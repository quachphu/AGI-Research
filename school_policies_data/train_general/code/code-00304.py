        import. It is required to decouple the function from assuming importlib's
        import implementation is desired.

        """
        module_name = demangle(module.__name__)
        # The hell that is fromlist ...
        # If a package was imported, try to import stuff from fromlist.
        if hasattr(module, "__path__"):
            for x in fromlist:
                if not isinstance(x, str):
                    if recursive:
                        where = module_name + ".__all__"
                    else:
                        where = "``from list''"
                    raise TypeError(
                        f"Item in {where} must be str, not {type(x).__name__}"
                    )
                elif x == "*":
                    if not recursive and hasattr(module, "__all__"):
                        self._handle_fromlist(module, module.__all__, recursive=True)
                elif not hasattr(module, x):
                    from_name = f"{module_name}.{x}"
                    try:
                        self._gcd_import(from_name)
                    except ModuleNotFoundError as exc:
                        # Backwards-compatibility dictates we ignore failed
                        # imports triggered by fromlist for modules that don't
                        # exist.
                        if (
                            exc.name == from_name
                            and self.modules.get(from_name, _NEEDS_LOADING) is not None
                        ):
                            continue
                        raise
        return module

    def __import__(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            module = self._gcd_import(name)
        else:
            globals_ = globals if globals is not None else {}
            package = _calc___package__(globals_)
            module = self._gcd_import(name, package, level)
        if not fromlist:
            # Return up to the first dot in 'name'. This is complicated by the fact
            # that 'name' may be relative.
            if level == 0:
                return self._gcd_import(name.partition(".")[0])
            elif not name:
                return module
            else:
                # Figure out where to slice the module's name up to the first dot
                # in 'name'.
                cut_off = len(name) - len(name.partition(".")[0])
                # Slice end needs to be positive to alleviate need to special-case
                # when ``'.' not in name``.
                module_name = demangle(module.__name__)
                return self.modules[module_name[: len(module_name) - cut_off]]
        else:
            return self._handle_fromlist(module, fromlist)

    def _get_package(self, package):
        """Take a package name or module object and return the module.

        If a name, the module is imported.  If the passed or imported module
        object is not a package, raise an exception.
        """
        if hasattr(package, "__spec__"):
            if package.__spec__.submodule_search_locations is None:
                raise TypeError(f"{package.__spec__.name!r} is not a package")
            else:
                return package
        else:
            module = self.import_module(package)
            if module.__spec__.submodule_search_locations is None:
                raise TypeError(f"{package!r} is not a package")
            else:
                return module

    def _zipfile_path(self, package, resource=None):
        package = self._get_package(package)
        if package.__loader__ is not self:
            raise AssertionError(
                f"package.__loader__ must be self, got {package.__loader__}"
            )
        name = demangle(package.__name__)
        if resource is not None:
            resource = _normalize_path(resource)
            return f"{name.replace('.', '/')}/{resource}"
        else:
            return f"{name.replace('.', '/')}"

    def _get_or_create_package(self, atoms: list[str]) -> "_PackageNode | _ExternNode":
        cur = self.root
        for i, atom in enumerate(atoms):
            node = cur.children.get(atom, None)
            if node is None:
                node = cur.children[atom] = _PackageNode(None)
            if isinstance(node, _ExternNode):
                return node
            if isinstance(node, _ModuleNode):
                name = ".".join(atoms[:i])
                raise ImportError(
                    f"inconsistent module structure. module {name} is not a package, but has submodules"
                )
            if not isinstance(node, _PackageNode):
                raise AssertionError(
                    f"expected _PackageNode, got {type(node).__name__}"
                )
            cur = node
        return cur

    def _add_file(self, filename: str):
        """Assembles a Python module out of the given file. Will ignore files in the .data directory.

        Args:
            filename (str): the name of the file inside of the package archive to be added
        """
        *prefix, last = filename.split("/")
        if len(prefix) > 1 and prefix[0] == ".data":
            return
        package = self._get_or_create_package(prefix)
        if isinstance(package, _ExternNode):
            raise ImportError(
                f"inconsistent module structure. package contains a module file {filename}"
                f" that is a subpackage of a module marked external."
            )
        if last == "__init__.py":
            package.source_file = filename
        elif last.endswith(".py"):
            package_name = last[: -len(".py")]
            package.children[package_name] = _ModuleNode(filename)

    def _add_extern(self, extern_name: str):
        *prefix, last = extern_name.split(".")
        package = self._get_or_create_package(prefix)
        if isinstance(package, _ExternNode):
            return  # the shorter extern covers this extern case
        package.children[last] = _ExternNode()


_NEEDS_LOADING = object()
_ERR_MSG_PREFIX = "No module named "
_ERR_MSG = _ERR_MSG_PREFIX + "{!r}"


class _PathNode:
    __slots__ = []


class _PackageNode(_PathNode):
    def __init__(self, source_file: str | None):
        self.source_file = source_file
        self.children: dict[str, _PathNode] = {}


class _ModuleNode(_PathNode):
    __slots__ = ["source_file"]

    def __init__(self, source_file: str):
        self.source_file = source_file


class _ExternNode(_PathNode):
    pass


# A private global registry of all modules that have been package-imported.
_package_imported_modules: WeakValueDictionary = WeakValueDictionary()

# `inspect` by default only looks in `sys.modules` to find source files for classes.
# Patch it to check our private registry of package-imported modules as well.
_orig_getfile = inspect.getfile


def _patched_getfile(object):
    if inspect.isclass(object):
        if object.__module__ in _package_imported_modules:
            return _package_imported_modules[object.__module__].__file__
    return _orig_getfile(object)


inspect.getfile = _patched_getfile


class _PackageResourceReader:
    """Private class used to support PackageImporter.get_resource_reader().

    Conforms to the importlib.abc.ResourceReader interface. Allowed to access
    the innards of PackageImporter.
    """

    def __init__(self, importer, fullname):
        self.importer = importer
        self.fullname = fullname

    def open_resource(self, resource):
        from io import BytesIO

        return BytesIO(self.importer.load_binary(self.fullname, resource))

    def resource_path(self, resource):
        # The contract for resource_path is that it either returns a concrete
        # file system path or raises FileNotFoundError.
        if isinstance(
            self.importer.zip_reader, DirectoryReader
        ) and self.importer.zip_reader.has_record(
            os.path.join(self.fullname, resource)
        ):
            return os.path.join(
                self.importer.zip_reader.directory, self.fullname, resource
            )
        raise FileNotFoundError

    def is_resource(self, name):
        path = self.importer._zipfile_path(self.fullname, name)
        return self.importer.zip_reader.has_record(path)

    def contents(self):
        from pathlib import Path

        filename = self.fullname.replace(".", "/")

        fullname_path = Path(self.importer._zipfile_path(self.fullname))
        files = self.importer.zip_reader.get_all_records()
        subdirs_seen = set()
        for filename in files:
            try:
                relative = Path(filename).relative_to(fullname_path)
            except ValueError:
                continue
            # If the path of the file (which is relative to the top of the zip
            # namespace), relative to the package given when the resource
            # reader was created, has a parent, then it's a name in a
            # subdirectory and thus we skip it.
            parent_name = relative.parent.name
            if len(parent_name) == 0:
                yield relative.name
            elif parent_name not in subdirs_seen:
                subdirs_seen.add(parent_name)
                yield parent_name
<|endoftext|>import sys
from pickle import (
    _compat_pickle,  # pyrefly: ignore [missing-module-attribute]
    _extension_registry,  # pyrefly: ignore [missing-module-attribute]
    _getattribute,  # pyrefly: ignore [missing-module-attribute]
    _Pickler,
    EXT1,
    EXT2,
    EXT4,
    GLOBAL,
    PicklingError,
    STACK_GLOBAL,
)
from struct import pack
from types import FunctionType

from .importer import Importer, ObjMismatchError, ObjNotFoundError, sys_importer


class _PyTorchLegacyPickler(_Pickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._persistent_id = None

    def persistent_id(self, obj):
        if self._persistent_id is None:
            return super().persistent_id(obj)
        return self._persistent_id(obj)


class PackagePickler(_PyTorchLegacyPickler):
    """Package-aware pickler.

    This behaves the same as a normal pickler, except it uses an `Importer`
    to find objects and modules to save.
    """

    def __init__(self, importer: Importer, *args, **kwargs):
        self.importer = importer
        super().__init__(*args, **kwargs)

        # Make sure the dispatch table copied from _Pickler is up-to-date.
        # Previous issues have been encountered where a library (e.g. dill)
        # mutate _Pickler.dispatch, PackagePickler makes a copy when this lib
        # is imported, then the offending library removes its dispatch entries,
        # leaving PackagePickler with a stale dispatch table that may cause
        # unwanted behavior.
        self.dispatch = _Pickler.dispatch.copy()  # type: ignore[misc]
        self.dispatch[FunctionType] = PackagePickler.save_global  # type: ignore[assignment]

    def save_global(self, obj, name=None):
        # ruff: noqa: F841
        # unfortunately the pickler code is factored in a way that
        # forces us to copy/paste this function. The only change is marked
        # CHANGED below.
        write = self.write  # type: ignore[attr-defined]
        memo = self.memo  # type: ignore[attr-defined]

        # CHANGED: import module from module environment instead of __import__
        try:
            module_name, name = self.importer.get_name(obj, name)
        except (ObjNotFoundError, ObjMismatchError) as err:
            raise PicklingError(f"Can't pickle {obj}: {str(err)}") from err

        module = self.importer.import_module(module_name)
        if sys.version_info >= (3, 14):
            # pickle._getattribute signature changes in 3.14
            # to take iterable and return just the object (not tuple)
            # We need to get the parent object that contains the attribute
            name_parts = name.split(".")
            if "<locals>" in name_parts:
                raise PicklingError(f"Can't pickle local object {obj!r}")
            if len(name_parts) == 1:
                parent = module
            else:
                parent = _getattribute(module, name_parts[:-1])
        else:
            _, parent = _getattribute(module, name)
        # END CHANGED

        if self.proto >= 2:  # type: ignore[attr-defined]
            code = _extension_registry.get((module_name, name))
            if code:
                if code <= 0:
                    raise AssertionError(
                        f"expected positive extension code, got {code}"
                    )
                if code <= 0xFF:
                    write(EXT1 + pack("<B", code))
                elif code <= 0xFFFF:
                    write(EXT2 + pack("<H", code))
                else:
                    write(EXT4 + pack("<i", code))
                return
        lastname = name.rpartition(".")[2]
        if parent is module:
            name = lastname
        # Non-ASCII identifiers are supported only with protocols >= 3.
        if self.proto >= 4:  # type: ignore[attr-defined]
            self.save(module_name)  # type: ignore[attr-defined]
            self.save(name)  # type: ignore[attr-defined]
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))  # type: ignore[attr-defined]
        elif self.proto >= 3:  # type: ignore[attr-defined]
            write(
                GLOBAL
                + bytes(module_name, "utf-8")
                + b"\n"
                + bytes(name, "utf-8")
                + b"\n"
            )
        else:
            if self.fix_imports:  # type: ignore[attr-defined]
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[(module_name, name)]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(
                    GLOBAL
                    + bytes(module_name, "ascii")
                    + b"\n"
                    + bytes(name, "ascii")
                    + b"\n"
                )
            except UnicodeEncodeError as exc:
                raise PicklingError(
                    f"can't pickle global identifier '{module}.{name}' using "
                    f"pickle protocol {self.proto:d}"  # type: ignore[attr-defined]
                ) from exc

        self.memoize(obj)  # type: ignore[attr-defined]


def create_pickler(data_buf, importer, protocol=4):
    if importer is sys_importer:
        # if we are using the normal import library system, then
        # we can use the C implementation of pickle which is faster
        return _PyTorchLegacyPickler(data_buf, protocol=protocol)
    else:
        return PackagePickler(importer, data_buf, protocol=protocol)
<|endoftext|># mypy: allow-untyped-defs
import _warnings
import os.path


# note: implementations
# copied from cpython's import code


# _zip_searchorder defines how we search for a module in the Zip
# archive: we first search for a package __init__, then for
# non-package .pyc, and .py entries. The .pyc entries
# are swapped by initzipimport() if we run in optimized mode. Also,
# '/' is replaced by path_sep there.

_zip_searchorder = (
    ("/__init__.py", True),
    (".py", False),
)


# Replace any occurrences of '\r\n?' in the input string with '\n'.
# This converts DOS and Mac line endings to Unix line endings.
def _normalize_line_endings(source):
    source = source.replace(b"\r\n", b"\n")
    source = source.replace(b"\r", b"\n")
    return source


def _resolve_name(name, package, level):
    """Resolve a relative module name to an absolute one."""
    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        raise ValueError("attempted relative import beyond top-level package")
    base = bits[0]
    return f"{base}.{name}" if name else base


def _sanity_check(name, package, level):
    """Verify arguments are "sane"."""
    if not isinstance(name, str):
        raise TypeError(f"module name must be str, not {type(name)}")
    if level < 0:
        raise ValueError("level must be >= 0")
    if level > 0:
        if not isinstance(package, str):
            raise TypeError("__package__ not set to a string")
        elif not package:
            raise ImportError("attempted relative import with no known parent package")
    if not name and level == 0:
        raise ValueError("Empty module name")


def _calc___package__(globals):
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    """
    package = globals.get("__package__")
    spec = globals.get("__spec__")
    if package is not None:
        if spec is not None and package != spec.parent:
            _warnings.warn(
                f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})",
                ImportWarning,
                stacklevel=3,
            )
        return package
    elif spec is not None:
        return spec.parent
    else:
        _warnings.warn(
            "can't resolve package from __spec__ or __package__, "
            "falling back on __name__ and __path__",
            ImportWarning,
            stacklevel=3,
        )
        package = globals["__name__"]
        if "__path__" not in globals:
            package = package.rpartition(".")[0]
    return package


def _normalize_path(path):
    """Normalize a path by ensuring it is a string.

    If the resulting string contains path separators, an exception is raised.
    """
    parent, file_name = os.path.split(path)
    if parent:
        raise ValueError(f"{path!r} must be only a file name")
    else:
        return file_name
<|endoftext|># mypy: allow-untyped-defs
import collections
import importlib.machinery
import io
import linecache
import os
import pickletools
import platform
import types
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, cast, IO

import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import FileLike, Storage
from torch.utils.hooks import RemovableHandle

from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import demangle, is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer


__all__ = [
    "PackagingErrorReason",
    "EmptyMatchError",
    "PackagingError",
    "PackageExporter",
]

_gate_torchscript_serialization = True

ActionHook = Callable[["PackageExporter", str], None]


class _ModuleProviderAction(Enum):
    """Represents one of the actions that :class:`PackageExporter` can take on a module.

    See :meth:`PackageExporter.extern` and friends for a description of what the actions do.
    """

    INTERN = 1
    EXTERN = 2
    MOCK = 3
    DENY = 4
    # Special case: when a module is mocked, PackageExporter writes out a
    # `_mock` module that implements our mocking stubs. If we re-package code,
    # we may encounter a `_mock` module from the original package. If we do,
    # just ignore it and write a `_mock` module once.
    REPACKAGED_MOCK_MODULE = 5
    # Special case: PackageImporter adds a fake module
    # (`torch_package_importer`) that allows packaged code to access it. Don't
    # re-export this.
    SKIP = 6


class PackagingErrorReason(Enum):
    """Listing of different reasons a dependency may fail to package.

    This enum is used to provide good error messages when
    :class:`PackagingError` is raised.
    """

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}>"

    IS_EXTENSION_MODULE = (
        "Module is a C extension module. torch.package supports Python modules only."
    )
    NO_DUNDER_FILE = "Module had no __file__ defined."
    SOURCE_FILE_NOT_FOUND = (
        "Module had a __file__, but we could not find it in your filesystem."
    )
    DEPENDENCY_RESOLUTION_FAILED = "Dependency resolution failed."
    NO_ACTION = (
        "Module did not match against any action pattern. Extern, mock, or intern it."
    )
    DENIED = "Module was denied by a pattern."
    MOCKED_BUT_STILL_USED = (
        "Module was mocked out, but is still being used in the package. "
        "Please intern or extern the mocked modules if objects are supposed to be in "
        "the package."
    )


@dataclass
class _PatternInfo:
    """Holds :class:`PackageExporter`-specific info about how to execute matches against"""

    # What action to take on a module that matches this pattern.
    action: _ModuleProviderAction
    # The value of `allow_empty` the user gave when specifying the pattern.
    allow_empty: bool
    # Whether this pattern has been matched during packaging.
    was_matched: bool

    def __init__(self, action, allow_empty):
        self.action = action
        self.allow_empty = allow_empty
        self.was_matched = False


class EmptyMatchError(Exception):
    """This is an exception that is thrown when a mock or extern is marked as
    ``allow_empty=False``, and is not matched with any module during packaging.
    """


class PackagingError(Exception):
    """This exception is raised when there is an issue with exporting a package.
    ``PackageExporter`` will attempt to gather up all the errors and present
    them to you at once.
    """

    def __init__(self, dependency_graph: DiGraph, debug=False):
        # Group errors by reason.
        broken: dict[PackagingErrorReason, list[str]] = defaultdict(list)
        for module_name, attrs in dependency_graph.nodes.items():
            error = attrs.get("error")
            if error is None:
                continue
            if error == PackagingErrorReason.NO_ACTION:
                if "action" in attrs:
                    raise AssertionError(
                        f"module {module_name} has NO_ACTION error but action is set"
                    )
            broken[error].append(module_name)

        message = io.StringIO()
        message.write("\n")

        for reason, module_names in broken.items():
            message.write(f"* {reason.value}\n")
            for module_name in module_names:
                message.write(f"    {module_name}\n")

                # Print additional context if it's provided.
                error_context = dependency_graph.nodes[module_name].get("error_context")
                if error_context is not None:
                    message.write(f"      Context: {error_context}\n")
                if module_name in _DISALLOWED_MODULES:
                    message.write(
                        "      Note: While we usually use modules in the python standard library "
                        f"from the local environment, `{module_name}` has a lot of system "
                        "level access and therefore can pose a security risk. We heavily "
                        f"recommend removing `{module_name}` from your packaged code. However, if that "
                        "is not possible, add it to the extern list by calling "
                        f'PackageExporter.extern("`{module_name}`")\n'
                    )
                if debug:
                    module_path = dependency_graph.first_path(module_name)
                    message.write(
                        f"      A path to {module_name}: {' -> '.join(module_path)}\n"
                    )
        if not debug:
            message.write("\n")
            message.write(
                "Set debug=True when invoking PackageExporter for a visualization of where "
                "broken modules are coming from!\n"
            )
        # Save the dependency graph so that tooling can get at it.
        self.dependency_graph = dependency_graph
        super().__init__(message.getvalue())


class PackageExporter:
    """Exporters allow you to write packages of code, pickled Python data, and
    arbitrary binary and text resources into a self-contained package.

    Imports can load this code in a hermetic way, such that code is loaded
    from the package rather than the normal Python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The code contained in packages is copied file-by-file from the original
    source when it is created, and the file format is a specially organized
    zip file. Future users of the package can unzip the package, and edit the code
    in order to perform custom modifications to it.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external using :meth:`extern`.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.

    When source code is added to the package, the exporter can optionally scan it
    for further code dependencies (``dependencies=True``). It looks for import statements,
    resolves relative references to qualified module names, and performs an action specified by the user
    (See: :meth:`extern`, :meth:`mock`, and :meth:`intern`).
    """

    """A importer that will be searched in order to find the modules referenced by other modules or by
    pickled objects. The default module environment just uses sys_importer, which searches the Python environment.
    """
    importer: Importer

    def __init__(
        self,
        f: FileLike,
        importer: Importer | Sequence[Importer] = sys_importer,
        debug: bool = False,
    ) -> None:
        """
        Create an exporter.

        Args:
            f: The location to export to. Can be a  ``string``/``Path`` object containing a filename
                or a binary I/O object.
            importer: If a single Importer is passed, use that to search for modules.
                If a sequence of importers are passed, an ``OrderedImporter`` will be constructed out of them.
            debug: If set to True, add path of broken modules to PackagingErrors.
        """
        torch._C._log_api_usage_once("torch.package.PackageExporter")
        self.debug = debug
        if isinstance(f, (str, os.PathLike)):
            f = os.fspath(f)
            self.buffer: IO[bytes] | None = None
        else:  # is a byte buffer
            self.buffer = f

        self.zip_file = torch._C.PyTorchFileWriter(f)
        self.zip_file.set_min_version(6)
        self._written_files: set[str] = set()

        self.serialized_reduces: dict[int, Any] = {}

        # A graph tracking all the modules and pickle objects added to this
        # package and the dependencies between them.
        # - Each node is a module name (or a pickle name that looks like '<foo.obj.pkl>')
        # - Each directed edge (u, v) means u depends on v.
        # - Nodes may contain metadata that describe how to write the thing to the zipfile.
        self.dependency_graph = DiGraph()
        self.script_module_serializer = torch._C.ScriptModuleSerializer(self.zip_file)
        self.storage_context = self.script_module_serializer.storage_context()

        # These are OrderedDicts for compatibility with RemovableHandle.
        # Generic OrderedDict type annotations are not present until 3.7.
        # The real type signature is OrderedDict[int, Callable[[PackageExporter, str], None]]
        self._extern_hooks: OrderedDict = OrderedDict()
        self._mock_hooks: OrderedDict = OrderedDict()
        self._intern_hooks: OrderedDict = OrderedDict()

        if isinstance(importer, Importer):
            self.importer = importer
        else:
            if not isinstance(importer, collections.abc.Sequence):
                raise TypeError(
                    "importer arg should be an Importer or a sequence of Importers, "
                    f"got {type(importer)} instead."
                )
            self.importer = OrderedImporter(*importer)

        self.patterns: dict[GlobGroup, _PatternInfo] = {}
        self._unique_id = 0

    def save_source_file(
        self, module_name: str, file_or_directory: str, dependencies=True
    ):
        """Adds the local file system ``file_or_directory`` to the source package to provide the code
        for ``module_name``.

        Args:
            module_name (str): e.g. ``"my_package.my_subpackage"``, code will be saved to provide code for this package.
            file_or_directory (str): the path to a file or directory of code. When a directory, all python files in the directory
                are recursively copied using :meth:`save_source_file`. If a file is named ``"/__init__.py"`` the code is treated
                as a package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        path = Path(file_or_directory)
        if path.is_dir():
            to_save = []  # list of tuples with arguments to save_source_string
            module_path = module_name.replace(".", "/")
            for filename in path.glob("**/*.py"):
                relative_path = filename.relative_to(path).as_posix()
                archivename = module_path + "/" + relative_path
                submodule_name = None
                if filename.name == "__init__.py":
                    submodule_name = archivename[: -len("/__init__.py")].replace(
                        "/", "."
                    )
                    is_package = True
                else:
                    submodule_name = archivename[: -len(".py")].replace("/", ".")
                    is_package = False

                # we delay the call to save_source_string so that we record all the source files
                # being provided by this directory structure _before_ attempting to resolve the dependencies
                # on the source. This makes sure we don't try to copy over modules that will just get
                # overwritten by this directory blob
                to_save.append(
                    (
                        submodule_name,
                        _read_file(str(filename)),
                        is_package,
                        dependencies,
                    )
                )

            for item in to_save:
                self.save_source_string(*item)
        else:
            is_package = path.name == "__init__.py"
            self.save_source_string(
                module_name,
                _read_file(file_or_directory),
                is_package,
                dependencies,
            )

    def get_unique_id(self) -> str:
        """Get an id. This id is guaranteed to only be handed out once for this package."""
        ret = str(self._unique_id)
        self._unique_id += 1
        return ret

    def _get_dependencies(
        self, src: str, module_name: str, is_package: bool
    ) -> list[str]:
        """Return all modules that this source code depends on.

        Dependencies are found by scanning the source code for import-like statements.

        Arguments:
            src: The Python source code to analyze for dependencies.
            module_name: The name of the module that ``src`` corresponds to.
            is_package: Whether this module should be treated as a package.
                See :py:meth:`save_source_string` for more info.

        Returns:
            A list containing modules detected as direct dependencies in
            ``src``.  The items in the list are guaranteed to be unique.
        """
        package_name = (
            module_name if is_package else module_name.rsplit(".", maxsplit=1)[0]
        )
        try:
            dep_pairs = find_files_source_depends_on(src, package_name)
        except Exception as e:
            self.dependency_graph.add_node(
                module_name,
                error=PackagingErrorReason.DEPENDENCY_RESOLUTION_FAILED,
                error_context=str(e),
            )
            return []

        # Use a dict to get uniquing but also deterministic order
        dependencies = {}
        for dep_module_name, dep_module_obj in dep_pairs:
            # handle the case where someone did something like `from pack import sub`
            # where `sub` is a submodule. In this case we don't have to save pack, just sub.
            # this ensures we don't pick up additional dependencies on pack.
            # However, in the case where `sub` is not a submodule but an object, then we do have
            # to save pack.
            if dep_module_obj is not None:
                possible_submodule = f"{dep_module_name}.{dep_module_obj}"
                if self._module_exists(possible_submodule):
                    dependencies[possible_submodule] = True
                    # we don't need to save `pack`
                    continue
            if self._module_exists(dep_module_name):
                dependencies[dep_module_name] = True

        return list(dependencies.keys())

    def save_source_string(
        self,
        module_name: str,
        src: str,
        is_package: bool = False,
        dependencies: bool = True,
    ):
        """Adds ``src`` as the source code for ``module_name`` in the exported package.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code for this package.
            src (str): The Python source code to save for this package.
            is_package (bool, optional): If ``True``, this module is treated as a package. Packages are allowed to have submodules
                (e.g. ``my_package.my_subpackage.my_subsubpackage``), and resources can be saved inside them. Defaults to ``False``.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        self.dependency_graph.add_node(
            module_name,
            source=src,
            is_package=is_package,
            provided=True,
            action=_ModuleProviderAction.INTERN,
        )

        if dependencies:
            deps = self._get_dependencies(src, module_name, is_package)

            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def _write_source_string(
        self,
        module_name: str,
        src: str,
        is_package: bool = False,
    ):
        """Write ``src`` as the source code for ``module_name`` in the zip archive.

        Arguments are otherwise the same as for :meth:`save_source_string`.
        """
        extension = "/__init__.py" if is_package else ".py"
        filename = module_name.replace(".", "/") + extension

        self._write(filename, src)

    def _import_module(self, module_name: str):
        try:
            return self.importer.import_module(module_name)
        except ModuleNotFoundError:
            if not is_mangled(module_name):
                raise
            msg = (
                f"Module not found: '{module_name}'. Make sure the PackageImporter that "
                "created this module is present in `self.importer`"
            )
            raise ModuleNotFoundError(msg) from None

    def _module_exists(self, module_name: str) -> bool:
        try:
            self._import_module(module_name)
            return True
        except Exception:
            return False

    def _get_source_of_module(self, module: types.ModuleType) -> str | None:
        filename = None
        spec = getattr(module, "__spec__", None)
        if spec is not None:
            loader = getattr(spec, "loader", None)
            if loader is not None and isinstance(loader, SourceFileLoader):
                try:
                    filename = loader.get_filename(module.__name__)
                except ImportError:
                    pass
        if filename is None:
            filename = getattr(module, "__file__", None)
        if isinstance(filename, str) and filename.endswith(".py"):
            return "".join(linecache.getlines(filename, module.__dict__))
        return None

    def add_dependency(self, module_name: str, dependencies=True):
        """Given a module, add it to the dependency graph according to patterns
        specified by the user.
        """
        if (
            module_name in self.dependency_graph
            and self.dependency_graph.nodes[module_name].get("provided") is True
        ):
            return

        # Special case: PackageImporter provides a special module called
        # `torch_package_importer` that allows packaged modules to reference
        # their PackageImporter. We don't want to re-export this.
        if module_name == "torch_package_importer":
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.SKIP,
                provided=True,
            )
            return

        if module_name == "_mock":
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.REPACKAGED_MOCK_MODULE,
                provided=True,
            )
            return

        if self._can_implicitly_extern(module_name):
            self.dependency_graph.add_node(
                module_name, action=_ModuleProviderAction.EXTERN, provided=True
            )
            return

        for pattern, pattern_info in self.patterns.items():
            if pattern.matches(module_name):
                pattern_info.was_matched = True
                self.dependency_graph.add_node(
                    module_name, action=pattern_info.action, provided=True
                )

                if pattern_info.action == _ModuleProviderAction.DENY:
                    # Requiring a denied module just adds an error to the graph.
                    self.dependency_graph.add_node(
                        module_name, error=PackagingErrorReason.DENIED
                    )

                # If we are interning this module, we need to retrieve its
                # dependencies and package those as well.
                if pattern_info.action == _ModuleProviderAction.INTERN:
                    self._intern_module(module_name, dependencies)
                return

        # No patterns have matched. Explicitly add this as an error.
        self.dependency_graph.add_node(
            module_name, error=PackagingErrorReason.NO_ACTION
        )

    def save_module(self, module_name: str, dependencies=True):
        """Save the code for ``module`` into the package. Code for the module is resolved using the ``importers`` path to find the
        module object, and then using its ``__file__`` attribute to find the source code.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code
                for this package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        if not isinstance(module_name, str):
            raise TypeError(
                "save_module() expects a string input, did you perhaps mean to pass `__name__`?"
            )

        self._intern_module(module_name, dependencies)

    def _intern_module(
        self,
        module_name: str,
        dependencies: bool,
    ):
        """Adds the module to the dependency graph as an interned module,
        along with any metadata needed to write it out to the zipfile at serialization time.
        """
        module_obj = self._import_module(module_name)
        # Subtle: if the import above succeeded, either:
        #   1. The module name is not mangled, and this was just a regular import, or
        #   2. The module name is mangled, but one of the importers was able to
        #      recognize the mangling and import it.
        # Either way, it is now safe to demangle this name so that we don't
        # serialize the mangled version to the package.
        module_name = demangle(module_name)

        # Find dependencies of this module and require them as well.
        is_package = hasattr(module_obj, "__path__")
        source = self._get_source_of_module(module_obj)
        if source is None:
            # Couldn't find a source!  Add it to our dependency graph as broken
            # and continue.
            filename = getattr(module_obj, "__file__", None)
            error_context = None
            if filename is None:
                packaging_error = PackagingErrorReason.NO_DUNDER_FILE
            elif filename.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
                packaging_error = PackagingErrorReason.IS_EXTENSION_MODULE
            else:
                packaging_error = PackagingErrorReason.SOURCE_FILE_NOT_FOUND
                error_context = f"filename: {filename}"
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.INTERN,
                is_package=is_package,
                error=packaging_error,
                error_context=error_context,
                provided=True,
            )
            return

        self.dependency_graph.add_node(
            module_name,
            action=_ModuleProviderAction.INTERN,
            is_package=is_package,
            source=source,
            provided=True,
        )

        if dependencies:
            deps = self._get_dependencies(source, module_name, is_package)
            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def save_pickle(
        self,
        package: str,
        resource: str,
        obj: Any,
        dependencies: bool = True,
        pickle_protocol: int = 3,
    ):
        """Save a python object to the archive using pickle. Equivalent to :func:`torch.save` but saving into
        the archive rather than a stand-alone file. Standard pickle does not save the code, only the objects.
        If ``dependencies`` is true, this method will also scan the pickled objects for which modules are required
        to reconstruct them and save the relevant code.

        To be able to save an object where ``type(obj).__name__`` is ``my_module.MyObject``,
        ``my_module.MyObject`` must resolve to the class of the object according to the ``importer`` order. When saving objects that
        have previously been packaged, the importer's ``import_module`` method will need to be present in the ``importer`` list
        for this to work.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            obj (Any): The object to save, must be picklable.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """

        if pickle_protocol not in (3, 4):
            raise AssertionError(
                f"torch.package only supports pickle protocols 3 and 4, got {pickle_protocol}"
            )

        filename = self._filename(package, resource)
        # Write the pickle data for `obj`
        data_buf = io.BytesIO()
        pickler = create_pickler(data_buf, self.importer, protocol=pickle_protocol)
        pickler.persistent_id = self._persistent_id
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        mocked_modules = defaultdict(list)
        name_in_dependency_graph = f"<{package}.{resource}>"
        self.dependency_graph.add_node(
            name_in_dependency_graph,
            action=_ModuleProviderAction.INTERN,
            provided=True,
            is_pickle=True,
        )

        def _check_mocked_error(module: str | None, field: str | None):
            """
            checks if an object (field) comes from a mocked module and then adds
            the pair to mocked_modules which contains mocked modules paired with their
            list of mocked objects present in the pickle.

            We also hold the invariant that the first user defined rule that applies
            to the module is the one we use.
            """

            if not isinstance(module, str):
                raise AssertionError(f"module must be str, got {type(module).__name__}")
            if not isinstance(field, str):
                raise AssertionError(f"field must be str, got {type(field).__name__}")
            if self._can_implicitly_extern(module):
                return
            for pattern, pattern_info in self.patterns.items():
                if pattern.matches(module):
                    if pattern_info.action == _ModuleProviderAction.MOCK:
                        mocked_modules[module].append(field)
                    return

        if dependencies:
            all_dependencies = []
            module = None
            field = None
            memo: defaultdict[int, str] = defaultdict(None)
            memo_count = 0
            # pickletools.dis(data_value)
            # pyrefly: ignore [bad-assignment]
            for opcode, arg, _pos in pickletools.genops(data_value):
                if pickle_protocol == 4:
                    if (
                        opcode.name == "SHORT_BINUNICODE"
                        or opcode.name == "BINUNICODE"
                        or opcode.name == "BINUNICODE8"
                    ):
                        if not isinstance(arg, str):
                            raise AssertionError(
                                f"expected str arg for {opcode.name}, got {type(arg).__name__}"
                            )
                        module = field
                        field = arg
                        memo[memo_count] = arg
                    elif (
                        opcode.name == "LONG_BINGET"
                        or opcode.name == "BINGET"
                        or opcode.name == "GET"
                    ):
                        if not isinstance(arg, int):
                            raise AssertionError(
                                f"expected int arg for {opcode.name}, got {type(arg).__name__}"
                            )
                        module = field
                        field = memo.get(arg, None)
                    elif opcode.name == "MEMOIZE":
                        memo_count += 1
                    elif opcode.name == "STACK_GLOBAL":
                        if module is None:
                            # If no module was passed on in the entries preceding this one, continue.
                            continue
                        if not isinstance(module, str):
                            raise AssertionError(
                                f"module must be str, got {type(module).__name__}"
                            )
                        if module not in all_dependencies:
                            all_dependencies.append(module)
                        _check_mocked_error(module, field)
                elif (
                    pickle_protocol == 3 and opcode.name == "GLOBAL"
                ):  # a global reference
                    if not isinstance(arg, str):
                        raise AssertionError(
                            f"expected str arg for GLOBAL, got {type(arg).__name__}"
                        )
                    module, field = arg.split(" ")
                    if module not in all_dependencies:
                        all_dependencies.append(module)
                    _check_mocked_error(module, field)
            for module_name in all_dependencies:
                self.dependency_graph.add_edge(name_in_dependency_graph, module_name)

                """ If an object happens to come from a mocked module, then we collect these errors and spit them
                    out with the other errors found by package exporter.
                """
                if module_name in mocked_modules:
                    if not isinstance(module_name, str):
                        raise AssertionError(
                            f"module_name must be str, got {type(module_name).__name__}"
                        )
                    fields = mocked_modules[module_name]
                    self.dependency_graph.add_node(
                        module_name,
                        action=_ModuleProviderAction.MOCK,
                        error=PackagingErrorReason.MOCKED_BUT_STILL_USED,
                        error_context=f"Object(s) '{fields}' from module `{module_name}` was mocked out during packaging "
                        f"but is being used in resource - `{resource}` in package `{package}`. ",
                        provided=True,
                    )
                else:
                    self.add_dependency(module_name)

        self._write(filename, data_value)

    def save_text(self, package: str, resource: str, text: str):
        """Save text data to the package.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            text (str): The contents to save.
        """
        return self.save_binary(package, resource, text.encode("utf-8"))

    def save_binary(self, package, resource, binary: bytes):
        """Save raw bytes to the package.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            binary (str): The data to save.
        """
        filename = self._filename(package, resource)
        self._write(filename, binary)

    def register_extern_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers an extern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`extern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._extern_hooks)
        self._extern_hooks[handle.id] = hook
        return handle

    def register_mock_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers a mock hook on the exporter.

        The hook will be called each time a module matches against a :meth:`mock` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._mock_hooks)
        self._mock_hooks[handle.id] = hook
        return handle

    def register_intern_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers an intern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`intern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        handle = RemovableHandle(self._intern_hooks)
        self._intern_hooks[handle.id] = hook
        return handle

    def intern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        """Specify modules that should be packaged. A module must match some ``intern`` pattern in order to be
        included in the package and have its dependencies processed recursively.

        Args:
            include (list[str] | str): A string e.g. "my_package.my_subpackage", or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.

            exclude (list[str] | str): An optional pattern that excludes some patterns that match the include string.

            allow_empty (bool): An optional flag that specifies whether the intern modules specified by this call
                to the ``intern`` method must be matched to some module during packaging. If an ``intern`` module glob
                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``)
                before any modules match that pattern, an exception is thrown. If ``allow_empty=True``, no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.INTERN, allow_empty
        )

    def mock(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        """Replace some required modules with a mock implementation.  Mocked modules will return a fake
        object for any attribute accessed from it. Because we copy file-by-file, the dependency resolution will sometimes
        find files that are imported by model files but whose functionality is never used
        (e.g. custom serialization code or training helpers).
        Use this function to mock this functionality out without having to modify the original code.

        Args:
            include (list[str] | str): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be mocked out. Strings can also be a glob-style pattern
                string that may match multiple modules. Any required dependencies that match this pattern
                string will be mocked out automatically.

                Examples :
                    ``'torch.**'`` -- matches ``torch`` and all submodules of torch, e.g. ``'torch.nn'``
                    and ``'torch.nn.functional'``

                    ``'torch.*'`` -- matches ``'torch.nn'`` or ``'torch.functional'``, but not
                    ``'torch.nn.functional'``

            exclude (list[str] | str): An optional pattern that excludes some patterns that match the include string.
                e.g. ``include='torch.**', exclude='torch.foo'`` will mock all torch packages except ``'torch.foo'``,
                Default: is ``[]``.

            allow_empty (bool): An optional flag that specifies whether the mock implementation(s) specified by this call
                to the :meth:`mock` method must be matched to some module during packaging. If a mock is added with
                ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``) and the mock has
                not been matched to a module used by the package being exported, an exception is thrown.
                If ``allow_empty=True``, no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.MOCK, allow_empty
        )

    def extern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
    ):
        """Include ``module`` in the list of external modules the package can import.
        This will prevent dependency discovery from saving
        it in the package. The importer will load an external module directly from the standard import system.
        Code for extern modules must also exist in the process loading the package.

        Args:
            include (list[str] | str): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as
                described in :meth:`mock`.

            exclude (list[str] | str): An optional pattern that excludes some patterns that match the
                include string.

            allow_empty (bool): An optional flag that specifies whether the extern modules specified by this call
                to the ``extern`` method must be matched to some module during packaging. If an extern module glob
                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via
                ``__exit__``) before any modules match that pattern, an exception is thrown. If ``allow_empty=True``,
                no such exception is thrown.

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.EXTERN, allow_empty
        )

    def deny(self, include: "GlobPattern", *, exclude: "GlobPattern" = ()):
        """Blocklist modules whose names match the given glob patterns from the list of modules the package can import.
        If a dependency on any matching packages is found, a :class:`PackagingError` is raised.

        Args:
            include (list[str] | str): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.

            exclude (list[str] | str): An optional pattern that excludes some patterns that match the include string.
        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.DENY, allow_empty=True
        )

    def _persistent_id(self, obj):
        if torch.is_storage(obj) or isinstance(obj, torch.storage.TypedStorage):
            storage: Storage
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, we can
                # remove this case
                untyped_storage = obj._untyped_storage
                storage_type_str = obj.pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage = cast(Storage, untyped_storage)
                storage_numel = obj.size()

            elif isinstance(obj, torch.UntypedStorage):
                untyped_storage = obj
                storage = cast(Storage, untyped_storage)
                storage_type = normalize_storage_type(type(storage))
                storage_numel = storage.nbytes()
            else:
                raise RuntimeError(f"storage type not recognized: {type(obj)}")

            location = location_tag(storage)

            # serialize storage if not already written
            storage_present = self.storage_context.has_storage(storage)
            storage_id = self.storage_context.get_or_add_storage(storage)
            if not storage_present:
                if storage.device.type != "cpu":
                    storage = storage.cpu()
                num_bytes = storage.nbytes()
                self.zip_file.write_record(
                    f".data/{storage_id}.storage", storage, num_bytes
                )
            return ("storage", storage_type, storage_id, location, storage_numel)

        if hasattr(obj, "__reduce_package__"):
            if _gate_torchscript_serialization and isinstance(
                obj, torch.jit.RecursiveScriptModule
            ):
                raise Exception(  # noqa: TRY002
                    "Serializing ScriptModules directly into a package is a beta feature. "
                    "To use, set global "
                    "`torch.package.package_exporter._gate_torchscript_serialization` to `False`."
                )
            if self.serialized_reduces.get(id(obj)) is None:
                self.serialized_reduces[id(obj)] = (
                    "reduce_package",
                    id(obj),
                    *obj.__reduce_package__(self),
                )

            return self.serialized_reduces[id(obj)]

        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If __exit__ was called because an exception was raised, we do not
        # attempt to finalize the package. Instead, control is returned to the
        # caller to continue raising the exception.
        if exc_type is not None:
            # Do the bare minimum to leave the open buffer in a valid state.
            self._finalize_zip()
            return

        self.close()

    def _write(self, filename, str_or_bytes):
        if filename in self._written_files:
            raise AssertionError(
                f"Tried to write file '{filename}', but it already exists in this archive. "
                "Please file a bug."
            )
        self._written_files.add(filename)

        if is_mangled(filename):
            raise AssertionError(
                f"Tried to save a torch.package'd module as '{filename}'. "
                "Directly saving torch.package'd modules is not allowed."
            )
        if isinstance(str_or_bytes, str):
            str_or_bytes = str_or_bytes.encode("utf-8")
        self.zip_file.write_record(filename, str_or_bytes, len(str_or_bytes))

    def _validate_dependency_graph(self):
        # 1. Check the graph for any errors inserted during dependency analysis.
        for attrs in self.dependency_graph.nodes.values():
            if "error" in attrs:
                raise PackagingError(self.dependency_graph, debug=self.debug)

        # 2. Check that all patterns for which allow_empty=False have been matched at least once.
        for pattern, pattern_info in self.patterns.items():
            if not pattern_info.allow_empty and not pattern_info.was_matched:
                raise EmptyMatchError(
                    f"Exporter did not match any modules to {pattern}, which was marked as allow_empty=False"
                )

    def _write_mock_file(self):
        if "_mock.py" not in self._written_files:
            mock_file = str(Path(__file__).parent / "_mock.py")
            self._write_source_string("_mock", _read_file(mock_file), is_package=False)

    def _execute_dependency_graph(self):
        """Takes a finalized dependency graph describing how to package all
        modules and executes it, writing to the ZIP archive.
        """
        self._validate_dependency_graph()

        extern_modules = []
        for module_name, attrs in self.dependency_graph.nodes.items():
            action = attrs["action"]

            if action == _ModuleProviderAction.EXTERN:
                for hook in self._extern_hooks.values():
                    hook(self, module_name)

                extern_modules.append(module_name)

            elif action == _ModuleProviderAction.MOCK:
                for hook in self._mock_hooks.values():
                    hook(self, module_name)

                self._write_mock_file()

                is_package = hasattr(self._import_module(module_name), "__path__")
                self._write_source_string(module_name, _MOCK_IMPL, is_package)

            elif action == _ModuleProviderAction.INTERN:
                for hook in self._intern_hooks.values():
                    hook(self, module_name)

                # The node in the dependency graph contains metadata that tells us
                # how to intern the module.
                if "provided" not in attrs:
                    raise AssertionError(
                        f"Module was marked `intern` but not provided: {module_name}"
                    )

                if attrs.get("is_pickle") is True:
                    # This node came from save_pickle, we don't need to write any source for it.
                    continue

                is_package = attrs["is_package"]
                source = attrs["source"]
                self._write_source_string(module_name, source, is_package)

            elif action == _ModuleProviderAction.REPACKAGED_MOCK_MODULE:
                self._write_mock_file()
            elif action == _ModuleProviderAction.SKIP:
                continue
            else:
                raise AssertionError(
                    f"Invalid action: {module_name}, {action}. Please report a bug to PyTorch."
                )

        extern_file_contents = "\n".join(extern_modules) + "\n"
        self._write(".data/extern_modules", extern_file_contents)

    def _write_python_version(self):
        """Writes the python version that the package was created with to .data/python_version"""
        self._write(".data/python_version", platform.python_version())

    def close(self):
        """Write the package to the filesystem. Any calls after :meth:`close` are now invalid.
        It is preferable to use resource guard syntax instead::

            with PackageExporter("file.zip") as e:
                ...
        """
        self._execute_dependency_graph()
        self._write_python_version()

        self.script_module_serializer.write_files()
        self._finalize_zip()

    def _finalize_zip(self):
        """Called at the very end of packaging to leave the zipfile in a closed but valid state."""
        del self.zip_file
        if self.buffer:
            self.buffer.flush()

    def _filename(self, package, resource):
        package_path = package.replace(".", "/")
        resource = _normalize_path(resource)
        return f"{package_path}/{resource}"

    def _can_implicitly_extern(self, module_name: str):
        top_level_package_name = module_name.partition(".")[0]
        return top_level_package_name == "torch" or (
            top_level_package_name not in _DISALLOWED_MODULES
            and is_stdlib_module(top_level_package_name)
        )

    def dependency_graph_string(self) -> str:
        """Returns digraph string representation of dependencies in package.

        Returns:
            A string representation of dependencies in package.
        """
        return self.dependency_graph.to_dot()

    def _nodes_with_action_type(
        self, action: _ModuleProviderAction | None
    ) -> list[str]:
        result = []
        for name, node_dict in self.dependency_graph.nodes.items():
            node_action = node_dict.get("action", None)
            if node_action == action and "is_pickle" not in node_dict:
                result.append(name)
        result.sort()
        return result

    def externed_modules(self) -> list[str]:
        """Return all modules that are currently externed.

        Returns:
            A list containing the names of modules which will be
            externed in this package.
        """
        return self._nodes_with_action_type(_ModuleProviderAction.EXTERN)

    def interned_modules(self) -> list[str]:
        """Return all modules that are currently interned.

        Returns:
            A list containing the names of modules which will be
            interned in this package.
        """
        return self._nodes_with_action_type(_ModuleProviderAction.INTERN)

    def mocked_modules(self) -> list[str]:
        """Return all modules that are currently mocked.

        Returns:
            A list containing the names of modules which will be
            mocked in this package.
        """
        return self._nodes_with_action_type(_ModuleProviderAction.MOCK)

    def denied_modules(self) -> list[str]:
        """Return all modules that are currently denied.

        Returns:
            A list containing the names of modules which will be
            denied in this package.
        """
        return self._nodes_with_action_type(_ModuleProviderAction.DENY)

    def get_rdeps(self, module_name: str) -> list[str]:
        """Return a list of all modules which depend on the module ``module_name``.

        Returns:
            A list containing the names of modules which depend on ``module_name``.
        """
        if module_name in self.dependency_graph._pred:
            return list(self.dependency_graph._pred[module_name].keys())
        else:
            return []

    def all_paths(self, src: str, dst: str) -> str:
        """Return a dot representation of the subgraph
           that has all paths from src to dst.

        Returns:
            A dot representation containing all paths from src to dst.
            (https://graphviz.org/doc/info/lang.html)
        """
        return self.dependency_graph.all_paths(src, dst)


# even though these are in the standard library, we do not allow them to be
# automatically externed since they offer a lot of system level access
_DISALLOWED_MODULES = ["sys", "io"]

_MOCK_IMPL = """\
from _mock import MockedObject
def __getattr__(attr: str):
    return MockedObject(__name__ + '.' + attr, _suppress_err=True)
"""


def _read_file(filename: str) -> str:
    with open(filename, "rb") as f:
        b = f.read()
        return b.decode("utf-8")
<|endoftext|># mypy: allow-untyped-defs
import ast

from ._importlib import _resolve_name


class _ExtractModuleReferences(ast.NodeVisitor):
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    def run(cls, src: str, package: str) -> list[tuple[str, str | None]]:
        visitor = cls(package)
        tree = ast.parse(src)
        visitor.visit(tree)
        return list(visitor.references.keys())

    def __init__(self, package):
        super().__init__()
        self.package = package
        self.references = {}

    def _absmodule(self, module_name: str, level: int) -> str:
        if level > 0:
            return _resolve_name(module_name, self.package, level)
        return module_name

    def visit_Import(self, node):
        for alias in node.names:
            self.references[(alias.name, None)] = True

    def visit_ImportFrom(self, node):
        name = self._absmodule(node.module, 0 if node.level is None else node.level)
        for alias in node.names:
            # from my_package import foo
            # foo may be a module, so we have to add it to the list of
            # potential references, if import of it fails, we will ignore it
            if alias.name != "*":
                self.references[(name, alias.name)] = True
            else:
                self.references[(name, None)] = True

    def _grab_node_int(self, node):
        return node.value

    def _grab_node_str(self, node):
        return node.value

    def visit_Call(self, node):
        # __import__ calls aren't routed to the visit_Import/From nodes
        if hasattr(node.func, "id") and node.func.id == "__import__":
            try:
                name = self._grab_node_str(node.args[0])
                fromlist: list[str] = []
                level = 0
                if len(node.args) > 3:
                    fromlist.extend(self._grab_node_str(v) for v in node.args[3].elts)
                elif hasattr(node, "keywords"):
                    for keyword in node.keywords:
                        if keyword.arg == "fromlist":
                            fromlist.extend(
                                self._grab_node_str(v) for v in keyword.value.elts
                            )
                if len(node.args) > 4:
                    level = self._grab_node_int(node.args[4])
                elif hasattr(node, "keywords"):
                    for keyword in node.keywords:
                        if keyword.arg == "level":
                            level = self._grab_node_int(keyword.value)
                if fromlist == []:
                    # the top-level package (the name up till the first dot) is returned
                    # when the fromlist argument is empty in normal import system,
                    # we need to include top level package to match this behavior and last
                    # level package to capture the intended dependency of user
                    self.references[(name, None)] = True
                    top_name = name.rsplit(".", maxsplit=1)[0]
                    if top_name != name:
                        top_name = self._absmodule(top_name, level)
                        self.references[(top_name, None)] = True
                else:
                    name = self._absmodule(name, level)
                    for alias in fromlist:
                        # fromlist args may be submodules, so we have to add the fromlist args
                        # to the list of potential references. If import of an arg fails we
                        # will ignore it, similar to visit_ImportFrom
                        if alias != "*":
                            self.references[(name, alias)] = True
                        else:
                            self.references[(name, None)] = True
            except Exception:
                return


find_files_source_depends_on = _ExtractModuleReferences.run
<|endoftext|># mypy: allow-untyped-defs

from .glob_group import GlobGroup, GlobPattern


__all__ = ["Directory"]


class Directory:
    """A file structure representation. Organized as Directory nodes that have lists of
    their Directory children. Directories for a package are created by calling
    :meth:`PackageImporter.file_structure`."""

    def __init__(self, name: str, is_dir: bool):
        self.name = name
        self.is_dir = is_dir
        self.children: dict[str, Directory] = {}

    def _get_dir(self, dirs: list[str]) -> "Directory":
        """Builds path of Directories if not yet built and returns last directory
        in list.

        Args:
            dirs (List[str]): List of directory names that are treated like a path.

        Returns:
            :class:`Directory`: The last Directory specified in the dirs list.
        """
        if len(dirs) == 0:
            return self
        dir_name = dirs[0]
        if dir_name not in self.children:
            self.children[dir_name] = Directory(dir_name, True)
        return self.children[dir_name]._get_dir(dirs[1:])

    def _add_file(self, file_path: str):
        """Adds a file to a Directory.

        Args:
            file_path (str): Path of file to add. Last element is added as a file while
                other paths items are added as directories.
        """
        *dirs, file = file_path.split("/")
        dir = self._get_dir(dirs)
        dir.children[file] = Directory(file, False)

    def has_file(self, filename: str) -> bool:
        """Checks if a file is present in a :class:`Directory`.

        Args:
            filename (str): Path of file to search for.
        Returns:
            bool: If a :class:`Directory` contains the specified file.
        """
        lineage = filename.split("/", maxsplit=1)
        child = lineage[0]
        grandchildren = lineage[1] if len(lineage) > 1 else None
        if child in self.children:
            if grandchildren is None:
                return True
            else:
                return self.children[child].has_file(grandchildren)
        return False

    def __str__(self):
        str_list: list[str] = []
        self._stringify_tree(str_list)
        return "".join(str_list)

    def _stringify_tree(
        self,
        str_list: list[str],
        preamble: str = "",
        dir_ptr: str = "\u2500\u2500\u2500 ",
    ):
        """Recursive method to generate print-friendly version of a Directory."""
        space = "    "
        branch = "\u2502   "
        tee = "\u251c\u2500\u2500 "
        last = "\u2514\u2500\u2500 "

        # add this directory's representation
        str_list.append(f"{preamble}{dir_ptr}{self.name}\n")

        # add directory's children representations
        if dir_ptr == tee:
            preamble = preamble + branch
        else:
            preamble = preamble + space

        file_keys: list[str] = []
        dir_keys: list[str] = []
        for key, val in self.children.items():
            if val.is_dir:
                dir_keys.append(key)
            else:
                file_keys.append(key)

        for index, key in enumerate(sorted(dir_keys)):
            if (index == len(dir_keys) - 1) and len(file_keys) == 0:
                self.children[key]._stringify_tree(str_list, preamble, last)
            else:
                self.children[key]._stringify_tree(str_list, preamble, tee)
        for index, file in enumerate(sorted(file_keys)):
            pointer = last if (index == len(file_keys) - 1) else tee
            str_list.append(f"{preamble}{pointer}{file}\n")


def _create_directory_from_file_list(
    filename: str,
    file_list: list[str],
    include: "GlobPattern" = "**",
    exclude: "GlobPattern" = (),
) -> Directory:
    """Return a :class:`Directory` file structure representation created from a list of files.

    Args:
        filename (str): The name given to the top-level directory that will be the
            relative root for all file paths found in the file_list.

        file_list (List[str]): List of files to add to the top-level directory.

        include (list[str] | str): An optional pattern that limits what is included from the file_list to
            files whose name matches the pattern.

        exclude (list[str] | str): An optional pattern that excludes files whose name match the pattern.

    Returns:
            :class:`Directory`: a :class:`Directory` file structure representation created from a list of files.
    """
    glob_pattern = GlobGroup(include, exclude=exclude, separator="/")

    top_dir = Directory(filename, True)
    for file in file_list:
        if glob_pattern.matches(file):
            top_dir._add_file(file)
    return top_dir
<|endoftext|># mypy: allow-untyped-defs
"""Import mangling.
See mangling.md for details.
"""

import re


_mangle_index = 0


class PackageMangler:
    """
    Used on import, to ensure that all modules imported have a shared mangle parent.
    """

    def __init__(self) -> None:
        global _mangle_index
        self._mangle_index = _mangle_index
        # Increment the global index
        _mangle_index += 1
        # Angle brackets are used so that there is almost no chance of
        # confusing this module for a real module. Plus, it is Python's
        # preferred way of denoting special modules.
        self._mangle_parent = f"<torch_package_{self._mangle_index}>"

    def mangle(self, name) -> str:
        if len(name) == 0:
            raise AssertionError("name must not be empty")
        return self._mangle_parent + "." + name

    def demangle(self, mangled: str) -> str:
        """
        Note: This only demangles names that were mangled by this specific
        PackageMangler. It will pass through names created by a different
        PackageMangler instance.
        """
        if mangled.startswith(self._mangle_parent + "."):
            return mangled.partition(".")[2]

        # wasn't a mangled name
        return mangled

    def parent_name(self):
        return self._mangle_parent


def is_mangled(name: str) -> bool:
    return bool(re.match(r"<torch_package_\d+>", name))


def demangle(name: str) -> str:
    """
    Note: Unlike PackageMangler.demangle, this version works on any
    mangled name, irrespective of which PackageMangler created it.
    """
    if is_mangled(name):
        _first, sep, last = name.partition(".")
        # If there is only a base mangle prefix, e.g. '<torch_package_0>',
        # then return an empty string.
        return last if len(sep) != 0 else ""
    return name


def get_mangle_prefix(name: str) -> str:
    return name.partition(".")[0] if is_mangled(name) else name
<|endoftext|>from .analyze.is_from_package import is_from_package
from .file_structure_representation import Directory
from .glob_group import GlobGroup
from .importer import (
    Importer,
    ObjMismatchError,
    ObjNotFoundError,
    OrderedImporter,
    sys_importer,
)
from .package_exporter import EmptyMatchError, PackageExporter, PackagingError
from .package_importer import PackageImporter
<|endoftext|># mypy: allow-untyped-defs
_magic_methods = [
    "__subclasscheck__",
    "__hex__",
    "__rmul__",
    "__float__",
    "__idiv__",
    "__setattr__",
    "__div__",
    "__invert__",
    "__nonzero__",
    "__rshift__",
    "__eq__",
    "__pos__",
    "__round__",
    "__rand__",
    "__or__",
    "__complex__",
    "__divmod__",
    "__len__",
    "__reversed__",
    "__copy__",
    "__reduce__",
    "__deepcopy__",
    "__rdivmod__",
    "__rrshift__",
    "__ifloordiv__",
    "__hash__",
    "__iand__",
    "__xor__",
    "__isub__",
    "__oct__",
    "__ceil__",
    "__imod__",
    "__add__",
    "__truediv__",
    "__unicode__",
    "__le__",
    "__delitem__",
    "__sizeof__",
    "__sub__",
    "__ne__",
    "__pow__",
    "__bytes__",
    "__mul__",
    "__itruediv__",
    "__bool__",
    "__iter__",
    "__abs__",
    "__gt__",
    "__iadd__",
    "__enter__",
    "__floordiv__",
    "__call__",
    "__neg__",
    "__and__",
    "__ixor__",
    "__getitem__",
    "__exit__",
    "__cmp__",
    "__getstate__",
    "__index__",
    "__contains__",
    "__floor__",
    "__lt__",
    "__getattr__",
    "__mod__",
    "__trunc__",
    "__delattr__",
    "__instancecheck__",
    "__setitem__",
    "__ipow__",
    "__ilshift__",
    "__long__",
    "__irshift__",
    "__imul__",
    "__lshift__",
    "__dir__",
    "__ge__",
    "__int__",
    "__ior__",
]


class MockedObject:
    _name: str

    def __new__(cls, *args, **kwargs):
        # _suppress_err is set by us in the mocked module impl, so that we can
        # construct instances of MockedObject to hand out to people looking up
        # module attributes.

        # Any other attempt to construct a MockedObject instance (say, in the
        # unpickling process) should give an error.
        if not kwargs.get("_suppress_err"):
            raise NotImplementedError(
                f"Object '{cls._name}' was mocked out during packaging "
                f"but it is being used in '__new__'. If this error is "
                "happening during 'load_pickle', please ensure that your "
                "pickled object doesn't contain any mocked objects."
            )
        # Otherwise, this is just a regular object creation
        # (e.g. `x = MockedObject("foo")`), so pass it through normally.
        return super().__new__(cls)

    def __init__(self, name: str, _suppress_err: bool):
        self.__dict__["_name"] = name

    def __repr__(self):
        return f"MockedObject({self._name})"


def install_method(method_name):
    def _not_implemented(self, *args, **kwargs):
        raise NotImplementedError(
            f"Object '{self._name}' was mocked out during packaging but it is being used in {method_name}"
        )

    setattr(MockedObject, method_name, _not_implemented)


for method_name in _magic_methods:
    install_method(method_name)
<|endoftext|># mypy: allow-untyped-defs
"""List of Python standard library modules.

Sadly, there is no reliable way to tell whether a module is part of the
standard library except by comparing to a canonical list.

This is taken from https://github.com/PyCQA/isort/tree/develop/isort/stdlibs,
which itself is sourced from the Python documentation.
"""

import sys


def is_stdlib_module(module: str) -> bool:
    base_module = module.partition(".")[0]
    return base_module in _get_stdlib_modules()


def _get_stdlib_modules():
    return sys.stdlib_module_names
<|endoftext|># mypy: allow-untyped-defs
from collections import deque


class DiGraph:
    """Really simple unweighted directed graph data structure to track dependencies.

    The API is pretty much the same as networkx so if you add something just
    copy their API.
    """

    def __init__(self):
        # Dict of node -> dict of arbitrary attributes
        self._node = {}
        # Nested dict of node -> successor node -> nothing.
        # (didn't implement edge data)
        self._succ = {}
        # Nested dict of node -> predecessor node -> nothing.
        self._pred = {}

        # Keep track of the order in which nodes are added to
        # the graph.
        self._node_order = {}
        self._insertion_idx = 0

    def add_node(self, n, **kwargs):
        """Add a node to the graph.

        Args:
            n: the node. Can be any object that is a valid dict key.
            **kwargs: any attributes you want to attach to the node.
        """
        if n not in self._node:
            self._node[n] = kwargs
            self._succ[n] = {}
            self._pred[n] = {}
            self._node_order[n] = self._insertion_idx
            self._insertion_idx += 1
        else:
            self._node[n].update(kwargs)

    def add_edge(self, u, v):
        """Add an edge to graph between nodes ``u`` and ``v``

        ``u`` and ``v`` will be created if they do not already exist.
        """
        # add nodes
        self.add_node(u)
        self.add_node(v)

        # add the edge
        self._succ[u][v] = True
        self._pred[v][u] = True

    def successors(self, n):
        """Returns an iterator over successor nodes of n."""
        try:
            return iter(self._succ[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    def predecessors(self, n):
        """Returns an iterator over predecessor nodes of n."""
        try:
            return iter(self._pred[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    @property
    def edges(self):
        """Returns an iterator over all edges (u, v) in the graph"""
        for n, successors in self._succ.items():
            for succ in successors:
                yield n, succ

    @property
    def nodes(self):
        """Returns a dictionary of all nodes to their attributes."""
        return self._node

    def __iter__(self):
        """Iterate over the nodes."""
        return iter(self._node)

    def __contains__(self, n):
        """Returns True if ``n`` is a node in the graph, False otherwise."""
        try:
            return n in self._node
        except TypeError:
            return False

    def forward_transitive_closure(self, src: str) -> set[str]:
        """Returns a set of nodes that are reachable from src"""

        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.successors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def backward_transitive_closure(self, src: str) -> set[str]:
        """Returns a set of nodes that are reachable from src in reverse direction"""

        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def all_paths(self, src: str, dst: str):
        """Returns a subgraph rooted at src that shows all the paths to dst."""

        result_graph = DiGraph()
        # First compute forward transitive closure of src (all things reachable from src).
        forward_reachable_from_src = self.forward_transitive_closure(src)

        if dst not in forward_reachable_from_src:
            return result_graph

        # Second walk the reverse dependencies of dst, adding each node to
        # the output graph iff it is also present in forward_reachable_from_src.
        # we don't use backward_transitive_closures for optimization purposes
        working_set = deque(dst)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n in forward_reachable_from_src:
                    result_graph.add_edge(n, cur)
                    # only explore further if its reachable from src
                    working_set.append(n)

        return result_graph.to_dot()

    def first_path(self, dst: str) -> list[str]:
        """Returns a list of nodes that show the first path that resulted in dst being added to the graph."""
        path = []

        while dst:
            path.append(dst)
            candidates = self._pred[dst].keys()
            dst, min_idx = "", None
            for candidate in candidates:
                idx = self._node_order.get(candidate, None)
                if idx is None:
                    break
                if min_idx is None or idx < min_idx:
                    min_idx = idx
                    dst = candidate

        return list(reversed(path))

    def to_dot(self) -> str:
        """Returns the dot representation of the graph.

        Returns:
            A dot representation of the graph.
        """
        edges = "\n".join(f'"{f}" -> "{t}";' for f, t in self.edges)
        return f"""\
digraph G {{
rankdir = LR;
node [shape=box];
{edges}
}}
"""
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import re
from collections.abc import Iterable


GlobPattern = str | Iterable[str]


class GlobGroup:
    """A set of patterns that candidate strings will be matched against.

    A candidate is composed of a list of segments separated by ``separator``, e.g. "foo.bar.baz".

    A pattern contains one or more segments. Segments can be:
        - A literal string (e.g. "foo"), which matches exactly.
        - A string containing a wildcard (e.g. "torch*", or "foo*baz*"). The wildcard matches
          any string, including the empty string.
        - A double wildcard ("**"). This matches against zero or more complete segments.

    Examples:
        ``torch.**``: matches ``torch`` and all its submodules, e.g. ``torch.nn`` and ``torch.nn.functional``.
        ``torch.*``: matches ``torch.nn`` or ``torch.functional``, but not ``torch.nn.functional``.
        ``torch*.**``: matches ``torch``, ``torchvision``, and all their submodules.

    A candidate will match the ``GlobGroup`` if it matches any of the ``include`` patterns and
    none of the ``exclude`` patterns.

    Args:
        include (str | Iterable[str]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will match if it matches *any* include pattern
        exclude (str | Iterable[str]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will be excluded from matching if it matches *any* exclude pattern.
        separator (str): A string that delimits segments in candidates and
            patterns. By default this is "." which corresponds to how modules are
            named in Python. Another common value for this is "/", which is
            the Unix path separator.
    """

    def __init__(
        self, include: GlobPattern, *, exclude: GlobPattern = (), separator: str = "."
    ):
        self._dbg = f"GlobGroup(include={include}, exclude={exclude})"
        self.include = GlobGroup._glob_list(include, separator)
        self.exclude = GlobGroup._glob_list(exclude, separator)
        self.separator = separator

    def __str__(self):
        return self._dbg

    def __repr__(self):
        return self._dbg

    def matches(self, candidate: str) -> bool:
        candidate = self.separator + candidate
        return any(p.fullmatch(candidate) for p in self.include) and all(
            not p.fullmatch(candidate) for p in self.exclude
        )

    @staticmethod
    def _glob_list(elems: GlobPattern, separator: str = "."):
        if isinstance(elems, str):
            return [GlobGroup._glob_to_re(elems, separator)]
        else:
            return [GlobGroup._glob_to_re(e, separator) for e in elems]

    @staticmethod
    def _glob_to_re(pattern: str, separator: str = "."):
        # to avoid corner cases for the first component, we prefix the candidate string
        # with '.' so `import torch` will regex against `.torch`, assuming '.' is the separator
        def component_to_re(component):
            if "**" in component:
                if component == "**":
                    return "(" + re.escape(separator) + "[^" + separator + "]+)*"
                else:
                    raise ValueError("** can only appear as an entire path segment")
            else:
                return re.escape(separator) + ("[^" + separator + "]*").join(
                    re.escape(x) for x in component.split("*")
                )

        result = "".join(component_to_re(c) for c in pattern.split(separator))
        return re.compile(result)
<|endoftext|># mypy: allow-untyped-defs
import _compat_pickle
import pickle

from .importer import Importer


class PackageUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    """Package-aware unpickler.

    This behaves the same as a normal unpickler, except it uses `importer` to
    find any global names that it encounters while unpickling.
    """

    def __init__(self, importer: Importer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._importer = importer

    def find_class(self, module, name):
        # Subclasses may override this.
        if self.proto < 3 and self.fix_imports:  # type: ignore[attr-defined]
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
        mod = self._importer.import_module(module)
        return getattr(mod, name)
<|endoftext|>import importlib
import logging
import sys
from abc import ABC, abstractmethod
from pickle import (
    _getattribute,  # pyrefly: ignore [missing-module-attribute]
    _Pickler,
    whichmodule as _pickle_whichmodule,  # pyrefly: ignore [missing-module-attribute]
)
from types import ModuleType
from typing import Any

from ._mangling import demangle, get_mangle_prefix, is_mangled


__all__ = ["ObjNotFoundError", "ObjMismatchError", "Importer", "OrderedImporter"]
log = logging.getLogger(__name__)


class ObjNotFoundError(Exception):
    """Raised when an importer cannot find an object by searching for its name."""


class ObjMismatchError(Exception):
    """Raised when an importer found a different object with the same name as the user-provided one."""


class Importer(ABC):
    """Represents an environment to import modules from.

    By default, you can figure out what module an object belongs by checking
    __module__ and importing the result using __import__ or importlib.import_module.

    torch.package introduces module importers other than the default one.
    Each PackageImporter introduces a new namespace. Potentially a single
    name (e.g. 'foo.bar') is present in multiple namespaces.

    It supports two main operations:
        import_module: module_name -> module object
        get_name: object -> (parent module name, name of obj within module)

    The guarantee is that following round-trip will succeed or throw an ObjNotFoundError/ObjMisMatchError.
        module_name, obj_name = env.get_name(obj)
        module = env.import_module(module_name)
        obj2 = getattr(module, obj_name)
        assert obj1 is obj2
    """

    modules: dict[str, ModuleType]

    @abstractmethod
    def import_module(self, module_name: str) -> ModuleType:
        """Import `module_name` from this environment.

        The contract is the same as for importlib.import_module.
        """

    def get_name(self, obj: Any, name: str | None = None) -> tuple[str, str]:
        """Given an object, return a name that can be used to retrieve the
        object from this environment.

        Args:
            obj: An object to get the module-environment-relative name for.
            name: If set, use this name instead of looking up __name__ or __qualname__ on `obj`.
                This is only here to match how Pickler handles __reduce__ functions that return a string,
                don't use otherwise.
        Returns:
            A tuple (parent_module_name, attr_name) that can be used to retrieve `obj` from this environment.
            Use it like:
                mod = importer.import_module(parent_module_name)
                obj = getattr(mod, attr_name)

        Raises:
            ObjNotFoundError: we couldn't retrieve `obj by name.
            ObjMisMatchError: we found a different object with the same name as `obj`.
        """
        if name is None and obj and _Pickler.dispatch.get(type(obj)) is None:
            # Honor the string return variant of __reduce__, which will give us
            # a global name to search for in this environment.
            # TODO: I guess we should do copyreg too?
            reduce = getattr(obj, "__reduce__", None)
            if reduce is not None:
                try:
                    rv = reduce()
                    if isinstance(rv, str):
                        name = rv
                except Exception:
                    pass
        if name is None:
            name = getattr(obj, "__qualname__", None)
        if name is None:
            name = obj.__name__

        orig_module_name = self.whichmodule(obj, name)
        # Demangle the module name before importing. If this obj came out of a
        # PackageImporter, `__module__` will be mangled. See mangling.md for
        # details.
        module_name = demangle(orig_module_name)

        # Check that this name will indeed return the correct object
        try:
            module = self.import_module(module_name)
            if sys.version_info >= (3, 14):
                # pickle._getattribute signature changes in 3.14
                # to take iterable and return just one object
                obj2 = _getattribute(module, name.split("."))
            else:
                obj2, _ = _getattribute(module, name)
        except (ImportError, KeyError, AttributeError):
            raise ObjNotFoundError(
                f"{obj} was not found as {module_name}.{name}"
            ) from None

        if obj is obj2:
            return module_name, name

        def get_obj_info(obj):
            if name is None:
                raise AssertionError("name must not be None")
            module_name = self.whichmodule(obj, name)
            is_mangled_ = is_mangled(module_name)
            location = (
                get_mangle_prefix(module_name)
                if is_mangled_
                else "the current Python environment"
            )
            importer_name = (
                f"the importer for {get_mangle_prefix(module_name)}"
                if is_mangled_
                else "'sys_importer'"
            )
            return module_name, location, importer_name

        obj_module_name, obj_location, obj_importer_name = get_obj_info(obj)
        obj2_module_name, obj2_location, obj2_importer_name = get_obj_info(obj2)
        msg = (
            f"\n\nThe object provided is from '{obj_module_name}', "
            f"which is coming from {obj_location}."
            f"\nHowever, when we import '{obj2_module_name}', it's coming from {obj2_location}."
            "\nTo fix this, make sure this 'PackageExporter's importer lists "
            f"{obj_importer_name} before {obj2_importer_name}."
        )
        raise ObjMismatchError(msg)

    def whichmodule(self, obj: Any, name: str) -> str:
        """Find the module name an object belongs to.

        This should be considered internal for end-users, but developers of
        an importer can override it to customize the behavior.

        Taken from pickle.py, but modified to exclude the search into sys.modules
        """
        module_name = getattr(obj, "__module__", None)
        if module_name is not None:
            return module_name

        # Protect the iteration by using a list copy of self.modules against dynamic
        # modules that trigger imports of other modules upon calls to getattr.
        for module_name, module in self.modules.copy().items():
            if (
                module_name == "__main__"
                or module_name == "__mp_main__"  # bpo-42406
                or module is None
            ):
                continue
            try:
                if _getattribute(module, name)[0] is obj:
                    return module_name
            except AttributeError:
                pass

        return "__main__"


class _SysImporter(Importer):
    """An importer that implements the default behavior of Python."""

    def import_module(self, module_name: str):
        return importlib.import_module(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        # In Python 3.14+, pickle.whichmodule tries to import the module,
        # which fails for mangled package names like '<torch_package_0>'.
        # Check __module__ first before calling pickle.whichmodule.
        module_name = getattr(obj, "__module__", None)
        if module_name is not None:
            return module_name
        return _pickle_whichmodule(obj, name)


sys_importer = _SysImporter()


class OrderedImporter(Importer):
    """A compound importer that takes a list of importers and tries them one at a time.

    The first importer in the list that returns a result "wins".
    """

    def __init__(self, *args):
        self._importers: list[Importer] = list(args)

    def _is_torchpackage_dummy(self, module):
        """Returns true iff this module is an empty PackageNode in a torch.package.

        If you intern `a.b` but never use `a` in your code, then `a` will be an
        empty module with no source. This can break cases where we are trying to
        re-package an object after adding a real dependency on `a`, since
        OrderedImporter will resolve `a` to the dummy package and stop there.

        See: https://github.com/pytorch/pytorch/pull/71520#issuecomment-1029603769
        """
        if not getattr(module, "__torch_package__", False):
            return False
        if not hasattr(module, "__path__"):
            return False
        if not hasattr(module, "__file__"):
            return True
        return module.__file__ is None

    def get_name(self, obj: Any, name: str | None = None) -> tuple[str, str]:
        for importer in self._importers:
            try:
                return importer.get_name(obj, name)
            except (ObjNotFoundError, ObjMismatchError) as e:
                warning_message = (
                    f"Tried to call get_name with obj {obj}, "
                    f"and name {name} on {importer} and got {e}"
                )
                log.warning(warning_message)
        raise ObjNotFoundError(
            f"Could not find obj {obj} and name {name} in any of the importers {self._importers}"
        )

    def import_module(self, module_name: str) -> ModuleType:
        last_err = None
        for importer in self._importers:
            if not isinstance(importer, Importer):
                raise TypeError(
                    f"{importer} is not an Importer. "
                    "All importers in OrderedImporter must inherit from Importer."
                )
            try:
                module = importer.import_module(module_name)
                if self._is_torchpackage_dummy(module):
                    continue
                return module
            except ModuleNotFoundError as err:
                last_err = err

        if last_err is not None:
            raise last_err
        else:
            raise ModuleNotFoundError(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        for importer in self._importers:
            module_name = importer.whichmodule(obj, name)
            if module_name != "__main__":
                return module_name

        return "__main__"
<|endoftext|># mypy: allow-untyped-defs
import sys
from collections.abc import Callable, Iterable
from typing import Any


__all__ = ["trace_dependencies"]


def trace_dependencies(
    callable: Callable[[Any], Any], inputs: Iterable[tuple[Any, ...]]
) -> list[str]:
    """Trace the execution of a callable in order to determine which modules it uses.

    Args:
        callable: The callable to execute and trace.
        inputs: The input to use during tracing. The modules used by 'callable' when invoked by each set of inputs
            are union-ed to determine all modules used by the callable for the purposes of packaging.

    Returns: A list of the names of all modules used during callable execution.
    """
    modules_used = set()

    def record_used_modules(frame, event, arg):
        # If the event being profiled is not a Python function
        # call, there is nothing to do.
        if event != "call":
            return

        # This is the name of the function that was called.
        name = frame.f_code.co_name
        module = None

        # Try to determine the name of the module that the function
        # is in:
        #   1) Check the global namespace of the frame.
        #   2) Check the local namespace of the frame.
        #   3) To handle class instance method calls, check
        #       the attribute named 'name' of the object
        #       in the local namespace corresponding to "self".
        if name in frame.f_globals:
            module = frame.f_globals[name].__module__
        elif name in frame.f_locals:
            module = frame.f_locals[name].__module__
        elif "self" in frame.f_locals:
            method = getattr(frame.f_locals["self"], name, None)
            module = method.__module__ if method else None

        # If a module was found, add it to the set of used modules.
        if module:
            modules_used.add(module)

    try:
        # Attach record_used_modules as the profiler function.
        sys.setprofile(record_used_modules)

        # Execute the callable with all inputs.
        for inp in inputs:
            callable(*inp)

    finally:
        # Detach the profiler function.
        sys.setprofile(None)

    return list(modules_used)
<|endoftext|>from torch.package.package_exporter import PackagingError


__all__ = ["find_first_use_of_broken_modules"]


def find_first_use_of_broken_modules(exc: PackagingError) -> dict[str, list[str]]:
    """
    Find all broken modules in a PackagingError, and for each one, return the
    dependency path in which the module was first encountered.

    E.g. broken module m.n.o was added to a dependency graph while processing a.b.c,
    then re-encountered while processing d.e.f. This method would return
    {'m.n.o': ['a', 'b', 'c']}

    Args:
        exc: a PackagingError

    Returns: A dict from broken module names to lists of module names in the path.
    """

    if not isinstance(exc, PackagingError):
        raise AssertionError(
            f"exception must be a PackagingError, got {type(exc).__name__}"
        )
    uses = {}
    broken_module_names = [
        m for m, attr in exc.dependency_graph.nodes.items() if attr.get("error", False)
    ]
    for module_name in broken_module_names:
        path = exc.dependency_graph.first_path(module_name)
        uses[module_name] = path
    return uses
<|endoftext|>from types import ModuleType
from typing import Any

from .._mangling import is_mangled


def is_from_package(obj: Any) -> bool:
    """
    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    """
    if type(obj) is ModuleType:
        return is_mangled(obj.__name__)
    else:
        return is_mangled(type(obj).__module__)
<|endoftext|>from collections.abc import Callable
from typing import TypeAlias


try:
    from fbscribelogger import (  # type: ignore[import-untyped, import-not-found]
        make_scribe_logger,
    )
except ImportError:
    TAtom: TypeAlias = int | float | bool | str
    TField: TypeAlias = TAtom | list[TAtom]
    TLazyField: TypeAlias = TField | Callable[[], TField]

    def make_scribe_logger(name: str, thrift_src: str) -> Callable[..., None]:
        def inner(**kwargs: TLazyField) -> None:
            pass

        return inner


open_source_signpost = make_scribe_logger(
    "TorchOpenSourceSignpost",
    """
struct TorchOpenSourceSignpostLogEntry {

  # The commit SHA that triggered the workflow, e.g., 02a6b1d30f338206a71d0b75bfa09d85fac0028a. Derived from GITHUB_SHA.
  4: optional string commit_sha;

  # Commit date (not author date) of the commit in commit_sha as timestamp, e.g., 1724208105.  Increasing if merge bot is used, though not monotonic; duplicates occur when stack is landed.
  5: optional i64 commit_date;

  # The fully-formed ref of the branch or tag that triggered the workflow run, e.g., refs/pull/133891/merge or refs/heads/main. Derived from GITHUB_REF.
  6: optional string github_ref;

  # Indicates if branch protections or rulesets are configured for the ref that triggered the workflow run. Derived from GITHUB_REF_PROTECTED.
  7: optional bool github_ref_protected;

  # A unique number for each attempt of a particular workflow run in a repository, e.g., 1. Derived from GITHUB_RUN_ATTEMPT.
  8: optional string github_run_attempt;

  # A unique number for each workflow run within a repository, e.g., 19471190684. Derived from GITHUB_RUN_ID.
  9: optional string github_run_id;

  # A unique number for each run of a particular workflow in a repository, e.g., 238742. Derived from GITHUB_RUN_NUMBER.
  10: optional string github_run_number_str;

  # The name of the current job. Derived from JOB_NAME, e.g., linux-jammy-py3.8-gcc11 / test (default, 3, 4, linux.2xlarge).
  11: optional string job_name;

  # The GitHub user who triggered the job.  Derived from GITHUB_TRIGGERING_ACTOR.
  12: optional string github_triggering_actor;
  13: optional string name; # Event name
  14: optional string parameters; # Parameters (JSON data)
  16: optional string subsystem; # Subsystem the event is associated with

  # The unit timestamp in second for the Scuba Time Column override
  17: optional i64 time;

  # The weight of the record according to current sampling rate
  18: optional i64 weight;
}
""",
)
<|endoftext|>"""
Utilities for converting data types into structured JSON for dumping.
"""

import inspect
import os
import traceback
from collections.abc import Sequence
from typing import Any

import torch._logging._internal


INTERN_TABLE: dict[str, int] = {}


DUMPED_FILES: set[str] = set()


def intern_string(s: str | None) -> int:
    if s is None:
        return -1

    r = INTERN_TABLE.get(s)
    if r is None:
        r = len(INTERN_TABLE)
        INTERN_TABLE[s] = r
        torch._logging._internal.trace_structured(
            "str", lambda: (s, r), suppress_context=True
        )
    return r


def dump_file(filename: str) -> None:
    if "eval_with_key" not in filename:
        return
    if filename in DUMPED_FILES:
        return
    DUMPED_FILES.add(filename)
    from torch.fx.graph_module import _loader

    torch._logging._internal.trace_structured(
        "dump_file",
        metadata_fn=lambda: {
            "name": filename,
        },
        payload_fn=lambda: _loader.get_source(filename),
    )


def from_traceback(tb: Sequence[traceback.FrameSummary]) -> list[dict[str, Any]]:
    # dict naming convention here coincides with
    # python/combined_traceback.cpp
    r = [
        {
            "line": frame.lineno,
            "name": frame.name,
            "filename": intern_string(frame.filename),
            "loc": frame.line,
        }
        for frame in tb
    ]
    return r


def get_user_stack(num_frames: int) -> list[dict[str, Any]]:
    from torch._guards import TracingContext
    from torch.utils._traceback import CapturedTraceback

    user_tb = TracingContext.extract_stack()
    if user_tb:
        return from_traceback(user_tb[-1 * num_frames :])

    tb = CapturedTraceback.extract().summary()

    # Filter out frames that are within the torch/ codebase
    torch_filepath = os.path.dirname(inspect.getfile(torch)) + os.path.sep
    for i, frame in enumerate(reversed(tb)):
        if torch_filepath not in frame.filename:
            # Only display `num_frames` frames in the traceback
            filtered_tb = tb[len(tb) - i - num_frames : len(tb) - i]
            return from_traceback(filtered_tb)

    return from_traceback(tb[-1 * num_frames :])


def get_framework_stack(
    num_frames: int = 25, cpp: bool = False
) -> list[dict[str, Any]]:
    """
    Returns the traceback for the user stack and the framework stack
    """
    from torch.fx.experimental.symbolic_shapes import uninteresting_files
    from torch.utils._traceback import CapturedTraceback

    tb = CapturedTraceback.extract(cpp=cpp).summary()
    tb = [
        frame
        for frame in tb
        if (
            (
                frame.filename.endswith(".py")
                and frame.filename not in uninteresting_files()
            )
            or ("at::" in frame.name or "torch::" in frame.name)
        )
    ]

    return from_traceback(tb[-1 * num_frames :])
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import errno
import functools
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import os.path
import pathlib
import pkgutil
import re
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Optional
from typing_extensions import ParamSpec
from weakref import WeakSet

import torch._logging.structured
from torch._guards import CompileId
from torch._utils_internal import log_trace_structured_event
from torch.utils._traceback import CapturedTraceback


_P = ParamSpec("_P")

log = logging.getLogger(__name__)

# This is a synthetic logger which doesn't correspond to an actual logger,
# but handles all of our "tracing" logging, which is structured and doesn't go
# to stderr but always goes to a dedicated log file.  We don't put these
# loggers in the classic module hierarchy, because we don't want a suppression
# of logs to also cause a trace to get suppressed (traces typically are not
# collected, unless we are in prod, in which case they always are collected.)
#
# TODO: Maybe we should allow for some sub-hierarchy so you can control which
# traces you want to collect, for performance reasons.
#
# See https://docs.google.com/document/d/1CX_hJ0PNy9f3R1y8TJrfkSeLkvGjjjLU84BSXgS2AZ8/edit
trace_log = logging.getLogger("torch.__trace")

DEFAULT_LOG_LEVEL = logging.WARNING
LOG_ENV_VAR = "TORCH_LOGS"
LOG_OUT_ENV_VAR = "TORCH_LOGS_OUT"
LOG_FORMAT_ENV_VAR = "TORCH_LOGS_FORMAT"
LOG_TRACE_ID_FILTER = "TORCH_LOGS_TRACE_ID_FILTER"
TRACE_ENV_VAR = "TORCH_TRACE"
DTRACE_ENV_VAR = "TORCH_DTRACE"

LOG_TRACE_HANDLER: Optional["LazyTraceHandler"] = None

GET_DTRACE_STRUCTURED = False

LOG_PREFIX = "dedicated_log_torch_trace_"


@dataclass
class LogRegistry:
    # shorthand name to log qualified name
    # Note: this only contains loggers registered
    # from register_log
    # e.g. "dynamo" -> "torch._dynamo"
    log_alias_to_log_qnames: dict[str, list[str]] = field(default_factory=dict)

    # artifact logger qualified names,
    # this is populated lazily, as calls to getArtifactLogger
    # currently formatted as <module>.__<artifact_name>
    # e.g. "torch._dynamo.convert_frame.__guards"
    artifact_log_qnames: set[str] = field(default_factory=set)

    # child logs of registered logs if specified via open
    # registration by the user (ie placing "torch._dynamo.output_graph" in the env var)
    # these need to be tracked so their levels can be reset properly
    # e.g. "torch._dynamo.output_graph"
    child_log_qnames: set[str] = field(default_factory=set)

    # artifact names, populated by register_artifact
    # e.g. "guards"
    artifact_names: set[str] = field(default_factory=set)

    # Artifacts that should be visible by default in the error message
    visible_artifacts: set[str] = field(default_factory=set)

    # A short description of each artifact
    artifact_descriptions: dict[str, str] = field(default_factory=dict)

    # artifacts which are not displayed unless explicitly named in the
    # settings. Ex. output_code is NOT displayed even if the inductor
    # log level is set to DEBUG. It must be explicitly named in the settings
    off_by_default_artifact_names: set[str] = field(default_factory=set)

    # logging format string for artifacts
    artifact_log_formatters: dict[str, logging.Formatter] = field(default_factory=dict)

    def is_artifact(self, name):
        return name in self.artifact_names

    def is_log(self, alias):
        return alias in self.log_alias_to_log_qnames

    # register a log with an alias
    def register_log(self, alias, log_qnames: str | list[str]) -> None:
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        self.log_alias_to_log_qnames[alias] = log_qnames

    # register an artifact name
    def register_artifact_name(
        self, name, description, visible, off_by_default, log_format
    ) -> None:
        self.artifact_names.add(name)
        if visible:
            self.visible_artifacts.add(name)
        self.artifact_descriptions[name] = description

        # if off by default, don't enable it
        # when log_name's log_level is set to DEBUG
        if off_by_default:
            self.off_by_default_artifact_names.add(name)

        if log_format is not None:
            self.artifact_log_formatters[name] = logging.Formatter(log_format)

    # register the qualified name of an artifact log
    # this is needed to know which logs need to be reset
    # whenever the log_state is changed
    def register_artifact_log(self, artifact_log_qname) -> None:
        self.artifact_log_qnames.add(artifact_log_qname)

    def register_child_log(self, log_qname) -> None:
        self.child_log_qnames.add(log_qname)

    # flattens all the qnames together (TODO: consider memoizing?)
    def get_log_qnames(self) -> set[str]:
        return set(itertools.chain.from_iterable(self.log_alias_to_log_qnames.values()))

    def get_artifact_log_qnames(self):
        return set(self.artifact_log_qnames)

    def get_child_log_qnames(self):
        return set(self.child_log_qnames)

    def is_off_by_default(self, artifact_qname):
        return artifact_qname in self.off_by_default_artifact_names


@dataclass
class LogState:
    # qualified log names -> currently set log level
    log_qname_to_level: dict[str, str] = field(default_factory=dict)

    # the set of currently enabled artifacts
    artifact_names: set[str] = field(default_factory=set)

    def enable_artifact(self, artifact_name) -> None:
        self.artifact_names.add(artifact_name)

    def is_artifact_enabled(self, name):
        return name in self.artifact_names

    def enable_log(self, log_qnames, log_level) -> None:
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        for log_qname in log_qnames:
            self.log_qname_to_level[log_qname] = log_level

    def get_log_level_pairs(self):
        """Returns all qualified module names for which the user requested
        explicit logging settings.

        .. warning:

            This function used to return all loggers, regardless of whether
            or not the user specified them or not; it now only returns logs
            which were explicitly mentioned by the user (and torch, which
            always is implicitly requested when we initialize our logging
            subsystem.)
        """
        return self.log_qname_to_level.items()

    def clear(self) -> None:
        self.log_qname_to_level.clear()
        self.artifact_names.clear()


log_registry = LogRegistry()
log_state = LogState()

# sample usage: torch._logging.set_logs(**torch._logging.DEFAULT_LOGGING)
DEFAULT_LOGGING = {
    "dynamo": logging.INFO,
    "aot": logging.INFO,
    "inductor": logging.INFO,
    "fsdp": logging.INFO,
    "ddp_graphs": True,
    "graph_breaks": True,
    "side_effects": True,
    "guards": True,
    "recompiles": True,
    "dynamic": logging.INFO,
}


def set_logs(
    *,
    all: int | None = None,
    dynamo: int | None = None,
    aot: int | None = None,
    autograd: int | None = None,
    dynamic: int | None = None,
    inductor: int | None = None,
    distributed: int | None = None,
    c10d: int | None = None,
    ddp: int | None = None,
    fsdp: int | None = None,
    dtensor: int | None = None,
    onnx: int | None = None,
    bytecode: bool = False,
    aot_graphs: bool = False,
    aot_joint_graph: bool = False,
    ddp_graphs: bool = False,
    graph: bool = False,
    graph_code: bool = False,
    graph_code_verbose: bool = False,
    graph_breaks: bool = False,
    side_effects: bool = False,
    graph_sizes: bool = False,
    guards: bool = False,
    recompiles: bool = False,
    recompiles_verbose: bool = False,
    trace_source: bool = False,
    trace_call: bool = False,
    trace_bytecode: bool = False,
    output_code: bool = False,
    kernel_code: bool = False,
    schedule: bool = False,
    perf_hints: bool = False,
    pre_grad_graphs: bool = False,
    post_grad_graphs: bool = False,
    ir_pre_fusion: bool = False,
    ir_post_fusion: bool = False,
    onnx_diagnostics: bool = False,
    fusion: bool = False,
    overlap: bool = False,
    export: int | None = None,
    modules: dict[str, int | bool] | None = None,
    cudagraphs: bool = False,
    sym_node: bool = False,
    compiled_autograd: bool = False,
    compiled_autograd_verbose: bool = False,
    cudagraph_static_inputs: bool = False,
    benchmarking: bool = False,
    autotuning: bool = False,
    autotuning_inputs: bool = False,
    incremental: bool = False,
    graph_region_expansion: bool = False,
    inductor_metrics: bool = False,
    hierarchical_compile: bool = False,
    compute_dependencies: bool = False,
    caching: bool = False,
    partitioned_scatter: bool = False,
) -> None:
    """
    Sets the log level for individual components and toggles individual log
    artifact types.

    .. warning:: This feature is a prototype and may have compatibility
        breaking changes in the future.

    .. note:: The ``TORCH_LOGS`` environment variable has complete precedence
        over this function, so if it was set, this function does nothing.

    A component is a set of related features in PyTorch. All of the log
    messages emitted from a given component have their own log levels. If the
    log level of a particular message has priority greater than or equal to its
    component's log level setting, it is emitted. Otherwise, it is suppressed.
    This allows you to, for instance, silence large groups of log messages that
    are not relevant to you and increase verbosity of logs for components that
    are relevant. The expected log level values, ordered from highest to lowest
    priority, are:

        * ``logging.CRITICAL``
        * ``logging.ERROR``
        * ``logging.WARNING``
        * ``logging.INFO``
        * ``logging.DEBUG``
        * ``logging.NOTSET``

    See documentation for the Python ``logging`` module for more information on
    log levels: `<https://docs.python.org/3/library/logging.html#logging-levels>`_

    An artifact is a particular type of log message. Each artifact is assigned
    to a parent component. A component can emit many different kinds of
    artifacts. In general, an artifact is emitted if either its corresponding
    setting in the argument list below is turned on or if its parent component
    is set to a log level less than or equal to the log level of the artifact.

    Keyword args:
        all (:class:`Optional[int]`):
            The default log level for all components. Default: ``logging.WARN``

        dynamo (:class:`Optional[int]`):
            The log level for the TorchDynamo component. Default: ``logging.WARN``

        aot (:class:`Optional[int]`):
            The log level for the AOTAutograd component. Default: ``logging.WARN``

        autograd (:class:`Optional[int]`):
            The log level for autograd. Default: ``logging.WARN``

        inductor (:class:`Optional[int]`):
            The log level for the TorchInductor component. Default: ``logging.WARN``

        dynamic (:class:`Optional[int]`):
            The log level for dynamic shapes. Default: ``logging.WARN``

        distributed (:class:`Optional[int]`):
            Whether to log c10d communication operations and other debug info from PyTorch Distributed components.
            Default: ``logging.WARN``

        c10d (:class:`Optional[int]`):
            Whether to log c10d communication operations related debug info in PyTorch Distributed components.
            Default: ``logging.WARN``

        ddp (:class:`Optional[int]`):
            Whether to log debug info related to ``DistributedDataParallel``(DDP) from PyTorch Distributed components.
            Default: ``logging.WARN``

        fsdp (:class:`Optional[int]`):
            Whether to log debug info related to ``FullyShardedDataParallel``(FSDP) in PyTorch Distributed components.
            Default: ``logging.WARN``

        dtensor (:class:`Optional[int]`):
            Whether to log debug info related to ``DTensor``(DTensor) in PyTorch Distributed components.
            Default: ``logging.WARN``

        onnx (:class:`Optional[int]`):
            The log level for the ONNX exporter component. Default: ``logging.WARN``

        bytecode (:class:`bool`):
            Whether to emit the original and generated bytecode from TorchDynamo.
            Default: ``False``

        aot_graphs (:class:`bool`):
            Whether to emit the graphs generated by AOTAutograd. Default: ``False``

        aot_joint_graph (:class:`bool`):
            Whether to emit the joint forward-backward graph generated by AOTAutograd. Default: ``False``

        ddp_graphs (:class:`bool`):
            Whether to emit graphs generated by DDPOptimizer. Default: ``False``

        graph (:class:`bool`):
            Whether to emit the graph captured by TorchDynamo in tabular format.
            Default: ``False``

        graph_code (:class:`bool`):
            Whether to emit the python source of the graph captured by TorchDynamo.
            Default: ``False``

        graph_code_verbose (:class:`bool`):
            Whether to emit verbose/intermediate FX pass logs for graph code. Default: ``False``

        graph_breaks (:class:`bool`):
            Whether to emit the graph breaks encountered by TorchDynamo.
            Default: ``False``

        side_effects (:class:`bool`):
            Whether to emit side effects (mutations, hooks, etc.) that TorchDynamo
            codegenerates in the output graph. Default: ``False``

        graph_sizes (:class:`bool`):
            Whether to emit tensor sizes of the graph captured by TorchDynamo.
            Default: ``False``

        guards (:class:`bool`):
            Whether to emit the guards generated by TorchDynamo for each compiled
            function. Default: ``False``

        recompiles (:class:`bool`):
            Whether to emit a guard failure reason and message every time
            TorchDynamo recompiles a function. Default: ``False``

        recompiles_verbose (:class:`bool`):
            Whether to emit all guard failure reasons when TorchDynamo recompiles
            a function, even those that are not actually run. Default: ``False``

        trace_source (:class:`bool`):
            Whether to emit when TorchDynamo begins tracing a new line. Default: ``False``

        trace_call (:class:`bool`):
            Whether to emit detailed line location when TorchDynamo creates an FX node
            corresponding to function call. Python 3.11+ only. Default: ``False``

        trace_bytecode (:class:`bool`):
            Whether to emit bytecode instructions and traced stack state as TorchDynamo
            traces bytecode. Default: ``False``

        output_code (:class:`bool`):
            Whether to emit the TorchInductor output code on a per-graph basis. Default: ``False``

        kernel_code (:class:`bool`):
            Whether to emit the TorchInductor output code on a per-kernel basis. Default: ``False``

        schedule (:class:`bool`):
            Whether to emit the TorchInductor schedule. Default: ``False``

        perf_hints (:class:`bool`):
            Whether to emit the TorchInductor perf hints. Default: ``False``

        pre_grad_graphs (:class:`bool`):
            Whether to emit the graphs before inductor grad passes. Default: ``False``

        post_grad_graphs (:class:`bool`):
            Whether to emit the graphs generated by after post grad passes. Default: ``False``

        ir_pre_fusion (:class:`bool`):
            Whether to emit the graphs before inductor fusion passes. Default: ``False``

        ir_post_fusion (:class:`bool`):
            Whether to emit the graphs after inductor fusion passes. Default: ``False``

        onnx_diagnostics (:class:`bool`):
            Whether to emit the ONNX exporter diagnostics in logging. Default: ``False``

        fusion (:class:`bool`):
            Whether to emit detailed Inductor fusion decisions. Default: ``False``

        overlap (:class:`bool`):
            Whether to emit detailed Inductor compute/comm overlap decisions. Default: ``False``

        sym_node (:class:`bool`):
            Whether to emit debug info for various SymNode operations. Default: ``False``

        export (:class:`Optional[int]`):
            The log level for export. Default: ``logging.WARN``

        benchmarking (:class:`bool`):
            Whether to emit detailed Inductor benchmarking information. Default: ``False``

        modules (dict):
            This argument provides an alternate way to specify the above log
            component and artifact settings, in the format of a keyword args
            dictionary given as a single argument. There are two cases
            where this is useful (1) if a new log component or artifact has
            been registered but a keyword argument for it has not been added
            to this function and (2) if the log level for an unregistered module
            needs to be set. This can be done by providing the fully-qualified module
            name as the key, with the log level as the value. Default: ``None``

        cudagraph_static_inputs (:class:`bool`):
            Whether to emit debug info for cudagraph static input detection. Default: ``False``

        autotuning (:class:`bool`):
            Autotuning choice logs, such as kernel source, perf, and tuning parameters. Default: ``False``

        autotuning_inputs (:class:`bool`):
            Per-kernel input tensor shapes/dtypes/strides logged during autotuning. Default: ``False``

        incremental (:class:`bool`):
            Incremental autotuning logs. Default: ``False``

        graph_region_expansion (:class:`bool`):
            Whether to emit the detailed steps of the duplicate graph region tracker expansion algorithm. Default: ``False``

        inductor_metrics (:class:`bool`):
            Whether to estimate the runtimes of the nodes in a graph and log them to the metrics table. Default: ``False``

        hierarchical_compile (:class:`bool`):
            Whether to emit debug info for hierarchical compilation. Default: ``False``

        caching (:class:`bool`):
            Whether to emit detailed Inductor caching information. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> import logging

        # The following changes the "dynamo" component to emit DEBUG-level
        # logs, and to emit "graph_code" artifacts.

        >>> torch._logging.set_logs(dynamo=logging.DEBUG, graph_code=True)

        # The following enables the logs for a different module

        >>> torch._logging.set_logs(modules={"unregistered.module.name": logging.DEBUG})
    """
    # ignore if env var is set
    if LOG_ENV_VAR in os.environ:
        log.warning(
            "Using TORCH_LOGS environment variable for log settings, ignoring call to set_logs"
        )
        return

    log_state.clear()

    modules = modules or {}

    def _set_logs(**kwargs) -> None:
        for alias, val in itertools.chain(kwargs.items(), modules.items()):  # type: ignore[union-attr]
            if val is None:
                continue

            if log_registry.is_artifact(alias):
                if not isinstance(val, bool):
                    raise ValueError(
                        f"Expected bool to enable artifact {alias}, received {val}"
                    )

                if val:
                    log_state.enable_artifact(alias)
            elif log_registry.is_log(alias) or alias in log_registry.child_log_qnames:
                if val not in logging._levelToName:
                    raise ValueError(
                        f"Unrecognized log level for log {alias}: {val}, valid level values "
                        f"are: {','.join([str(k) for k in logging._levelToName])}"
                    )

                log_state.enable_log(
                    log_registry.log_alias_to_log_qnames.get(alias, alias), val
                )
            elif _is_valid_module(alias):
                found_modules = _get_module_and_submodules(alias) or (alias,)
                for module_name in found_modules:
                    if not _has_registered_parent(module_name):
                        log_registry.register_log(module_name, module_name)
                    else:
                        log_registry.register_child_log(module_name)
                    log_state.enable_log(
                        log_registry.log_alias_to_log_qnames.get(
                            module_name, module_name
                        ),
                        val,
                    )
            else:
                raise ValueError(
                    f"Unrecognized log or artifact name passed to set_logs: {alias}"
                )

        _init_logs()

    _set_logs(
        torch=all,
        dynamo=dynamo,
        aot=aot,
        autograd=autograd,
        inductor=inductor,
        dynamic=dynamic,
        bytecode=bytecode,
        aot_graphs=aot_graphs,
        aot_joint_graph=aot_joint_graph,
        ddp_graphs=ddp_graphs,
        distributed=distributed,
        c10d=c10d,
        ddp=ddp,
        fsdp=fsdp,
        dtensor=dtensor,
        graph=graph,
        graph_code=graph_code,
        graph_code_verbose=graph_code_verbose,
        graph_breaks=graph_breaks,
        side_effects=side_effects,
        graph_sizes=graph_sizes,
        guards=guards,
        recompiles=recompiles,
        recompiles_verbose=recompiles_verbose,
        trace_source=trace_source,
        trace_call=trace_call,
        trace_bytecode=trace_bytecode,
        output_code=output_code,
        kernel_code=kernel_code,
        schedule=schedule,
        perf_hints=perf_hints,
        pre_grad_graphs=pre_grad_graphs,
        post_grad_graphs=post_grad_graphs,
        ir_pre_fusion=ir_pre_fusion,
        ir_post_fusion=ir_post_fusion,
        onnx=onnx,
        onnx_diagnostics=onnx_diagnostics,
        fusion=fusion,
        overlap=overlap,
        sym_node=sym_node,
        export=export,
        cudagraphs=cudagraphs,
        compiled_autograd=compiled_autograd,
        compiled_autograd_verbose=compiled_autograd_verbose,
        cudagraph_static_inputs=cudagraph_static_inputs,
        benchmarking=benchmarking,
        autotuning=autotuning,
        autotuning_inputs=autotuning_inputs,
        incremental=incremental,
        graph_region_expansion=graph_region_expansion,
        inductor_metrics=inductor_metrics,
        hierarchical_compile=hierarchical_compile,
        compute_dependencies=compute_dependencies,
        caching=caching,
        partitioned_scatter=partitioned_scatter,
    )


def get_loggers() -> list[logging.Logger]:
    """
    Returns: a list of all registered loggers
    """
    return [logging.getLogger(qname) for qname in log_registry.get_log_qnames()]


def register_log(setting_name, log_name) -> None:
    """
    Enables a log to be controlled by the env var and user API with the setting_name
    Args:
        setting_name:  the shorthand name used in the env var and user API
        log_name:  the log name that the setting_name is associated with
    """
    log_registry.register_log(setting_name, log_name)


def register_artifact(
    setting_name, description, visible=False, off_by_default=False, log_format=None
) -> None:
    """
    Enables an artifact to be controlled by the env var and user API with name
    Args:
        setting_name: the shorthand name used in the env var and user API
        description: A description of what this outputs
        visible: Whether it gets suggested to users by default
        off_by_default: whether this artifact should be logged when the ancestor loggers
            are enabled at level DEBUG
    """
    log_registry.register_artifact_name(
        setting_name, description, visible, off_by_default, log_format
    )


def getArtifactLogger(module_qname, artifact_name) -> logging.Logger:
    if artifact_name not in log_registry.artifact_names:
        raise ValueError(
            f"Artifact name: {repr(artifact_name)} not registered,"
            f"please call register_artifact({repr(artifact_name)}) in torch._logging.registrations."
        )
    qname = module_qname + f".__{artifact_name}"
    log = logging.getLogger(qname)
    log.artifact_name = artifact_name  # type: ignore[attr-defined]
    log_registry.register_artifact_log(qname)
    configure_artifact_log(log)
    return log


INCR_VERBOSITY_CHAR = "+"
DECR_VERBOSITY_CHAR = "-"
VERBOSITY_REGEX = (
    "("
    + "|".join([re.escape(INCR_VERBOSITY_CHAR), re.escape(DECR_VERBOSITY_CHAR)])
    + "?)"
)


def configure_artifact_log(log) -> None:
    # If the artifact is off by default, then it should only be logged when explicitly
    # enabled; set propagate to False so that this artifact is not propagated
    # to its ancestor logger
    if log_registry.is_off_by_default(log.artifact_name):
        log.propagate = False

    # enable artifact logging when explicitly enabled
    if log_state.is_artifact_enabled(log.artifact_name):
        log.setLevel(logging.DEBUG)
        log.propagate = True


# match a comma separated list of loggable names (whitespace allowed after commas)
def _gen_settings_regex():
    return re.compile(r"((\+|-)?[\w\.]+,\s*)*(\+|-)?[\w\.]+?")


def _validate_settings(settings):
    return re.fullmatch(_gen_settings_regex(), settings) is not None


def help_message(verbose=False):
    def pad_to(s, length=30):
        if len(s) > length:
            raise AssertionError(f"string length {len(s)} exceeds max {length}")
        return s + " " * (length - len(s))

    if verbose:
        printed_artifacts = log_registry.artifact_names
    else:
        printed_artifacts = log_registry.visible_artifacts
    if verbose:
        heading = "All registered names"
    else:
        heading = "Visible registered names (use TORCH_LOGS='+help' for full list)"
    lines = (
        ["all"]
        + sorted(log_registry.log_alias_to_log_qnames.keys())
        + sorted(
            [
                f"{pad_to(name)}\t{log_registry.artifact_descriptions[name]}"
                for name in printed_artifacts
            ]
        )
    )
    setting_info = "  " + "\n  ".join(lines)
    examples = """
Examples:
  TORCH_LOGS="+dynamo,aot" will set the log level of TorchDynamo to
  logging.DEBUG and AOT to logging.INFO

  TORCH_LOGS="-dynamo,+inductor" will set the log level of TorchDynamo to
  logging.ERROR and TorchInductor to logging.DEBUG

  TORCH_LOGS="aot_graphs" will enable the aot_graphs artifact

  TORCH_LOGS="+dynamo,schedule" will enable set the log level of TorchDynamo
  to logging.DEBUG and enable the schedule artifact

  TORCH_LOGS="+some.random.module,schedule" will set the log level of
  some.random.module to logging.DEBUG and enable the schedule artifact

  TORCH_LOGS="+torch._functorch._aot_autograd" will set the log level of
  torch._functorch._aot_autograd and all its submodules to logging.DEBUG
  (directory-based logging)

  TORCH_LOGS_FORMAT="%(levelname)s: %(message)s" or any provided format
  string will set the output format
  Valid keys are "levelname", "message", "pathname", "levelno", "lineno",
  "filename" and "name".

  TORCH_LOGS_OUT=/tmp/output.txt will output the logs to /tmp/output.txt as
  well. This is useful when the output is long.
"""
    msg = f"""
TORCH_LOGS Info
{examples}

{heading}
{setting_info}
"""
    return msg


def _invalid_settings_err_msg(settings, verbose=False):
    valid_settings = (
        ["all"]
        + list(log_registry.log_alias_to_log_qnames.keys())
        + list(log_registry.artifact_names)
    )
    valid_settings = ", ".join(sorted(valid_settings))
    msg = f"""
Invalid log settings: {settings}, must be a comma separated list of fully
qualified module names, registered log names or registered artifact names.
For more info on various settings, try TORCH_LOGS="help"
Valid settings:
{valid_settings}
"""
    return msg


def process_env_var_string_for_windows(env_var_str: str) -> str:
    """
    When we setup logging config as guide: https://docs.pytorch.org/docs/stable/logging.html
    Such as:
        TORCH_LOGS="+schedule,+inductor,+output_code"

    On Linux, it shows as:
        declare -x SSH_TTY="/dev/pts/0"
        declare -x TERM="xterm"
        declare -x TORCH_LOGS="+schedule,+inductor,+output_code"
        declare -x USER="xu"

    On Windows, it shows as:
        TORCHINDUCTOR_WINDOWS_TESTS=1
        TORCH_LOGS="+schedule,+inductor,+output_code"
        UCRTVersion=10.0.22000.0

    For Linux, it shows quotes by default, And Windows is not shows quotes.
    Besides that, Windows would auto assemble quotes when env var processing.
    On Linux, we will get variable: "+schedule,+inductor,+output_code"
    On Windows, we will get variable: '"+schedule,+inductor,+output_code"'

    So, we need remove the outer quotes for Windows.
    """
    _IS_WINDOWS = sys.platform == "win32"

    def remove_outer_quotes(s: str) -> str:
        if len(s) >= 2 and (
            (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
        ):
            return s[1:-1]
        return s

    if _IS_WINDOWS:
        env_var_str = remove_outer_quotes(env_var_str)

    return env_var_str


@functools.lru_cache
def _parse_log_settings(settings):
    settings = process_env_var_string_for_windows(settings)

    if settings == "":
        return {}

    if settings == "help":
        raise ValueError(help_message(verbose=False))
    elif settings == "+help":
        raise ValueError(help_message(verbose=True))
    if not _validate_settings(settings):
        raise ValueError(_invalid_settings_err_msg(settings))

    settings = re.sub(r"\s+", "", settings)
    log_names = settings.split(",")

    def get_name_level_pair(name):
        clean_name = name.replace(INCR_VERBOSITY_CHAR, "")
        clean_name = clean_name.replace(DECR_VERBOSITY_CHAR, "")

        if name[0] == INCR_VERBOSITY_CHAR:
            level = logging.DEBUG
        elif name[0] == DECR_VERBOSITY_CHAR:
            level = logging.ERROR
        else:
            level = logging.INFO

        return clean_name, level

    log_state = LogState()

    for name in log_names:
        name, level = get_name_level_pair(name)

        if name == "all":
            name = "torch"

        if log_registry.is_log(name):
            if level is None:
                raise AssertionError("level must not be None for log name")
            log_qnames = log_registry.log_alias_to_log_qnames[name]
            log_state.enable_log(log_qnames, level)
        elif log_registry.is_artifact(name):
            log_state.enable_artifact(name)
        elif _is_valid_module(name):
            # Get the module and all its submodules if it's a package
            found_modules = _get_module_and_submodules(name) or (name,)
            for module_name in found_modules:
                if not _has_registered_parent(module_name):
                    log_registry.register_log(module_name, module_name)
                else:
                    log_registry.register_child_log(module_name)
                log_state.enable_log(module_name, level)
        else:
            raise ValueError(_invalid_settings_err_msg(settings))

    return log_state


def _is_valid_module(qname):
    spec = importlib.util.find_spec(qname)
    return spec is not None


def _get_module_and_submodules(qname: str) -> Sequence[str] | None:
    """
    Get a module and all its submodules (recursively).

    If qname is a package, this returns a list of all modules and submodules.
    If qname is a simple module, this returns a list containing just that module.

    Args:
        qname: The fully qualified module name

    Returns:
        A list of fully qualified module names, or None if the module doesn't exist
    """
    spec = importlib.util.find_spec(qname)
    if spec is None:
        return None

    modules = [qname]

    if spec.submodule_search_locations is not None:
        package = importlib.import_module(qname)
        if hasattr(package, "__path__"):
            for importer, modname, ispkg in pkgutil.walk_packages(
                path=package.__path__,
                prefix=qname + ".",
                onerror=lambda x: None,
            ):
                modules.append(modname)

    return modules


def _update_log_state_from_env() -> None:
    global log_state
    log_setting = os.environ.get(LOG_ENV_VAR, None)
    if log_setting is not None:
        log_state = _parse_log_settings(log_setting)


def _has_registered_parent(log_qname) -> bool:
    cur_log = logging.getLogger(log_qname)

    registered_log_qnames = log_registry.get_log_qnames()

    while cur_log.parent:
        if cur_log.name in registered_log_qnames:
            return True
        cur_log = cur_log.parent

    return False


@functools.lru_cache
def _make_module_path_relative(abs_path: str, sys_path: tuple[str, ...]) -> str:
    """
    The string manipulation here is fairly expensive and we expect very high
    cache hit rates. Because `sys.path` changes very infrequently it's most
    performant to convert it into a tuple of strings and use that for the cache
    key because python has dedicated fast paths for list to tuple conversion
    and tuple hashing. (Empirically, a single top-level cache is about 2x faster
    than trying to cache individual parts.)
    """

    abs_path_resolved = pathlib.Path(abs_path).resolve()

    for path in sys_path:
        try:
            rel_path = abs_path_resolved.relative_to(path)
        except ValueError:
            continue
        else:
            return str(rel_path)

    return str(abs_path_resolved)


def make_module_path_relative(abs_path: str) -> str:
    """
    Given an absolute filepath corresponding to a Python module which was
    loaded via normal import mechanisms using sys.path, convert it into
    a relative path relative to one of the Python search paths.
    """
    return _make_module_path_relative(abs_path, tuple(sys.path))


# apply custom formats to artifacts when necessary
class TorchLogsFormatter(logging.Formatter):
    def __init__(
        self, *, trace: bool = False, trace_id_filter: set[str] | None = None
    ) -> None:
        super().__init__()
        self._is_trace = trace
        self._trace_id_filter = trace_id_filter

    def format(self, record):
        artifact_name = getattr(logging.getLogger(record.name), "artifact_name", None)
        if artifact_name is not None:
            artifact_formatter = log_registry.artifact_log_formatters.get(
                artifact_name, None
            )
            if artifact_formatter is not None:
                return artifact_formatter.format(record)

        record.message = record.getMessage()
        record.asctime = self.formatTime(record, "%m%d %H:%M:%S")

        # exception handling - copied from logging.Formatter.format
        s = record.message
        if record.exc_info:
            from torch._dynamo import config

            should_format_exc = config.verbose or artifact_name != "graph_breaks"
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if should_format_exc:
                if not record.exc_text:
                    record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)

        record.rankprefix = ""
        if not self._is_trace and dist.is_available() and dist.is_initialized():
            record.rankprefix = f"[rank{dist.get_rank()}]:"

        record.traceid = ""
        if (
            not self._is_trace
            and (trace_id := torch._guards.CompileContext.current_trace_id())
            is not None
        ):
            record.traceid = f" [{trace_id}]"

        glog_level_to_abbr = {
            "DEBUG": "V",  # V is for VERBOSE in glog
            "INFO": "I",
            "WARNING": "W",
            "ERROR": "E",
            "CRITICAL": "C",
        }

        shortlevel = glog_level_to_abbr.get(record.levelname, record.levelname)

        record.artifactprefix = ""
        if artifact_name is not None:
            record.artifactprefix = f" [__{artifact_name}]"

        filepath = make_module_path_relative(record.pathname)

        if (
            self._trace_id_filter
            and record.traceid.strip() not in self._trace_id_filter
        ):
            return ""

        prefix = (
            f"{record.rankprefix}{shortlevel}{record.asctime}.{int(record.msecs * 1000):06d} {record.process} "
            f"{filepath}:"
            f"{record.lineno}]{record.traceid}{record.artifactprefix}"
        )
        if self._is_trace:
            if s != "":
                raise AssertionError(f"expected empty string for trace, got {s!r}")
            r = f"{prefix} {json.dumps(record.metadata, default=repr)}"
            if record.payload is not None:
                r += "".join(f"\n\t{l}" for l in record.payload.split("\n"))
            return r
        else:
            lines = s.split("\n")
            return "\n".join(f"{prefix} {l}" for l in lines)


def _default_formatter():
    fmt = os.environ.get(LOG_FORMAT_ENV_VAR, None)
    trace_id_filter = {
        item.strip()
        for item in os.environ.get(LOG_TRACE_ID_FILTER, "").split(",")
        if item.strip()
    }
    if fmt is None:
        return TorchLogsFormatter(trace_id_filter=trace_id_filter)
    else:
        if fmt in ("short", "basic"):
            fmt = logging.BASIC_FORMAT
        return logging.Formatter(fmt)


DEFAULT_FORMATTER = _default_formatter()


class _StderrHandler(logging.Handler):
    """
    A logging handler that follows the current sys.stderr.

    Test capture tools can replace and later close sys.stderr before Python
    atexit handlers run.  Some PT2 logs intentionally emit from atexit, so the
    console handler must not retain a stale captured stream.
    """

    terminator = "\n"

    @property
    def stream(self) -> Any:
        return sys.stderr

    def flush(self) -> None:
        self.acquire()
        try:
            stream = self.stream
            if stream and hasattr(stream, "flush"):
                stream.flush()
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _setup_handlers(create_handler_fn, log) -> None:
    debug_handler = _track_handler(create_handler_fn())
    debug_handler.setFormatter(DEFAULT_FORMATTER)
    debug_handler.setLevel(logging.DEBUG)
    log.addHandler(debug_handler)


handlers = WeakSet()  # type: ignore[var-annotated]
_TORCH_HANDLER_MARKER = "_torch_logging_internal_handler"


# mark handlers that we've created
# so we don't modify user handlers
def _track_handler(handler):
    setattr(handler, _TORCH_HANDLER_MARKER, True)
    handlers.add(handler)
    return handler


def _is_torch_handler(handler):
    return handler in handlers or getattr(handler, _TORCH_HANDLER_MARKER, False)


# clears all torch handlers on specified loggers
def _clear_handlers(log, *, close: bool = True) -> None:
    to_remove = [handler for handler in log.handlers if _is_torch_handler(handler)]
    for handler in to_remove:
        log.removeHandler(handler)
        if close:
            handler.close()


def _reset_logs() -> None:
    # reset all registered logs
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        log.setLevel(logging.WARNING)
        log.propagate = False
        _clear_handlers(log)

    # reset all artifact and child logs
    for artifact_log_qname in itertools.chain(
        log_registry.get_artifact_log_qnames(), log_registry.get_child_log_qnames()
    ):
        log = logging.getLogger(artifact_log_qname)
        log.setLevel(logging.NOTSET)
        log.propagate = True

    trace_log.propagate = False
    # LOG_TRACE_HANDLER is a reusable singleton.  Keep its stream lifecycle
    # unchanged when _init_logs temporarily removes it from trace_log.
    _clear_handlers(trace_log, close=False)


def _get_log_state():
    return log_state


def _set_log_state(state) -> None:
    global log_state
    log_state = state


def _init_logs(log_file_name=None) -> None:
    global GET_DTRACE_STRUCTURED

    _reset_logs()
    _update_log_state_from_env()

    out = os.environ.get(LOG_OUT_ENV_VAR, None)
    if out is not None:
        log_file_name = out

    # First, reset all known (registered) loggers to NOTSET, so that they
    # respect their parent log level
    for log_qname in log_registry.get_log_qnames():
        # But not the top level torch level: this defaults to WARNING so
        # that our log messages don't leak to the lower levels
        if log_qname == "torch":
            continue
        log = logging.getLogger(log_qname)
        log.setLevel(logging.NOTSET)

    # Now, for all loggers which the user requested to have non-standard
    # logging behavior, modify their log levels
    for log_qname, level in log_state.get_log_level_pairs():
        log = logging.getLogger(log_qname)
        log.setLevel(level)

    # Finally, setup handlers for all registered loggers
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        _setup_handlers(
            _StderrHandler,
            log,
        )

        if log_file_name is not None:
            _setup_handlers(
                lambda: logging.FileHandler(log_file_name),
                log,
            )

    # configure artifact loggers, note: this must happen last
    # since the levels of ancestor loggers are taken into account
    for artifact_log_qname in log_registry.get_artifact_log_qnames():
        log = logging.getLogger(artifact_log_qname)
        configure_artifact_log(log)

    # Setup handler for the special trace_log, with different default
    # configuration
    trace_dir_name = os.environ.get(TRACE_ENV_VAR, None)

    # If TORCH_COMPILE_DEBUG=1 is set but no TORCH_TRACE, automatically use
    # the torch_compile_debug directory for trace logs (to simplify tlparse usage)
    if trace_dir_name is None and os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1":
        import torch._dynamo.config as dynamo_config

        trace_dir_name = os.path.join(dynamo_config.debug_dir_root, "tlparse")

    if dtrace_dir_name := os.environ.get(DTRACE_ENV_VAR, None):
        GET_DTRACE_STRUCTURED = True
        trace_dir_name = dtrace_dir_name

    # This handler may remove itself if trace_dir_name is None and we are not
    # actually in an FB environment.  This allows us to defer actually
    # initializing it until we actually need to log anything.  This is
    # important because JK initializes a C++ singleton, which will pork our
    # process if we subsequently fork.
    global LOG_TRACE_HANDLER
    if LOG_TRACE_HANDLER is None:
        LOG_TRACE_HANDLER = LazyTraceHandler(trace_dir_name)
    # This log is ALWAYS at debug level.  We will additionally test if there
    # are any handlers before deciding to actually call logging on this.  Do
    # not manually call
    trace_log.setLevel(logging.DEBUG)
    # Override isEnabledFor so that logging.disable() cannot suppress trace
    # events.  When TORCH_TRACE is set we always want output regardless of
    # the global disable threshold.
    trace_log.isEnabledFor = lambda level: level >= trace_log.level
    trace_log_handler = _track_handler(LOG_TRACE_HANDLER)
    trace_log_handler.setFormatter(TorchLogsFormatter(trace=True))
    trace_log.addHandler(trace_log_handler)


class LazyTraceHandler(logging.StreamHandler):
    """Like FileHandler, but the file is allocated lazily only upon the first log message"""

    def __init__(self, root_dir: str | None) -> None:
        # This is implemented in the same way that delay is implemented on
        # FileHandler
        self.root_dir = root_dir
        logging.Handler.__init__(self)
        self.stream = None
        self._pid: int | None = None
        self._stream_path: str | None = None
        self._builtin_open = open
        self._pending_log_version = False

    def _close_stream(self, *, flush: bool = True) -> None:
        if self.stream:
            stream = self.stream
            try:
                if flush:
                    self.flush()
            finally:
                self.stream = None
                self._pid = None
                self._stream_path = None
                self._pending_log_version = False
                if flush and hasattr(stream, "close"):
                    stream.close()
                elif not flush:
                    # stream.close() flushes Python buffers.  In a post-fork
                    # child, close only the inherited fd so parent buffers are
                    # discarded instead of written to the parent's trace file.
                    os.close(stream.fileno())
                    try:
                        stream.close()
                    except OSError as exc:
                        if exc.errno != errno.EBADF:
                            raise

    # cloned from FileHandler in cpython
    def close(self) -> None:
        self.acquire()
        try:
            try:
                self._close_stream(flush=self._pid is None or self._pid == os.getpid())
            finally:
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                # Also see Issue #42378: we also rely on
                # self._closed being set to True there
                logging.StreamHandler.close(self)
        finally:
            self.release()

    def emit(self, record) -> None:
        current_pid = os.getpid()
        if self.stream is not None and self._pid != current_pid:
            self._close_stream(flush=False)

        if self.stream is None:
            if self.root_dir is None:
                TRACE_LOG_DIR = "/logs"

                import torch.version as torch_version

                if (
                    hasattr(torch_version, "git_version")
                    and os.getenv("MAST_HPC_JOB_NAME") is None
                ):
                    log.info(
                        "LazyTraceHandler: disabled because not fbcode or conda on mast"
                    )
                elif not torch._utils_internal.justknobs_check("pytorch/trace:enable"):
                    log.info(
                        "LazyTraceHandler: disabled because justknobs_check('pytorch/trace:enable') returned False"
                    )
                elif not os.path.exists(TRACE_LOG_DIR):
                    log.info(
                        "LazyTraceHandler: disabled because %s does not exist",
                        TRACE_LOG_DIR,
                    )
                elif not os.access(TRACE_LOG_DIR, os.W_OK):
                    log.info(
                        "LazyTraceHandler: disabled because %s is not writeable",
                        TRACE_LOG_DIR,
                    )
                else:
                    self.root_dir = TRACE_LOG_DIR

            if self.root_dir is not None:
                os.makedirs(self.root_dir, exist_ok=True)
                ranksuffix = ""
                if dist.is_available() and dist.is_initialized():
                    ranksuffix = f"rank_{dist.get_rank()}_"
                fd, path = tempfile.mkstemp(
                    suffix=".log",
                    prefix=LOG_PREFIX + ranksuffix,
                    dir=self.root_dir,
                )
                self.stream = os.fdopen(fd, mode="w+")
                self._pid = current_pid
                self._stream_path = path
                log.info("LazyTraceHandler: logging to %s", self._stream_path)
                # Log tlparse path via inductor logger so it shows when
                # TORCH_LOGS="inductor" is enabled
                inductor_log = logging.getLogger("torch._inductor")
                inductor_log.info("tlparse raw data: %s", self._stream_path)
                self._pending_log_version = True
            else:
                # We go poof, remove and no-op
                trace_log.removeHandler(self)
                return
        if self.stream:
            super().emit(record)
            if self._pending_log_version:
                self._pending_log_version = False
                _log_torch_version()


def _log_torch_version() -> None:
    import torch
    from torch._environment import is_fbcode
    from torch._utils_internal import get_torch_source_version

    version_info: dict[str, object] = {
        "pytorch_version": torch.__version__,
        "commit": get_torch_source_version(),
        "oss": not is_fbcode(),
    }

    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": "torch_version", "encoding": "json"},
        payload_fn=lambda: version_info,
        suppress_context=True,
        expect_trace_id=False,
    )


@functools.cache
def warning_once(logger_obj, *args, **kwargs) -> None:
    """
    This function is similar to `logger.warning()`, but will emit the warning with the same message only once
    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    logger_obj.warning(*args, **kwargs)


def safe_grad_filter(message, category, filename, lineno, file=None, line=None) -> bool:
    return "The .grad attribute of a Tensor" not in str(message)


def user_warning_filter(
    message, category, filename, lineno, file=None, line=None
) -> bool:
    return category is not UserWarning


@contextlib.contextmanager
def hide_warnings(filter_fn=lambda *args, **kwargs: True):
    """
    A context manager that temporarily suppresses warnings,
    using public API: https://docs.python.org/3/library/warnings.html#warnings.showwarning.

    Useful to hide warnings without mutating warnings module state, see:
    https://github.com/pytorch/pytorch/issues/128427#issuecomment-2161496162.

    NOTE: Warnings issued under this context will still be cached in the __warningregistry__
    and count towards the once/default rule. So you should NEVER use this on a user-land function.

    Filter must implement the showwarning API:
    def filter_fn(message, category, filename, lineno, file=None, line=None) -> bool:
        return True  # show this warning entry
    """
    prior = warnings.showwarning

    def _showwarning(*args, **kwargs):
        if filter_fn(*args, **kwargs):
            prior(*args, **kwargs)

    try:
        warnings.showwarning = _showwarning
        yield
    finally:
        warnings.showwarning = prior


class LazyString(Generic[_P]):
    def __init__(
        self, func: Callable[_P, str], *args: _P.args, **kwargs: _P.kwargs
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.func(*self.args, **self.kwargs)


# Logs the time it takes to do structured logging by frame/compile id
# key is always {frame_id}_{frame_compile_id}
structured_logging_overhead: dict[str, float] = defaultdict(float)


def add_structured_logging_overhead(time_spent: float) -> None:
    global structured_logging_overhead
    key = None
    if (trace_id := torch._guards.CompileContext.current_trace_id()) is not None:
        frame_id = trace_id.compile_id.frame_id
        frame_compile_id = trace_id.compile_id.frame_compile_id
        # Why not trace_id.attempt, like structured logging?
        # We aggregate across all attempts because
        # a compilation metric is logged per successful attempt
        key = f"{frame_id}_{frame_compile_id}"
    # TODO: deal with structured logging that occurs outside of specific compile ids
    # It's hard to figure out where we would log that if we want it in compilation metrics
    # itself.
    if key is not None:
        key = str(key)
        structured_logging_overhead[key] += time_spent


def get_structured_logging_overhead() -> float | None:
    key = None
    if (trace_id := torch._guards.CompileContext.current_trace_id()) is not None:
        frame_id = trace_id.compile_id.frame_id
        frame_compile_id = trace_id.compile_id.frame_compile_id
        key = f"{frame_id}_{frame_compile_id}"
    if key is not None:
        return structured_logging_overhead.get(key)
    else:
        return None


def trace_structured_artifact(
    name: str,  # this will go in metadata
    encoding: str,
    payload_fn: Callable[[], str | object | None] = lambda: None,
    compile_id: CompileId | None = None,
) -> None:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": name,
            "encoding": encoding,
        },
        payload_fn=payload_fn,
        compile_id=compile_id,
    )


def trace_structured(
    name: str,
    # NB: metadata expected to be dict so adding more info is forward compatible
    # Tuple[str, int] is a special case for string interning
    metadata_fn: Callable[[], dict[str, Any] | tuple[str, int]] = dict,
    *,
    payload_fn: Callable[[], str | object | None] = lambda: None,
    suppress_context: bool = False,
    expect_trace_id: bool = True,  # Whether or not we expect to have a current trace id
    record_logging_overhead: bool = True,  # Whether or not to record the time spent on structured logging
    compile_id: CompileId | None = None,  # Optional if unavailable in the trace
) -> None:
    """
    metadata is an arbitrary JSON compatible struct, but it's expected to not be
    too long (e.g., less than 1MB)

    payload is an arbitrary string, which can be arbitrarily long (but expected to have
    newlines so no lines are too long)
    """
    reserved_names = [
        "rank",
        "compiled_autograd_id",
        "frame_id",
        "frame_compile_id",
        "attempt",
        "severity",
        "timestamp",
        "pathname",
        "thread",
        "subgraph_name",
    ]
    if name in reserved_names:
        raise AssertionError(f"name {name!r} is reserved and cannot be used")
    if not callable(metadata_fn):
        raise AssertionError(
            f"metadata_fn should be callable, but got {type(metadata_fn)}"
        )
    if not callable(payload_fn):
        raise AssertionError(
            f"payload_fn should be callable, but got {type(payload_fn)}"
        )
    # trace_log never propagates and is ALWAYS DEBUG, so also check that there
    # are handlers instead of checking the log level
    if trace_log.handlers:
        start_time = time.time_ns()
        record: dict[str, object] = {}
        record[name] = metadata_fn()
        if not suppress_context:
            # TODO: Actually, the rank probably should just be emitted once at
            # the top, and not repeatedly spammed in all the logs, since it
            # never changes and we assume no interleaving
            if dist.is_available() and dist.is_initialized():
                record["rank"] = dist.get_rank()

            trace_id = torch._guards.CompileContext.current_trace_id()
            if expect_trace_id and trace_id is None and compile_id is None:
                # Record the stack of the log call to better diagnose why we
                # don't have a frame id for it
                record["stack"] = torch._logging.structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                )
            else:
                cid = trace_id.compile_id if trace_id else compile_id
                if cid is not None:
                    if cid.compiled_autograd_id is not None:
                        record["compiled_autograd_id"] = cid.compiled_autograd_id
                    if cid.frame_id is not None:
                        record["frame_id"] = cid.frame_id
                    if cid.frame_compile_id is not None:
                        record["frame_compile_id"] = cid.frame_compile_id
                if trace_id:
                    record["attempt"] = trace_id.attempt

            from torch.fx.traceback import _get_regional_inductor_subgraph_name

            subgraph_name = _get_regional_inductor_subgraph_name()
            if subgraph_name is not None:
                record["subgraph_name"] = subgraph_name

        payload = payload_fn()
        if payload is not None:
            if not isinstance(payload, str):
                if isinstance(payload, list):
                    # special case to look better
                    payload = "[\n" + ",\n".join(json.dumps(i) for i in payload) + "\n]"
                else:

                    def json_default(obj):
                        # Sets aren't json serializable
                        if isinstance(obj, set):
                            return list(obj)
                        raise TypeError(
                            f"Object of type {type(obj)} is not JSON serializable"
                        )

                    # force newlines so we are unlikely to overflow line limit
                    payload = json.dumps(payload, default=json_default, indent=0)
            h = hashlib.md5(usedforsecurity=False)
            h.update(payload.encode("utf-8"))
            record["has_payload"] = h.hexdigest()
        trace_log.debug(
            "", extra={"metadata": record, "payload": payload}, stacklevel=2
        )
        log_trace_structured_event(name, record)

        if record_logging_overhead:
            # Convert to seconds from nanoseconds, add it to the frame compile total
            structured_logging_overhead_s = (time.time_ns() - start_time) / 1e9
            add_structured_logging_overhead(structured_logging_overhead_s)


def dtrace_structured(
    name: str,
    # NB: metadata expected to be dict so adding more info is forward compatible
    # Tuple[str, int] is a special case for string interning
    metadata_fn: Callable[[], dict[str, Any] | tuple[str, int]] = dict,
    *,
    payload_fn: Callable[[], str | object | None] = lambda: None,
    suppress_context: bool = False,
    expect_trace_id: bool = False,  # Whether or not we expect to have a current trace id
    record_logging_overhead: bool = True,  # Whether or not to record the time spent on structured logging
) -> None:
    """
    For logging more detailed information used for debugging. This may result in
    the program becoming slow.
    """
    if GET_DTRACE_STRUCTURED:
        trace_structured(
            name,
            metadata_fn,
            payload_fn=payload_fn,
            suppress_context=suppress_context,
            expect_trace_id=expect_trace_id,
            record_logging_overhead=record_logging_overhead,
        )


import torch._guards
import torch._utils_internal
import torch.distributed as dist
import torch.fx.traceback
<|endoftext|># mypy: allow-untyped-defs
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import SymInt, Tensor
from torch._C import _add_docstr, _nested  # type: ignore[attr-defined]
from torch.types import _device as Device, _dtype as DType


__all__ = [
    "to_padded_tensor",
    "as_nested_tensor",
    "nested_tensor",
    "nested_tensor_from_jagged",
    "narrow",
    "masked_select",
]

# Allowlist these for weights_only load of NJT
from ._internal.nested_tensor import _rebuild_njt, NestedTensor as _NestedTensor


torch.serialization.add_safe_globals([_NestedTensor, _rebuild_njt])


def as_nested_tensor(
    ts: Tensor | list[Tensor] | tuple[Tensor, ...],
    dtype: DType | None = None,
    device: Device | None = None,
    layout=None,
) -> Tensor:
    r"""
    Constructs a nested tensor preserving autograd history from a tensor or a list / tuple of
    tensors.

    If a nested tensor is passed, it will be returned directly unless the device / dtype / layout
    differ. Note that converting device / dtype will result in a copy, while converting layout
    is not currently supported by this function.

    If a non-nested tensor is passed, it is treated as a batch of constituents of consistent size.
    A copy will be incurred if the passed device / dtype differ from those of the input OR if
    the input is non-contiguous. Otherwise, the input's storage will be used directly.

    If a tensor list is provided, tensors in the list are always copied during construction of
    the nested tensor.

    Args:
        ts (Tensor or List[Tensor] or Tuple[Tensor]): a tensor to treat as a nested tensor OR a
            list / tuple of tensors with the same ndim

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
            Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
        device (:class:`torch.device`, optional): the desired device of returned nested tensor.
            Default: if None, same :class:`torch.device` as leftmost tensor in the list
        layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
            Only strided and jagged layouts are supported. Default: if None, the strided layout.

    Example::

        >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
        >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
        >>> nt = torch.nested.as_nested_tensor([a, b], layout=torch.jagged)
        >>> nt.is_leaf
        False
        >>> buffer = torch.cat([torch.ones_like(a), torch.zeros_like(b)])
        >>> fake_grad = torch.nested.nested_tensor_from_jagged(buffer, nt.offsets())
        >>> nt.backward(fake_grad)
        >>> a.grad
        tensor([1., 1., 1.])
        >>> b.grad
        tensor([0., 0., 0., 0., 0.])
        >>> c = torch.randn(3, 5, requires_grad=True)
        >>> nt2 = torch.nested.as_nested_tensor(c, layout=torch.jagged)
    """
    is_tensor_list = isinstance(ts, (list, tuple)) and all(
        isinstance(t, Tensor) for t in ts
    )
    if not isinstance(ts, Tensor) and not is_tensor_list:
        raise TypeError(
            "as_nested_tensor(): Expected first argument to be a tensor or a list / tuple of tensors "
        )
    # convert tuple -> list if needed
    if is_tensor_list and not isinstance(ts, list):
        ts = list(ts)

    if isinstance(ts, Tensor) and ts.dim() < 2:
        raise RuntimeError(
            "as_nested_tensor(): Expected tensor argument to have dim() > 1"
        )

    if isinstance(ts, Tensor) and ts.is_nested:
        if layout == ts.layout:
            # return input directly or input copied to device / dtype
            return ts.to(device=device, dtype=dtype)
        else:
            # TODO: Just use nt.to(layout=layout) when it exists.
            raise RuntimeError(
                "as_nested_tensor(): Converting between nested tensor layouts is not supported"
            )

    if layout is None:
        layout = torch.strided
    if layout == torch.strided:
        if isinstance(ts, Tensor):
            buffer = ts.reshape(-1).to(device=device, dtype=dtype)
            nested_sizes = torch.tensor([t.shape for t in ts])
            return torch._nested_view_from_buffer(
                buffer,
                nested_sizes,
                *torch._nested_compute_contiguous_strides_offsets(nested_sizes),
            )
        else:
            if not isinstance(ts, list):
                raise AssertionError(
                    f"Expected ts to be a list, but got {type(ts).__name__}"
                )
            return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    elif layout == torch.jagged:
        if isinstance(ts, Tensor):
            if device is None:
                device = ts.device

            # contiguous() might be necessary to get flattened view.
            # we could probably be more precise about when to do this as an optimization
            values = ts.contiguous().flatten(0, 1).to(device=device, dtype=dtype)
            batch_size = ts.shape[0]
            seq_len = ts.shape[1]
            offsets = torch.arange(
                0, batch_size * seq_len + 1, seq_len, device=device, dtype=torch.int64
            )

            from torch.nested._internal.nested_tensor import (
                nested_view_from_values_offsets,
            )

            return nested_view_from_values_offsets(
                values, offsets, min_seqlen=seq_len, max_seqlen=seq_len
            )
        else:
            from torch.nested._internal.nested_tensor import jagged_from_list

            if not isinstance(ts, list):
                raise AssertionError(
                    f"Expected ts to be a list, but got {type(ts).__name__}"
                )
            nt, _ = jagged_from_list(ts, offsets=None, device=device, dtype=dtype)
            return nt
    else:
        raise RuntimeError(
            f"Specified layout is unsupported for nested tensors: {layout}"
        )


# Note: This not only adds doc strings for the nested ops, but
# also connects the torch.nested Python namespace to the torch._C._nested builtins.

to_padded_tensor = _add_docstr(
    _nested.nested_to_padded_tensor,
    r"""
to_padded_tensor(input, padding, output_size=None, out=None) -> Tensor

Returns a new (non-nested) Tensor by padding the :attr:`input` nested tensor.
The leading entries will be filled with the nested data,
while the trailing entries will be padded.

.. warning::

    :func:`to_padded_tensor` always copies the underlying data,
    since the nested and the non-nested tensors differ in memory layout.

Args:
    padding (float): The padding value for the trailing entries.

Keyword args:
    output_size (Tuple[int]): The size of the output tensor.
                              If given, it must be large enough to contain all nested data;
                              else, will infer by taking the max size of each nested sub-tensor along each dimension.
    out (Tensor, optional): the output tensor.

Example::

    >>> nt = torch.nested.nested_tensor([torch.randn((2, 5)), torch.randn((3, 4))])
    nested_tensor([
      tensor([[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276],
              [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995]]),
      tensor([[-1.8546, -0.7194, -0.2918, -0.1846],
              [ 0.2773,  0.8793, -0.5183, -0.6447],
              [ 1.8009,  1.8468, -0.9832, -1.5272]])
    ])
    >>> pt_infer = torch.nested.to_padded_tensor(nt, 0.0)
    tensor([[[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276],
             [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995],
             [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
            [[-1.8546, -0.7194, -0.2918, -0.1846,  0.0000],
             [ 0.2773,  0.8793, -0.5183, -0.6447,  0.0000],
             [ 1.8009,  1.8468, -0.9832, -1.5272,  0.0000]]])
    >>> pt_large = torch.nested.to_padded_tensor(nt, 1.0, (2, 4, 6))
    tensor([[[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276,  1.0000],
             [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]],
            [[-1.8546, -0.7194, -0.2918, -0.1846,  1.0000,  1.0000],
             [ 0.2773,  0.8793, -0.5183, -0.6447,  1.0000,  1.0000],
             [ 1.8009,  1.8468, -0.9832, -1.5272,  1.0000,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]]])
    >>> pt_small = torch.nested.to_padded_tensor(nt, 2.0, (2, 2, 2))
    RuntimeError: Value in output_size is less than NestedTensor padded size. Truncation is not supported.

""",
)


def nested_tensor(
    tensor_list,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
) -> Tensor:
    r"""
    Constructs a nested tensor with no autograd history (also known as a "leaf tensor", see
    :ref:`Autograd mechanics <autograd-mechanics>`) from :attr:`tensor_list` a list of tensors.

    Args:
        tensor_list (List[array_like]): a list of tensors, or anything that can be passed to torch.tensor,
        where each element of the list has the same dimensionality.

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
            Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
        layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
            Only strided and jagged layouts are supported. Default: if None, the strided layout.
        device (:class:`torch.device`, optional): the desired device of returned nested tensor.
            Default: if None, same :class:`torch.device` as leftmost tensor in the list
        requires_grad (bool, optional): If autograd should record operations on the
            returned nested tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned nested tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.

    Example::

        >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
        >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
        >>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged, requires_grad=True)
        >>> nt.is_leaf
        True
    """
    if layout is None:
        layout = torch.strided
    if layout == torch.strided:
        return _nested.nested_tensor(
            tensor_list,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
    elif layout == torch.jagged:
        # Need to wrap lists of scalars as tensors
        list_of_tensors = [
            t if isinstance(t, Tensor) else torch.as_tensor(t) for t in tensor_list
        ]

        from torch.nested._internal.nested_tensor import jagged_from_list

        with torch.no_grad():
            nt, _ = jagged_from_list(
                list_of_tensors, offsets=None, device=device, dtype=dtype
            )

        nt.requires_grad_(requires_grad)
        if pin_memory:
            nt = nt.pin_memory()  # type: ignore[assignment]

        return nt
    else:
        raise RuntimeError(
            f"Specified layout is unsupported for nested tensors: {layout}"
        )


def narrow(
    tensor: Tensor,
    dim: int,
    start: int | Tensor,
    length: int | Tensor,
    layout=torch.strided,
) -> Tensor:
    r"""
    Constructs a nested tensor (which might be a view) from :attr:`tensor`, a strided tensor. This follows
    similar semantics to torch.Tensor.narrow, where in the :attr:`dim`-th dimension the new nested tensor
    shows only the elements in the interval `[start, start+length)`. As nested representations
    allow for a different `start` and `length` at each 'row' of that dimension, :attr:`start` and :attr:`length`
    can also be tensors of shape `tensor.shape[0]`.

    There's some differences depending on the layout you use for the nested tensor. If using strided layout,
    torch.narrow will do a copy of the narrowed data into a contiguous NT with strided layout, while
    jagged layout narrow() will create a non-contiguous view of your original strided tensor. This particular
    representation is really useful for representing kv-caches in Transformer models, as specialized
    SDPA kernels can deal with format easily, resulting in performance improvements.


    Args:
        tensor (:class:`torch.Tensor`): a strided tensor, which will be used as the underlying data
            for the nested tensor if using the jagged layout or will be copied for the strided layout.
        dim (int): the dimension where narrow will be applied. Only `dim=1` is supported for the
            jagged layout, while strided supports all dim
        start (Union[int, :class:`torch.Tensor`]): starting element for the narrow operation
        length (Union[int, :class:`torch.Tensor`]): number of elements taken during the narrow op

    Keyword arguments:
        layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
            Only strided and jagged layouts are supported. Default: if None, the strided layout.

    Example::

        >>> starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        >>> lengths = torch.tensor([3, 2, 2, 1, 5], dtype=torch.int64)
        >>> narrow_base = torch.randn(5, 10, 20)
        >>> nt_narrowed = torch.nested.narrow(narrow_base, 1, starts, lengths, layout=torch.jagged)
        >>> nt_narrowed.is_contiguous()
        False
    """
    if not isinstance(start, (int, SymInt, Tensor)):
        raise RuntimeError("start must be an integer or a tensor")

    if not isinstance(length, (int, SymInt, Tensor)):
        raise RuntimeError("length must be an integer or a tensor")

    if layout == torch.strided:
        if isinstance(start, Tensor) or isinstance(length, Tensor):
            raise RuntimeError(
                "start and length must be integers for the strided layout NT impl"
            )
        # TODO: switch to as_nested_tensor(tensor) when it is available
        nt = as_nested_tensor(torch.unbind(tensor), layout=torch.strided).narrow(
            dim, start, length
        )
    elif layout == torch.jagged:
        if dim != 1:
            raise RuntimeError("jagged layout only supports dim=1")

        from torch.nested._internal.nested_tensor import jagged_from_tensor_and_lengths

        if isinstance(start, (int, SymInt)):
            start = torch.tensor([start], device=tensor.device, dtype=torch.int64)

        if isinstance(length, (int, SymInt)):
            length = torch.tensor([length], device=tensor.device, dtype=torch.int64)

        nt, _, _ = jagged_from_tensor_and_lengths(tensor, start, length)
    else:
        raise RuntimeError(
            f"Specified layout is unsupported for nested narrow: {layout}"
        )

    return nt


def nested_tensor_from_jagged(
    values: Tensor,
    offsets: Tensor | None = None,
    lengths: Tensor | None = None,
    jagged_dim: int | None = None,
    min_seqlen: int | None = None,
    max_seqlen: int | None = None,
) -> Tensor:
    r"""
    Constructs a jagged layout nested tensor from the given jagged components. The jagged layout
    consists of a required values buffer with the jagged dimension packed into a single dimension.
    The offsets / lengths metadata determines how this dimension is split into batch elements
    and are expected to be allocated on the same device as the values buffer.

    Expected metadata formats:
        * offsets: Indices within the packed dimension splitting it into heterogeneously-sized
          batch elements. Example: [0, 2, 3, 6] indicates that a packed jagged dim of size 6
          should be conceptually split into batch elements of length [2, 1, 3]. Note that both the
          beginning and ending offsets are required for kernel convenience (i.e. shape batch_size + 1).
        * lengths: Lengths of the individual batch elements; shape == batch_size. Example: [2, 1, 3]
          indicates that a packed jagged dim of size 6 should be conceptually split into batch
          elements of length [2, 1, 3].

    Note that it can be useful to provide both offsets and lengths. This describes a nested tensor
    with "holes", where the offsets indicate the start position of each batch item and the length
    specifies the total number of elements (see example below).

    The returned jagged layout nested tensor will be a view of the input values tensor.

    Args:
        values (:class:`torch.Tensor`): The underlying buffer in the shape of
            (sum_B(*), D_1, ..., D_N). The jagged dimension is packed into a single dimension,
            with the offsets / lengths metadata used to distinguish batch elements.
        offsets (optional :class:`torch.Tensor`): Offsets into the jagged dimension of shape B + 1.
        lengths (optional :class:`torch.Tensor`): Lengths of the batch elements of shape B.
        jagged_dim (optional int): Indicates which dimension in values is the packed jagged
            dimension. Must be >= 1 as the batch dimension (dim=0) cannot be ragged.
            If None, this is set to dim=1 (i.e. the dimension immediately following the batch dimension). Default: None
        min_seqlen (optional int): If set, uses the specified value as the cached minimum sequence
            length for the returned nested tensor. This can be a useful alternative to computing
            this value on-demand, possibly avoiding a GPU -> CPU sync. Default: None
        max_seqlen (optional int): If set, uses the specified value as the cached maximum sequence
            length for the returned nested tensor. This can be a useful alternative to computing
            this value on-demand, possibly avoiding a GPU -> CPU sync. Default: None

    Example::

        >>> values = torch.randn(12, 5)
        >>> offsets = torch.tensor([0, 3, 5, 6, 10, 12])
        >>> nt = nested_tensor_from_jagged(values, offsets)
        >>> # 3D shape with the middle dimension jagged
        >>> nt.shape  # xdoctest: +ELLIPSIS
        torch.Size([5, j..., 5])
        >>> # Length of each item in the batch:
        >>> offsets.diff()
        tensor([3, 2, 1, 4, 2])

        >>> values = torch.randn(6, 5)
        >>> offsets = torch.tensor([0, 2, 3, 6])
        >>> lengths = torch.tensor([1, 1, 2])
        >>> # NT with holes
        >>> nt = nested_tensor_from_jagged(values, offsets, lengths)
        >>> a, b, c = nt.unbind()
        >>> # Batch item 1 consists of indices [0, 1)
        >>> torch.equal(a, values[0:1, :])
        True
        >>> # Batch item 2 consists of indices [2, 3)
        >>> torch.equal(b, values[2:3, :])
        True
        >>> # Batch item 3 consists of indices [3, 5)
        >>> torch.equal(c, values[3:5, :])
        True
    """
    from torch.fx._symbolic_trace import is_fx_tracing

    if is_fx_tracing():
        raise RuntimeError(
            "torch.nested.nested_tensor_from_jagged does not support tracing with fx.symbolic_trace. "
            "Use fx.wrap to wrap the function that calls nested_tensor_from_jagged."
        )

    if offsets is None:
        if lengths is None:
            raise RuntimeError(
                "nested_tensor_from_jagged(): At least one of offsets or lengths is required."
            )
        else:
            # TODO: Truly support offsets=None at some point?
            # For now, just convert lengths -> offsets for kernel convenience
            offsets = F.pad(lengths.cumsum(0), (1, 0))
            lengths = None

    if jagged_dim is None:
        jagged_dim = 1
    elif jagged_dim < 1:
        raise ValueError(f"Expected jagged_dim >=1, but got {jagged_dim}.")

    from torch.nested._internal.nested_tensor import (
        nested_view_from_values_offsets_lengths,
    )

    return nested_view_from_values_offsets_lengths(
        values,
        offsets,
        lengths,
        ragged_idx=jagged_dim,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
    )


def masked_select(tensor: Tensor, mask: Tensor) -> Tensor:
    r"""
    Constructs a nested tensor given a strided tensor input and a strided mask, the resulting jagged layout nested tensor
    will have values retain values where the mask is equal to True. The dimensionality of the mask is preserved and is
    represented with the offsets, this is unlike :func:`masked_select` where the output is collapsed to a 1D tensor.

    Args:
    tensor (:class:`torch.Tensor`): a strided tensor from which the jagged layout nested tensor is constructed from.
    mask (:class:`torch.Tensor`): a strided mask tensor which is applied to the tensor input

    Example::

        >>> tensor = torch.randn(3, 3)
        >>> mask = torch.tensor([[False, False, True], [True, False, True], [False, False, True]])
        >>> nt = torch.nested.masked_select(tensor, mask)
        >>> nt.shape  # xdoctest: +ELLIPSIS
        torch.Size([3, j...])
        >>> # Length of each item in the batch:
        >>> nt.offsets().diff()
        tensor([1, 2, 1])

        >>> tensor = torch.randn(6, 5)
        >>> mask = torch.tensor([False])
        >>> nt = torch.nested.masked_select(tensor, mask)
        >>> nt.shape  # xdoctest: +ELLIPSIS
        torch.Size([6, j...])
        >>> # Length of each item in the batch:
        >>> nt.offsets().diff()
        tensor([0, 0, 0, 0, 0, 0])
    """
    if tensor.layout != torch.strided:
        raise RuntimeError(
            f"torch.nested.masked_select requires a strided tensor, given {tensor.layout}"
        )

    if mask.layout != torch.strided:
        raise RuntimeError(
            f"torch.nested.masked_select requires a strided mask, given: {mask.layout}"
        )
    res_values = tensor.masked_select(mask)
    expanded_mask = mask.expand(tensor.shape)
    res_lengths = expanded_mask.sum(dim=tensor.ndim - 1).view(-1)

    from torch.nested._internal.nested_tensor import nested_view_from_values_offsets

    return nested_view_from_values_offsets(
        values=res_values,
        offsets=F.pad(res_lengths.cumsum(dim=0), (1, 0)),
    )
<|endoftext|>from typing import *  # noqa: F403

import torch
from torch.fx.experimental._constant_symnode import ConstantIntNode


__all__ = ["NestedIntNode"]


# Python version of aten/src/ATen/core/NestedIntSymNodeImpl.cpp
def _eq(lhs: Any, rhs: Any) -> bool:
    return (
        isinstance(lhs, NestedIntNode)
        and isinstance(rhs, NestedIntNode)
        and lhs.t_id == rhs.t_id
        and lhs.coeff == rhs.coeff
    )


def _ge(lhs: Any, rhs: Any) -> bool:
    if isinstance(rhs, NestedIntNode) and isinstance(lhs, NestedIntNode):
        if lhs.t_id == rhs.t_id:
            return lhs.coeff >= rhs.coeff
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(lhs, NestedIntNode):
        if rhs.is_constant() and rhs.constant_int() <= 2:
            return True
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(rhs, NestedIntNode):
        if lhs.is_constant() and lhs.constant_int() < 2:
            return False
        raise ValueError("ge: relation is indeterminate")
    else:
        raise ValueError("inputs unsupported")


class NestedIntNode:
    def __init__(self, t_id: int, coeff: int) -> None:
        self.t_id = t_id
        self.coeff = coeff

    def nested_int_coeff(self) -> int:
        return self.coeff

    def maybe_as_int(self) -> Optional[int]:
        return None

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return True

    def clone(self) -> "NestedIntNode":
        return self

    def _str(self) -> Any:
        if self.coeff == 1:
            return f"j{self.t_id}"
        return f"{self.coeff}*j{self.t_id}"

    def str(self) -> Any:
        return self._str()

    def __str__(self) -> Any:
        return self._str()

    def __repr__(self) -> Any:
        return self._str()

    def _graph_repr(self) -> Any:
        return self._str()

    def mul(self, other: Any) -> "NestedIntNode":
        if other.is_constant():
            other = other.constant_int()
        else:
            raise ValueError(f"unsupported: {type(other)}")
        return NestedIntNode(self.t_id, self.coeff * other)

    def eq(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_eq(self, other))

    def ne(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _eq(self, other))

    def gt(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _ge(other, self))

    def lt(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _ge(self, other))

    def le(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_ge(other, self))

    def ge(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_ge(self, other))

    def is_symbolic(self) -> bool:
        return False

    def nested_int(self) -> int:
        return self.t_id

    def is_constant(self) -> bool:
        return False

    def wrap_int(self, num: int) -> ConstantIntNode:
        if type(num) is not int:
            raise AssertionError(
                f"Expected num to be int, but got {type(num).__name__}"
            )
        return ConstantIntNode(num)
<|endoftext|># mypy: allow-untyped-defs
import logging

