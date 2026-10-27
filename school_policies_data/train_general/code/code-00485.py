                _Filter(*cls.ALWAYS_EXCLUDE, *exclude),
                _Filter(*include),
            )
        )

    @classmethod
    def _find_iter(
        cls, where: StrPath, exclude: _Filter, include: _Filter
    ) -> Iterator[str]:
        raise NotImplementedError


class PackageFinder(_Finder):
    """
    Generate a list of all Python packages found within a directory
    """

    ALWAYS_EXCLUDE = ("ez_setup", "*__pycache__")

    @classmethod
    def _find_iter(
        cls, where: StrPath, exclude: _Filter, include: _Filter
    ) -> Iterator[str]:
        """
        All the packages found in 'where' that pass the 'include' filter, but
        not the 'exclude' filter.
        """
        for root, dirs, files in os.walk(str(where), followlinks=True):
            # Copy dirs to iterate over it, then empty dirs.
            all_dirs = dirs[:]
            dirs[:] = []

            for dir in all_dirs:
                full_path = os.path.join(root, dir)
                rel_path = os.path.relpath(full_path, where)
                package = rel_path.replace(os.path.sep, '.')

                # Skip directory trees that are not valid packages
                if '.' in dir or not cls._looks_like_package(full_path, package):
                    continue

                # Should this package be included?
                if include(package) and not exclude(package):
                    yield package

                # Early pruning if there is nothing else to be scanned
                if f"{package}*" in exclude or f"{package}.*" in exclude:
                    continue

                # Keep searching subdirectories, as there may be more packages
                # down there, even if the parent was excluded.
                dirs.append(dir)

    @staticmethod
    def _looks_like_package(path: StrPath, _package_name: str) -> bool:
        """Does a directory look like a package?"""
        return os.path.isfile(os.path.join(path, '__init__.py'))


class PEP420PackageFinder(PackageFinder):
    @staticmethod
    def _looks_like_package(_path: StrPath, _package_name: str) -> bool:
        return True


class ModuleFinder(_Finder):
    """Find isolated Python modules.
    This function will **not** recurse subdirectories.
    """

    @classmethod
    def _find_iter(
        cls, where: StrPath, exclude: _Filter, include: _Filter
    ) -> Iterator[str]:
        for file in glob(os.path.join(where, "*.py")):
            module, _ext = os.path.splitext(os.path.basename(file))

            if not cls._looks_like_module(module):
                continue

            if include(module) and not exclude(module):
                yield module

    _looks_like_module = staticmethod(_valid_name)


# We have to be extra careful in the case of flat layout to not include files
# and directories not meant for distribution (e.g. tool-related)


class FlatLayoutPackageFinder(PEP420PackageFinder):
    _EXCLUDE = (
        "ci",
        "bin",
        "debian",
        "doc",
        "docs",
        "documentation",
        "manpages",
        "news",
        "newsfragments",
        "changelog",
        "test",
        "tests",
        "unit_test",
        "unit_tests",
        "example",
        "examples",
        "scripts",
        "tools",
        "util",
        "utils",
        "python",
        "build",
        "dist",
        "venv",
        "env",
        "requirements",
        # ---- Task runners / Build tools ----
        "tasks",  # invoke
        "fabfile",  # fabric
        "site_scons",  # SCons
        # ---- Other tools ----
        "benchmark",
        "benchmarks",
        "exercise",
        "exercises",
        "htmlcov",  # Coverage.py
        # ---- Hidden directories/Private packages ----
        "[._]*",
    )

    DEFAULT_EXCLUDE = tuple(chain_iter((p, f"{p}.*") for p in _EXCLUDE))
    """Reserved package names"""

    @staticmethod
    def _looks_like_package(_path: StrPath, package_name: str) -> bool:
        names = package_name.split('.')
        # Consider PEP 561
        root_pkg_is_valid = names[0].isidentifier() or names[0].endswith("-stubs")
        return root_pkg_is_valid and all(name.isidentifier() for name in names[1:])


class FlatLayoutModuleFinder(ModuleFinder):
    DEFAULT_EXCLUDE = (
        "setup",
        "conftest",
        "test",
        "tests",
        "example",
        "examples",
        "build",
        # ---- Task runners ----
        "toxfile",
        "noxfile",
        "pavement",
        "dodo",
        "tasks",
        "fabfile",
        # ---- Other tools ----
        "[Ss][Cc]onstruct",  # SCons
        "conanfile",  # Connan: C/C++ build tool
        "manage",  # Django
        "benchmark",
        "benchmarks",
        "exercise",
        "exercises",
        # ---- Hidden files/Private modules ----
        "[._]*",
    )
    """Reserved top-level module names"""


def _find_packages_within(root_pkg: str, pkg_dir: StrPath) -> list[str]:
    nested = PEP420PackageFinder.find(pkg_dir)
    return [root_pkg] + [f'{root_pkg}.{n}' for n in nested]


class ConfigDiscovery:
    """Fill-in metadata and options that can be automatically derived
    (from other metadata/options, the file system or conventions)
    """

    def __init__(self, distribution: Distribution) -> None:
        self.dist = distribution
        self._called = False
        self._disabled = False
        self._skip_ext_modules = False

    def _disable(self):
        """Internal API to disable automatic discovery"""
        self._disabled = True

    def _ignore_ext_modules(self):
        """Internal API to disregard ext_modules.

        Normally auto-discovery would not be triggered if ``ext_modules`` are set
        (this is done for backward compatibility with existing packages relying on
        ``setup.py`` or ``setup.cfg``). However, ``setuptools`` can call this function
        to ignore given ``ext_modules`` and proceed with the auto-discovery if
        ``packages`` and ``py_modules`` are not given (e.g. when using pyproject.toml
        metadata).
        """
        self._skip_ext_modules = True

    @property
    def _root_dir(self) -> StrPath:
        # The best is to wait until `src_root` is set in dist, before using _root_dir.
        return self.dist.src_root or os.curdir

    @property
    def _package_dir(self) -> dict[str, str]:
        if self.dist.package_dir is None:
            return {}
        return self.dist.package_dir

    def __call__(
        self, force: bool = False, name: bool = True, ignore_ext_modules: bool = False
    ) -> None:
        """Automatically discover missing configuration fields
        and modifies the given ``distribution`` object in-place.

        Note that by default this will only have an effect the first time the
        ``ConfigDiscovery`` object is called.

        To repeatedly invoke automatic discovery (e.g. when the project
        directory changes), please use ``force=True`` (or create a new
        ``ConfigDiscovery`` instance).
        """
        if force is False and (self._called or self._disabled):
            # Avoid overhead of multiple calls
            return

        self._analyse_package_layout(ignore_ext_modules)
        if name:
            self.analyse_name()  # depends on ``packages`` and ``py_modules``

        self._called = True

    def _explicitly_specified(self, ignore_ext_modules: bool) -> bool:
        """``True`` if the user has specified some form of package/module listing"""
        ignore_ext_modules = ignore_ext_modules or self._skip_ext_modules
        ext_modules = not (self.dist.ext_modules is None or ignore_ext_modules)
        return (
            self.dist.packages is not None
            or self.dist.py_modules is not None
            or ext_modules
            or hasattr(self.dist, "configuration")
            and self.dist.configuration
            # ^ Some projects use numpy.distutils.misc_util.Configuration
        )

    def _analyse_package_layout(self, ignore_ext_modules: bool) -> bool:
        if self._explicitly_specified(ignore_ext_modules):
            # For backward compatibility, just try to find modules/packages
            # when nothing is given
            return True

        log.debug(
            "No `packages` or `py_modules` configuration, performing "
            "automatic discovery."
        )

        return (
            self._analyse_explicit_layout()
            or self._analyse_src_layout()
            # flat-layout is the trickiest for discovery so it should be last
            or self._analyse_flat_layout()
        )

    def _analyse_explicit_layout(self) -> bool:
        """The user can explicitly give a package layout via ``package_dir``"""
        package_dir = self._package_dir.copy()  # don't modify directly
        package_dir.pop("", None)  # This falls under the "src-layout" umbrella
        root_dir = self._root_dir

        if not package_dir:
            return False

        log.debug(f"`explicit-layout` detected -- analysing {package_dir}")
        pkgs = chain_iter(
            _find_packages_within(pkg, os.path.join(root_dir, parent_dir))
            for pkg, parent_dir in package_dir.items()
        )
        self.dist.packages = list(pkgs)
        log.debug(f"discovered packages -- {self.dist.packages}")
        return True

    def _analyse_src_layout(self) -> bool:
        """Try to find all packages or modules under the ``src`` directory
        (or anything pointed by ``package_dir[""]``).

        The "src-layout" is relatively safe for automatic discovery.
        We assume that everything within is meant to be included in the
        distribution.

        If ``package_dir[""]`` is not given, but the ``src`` directory exists,
        this function will set ``package_dir[""] = "src"``.
        """
        package_dir = self._package_dir
        src_dir = os.path.join(self._root_dir, package_dir.get("", "src"))
        if not os.path.isdir(src_dir):
            return False

        log.debug(f"`src-layout` detected -- analysing {src_dir}")
        package_dir.setdefault("", os.path.basename(src_dir))
        self.dist.package_dir = package_dir  # persist eventual modifications
        self.dist.packages = PEP420PackageFinder.find(src_dir)
        self.dist.py_modules = ModuleFinder.find(src_dir)
        log.debug(f"discovered packages -- {self.dist.packages}")
        log.debug(f"discovered py_modules -- {self.dist.py_modules}")
        return True

    def _analyse_flat_layout(self) -> bool:
        """Try to find all packages and modules under the project root.

        Since the ``flat-layout`` is more dangerous in terms of accidentally including
        extra files/directories, this function is more conservative and will raise an
        error if multiple packages or modules are found.

        This assumes that multi-package dists are uncommon and refuse to support that
        use case in order to be able to prevent unintended errors.
        """
        log.debug(f"`flat-layout` detected -- analysing {self._root_dir}")
        return self._analyse_flat_packages() or self._analyse_flat_modules()

    def _analyse_flat_packages(self) -> bool:
        self.dist.packages = FlatLayoutPackageFinder.find(self._root_dir)
        top_level = remove_nested_packages(remove_stubs(self.dist.packages))
        log.debug(f"discovered packages -- {self.dist.packages}")
        self._ensure_no_accidental_inclusion(top_level, "packages")
        return bool(top_level)

    def _analyse_flat_modules(self) -> bool:
        self.dist.py_modules = FlatLayoutModuleFinder.find(self._root_dir)
        log.debug(f"discovered py_modules -- {self.dist.py_modules}")
        self._ensure_no_accidental_inclusion(self.dist.py_modules, "modules")
        return bool(self.dist.py_modules)

    def _ensure_no_accidental_inclusion(self, detected: list[str], kind: str):
        if len(detected) > 1:
            from inspect import cleandoc

            from setuptools.errors import PackageDiscoveryError

            msg = f"""Multiple top-level {kind} discovered in a flat-layout: {detected}.

            To avoid accidental inclusion of unwanted files or directories,
            setuptools will not proceed with this build.

            If you are trying to create a single distribution with multiple {kind}
            on purpose, you should not rely on automatic discovery.
            Instead, consider the following options:

            1. set up custom discovery (`find` directive with `include` or `exclude`)
            2. use a `src-layout`
            3. explicitly set `py_modules` or `packages` with a list of names

            To find more information, look for "package discovery" on setuptools docs.
            """
            raise PackageDiscoveryError(cleandoc(msg))

    def analyse_name(self) -> None:
        """The packages/modules are the essential contribution of the author.
        Therefore the name of the distribution can be derived from them.
        """
        if self.dist.metadata.name or self.dist.name:
            # get_name() is not reliable (can return "UNKNOWN")
            return

        log.debug("No `name` configuration, performing automatic discovery")

        name = (
            self._find_name_single_package_or_module()
            or self._find_name_from_packages()
        )
        if name:
            self.dist.metadata.name = name

    def _find_name_single_package_or_module(self) -> str | None:
        """Exactly one module or package"""
        for field in ('packages', 'py_modules'):
            items = getattr(self.dist, field, None) or []
            if items and len(items) == 1:
                log.debug(f"Single module/package detected, name: {items[0]}")
                return items[0]

        return None

    def _find_name_from_packages(self) -> str | None:
        """Try to find the root package that is not a PEP 420 namespace"""
        if not self.dist.packages:
            return None

        packages = remove_stubs(sorted(self.dist.packages, key=len))
        package_dir = self.dist.package_dir or {}

        parent_pkg = find_parent_package(packages, package_dir, self._root_dir)
        if parent_pkg:
            log.debug(f"Common parent package detected, name: {parent_pkg}")
            return parent_pkg

        log.warn("No parent package detected, impossible to derive `name`")
        return None


def remove_nested_packages(packages: list[str]) -> list[str]:
    """Remove nested packages from a list of packages.

    >>> remove_nested_packages(["a", "a.b1", "a.b2", "a.b1.c1"])
    ['a']
    >>> remove_nested_packages(["a", "b", "c.d", "c.d.e.f", "g.h", "a.a1"])
    ['a', 'b', 'c.d', 'g.h']
    """
    pkgs = sorted(packages, key=len)
    top_level = pkgs[:]
    size = len(pkgs)
    for i, name in enumerate(reversed(pkgs)):
        if any(name.startswith(f"{other}.") for other in top_level):
            top_level.pop(size - i - 1)

    return top_level


def remove_stubs(packages: list[str]) -> list[str]:
    """Remove type stubs (:pep:`561`) from a list of packages.

    >>> remove_stubs(["a", "a.b", "a-stubs", "a-stubs.b.c", "b", "c-stubs"])
    ['a', 'a.b', 'b']
    """
    return [pkg for pkg in packages if not pkg.split(".")[0].endswith("-stubs")]


def find_parent_package(
    packages: list[str], package_dir: Mapping[str, str], root_dir: StrPath
) -> str | None:
    """Find the parent package that is not a namespace."""
    packages = sorted(packages, key=len)
    common_ancestors = []
    for i, name in enumerate(packages):
        if not all(n.startswith(f"{name}.") for n in packages[i + 1 :]):
            # Since packages are sorted by length, this condition is able
            # to find a list of all common ancestors.
            # When there is divergence (e.g. multiple root packages)
            # the list will be empty
            break
        common_ancestors.append(name)

    for name in common_ancestors:
        pkg_path = find_package_path(name, package_dir, root_dir)
        init = os.path.join(pkg_path, "__init__.py")
        if os.path.isfile(init):
            return name

    return None


def find_package_path(
    name: str, package_dir: Mapping[str, str], root_dir: StrPath
) -> str:
    """Given a package name, return the path where it should be found on
    disk, considering the ``package_dir`` option.

    >>> path = find_package_path("my.pkg", {"": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested/my/pkg'

    >>> path = find_package_path("my.pkg", {"my": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested/pkg'

    >>> path = find_package_path("my.pkg", {"my.pkg": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested'

    >>> path = find_package_path("other.pkg", {"my.pkg": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './other/pkg'
    """
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        # Look backwards, the most specific package_dir first
        partial_name = ".".join(parts[:i])
        if partial_name in package_dir:
            parent = package_dir[partial_name]
            return os.path.join(root_dir, parent, *parts[i:])

    parent = package_dir.get("") or ""
    return os.path.join(root_dir, *parent.split("/"), *parts)


def construct_package_dir(packages: list[str], package_path: StrPath) -> dict[str, str]:
    parent_pkgs = remove_nested_packages(packages)
    prefix = Path(package_path).parts
    return {pkg: "/".join([*prefix, *pkg.split(".")]) for pkg in parent_pkgs}
<|endoftext|>import functools
import operator

import packaging.requirements


# from coherent.build.discovery
def extras_from_dep(dep):
    try:
        markers = packaging.requirements.Requirement(dep).marker._markers
    except AttributeError:
        markers = ()
    return {
        marker[2].value
        for marker in markers
        if isinstance(marker, tuple) and marker[0].value == 'extra'
    }


def extras_from_deps(deps):
    """
    >>> extras_from_deps(['requests'])
    set()
    >>> extras_from_deps(['pytest; extra == "test"'])
    {'test'}
    >>> sorted(extras_from_deps([
    ...     'requests',
    ...     'pytest; extra == "test"',
    ...     'pytest-cov; extra == "test"',
    ...     'sphinx; extra=="doc"']))
    ['doc', 'test']
    """
    return functools.reduce(operator.or_, map(extras_from_dep, deps), set())
<|endoftext|>"""Wheels support."""

import contextlib
import email
import functools
import itertools
import os
import posixpath
import re
import zipfile
from collections.abc import Iterator

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name
from packaging.version import Version as parse_version

import setuptools
from setuptools.archive_util import _unpack_zipfile_obj
from setuptools.command.egg_info import _egg_basename, write_requirements

from ._discovery import extras_from_deps
from ._importlib import metadata
from .unicode_utils import _read_utf8_with_fallback

from distutils.util import get_platform

WHEEL_NAME = re.compile(
    r"""^(?P<project_name>.+?)-(?P<version>\d.*?)
    ((-(?P<build>\d.*?))?-(?P<py_version>.+?)-(?P<abi>.+?)-(?P<platform>.+?)
    )\.whl$""",
    re.VERBOSE,
).match

NAMESPACE_PACKAGE_INIT = "__import__('pkg_resources').declare_namespace(__name__)\n"


@functools.cache
def _get_supported_tags():
    # We calculate the supported tags only once, otherwise calling
    # this method on thousands of wheels takes seconds instead of
    # milliseconds.
    return {(t.interpreter, t.abi, t.platform) for t in sys_tags()}


def unpack(src_dir, dst_dir) -> None:
    """Move everything under `src_dir` to `dst_dir`, and delete the former."""
    for dirpath, dirnames, filenames in os.walk(src_dir):
        subdir = os.path.relpath(dirpath, src_dir)
        for f in filenames:
            src = os.path.join(dirpath, f)
            dst = os.path.join(dst_dir, subdir, f)
            os.renames(src, dst)
        for n, d in reversed(list(enumerate(dirnames))):
            src = os.path.join(dirpath, d)
            dst = os.path.join(dst_dir, subdir, d)
            if not os.path.exists(dst):
                # Directory does not exist in destination,
                # rename it and prune it from os.walk list.
                os.renames(src, dst)
                del dirnames[n]
    # Cleanup.
    for dirpath, dirnames, filenames in os.walk(src_dir, topdown=True):
        assert not filenames
        os.rmdir(dirpath)


@contextlib.contextmanager
def disable_info_traces() -> Iterator[None]:
    """
    Temporarily disable info traces.
    """
    from distutils import log

    saved = log.set_threshold(log.WARN)
    try:
        yield
    finally:
        log.set_threshold(saved)


class Wheel:
    def __init__(self, filename) -> None:
        match = WHEEL_NAME(os.path.basename(filename))
        if match is None:
            raise ValueError(f'invalid wheel name: {filename!r}')
        self.filename = filename
        for k, v in match.groupdict().items():
            setattr(self, k, v)

    def tags(self):
        """List tags (py_version, abi, platform) supported by this wheel."""
        return itertools.product(
            self.py_version.split('.'),
            self.abi.split('.'),
            self.platform.split('.'),
        )

    def is_compatible(self):
        """Is the wheel compatible with the current platform?"""
        return next((True for t in self.tags() if t in _get_supported_tags()), False)

    def egg_name(self):
        return (
            _egg_basename(
                self.project_name,
                self.version,
                platform=(None if self.platform == 'any' else get_platform()),
            )
            + ".egg"
        )

    def get_dist_info(self, zf):
        # find the correct name of the .dist-info dir in the wheel file
        for member in zf.namelist():
            dirname = posixpath.dirname(member)
            if dirname.endswith('.dist-info') and canonicalize_name(dirname).startswith(
                canonicalize_name(self.project_name)
            ):
                return dirname
        raise ValueError("unsupported wheel format. .dist-info not found")

    def install_as_egg(self, destination_eggdir) -> None:
        """Install wheel as an egg directory."""
        with zipfile.ZipFile(self.filename) as zf:
            self._install_as_egg(destination_eggdir, zf)

    def _install_as_egg(self, destination_eggdir, zf):
        dist_basename = f'{self.project_name}-{self.version}'
        dist_info = self.get_dist_info(zf)
        dist_data = f'{dist_basename}.data'
        egg_info = os.path.join(destination_eggdir, 'EGG-INFO')

        self._convert_metadata(zf, destination_eggdir, dist_info, egg_info)
        self._move_data_entries(destination_eggdir, dist_data)
        self._fix_namespace_packages(egg_info, destination_eggdir)

    @staticmethod
    def _convert_metadata(zf, destination_eggdir, dist_info, egg_info):
        def get_metadata(name):
            with zf.open(posixpath.join(dist_info, name)) as fp:
                value = fp.read().decode('utf-8')
                return email.parser.Parser().parsestr(value)

        wheel_metadata = get_metadata('WHEEL')
        # Check wheel format version is supported.
        wheel_version = parse_version(wheel_metadata.get('Wheel-Version'))
        wheel_v1 = parse_version('1.0') <= wheel_version < parse_version('2.0dev0')
        if not wheel_v1:
            raise ValueError(f'unsupported wheel format version: {wheel_version}')
        # Extract to target directory.
        _unpack_zipfile_obj(zf, destination_eggdir)
        dist_info = os.path.join(destination_eggdir, dist_info)
        install_requires, extras_require = Wheel._convert_requires(
            destination_eggdir, dist_info
        )
        os.rename(dist_info, egg_info)
        os.rename(
            os.path.join(egg_info, 'METADATA'),
            os.path.join(egg_info, 'PKG-INFO'),
        )
        setup_dist = setuptools.Distribution(
            attrs=dict(
                install_requires=install_requires,
                extras_require=extras_require,
            ),
        )
        with disable_info_traces():
            write_requirements(
                setup_dist.get_command_obj('egg_info'),
                None,
                os.path.join(egg_info, 'requires.txt'),
            )

    @staticmethod
    def _convert_requires(destination_eggdir, dist_info):
        md = metadata.Distribution.at(dist_info).metadata
        deps = md.get_all('Requires-Dist') or []
        reqs = list(map(Requirement, deps))

        extras = extras_from_deps(deps)

        # Note: Evaluate and strip markers now,
        # as it's difficult to convert back from the syntax:
        # foobar; "linux" in sys_platform and extra == 'test'
        def raw_req(req):
            req = Requirement(str(req))
            req.marker = None
            return str(req)

        def eval(req, **env):
            return not req.marker or req.marker.evaluate(env)

        def for_extra(req):
            try:
                markers = req.marker._markers
            except AttributeError:
                markers = ()
            return {
                marker[2].value
                for marker in markers
                if isinstance(marker, tuple) and marker[0].value == 'extra'
            }

        install_requires = list(
            map(raw_req, filter(eval, itertools.filterfalse(for_extra, reqs)))
        )
        extras_require = {
            extra: list(
                map(
                    raw_req,
                    (req for req in reqs if for_extra(req) and eval(req, extra=extra)),
                )
            )
            for extra in extras
        }
        return install_requires, extras_require

    @staticmethod
    def _move_data_entries(destination_eggdir, dist_data):
        """Move data entries to their correct location."""
        dist_data = os.path.join(destination_eggdir, dist_data)
        dist_data_scripts = os.path.join(dist_data, 'scripts')
        if os.path.exists(dist_data_scripts):
            egg_info_scripts = os.path.join(destination_eggdir, 'EGG-INFO', 'scripts')
            os.mkdir(egg_info_scripts)
            for entry in os.listdir(dist_data_scripts):
                # Remove bytecode, as it's not properly handled
                # during easy_install scripts install phase.
                if entry.endswith('.pyc'):
                    os.unlink(os.path.join(dist_data_scripts, entry))
                else:
                    os.rename(
                        os.path.join(dist_data_scripts, entry),
                        os.path.join(egg_info_scripts, entry),
                    )
            os.rmdir(dist_data_scripts)
        for subdir in filter(
            os.path.exists,
            (
                os.path.join(dist_data, d)
                for d in ('data', 'headers', 'purelib', 'platlib')
            ),
        ):
            unpack(subdir, destination_eggdir)
        if os.path.exists(dist_data):
            os.rmdir(dist_data)

    @staticmethod
    def _fix_namespace_packages(egg_info, destination_eggdir):
        namespace_packages = os.path.join(egg_info, 'namespace_packages.txt')
        if os.path.exists(namespace_packages):
            namespace_packages = _read_utf8_with_fallback(namespace_packages).split()

            for mod in namespace_packages:
                mod_dir = os.path.join(destination_eggdir, *mod.split('.'))
                mod_init = os.path.join(mod_dir, '__init__.py')
                if not os.path.exists(mod_dir):
                    os.mkdir(mod_dir)
                if not os.path.exists(mod_init):
                    with open(mod_init, 'w', encoding="utf-8") as fp:
                        fp.write(NAMESPACE_PACKAGE_INIT)
<|endoftext|>from __future__ import annotations

import contextlib
import dis
import marshal
import sys
from types import CodeType
from typing import Any, Literal, TypeVar

from packaging.version import Version

from . import _imp
from ._imp import PY_COMPILED, PY_FROZEN, PY_SOURCE, find_module

_T = TypeVar("_T")

__all__ = ['Require', 'find_module']


class Require:
    """A prerequisite to building or installing a distribution"""

    def __init__(
        self,
        name,
        requested_version,
        module,
        homepage: str = '',
        attribute=None,
        format=None,
    ) -> None:
        if format is None and requested_version is not None:
            format = Version

        if format is not None:
            requested_version = format(requested_version)
            if attribute is None:
                attribute = '__version__'

        self.__dict__.update(locals())
        del self.self

    def full_name(self):
        """Return full package/distribution name, w/version"""
        if self.requested_version is not None:
            return f'{self.name}-{self.requested_version}'
        return self.name

    def version_ok(self, version):
        """Is 'version' sufficiently up-to-date?"""
        return (
            self.attribute is None
            or self.format is None
            or str(version) != "unknown"
            and self.format(version) >= self.requested_version
        )

    def get_version(
        self, paths=None, default: _T | Literal["unknown"] = "unknown"
    ) -> _T | Literal["unknown"] | None | Any:
        """Get version number of installed module, 'None', or 'default'

        Search 'paths' for module.  If not found, return 'None'.  If found,
        return the extracted version attribute, or 'default' if no version
        attribute was specified, or the value cannot be determined without
        importing the module.  The version is formatted according to the
        requirement's version format (if any), unless it is 'None' or the
        supplied 'default'.
        """

        if self.attribute is None:
            try:
                f, _p, _i = find_module(self.module, paths)
            except ImportError:
                return None
            if f:
                f.close()
            return default

        v = get_module_constant(self.module, self.attribute, default, paths)

        if v is not None and v is not default and self.format is not None:
            return self.format(v)

        return v

    def is_present(self, paths=None):
        """Return true if dependency is present on 'paths'"""
        return self.get_version(paths) is not None

    def is_current(self, paths=None):
        """Return true if dependency is present and up-to-date on 'paths'"""
        version = self.get_version(paths)
        if version is None:
            return False
        return self.version_ok(str(version))


def maybe_close(f):
    @contextlib.contextmanager
    def empty():
        yield

    if not f:
        return empty()

    return contextlib.closing(f)


# Some objects are not available on some platforms.
# XXX it'd be better to test assertions about bytecode instead.
if not sys.platform.startswith('java') and sys.platform != 'cli':

    def get_module_constant(
        module, symbol, default: _T | int = -1, paths=None
    ) -> _T | int | None | Any:
        """Find 'module' by searching 'paths', and extract 'symbol'

        Return 'None' if 'module' does not exist on 'paths', or it does not define
        'symbol'.  If the module defines 'symbol' as a constant, return the
        constant.  Otherwise, return 'default'."""

        try:
            f, path, (_suffix, _mode, kind) = info = find_module(module, paths)
        except ImportError:
            # Module doesn't exist
            return None

        with maybe_close(f):
            if kind == PY_COMPILED:
                f.read(8)  # skip magic & date
                code = marshal.load(f)
            elif kind == PY_FROZEN:
                code = _imp.get_frozen_object(module, paths)
            elif kind == PY_SOURCE:
                code = compile(f.read(), path, 'exec')
            else:
                # Not something we can parse; we'll have to import it.  :(
                imported = _imp.get_module(module, paths, info)
                return getattr(imported, symbol, None)

        return extract_constant(code, symbol, default)

    def extract_constant(
        code: CodeType, symbol: str, default: _T | int = -1
    ) -> _T | int | None | Any:
        """Extract the constant value of 'symbol' from 'code'

        If the name 'symbol' is bound to a constant value by the Python code
        object 'code', return that value.  If 'symbol' is bound to an expression,
        return 'default'.  Otherwise, return 'None'.

        Return value is based on the first assignment to 'symbol'.  'symbol' must
        be a global, or at least a non-"fast" local in the code block.  That is,
        only 'STORE_NAME' and 'STORE_GLOBAL' opcodes are checked, and 'symbol'
        must be present in 'code.co_names'.
        """
        if symbol not in code.co_names:
            # name's not there, can't possibly be an assignment
            return None

        name_idx = list(code.co_names).index(symbol)

        STORE_NAME = dis.opmap['STORE_NAME']
        STORE_GLOBAL = dis.opmap['STORE_GLOBAL']
        LOAD_CONST = dis.opmap['LOAD_CONST']

        const = default

        for byte_code in dis.Bytecode(code):
            op = byte_code.opcode
            arg = byte_code.arg

            if op == LOAD_CONST:
                assert arg is not None
                const = code.co_consts[arg]
            elif arg == name_idx and (op == STORE_NAME or op == STORE_GLOBAL):
                return const
            else:
                const = default

        return None

    __all__ += ['extract_constant', 'get_module_constant']
<|endoftext|>"""Extensions to the 'distutils' for large or complex distributions"""
# mypy: disable_error_code=override
# Command.reinitialize_command has an extra **kw param that distutils doesn't have
# Can't disable on the exact line because distutils doesn't exists on Python 3.12
# and mypy isn't aware of distutils_hack, causing distutils.core.Command to be Any,
# and a [unused-ignore] to be raised on 3.12+

from __future__ import annotations

import functools
import os
import sys
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar, overload

sys.path.extend(((vendor_path := os.path.join(os.path.dirname(os.path.dirname(__file__)), 'setuptools', '_vendor')) not in sys.path) * [vendor_path])  # fmt: skip
# workaround for #4476
sys.modules.pop('backports', None)

import _distutils_hack.override  # noqa: F401

from . import logging, monkey
from .depends import Require
from .discovery import PackageFinder, PEP420PackageFinder
from .dist import Distribution
from .extension import Extension
from .version import __version__ as __version__
from .warnings import SetuptoolsDeprecationWarning

import distutils.core

__all__ = [
    'Command',
    'Distribution',
    'Extension',
    'Require',
    'SetuptoolsDeprecationWarning',
    'find_namespace_packages',
    'find_packages',
    'setup',
]

_CommandT = TypeVar("_CommandT", bound="_Command")

bootstrap_install_from = None

find_packages = PackageFinder.find
find_namespace_packages = PEP420PackageFinder.find


def _install_setup_requires(attrs):
    # Note: do not use `setuptools.Distribution` directly, as
    # our PEP 517 backend patch `distutils.core.Distribution`.
    class MinimalDistribution(distutils.core.Distribution):
        """
        A minimal version of a distribution for supporting the
        fetch_build_eggs interface.
        """

        def __init__(self, attrs: Mapping[str, object]) -> None:
            _incl = 'dependency_links', 'setup_requires'
            filtered = {k: attrs[k] for k in set(_incl) & set(attrs)}
            super().__init__(filtered)
            # Prevent accidentally triggering discovery with incomplete set of attrs
            self.set_defaults._disable()

        def _get_project_config_files(self, filenames=None):
            """Ignore ``pyproject.toml``, they are not related to setup_requires"""
            try:
                cfg, _toml = super()._split_standard_project_metadata(filenames)
            except Exception:  # noqa: BLE001 # intentional broad fallback
                return filenames, ()
            return cfg, ()

        def finalize_options(self):
            """
            Disable finalize_options to avoid building the working set.
            Ref #2158.
            """

    dist = MinimalDistribution(attrs)

    # Honor setup.cfg's options.
    dist.parse_config_files(ignore_option_errors=True)
    if dist.setup_requires:
        _fetch_build_eggs(dist)


def _fetch_build_eggs(dist: Distribution):
    try:
        dist.fetch_build_eggs(dist.setup_requires)
    except Exception as ex:
        msg = """
        It is possible a package already installed in your system
        contains an version that is invalid according to PEP 440.
        You can try `pip install --use-pep517` as a workaround for this problem,
        or rely on a new virtual environment.

        If the problem refers to a package that is not installed yet,
        please contact that package's maintainers or distributors.
        """
        if "InvalidVersion" in ex.__class__.__name__:
            if hasattr(ex, "add_note"):
                ex.add_note(msg)  # PEP 678
            else:
                dist.announce(f"\n{msg}\n")
        raise


def setup(**attrs) -> Distribution:
    logging.configure()
    # Make sure we have any requirements needed to interpret 'attrs'.
    _install_setup_requires(attrs)
    # Override return type of distutils.core.Distribution with setuptools.dist.Distribution
    # (implicitly implemented via `setuptools.monkey.patch_all`).
    return distutils.core.setup(**attrs)  # type: ignore[return-value]


setup.__doc__ = distutils.core.setup.__doc__

if TYPE_CHECKING:
    # Work around a mypy issue where type[T] can't be used as a base: https://github.com/python/mypy/issues/10962
    from distutils.core import Command as _Command
else:
    _Command = monkey.get_unpatched(distutils.core.Command)


class Command(_Command):
    """
    Setuptools internal actions are organized using a *command design pattern*.
    This means that each action (or group of closely related actions) executed during
    the build should be implemented as a ``Command`` subclass.

    These commands are abstractions and do not necessarily correspond to a command that
    can (or should) be executed via a terminal, in a CLI fashion (although historically
    they would).

    When creating a new command from scratch, custom defined classes **SHOULD** inherit
    from ``setuptools.Command`` and implement a few mandatory methods.
    Between these mandatory methods, are listed:
    :meth:`initialize_options`, :meth:`finalize_options` and :meth:`run`.

    A useful analogy for command classes is to think of them as subroutines with local
    variables called "options".  The options are "declared" in :meth:`initialize_options`
    and "defined" (given their final values, aka "finalized") in :meth:`finalize_options`,
    both of which must be defined by every command class. The "body" of the subroutine,
    (where it does all the work) is the :meth:`run` method.
    Between :meth:`initialize_options` and :meth:`finalize_options`, ``setuptools`` may set
    the values for options/attributes based on user's input (or circumstance),
    which means that the implementation should be careful to not overwrite values in
    :meth:`finalize_options` unless necessary.

    Please note that other commands (or other parts of setuptools) may also overwrite
    the values of the command's options/attributes multiple times during the build
    process.
    Therefore it is important to consistently implement :meth:`initialize_options` and
    :meth:`finalize_options`. For example, all derived attributes (or attributes that
    depend on the value of other attributes) **SHOULD** be recomputed in
    :meth:`finalize_options`.

    When overwriting existing commands, custom defined classes **MUST** abide by the
    same APIs implemented by the original class. They also **SHOULD** inherit from the
    original class.
    """

    command_consumes_arguments = False
    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution

    dry_run = False  # type: ignore[assignment] # pyright: ignore[reportAssignmentType] (until #4689; see #4872)
    """
    For compatibility with vendored bdist_wheel.
    https://github.com/pypa/setuptools/pull/4872/files#r1986395142
    """

    def __init__(self, dist: Distribution, **kw) -> None:
        """
        Construct the command for dist, updating
        vars(self) with any keyword parameters.
        """
        super().__init__(dist)
        vars(self).update(kw)

    @overload
    def reinitialize_command(
        self, command: str, reinit_subcommands: bool = False, **kw
    ) -> Command: ...  # override distutils.cmd.Command with setuptools.Command
    @overload
    def reinitialize_command(
        self, command: _CommandT, reinit_subcommands: bool = False, **kw
    ) -> _CommandT: ...
    def reinitialize_command(
        self, command: str | _Command, reinit_subcommands: bool = False, **kw
    ) -> Command | _Command:
        cmd = _Command.reinitialize_command(self, command, reinit_subcommands)
        vars(cmd).update(kw)
        return cmd  # pyright: ignore[reportReturnType] # pypa/distutils#307

    @abstractmethod
    def initialize_options(self) -> None:
        """
        Set or (reset) all options/attributes/caches used by the command
        to their default values. Note that these values may be overwritten during
        the build.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize_options(self) -> None:
        """
        Set final values for all options/attributes used by the command.
        Most of the time, each option/attribute/cache should only be set if it does not
        have any value yet (e.g. ``if self.attr is None: self.attr = val``).
        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """
        Execute the actions intended by the command.
        (Side effects **SHOULD** only take place when :meth:`run` is executed,
        for example, creating new files or writing to the terminal output).
        """
        raise NotImplementedError


def _find_all_simple(path):
    """
    Find all files under 'path'
    """
    results = (
        os.path.join(base, file)
        for base, dirs, files in os.walk(path, followlinks=True)
        for file in files
    )
    return filter(os.path.isfile, results)


def findall(dir=os.curdir):
    """
    Find all files under 'dir' and return the list of full filenames.
    Unless dir is '.', return full filenames with dir prepended.
    """
    files = _find_all_simple(dir)
    if dir == os.curdir:
        make_rel = functools.partial(os.path.relpath, start=dir)
        files = map(make_rel, files)
    return list(files)


class sic(str):
    """Treat this string as-is (https://en.wikipedia.org/wiki/Sic)"""


# Apply monkey patches
monkey.patch_all()
<|endoftext|>from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar, overload

from jaraco import text
from packaging.requirements import Requirement

if TYPE_CHECKING:
    from typing import TypeAlias

_T = TypeVar("_T")
_StrOrIter: TypeAlias = str | Iterable[str]


parse_req: Callable[[str], Requirement] = lru_cache()(Requirement)
# Setuptools parses the same requirement many times
# (e.g. first for validation than for normalisation),
# so it might be worth to cache.


def parse_strings(strs: _StrOrIter) -> Iterator[str]:
    """
    Yield requirement strings for each specification in `strs`.

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    return text.join_continuation(map(text.drop_comment, text.yield_lines(strs)))


# These overloads are only needed because of a mypy false-positive, pyright gets it right
# https://github.com/python/mypy/issues/3737
@overload
def parse(strs: _StrOrIter) -> Iterator[Requirement]: ...
@overload
def parse(strs: _StrOrIter, parser: Callable[[str], _T]) -> Iterator[_T]: ...
def parse(strs: _StrOrIter, parser: Callable[[str], _T] = parse_req) -> Iterator[_T]:  # type: ignore[assignment]
    """
    Parse requirements.
    """
    return map(parser, parse_strings(strs))
<|endoftext|>"""Provide basic warnings used by setuptools modules.

Using custom classes (other than ``UserWarning``) allow users to set
``PYTHONWARNINGS`` filters to run tests and prepare for upcoming changes in
setuptools.
"""

from __future__ import annotations

import os
import warnings
from datetime import date
from inspect import cleandoc
from textwrap import indent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeAlias

_DueDate: TypeAlias = tuple[int, int, int]  # time tuple
_INDENT = 8 * " "
_TEMPLATE = f"""{80 * '*'}\n{{details}}\n{80 * '*'}"""


class SetuptoolsWarning(UserWarning):
    """Base class in ``setuptools`` warning hierarchy."""

    @classmethod
    def emit(
        cls,
        summary: str | None = None,
        details: str | None = None,
        due_date: _DueDate | None = None,
        see_docs: str | None = None,
        see_url: str | None = None,
        stacklevel: int = 2,
        **kwargs,
    ) -> None:
        """Private: reserved for ``setuptools`` internal use only"""
        # Default values:
        summary_ = summary or getattr(cls, "_SUMMARY", None) or ""
        details_ = details or getattr(cls, "_DETAILS", None) or ""
        due_date = due_date or getattr(cls, "_DUE_DATE", None)
        docs_ref = see_docs or getattr(cls, "_SEE_DOCS", None)
        docs_url = docs_ref and f"https://setuptools.pypa.io/en/latest/{docs_ref}"
        see_url = see_url or getattr(cls, "_SEE_URL", None)
        due = date(*due_date) if due_date else None

        text = cls._format(summary_, details_, due, see_url or docs_url, kwargs)
        if due and due < date.today() and _should_enforce():  # noqa: DTZ011 # local date intended for due-date comparison
            raise cls(text)
        warnings.warn(text, cls, stacklevel=stacklevel + 1)

    @classmethod
    def _format(
        cls,
        summary: str,
        details: str,
        due_date: date | None = None,
        see_url: str | None = None,
        format_args: dict | None = None,
    ) -> str:
        """Private: reserved for ``setuptools`` internal use only"""
        today = date.today()  # noqa: DTZ011 # local date intended for due-date comparison
        summary = cleandoc(summary).format_map(format_args or {})
        possible_parts = [
            cleandoc(details).format_map(format_args or {}),
            (
                f"\nBy {due_date:%Y-%b-%d}, you need to update your project and remove "
                "deprecated calls\nor your builds will no longer be supported."
                if due_date and due_date > today
                else None
            ),
            (
                "\nThis deprecation is overdue, please update your project and remove "
                "deprecated\ncalls to avoid build errors in the future."
                if due_date and due_date < today
                else None
            ),
            (f"\nSee {see_url} for details." if see_url else None),
        ]
        parts = [x for x in possible_parts if x]
        if parts:
            body = indent(_TEMPLATE.format(details="\n".join(parts)), _INDENT)
            return f"{summary}\n!!\n\n{body}\n\n!!"
        return summary


class InformationOnly(SetuptoolsWarning):
    """Currently there is no clear way of displaying messages to the users
    that use the setuptools backend directly via ``pip``.
    The only thing that might work is a warning, although it is not the
    most appropriate tool for the job...

    See pypa/packaging-problems#558.
    """


class SetuptoolsDeprecationWarning(SetuptoolsWarning):
    """
    Base class for warning deprecations in ``setuptools``

    This class is not derived from ``DeprecationWarning``, and as such is
    visible by default.
    """


def _should_enforce():
    enforce = os.getenv("SETUPTOOLS_ENFORCE_DEPRECATION", "false").lower()
    return enforce in ("true", "on", "ok", "1")
<|endoftext|>import sys
import unicodedata
from configparser import RawConfigParser

from .compat import py39
from .warnings import SetuptoolsDeprecationWarning


# HFS Plus uses decomposed UTF-8
def decompose(path):
    if isinstance(path, str):
        return unicodedata.normalize('NFD', path)
    try:
        path = path.decode('utf-8')
        path = unicodedata.normalize('NFD', path)
        path = path.encode('utf-8')
    except UnicodeError:
        pass  # Not UTF-8
    return path


def normalize(text):
    """
    Return *text* in a canonical Unicode form (NFC) so that names which are
    visually identical but encoded differently compare equal.

    macOS APFS/HFS+ store file names in decomposed form (NFD), while patterns
    in ``MANIFEST.in`` are typically authored composed (NFC). The two denote
    the same file but differ byte-for-byte, so matching them directly lets an
    exclusion silently fail. Normalizing both the walked path and the pattern
    to a single form before matching avoids that (GHSA-h35f-9h28-mq5c).
    """
    return unicodedata.normalize('NFC', text) if isinstance(text, str) else text


def filesys_decode(path):
    """
    Ensure that the given path is decoded,
    ``None`` when no expected encoding works
    """

    if isinstance(path, str):
        return path

    fs_enc = sys.getfilesystemencoding() or 'utf-8'
    candidates = fs_enc, 'utf-8'

    for enc in candidates:
        try:
            return path.decode(enc)
        except UnicodeDecodeError:
            continue

    return None


def try_encode(string, enc):
    "turn unicode encoding into a functional routine"
    try:
        return string.encode(enc)
    except UnicodeEncodeError:
        return None


def _read_utf8_with_fallback(file: str, fallback_encoding=py39.LOCALE_ENCODING) -> str:
    """
    First try to read the file with UTF-8, if there is an error fallback to a
    different encoding ("locale" by default). Returns the content of the file.
    Also useful when reading files that might have been produced by an older version of
    setuptools.
    """
    try:
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:  # pragma: no cover
        _Utf8EncodingNeeded.emit(file=file, fallback_encoding=fallback_encoding)
        with open(file, "r", encoding=fallback_encoding) as f:
            return f.read()


def _cfg_read_utf8_with_fallback(
    cfg: RawConfigParser, file: str, fallback_encoding=py39.LOCALE_ENCODING
) -> None:
    """Same idea as :func:`_read_utf8_with_fallback`, but for the
    :meth:`RawConfigParser.read` method.

    This method may call ``cfg.clear()``.
    """
    try:
        cfg.read(file, encoding="utf-8")
    except UnicodeDecodeError:  # pragma: no cover
        _Utf8EncodingNeeded.emit(file=file, fallback_encoding=fallback_encoding)
        cfg.clear()
        cfg.read(file, encoding=fallback_encoding)


class _Utf8EncodingNeeded(SetuptoolsDeprecationWarning):
    _SUMMARY = """
    `encoding="utf-8"` fails with {file!r}, trying `encoding={fallback_encoding!r}`.
    """

    _DETAILS = """
    Fallback behavior for UTF-8 is considered **deprecated** and future versions of
    `setuptools` may not implement it.

    Please encode {file!r} with "utf-8" to ensure future builds will succeed.

    If this file was produced by `setuptools` itself, cleaning up the cached files
    and re-building/re-installing the package with a newer version of `setuptools`
    (e.g. by updating `build-system.requires` in its `pyproject.toml`)
    might solve the problem.
    """
    # TODO: Add a deadline?
    #       Will we be able to remove this?
    #       The question comes to mind mainly because of sdists that have been produced
    #       by old versions of setuptools and published to PyPI...
<|endoftext|>"""Utilities for extracting common archive formats"""

import contextlib
import os
import posixpath
import shutil
import tarfile
import zipfile

from ._path import ensure_directory

from distutils.errors import DistutilsError

__all__ = [
    "UnrecognizedFormat",
    "default_filter",
    "extraction_drivers",
    "unpack_archive",
    "unpack_directory",
    "unpack_tarfile",
    "unpack_zipfile",
]


class UnrecognizedFormat(DistutilsError):
    """Couldn't recognize the archive type"""


def default_filter(src, dst):
    """The default progress/filter callback; returns True for all files"""
    return dst


def unpack_archive(
    filename, extract_dir, progress_filter=default_filter, drivers=None
) -> None:
    """Unpack `filename` to `extract_dir`, or raise ``UnrecognizedFormat``

    `progress_filter` is a function taking two arguments: a source path
    internal to the archive ('/'-separated), and a filesystem path where it
    will be extracted.  The callback must return the desired extract path
    (which may be the same as the one passed in), or else ``None`` to skip
    that file or directory.  The callback can thus be used to report on the
    progress of the extraction, as well as to filter the items extracted or
    alter their extraction paths.

    `drivers`, if supplied, must be a non-empty sequence of functions with the
    same signature as this function (minus the `drivers` argument), that raise
    ``UnrecognizedFormat`` if they do not support extracting the designated
    archive type.  The `drivers` are tried in sequence until one is found that
    does not raise an error, or until all are exhausted (in which case
    ``UnrecognizedFormat`` is raised).  If you do not supply a sequence of
    drivers, the module's ``extraction_drivers`` constant will be used, which
    means that ``unpack_zipfile`` and ``unpack_tarfile`` will be tried, in that
    order.
    """
    for driver in drivers or extraction_drivers:
        try:
            driver(filename, extract_dir, progress_filter)
        except UnrecognizedFormat:
            continue
        else:
            return
    raise UnrecognizedFormat(f"Not a recognized archive type: {filename}")


def unpack_directory(filename, extract_dir, progress_filter=default_filter) -> None:
    """ "Unpack" a directory, using the same interface as for archives

    Raises ``UnrecognizedFormat`` if `filename` is not a directory
    """
    if not os.path.isdir(filename):
        raise UnrecognizedFormat(f"{filename} is not a directory")

    paths = {
        filename: ('', extract_dir),
    }
    for base, dirs, files in os.walk(filename):
        src, dst = paths[base]
        for d in dirs:
            paths[os.path.join(base, d)] = src + d + '/', os.path.join(dst, d)
        for f in files:
            target = os.path.join(dst, f)
            target = progress_filter(src + f, target)
            if not target:
                # skip non-files
                continue
            ensure_directory(target)
            f = os.path.join(base, f)
            shutil.copyfile(f, target)
            shutil.copystat(f, target)


def unpack_zipfile(filename, extract_dir, progress_filter=default_filter) -> None:
    """Unpack zip `filename` to `extract_dir`

    Raises ``UnrecognizedFormat`` if `filename` is not a zipfile (as determined
    by ``zipfile.is_zipfile()``).  See ``unpack_archive()`` for an explanation
    of the `progress_filter` argument.
    """

    if not zipfile.is_zipfile(filename):
        raise UnrecognizedFormat(f"{filename} is not a zip file")

    with zipfile.ZipFile(filename) as z:
        _unpack_zipfile_obj(z, extract_dir, progress_filter)


def _unpack_zipfile_obj(zipfile_obj, extract_dir, progress_filter=default_filter):
    """Internal/private API used by other parts of setuptools.
    Similar to ``unpack_zipfile``, but receives an already opened :obj:`zipfile.ZipFile`
    object instead of a filename.
    """
    for info in zipfile_obj.infolist():
        name = info.filename

        # don't extract absolute paths or ones with .. in them
        if name.startswith('/') or '..' in name.split('/'):
            continue

        target = os.path.join(extract_dir, *name.split('/'))
        target = progress_filter(name, target)
        if not target:
            continue
        if name.endswith('/'):
            # directory
            ensure_directory(target)
        else:
            # file
            ensure_directory(target)
            data = zipfile_obj.read(info.filename)
            with open(target, 'wb') as f:
                f.write(data)
        unix_attributes = info.external_attr >> 16
        if unix_attributes:
            os.chmod(target, unix_attributes)


def _resolve_tar_file_or_dir(tar_obj, tar_member_obj):
    """Resolve any links and extract link targets as normal files."""
    while tar_member_obj is not None and (
        tar_member_obj.islnk() or tar_member_obj.issym()
    ):
        linkpath = tar_member_obj.linkname
        if tar_member_obj.issym():
            base = posixpath.dirname(tar_member_obj.name)
            linkpath = posixpath.join(base, linkpath)
            linkpath = posixpath.normpath(linkpath)
        tar_member_obj = tar_obj._getmember(linkpath)

    is_file_or_dir = tar_member_obj is not None and (
        tar_member_obj.isfile() or tar_member_obj.isdir()
    )
    if is_file_or_dir:
        return tar_member_obj

    raise LookupError('Got unknown file type')


def _iter_open_tar(tar_obj, extract_dir, progress_filter):
    """Emit member-destination pairs from a tar archive."""
    # don't do any chowning!
    tar_obj.chown = lambda *args: None

    with contextlib.closing(tar_obj):
        for member in tar_obj:
            name = member.name
            # don't extract absolute paths or ones with .. in them
            if name.startswith('/') or '..' in name.split('/'):
                continue

            prelim_dst = os.path.join(extract_dir, *name.split('/'))

            try:
                member = _resolve_tar_file_or_dir(tar_obj, member)
            except LookupError:
                continue

            final_dst = progress_filter(name, prelim_dst)
            if not final_dst:
                continue

            if final_dst.endswith(os.sep):
                final_dst = final_dst[:-1]

            yield member, final_dst


def unpack_tarfile(filename, extract_dir, progress_filter=default_filter) -> bool:
    """Unpack tar/tar.gz/tar.bz2 `filename` to `extract_dir`

    Raises ``UnrecognizedFormat`` if `filename` is not a tarfile (as determined
    by ``tarfile.open()``).  See ``unpack_archive()`` for an explanation
    of the `progress_filter` argument.
    """
    try:
        tarobj = tarfile.open(filename)  # noqa: SIM115 # handle managed explicitly
    except tarfile.TarError as e:
        raise UnrecognizedFormat(
            f"{filename} is not a compressed or uncompressed tar file"
        ) from e

    for member, final_dst in _iter_open_tar(
        tarobj,
        extract_dir,
        progress_filter,
    ):
        try:
            # XXX Ugh
            tarobj._extract_member(member, final_dst)
        except tarfile.ExtractError:
            # chown/chmod/mkfifo/mknode/makedev failed
            pass

    return True


extraction_drivers = unpack_directory, unpack_zipfile, unpack_tarfile
<|endoftext|>"""Convenience layer on top of stdlib's shutil and os"""

import os
import stat
from collections.abc import Callable
from typing import TypeVar

from .compat import py311

from distutils import log

try:
    from os import chmod  # pyright: ignore[reportAssignmentType]
    # Losing type-safety w/ pyright, but that's ok
except ImportError:  # pragma: no cover
    # Jython compatibility
    def chmod(*args: object, **kwargs: object) -> None:  # type: ignore[misc] # Mypy reuses the imported definition anyway
        pass


_T = TypeVar("_T")


def attempt_chmod_verbose(path, mode):
    log.debug("changing mode of %s to %o", path, mode)
    try:
        chmod(path, mode)
    except OSError as e:  # pragma: no cover
        log.debug("chmod failed: %s", e)


# Must match shutil._OnExcCallback
def _auto_chmod(
    func: Callable[..., _T], arg: str, exc: BaseException
) -> _T:  # pragma: no cover
    """shutils onexc callback to automatically call chmod for certain functions."""
    # Only retry for scenarios known to have an issue
    if func in [os.unlink, os.remove] and os.name == 'nt':
        attempt_chmod_verbose(arg, stat.S_IWRITE)
        return func(arg)
    raise exc


def rmtree(path, ignore_errors=False, onexc=_auto_chmod):
    """
    Similar to ``shutil.rmtree`` but automatically executes ``chmod``
    for well know Windows failure scenarios.
    """
    return py311.shutil_rmtree(path, ignore_errors, onexc)


def rmdir(path, **opts):
    if os.path.isdir(path):
        rmtree(path, **opts)


def current_umask():
    tmp = os.umask(0o022)
    os.umask(tmp)
    return tmp
<|endoftext|>from __future__ import annotations

import functools
import io
import itertools
import numbers
import os
import re
import sys
from collections.abc import Iterable, Iterator, MutableMapping, Sequence
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from more_itertools import partition, unique_everseen
from packaging.markers import InvalidMarker, Marker
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from . import (
    _entry_points,
    _reqs,
    _static,
    command as _,  # noqa: F401 # imported for side-effects
)
from ._importlib import metadata
from ._normalization import _canonicalize_license_expression
from ._path import StrPath
from ._reqs import _StrOrIter
from .config import pyprojecttoml, setupcfg
from .discovery import ConfigDiscovery
from .errors import InvalidConfigError
from .monkey import get_unpatched
from .warnings import InformationOnly, SetuptoolsDeprecationWarning

import distutils.cmd
import distutils.command
import distutils.core
import distutils.dist
import distutils.log
from distutils.debug import DEBUG
from distutils.errors import DistutilsOptionError, DistutilsSetupError
from distutils.fancy_getopt import translate_longopt
from distutils.util import strtobool

if TYPE_CHECKING:
    from typing import TypeAlias


__all__ = ['Distribution']

_sequence = tuple, list
"""
:meta private:

Supported iterable types that are known to be:
- ordered (which `set` isn't)
- not match a str (which `Sequence[str]` does)
- not imply a nested type (like `dict`)
for use with `isinstance`.
"""
_Sequence: TypeAlias = tuple[str, ...] | list[str]
# This is how stringifying _Sequence would look in Python 3.10
_sequence_type_repr = "tuple[str, ...] | list[str]"
_OrderedStrSequence: TypeAlias = str | dict[str, Any] | Sequence[str]
"""
:meta private:
Avoid single-use iterable. Disallow sets.
A poor approximation of an OrderedSequence (dict doesn't match a Sequence).
"""


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "sequence":
        SetuptoolsDeprecationWarning.emit(
            "`setuptools.dist.sequence` is an internal implementation detail.",
            "Please define your own `sequence = tuple, list` instead.",
            due_date=(2025, 8, 28),  # Originally added on 2024-08-27
        )
        return _sequence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def check_importable(dist, attr, value):
    try:
        ep = metadata.EntryPoint(value=value, name=None, group=None)
        assert not ep.extras
    except (TypeError, ValueError, AttributeError, AssertionError) as e:
        raise DistutilsSetupError(
            f"{attr!r} must be importable 'module:attrs' string (got {value!r})"
        ) from e


def assert_string_list(dist, attr: str, value: _Sequence) -> None:
    """Verify that value is a string list"""
    try:
        # verify that value is a list or tuple to exclude unordered
        # or single-use iterables
        assert isinstance(value, _sequence)
        # verify that elements of value are strings
        assert ''.join(value) != value
    except (TypeError, ValueError, AttributeError, AssertionError) as e:
        raise DistutilsSetupError(
            f"{attr!r} must be of type <{_sequence_type_repr}> (got {value!r})"
        ) from e


def check_nsp(dist, attr, value):
    """Verify that namespace packages are valid"""
    ns_packages = value
    assert_string_list(dist, attr, ns_packages)
    for nsp in ns_packages:
        if not dist.has_contents_for(nsp):
            raise DistutilsSetupError(
                f"Distribution contains no modules or packages for namespace package {nsp!r}"
            )
        parent, _sep, _child = nsp.rpartition('.')
        if parent and parent not in ns_packages:
            distutils.log.warn(
                "WARNING: %r is declared as a package namespace, but %r"
                " is not: please correct this in setup.py",
                nsp,
                parent,
            )
        SetuptoolsDeprecationWarning.emit(
            "The namespace_packages parameter is deprecated.",
            "Please replace its usage with implicit namespaces (PEP 420).",
            see_docs="references/keywords.html#keyword-namespace-packages",
            # TODO: define due_date, it may break old packages that are no longer
            # maintained (e.g. sphinxcontrib extensions) when installed from source.
            # Warning officially introduced in May 2022, however the deprecation
            # was mentioned much earlier in the docs (May 2020, see #2149).
        )


def check_extras(dist, attr, value):
    """Verify that extras_require mapping is valid"""
    try:
        list(itertools.starmap(_check_extra, value.items()))
    except (TypeError, ValueError, AttributeError) as e:
        raise DistutilsSetupError(
            "'extras_require' must be a dictionary whose values are "
            "strings or lists of strings containing valid project/version "
            "requirement specifiers."
        ) from e


def _check_extra(extra, reqs):
    _name, _sep, marker = extra.partition(':')
    try:
        _check_marker(marker)
    except InvalidMarker:
        msg = f"Invalid environment marker: {marker} ({extra!r})"
        raise DistutilsSetupError(msg) from None
    list(_reqs.parse(reqs))


def _check_marker(marker):
    if not marker:
        return
    m = Marker(marker)
    m.evaluate()


def assert_bool(dist, attr, value):
    """Verify that value is True, False, 0, or 1"""
    if bool(value) != value:
        raise DistutilsSetupError(f"{attr!r} must be a boolean value (got {value!r})")


def invalid_unless_false(dist, attr, value):
    if not value:
        DistDeprecationWarning.emit(f"{attr} is ignored.")
        # TODO: should there be a `due_date` here?
        return
    raise DistutilsSetupError(f"{attr} is invalid.")


def check_requirements(dist, attr: str, value: _OrderedStrSequence) -> None:
    """Verify that install_requires is a valid requirements list"""
    try:
        list(_reqs.parse(value))
        if isinstance(value, set):
            raise TypeError("Unordered types are not allowed")
    except (TypeError, ValueError) as error:
        msg = (
            f"{attr!r} must be a string or iterable of strings "
            f"containing valid project/version requirement specifiers; {error}"
        )
        raise DistutilsSetupError(msg) from error


def check_specifier(dist, attr, value):
    """Verify that value is a valid version specifier"""
    try:
        SpecifierSet(value)
    except (InvalidSpecifier, AttributeError) as error:
        msg = f"{attr!r} must be a string containing valid version specifiers; {error}"
        raise DistutilsSetupError(msg) from error


def check_entry_points(dist, attr, value):
    """Verify that entry_points map is parseable"""
    try:
        _entry_points.load(value)
    except Exception as e:
        raise DistutilsSetupError(e) from e


def check_package_data(dist, attr, value):
    """Verify that value is a dictionary of package names to glob lists"""
    if not isinstance(value, dict):
        raise DistutilsSetupError(
            f"{attr!r} must be a dictionary mapping package names to lists of "
            "string wildcard patterns"
        )
    for k, v in value.items():
        if not isinstance(k, str):
            raise DistutilsSetupError(
                f"keys of {attr!r} dict must be strings (got {k!r})"
            )
        assert_string_list(dist, f'values of {attr!r} dict', v)


def check_packages(dist, attr, value):
    for pkgname in value:
        if not re.match(r'\w+(\.\w+)*', pkgname):
            distutils.log.warn(
                "WARNING: %r not a valid package name; please use only "
                ".-separated package names in setup.py",
                pkgname,
            )


if TYPE_CHECKING:
    # Work around a mypy issue where type[T] can't be used as a base: https://github.com/python/mypy/issues/10962
    from distutils.core import Distribution as _Distribution
else:
    _Distribution = get_unpatched(distutils.core.Distribution)


class Distribution(_Distribution):
    """Distribution with support for tests and package data

    This is an enhanced version of 'distutils.dist.Distribution' that
    effectively adds the following new optional keyword arguments to 'setup()':

     'install_requires' -- a string or sequence of strings specifying project
        versions that the distribution requires when installed, in the format
        used by 'pkg_resources.require()'.  They will be installed
        automatically when the package is installed.  If you wish to use
        packages that are not available in PyPI, or want to give your users an
        alternate download location, you can add a 'find_links' option to the
        '[easy_install]' section of your project's 'setup.cfg' file, and then
        setuptools will scan the listed web pages for links that satisfy the
        requirements.

     'extras_require' -- a dictionary mapping names of optional "extras" to the
        additional requirement(s) that using those extras incurs. For example,
        this::

            extras_require = dict(reST = ["docutils>=0.3", "reSTedit"])

        indicates that the distribution can optionally provide an extra
        capability called "reST", but it can only be used if docutils and
        reSTedit are installed.  If the user installs your package using
        EasyInstall and requests one of your extras, the corresponding
        additional requirements will be installed if needed.

     'package_data' -- a dictionary mapping package names to lists of filenames
        or globs to use to find data files contained in the named packages.
        If the dictionary has filenames or globs listed under '""' (the empty
        string), those names will be searched for in every package, in addition
        to any names for the specific package.  Data files found using these
        names/globs will be installed along with the package, in the same
        location as the package.  Note that globs are allowed to reference
        the contents of non-package subdirectories, as long as you use '/' as
        a path separator.  (Globs are automatically converted to
        platform-specific paths at runtime.)

    In addition to these new keywords, this class also has several new methods
    for manipulating the distribution's contents.  For example, the 'include()'
    and 'exclude()' methods can be thought of as in-place add and subtract
    commands that add or remove packages, modules, extensions, and so on from
    the distribution.
    """

    _DISTUTILS_UNSUPPORTED_METADATA: ClassVar[dict] = {
        'long_description_content_type': lambda: None,
        'project_urls': dict,
        'provides_extras': dict,  # behaves like an ordered set
        'license_expression': lambda: None,
        'license_file': lambda: None,
        'license_files': lambda: None,
        'install_requires': list,
        'extras_require': dict,
    }

    # Used by build_py, editable_wheel and install_lib commands for legacy namespaces
    namespace_packages: list[str]  #: :meta private: DEPRECATED

    # Any: Dynamic assignment results in Incompatible types in assignment
    def __init__(self, attrs: MutableMapping[str, Any] | None = None) -> None:
        have_package_data = hasattr(self, "package_data")
        if not have_package_data:
            self.package_data: dict[str, list[str]] = {}
        attrs = attrs or {}
        self.dist_files: list[tuple[str, str, str]] = []
        self.include_package_data: bool | None = None
        self.exclude_package_data: dict[str, list[str]] | None = None
        # Filter-out setuptools' specific options.
        self.src_root: str | None = attrs.pop("src_root", None)
        self.dependency_links: list[str] = attrs.pop('dependency_links', [])
        self.setup_requires: list[str] = attrs.pop('setup_requires', [])
        for ep in metadata.entry_points(group='distutils.setup_keywords'):
            vars(self).setdefault(ep.name, None)

        metadata_only = set(self._DISTUTILS_UNSUPPORTED_METADATA)
        metadata_only -= {"install_requires", "extras_require"}
        dist_attrs = {k: v for k, v in attrs.items() if k not in metadata_only}
        _Distribution.__init__(self, dist_attrs)

        # Private API (setuptools-use only, not restricted to Distribution)
        # Stores files that are referenced by the configuration and need to be in the
        # sdist (e.g. `version = file: VERSION.txt`)
        self._referenced_files = set[str]()

        self.set_defaults = ConfigDiscovery(self)

        self._set_metadata_defaults(attrs)

        self.metadata.version = self._normalize_version(self.metadata.version)
        self._finalize_requires()

    def _validate_metadata(self):
        required = {"name"}
        provided = {
            key
            for key in vars(self.metadata)
            if getattr(self.metadata, key, None) is not None
        }
        missing = required - provided

        if missing:
            msg = f"Required package metadata is missing: {missing}"
            raise DistutilsSetupError(msg)

    def _set_metadata_defaults(self, attrs):
        """
        Fill-in missing metadata fields not supported by distutils.
        Some fields may have been set by other tools (e.g. pbr).
        Those fields (vars(self.metadata)) take precedence to
        supplied attrs.
        """
        for option, default in self._DISTUTILS_UNSUPPORTED_METADATA.items():
            vars(self.metadata).setdefault(option, attrs.get(option, default()))

    @staticmethod
    def _normalize_version(version):
        from . import sic

        if isinstance(version, numbers.Number):
            # Some people apparently take "version number" too literally :)
            version = str(version)
        elif isinstance(version, sic) or version is None:
            return version

        normalized = str(Version(version))
        if version != normalized:
            InformationOnly.emit(f"Normalizing '{version}' to '{normalized}'")
            return normalized
        return version

    def _finalize_requires(self):
        """
        Set `metadata.python_requires` and fix environment markers
        in `install_requires` and `extras_require`.
        """
        if getattr(self, 'python_requires', None):
            self.metadata.python_requires = self.python_requires

        self._normalize_requires()
        self.metadata.install_requires = self.install_requires
        self.metadata.extras_require = self.extras_require

        if self.extras_require:
            for extra in self.extras_require:
                # Setuptools allows a weird "<name>:<env markers> syntax for extras
                extra = extra.split(':')[0]
                if extra:
                    self.metadata.provides_extras.setdefault(extra)

    def _normalize_requires(self):
        """Make sure requirement-related attributes exist and are normalized"""
        install_requires = getattr(self, "install_requires", None) or []
        extras_require = getattr(self, "extras_require", None) or {}

        # Preserve the "static"-ness of values parsed from config files
        list_ = _static.List if _static.is_static(install_requires) else list
        self.install_requires = list_(map(str, _reqs.parse(install_requires)))

        dict_ = _static.Dict if _static.is_static(extras_require) else dict
        self.extras_require = dict_(
            (k, list(map(str, _reqs.parse(v or [])))) for k, v in extras_require.items()
        )

    def _finalize_license_expression(self) -> None:
        """
        Normalize license and license_expression.
        >>> dist = Distribution({"license_expression": _static.Str("mit aNd  gpl-3.0-OR-later")})
        >>> _static.is_static(dist.metadata.license_expression)
        True
        >>> dist._finalize_license_expression()
        >>> _static.is_static(dist.metadata.license_expression)  # preserve "static-ness"
        True
        >>> print(dist.metadata.license_expression)
        MIT AND GPL-3.0-or-later
        """
        classifiers = self.metadata.get_classifiers()
        license_classifiers = [cl for cl in classifiers if cl.startswith("License :: ")]

        license_expr = self.metadata.license_expression
        if license_expr:
            str_ = _static.Str if _static.is_static(license_expr) else str
            normalized = str_(_canonicalize_license_expression(license_expr))
            if license_expr != normalized:
                InformationOnly.emit(f"Normalizing '{license_expr}' to '{normalized}'")
                self.metadata.license_expression = normalized
            if license_classifiers:
                raise InvalidConfigError(
                    "License classifiers have been superseded by license expressions "
                    "(see https://peps.python.org/pep-0639/). Please remove:\n\n"
                    + "\n".join(license_classifiers),
                )
        elif license_classifiers:
            pypa_guides = "guides/writing-pyproject-toml/#license"
            SetuptoolsDeprecationWarning.emit(
                "License classifiers are deprecated.",
                "Please consider removing the following classifiers in favor of a "
                "SPDX license expression:\n\n" + "\n".join(license_classifiers),
                see_url=f"https://packaging.python.org/en/latest/{pypa_guides}",
                # Warning introduced on 2025-02-17
                # TODO: Should we add a due date? It may affect old/unmaintained
                #       packages in the ecosystem and cause problems...
            )

    def _finalize_license_files(self) -> None:
        """Compute names of all license files which should be included."""
        license_files: list[str] | None = self.metadata.license_files
        patterns = license_files or []

        license_file: str | None = self.metadata.license_file
        if license_file and license_file not in patterns:
            patterns.append(license_file)

        if license_files is None and license_file is None:
            # Default patterns match the ones wheel uses
            # See https://wheel.readthedocs.io/en/stable/user_guide.html
            # -> 'Including license files in the generated wheel file'
            patterns = ['LICEN[CS]E*', 'COPYING*', 'NOTICE*', 'AUTHORS*']
            files = self._expand_patterns(patterns, enforce_match=False)
        else:  # Patterns explicitly given by the user
            files = self._expand_patterns(patterns, enforce_match=True)

        self.metadata.license_files = list(unique_everseen(files))

    @classmethod
    def _expand_patterns(
        cls, patterns: list[str], enforce_match: bool = True
    ) -> Iterator[str]:
        """
        >>> getfixture('sample_project_cwd')
        >>> list(Distribution._expand_patterns(['LICENSE.txt']))
        ['LICENSE.txt']
        >>> list(Distribution._expand_patterns(['pyproject.toml', 'LIC*']))
        ['pyproject.toml', 'LICENSE.txt']
        >>> list(Distribution._expand_patterns(['src/**/*.dat']))
        ['src/sample/package_data.dat']
        """
        return (
            path.replace(os.sep, "/")
            for pattern in patterns
            for path in sorted(cls._find_pattern(pattern, enforce_match))
            if not path.endswith('~') and os.path.isfile(path)
        )

    @staticmethod
    def _find_pattern(pattern: str, enforce_match: bool = True) -> list[str]:
        r"""
        >>> getfixture('sample_project_cwd')
        >>> Distribution._find_pattern("LICENSE.txt")
        ['LICENSE.txt']
        >>> Distribution._find_pattern("/LICENSE.MIT")
        Traceback (most recent call last):
        ...
        setuptools.errors.InvalidConfigError: Pattern '/LICENSE.MIT' should be relative...
        >>> Distribution._find_pattern("../LICENSE.MIT")
        Traceback (most recent call last):
        ...
        setuptools.warnings.SetuptoolsDeprecationWarning: ...Pattern '../LICENSE.MIT' cannot contain '..'...
        >>> Distribution._find_pattern("LICEN{CSE*")
        Traceback (most recent call last):
        ...
        setuptools.warnings.SetuptoolsDeprecationWarning: ...Pattern 'LICEN{CSE*' contains invalid characters...
        """
        pypa_guides = "specifications/glob-patterns/"
        if ".." in pattern:
            SetuptoolsDeprecationWarning.emit(
                f"Pattern {pattern!r} cannot contain '..'",
                """
                Please ensure the files specified are contained by the root
                of the Python package (normally marked by `pyproject.toml`).
                """,
                see_url=f"https://packaging.python.org/en/latest/{pypa_guides}",
                due_date=(2027, 2, 18),  # Introduced in 2025-03-20
                # Replace with InvalidConfigError after deprecation
            )
        if pattern.startswith((os.sep, "/")) or ":\\" in pattern:
            raise InvalidConfigError(
                f"Pattern {pattern!r} should be relative and must not start with '/'"
            )
        if re.match(r'^[\w\-\.\/\*\?\[\]]+$', pattern) is None:
            SetuptoolsDeprecationWarning.emit(
                "Please provide a valid glob pattern.",
                "Pattern {pattern!r} contains invalid characters.",
                pattern=pattern,
                see_url=f"https://packaging.python.org/en/latest/{pypa_guides}",
                due_date=(2027, 2, 18),  # Introduced in 2025-02-20
            )

        found = glob(pattern, recursive=True)

        if enforce_match and not found:
            SetuptoolsDeprecationWarning.emit(
                "Cannot find any files for the given pattern.",
                "Pattern {pattern!r} did not match any files.",
                pattern=pattern,
                due_date=(2027, 2, 18),  # Introduced in 2025-02-20
                # PEP 639 requires us to error, but as a transition period
                # we will only issue a warning to give people time to prepare.
                # After the transition, this should raise an InvalidConfigError.
            )
        return found

    # FIXME: 'Distribution._parse_config_files' is too complex (14)
    def _parse_config_files(self, filenames=None):  # noqa: C901
        """
        Adapted from distutils.dist.Distribution.parse_config_files,
        this method provides the same functionality in subtly-improved
        ways.
        """
        from configparser import ConfigParser

        # Ignore install directory options if we have a venv
        ignore_options = (
            []
            if sys.prefix == sys.base_prefix
            else [
                'install-base',
                'install-platbase',
                'install-lib',
                'install-platlib',
                'install-purelib',
                'install-headers',
                'install-scripts',
                'install-data',
                'prefix',
                'exec-prefix',
                'home',
                'user',
                'root',
            ]
        )

        ignore_options = frozenset(ignore_options)

        if filenames is None:
            filenames = self.find_config_files()

        if DEBUG:
            self.announce("Distribution.parse_config_files():")

        parser = ConfigParser()
        parser.optionxform = str
        for filename in filenames:
            with open(filename, encoding='utf-8') as reader:
                if DEBUG:
                    self.announce("  reading {filename}".format(**locals()))
                parser.read_file(reader)
            for section in parser.sections():
                options = parser.options(section)
                opt_dict = self.get_option_dict(section)

                for opt in options:
                    if opt == '__name__' or opt in ignore_options:
                        continue

                    val = parser.get(section, opt)
                    opt = self._enforce_underscore(opt, section)
                    opt = self._enforce_option_lowercase(opt, section)
                    opt_dict[opt] = (filename, val)

            # Make the ConfigParser forget everything (so we retain
            # the original filenames that options come from)
            parser.__init__()

        if 'global' not in self.command_options:
            return

        # If there was a "global" section in the config file, use it
        # to set Distribution options.

        for opt, (src, val) in self.command_options['global'].items():
            alias = self.negative_opt.get(opt)
            if alias:
                val = not strtobool(val)
            elif opt == 'verbose':
                val = strtobool(val)

            try:
                setattr(self, alias or opt, val)
            except ValueError as e:
                raise DistutilsOptionError(e) from e

    def _enforce_underscore(self, opt: str, section: str) -> str:
        if "-" not in opt or self._skip_setupcfg_normalization(section):
            return opt

        underscore_opt = opt.replace('-', '_')
        affected = f"(Affected: {self.metadata.name})." if self.metadata.name else ""
        SetuptoolsDeprecationWarning.emit(
            f"Invalid dash-separated key {opt!r} in {section!r} (setup.cfg), "
            f"please use the underscore name {underscore_opt!r} instead.",
            f"""
            Usage of dash-separated {opt!r} will not be supported in future
            versions. Please use the underscore name {underscore_opt!r} instead.
            {affected}

            Available configuration options are listed in:
            https://setuptools.pypa.io/en/latest/userguide/declarative_config.html
            """,
            see_url="https://github.com/pypa/setuptools/discussions/5011",
            due_date=(2026, 3, 3),
            # Warning initially introduced in 3 Mar 2021
        )
        return underscore_opt

    def _enforce_option_lowercase(self, opt: str, section: str) -> str:
        if opt.islower() or self._skip_setupcfg_normalization(section):
            return opt

        lowercase_opt = opt.lower()
        affected = f"(Affected: {self.metadata.name})." if self.metadata.name else ""
        SetuptoolsDeprecationWarning.emit(
            f"Invalid uppercase key {opt!r} in {section!r} (setup.cfg), "
            f"please use lowercase {lowercase_opt!r} instead.",
            f"""
            Usage of uppercase key {opt!r} in {section!r} will not be supported in
            future versions. Please use lowercase {lowercase_opt!r} instead.
            {affected}

            Available configuration options are listed in:
            https://setuptools.pypa.io/en/latest/userguide/declarative_config.html
            """,
            see_url="https://github.com/pypa/setuptools/discussions/5011",
            due_date=(2026, 3, 3),
            # Warning initially introduced in 6 Mar 2021
        )
        return lowercase_opt

    def _skip_setupcfg_normalization(self, section: str) -> bool:
        skip = (
            'options.extras_require',
            'options.data_files',
            'options.entry_points',
            'options.package_data',
            'options.exclude_package_data',
        )
        return section in skip or not self._is_setuptools_section(section)

    def _is_setuptools_section(self, section: str) -> bool:
        return (
            section == "metadata"
            or section.startswith("options")
            or section in _setuptools_commands()
        )

    # FIXME: 'Distribution._set_command_options' is too complex (14)
    def _set_command_options(self, command_obj, option_dict=None):  # noqa: C901
        """
        Set the options for 'command_obj' from 'option_dict'.  Basically
        this means copying elements of a dictionary ('option_dict') to
        attributes of an instance ('command').

        'command_obj' must be a Command instance.  If 'option_dict' is not
        supplied, uses the standard option dictionary for this command
        (from 'self.command_options').

        (Adopted from distutils.dist.Distribution._set_command_options)
        """
        command_name = command_obj.get_command_name()
        if option_dict is None:
            option_dict = self.get_option_dict(command_name)

        if DEBUG:
            self.announce(f"  setting options for '{command_name}' command:")
        for option, (source, value) in option_dict.items():
            if DEBUG:
                self.announce(f"    {option} = {value} (from {source})")
            try:
                bool_opts = [translate_longopt(o) for o in command_obj.boolean_options]
            except AttributeError:
                bool_opts = []
            try:
                neg_opt = command_obj.negative_opt
            except AttributeError:
                neg_opt = {}

            try:
                is_string = isinstance(value, str)
                if option in neg_opt and is_string:
                    setattr(command_obj, neg_opt[option], not strtobool(value))
                elif option in bool_opts and is_string:
                    setattr(command_obj, option, strtobool(value))
                elif hasattr(command_obj, option):
                    setattr(command_obj, option, value)
                else:
                    raise DistutilsOptionError(
                        f"error in {source}: command '{command_name}' has no such option '{option}'"
                    )
            except ValueError as e:
                raise DistutilsOptionError(e) from e

    def _get_project_config_files(self, filenames: Iterable[StrPath] | None):
        """Add default file and split between INI and TOML"""
        tomlfiles = []
        standard_project_metadata = Path(self.src_root or os.curdir, "pyproject.toml")
        if filenames is not None:
            parts = partition(lambda f: Path(f).suffix == ".toml", filenames)
            filenames = list(parts[0])  # 1st element => predicate is False
            tomlfiles = list(parts[1])  # 2nd element => predicate is True
        elif standard_project_metadata.exists():
            tomlfiles = [standard_project_metadata]
        return filenames, tomlfiles

    def parse_config_files(
        self,
        filenames: Iterable[StrPath] | None = None,
        ignore_option_errors: bool = False,
    ) -> None:
        """Parses configuration files from various levels
        and loads configuration.
        """
        inifiles, tomlfiles = self._get_project_config_files(filenames)

        self._parse_config_files(filenames=inifiles)

        setupcfg.parse_configuration(
            self, self.command_options, ignore_option_errors=ignore_option_errors
        )
        for filename in tomlfiles:
            pyprojecttoml.apply_configuration(self, filename, ignore_option_errors)

        self._finalize_requires()
        self._finalize_license_expression()
        self._finalize_license_files()

    def fetch_build_eggs(self, requires: _StrOrIter) -> list[metadata.Distribution]:
        """Resolve pre-setup requirements"""
        from .installer import _fetch_build_eggs

        return _fetch_build_eggs(self, requires)

    def finalize_options(self) -> None:
        """
        Allow plugins to apply arbitrary operations to the
        distribution. Each hook may optionally define a 'order'
        to influence the order of execution. Smaller numbers
        go first and the default is 0.
        """
        group = 'setuptools.finalize_distribution_options'

        def by_order(hook):
            return getattr(hook, 'order', 0)

        defined = metadata.entry_points(group=group)
        filtered = itertools.filterfalse(self._removed, defined)
        loaded = (e.load() for e in filtered)
        for ep in sorted(loaded, key=by_order):
            ep(self)

    @staticmethod
    def _removed(ep):
        """
        When removing an entry point, if metadata is loaded
        from an older version of Setuptools, that removed
        entry point will attempt to be loaded and will fail.
        See #2765 for more details.
        """
        removed = {
            # removed 2021-09-05
            '2to3_doctests',
        }
        return ep.name in removed

    def _finalize_setup_keywords(self):
        for ep in metadata.entry_points(group='distutils.setup_keywords'):
            value = getattr(self, ep.name, None)
            if value is not None:
                ep.load()(self, ep.name, value)

    def get_egg_cache_dir(self) -> str:
        from . import windows_support

        egg_cache_dir = os.path.join(os.curdir, '.eggs')
        if not os.path.exists(egg_cache_dir):
            os.mkdir(egg_cache_dir)
            windows_support.hide_file(egg_cache_dir)
            readme_txt_filename = os.path.join(egg_cache_dir, 'README.txt')
            with open(readme_txt_filename, 'w', encoding="utf-8") as f:
                f.write(
                    'This directory contains eggs that were downloaded '
                    'by setuptools to build, test, and run plug-ins.\n\n'
                )
                f.write(
                    'This directory caches those eggs to prevent '
                    'repeated downloads.\n\n'
                )
                f.write('However, it is safe to delete this directory.\n\n')

        return egg_cache_dir

    def fetch_build_egg(self, req):
        """Fetch an egg needed for building"""
        from .installer import fetch_build_egg

        return fetch_build_egg(self, req)

    def get_command_class(self, command: str) -> type[distutils.cmd.Command]:  # type: ignore[override] # Not doing complex overrides yet
        """Pluggable version of get_command_class()"""
        if command in self.cmdclass:
            return self.cmdclass[command]

        # Special case bdist_wheel so it's never loaded from "wheel"
        if command == 'bdist_wheel':
            from .command.bdist_wheel import bdist_wheel

            return bdist_wheel

        eps = metadata.entry_points(group='distutils.commands', name=command)
        for ep in eps:
            self.cmdclass[command] = cmdclass = ep.load()
            return cmdclass
        return _Distribution.get_command_class(self, command)

    def print_commands(self):
        for ep in metadata.entry_points(group='distutils.commands'):
            if ep.name not in self.cmdclass:
                cmdclass = ep.load()
                self.cmdclass[ep.name] = cmdclass
        return _Distribution.print_commands(self)

    def get_command_list(self):
        for ep in metadata.entry_points(group='distutils.commands'):
            if ep.name not in self.cmdclass:
                cmdclass = ep.load()
                self.cmdclass[ep.name] = cmdclass
        return _Distribution.get_command_list(self)

    def include(self, **attrs) -> None:
        """Add items to distribution that are named in keyword arguments

        For example, 'dist.include(py_modules=["x"])' would add 'x' to
        the distribution's 'py_modules' attribute, if it was not already
        there.

        Currently, this method only supports inclusion for attributes that are
        lists or tuples.  If you need to add support for adding to other
        attributes in this or a subclass, you can add an '_include_X' method,
        where 'X' is the name of the attribute.  The method will be called with
        the value passed to 'include()'.  So, 'dist.include(foo={"bar":"baz"})'
        will try to call 'dist._include_foo({"bar":"baz"})', which can then
        handle whatever special inclusion logic is needed.
        """
        for k, v in attrs.items():
            include = getattr(self, '_include_' + k, None)
            if include:
                include(v)
            else:
                self._include_misc(k, v)

    def exclude_package(self, package: str) -> None:
        """Remove packages, modules, and extensions in named package"""

        pfx = package + '.'
        if self.packages:
            self.packages = [
                p for p in self.packages if p != package and not p.startswith(pfx)
            ]

        if self.py_modules:
            self.py_modules = [
                p for p in self.py_modules if p != package and not p.startswith(pfx)
            ]

        if self.ext_modules:
            self.ext_modules = [
                p
                for p in self.ext_modules
                if p.name != package and not p.name.startswith(pfx)
            ]

    def has_contents_for(self, package: str) -> bool:
        """Return true if 'exclude_package(package)' would do something"""

        pfx = package + '.'

        for p in self.iter_distribution_names():
            if p == package or p.startswith(pfx):
                return True

        return False

    def _exclude_misc(self, name: str, value: _Sequence) -> None:
        """Handle 'exclude()' for list/tuple attrs without a special handler"""
        if not isinstance(value, _sequence):
            raise DistutilsSetupError(
                f"{name}: setting must be of type <{_sequence_type_repr}> (got {value!r})"
            )
        try:
            old = getattr(self, name)
        except AttributeError as e:
            raise DistutilsSetupError(f"{name}: No such distribution setting") from e
        if old is not None and not isinstance(old, _sequence):
            raise DistutilsSetupError(
                name + ": this setting cannot be changed via include/exclude"
            )
        elif old:
            setattr(self, name, [item for item in old if item not in value])

    def _include_misc(self, name: str, value: _Sequence) -> None:
        """Handle 'include()' for list/tuple attrs without a special handler"""

        if not isinstance(value, _sequence):
            raise DistutilsSetupError(
                f"{name}: setting must be of type <{_sequence_type_repr}> (got {value!r})"
            )
        try:
            old = getattr(self, name)
        except AttributeError as e:
            raise DistutilsSetupError(f"{name}: No such distribution setting") from e
        if old is None:
            setattr(self, name, value)
        elif not isinstance(old, _sequence):
            raise DistutilsSetupError(
                name + ": this setting cannot be changed via include/exclude"
            )
        else:
            new = [item for item in value if item not in old]
            setattr(self, name, list(old) + new)

    def exclude(self, **attrs) -> None:
        """Remove items from distribution that are named in keyword arguments

        For example, 'dist.exclude(py_modules=["x"])' would remove 'x' from
        the distribution's 'py_modules' attribute.  Excluding packages uses
        the 'exclude_package()' method, so all of the package's contained
        packages, modules, and extensions are also excluded.

        Currently, this method only supports exclusion from attributes that are
        lists or tuples.  If you need to add support for excluding from other
        attributes in this or a subclass, you can add an '_exclude_X' method,
        where 'X' is the name of the attribute.  The method will be called with
        the value passed to 'exclude()'.  So, 'dist.exclude(foo={"bar":"baz"})'
        will try to call 'dist._exclude_foo({"bar":"baz"})', which can then
        handle whatever special exclusion logic is needed.
        """
        for k, v in attrs.items():
            exclude = getattr(self, '_exclude_' + k, None)
            if exclude:
                exclude(v)
            else:
                self._exclude_misc(k, v)

    def _exclude_packages(self, packages: _Sequence) -> None:
        if not isinstance(packages, _sequence):
            raise DistutilsSetupError(
                f"packages: setting must be of type <{_sequence_type_repr}> (got {packages!r})"
            )
        list(map(self.exclude_package, packages))

    def _parse_command_opts(self, parser, args):
        # Remove --with-X/--without-X options when processing command args
        self.global_options = self.__class__.global_options
        self.negative_opt = self.__class__.negative_opt

        # First, expand any aliases
        command = args[0]
        aliases = self.get_option_dict('aliases')
        while command in aliases:
            _src, alias = aliases[command]
            del aliases[command]  # ensure each alias can expand only once!
            import shlex

            args[:1] = shlex.split(alias, True)
            command = args[0]

        nargs = _Distribution._parse_command_opts(self, parser, args)

        # Handle commands that want to consume all remaining arguments
        cmd_class = self.get_command_class(command)
        if getattr(cmd_class, 'command_consumes_arguments', None):
            self.get_option_dict(command)['args'] = ("command line", nargs)
            if nargs is not None:
                return []

        return nargs

    def get_cmdline_options(self) -> dict[str, dict[str, str | None]]:
        """Return a '{cmd: {opt:val}}' map of all command-line options

        Option names are all long, but do not include the leading '--', and
        contain dashes rather than underscores.  If the option doesn't take
        an argument (e.g. '--quiet'), the 'val' is 'None'.

        Note that options provided by config files are intentionally excluded.
        """

        d: dict[str, dict[str, str | None]] = {}

        for cmd, opts in self.command_options.items():
            val: str | None
            for opt, (src, val) in opts.items():
                if src != "command line":
                    continue

                opt = opt.replace('_', '-')

                if val == 0:
                    cmdobj = self.get_command_obj(cmd)
                    neg_opt = self.negative_opt.copy()
                    neg_opt.update(getattr(cmdobj, 'negative_opt', {}))
                    for neg, pos in neg_opt.items():
                        if pos == opt:
                            opt = neg
                            val = None
                            break
                    else:
                        raise AssertionError("Shouldn't be able to get here")

                elif val == 1:
                    val = None

                d.setdefault(cmd, {})[opt] = val

        return d

    def iter_distribution_names(self) -> Iterator[str]:
        """Yield all packages, modules, and extension names in distribution"""

        yield from self.packages or ()

        yield from self.py_modules or ()

        for ext in self.ext_modules or ():
            if isinstance(ext, tuple):
                name, _buildinfo = ext
            else:
                name = ext.name
            name = name.removesuffix('module')
            yield name

    def handle_display_options(self, option_order):
        """If there were any non-global "display-only" options
        (--help-commands or the metadata display options) on the command
        line, display the requested info and return true; else return
        false.
        """
        import sys

        if self.help_commands:
            return _Distribution.handle_display_options(self, option_order)

        # Stdout may be StringIO (e.g. in tests)
        if not isinstance(sys.stdout, io.TextIOWrapper):
            return _Distribution.handle_display_options(self, option_order)

        # Don't wrap stdout if utf-8 is already the encoding. Provides
        #  workaround for #334.
        if sys.stdout.encoding.lower() in ('utf-8', 'utf8'):
            return _Distribution.handle_display_options(self, option_order)

        # Print metadata in UTF-8 no matter the platform
        encoding = sys.stdout.encoding
        sys.stdout.reconfigure(encoding='utf-8')
        try:
            return _Distribution.handle_display_options(self, option_order)
        finally:
            sys.stdout.reconfigure(encoding=encoding)

    def run_command(self, command) -> None:
        self.set_defaults()
        # Postpone defaults until all explicit configuration is considered
        # (setup() args, config files, command line and plugins)

        super().run_command(command)


@functools.cache
def _setuptools_commands() -> set[str]:
    try:
        # Use older API for importlib.metadata compatibility
        entry_points = metadata.distribution('setuptools').entry_points
        eps: Iterable[str] = (ep.name for ep in entry_points)
    except metadata.PackageNotFoundError:
        # during bootstrapping, distribution doesn't exist
        eps = []
    return {*distutils.command.__all__, *eps}


class DistDeprecationWarning(SetuptoolsDeprecationWarning):
    """Class for warning about deprecations in dist in
    setuptools. Not ignored by default, unlike DeprecationWarning."""
<|endoftext|>from __future__ import annotations

import glob
import itertools
import os
import subprocess
import sys
import tempfile

import packaging.requirements
import packaging.utils

from . import _reqs
from ._importlib import metadata
from .warnings import SetuptoolsDeprecationWarning
from .wheel import Wheel

from distutils import log
from distutils.errors import DistutilsError


def _fixup_find_links(find_links):
    """Ensure find-links option end-up being a list of strings."""
    if isinstance(find_links, str):
        return find_links.split()
    assert isinstance(find_links, (tuple, list))
    return find_links


def fetch_build_egg(dist, req) -> metadata.Distribution | metadata.PathDistribution:
    """Fetch an egg needed for building.

    Use pip/wheel to fetch/build a wheel."""
    _DeprecatedInstaller.emit()
    _warn_wheel_not_available(dist)
    return _fetch_build_egg_no_warn(dist, req)


def _present(req):
    return any(_dist_matches_req(dist, req) for dist in metadata.distributions())


def _fetch_build_eggs(dist, requires: _reqs._StrOrIter) -> list[metadata.Distribution]:
    _DeprecatedInstaller.emit(stacklevel=3)
    _warn_wheel_not_available(dist)

    parsed_reqs = _reqs.parse(requires)

    missing_reqs = itertools.filterfalse(_present, parsed_reqs)

    needed_reqs = (
        req for req in missing_reqs if not req.marker or req.marker.evaluate()
    )
    resolved_dists = [_fetch_build_egg_no_warn(dist, req) for req in needed_reqs]
    for resolved_dist in resolved_dists:
        # locate_file('') is the directory containing EGG-INFO, where the importable
        # contents can be found.
        sys.path.insert(0, str(resolved_dist.locate_file('')))
    return resolved_dists


def _dist_matches_req(egg_dist, req):
    return (
        packaging.utils.canonicalize_name(egg_dist.name)
        == packaging.utils.canonicalize_name(req.name)
        and egg_dist.version in req.specifier
    )


def _fetch_build_egg_no_warn(dist, req):  # noqa: C901  # is too complex (16)  # FIXME
    # Ignore environment markers; if supplied, it is required.
    req = strip_marker(req)
    # Take easy_install options into account, but do not override relevant
    # pip environment variables (like PIP_INDEX_URL or PIP_QUIET); they'll
    # take precedence.
    opts = dist.get_option_dict('easy_install')
    if 'allow_hosts' in opts:
        raise DistutilsError(
            'the `allow-hosts` option is not supported '
            'when using pip to install requirements.'
        )
    quiet = 'PIP_QUIET' not in os.environ and 'PIP_VERBOSE' not in os.environ
    if 'PIP_INDEX_URL' in os.environ:
        index_url = None
    elif 'index_url' in opts:
        index_url = opts['index_url'][1]
    else:
        index_url = None
    find_links = (
        _fixup_find_links(opts['find_links'][1])[:] if 'find_links' in opts else []
    )
    if dist.dependency_links:
        find_links.extend(dist.dependency_links)
    eggs_dir = os.path.realpath(dist.get_egg_cache_dir())
    cached_dists = metadata.Distribution.discover(path=glob.glob(f'{eggs_dir}/*.egg'))
    for egg_dist in cached_dists:
        if _dist_matches_req(egg_dist, req):
            return egg_dist
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            '-m',
            'pip',
            '--disable-pip-version-check',
            'wheel',
            '--no-deps',
            '-w',
            tmpdir,
        ]
        if quiet:
            cmd.append('--quiet')
        if index_url is not None:
            cmd.extend(('--index-url', index_url))
        for link in find_links or []:
            cmd.extend(('--find-links', link))
        # If requirement is a PEP 508 direct URL, directly pass
        # the URL to pip, as `req @ url` does not work on the
        # command line.
        cmd.append(req.url or str(req))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            raise DistutilsError(str(e)) from e
        wheel = Wheel(glob.glob(os.path.join(tmpdir, '*.whl'))[0])
        dist_location = os.path.join(eggs_dir, wheel.egg_name())
        wheel.install_as_egg(dist_location)
        return metadata.Distribution.at(dist_location + '/EGG-INFO')


def strip_marker(req) -> packaging.requirements.Requirement:
    """
    Return a new requirement without the environment marker to avoid
    calling pip with something like `babel; extra == "i18n"`, which
    would always be ignored.
    """
    # create a copy to avoid mutating the input
    req = packaging.requirements.Requirement(str(req))
    req.marker = None
    return req


def _warn_wheel_not_available(dist):
    try:
        metadata.distribution('wheel')
    except metadata.PackageNotFoundError:
        dist.announce('WARNING: The wheel package is not available.', log.WARN)


class _DeprecatedInstaller(SetuptoolsDeprecationWarning):
    _SUMMARY = "setuptools.installer and fetch_build_eggs are deprecated."
    _DETAILS = """
    Requirements should be satisfied by a PEP 517 installer.
    If you are using pip, you can try `pip install --use-pep517`.
    """
    _DUE_DATE = 2025, 10, 31
<|endoftext|>try:
    # Ensure a DistutilsError raised by these methods is the same as distutils.errors.DistutilsError
    from distutils._modified import (
        newer,
        newer_group,
        newer_pairwise,
        newer_pairwise_group,
    )
except ImportError:
    # fallback for SETUPTOOLS_USE_DISTUTILS=stdlib, because _modified never existed in stdlib
    from ._distutils._modified import (
        newer,
        newer_group,
        newer_pairwise,
        newer_pairwise_group,
    )

__all__ = ['newer', 'newer_group', 'newer_pairwise', 'newer_pairwise_group']
<|endoftext|>"""
Re-implementation of find_module and get_frozen_object
from the deprecated imp module.
"""

import importlib.machinery
import importlib.util
import os
import tokenize
from importlib.util import module_from_spec

PY_SOURCE = 1
PY_COMPILED = 2
C_EXTENSION = 3
C_BUILTIN = 6
PY_FROZEN = 7


def find_spec(module, paths):
    finder = (
        importlib.machinery.PathFinder().find_spec
        if isinstance(paths, list)
        else importlib.util.find_spec
    )
    return finder(module, paths)


def find_module(module, paths=None):
    """Just like 'imp.find_module()', but with package support"""
    spec = find_spec(module, paths)
    if spec is None:
        raise ImportError(f"Can't find {module}")
    if not spec.has_location and hasattr(spec, 'submodule_search_locations'):
        spec = importlib.util.spec_from_loader('__init__.py', spec.loader)

    kind = -1
    file = None
    static = isinstance(spec.loader, type)
    if (
        spec.origin == 'frozen'
        or static
        and issubclass(spec.loader, importlib.machinery.FrozenImporter)
    ):
        kind = PY_FROZEN
        path = None  # imp compabilty
        suffix = mode = ''  # imp compatibility
    elif (
        spec.origin == 'built-in'
        or static
        and issubclass(spec.loader, importlib.machinery.BuiltinImporter)
    ):
        kind = C_BUILTIN
        path = None  # imp compabilty
        suffix = mode = ''  # imp compatibility
    elif spec.has_location:
        path = spec.origin
        suffix = os.path.splitext(path)[1]
        mode = 'r' if suffix in importlib.machinery.SOURCE_SUFFIXES else 'rb'

        if suffix in importlib.machinery.SOURCE_SUFFIXES:
            kind = PY_SOURCE
            file = tokenize.open(path)  # noqa: SIM115 # handle managed explicitly
        elif suffix in importlib.machinery.BYTECODE_SUFFIXES:
            kind = PY_COMPILED
            file = open(path, 'rb')  # noqa: SIM115 # handle managed explicitly
        elif suffix in importlib.machinery.EXTENSION_SUFFIXES:
            kind = C_EXTENSION

    else:
        path = None
        suffix = mode = ''

    return file, path, (suffix, mode, kind)


def get_frozen_object(module, paths=None):
    spec = find_spec(module, paths)
    if not spec:
        raise ImportError(f"Can't find {module}")
    return spec.loader.get_code(module)


def get_module(module, paths, info):
    spec = find_spec(module, paths)
    if not spec:
        raise ImportError(f"Can't find {module}")
    return module_from_spec(spec)
<|endoftext|>from more_itertools import consume  # noqa: F401


# copied from jaraco.itertools 6.1
def ensure_unique(iterable, key=lambda x: x):
    """
    Wrap an iterable to raise a ValueError if non-unique values are encountered.

    >>> list(ensure_unique('abc'))
    ['a', 'b', 'c']
    >>> consume(ensure_unique('abca'))
    Traceback (most recent call last):
    ...
    ValueError: Duplicate element 'a' encountered.
    """
    seen = set()
    seen_add = seen.add
    for element in iterable:
        k = key(element)
        if k in seen:
            raise ValueError(f"Duplicate element {element!r} encountered.")
        seen_add(k)
        yield element
<|endoftext|>"""
Environment info about Microsoft Compilers.

>>> getfixture('windows_only')
>>> ei = EnvironmentInfo('amd64')
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import os.path
import platform
from typing import TYPE_CHECKING, TypedDict, overload

from more_itertools import unique_everseen

from ._path import StrPath
from .compat import py310

import distutils.errors

if TYPE_CHECKING:
    from typing_extensions import LiteralString, NotRequired

# https://github.com/python/mypy/issues/8166
if not TYPE_CHECKING and platform.system() == 'Windows':
    import winreg
    from os import environ
else:
    # Mock winreg and environ so the module can be imported on this platform.

    class winreg:
        HKEY_USERS = None
        HKEY_CURRENT_USER = None
        HKEY_LOCAL_MACHINE = None
        HKEY_CLASSES_ROOT = None

    environ: dict[str, str] = {}


class PlatformInfo:
    """
    Current and Target Architectures information.

    Parameters
    ----------
    arch: str
        Target architecture.
    """

    current_cpu = environ.get('processor_architecture', '').lower()

    def __init__(self, arch: str) -> None:
        self.arch = arch.lower().replace('x64', 'amd64')

    @property
    def target_cpu(self) -> str:
        """
        Return Target CPU architecture.

        Return
        ------
        str
            Target CPU
        """
        return self.arch[self.arch.find('_') + 1 :]

    def target_is_x86(self) -> bool:
        """
        Return True if target CPU is x86 32 bits..

        Return
        ------
        bool
            CPU is x86 32 bits
        """
        return self.target_cpu == 'x86'

    def current_is_x86(self) -> bool:
        """
        Return True if current CPU is x86 32 bits..

        Return
        ------
        bool
            CPU is x86 32 bits
        """
        return self.current_cpu == 'x86'

    def current_dir(self, hidex86=False, x64=False) -> str:
        """
        Current platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\x86' if architecture is x86.
        x64: bool
            return '\x64' and not '\amd64' if architecture is amd64.

        Return
        ------
        str
            subfolder: '\target', or '' (see hidex86 parameter)
        """
        return (
            ''
            if (self.current_cpu == 'x86' and hidex86)
            else r'\x64'
            if (self.current_cpu == 'amd64' and x64)
            else rf'\{self.current_cpu}'
        )

    def target_dir(self, hidex86=False, x64=False) -> str:
        r"""
        Target platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\x86' if architecture is x86.
        x64: bool
            return '\x64' and not '\amd64' if architecture is amd64.

        Return
        ------
        str
            subfolder: '\current', or '' (see hidex86 parameter)
        """
        return (
            ''
            if (self.target_cpu == 'x86' and hidex86)
            else r'\x64'
            if (self.target_cpu == 'amd64' and x64)
            else rf'\{self.target_cpu}'
        )

    def cross_dir(self, forcex86=False) -> str:
        r"""
        Cross platform specific subfolder.

        Parameters
        ----------
        forcex86: bool
            Use 'x86' as current architecture even if current architecture is
            not x86.

        Return
        ------
        str
            subfolder: '' if target architecture is current architecture,
            '\current_target' if not.
        """
        current = 'x86' if forcex86 else self.current_cpu
        return (
            ''
            if self.target_cpu == current
            else self.target_dir().replace('\\', f'\\{current}_')
        )


class RegistryInfo:
    """
    Microsoft Visual Studio related registry information.

    Parameters
    ----------
    platform_info: PlatformInfo
        "PlatformInfo" instance.
    """

    HKEYS = (
        winreg.HKEY_USERS,
        winreg.HKEY_CURRENT_USER,
        winreg.HKEY_LOCAL_MACHINE,
        winreg.HKEY_CLASSES_ROOT,
    )

    def __init__(self, platform_info: PlatformInfo) -> None:
        self.pi = platform_info

    @property
    def visualstudio(self) -> LiteralString:
        """
        Microsoft Visual Studio root registry key.

        Return
        ------
        str
            Registry key
        """
        return 'VisualStudio'

    @property
    def sxs(self) -> LiteralString:
        """
        Microsoft Visual Studio SxS registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.visualstudio, 'SxS')

    @property
    def vc(self) -> LiteralString:
        """
        Microsoft Visual C++ VC7 registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.sxs, 'VC7')

    @property
    def vs(self) -> LiteralString:
        """
        Microsoft Visual Studio VS7 registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.sxs, 'VS7')

    @property
    def vc_for_python(self) -> LiteralString:
        """
        Microsoft Visual C++ for Python registry key.

        Return
        ------
        str
            Registry key
        """
        return r'DevDiv\VCForPython'

    @property
    def microsoft_sdk(self) -> LiteralString:
        """
        Microsoft SDK registry key.

        Return
        ------
        str
            Registry key
        """
        return 'Microsoft SDKs'

    @property
    def windows_sdk(self) -> LiteralString:
        """
        Microsoft Windows/Platform SDK registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.microsoft_sdk, 'Windows')

    @property
    def netfx_sdk(self) -> LiteralString:
        """
        Microsoft .NET Framework SDK registry key.

        Return
        ------
        str
            Registry key
        """
        return os.path.join(self.microsoft_sdk, 'NETFXSDK')

    @property
    def windows_kits_roots(self) -> LiteralString:
        """
        Microsoft Windows Kits Roots registry key.

        Return
        ------
        str
            Registry key
        """
        return r'Windows Kits\Installed Roots'

    @overload
    def microsoft(self, key: LiteralString, x86: bool = False) -> LiteralString: ...
    @overload
    def microsoft(self, key: str, x86: bool = False) -> str: ...  # type: ignore[misc]
    def microsoft(self, key: str, x86: bool = False) -> str:
        """
        Return key in Microsoft software registry.

        Parameters
        ----------
        key: str
            Registry key path where look.
        x86: bool
            Force x86 software registry.

        Return
        ------
        str
            Registry key
        """
        node64 = '' if self.pi.current_is_x86() or x86 else 'Wow6432Node'
        return os.path.join('Software', node64, 'Microsoft', key)

    def lookup(self, key: str, name: str) -> str | None:
        """
        Look for values in registry in Microsoft software registry.

        Parameters
        ----------
        key: str
            Registry key path where look.
        name: str
            Value name to find.

        Return
        ------
        str | None
            value
        """
        key_read = winreg.KEY_READ
        openkey = winreg.OpenKey
        closekey = winreg.CloseKey
        ms = self.microsoft
        for hkey in self.HKEYS:
            bkey = None
            try:
                bkey = openkey(hkey, ms(key), 0, key_read)
            except OSError:
                if not self.pi.current_is_x86():
                    try:
                        bkey = openkey(hkey, ms(key, True), 0, key_read)
                    except OSError:
                        continue
                else:
                    continue
            try:
                return winreg.QueryValueEx(bkey, name)[0]
            except OSError:
                pass
            finally:
                if bkey:
                    closekey(bkey)
        return None


class SystemInfo:
    """
    Microsoft Windows and Visual Studio related system information.

    Parameters
    ----------
    registry_info: RegistryInfo
        "RegistryInfo" instance.
    vc_ver: float
        Required Microsoft Visual C++ version.
    """

    # Variables and properties in this class use originals CamelCase variables
    # names from Microsoft source files for more easy comparison.
    WinDir = environ.get('WinDir', '')
    ProgramFiles = environ.get('ProgramFiles', '')
    ProgramFilesx86 = environ.get('ProgramFiles(x86)', ProgramFiles)

    def __init__(
        self, registry_info: RegistryInfo, vc_ver: float | None = None
    ) -> None:
        self.ri = registry_info
        self.pi = self.ri.pi

        self.known_vs_paths = self.find_programdata_vs_vers()

        # Except for VS15+, VC version is aligned with VS version
        self.vs_ver = self.vc_ver = vc_ver or self._find_latest_available_vs_ver()

    def _find_latest_available_vs_ver(self):
        """
        Find the latest VC version

        Return
        ------
        float
            version
        """
        reg_vc_vers = self.find_reg_vs_vers()

        if not (reg_vc_vers or self.known_vs_paths):
            raise distutils.errors.DistutilsPlatformError(
                'No Microsoft Visual C++ version found'
            )

        vc_vers = set(reg_vc_vers)
        vc_vers.update(self.known_vs_paths)
        return max(vc_vers)

    def find_reg_vs_vers(self) -> list[float]:
        """
        Find Microsoft Visual Studio versions available in registry.

        Return
        ------
        list of float
            Versions
        """
        ms = self.ri.microsoft
        vckeys = (self.ri.vc, self.ri.vc_for_python, self.ri.vs)
        vs_vers = []
        for hkey, key in itertools.product(self.ri.HKEYS, vckeys):
            try:
                bkey = winreg.OpenKey(hkey, ms(key), 0, winreg.KEY_READ)
            except OSError:
                continue
            with bkey:
                subkeys, values, _ = winreg.QueryInfoKey(bkey)
                for i in range(values):
                    with contextlib.suppress(ValueError):
                        ver = float(winreg.EnumValue(bkey, i)[0])
                        if ver not in vs_vers:
                            vs_vers.append(ver)
                for i in range(subkeys):
                    with contextlib.suppress(ValueError):
                        ver = float(winreg.EnumKey(bkey, i))
                        if ver not in vs_vers:
                            vs_vers.append(ver)
        return sorted(vs_vers)

    def find_programdata_vs_vers(self) -> dict[float, str]:
        r"""
        Find Visual studio 2017+ versions from information in
        "C:\ProgramData\Microsoft\VisualStudio\Packages\_Instances".

        Return
        ------
        dict
            float version as key, path as value.
        """
        vs_versions: dict[float, str] = {}
        instances_dir = r'C:\ProgramData\Microsoft\VisualStudio\Packages\_Instances'

        try:
            hashed_names = os.listdir(instances_dir)

        except OSError:
            # Directory not exists with all Visual Studio versions
            return vs_versions

        for name in hashed_names:
            try:
                # Get VS installation path from "state.json" file
                state_path = os.path.join(instances_dir, name, 'state.json')
                with open(state_path, 'rt', encoding='utf-8') as state_file:
                    state = json.load(state_file)
                vs_path = state['installationPath']

                # Raises OSError if this VS installation does not contain VC
                os.listdir(os.path.join(vs_path, r'VC\Tools\MSVC'))

                # Store version and path
                vs_versions[self._as_float_version(state['installationVersion'])] = (
                    vs_path
                )

            except (OSError, KeyError):
                # Skip if "state.json" file is missing or bad format
                continue

        return vs_versions

    @staticmethod
    def _as_float_version(version):
        """
        Return a string version as a simplified float version (major.minor)

        Parameters
        ----------
        version: str
            Version.

        Return
        ------
        float
            version
        """
        return float('.'.join(version.split('.')[:2]))

    @property
    def VSInstallDir(self) -> str:
        """
        Microsoft Visual Studio directory.

        Return
        ------
        str
            path
        """
        # Default path
        default = os.path.join(
            self.ProgramFilesx86, f'Microsoft Visual Studio {self.vs_ver:0.1f}'
        )

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vs, f'{self.vs_ver:0.1f}') or default

    @property
    def VCInstallDir(self) -> str:
        """
        Microsoft Visual C++ directory.

        Return
        ------
        str
            path
        """
        path = self._guess_vc() or self._guess_vc_legacy()

        if not os.path.isdir(path):
            msg = 'Microsoft Visual C++ directory not found'
            raise distutils.errors.DistutilsPlatformError(msg)

        return path

    def _guess_vc(self):
        """
        Locate Visual C++ for VS2017+.

        Return
        ------
        str
            path
        """
        if self.vs_ver <= 14.0:
            return ''

        try:
            # First search in known VS paths
            vs_dir = self.known_vs_paths[self.vs_ver]
        except KeyError:
            # Else, search with path from registry
            vs_dir = self.VSInstallDir

        guess_vc = os.path.join(vs_dir, r'VC\Tools\MSVC')

        # Subdir with VC exact version as name
        try:
            # Update the VC version with real one instead of VS version
            vc_ver = os.listdir(guess_vc)[-1]
            self.vc_ver = self._as_float_version(vc_ver)
            return os.path.join(guess_vc, vc_ver)
        except (OSError, IndexError):
            return ''

    def _guess_vc_legacy(self):
        """
        Locate Visual C++ for versions prior to 2017.

        Return
        ------
        str
            path
        """
        default = os.path.join(
            self.ProgramFilesx86,
            rf'Microsoft Visual Studio {self.vs_ver:0.1f}\VC',
        )

        # Try to get "VC++ for Python" path from registry as default path
        reg_path = os.path.join(self.ri.vc_for_python, f'{self.vs_ver:0.1f}')
        python_vc = self.ri.lookup(reg_path, 'installdir')
        default_vc = os.path.join(python_vc, 'VC') if python_vc else default

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vc, f'{self.vs_ver:0.1f}') or default_vc

    @property
    def WindowsSdkVersion(self) -> tuple[LiteralString, ...]:
        """
        Microsoft Windows SDK versions for specified MSVC++ version.

        Return
        ------
        tuple of str
            versions
        """
        if self.vs_ver <= 9.0:
            return '7.0', '6.1', '6.0a'
        elif self.vs_ver == 10.0:
            return '7.1', '7.0a'
        elif self.vs_ver == 11.0:
            return '8.0', '8.0a'
        elif self.vs_ver == 12.0:
            return '8.1', '8.1a'
        elif self.vs_ver >= 14.0:
            return '10.0', '8.1'
        return ()

    @property
    def WindowsSdkLastVersion(self) -> str:
        """
        Microsoft Windows SDK last version.

        Return
        ------
        str
            version
        """
        return self._use_last_dir_name(os.path.join(self.WindowsSdkDir, 'lib'))

    @property
    def WindowsSdkDir(self) -> str:  # noqa: C901  # is too complex (12)  # FIXME
        """
        Microsoft Windows SDK directory.

        Return
        ------
        str
            path
        """
        sdkdir: str | None = ''
        for ver in self.WindowsSdkVersion:
            # Try to get it from registry
            loc = os.path.join(self.ri.windows_sdk, f'v{ver}')
            sdkdir = self.ri.lookup(loc, 'installationfolder')
            if sdkdir:
                break
        if not sdkdir or not os.path.isdir(sdkdir):
            # Try to get "VC++ for Python" version from registry
            path = os.path.join(self.ri.vc_for_python, f'{self.vc_ver:0.1f}')
            install_base = self.ri.lookup(path, 'installdir')
            if install_base:
                sdkdir = os.path.join(install_base, 'WinSDK')
        if not sdkdir or not os.path.isdir(sdkdir):
            # If fail, use default new path
            for ver in self.WindowsSdkVersion:
                intver = ver[: ver.rfind('.')]
                path = rf'Microsoft SDKs\Windows Kits\{intver}'
                d = os.path.join(self.ProgramFiles, path)
                if os.path.isdir(d):
                    sdkdir = d
        if not sdkdir or not os.path.isdir(sdkdir):
            # If fail, use default old path
            for ver in self.WindowsSdkVersion:
                path = rf'Microsoft SDKs\Windows\v{ver}'
                d = os.path.join(self.ProgramFiles, path)
                if os.path.isdir(d):
                    sdkdir = d
        if not sdkdir:
            # If fail, use Platform SDK
            sdkdir = os.path.join(self.VCInstallDir, 'PlatformSDK')
        return sdkdir

    @property
    def WindowsSDKExecutablePath(self) -> str | None:
        """
        Microsoft Windows SDK executable directory.

        Return
        ------
        str | None
            path
        """
        # Find WinSDK NetFx Tools registry dir name
        if self.vs_ver <= 11.0:
            netfxver = 35
            arch = ''
        else:
            netfxver = 40
            hidex86 = self.vs_ver <= 12.0
            arch = self.pi.current_dir(x64=True, hidex86=hidex86).replace('\\', '-')
        fx = f'WinSDK-NetFx{netfxver}Tools{arch}'

        # list all possibles registry paths
        regpaths = []
        if self.vs_ver >= 14.0:
            for ver in self.NetFxSdkVersion:
                regpaths += [os.path.join(self.ri.netfx_sdk, ver, fx)]

        for ver in self.WindowsSdkVersion:
            regpaths += [os.path.join(self.ri.windows_sdk, f'v{ver}A', fx)]

        # Return installation folder from the more recent path
        for path in regpaths:
            execpath = self.ri.lookup(path, 'installationfolder')
            if execpath:
                return execpath

        return None

    @property
    def FSharpInstallDir(self) -> str:
        """
        Microsoft Visual F# directory.

        Return
        ------
        str
            path
        """
        path = os.path.join(self.ri.visualstudio, rf'{self.vs_ver:0.1f}\Setup\F#')
        return self.ri.lookup(path, 'productdir') or ''

    @property
    def UniversalCRTSdkDir(self) -> str | None:
        """
        Microsoft Universal CRT SDK directory.

        Return
        ------
        str | None
            path
        """
        # Set Kit Roots versions for specified MSVC++ version
        vers = ('10', '81') if self.vs_ver >= 14.0 else ()

        # Find path of the more recent Kit
        for ver in vers:
            sdkdir = self.ri.lookup(self.ri.windows_kits_roots, f'kitsroot{ver}')
            if sdkdir:
                return sdkdir

        return None

    @property
    def UniversalCRTSdkLastVersion(self) -> str:
        """
        Microsoft Universal C Runtime SDK last version.

        Return
        ------
        str
            version
        """
        try:
            return self._use_last_dir_name(os.path.join(self.UniversalCRTSdkDir, 'lib'))  # type: ignore[arg-type] # Expected TypeError
        except TypeError as ex:
            py310.add_note(ex, "Cannot find UniversalCRTSdkDir")
            raise

    @property
    def NetFxSdkVersion(self) -> tuple[LiteralString, ...]:
        """
        Microsoft .NET Framework SDK versions.

        Return
        ------
        tuple of str
            versions
        """
        # Set FxSdk versions for specified VS version
        return (
            ('4.7.2', '4.7.1', '4.7', '4.6.2', '4.6.1', '4.6', '4.5.2', '4.5.1', '4.5')
            if self.vs_ver >= 14.0
            else ()
        )

    @property
    def NetFxSdkDir(self) -> str | None:
        """
        Microsoft .NET Framework SDK directory.

        Return
        ------
        str | None
            path
        """
        sdkdir: str | None = ''
        for ver in self.NetFxSdkVersion:
            loc = os.path.join(self.ri.netfx_sdk, ver)
            sdkdir = self.ri.lookup(loc, 'kitsinstallationfolder')
            if sdkdir:
                break
        return sdkdir

    @property
    def FrameworkDir32(self) -> str:
        """
        Microsoft .NET Framework 32bit directory.

        Return
        ------
        str
            path
        """
        # Default path
        guess_fw = os.path.join(self.WinDir, r'Microsoft.NET\Framework')

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vc, 'frameworkdir32') or guess_fw

    @property
    def FrameworkDir64(self) -> str:
        """
        Microsoft .NET Framework 64bit directory.

        Return
        ------
        str
            path
        """
        # Default path
        guess_fw = os.path.join(self.WinDir, r'Microsoft.NET\Framework64')

        # Try to get path from registry, if fail use default path
        return self.ri.lookup(self.ri.vc, 'frameworkdir64') or guess_fw

    @property
    def FrameworkVersion32(self) -> tuple[str, ...]:
        """
        Microsoft .NET Framework 32bit versions.

        Return
        ------
        tuple of str
            versions
        """
        return self._find_dot_net_versions(32)

    @property
    def FrameworkVersion64(self) -> tuple[str, ...]:
        """
        Microsoft .NET Framework 64bit versions.

        Return
        ------
        tuple of str
            versions
        """
        return self._find_dot_net_versions(64)

    def _find_dot_net_versions(self, bits) -> tuple[str, ...]:
        """
        Find Microsoft .NET Framework versions.

        Parameters
        ----------
        bits: int
            Platform number of bits: 32 or 64.

        Return
        ------
        tuple of str
            versions
        """
        # Find actual .NET version in registry
        reg_ver = self.ri.lookup(self.ri.vc, f'frameworkver{bits}')
        dot_net_dir = getattr(self, f'FrameworkDir{bits}')
        ver = reg_ver or self._use_last_dir_name(dot_net_dir, 'v') or ''

        # Set .NET versions for specified MSVC++ version
        if self.vs_ver >= 12.0:
            return ver, 'v4.0'
        elif self.vs_ver >= 10.0:
            return 'v4.0.30319' if ver.lower()[:2] != 'v4' else ver, 'v3.5'
        elif self.vs_ver == 9.0:
            return 'v3.5', 'v2.0.50727'
        elif self.vs_ver == 8.0:
            return 'v3.0', 'v2.0.50727'
        return ()

    @staticmethod
    def _use_last_dir_name(path: StrPath, prefix: str = '') -> str:
        """
        Return name of the last dir in path or '' if no dir found.

        Parameters
        ----------
        path: StrPath
            Use dirs in this path
        prefix: str
            Use only dirs starting by this prefix

        Return
        ------
        str
            name
        """
        matching_dirs = (
            dir_name
            for dir_name in reversed(os.listdir(path))
            if os.path.isdir(os.path.join(path, dir_name))
            and dir_name.startswith(prefix)
        )
        return next(matching_dirs, '')


class _EnvironmentDict(TypedDict):
    include: str
    lib: str
    libpath: str
    path: str
    py_vcruntime_redist: NotRequired[str | None]


class EnvironmentInfo:
    """
    Return environment variables for specified Microsoft Visual C++ version
    and platform : Lib, Include, Path and libpath.

    This function is compatible with Microsoft Visual C++ 9.0 to 14.X.

    Script created by analysing Microsoft environment configuration files like
    "vcvars[...].bat", "SetEnv.Cmd", "vcbuildtools.bat", ...

    Parameters
    ----------
    arch: str
        Target architecture.
    vc_ver: float
        Required Microsoft Visual C++ version. If not set, autodetect the last
        version.
    vc_min_ver: float
        Minimum Microsoft Visual C++ version.
    """

    # Variables and properties in this class use originals CamelCase variables
    # names from Microsoft source files for more easy comparison.

    def __init__(self, arch, vc_ver=None, vc_min_ver=0) -> None:
        self.pi = PlatformInfo(arch)
        self.ri = RegistryInfo(self.pi)
        self.si = SystemInfo(self.ri, vc_ver)

        if self.vc_ver < vc_min_ver:
            err = 'No suitable Microsoft Visual C++ version found'
            raise distutils.errors.DistutilsPlatformError(err)

    @property
    def vs_ver(self):
        """
        Microsoft Visual Studio.

        Return
        ------
        float
            version
        """
        return self.si.vs_ver

    @property
    def vc_ver(self):
        """
        Microsoft Visual C++ version.

        Return
        ------
        float
            version
        """
        return self.si.vc_ver

    @property
    def VSTools(self):
        """
        Microsoft Visual Studio Tools.

        Return
        ------
        list of str
            paths
        """
        paths = [r'Common7\IDE', r'Common7\Tools']

        if self.vs_ver >= 14.0:
            arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            paths += [r'Common7\IDE\CommonExtensions\Microsoft\TestWindow']
            paths += [r'Team Tools\Performance Tools']
            paths += [rf'Team Tools\Performance Tools{arch_subdir}']

        return [os.path.join(self.si.VSInstallDir, path) for path in paths]

    @property
    def VCIncludes(self):
        """
        Microsoft Visual C++ & Microsoft Foundation Class Includes.

        Return
        ------
        list of str
            paths
        """
        return [
            os.path.join(self.si.VCInstallDir, 'Include'),
            os.path.join(self.si.VCInstallDir, r'ATLMFC\Include'),
        ]

    @property
    def VCLibraries(self):
        """
        Microsoft Visual C++ & Microsoft Foundation Class Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver >= 15.0:
            arch_subdir = self.pi.target_dir(x64=True)
        else:
            arch_subdir = self.pi.target_dir(hidex86=True)
        paths = [f'Lib{arch_subdir}', rf'ATLMFC\Lib{arch_subdir}']

        if self.vs_ver >= 14.0:
            paths += [rf'Lib\store{arch_subdir}']

        return [os.path.join(self.si.VCInstallDir, path) for path in paths]

    @property
    def VCStoreRefs(self):
        """
        Microsoft Visual C++ store references Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []
        return [os.path.join(self.si.VCInstallDir, r'Lib\store\references')]

    @property
    def VCTools(self):
        """
        Microsoft Visual C++ Tools.

        Return
        ------
        list of str
            paths

        When host CPU is ARM, the tools should be found for ARM.

        >>> getfixture('windows_only')
        >>> mp = getfixture('monkeypatch')
        >>> mp.setattr(PlatformInfo, 'current_cpu', 'arm64')
        >>> ei = EnvironmentInfo(arch='irrelevant')
        >>> paths = ei.VCTools
        >>> any('HostARM64' in path for path in paths)
        True
        """
        si = self.si
        tools = [os.path.join(si.VCInstallDir, 'VCPackages')]

        forcex86 = self.vs_ver <= 10.0
        arch_subdir = self.pi.cross_dir(forcex86)
        if arch_subdir:
            tools += [os.path.join(si.VCInstallDir, f'Bin{arch_subdir}')]

        if self.vs_ver == 14.0:
            path = f'Bin{self.pi.current_dir(hidex86=True)}'
            tools += [os.path.join(si.VCInstallDir, path)]

        elif self.vs_ver >= 15.0:
            host_id = self.pi.current_cpu.replace('amd64', 'x64').upper()
            host_dir = os.path.join('bin', f'Host{host_id}%s')
            tools += [
                os.path.join(si.VCInstallDir, host_dir % self.pi.target_dir(x64=True))
            ]

            if self.pi.current_cpu != self.pi.target_cpu:
                tools += [
                    os.path.join(
                        si.VCInstallDir, host_dir % self.pi.current_dir(x64=True)
                    )
                ]

        else:
            tools += [os.path.join(si.VCInstallDir, 'Bin')]

        return tools

    @property
    def OSLibraries(self):
        """
        Microsoft Windows SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver <= 10.0:
            arch_subdir = self.pi.target_dir(hidex86=True, x64=True)
            return [os.path.join(self.si.WindowsSdkDir, f'Lib{arch_subdir}')]

        else:
            arch_subdir = self.pi.target_dir(x64=True)
            lib = os.path.join(self.si.WindowsSdkDir, 'lib')
            libver = self._sdk_subdir
            return [os.path.join(lib, f'{libver}um{arch_subdir}')]

    @property
    def OSIncludes(self):
        """
        Microsoft Windows SDK Include.

        Return
        ------
        list of str
            paths
        """
        include = os.path.join(self.si.WindowsSdkDir, 'include')

        if self.vs_ver <= 10.0:
            return [include, os.path.join(include, 'gl')]

        else:
            if self.vs_ver >= 14.0:
                sdkver = self._sdk_subdir
            else:
                sdkver = ''
            return [
                os.path.join(include, f'{sdkver}shared'),
                os.path.join(include, f'{sdkver}um'),
                os.path.join(include, f'{sdkver}winrt'),
            ]

    @property
    def OSLibpath(self):
        """
        Microsoft Windows SDK Libraries Paths.

        Return
        ------
        list of str
            paths
        """
        ref = os.path.join(self.si.WindowsSdkDir, 'References')
        libpath = []

        if self.vs_ver <= 9.0:
            libpath += self.OSLibraries

        if self.vs_ver >= 11.0:
            libpath += [os.path.join(ref, r'CommonConfiguration\Neutral')]

        if self.vs_ver >= 14.0:
            libpath += [
                ref,
                os.path.join(self.si.WindowsSdkDir, 'UnionMetadata'),
                os.path.join(ref, 'Windows.Foundation.UniversalApiContract', '1.0.0.0'),
                os.path.join(ref, 'Windows.Foundation.FoundationContract', '1.0.0.0'),
                os.path.join(
                    ref, 'Windows.Networking.Connectivity.WwanContract', '1.0.0.0'
                ),
                os.path.join(
                    self.si.WindowsSdkDir,
                    'ExtensionSDKs',
                    'Microsoft.VCLibs',
                    f'{self.vs_ver:0.1f}',
                    'References',
                    'CommonConfiguration',
                    'neutral',
                ),
            ]
        return libpath

    @property
    def SdkTools(self):
        """
        Microsoft Windows SDK Tools.

        Return
        ------
        list of str
            paths
        """
        return list(self._sdk_tools())

    def _sdk_tools(self):
        """
        Microsoft Windows SDK Tools paths generator.

        Return
        ------
        generator of str
            paths
        """
        if self.vs_ver < 15.0:
            bin_dir = 'Bin' if self.vs_ver <= 11.0 else r'Bin\x86'
            yield os.path.join(self.si.WindowsSdkDir, bin_dir)

        if not self.pi.current_is_x86():
            arch_subdir = self.pi.current_dir(x64=True)
            path = f'Bin{arch_subdir}'
            yield os.path.join(self.si.WindowsSdkDir, path)

        if self.vs_ver in (10.0, 11.0):
            if self.pi.target_is_x86():
                arch_subdir = ''
            else:
                arch_subdir = self.pi.current_dir(hidex86=True, x64=True)
            path = rf'Bin\NETFX 4.0 Tools{arch_subdir}'
            yield os.path.join(self.si.WindowsSdkDir, path)

        elif self.vs_ver >= 15.0:
            path = os.path.join(self.si.WindowsSdkDir, 'Bin')
            arch_subdir = self.pi.current_dir(x64=True)
            sdkver = self.si.WindowsSdkLastVersion
            yield os.path.join(path, f'{sdkver}{arch_subdir}')

        if self.si.WindowsSDKExecutablePath:
            yield self.si.WindowsSDKExecutablePath

    @property
    def _sdk_subdir(self) -> str:
        """
        Microsoft Windows SDK version subdir.

        Return
        ------
        str
            subdir
        """
        ucrtver = self.si.WindowsSdkLastVersion
        return (f'{ucrtver}\\') if ucrtver else ''

    @property
    def SdkSetup(self):
        """
        Microsoft Windows SDK Setup.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver > 9.0:
            return []

        return [os.path.join(self.si.WindowsSdkDir, 'Setup')]

    @property
    def FxTools(self):
        """
        Microsoft .NET Framework Tools.

        Return
        ------
        list of str
            paths
        """
        pi = self.pi
        si = self.si

        if self.vs_ver <= 10.0:
            include32 = True
            include64 = not pi.target_is_x86() and not pi.current_is_x86()
        else:
            include32 = pi.target_is_x86() or pi.current_is_x86()
            include64 = pi.current_cpu == 'amd64' or pi.target_cpu == 'amd64'

        tools = []
        if include32:
            tools += [
                os.path.join(si.FrameworkDir32, ver) for ver in si.FrameworkVersion32
            ]
        if include64:
            tools += [
                os.path.join(si.FrameworkDir64, ver) for ver in si.FrameworkVersion64
            ]
        return tools

    @property
    def NetFxSDKLibraries(self):
        """
        Microsoft .Net Framework SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
            return []

        arch_subdir = self.pi.target_dir(x64=True)
        return [os.path.join(self.si.NetFxSdkDir, rf'lib\um{arch_subdir}')]

    @property
    def NetFxSDKIncludes(self):
        """
        Microsoft .Net Framework SDK Includes.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0 or not self.si.NetFxSdkDir:
            return []

        return [os.path.join(self.si.NetFxSdkDir, r'include\um')]

    @property
    def VsTDb(self):
        """
        Microsoft Visual Studio Team System Database.

        Return
        ------
        list of str
            paths
        """
        return [os.path.join(self.si.VSInstallDir, r'VSTSDB\Deploy')]

    @property
    def MSBuild(self):
        """
        Microsoft Build Engine.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 12.0:
            return []
        elif self.vs_ver < 15.0:
            base_path = self.si.ProgramFilesx86
            arch_subdir = self.pi.current_dir(hidex86=True)
        else:
            base_path = self.si.VSInstallDir
            arch_subdir = ''

        path = rf'MSBuild\{self.vs_ver:0.1f}\bin{arch_subdir}'
        build = [os.path.join(base_path, path)]

        if self.vs_ver >= 15.0:
            # Add Roslyn C# & Visual Basic Compiler
            build += [os.path.join(base_path, path, 'Roslyn')]

        return build

    @property
    def HTMLHelpWorkshop(self):
        """
        Microsoft HTML Help Workshop.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 11.0:
            return []

        return [os.path.join(self.si.ProgramFilesx86, 'HTML Help Workshop')]

    @property
    def UCRTLibraries(self) -> list[str]:
        """
        Microsoft Universal C Runtime SDK Libraries.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []

        arch_subdir = self.pi.target_dir(x64=True)
        try:
            lib = os.path.join(self.si.UniversalCRTSdkDir, 'lib')  # type: ignore[arg-type] # Expected TypeError
        except TypeError as ex:
            py310.add_note(ex, "Cannot find UniversalCRTSdkDir")
            raise
        ucrtver = self._ucrt_subdir
        return [os.path.join(lib, f'{ucrtver}ucrt{arch_subdir}')]

    @property
    def UCRTIncludes(self) -> list[str]:
        """
        Microsoft Universal C Runtime SDK Include.

        Return
        ------
        list of str
            paths
        """
        if self.vs_ver < 14.0:
            return []

        try:
            include = os.path.join(self.si.UniversalCRTSdkDir, 'include')  # type: ignore[arg-type] # Expected TypeError
        except TypeError as ex:
            py310.add_note(ex, "Cannot find UniversalCRTSdkDir")
            raise
        return [os.path.join(include, f'{self._ucrt_subdir}ucrt')]

    @property
    def _ucrt_subdir(self) -> str:
        """
        Microsoft Universal C Runtime SDK version subdir.

        Return
        ------
        str
            subdir
        """
        ucrtver = self.si.UniversalCRTSdkLastVersion
        return (f'{ucrtver}\\') if ucrtver else ''

    @property
    def FSharp(self):
        """
        Microsoft Visual F#.

        Return
        ------
        list of str
            paths
        """
        if 11.0 > self.vs_ver > 12.0:
            return []

        return [self.si.FSharpInstallDir]

    @property
    def VCRuntimeRedist(self) -> str | None:
        """
        Microsoft Visual C++ runtime redistributable dll.

        Returns the first suitable path found or None.
        """
        vcruntime = f'vcruntime{self.vc_ver}0.dll'
        arch_subdir = self.pi.target_dir(x64=True).strip('\\')

        # Installation prefixes candidates
        prefixes = []
        tools_path = self.si.VCInstallDir
        redist_path = os.path.dirname(tools_path.replace(r'\Tools', r'\Redist'))
        if os.path.isdir(redist_path):
            # Redist version may not be exactly the same as tools
            redist_path = os.path.join(redist_path, os.listdir(redist_path)[-1])
            prefixes += [redist_path, os.path.join(redist_path, 'onecore')]

        prefixes += [os.path.join(tools_path, 'redist')]  # VS14 legacy path

        # CRT directory
        crt_dirs = (
            f'Microsoft.VC{self.vc_ver * 10}.CRT',
            # Sometime store in directory with VS version instead of VC
            f'Microsoft.VC{int(self.vs_ver) * 10}.CRT',
        )

        # vcruntime path
        candidate_paths = (
            os.path.join(prefix, arch_subdir, crt_dir, vcruntime)
            for (prefix, crt_dir) in itertools.product(prefixes, crt_dirs)
        )
        return next(filter(os.path.isfile, candidate_paths), None)  # type: ignore[arg-type] #python/mypy#12682

    def return_env(self, exists: bool = True) -> _EnvironmentDict:
        """
        Return environment dict.

        Parameters
        ----------
        exists: bool
            It True, only return existing paths.

        Return
        ------
        dict
            environment
        """
        env = _EnvironmentDict(
            include=self._build_paths(
                'include',
                [
                    self.VCIncludes,
                    self.OSIncludes,
                    self.UCRTIncludes,
                    self.NetFxSDKIncludes,
                ],
                exists,
            ),
            lib=self._build_paths(
                'lib',
                [
                    self.VCLibraries,
                    self.OSLibraries,
                    self.FxTools,
                    self.UCRTLibraries,
                    self.NetFxSDKLibraries,
                ],
                exists,
            ),
            libpath=self._build_paths(
                'libpath',
                [self.VCLibraries, self.FxTools, self.VCStoreRefs, self.OSLibpath],
                exists,
            ),
            path=self._build_paths(
                'path',
                [
                    self.VCTools,
                    self.VSTools,
                    self.VsTDb,
                    self.SdkTools,
                    self.SdkSetup,
                    self.FxTools,
                    self.MSBuild,
                    self.HTMLHelpWorkshop,
                    self.FSharp,
                ],
                exists,
            ),
        )
        if self.vs_ver >= 14 and self.VCRuntimeRedist:
            env['py_vcruntime_redist'] = self.VCRuntimeRedist
        return env

    def _build_paths(self, name, spec_path_lists, exists):
        """
        Given an environment variable name and specified paths,
        return a pathsep-separated string of paths containing
        unique, extant, directories from those paths and from
        the environment variable. Raise an error if no paths
        are resolved.

        Parameters
        ----------
        name: str
            Environment variable name
        spec_path_lists: list of str
            Paths
        exists: bool
            It True, only return existing paths.

        Return
        ------
        str
            Pathsep-separated paths
        """
        # flatten spec_path_lists
        spec_paths = itertools.chain.from_iterable(spec_path_lists)
        env_paths = environ.get(name, '').split(os.pathsep)
        paths = itertools.chain(spec_paths, env_paths)
        extant_paths = list(filter(os.path.isdir, paths)) if exists else paths
        if not extant_paths:
            msg = f"{name.upper()} environment variable is empty"
            raise distutils.errors.DistutilsPlatformError(msg)
        unique_paths = unique_everseen(extant_paths)
        return os.pathsep.join(unique_paths)
<|endoftext|>"""
Create a wheel (.whl) distribution.

A wheel is a built archive format.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import sys
import sysconfig
import warnings
from collections.abc import Iterable, Sequence
from email.generator import BytesGenerator
from glob import iglob
from typing import ClassVar, Literal, cast
from zipfile import ZIP_DEFLATED, ZIP_STORED

from packaging import tags, version as _packaging_version
from wheel.wheelfile import WheelFile

from .. import Command, __version__, _shutil
from .._core_metadata import _safe_license_file
from .._normalization import safer_name
from ..warnings import SetuptoolsDeprecationWarning
from .egg_info import egg_info as egg_info_cls

from distutils import log


def safe_version(version: str) -> str:
    """
    Convert an arbitrary string to a standard version string
    """
    try:
        # normalize the version
        return str(_packaging_version.Version(version))
    except _packaging_version.InvalidVersion:
        version = version.replace(" ", ".")
        return re.sub("[^A-Za-z0-9.]+", "-", version)


setuptools_major_version = int(__version__.split(".")[0])

PY_LIMITED_API_PATTERN = r"cp3\d"


def _is_32bit_interpreter() -> bool:
    return struct.calcsize("P") == 4


def python_tag() -> str:
    return f"py{sys.version_info.major}"


def get_platform(archive_root: str | None) -> str:
    """Return our platform name 'win32', 'linux_x86_64'"""
    result = sysconfig.get_platform()
    if result.startswith("macosx") and archive_root is not None:  # pragma: no cover
        from wheel.macosx_libfile import calculate_macosx_platform_tag

        result = calculate_macosx_platform_tag(archive_root, result)
    elif _is_32bit_interpreter():
        if result == "linux-x86_64":
            # pip pull request #3497
            result = "linux-i686"
        elif result == "linux-aarch64":
            # packaging pull request #234
            # TODO armv8l, packaging pull request #690 => this did not land
            # in pip/packaging yet
            result = "linux-armv7l"

    return result.replace("-", "_")


def get_flag(
    var: str, fallback: bool, expected: bool = True, warn: bool = True
) -> bool:
    """Use a fallback value for determining SOABI flags if the needed config
    var is unset or unavailable."""
    val = sysconfig.get_config_var(var)
    if val is None:
        if warn:
            warnings.warn(
                f"Config variable '{var}' is unset, Python ABI tag may be incorrect",
                RuntimeWarning,
                stacklevel=2,
            )
        return fallback
    return val == expected


def get_abi_tag() -> str | None:
    """Return the ABI tag based on SOABI (if available) or emulate SOABI (PyPy2)."""
    soabi: str = sysconfig.get_config_var("SOABI")
    impl = tags.interpreter_name()
    if not soabi and impl in ("cp", "pp") and hasattr(sys, "maxunicode"):
        d = ""
        u = ""
        if get_flag("Py_DEBUG", hasattr(sys, "gettotalrefcount"), warn=(impl == "cp")):
            d = "d"

        abi = f"{impl}{tags.interpreter_version()}{d}{u}"
    elif soabi and impl == "cp" and soabi.startswith("cpython"):
        # non-Windows
        abi = "cp" + soabi.split("-")[1]
    elif soabi and impl == "cp" and soabi.startswith("cp"):
        # Windows
        abi = soabi.split("-")[0]
        if hasattr(sys, "gettotalrefcount"):
            # using debug build; append "d" flag
            abi += "d"
    elif soabi and impl == "pp":
        # we want something like pypy36-pp73
        abi = "-".join(soabi.split("-")[:2])
        abi = abi.replace(".", "_").replace("-", "_")
    elif soabi and impl == "graalpy":
        abi = "-".join(soabi.split("-")[:3])
        abi = abi.replace(".", "_").replace("-", "_")
    elif soabi:
        abi = soabi.replace(".", "_").replace("-", "_")
    else:
        abi = None

    return abi


def safer_version(version: str) -> str:
    return safe_version(version).replace("-", "_")


class bdist_wheel(Command):
    description = "create a wheel distribution"

    supported_compressions: ClassVar[dict[str, int]] = {
        "stored": ZIP_STORED,
        "deflated": ZIP_DEFLATED,
    }

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ("bdist-dir=", "b", "temporary directory for creating the distribution"),
        (
            "plat-name=",
            "p",
            (
                "platform name to embed in generated filenames "
                f"[default: {get_platform(None)}]"
            ),
        ),
        (
            "keep-temp",
            "k",
            (
                "keep the pseudo-installation tree around after "
                "creating the distribution archive"
            ),
        ),
        ("dist-dir=", "d", "directory to put final built distributions in"),
        ("skip-build", None, "skip rebuilding everything (for testing/debugging)"),
        (
            "relative",
            None,
            "build the archive using relative paths [default: false]",
        ),
        (
            "owner=",
            "u",
            "Owner name used when creating a tar file [default: current user]",
        ),
        (
            "group=",
            "g",
            "Group name used when creating a tar file [default: current group]",
        ),
        ("universal", None, "*DEPRECATED* make a universal wheel [default: false]"),
        (
            "compression=",
            None,
            f"zipfile compression (one of: {', '.join(supported_compressions)}) [default: 'deflated']",
        ),
        (
            "python-tag=",
            None,
            f"Python implementation compatibility tag [default: '{python_tag()}']",
        ),
        (
            "build-number=",
            None,
            (
                "Build number for this particular version. "
                "As specified in PEP-0427, this must start with a digit. "
                "[default: None]"
            ),
        ),
        (
            "py-limited-api=",
            None,
            "Python tag (cp32|cp33|cpNN) for abi3 wheel tag [default: false]",
        ),
        (
            "dist-info-dir=",
            None,
            (
                "directory where a pre-generated dist-info can be found (e.g. as a "
                "result of calling the PEP517 'prepare_metadata_for_build_wheel' "
                "method)"
            ),
        ),
    ]

    boolean_options: ClassVar[list[str]] = [
        "keep-temp",
        "skip-build",
        "relative",
        "universal",
    ]

    def initialize_options(self) -> None:
        self.bdist_dir: str | None = None
        self.data_dir = ""
        self.plat_name: str | None = None
        self.plat_tag: str | None = None
        self.format = "zip"
        self.keep_temp = False
        self.dist_dir: str | None = None
        self.dist_info_dir = None
        self.egginfo_dir: str | None = None
        self.root_is_pure: bool | None = None
        self.skip_build = False
        self.relative = False
        self.owner = None
        self.group = None
        self.universal = False
        self.compression: str | int = "deflated"
        self.python_tag = python_tag()
        self.build_number: str | None = None
        self.py_limited_api: str | Literal[False] = False
        self.plat_name_supplied = False

    def finalize_options(self) -> None:
        if not self.bdist_dir:
            bdist_base = self.get_finalized_command("bdist").bdist_base
            self.bdist_dir = os.path.join(bdist_base, "wheel")

        if self.dist_info_dir is None:
            egg_info = cast(egg_info_cls, self.distribution.get_command_obj("egg_info"))
            egg_info.ensure_finalized()  # needed for correct `wheel_dist_name`

        self.data_dir = self.wheel_dist_name + ".data"
        self.plat_name_supplied = bool(self.plat_name)

        need_options = ("dist_dir", "plat_name", "skip_build")

        self.set_undefined_options("bdist", *zip(need_options, need_options))

        self.root_is_pure = not (
            self.distribution.has_ext_modules() or self.distribution.has_c_libraries()
        )

        self._validate_py_limited_api()

        # Support legacy [wheel] section for setting universal
        wheel = self.distribution.get_option_dict("wheel")
        if "universal" in wheel:  # pragma: no cover
            # please don't define this in your global configs
            log.warn("The [wheel] section is deprecated. Use [bdist_wheel] instead.")
            val = wheel["universal"][1].strip()
            if val.lower() in ("1", "true", "yes"):
                self.universal = True

        if self.universal:
            SetuptoolsDeprecationWarning.emit(
                "bdist_wheel.universal is deprecated",
                """
                With Python 2.7 end-of-life, support for building universal wheels
                (i.e., wheels that support both Python 2 and Python 3)
                is being obviated.
                Please discontinue using this option, or if you still need it,
                file an issue with pypa/setuptools describing your use case.
                """,
                due_date=(2025, 8, 30),  # Introduced in 2024-08-30
            )

        if self.build_number is not None and not self.build_number[:1].isdigit():
            raise ValueError("Build tag (build-number) must start with a digit.")

    def _validate_py_limited_api(self) -> None:
        if not self.py_limited_api:
            return

        if not re.match(PY_LIMITED_API_PATTERN, self.py_limited_api):
            raise ValueError(f"py-limited-api must match '{PY_LIMITED_API_PATTERN}'")

        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            raise ValueError(
                f"`py_limited_api={self.py_limited_api!r}` not supported. "
                "`Py_LIMITED_API` is currently incompatible with "
                "`Py_GIL_DISABLED`. "
                "See https://github.com/python/cpython/issues/111506."
            )

    @property
    def wheel_dist_name(self) -> str:
        """Return distribution full name with - replaced with _"""
        components = [
            safer_name(self.distribution.get_name()),
            safer_version(self.distribution.get_version()),
        ]
        if self.build_number:
            components.append(self.build_number)
        return "-".join(components)

    def get_tag(self) -> tuple[str, str, str]:
        # bdist sets self.plat_name if unset, we should only use it for purepy
        # wheels if the user supplied it.
        if self.plat_name_supplied and self.plat_name:
            plat_name = self.plat_name
        elif self.root_is_pure:
            plat_name = "any"
        else:
            # macosx contains system version in platform name so need special handle
            if self.plat_name and not self.plat_name.startswith("macosx"):
                plat_name = self.plat_name
            else:
                # on macosx always limit the platform name to comply with any
                # c-extension modules in bdist_dir, since the user can specify
                # a higher MACOSX_DEPLOYMENT_TARGET via tools like CMake

                # on other platforms, and on macosx if there are no c-extension
                # modules, use the default platform name.
                plat_name = get_platform(self.bdist_dir)

            if _is_32bit_interpreter():
                if plat_name in ("linux-x86_64", "linux_x86_64"):
                    plat_name = "linux_i686"
                if plat_name in ("linux-aarch64", "linux_aarch64"):
                    # TODO armv8l, packaging pull request #690 => this did not land
                    # in pip/packaging yet
                    plat_name = "linux_armv7l"

        plat_name = (
            plat_name.lower().replace("-", "_").replace(".", "_").replace(" ", "_")
        )

        if self.root_is_pure:
            if self.universal:
                impl = "py2.py3"
            else:
                impl = self.python_tag
            tag = (impl, "none", plat_name)
        else:
            impl_name = tags.interpreter_name()
            impl_ver = tags.interpreter_version()
            impl = impl_name + impl_ver
            # We don't work on CPython 3.1, 3.0.
            if self.py_limited_api and (impl_name + impl_ver).startswith("cp3"):
                impl = self.py_limited_api
                abi_tag = "abi3"
            else:
                abi_tag = str(get_abi_tag()).lower()
            tag = (impl, abi_tag, plat_name)
            # issue gh-374: allow overriding plat_name
            supported_tags = [
                (t.interpreter, t.abi, plat_name) for t in tags.sys_tags()
            ]
            assert tag in supported_tags, (
                f"would build wheel with unsupported tag {tag}"
            )
        return tag

    def run(self):
        build_scripts = self.reinitialize_command("build_scripts")
        build_scripts.executable = "python"
        build_scripts.force = True

        build_ext = self.reinitialize_command("build_ext")
        build_ext.inplace = False

        if not self.skip_build:
            self.run_command("build")

        install = self.reinitialize_command("install", reinit_subcommands=True)
        install.root = self.bdist_dir
        install.compile = False
        install.skip_build = self.skip_build
        install.warn_dir = False

        # A wheel without setuptools scripts is more cross-platform.
        # Use the (undocumented) `no_ep` option to setuptools'
        # install_scripts command to avoid creating entry point scripts.
        install_scripts = self.reinitialize_command("install_scripts")
        install_scripts.no_ep = True

        # Use a custom scheme for the archive, because we have to decide
        # at installation time which scheme to use.
        for key in ("headers", "scripts", "data", "purelib", "platlib"):
            setattr(install, "install_" + key, os.path.join(self.data_dir, key))

        basedir_observed = ""

        if os.name == "nt":
            # win32 barfs if any of these are ''; could be '.'?
            # (distutils.command.install:change_roots bug)
            basedir_observed = os.path.normpath(os.path.join(self.data_dir, ".."))
            self.install_libbase = self.install_lib = basedir_observed

        setattr(
            install,
            "install_purelib" if self.root_is_pure else "install_platlib",
            basedir_observed,
        )

        log.info(f"installing to {self.bdist_dir}")

        self.run_command("install")

        impl_tag, abi_tag, plat_tag = self.get_tag()
        archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
        if not self.relative:
            archive_root = self.bdist_dir
        else:
            archive_root = os.path.join(
                self.bdist_dir, self._ensure_relative(install.install_base)
            )

        self.set_undefined_options("install_egg_info", ("target", "egginfo_dir"))
        distinfo_dirname = (
            f"{safer_name(self.distribution.get_name())}-"
            f"{safer_version(self.distribution.get_version())}.dist-info"
        )
        distinfo_dir = os.path.join(self.bdist_dir, distinfo_dirname)
        if self.dist_info_dir:
            # Use the given dist-info directly.
            log.debug(f"reusing {self.dist_info_dir}")
            shutil.copytree(self.dist_info_dir, distinfo_dir)
            # Egg info is still generated, so remove it now to avoid it getting
            # copied into the wheel.
            _shutil.rmtree(self.egginfo_dir)
        else:
            # Convert the generated egg-info into dist-info.
            self.egg2dist(self.egginfo_dir, distinfo_dir)

        self.write_wheelfile(distinfo_dir)

        # Make the archive
        if not os.path.exists(self.dist_dir):
            os.makedirs(self.dist_dir)

        wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
        with WheelFile(wheel_path, "w", self._zip_compression()) as wf:
            wf.write_files(archive_root)

        # Add to 'Distribution.dist_files' so that the "upload" command works
        getattr(self.distribution, "dist_files", []).append((
            "bdist_wheel",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            wheel_path,
        ))

        if not self.keep_temp:
            log.info(f"removing {self.bdist_dir}")
            _shutil.rmtree(self.bdist_dir)

    def write_wheelfile(
        self, wheelfile_base: str, generator: str = f"setuptools ({__version__})"
    ) -> None:
        from email.message import Message

        msg = Message()
        msg["Wheel-Version"] = "1.0"  # of the spec
        msg["Generator"] = generator
        msg["Root-Is-Purelib"] = str(self.root_is_pure).lower()
        if self.build_number is not None:
            msg["Build"] = self.build_number

        # Doesn't work for bdist_wininst
        impl_tag, abi_tag, plat_tag = self.get_tag()
        for impl in impl_tag.split("."):
            for abi in abi_tag.split("."):
                for plat in plat_tag.split("."):
                    msg["Tag"] = f"{impl}-{abi}-{plat}"

        wheelfile_path = os.path.join(wheelfile_base, "WHEEL")
        log.info(f"creating {wheelfile_path}")
        with open(wheelfile_path, "wb") as f:
            BytesGenerator(f, maxheaderlen=0).flatten(msg)

    def _ensure_relative(self, path: str) -> str:
        # copied from dir_util, deleted
        drive, path = os.path.splitdrive(path)
        if path[0:1] == os.sep:
            path = drive + path[1:]
        return path

    @property
    def license_paths(self) -> Iterable[str]:
        if setuptools_major_version >= 57:
            # Setuptools has resolved any patterns to actual file names
            return self.distribution.metadata.license_files or ()

        files = set[str]()
        metadata = self.distribution.get_option_dict("metadata")
        if setuptools_major_version >= 42:
            # Setuptools recognizes the license_files option but does not do globbing
            patterns = cast(Sequence[str], self.distribution.metadata.license_files)
        else:
            # Prior to those, wheel is entirely responsible for handling license files
            if "license_files" in metadata:
                patterns = metadata["license_files"][1].split()
            else:
                patterns = ()

        if "license_file" in metadata:
            warnings.warn(
                'The "license_file" option is deprecated. Use "license_files" instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            files.add(metadata["license_file"][1])

        if not files and not patterns and not isinstance(patterns, list):
            patterns = ("LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*")

        for pattern in patterns:
            for path in iglob(pattern):
                if path.endswith("~"):
                    log.debug(
                        f'ignoring license file "{path}" as it looks like a backup'
                    )
                    continue

                if path not in files and os.path.isfile(path):
                    log.info(
                        f'adding license file "{path}" (matched pattern "{pattern}")'
                    )
                    files.add(path)

        return files

    def egg2dist(self, egginfo_path: str, distinfo_path: str) -> None:
        """Convert an .egg-info directory into a .dist-info directory"""

        def adios(p: str) -> None:
            """Appropriately delete directory, file or link."""
            if os.path.exists(p) and not os.path.islink(p) and os.path.isdir(p):
                _shutil.rmtree(p)
            elif os.path.exists(p):
                os.unlink(p)

        adios(distinfo_path)

        if not os.path.exists(egginfo_path):
            # There is no egg-info. This is probably because the egg-info
            # file/directory is not named matching the distribution name used
            # to name the archive file. Check for this case and report
            # accordingly.
            import glob

            pat = os.path.join(os.path.dirname(egginfo_path), "*.egg-info")
            possible = glob.glob(pat)
            err = f"Egg metadata expected at {egginfo_path} but not found"
            if possible:
                alt = os.path.basename(possible[0])
                err += f" ({alt} found - possible misnamed archive file?)"

            raise ValueError(err)

        # .egg-info is a directory
        pkginfo_path = os.path.join(egginfo_path, "PKG-INFO")

        # ignore common egg metadata that is useless to wheel
        shutil.copytree(
            egginfo_path,
            distinfo_path,
            ignore=lambda x, y: {
                "PKG-INFO",
                "requires.txt",
                "SOURCES.txt",
                "not-zip-safe",
            },
        )

        # delete dependency_links if it is only whitespace
        dependency_links_path = os.path.join(distinfo_path, "dependency_links.txt")
        with open(dependency_links_path, encoding="utf-8") as dependency_links_file:
            dependency_links = dependency_links_file.read().strip()
        if not dependency_links:
            adios(dependency_links_path)

        metadata_path = os.path.join(distinfo_path, "METADATA")
        shutil.copy(pkginfo_path, metadata_path)

        licenses_folder_path = os.path.join(distinfo_path, "licenses")
        for license_path in self.license_paths:
            safe_path = _safe_license_file(license_path)
            dist_info_license_path = os.path.join(licenses_folder_path, safe_path)
            os.makedirs(os.path.dirname(dist_info_license_path), exist_ok=True)
            shutil.copy(license_path, dist_info_license_path)

        adios(egginfo_path)

    def _zip_compression(self) -> int:
        if (
            isinstance(self.compression, int)
            and self.compression in self.supported_compressions.values()
        ):
            return self.compression

        compression = self.supported_compressions.get(str(self.compression))
        if compression is not None:
            return compression

        raise ValueError(f"Unsupported compression: {self.compression!r}")
<|endoftext|>from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterator
from itertools import chain
from typing import ClassVar

from .._importlib import metadata
from ..dist import Distribution
from .build import _ORIGINAL_SUBCOMMANDS

import distutils.command.sdist as orig
from distutils import log

_default_revctrl = list


def walk_revctrl(dirname='') -> Iterator:
    """Find all files under revision control"""
    for ep in metadata.entry_points(group='setuptools.file_finders'):
        yield from ep.load()(dirname)


class sdist(orig.sdist):
    """Smart sdist that finds anything supported by revision control"""

    # Match the (narrower) type of the distutils sdist base rather than the
    # Command base union, which mypy rejects as an incompatible override when
    # distutils stubs are present (Python < 3.12).
    user_options: ClassVar[list[tuple[str, str | None, str]]] = [
        ('formats=', None, "formats for source distribution (comma-separated list)"),
        (
            'keep-temp',
            'k',
            "keep the distribution tree around after creating archive file(s)",
        ),
        (
            'dist-dir=',
            'd',
            "directory to put the source distribution archive(s) in [default: dist]",
        ),
        (
            'owner=',
            'u',
            "Owner name used when creating a tar file [default: current user]",
        ),
        (
            'group=',
            'g',
            "Group name used when creating a tar file [default: current group]",
        ),
    ]

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution
    negative_opt: ClassVar[dict[str, str]] = {}

    README_EXTENSIONS: ClassVar[list[str]] = ['', '.rst', '.txt', '.md']
    READMES = tuple(f'README{ext}' for ext in README_EXTENSIONS)

    def run(self) -> None:
        self.run_command('egg_info')
        ei_cmd = self.get_finalized_command('egg_info')
        self.filelist = ei_cmd.filelist
        self.filelist.append(os.path.join(ei_cmd.egg_info, 'SOURCES.txt'))
        self.check_readme()

        # Run sub commands
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

        self.make_distribution()

        dist_files = getattr(self.distribution, 'dist_files', [])
        for file in self.archive_files:
            data = ('sdist', '', file)
            if data not in dist_files:
                dist_files.append(data)

    def initialize_options(self) -> None:
        orig.sdist.initialize_options(self)

    def make_distribution(self) -> None:
        """
        Workaround for #516
        """
        with self._remove_os_link():
            orig.sdist.make_distribution(self)

    @staticmethod
    @contextlib.contextmanager
    def _remove_os_link():
        """
        In a context, remove and restore os.link if it exists
        """

        class NoValue:
            pass

        orig_val = getattr(os, 'link', NoValue)
        try:
            del os.link
        except Exception:  # noqa: BLE001, S110 # best-effort; failure is non-fatal
            pass
        try:
            yield
        finally:
            if orig_val is not NoValue:
                os.link = orig_val

    def add_defaults(self) -> None:
        super().add_defaults()
        self._add_defaults_build_sub_commands()

    def _add_defaults_optional(self):
        super()._add_defaults_optional()
        if os.path.isfile('pyproject.toml'):
            self.filelist.append('pyproject.toml')

    def _add_defaults_python(self):
        """getting python files"""
        if self.distribution.has_pure_modules():
            build_py = self.get_finalized_command('build_py')
            self.filelist.extend(build_py.get_source_files())
            self._add_data_files(self._safe_data_files(build_py))

    def _add_defaults_build_sub_commands(self):
        build = self.get_finalized_command("build")
        missing_cmds = set(build.get_sub_commands()) - _ORIGINAL_SUBCOMMANDS
        # ^-- the original built-in sub-commands are already handled by default.
        cmds = (self.get_finalized_command(c) for c in missing_cmds)
        files = (c.get_source_files() for c in cmds if hasattr(c, "get_source_files"))
        self.filelist.extend(chain.from_iterable(files))

    def _safe_data_files(self, build_py):
        """
        Since the ``sdist`` class is also used to compute the MANIFEST
        (via :obj:`setuptools.command.egg_info.manifest_maker`),
        there might be recursion problems when trying to obtain the list of
        data_files and ``include_package_data=True`` (which in turn depends on
        the files included in the MANIFEST).

        To avoid that, ``manifest_maker`` should be able to overwrite this
        method and avoid recursive attempts to build/analyze the MANIFEST.
        """
        return build_py.data_files

    def _add_data_files(self, data_files):
        """
        Add data files as found in build_py.data_files.
        """
        self.filelist.extend(
            os.path.join(src_dir, name)
            for _, src_dir, _, filenames in data_files
            for name in filenames
        )

    def _add_defaults_data_files(self):
        try:
            super()._add_defaults_data_files()
        except TypeError:
            log.warn("data_files contains unexpected objects")

    def prune_file_list(self) -> None:
        super().prune_file_list()
        # Prevent accidental inclusion of test-related cache dirs at the project root
        sep = re.escape(os.sep)
        self.filelist.exclude_pattern(r"^(\.tox|\.nox|\.venv)" + sep, is_regex=True)

    def check_readme(self) -> None:
        for f in self.READMES:
            if os.path.exists(f):
                return
        self.warn(
            "standard file not found: should have one of " + ', '.join(self.READMES)
        )

    def make_release_tree(self, base_dir, files) -> None:
        orig.sdist.make_release_tree(self, base_dir, files)

        # Save any egg_info command line options used to create this sdist
        dest = os.path.join(base_dir, 'setup.cfg')
        if hasattr(os, 'link') and os.path.exists(dest):
            # unlink and re-copy, since it might be hard-linked, and
            # we don't want to change the source version
            os.unlink(dest)
            self.copy_file('setup.cfg', dest)

        self.get_finalized_command('egg_info').save_version_info(dest)

    def _manifest_is_not_generated(self):
        # check for special comment used in 2.7.1 and higher
        if not os.path.isfile(self.manifest):
            return False

        with open(self.manifest, 'rb') as fp:
            first_line = fp.readline()
        return first_line != b'# file GENERATED by distutils, do NOT edit\n'

    def read_manifest(self) -> None:
        """Read the manifest file (named by 'self.manifest') and use it to
        fill in 'self.filelist', the list of files to include in the source
        distribution.
        """
        log.info("reading manifest file '%s'", self.manifest)
        manifest = open(self.manifest, 'rb')  # noqa: SIM115 # handle managed explicitly
        for bytes_line in manifest:
            # The manifest must contain UTF-8. See #303.
            try:
                line = bytes_line.decode('UTF-8')
            except UnicodeDecodeError:
                log.warn(f"{line!r} not UTF-8 decodable -- skipping")
                continue
            # ignore comments and blank lines
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            self.filelist.append(line)
        manifest.close()
<|endoftext|>"""setuptools.command.egg_info

Create a distribution's .egg-info directory and contents"""

from __future__ import annotations

import functools
import os
import re
import sys
import time
from collections.abc import Callable
from typing import ClassVar

import packaging
import packaging.requirements
import packaging.version

from setuptools import Command, unicode_utils
from setuptools.command import bdist_egg
from setuptools.command.sdist import sdist, walk_revctrl
from setuptools.command.setopt import edit_config
from setuptools.glob import glob

from .. import _entry_points, _normalization
from .._importlib import metadata
from ..warnings import SetuptoolsDeprecationWarning
from . import _requirestxt

import distutils.errors
import distutils.filelist
from distutils import log
from distutils.errors import DistutilsInternalError
from distutils.filelist import FileList as _FileList
from distutils.util import convert_path

PY_MAJOR = f'{sys.version_info.major}.{sys.version_info.minor}'


class _NormalizedMatcher:
    """
    Wrap a compiled pattern so that matching is insensitive to Unicode
    normalization form.

    File names walked from disk (NFD on macOS APFS/HFS+) and patterns from
    ``MANIFEST.in`` (typically NFC) can denote the same file while differing
    byte-for-byte. Normalizing both sides before matching keeps an exclusion
    (or inclusion) from silently failing. See GHSA-h35f-9h28-mq5c.
    """

    def __init__(self, pattern: re.Pattern) -> None:
        self._pattern = pattern

    def match(self, path):
        return self._pattern.match(unicode_utils.normalize(path))

    def search(self, path):
        return self._pattern.search(unicode_utils.normalize(path))


def translate_pattern(glob):  # noqa: C901  # is too complex (14)  # FIXME
    """
    Translate a file path glob like '*.txt' in to a regular expression.
    This differs from fnmatch.translate which allows wildcards to match
    directory separators. It also knows about '**/' which matches any number of
    directories.
    """
    pat = ''

    # Normalize the pattern so it matches paths regardless of the Unicode
    # normalization form used on disk (GHSA-h35f-9h28-mq5c). Candidate paths
    # are normalized to the same form by ``_NormalizedMatcher``.
    glob = unicode_utils.normalize(glob)

    # This will split on '/' within [character classes]. This is deliberate.
    chunks = glob.split(os.path.sep)

    sep = re.escape(os.sep)
    valid_char = f'[^{sep}]'

    for c, chunk in enumerate(chunks):
        last_chunk = c == len(chunks) - 1

        # Chunks that are a literal ** are globstars. They match anything.
        if chunk == '**':
            if last_chunk:
                # Match anything if this is the last component
                pat += '.*'
            else:
                # Match '(name/)*'
                pat += f'(?:{valid_char}+{sep})*'
            continue  # Break here as the whole path component has been handled

        # Find any special characters in the remainder
        i = 0
        chunk_len = len(chunk)
        while i < chunk_len:
            char = chunk[i]
            if char == '*':
                # Match any number of name characters
                pat += valid_char + '*'
            elif char == '?':
                # Match a name character
                pat += valid_char
            elif char == '[':
                # Character class
                inner_i = i + 1
                # Skip initial !/] chars
                if inner_i < chunk_len and chunk[inner_i] == '!':
                    inner_i = inner_i + 1
                if inner_i < chunk_len and chunk[inner_i] == ']':
                    inner_i = inner_i + 1

                # Loop till the closing ] is found
                while inner_i < chunk_len and chunk[inner_i] != ']':
                    inner_i = inner_i + 1

                if inner_i >= chunk_len:
                    # Got to the end of the string without finding a closing ]
                    # Do not treat this as a matching group, but as a literal [
                    pat += re.escape(char)
                else:
                    # Grab the insides of the [brackets]
                    inner = chunk[i + 1 : inner_i]
                    char_class = ''

                    # Class negation
                    if inner[0] == '!':
                        char_class = '^'
                        inner = inner[1:]

                    char_class += re.escape(inner)
                    pat += f'[{char_class}]'

                    # Skip to the end ]
                    i = inner_i
            else:
                pat += re.escape(char)
            i += 1

        # Join each chunk with the dir separator
        if not last_chunk:
            pat += sep

    pat += r'\Z'
    return _NormalizedMatcher(re.compile(pat, flags=re.MULTILINE | re.DOTALL))


class InfoCommon:
    tag_build = None
    tag_date = None

    @property
    def name(self):
        return _normalization.safe_name(self.distribution.get_name())

    def tagged_version(self):
        tagged = self._maybe_tag(self.distribution.get_version())
        return _normalization.safe_version(tagged)

    def _maybe_tag(self, version):
        """
        egg_info may be called more than once for a distribution,
        in which case the version string already contains all tags.
        """
        return (
            version
            if self.vtags and self._already_tagged(version)
            else version + self.vtags
        )

    def _already_tagged(self, version: str) -> bool:
        # Depending on their format, tags may change with version normalization.
        # So in addition the regular tags, we have to search for the normalized ones.
        return version.endswith((self.vtags, self._safe_tags()))

    def _safe_tags(self) -> str:
        # To implement this we can rely on `safe_version` pretending to be version 0
        # followed by tags. Then we simply discard the starting 0 (fake version number)
        try:
            return _normalization.safe_version(f"0{self.vtags}")[1:]
        except packaging.version.InvalidVersion:
            return _normalization.safe_name(self.vtags.replace(' ', '.'))

    def tags(self) -> str:
        version = ''
        if self.tag_build:
            version += self.tag_build
        if self.tag_date:
            version += time.strftime("%Y%m%d")
        return version

    vtags = property(tags)


class egg_info(InfoCommon, Command):
    description = "create a distribution's .egg-info directory"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        (
            'egg-base=',
            'e',
            (
                "directory containing .egg-info directories"
                " [default: top of the source tree]"
            ),
        ),
        ('tag-date', 'd', "Add date stamp (e.g. 20050528) to version number"),
        ('tag-build=', 'b', "Specify explicit tag to add to version number"),
        ('no-date', 'D', "Don't include date stamp [default]"),
    ]

    boolean_options: ClassVar[list[str]] = ['tag-date']
    negative_opt: ClassVar[dict[str, str]] = {
        'no-date': 'tag-date',
    }

    def initialize_options(self):
        self.egg_base = None
        self.egg_name = None
        self.egg_info = None
        self.egg_version = None
        self.ignore_egg_info_in_manifest = False

    ####################################
    # allow the 'tag_svn_revision' to be detected and
    # set, supporting sdists built on older Setuptools.
    @property
    def tag_svn_revision(self) -> int | None:
        pass

    @tag_svn_revision.setter
    def tag_svn_revision(self, value) -> None:
        pass

    ####################################

    def save_version_info(self, filename) -> None:
        """
        Materialize the value of date into the
        build tag. Install build keys in a deterministic order
        to avoid arbitrary reordering on subsequent builds.
        """
        # follow the order these keys would have been added
        # when PYTHONHASHSEED=0
        egg_info = dict(tag_build=self.tags(), tag_date=0)
        edit_config(filename, dict(egg_info=egg_info))

    def finalize_options(self) -> None:
        # Note: we need to capture the current value returned
        # by `self.tagged_version()`, so we can later update
        # `self.distribution.metadata.version` without
        # repercussions.
        self.egg_name = self.name
        self.egg_version = self.tagged_version()
        parsed_version = packaging.version.Version(self.egg_version)

        try:
            is_version = isinstance(parsed_version, packaging.version.Version)
            spec = "%s==%s" if is_version else "%s===%s"
            packaging.requirements.Requirement(spec % (self.egg_name, self.egg_version))
        except ValueError as e:
            raise distutils.errors.DistutilsOptionError(
                f"Invalid distribution name or version syntax: {self.egg_name}-{self.egg_version}"
            ) from e

        if self.egg_base is None:
            dirs = self.distribution.package_dir
            self.egg_base = (dirs or {}).get('', os.curdir)

        self.ensure_dirname('egg_base')
        self.egg_info = _normalization.filename_component(self.egg_name) + '.egg-info'
        if self.egg_base != os.curdir:
            self.egg_info = os.path.join(self.egg_base, self.egg_info)

        # Set package version for the benefit of dumber commands
        # (e.g. sdist, bdist_wininst, etc.)
        #
        self.distribution.metadata.version = self.egg_version

    def _get_egg_basename(self, py_version=PY_MAJOR, platform=None):
        """Compute filename of the output egg. Private API."""
        return _egg_basename(self.egg_name, self.egg_version, py_version, platform)

    def write_or_delete_file(self, what, filename, data, force: bool = False) -> None:
        """Write `data` to `filename` or delete if empty

        If `data` is non-empty, this routine is the same as ``write_file()``.
        If `data` is empty but not ``None``, this is the same as calling
        ``delete_file(filename)`.  If `data` is ``None``, then this is a no-op
        unless `filename` exists, in which case a warning is issued about the
        orphaned file (if `force` is false), or deleted (if `force` is true).
        """
        if data:
            self.write_file(what, filename, data)
        elif os.path.exists(filename):
            if data is None and not force:
                log.warn("%s not set in setup(), but %s exists", what, filename)
                return
            else:
                self.delete_file(filename)

    def write_file(self, what, filename, data) -> None:
