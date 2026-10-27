        """Write `data` to `filename` (if not a dry run) after announcing it

        `what` is used in a log message to identify what is being written
        to the file.
        """
        log.info("writing %s to %s", what, filename)
        data = data.encode("utf-8")
        f = open(filename, 'wb')  # noqa: SIM115 # handle managed explicitly
        f.write(data)
        f.close()

    def delete_file(self, filename) -> None:
        """Delete `filename` (if not a dry run) after announcing it"""
        log.info("deleting %s", filename)
        os.unlink(filename)

    def run(self) -> None:
        # Pre-load to avoid iterating over entry-points while an empty .egg-info
        # exists in sys.path. See pypa/pyproject-hooks#206
        writers = list(metadata.entry_points(group='egg_info.writers'))

        self.mkpath(self.egg_info)
        try:
            os.utime(self.egg_info, None)
        except OSError as e:
            msg = f"Cannot update time stamp of directory '{self.egg_info}'"
            raise distutils.errors.DistutilsFileError(msg) from e
        for ep in writers:
            writer = ep.load()
            writer(self, ep.name, os.path.join(self.egg_info, ep.name))

        # Get rid of native_libs.txt if it was put there by older bdist_egg
        nl = os.path.join(self.egg_info, "native_libs.txt")
        if os.path.exists(nl):
            self.delete_file(nl)

        self.find_sources()

    def find_sources(self) -> None:
        """Generate SOURCES.txt manifest file"""
        manifest_filename = os.path.join(self.egg_info, "SOURCES.txt")
        mm = manifest_maker(self.distribution)
        mm.ignore_egg_info_dir = self.ignore_egg_info_in_manifest
        mm.manifest = manifest_filename
        mm.run()
        self.filelist = mm.filelist


class FileList(_FileList):
    # Implementations of the various MANIFEST.in commands

    def __init__(
        self, warn=None, debug_print=None, ignore_egg_info_dir: bool = False
    ) -> None:
        super().__init__(warn, debug_print)
        self.ignore_egg_info_dir = ignore_egg_info_dir

    def process_template_line(self, line) -> None:
        # Parse the line: split it up, make sure the right number of words
        # is there, and return the relevant words.  'action' is always
        # defined: it's the first word of the line.  Which of the other
        # three are defined depends on the action; it'll be either
        # patterns, (dir and patterns), or (dir_pattern).
        (action, patterns, dir, dir_pattern) = self._parse_template_line(line)

        action_map: dict[str, Callable] = {
            'include': self.include,
            'exclude': self.exclude,
            'global-include': self.global_include,
            'global-exclude': self.global_exclude,
            'recursive-include': functools.partial(
                self.recursive_include,
                dir,
            ),
            'recursive-exclude': functools.partial(
                self.recursive_exclude,
                dir,
            ),
            'graft': self.graft,
            'prune': self.prune,
        }
        log_map = {
            'include': "warning: no files found matching '%s'",
            'exclude': ("warning: no previously-included files found matching '%s'"),
            'global-include': (
                "warning: no files found matching '%s' anywhere in distribution"
            ),
            'global-exclude': (
                "warning: no previously-included files matching "
                "'%s' found anywhere in distribution"
            ),
            'recursive-include': (
                "warning: no files found matching '%s' under directory '%s'"
            ),
            'recursive-exclude': (
                "warning: no previously-included files matching "
                "'%s' found under directory '%s'"
            ),
            'graft': "warning: no directories found matching '%s'",
            'prune': "no previously-included directories found matching '%s'",
        }

        try:
            process_action = action_map[action]
        except KeyError:
            msg = f"Invalid MANIFEST.in: unknown action {action!r} in {line!r}"
            raise DistutilsInternalError(msg) from None

        # OK, now we know that the action is valid and we have the
        # right number of words on the line for that action -- so we
        # can proceed with minimal error-checking.

        action_is_recursive = action.startswith('recursive-')
        if action in {'graft', 'prune'}:
            patterns = [dir_pattern]
        extra_log_args = (dir,) if action_is_recursive else ()
        log_tmpl = log_map[action]

        self.debug_print(
            ' '.join(
                [action] + ([dir] if action_is_recursive else []) + patterns,
            )
        )
        for pattern in patterns:
            if not process_action(pattern):
                log.warn(log_tmpl, pattern, *extra_log_args)

    def _remove_files(self, predicate):
        """
        Remove all files from the file list that match the predicate.
        Return True if any matching files were removed
        """
        found = False
        for i in range(len(self.files) - 1, -1, -1):
            if predicate(self.files[i]):
                self.debug_print(" removing " + self.files[i])
                del self.files[i]
                found = True
        return found

    def include(self, pattern):
        """Include files that match 'pattern'."""
        found = [f for f in glob(pattern) if not os.path.isdir(f)]
        self.extend(found)
        return bool(found)

    def exclude(self, pattern):
        """Exclude files that match 'pattern'."""
        match = translate_pattern(pattern)
        return self._remove_files(match.match)

    def recursive_include(self, dir, pattern):
        """
        Include all files anywhere in 'dir/' that match the pattern.
        """
        full_pattern = os.path.join(dir, '**', pattern)
        found = [f for f in glob(full_pattern, recursive=True) if not os.path.isdir(f)]
        self.extend(found)
        return bool(found)

    def recursive_exclude(self, dir, pattern):
        """
        Exclude any file anywhere in 'dir/' that match the pattern.
        """
        match = translate_pattern(os.path.join(dir, '**', pattern))
        return self._remove_files(match.match)

    def graft(self, dir):
        """Include all files from 'dir/'."""
        found = [
            item
            for match_dir in glob(dir)
            for item in distutils.filelist.findall(match_dir)
        ]
        self.extend(found)
        return bool(found)

    def prune(self, dir):
        """Filter out files from 'dir/'."""
        match = translate_pattern(os.path.join(dir, '**'))
        return self._remove_files(match.match)

    def global_include(self, pattern):
        """
        Include all files anywhere in the current directory that match the
        pattern. This is very inefficient on large file trees.
        """
        if self.allfiles is None:
            self.findall()
        match = translate_pattern(os.path.join('**', pattern))
        found = [f for f in self.allfiles if match.match(f)]
        self.extend(found)
        return bool(found)

    def global_exclude(self, pattern):
        """
        Exclude all files anywhere that match the pattern.
        """
        match = translate_pattern(os.path.join('**', pattern))
        return self._remove_files(match.match)

    def append(self, item) -> None:
        item = item.removesuffix('\r')  # Fix older sdists built on Windows
        path = convert_path(item)

        if self._safe_path(path):
            self.files.append(path)

    def extend(self, paths) -> None:
        self.files.extend(filter(self._safe_path, paths))

    def _repair(self):
        """
        Replace self.files with only safe paths

        Because some owners of FileList manipulate the underlying
        ``files`` attribute directly, this method must be called to
        repair those paths.
        """
        self.files = list(filter(self._safe_path, self.files))

    def _safe_path(self, path):
        enc_warn = "'%s' not %s encodable -- skipping"

        # To avoid accidental trans-codings errors, first to unicode
        u_path = unicode_utils.filesys_decode(path)
        if u_path is None:
            log.warn(f"'{path}' in unexpected encoding -- skipping")
            return False

        # Must ensure utf-8 encodability
        utf8_path = unicode_utils.try_encode(u_path, "utf-8")
        if utf8_path is None:
            log.warn(enc_warn, path, 'utf-8')
            return False

        try:
            # ignore egg-info paths
            is_egg_info = ".egg-info" in u_path or b".egg-info" in utf8_path
            if self.ignore_egg_info_dir and is_egg_info:
                return False
            # accept is either way checks out
            if os.path.exists(u_path) or os.path.exists(utf8_path):
                return True
        # this will catch any encode errors decoding u_path
        except UnicodeEncodeError:
            log.warn(enc_warn, path, sys.getfilesystemencoding())


class manifest_maker(sdist):
    template = "MANIFEST.in"

    def initialize_options(self) -> None:
        self.use_defaults = True
        self.prune = True
        self.manifest_only = True
        self.force_manifest = True
        self.ignore_egg_info_dir = False

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        self.filelist = FileList(ignore_egg_info_dir=self.ignore_egg_info_dir)
        if not os.path.exists(self.manifest):
            self.write_manifest()  # it must exist so it'll get in the list
        self.add_defaults()
        if os.path.exists(self.template):
            self.read_template()
        self.add_license_files()
        self._add_referenced_files()
        self.prune_file_list()
        self.filelist.sort()
        self.filelist.remove_duplicates()
        self.write_manifest()

    def _manifest_normalize(self, path):
        path = unicode_utils.filesys_decode(path)
        return path.replace(os.sep, '/')

    def write_manifest(self) -> None:
        """
        Write the file list in 'self.filelist' to the manifest file
        named by 'self.manifest'.
        """
        self.filelist._repair()

        # Now _repairs should encodability, but not unicode
        files = [self._manifest_normalize(f) for f in self.filelist.files]
        msg = f"writing manifest file '{self.manifest}'"
        self.execute(write_file, (self.manifest, files), msg)

    def warn(self, msg) -> None:
        if not self._should_suppress_warning(msg):
            sdist.warn(self, msg)

    @staticmethod
    def _should_suppress_warning(msg):
        """
        suppress missing-file warnings from sdist
        """
        return re.match(r"standard file .*not found", msg)

    def add_defaults(self) -> None:
        sdist.add_defaults(self)
        self.filelist.append(self.template)
        self.filelist.append(self.manifest)
        rcfiles = list(walk_revctrl())
        if rcfiles:
            self.filelist.extend(rcfiles)
        elif os.path.exists(self.manifest):
            self.read_manifest()

        if os.path.exists("setup.py"):
            # setup.py should be included by default, even if it's not
            # the script called to create the sdist
            self.filelist.append("setup.py")

        ei_cmd = self.get_finalized_command('egg_info')
        self.filelist.graft(ei_cmd.egg_info)

    def add_license_files(self) -> None:
        license_files = self.distribution.metadata.license_files or []
        for lf in license_files:
            log.info("adding license file '%s'", lf)
        self.filelist.extend(license_files)

    def _add_referenced_files(self):
        """Add files referenced by the config (e.g. `file:` directive) to filelist"""
        referenced = getattr(self.distribution, '_referenced_files', [])
        # ^-- fallback if dist comes from distutils or is a custom class
        for rf in referenced:
            log.debug("adding file referenced by config '%s'", rf)
        self.filelist.extend(referenced)

    def _safe_data_files(self, build_py):
        """
        The parent class implementation of this method
        (``sdist``) will try to include data files, which
        might cause recursion problems when
        ``include_package_data=True``.

        Therefore, avoid triggering any attempt of
        analyzing/building the manifest again.
        """
        if hasattr(build_py, 'get_data_files_without_manifest'):
            return build_py.get_data_files_without_manifest()

        SetuptoolsDeprecationWarning.emit(
            "`build_py` command does not inherit from setuptools' `build_py`.",
            """
            Custom 'build_py' does not implement 'get_data_files_without_manifest'.
            Please extend command classes from setuptools instead of distutils.
            """,
            see_url="https://peps.python.org/pep-0632/",
            # due_date not defined yet, old projects might still do it?
        )
        return build_py.get_data_files()


def write_file(filename, contents) -> None:
    """Create a file with the specified name and write 'contents' (a
    sequence of strings without line terminators) to it.
    """
    contents = "\n".join(contents)

    # assuming the contents has been vetted for utf-8 encoding
    contents = contents.encode("utf-8")

    with open(filename, "wb") as f:  # always write POSIX-style manifest
        f.write(contents)


def write_pkg_info(cmd, basename, filename) -> None:
    log.info("writing %s", filename)
    metadata = cmd.distribution.metadata
    metadata.version, oldver = cmd.egg_version, metadata.version
    metadata.name, oldname = cmd.egg_name, metadata.name

    try:
        metadata.write_pkg_info(cmd.egg_info)
    finally:
        metadata.name, metadata.version = oldname, oldver

    safe = getattr(cmd.distribution, 'zip_safe', None)

    bdist_egg.write_safety_flag(cmd.egg_info, safe)


def warn_depends_obsolete(cmd, basename, filename) -> None:
    """
    Unused: left to avoid errors when updating (from source) from <= 67.8.
    Old installations have a .dist-info directory with the entry-point
    ``depends.txt = setuptools.command.egg_info:warn_depends_obsolete``.
    This may trigger errors when running the first egg_info in build_meta.
    TODO: Remove this function in a version sufficiently > 68.
    """


# Export API used in entry_points
write_requirements = _requirestxt.write_requirements
write_setup_requirements = _requirestxt.write_setup_requirements


def write_toplevel_names(cmd, basename, filename) -> None:
    pkgs = dict.fromkeys([
        k.split('.', 1)[0] for k in cmd.distribution.iter_distribution_names()
    ])
    cmd.write_file("top-level names", filename, '\n'.join(sorted(pkgs)) + '\n')


def overwrite_arg(cmd, basename, filename) -> None:
    write_arg(cmd, basename, filename, True)


def write_arg(cmd, basename, filename, force: bool = False) -> None:
    argname = os.path.splitext(basename)[0]
    value = getattr(cmd.distribution, argname, None)
    if value is not None:
        value = '\n'.join(value) + '\n'
    cmd.write_or_delete_file(argname, filename, value, force)


def write_entries(cmd, basename, filename) -> None:
    eps = _entry_points.load(cmd.distribution.entry_points)
    defn = _entry_points.render(eps)
    cmd.write_or_delete_file('entry points', filename, defn, True)


def _egg_basename(egg_name, egg_version, py_version=None, platform=None):
    """Compute filename of the output egg. Private API."""
    name = _normalization.filename_component(egg_name)
    version = _normalization.filename_component(egg_version)
    egg = f"{name}-{version}-py{py_version or PY_MAJOR}"
    if platform:
        egg += f"-{platform}"
    return egg


class EggInfoDeprecationWarning(SetuptoolsDeprecationWarning):
    """Deprecated behavior warning for EggInfo, bypassing suppression."""
<|endoftext|>"""setuptools.command.bdist_egg

Build .egg distributions"""

from __future__ import annotations

import marshal
import os
import re
import sys
import textwrap
from collections.abc import Iterator
from sysconfig import get_path, get_platform, get_python_version
from types import CodeType
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar

from setuptools import Command
from setuptools.extension import Library

from .._path import StrPath, StrPathT, ensure_directory

from distutils import log
from distutils.dir_util import mkpath, remove_tree

if TYPE_CHECKING:
    from typing import TypeAlias

    from _typeshed import GenericPath

    _StrOrBytesT = TypeVar("_StrOrBytesT", str, bytes)

# Same as zipfile._ZipFileMode from typeshed
_ZipFileMode: TypeAlias = Literal["r", "w", "x", "a"]


def _get_purelib():
    return get_path("purelib")


def strip_module(filename):
    if '.' in filename:
        filename = os.path.splitext(filename)[0]
    filename = filename.removesuffix('module')
    return filename


def sorted_walk(
    dir: GenericPath[_StrOrBytesT],
) -> Iterator[tuple[_StrOrBytesT, list[_StrOrBytesT], list[_StrOrBytesT]]]:
    """Do os.walk in a reproducible way,
    independent of indeterministic filesystem readdir order
    """
    for base, dirs, files in os.walk(dir):
        dirs.sort()
        files.sort()
        yield base, dirs, files


def write_stub(resource, pyfile) -> None:
    _stub_template = textwrap.dedent(
        """
        def __bootstrap__():
            global __bootstrap__, __loader__, __file__
            import sys, importlib.resources as irs, importlib.util
            with irs.as_file(irs.files(__name__).joinpath(%r)) as __file__:
                __loader__ = None; del __bootstrap__, __loader__
                spec = importlib.util.spec_from_file_location(__name__,__file__)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
        __bootstrap__()
        """
    ).lstrip()
    with open(pyfile, 'w', encoding="utf-8") as f:
        f.write(_stub_template % resource)


class bdist_egg(Command):
    description = 'create an "egg" distribution'

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('bdist-dir=', 'b', "temporary directory for creating the distribution"),
        (
            'plat-name=',
            'p',
            (
                "platform name to embed in generated filenames "
                "(by default uses `sysconfig.get_platform()`)"
            ),
        ),
        ('exclude-source-files', None, "remove all .py files from the generated egg"),
        (
            'keep-temp',
            'k',
            (
                "keep the pseudo-installation tree around after "
                "creating the distribution archive"
            ),
        ),
        ('dist-dir=', 'd', "directory to put final built distributions in"),
        ('skip-build', None, "skip rebuilding everything (for testing/debugging)"),
    ]

    boolean_options: ClassVar[list[str]] = [
        'keep-temp',
        'skip-build',
        'exclude-source-files',
    ]

    def initialize_options(self):
        self.bdist_dir = None
        self.plat_name = None
        self.keep_temp = False
        self.dist_dir = None
        self.skip_build = False
        self.egg_output = None
        self.exclude_source_files = None

    def finalize_options(self) -> None:
        ei_cmd = self.ei_cmd = self.get_finalized_command("egg_info")
        self.egg_info = ei_cmd.egg_info

        if self.bdist_dir is None:
            bdist_base = self.get_finalized_command('bdist').bdist_base
            self.bdist_dir = os.path.join(bdist_base, 'egg')

        if self.plat_name is None:
            self.plat_name = get_platform()

        self.set_undefined_options('bdist', ('dist_dir', 'dist_dir'))

        if self.egg_output is None:
            # Compute filename of the output egg
            basename = ei_cmd._get_egg_basename(
                py_version=get_python_version(),
                platform=self.distribution.has_ext_modules() and self.plat_name,
            )

            self.egg_output = os.path.join(self.dist_dir, basename + '.egg')

    def do_install_data(self) -> None:
        # Hack for packages that install data to install's --install-lib
        self.get_finalized_command('install').install_lib = self.bdist_dir

        site_packages = os.path.normcase(os.path.realpath(_get_purelib()))
        old, self.distribution.data_files = self.distribution.data_files, []

        for item in old:
            if isinstance(item, tuple) and len(item) == 2 and os.path.isabs(item[0]):
                realpath = os.path.realpath(item[0])
                normalized = os.path.normcase(realpath)
                if normalized == site_packages or normalized.startswith(
                    site_packages + os.sep
                ):
                    item = realpath[len(site_packages) + 1 :], item[1]
                    # XXX else: raise ???
            self.distribution.data_files.append(item)

        try:
            log.info("installing package data to %s", self.bdist_dir)
            self.call_command('install_data', force=False, root=None)
        finally:
            self.distribution.data_files = old

    def get_outputs(self):
        return [self.egg_output]

    def call_command(self, cmdname, **kw):
        """Invoke reinitialized command `cmdname` with keyword args"""
        for dirname in INSTALL_DIRECTORY_ATTRS:
            kw.setdefault(dirname, self.bdist_dir)
        kw.setdefault('skip_build', self.skip_build)
        cmd = self.reinitialize_command(cmdname, **kw)
        self.run_command(cmdname)
        return cmd

    def run(self) -> None:  # noqa: C901  # is too complex (14)  # FIXME
        # Generate metadata first
        self.run_command("egg_info")
        # We run install_lib before install_data, because some data hacks
        # pull their data path from the install_lib command.
        log.info("installing library code to %s", self.bdist_dir)
        instcmd = self.get_finalized_command('install')
        old_root = instcmd.root
        instcmd.root = None
        if self.distribution.has_c_libraries() and not self.skip_build:
            self.run_command('build_clib')
        cmd = self.call_command('install_lib', warn_dir=False)
        instcmd.root = old_root

        all_outputs, ext_outputs = self.get_ext_outputs()
        self.stubs = []
        to_compile = []
        for p, ext_name in enumerate(ext_outputs):
            filename, _ext = os.path.splitext(ext_name)
            pyfile = os.path.join(self.bdist_dir, strip_module(filename) + '.py')
            self.stubs.append(pyfile)
            log.info("creating stub loader for %s", ext_name)
            write_stub(os.path.basename(ext_name), pyfile)
            to_compile.append(pyfile)
            ext_outputs[p] = ext_name.replace(os.sep, '/')

        if to_compile:
            cmd.byte_compile(to_compile)
        if self.distribution.data_files:
            self.do_install_data()

        # Make the EGG-INFO directory
        archive_root = self.bdist_dir
        egg_info = os.path.join(archive_root, 'EGG-INFO')
        self.mkpath(egg_info)
        if self.distribution.scripts:
            script_dir = os.path.join(egg_info, 'scripts')
            log.info("installing scripts to %s", script_dir)
            self.call_command('install_scripts', install_dir=script_dir, no_ep=True)

        self.copy_metadata_to(egg_info)
        native_libs = os.path.join(egg_info, "native_libs.txt")
        if all_outputs:
            log.info("writing %s", native_libs)
            ensure_directory(native_libs)
            with open(native_libs, 'wt', encoding="utf-8") as libs_file:
                libs_file.write('\n'.join(all_outputs))
                libs_file.write('\n')
        elif os.path.isfile(native_libs):
            log.info("removing %s", native_libs)
            os.unlink(native_libs)

        write_safety_flag(os.path.join(archive_root, 'EGG-INFO'), self.zip_safe())

        if os.path.exists(os.path.join(self.egg_info, 'depends.txt')):
            log.warn(
                "WARNING: 'depends.txt' will not be used by setuptools 0.6!\n"
                "Use the install_requires/extras_require setup() args instead."
            )

        if self.exclude_source_files:
            self.zap_pyfiles()

        # Make the archive
        make_zipfile(
            self.egg_output,
            archive_root,
            verbose=self.verbose,
            mode=self.gen_header(),
        )
        if not self.keep_temp:
            remove_tree(self.bdist_dir)

        # Add to 'Distribution.dist_files' so that the "upload" command works
        getattr(self.distribution, 'dist_files', []).append((
            'bdist_egg',
            get_python_version(),
            self.egg_output,
        ))

    def zap_pyfiles(self) -> None:
        log.info("Removing .py files from temporary directory")
        for base, dirs, files in walk_egg(self.bdist_dir):
            for name in files:
                path = os.path.join(base, name)

                if name.endswith('.py'):
                    log.debug("Deleting %s", path)
                    os.unlink(path)

                if base.endswith('__pycache__'):
                    path_old = path

                    pattern = r'(?P<name>.+)\.(?P<magic>[^.]+)\.pyc'
                    m = re.match(pattern, name)
                    # We shouldn't find any non-pyc files in __pycache__
                    assert m is not None
                    path_new = os.path.join(base, os.pardir, m.group('name') + '.pyc')
                    log.info(f"Renaming file from [{path_old}] to [{path_new}]")
                    try:
                        os.remove(path_new)
                    except OSError:
                        pass
                    os.rename(path_old, path_new)

    def zip_safe(self):
        safe = getattr(self.distribution, 'zip_safe', None)
        if safe is not None:
            return safe
        log.warn("zip_safe flag not set; analyzing archive contents...")
        return analyze_egg(self.bdist_dir, self.stubs)

    def gen_header(self) -> Literal["w"]:
        return 'w'

    def copy_metadata_to(self, target_dir) -> None:
        "Copy metadata (egg info) to the target_dir"
        # normalize the path (so that a forward-slash in egg_info will
        # match using startswith below)
        norm_egg_info = os.path.normpath(self.egg_info)
        prefix = os.path.join(norm_egg_info, '')
        for path in self.ei_cmd.filelist.files:
            if path.startswith(prefix):
                target = os.path.join(target_dir, path[len(prefix) :])
                ensure_directory(target)
                self.copy_file(path, target)

    def get_ext_outputs(self):
        """Get a list of relative paths to C extensions in the output distro"""

        all_outputs = []
        ext_outputs = []

        paths = {self.bdist_dir: ''}
        for base, dirs, files in sorted_walk(self.bdist_dir):
            all_outputs.extend(
                paths[base] + filename
                for filename in files
                if os.path.splitext(filename)[1].lower() in NATIVE_EXTENSIONS
            )
            for filename in dirs:
                paths[os.path.join(base, filename)] = paths[base] + filename + '/'

        if self.distribution.has_ext_modules():
            build_cmd = self.get_finalized_command('build_ext')
            for ext in build_cmd.extensions:
                if isinstance(ext, Library):
                    continue
                fullname = build_cmd.get_ext_fullname(ext.name)
                filename = build_cmd.get_ext_filename(fullname)
                if not os.path.basename(filename).startswith('dl-') and os.path.exists(
                    os.path.join(self.bdist_dir, filename)
                ):
                    ext_outputs.append(filename)

        return all_outputs, ext_outputs


NATIVE_EXTENSIONS: dict[str, None] = dict.fromkeys(['.dll', '.so', '.dylib', '.pyd'])


def walk_egg(egg_dir: StrPath) -> Iterator[tuple[str, list[str], list[str]]]:
    """Walk an unpacked egg's contents, skipping the metadata directory"""
    walker = sorted_walk(egg_dir)
    base, dirs, files = next(walker)
    if 'EGG-INFO' in dirs:
        dirs.remove('EGG-INFO')
    yield base, dirs, files
    yield from walker


def analyze_egg(egg_dir, stubs):
    # check for existing flag in EGG-INFO
    for flag, fn in safety_flags.items():
        if os.path.exists(os.path.join(egg_dir, 'EGG-INFO', fn)):
            return flag
    if not can_scan():
        return False
    safe = True
    for base, dirs, files in walk_egg(egg_dir):
        for name in files:
            if name.endswith(('.py', '.pyw')):
                continue
            elif name.endswith(('.pyc', '.pyo')):
                # always scan, even if we already know we're not safe
                safe = scan_module(egg_dir, base, name, stubs) and safe
    return safe


def write_safety_flag(egg_dir, safe) -> None:
    # Write or remove zip safety flag file(s)
    for flag, fn in safety_flags.items():
        fn = os.path.join(egg_dir, fn)
        if os.path.exists(fn):
            if safe is None or bool(safe) != flag:
                os.unlink(fn)
        elif safe is not None and bool(safe) == flag:
            with open(fn, 'wt', encoding="utf-8") as f:
                f.write('\n')


safety_flags = {
    True: 'zip-safe',
    False: 'not-zip-safe',
}


def scan_module(egg_dir, base, name, stubs):
    """Check whether module possibly uses unsafe-for-zipfile stuff"""

    filename = os.path.join(base, name)
    if filename[:-1] in stubs:
        return True  # Extension module
    pkg = base[len(egg_dir) + 1 :].replace(os.sep, '.')
    module = pkg + (pkg and '.' or '') + os.path.splitext(name)[0]
    skip = 16  # skip magic & reserved? & date & file size
    f = open(filename, 'rb')  # noqa: SIM115 # handle managed explicitly
    f.read(skip)
    code = marshal.load(f)
    f.close()
    safe = True
    symbols = dict.fromkeys(iter_symbols(code))
    for bad in ['__file__', '__path__']:
        if bad in symbols:
            log.warn("%s: module references %s", module, bad)
            safe = False
    if 'inspect' in symbols:
        for bad in [
            'getsource',
            'getabsfile',
            'getfile',
            'getsourcefile',
            'getsourcelines',
            'findsource',
            'getcomments',
            'getframeinfo',
            'getinnerframes',
            'getouterframes',
            'stack',
            'trace',
        ]:
            if bad in symbols:
                log.warn("%s: module MAY be using inspect.%s", module, bad)
                safe = False
    return safe


def iter_symbols(code: CodeType) -> Iterator[str]:
    """Yield names and strings used by `code` and its nested code objects"""
    yield from code.co_names
    for const in code.co_consts:
        if isinstance(const, str):
            yield const
        elif isinstance(const, CodeType):
            yield from iter_symbols(const)


def can_scan() -> bool:
    if not sys.platform.startswith('java') and sys.platform != 'cli':
        # CPython, PyPy, etc.
        return True
    log.warn("Unable to analyze compiled code on this platform.")
    log.warn(
        "Please ask the author to include a 'zip_safe'"
        " setting (either True or False) in the package's setup.py"
    )
    return False


# Attribute names of options for commands that might need to be convinced to
# install to the egg build directory

INSTALL_DIRECTORY_ATTRS = ['install_lib', 'install_dir', 'install_data', 'install_base']


def make_zipfile(
    zip_filename: StrPathT,
    base_dir,
    verbose: bool = False,
    compress=True,
    mode: _ZipFileMode = 'w',
) -> StrPathT:
    """Create a zip file from all the files under 'base_dir'.  The output
    zip file will be named 'base_dir' + ".zip".  Uses either the "zipfile"
    Python module (if available) or the InfoZIP "zip" utility (if installed
    and found on the default search path).  If neither tool is available,
    raises DistutilsExecError.  Returns the name of the output zip file.
    """
    import zipfile

    mkpath(os.path.dirname(zip_filename))  # type: ignore[arg-type] # python/mypy#18075
    log.info("creating '%s' and adding '%s' to it", zip_filename, base_dir)

    def visit(z, dirname, names):
        for name in names:
            path = os.path.normpath(os.path.join(dirname, name))
            if os.path.isfile(path):
                p = path[len(base_dir) + 1 :]
                z.write(path, p)
                log.debug("adding '%s'", p)

    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    z = zipfile.ZipFile(zip_filename, mode, compression=compression)
    for dirname, dirs, files in sorted_walk(base_dir):
        visit(z, dirname, files)
    z.close()
    return zip_filename
<|endoftext|>from __future__ import annotations

import os
from typing import ClassVar

from .. import Command, _shutil

from distutils import log
from distutils.errors import DistutilsOptionError
from distutils.util import convert_path


class rotate(Command):
    """Delete older distributions"""

    description = "delete older distributions, keeping N newest files"
    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('match=', 'm', "patterns to match (required)"),
        ('dist-dir=', 'd', "directory where the distributions are"),
        ('keep=', 'k', "number of matching distributions to keep"),
    ]

    boolean_options: ClassVar[list[str]] = []

    def initialize_options(self):
        self.match = None
        self.dist_dir = None
        self.keep = None

    def finalize_options(self) -> None:
        if self.match is None:
            raise DistutilsOptionError(
                "Must specify one or more (comma-separated) match patterns "
                "(e.g. '.zip' or '.egg')"
            )
        if self.keep is None:
            raise DistutilsOptionError("Must specify number of files to keep")
        try:
            self.keep = int(self.keep)
        except ValueError as e:
            raise DistutilsOptionError("--keep must be an integer") from e
        if isinstance(self.match, str):
            self.match = [convert_path(p.strip()) for p in self.match.split(',')]
        self.set_undefined_options('bdist', ('dist_dir', 'dist_dir'))

    def run(self) -> None:
        self.run_command("egg_info")
        from glob import glob

        for pattern in self.match:
            pattern = self.distribution.get_name() + '*' + pattern
            files = glob(os.path.join(self.dist_dir, pattern))
            files = [(os.path.getmtime(f), f) for f in files]
            files.sort()
            files.reverse()

            log.info("%d file(s) matching %s", len(files), pattern)
            files = files[self.keep :]
            for t, f in files:
                log.info("Deleting %s", f)
                if os.path.isdir(f):
                    _shutil.rmtree(f)
                else:
                    os.unlink(f)
<|endoftext|>"""
Create a wheel that, when installed, will make the source package 'editable'
(add it to the interpreter's path, including metadata) per PEP 660. Replaces
'setup.py develop'.

.. note::
   One of the mechanisms briefly mentioned in PEP 660 to implement editable installs is
   to create a separated directory inside ``build`` and use a .pth file to point to that
   directory. In the context of this file such directory is referred as
   *auxiliary build directory* or ``auxiliary_dir``.
"""

from __future__ import annotations

import io
import logging
import operator
import os
import shutil
import traceback
from collections.abc import Iterable, Iterator, Mapping
from contextlib import suppress
from enum import Enum
from inspect import cleandoc
from itertools import chain, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar, cast

from .. import Command, _normalization, _path, _shutil, errors, namespaces
from .._path import StrPath
from ..compat import py310, py312
from ..discovery import find_package_path
from ..dist import Distribution
from ..warnings import InformationOnly, SetuptoolsDeprecationWarning
from .build import build as build_cls
from .build_py import build_py as build_py_cls
from .dist_info import dist_info as dist_info_cls
from .egg_info import egg_info as egg_info_cls
from .install import install as install_cls
from .install_scripts import install_scripts as install_scripts_cls

if TYPE_CHECKING:
    from typing_extensions import Self

    from .._vendor.wheel.wheelfile import WheelFile

_P = TypeVar("_P", bound=StrPath)
_logger = logging.getLogger(__name__)


class _EditableMode(Enum):
    """
    Possible editable installation modes:
    `lenient` (new files automatically added to the package - DEFAULT);
    `strict` (requires a new installation when files are added/removed); or
    `compat` (attempts to emulate `python setup.py develop` - DEPRECATED).
    """

    STRICT = "strict"
    LENIENT = "lenient"
    COMPAT = "compat"  # TODO: Remove `compat` after Dec/2022.

    @classmethod
    def convert(cls, mode: str | None) -> _EditableMode:
        if not mode:
            return _EditableMode.LENIENT  # default

        _mode = mode.upper()
        if _mode not in _EditableMode.__members__:
            raise errors.OptionError(f"Invalid editable mode: {mode!r}. Try: 'strict'.")

        if _mode == "COMPAT":
            SetuptoolsDeprecationWarning.emit(
                "Compat editable installs",
                """
                The 'compat' editable mode is transitional and will be removed
                in future versions of `setuptools`.
                Please adapt your code accordingly to use either the 'strict' or the
                'lenient' modes.
                """,
                see_docs="userguide/development_mode.html",
                # TODO: define due_date
                # There is a series of shortcomings with the available editable install
                # methods, and they are very controversial. This is something that still
                # needs work.
                # Moreover, `pip` is still hiding this warning, so users are not aware.
            )

        return _EditableMode[_mode]


_STRICT_WARNING = """
New or renamed files may not be automatically picked up without a new installation.
"""

_LENIENT_WARNING = """
Options like `package-data`, `include/exclude-package-data` or
`packages.find.exclude/include` may have no effect.
"""


class editable_wheel(Command):
    """Build 'editable' wheel for development.
    This command is private and reserved for internal use of setuptools,
    users should rely on ``setuptools.build_meta`` APIs.
    """

    description = "DO NOT CALL DIRECTLY, INTERNAL ONLY: create PEP 660 editable wheel"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ("dist-dir=", "d", "directory to put final built distributions in"),
        ("dist-info-dir=", "I", "path to a pre-build .dist-info directory"),
        ("mode=", None, cleandoc(_EditableMode.__doc__ or "")),
    ]

    def initialize_options(self):
        self.dist_dir = None
        self.dist_info_dir = None
        self.project_dir = None
        self.mode = None

    def finalize_options(self) -> None:
        dist = self.distribution
        self.project_dir = dist.src_root or os.curdir
        self.package_dir = dist.package_dir or {}
        self.dist_dir = Path(self.dist_dir or os.path.join(self.project_dir, "dist"))

    def run(self) -> None:
        try:
            self.dist_dir.mkdir(exist_ok=True)
            self._ensure_dist_info()

            # Add missing dist_info files
            self.reinitialize_command("bdist_wheel")
            bdist_wheel = self.get_finalized_command("bdist_wheel")
            bdist_wheel.write_wheelfile(self.dist_info_dir)

            self._create_wheel_file(bdist_wheel)
        except Exception as ex:
            project = self.distribution.name or self.distribution.get_name()
            py310.add_note(
                ex,
                f"An error occurred when building editable wheel for {project}.\n"
                "See debugging tips in: "
                "https://setuptools.pypa.io/en/latest/userguide/development_mode.html#debugging-tips",
            )
            raise

    def _ensure_dist_info(self):
        if self.dist_info_dir is None:
            dist_info = cast(dist_info_cls, self.reinitialize_command("dist_info"))
            dist_info.output_dir = self.dist_dir
            dist_info.ensure_finalized()
            dist_info.run()
            self.dist_info_dir = dist_info.dist_info_dir
        else:
            assert str(self.dist_info_dir).endswith(".dist-info")
            assert Path(self.dist_info_dir, "METADATA").exists()

    def _install_namespaces(self, installation_dir, pth_prefix):
        # XXX: Only required to support the deprecated namespace practice
        dist = self.distribution
        if not dist.namespace_packages:
            return

        src_root = Path(self.project_dir, self.package_dir.get("", ".")).resolve()
        installer = _NamespaceInstaller(dist, installation_dir, pth_prefix, src_root)
        installer.install_namespaces()

    def _find_egg_info_dir(self) -> str | None:
        parent_dir = Path(self.dist_info_dir).parent if self.dist_info_dir else Path()
        candidates = map(str, parent_dir.glob("*.egg-info"))
        return next(candidates, None)

    def _configure_build(
        self, name: str, unpacked_wheel: StrPath, build_lib: StrPath, tmp_dir: StrPath
    ):
        """Configure commands to behave in the following ways:

        - Build commands can write to ``build_lib`` if they really want to...
          (but this folder is expected to be ignored and modules are expected to live
          in the project directory...)
        - Binary extensions should be built in-place (editable_mode = True)
        - Data/header/script files are not part of the "editable" specification
          so they are written directly to the unpacked_wheel directory.
        """
        # Non-editable files (data, headers, scripts) are written directly to the
        # unpacked_wheel

        dist = self.distribution
        wheel = str(unpacked_wheel)
        build_lib = str(build_lib)
        data = str(Path(unpacked_wheel, f"{name}.data", "data"))
        headers = str(Path(unpacked_wheel, f"{name}.data", "headers"))
        scripts = str(Path(unpacked_wheel, f"{name}.data", "scripts"))

        # egg-info may be generated again to create a manifest (used for package data)
        egg_info = cast(
            egg_info_cls, dist.reinitialize_command("egg_info", reinit_subcommands=True)
        )
        egg_info.egg_base = str(tmp_dir)
        egg_info.ignore_egg_info_in_manifest = True

        build = cast(
            build_cls, dist.reinitialize_command("build", reinit_subcommands=True)
        )
        install = cast(
            install_cls, dist.reinitialize_command("install", reinit_subcommands=True)
        )

        build.build_platlib = build.build_purelib = build.build_lib = build_lib
        install.install_purelib = install.install_platlib = install.install_lib = wheel
        install.install_scripts = build.build_scripts = scripts
        install.install_headers = headers
        install.install_data = data

        # For portability, ensure scripts are built with #!python shebang
        # pypa/setuptools#4863
        build_scripts = dist.get_command_obj("build_scripts")
        build_scripts.executable = 'python'

        install_scripts = cast(
            install_scripts_cls, dist.get_command_obj("install_scripts")
        )
        install_scripts.no_ep = True

        build.build_temp = str(tmp_dir)

        build_py = cast(build_py_cls, dist.get_command_obj("build_py"))
        build_py.compile = False
        build_py.existing_egg_info_dir = self._find_egg_info_dir()

        self._set_editable_mode()

        build.ensure_finalized()
        install.ensure_finalized()

    def _set_editable_mode(self):
        """Set the ``editable_mode`` flag in the build sub-commands"""
        dist = self.distribution
        build = dist.get_command_obj("build")
        for cmd_name in build.get_sub_commands():
            cmd = dist.get_command_obj(cmd_name)
            if hasattr(cmd, "editable_mode"):
                cmd.editable_mode = True
            elif hasattr(cmd, "inplace"):
                cmd.inplace = True  # backward compatibility with distutils

    def _collect_build_outputs(self) -> tuple[list[str], dict[str, str]]:
        files: list[str] = []
        mapping: dict[str, str] = {}
        build = self.get_finalized_command("build")

        for cmd_name in build.get_sub_commands():
            cmd = self.get_finalized_command(cmd_name)
            if hasattr(cmd, "get_outputs"):
                files.extend(cmd.get_outputs() or [])
            if hasattr(cmd, "get_output_mapping"):
                mapping.update(cmd.get_output_mapping() or {})

        return files, mapping

    def _run_build_commands(
        self,
        dist_name: str,
        unpacked_wheel: StrPath,
        build_lib: StrPath,
        tmp_dir: StrPath,
    ) -> tuple[list[str], dict[str, str]]:
        self._configure_build(dist_name, unpacked_wheel, build_lib, tmp_dir)
        self._run_build_subcommands()
        files, mapping = self._collect_build_outputs()
        self._run_install("headers")
        self._run_install("scripts")
        self._run_install("data")
        return files, mapping

    def _run_build_subcommands(self) -> None:
        """
        Issue #3501 indicates that some plugins/customizations might rely on:

        1. ``build_py`` not running
        2. ``build_py`` always copying files to ``build_lib``

        However both these assumptions may be false in editable_wheel.
        This method implements a temporary workaround to support the ecosystem
        while the implementations catch up.
        """
        # TODO: Once plugins/customizations had the chance to catch up, replace
        #       `self._run_build_subcommands()` with `self.run_command("build")`.
        #       Also remove _safely_run, TestCustomBuildPy. Suggested date: Aug/2023.
        build = self.get_finalized_command("build")
        for name in build.get_sub_commands():
            cmd = self.get_finalized_command(name)
            if name == "build_py" and type(cmd) is not build_py_cls:
                self._safely_run(name)
            else:
                self.run_command(name)

    def _safely_run(self, cmd_name: str):
        try:
            return self.run_command(cmd_name)
        except Exception:  # noqa: BLE001 # intentional broad fallback
            SetuptoolsDeprecationWarning.emit(
                "Customization incompatible with editable install",
                f"""
                {traceback.format_exc()}

                If you are seeing this warning it is very likely that a setuptools
                plugin or customization overrides the `{cmd_name}` command, without
                taking into consideration how editable installs run build steps
                starting from setuptools v64.0.0.

                Plugin authors and developers relying on custom build steps are
                encouraged to update their `{cmd_name}` implementation considering the
                information about editable installs in
                https://setuptools.pypa.io/en/latest/userguide/extension.html.

                For the time being `setuptools` will silence this error and ignore
                the faulty command, but this behavior will change in future versions.
                """,
                # TODO: define due_date
                # There is a series of shortcomings with the available editable install
                # methods, and they are very controversial. This is something that still
                # needs work.
            )

    def _create_wheel_file(self, bdist_wheel):
        from wheel.wheelfile import WheelFile

        dist_info = self.get_finalized_command("dist_info")
        dist_name = dist_info.name
        tag = "-".join(bdist_wheel.get_tag())
        build_tag = "0.editable"  # According to PEP 427 needs to start with digit
        archive_name = f"{dist_name}-{build_tag}-{tag}.whl"
        wheel_path = Path(self.dist_dir, archive_name)
        if wheel_path.exists():
            wheel_path.unlink()

        unpacked_wheel = TemporaryDirectory(suffix=archive_name)
        build_lib = TemporaryDirectory(suffix=".build-lib")
        build_tmp = TemporaryDirectory(suffix=".build-temp")

        with unpacked_wheel as unpacked, build_lib as lib, build_tmp as tmp:
            unpacked_dist_info = Path(unpacked, Path(self.dist_info_dir).name)
            shutil.copytree(self.dist_info_dir, unpacked_dist_info)
            self._install_namespaces(unpacked, dist_name)
            files, mapping = self._run_build_commands(dist_name, unpacked, lib, tmp)
            strategy = self._select_strategy(dist_name, tag, lib)
            with strategy, WheelFile(wheel_path, "w") as wheel_obj:
                strategy(wheel_obj, files, mapping)
                wheel_obj.write_files(unpacked)

        return wheel_path

    def _run_install(self, category: str):
        has_category = getattr(self.distribution, f"has_{category}", None)
        if has_category and has_category():
            _logger.info(f"Installing {category} as non editable")
            self.run_command(f"install_{category}")

    def _select_strategy(
        self,
        name: str,
        tag: str,
        build_lib: StrPath,
    ) -> EditableStrategy:
        """Decides which strategy to use to implement an editable installation."""
        build_name = f"__editable__.{name}-{tag}"
        project_dir = Path(self.project_dir)
        mode = _EditableMode.convert(self.mode)

        if mode is _EditableMode.STRICT:
            auxiliary_dir = _empty_dir(Path(self.project_dir, "build", build_name))
            return _LinkTree(self.distribution, name, auxiliary_dir, build_lib)

        packages = _find_packages(self.distribution)
        has_simple_layout = _simple_layout(packages, self.package_dir, project_dir)
        is_compat_mode = mode is _EditableMode.COMPAT
        if set(self.package_dir) == {""} and has_simple_layout or is_compat_mode:
            # src-layout(ish) is relatively safe for a simple pth file
            src_dir = self.package_dir.get("", ".")
            return _StaticPth(self.distribution, name, [Path(project_dir, src_dir)])

        # Use a MetaPathFinder to avoid adding accidental top-level packages/modules
        return _TopLevelFinder(self.distribution, name)


class EditableStrategy(Protocol):
    def __call__(
        self, wheel: WheelFile, files: list[str], mapping: Mapping[str, str]
    ) -> object: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> object: ...


class _StaticPth:
    def __init__(self, dist: Distribution, name: str, path_entries: list[Path]) -> None:
        self.dist = dist
        self.name = name
        self.path_entries = path_entries

    def __call__(
        self, wheel: WheelFile, files: list[str], mapping: Mapping[str, str]
    ) -> None:
        entries = "\n".join(str(p.resolve()) for p in self.path_entries)
        contents = _encode_pth(f"{entries}\n")
        wheel.writestr(f"__editable__.{self.name}.pth", contents)

    def __enter__(self) -> Self:
        msg = f"""
        Editable install will be performed using .pth file to extend `sys.path` with:
        {list(map(os.fspath, self.path_entries))!r}
        """
        _logger.warning(msg + _LENIENT_WARNING)
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        pass


class _LinkTree(_StaticPth):
    """
    Creates a ``.pth`` file that points to a link tree in the ``auxiliary_dir``.

    This strategy will only link files (not dirs), so it can be implemented in
    any OS, even if that means using hardlinks instead of symlinks.

    By collocating ``auxiliary_dir`` and the original source code, limitations
    with hardlinks should be avoided.
    """

    def __init__(
        self,
        dist: Distribution,
        name: str,
        auxiliary_dir: StrPath,
        build_lib: StrPath,
    ) -> None:
        self.auxiliary_dir = Path(auxiliary_dir)
        self.build_lib = Path(build_lib).resolve()
        self._file = dist.get_command_obj("build_py").copy_file
        super().__init__(dist, name, [self.auxiliary_dir])

    def __call__(
        self, wheel: WheelFile, files: list[str], mapping: Mapping[str, str]
    ) -> None:
        self._create_links(files, mapping)
        super().__call__(wheel, files, mapping)

    def _normalize_output(self, file: str) -> str | None:
        # Files relative to build_lib will be normalized to None
        with suppress(ValueError):
            path = Path(file).resolve().relative_to(self.build_lib)
            return str(path).replace(os.sep, '/')
        return None

    def _create_file(self, relative_output: str, src_file: str, link=None):
        dest = self.auxiliary_dir / relative_output
        if not dest.parent.is_dir():
            dest.parent.mkdir(parents=True)
        self._file(src_file, dest, link=link)

    def _create_links(self, outputs, output_mapping: Mapping[str, str]):
        self.auxiliary_dir.mkdir(parents=True, exist_ok=True)
        link_type = "sym" if _can_symlink_files(self.auxiliary_dir) else "hard"
        normalised = ((self._normalize_output(k), v) for k, v in output_mapping.items())
        # remove files that are not relative to build_lib
        mappings = {k: v for k, v in normalised if k is not None}

        for output in outputs:
            relative = self._normalize_output(output)
            if relative and relative not in mappings:
                self._create_file(relative, output)

        for relative, src in mappings.items():
            self._create_file(relative, src, link=link_type)

    def __enter__(self) -> Self:
        msg = "Strict editable install will be performed using a link tree.\n"
        _logger.warning(msg + _STRICT_WARNING)
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        msg = f"""\n
        Strict editable installation performed using the auxiliary directory:
            {self.auxiliary_dir}

        Please be careful to not remove this directory, otherwise you might not be able
        to import/use your package.
        """
        InformationOnly.emit("Editable installation.", msg)


class _TopLevelFinder:
    def __init__(self, dist: Distribution, name: str) -> None:
        self.dist = dist
        self.name = name

    def template_vars(self) -> tuple[str, str, dict[str, str], dict[str, list[str]]]:
        src_root = self.dist.src_root or os.curdir
        top_level = chain(_find_packages(self.dist), _find_top_level_modules(self.dist))
        package_dir = self.dist.package_dir or {}
        roots = _find_package_roots(top_level, package_dir, src_root)

        namespaces_ = dict(
            chain(
                _find_namespaces(self.dist.packages or [], roots),
                ((ns, []) for ns in _find_virtual_namespaces(roots)),
            )
        )

        legacy_namespaces = {
            pkg: find_package_path(pkg, roots, self.dist.src_root or "")
            for pkg in self.dist.namespace_packages or []
        }

        mapping = {**roots, **legacy_namespaces}
        # ^-- We need to explicitly add the legacy_namespaces to the mapping to be
        #     able to import their modules even if another package sharing the same
        #     namespace is installed in a conventional (non-editable) way.

        name = f"__editable__.{self.name}.finder"
        finder = _normalization.safe_identifier(name)
        return finder, name, mapping, namespaces_

    def get_implementation(self) -> Iterator[tuple[str, bytes]]:
        finder, name, mapping, namespaces_ = self.template_vars()

        content = bytes(_finder_template(name, mapping, namespaces_), "utf-8")
        yield (f"{finder}.py", content)

        content = _encode_pth(f"import {finder}; {finder}.install()")
        yield (f"__editable__.{self.name}.pth", content)

    def __call__(
        self, wheel: WheelFile, files: list[str], mapping: Mapping[str, str]
    ) -> None:
        for file, content in self.get_implementation():
            wheel.writestr(file, content)

    def __enter__(self) -> Self:
        msg = "Editable install will be performed using a meta path finder.\n"
        _logger.warning(msg + _LENIENT_WARNING)
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        msg = """\n
        Please be careful with folders in your working directory with the same
        name as your package as they may take precedence during imports.
        """
        InformationOnly.emit("Editable installation.", msg)


def _encode_pth(content: str) -> bytes:
    """
    Prior to Python 3.13 (see https://github.com/python/cpython/issues/77102),
    .pth files are always read with 'locale' encoding, the recommendation
    from the cpython core developers is to write them as ``open(path, "w")``
    and ignore warnings (see python/cpython#77102, pypa/setuptools#3937).
    This function tries to simulate this behavior without having to create an
    actual file, in a way that supports a range of active Python versions.
    (There seems to be some variety in the way different version of Python handle
    ``encoding=None``, not all of them use ``locale.getpreferredencoding(False)``
    or ``locale.getencoding()``).
    """
    with io.BytesIO() as buffer:
        wrapper = io.TextIOWrapper(buffer, encoding=py312.PTH_ENCODING)
        # TODO: Python 3.13 replace the whole function with `bytes(content, "utf-8")`
        wrapper.write(content)
        wrapper.flush()
        buffer.seek(0)
        return buffer.read()


def _can_symlink_files(base_dir: Path) -> bool:
    with TemporaryDirectory(dir=str(base_dir.resolve())) as tmp:
        path1, path2 = Path(tmp, "file1.txt"), Path(tmp, "file2.txt")
        path1.write_text("file1", encoding="utf-8")
        with suppress(AttributeError, NotImplementedError, OSError):
            os.symlink(path1, path2)
            if path2.is_symlink() and path2.read_text(encoding="utf-8") == "file1":
                return True

        try:
            os.link(path1, path2)  # Ensure hard links can be created
        except Exception as ex:
            msg = (
                "File system does not seem to support either symlinks or hard links. "
                "Strict editable installs require one of them to be supported."
            )
            raise LinksNotSupported(msg) from ex
        return False


def _simple_layout(
    packages: Iterable[str], package_dir: dict[str, str], project_dir: StrPath
) -> bool:
    """Return ``True`` if:
    - all packages are contained by the same parent directory, **and**
    - all packages become importable if the parent directory is added to ``sys.path``.

    >>> _simple_layout(['a'], {"": "src"}, "/tmp/myproj")
    True
    >>> _simple_layout(['a', 'a.b'], {"": "src"}, "/tmp/myproj")
    True
    >>> _simple_layout(['a', 'a.b'], {}, "/tmp/myproj")
    True
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"": "src"}, "/tmp/myproj")
    True
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"a": "a", "b": "b"}, ".")
    True
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"a": "_a", "b": "_b"}, ".")
    False
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"a": "_a"}, "/tmp/myproj")
    False
    >>> _simple_layout(['a', 'a.a1', 'a.a1.a2', 'b'], {"a.a1.a2": "_a2"}, ".")
    False
    >>> _simple_layout(['a', 'a.b'], {"": "src", "a.b": "_ab"}, "/tmp/myproj")
    False
    >>> # Special cases, no packages yet:
    >>> _simple_layout([], {"": "src"}, "/tmp/myproj")
    True
    >>> _simple_layout([], {"a": "_a", "": "src"}, "/tmp/myproj")
    False
    """
    layout = {pkg: find_package_path(pkg, package_dir, project_dir) for pkg in packages}
    if not layout:
        return set(package_dir) in ({}, {""})
    parent = os.path.commonpath(starmap(_parent_path, layout.items()))
    return all(
        _path.same_path(Path(parent, *key.split('.')), value)
        for key, value in layout.items()
    )


def _parent_path(pkg, pkg_path):
    """Infer the parent path containing a package, that if added to ``sys.path`` would
    allow importing that package.
    When ``pkg`` is directly mapped into a directory with a different name, return its
    own path.
    >>> _parent_path("a", "src/a")
    'src'
    >>> _parent_path("b", "src/c")
    'src/c'
    """
    parent = pkg_path.removesuffix(pkg)
    return parent.rstrip("/" + os.sep)


def _find_packages(dist: Distribution) -> Iterator[str]:
    yield from iter(dist.packages or [])

    py_modules = dist.py_modules or []
    nested_modules = [mod for mod in py_modules if "." in mod]
    if dist.ext_package:
        yield dist.ext_package
    else:
        ext_modules = dist.ext_modules or []
        nested_modules += [x.name for x in ext_modules if "." in x.name]

    for module in nested_modules:
        package, _, _ = module.rpartition(".")
        yield package


def _find_top_level_modules(dist: Distribution) -> Iterator[str]:
    py_modules = dist.py_modules or []
    yield from (mod for mod in py_modules if "." not in mod)

    if not dist.ext_package:
        ext_modules = dist.ext_modules or []
        yield from (x.name for x in ext_modules if "." not in x.name)


def _find_package_roots(
    packages: Iterable[str],
    package_dir: Mapping[str, str],
    src_root: StrPath,
) -> dict[str, str]:
    pkg_roots: dict[str, str] = {
        pkg: _absolute_root(find_package_path(pkg, package_dir, src_root))
        for pkg in sorted(packages)
    }

    return _remove_nested(pkg_roots)


def _absolute_root(path: StrPath) -> str:
    """Works for packages and top-level modules"""
    path_ = Path(path)
    parent = path_.parent

    if path_.exists():
        return str(path_.resolve())
    else:
        return str(parent.resolve() / path_.name)


def _find_virtual_namespaces(pkg_roots: dict[str, str]) -> Iterator[str]:
    """By carefully designing ``package_dir``, it is possible to implement the logical
    structure of PEP 420 in a package without the corresponding directories.

    Moreover a parent package can be purposefully/accidentally skipped in the discovery
    phase (e.g. ``find_packages(include=["mypkg.*"])``, when ``mypkg.foo`` is included
    by ``mypkg`` itself is not).
    We consider this case to also be a virtual namespace (ignoring the original
    directory) to emulate a non-editable installation.

    This function will try to find these kinds of namespaces.
    """
    for pkg in pkg_roots:
        if "." not in pkg:
            continue
        parts = pkg.split(".")
        for i in range(len(parts) - 1, 0, -1):
            partial_name = ".".join(parts[:i])
            path = Path(find_package_path(partial_name, pkg_roots, ""))
            if not path.exists() or partial_name not in pkg_roots:
                # partial_name not in pkg_roots ==> purposefully/accidentally skipped
                yield partial_name


def _find_namespaces(
    packages: list[str], pkg_roots: dict[str, str]
) -> Iterator[tuple[str, list[str]]]:
    for pkg in packages:
        path = find_package_path(pkg, pkg_roots, "")
        if Path(path).exists() and not Path(path, "__init__.py").exists():
            yield (pkg, [path])


def _remove_nested(pkg_roots: dict[str, str]) -> dict[str, str]:
    output = dict(pkg_roots.copy())

    for pkg, path in reversed(list(pkg_roots.items())):
        if any(
            pkg != other and _is_nested(pkg, path, other, other_path)
            for other, other_path in pkg_roots.items()
        ):
            output.pop(pkg)

    return output


def _is_nested(pkg: str, pkg_path: str, parent: str, parent_path: str) -> bool:
    """
    Return ``True`` if ``pkg`` is nested inside ``parent`` both logically and in the
    file system.
    >>> _is_nested("a.b", "path/a/b", "a", "path/a")
    True
    >>> _is_nested("a.b", "path/a/b", "a", "otherpath/a")
    False
    >>> _is_nested("a.b", "path/a/b", "c", "path/c")
    False
    >>> _is_nested("a.a", "path/a/a", "a", "path/a")
    True
    >>> _is_nested("b.a", "path/b/a", "a", "path/a")
    False
    """
    norm_pkg_path = _path.normpath(pkg_path)
    rest = pkg.replace(parent, "", 1).strip(".").split(".")
    return pkg.startswith(parent) and norm_pkg_path == _path.normpath(
        Path(parent_path, *rest)
    )


def _empty_dir(dir_: _P) -> _P:
    """Create a directory ensured to be empty. Existing files may be removed."""
    _shutil.rmtree(dir_, ignore_errors=True)
    os.makedirs(dir_)
    return dir_


class _NamespaceInstaller(namespaces.Installer):
    def __init__(self, distribution, installation_dir, editable_name, src_root) -> None:
        self.distribution = distribution
        self.src_root = src_root
        self.installation_dir = installation_dir
        self.editable_name = editable_name
        self.outputs: list[str] = []

    def _get_nspkg_file(self):
        """Installation target."""
        return os.path.join(self.installation_dir, self.editable_name + self.nspkg_ext)

    def _get_root(self):
        """Where the modules/packages should be loaded from."""
        return repr(str(self.src_root))


_FINDER_TEMPLATE = """\
from __future__ import annotations
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from itertools import chain
from pathlib import Path

MAPPING: dict[str, str] = {mapping!r}
NAMESPACES: dict[str, list[str]] = {namespaces!r}
PATH_PLACEHOLDER = {name!r} + ".__path_hook__"


class _EditableFinder:  # MetaPathFinder
    @classmethod
    def find_spec(cls, fullname: str, path=None, target=None) -> ModuleSpec | None:  # type: ignore
        # Top-level packages and modules (we know these exist in the FS)
        if fullname in MAPPING:
            pkg_path = MAPPING[fullname]
            return cls._find_spec(fullname, Path(pkg_path))

        # Handle immediate children modules (required for namespaces to work)
        # To avoid problems with case sensitivity in the file system we delegate
        # to the importlib.machinery implementation.
        parent, _, child = fullname.rpartition(".")
        if parent and parent in MAPPING:
            return PathFinder.find_spec(fullname, path=[MAPPING[parent]])

        # Other levels of nesting should be handled automatically by importlib
        # using the parent path.
        return None

    @classmethod
    def _find_spec(cls, fullname: str, candidate_path: Path) -> ModuleSpec | None:
        init = candidate_path / "__init__.py"
        candidates = (candidate_path.with_suffix(x) for x in module_suffixes())
        for candidate in chain([init], candidates):
            if candidate.exists():
                return spec_from_file_location(fullname, candidate)
        return None


class _EditableNamespaceFinder:  # PathEntryFinder
    @classmethod
    def _path_hook(cls, path) -> type[_EditableNamespaceFinder]:
        if path == PATH_PLACEHOLDER:
            return cls
        raise ImportError

    @classmethod
    def _paths(cls, fullname: str) -> list[str]:
        paths = NAMESPACES[fullname]
        if not paths and fullname in MAPPING:
            paths = [MAPPING[fullname]]
        # Always add placeholder, for 2 reasons:
        # 1. __path__ cannot be empty for the spec to be considered namespace.
        # 2. In the case of nested namespaces, we need to force
        #    import machinery to query _EditableNamespaceFinder again.
        return [*paths, PATH_PLACEHOLDER]

    @classmethod
    def find_spec(cls, fullname: str, target=None) -> ModuleSpec | None:  # type: ignore
        if fullname in NAMESPACES:
            spec = ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = cls._paths(fullname)
            return spec
        return None

    @classmethod
    def find_module(cls, _fullname) -> None:
        return None


def install():
    if not any(finder == _EditableFinder for finder in sys.meta_path):
        sys.meta_path.append(_EditableFinder)

    if not NAMESPACES:
        return

    if not any(hook == _EditableNamespaceFinder._path_hook for hook in sys.path_hooks):
        # PathEntryFinder is needed to create NamespaceSpec without private APIS
        sys.path_hooks.append(_EditableNamespaceFinder._path_hook)
    if PATH_PLACEHOLDER not in sys.path:
        sys.path.append(PATH_PLACEHOLDER)  # Used just to trigger the path hook
"""


def _finder_template(
    name: str, mapping: Mapping[str, str], namespaces: dict[str, list[str]]
) -> str:
    """Create a string containing the code for the``MetaPathFinder`` and
    ``PathEntryFinder``.
    """
    mapping = dict(sorted(mapping.items(), key=operator.itemgetter(0)))
    return _FINDER_TEMPLATE.format(name=name, mapping=mapping, namespaces=namespaces)


class LinksNotSupported(errors.FileError):
    """File system does not seem to support either symlinks or hard links."""
<|endoftext|>from __future__ import annotations

import inspect
import platform
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from ..dist import Distribution
from ..warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning

import distutils.command.install as orig
from distutils.errors import DistutilsArgError

if TYPE_CHECKING:
    # This is only used for a type-cast, don't import at runtime or it'll cause deprecation warnings
    from .easy_install import easy_install as easy_install_cls
else:
    easy_install_cls = None


def __getattr__(name: str):  # pragma: no cover
    if name == "_install":
        SetuptoolsDeprecationWarning.emit(
            "`setuptools.command._install` was an internal implementation detail "
            "that was left in for numpy<1.9 support.",
            due_date=(2025, 5, 2),  # Originally added on 2024-11-01
        )
        return orig.install
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class install(orig.install):
    """Use easy_install to install the package, w/dependencies"""

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution

    user_options = orig.install.user_options + [
        ('old-and-unmanageable', None, "Try not to use this!"),
        (
            'single-version-externally-managed',
            None,
            "used by system package builders to create 'flat' eggs",
        ),
    ]
    boolean_options = orig.install.boolean_options + [
        'old-and-unmanageable',
        'single-version-externally-managed',
    ]
    # Type the same as distutils.command.install.install.sub_commands
    # Must keep the second tuple item potentially None due to invariance
    new_commands: ClassVar[list[tuple[str, Callable[[Any], bool] | None]]] = [
        ('install_egg_info', lambda self: True),
        ('install_scripts', lambda self: True),
    ]
    _nc: ClassVar[dict] = dict(new_commands)

    def initialize_options(self):
        SetuptoolsDeprecationWarning.emit(
            "setup.py install is deprecated.",
            """
            Please avoid running ``setup.py`` directly.
            Instead, use pypa/build, pypa/installer or other
            standards-based tools.
            """,
            see_url="https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html",
            due_date=(2025, 10, 31),
        )

        super().initialize_options()
        self.old_and_unmanageable = None
        self.single_version_externally_managed = None

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.root:
            self.single_version_externally_managed = True
        elif (
            self.single_version_externally_managed and not self.root and not self.record
        ):
            raise DistutilsArgError(
                "You must specify --record or --root when building system packages"
            )

    def handle_extra_path(self):
        if self.root or self.single_version_externally_managed:
            # explicit backward-compatibility mode, allow extra_path to work
            return orig.install.handle_extra_path(self)

        # Ignore extra_path when installing an egg (or being run by another
        # command without --root or --single-version-externally-managed
        self.path_file = None
        self.extra_dirs = ''
        return None

    @staticmethod
    def _called_from_setup(run_frame):
        """
        Attempt to detect whether run() was called from setup() or by another
        command.  If called by setup(), the parent caller will be the
        'run_command' method in 'distutils.dist', and *its* caller will be
        the 'run_commands' method.  If called any other way, the
        immediate caller *might* be 'run_command', but it won't have been
        called by 'run_commands'. Return True in that case or if a call stack
        is unavailable. Return False otherwise.
        """
        if run_frame is None:
            msg = "Call stack not available. bdist_* commands may fail."
            SetuptoolsWarning.emit(msg)
            if platform.python_implementation() == 'IronPython':
                msg = "For best results, pass -X:Frames to enable call stack."
                SetuptoolsWarning.emit(msg)
            return True

        frames = inspect.getouterframes(run_frame)
        for frame in frames[2:4]:
            (caller,) = frame[:1]
            info = inspect.getframeinfo(caller)
            caller_module = caller.f_globals.get('__name__', '')

            if caller_module == "setuptools.dist" and info.function == "run_command":
                # Starting from v61.0.0 setuptools overwrites dist.run_command
                continue

            return caller_module == 'distutils.dist' and info.function == 'run_commands'

        return False


# XXX Python 3.1 doesn't see _nc if this is inside the class
install.sub_commands = [
    cmd for cmd in orig.install.sub_commands if cmd[0] not in install._nc
] + install.new_commands
<|endoftext|>from ..dist import Distribution
from ..warnings import SetuptoolsDeprecationWarning

import distutils.command.bdist_rpm as orig


class bdist_rpm(orig.bdist_rpm):
    """
    Override the default bdist_rpm behavior to do the following:

    1. Run egg_info to ensure the name and version are properly calculated.
    2. Always run 'install' using --single-version-externally-managed to
       disable eggs in RPM distributions.
    """

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution

    def run(self) -> None:
        SetuptoolsDeprecationWarning.emit(
            "Deprecated command",
            """
            bdist_rpm is deprecated and will be removed in a future version.
            Use bdist_wheel (wheel packages) instead.
            """,
            see_url="https://github.com/pypa/setuptools/issues/1988",
            due_date=(2023, 10, 30),  # Deprecation introduced in 22 Oct 2021.
        )

        # ensure distro name is up-to-date
        self.run_command('egg_info')

        orig.bdist_rpm.run(self)

    def _make_spec_file(self):
        spec = orig.bdist_rpm._make_spec_file(self)
        return [
            line.replace(
                "setup.py install ",
                "setup.py install --single-version-externally-managed ",
            ).replace("%setup", "%setup -n %{name}-%{unmangled_version}")
            for line in spec
        ]
<|endoftext|>import site
import subprocess
import sys
from typing import ClassVar, cast

from setuptools import Command
from setuptools.warnings import SetuptoolsDeprecationWarning


class develop(Command):
    """Set up package for development"""

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ("install-dir=", "d", "install package to DIR"),
        ('no-deps', 'N', "don't install dependencies"),
        ('user', None, f"install in user site-package '{site.USER_SITE}'"),
        ('prefix=', None, "installation prefix"),
        ("index-url=", "i", "base URL of Python Package Index"),
    ]
    boolean_options: ClassVar[list[str]] = [
        'no-deps',
        'user',
    ]

    install_dir = None
    no_deps = False
    user = False
    prefix = None
    index_url = None

    def run(self) -> None:
        # Casting because mypy doesn't understand bool mult conditionals
        cmd = cast(
            list[str],
            [sys.executable, '-m', 'pip', 'install', '-e', '.', '--use-pep517']
            + ['--target', self.install_dir] * bool(self.install_dir)
            + ['--no-deps'] * self.no_deps
            + ['--user'] * self.user
            + ['--prefix', self.prefix] * bool(self.prefix)
            + ['--index-url', self.index_url] * bool(self.index_url),
        )
        subprocess.check_call(cmd)

    def initialize_options(self) -> None:
        DevelopDeprecationWarning.emit()

    def finalize_options(self) -> None:
        pass


class DevelopDeprecationWarning(SetuptoolsDeprecationWarning):
    _SUMMARY = "develop command is deprecated."
    _DETAILS = """
    Please avoid running ``setup.py`` and ``develop``.
    Instead, use standards-based tools like pip or uv.
    """
    _SEE_URL = "https://github.com/pypa/setuptools/issues/917"
    _DUE_DATE = 2025, 10, 31
<|endoftext|>import configparser
import os
from typing import ClassVar

from .. import Command
from ..unicode_utils import _cfg_read_utf8_with_fallback

import distutils
from distutils import log
from distutils.errors import DistutilsOptionError
from distutils.util import convert_path

__all__ = ['config_file', 'edit_config', 'option_base', 'setopt']


def config_file(kind="local"):
    """Get the filename of the distutils, local, global, or per-user config

    `kind` must be one of "local", "global", or "user"
    """
    if kind == 'local':
        return 'setup.cfg'
    if kind == 'global':
        return os.path.join(os.path.dirname(distutils.__file__), 'distutils.cfg')
    if kind == 'user':
        dot = os.name == 'posix' and '.' or ''
        return os.path.expanduser(convert_path(f"~/{dot}pydistutils.cfg"))
    raise ValueError("config_file() type must be 'local', 'global', or 'user'", kind)


def edit_config(filename, settings) -> None:
    """Edit a configuration file to include `settings`

    `settings` is a dictionary of dictionaries or ``None`` values, keyed by
    command/section name.  A ``None`` value means to delete the entire section,
    while a dictionary lists settings to be changed or deleted in that section.
    A setting of ``None`` means to delete that setting.
    """
    log.debug("Reading configuration from %s", filename)
    opts = configparser.RawConfigParser()
    opts.optionxform = lambda optionstr: optionstr  # type: ignore[method-assign] # overriding method
    _cfg_read_utf8_with_fallback(opts, filename)

    for section, options in settings.items():
        if options is None:
            log.info("Deleting section [%s] from %s", section, filename)
            opts.remove_section(section)
        else:
            if not opts.has_section(section):
                log.debug("Adding new section [%s] to %s", section, filename)
                opts.add_section(section)
            for option, value in options.items():
                if value is None:
                    log.debug("Deleting %s.%s from %s", section, option, filename)
                    opts.remove_option(section, option)
                    if not opts.options(section):
                        log.info(
                            "Deleting empty [%s] section from %s", section, filename
                        )
                        opts.remove_section(section)
                else:
                    log.debug(
                        "Setting %s.%s to %r in %s", section, option, value, filename
                    )
                    opts.set(section, option, value)

    log.info("Writing %s", filename)
    with open(filename, 'w', encoding="utf-8") as f:
        opts.write(f)


class option_base(Command):
    """Abstract base class for commands that mess with config files"""

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('global-config', 'g', "save options to the site-wide distutils.cfg file"),
        ('user-config', 'u', "save options to the current user's pydistutils.cfg file"),
        ('filename=', 'f', "configuration file to use (default=setup.cfg)"),
    ]

    boolean_options: ClassVar[list[str]] = [
        'global-config',
        'user-config',
    ]

    def initialize_options(self):
        self.global_config = None
        self.user_config = None
        self.filename = None

    def finalize_options(self) -> None:
        filenames = []
        if self.global_config:
            filenames.append(config_file('global'))
        if self.user_config:
            filenames.append(config_file('user'))
        if self.filename is not None:
            filenames.append(self.filename)
        if not filenames:
            filenames.append(config_file('local'))
        if len(filenames) > 1:
            raise DistutilsOptionError(
                "Must specify only one configuration file option", filenames
            )
        (self.filename,) = filenames


class setopt(option_base):
    """Save command-line options to a file"""

    description = "set an option in setup.cfg or another config file"

    user_options = [
        ('command=', 'c', 'command to set an option for'),
        ('option=', 'o', 'option to set'),
        ('set-value=', 's', 'value of the option'),
        ('remove', 'r', 'remove (unset) the value'),
    ] + option_base.user_options

    boolean_options = option_base.boolean_options + ['remove']

    def initialize_options(self):
        option_base.initialize_options(self)
        self.command = None
        self.option = None
        self.set_value = None
        self.remove = None

    def finalize_options(self) -> None:
        option_base.finalize_options(self)
        if self.command is None or self.option is None:
            raise DistutilsOptionError("Must specify --command *and* --option")
        if self.set_value is None and not self.remove:
            raise DistutilsOptionError("Must specify --set-value or --remove")

    def run(self) -> None:
        edit_config(
            self.filename,
            {self.command: {self.option.replace('-', '_'): self.set_value}},
        )
<|endoftext|>import os
import sys
import types

from setuptools import Command

from .. import _scripts, warnings


class easy_install(Command):
    """Stubbed command for temporary pbr compatibility."""


def __getattr__(name):
    attr = getattr(
        types.SimpleNamespace(
            ScriptWriter=_scripts.ScriptWriter,
            sys_executable=os.environ.get(
                "__PYVENV_LAUNCHER__", os.path.normpath(sys.executable)
            ),
        ),
        name,
    )
    warnings.SetuptoolsDeprecationWarning.emit(
        summary="easy_install module is deprecated",
        details="Avoid accessing attributes of setuptools.command.easy_install.",
        due_date=(2025, 10, 31),
        see_url="https://github.com/pypa/setuptools/issues/4976",
    )
    return attr
<|endoftext|>"""Helper code used to generate ``requires.txt`` files in the egg-info directory.

The ``requires.txt`` file has an specific format:
    - Environment markers need to be part of the section headers and
      should not be part of the requirement spec itself.

See https://setuptools.pypa.io/en/latest/deprecated/python_eggs.html#requires-txt
"""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import Mapping
from itertools import filterfalse
from typing import TypeVar

from jaraco.text import yield_lines
from packaging.requirements import Requirement

from .. import _reqs
from .._reqs import _StrOrIter

# dict can work as an ordered set
_T = TypeVar("_T")
_Ordered = dict[_T, None]


def _prepare(
    install_requires: _StrOrIter, extras_require: Mapping[str, _StrOrIter]
) -> tuple[list[str], dict[str, list[str]]]:
    """Given values for ``install_requires`` and ``extras_require``
    create modified versions in a way that can be written in ``requires.txt``
    """
    extras = _convert_extras_requirements(extras_require)
    return _move_install_requirements_markers(install_requires, extras)


def _convert_extras_requirements(
    extras_require: Mapping[str, _StrOrIter],
) -> defaultdict[str, _Ordered[Requirement]]:
    """
    Convert requirements in `extras_require` of the form
    `"extra": ["barbazquux; {marker}"]` to
    `"extra:{marker}": ["barbazquux"]`.
    """
    output = defaultdict[str, _Ordered[Requirement]](dict)
    for section, v in extras_require.items():
        # Do not strip empty sections.
        output[section]
        for r in _reqs.parse(v):
            output[section + _suffix_for(r)].setdefault(r)

    return output


def _move_install_requirements_markers(
    install_requires: _StrOrIter, extras_require: Mapping[str, _Ordered[Requirement]]
) -> tuple[list[str], dict[str, list[str]]]:
    """
    The ``requires.txt`` file has an specific format:
        - Environment markers need to be part of the section headers and
          should not be part of the requirement spec itself.

    Move requirements in ``install_requires`` that are using environment
    markers ``extras_require``.
    """

    # divide the install_requires into two sets, simple ones still
    # handled by install_requires and more complex ones handled by extras_require.

    inst_reqs = list(_reqs.parse(install_requires))
    simple_reqs = filter(_no_marker, inst_reqs)
    complex_reqs = filterfalse(_no_marker, inst_reqs)
    simple_install_requires = list(map(str, simple_reqs))

    for r in complex_reqs:
        extras_require[':' + str(r.marker)].setdefault(r)

    expanded_extras = {
        # list(dict.fromkeys(...))  ensures a list of unique strings
        k: list(dict.fromkeys(str(r) for r in map(_clean_req, v)))
        for k, v in extras_require.items()
    }

    return simple_install_requires, expanded_extras


def _suffix_for(req):
    """Return the 'extras_require' suffix for a given requirement."""
    return ':' + str(req.marker) if req.marker else ''


def _clean_req(req):
    """Given a Requirement, remove environment markers and return it"""
    r = Requirement(str(req))  # create a copy before modifying
    r.marker = None
    return r


def _no_marker(req):
    return not req.marker


def _write_requirements(stream, reqs):
    lines = yield_lines(reqs or ())

    def append_cr(line):
        return line + '\n'

    lines = map(append_cr, lines)
    stream.writelines(lines)


def write_requirements(cmd, basename, filename):
    dist = cmd.distribution
    data = io.StringIO()
    install_requires, extras_require = _prepare(
        dist.install_requires or (), dist.extras_require or {}
    )
    _write_requirements(data, install_requires)
    for extra in sorted(extras_require):
        data.write('\n[{extra}]\n'.format(**vars()))
        _write_requirements(data, extras_require[extra])
    cmd.write_or_delete_file("requirements", filename, data.getvalue())


def write_setup_requirements(cmd, basename, filename):
    data = io.StringIO()
    _write_requirements(data, cmd.distribution.setup_requires)
    cmd.write_or_delete_file("setup-requirements", filename, data.getvalue())
<|endoftext|># mypy: disable_error_code=call-overload
# pyright: reportCallIssue=false, reportArgumentType=false
# Can't disable on the exact line because distutils doesn't exists on Python 3.12
# and type-checkers aren't aware of distutils_hack,
# causing distutils.command.bdist.bdist.format_commands to be Any.

import sys

from distutils.command.bdist import bdist

if 'egg' not in bdist.format_commands:
    try:
        # format_commands is a dict in vendored distutils
        # It used to be a list in older (stdlib) distutils
        # We support both for backwards compatibility
        bdist.format_commands['egg'] = ('bdist_egg', "Python .egg file")
    except TypeError:
        bdist.format_command['egg'] = ('bdist_egg', "Python .egg file")
        bdist.format_commands.append('egg')

del bdist, sys
<|endoftext|>from setuptools.command.setopt import config_file, edit_config, option_base

from distutils.errors import DistutilsOptionError


def shquote(arg):
    """Quote an argument for later parsing by shlex.split()"""
    for c in '"', "'", "\\", "#":
        if c in arg:
            return repr(arg)
    if arg.split() != [arg]:
        return repr(arg)
    return arg


class alias(option_base):
    """Define a shortcut that invokes one or more commands"""

    description = "define a shortcut to invoke one or more commands"
    command_consumes_arguments = True

    user_options = [
        ('remove', 'r', 'remove (unset) the alias'),
    ] + option_base.user_options

    boolean_options = option_base.boolean_options + ['remove']

    def initialize_options(self):
        option_base.initialize_options(self)
        self.args = None
        self.remove = None

    def finalize_options(self) -> None:
        option_base.finalize_options(self)
        if self.remove and len(self.args) != 1:
            raise DistutilsOptionError(
                "Must specify exactly one argument (the alias name) when using --remove"
            )

    def run(self) -> None:
        aliases = self.distribution.get_option_dict('aliases')

        if not self.args:
            print("Command Aliases")
            print("---------------")
            for alias in aliases:
                print("setup.py alias", format_alias(alias, aliases))
            return

        elif len(self.args) == 1:
            (alias,) = self.args
            if self.remove:
                command = None
            elif alias in aliases:
                print("setup.py alias", format_alias(alias, aliases))
                return
            else:
                print(f"No alias definition found for {alias!r}")
                return
        else:
            alias = self.args[0]
            command = ' '.join(map(shquote, self.args[1:]))

        edit_config(self.filename, {'aliases': {alias: command}})


def format_alias(name, aliases):
    source, command = aliases[name]
    if source == config_file('global'):
        source = '--global-config '
    elif source == config_file('user'):
        source = '--user-config '
    elif source == config_file('local'):
        source = ''
    else:
        source = f'--filename={source!r}'
    return source + name + ' ' + command
<|endoftext|>from __future__ import annotations

from typing import Protocol

from ..dist import Distribution

from distutils.command.build import build as _build

_ORIGINAL_SUBCOMMANDS = {"build_py", "build_clib", "build_ext", "build_scripts"}


class build(_build):
    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution

    # copy to avoid sharing the object with parent class
    sub_commands = _build.sub_commands[:]


class SubCommand(Protocol):
    """In order to support editable installations (see :pep:`660`) all
    build subcommands **SHOULD** implement this protocol. They also **MUST** inherit
    from ``setuptools.Command``.

    When creating an :pep:`editable wheel <660>`, ``setuptools`` will try to evaluate
    custom ``build`` subcommands using the following procedure:

    1. ``setuptools`` will set the ``editable_mode`` attribute to ``True``
    2. ``setuptools`` will execute the ``run()`` command.

       .. important::
          Subcommands **SHOULD** take advantage of ``editable_mode=True`` to adequate
          its behaviour or perform optimisations.

          For example, if a subcommand doesn't need to generate an extra file and
          all it does is to copy a source file into the build directory,
          ``run()`` **SHOULD** simply "early return".

          Similarly, if the subcommand creates files that would be placed alongside
          Python files in the final distribution, during an editable install
          the command **SHOULD** generate these files "in place" (i.e. write them to
          the original source directory, instead of using the build directory).
          Note that ``get_output_mapping()`` should reflect that and include mappings
          for "in place" builds accordingly.

    3. ``setuptools`` use any knowledge it can derive from the return values of
       ``get_outputs()`` and ``get_output_mapping()`` to create an editable wheel.
       When relevant ``setuptools`` **MAY** attempt to use file links based on the value
       of ``get_output_mapping()``. Alternatively, ``setuptools`` **MAY** attempt to use
       :doc:`import hooks <python:reference/import>` to redirect any attempt to import
       to the directory with the original source code and other files built in place.

    Please note that custom sub-commands **SHOULD NOT** rely on ``run()`` being
    executed (or not) to provide correct return values for ``get_outputs()``,
    ``get_output_mapping()`` or ``get_source_files()``. The ``get_*`` methods should
    work independently of ``run()``.
    """

    editable_mode: bool = False
    """Boolean flag that will be set to ``True`` when setuptools is used for an
    editable installation (see :pep:`660`).
    Implementations **SHOULD** explicitly set the default value of this attribute to
    ``False``.
    When subcommands run, they can use this flag to perform optimizations or change
    their behaviour accordingly.
    """

    build_lib: str
    """String representing the directory where the build artifacts should be stored,
    e.g. ``build/lib``.
    For example, if a distribution wants to provide a Python module named ``pkg.mod``,
    then a corresponding file should be written to ``{build_lib}/package/module.py``.
    A way of thinking about this is that the files saved under ``build_lib``
    would be eventually copied to one of the directories in :obj:`site.PREFIXES`
    upon installation.

    A command that produces platform-independent files (e.g. compiling text templates
    into Python functions), **CAN** initialize ``build_lib`` by copying its value from
    the ``build_py`` command. On the other hand, a command that produces
    platform-specific files **CAN** initialize ``build_lib`` by copying its value from
    the ``build_ext`` command. In general this is done inside the ``finalize_options``
    method with the help of the ``set_undefined_options`` command::

        def finalize_options(self):
            self.set_undefined_options("build_py", ("build_lib", "build_lib"))
            ...
    """

    def initialize_options(self) -> None:
        """(Required by the original :class:`setuptools.Command` interface)"""
        ...

    def finalize_options(self) -> None:
        """(Required by the original :class:`setuptools.Command` interface)"""
        ...

    def run(self) -> None:
        """(Required by the original :class:`setuptools.Command` interface)"""
        ...

    def get_source_files(self) -> list[str]:
        """
        Return a list of all files that are used by the command to create the expected
        outputs.
        For example, if your build command transpiles Java files into Python, you should
        list here all the Java files.
        The primary purpose of this function is to help populating the ``sdist``
        with all the files necessary to build the distribution.
        All files should be strings relative to the project root directory.
        """
        ...

    def get_outputs(self) -> list[str]:
        """
        Return a list of files intended for distribution as they would have been
        produced by the build.
        These files should be strings in the form of
        ``"{build_lib}/destination/file/path"``.

        .. note::
           The return value of ``get_output()`` should include all files used as keys
           in ``get_output_mapping()`` plus files that are generated during the build
           and don't correspond to any source file already present in the project.
        """
        ...

    def get_output_mapping(self) -> dict[str, str]:
        """
        Return a mapping between destination files as they would be produced by the
        build (dict keys) into the respective existing (source) files (dict values).
        Existing (source) files should be represented as strings relative to the project
        root directory.
        Destination files should be strings in the form of
        ``"{build_lib}/destination/file/path"``.
        """
        ...
<|endoftext|>from __future__ import annotations

import itertools
import operator
import os
import sys
import textwrap
from collections.abc import Iterator
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import cache_from_source as _compiled_file_name
from pathlib import Path
from typing import TYPE_CHECKING

from setuptools.dist import Distribution
from setuptools.errors import BaseError
from setuptools.extension import Extension, Library

from distutils import log
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler, get_config_var

if TYPE_CHECKING:
    # Cython not installed on CI tests, causing _build_ext to be `Any`
    from distutils.command.build_ext import build_ext as _build_ext
else:
    try:
        # Attempt to use Cython for building extensions, if available
        from Cython.Distutils.build_ext import build_ext as _build_ext

        # Additionally, assert that the compiler module will load
        # also. Ref #1229.
        __import__('Cython.Compiler.Main')
    except ImportError:
        from distutils.command.build_ext import build_ext as _build_ext

# make sure _config_vars is initialized
get_config_var("LDSHARED")
# Not publicly exposed in typeshed distutils stubs, but this is done on purpose
# See https://github.com/pypa/setuptools/pull/4228#issuecomment-1959856400
from distutils.sysconfig import _config_vars as _CONFIG_VARS


def _customize_compiler_for_shlib(compiler):
    if sys.platform == "darwin":
        # building .dylib requires additional compiler flags on OSX; here we
        # temporarily substitute the pyconfig.h variables so that distutils'
        # 'customize_compiler' uses them before we build the shared libraries.
        tmp = _CONFIG_VARS.copy()
        try:
            # XXX Help!  I don't have any idea whether these are right...
            _CONFIG_VARS['LDSHARED'] = (
                "gcc -Wl,-x -dynamiclib -undefined dynamic_lookup"
            )
            _CONFIG_VARS['CCSHARED'] = " -dynamiclib"
            _CONFIG_VARS['SO'] = ".dylib"
            customize_compiler(compiler)
        finally:
            _CONFIG_VARS.clear()
            _CONFIG_VARS.update(tmp)
    else:
        customize_compiler(compiler)


have_rtld = False
use_stubs = False
libtype = 'shared'

if sys.platform == "darwin":
    use_stubs = True
elif os.name != 'nt':
    try:
        import dl  # type: ignore[import-not-found] # https://github.com/python/mypy/issues/13002

        use_stubs = have_rtld = hasattr(dl, 'RTLD_NOW')
    except ImportError:
        pass


def get_abi3_suffix():
    """Return the file extension for an abi3-compliant Extension()"""
    for suffix in EXTENSION_SUFFIXES:
        if '.abi3' in suffix or suffix == '.pyd':  # Unix
            return suffix
    return None


class build_ext(_build_ext):
    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution
    editable_mode = False
    inplace = False

    def run(self) -> None:
        """Build extensions in build directory, then copy if --inplace"""
        old_inplace, self.inplace = self.inplace, False
        _build_ext.run(self)
        self.inplace = old_inplace
        if old_inplace:
            self.copy_extensions_to_source()

    def _get_inplace_equivalent(self, build_py, ext: Extension) -> tuple[str, str]:
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)
        modpath = fullname.split('.')
        package = '.'.join(modpath[:-1])
        package_dir = build_py.get_package_dir(package)
        inplace_file = os.path.join(package_dir, os.path.basename(filename))
        regular_file = os.path.join(self.build_lib, filename)
        return (inplace_file, regular_file)

    def copy_extensions_to_source(self) -> None:
        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)

            # Always copy, even if source is older than destination, to ensure
            # that the right extensions for the current Python/platform are
            # used.
            if os.path.exists(regular_file) or not ext.optional:
                self.copy_file(regular_file, inplace_file, level=self.verbose)

            if ext._needs_stub:
                inplace_stub = self._get_equivalent_stub(ext, inplace_file)
                self._write_stub_file(inplace_stub, ext, compile=True)
                # Always compile stub and remove the original (leave the cache behind)
                # (this behaviour was observed in previous iterations of the code)

    def _get_equivalent_stub(self, ext: Extension, output_file: str) -> str:
        dir_ = os.path.dirname(output_file)
        _, _, name = ext.name.rpartition(".")
        return f"{os.path.join(dir_, name)}.py"

    def _get_output_mapping(self) -> Iterator[tuple[str, str]]:
        if not self.inplace:
            return

        build_py = self.get_finalized_command('build_py')
        opt = self.get_finalized_command('install_lib').optimize or ""

        for ext in self.extensions:
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)
            yield (regular_file, inplace_file)

            if ext._needs_stub:
                # This version of `build_ext` always builds artifacts in another dir,
                # when "inplace=True" is given it just copies them back.
                # This is done in the `copy_extensions_to_source` function, which
                # always compile stub files via `_compile_and_remove_stub`.
                # At the end of the process, a `.pyc` stub file is created without the
                # corresponding `.py`.

                inplace_stub = self._get_equivalent_stub(ext, inplace_file)
                regular_stub = self._get_equivalent_stub(ext, regular_file)
                inplace_cache = _compiled_file_name(inplace_stub, optimization=opt)
                output_cache = _compiled_file_name(regular_stub, optimization=opt)
                yield (output_cache, inplace_cache)

    def get_ext_filename(self, fullname: str) -> str:
        so_ext = os.getenv('SETUPTOOLS_EXT_SUFFIX')
        if so_ext:
            filename = os.path.join(*fullname.split('.')) + so_ext
        else:
            filename = _build_ext.get_ext_filename(self, fullname)
            ext_suffix = get_config_var('EXT_SUFFIX')
            if not isinstance(ext_suffix, str):
                raise OSError(
                    "Configuration variable EXT_SUFFIX not found for this platform "
                    "and environment variable SETUPTOOLS_EXT_SUFFIX is missing"
                )
            so_ext = ext_suffix

        if fullname in self.ext_map:
            ext = self.ext_map[fullname]
            abi3_suffix = get_abi3_suffix()
            if ext.py_limited_api and abi3_suffix:  # Use abi3
                filename = filename[: -len(so_ext)] + abi3_suffix
            if isinstance(ext, Library):
                fn, ext = os.path.splitext(filename)
                return self.shlib_compiler.library_filename(fn, libtype)
            elif use_stubs and ext._links_to_dynamic:
                d, fn = os.path.split(filename)
                return os.path.join(d, 'dl-' + fn)
        return filename

    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.shlib_compiler = None
        self.shlibs = []
        self.ext_map = {}
        self.editable_mode = False

    def finalize_options(self) -> None:
        _build_ext.finalize_options(self)
        self.extensions = self.extensions or []
        self.check_extensions_list(self.extensions)
        self.shlibs = [ext for ext in self.extensions if isinstance(ext, Library)]
        if self.shlibs:
            self.setup_shlib_compiler()
        for ext in self.extensions:
            ext._full_name = self.get_ext_fullname(ext.name)
        for ext in self.extensions:
            fullname = ext._full_name
            self.ext_map[fullname] = ext

            # distutils 3.1 will also ask for module names
            # XXX what to do with conflicts?
            self.ext_map[fullname.split('.')[-1]] = ext

            ltd = self.shlibs and self.links_to_dynamic(ext) or False
            ns = ltd and use_stubs and not isinstance(ext, Library)
            ext._links_to_dynamic = ltd
            ext._needs_stub = ns
            filename = ext._file_name = self.get_ext_filename(fullname)
            libdir = os.path.dirname(os.path.join(self.build_lib, filename))
            if ltd and libdir not in ext.library_dirs:
                ext.library_dirs.append(libdir)
            if ltd and use_stubs and os.curdir not in ext.runtime_library_dirs:
                ext.runtime_library_dirs.append(os.curdir)

        if self.editable_mode:
            self.inplace = True

    def setup_shlib_compiler(self) -> None:
        compiler = self.shlib_compiler = new_compiler(
            compiler=self.compiler, force=self.force
        )
        _customize_compiler_for_shlib(compiler)

        if self.include_dirs is not None:
            compiler.set_include_dirs(self.include_dirs)
        if self.define is not None:
            # 'define' option is a list of (name,value) tuples
            for name, value in self.define:
                compiler.define_macro(name, value)
        if self.undef is not None:
            for macro in self.undef:
                compiler.undefine_macro(macro)
        if self.libraries is not None:
            compiler.set_libraries(self.libraries)
        if self.library_dirs is not None:
            compiler.set_library_dirs(self.library_dirs)
        if self.rpath is not None:
            compiler.set_runtime_library_dirs(self.rpath)
        if self.link_objects is not None:
            compiler.set_link_objects(self.link_objects)

        # hack so distutils' build_extension() builds a library instead
        compiler.link_shared_object = link_shared_object.__get__(compiler)  # type: ignore[method-assign]

    def get_export_symbols(self, ext):
        if isinstance(ext, Library):
            return ext.export_symbols
        return _build_ext.get_export_symbols(self, ext)

    def build_extension(self, ext) -> None:
        ext._convert_pyx_sources_to_lang()
        _compiler = self.compiler
        try:
            if isinstance(ext, Library):
                self.compiler = self.shlib_compiler
            _build_ext.build_extension(self, ext)
            if ext._needs_stub:
                build_lib = self.get_finalized_command('build_py').build_lib
                self.write_stub(build_lib, ext)
        finally:
            self.compiler = _compiler

    def links_to_dynamic(self, ext):
        """Return true if 'ext' links to a dynamic lib in the same package"""
        # XXX this should check to ensure the lib is actually being built
        # XXX as dynamic, and not just using a locally-found version or a
        # XXX static-compiled version
        libnames = dict.fromkeys([lib._full_name for lib in self.shlibs])
        pkg = '.'.join(ext._full_name.split('.')[:-1] + [''])
        return any(pkg + libname in libnames for libname in ext.libraries)

    def get_source_files(self) -> list[str]:
        return [*_build_ext.get_source_files(self), *self._get_internal_depends()]

    def _get_internal_depends(self) -> Iterator[str]:
        """Yield ``ext.depends`` that are contained by the project directory"""
        project_root = Path(self.distribution.src_root or os.curdir).resolve()
        depends = (dep for ext in self.extensions for dep in ext.depends)

        def skip(orig_path: str, reason: str) -> None:
            log.info(
                "dependency %s won't be automatically "
                "included in the manifest: the path %s",
                orig_path,
                reason,
            )

        for dep in depends:
            path = Path(dep)

            if path.is_absolute():
                skip(dep, "must be relative")
                continue

            if ".." in path.parts:
                skip(dep, "can't have `..` segments")
                continue

            try:
                resolved = (project_root / path).resolve(strict=True)
            except OSError:
                skip(dep, "doesn't exist")
                continue

            try:
                resolved.relative_to(project_root)
            except ValueError:
                skip(dep, "must be inside the project root")
                continue

            yield path.as_posix()

    def get_outputs(self) -> list[str]:
        if self.inplace:
            return list(self.get_output_mapping().keys())
        return sorted(_build_ext.get_outputs(self) + self.__get_stubs_outputs())

    def get_output_mapping(self) -> dict[str, str]:
        """See :class:`setuptools.commands.build.SubCommand`"""
        mapping = self._get_output_mapping()
        return dict(sorted(mapping, key=operator.itemgetter(0)))

    def __get_stubs_outputs(self):
        # assemble the base name for each extension that needs a stub
        ns_ext_bases = (
            os.path.join(self.build_lib, *ext._full_name.split('.'))
            for ext in self.extensions
            if ext._needs_stub
        )
        # pair each base with the extension
        pairs = itertools.product(ns_ext_bases, self.__get_output_extensions())
        return [base + fnext for base, fnext in pairs]

    def __get_output_extensions(self):
        yield '.py'
        yield '.pyc'
        if self.get_finalized_command('build_py').optimize:
            yield '.pyo'

    def write_stub(self, output_dir, ext, compile=False) -> None:
        stub_file = os.path.join(output_dir, *ext._full_name.split('.')) + '.py'
        self._write_stub_file(stub_file, ext, compile)

    def _write_stub_file(self, stub_file: str, ext: Extension, compile=False):
        log.info("writing stub loader for %s to %s", ext._full_name, stub_file)
        if compile and os.path.exists(stub_file):
            raise BaseError(stub_file + " already exists! Please delete.")
        with open(stub_file, 'w', encoding="utf-8") as f:
            content = (
                textwrap
                .dedent(f"""
                def __bootstrap__():
                   global __bootstrap__, __file__, __loader__
                   import sys, os, importlib.resources as irs, importlib.util
                #rtld   import dl
                   with irs.files(__name__).joinpath(
                     {os.path.basename(ext._file_name)!r}) as __file__:
                      del __bootstrap__
                      if '__loader__' in globals():
                          del __loader__
                #rtld      old_flags = sys.getdlopenflags()
                      old_dir = os.getcwd()
                      try:
                        os.chdir(os.path.dirname(__file__))
                #rtld        sys.setdlopenflags(dl.RTLD_NOW)
                        spec = importlib.util.spec_from_file_location(
                                   __name__, __file__)
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                      finally:
                #rtld        sys.setdlopenflags(old_flags)
                        os.chdir(old_dir)
                __bootstrap__()
                """)
                .lstrip()
                .replace('#rtld', '#rtld' * (not have_rtld))
            )
            f.write(content)
        if compile:
            self._compile_and_remove_stub(stub_file)

    def _compile_and_remove_stub(self, stub_file: str):
        from distutils.util import byte_compile

        byte_compile([stub_file], optimize=0, force=True)
        optimize = self.get_finalized_command('install_lib').optimize
        if optimize > 0:
            byte_compile(
                [stub_file],
                optimize=optimize,
                force=True,
            )
        if os.path.exists(stub_file):
            os.unlink(stub_file)


if use_stubs or os.name == 'nt':
    # Build shared libraries
    #
    def link_shared_object(
        self,
        objects,
        output_libname,
        output_dir=None,
        libraries=None,
        library_dirs=None,
        runtime_library_dirs=None,
        export_symbols=None,
        debug: bool = False,
        extra_preargs=None,
        extra_postargs=None,
        build_temp=None,
        target_lang=None,
    ) -> None:
        self.link(
            self.SHARED_LIBRARY,
            objects,
            output_libname,
            output_dir,
            libraries,
            library_dirs,
            runtime_library_dirs,
            export_symbols,
            debug,
            extra_preargs,
            extra_postargs,
            build_temp,
            target_lang,
        )

else:
    # Build static libraries everywhere else
    libtype = 'static'

    def link_shared_object(
        self,
        objects,
        output_libname,
        output_dir=None,
        libraries=None,
        library_dirs=None,
        runtime_library_dirs=None,
        export_symbols=None,
        debug: bool = False,
        extra_preargs=None,
        extra_postargs=None,
        build_temp=None,
        target_lang=None,
    ) -> None:
        # XXX we need to either disallow these attrs on Library instances,
        # or warn/abort here if set, or something...
        # libraries=None, library_dirs=None, runtime_library_dirs=None,
        # export_symbols=None, extra_preargs=None, extra_postargs=None,
        # build_temp=None

        assert output_dir is None  # distutils build_ext doesn't pass this
        output_dir, filename = os.path.split(output_libname)
        basename, _ext = os.path.splitext(filename)
        if self.library_filename("x").startswith('lib'):
            # strip 'lib' prefix; this is kludgy if some platform uses
            # a different prefix
            basename = basename[3:]

        self.create_static_lib(objects, basename, output_dir, debug, target_lang)
<|endoftext|>from ..dist import Distribution
from ..modified import newer_pairwise_group

import distutils.command.build_clib as orig
from distutils import log
from distutils.errors import DistutilsSetupError


class build_clib(orig.build_clib):
    """
    Override the default build_clib behaviour to do the following:

    1. Implement a rudimentary timestamp-based dependency system
       so 'compile()' doesn't run every time.
    2. Add more keys to the 'build_info' dictionary:
        * obj_deps - specify dependencies for each object compiled.
                     this should be a dictionary mapping a key
                     with the source filename to a list of
                     dependencies. Use an empty string for global
                     dependencies.
        * cflags   - specify a list of additional flags to pass to
                     the compiler.
    """

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution

    def build_libraries(self, libraries) -> None:
        for lib_name, build_info in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                    f"in 'libraries' option (library '{lib_name}'), "
                    "'sources' must be present and must be "
                    "a list of source filenames"
                )
            sources = sorted(sources)

            log.info("building '%s' library", lib_name)

            # Make sure everything is the correct type.
            # obj_deps should be a dictionary of keys as sources
            # and a list/tuple of files that are its dependencies.
            obj_deps = build_info.get('obj_deps', {})
            if not isinstance(obj_deps, dict):
                raise DistutilsSetupError(
                    f"in 'libraries' option (library '{lib_name}'), "
                    "'obj_deps' must be a dictionary of "
                    "type 'source: list'"
                )
            dependencies = []

            # Get the global dependencies that are specified by the '' key.
            # These will go into every source's dependency list.
            global_deps = obj_deps.get('', [])
            if not isinstance(global_deps, (list, tuple)):
                raise DistutilsSetupError(
                    f"in 'libraries' option (library '{lib_name}'), "
                    "'obj_deps' must be a dictionary of "
                    "type 'source: list'"
                )

            # Build the list to be used by newer_pairwise_group
            # each source will be auto-added to its dependencies.
            for source in sources:
                src_deps = [source]
                src_deps.extend(global_deps)
                extra_deps = obj_deps.get(source, [])
                if not isinstance(extra_deps, (list, tuple)):
                    raise DistutilsSetupError(
                        f"in 'libraries' option (library '{lib_name}'), "
                        "'obj_deps' must be a dictionary of "
                        "type 'source: list'"
                    )
                src_deps.extend(extra_deps)
                dependencies.append(src_deps)

            expected_objects = self.compiler.object_filenames(
                sources,
                output_dir=self.build_temp,
            )

            if newer_pairwise_group(dependencies, expected_objects) != ([], []):
                # First, compile the source code to object files in the library
                # directory.  (This should probably change to putting object
                # files in a temporary build directory.)
                macros = build_info.get('macros')
                include_dirs = build_info.get('include_dirs')
                cflags = build_info.get('cflags')
                self.compiler.compile(
                    sources,
                    output_dir=self.build_temp,
                    macros=macros,
                    include_dirs=include_dirs,
                    extra_postargs=cflags,
                    debug=self.debug,
                )

            # Now "link" the object files together into a static library.
            # (On Unix at least, this isn't really linking -- it just
            # builds an archive.  Whatever.)
            self.compiler.create_static_lib(
                expected_objects, lib_name, output_dir=self.build_clib, debug=self.debug
            )
<|endoftext|>from __future__ import annotations

import os
import sys
from itertools import product, starmap

from .._path import StrPath
from ..dist import Distribution

import distutils.command.install_lib as orig


class install_lib(orig.install_lib):
    """Don't add compiled flags to filenames of non-Python files"""

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution

    def run(self) -> None:
        self.build()
        outfiles = self.install()
        if outfiles is not None:
            # always compile, in case we have any extension stubs to deal with
            self.byte_compile(outfiles)

    def get_exclusions(self):
        """
        Return a collections.Sized collections.Container of paths to be
        excluded for single_version_externally_managed installations.
        """
        all_packages = (
            pkg
            for ns_pkg in self._get_SVEM_NSPs()
            for pkg in self._all_packages(ns_pkg)
        )

        excl_specs = product(all_packages, self._gen_exclusion_paths())
        return set(starmap(self._exclude_pkg_path, excl_specs))

    def _exclude_pkg_path(self, pkg, exclusion_path):
        """
        Given a package name and exclusion path within that package,
        compute the full exclusion path.
        """
        parts = pkg.split('.') + [exclusion_path]
        return os.path.join(self.install_dir, *parts)

    @staticmethod
    def _all_packages(pkg_name):
        """
        >>> list(install_lib._all_packages('foo.bar.baz'))
        ['foo.bar.baz', 'foo.bar', 'foo']
        """
        while pkg_name:
            yield pkg_name
            pkg_name, _sep, _child = pkg_name.rpartition('.')

    def _get_SVEM_NSPs(self):
        """
        Get namespace packages (list) but only for
        single_version_externally_managed installations and empty otherwise.
        """
        # TODO: is it necessary to short-circuit here? i.e. what's the cost
        # if get_finalized_command is called even when namespace_packages is
        # False?
        if not self.distribution.namespace_packages:
            return []

        install_cmd = self.get_finalized_command('install')
        svem = install_cmd.single_version_externally_managed

        return self.distribution.namespace_packages if svem else []

    @staticmethod
    def _gen_exclusion_paths():
        """
        Generate file paths to be excluded for namespace packages (bytecode
        cache files).
        """
        # always exclude the package module itself
        yield '__init__.py'

        yield '__init__.pyc'
        yield '__init__.pyo'

        if not hasattr(sys, 'implementation'):
            return

        base = os.path.join('__pycache__', '__init__.' + sys.implementation.cache_tag)
        yield base + '.pyc'
        yield base + '.pyo'
        yield base + '.opt-1.pyc'
        yield base + '.opt-2.pyc'

    def copy_tree(
        self,
        infile: StrPath,
        outfile: str,
        # override: Using actual booleans
        preserve_mode: bool = True,  # type: ignore[override]
        preserve_times: bool = True,  # type: ignore[override]
        preserve_symlinks: bool = False,  # type: ignore[override]
        level: object = 1,
    ) -> list[str]:
        assert preserve_mode
        assert preserve_times
        assert not preserve_symlinks
        exclude = self.get_exclusions()

        if not exclude:
            return orig.install_lib.copy_tree(self, infile, outfile)

        # Exclude namespace package __init__.py* files from the output

        from setuptools.archive_util import unpack_directory

        from distutils import log

        outfiles: list[str] = []

        def pf(src: str, dst: str):
            if dst in exclude:
                log.warn("Skipping installation of %s (namespace package)", dst)
                return False

            log.info("copying %s -> %s", src, os.path.dirname(dst))
            outfiles.append(dst)
            return dst

        unpack_directory(infile, outfile, pf)
        return outfiles

    def get_outputs(self):
        outputs = orig.install_lib.get_outputs(self)
        exclude = self.get_exclusions()
        if exclude:
            return [f for f in outputs if f not in exclude]
        return outputs
<|endoftext|>import os
from typing import ClassVar

from setuptools import Command, namespaces
from setuptools.archive_util import unpack_archive

from .._path import ensure_directory

from distutils import dir_util, log


class install_egg_info(namespaces.Installer, Command):
    """Install an .egg-info directory for the package"""

    description = "Install an .egg-info directory for the package"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('install-dir=', 'd', "directory to install to"),
    ]

    def initialize_options(self):
        self.install_dir = None

    def finalize_options(self) -> None:
        self.set_undefined_options('install_lib', ('install_dir', 'install_dir'))
        ei_cmd = self.get_finalized_command("egg_info")
        basename = f"{ei_cmd._get_egg_basename()}.egg-info"
        self.source = ei_cmd.egg_info
        self.target = os.path.join(self.install_dir, basename)
        self.outputs: list[str] = []

    def run(self) -> None:
        self.run_command('egg_info')
        if os.path.isdir(self.target) and not os.path.islink(self.target):
            dir_util.remove_tree(self.target)
        elif os.path.exists(self.target):
            self.execute(os.unlink, (self.target,), "Removing " + self.target)
        ensure_directory(self.target)
        self.execute(self.copytree, (), f"Copying {self.source} to {self.target}")
        self.install_namespaces()

    def get_outputs(self):
        return self.outputs

    def copytree(self) -> None:
        # Copy the .egg-info tree to site-packages
        def skimmer(src, dst):
            # filter out source-control directories; note that 'src' is always
            # a '/'-separated path, regardless of platform.  'dst' is a
            # platform-specific path.
            for skip in '.svn/', 'CVS/':
                if src.startswith(skip) or '/' + skip in src:
                    return None
            self.outputs.append(dst)
            log.debug("Copying %s to %s", src, dst)
            return dst

        unpack_archive(self.source, self.target, skimmer)
<|endoftext|>from setuptools.command.setopt import edit_config, option_base


class saveopts(option_base):
    """Save command-line options to a file"""

    description = "save supplied options to setup.cfg or other config file"

    def run(self) -> None:
        dist = self.distribution
        settings: dict[str, dict[str, str]] = {}

        for cmd in dist.command_options:
            if cmd == 'saveopts':
                continue  # don't save our own options!

            for opt, (src, val) in dist.get_option_dict(cmd).items():
                if src == "command line":
                    settings.setdefault(cmd, {})[opt] = val

        edit_config(self.filename, settings)
<|endoftext|>from __future__ import annotations

import fnmatch
import itertools
import operator
import os
import stat
import textwrap
from collections.abc import Iterable, Iterator
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any

from more_itertools import unique_everseen

from .._path import StrPath, StrPathT
from ..dist import Distribution
from ..warnings import SetuptoolsDeprecationWarning

import distutils.command.build_py as orig
import distutils.errors
from distutils.util import convert_path

_IMPLICIT_DATA_FILES = ('*.pyi', 'py.typed')


def make_writable(target) -> None:
    os.chmod(target, os.stat(target).st_mode | stat.S_IWRITE)


class build_py(orig.build_py):
    """Enhanced 'build_py' command that includes data files with packages

    The data files are specified via a 'package_data' argument to 'setup()'.
    See 'setuptools.dist.Distribution' for more details.

    Also, this version of the 'build_py' command allows you to specify both
    'py_modules' and 'packages' in the same setup operation.
    """

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution
    editable_mode: bool = False
    existing_egg_info_dir: StrPath | None = None  #: Private API, internal use only.

    def finalize_options(self) -> None:
        orig.build_py.finalize_options(self)
        self.package_data = self.distribution.package_data
        self.exclude_package_data = self.distribution.exclude_package_data or {}
        if 'data_files' in self.__dict__:
            del self.__dict__['data_files']

    def copy_file(  # type: ignore[override] # No overload, no bytes support
        self,
        infile: StrPath,
        outfile: StrPathT,
        preserve_mode: bool = True,
        preserve_times: bool = True,
        link: str | None = None,
        level: object = 1,
    ) -> tuple[StrPathT | str, bool]:
        # Overwrite base class to allow using links
        if link:
            infile = str(Path(infile).resolve())
            outfile = str(Path(outfile).resolve())  # type: ignore[assignment] # Re-assigning a str when outfile is StrPath is ok
        return super().copy_file(  # pyright: ignore[reportReturnType] # pypa/distutils#309
            infile, outfile, preserve_mode, preserve_times, link, level
        )

    def run(self) -> None:
        """Build modules, packages, and copy data files to build directory"""
        if not (self.py_modules or self.packages) or self.editable_mode:
            return

        if self.py_modules:
            self.build_modules()

        if self.packages:
            self.build_packages()
            self.build_package_data()

        # Only compile actual .py files, using our base class' idea of what our
        # output files are.
        self.byte_compile(orig.build_py.get_outputs(self, include_bytecode=False))

    # Should return "list[tuple[str, str, str, list[str]]] | Any" but can't do without typed distutils on Python 3.12+
    def __getattr__(self, attr: str) -> Any:
        "lazily compute data files"
        if attr == 'data_files':
            self.data_files = self._get_data_files()
            return self.data_files
        return orig.build_py.__getattr__(self, attr)

    def _get_data_files(self):
        """Generate list of '(package,src_dir,build_dir,filenames)' tuples"""
        self.analyze_manifest()
        return list(map(self._get_pkg_data_files, self.packages or ()))

    def get_data_files_without_manifest(self) -> list[tuple[str, str, str, list[str]]]:
        """
        Generate list of ``(package,src_dir,build_dir,filenames)`` tuples,
        but without triggering any attempt to analyze or build the manifest.
        """
        # Prevent eventual errors from unset `manifest_files`
        # (that would otherwise be set by `analyze_manifest`)
        self.__dict__.setdefault('manifest_files', {})
        return list(map(self._get_pkg_data_files, self.packages or ()))

    def _get_pkg_data_files(self, package: str) -> tuple[str, str, str, list[str]]:
        # Locate package source directory
        src_dir = self.get_package_dir(package)

        # Compute package build directory
        build_dir = os.path.join(*([self.build_lib] + package.split('.')))

        # Strip directory from globbed filenames
        filenames = [
            os.path.relpath(file, src_dir)
            for file in self.find_data_files(package, src_dir)
        ]
        return package, src_dir, build_dir, filenames

    def find_data_files(self, package, src_dir):
        """Return filenames for package's data files in 'src_dir'"""
        patterns = self._get_platform_patterns(
            self.package_data,
            package,
            src_dir,
            extra_patterns=_IMPLICIT_DATA_FILES,
        )
        globs_expanded = map(partial(glob, recursive=True), patterns)
        # flatten the expanded globs into an iterable of matches
        globs_matches = itertools.chain.from_iterable(globs_expanded)
        glob_files = filter(os.path.isfile, globs_matches)
        files = itertools.chain(
            self.manifest_files.get(package, []),
            glob_files,
        )
        return self.exclude_data_files(package, src_dir, files)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:  # type: ignore[override] # Using a real boolean instead of 0|1
        """See :class:`setuptools.commands.build.SubCommand`"""
        if self.editable_mode:
            return list(self.get_output_mapping().keys())
        return super().get_outputs(include_bytecode)

    def get_output_mapping(self) -> dict[str, str]:
        """See :class:`setuptools.commands.build.SubCommand`"""
        mapping = itertools.chain(
            self._get_package_data_output_mapping(),
            self._get_module_mapping(),
        )
        return dict(sorted(mapping, key=operator.itemgetter(0)))

    def _get_module_mapping(self) -> Iterator[tuple[str, str]]:
        """Iterate over all modules producing (dest, src) pairs."""
        for package, module, module_file in self.find_all_modules():
            package = package.split('.')
            filename = self.get_module_outfile(self.build_lib, package, module)
            yield (filename, module_file)

    def _get_package_data_output_mapping(self) -> Iterator[tuple[str, str]]:
        """Iterate over package data producing (dest, src) pairs."""
        for package, src_dir, build_dir, filenames in self.data_files:
            for filename in filenames:
                target = os.path.join(build_dir, filename)
                srcfile = os.path.join(src_dir, filename)
                yield (target, srcfile)

    def build_package_data(self) -> None:
        """Copy data files into build directory"""
        for target, srcfile in self._get_package_data_output_mapping():
            self.mkpath(os.path.dirname(target))
            _outf, _copied = self.copy_file(srcfile, target)
            make_writable(target)

    def analyze_manifest(self) -> None:
        self.manifest_files: dict[str, list[str]] = {}
        if not self.distribution.include_package_data:
            return
        src_dirs: dict[str, str] = {}
        for package in self.packages or ():
            # Locate package source directory
            src_dirs[assert_relative(self.get_package_dir(package))] = package

        if (
            self.existing_egg_info_dir
            and Path(self.existing_egg_info_dir, "SOURCES.txt").exists()
        ):
            egg_info_dir = self.existing_egg_info_dir
            manifest = Path(egg_info_dir, "SOURCES.txt")
            files = manifest.read_text(encoding="utf-8").splitlines()
        else:
            self.run_command('egg_info')
            ei_cmd = self.get_finalized_command('egg_info')
            egg_info_dir = ei_cmd.egg_info
            files = ei_cmd.filelist.files

        check = _IncludePackageDataAbuse()
        for path in self._filter_build_files(files, egg_info_dir):
            d, f = os.path.split(assert_relative(path))
            prev = None
            oldf = f
            while d and d != prev and d not in src_dirs:
                prev = d
                d, df = os.path.split(d)
                f = os.path.join(df, f)
            if d in src_dirs:
                if f == oldf:
                    if check.is_module(f):
                        continue  # it's a module, not data
                else:
                    importable = check.importable_subpackage(src_dirs[d], f)
                    if importable:
                        check.warn(importable)
                self.manifest_files.setdefault(src_dirs[d], []).append(path)

    def _filter_build_files(
        self, files: Iterable[str], egg_info: StrPath
    ) -> Iterator[str]:
        """
        ``build_meta`` may try to create egg_info outside of the project directory,
        and this can be problematic for certain plugins (reported in issue #3500).

        Extensions might also include between their sources files created on the
        ``build_lib`` and ``build_temp`` directories.

        This function should filter this case of invalid files out.
        """
        build = self.get_finalized_command("build")
        build_dirs = (egg_info, self.build_lib, build.build_temp, build.build_base)
        norm_dirs = [os.path.normpath(p) for p in build_dirs if p]

        for file in files:
            norm_path = os.path.normpath(file)
            if not os.path.isabs(file) or all(d not in norm_path for d in norm_dirs):
                yield file

    def get_data_files(self) -> None:
        pass  # Lazily compute data files in _get_data_files() function.

    def check_package(self, package, package_dir):
        """Check namespace packages' __init__ for declare_namespace"""
        try:
            return self.packages_checked[package]
        except KeyError:
            pass

        init_py = orig.build_py.check_package(self, package, package_dir)
        self.packages_checked[package] = init_py

        if not init_py or not self.distribution.namespace_packages:
            return init_py

        for pkg in self.distribution.namespace_packages:
            if pkg == package or pkg.startswith(package + '.'):
                break
        else:
            return init_py

        with open(init_py, 'rb') as f:
            contents = f.read()
        if b'declare_namespace' not in contents:
            raise distutils.errors.DistutilsError(
                f"Namespace package problem: {package} is a namespace package, but "
                "its\n__init__.py does not call declare_namespace()! Please "
                'fix it.\n(See the setuptools manual under '
                '"Namespace Packages" for details.)\n"'
            )
        return init_py

    def initialize_options(self):
        self.packages_checked = {}
        orig.build_py.initialize_options(self)
        self.editable_mode = False
        self.existing_egg_info_dir = None

    def get_package_dir(self, package: str) -> str:
        res = orig.build_py.get_package_dir(self, package)
        if self.distribution.src_root is not None:
            return os.path.join(self.distribution.src_root, res)
        return res

    def exclude_data_files(self, package, src_dir, files):
        """Filter filenames for package's data files in 'src_dir'"""
        files = list(files)
        patterns = self._get_platform_patterns(
            self.exclude_package_data,
            package,
            src_dir,
        )
        match_groups = (fnmatch.filter(files, pattern) for pattern in patterns)
        # flatten the groups of matches into an iterable of matches
        matches = itertools.chain.from_iterable(match_groups)
        bad = set(matches)
        keepers = (fn for fn in files if fn not in bad)
        # ditch dupes
        return list(unique_everseen(keepers))

    @staticmethod
    def _get_platform_patterns(spec, package, src_dir, extra_patterns=()):
        """
        yield platform-specific path patterns (suitable for glob
        or fn_match) from a glob-based spec (such as
        self.package_data or self.exclude_package_data)
        matching package in src_dir.
        """
        raw_patterns = itertools.chain(
            extra_patterns,
            spec.get('', []),
            spec.get(package, []),
        )
        return (
            # Each pattern has to be converted to a platform-specific path
            os.path.join(src_dir, convert_path(pattern))
            for pattern in raw_patterns
        )


def assert_relative(path):
    if not os.path.isabs(path):
        return path
    from distutils.errors import DistutilsSetupError

    msg = (
        textwrap.dedent(
            """
        Error: setup script specifies an absolute path:

            %s

        setup() arguments must *always* be /-separated paths relative to the
        setup.py directory, *never* absolute paths.
        """
        ).lstrip()
        % path
    )
    raise DistutilsSetupError(msg)


class _IncludePackageDataAbuse:
    """Inform users that package or module is included as 'data file'"""

    class _Warning(SetuptoolsDeprecationWarning):
        _SUMMARY = """
        Package {importable!r} is absent from the `packages` configuration.
        """

        _DETAILS = """
        ############################
        # Package would be ignored #
        ############################
        Python recognizes {importable!r} as an importable package[^1],
        but it is absent from setuptools' `packages` configuration.

        This leads to an ambiguous overall configuration. If you want to distribute this
        package, please make sure that {importable!r} is explicitly added
        to the `packages` configuration field.

        Alternatively, you can also rely on setuptools' discovery methods
        (for example by using `find_namespace_packages(...)`/`find_namespace:`
        instead of `find_packages(...)`/`find:`).

        You can read more about "package discovery" on setuptools documentation page:

        - https://setuptools.pypa.io/en/latest/userguide/package_discovery.html

        If you don't want {importable!r} to be distributed and are
        already explicitly excluding {importable!r} via
        `find_namespace_packages(...)/find_namespace` or `find_packages(...)/find`,
        you can try to use `exclude_package_data`, or `include-package-data=False` in
        combination with a more fine grained `package-data` configuration.

        You can read more about "package data files" on setuptools documentation page:

        - https://setuptools.pypa.io/en/latest/userguide/datafiles.html


        [^1]: For Python, any directory (with suitable naming) can be imported,
              even if it does not contain any `.py` files.
              On the other hand, currently there is no concept of package data
              directory, all directories are treated like packages.
        """
        # _DUE_DATE: still not defined as this is particularly controversial.
        # Warning initially introduced in May 2022. See issue #3340 for discussion.

    def __init__(self) -> None:
        self._already_warned = set[str]()

    def is_module(self, file):
        return file.endswith(".py") and file[: -len(".py")].isidentifier()

    def importable_subpackage(self, parent, file):
        pkg = Path(file).parent
        parts = list(itertools.takewhile(str.isidentifier, pkg.parts))
        if parts:
            return ".".join([parent, *parts])
        return None

    def warn(self, importable):
        if importable not in self._already_warned:
            self._Warning.emit(importable=importable)
            self._already_warned.add(importable)
<|endoftext|>from __future__ import annotations

from typing import ClassVar, NoReturn

from setuptools import Command
from setuptools.warnings import SetuptoolsDeprecationWarning


# Would restrict to Literal["test"], but mypy doesn't support it: https://github.com/python/mypy/issues/8203
def __getattr__(name: str) -> type[_test]:
    if name == 'test':
        SetuptoolsDeprecationWarning.emit(
            "The test command is disabled and references to it are deprecated.",
            "Please remove any references to `setuptools.command.test` in all "
            "supported versions of the affected package.",
            due_date=(2024, 11, 15),
            stacklevel=2,
        )
        return _test
    raise AttributeError(name)


class _test(Command):
    """
    Stub to warn when test command is referenced or used.
    """

    description = "stub for old test command (do not use)"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('test-module=', 'm', "Run 'test_suite' in specified module"),
        (
            'test-suite=',
            's',
            "Run single test, case or suite (e.g. 'module.test_suite')",
        ),
        ('test-runner=', 'r', "Test runner to use"),
    ]

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> NoReturn:
        raise RuntimeError("Support for the test command was removed in Setuptools 72")
<|endoftext|>from __future__ import annotations

import os
import sys

from .._path import ensure_directory
from ..dist import Distribution

import distutils.command.install_scripts as orig
from distutils import log


class install_scripts(orig.install_scripts):
    """Do normal script install, plus any egg_info wrapper scripts"""

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution

    def initialize_options(self) -> None:
        orig.install_scripts.initialize_options(self)
        self.no_ep = False

    def run(self) -> None:
        self.run_command("egg_info")
        if self.distribution.scripts:
            orig.install_scripts.run(self)  # run first to set up self.outfiles
        else:
            self.outfiles: list[str] = []
        if self.no_ep:
            # don't install entry point scripts into .egg file!
            return
        self._install_ep_scripts()

    def _install_ep_scripts(self):
        # Delay import side-effects
        from .. import _scripts
        from .._importlib import metadata

        ei_cmd = self.get_finalized_command("egg_info")
        dist = metadata.Distribution.at(path=ei_cmd.egg_info)
        bs_cmd = self.get_finalized_command('build_scripts')
        exec_param = getattr(bs_cmd, 'executable', None)
        writer = _scripts.ScriptWriter
        if exec_param == sys.executable:
            # In case the path to the Python executable contains a space, wrap
            # it so it's not split up.
            exec_param = [exec_param]
        # resolve the writer to the environment
        writer = writer.best()
        cmd = writer.command_spec_class.best().from_param(exec_param)
        for args in writer.get_args(dist, cmd.as_header()):
            self.write_script(*args)

    def write_script(self, script_name, contents, mode: str = "t", *ignored) -> None:
        """Write an executable file to the scripts directory"""
        from .._shutil import attempt_chmod_verbose as chmod, current_umask

        log.info("Installing %s script to %s", script_name, self.install_dir)
        target = os.path.join(self.install_dir, script_name)
        self.outfiles.append(target)

        encoding = None if "b" in mode else "utf-8"
        mask = current_umask()
        ensure_directory(target)
        with open(target, "w" + mode, encoding=encoding) as f:
            f.write(contents)
        chmod(target, 0o777 - mask)
<|endoftext|>from .compat.numpy import (  # noqa: F401
    _default_compilers,
    compiler_class,
)
from .compilers.C import base
from .compilers.C.base import (
    gen_lib_options,
    gen_preprocess_options,
    get_compilers,
    get_default_compiler,
    new_compiler,
)
from .compilers.C.errors import CompileError, LinkError

__all__ = [
    'CompileError',
    'LinkError',
    'gen_lib_options',
    'gen_preprocess_options',
    'get_default_compiler',
    'new_compiler',
    'show_compilers',
]


CCompiler = base.Compiler


def show_compilers() -> None:
    """Print list of available compilers (used by the "--help-compiler"
    options to "build", "build_ext", "build_clib").
    """
    # XXX this "knows" that the compiler option it's describing is
    # "--compiler", which just happens to be the case for the three
    # commands that use it.
    from .fancy_getopt import FancyGetopt

    compilers = sorted(
        ("compiler=" + name, None, cls.description)
        for name, cls in get_compilers().items()
    )
    FancyGetopt(compilers).print_help("List of available compilers:")
<|endoftext|>"""Module for parsing and testing package version predicate strings."""

import operator
import re

from . import version

re_validPackage = re.compile(r"(?i)^\s*([a-z_]\w*(?:\.[a-z_]\w*)*)(.*)", re.ASCII)
# (package) (rest)

re_paren = re.compile(r"^\s*\((.*)\)\s*$")  # (list) inside of parentheses
re_splitComparison = re.compile(r"^\s*(<=|>=|<|>|!=|==)\s*([^\s,]+)\s*$")
# (comp) (version)


def splitUp(pred):
    """Parse a single version comparison.

    Return (comparison string, StrictVersion)
    """
    res = re_splitComparison.match(pred)
    if not res:
        raise ValueError(f"bad package restriction syntax: {pred!r}")
    comp, verStr = res.groups()
    with version.suppress_known_deprecation():
        other = version.StrictVersion(verStr)
    return (comp, other)


compmap = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    ">=": operator.ge,
    "!=": operator.ne,
}


class VersionPredicate:
    """Parse and test package version predicates.

    >>> v = VersionPredicate('pyepat.abc (>1.0, <3333.3a1, !=1555.1b3)')

    The `name` attribute provides the full dotted name that is given::

    >>> v.name
    'pyepat.abc'

    The str() of a `VersionPredicate` provides a normalized
    human-readable version of the expression::

    >>> print(v)
    pyepat.abc (> 1.0, < 3333.3a1, != 1555.1b3)

    The `satisfied_by()` method can be used to determine with a given
    version number is included in the set described by the version
    restrictions::

    >>> v.satisfied_by('1.1')
    True
    >>> v.satisfied_by('1.4')
    True
    >>> v.satisfied_by('1.0')
    False
    >>> v.satisfied_by('4444.4')
    False
    >>> v.satisfied_by('1555.1b3')
    False

    `VersionPredicate` is flexible in accepting extra whitespace::

    >>> v = VersionPredicate(' pat( ==  0.1  )  ')
    >>> v.name
    'pat'
    >>> v.satisfied_by('0.1')
    True
    >>> v.satisfied_by('0.2')
    False

    If any version numbers passed in do not conform to the
    restrictions of `StrictVersion`, a `ValueError` is raised::

    >>> v = VersionPredicate('p1.p2.p3.p4(>=1.0, <=1.3a1, !=1.2zb3)')
    Traceback (most recent call last):
      ...
    ValueError: invalid version number '1.2zb3'

    It the module or package name given does not conform to what's
    allowed as a legal module or package name, `ValueError` is
    raised::

    >>> v = VersionPredicate('foo-bar')
    Traceback (most recent call last):
      ...
    ValueError: expected parenthesized list: '-bar'

    >>> v = VersionPredicate('foo bar (12.21)')
    Traceback (most recent call last):
      ...
    ValueError: expected parenthesized list: 'bar (12.21)'

    """

    def __init__(self, versionPredicateStr):
        """Parse a version predicate string."""
        # Fields:
        #    name:  package name
        #    pred:  list of (comparison string, StrictVersion)

        versionPredicateStr = versionPredicateStr.strip()
        if not versionPredicateStr:
            raise ValueError("empty package restriction")
        match = re_validPackage.match(versionPredicateStr)
        if not match:
            raise ValueError(f"bad package name in {versionPredicateStr!r}")
        self.name, paren = match.groups()
        paren = paren.strip()
        if paren:
            match = re_paren.match(paren)
            if not match:
                raise ValueError(f"expected parenthesized list: {paren!r}")
            str = match.groups()[0]
            self.pred = [splitUp(aPred) for aPred in str.split(",")]
            if not self.pred:
                raise ValueError(f"empty parenthesized list in {versionPredicateStr!r}")
        else:
            self.pred = []

    def __str__(self):
        if self.pred:
            seq = ", ".join(f"{cond} {ver}" for cond, ver in self.pred)
            return f"{self.name} ({seq})"
        else:
            return self.name

    def satisfied_by(self, version):
        """True if version is compatible with all the predicates in self.
        The parameter version must be acceptable to the StrictVersion
        constructor.  It may be either a string or StrictVersion.
        """
        for cond, ver in self.pred:
            if not compmap[cond](version, ver):
                return False
        return True


_provision_rx = None


def split_provision(value):
    """Return the name and optional version number of a provision.

    The version number, if given, will be returned as a `StrictVersion`
    instance, otherwise it will be `None`.

    >>> split_provision('mypkg')
    ('mypkg', None)
    >>> split_provision(' mypkg( 1.2 ) ')
    ('mypkg', StrictVersion ('1.2'))
    """
    global _provision_rx
    if _provision_rx is None:
        _provision_rx = re.compile(
            r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)(?:\s*\(\s*([^)\s]+)\s*\))?$", re.ASCII
        )
    value = value.strip()
    m = _provision_rx.match(value)
    if not m:
        raise ValueError(f"illegal provides specification: {value!r}")
    ver = m.group(2) or None
    if ver:
        with version.suppress_known_deprecation():
            ver = version.StrictVersion(ver)
    return m.group(1), ver
<|endoftext|>"""distutils.fancy_getopt

Wrapper around the standard getopt module that provides the following
additional features:
  * short and long options are tied together
  * options have help strings, so fancy_getopt could potentially
    create a complete usage summary
  * options set attributes of a passed-in object
"""

from __future__ import annotations

import getopt
import re
import string
import sys
from collections.abc import Sequence
from typing import Any

from .errors import DistutilsArgError, DistutilsGetoptError

# Much like command_re in distutils.core, this is close to but not quite
# the same as a Python NAME -- except, in the spirit of most GNU
# utilities, we use '-' in place of '_'.  (The spirit of LISP lives on!)
# The similarities to NAME are again not a coincidence...
longopt_pat = r'[a-zA-Z](?:[a-zA-Z0-9-]*)'
longopt_re = re.compile(rf'^{longopt_pat}$')

# For recognizing "negative alias" options, eg. "quiet=!verbose"
neg_alias_re = re.compile(f"^({longopt_pat})=!({longopt_pat})$")

# This is used to translate long options to legitimate Python identifiers
# (for use as attributes of some object).
longopt_xlate = str.maketrans('-', '_')


class FancyGetopt:
    """Wrapper around the standard 'getopt()' module that provides some
    handy extra functionality:
      * short and long options are tied together
      * options have help strings, and help text can be assembled
        from them
      * options set attributes of a passed-in object
      * boolean options can have "negative aliases" -- eg. if
        --quiet is the "negative alias" of --verbose, then "--quiet"
        on the command line sets 'verbose' to false
    """

    def __init__(self, option_table=None):
        # The option table is (currently) a list of tuples.  The
        # tuples may have 3 or four values:
        #   (long_option, short_option, help_string [, repeatable])
        # if an option takes an argument, its long_option should have '='
        # appended; short_option should just be a single character, no ':'
        # in any case.  If a long_option doesn't have a corresponding
        # short_option, short_option should be None.  All option tuples
        # must have long options.
        self.option_table = option_table

        # 'option_index' maps long option names to entries in the option
        # table (ie. those 3-tuples).
        self.option_index = {}
        if self.option_table:
            self._build_index()

        # 'alias' records (duh) alias options; {'foo': 'bar'} means
        # --foo is an alias for --bar
        self.alias = {}

        # 'negative_alias' keeps track of options that are the boolean
        # opposite of some other option
        self.negative_alias = {}

        # These keep track of the information in the option table.  We
        # don't actually populate these structures until we're ready to
        # parse the command-line, since the 'option_table' passed in here
        # isn't necessarily the final word.
        self.short_opts = []
        self.long_opts = []
        self.short2long = {}
        self.attr_name = {}
        self.takes_arg = {}

        # And 'option_order' is filled up in 'getopt()'; it records the
        # original order of options (and their values) on the command-line,
        # but expands short options, converts aliases, etc.
        self.option_order = []

    def _build_index(self):
        self.option_index.clear()
        for option in self.option_table:
            self.option_index[option[0]] = option

    def set_option_table(self, option_table):
        self.option_table = option_table
        self._build_index()

    def add_option(self, long_option, short_option=None, help_string=None):
        if long_option in self.option_index:
            raise DistutilsGetoptError(
                f"option conflict: already an option '{long_option}'"
            )
        else:
            option = (long_option, short_option, help_string)
            self.option_table.append(option)
            self.option_index[long_option] = option

    def has_option(self, long_option: str) -> bool:
        """Return true if the option table for this parser has an
        option with long name 'long_option'."""
        return long_option in self.option_index

    def get_attr_name(self, long_option):
        """Translate long option name 'long_option' to the form it
        has as an attribute of some object: ie., translate hyphens
        to underscores."""
        return long_option.translate(longopt_xlate)

    def _check_alias_dict(self, aliases, what):
        assert isinstance(aliases, dict)
        for alias, opt in aliases.items():
            if alias not in self.option_index:
                raise DistutilsGetoptError(
                    f"invalid {what} '{alias}': option '{alias}' not defined"
                )
            if opt not in self.option_index:
                raise DistutilsGetoptError(
                    f"invalid {what} '{alias}': aliased option '{opt}' not defined"
                )

    def set_aliases(self, alias):
        """Set the aliases for this option parser."""
        self._check_alias_dict(alias, "alias")
        self.alias = alias

    def set_negative_aliases(self, negative_alias):
        """Set the negative aliases for this option parser.
        'negative_alias' should be a dictionary mapping option names to
        option names, both the key and value must already be defined
        in the option table."""
        self._check_alias_dict(negative_alias, "negative alias")
        self.negative_alias = negative_alias

    def _grok_option_table(self):  # noqa: C901
        """Populate the various data structures that keep tabs on the
        option table.  Called by 'getopt()' before it can do anything
        worthwhile.
        """
        self.long_opts = []
        self.short_opts = []
        self.short2long.clear()
        self.repeat = {}

        for option in self.option_table:
            if len(option) == 3:
                long, short, _help = option
                repeat = 0
            elif len(option) == 4:
                long, short, _help, repeat = option
            else:
                # the option table is part of the code, so simply
                # assert that it is correct
                raise ValueError(f"invalid option tuple: {option!r}")

            # Type- and value-check the option names
            if not isinstance(long, str) or len(long) < 2:
                raise DistutilsGetoptError(
                    f"invalid long option '{long}': must be a string of length >= 2"
                )

            if not ((short is None) or (isinstance(short, str) and len(short) == 1)):
                raise DistutilsGetoptError(
                    f"invalid short option '{short}': must a single character or None"
                )

            self.repeat[long] = repeat
            self.long_opts.append(long)

            if long[-1] == '=':  # option takes an argument?
                if short:
                    short = short + ':'
                long = long[0:-1]
                self.takes_arg[long] = True
            else:
                # Is option is a "negative alias" for some other option (eg.
                # "quiet" == "!verbose")?
                alias_to = self.negative_alias.get(long)
                if alias_to is not None:
                    if self.takes_arg[alias_to]:
                        raise DistutilsGetoptError(
                            f"invalid negative alias '{long}': "
                            f"aliased option '{alias_to}' takes a value"
                        )

                    self.long_opts[-1] = long  # XXX redundant?!
                self.takes_arg[long] = False

            # If this is an alias option, make sure its "takes arg" flag is
            # the same as the option it's aliased to.
            alias_to = self.alias.get(long)
            if (
                alias_to is not None
                and self.takes_arg[long] != self.takes_arg[alias_to]
            ):
                raise DistutilsGetoptError(
                    f"invalid alias '{long}': inconsistent with "
                    f"aliased option '{alias_to}' (one of them takes a value, "
                    "the other doesn't"
                )

            # Now enforce some bondage on the long option name, so we can
            # later translate it to an attribute name on some object.  Have
            # to do this a bit late to make sure we've removed any trailing
            # '='.
            if not longopt_re.match(long):
                raise DistutilsGetoptError(
                    f"invalid long option name '{long}' "
                    "(must be letters, numbers, hyphens only"
                )

            self.attr_name[long] = self.get_attr_name(long)
            if short:
                self.short_opts.append(short)
                self.short2long[short[0]] = long

    def getopt(self, args: Sequence[str] | None = None, object=None):  # noqa: C901
        """Parse command-line options in args. Store as attributes on object.

        If 'args' is None or not supplied, uses 'sys.argv[1:]'.  If
        'object' is None or not supplied, creates a new OptionDummy
        object, stores option values there, and returns a tuple (args,
        object).  If 'object' is supplied, it is modified in place and
        'getopt()' just returns 'args'; in both cases, the returned
        'args' is a modified copy of the passed-in 'args' list, which
        is left untouched.
        """
        if args is None:
            args = sys.argv[1:]
        if object is None:
            object = OptionDummy()
            created_object = True
        else:
            created_object = False

        self._grok_option_table()

        short_opts = ' '.join(self.short_opts)
        try:
            opts, args = getopt.getopt(args, short_opts, self.long_opts)
        except getopt.error as msg:
            raise DistutilsArgError(msg)

        for opt, val in opts:
            if len(opt) == 2 and opt[0] == '-':  # it's a short option
                opt = self.short2long[opt[1]]
            else:
                assert len(opt) > 2 and opt[:2] == '--'
                opt = opt[2:]

            alias = self.alias.get(opt)
            if alias:
                opt = alias

            if not self.takes_arg[opt]:  # boolean option?
                assert val == '', "boolean option can't have value"
                alias = self.negative_alias.get(opt)
                if alias:
                    opt = alias
                    val = 0
                else:
                    val = 1

            attr = self.attr_name[opt]
            # The only repeating option at the moment is 'verbose'.
            # It has a negative option -q quiet, which should set verbose = False.
            if val and self.repeat.get(attr) is not None:
                val = getattr(object, attr, 0) + 1
            setattr(object, attr, val)
            self.option_order.append((opt, val))

        # for opts
        if created_object:
            return args, object
        else:
            return args

    def get_option_order(self):
        """Returns the list of (option, value) tuples processed by the
        previous run of 'getopt()'.  Raises RuntimeError if
        'getopt()' hasn't been called yet.
        """
        if self.option_order is None:
            raise RuntimeError("'getopt()' hasn't been called yet")
        else:
            return self.option_order

    def generate_help(self, header=None):  # noqa: C901
        """Generate help text (a list of strings, one per suggested line of
        output) from the option table for this FancyGetopt object.
        """
        # Blithely assume the option table is good: probably wouldn't call
        # 'generate_help()' unless you've already called 'getopt()'.

        # First pass: determine maximum length of long option names
        max_opt = 0
        for option in self.option_table:
            long = option[0]
            short = option[1]
            ell = len(long)
            if long[-1] == '=':
                ell = ell - 1
            if short is not None:
                ell = ell + 5  # " (-x)" where short == 'x'
            max_opt = max(max_opt, ell)

        opt_width = max_opt + 2 + 2 + 2  # room for indent + dashes + gutter

        # Typical help block looks like this:
        #   --foo       controls foonabulation
        # Help block for longest option looks like this:
        #   --flimflam  set the flim-flam level
        # and with wrapped text:
        #   --flimflam  set the flim-flam level (must be between
        #               0 and 100, except on Tuesdays)
        # Options with short names will have the short name shown (but
        # it doesn't contribute to max_opt):
        #   --foo (-f)  controls foonabulation
        # If adding the short option would make the left column too wide,
        # we push the explanation off to the next line
        #   --flimflam (-l)
        #               set the flim-flam level
        # Important parameters:
        #   - 2 spaces before option block start lines
        #   - 2 dashes for each long option name
        #   - min. 2 spaces between option and explanation (gutter)
        #   - 5 characters (incl. space) for short option name

        # Now generate lines of help text.  (If 80 columns were good enough
        # for Jesus, then 78 columns are good enough for me!)
        line_width = 78
        text_width = line_width - opt_width
        big_indent = ' ' * opt_width
        if header:
            lines = [header]
        else:
            lines = ['Option summary:']

        for option in self.option_table:
            long, short, help = option[:3]
            text = wrap_text(help, text_width)
            if long[-1] == '=':
                long = long[0:-1]

            # Case 1: no short option at all (makes life easy)
            if short is None:
                if text:
                    lines.append(f"  --{long:<{max_opt}}  {text[0]}")
                else:
                    lines.append(f"  --{long:<{max_opt}}")

            # Case 2: we have a short option, so we have to include it
            # just after the long option
            else:
                opt_names = f"{long} (-{short})"
                if text:
                    lines.append(f"  --{opt_names:<{max_opt}}  {text[0]}")
                else:
                    lines.append(f"  --{opt_names:<{max_opt}}")

            for ell in text[1:]:
                lines.append(big_indent + ell)
        return lines

    def print_help(self, header=None, file=None):
        if file is None:
            file = sys.stdout
        for line in self.generate_help(header):
            file.write(line + "\n")


def fancy_getopt(options, negative_opt, object, args: Sequence[str] | None):
    parser = FancyGetopt(options)
    parser.set_negative_aliases(negative_opt)
    return parser.getopt(args, object)


WS_TRANS = {ord(_wschar): ' ' for _wschar in string.whitespace}


def wrap_text(text, width):
    """wrap_text(text : string, width : int) -> [string]

    Split 'text' into multiple lines of no more than 'width' characters
    each, and return the list of strings that results.
    """
    if text is None:
        return []
    if len(text) <= width:
        return [text]

    text = text.expandtabs()
    text = text.translate(WS_TRANS)
    chunks = re.split(r'( +|-+)', text)
    chunks = [ch for ch in chunks if ch]  # ' - ' results in empty strings
    lines = []

    while chunks:
        cur_line = []  # list of chunks (to-be-joined)
        cur_len = 0  # length of current line

        while chunks:
            ell = len(chunks[0])
            if cur_len + ell <= width:  # can squeeze (at least) this chunk in
                cur_line.append(chunks[0])
                del chunks[0]
                cur_len = cur_len + ell
            else:  # this line is full
                # drop last chunk if all space
                if cur_line and cur_line[-1][0] == ' ':
                    del cur_line[-1]
                break

        if chunks:  # any chunks left to process?
            # if the current line is still empty, then we had a single
            # chunk that's too big too fit on a line -- so we break
            # down and break it up at the line width
            if cur_len == 0:
                cur_line.append(chunks[0][0:width])
                chunks[0] = chunks[0][width:]

            # all-whitespace chunks at the end of a line can be discarded
            # (and we know from the re.split above that if a chunk has
            # *any* whitespace, it is *all* whitespace)
            if chunks[0][0] == ' ':
                del chunks[0]

        # and store this line in the list-of-all-lines -- as a single
        # string, of course!
        lines.append(''.join(cur_line))

    return lines


def translate_longopt(opt):
    """Convert a long option name to a valid Python identifier by
    changing "-" to "_".
    """
    return opt.translate(longopt_xlate)


class OptionDummy:
    """Dummy class just used as a place to hold command-line option
    values as instance attributes."""

    def __init__(self, options: Sequence[Any] = []):
        """Create a new OptionDummy instance.  The attributes listed in
        'options' will be initialized to None."""
        for opt in options:
            setattr(self, opt, None)


if __name__ == "__main__":
    text = """\
Tra-la-la, supercalifragilisticexpialidocious.
How *do* you spell that odd word, anyways?
(Someone ask Mary -- she'll know [or she'll
say, "How should I know?"].)"""

    for w in (10, 20, 30, 40):
        print(f"width: {w}")
        print("\n".join(wrap_text(text, w)))
        print()
<|endoftext|>"""text_file

provides the TextFile class, which gives an interface to text files
that (optionally) takes care of stripping comments, ignoring blank
lines, and joining lines with backslashes."""

import sys
from typing import ClassVar


class TextFile:
    """Provides a file-like object that takes care of all the things you
    commonly want to do when processing a text file that has some
    line-by-line syntax: strip comments (as long as "#" is your
    comment character), skip blank lines, join adjacent lines by
    escaping the newline (ie. backslash at end of line), strip
    leading and/or trailing whitespace.  All of these are optional
    and independently controllable.

    Provides a 'warn()' method so you can generate warning messages that
    report physical line number, even if the logical line in question
    spans multiple physical lines.  Also provides 'unreadline()' for
    implementing line-at-a-time lookahead.

    Constructor is called as:

        TextFile (filename=None, file=None, **options)

    It bombs (RuntimeError) if both 'filename' and 'file' are None;
    'filename' should be a string, and 'file' a file object (or
    something that provides 'readline()' and 'close()' methods).  It is
    recommended that you supply at least 'filename', so that TextFile
    can include it in warning messages.  If 'file' is not supplied,
    TextFile creates its own using 'io.open()'.

    The options are all boolean, and affect the value returned by
    'readline()':
      strip_comments [default: true]
        strip from "#" to end-of-line, as well as any whitespace
        leading up to the "#" -- unless it is escaped by a backslash
      lstrip_ws [default: false]
        strip leading whitespace from each line before returning it
      rstrip_ws [default: true]
        strip trailing whitespace (including line terminator!) from
        each line before returning it
      skip_blanks [default: true}
        skip lines that are empty *after* stripping comments and
        whitespace.  (If both lstrip_ws and rstrip_ws are false,
        then some lines may consist of solely whitespace: these will
        *not* be skipped, even if 'skip_blanks' is true.)
      join_lines [default: false]
        if a backslash is the last non-newline character on a line
        after stripping comments and whitespace, join the following line
        to it to form one "logical line"; if N consecutive lines end
        with a backslash, then N+1 physical lines will be joined to
        form one logical line.
      collapse_join [default: false]
        strip leading whitespace from lines that are joined to their
        predecessor; only matters if (join_lines and not lstrip_ws)
      errors [default: 'strict']
        error handler used to decode the file content

    Note that since 'rstrip_ws' can strip the trailing newline, the
    semantics of 'readline()' must differ from those of the builtin file
    object's 'readline()' method!  In particular, 'readline()' returns
    None for end-of-file: an empty string might just be a blank line (or
    an all-whitespace line), if 'rstrip_ws' is true but 'skip_blanks' is
    not."""

    default_options: ClassVar[dict[str, str | int]] = {
        'strip_comments': 1,
        'skip_blanks': 1,
        'lstrip_ws': 0,
        'rstrip_ws': 1,
        'join_lines': 0,
        'collapse_join': 0,
        'errors': 'strict',
    }

    def __init__(self, filename=None, file=None, **options):
        """Construct a new TextFile object.  At least one of 'filename'
        (a string) and 'file' (a file-like object) must be supplied.
        They keyword argument options are described above and affect
        the values returned by 'readline()'."""
        if filename is None and file is None:
            raise RuntimeError(
                "you must supply either or both of 'filename' and 'file'"
            )

        # set values for all options -- either from client option hash
        # or fallback to default_options
        for opt in self.default_options:
            if opt in options:
                setattr(self, opt, options[opt])
            else:
                setattr(self, opt, self.default_options[opt])

        # sanity check client option hash
        for opt in options:
            if opt not in self.default_options:
                raise KeyError(f"invalid TextFile option '{opt}'")

        if file is None:
            self.open(filename)
        else:
            self.filename = filename
            self.file = file
            self.current_line = 0  # assuming that file is at BOF!

        # 'linebuf' is a stack of lines that will be emptied before we
        # actually read from the file; it's only populated by an
        # 'unreadline()' operation
        self.linebuf = []

    def open(self, filename):
        """Open a new file named 'filename'.  This overrides both the
        'filename' and 'file' arguments to the constructor."""
        self.filename = filename
        self.file = open(  # noqa: SIM115 # handle owned by TextFile, closed in close()
            self.filename, errors=self.errors, encoding='utf-8'
        )
        self.current_line = 0

    def close(self):
        """Close the current file and forget everything we know about it
        (filename, current line number)."""
        file = self.file
        self.file = None
        self.filename = None
        self.current_line = None
        file.close()

    def gen_error(self, msg, line=None):
        outmsg = []
        if line is None:
            line = self.current_line
        outmsg.append(self.filename + ", ")
        if isinstance(line, (list, tuple)):
            outmsg.append("lines {}-{}: ".format(*line))
        else:
            outmsg.append(f"line {int(line)}: ")
        outmsg.append(str(msg))
        return "".join(outmsg)

    def error(self, msg, line=None):
        raise ValueError("error: " + self.gen_error(msg, line))

    def warn(self, msg, line=None):
        """Print (to stderr) a warning message tied to the current logical
        line in the current file.  If the current logical line in the
        file spans multiple physical lines, the warning refers to the
        whole range, eg. "lines 3-5".  If 'line' supplied, it overrides
        the current line number; it may be a list or tuple to indicate a
        range of physical lines, or an integer for a single physical
        line."""
        sys.stderr.write("warning: " + self.gen_error(msg, line) + "\n")

    def readline(self):  # noqa: C901
        """Read and return a single logical line from the current file (or
        from an internal buffer if lines have previously been "unread"
        with 'unreadline()').  If the 'join_lines' option is true, this
        may involve reading multiple physical lines concatenated into a
        single string.  Updates the current line number, so calling
        'warn()' after 'readline()' emits a warning about the physical
        line(s) just read.  Returns None on end-of-file, since the empty
        string can occur if 'rstrip_ws' is true but 'strip_blanks' is
        not."""
        # If any "unread" lines waiting in 'linebuf', return the top
        # one.  (We don't actually buffer read-ahead data -- lines only
        # get put in 'linebuf' if the client explicitly does an
        # 'unreadline()'.
        if self.linebuf:
            line = self.linebuf[-1]
            del self.linebuf[-1]
            return line

        buildup_line = ''

        while True:
            # read the line, make it None if EOF
            line = self.file.readline()
            if line == '':
                line = None

            if self.strip_comments and line:
                # Look for the first "#" in the line.  If none, never
                # mind.  If we find one and it's the first character, or
                # is not preceded by "\", then it starts a comment --
                # strip the comment, strip whitespace before it, and
                # carry on.  Otherwise, it's just an escaped "#", so
                # unescape it (and any other escaped "#"'s that might be
                # lurking in there) and otherwise leave the line alone.

                pos = line.find("#")
                if pos == -1:  # no "#" -- no comments
                    pass

                # It's definitely a comment -- either "#" is the first
                # character, or it's elsewhere and unescaped.
                elif pos == 0 or line[pos - 1] != "\\":
                    # Have to preserve the trailing newline, because it's
                    # the job of a later step (rstrip_ws) to remove it --
                    # and if rstrip_ws is false, we'd better preserve it!
                    # (NB. this means that if the final line is all comment
                    # and has no trailing newline, we will think that it's
                    # EOF; I think that's OK.)
                    eol = (line[-1] == '\n') and '\n' or ''
                    line = line[0:pos] + eol

                    # If all that's left is whitespace, then skip line
                    # *now*, before we try to join it to 'buildup_line' --
                    # that way constructs like
                    #   hello \\
                    #   # comment that should be ignored
                    #   there
                    # result in "hello there".
                    if line.strip() == "":
                        continue
                else:  # it's an escaped "#"
                    line = line.replace("\\#", "#")

            # did previous line end with a backslash? then accumulate
            if self.join_lines and buildup_line:
                # oops: end of file
                if line is None:
                    self.warn("continuation line immediately precedes end-of-file")
                    return buildup_line

                if self.collapse_join:
                    line = line.lstrip()
                line = buildup_line + line

                # careful: pay attention to line number when incrementing it
                if isinstance(self.current_line, list):
                    self.current_line[1] = self.current_line[1] + 1
                else:
                    self.current_line = [self.current_line, self.current_line + 1]
            # just an ordinary line, read it as usual
            else:
                if line is None:  # eof
                    return None

                # still have to be careful about incrementing the line number!
                if isinstance(self.current_line, list):
                    self.current_line = self.current_line[1] + 1
                else:
                    self.current_line = self.current_line + 1

            # strip whitespace however the client wants (leading and
            # trailing, or one or the other, or neither)
            if self.lstrip_ws and self.rstrip_ws:
                line = line.strip()
            elif self.lstrip_ws:
                line = line.lstrip()
            elif self.rstrip_ws:
                line = line.rstrip()

            # blank line (whether we rstrip'ed or not)? skip to next line
            # if appropriate
            if line in ('', '\n') and self.skip_blanks:
                continue

            if self.join_lines:
                if line[-1] == '\\':
                    buildup_line = line[:-1]
                    continue

                if line[-2:] == '\\\n':
                    buildup_line = line[0:-2] + '\n'
                    continue

            # well, I guess there's some actual content there: return it
            return line

    def readlines(self):
        """Read and return the list of all logical lines remaining in the
        current file."""
        lines = []
        while True:
            line = self.readline()
            if line is None:
                return lines
            lines.append(line)

    def unreadline(self, line):
        """Push 'line' (a string) onto an internal buffer that will be
        checked by future 'readline()' calls.  Handy for implementing
        a parser with line-at-a-time lookahead."""
        self.linebuf.append(line)
<|endoftext|>"""distutils.extension

Provides the Extension class, used to describe C/C++ extension
modules in setup scripts."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import field, fields

from ._dataclass import lenient_dataclass

# This class is really only used by the "build_ext" command, so it might
# make sense to put it in distutils.command.build_ext.  However, that
# module is already big enough, and I want to make this class a bit more
# complex to simplify some common cases ("foo" module in "foo.c") and do
# better error-checking ("foo.c" actually exists).
#
# Also, putting this in build_ext.py means every setup script would have to
# import that large-ish module (indirectly, through distutils.core) in
# order to do anything.


@lenient_dataclass()
class Extension:
    """Just a collection of attributes that describes an extension
    module and everything needed to build it (hopefully in a portable
    way, but there are hooks that let you be as unportable as you need).
    """

    name: str
    """
    the full name of the extension, including any packages -- ie.
    *not* a filename or pathname, but Python dotted name
    """

    sources: Iterable[str | os.PathLike[str]]
    """
    iterable of source filenames (except strings, which could be misinterpreted
    as a single filename), relative to the distribution root (where the setup
    script lives), in Unix form (slash-separated) for portability. Can be any
    non-string iterable (list, tuple, set, etc.) containing strings or
    PathLike objects. Source files may be C, C++, SWIG (.i), platform-specific
    resource files, or whatever else is recognized by the "build_ext" command
    as source for a Python extension.
    """

    include_dirs: list[str] = field(default_factory=list)
    """
    list of directories to search for C/C++ header files (in Unix
    form for portability)
    """

    define_macros: list[tuple[str, str | None]] = field(default_factory=list)
    """
    list of macros to define; each macro is defined using a 2-tuple,
    where 'value' is either the string to define it to or None to
    define it without a particular value (equivalent of "#define
    FOO" in source or -DFOO on Unix C compiler command line)
    """

    undef_macros: list[str] = field(default_factory=list)
    """list of macros to undefine explicitly"""

    library_dirs: list[str] = field(default_factory=list)
    """list of directories to search for C/C++ libraries at link time"""

    libraries: list[str] = field(default_factory=list)
    """list of library names (not filenames or paths) to link against"""

    runtime_library_dirs: list[str] = field(default_factory=list)
    """
    list of directories to search for C/C++ libraries at run time
    (for shared extensions, this is when the extension is loaded)
    """

    extra_objects: list[str] = field(default_factory=list)
    """
    list of extra files to link with (eg. object files not implied
    by 'sources', static library that must be explicitly specified,
    binary resource files, etc.)
    """

    extra_compile_args: list[str] = field(default_factory=list)
    """
    any extra platform- and compiler-specific information to use
    when compiling the source files in 'sources'.  For platforms and
    compilers where "command line" makes sense, this is typically a
    list of command-line arguments, but for other platforms it could
    be anything.
    """

    extra_link_args: list[str] = field(default_factory=list)
    """
    any extra platform- and compiler-specific information to use
    when linking object files together to create the extension (or
    to create a new static Python interpreter).  Similar
    interpretation as for 'extra_compile_args'.
    """

    export_symbols: list[str] = field(default_factory=list)
    """
    list of symbols to be exported from a shared extension.  Not
    used on all platforms, and not generally necessary for Python
    extensions, which typically export exactly one symbol: "init" +
    extension_name.
    """

    swig_opts: list[str] = field(default_factory=list)
    """
    any extra options to pass to SWIG if a source file has the .i
    extension.
    """

    depends: list[str] = field(default_factory=list)
    """list of files that the extension depends on"""

    language: str | None = None
    """
    extension language (i.e. "c", "c++", "objc"). Will be detected
    from the source extensions if not provided.
    """

    optional: bool = False
    """
    specifies that a build failure in the extension should not abort the
    build process, but simply not install the failing extension.
    """

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("'name' must be a string")

        # handle the string case first; since strings are iterable, disallow them
        if isinstance(self.sources, str):
            raise TypeError(
                "'sources' must be an iterable of strings or PathLike objects, not a string"
            )

        # now we check if it's iterable and contains valid types
        try:
            self.sources = list(map(os.fspath, self.sources))
        except TypeError:
            raise TypeError(
                "'sources' must be an iterable of strings or PathLike objects"
            )


_safe = tuple(f.name for f in fields(Extension))
"""Legal keyword arguments for the Extension constructor."""


def read_setup_file(filename):  # noqa: C901
    """Reads a Setup file and returns Extension instances."""
    from distutils.sysconfig import _variable_rx, expand_makefile_vars, parse_makefile
    from distutils.text_file import TextFile
    from distutils.util import split_quoted

    # First pass over the file to gather "VAR = VALUE" assignments.
    vars = parse_makefile(filename)

    # Second pass to gobble up the real content: lines of the form
    #   <module> ... [<sourcefile> ...] [<cpparg> ...] [<library> ...]
    file = TextFile(
        filename,
        strip_comments=True,
        skip_blanks=True,
        join_lines=True,
        lstrip_ws=True,
        rstrip_ws=True,
    )
    try:
        extensions = []

        while True:
            line = file.readline()
            if line is None:  # eof
                break
            if _variable_rx.match(line):  # VAR=VALUE, handled in first pass
                continue

            if line[0] == line[-1] == "*":
                file.warn(f"'{line}' lines not handled yet")
                continue

            line = expand_makefile_vars(line, vars)
            words = split_quoted(line)

            # NB. this parses a slightly different syntax than the old
            # makesetup script: here, there must be exactly one extension per
            # line, and it must be the first word of the line.  I have no idea
            # why the old syntax supported multiple extensions per line, as
            # they all wind up being the same.

            module = words[0]
            ext = Extension(module, [])
            append_next_word = None

            for word in words[1:]:
                if append_next_word is not None:
                    append_next_word.append(word)
                    append_next_word = None
                    continue

                suffix = os.path.splitext(word)[1]
                switch = word[0:2]
                value = word[2:]

                if suffix in (".c", ".cc", ".cpp", ".cxx", ".c++", ".m", ".mm"):
                    # hmm, should we do something about C vs. C++ sources?
                    # or leave it up to the CCompiler implementation to
                    # worry about?
                    ext.sources.append(word)
                elif switch == "-I":
                    ext.include_dirs.append(value)
                elif switch == "-D":
                    equals = value.find("=")
                    if equals == -1:  # bare "-DFOO" -- no value
                        ext.define_macros.append((value, None))
                    else:  # "-DFOO=blah"
                        ext.define_macros.append((value[0:equals], value[equals + 2 :]))
                elif switch == "-U":
                    ext.undef_macros.append(value)
                elif switch == "-C":  # only here 'cause makesetup has it!
                    ext.extra_compile_args.append(word)
                elif switch == "-l":
                    ext.libraries.append(value)
                elif switch == "-L":
                    ext.library_dirs.append(value)
                elif switch == "-R":
                    ext.runtime_library_dirs.append(value)
                elif word == "-rpath":
                    append_next_word = ext.runtime_library_dirs
                elif word == "-Xlinker":
                    append_next_word = ext.extra_link_args
                elif word == "-Xcompiler":
                    append_next_word = ext.extra_compile_args
                elif switch == "-u":
                    ext.extra_link_args.append(word)
                    if not value:
                        append_next_word = ext.extra_link_args
                elif suffix in (".a", ".so", ".sl", ".o", ".dylib"):
                    # NB. a really faithful emulation of makesetup would
                    # append a .o file to extra_objects only if it
                    # had a slash in it; otherwise, it would s/.o/.c/
                    # and append it to sources.  Hmmmm.
                    ext.extra_objects.append(word)
                else:
                    file.warn(f"unrecognized argument '{word}'")

            extensions.append(ext)
    finally:
        file.close()

    return extensions
<|endoftext|>"""
Exceptions used by the Distutils modules.

Distutils modules may raise these or standard exceptions,
including :exc:`SystemExit`.
"""

# compiler exceptions re-exported for backward compatibility; listed in
# __all__ below so they're recognized as intentional public re-exports.
from .compilers.C.errors import CompileError, LibError, LinkError, PreprocessError
from .compilers.errors import Error as _Error
from .compilers.errors import UnknownFileType as _UnknownFileType

CCompilerError = _Error
UnknownFileError = _UnknownFileType

__all__ = [
    'CCompilerError',
    'CompileError',
    'DistutilsArgError',
    'DistutilsByteCompileError',
    'DistutilsClassError',
    'DistutilsError',
    'DistutilsExecError',
    'DistutilsFileError',
    'DistutilsGetoptError',
    'DistutilsInternalError',
    'DistutilsModuleError',
    'DistutilsOptionError',
    'DistutilsPlatformError',
    'DistutilsSetupError',
    'DistutilsTemplateError',
    'LibError',
    'LinkError',
    'PreprocessError',
    'UnknownFileError',
]


class DistutilsError(Exception):
    """The root of all Distutils evil."""


class DistutilsModuleError(DistutilsError):
    """Unable to load an expected module, or to find an expected class
    within some module (in particular, command modules and classes)."""


class DistutilsClassError(DistutilsError):
    """Some command class (or possibly distribution class, if anyone
    feels a need to subclass Distribution) is found not to be holding
    up its end of the bargain, ie. implementing some part of the
    "command "interface."""


class DistutilsGetoptError(DistutilsError):
    """The option table provided to 'fancy_getopt()' is bogus."""


class DistutilsArgError(DistutilsError):
    """Raised by fancy_getopt in response to getopt.error -- ie. an
    error in the command line usage."""


class DistutilsFileError(DistutilsError):
    """Any problems in the filesystem: expected file not found, etc.
    Typically this is for problems that we detect before OSError
    could be raised."""


class DistutilsOptionError(DistutilsError):
    """Syntactic/semantic errors in command options, such as use of
    mutually conflicting options, or inconsistent options,
    badly-spelled values, etc.  No distinction is made between option
    values originating in the setup script, the command line, config
    files, or what-have-you -- but if we *know* something originated in
    the setup script, we'll raise DistutilsSetupError instead."""


class DistutilsSetupError(DistutilsError):
    """For errors that can be definitely blamed on the setup script,
    such as invalid keyword arguments to 'setup()'."""


class DistutilsPlatformError(DistutilsError):
    """We don't know how to do something on the current platform (but
    we do know how to do it on some platform) -- eg. trying to compile
    C files on a platform not supported by a CCompiler subclass."""


class DistutilsExecError(DistutilsError):
    """Any problems executing an external program (such as the C
    compiler, when compiling C files)."""


class DistutilsInternalError(DistutilsError):
    """Internal inconsistencies or impossibilities (obviously, this
    should never be seen if the code is working!)."""


class DistutilsTemplateError(DistutilsError):
    """Syntax error in a file list template."""


class DistutilsByteCompileError(DistutilsError):
    """Byte compile error."""
<|endoftext|>"""distutils.cmd

Provides the Command class, the base class for the command classes
in the distutils.command package.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from abc import abstractmethod
from collections.abc import Callable, MutableSequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast, overload

from . import _modified, archive_util, dir_util, file_util, util
from ._log import log
from .errors import DistutilsOptionError

if TYPE_CHECKING:
    # type-only import because of mutual dependence between these classes
    from distutils.dist import Distribution

    from typing_extensions import TypeVarTuple, Unpack

    _Ts = TypeVarTuple("_Ts")

_StrPathT = TypeVar("_StrPathT", bound="str | os.PathLike[str]")
_BytesPathT = TypeVar("_BytesPathT", bound="bytes | os.PathLike[bytes]")
_CommandT = TypeVar("_CommandT", bound="Command")


class Command:
    """Abstract base class for defining command classes, the "worker bees"
    of the Distutils.  A useful analogy for command classes is to think of
    them as subroutines with local variables called "options".  The options
    are "declared" in 'initialize_options()' and "defined" (given their
    final values, aka "finalized") in 'finalize_options()', both of which
    must be defined by every command class.  The distinction between the
    two is necessary because option values might come from the outside
    world (command line, config file, ...), and any options dependent on
    other options must be computed *after* these outside influences have
    been processed -- hence 'finalize_options()'.  The "body" of the
    subroutine, where it does all its work based on the values of its
    options, is the 'run()' method, which must also be implemented by every
    command class.
    """

    # 'sub_commands' formalizes the notion of a "family" of commands,
    # eg. "install" as the parent with sub-commands "install_lib",
    # "install_headers", etc.  The parent of a family of commands
    # defines 'sub_commands' as a class attribute; it's a list of
