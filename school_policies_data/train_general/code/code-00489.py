    # when implementing a real-world config command!)

    def check_func(
        self,
        func,
        headers=None,
        include_dirs=None,
        libraries=None,
        library_dirs=None,
        decl=False,
        call=False,
    ):
        """Determine if function 'func' is available by constructing a
        source file that refers to 'func', and compiles and links it.
        If everything succeeds, returns true; otherwise returns false.

        The constructed source file starts out by including the header
        files listed in 'headers'.  If 'decl' is true, it then declares
        'func' (as "int func()"); you probably shouldn't supply 'headers'
        and set 'decl' true in the same call, or you might get errors about
        a conflicting declarations for 'func'.  Finally, the constructed
        'main()' function either references 'func' or (if 'call' is true)
        calls it.  'libraries' and 'library_dirs' are used when
        linking.
        """
        self._check_compiler()
        body = []
        if decl:
            body.append(f"int {func} ();")
        body.append("int main () {")
        if call:
            body.append(f"  {func}();")
        else:
            body.append(f"  {func};")
        body.append("}")
        body = "\n".join(body) + "\n"

        return self.try_link(body, headers, include_dirs, libraries, library_dirs)

    def check_lib(
        self,
        library,
        library_dirs=None,
        headers=None,
        include_dirs=None,
        other_libraries: Sequence[str] = [],
    ):
        """Determine if 'library' is available to be linked against,
        without actually checking that any particular symbols are provided
        by it.  'headers' will be used in constructing the source file to
        be compiled, but the only effect of this is to check if all the
        header files listed are available.  Any libraries listed in
        'other_libraries' will be included in the link, in case 'library'
        has symbols that depend on other libraries.
        """
        self._check_compiler()
        return self.try_link(
            "int main (void) { }",
            headers,
            include_dirs,
            [library] + list(other_libraries),
            library_dirs,
        )

    def check_header(self, header, include_dirs=None, library_dirs=None, lang="c"):
        """Determine if the system header file named by 'header_file'
        exists and can be found by the preprocessor; return true if so,
        false otherwise.
        """
        return self.try_cpp(
            body="/* No body */", headers=[header], include_dirs=include_dirs
        )


def dump_file(filename, head=None):
    """Dumps a file content into log.info.

    If head is not None, will be dumped before the file content.
    """
    if head is None:
        log.info('%s', filename)
    else:
        log.info(head)
    log.info(pathlib.Path(filename).read_text(encoding='utf-8'))
<|endoftext|>"""
Backward compatibility for homebrew builds on macOS.
"""

import functools
import os
import subprocess
import sys
import sysconfig


@functools.lru_cache
def enabled():
    """
    Only enabled for Python 3.9 framework homebrew builds
    except ensurepip and venv.
    """
    PY39 = (3, 9) < sys.version_info < (3, 10)
    framework = sys.platform == 'darwin' and sys._framework
    homebrew = "Cellar" in sysconfig.get_config_var('projectbase')
    venv = sys.prefix != sys.base_prefix
    ensurepip = os.environ.get("ENSUREPIP_OPTIONS")
    return PY39 and framework and homebrew and not venv and not ensurepip


schemes = {
    'osx_framework_library': {
        'stdlib': '{installed_base}/{platlibdir}/python{py_version_short}',
        'platstdlib': '{platbase}/{platlibdir}/python{py_version_short}',
        'purelib': '{homebrew_prefix}/lib/python{py_version_short}/site-packages',
        'platlib': '{homebrew_prefix}/{platlibdir}/python{py_version_short}/site-packages',
        'include': '{installed_base}/include/python{py_version_short}{abiflags}',
        'platinclude': '{installed_platbase}/include/python{py_version_short}{abiflags}',
        'scripts': '{homebrew_prefix}/bin',
        'data': '{homebrew_prefix}',
    }
}


@functools.lru_cache
def vars():
    if not enabled():
        return {}
    homebrew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
    return locals()


def scheme(name):
    """
    Override the selected scheme for posix_prefix.
    """
    if not enabled() or not name.endswith('_prefix'):
        return name
    return 'osx_framework_library'
<|endoftext|>"""distutils.command.build

Implements the Distutils 'build' command."""

from __future__ import annotations

import os
import sys
import sysconfig
from collections.abc import Callable
from typing import Any, ClassVar

from ..ccompiler import show_compilers
from ..core import Command
from ..errors import DistutilsOptionError
from ..util import get_platform


class build(Command):
    description = "build everything needed to install"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('build-base=', 'b', "base directory for build library"),
        ('build-purelib=', None, "build directory for platform-neutral distributions"),
        ('build-platlib=', None, "build directory for platform-specific distributions"),
        (
            'build-lib=',
            None,
            "build directory for all distribution (defaults to either build-purelib or build-platlib",
        ),
        ('build-scripts=', None, "build directory for scripts"),
        ('build-temp=', 't', "temporary build directory"),
        (
            'plat-name=',
            'p',
            f"platform name to build for, if supported [default: {get_platform()}]",
        ),
        ('compiler=', 'c', "specify the compiler type"),
        ('parallel=', 'j', "number of parallel build jobs"),
        ('debug', 'g', "compile extensions and libraries with debugging information"),
        ('force', 'f', "forcibly build everything (ignore file timestamps)"),
        ('executable=', 'e', "specify final destination interpreter path (build.py)"),
    ]

    boolean_options: ClassVar[list[str]] = ['debug', 'force']

    help_options: ClassVar[list[tuple[str, str | None, str, Callable[[], object]]]] = [
        ('help-compiler', None, "list available compilers", show_compilers),
    ]

    def initialize_options(self):
        self.build_base = 'build'
        # these are decided only after 'build_base' has its final value
        # (unless overridden by the user or client)
        self.build_purelib = None
        self.build_platlib = None
        self.build_lib = None
        self.build_temp = None
        self.build_scripts = None
        self.compiler = None
        self.plat_name = None
        self.debug = None
        self.force = False
        self.executable = None
        self.parallel = None

    def finalize_options(self) -> None:  # noqa: C901
        if self.plat_name is None:
            self.plat_name = get_platform()
        else:
            # plat-name only supported for windows (other platforms are
            # supported via ./configure flags, if at all).  Avoid misleading
            # other platforms.
            if os.name != 'nt':
                raise DistutilsOptionError(
                    "--plat-name only supported on Windows (try "
                    "using './configure --help' on your platform)"
                )

        plat_specifier = f".{self.plat_name}-{sys.implementation.cache_tag}"

        # Python 3.13+ with --disable-gil shouldn't share build directories
        if sysconfig.get_config_var('Py_GIL_DISABLED'):
            plat_specifier += 't'

        # Make it so Python 2.x and Python 2.x with --with-pydebug don't
        # share the same build directories. Doing so confuses the build
        # process for C modules
        if hasattr(sys, 'gettotalrefcount'):
            plat_specifier += '-pydebug'

        # 'build_purelib' and 'build_platlib' just default to 'lib' and
        # 'lib.<plat>' under the base build directory.  We only use one of
        # them for a given distribution, though --
        if self.build_purelib is None:
            self.build_purelib = os.path.join(self.build_base, 'lib')
        if self.build_platlib is None:
            self.build_platlib = os.path.join(self.build_base, 'lib' + plat_specifier)

        # 'build_lib' is the actual directory that we will use for this
        # particular module distribution -- if user didn't supply it, pick
        # one of 'build_purelib' or 'build_platlib'.
        if self.build_lib is None:
            if self.distribution.has_ext_modules():
                self.build_lib = self.build_platlib
            else:
                self.build_lib = self.build_purelib

        # 'build_temp' -- temporary directory for compiler turds,
        # "build/temp.<plat>"
        if self.build_temp is None:
            self.build_temp = os.path.join(self.build_base, 'temp' + plat_specifier)
        if self.build_scripts is None:
            self.build_scripts = os.path.join(
                self.build_base,
                f'scripts-{sys.version_info.major}.{sys.version_info.minor}',
            )

        if self.executable is None and sys.executable:
            self.executable = os.path.normpath(sys.executable)

        if isinstance(self.parallel, str):
            try:
                self.parallel = int(self.parallel)
            except ValueError:
                raise DistutilsOptionError("parallel should be an integer")

    def run(self) -> None:
        # Run all relevant sub-commands.  This will be some subset of:
        #  - build_py      - pure Python modules
        #  - build_clib    - standalone C libraries
        #  - build_ext     - Python extensions
        #  - build_scripts - (Python) scripts
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

    # -- Predicates for the sub-command list ---------------------------

    def has_pure_modules(self) -> bool:
        return self.distribution.has_pure_modules()

    def has_c_libraries(self) -> bool:
        return self.distribution.has_c_libraries()

    def has_ext_modules(self) -> bool:
        return self.distribution.has_ext_modules()

    def has_scripts(self) -> bool:
        return self.distribution.has_scripts()

    sub_commands: ClassVar[list[tuple[str, Callable[[Any], bool] | None]]] = [
        ('build_py', has_pure_modules),
        ('build_clib', has_c_libraries),
        ('build_ext', has_ext_modules),
        ('build_scripts', has_scripts),
    ]
<|endoftext|>"""distutils.command.build_ext

Implements the Distutils 'build_ext' command, for building extension
modules (currently limited to C extensions, should accommodate C++
extensions ASAP)."""

from __future__ import annotations

import contextlib
import os
import re
import sys
from collections.abc import Callable
from distutils._log import log
from site import USER_BASE
from typing import ClassVar

from .._modified import newer_group
from ..ccompiler import CCompiler, new_compiler, show_compilers
from ..core import Command
from ..errors import (
    CCompilerError,
    CompileError,
    DistutilsError,
    DistutilsOptionError,
    DistutilsPlatformError,
    DistutilsSetupError,
)
from ..extension import Extension
from ..sysconfig import customize_compiler, get_config_h_filename, get_python_version
from ..util import get_platform, is_freethreaded, is_mingw

# An extension name is just a dot-separated list of Python NAMEs (ie.
# the same as a fully-qualified module name).
extension_name_re = re.compile(r'^[a-zA-Z_][a-zA-Z_0-9]*(\.[a-zA-Z_][a-zA-Z_0-9]*)*$')


class build_ext(Command):
    description = "build C/C++ extensions (compile/link to build directory)"

    # XXX thoughts on how to deal with complex command-line options like
    # these, i.e. how to make it so fancy_getopt can suck them off the
    # command line and make it look like setup.py defined the appropriate
    # lists of tuples of what-have-you.
    #   - each command needs a callback to process its command-line options
    #   - Command.__init__() needs access to its share of the whole
    #     command line (must ultimately come from
    #     Distribution.parse_command_line())
    #   - it then calls the current command class' option-parsing
    #     callback to deal with weird options like -D, which have to
    #     parse the option text and churn out some custom data
    #     structure
    #   - that data structure (in this case, a list of 2-tuples)
    #     will then be present in the command object by the time
    #     we get to finalize_options() (i.e. the constructor
    #     takes care of both command-line and client options
    #     in between initialize_options() and finalize_options())

    sep_by = f" (separated by '{os.pathsep}')"
    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('build-lib=', 'b', "directory for compiled extension modules"),
        ('build-temp=', 't', "directory for temporary files (build by-products)"),
        (
            'plat-name=',
            'p',
            (
                "platform name to cross-compile for, if supported "
                f"[default: {get_platform()}]"
            ),
        ),
        (
            'inplace',
            'i',
            (
                "ignore build-lib and put compiled extensions into the source "
                "directory alongside your pure Python modules"
            ),
        ),
        (
            'include-dirs=',
            'I',
            "list of directories to search for header files" + sep_by,
        ),
        ('define=', 'D', "C preprocessor macros to define"),
        ('undef=', 'U', "C preprocessor macros to undefine"),
        ('libraries=', 'l', "external C libraries to link with"),
        (
            'library-dirs=',
            'L',
            "directories to search for external C libraries" + sep_by,
        ),
        ('rpath=', 'R', "directories to search for shared C libraries at runtime"),
        ('link-objects=', 'O', "extra explicit link objects to include in the link"),
        ('debug', 'g', "compile/link with debugging information"),
        ('force', 'f', "forcibly build everything (ignore file timestamps)"),
        ('compiler=', 'c', "specify the compiler type"),
        ('parallel=', 'j', "number of parallel build jobs"),
        ('swig-cpp', None, "make SWIG create C++ files (default is C)"),
        ('swig-opts=', None, "list of SWIG command line options"),
        ('swig=', None, "path to the SWIG executable"),
        ('user', None, "add user include, library and rpath"),
    ]

    boolean_options: ClassVar[list[str]] = [
        'inplace',
        'debug',
        'force',
        'swig-cpp',
        'user',
    ]

    help_options: ClassVar[list[tuple[str, str | None, str, Callable[[], object]]]] = [
        ('help-compiler', None, "list available compilers", show_compilers),
    ]

    def initialize_options(self) -> None:
        self.extensions: list[Extension] | None = None
        self.build_lib: str = None  # type: ignore[assignment] # Should always be set in finalize_options by set_undefined_options from build Command
        self.plat_name: str = None  # type: ignore[assignment] # Should always be set in finalize_options by set_undefined_options from build Command
        self.build_temp: str = None  # type: ignore[assignment] # Should always be set in finalize_options by set_undefined_options from build Command
        self.inplace = False
        self.package: str | None = None

        self.include_dirs: list[str] = None  # type: ignore[assignment] # Should always be set in finalize_options
        self.define = None
        self.undef = None
        self.libraries: list[str] = None  # type: ignore[assignment] # Should always be set in finalize_options
        self.library_dirs: list[str] = None  # type: ignore[assignment] # Should always be set in finalize_options
        self.rpath: list[str] = None  # type: ignore[assignment] # Should always be set in finalize_options
        self.link_objects = None
        self.debug = None
        # Inherit ``Command.force``'s ``bool | None`` type; command-line
        # boolean options aren't coerced to ``bool``, so don't claim they are.
        self.force: bool | None = None
        # ``compiler`` holds three types across the command's lifecycle:
        # ``None`` initially, the compiler name (``str``) after
        # ``finalize_options`` (via ``set_undefined_options``), then a
        # ``CCompiler`` once ``run`` calls ``new_compiler``.
        # See https://github.com/pypa/distutils/pull/368#discussion_r3559726265
        self.compiler: str | CCompiler | None = None
        self.swig = None
        self.swig_cpp = None
        self.swig_opts: list[str] = None  # type: ignore[assignment] # Should always be set in finalize_options
        self.user = None
        self.parallel: int | None = None

    @staticmethod
    def _python_lib_dir(sysconfig):
        """
        Resolve Python's library directory for building extensions
        that rely on a shared Python library.

        See python/cpython#44264 and python/cpython#48686
        """
        if not sysconfig.get_config_var('Py_ENABLE_SHARED'):
            return

        if sysconfig.python_build:
            yield '.'
            return

        if sys.platform == 'zos':
            # On z/OS, a user is not required to install Python to
            # a predetermined path, but can use Python portably
            installed_dir = sysconfig.get_config_var('base')
            lib_dir = sysconfig.get_config_var('platlibdir')
            yield os.path.join(installed_dir, lib_dir)
        else:
            # building third party extensions
            yield sysconfig.get_config_var('LIBDIR')

    def finalize_options(self) -> None:  # noqa: C901
        from distutils import sysconfig

        self.set_undefined_options(
            'build',
            ('build_lib', 'build_lib'),
            ('build_temp', 'build_temp'),
            ('compiler', 'compiler'),
            ('debug', 'debug'),
            ('force', 'force'),
            ('parallel', 'parallel'),
            ('plat_name', 'plat_name'),
        )

        if self.package is None:
            self.package = self.distribution.ext_package

        self.extensions = self.distribution.ext_modules

        # Make sure Python's include directories (for Python.h, pyconfig.h,
        # etc.) are in the include search path.
        py_include = sysconfig.get_python_inc()
        plat_py_include = sysconfig.get_python_inc(plat_specific=True)
        if self.include_dirs is None:
            self.include_dirs = self.distribution.include_dirs or []
        if isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)

        # If in a virtualenv, add its include directory
        # Issue 16116
        if sys.exec_prefix != sys.base_exec_prefix:
            self.include_dirs.append(os.path.join(sys.exec_prefix, 'include'))

        # Put the Python "system" include dir at the end, so that
        # any local include dirs take precedence.
        self.include_dirs.extend(py_include.split(os.path.pathsep))
        if plat_py_include != py_include:
            self.include_dirs.extend(plat_py_include.split(os.path.pathsep))

        self.ensure_string_list('libraries')
        self.ensure_string_list('link_objects')

        # Life is easier if we're not forever checking for None, so
        # simplify these options to empty lists if unset
        if self.libraries is None:
            self.libraries = []
        if self.library_dirs is None:
            self.library_dirs = []
        elif isinstance(self.library_dirs, str):
            self.library_dirs = self.library_dirs.split(os.pathsep)

        if self.rpath is None:
            self.rpath = []
        elif isinstance(self.rpath, str):
            self.rpath = self.rpath.split(os.pathsep)

        # for extensions under windows use different directories
        # for Release and Debug builds.
        # also Python's library directory must be appended to library_dirs
        if os.name == 'nt' and not is_mingw():
            # the 'libs' directory is for binary installs - we assume that
            # must be the *native* platform.  But we don't really support
            # cross-compiling via a binary install anyway, so we let it go.
            self.library_dirs.append(os.path.join(sys.exec_prefix, 'libs'))
            if sys.base_exec_prefix != sys.prefix:  # Issue 16116
                self.library_dirs.append(os.path.join(sys.base_exec_prefix, 'libs'))
            if self.debug:
                self.build_temp = os.path.join(self.build_temp, "Debug")
            else:
                self.build_temp = os.path.join(self.build_temp, "Release")

            # Append the source distribution include and library directories,
            # this allows distutils on windows to work in the source tree
            self.include_dirs.append(os.path.dirname(get_config_h_filename()))
            self.library_dirs.append(sys.base_exec_prefix)

            # Use the .lib files for the correct architecture
            if self.plat_name == 'win32':
                suffix = 'win32'
            else:
                # win-amd64
                suffix = self.plat_name[4:]
            new_lib = os.path.join(sys.exec_prefix, 'PCbuild')
            if suffix:
                new_lib = os.path.join(new_lib, suffix)
            self.library_dirs.append(new_lib)

        # For extensions under Cygwin, Python's library directory must be
        # appended to library_dirs
        if sys.platform[:6] == 'cygwin':
            if not sysconfig.python_build:
                # building third party extensions
                self.library_dirs.append(
                    os.path.join(
                        sys.prefix, "lib", "python" + get_python_version(), "config"
                    )
                )
            else:
                # building python standard extensions
                self.library_dirs.append('.')

        self.library_dirs.extend(self._python_lib_dir(sysconfig))

        # The argument parsing will result in self.define being a string, but
        # it has to be a list of 2-tuples.  All the preprocessor symbols
        # specified by the 'define' option will be set to '1'.  Multiple
        # symbols can be separated with commas.

        if self.define:
            defines = self.define.split(',')
            self.define = [(symbol, '1') for symbol in defines]

        # The option for macros to undefine is also a string from the
        # option parsing, but has to be a list.  Multiple symbols can also
        # be separated with commas here.
        if self.undef:
            self.undef = self.undef.split(',')

        if self.swig_opts is None:
            self.swig_opts = []
        else:
            self.swig_opts = self.swig_opts.split(' ')

        # Finally add the user include and library directories if requested
        if self.user:
            user_include = os.path.join(USER_BASE, "include")
            user_lib = os.path.join(USER_BASE, "lib")
            if os.path.isdir(user_include):
                self.include_dirs.append(user_include)
            if os.path.isdir(user_lib):
                self.library_dirs.append(user_lib)
                self.rpath.append(user_lib)

        if isinstance(self.parallel, str):
            try:
                self.parallel = int(self.parallel)
            except ValueError:
                raise DistutilsOptionError("parallel should be an integer")

    def run(self) -> None:  # noqa: C901
        # 'self.extensions', as supplied by setup.py, is a list of
        # Extension instances.  See the documentation for Extension (in
        # distutils.extension) for details.
        #
        # For backwards compatibility with Distutils 0.8.2 and earlier, we
        # also allow the 'extensions' list to be a list of tuples:
        #    (ext_name, build_info)
        # where build_info is a dictionary containing everything that
        # Extension instances do except the name, with a few things being
        # differently named.  We convert these 2-tuples to Extension
        # instances as needed.

        if not self.extensions:
            return

        # If we were asked to build any C/C++ libraries, make sure that the
        # directory where we put them is in the library search path for
        # linking extensions.
        if self.distribution.has_c_libraries():
            build_clib = self.get_finalized_command('build_clib')
            self.libraries.extend(build_clib.get_library_names() or [])
            self.library_dirs.append(build_clib.build_clib)

        # Setup the CCompiler object that we'll use to do all the
        # compiling and linking
        self.compiler = new_compiler(
            compiler=self.compiler,
            verbose=self.verbose,
            force=self.force,
        )
        customize_compiler(self.compiler)
        # When cross-compiling, initialize the compiler now so it targets
        # plat_name rather than the host platform. For a native build,
        # initialization is left lazy, since some builds may rely on it
        # (e.g. to avoid requiring a compiler that never compiles).
        # See pypa/distutils#410.
        if self.plat_name != get_platform():
            self.compiler.initialize(self.plat_name)

        # The official Windows free threaded Python installer doesn't set
        # Py_GIL_DISABLED because its pyconfig.h is shared with the
        # default build, so define it here (pypa/setuptools#4662).
        if os.name == 'nt' and is_freethreaded():
            self.compiler.define_macro('Py_GIL_DISABLED', '1')

        # And make sure that any compile/link-related options (which might
        # come from the command-line or from the setup script) are set in
        # that CCompiler object -- that way, they automatically apply to
        # all compiling and linking done here.
        if self.include_dirs is not None:
            self.compiler.set_include_dirs(self.include_dirs)
        if self.define is not None:
            # 'define' option is a list of (name,value) tuples
            for name, value in self.define:
                self.compiler.define_macro(name, value)
        if self.undef is not None:
            for macro in self.undef:
                self.compiler.undefine_macro(macro)
        if self.libraries is not None:
            self.compiler.set_libraries(self.libraries)
        if self.library_dirs is not None:
            self.compiler.set_library_dirs(self.library_dirs)
        if self.rpath is not None:
            self.compiler.set_runtime_library_dirs(self.rpath)
        if self.link_objects is not None:
            self.compiler.set_link_objects(self.link_objects)

        # Now actually compile and link everything.
        self.build_extensions()

    def check_extensions_list(self, extensions) -> None:  # noqa: C901
        """Ensure that the list of extensions (presumably provided as a
        command option 'extensions') is valid, i.e. it is a list of
        Extension objects.  We also support the old-style list of 2-tuples,
        where the tuples are (ext_name, build_info), which are converted to
        Extension instances here.

        Raise DistutilsSetupError if the structure is invalid anywhere;
        just returns otherwise.
        """
        if not isinstance(extensions, list):
            raise DistutilsSetupError(
                "'ext_modules' option must be a list of Extension instances"
            )

        for i, ext in enumerate(extensions):
            if isinstance(ext, Extension):
                continue  # OK! (assume type-checking done
                # by Extension constructor)

            if not isinstance(ext, tuple) or len(ext) != 2:
                raise DistutilsSetupError(
                    "each element of 'ext_modules' option must be an "
                    "Extension instance or 2-tuple"
                )

            ext_name, build_info = ext

            log.warning(
                "old-style (ext_name, build_info) tuple found in "
                "ext_modules for extension '%s' "
                "-- please convert to Extension instance",
                ext_name,
            )

            if not (isinstance(ext_name, str) and extension_name_re.match(ext_name)):
                raise DistutilsSetupError(
                    "first element of each tuple in 'ext_modules' "
                    "must be the extension name (a string)"
                )

            if not isinstance(build_info, dict):
                raise DistutilsSetupError(
                    "second element of each tuple in 'ext_modules' "
                    "must be a dictionary (build info)"
                )

            # OK, the (ext_name, build_info) dict is type-safe: convert it
            # to an Extension instance.
            ext = Extension(ext_name, build_info['sources'])

            # Easy stuff: one-to-one mapping from dict elements to
            # instance attributes.
            for key in (
                'include_dirs',
                'library_dirs',
                'libraries',
                'extra_objects',
                'extra_compile_args',
                'extra_link_args',
            ):
                val = build_info.get(key)
                if val is not None:
                    setattr(ext, key, val)

            # Medium-easy stuff: same syntax/semantics, different names.
            ext.runtime_library_dirs = build_info.get('rpath') or []
            if 'def_file' in build_info:
                log.warning("'def_file' element of build info dict no longer supported")

            # Non-trivial stuff: 'macros' split into 'define_macros'
            # and 'undef_macros'.
            macros = build_info.get('macros')
            if macros:
                ext.define_macros = []
                ext.undef_macros = []
                for macro in macros:
                    if not (isinstance(macro, tuple) and len(macro) in (1, 2)):
                        raise DistutilsSetupError(
                            "'macros' element of build info dict must be 1- or 2-tuple"
                        )
                    if len(macro) == 1:
                        ext.undef_macros.append(macro[0])
                    elif len(macro) == 2:
                        ext.define_macros.append(macro)

            extensions[i] = ext

    def get_source_files(self):
        self.check_extensions_list(self.extensions)
        filenames = []

        # Wouldn't it be neat if we knew the names of header files too...
        for ext in self.extensions:
            filenames.extend(ext.sources)
        return filenames

    def get_outputs(self):
        # Sanity check the 'extensions' list -- can't assume this is being
        # done in the same run as a 'build_extensions()' call (in fact, we
        # can probably assume that it *isn't*!).
        self.check_extensions_list(self.extensions)

        # And build the list of output (built) filenames.  Note that this
        # ignores the 'inplace' flag, and assumes everything goes in the
        # "build" tree.
        return [self.get_ext_fullpath(ext.name) for ext in self.extensions]

    def build_extensions(self) -> None:
        # First, sanity-check the 'extensions' list
        self.check_extensions_list(self.extensions)
        if self.parallel:
            self._build_extensions_parallel()
        else:
            self._build_extensions_serial()

    def _build_extensions_parallel(self):
        workers = self.parallel
        if self.parallel is True:
            workers = os.cpu_count()  # may return None
        try:
            from concurrent.futures import ThreadPoolExecutor
        except ImportError:
            workers = None

        if workers is None:
            self._build_extensions_serial()
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self.build_extension, ext) for ext in self.extensions
            ]
            for ext, fut in zip(self.extensions, futures, strict=False):
                with self._filter_build_errors(ext):
                    fut.result()

    def _build_extensions_serial(self):
        for ext in self.extensions:
            with self._filter_build_errors(ext):
                self.build_extension(ext)

    @contextlib.contextmanager
    def _filter_build_errors(self, ext):
        try:
            yield
        except (CCompilerError, DistutilsError, CompileError) as e:
            if not ext.optional:
                raise
            self.warn(f'building extension "{ext.name}" failed: {e}')

    def build_extension(self, ext) -> None:
        sources = ext.sources
        if sources is None or not isinstance(sources, (list, tuple)):
            raise DistutilsSetupError(
                f"in 'ext_modules' option (extension '{ext.name}'), "
                "'sources' must be present and must be "
                "a list of source filenames"
            )
        # sort to make the resulting .so file build reproducible
        sources = sorted(sources)

        ext_path = self.get_ext_fullpath(ext.name)
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_path, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        # First, scan the sources for SWIG definition files (.i), run
        # SWIG on 'em to create .c files, and modify the sources list
        # accordingly.
        sources = self.swig_sources(sources, ext)

        # Next, compile the source code to object files.

        # XXX not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        extra_args = ext.extra_compile_args or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        # run() replaces the compiler name with a CCompiler instance before
        # extensions are built.
        # https://github.com/pypa/distutils/pull/368#discussion_r3559726265
        assert isinstance(self.compiler, CCompiler), (
            "run() must precede build_extension()"
        )
        objects = self.compiler.compile(
            sources,
            output_dir=self.build_temp,
            macros=macros,
            include_dirs=ext.include_dirs,
            debug=self.debug,
            extra_postargs=extra_args,
            depends=ext.depends,
        )

        # XXX outdated variable, kept here in case third-part code
        # needs it.
        self._built_objects = objects[:]

        # Now link the object files together into a "shared object" --
        # of course, first we have to figure out all the other things
        # that go into the mix.
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []

        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)

        self.compiler.link_shared_object(
            objects,
            ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=language,
        )

    def swig_sources(self, sources, extension):
        """Walk the list of source files in 'sources', looking for SWIG
        interface (.i) files.  Run SWIG on all that are found, and
        return a modified 'sources' list with SWIG source files replaced
        by the generated C (or C++) files.
        """
        new_sources = []
        swig_sources = []
        swig_targets = {}

        # XXX this drops generated C/C++ files into the source tree, which
        # is fine for developers who want to distribute the generated
        # source -- but there should be an option to put SWIG output in
        # the temp dir.

        if self.swig_cpp:
            log.warning("--swig-cpp is deprecated - use --swig-opts=-c++")

        if (
            self.swig_cpp
            or ('-c++' in self.swig_opts)
            or ('-c++' in extension.swig_opts)
        ):
            target_ext = '.cpp'
        else:
            target_ext = '.c'

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == ".i":  # SWIG interface file
                new_sources.append(base + '_wrap' + target_ext)
                swig_sources.append(source)
                swig_targets[source] = new_sources[-1]
            else:
                new_sources.append(source)

        if not swig_sources:
            return new_sources

        swig = self.swig or self.find_swig()
        swig_cmd = [swig, "-python"]
        swig_cmd.extend(self.swig_opts)
        if self.swig_cpp:
            swig_cmd.append("-c++")

        # Do not override commandline arguments
        if not self.swig_opts:
            swig_cmd.extend(extension.swig_opts)

        for source in swig_sources:
            target = swig_targets[source]
            log.info("swigging %s to %s", source, target)
            self.spawn(swig_cmd + ["-o", target, source])

        return new_sources

    def find_swig(self):
        """Return the name of the SWIG executable.  On Unix, this is
        just "swig" -- it should be in the PATH.  Tries a bit harder on
        Windows.
        """
        if os.name == "posix":
            return "swig"
        elif os.name == "nt":
            # Look for SWIG in its standard installation directory on
            # Windows (or so I presume!).  If we find it there, great;
            # if not, act like Unix and assume it's in the PATH.
            for vers in ("1.3", "1.2", "1.1"):
                fn = os.path.join(f"c:\\swig{vers}", "swig.exe")
                if os.path.isfile(fn):
                    return fn
            return "swig.exe"
        else:
            raise DistutilsPlatformError(
                f"I don't know how to find (much less run) SWIG on platform '{os.name}'"
            )

    # -- Name generators -----------------------------------------------
    # (extension names, filenames, whatever)
    def get_ext_fullpath(self, ext_name: str) -> str:
        """Returns the path of the filename for a given extension.

        The file is located in `build_lib` or directly in the package
        (inplace option).
        """
        fullname = self.get_ext_fullname(ext_name)
        modpath = fullname.split('.')
        filename = self.get_ext_filename(modpath[-1])

        if not self.inplace:
            # no further work needed
            # returning :
            #   build_dir/package/path/filename
            filename = os.path.join(*modpath[:-1] + [filename])
            return os.path.join(self.build_lib, filename)

        # the inplace option requires to find the package directory
        # using the build_py command for that
        package = '.'.join(modpath[0:-1])
        build_py = self.get_finalized_command('build_py')
        package_dir = os.path.abspath(build_py.get_package_dir(package))

        # returning
        #   package_dir/filename
        return os.path.join(package_dir, filename)

    def get_ext_fullname(self, ext_name: str) -> str:
        """Returns the fullname of a given extension name.

        Adds the `package.` prefix"""
        if self.package is None:
            return ext_name
        else:
            return self.package + '.' + ext_name

    def get_ext_filename(self, ext_name: str) -> str:
        r"""Convert the name of an extension (eg. "foo.bar") into the name
        of the file from which it will be loaded (eg. "foo/bar.so", or
        "foo\bar.pyd").
        """
        from ..sysconfig import get_config_var

        ext_path = ext_name.split('.')
        ext_suffix = get_config_var('EXT_SUFFIX')
        return os.path.join(*ext_path) + ext_suffix

    def get_export_symbols(self, ext: Extension) -> list[str]:
        """Return the list of symbols that a shared extension has to
        export.  This either uses 'ext.export_symbols' or, if it's not
        provided, "PyInit_" + module_name.  Only relevant on Windows, where
        the .pyd file (DLL) must export the module "PyInit_" function.
        """
        name = self._get_module_name_for_symbol(ext)
        try:
            # Unicode module name support as defined in PEP-489
            # https://peps.python.org/pep-0489/#export-hook-name
            name.encode('ascii')
        except UnicodeEncodeError:
            suffix = 'U_' + name.encode('punycode').replace(b'-', b'_').decode('ascii')
        else:
            suffix = "_" + name

        initfunc_name = "PyInit" + suffix
        if initfunc_name not in ext.export_symbols:
            ext.export_symbols.append(initfunc_name)
        return ext.export_symbols

    def _get_module_name_for_symbol(self, ext):
        # Package name should be used for `__init__` modules
        # https://github.com/python/cpython/issues/80074
        # https://github.com/pypa/setuptools/issues/4826
        parts = ext.name.split(".")
        if parts[-1] == "__init__" and len(parts) >= 2:
            return parts[-2]
        return parts[-1]

    def get_libraries(self, ext: Extension) -> list[str]:
        """Return the list of libraries to link against when building a
        shared extension.  On most platforms, this is just 'ext.libraries';
        on Windows, we add the Python library (eg. python20.dll).
        """
        # The python library is always needed on Windows.  For MSVC, this
        # is redundant, since the library is mentioned in a pragma in
        # pyconfig.h that MSVC groks.  The other Windows compilers all seem
        # to need it mentioned explicitly, though, so that's what we do.
        # Append '_d' to the python import library on debug builds.
        if sys.platform == "win32" and not is_mingw():
            from .._msvccompiler import MSVCCompiler

            if not isinstance(self.compiler, MSVCCompiler):
                pythonlib = (
                    f"python{sys.hexversion >> 24}{(sys.hexversion >> 16) & 0xFF}"
                    + 't' * is_freethreaded()
                    + '_d' * bool(self.debug)
                )
                # don't extend ext.libraries, it may be shared with other
                # extensions, it is a reference to the original list
                return ext.libraries + [pythonlib]
        else:
            # On Android only the main executable and LD_PRELOADs are considered
            # to be RTLD_GLOBAL, all the dependencies of the main executable
            # remain RTLD_LOCAL and so the shared libraries must be linked with
            # libpython when python is built with a shared python library (issue
            # bpo-21536).
            # On Cygwin (and if required, other POSIX-like platforms based on
            # Windows like MinGW) it is simply necessary that all symbols in
            # shared libraries are resolved at link time.
            from ..sysconfig import get_config_var

            link_libpython = False
            if get_config_var('Py_ENABLE_SHARED'):
                # A native build on an Android device or on Cygwin
                if (
                    hasattr(sys, 'getandroidapilevel')
                    or sys.platform == 'cygwin'
                    or is_mingw()
                ):
                    link_libpython = True
                elif '_PYTHON_HOST_PLATFORM' in os.environ:
                    # We are cross-compiling for one of the relevant platforms
                    if (
                        get_config_var('ANDROID_API_LEVEL') != 0
                        or get_config_var('MACHDEP') == 'cygwin'
                    ):
                        link_libpython = True

            if link_libpython:
                ldversion = get_config_var('LDVERSION')
                return ext.libraries + ['python' + ldversion]

        return ext.libraries
<|endoftext|>"""distutils.command.build_clib

Implements the Distutils 'build_clib' command, to build a C/C++ library
that is included in the module distribution and needed by an extension
module."""

# XXX this module has *lots* of code ripped-off quite transparently from
# build_ext.py -- not surprisingly really, as the work required to build
# a static library from a collection of C source files is not really all
# that different from what's required to build a shared object file from
# a collection of C source files.  Nevertheless, I haven't done the
# necessary refactoring to account for the overlap in code between the
# two modules, mainly because a number of subtle details changed in the
# cut 'n paste.  Sigh.
from __future__ import annotations

import os
from collections.abc import Callable
from distutils._log import log
from typing import ClassVar

from ..ccompiler import new_compiler, show_compilers
from ..core import Command
from ..errors import DistutilsSetupError
from ..sysconfig import customize_compiler


class build_clib(Command):
    description = "build C/C++ libraries used by Python extensions"

    user_options: ClassVar[list[tuple[str, str, str]]] = [
        ('build-clib=', 'b', "directory to build C/C++ libraries to"),
        ('build-temp=', 't', "directory to put temporary build by-products"),
        ('debug', 'g', "compile with debugging information"),
        ('force', 'f', "forcibly build everything (ignore file timestamps)"),
        ('compiler=', 'c', "specify the compiler type"),
    ]

    boolean_options: ClassVar[list[str]] = ['debug', 'force']

    help_options: ClassVar[list[tuple[str, str | None, str, Callable[[], object]]]] = [
        ('help-compiler', None, "list available compilers", show_compilers),
    ]

    def initialize_options(self):
        self.build_clib = None
        self.build_temp = None

        # List of libraries to build
        self.libraries = None

        # Compilation options for all libraries
        self.include_dirs = None
        self.define = None
        self.undef = None
        self.debug = None
        self.force = False
        self.compiler = None

    def finalize_options(self) -> None:
        # This might be confusing: both build-clib and build-temp default
        # to build-temp as defined by the "build" command.  This is because
        # I think that C libraries are really just temporary build
        # by-products, at least from the point of view of building Python
        # extensions -- but I want to keep my options open.
        self.set_undefined_options(
            'build',
            ('build_temp', 'build_clib'),
            ('build_temp', 'build_temp'),
            ('compiler', 'compiler'),
            ('debug', 'debug'),
            ('force', 'force'),
        )

        self.libraries = self.distribution.libraries
        if self.libraries:
            self.check_library_list(self.libraries)

        if self.include_dirs is None:
            self.include_dirs = self.distribution.include_dirs or []
        if isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)

        # XXX same as for build_ext -- what about 'self.define' and
        # 'self.undef' ?

    def run(self) -> None:
        if not self.libraries:
            return

        self.compiler = new_compiler(compiler=self.compiler, force=self.force)
        customize_compiler(self.compiler)

        if self.include_dirs is not None:
            self.compiler.set_include_dirs(self.include_dirs)
        if self.define is not None:
            # 'define' option is a list of (name,value) tuples
            for name, value in self.define:
                self.compiler.define_macro(name, value)
        if self.undef is not None:
            for macro in self.undef:
                self.compiler.undefine_macro(macro)

        self.build_libraries(self.libraries)

    def check_library_list(self, libraries) -> None:
        """Ensure that the list of libraries is valid.

        `library` is presumably provided as a command option 'libraries'.
        This method checks that it is a list of 2-tuples, where the tuples
        are (library_name, build_info_dict).

        Raise DistutilsSetupError if the structure is invalid anywhere;
        just returns otherwise.
        """
        if not isinstance(libraries, list):
            raise DistutilsSetupError("'libraries' option must be a list of tuples")

        for lib in libraries:
            if not isinstance(lib, tuple) and len(lib) != 2:
                raise DistutilsSetupError("each element of 'libraries' must a 2-tuple")

            name, build_info = lib

            if not isinstance(name, str):
                raise DistutilsSetupError(
                    "first element of each tuple in 'libraries' "
                    "must be a string (the library name)"
                )

            if '/' in name or (os.sep != '/' and os.sep in name):
                raise DistutilsSetupError(
                    f"bad library name '{lib[0]}': may not contain directory separators"
                )

            if not isinstance(build_info, dict):
                raise DistutilsSetupError(
                    "second element of each tuple in 'libraries' "
                    "must be a dictionary (build info)"
                )

    def get_library_names(self):
        # Assume the library list is valid -- 'check_library_list()' is
        # called from 'finalize_options()', so it should be!
        if not self.libraries:
            return None

        lib_names = []
        for lib_name, _build_info in self.libraries:
            lib_names.append(lib_name)
        return lib_names

    def get_source_files(self):
        self.check_library_list(self.libraries)
        filenames = []
        for lib_name, build_info in self.libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                    f"in 'libraries' option (library '{lib_name}'), "
                    "'sources' must be present and must be "
                    "a list of source filenames"
                )

            filenames.extend(sources)
        return filenames

    def build_libraries(self, libraries) -> None:
        for lib_name, build_info in libraries:
            sources = build_info.get('sources')
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                    f"in 'libraries' option (library '{lib_name}'), "
                    "'sources' must be present and must be "
                    "a list of source filenames"
                )
            sources = list(sources)

            log.info("building '%s' library", lib_name)

            # First, compile the source code to object files in the library
            # directory.  (This should probably change to putting object
            # files in a temporary build directory.)
            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')
            objects = self.compiler.compile(
                sources,
                output_dir=self.build_temp,
                macros=macros,
                include_dirs=include_dirs,
                debug=self.debug,
            )

            # Now "link" the object files together into a static library.
            # (On Unix at least, this isn't really linking -- it just
            # builds an archive.  Whatever.)
            self.compiler.create_static_lib(
                objects, lib_name, output_dir=self.build_clib, debug=self.debug
            )
<|endoftext|>"""distutils.command.install_data

Implements the Distutils 'install_data' command, for installing
platform-independent data files."""

# contributed by Bastian Kleineidam

from __future__ import annotations

import functools
import os
from collections.abc import Iterable
from typing import ClassVar

from ..core import Command
from ..util import change_root, convert_path


class install_data(Command):
    description = "install data files"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        (
            'install-dir=',
            'd',
            "base directory for installing data files [default: installation base dir]",
        ),
        ('root=', None, "install everything relative to this alternate root directory"),
        ('force', 'f', "force installation (overwrite existing files)"),
    ]

    boolean_options: ClassVar[list[str]] = ['force']

    def initialize_options(self):
        self.install_dir = None
        self.outfiles = []
        self.root = None
        self.force = False
        self.data_files = self.distribution.data_files
        self.warn_dir = True

    def finalize_options(self) -> None:
        self.set_undefined_options(
            'install',
            ('install_data', 'install_dir'),
            ('root', 'root'),
            ('force', 'force'),
        )

    def run(self) -> None:
        self.mkpath(self.install_dir)
        for f in self.data_files:
            self._copy(f)

    @functools.singledispatchmethod
    def _copy(self, f: tuple[str | os.PathLike, Iterable[str | os.PathLike]]):
        # it's a tuple with path to install to and a list of files
        dir = convert_path(f[0])
        if not os.path.isabs(dir):
            dir = os.path.join(self.install_dir, dir)
        elif self.root:
            dir = change_root(self.root, dir)
        self.mkpath(dir)

        if f[1] == []:
            # If there are no files listed, the user must be
            # trying to create an empty directory, so add the
            # directory to the list of output files.
            self.outfiles.append(dir)
        else:
            # Copy files, adding them to the list of output files.
            for data in f[1]:
                data = convert_path(data)
                (out, _) = self.copy_file(data, dir)
                self.outfiles.append(out)

    @_copy.register(str)
    @_copy.register(os.PathLike)
    def _(self, f: str | os.PathLike):
        # it's a simple file, so copy it
        f = convert_path(f)
        if self.warn_dir:
            self.warn(
                "setup script did not provide a directory for "
                f"'{f}' -- installing right in '{self.install_dir}'"
            )
        (out, _) = self.copy_file(f, self.install_dir)
        self.outfiles.append(out)

    def get_inputs(self):
        return self.data_files or []

    def get_outputs(self):
        return self.outfiles
<|endoftext|>"""distutils.command.install_lib

Implements the Distutils 'install_lib' command
(install all Python modules)."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, ClassVar

from ..core import Command
from ..errors import DistutilsOptionError

# Extension for Python source files.
PYTHON_SOURCE_EXTENSION = ".py"


class install_lib(Command):
    description = "install all Python modules (extensions and pure Python)"

    # The byte-compilation options are a tad confusing.  Here are the
    # possible scenarios:
    #   1) no compilation at all (--no-compile --no-optimize)
    #   2) compile .pyc only (--compile --no-optimize; default)
    #   3) compile .pyc and "opt-1" .pyc (--compile --optimize)
    #   4) compile "opt-1" .pyc only (--no-compile --optimize)
    #   5) compile .pyc and "opt-2" .pyc (--compile --optimize-more)
    #   6) compile "opt-2" .pyc only (--no-compile --optimize-more)
    #
    # The UI for this is two options, 'compile' and 'optimize'.
    # 'compile' is strictly boolean, and only decides whether to
    # generate .pyc files.  'optimize' is three-way (0, 1, or 2), and
    # decides both whether to generate .pyc files and what level of
    # optimization to use.

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('install-dir=', 'd', "directory to install to"),
        ('build-dir=', 'b', "build directory (where to install from)"),
        ('force', 'f', "force installation (overwrite existing files)"),
        ('compile', 'c', "compile .py to .pyc [default]"),
        ('no-compile', None, "don't compile .py files"),
        (
            'optimize=',
            'O',
            (
                "also compile with optimization: -O1 for \"python -O\", "
                "-O2 for \"python -OO\", and -O0 to disable [default: -O0]"
            ),
        ),
        ('skip-build', None, "skip the build steps"),
    ]

    boolean_options: ClassVar[list[str]] = ['force', 'compile', 'skip-build']
    negative_opt: ClassVar[dict[str, str]] = {'no-compile': 'compile'}

    def initialize_options(self):
        # let the 'install' command dictate our installation directory
        self.install_dir = None
        self.build_dir = None
        self.force = False
        self.compile = None
        self.optimize = None
        self.skip_build = None

    def finalize_options(self) -> None:
        # Get all the information we need to install pure Python modules
        # from the umbrella 'install' command -- build (source) directory,
        # install (target) directory, and whether to compile .py files.
        self.set_undefined_options(
            'install',
            ('build_lib', 'build_dir'),
            ('install_lib', 'install_dir'),
            ('force', 'force'),
            ('compile', 'compile'),
            ('optimize', 'optimize'),
            ('skip_build', 'skip_build'),
        )

        if self.compile is None:
            self.compile = True
        if self.optimize is None:
            self.optimize = False

        if not isinstance(self.optimize, int):
            try:
                self.optimize = int(self.optimize)
            except ValueError:
                pass
            if self.optimize not in (0, 1, 2):
                raise DistutilsOptionError("optimize must be 0, 1, or 2")

    def run(self) -> None:
        # Make sure we have built everything we need first
        self.build()

        # Install everything: simply dump the entire contents of the build
        # directory to the installation directory (that's the beauty of
        # having a build directory!)
        outfiles = self.install()

        # (Optionally) compile .py to .pyc
        if outfiles is not None and self.distribution.has_pure_modules():
            self.byte_compile(outfiles)

    # -- Top-level worker functions ------------------------------------
    # (called from 'run()')

    def build(self) -> None:
        if not self.skip_build:
            if self.distribution.has_pure_modules():
                self.run_command('build_py')
            if self.distribution.has_ext_modules():
                self.run_command('build_ext')

    # Any: https://typing.readthedocs.io/en/latest/guides/writing_stubs.html#the-any-trick
    def install(self) -> list[str] | Any:
        if os.path.isdir(self.build_dir):
            outfiles = self.copy_tree(self.build_dir, self.install_dir)
        else:
            self.warn(
                f"'{self.build_dir}' does not exist -- no Python modules to install"
            )
            return
        return outfiles

    def byte_compile(self, files) -> None:
        if sys.dont_write_bytecode:
            self.warn('byte-compiling is disabled, skipping.')
            return

        from ..util import byte_compile

        # Get the "--root" directory supplied to the "install" command,
        # and use it as a prefix to strip off the purported filename
        # encoded in bytecode files.  This is far from complete, but it
        # should at least generate usable bytecode in RPM distributions.
        install_root = self.get_finalized_command('install').root

        if self.compile:
            byte_compile(
                files,
                optimize=0,
                force=self.force,
                prefix=install_root,
            )
        if self.optimize > 0:
            byte_compile(
                files,
                optimize=self.optimize,
                force=self.force,
                prefix=install_root,
                verbose=self.verbose,
            )

    # -- Utility methods -----------------------------------------------

    def _mutate_outputs(self, has_any, build_cmd, cmd_option, output_dir):
        if not has_any:
            return []

        build_cmd = self.get_finalized_command(build_cmd)
        build_files = build_cmd.get_outputs()
        build_dir = getattr(build_cmd, cmd_option)

        prefix_len = len(build_dir) + len(os.sep)
        outputs = [os.path.join(output_dir, file[prefix_len:]) for file in build_files]

        return outputs

    def _bytecode_filenames(self, py_filenames):
        bytecode_files = []
        for py_file in py_filenames:
            # Since build_py handles package data installation, the
            # list of outputs can contain more than just .py files.
            # Make sure we only report bytecode for the .py files.
            ext = os.path.splitext(os.path.normcase(py_file))[1]
            if ext != PYTHON_SOURCE_EXTENSION:
                continue
            if self.compile:
                bytecode_files.append(
                    importlib.util.cache_from_source(py_file, optimization='')
                )
            if self.optimize > 0:
                bytecode_files.append(
                    importlib.util.cache_from_source(
                        py_file, optimization=self.optimize
                    )
                )

        return bytecode_files

    # -- External interface --------------------------------------------
    # (called by outsiders)

    def get_outputs(self):
        """Return the list of files that would be installed if this command
        were actually run.  Not affected by the "dry-run" flag or whether
        modules have actually been built yet.
        """
        pure_outputs = self._mutate_outputs(
            self.distribution.has_pure_modules(),
            'build_py',
            'build_lib',
            self.install_dir,
        )
        if self.compile:
            bytecode_outputs = self._bytecode_filenames(pure_outputs)
        else:
            bytecode_outputs = []

        ext_outputs = self._mutate_outputs(
            self.distribution.has_ext_modules(),
            'build_ext',
            'build_lib',
            self.install_dir,
        )

        return pure_outputs + bytecode_outputs + ext_outputs

    def get_inputs(self):
        """Get the list of files that are input to this command, ie. the
        files that get installed as they are named in the build tree.
        The files in this list correspond one-to-one to the output
        filenames returned by 'get_outputs()'.
        """
        inputs = []

        if self.distribution.has_pure_modules():
            build_py = self.get_finalized_command('build_py')
            inputs.extend(build_py.get_outputs())

        if self.distribution.has_ext_modules():
            build_ext = self.get_finalized_command('build_ext')
            inputs.extend(build_ext.get_outputs())

        return inputs
<|endoftext|>"""distutils.command.bdist_dumb

Implements the Distutils 'bdist_dumb' command (create a "dumb" built
distribution -- i.e., just an archive to be unpacked under $prefix or
$exec_prefix)."""

import os
from distutils._log import log
from typing import ClassVar

from ..core import Command
from ..dir_util import ensure_relative, remove_tree
from ..errors import DistutilsPlatformError
from ..sysconfig import get_python_version
from ..util import get_platform


class bdist_dumb(Command):
    description = "create a \"dumb\" built distribution"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('bdist-dir=', 'd', "temporary directory for creating the distribution"),
        (
            'plat-name=',
            'p',
            (
                "platform name to embed in generated filenames "
                f"[default: {get_platform()}]"
            ),
        ),
        (
            'format=',
            'f',
            "archive format to create (tar, gztar, bztar, xztar, ztar, zip)",
        ),
        (
            'keep-temp',
            'k',
            "keep the pseudo-installation tree around after creating the distribution archive",
        ),
        ('dist-dir=', 'd', "directory to put final built distributions in"),
        ('skip-build', None, "skip rebuilding everything (for testing/debugging)"),
        (
            'relative',
            None,
            "build the archive using relative paths [default: false]",
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

    boolean_options: ClassVar[list[str]] = ['keep-temp', 'skip-build', 'relative']

    default_format: ClassVar[dict[str, str]] = {'posix': 'gztar', 'nt': 'zip'}

    def initialize_options(self):
        self.bdist_dir = None
        self.plat_name = None
        self.format = None
        self.keep_temp = False
        self.dist_dir = None
        self.skip_build = None
        self.relative = False
        self.owner = None
        self.group = None

    def finalize_options(self):
        if self.bdist_dir is None:
            bdist_base = self.get_finalized_command('bdist').bdist_base
            self.bdist_dir = os.path.join(bdist_base, 'dumb')

        if self.format is None:
            try:
                self.format = self.default_format[os.name]
            except KeyError:
                raise DistutilsPlatformError(
                    "don't know how to create dumb built distributions "
                    f"on platform {os.name}"
                )

        self.set_undefined_options(
            'bdist',
            ('dist_dir', 'dist_dir'),
            ('plat_name', 'plat_name'),
            ('skip_build', 'skip_build'),
        )

    def run(self):
        if not self.skip_build:
            self.run_command('build')

        install = self.reinitialize_command('install', reinit_subcommands=True)
        install.root = self.bdist_dir
        install.skip_build = self.skip_build
        install.warn_dir = False

        log.info("installing to %s", self.bdist_dir)
        self.run_command('install')

        # And make an archive relative to the root of the
        # pseudo-installation tree.
        archive_basename = f"{self.distribution.get_fullname()}.{self.plat_name}"

        pseudoinstall_root = os.path.join(self.dist_dir, archive_basename)
        if not self.relative:
            archive_root = self.bdist_dir
        else:
            if self.distribution.has_ext_modules() and (
                install.install_base != install.install_platbase
            ):
                raise DistutilsPlatformError(
                    "can't make a dumb built distribution where "
                    f"base and platbase are different ({install.install_base!r}, {install.install_platbase!r})"
                )
            else:
                archive_root = os.path.join(
                    self.bdist_dir, ensure_relative(install.install_base)
                )

        # Make the archive
        filename = self.make_archive(
            pseudoinstall_root,
            self.format,
            root_dir=archive_root,
            owner=self.owner,
            group=self.group,
        )
        if self.distribution.has_ext_modules():
            pyversion = get_python_version()
        else:
            pyversion = 'any'
        self.distribution.dist_files.append(('bdist_dumb', pyversion, filename))

        if not self.keep_temp:
            remove_tree(self.bdist_dir)
<|endoftext|>"""
distutils.command.install_egg_info

Implements the Distutils 'install_egg_info' command, for installing
a package's PKG-INFO metadata.
"""

import os
import re
import sys
from typing import ClassVar

from .. import dir_util
from .._log import log
from ..cmd import Command


class install_egg_info(Command):
    """Install an .egg-info file for the package"""

    description = "Install package's PKG-INFO metadata as an .egg-info file"
    user_options: ClassVar[list[tuple[str, str, str]]] = [
        ('install-dir=', 'd', "directory to install to"),
    ]

    def initialize_options(self):
        self.install_dir = None

    @property
    def basename(self):
        """
        Allow basename to be overridden by child class.
        Ref pypa/distutils#2.
        """
        name = to_filename(safe_name(self.distribution.get_name()))
        version = to_filename(safe_version(self.distribution.get_version()))
        return f"{name}-{version}-py{sys.version_info.major}.{sys.version_info.minor}.egg-info"

    def finalize_options(self):
        self.set_undefined_options('install_lib', ('install_dir', 'install_dir'))
        self.target = os.path.join(self.install_dir, self.basename)
        self.outputs = [self.target]

    def run(self):
        target = self.target
        if os.path.isdir(target) and not os.path.islink(target):
            dir_util.remove_tree(target)
        elif os.path.exists(target):
            self.execute(os.unlink, (self.target,), "Removing " + target)
        elif not os.path.isdir(self.install_dir):
            self.execute(
                os.makedirs, (self.install_dir,), "Creating " + self.install_dir
            )
        log.info("Writing %s", target)
        with open(target, 'w', encoding='UTF-8') as f:
            self.distribution.metadata.write_pkg_file(f)

    def get_outputs(self):
        return self.outputs


# The following routines were originally copied from setuptools' pkg_resources
# module and intended to be replaced by stdlib versions. They're now just legacy
# cruft.


def safe_name(name):
    """Convert an arbitrary string to a standard distribution name

    Any runs of non-alphanumeric/. characters are replaced with a single '-'.
    """
    return re.sub('[^A-Za-z0-9.]+', '-', name)


def safe_version(version):
    """Convert an arbitrary string to a standard version string

    Spaces become dots, and all other non-alphanumeric characters become
    dashes, with runs of multiple dashes condensed to a single dash.
    """
    version = version.replace(' ', '.')
    return re.sub('[^A-Za-z0-9.]+', '-', version)


def to_filename(name):
    """Convert a project or version name to its filename-escaped form

    Any '-' characters are currently replaced with '_'.
    """
    return name.replace('-', '_')
<|endoftext|>"""distutils.command.install_headers

Implements the Distutils 'install_headers' command, to install C/C++ header
files to the Python include directory."""

from typing import ClassVar

from ..core import Command


# XXX force is never used
class install_headers(Command):
    description = "install C/C++ header files"

    user_options: ClassVar[list[tuple[str, str, str]]] = [
        ('install-dir=', 'd', "directory to install header files to"),
        ('force', 'f', "force installation (overwrite existing files)"),
    ]

    boolean_options: ClassVar[list[str]] = ['force']

    def initialize_options(self):
        self.install_dir = None
        self.force = False
        self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options(
            'install', ('install_headers', 'install_dir'), ('force', 'force')
        )

    def run(self):
        headers = self.distribution.headers
        if not headers:
            return

        self.mkpath(self.install_dir)
        for header in headers:
            (out, _) = self.copy_file(header, self.install_dir)
            self.outfiles.append(out)

    def get_inputs(self):
        return self.distribution.headers or []

    def get_outputs(self):
        return self.outfiles
<|endoftext|>"""distutils.command.build_py

Implements the Distutils 'build_py' command."""

import glob
import importlib.util
import os
import sys
from distutils._log import log
from typing import ClassVar

from ..core import Command
from ..errors import DistutilsFileError, DistutilsOptionError
from ..util import convert_path


class build_py(Command):
    description = "\"build\" pure Python modules (copy to build directory)"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('build-lib=', 'd', "directory to \"build\" (copy) to"),
        ('compile', 'c', "compile .py to .pyc"),
        ('no-compile', None, "don't compile .py files [default]"),
        (
            'optimize=',
            'O',
            (
                "also compile with optimization: -O1 for \"python -O\", "
                "-O2 for \"python -OO\", and -O0 to disable [default: -O0]"
            ),
        ),
        ('force', 'f', "forcibly build everything (ignore file timestamps)"),
    ]

    boolean_options: ClassVar[list[str]] = ['compile', 'force']
    negative_opt: ClassVar[dict[str, str]] = {'no-compile': 'compile'}

    def initialize_options(self):
        self.build_lib = None
        self.py_modules = None
        self.package = None
        self.package_data = None
        self.package_dir = None
        self.compile = False
        self.optimize = 0
        self.force = None

    def finalize_options(self) -> None:
        self.set_undefined_options(
            'build', ('build_lib', 'build_lib'), ('force', 'force')
        )

        # Get the distribution options that are aliases for build_py
        # options -- list of packages and list of modules.
        self.packages = self.distribution.packages
        self.py_modules = self.distribution.py_modules
        self.package_data = self.distribution.package_data
        self.package_dir = {}
        if self.distribution.package_dir:
            for name, path in self.distribution.package_dir.items():
                self.package_dir[name] = convert_path(path)
        self.data_files = self.get_data_files()

        # Ick, copied straight from install_lib.py (fancy_getopt needs a
        # type system!  Hell, *everything* needs a type system!!!)
        if not isinstance(self.optimize, int):
            try:
                self.optimize = int(self.optimize)
                assert 0 <= self.optimize <= 2
            except (ValueError, AssertionError):
                raise DistutilsOptionError("optimize must be 0, 1, or 2")

    def run(self) -> None:
        # XXX copy_file by default preserves atime and mtime.  IMHO this is
        # the right thing to do, but perhaps it should be an option -- in
        # particular, a site administrator might want installed files to
        # reflect the time of installation rather than the last
        # modification time before the installed release.

        # XXX copy_file by default preserves mode, which appears to be the
        # wrong thing to do: if a file is read-only in the working
        # directory, we want it to be installed read/write so that the next
        # installation of the same module distribution can overwrite it
        # without problems.  (This might be a Unix-specific issue.)  Thus
        # we turn off 'preserve_mode' when copying to the build directory,
        # since the build directory is supposed to be exactly what the
        # installation will look like (ie. we preserve mode when
        # installing).

        # Two options control which modules will be installed: 'packages'
        # and 'py_modules'.  The former lets us work with whole packages, not
        # specifying individual modules at all; the latter is for
        # specifying modules one-at-a-time.

        if self.py_modules:
            self.build_modules()
        if self.packages:
            self.build_packages()
            self.build_package_data()

        self.byte_compile(self.get_outputs(include_bytecode=False))

    def get_data_files(self):
        """Generate list of '(package,src_dir,build_dir,filenames)' tuples"""
        data = []
        if not self.packages:
            return data
        for package in self.packages:
            # Locate package source directory
            src_dir = self.get_package_dir(package)

            # Compute package build directory
            build_dir = os.path.join(*([self.build_lib] + package.split('.')))

            # Length of path to strip from found files
            plen = 0
            if src_dir:
                plen = len(src_dir) + 1

            # Strip directory from globbed filenames
            filenames = [file[plen:] for file in self.find_data_files(package, src_dir)]
            data.append((package, src_dir, build_dir, filenames))
        return data

    def find_data_files(self, package, src_dir):
        """Return filenames for package's data files in 'src_dir'"""
        globs = self.package_data.get('', []) + self.package_data.get(package, [])
        files = []
        for pattern in globs:
            # Each pattern has to be converted to a platform-specific path
            filelist = glob.glob(
                os.path.join(glob.escape(src_dir), convert_path(pattern))
            )
            # Files that match more than one pattern are only added once
            files.extend([
                fn for fn in filelist if fn not in files and os.path.isfile(fn)
            ])
        return files

    def build_package_data(self) -> None:
        """Copy data files into build directory"""
        for _package, src_dir, build_dir, filenames in self.data_files:
            for filename in filenames:
                target = os.path.join(build_dir, filename)
                self.mkpath(os.path.dirname(target))
                self.copy_file(
                    os.path.join(src_dir, filename), target, preserve_mode=False
                )

    def get_package_dir(self, package):
        """Return the directory, relative to the top of the source
        distribution, where package 'package' should be found
        (at least according to the 'package_dir' option, if any)."""
        path = package.split('.')

        if not self.package_dir:
            if path:
                return os.path.join(*path)
            else:
                return ''
        else:
            tail = []
            while path:
                try:
                    pdir = self.package_dir['.'.join(path)]
                except KeyError:
                    tail.insert(0, path[-1])
                    del path[-1]
                else:
                    tail.insert(0, pdir)
                    return os.path.join(*tail)
            # Oops, got all the way through 'path' without finding a
            # match in package_dir.  If package_dir defines a directory
            # for the root (nameless) package, then fallback on it;
            # otherwise, we might as well have not consulted
            # package_dir at all, as we just use the directory implied
            # by 'tail' (which should be the same as the original value
            # of 'path' at this point).
            pdir = self.package_dir.get('')
            if pdir is not None:
                tail.insert(0, pdir)

            if tail:
                return os.path.join(*tail)
            else:
                return ''

    def check_package(self, package, package_dir):
        # Empty dir name means current directory, which we can probably
        # assume exists.  Also, os.path.exists and isdir don't know about
        # my "empty string means current dir" convention, so we have to
        # circumvent them.
        if package_dir != "":
            if not os.path.exists(package_dir):
                raise DistutilsFileError(
                    f"package directory '{package_dir}' does not exist"
                )
            if not os.path.isdir(package_dir):
                raise DistutilsFileError(
                    f"supposed package directory '{package_dir}' exists, "
                    "but is not a directory"
                )

        # Directories without __init__.py are namespace packages (PEP 420).
        if package:
            init_py = os.path.join(package_dir, "__init__.py")
            if os.path.isfile(init_py):
                return init_py

        # Either not in a package at all (__init__.py not expected), or
        # __init__.py doesn't exist -- so don't return the filename.
        return None

    def check_module(self, module, module_file):
        if not os.path.isfile(module_file):
            log.warning("file %s (for module %s) not found", module_file, module)
            return False
        else:
            return True

    def find_package_modules(self, package, package_dir):
        self.check_package(package, package_dir)
        module_files = glob.glob(os.path.join(glob.escape(package_dir), "*.py"))
        modules = []
        setup_script = os.path.abspath(self.distribution.script_name)

        for f in module_files:
            abs_f = os.path.abspath(f)
            if abs_f != setup_script:
                module = os.path.splitext(os.path.basename(f))[0]
                modules.append((package, module, f))
            else:
                self.debug_print(f"excluding {setup_script}")
        return modules

    def find_modules(self):
        """Finds individually-specified Python modules, ie. those listed by
        module name in 'self.py_modules'.  Returns a list of tuples (package,
        module_base, filename): 'package' is a tuple of the path through
        package-space to the module; 'module_base' is the bare (no
        packages, no dots) module name, and 'filename' is the path to the
        ".py" file (relative to the distribution root) that implements the
        module.
        """
        # Map package names to tuples of useful info about the package:
        #    (package_dir, checked)
        # package_dir - the directory where we'll find source files for
        #   this package
        # checked - true if we have checked that the package directory
        #   is valid (exists, contains __init__.py, ... ?)
        packages = {}

        # List of (package, module, filename) tuples to return
        modules = []

        # We treat modules-in-packages almost the same as toplevel modules,
        # just the "package" for a toplevel is empty (either an empty
        # string or empty list, depending on context).  Differences:
        #   - don't check for __init__.py in directory for empty package
        for module in self.py_modules:
            path = module.split('.')
            package = '.'.join(path[0:-1])
            module_base = path[-1]

            try:
                (package_dir, checked) = packages[package]
            except KeyError:
                package_dir = self.get_package_dir(package)
                checked = False

            if not checked:
                init_py = self.check_package(package, package_dir)
                packages[package] = (package_dir, 1)
                if init_py:
                    modules.append((package, "__init__", init_py))

            # XXX perhaps we should also check for just .pyc files
            # (so greedy closed-source bastards can distribute Python
            # modules too)
            module_file = os.path.join(package_dir, module_base + ".py")
            if not self.check_module(module, module_file):
                continue

            modules.append((package, module_base, module_file))

        return modules

    def find_all_modules(self):
        """Compute the list of all modules that will be built, whether
        they are specified one-module-at-a-time ('self.py_modules') or
        by whole packages ('self.packages').  Return a list of tuples
        (package, module, module_file), just like 'find_modules()' and
        'find_package_modules()' do."""
        modules = []
        if self.py_modules:
            modules.extend(self.find_modules())
        if self.packages:
            for package in self.packages:
                package_dir = self.get_package_dir(package)
                m = self.find_package_modules(package, package_dir)
                modules.extend(m)
        return modules

    def get_source_files(self):
        return [module[-1] for module in self.find_all_modules()]

    def get_module_outfile(self, build_dir, package, module):
        outfile_path = [build_dir] + list(package) + [module + ".py"]
        return os.path.join(*outfile_path)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        modules = self.find_all_modules()
        outputs = []
        for package, module, _module_file in modules:
            package = package.split('.')
            filename = self.get_module_outfile(self.build_lib, package, module)
            outputs.append(filename)
            if include_bytecode:
                if self.compile:
                    outputs.append(
                        importlib.util.cache_from_source(filename, optimization='')
                    )
                if self.optimize > 0:
                    outputs.append(
                        importlib.util.cache_from_source(
                            filename, optimization=self.optimize
                        )
                    )

        outputs += [
            os.path.join(build_dir, filename)
            for package, src_dir, build_dir, filenames in self.data_files
            for filename in filenames
        ]

        return outputs

    def build_module(self, module, module_file, package):
        if isinstance(package, str):
            package = package.split('.')
        elif not isinstance(package, (list, tuple)):
            raise TypeError(
                "'package' must be a string (dot-separated), list, or tuple"
            )

        # Now put the module source file into the "build" area -- this is
        # easy, we just copy it somewhere under self.build_lib (the build
        # directory for Python source).
        outfile = self.get_module_outfile(self.build_lib, package, module)
        dir = os.path.dirname(outfile)
        self.mkpath(dir)
        return self.copy_file(module_file, outfile, preserve_mode=False)

    def build_modules(self) -> None:
        modules = self.find_modules()
        for package, module, module_file in modules:
            # Now "build" the module -- ie. copy the source file to
            # self.build_lib (the build directory for Python source).
            # (Actually, it gets copied to the directory for this package
            # under self.build_lib.)
            self.build_module(module, module_file, package)

    def build_packages(self) -> None:
        if self.packages is None:
            raise TypeError(
                f"{type(self).__name__}.packages is None. Is the Distribution missing packages ?"
            )
        for package in self.packages:
            # Get list of (package, module, module_file) tuples based on
            # scanning the package directory.  'package' is only included
            # in the tuple so that 'find_modules()' and
            # 'find_package_tuples()' have a consistent interface; it's
            # ignored here (apart from a sanity check).  Also, 'module' is
            # the *unqualified* module name (ie. no dots, no package -- we
            # already know its package!), and 'module_file' is the path to
            # the .py file, relative to the current directory
            # (ie. including 'package_dir').
            package_dir = self.get_package_dir(package)
            modules = self.find_package_modules(package, package_dir)

            # Now loop over the modules we found, "building" each one (just
            # copy it to self.build_lib).
            for package_, module, module_file in modules:
                assert package == package_
                self.build_module(module, module_file, package)

    def byte_compile(self, files) -> None:
        if sys.dont_write_bytecode:
            self.warn('byte-compiling is disabled, skipping.')
            return

        from ..util import byte_compile

        prefix = self.build_lib
        if prefix[-1] != os.sep:
            prefix = prefix + os.sep

        # XXX this code is essentially the same as the 'byte_compile()
        # method of the "install_lib" command, except for the determination
        # of the 'prefix' string.  Hmmm.
        if self.compile:
            byte_compile(files, optimize=0, force=self.force, prefix=prefix)
        if self.optimize > 0:
            byte_compile(
                files,
                optimize=self.optimize,
                force=self.force,
                prefix=prefix,
            )
<|endoftext|>"""distutils.command.bdist

Implements the Distutils 'bdist' command (create a built [binary]
distribution)."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from ..core import Command
from ..errors import DistutilsOptionError, DistutilsPlatformError
from ..util import get_platform

if TYPE_CHECKING:
    from typing_extensions import deprecated
else:

    def deprecated(message):
        return lambda fn: fn


def show_formats():
    """Print list of available formats (arguments to "--format" option)."""
    from ..fancy_getopt import FancyGetopt

    formats = [
        ("formats=" + format, None, bdist.format_commands[format][1])
        for format in bdist.format_commands
    ]
    pretty_printer = FancyGetopt(formats)
    pretty_printer.print_help("List of available distribution formats:")


class ListCompat(dict[str, tuple[str, str]]):
    # adapter to allow for Setuptools compatibility in format_commands
    @deprecated("format_commands is now a dict. append is deprecated.")
    def append(self, item: object) -> None:
        warnings.warn(
            "format_commands is now a dict. append is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )


class bdist(Command):
    description = "create a built (binary) distribution"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('bdist-base=', 'b', "temporary directory for creating built distributions"),
        (
            'plat-name=',
            'p',
            (
                "platform name to embed in generated filenames "
                f"[default: {get_platform()}]"
            ),
        ),
        ('formats=', None, "formats for distribution (comma-separated list)"),
        (
            'dist-dir=',
            'd',
            "directory to put final built distributions in [default: dist]",
        ),
        ('skip-build', None, "skip rebuilding everything (for testing/debugging)"),
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

    boolean_options: ClassVar[list[str]] = ['skip-build']

    help_options: ClassVar[list[tuple[str, str | None, str, Callable[[], object]]]] = [
        ('help-formats', None, "lists available distribution formats", show_formats),
    ]

    # The following commands do not take a format option from bdist
    no_format_option: ClassVar[tuple[str, ...]] = ('bdist_rpm',)

    # This won't do in reality: will need to distinguish RPM-ish Linux,
    # Debian-ish Linux, Solaris, FreeBSD, ..., Windows, Mac OS.
    default_format: ClassVar[dict[str, str]] = {'posix': 'gztar', 'nt': 'zip'}

    # Define commands in preferred order for the --help-formats option
    format_commands = ListCompat({
        'rpm': ('bdist_rpm', "RPM distribution"),
        'gztar': ('bdist_dumb', "gzip'ed tar file"),
        'bztar': ('bdist_dumb', "bzip2'ed tar file"),
        'xztar': ('bdist_dumb', "xz'ed tar file"),
        'ztar': ('bdist_dumb', "compressed tar file"),
        'tar': ('bdist_dumb', "tar file"),
        'zip': ('bdist_dumb', "ZIP file"),
    })

    # for compatibility until consumers only reference format_commands
    format_command = format_commands

    def initialize_options(self):
        self.bdist_base = None
        self.plat_name = None
        self.formats = None
        self.dist_dir = None
        self.skip_build = False
        self.group = None
        self.owner = None

    def finalize_options(self) -> None:
        # have to finalize 'plat_name' before 'bdist_base'
        if self.plat_name is None:
            if self.skip_build:
                self.plat_name = get_platform()
            else:
                self.plat_name = self.get_finalized_command('build').plat_name

        # 'bdist_base' -- parent of per-built-distribution-format
        # temporary directories (eg. we'll probably have
        # "build/bdist.<plat>/dumb", "build/bdist.<plat>/rpm", etc.)
        if self.bdist_base is None:
            build_base = self.get_finalized_command('build').build_base
            self.bdist_base = os.path.join(build_base, 'bdist.' + self.plat_name)

        self.ensure_string_list('formats')
        if self.formats is None:
            try:
                self.formats = [self.default_format[os.name]]
            except KeyError:
                raise DistutilsPlatformError(
                    "don't know how to create built distributions "
                    f"on platform {os.name}"
                )

        if self.dist_dir is None:
            self.dist_dir = "dist"

    def run(self) -> None:
        # Figure out which sub-commands we need to run.
        commands = []
        for format in self.formats:
            try:
                commands.append(self.format_commands[format][0])
            except KeyError:
                raise DistutilsOptionError(f"invalid format '{format}'")

        # Reinitialize and run each command.
        for i in range(len(self.formats)):
            cmd_name = commands[i]
            sub_cmd = self.reinitialize_command(cmd_name)
            if cmd_name not in self.no_format_option:
                sub_cmd.format = self.formats[i]

            # passing the owner and group names for tar archiving
            if cmd_name == 'bdist_dumb':
                sub_cmd.owner = self.owner
                sub_cmd.group = self.group

            # If we're going to need to run this command again, tell it to
            # keep its temporary files around so subsequent runs go faster.
            if cmd_name in commands[i + 1 :]:
                sub_cmd.keep_temp = True
            self.run_command(cmd_name)
<|endoftext|>"""distutils.command.install_scripts

Implements the Distutils 'install_scripts' command, for installing
Python scripts."""

# contributed by Bastian Kleineidam

import os
from distutils._log import log
from stat import ST_MODE
from typing import ClassVar

from ..core import Command


class install_scripts(Command):
    description = "install scripts (Python or otherwise)"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('install-dir=', 'd', "directory to install scripts to"),
        ('build-dir=', 'b', "build directory (where to install from)"),
        ('force', 'f', "force installation (overwrite existing files)"),
        ('skip-build', None, "skip the build steps"),
    ]

    boolean_options: ClassVar[list[str]] = ['force', 'skip-build']

    def initialize_options(self):
        self.install_dir = None
        self.force = False
        self.build_dir = None
        self.skip_build = None

    def finalize_options(self) -> None:
        self.set_undefined_options('build', ('build_scripts', 'build_dir'))
        self.set_undefined_options(
            'install',
            ('install_scripts', 'install_dir'),
            ('force', 'force'),
            ('skip_build', 'skip_build'),
        )

    def run(self) -> None:
        if not self.skip_build:
            self.run_command('build_scripts')
        self.outfiles = self.copy_tree(self.build_dir, self.install_dir)
        if os.name == 'posix':
            # Set the executable bits (owner, group, and world) on
            # all the scripts we just installed.
            for file in self.get_outputs():
                mode = ((os.stat(file)[ST_MODE]) | 0o555) & 0o7777
                log.info("changing mode of %s to %o", file, mode)
                os.chmod(file, mode)

    def get_inputs(self):
        return self.distribution.scripts or []

    def get_outputs(self):
        return self.outfiles or []
<|endoftext|>from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

_T = TypeVar("_T")

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:
    if TYPE_CHECKING:
        # typing_extensions usually "exist" when type-checking,
        # without the need for extra runtime dependencies
        from typing_extensions import dataclass_transform
    else:
        # Runtime no-op
        def dataclass_transform(  # type: ignore[misc]
            *,
            eq_default: bool | None = None,
            order_default: bool | None = None,
            kw_only_default: bool | None = None,
            field_specifiers: tuple[type[Any], ...] = (),
            **_: Any,
        ) -> Callable[[_T], _T]:
            def _decorator(obj: _T) -> _T:
                return obj

            return _decorator
<|endoftext|>from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import TypeAlias

    from _typeshed import ExcInfo, StrOrBytesPath

# Same as shutil._OnExcCallback from typeshed
_OnExcCallback: TypeAlias = Callable[[Callable[..., Any], str, BaseException], object]


def shutil_rmtree(
    path: StrOrBytesPath,
    ignore_errors: bool = False,
    onexc: _OnExcCallback | None = None,
) -> None:
    if sys.version_info >= (3, 12):
        return shutil.rmtree(path, ignore_errors, onexc=onexc)

    def _handler(fn: Callable[..., Any], path: str, excinfo: ExcInfo) -> None:
        if onexc:
            onexc(fn, path, excinfo[1])

    return shutil.rmtree(path, ignore_errors, onerror=_handler)
<|endoftext|>from __future__ import annotations

import sys

if sys.version_info >= (3, 12, 4):
    # Python 3.13 should support `.pth` files encoded in UTF-8
    # See discussion in https://github.com/python/cpython/issues/77102
    PTH_ENCODING: str | None = "utf-8"
else:
    from .py39 import LOCALE_ENCODING

    # PTH_ENCODING = "locale"
    PTH_ENCODING = LOCALE_ENCODING
<|endoftext|>import sys

__all__ = ['tomllib']


if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


if sys.version_info >= (3, 11):

    def add_note(ex, note):
        ex.add_note(note)

else:  # pragma: no cover

    def add_note(ex, note):
        vars(ex).setdefault('__notes__', []).append(note)
<|endoftext|>"""Utility functions to expand configuration directives or special values
(such glob patterns).

We can split the process of interpreting configuration files into 2 steps:

1. The parsing the file contents from strings to value objects
   that can be understand by Python (for example a string with a comma
   separated list of keywords into an actual Python list of strings).

2. The expansion (or post-processing) of these values according to the
   semantics ``setuptools`` assign to them (for example a configuration field
   with the ``file:`` directive should be expanded from a list of file paths to
   a single string with the contents of those files concatenated)

This module focus on the second step, and therefore allow sharing the expansion
functions among several configuration file formats.

**PRIVATE MODULE**: API reserved for setuptools internal usage only.
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from configparser import ConfigParser
from glob import iglob
from importlib.machinery import ModuleSpec, all_suffixes
from itertools import chain
from pathlib import Path
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Any, TypeVar

from .. import _static
from .._path import StrPath, same_path as _same_path
from ..discovery import find_package_path
from ..warnings import SetuptoolsWarning

from distutils.errors import DistutilsOptionError

if TYPE_CHECKING:
    from typing_extensions import Self

    from setuptools.dist import Distribution

_K = TypeVar("_K")
_V_co = TypeVar("_V_co", covariant=True)


class StaticModule:
    """Proxy to a module object that avoids executing arbitrary code."""

    def __init__(self, name: str, spec: ModuleSpec) -> None:
        module = ast.parse(pathlib.Path(spec.origin).read_bytes())  # type: ignore[arg-type] # Let it raise an error on None
        vars(self).update(locals())
        del self.self

    def _find_assignments(self) -> Iterator[tuple[ast.AST, ast.AST]]:
        for statement in self.module.body:
            if isinstance(statement, ast.Assign):
                yield from ((target, statement.value) for target in statement.targets)
            elif isinstance(statement, ast.AnnAssign) and statement.value:
                yield (statement.target, statement.value)

    def __getattr__(self, attr: str) -> Any:
        """Attempt to load an attribute "statically", via :func:`ast.literal_eval`."""
        try:
            return next(
                ast.literal_eval(value)
                for target, value in self._find_assignments()
                if isinstance(target, ast.Name) and target.id == attr
            )
        except Exception as e:
            raise AttributeError(f"{self.name} has no attribute {attr}") from e


def glob_relative(
    patterns: Iterable[str], root_dir: StrPath | None = None
) -> list[str]:
    """Expand the list of glob patterns, but preserving relative paths.

    :param list[str] patterns: List of glob patterns
    :param str root_dir: Path to which globs should be relative
                         (current directory by default)
    :rtype: list
    """
    glob_characters = {'*', '?', '[', ']', '{', '}'}
    expanded_values = []
    root_dir = root_dir or os.getcwd()
    for value in patterns:
        # Has globby characters?
        if any(char in value for char in glob_characters):
            # then expand the glob pattern while keeping paths *relative*:
            glob_path = os.path.abspath(os.path.join(root_dir, value))
            expanded_values.extend(
                sorted(
                    os.path.relpath(path, root_dir).replace(os.sep, "/")
                    for path in iglob(glob_path, recursive=True)
                )
            )

        else:
            # take the value as-is
            path = os.path.relpath(value, root_dir).replace(os.sep, "/")
            expanded_values.append(path)

    return expanded_values


def read_files(
    filepaths: StrPath | Iterable[StrPath], root_dir: StrPath | None = None
) -> str:
    """Return the content of the files concatenated using ``\n`` as str

    This function is sandboxed and won't reach anything outside ``root_dir``

    (By default ``root_dir`` is the current directory).
    """
    from more_itertools import always_iterable

    root_dir = os.path.abspath(root_dir or os.getcwd())
    _filepaths = (os.path.join(root_dir, path) for path in always_iterable(filepaths))
    return '\n'.join(
        _read_file(path)
        for path in _filter_existing_files(_filepaths)
        if _assert_local(path, root_dir)
    )


def _filter_existing_files(filepaths: Iterable[StrPath]) -> Iterator[StrPath]:
    for path in filepaths:
        if os.path.isfile(path):
            yield path
        else:
            SetuptoolsWarning.emit(f"File {path!r} cannot be found")


def _read_file(filepath: bytes | StrPath) -> str:
    with open(filepath, encoding='utf-8') as f:
        return f.read()


def _assert_local(filepath: StrPath, root_dir: str):
    if Path(os.path.abspath(root_dir)) not in Path(os.path.abspath(filepath)).parents:
        msg = f"Cannot access {filepath!r} (or anything outside {root_dir!r})"
        raise DistutilsOptionError(msg)

    return True


def read_attr(
    attr_desc: str,
    package_dir: Mapping[str, str] | None = None,
    root_dir: StrPath | None = None,
) -> Any:
    """Reads the value of an attribute from a module.

    This function will try to read the attributed statically first
    (via :func:`ast.literal_eval`), and only evaluate the module if it fails.

    Examples:
        read_attr("package.attr")
        read_attr("package.module.attr")

    :param str attr_desc: Dot-separated string describing how to reach the
        attribute (see examples above)
    :param dict[str, str] package_dir: Mapping of package names to their
        location in disk (represented by paths relative to ``root_dir``).
    :param str root_dir: Path to directory containing all the packages in
        ``package_dir`` (current directory by default).
    :rtype: str
    """
    root_dir = root_dir or os.getcwd()
    attrs_path = attr_desc.strip().split('.')
    attr_name = attrs_path.pop()
    module_name = '.'.join(attrs_path)
    module_name = module_name or '__init__'
    path = _find_module(module_name, package_dir, root_dir)
    spec = _find_spec(module_name, path)

    try:
        value = getattr(StaticModule(module_name, spec), attr_name)
        # XXX: Is marking as static contents coming from modules too optimistic?
        return _static.attempt_conversion(value)
    except Exception:  # noqa: BLE001 # intentional broad fallback
        # fallback to evaluate module
        module = _load_spec(spec, module_name)
        return getattr(module, attr_name)


def _find_spec(module_name: str, module_path: StrPath | None) -> ModuleSpec:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    spec = spec or importlib.util.find_spec(module_name)

    if spec is None:
        raise ModuleNotFoundError(module_name)

    return spec


def _load_spec(spec: ModuleSpec, module_name: str) -> ModuleType:
    name = getattr(spec, "__name__", module_name)
    if name in sys.modules:
        return sys.modules[name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # cache (it also ensures `==` works on loaded items)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _find_module(
    module_name: str, package_dir: Mapping[str, str] | None, root_dir: StrPath
) -> str | None:
    """Find the path to the module named ``module_name``,
    considering the ``package_dir`` in the build configuration and ``root_dir``.

    >>> tmp = getfixture('tmpdir')
    >>> _ = tmp.ensure("a/b/c.py")
    >>> _ = tmp.ensure("a/b/d/__init__.py")
    >>> r = lambda x: x.replace(str(tmp), "tmp").replace(os.sep, "/")
    >>> r(_find_module("a.b.c", None, tmp))
    'tmp/a/b/c.py'
    >>> r(_find_module("f.g.h", {"": "1", "f": "2", "f.g": "3", "f.g.h": "a/b/d"}, tmp))
    'tmp/a/b/d/__init__.py'
    """
    path_start = find_package_path(module_name, package_dir or {}, root_dir)
    candidates = chain.from_iterable(
        (f"{path_start}{ext}", os.path.join(path_start, f"__init__{ext}"))
        for ext in all_suffixes()
    )
    return next((x for x in candidates if os.path.isfile(x)), None)


def resolve_class(
    qualified_class_name: str,
    package_dir: Mapping[str, str] | None = None,
    root_dir: StrPath | None = None,
) -> Callable:
    """Given a qualified class name, return the associated class object"""
    root_dir = root_dir or os.getcwd()
    idx = qualified_class_name.rfind('.')
    class_name = qualified_class_name[idx + 1 :]
    pkg_name = qualified_class_name[:idx]

    path = _find_module(pkg_name, package_dir, root_dir)
    module = _load_spec(_find_spec(pkg_name, path), pkg_name)
    return getattr(module, class_name)


def cmdclass(
    values: dict[str, str],
    package_dir: Mapping[str, str] | None = None,
    root_dir: StrPath | None = None,
) -> dict[str, Callable]:
    """Given a dictionary mapping command names to strings for qualified class
    names, apply :func:`resolve_class` to the dict values.
    """
    return {k: resolve_class(v, package_dir, root_dir) for k, v in values.items()}


def find_packages(
    *,
    namespaces=True,
    fill_package_dir: dict[str, str] | None = None,
    root_dir: StrPath | None = None,
    **kwargs,
) -> list[str]:
    """Works similarly to :func:`setuptools.find_packages`, but with all
    arguments given as keyword arguments. Moreover, ``where`` can be given
    as a list (the results will be simply concatenated).

    When the additional keyword argument ``namespaces`` is ``True``, it will
    behave like :func:`setuptools.find_namespace_packages`` (i.e. include
    implicit namespaces as per :pep:`420`).

    The ``where`` argument will be considered relative to ``root_dir`` (or the current
    working directory when ``root_dir`` is not given).

    If the ``fill_package_dir`` argument is passed, this function will consider it as a
    similar data structure to the ``package_dir`` configuration parameter add fill-in
    any missing package location.

    :rtype: list
    """
    from more_itertools import always_iterable, unique_everseen

    from setuptools.discovery import construct_package_dir

    # check "not namespaces" first due to python/mypy#6232
    if not namespaces:
        from setuptools.discovery import PackageFinder
    else:
        from setuptools.discovery import PEP420PackageFinder as PackageFinder

    root_dir = root_dir or os.curdir
    where = kwargs.pop('where', ['.'])
    packages: list[str] = []
    fill_package_dir = {} if fill_package_dir is None else fill_package_dir
    search = list(unique_everseen(always_iterable(where)))

    if len(search) == 1 and all(not _same_path(search[0], x) for x in (".", root_dir)):
        fill_package_dir.setdefault("", search[0])

    for path in search:
        package_path = _nest_path(root_dir, path)
        pkgs = PackageFinder.find(package_path, **kwargs)
        packages.extend(pkgs)
        if pkgs and not (
            fill_package_dir.get("") == path or os.path.samefile(package_path, root_dir)
        ):
            fill_package_dir.update(construct_package_dir(pkgs, path))

    return packages


def _nest_path(parent: StrPath, path: StrPath) -> str:
    path = parent if path in {".", ""} else os.path.join(parent, path)
    return os.path.normpath(path)


def version(value: Callable | Iterable[str | int] | str) -> str:
    """When getting the version directly from an attribute,
    it should be normalised to string.
    """
    _value = value() if callable(value) else value

    if isinstance(_value, str):
        return _value
    if hasattr(_value, '__iter__'):
        return '.'.join(map(str, _value))
    return f'{_value}'


def canonic_package_data(package_data: dict) -> dict:
    if "*" in package_data:
        package_data[""] = package_data.pop("*")
    return package_data


def canonic_data_files(
    data_files: list | dict, root_dir: StrPath | None = None
) -> list[tuple[str, list[str]]]:
    """For compatibility with ``setup.py``, ``data_files`` should be a list
    of pairs instead of a dict.

    This function also expands glob patterns.
    """
    if isinstance(data_files, list):
        return data_files

    return [
        (dest, glob_relative(patterns, root_dir))
        for dest, patterns in data_files.items()
    ]


def entry_points(
    text: str, text_source: str = "entry-points"
) -> dict[str, dict[str, str]]:
    """Given the contents of entry-points file,
    process it into a 2-level dictionary (``dict[str, dict[str, str]]``).
    The first level keys are entry-point groups, the second level keys are
    entry-point names, and the second level values are references to objects
    (that correspond to the entry-point value).
    """
    # Using undocumented behaviour, see python/typeshed#12700
    parser = ConfigParser(default_section=None, delimiters=("=",))  # type: ignore[call-overload]
    parser.optionxform = str  # case sensitive
    parser.read_string(text, text_source)
    groups = {k: dict(v.items()) for k, v in parser.items()}
    groups.pop(parser.default_section, None)
    return groups


class EnsurePackagesDiscovered:
    """Some expand functions require all the packages to already be discovered before
    they run, e.g. :func:`read_attr`, :func:`resolve_class`, :func:`cmdclass`.

    Therefore in some cases we will need to run autodiscovery during the evaluation of
    the configuration. However, it is better to postpone calling package discovery as
    much as possible, because some parameters can influence it (e.g. ``package_dir``),
    and those might not have been processed yet.
    """

    def __init__(self, distribution: Distribution) -> None:
        self._dist = distribution
        self._called = False

    def __call__(self) -> None:
        """Trigger the automatic package discovery, if it is still necessary."""
        if not self._called:
            self._called = True
            self._dist.set_defaults(name=False)  # Skip name, we can still be parsing

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._called:
            self._dist.set_defaults.analyse_name()  # Now we can set a default name

    def _get_package_dir(self) -> Mapping[str, str]:
        self()
        pkg_dir = self._dist.package_dir
        return {} if pkg_dir is None else pkg_dir

    @property
    def package_dir(self) -> Mapping[str, str]:
        """Proxy to ``package_dir`` that may trigger auto-discovery when used."""
        return LazyMappingProxy(self._get_package_dir)


class LazyMappingProxy(Mapping[_K, _V_co]):
    """Mapping proxy that delays resolving the target object, until really needed.

    >>> def obtain_mapping():
    ...     print("Running expensive function!")
    ...     return {"key": "value", "other key": "other value"}
    >>> mapping = LazyMappingProxy(obtain_mapping)
    >>> mapping["key"]
    Running expensive function!
    'value'
    >>> mapping["other key"]
    'other value'
    """

    def __init__(self, obtain_mapping_value: Callable[[], Mapping[_K, _V_co]]) -> None:
        self._obtain = obtain_mapping_value
        self._value: Mapping[_K, _V_co] | None = None

    def _target(self) -> Mapping[_K, _V_co]:
        if self._value is None:
            self._value = self._obtain()
        return self._value

    def __getitem__(self, key: _K) -> _V_co:
        return self._target()[key]

    def __len__(self) -> int:
        return len(self._target())

    def __iter__(self) -> Iterator[_K]:
        return iter(self._target())
<|endoftext|>"""Translation layer between pyproject config and setuptools distribution and
metadata objects.

The distribution and metadata objects are modeled after (an old version of)
core metadata, therefore configs in the format specified for ``pyproject.toml``
need to be processed before being applied.

**PRIVATE MODULE**: API reserved for setuptools internal usage only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

from .. import _static
from .._path import StrPath
from ..errors import InvalidConfigError, RemovedConfigError
from ..extension import Extension
from ..warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning

if TYPE_CHECKING:
    from typing import TypeAlias

    from setuptools._importlib import metadata
    from setuptools.dist import Distribution

    from distutils.dist import _OptionsList  # Comes from typeshed


EMPTY: Mapping = MappingProxyType({})  # Immutable dict-like
_ProjectReadmeValue: TypeAlias = str | dict[str, str]
_Correspondence: TypeAlias = Callable[["Distribution", Any, StrPath | None], None]
_T = TypeVar("_T")

_logger = logging.getLogger(__name__)


def apply(dist: Distribution, config: dict, filename: StrPath) -> Distribution:
    """Apply configuration dict read with :func:`read_configuration`"""

    if not config:
        return dist  # short-circuit unrelated pyproject.toml file

    root_dir = os.path.dirname(filename) or "."

    _apply_project_table(dist, config, root_dir)
    _apply_tool_table(dist, config, filename)

    current_directory = os.getcwd()
    os.chdir(root_dir)
    try:
        dist._finalize_requires()
        dist._finalize_license_expression()
        dist._finalize_license_files()
    finally:
        os.chdir(current_directory)

    return dist


def _apply_project_table(dist: Distribution, config: dict, root_dir: StrPath):
    orig_config = config.get("project", {})
    if not orig_config:
        return  # short-circuit

    project_table = {k: _static.attempt_conversion(v) for k, v in orig_config.items()}
    _handle_missing_dynamic(dist, project_table)
    _unify_entry_points(project_table)

    for field, value in project_table.items():
        norm_key = json_compatible_key(field)
        corresp = PYPROJECT_CORRESPONDENCE.get(norm_key, norm_key)
        if callable(corresp):
            corresp(dist, value, root_dir)
        else:
            _set_config(dist, corresp, value)


def _apply_tool_table(dist: Distribution, config: dict, filename: StrPath):
    tool_table = config.get("tool", {}).get("setuptools", {})
    if not tool_table:
        return  # short-circuit

    if "license-files" in tool_table:
        if "license-files" in config.get("project", {}):
            # https://github.com/pypa/setuptools/pull/4837#discussion_r2004983349
            raise InvalidConfigError(
                "'project.license-files' is defined already. "
                "Remove 'tool.setuptools.license-files'."
            )

        pypa_guides = "guides/writing-pyproject-toml/#license-files"
        SetuptoolsDeprecationWarning.emit(
            "'tool.setuptools.license-files' is deprecated in favor of "
            "'project.license-files' (available on setuptools>=77.0.0).",
            see_url=f"https://packaging.python.org/en/latest/{pypa_guides}",
            due_date=(2027, 2, 18),  # Warning introduced on 2025-02-18
        )

    for field, value in tool_table.items():
        norm_key = json_compatible_key(field)

        if norm_key in TOOL_TABLE_REMOVALS:
            suggestion = cleandoc(TOOL_TABLE_REMOVALS[norm_key])
            msg = f"""
            The parameter `tool.setuptools.{field}` was long deprecated
            and has been removed from `pyproject.toml`.
            """
            raise RemovedConfigError("\n".join([cleandoc(msg), suggestion]))

        norm_key = TOOL_TABLE_RENAMES.get(norm_key, norm_key)
        corresp = TOOL_TABLE_CORRESPONDENCE.get(norm_key, norm_key)
        if callable(corresp):
            corresp(dist, value)
        else:
            _set_config(dist, corresp, value)

    _copy_command_options(config, dist, filename)


def _handle_missing_dynamic(dist: Distribution, project_table: dict):
    """Be temporarily forgiving with ``dynamic`` fields not listed in ``dynamic``"""
    dynamic = set(project_table.get("dynamic", []))
    for field, getter in _PREVIOUSLY_DEFINED.items():
        if not (field in project_table or field in dynamic):
            value = getter(dist)
            if value:
                _MissingDynamic.emit(field=field, value=value)
                project_table[field] = _RESET_PREVIOUSLY_DEFINED.get(field)


def json_compatible_key(key: str) -> str:
    """As defined in :pep:`566#json-compatible-metadata`"""
    return key.lower().replace("-", "_")


def _set_config(dist: Distribution, field: str, value: Any):
    val = _PREPROCESS.get(field, _noop)(dist, value)
    setter = getattr(dist.metadata, f"set_{field}", None)
    if setter:
        setter(val)
    elif hasattr(dist.metadata, field) or field in SETUPTOOLS_PATCHES:
        setattr(dist.metadata, field, val)
    else:
        setattr(dist, field, val)


_CONTENT_TYPES = {
    ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
}


def _guess_content_type(file: str) -> str | None:
    _, ext = os.path.splitext(file.lower())
    if not ext:
        return None

    if ext in _CONTENT_TYPES:
        return _static.Str(_CONTENT_TYPES[ext])

    valid = ", ".join(f"{k} ({v})" for k, v in _CONTENT_TYPES.items())
    msg = f"only the following file extensions are recognized: {valid}."
    raise ValueError(f"Undefined content type for {file}, {msg}")


def _long_description(
    dist: Distribution, val: _ProjectReadmeValue, root_dir: StrPath | None
):
    from setuptools.config import expand

    file: str | tuple[()]
    if isinstance(val, str):
        file = val
        text = expand.read_files(file, root_dir)
        ctype = _guess_content_type(file)
    else:
        file = val.get("file") or ()
        text = val.get("text") or expand.read_files(file, root_dir)
        ctype = val["content-type"]

    # XXX: Is it completely safe to assume static?
    _set_config(dist, "long_description", _static.Str(text))

    if ctype:
        _set_config(dist, "long_description_content_type", _static.Str(ctype))

    if file:
        dist._referenced_files.add(file)


def _license(dist: Distribution, val: str | dict, root_dir: StrPath | None):
    from setuptools.config import expand

    if isinstance(val, str):
        if getattr(dist.metadata, "license", None):
            SetuptoolsWarning.emit("`license` overwritten by `pyproject.toml`")
            dist.metadata.license = None
        _set_config(dist, "license_expression", _static.Str(val))
    else:
        pypa_guides = "guides/writing-pyproject-toml/#license"
        SetuptoolsDeprecationWarning.emit(
            "`project.license` as a TOML table is deprecated",
            "Please use a simple string containing a SPDX expression for "
            "`project.license`. You can also use `project.license-files`. "
            "(Both options available on setuptools>=77.0.0).",
            see_url=f"https://packaging.python.org/en/latest/{pypa_guides}",
            due_date=(2027, 2, 18),  # Introduced on 2025-02-18
        )
        if "file" in val:
            # XXX: Is it completely safe to assume static?
            value = expand.read_files([val["file"]], root_dir)
            _set_config(dist, "license", _static.Str(value))
            dist._referenced_files.add(val["file"])
        else:
            _set_config(dist, "license", _static.Str(val["text"]))


def _people(dist: Distribution, val: list[dict], _root_dir: StrPath | None, kind: str):
    field = []
    email_field = []
    for person in val:
        if "name" not in person:
            email_field.append(person["email"])
        elif "email" not in person:
            field.append(person["name"])
        else:
            addr = Address(display_name=person["name"], addr_spec=person["email"])
            email_field.append(str(addr))

    if field:
        _set_config(dist, kind, _static.Str(", ".join(field)))
    if email_field:
        _set_config(dist, f"{kind}_email", _static.Str(", ".join(email_field)))


def _project_urls(dist: Distribution, val: dict, _root_dir: StrPath | None):
    _set_config(dist, "project_urls", val)


def _python_requires(dist: Distribution, val: str, _root_dir: StrPath | None):
    _set_config(dist, "python_requires", _static.SpecifierSet(val))


def _dependencies(dist: Distribution, val: list, _root_dir: StrPath | None):
    if getattr(dist, "install_requires", []):
        msg = "`install_requires` overwritten in `pyproject.toml` (dependencies)"
        SetuptoolsWarning.emit(msg)
    dist.install_requires = val


def _optional_dependencies(dist: Distribution, val: dict, _root_dir: StrPath | None):
    if getattr(dist, "extras_require", None):
        msg = "`extras_require` overwritten in `pyproject.toml` (optional-dependencies)"
        SetuptoolsWarning.emit(msg)
    dist.extras_require = val


def _ext_modules(dist: Distribution, val: list[dict]) -> list[Extension]:
    existing = dist.ext_modules or []
    args = ({k.replace("-", "_"): v for k, v in x.items()} for x in val)
    new = (Extension(**_adjust_ext_attrs(kw)) for kw in args)
    return [*existing, *new]


def _adjust_ext_attrs(attrs: dict) -> dict:
    # https://github.com/pypa/setuptools/issues/4810
    # In TOML there is no differentiation between tuples and lists,
    # and distutils requires tuples...
    attrs["define_macros"] = list(map(tuple, attrs.get("define_macros") or []))
    return attrs


def _noop(_dist: Distribution, val: _T) -> _T:
    return val


def _identity(val: _T) -> _T:
    return val


def _unify_entry_points(project_table: dict):
    project = project_table
    given = project.pop("entry-points", project.pop("entry_points", {}))
    entry_points = dict(given)  # Avoid problems with static
    renaming = {"scripts": "console_scripts", "gui_scripts": "gui_scripts"}
    for key, value in list(project.items()):  # eager to allow modifications
        norm_key = json_compatible_key(key)
        if norm_key in renaming:
            # Don't skip even if value is empty (reason: reset missing `dynamic`)
            entry_points[renaming[norm_key]] = project.pop(key)

    if entry_points:
        project["entry-points"] = {
            name: [f"{k} = {v}" for k, v in group.items()]
            for name, group in entry_points.items()
            if group  # now we can skip empty groups
        }
        # Sometimes this will set `project["entry-points"] = {}`, and that is
        # intentional (for resetting configurations that are missing `dynamic`).


def _copy_command_options(pyproject: dict, dist: Distribution, filename: StrPath):
    tool_table = pyproject.get("tool", {})
    cmdclass = tool_table.get("setuptools", {}).get("cmdclass", {})
    valid_options = _valid_command_options(cmdclass)

    cmd_opts = dist.command_options
    for cmd, config in pyproject.get("tool", {}).get("distutils", {}).items():
        cmd = json_compatible_key(cmd)
        valid = valid_options.get(cmd, set())
        cmd_opts.setdefault(cmd, {})
        for key, value in config.items():
            key = json_compatible_key(key)
            cmd_opts[cmd][key] = (str(filename), value)
            if key not in valid:
                # To avoid removing options that are specified dynamically we
                # just log a warn...
                _logger.warning(f"Command option {cmd}.{key} is not defined")


def _valid_command_options(cmdclass: Mapping = EMPTY) -> dict[str, set[str]]:
    from setuptools.dist import Distribution

    from .._importlib import metadata

    valid_options = {"global": _normalise_cmd_options(Distribution.global_options)}

    unloaded_entry_points = metadata.entry_points(group='distutils.commands')
    loaded_entry_points = (_load_ep(ep) for ep in unloaded_entry_points)
    entry_points = (ep for ep in loaded_entry_points if ep)
    for cmd, cmd_class in chain(entry_points, cmdclass.items()):
        opts = valid_options.get(cmd, set())
        opts = opts | _normalise_cmd_options(getattr(cmd_class, "user_options", []))
        valid_options[cmd] = opts

    return valid_options


def _load_ep(ep: metadata.EntryPoint) -> tuple[str, type] | None:
    if ep.value.startswith("wheel.bdist_wheel"):
        # Ignore deprecated entrypoint from wheel and avoid warning pypa/wheel#631
        # TODO: remove check when `bdist_wheel` has been fully removed from pypa/wheel
        return None

    # Ignore all the errors
    try:
        return (ep.name, ep.load())
    except Exception as ex:  # noqa: BLE001 # intentional broad fallback
        msg = f"{ex.__class__.__name__} while trying to load entry-point {ep.name}"
        _logger.warning(f"{msg}: {ex}")
        return None


def _normalise_cmd_option_key(name: str) -> str:
    return json_compatible_key(name).strip("_=")


def _normalise_cmd_options(desc: _OptionsList) -> set[str]:
    return {_normalise_cmd_option_key(fancy_option[0]) for fancy_option in desc}


def _get_previous_entrypoints(dist: Distribution) -> dict[str, list]:
    ignore = ("console_scripts", "gui_scripts")
    value = getattr(dist, "entry_points", None) or {}
    return {k: v for k, v in value.items() if k not in ignore}


def _get_previous_scripts(dist: Distribution) -> list | None:
    value = getattr(dist, "entry_points", None) or {}
    return value.get("console_scripts")


def _get_previous_gui_scripts(dist: Distribution) -> list | None:
    value = getattr(dist, "entry_points", None) or {}
    return value.get("gui_scripts")


def _set_static_list_metadata(attr: str, dist: Distribution, val: list) -> None:
    """Apply distutils metadata validation but preserve "static" behaviour"""
    meta = dist.metadata
    setter, getter = getattr(meta, f"set_{attr}"), getattr(meta, f"get_{attr}")
    setter(val)
    setattr(meta, attr, _static.List(getter()))


def _attrgetter(attr):
    """
    Similar to ``operator.attrgetter`` but returns None if ``attr`` is not found
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace(a=42, b=SimpleNamespace(c=13))
    >>> _attrgetter("a")(obj)
    42
    >>> _attrgetter("b.c")(obj)
    13
    >>> _attrgetter("d")(obj) is None
    True
    """
    return partial(reduce, lambda acc, x: getattr(acc, x, None), attr.split("."))


def _some_attrgetter(*items):
    """
    Return the first "truth-y" attribute or None
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace(a=42, b=SimpleNamespace(c=13))
    >>> _some_attrgetter("d", "a", "b.c")(obj)
    42
    >>> _some_attrgetter("d", "e", "b.c", "a")(obj)
    13
    >>> _some_attrgetter("d", "e", "f")(obj) is None
    True
    """

    def _acessor(obj):
        values = (_attrgetter(i)(obj) for i in items)
        return next((i for i in values if i is not None), None)

    return _acessor


PYPROJECT_CORRESPONDENCE: dict[str, _Correspondence] = {
    "readme": _long_description,
    "license": _license,
    "authors": partial(_people, kind="author"),
    "maintainers": partial(_people, kind="maintainer"),
    "urls": _project_urls,
    "dependencies": _dependencies,
    "optional_dependencies": _optional_dependencies,
    "requires_python": _python_requires,
}

TOOL_TABLE_RENAMES = {"script_files": "scripts"}
TOOL_TABLE_REMOVALS = {
    "namespace_packages": """
        Please migrate to implicit native namespaces instead.
        See https://packaging.python.org/en/latest/guides/packaging-namespace-packages/.
        """,
}
TOOL_TABLE_CORRESPONDENCE = {
    # Fields with corresponding core metadata need to be marked as static:
    "obsoletes": partial(_set_static_list_metadata, "obsoletes"),
    "provides": partial(_set_static_list_metadata, "provides"),
    "platforms": partial(_set_static_list_metadata, "platforms"),
}

SETUPTOOLS_PATCHES = {
    "long_description_content_type",
    "project_urls",
    "provides_extras",
    "license_file",
    "license_files",
    "license_expression",
}

_PREPROCESS = {
    "ext_modules": _ext_modules,
}

_PREVIOUSLY_DEFINED = {
    "name": _attrgetter("metadata.name"),
    "version": _attrgetter("metadata.version"),
    "description": _attrgetter("metadata.description"),
    "readme": _attrgetter("metadata.long_description"),
    "requires-python": _some_attrgetter("python_requires", "metadata.python_requires"),
    "license": _some_attrgetter("metadata.license_expression", "metadata.license"),
    # XXX: `license-file` is currently not considered in the context of `dynamic`.
    #      See TestPresetField.test_license_files_exempt_from_dynamic
    "authors": _some_attrgetter("metadata.author", "metadata.author_email"),
    "maintainers": _some_attrgetter("metadata.maintainer", "metadata.maintainer_email"),
    "keywords": _attrgetter("metadata.keywords"),
    "classifiers": _attrgetter("metadata.classifiers"),
    "urls": _attrgetter("metadata.project_urls"),
    "entry-points": _get_previous_entrypoints,
    "scripts": _get_previous_scripts,
    "gui-scripts": _get_previous_gui_scripts,
    "dependencies": _attrgetter("install_requires"),
    "optional-dependencies": _attrgetter("extras_require"),
}


_RESET_PREVIOUSLY_DEFINED: dict = {
    # Fix improper setting: given in `setup.py`, but not listed in `dynamic`
    # Use "immutable" data structures to avoid in-place modification.
    # dict: pyproject name => value to which reset
    "license": "",
    # XXX: `license-file` is currently not considered in the context of `dynamic`.
    #      See TestPresetField.test_license_files_exempt_from_dynamic
    "authors": _static.EMPTY_LIST,
    "maintainers": _static.EMPTY_LIST,
    "keywords": _static.EMPTY_LIST,
    "classifiers": _static.EMPTY_LIST,
    "urls": _static.EMPTY_DICT,
    "entry-points": _static.EMPTY_DICT,
    "scripts": _static.EMPTY_DICT,
    "gui-scripts": _static.EMPTY_DICT,
    "dependencies": _static.EMPTY_LIST,
    "optional-dependencies": _static.EMPTY_DICT,
}


class _MissingDynamic(SetuptoolsWarning):
    _SUMMARY = "`{field}` defined outside of `pyproject.toml` is ignored."

    _DETAILS = """
    The following seems to be defined outside of `pyproject.toml`:

    `{field} = {value!r}`

    According to the spec (see the link below), however, setuptools CANNOT
    consider this value unless `{field}` is listed as `dynamic`.

    https://packaging.python.org/en/latest/specifications/pyproject-toml/#declaring-project-metadata-the-project-table

    To prevent this problem, you can list `{field}` under `dynamic` or alternatively
    remove the `[project]` table from your file and rely entirely on other means of
    configuration.
    """
    # TODO: Consider removing this check in the future?
    #       There is a trade-off here between improving "debug-ability" and the cost
    #       of running/testing/maintaining these unnecessary checks...

    @classmethod
    def details(cls, field: str, value: Any) -> str:
        return cls._DETAILS.format(field=field, value=value)
<|endoftext|>"""
Load setuptools configuration from ``setup.cfg`` files.

**API will be made private in the future**

To read project metadata, consider using
``build.util.project_wheel_metadata`` (https://pypi.org/project/build/).
For simple scenarios, you can also try parsing the file directly
with the help of ``configparser``.
"""

from __future__ import annotations

import contextlib
import functools
import os
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from packaging.markers import default_environment as marker_env
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from .. import _static
from .._path import StrPath
from ..errors import FileError, OptionError
from ..warnings import SetuptoolsDeprecationWarning
from . import expand

if TYPE_CHECKING:
    from typing import TypeAlias

    from setuptools.dist import Distribution

    from distutils.dist import DistributionMetadata

SingleCommandOptions: TypeAlias = dict[str, tuple[str, Any]]
"""Dict that associate the name of the options of a particular command to a
tuple. The first element of the tuple indicates the origin of the option value
(e.g. the name of the configuration file where it was read from),
while the second element of the tuple is the option value itself
"""
AllCommandOptions: TypeAlias = dict[str, SingleCommandOptions]
"""cmd name => its options"""
Target = TypeVar("Target", "Distribution", "DistributionMetadata")


def read_configuration(
    filepath: StrPath, find_others: bool = False, ignore_option_errors: bool = False
) -> dict:
    """Read given configuration file and returns options from it as a dict.

    :param str|unicode filepath: Path to configuration file
        to get options from.

    :param bool find_others: Whether to search for other configuration files
        which could be on in various places.

    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.

    :rtype: dict
    """
    from setuptools.dist import Distribution

    dist = Distribution()
    filenames = dist.find_config_files() if find_others else []
    handlers = _apply(dist, filepath, filenames, ignore_option_errors)
    return configuration_to_dict(handlers)


def apply_configuration(dist: Distribution, filepath: StrPath) -> Distribution:
    """Apply the configuration from a ``setup.cfg`` file into an existing
    distribution object.
    """
    _apply(dist, filepath)
    dist._finalize_requires()
    return dist


def _apply(
    dist: Distribution,
    filepath: StrPath,
    other_files: Iterable[StrPath] = (),
    ignore_option_errors: bool = False,
) -> tuple[ConfigMetadataHandler, ConfigOptionsHandler]:
    """Read configuration from ``filepath`` and applies to the ``dist`` object."""
    from setuptools.dist import _Distribution

    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        raise FileError(f'Configuration file {filepath} does not exist.')

    current_directory = os.getcwd()
    os.chdir(os.path.dirname(filepath))
    filenames = [*other_files, filepath]

    try:
        # TODO: Temporary cast until mypy 1.12 is released with upstream fixes from typeshed
        _Distribution.parse_config_files(dist, filenames=cast(list[str], filenames))
        handlers = parse_configuration(
            dist, dist.command_options, ignore_option_errors=ignore_option_errors
        )
        dist._finalize_license_files()
    finally:
        os.chdir(current_directory)

    return handlers


def _get_option(target_obj: Distribution | DistributionMetadata, key: str):
    """
    Given a target object and option key, get that option from
    the target object, either through a get_{key} method or
    from an attribute directly.
    """
    getter_name = f'get_{key}'
    by_attribute = functools.partial(getattr, target_obj, key)
    getter = getattr(target_obj, getter_name, by_attribute)
    return getter()


def configuration_to_dict(
    handlers: Iterable[
        ConfigHandler[Distribution] | ConfigHandler[DistributionMetadata]
    ],
) -> dict:
    """Returns configuration data gathered by given handlers as a dict.

    :param Iterable[ConfigHandler] handlers: Handlers list,
        usually from parse_configuration()

    :rtype: dict
    """
    config_dict: dict = defaultdict(dict)

    for handler in handlers:
        for option in handler.set_options:
            value = _get_option(handler.target_obj, option)
            config_dict[handler.section_prefix][option] = value

    return config_dict


def parse_configuration(
    distribution: Distribution,
    command_options: AllCommandOptions,
    ignore_option_errors: bool = False,
) -> tuple[ConfigMetadataHandler, ConfigOptionsHandler]:
    """Performs additional parsing of configuration options
    for a distribution.

    Returns a list of used option handlers.

    :param Distribution distribution:
    :param dict command_options:
    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.
    :rtype: list
    """
    with expand.EnsurePackagesDiscovered(distribution) as ensure_discovered:
        options = ConfigOptionsHandler(
            distribution,
            command_options,
            ignore_option_errors,
            ensure_discovered,
        )

        options.parse()
        if not distribution.package_dir:
            distribution.package_dir = options.package_dir  # Filled by `find_packages`

        meta = ConfigMetadataHandler(
            distribution.metadata,
            command_options,
            ignore_option_errors,
            ensure_discovered,
            distribution.package_dir,
            distribution.src_root,
        )
        meta.parse()
        distribution._referenced_files.update(
            options._referenced_files, meta._referenced_files
        )

    return meta, options


def _warn_accidental_env_marker_misconfig(label: str, orig_value: str, parsed: list):
    """Because users sometimes misinterpret this configuration:

    [options.extras_require]
    foo = bar;python_version<"4"

    It looks like one requirement with an environment marker
    but because there is no newline, it's parsed as two requirements
    with a semicolon as separator.

    Therefore, if:
        * input string does not contain a newline AND
        * parsed result contains two requirements AND
        * parsing of the two parts from the result ("<first>;<second>")
        leads in a valid Requirement with a valid marker
    a UserWarning is shown to inform the user about the possible problem.
    """
    if "\n" in orig_value or len(parsed) != 2:
        return

    markers = marker_env().keys()

    try:
        req = Requirement(parsed[1])
        if req.name in markers:
            _AmbiguousMarker.emit(field=label, req=parsed[1])
    except InvalidRequirement as ex:
        if any(parsed[1].startswith(marker) for marker in markers):
            msg = _AmbiguousMarker.message(field=label, req=parsed[1])
            raise InvalidRequirement(msg) from ex


class ConfigHandler(Generic[Target]):
    """Handles metadata supplied in configuration files."""

    section_prefix: str
    """Prefix for config sections handled by this handler.
    Must be provided by class heirs.

    """

    aliases: ClassVar[dict[str, str]] = {}
    """Options aliases.
    For compatibility with various packages. E.g.: d2to1 and pbr.
    Note: `-` in keys is replaced with `_` by config parser.

    """

    def __init__(
        self,
        target_obj: Target,
        options: AllCommandOptions,
        ignore_option_errors,
        ensure_discovered: expand.EnsurePackagesDiscovered,
    ) -> None:
        self.ignore_option_errors = ignore_option_errors
        self.target_obj: Target = target_obj
        self.sections = dict(self._section_options(options))
        self.set_options: list[str] = []
        self.ensure_discovered = ensure_discovered
        self._referenced_files = set[str]()
        """After parsing configurations, this property will enumerate
        all files referenced by the "file:" directive. Private API for setuptools only.
        """

    @classmethod
    def _section_options(
        cls, options: AllCommandOptions
    ) -> Iterator[tuple[str, SingleCommandOptions]]:
        for full_name, value in options.items():
            pre, _sep, name = full_name.partition(cls.section_prefix)
            if pre:
                continue
            yield name.lstrip('.'), value

    @property
    @abstractmethod
    def parsers(self) -> dict[str, Callable]:
        """Metadata item name to parser function mapping."""
        raise NotImplementedError(
            f'{self.__class__.__name__} must provide .parsers property'
        )

    def __setitem__(self, option_name, value) -> None:
        target_obj = self.target_obj

        # Translate alias into real name.
        option_name = self.aliases.get(option_name, option_name)

        try:
            current_value = getattr(target_obj, option_name)
        except AttributeError as e:
            raise KeyError(option_name) from e

        if current_value:
            # Already inhabited. Skipping.
            return

        try:
            parsed = self.parsers.get(option_name, lambda x: x)(value)
        except (Exception,) * self.ignore_option_errors:  # noqa: B030 # dynamic exception tuple (0 or 1 members)
            return

        simple_setter = functools.partial(target_obj.__setattr__, option_name)
        setter = getattr(target_obj, f"set_{option_name}", simple_setter)
        setter(parsed)

        self.set_options.append(option_name)

    @classmethod
    def _parse_list(cls, value, separator=','):
        """Represents value as a list.

        Value is split either by separator (defaults to comma) or by lines.

        :param value:
        :param separator: List items separator character.
        :rtype: list
        """
        if isinstance(value, list):  # _get_parser_compound case
            return value

        if '\n' in value:
            value = value.splitlines()
        else:
            value = value.split(separator)

        return [chunk.strip() for chunk in value if chunk.strip()]

    @classmethod
    def _parse_dict(cls, value):
        """Represents value as a dict.

        :param value:
        :rtype: dict
        """
        separator = '='
        result = {}
        for line in cls._parse_list(value):
            key, sep, val = line.partition(separator)
            if sep != separator:
                raise OptionError(f"Unable to parse option value to dict: {value}")
            result[key.strip()] = val.strip()

        return result

    @classmethod
    def _parse_bool(cls, value):
        """Represents value as boolean.

        :param value:
        :rtype: bool
        """
        value = value.lower()
        return value in ('1', 'true', 'yes')

    @classmethod
    def _exclude_files_parser(cls, key):
        """Returns a parser function to make sure field inputs
        are not files.

        Parses a value after getting the key so error messages are
        more informative.

        :param key:
        :rtype: callable
        """

        def parser(value):
            exclude_directive = 'file:'
            if value.startswith(exclude_directive):
                raise ValueError(
                    f'Only strings are accepted for the {key} field, '
                    'files are not accepted'
                )
            return _static.Str(value)

        return parser

    def _parse_file(self, value, root_dir: StrPath | None):
        """Represents value as a string, allowing including text
        from nearest files using `file:` directive.

        Directive is sandboxed and won't reach anything outside
        directory with setup.py.

        Examples:
            file: README.rst, CHANGELOG.md, src/file.txt

        :param str value:
        :rtype: str
        """
        include_directive = 'file:'

        if not isinstance(value, str):
            return value

        if not value.startswith(include_directive):
            return _static.Str(value)

        spec = value[len(include_directive) :]
        filepaths = [path.strip() for path in spec.split(',')]
        self._referenced_files.update(filepaths)
        # XXX: Is marking as static contents coming from files too optimistic?
        return _static.Str(expand.read_files(filepaths, root_dir))

    def _parse_attr(self, value, package_dir, root_dir: StrPath):
        """Represents value as a module attribute.

        Examples:
            attr: package.attr
            attr: package.module.attr

        :param str value:
        :rtype: str
        """
        attr_directive = 'attr:'
        if not value.startswith(attr_directive):
            return _static.Str(value)

        attr_desc = value.replace(attr_directive, '')

        # Make sure package_dir is populated correctly, so `attr:` directives can work
        package_dir.update(self.ensure_discovered.package_dir)
        return expand.read_attr(attr_desc, package_dir, root_dir)

    @classmethod
    def _get_parser_compound(cls, *parse_methods):
        """Returns parser function to represents value as a list.

        Parses a value applying given methods one after another.

        :param parse_methods:
        :rtype: callable
        """

        def parse(value):
            parsed = value

            for method in parse_methods:
                parsed = method(parsed)

            return parsed

        return parse

    @classmethod
    def _parse_section_to_dict_with_key(cls, section_options, values_parser):
        """Parses section options into a dictionary.

        Applies a given parser to each option in a section.

        :param dict section_options:
        :param callable values_parser: function with 2 args corresponding to key, value
        :rtype: dict
        """
        value = {}
        for key, (_, val) in section_options.items():
            value[key] = values_parser(key, val)
        return value

    @classmethod
    def _parse_section_to_dict(cls, section_options, values_parser=None):
        """Parses section options into a dictionary.

        Optionally applies a given parser to each value.

        :param dict section_options:
        :param callable values_parser: function with 1 arg corresponding to option value
        :rtype: dict
        """
        parser = (lambda _, v: values_parser(v)) if values_parser else (lambda _, v: v)
        return cls._parse_section_to_dict_with_key(section_options, parser)

    def parse_section(self, section_options) -> None:
        """Parses configuration file section.

        :param dict section_options:
        """
        for name, (_, value) in section_options.items():
            with contextlib.suppress(KeyError):
                # Keep silent for a new option may appear anytime.
                self[name] = value

    def parse(self) -> None:
        """Parses configuration file items from one
        or more related sections.

        """
        for section_name, section_options in self.sections.items():
            method_postfix = ''
            if section_name:  # [section.option] variant
                method_postfix = f"_{section_name}"

            section_parser_method: Callable | None = getattr(
                self,
                # Dots in section names are translated into dunderscores.
                f'parse_section{method_postfix}'.replace('.', '__'),
                None,
            )

            if section_parser_method is None:
                raise OptionError(
                    "Unsupported distribution option section: "
                    f"[{self.section_prefix}.{section_name}]"
                )

            section_parser_method(section_options)

    def _deprecated_config_handler(self, func, msg, **kw):
        """this function will wrap around parameters that are deprecated

        :param msg: deprecation message
        :param func: function to be wrapped around
        """

        @wraps(func)
        def config_handler(*args, **kwargs):
            kw.setdefault("stacklevel", 2)
            _DeprecatedConfig.emit("Deprecated config in `setup.cfg`", msg, **kw)
            return func(*args, **kwargs)

        return config_handler


class ConfigMetadataHandler(ConfigHandler["DistributionMetadata"]):
    section_prefix = 'metadata'

    aliases: ClassVar[dict[str, str]] = {
        'home_page': 'url',
        'summary': 'description',
        'classifier': 'classifiers',
        'platform': 'platforms',
    }

    strict_mode = False
    """We need to keep it loose, to be partially compatible with
    `pbr` and `d2to1` packages which also uses `metadata` section.

    """

    def __init__(
        self,
        target_obj: DistributionMetadata,
        options: AllCommandOptions,
        ignore_option_errors: bool,
        ensure_discovered: expand.EnsurePackagesDiscovered,
        package_dir: dict | None = None,
        root_dir: StrPath | None = os.curdir,
    ) -> None:
        super().__init__(target_obj, options, ignore_option_errors, ensure_discovered)
        self.package_dir = package_dir
        self.root_dir = root_dir

    @property
    def parsers(self) -> dict[str, Callable]:
        """Metadata item name to parser function mapping."""
        parse_list_static = self._get_parser_compound(self._parse_list, _static.List)
        parse_dict_static = self._get_parser_compound(self._parse_dict, _static.Dict)
        parse_file = partial(self._parse_file, root_dir=self.root_dir)
        exclude_files_parser = self._exclude_files_parser

        return {
            'author': _static.Str,
            'author_email': _static.Str,
            'maintainer': _static.Str,
            'maintainer_email': _static.Str,
            'platforms': parse_list_static,
            'keywords': parse_list_static,
            'provides': parse_list_static,
            'obsoletes': parse_list_static,
            'classifiers': self._get_parser_compound(parse_file, parse_list_static),
            'license': exclude_files_parser('license'),
            'license_files': parse_list_static,
            'description': parse_file,
            'long_description': parse_file,
            'long_description_content_type': _static.Str,
            'version': self._parse_version,  # Cannot be marked as dynamic
            'url': _static.Str,
            'project_urls': parse_dict_static,
        }

    def _parse_version(self, value):
        """Parses `version` option value.

        :param value:
        :rtype: str

        """
        version = self._parse_file(value, self.root_dir)

        if version != value:
            version = version.strip()
            # Be strict about versions loaded from file because it's easy to
            # accidentally include newlines and other unintended content
            try:
                Version(version)
            except InvalidVersion as e:
                raise OptionError(
                    f'Version loaded from {value} does not '
                    f'comply with PEP 440: {version}'
                ) from e

            return version

        return expand.version(self._parse_attr(value, self.package_dir, self.root_dir))


class ConfigOptionsHandler(ConfigHandler["Distribution"]):
    section_prefix = 'options'

    def __init__(
        self,
        target_obj: Distribution,
        options: AllCommandOptions,
        ignore_option_errors: bool,
        ensure_discovered: expand.EnsurePackagesDiscovered,
    ) -> None:
        super().__init__(target_obj, options, ignore_option_errors, ensure_discovered)
        self.root_dir = target_obj.src_root
        self.package_dir: dict[str, str] = {}  # To be filled by `find_packages`

    @classmethod
    def _parse_list_semicolon(cls, value):
        return cls._parse_list(value, separator=';')

    def _parse_file_in_root(self, value):
        return self._parse_file(value, root_dir=self.root_dir)

    def _parse_requirements_list(self, label: str, value: str):
        # Parse a requirements list, either by reading in a `file:`, or a list.
        parsed = self._parse_list_semicolon(self._parse_file_in_root(value))
        _warn_accidental_env_marker_misconfig(label, value, parsed)
        # Filter it to only include lines that are not comments. `parse_list`
        # will have stripped each line and filtered out empties.
        return _static.List(line for line in parsed if not line.startswith("#"))
        # ^-- Use `_static.List` to mark a non-`Dynamic` Core Metadata

    @property
    def parsers(self) -> dict[str, Callable]:
        """Metadata item name to parser function mapping."""
        parse_list = self._parse_list
        parse_bool = self._parse_bool
        parse_cmdclass = self._parse_cmdclass

        return {
            'zip_safe': parse_bool,
            'include_package_data': parse_bool,
            'package_dir': self._parse_dict,
            'scripts': parse_list,
            'eager_resources': parse_list,
            'dependency_links': parse_list,
            'namespace_packages': self._deprecated_config_handler(
                parse_list,
                "The namespace_packages parameter is deprecated, "
                "consider using implicit namespaces instead (PEP 420).",
                # TODO: define due date, see setuptools.dist:check_nsp.
            ),
            'install_requires': partial(  # Core Metadata
                self._parse_requirements_list, "install_requires"
            ),
            'setup_requires': self._parse_list_semicolon,
            'packages': self._parse_packages,
            'entry_points': self._parse_file_in_root,
            'py_modules': parse_list,
            'python_requires': _static.SpecifierSet,  # Core Metadata
            'cmdclass': parse_cmdclass,
        }

    def _parse_cmdclass(self, value):
        package_dir = self.ensure_discovered.package_dir
        return expand.cmdclass(self._parse_dict(value), package_dir, self.root_dir)

    def _parse_packages(self, value):
        """Parses `packages` option value.

        :param value:
        :rtype: list
        """
        find_directives = ['find:', 'find_namespace:']
        trimmed_value = value.strip()

        if trimmed_value not in find_directives:
            return self._parse_list(value)

        # Read function arguments from a dedicated section.
        find_kwargs = self.parse_section_packages__find(
            self.sections.get('packages.find', {})
        )

        find_kwargs.update(
            namespaces=(trimmed_value == find_directives[1]),
            root_dir=self.root_dir,
            fill_package_dir=self.package_dir,
        )

        return expand.find_packages(**find_kwargs)

    def parse_section_packages__find(self, section_options):
        """Parses `packages.find` configuration file section.

        To be used in conjunction with _parse_packages().

        :param dict section_options:
        """
        section_data = self._parse_section_to_dict(section_options, self._parse_list)

        valid_keys = ['where', 'include', 'exclude']
        find_kwargs = {k: v for k, v in section_data.items() if k in valid_keys and v}

        where = find_kwargs.get('where')
        if where is not None:
            find_kwargs['where'] = where[0]  # cast list to single val

        return find_kwargs

    def parse_section_entry_points(self, section_options) -> None:
        """Parses `entry_points` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict(section_options, self._parse_list)
        self['entry_points'] = parsed

    def _parse_package_data(self, section_options):
        package_data = self._parse_section_to_dict(section_options, self._parse_list)
        return expand.canonic_package_data(package_data)

    def parse_section_package_data(self, section_options) -> None:
        """Parses `package_data` configuration file section.

        :param dict section_options:
        """
        self['package_data'] = self._parse_package_data(section_options)

    def parse_section_exclude_package_data(self, section_options) -> None:
        """Parses `exclude_package_data` configuration file section.

        :param dict section_options:
        """
        self['exclude_package_data'] = self._parse_package_data(section_options)

    def parse_section_extras_require(self, section_options) -> None:  # Core Metadata
        """Parses `extras_require` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict_with_key(
            section_options,
            lambda k, v: self._parse_requirements_list(f"extras_require[{k}]", v),
        )

        self['extras_require'] = _static.Dict(parsed)
        # ^-- Use `_static.Dict` to mark a non-`Dynamic` Core Metadata

    def parse_section_data_files(self, section_options) -> None:
        """Parses `data_files` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict(section_options, self._parse_list)
        self['data_files'] = expand.canonic_data_files(parsed, self.root_dir)


class _AmbiguousMarker(SetuptoolsDeprecationWarning):
    _SUMMARY = "Ambiguous requirement marker."
    _DETAILS = """
    One of the parsed requirements in `{field}` looks like a valid environment marker:

        {req!r}

    Please make sure that the configuration file is correct.
    You can use dangling lines to avoid this problem.
    """
    _SEE_DOCS = "userguide/declarative_config.html#opt-2"
    # TODO: should we include due_date here? Initially introduced in 6 Aug 2022.
    # Does this make sense with latest version of packaging?

    @classmethod
    def message(cls, **kw):
        docs = f"https://setuptools.pypa.io/en/latest/{cls._SEE_DOCS}"
        return cls._format(cls._SUMMARY, cls._DETAILS, see_url=docs, format_args=kw)


class _DeprecatedConfig(SetuptoolsDeprecationWarning):
    _SEE_DOCS = "userguide/declarative_config.html"
<|endoftext|>"""
Load setuptools configuration from ``pyproject.toml`` files.

**PRIVATE MODULE**: API reserved for setuptools internal usage only.

To read project metadata, consider using
``build.util.project_wheel_metadata`` (https://pypi.org/project/build/).
For simple scenarios, you can also try parsing the file directly
with the help of ``tomllib`` or ``tomli``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any

from .._path import StrPath
from ..errors import FileError, InvalidConfigError
from ..warnings import SetuptoolsWarning
from . import expand as _expand
from ._apply_pyprojecttoml import _PREVIOUSLY_DEFINED, _MissingDynamic, apply as _apply

if TYPE_CHECKING:
    from typing_extensions import Self

    from setuptools.dist import Distribution

_logger = logging.getLogger(__name__)


def load_file(filepath: StrPath) -> dict:
    from ..compat.py310 import tomllib

    with open(filepath, "rb") as file:
        return tomllib.load(file)


def validate(config: dict, filepath: StrPath) -> bool:
    from . import _validate_pyproject as validator

    trove_classifier = validator.FORMAT_FUNCTIONS.get("trove-classifier")
    if hasattr(trove_classifier, "_disable_download"):
        # Improve reproducibility by default. See abravalheri/validate-pyproject#31
        trove_classifier._disable_download()  # type: ignore[union-attr]

    try:
        return validator.validate(config)
    except validator.ValidationError as ex:
        summary = f"configuration error: {ex.summary}"
        if ex.name.strip("`") != "project":
            # Probably it is just a field missing/misnamed, not worthy the verbosity...
            _logger.debug(summary)
            _logger.debug(ex.details)

        error = f"invalid pyproject.toml config: {ex.name}."
        raise ValueError(f"{error}\n{summary}") from None


def apply_configuration(
    dist: Distribution,
    filepath: StrPath,
    ignore_option_errors: bool = False,
) -> Distribution:
    """Apply the configuration from a ``pyproject.toml`` file into an existing
    distribution object.
    """
    config = read_configuration(filepath, True, ignore_option_errors, dist)
    return _apply(dist, config, filepath)


def read_configuration(
    filepath: StrPath,
    expand: bool = True,
    ignore_option_errors: bool = False,
    dist: Distribution | None = None,
) -> dict[str, Any]:
    """Read given configuration file and returns options from it as a dict.

    :param str|unicode filepath: Path to configuration file in the ``pyproject.toml``
        format.

    :param bool expand: Whether to expand directives and other computed values
        (i.e. post-process the given configuration)

    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.

    :param Distribution|None: Distribution object to which the configuration refers.
        If not given a dummy object will be created and discarded after the
        configuration is read. This is used for auto-discovery of packages and in the
        case a dynamic configuration (e.g. ``attr`` or ``cmdclass``) is expanded.
        When ``expand=False`` this object is simply ignored.

    :rtype: dict
    """
    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        raise FileError(f"Configuration file {filepath!r} does not exist.")

    asdict = load_file(filepath) or {}
    project_table = asdict.get("project", {})
    tool_table = asdict.get("tool", {})
    setuptools_table = tool_table.get("setuptools", {})
    if not asdict or not (project_table or setuptools_table):
        return {}  # User is not using pyproject to configure setuptools

    if "setuptools" in asdict.get("tools", {}):
        # let the user know they probably have a typo in their metadata
        _ToolsTypoInMetadata.emit()

    if "distutils" in tool_table:
        _ExperimentalConfiguration.emit(subject="[tool.distutils]")

    # There is an overall sense in the community that making include_package_data=True
    # the default would be an improvement.
    # `ini2toml` backfills include_package_data=False when nothing is explicitly given,
    # therefore setting a default here is backwards compatible.
    if dist and dist.include_package_data is not None:
        setuptools_table.setdefault("include-package-data", dist.include_package_data)
    else:
        setuptools_table.setdefault("include-package-data", True)
    # Persist changes:
    asdict["tool"] = tool_table
    tool_table["setuptools"] = setuptools_table

    if "ext-modules" in setuptools_table:
        _ExperimentalConfiguration.emit(subject="[tool.setuptools.ext-modules]")

    fields = ("import-names", "import-namespaces")
    places = (project_table, project_table.get("dynamic", []))
    if any(field in place for field in fields for place in places):
        raise NotImplementedError(
            "Setuptools does not support `import-names` and `import-namespaces`"
            " in `pyproject.toml` yet. If your are interested in this feature, "
            " please consider submitting a contribution via pull requests."
        )

    with _ignore_errors(ignore_option_errors):
        # Don't complain about unrelated errors (e.g. tools not using the "tool" table)
        subset = {"project": project_table, "tool": {"setuptools": setuptools_table}}
        validate(subset, filepath)

    if expand:
        root_dir = os.path.dirname(filepath)
        return expand_configuration(asdict, root_dir, ignore_option_errors, dist)

    return asdict


def expand_configuration(
    config: dict,
    root_dir: StrPath | None = None,
    ignore_option_errors: bool = False,
    dist: Distribution | None = None,
) -> dict:
    """Given a configuration with unresolved fields (e.g. dynamic, cmdclass, ...)
    find their final values.

    :param dict config: Dict containing the configuration for the distribution
    :param str root_dir: Top-level directory for the distribution/project
        (the same directory where ``pyproject.toml`` is place)
    :param bool ignore_option_errors: see :func:`read_configuration`
    :param Distribution|None: Distribution object to which the configuration refers.
        If not given a dummy object will be created and discarded after the
        configuration is read. Used in the case a dynamic configuration
        (e.g. ``attr`` or ``cmdclass``).

    :rtype: dict
    """
    return _ConfigExpander(config, root_dir, ignore_option_errors, dist).expand()


class _ConfigExpander:
    def __init__(
        self,
        config: dict,
        root_dir: StrPath | None = None,
        ignore_option_errors: bool = False,
        dist: Distribution | None = None,
    ) -> None:
        self.config = config
        self.root_dir = root_dir or os.getcwd()
        self.project_cfg = config.get("project", {})
        self.dynamic = self.project_cfg.get("dynamic", [])
        self.setuptools_cfg = config.get("tool", {}).get("setuptools", {})
        self.dynamic_cfg = self.setuptools_cfg.get("dynamic", {})
        self.ignore_option_errors = ignore_option_errors
        self._dist = dist
        self._referenced_files = set[str]()

    def _ensure_dist(self) -> Distribution:
        from setuptools.dist import Distribution

        attrs = {"src_root": self.root_dir, "name": self.project_cfg.get("name", None)}
        return self._dist or Distribution(attrs)

    def _process_field(self, container: dict, field: str, fn: Callable):
        if field in container:
            with _ignore_errors(self.ignore_option_errors):
                container[field] = fn(container[field])

    def _canonic_package_data(self, field="package-data"):
        package_data = self.setuptools_cfg.get(field, {})
        return _expand.canonic_package_data(package_data)

    def expand(self):
        self._expand_packages()
        self._canonic_package_data()
        self._canonic_package_data("exclude-package-data")

        # A distribution object is required for discovering the correct package_dir
        dist = self._ensure_dist()
        ctx = _EnsurePackagesDiscovered(dist, self.project_cfg, self.setuptools_cfg)
        with ctx as ensure_discovered:
            package_dir = ensure_discovered.package_dir
            self._expand_data_files()
            self._expand_cmdclass(package_dir)
            self._expand_all_dynamic(dist, package_dir)

        dist._referenced_files.update(self._referenced_files)
        return self.config

    def _expand_packages(self):
        packages = self.setuptools_cfg.get("packages")
        if packages is None or isinstance(packages, (list, tuple)):
            return

        find = packages.get("find")
        if isinstance(find, dict):
            find["root_dir"] = self.root_dir
            find["fill_package_dir"] = self.setuptools_cfg.setdefault("package-dir", {})
            with _ignore_errors(self.ignore_option_errors):
                self.setuptools_cfg["packages"] = _expand.find_packages(**find)

    def _expand_data_files(self):
        data_files = partial(_expand.canonic_data_files, root_dir=self.root_dir)
        self._process_field(self.setuptools_cfg, "data-files", data_files)

    def _expand_cmdclass(self, package_dir: Mapping[str, str]):
        root_dir = self.root_dir
        cmdclass = partial(_expand.cmdclass, package_dir=package_dir, root_dir=root_dir)
        self._process_field(self.setuptools_cfg, "cmdclass", cmdclass)

    def _expand_all_dynamic(self, dist: Distribution, package_dir: Mapping[str, str]):
        special = (  # need special handling
            "version",
            "readme",
            "entry-points",
            "scripts",
            "gui-scripts",
            "classifiers",
            "dependencies",
            "optional-dependencies",
        )
        # `_obtain` functions are assumed to raise appropriate exceptions/warnings.
        obtained_dynamic = {
            field: self._obtain(dist, field, package_dir)
            for field in self.dynamic
            if field not in special
        }
        obtained_dynamic.update(
            self._obtain_entry_points(dist, package_dir) or {},
            version=self._obtain_version(dist, package_dir),
            readme=self._obtain_readme(dist),
            classifiers=self._obtain_classifiers(dist),
            dependencies=self._obtain_dependencies(dist),
            optional_dependencies=self._obtain_optional_dependencies(dist),
        )
        # `None` indicates there is nothing in `tool.setuptools.dynamic` but the value
        # might have already been set by setup.py/extensions, so avoid overwriting.
        updates = {k: v for k, v in obtained_dynamic.items() if v is not None}
        self.project_cfg.update(updates)

    def _ensure_previously_set(self, dist: Distribution, field: str):
        previous = _PREVIOUSLY_DEFINED[field](dist)
        if previous is None and not self.ignore_option_errors:
            msg = (
                f"No configuration found for dynamic {field!r}.\n"
                "Some dynamic fields need to be specified via `tool.setuptools.dynamic`"
                "\nothers must be specified via the equivalent attribute in `setup.py`."
            )
            raise InvalidConfigError(msg)

    def _expand_directive(
        self, specifier: str, directive, package_dir: Mapping[str, str]
    ):
        from more_itertools import always_iterable

        with _ignore_errors(self.ignore_option_errors):
            root_dir = self.root_dir
            if "file" in directive:
                self._referenced_files.update(always_iterable(directive["file"]))
                return _expand.read_files(directive["file"], root_dir)
            if "attr" in directive:
                return _expand.read_attr(directive["attr"], package_dir, root_dir)
            raise ValueError(f"invalid `{specifier}`: {directive!r}")
        return None

    def _obtain(self, dist: Distribution, field: str, package_dir: Mapping[str, str]):
        if field in self.dynamic_cfg:
            return self._expand_directive(
                f"tool.setuptools.dynamic.{field}",
                self.dynamic_cfg[field],
                package_dir,
            )
        self._ensure_previously_set(dist, field)
        return None

    def _obtain_version(self, dist: Distribution, package_dir: Mapping[str, str]):
        # Since plugins can set version, let's silently skip if it cannot be obtained
        if "version" in self.dynamic and "version" in self.dynamic_cfg:
            return _expand.version(
                # We already do an early check for the presence of "version"
                self._obtain(dist, "version", package_dir)  # pyright: ignore[reportArgumentType]
            )
        return None

    def _obtain_readme(self, dist: Distribution) -> dict[str, str] | None:
        if "readme" not in self.dynamic:
            return None

        dynamic_cfg = self.dynamic_cfg
        if "readme" in dynamic_cfg:
            return {
                # We already do an early check for the presence of "readme"
                "text": self._obtain(dist, "readme", {}),
                "content-type": dynamic_cfg["readme"].get("content-type", "text/x-rst"),
            }  # pyright: ignore[reportReturnType]

        self._ensure_previously_set(dist, "readme")
        return None

    def _obtain_entry_points(
        self, dist: Distribution, package_dir: Mapping[str, str]
    ) -> dict[str, dict[str, Any]] | None:
        fields = ("entry-points", "scripts", "gui-scripts")
        if not any(field in self.dynamic for field in fields):
            return None

        text = self._obtain(dist, "entry-points", package_dir)
        if text is None:
            return None

        groups = _expand.entry_points(text)
        # Any is str | dict[str, str], but causes variance issues
        expanded: dict[str, dict[str, Any]] = {"entry-points": groups}

        def _set_scripts(field: str, group: str):
            if group in groups:
                value = groups.pop(group)
                if field not in self.dynamic:
                    raise InvalidConfigError(_MissingDynamic.details(field, value))
                expanded[field] = value

        _set_scripts("scripts", "console_scripts")
        _set_scripts("gui-scripts", "gui_scripts")

        return expanded

    def _obtain_classifiers(self, dist: Distribution):
        if "classifiers" in self.dynamic:
            value = self._obtain(dist, "classifiers", {})
            if value:
                return value.splitlines()
        return None

    def _obtain_dependencies(self, dist: Distribution):
        if "dependencies" in self.dynamic:
            value = self._obtain(dist, "dependencies", {})
            if value:
                return _parse_requirements_list(value)
        return None

    def _obtain_optional_dependencies(self, dist: Distribution):
        if "optional-dependencies" not in self.dynamic:
            return None
        if "optional-dependencies" in self.dynamic_cfg:
            optional_dependencies_map = self.dynamic_cfg["optional-dependencies"]
            assert isinstance(optional_dependencies_map, dict)
            return {
                group: _parse_requirements_list(
                    self._expand_directive(
                        f"tool.setuptools.dynamic.optional-dependencies.{group}",
                        directive,
                        {},
                    )
                )
                for group, directive in optional_dependencies_map.items()
            }
        self._ensure_previously_set(dist, "optional-dependencies")
        return None


def _parse_requirements_list(value):
    return [
        line
        for line in value.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@contextmanager
def _ignore_errors(ignore_option_errors: bool):
    if not ignore_option_errors:
        yield
        return

    try:
        yield
    except Exception as ex:  # noqa: BLE001 # intentional broad fallback
        _logger.debug(f"ignored error: {ex.__class__.__name__} - {ex}")


class _EnsurePackagesDiscovered(_expand.EnsurePackagesDiscovered):
    def __init__(
        self, distribution: Distribution, project_cfg: dict, setuptools_cfg: dict
    ) -> None:
        super().__init__(distribution)
        self._project_cfg = project_cfg
        self._setuptools_cfg = setuptools_cfg

    def __enter__(self) -> Self:
        """When entering the context, the values of ``packages``, ``py_modules`` and
        ``package_dir`` that are missing in ``dist`` are copied from ``setuptools_cfg``.
        """
        dist, cfg = self._dist, self._setuptools_cfg
        package_dir: dict[str, str] = cfg.setdefault("package-dir", {})
        package_dir.update(dist.package_dir or {})
        dist.package_dir = package_dir  # needs to be the same object

        dist.set_defaults._ignore_ext_modules()  # pyproject.toml-specific behaviour

        # Set `name`, `py_modules` and `packages` in dist to short-circuit
        # auto-discovery, but avoid overwriting empty lists purposefully set by users.
        if dist.metadata.name is None:
            dist.metadata.name = self._project_cfg.get("name")
        if dist.py_modules is None:
            dist.py_modules = cfg.get("py-modules")
        if dist.packages is None:
            dist.packages = cfg.get("packages")

        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """When exiting the context, if values of ``packages``, ``py_modules`` and
        ``package_dir`` are missing in ``setuptools_cfg``, copy from ``dist``.
        """
        # If anything was discovered set them back, so they count in the final config.
        self._setuptools_cfg.setdefault("packages", self._dist.packages)
        self._setuptools_cfg.setdefault("py-modules", self._dist.py_modules)
        return super().__exit__(exc_type, exc_value, traceback)


class _ExperimentalConfiguration(SetuptoolsWarning):
    _SUMMARY = (
        "`{subject}` in `pyproject.toml` is still *experimental* "
        "and likely to change in future releases."
    )


class _ToolsTypoInMetadata(SetuptoolsWarning):
    _SUMMARY = (
        "Ignoring [tools.setuptools] in pyproject.toml, did you mean [tool.setuptools]?"
    )
<|endoftext|>"""For backward compatibility, expose main functions from
``setuptools.config.setupcfg``
"""

from collections.abc import Callable
from functools import wraps
from typing import TypeVar, cast

from ..warnings import SetuptoolsDeprecationWarning
from . import setupcfg

Fn = TypeVar("Fn", bound=Callable)

__all__ = ('parse_configuration', 'read_configuration')


def _deprecation_notice(fn: Fn) -> Fn:
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        SetuptoolsDeprecationWarning.emit(
            "Deprecated API usage.",
            f"""
            As setuptools moves its configuration towards `pyproject.toml`,
            `{__name__}.{fn.__name__}` became deprecated.

            For the time being, you can use the `{setupcfg.__name__}` module
            to access a backward compatible API, but this module is provisional
            and might be removed in the future.

            To read project metadata, consider using
            ``build.util.project_wheel_metadata`` (https://pypi.org/project/build/).
            For simple scenarios, you can also try parsing the file directly
            with the help of ``configparser``.
            """,
            # due_date not defined yet, because the community still heavily relies on it
            # Warning introduced in 24 Mar 2022
        )
        return fn(*args, **kwargs)

    return cast(Fn, _wrapper)


read_configuration = _deprecation_notice(setupcfg.read_configuration)
parse_configuration = _deprecation_notice(setupcfg.parse_configuration)
<|endoftext|>"""The purpose of this module is implement PEP 621 validations that are
difficult to express as a JSON Schema (or that are not supported by the current
JSON Schema library).
"""

import collections
import itertools
from inspect import cleandoc
from typing import Generator, Iterable, Mapping, TypeVar

from .error_reporting import ValidationError

T = TypeVar("T", bound=Mapping)


class RedefiningStaticFieldAsDynamic(ValidationError):
    _DESC = """According to PEP 621:

    Build back-ends MUST raise an error if the metadata specifies a field
    statically as well as being listed in dynamic.
    """
    __doc__ = _DESC
    _URL = (
        "https://packaging.python.org/en/latest/specifications/pyproject-toml/#dynamic"
    )


class IncludedDependencyGroupMustExist(ValidationError):
    _DESC = """An included dependency group must exist and must not be cyclic.
    """
    __doc__ = _DESC
    _URL = "https://peps.python.org/pep-0735/"


class ImportNameCollision(ValidationError):
    _DESC = """According to PEP 794:

    All import-names and import-namespaces items must be unique.
    """
    __doc__ = _DESC
    _URL = "https://peps.python.org/pep-0794/"


class ImportNameMissing(ValidationError):
    _DESC = """According to PEP 794:

    An import name must have all parents listed.
    """
    __doc__ = _DESC
    _URL = "https://peps.python.org/pep-0794/"


def validate_project_dynamic(pyproject: T) -> T:
    project_table = pyproject.get("project", {})
    dynamic = project_table.get("dynamic", [])

    for field in dynamic:
        if field in project_table:
            raise RedefiningStaticFieldAsDynamic(
                message=f"You cannot provide a value for `project.{field}` and "
                "list it under `project.dynamic` at the same time",
                value={
                    field: project_table[field],
                    "...": " # ...",
                    "dynamic": dynamic,
                },
                name=f"data.project.{field}",
                definition={
                    "description": cleandoc(RedefiningStaticFieldAsDynamic._DESC),
                    "see": RedefiningStaticFieldAsDynamic._URL,
                },
                rule="PEP 621",
            )

    return pyproject


def validate_include_depenency(pyproject: T) -> T:
    dependency_groups = pyproject.get("dependency-groups", {})
    for key, value in dependency_groups.items():
        for each in value:
            if (
                isinstance(each, dict)
                and (include_group := each.get("include-group"))
                and include_group not in dependency_groups
            ):
                raise IncludedDependencyGroupMustExist(
                    message=f"The included dependency group {include_group} doesn't exist",
                    value=each,
                    name=f"data.dependency_groups.{key}",
                    definition={
                        "description": cleandoc(IncludedDependencyGroupMustExist._DESC),
                        "see": IncludedDependencyGroupMustExist._URL,
                    },
                    rule="PEP 735",
                )
    # TODO: check for `include-group` cycles (can be conditional to graphlib)
    return pyproject


def _remove_private(items: Iterable[str]) -> Generator[str, None, None]:
    for item in items:
        yield item.partition(";")[0].rstrip()


def validate_import_name_issues(pyproject: T) -> T:
    project = pyproject.get("project", {})
    import_names = collections.Counter(_remove_private(project.get("import-names", [])))
    import_namespaces = collections.Counter(
        _remove_private(project.get("import-namespaces", []))
    )

    duplicated = [k for k, v in (import_names + import_namespaces).items() if v > 1]

    if duplicated:
        raise ImportNameCollision(
            message="Duplicated names are not allowed in import-names/import-namespaces",
            value=duplicated,
            name="data.project.importnames(paces)",
            definition={
                "description": cleandoc(ImportNameCollision._DESC),
                "see": ImportNameCollision._URL,
            },
            rule="PEP 794",
        )

    names = frozenset(import_names + import_namespaces)
    for name in names:
        for parent in itertools.accumulate(
            name.split(".")[:-1], lambda a, b: f"{a}.{b}"
        ):
            if parent not in names:
                raise ImportNameMissing(
                    message="All parents of an import name must also be listed in import-namespace/import-names",
                    value=name,
                    name="data.project.importnames(paces)",
                    definition={
                        "description": cleandoc(ImportNameMissing._DESC),
                        "see": ImportNameMissing._URL,
                    },
                    rule="PEP 794",
                )

    return pyproject


EXTRA_VALIDATIONS = (
    validate_project_dynamic,
    validate_include_depenency,
    validate_import_name_issues,
)
<|endoftext|>from __future__ import annotations

import io
import json
import logging
import os
import re
import typing
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Generator, Iterator, Sequence

from .fastjsonschema_exceptions import JsonSchemaValueException

if typing.TYPE_CHECKING:
    import sys

    if sys.version_info < (3, 11):
        from typing_extensions import Self
    else:
        from typing import Self

_logger = logging.getLogger(__name__)

_MESSAGE_REPLACEMENTS = {
    "must be named by propertyName definition": "keys must be named by",
    "one of contains definition": "at least one item that matches",
    " same as const definition:": "",
    "only specified items": "only items matching the definition",
}

_SKIP_DETAILS = (
    "must not be empty",
    "is always invalid",
    "must not be there",
)

_NEED_DETAILS = {"anyOf", "oneOf", "allOf", "contains", "propertyNames", "not", "items"}

_CAMEL_CASE_SPLITTER = re.compile(r"\W+|([A-Z][^A-Z\W]*)")
_IDENTIFIER = re.compile(r"^[\w_]+$", re.IGNORECASE)

_TOML_JARGON = {
    "object": "table",
    "property": "key",
    "properties": "keys",
    "property names": "keys",
}

_FORMATS_HELP = """
For more details about `format` see
https://validate-pyproject.readthedocs.io/en/latest/api/validate_pyproject.formats.html
"""


class ValidationError(JsonSchemaValueException):
    """Report violations of a given JSON schema.

    This class extends :exc:`~fastjsonschema.JsonSchemaValueException`
    by adding the following properties:

    - ``summary``: an improved version of the ``JsonSchemaValueException`` error message
      with only the necessary information)

    - ``details``: more contextual information about the error like the failing schema
      itself and the value that violates the schema.

    Depending on the level of the verbosity of the ``logging`` configuration
    the exception message will be only ``summary`` (default) or a combination of
    ``summary`` and ``details`` (when the logging level is set to :obj:`logging.DEBUG`).
    """

    summary = ""
    details = ""
    _original_message = ""

    @classmethod
    def _from_jsonschema(cls, ex: JsonSchemaValueException) -> Self:
        formatter = _ErrorFormatting(ex)
        obj = cls(str(formatter), ex.value, formatter.name, ex.definition, ex.rule)
        debug_code = os.getenv("JSONSCHEMA_DEBUG_CODE_GENERATION", "false").lower()
        if debug_code != "false":  # pragma: no cover
            obj.__cause__, obj.__traceback__ = ex.__cause__, ex.__traceback__
        obj._original_message = ex.message
        obj.summary = formatter.summary
        obj.details = formatter.details
        return obj


@contextmanager
def detailed_errors() -> Generator[None, None, None]:
    try:
        yield
    except JsonSchemaValueException as ex:
        raise ValidationError._from_jsonschema(ex) from None


class _ErrorFormatting:
    def __init__(self, ex: JsonSchemaValueException):
        self.ex = ex
        self.name = f"`{self._simplify_name(ex.name)}`"
        self._original_message: str = self.ex.message.replace(ex.name, self.name)
        self._summary = ""
        self._details = ""

    def __str__(self) -> str:
        if _logger.getEffectiveLevel() <= logging.DEBUG and self.details:
            return f"{self.summary}\n\n{self.details}"

        return self.summary

    @property
    def summary(self) -> str:
        if not self._summary:
            self._summary = self._expand_summary()

        return self._summary

    @property
    def details(self) -> str:
        if not self._details:
            self._details = self._expand_details()

        return self._details

    @staticmethod
    def _simplify_name(name: str) -> str:
        x = len("data.")
        return name[x:] if name.startswith("data.") else name

    def _expand_summary(self) -> str:
        msg = self._original_message

        for bad, repl in _MESSAGE_REPLACEMENTS.items():
            msg = msg.replace(bad, repl)

        if any(substring in msg for substring in _SKIP_DETAILS):
            return msg

        schema = self.ex.rule_definition
        if self.ex.rule in _NEED_DETAILS and schema:
            summary = _SummaryWriter(_TOML_JARGON)
            return f"{msg}:\n\n{indent(summary(schema), '    ')}"

        return msg

    def _expand_details(self) -> str:
        optional = []
        definition = self.ex.definition or {}
        desc_lines = definition.pop("$$description", [])
        desc = definition.pop("description", None) or " ".join(desc_lines)
        if desc:
            description = "\n".join(
                wrap(
                    desc,
                    width=80,
                    initial_indent="    ",
                    subsequent_indent="    ",
                    break_long_words=False,
                )
            )
            optional.append(f"DESCRIPTION:\n{description}")
        schema = json.dumps(definition, indent=4)
        value = json.dumps(self.ex.value, indent=4)
        defaults = [
            f"GIVEN VALUE:\n{indent(value, '    ')}",
            f"OFFENDING RULE: {self.ex.rule!r}",
            f"DEFINITION:\n{indent(schema, '    ')}",
        ]
        msg = "\n\n".join(optional + defaults)
        epilog = f"\n{_FORMATS_HELP}" if "format" in msg.lower() else ""
        return msg + epilog


class _SummaryWriter:
    _IGNORE = frozenset(("description", "default", "title", "examples"))

    def __init__(self, jargon: dict[str, str] | None = None):
        self.jargon: dict[str, str] = jargon or {}
        # Clarify confusing terms
        self._terms = {
            "anyOf": "at least one of the following",
            "oneOf": "exactly one of the following",
            "allOf": "all of the following",
            "not": "(*NOT* the following)",
            "prefixItems": f"{self._jargon('items')} (in order)",
            "items": "items",
            "contains": "contains at least one of",
            "propertyNames": (
                f"non-predefined acceptable {self._jargon('property names')}"
            ),
            "patternProperties": f"{self._jargon('properties')} named via pattern",
            "const": "predefined value",
            "enum": "one of",
        }
        # Attributes that indicate that the definition is easy and can be done
        # inline (e.g. string and number)
        self._guess_inline_defs = [
            "enum",
            "const",
            "maxLength",
            "minLength",
            "pattern",
            "format",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        ]

    def _jargon(self, term: str | list[str]) -> str | list[str]:
        if isinstance(term, list):
            return [self.jargon.get(t, t) for t in term]
        return self.jargon.get(term, term)

    def __call__(
        self,
        schema: dict | list[dict],
        prefix: str = "",
        *,
        _path: Sequence[str] = (),
    ) -> str:
        if isinstance(schema, list):
            return self._handle_list(schema, prefix, _path)

        filtered = self._filter_unecessary(schema, _path)
        simple = self._handle_simple_dict(filtered, _path)
        if simple:
            return f"{prefix}{simple}"

        child_prefix = self._child_prefix(prefix, "  ")
        item_prefix = self._child_prefix(prefix, "- ")
        indent = len(prefix) * " "
        with io.StringIO() as buffer:
            for i, (key, value) in enumerate(filtered.items()):
                child_path = [*_path, key]
                line_prefix = prefix if i == 0 else indent
                buffer.write(f"{line_prefix}{self._label(child_path)}:")
                # ^  just the first item should receive the complete prefix
                if isinstance(value, dict):
                    filtered = self._filter_unecessary(value, child_path)
                    simple = self._handle_simple_dict(filtered, child_path)
                    buffer.write(
                        f" {simple}"
                        if simple
                        else f"\n{self(value, child_prefix, _path=child_path)}"
                    )
                elif isinstance(value, list) and (
                    key != "type" or self._is_property(child_path)
                ):
                    children = self._handle_list(value, item_prefix, child_path)
                    sep = " " if children.startswith("[") else "\n"
                    buffer.write(f"{sep}{children}")
                else:
                    buffer.write(f" {self._value(value, child_path)}\n")
            return buffer.getvalue()

    def _is_unecessary(self, path: Sequence[str]) -> bool:
        if self._is_property(path) or not path:  # empty path => instruction @ root
            return False
        key = path[-1]
        return any(key.startswith(k) for k in "$_") or key in self._IGNORE

    def _filter_unecessary(
        self, schema: dict[str, Any], path: Sequence[str]
    ) -> dict[str, Any]:
        return {
            key: value
            for key, value in schema.items()
            if not self._is_unecessary([*path, key])
        }

    def _handle_simple_dict(self, value: dict, path: Sequence[str]) -> str | None:
        inline = any(p in value for p in self._guess_inline_defs)
        simple = not any(isinstance(v, (list, dict)) for v in value.values())
        if inline or simple:
            return f"{{{', '.join(self._inline_attrs(value, path))}}}\n"
        return None

    def _handle_list(
        self, schemas: list, prefix: str = "", path: Sequence[str] = ()
    ) -> str:
        if self._is_unecessary(path):
            return ""

        repr_ = repr(schemas)
        if all(not isinstance(e, (dict, list)) for e in schemas) and len(repr_) < 60:
            return f"{repr_}\n"

        item_prefix = self._child_prefix(prefix, "- ")
        return "".join(
            self(v, item_prefix, _path=[*path, f"[{i}]"]) for i, v in enumerate(schemas)
        )

    def _is_property(self, path: Sequence[str]) -> bool:
        """Check if the given path can correspond to an arbitrarily named property"""
        counter = 0
        for key in path[-2::-1]:
            if key not in {"properties", "patternProperties"}:
                break
            counter += 1

        # If the counter if even, the path correspond to a JSON Schema keyword
        # otherwise it can be any arbitrary string naming a property
        return counter % 2 == 1

    def _label(self, path: Sequence[str]) -> str:
        *parents, key = path
        if not self._is_property(path):
            norm_key = _separate_terms(key)
            return self._terms.get(key) or " ".join(self._jargon(norm_key))

        if parents[-1] == "patternProperties":
            return f"(regex {key!r})"
        return repr(key)  # property name

    def _value(self, value: Any, path: Sequence[str]) -> str:
        if path[-1] == "type" and not self._is_property(path):
            type_ = self._jargon(value)
            return f"[{', '.join(type_)}]" if isinstance(type_, list) else type_
        return repr(value)

