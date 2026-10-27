                    self.call(
                        self.compiler_so + cc_args + [src, '-o', obj] + extra_postargs
                    )
            except (subprocess.CalledProcessError, OSError) as msg:
                raise CompileError(msg)

    def link(
        self,
        target_desc,
        objects,
        output_filename,
        output_dir=None,
        libraries=None,
        library_dirs=None,
        runtime_library_dirs=None,
        export_symbols=None,
        debug=False,
        extra_preargs=None,
        extra_postargs=None,
        build_temp=None,
        target_lang=None,
    ):
        """Link the objects."""
        # use separate copies, so we can modify the lists
        extra_preargs = copy.copy(extra_preargs or [])
        libraries = copy.copy(libraries or [])
        objects = copy.copy(objects or [])

        if runtime_library_dirs:
            self.warn(_runtime_library_dirs_msg)

        # Additional libraries
        libraries.extend(self.dll_libraries)

        # handle export symbols by creating a def-file
        # with executables this only works with gcc/ld as linker
        if (export_symbols is not None) and (
            target_desc != self.EXECUTABLE or self.linker_dll == "gcc"
        ):
            # (The linker doesn't do anything if output is up-to-date.
            # So it would probably better to check if we really need this,
            # but for this we had to insert some unchanged parts of
            # UnixCCompiler, and this is not what we want.)

            # we want to put some files in the same directory as the
            # object files are, build_temp doesn't help much
            # where are the object files
            temp_dir = os.path.dirname(objects[0])
            # name of dll to give the helper files the same base name
            (dll_name, _dll_extension) = os.path.splitext(
                os.path.basename(output_filename)
            )

            # generate the filenames for these files
            def_file = os.path.join(temp_dir, dll_name + ".def")

            # Generate .def file
            contents = [f"LIBRARY {os.path.basename(output_filename)}", "EXPORTS"]
            contents.extend(export_symbols)
            log.info("writing %s", def_file)
            pathlib.Path(def_file).write_text('\n'.join(contents) + '\n')

            # next add options for def-file

            # for gcc/ld the def-file is specified as any object files
            objects.append(def_file)

        # end: if ((export_symbols is not None) and
        #        (target_desc != self.EXECUTABLE or self.linker_dll == "gcc")):

        # who wants symbols and a many times larger output file
        # should explicitly switch the debug mode on
        # otherwise we let ld strip the output file
        # (On my machine: 10KiB < stripped_file < ??100KiB
        #   unstripped_file = stripped_file + XXX KiB
        #  ( XXX=254 for a typical python extension))
        if not debug:
            extra_preargs.append("-s")

        super().link(
            target_desc,
            objects,
            output_filename,
            output_dir,
            libraries,
            library_dirs,
            runtime_library_dirs,
            None,  # export_symbols, we do this in our def-file
            debug,
            extra_preargs,
            extra_postargs,
            build_temp,
            target_lang,
        )

    def runtime_library_dir_option(self, dir):
        # cygwin doesn't support rpath. While in theory we could error
        # out like MSVC does, code might expect it to work like on Unix, so
        # just warn and hope for the best.
        self.warn(_runtime_library_dirs_msg)
        return []

    # -- Miscellaneous methods -----------------------------------------

    def _make_out_path(self, output_dir, strip_dir, src_name):
        # use normcase to make sure '.rc' is really '.rc' and not '.RC'
        norm_src_name = os.path.normcase(src_name)
        return super()._make_out_path(output_dir, strip_dir, norm_src_name)

    @property
    def out_extensions(self):
        """
        Add support for rc and res files.
        """
        return {
            **super().out_extensions,
            **{ext: ext + self.obj_extension for ext in ('.res', '.rc')},
        }


# the same as cygwin plus some additional parameters
class MinGW32Compiler(Compiler):
    """Handles the Mingw32 port of the GNU C compiler to Windows."""

    compiler_type = 'mingw32'
    description = "Mingw32 port of GNU C Compiler for Win32"

    def __init__(self, verbose=False, force=False):
        super().__init__(verbose, force)

        shared_option = "-shared"

        if is_cygwincc(self.cc):
            raise Error('Cygwin gcc cannot be used with --compiler=mingw32')

        self.set_executables(
            compiler=f'{self.cc} -O1 -Wall',
            compiler_so=f'{self.cc} -shared -O1 -Wall',
            compiler_so_cxx=f'{self.cxx} -shared -O1 -Wall',
            compiler_cxx=f'{self.cxx} -O1 -Wall',
            linker_exe=f'{self.cc}',
            linker_so=f'{self.linker_dll} {shared_option}',
            linker_exe_cxx=f'{self.cxx}',
            linker_so_cxx=f'{self.linker_dll_cxx} {shared_option}',
        )

    def configure_system(self) -> None:
        # Only apply the interpreter's Unix-style build configuration when
        # actually running under a mingw Python; on an MSVC Python those
        # settings don't apply.
        if is_mingw():
            super().configure_system()

    def runtime_library_dir_option(self, dir):
        raise PlatformError(_runtime_library_dirs_msg)


# Because these compilers aren't configured in Python's pyconfig.h file by
# default, we should at least warn the user if he is using an unmodified
# version.

CONFIG_H_OK = "ok"
CONFIG_H_NOTOK = "not ok"
CONFIG_H_UNCERTAIN = "uncertain"


def check_config_h():
    """Check if the current Python installation appears amenable to building
    extensions with GCC.

    Returns a tuple (status, details), where 'status' is one of the following
    constants:

    - CONFIG_H_OK: all is well, go ahead and compile
    - CONFIG_H_NOTOK: doesn't look good
    - CONFIG_H_UNCERTAIN: not sure -- unable to read pyconfig.h

    'details' is a human-readable string explaining the situation.

    Note there are two ways to conclude "OK": either 'sys.version' contains
    the string "GCC" (implying that this Python was built with GCC), or the
    installed "pyconfig.h" contains the string "__GNUC__".
    """

    # XXX since this function also checks sys.version, it's not strictly a
    # "pyconfig.h" check -- should probably be renamed...

    import sysconfig

    # if sys.version contains GCC then python was compiled with GCC, and the
    # pyconfig.h file should be OK
    if "GCC" in sys.version:
        return CONFIG_H_OK, "sys.version mentions 'GCC'"

    # Clang would also work
    if "Clang" in sys.version:
        return CONFIG_H_OK, "sys.version mentions 'Clang'"

    # let's see if __GNUC__ is mentioned in python.h
    fn = sysconfig.get_config_h_filename()
    try:
        config_h = pathlib.Path(fn).read_text(encoding='utf-8')
    except OSError as exc:
        return (CONFIG_H_UNCERTAIN, f"couldn't read '{fn}': {exc.strerror}")
    else:
        substring = '__GNUC__'
        if substring in config_h:
            code = CONFIG_H_OK
            mention_inflected = 'mentions'
        else:
            code = CONFIG_H_NOTOK
            mention_inflected = 'does not mention'
        return code, f"{fn!r} {mention_inflected} {substring!r}"


def is_cygwincc(cc: str | shlex._ShlexInstream) -> bool:
    """Try to determine if the compiler that would be used is from cygwin."""
    out_string = subprocess.check_output(shlex.split(cc) + ['-dumpmachine'])
    return out_string.strip().endswith(b'cygwin')


get_versions = None
"""
A stand-in for the previous get_versions() function to prevent failures
when monkeypatched. See pypa/setuptools#2969.
"""
<|endoftext|>"""distutils.ccompiler

Contains Compiler, an abstract base class that defines the interface
for the Distutils compiler abstraction model."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    TypeVar,
    overload,
)

from more_itertools import always_iterable

from .._modified import newer_group
from .._util import split_quoted
from ..errors import PlatformError, UnknownFileType
from ..logging import get_logger
from ..platform import macos
from ..platform.detect import is_mingw
from .errors import CompileError, LinkError

log = get_logger(__name__)

if TYPE_CHECKING:
    from subprocess import _ENV
    from typing import TypeAlias

    from typing_extensions import TypeVarTuple, Unpack

    _Ts = TypeVarTuple("_Ts")

_Macro: TypeAlias = tuple[str] | tuple[str, str | None]
_StrPathT = TypeVar("_StrPathT", bound="str | os.PathLike[str]")
_BytesPathT = TypeVar("_BytesPathT", bound="bytes | os.PathLike[bytes]")


class Compiler:
    """Abstract base class to define the interface that must be implemented
    by real compiler classes.  Also has some utility methods used by
    several compiler classes.

    The basic idea behind a compiler abstraction class is that each
    instance can be used for all the compile/link steps in building a
    single project.  Thus, attributes common to all of those compile and
    link steps -- include directories, macros to define, libraries to link
    against, etc. -- are attributes of the compiler instance.  To allow for
    variability in how individual files are treated, most of those
    attributes may be varied on a per-compilation or per-link basis.
    """

    compiler_type: ClassVar[str]
    """
    Short name identifying the kind of compiler, so callers can tell
    compilers apart without importing every compiler class for an
    ``isinstance`` check. Set this in each concrete subclass; it's the key
    under which the class is registered by ``get_compilers()``.
    """

    description: ClassVar[str]
    """
    Human-readable description of the compiler, shown by ``show_compilers()``.
    Set this in each concrete subclass.
    """

    # XXX things not handled by this compiler abstraction model:
    #   * client can't provide additional options for a compiler,
    #     e.g. warning, optimization, debugging flags.  Perhaps this
    #     should be the domain of concrete compiler abstraction classes
    #     (UnixCCompiler, MSVCCompiler, etc.) -- or perhaps the base
    #     class should have methods for the common ones.
    #   * can't completely override the include or library search
    #     path, ie. no "cc -I -Idir1 -Idir2" or "cc -L -Ldir1 -Ldir2".
    #     I'm not sure how widely supported this is even by Unix
    #     compilers, much less on other platforms.  And I'm even less
    #     sure how useful it is; maybe for cross-compiling, but
    #     support for that is a ways off.  (And anyways, cross
    #     compilers probably have a dedicated binary with the
    #     right paths compiled in.  I hope.)
    #   * can't do really freaky things with the library list/library
    #     dirs, e.g. "-Ldir1 -lfoo -Ldir2 -lfoo" to link against
    #     different versions of libfoo.a in different locations.  I
    #     think this is useless without the ability to null out the
    #     library search path anyways.

    executables: ClassVar[dict]

    # Subclasses that rely on the standard filename generation methods
    # implemented below should override these; see the comment near
    # those methods ('object_filenames()' et. al.) for details:
    src_extensions: ClassVar[list[str] | None] = None
    obj_extension: ClassVar[str | None] = None
    static_lib_extension: ClassVar[str | None] = None
    shared_lib_extension: ClassVar[str | None] = None
    static_lib_format: ClassVar[str | None] = None  # format string
    shared_lib_format: ClassVar[str | None] = None  # prob. same as static_lib_format
    exe_extension: ClassVar[str | None] = None

    # Default language settings. language_map is used to detect a source
    # file or Extension target language, checking source filenames.
    # language_order is used to detect the language precedence, when deciding
    # what language to use when mixing source types. For example, if some
    # extension has two files with ".c" extension, and one with ".cpp", it
    # is still linked as c++.
    language_map: ClassVar[dict[str, str]] = {
        ".c": "c",
        ".cc": "c++",
        ".cpp": "c++",
        ".cxx": "c++",
        ".m": "objc",
    }
    language_order: ClassVar[list[str]] = ["c++", "objc", "c"]

    # Not ClassVar: instances reassign these in __init__ and set_*_dirs.
    include_dirs: list[str] = []  # noqa: RUF012 # class-level default, overridden per instance
    """
    include dirs specific to this compiler class
    """

    library_dirs: list[str] = []  # noqa: RUF012 # class-level default, overridden per instance
    """
    library dirs specific to this compiler class
    """

    def __init__(self, verbose: bool = False, force: bool = False) -> None:
        self.force = force
        self.verbose = verbose

        # 'output_dir': a common output directory for object, library,
        # shared object, and shared library files
        self.output_dir: str | None = None

        # 'macros': a list of macro definitions (or undefinitions).  A
        # macro definition is a 2-tuple (name, value), where the value is
        # either a string or None (no explicit value).  A macro
        # undefinition is a 1-tuple (name,).
        self.macros: list[_Macro] = []

        # 'include_dirs': a list of directories to search for include files
        self.include_dirs = []

        # 'libraries': a list of libraries to include in any link
        # (library names, not filenames: eg. "foo" not "libfoo.a")
        self.libraries: list[str] = []

        # 'library_dirs': a list of directories to search for libraries
        self.library_dirs = []

        # 'runtime_library_dirs': a list of directories to search for
        # shared libraries/objects at runtime
        self.runtime_library_dirs: list[str] = []

        # 'objects': a list of object files (or similar, such as explicitly
        # named library files) to include on any link
        self.objects: list[str] = []

        for key in self.executables:
            self.set_executable(key, self.executables[key])

    def initialize(self, plat_name: str | None = None) -> None:
        """Prepare the compiler for use, targeting the given platform.

        Most compilers configure themselves lazily on first use and so
        need no explicit initialization; for those this is a no-op. A
        compiler that must be initialized before use (e.g. MSVCCompiler,
        which resolves a Visual Studio environment) overrides this to
        honor ``plat_name`` when cross-compiling.
        """

    def configure_system(self) -> None:
        """Configure this compiler from the interpreter's build configuration.

        The default is a no-op; Unix-style compilers override this to apply
        the compiler, flag, and archiver settings that CPython recorded in
        sysconfig when it was built, so extensions build consistently with the
        interpreter.
        """

    def set_executables(self, **kwargs: str) -> None:
        """Define the executables (and options for them) that will be run
        to perform the various stages of compilation.  The exact set of
        executables that may be specified here depends on the compiler
        class (via the 'executables' class attribute), but most will have:
          compiler      the C/C++ compiler
          linker_so     linker used to create shared objects and libraries
          linker_exe    linker used to create binary executables
          archiver      static library creator

        On platforms with a command-line (Unix, DOS/Windows), each of these
        is a string that will be split into executable name and (optional)
        list of arguments.  (Splitting the string is done similarly to how
        Unix shells operate: words are delimited by spaces, but quotes and
        backslashes can override this.  See
        'distutils.util.split_quoted()'.)
        """

        # Note that some CCompiler implementation classes will define class
        # attributes 'cpp', 'cc', etc. with hard-coded executable names;
        # this is appropriate when a compiler class is for exactly one
        # compiler/OS combination (eg. MSVCCompiler).  Other compiler
        # classes (UnixCCompiler, in particular) are driven by information
        # discovered at run-time, since there are many different ways to do
        # basically the same things with Unix C compilers.

        for key, value in kwargs.items():
            if key not in self.executables:
                raise ValueError(
                    f"unknown executable '{key}' for class {self.__class__.__name__}"
                )
            self.set_executable(key, value)

    def set_executable(self, key, value):
        if isinstance(value, str):
            setattr(self, key, split_quoted(value))
        else:
            setattr(self, key, value)

    def _find_macro(self, name):
        for i, defn in enumerate(self.macros):
            if defn[0] == name:
                return i
        return None

    def _check_macro_definitions(self, definitions):
        """Ensure that every element of 'definitions' is valid."""
        for defn in definitions:
            self._check_macro_definition(*defn)

    def _check_macro_definition(self, defn):
        """
        Raise a TypeError if defn is not valid.

        A valid definition is either a (name, value) 2-tuple or a (name,) tuple.
        """
        if not isinstance(defn, tuple) or not self._is_valid_macro(*defn):
            raise TypeError(
                f"invalid macro definition '{defn}': "
                "must be tuple (string,), (string, string), or (string, None)"
            )

    @staticmethod
    def _is_valid_macro(name, value=None):
        """
        A valid macro is a ``name : str`` and a ``value : str | None``.

        >>> Compiler._is_valid_macro('foo', None)
        True
        """
        return isinstance(name, str) and isinstance(value, (str, type(None)))

    # -- Bookkeeping methods -------------------------------------------

    def define_macro(self, name: str, value: str | None = None) -> None:
        """Define a preprocessor macro for all compilations driven by this
        compiler object.  The optional parameter 'value' should be a
        string; if it is not supplied, then the macro will be defined
        without an explicit value and the exact outcome depends on the
        compiler used (XXX true? does ANSI say anything about this?)
        """
        # Delete from the list of macro definitions/undefinitions if
        # already there (so that this one will take precedence).
        i = self._find_macro(name)
        if i is not None:
            del self.macros[i]

        self.macros.append((name, value))

    def undefine_macro(self, name: str) -> None:
        """Undefine a preprocessor macro for all compilations driven by
        this compiler object.  If the same macro is defined by
        'define_macro()' and undefined by 'undefine_macro()' the last call
        takes precedence (including multiple redefinitions or
        undefinitions).  If the macro is redefined/undefined on a
        per-compilation basis (ie. in the call to 'compile()'), then that
        takes precedence.
        """
        # Delete from the list of macro definitions/undefinitions if
        # already there (so that this one will take precedence).
        i = self._find_macro(name)
        if i is not None:
            del self.macros[i]

        undefn = (name,)
        self.macros.append(undefn)

    def add_include_dir(self, dir: str) -> None:
        """Add 'dir' to the list of directories that will be searched for
        header files.  The compiler is instructed to search directories in
        the order in which they are supplied by successive calls to
        'add_include_dir()'.
        """
        self.include_dirs.append(dir)

    def set_include_dirs(self, dirs: list[str]) -> None:
        """Set the list of directories that will be searched to 'dirs' (a
        list of strings).  Overrides any preceding calls to
        'add_include_dir()'; subsequence calls to 'add_include_dir()' add
        to the list passed to 'set_include_dirs()'.  This does not affect
        any list of standard include directories that the compiler may
        search by default.
        """
        self.include_dirs = dirs[:]

    def add_library(self, libname: str) -> None:
        """Add 'libname' to the list of libraries that will be included in
        all links driven by this compiler object.  Note that 'libname'
        should *not* be the name of a file containing a library, but the
        name of the library itself: the actual filename will be inferred by
        the linker, the compiler, or the compiler class (depending on the
        platform).

        The linker will be instructed to link against libraries in the
        order they were supplied to 'add_library()' and/or
        'set_libraries()'.  It is perfectly valid to duplicate library
        names; the linker will be instructed to link against libraries as
        many times as they are mentioned.
        """
        self.libraries.append(libname)

    def set_libraries(self, libnames: list[str]) -> None:
        """Set the list of libraries to be included in all links driven by
        this compiler object to 'libnames' (a list of strings).  This does
        not affect any standard system libraries that the linker may
        include by default.
        """
        self.libraries = libnames[:]

    def add_library_dir(self, dir: str) -> None:
        """Add 'dir' to the list of directories that will be searched for
        libraries specified to 'add_library()' and 'set_libraries()'.  The
        linker will be instructed to search for libraries in the order they
        are supplied to 'add_library_dir()' and/or 'set_library_dirs()'.
        """
        self.library_dirs.append(dir)

    def set_library_dirs(self, dirs: list[str]) -> None:
        """Set the list of library search directories to 'dirs' (a list of
        strings).  This does not affect any standard library search path
        that the linker may search by default.
        """
        self.library_dirs = dirs[:]

    def add_runtime_library_dir(self, dir: str) -> None:
        """Add 'dir' to the list of directories that will be searched for
        shared libraries at runtime.
        """
        self.runtime_library_dirs.append(dir)

    def set_runtime_library_dirs(self, dirs: list[str]) -> None:
        """Set the list of directories to search for shared libraries at
        runtime to 'dirs' (a list of strings).  This does not affect any
        standard search path that the runtime linker may search by
        default.
        """
        self.runtime_library_dirs = dirs[:]

    def add_link_object(self, object: str) -> None:
        """Add 'object' to the list of object files (or analogues, such as
        explicitly named library files or the output of "resource
        compilers") to be included in every link driven by this compiler
        object.
        """
        self.objects.append(object)

    def set_link_objects(self, objects: list[str]) -> None:
        """Set the list of object files (or analogues) to be included in
        every link to 'objects'.  This does not affect any standard object
        files that the linker may include by default (such as system
        libraries).
        """
        self.objects = objects[:]

    # -- Private utility methods --------------------------------------
    # (here for the convenience of subclasses)

    # Helper method to prep compiler in subclass compile() methods

    def _setup_compile(
        self,
        outdir: str | None,
        macros: list[_Macro] | None,
        incdirs: list[str] | tuple[str, ...] | None,
        sources,
        depends,
        extra,
    ):
        """Process arguments and decide which source files to compile."""
        outdir, macros, incdirs = self._fix_compile_args(outdir, macros, incdirs)

        if extra is None:
            extra = []

        # Get the list of expected output (object) files
        objects = self.object_filenames(sources, strip_dir=False, output_dir=outdir)
        assert len(objects) == len(sources)

        pp_opts = gen_preprocess_options(macros, incdirs)

        build = {}
        for i in range(len(sources)):
            src = sources[i]
            obj = objects[i]
            ext = os.path.splitext(src)[1]
            self.mkpath(os.path.dirname(obj))
            build[obj] = (src, ext)

        return macros, objects, extra, pp_opts, build

    def _get_cc_args(self, pp_opts, debug, before):
        # works for unixccompiler, cygwinccompiler
        cc_args = pp_opts + ['-c']
        if debug:
            cc_args[:0] = ['-g']
        if before:
            cc_args[:0] = before
        return cc_args

    def _fix_compile_args(
        self,
        output_dir: str | None,
        macros: list[_Macro] | None,
        include_dirs: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, list[_Macro], list[str]]:
        """Typecheck and fix-up some of the arguments to the 'compile()'
        method, and return fixed-up values.  Specifically: if 'output_dir'
        is None, replaces it with 'self.output_dir'; ensures that 'macros'
        is a list, and augments it with 'self.macros'; ensures that
        'include_dirs' is a list, and augments it with 'self.include_dirs'.
        Guarantees that the returned values are of the correct type,
        i.e. for 'output_dir' either string or None, and for 'macros' and
        'include_dirs' either list or None.
        """
        if output_dir is None:
            output_dir = self.output_dir
        elif not isinstance(output_dir, str):
            raise TypeError("'output_dir' must be a string or None")

        if macros is None:
            macros = list(self.macros)
        elif isinstance(macros, list):
            macros = macros + (self.macros or [])
        else:
            raise TypeError("'macros' (if supplied) must be a list of tuples")

        if include_dirs is None:
            include_dirs = list(self.include_dirs)
        elif isinstance(include_dirs, (list, tuple)):
            include_dirs = list(include_dirs) + (self.include_dirs or [])
        else:
            raise TypeError("'include_dirs' (if supplied) must be a list of strings")

        # add include dirs for class
        include_dirs += self.__class__.include_dirs

        return output_dir, macros, include_dirs

    def _prep_compile(self, sources, output_dir, depends=None):
        """Decide which source files must be recompiled.

        Determine the list of object files corresponding to 'sources',
        and figure out which ones really need to be recompiled.
        Return a list of all object files and a dictionary telling
        which source files can be skipped.
        """
        # Get the list of expected output (object) files
        objects = self.object_filenames(sources, output_dir=output_dir)
        assert len(objects) == len(sources)

        # Return an empty dict for the "which source files can be skipped"
        # return value to preserve API compatibility.
        return objects, {}

    def _fix_object_args(
        self, objects: list[str] | tuple[str, ...], output_dir: str | None
    ) -> tuple[list[str], str]:
        """Typecheck and fix up some arguments supplied to various methods.
        Specifically: ensure that 'objects' is a list; if output_dir is
        None, replace with self.output_dir.  Return fixed versions of
        'objects' and 'output_dir'.
        """
        if not isinstance(objects, (list, tuple)):
            raise TypeError("'objects' must be a list or tuple of strings")
        objects = list(objects)

        if output_dir is None:
            output_dir = self.output_dir
        elif not isinstance(output_dir, str):
            raise TypeError("'output_dir' must be a string or None")

        return (objects, output_dir)

    def _fix_lib_args(
        self,
        libraries: list[str] | tuple[str, ...] | None,
        library_dirs: list[str] | tuple[str, ...] | None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Typecheck and fix up some of the arguments supplied to the
        'link_*' methods.  Specifically: ensure that all arguments are
        lists, and augment them with their permanent versions
        (eg. 'self.libraries' augments 'libraries').  Return a tuple with
        fixed versions of all arguments.
        """
        if libraries is None:
            libraries = list(self.libraries)
        elif isinstance(libraries, (list, tuple)):
            libraries = list(libraries) + (self.libraries or [])
        else:
            raise TypeError("'libraries' (if supplied) must be a list of strings")

        if library_dirs is None:
            library_dirs = list(self.library_dirs)
        elif isinstance(library_dirs, (list, tuple)):
            library_dirs = list(library_dirs) + (self.library_dirs or [])
        else:
            raise TypeError("'library_dirs' (if supplied) must be a list of strings")

        # add library dirs for class
        library_dirs += self.__class__.library_dirs

        if runtime_library_dirs is None:
            runtime_library_dirs = list(self.runtime_library_dirs)
        elif isinstance(runtime_library_dirs, (list, tuple)):
            runtime_library_dirs = list(runtime_library_dirs) + (
                self.runtime_library_dirs or []
            )
        else:
            raise TypeError(
                "'runtime_library_dirs' (if supplied) must be a list of strings"
            )

        return (libraries, library_dirs, runtime_library_dirs)

    def _need_link(self, objects, output_file):
        """Return true if we need to relink the files listed in 'objects'
        to recreate 'output_file'.
        """
        if self.force:
            return True
        newer = newer_group(objects, output_file)
        return newer

    def detect_language(self, sources: str | list[str]) -> str | None:
        """Detect the language of a given file, or list of files. Uses
        language_map, and language_order to do the job.
        """
        if not isinstance(sources, list):
            sources = [sources]
        lang = None
        index = len(self.language_order)
        for source in sources:
            _base, ext = os.path.splitext(source)
            extlang = self.language_map.get(ext)
            try:
                extindex = self.language_order.index(extlang)
                if extindex < index:
                    lang = extlang
                    index = extindex
            except ValueError:
                pass
        return lang

    # -- Worker methods ------------------------------------------------
    # (must be implemented by subclasses)

    def preprocess(
        self,
        source: str | os.PathLike[str],
        output_file: str | os.PathLike[str] | None = None,
        macros: list[_Macro] | None = None,
        include_dirs: list[str] | tuple[str, ...] | None = None,
        extra_preargs: list[str] | None = None,
        extra_postargs: Iterable[str] | None = None,
    ):
        """Preprocess a single C/C++ source file, named in 'source'.
        Output will be written to file named 'output_file', or stdout if
        'output_file' not supplied.  'macros' is a list of macro
        definitions as for 'compile()', which will augment the macros set
        with 'define_macro()' and 'undefine_macro()'.  'include_dirs' is a
        list of directory names that will be added to the default list.

        Raises PreprocessError on failure.
        """

    def compile(
        self,
        sources: Sequence[str | os.PathLike[str]],
        output_dir: str | None = None,
        macros: list[_Macro] | None = None,
        include_dirs: list[str] | tuple[str, ...] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        depends: list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        """Compile one or more source files.

        'sources' must be a list of filenames, most likely C/C++
        files, but in reality anything that can be handled by a
        particular compiler and compiler class (eg. MSVCCompiler can
        handle resource files in 'sources').  Return a list of object
        filenames, one per source filename in 'sources'.  Depending on
        the implementation, not all source files will necessarily be
        compiled, but all corresponding object filenames will be
        returned.

        If 'output_dir' is given, object files will be put under it, while
        retaining their original path component.  That is, "foo/bar.c"
        normally compiles to "foo/bar.o" (for a Unix implementation); if
        'output_dir' is "build", then it would compile to
        "build/foo/bar.o".

        'macros', if given, must be a list of macro definitions.  A macro
        definition is either a (name, value) 2-tuple or a (name,) 1-tuple.
        The former defines a macro; if the value is None, the macro is
        defined without an explicit value.  The 1-tuple case undefines a
        macro.  Later definitions/redefinitions/ undefinitions take
        precedence.

        'include_dirs', if given, must be a list of strings, the
        directories to add to the default include file search path for this
        compilation only.

        'debug' is a boolean; if true, the compiler will be instructed to
        output debug symbols in (or alongside) the object file(s).

        'extra_preargs' and 'extra_postargs' are implementation- dependent.
        On platforms that have the notion of a command-line (e.g. Unix,
        DOS/Windows), they are most likely lists of strings: extra
        command-line arguments to prepend/append to the compiler command
        line.  On other platforms, consult the implementation class
        documentation.  In any event, they are intended as an escape hatch
        for those occasions when the abstract compiler framework doesn't
        cut the mustard.

        'depends', if given, is a list of filenames that all targets
        depend on.  If a source file is older than any file in
        depends, then the source file will be recompiled.  This
        supports dependency tracking, but only at a coarse
        granularity.

        Raises CompileError on failure.
        """
        # A concrete compiler class can either override this method
        # entirely or implement _compile().
        macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs
        )
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

        for obj in objects:
            try:
                src, ext = build[obj]
            except KeyError:
                continue
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        # Return *all* object filenames, not just the ones we just built.
        return objects

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        """Compile 'src' to product 'obj'."""
        # A concrete compiler class that does not override compile()
        # should implement _compile().

    def create_static_lib(
        self,
        objects: list[str] | tuple[str, ...],
        output_libname: str,
        output_dir: str | None = None,
        debug: bool = False,
        target_lang: str | None = None,
    ) -> None:
        """Link a bunch of stuff together to create a static library file.
        The "bunch of stuff" consists of the list of object files supplied
        as 'objects', the extra object files supplied to
        'add_link_object()' and/or 'set_link_objects()', the libraries
        supplied to 'add_library()' and/or 'set_libraries()', and the
        libraries supplied as 'libraries' (if any).

        'output_libname' should be a library name, not a filename; the
        filename will be inferred from the library name.  'output_dir' is
        the directory where the library file will be put.

        'debug' is a boolean; if true, debugging information will be
        included in the library (note that on most platforms, it is the
        compile step where this matters: the 'debug' flag is included here
        just for consistency).

        'target_lang' is the target language for which the given objects
        are being compiled. This allows specific linkage time treatment of
        certain languages.

        Raises LibError on failure.
        """

    # values for target_desc parameter in link()
    SHARED_OBJECT = "shared_object"
    SHARED_LIBRARY = "shared_library"
    EXECUTABLE = "executable"

    def link(
        self,
        target_desc: str,
        objects: list[str] | tuple[str, ...],
        output_filename: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        export_symbols: Iterable[str] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        build_temp: str | os.PathLike[str] | None = None,
        target_lang: str | None = None,
    ):
        """Link a bunch of stuff together to create an executable or
        shared library file.

        The "bunch of stuff" consists of the list of object files supplied
        as 'objects'.  'output_filename' should be a filename.  If
        'output_dir' is supplied, 'output_filename' is relative to it
        (i.e. 'output_filename' can provide directory components if
        needed).

        'libraries' is a list of libraries to link against.  These are
        library names, not filenames, since they're translated into
        filenames in a platform-specific way (eg. "foo" becomes "libfoo.a"
        on Unix and "foo.lib" on DOS/Windows).  However, they can include a
        directory component, which means the linker will look in that
        specific directory rather than searching all the normal locations.

        'library_dirs', if supplied, should be a list of directories to
        search for libraries that were specified as bare library names
        (ie. no directory component).  These are on top of the system
        default and those supplied to 'add_library_dir()' and/or
        'set_library_dirs()'.  'runtime_library_dirs' is a list of
        directories that will be embedded into the shared library and used
        to search for other shared libraries that *it* depends on at
        run-time.  (This may only be relevant on Unix.)

        'export_symbols' is a list of symbols that the shared library will
        export.  (This appears to be relevant only on Windows.)

        'debug' is as for 'compile()' and 'create_static_lib()', with the
        slight distinction that it actually matters on most platforms (as
        opposed to 'create_static_lib()', which includes a 'debug' flag
        mostly for form's sake).

        'extra_preargs' and 'extra_postargs' are as for 'compile()' (except
        of course that they supply command-line arguments for the
        particular linker being used).

        'target_lang' is the target language for which the given objects
        are being compiled. This allows specific linkage time treatment of
        certain languages.

        Raises LinkError on failure.
        """
        raise NotImplementedError

    # Old 'link_*()' methods, rewritten to use the new 'link()' method.

    def link_shared_lib(
        self,
        objects: list[str] | tuple[str, ...],
        output_libname: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        export_symbols: Iterable[str] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        build_temp: str | os.PathLike[str] | None = None,
        target_lang: str | None = None,
    ):
        self.link(
            Compiler.SHARED_LIBRARY,
            objects,
            self.library_filename(output_libname, lib_type='shared'),
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

    def link_shared_object(
        self,
        objects: list[str] | tuple[str, ...],
        output_filename: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        export_symbols: Iterable[str] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        build_temp: str | os.PathLike[str] | None = None,
        target_lang: str | None = None,
    ):
        self.link(
            Compiler.SHARED_OBJECT,
            objects,
            output_filename,
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

    def link_executable(
        self,
        objects: list[str] | tuple[str, ...],
        output_progname: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        target_lang: str | None = None,
    ):
        self.link(
            Compiler.EXECUTABLE,
            objects,
            self.executable_filename(output_progname),
            output_dir,
            libraries,
            library_dirs,
            runtime_library_dirs,
            None,
            debug,
            extra_preargs,
            extra_postargs,
            None,
            target_lang,
        )

    # -- Miscellaneous methods -----------------------------------------
    # These are all used by the 'gen_lib_options() function; there is
    # no appropriate default implementation so subclasses should
    # implement all of these.

    def library_dir_option(self, dir: str) -> str:
        """Return the compiler option to add 'dir' to the list of
        directories searched for libraries.
        """
        raise NotImplementedError

    def runtime_library_dir_option(self, dir: str) -> str | list[str]:
        """Return the compiler option to add 'dir' to the list of
        directories searched for runtime libraries.
        """
        raise NotImplementedError

    def library_option(self, lib: str) -> str:
        """Return the compiler option to add 'lib' to the list of libraries
        linked into the shared library or executable.
        """
        raise NotImplementedError

    def has_function(  # noqa: C901
        self,
        funcname: str,
        includes: Iterable[str] | None = None,
        include_dirs: list[str] | tuple[str, ...] | None = None,
        libraries: list[str] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        """Return a boolean indicating whether funcname is provided as
        a symbol on the current platform.  The optional arguments can
        be used to augment the compilation environment.

        The libraries argument is a list of flags to be passed to the
        linker to make additional symbol definitions available for
        linking.

        The includes and include_dirs arguments are deprecated.
        Usually, supplying include files with function declarations
        will cause function detection to fail even in cases where the
        symbol is available for linking.

        """
        # this can't be included at module scope because it tries to
        # import math which might not be available at that point - maybe
        # the necessary logic should just be inlined?
        import tempfile

        if includes is None:
            includes = []
        else:
            warnings.warn("includes is deprecated", DeprecationWarning)
        if include_dirs is None:
            include_dirs = []
        else:
            warnings.warn("include_dirs is deprecated", DeprecationWarning)
        if libraries is None:
            libraries = []
        if library_dirs is None:
            library_dirs = []
        fd, fname = tempfile.mkstemp(".c", funcname, text=True)
        with os.fdopen(fd, "w", encoding='utf-8') as f:
            for incl in includes:
                f.write(f"""#include "{incl}"\n""")
            if not includes:
                # Use "char func(void);" as the prototype to follow
                # what autoconf does.  This prototype does not match
                # any well-known function the compiler might recognize
                # as a builtin, so this ends up as a true link test.
                # Without a fake prototype, the test would need to
                # know the exact argument types, and the has_function
                # interface does not provide that level of information.
                f.write(
                    f"""\
#ifdef __cplusplus
extern "C"
#endif
char {funcname}(void);
"""
                )
            f.write(
                f"""\
int main (int argc, char **argv) {{
    {funcname}();
    return 0;
}}
"""
            )

        try:
            objects = self.compile([fname], include_dirs=include_dirs)
        except CompileError:
            return False
        finally:
            os.remove(fname)

        try:
            self.link_executable(
                objects, "a.out", libraries=libraries, library_dirs=library_dirs
            )
        except (LinkError, TypeError):
            return False
        else:
            os.remove(
                self.executable_filename("a.out", output_dir=self.output_dir or '')
            )
        finally:
            for fn in objects:
                os.remove(fn)
        return True

    def find_library_file(
        self, dirs: Iterable[str], lib: str, debug: bool = False
    ) -> str | None:
        """Search the specified list of directories for a static or shared
        library file 'lib' and return the full path to that file.  If
        'debug' true, look for a debugging version (if that makes sense on
        the current platform).  Return None if 'lib' wasn't found in any of
        the specified directories.
        """
        raise NotImplementedError

    # -- Filename generation methods -----------------------------------

    # The default implementation of the filename generating methods are
    # prejudiced towards the Unix/DOS/Windows view of the world:
    #   * object files are named by replacing the source file extension
    #     (eg. .c/.cpp -> .o/.obj)
    #   * library files (shared or static) are named by plugging the
    #     library name and extension into a format string, eg.
    #     "lib%s.%s" % (lib_name, ".a") for Unix static libraries
    #   * executables are named by appending an extension (possibly
    #     empty) to the program name: eg. progname + ".exe" for
    #     Windows
    #
    # To reduce redundant code, these methods expect to find
    # several attributes in the current object (presumably defined
    # as class attributes):
    #   * src_extensions -
    #     list of C/C++ source file extensions, eg. ['.c', '.cpp']
    #   * obj_extension -
    #     object file extension, eg. '.o' or '.obj'
    #   * static_lib_extension -
    #     extension for static library files, eg. '.a' or '.lib'
    #   * shared_lib_extension -
    #     extension for shared library/object files, eg. '.so', '.dll'
    #   * static_lib_format -
    #     format string for generating static library filenames,
    #     eg. 'lib%s.%s' or '%s.%s'
    #   * shared_lib_format
    #     format string for generating shared library filenames
    #     (probably same as static_lib_format, since the extension
    #     is one of the intended parameters to the format string)
    #   * exe_extension -
    #     extension for executable files, eg. '' or '.exe'

    def object_filenames(
        self,
        source_filenames: Iterable[str | os.PathLike[str]],
        strip_dir: bool = False,
        output_dir: str | os.PathLike[str] | None = '',
    ) -> list[str]:
        if output_dir is None:
            output_dir = ''
        return [
            self._make_out_path(output_dir, strip_dir, src_name)
            for src_name in source_filenames
        ]

    @property
    def out_extensions(self):
        return dict.fromkeys(self.src_extensions, self.obj_extension)

    def _make_out_path(self, output_dir, strip_dir, src_name):
        return self._make_out_path_exts(
            output_dir, strip_dir, src_name, self.out_extensions
        )

    @classmethod
    def _make_out_path_exts(cls, output_dir, strip_dir, src_name, extensions):
        r"""
        >>> exts = {'.c': '.o'}
        >>> Compiler._make_out_path_exts('.', False, '/foo/bar.c', exts).replace('\\', '/')
        './foo/bar.o'
        >>> Compiler._make_out_path_exts('.', True, '/foo/bar.c', exts).replace('\\', '/')
        './bar.o'
        """
        src = pathlib.PurePath(src_name)
        # Ensure base is relative to honor output_dir (python/cpython#37775).
        base = cls._make_relative(src)
        try:
            new_ext = extensions[src.suffix]
        except LookupError:
            raise UnknownFileType(f"unknown file type '{src.suffix}' (from '{src}')")
        if strip_dir:
            base = pathlib.PurePath(base.name)
        return os.path.join(output_dir, base.with_suffix(new_ext))

    @staticmethod
    def _make_relative(base: pathlib.Path):
        return base.relative_to(base.anchor)

    @overload
    def shared_object_filename(
        self,
        basename: str,
        strip_dir: Literal[False] = False,
        output_dir: str | os.PathLike[str] = "",
    ) -> str: ...
    @overload
    def shared_object_filename(
        self,
        basename: str | os.PathLike[str],
        strip_dir: Literal[True],
        output_dir: str | os.PathLike[str] = "",
    ) -> str: ...
    def shared_object_filename(
        self,
        basename: str | os.PathLike[str],
        strip_dir: bool = False,
        output_dir: str | os.PathLike[str] = '',
    ) -> str:
        assert output_dir is not None
        if strip_dir:
            basename = os.path.basename(basename)
        return os.path.join(output_dir, basename + self.shared_lib_extension)

    @overload
    def executable_filename(
        self,
        basename: str,
        strip_dir: Literal[False] = False,
        output_dir: str | os.PathLike[str] = "",
    ) -> str: ...
    @overload
    def executable_filename(
        self,
        basename: str | os.PathLike[str],
        strip_dir: Literal[True],
        output_dir: str | os.PathLike[str] = "",
    ) -> str: ...
    def executable_filename(
        self,
        basename: str | os.PathLike[str],
        strip_dir: bool = False,
        output_dir: str | os.PathLike[str] = '',
    ) -> str:
        assert output_dir is not None
        if strip_dir:
            basename = os.path.basename(basename)
        return os.path.join(output_dir, basename + (self.exe_extension or ''))

    def library_filename(
        self,
        libname: str,
        lib_type: str = "static",
        strip_dir: bool = False,
        output_dir: str | os.PathLike[str] = "",  # or 'shared'
    ):
        assert output_dir is not None
        expected = '"static", "shared", "dylib", "xcode_stub"'
        if lib_type not in eval(expected):
            raise ValueError(f"'lib_type' must be {expected}")
        fmt = getattr(self, lib_type + "_lib_format")
        ext = getattr(self, lib_type + "_lib_extension")

        dir, base = os.path.split(libname)
        filename = fmt % (base, ext)
        if strip_dir:
            dir = ''

        return os.path.join(output_dir, dir, filename)

    # -- Utility methods -----------------------------------------------

    def announce(self, msg: object, level: int = 1) -> None:
        log.debug(msg)

    def debug_print(self, msg: object) -> None:
        from distutils.debug import DEBUG

        if DEBUG:
            print(msg)

    def warn(self, msg: object) -> None:
        sys.stderr.write(f"warning: {msg}\n")

    def execute(
        self,
        func: Callable[[Unpack[_Ts]], object],
        args: tuple[Unpack[_Ts]],
        msg: object = None,
        level: int = 1,
    ) -> None:
        if msg is None:
            msg = f"{func.__name__}{args!r}"
        log.info(msg)
        func(*args)

    def call(
        self,
        cmd: MutableSequence[bytes | str | os.PathLike[str]],
        *,
        env: _ENV | None = None,
        **kwargs,
    ) -> None:
        """Run 'cmd' in a subprocess, letting subprocess exceptions propagate."""
        log.info(subprocess.list2cmdline(cmd))
        subprocess.check_call(cmd, env=macos.inject_ver(env), **kwargs)

    def spawn(
        self,
        cmd: MutableSequence[bytes | str | os.PathLike[str]],
        *,
        env: _ENV | None = None,
        **kwargs,
    ) -> None:
        warnings.warn(
            "Compiler.spawn is deprecated; use Compiler.call instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # translation shared with distutils.spawn.spawn; imported late so the
        # clean `call` path stays free of the distutils dependency.
        from ...spawn import _translate_errors

        with _translate_errors(cmd):
            self.call(cmd, env=env, **kwargs)

    @overload
    def move_file(
        self, src: str | os.PathLike[str], dst: _StrPathT
    ) -> _StrPathT | str: ...
    @overload
    def move_file(
        self, src: bytes | os.PathLike[bytes], dst: _BytesPathT
    ) -> _BytesPathT | bytes: ...
    def move_file(
        self,
        src: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        dst: str | os.PathLike[str] | bytes | os.PathLike[bytes],
    ) -> str | os.PathLike[str] | bytes | os.PathLike[bytes]:
        return shutil.move(src, dst)

    def mkpath(self, name, mode=0o777):
        if name:
            os.makedirs(name, mode, exist_ok=True)


# Map a sys.platform/os.name ('posix', 'nt') to the default compiler
# type for that platform. Keys are interpreted as re match
# patterns. Order is important; platform mappings are preferred over
# OS names.
_default_compilers = (
    # Platform string mappings
    # on a cygwin built python we can use gcc like an ordinary UNIXish
    # compiler
    ('cygwin.*', 'unix'),
    ('zos', 'zos'),
    # OS name mappings
    ('posix', 'unix'),
    ('nt', 'msvc'),
)


def get_default_compiler(osname: str | None = None, platform: str | None = None) -> str:
    """Determine the default compiler to use for the given platform.

    osname should be one of the standard Python OS names (i.e. the
    ones returned by os.name) and platform the common value
    returned by sys.platform for the platform in question.

    The default values are os.name and sys.platform in case the
    parameters are not given.
    """
    if osname is None:
        osname = os.name
    if platform is None:
        platform = sys.platform
    # Mingw is a special case where sys.platform is 'win32' but we
    # want to use the 'mingw32' compiler, so check it first
    if is_mingw():
        return 'mingw32'
    for pattern, compiler in _default_compilers:
        if (
            re.match(pattern, platform) is not None
            or re.match(pattern, osname) is not None
        ):
            return compiler
    # Default to Unix compiler
    return 'unix'


def _concrete_compilers(cls: type[Compiler] | None = None):
    """Yield every subclass of ``cls`` (recursively)."""
    for subclass in (cls or Compiler).__subclasses__():
        yield subclass
        yield from _concrete_compilers(subclass)


def get_compilers() -> dict[str, type[Compiler]]:
    """Map each compiler's short name (``compiler_type``) to its class.

    Imports the concrete compiler modules so their classes are defined, then
    collects every non-abstract ``Compiler`` subclass.
    """
    # ensure the concrete compiler classes are imported and thus registered
    # as subclasses of Compiler
    from . import cygwin, msvc, unix, zos  # noqa: F401

    return {
        subclass.compiler_type: subclass
        for subclass in _concrete_compilers()
        if getattr(subclass, 'compiler_type', None)
    }


def new_compiler(
    plat: str | None = None,
    compiler: str | None = None,
    verbose: bool = False,
    force: bool = False,
) -> Compiler:
    """Generate an instance of some Compiler subclass for the supplied
    platform/compiler combination.  'plat' defaults to 'os.name'
    (eg. 'posix', 'nt'), and 'compiler' defaults to the default compiler
    for that platform.  Currently only 'posix' and 'nt' are supported, and
    the default compilers are "traditional Unix interface" (UnixCCompiler
    class) and Visual C++ (MSVCCompiler class).  Note that it's perfectly
    possible to ask for a Unix compiler object under Windows, and a
    Microsoft compiler object under Unix -- if you supply a value for
    'compiler', 'plat' is ignored.
    """
    if plat is None:
        plat = os.name
    if compiler is None:
        compiler = get_default_compiler(plat)

    try:
        cls = get_compilers()[compiler]
    except KeyError:
        msg = f"don't know how to compile C/C++ code on platform '{plat}'"
        if compiler is not None:
            msg = msg + f" with '{compiler}' compiler"
        raise PlatformError(msg)

    # XXX The None is necessary to preserve backwards compatibility
    # with classes that expect verbose to be the first positional
    # argument.
    return cls(None, force=force)


def gen_preprocess_options(
    macros: Iterable[_Macro], include_dirs: Iterable[str]
) -> list[str]:
    """Generate C pre-processor options (-D, -U, -I) as used by at least
    two types of compilers: the typical Unix compiler and Visual C++.
    'macros' is the usual thing, a list of 1- or 2-tuples, where (name,)
    means undefine (-U) macro 'name', and (name,value) means define (-D)
    macro 'name' to 'value'.  'include_dirs' is just a list of directory
    names to be added to the header file search path (-I).  Returns a list
    of command-line options suitable for either Unix compilers or Visual
    C++.
    """
    # XXX it would be nice (mainly aesthetic, and so we don't generate
    # stupid-looking command lines) to go over 'macros' and eliminate
    # redundant definitions/undefinitions (ie. ensure that only the
    # latest mention of a particular macro winds up on the command
    # line).  I don't think it's essential, though, since most (all?)
    # Unix C compilers only pay attention to the latest -D or -U
    # mention of a macro on their command line.  Similar situation for
    # 'include_dirs'.  I'm punting on both for now.  Anyways, weeding out
    # redundancies like this should probably be the province of
    # CCompiler, since the data structures used are inherited from it
    # and therefore common to all CCompiler classes.
    pp_opts = []
    for macro in macros:
        if not (isinstance(macro, tuple) and 1 <= len(macro) <= 2):
            raise TypeError(
                f"bad macro definition '{macro}': "
                "each element of 'macros' list must be a 1- or 2-tuple"
            )

        if len(macro) == 1:  # undefine this macro
            pp_opts.append(f"-U{macro[0]}")
        elif len(macro) == 2:
            if macro[1] is None:  # define with no explicit value
                pp_opts.append(f"-D{macro[0]}")
            else:
                # XXX *don't* need to be clever about quoting the
                # macro value here, because we're going to avoid the
                # shell at all costs when we spawn the command!
                pp_opts.append("-D{}={}".format(*macro))

    pp_opts.extend(f"-I{dir}" for dir in include_dirs)
    return pp_opts


def gen_lib_options(
    compiler: Compiler,
    library_dirs: Iterable[str],
    runtime_library_dirs: Iterable[str],
    libraries: Iterable[str],
) -> list[str]:
    """Generate linker options for searching library directories and
    linking with specific libraries.  'libraries' and 'library_dirs' are,
    respectively, lists of library names (not filenames!) and search
    directories.  Returns a list of command-line options suitable for use
    with some compiler (depending on the two format strings passed in).
    """
    lib_opts = [compiler.library_dir_option(dir) for dir in library_dirs]

    for dir in runtime_library_dirs:
        lib_opts.extend(always_iterable(compiler.runtime_library_dir_option(dir)))

    # XXX it's important that we *not* remove redundant library mentions!
    # sometimes you really do have to say "-lfoo -lbar -lfoo" in order to
    # resolve all symbols.  I just hope we never have to say "-lfoo obj.o
    # -lbar" to get things to work -- that's certainly a possibility, but a
    # pretty nasty way to arrange your C code.

    for lib in libraries:
        (lib_dir, lib_name) = os.path.split(lib)
        if lib_dir:
            lib_file = compiler.find_library_file([lib_dir], lib_name)
            if lib_file:
                lib_opts.append(lib_file)
            else:
                compiler.warn(
                    f"no library file corresponding to '{lib}' found (skipping)"
                )
        else:
            lib_opts.append(compiler.library_option(lib))
    return lib_opts
<|endoftext|>"""distutils.unixccompiler

Contains the UnixCCompiler class, a subclass of CCompiler that handles
the "typical" Unix-style command-line C compiler:
  * macros defined with -Dname[=value]
  * macros undefined with -Uname
  * include search directories specified with -Idir
  * libraries specified with -lllib
  * library search directories specified with -Ldir
  * compile handled by 'cc' (or similar) executable with -c option:
    compiles .c to .o
  * link static library handled by 'ar' command (possibly with 'ranlib')
  * link shared library handled by 'cc -shared'
"""

from __future__ import annotations

import itertools
import os
import re
import shlex
import subprocess
import sys
import sysconfig
from collections.abc import Iterable
from typing import ClassVar

from jaraco.functools import pass_none

from .._modified import newer
from ..logging import get_logger
from ..platform import macos
from ..platform.macos import compiler_fixup
from . import base
from .base import _Macro, gen_lib_options, gen_preprocess_options
from .errors import CompileError, LibError, LinkError

log = get_logger(__name__)


def _require_config_vars(*names):
    values = sysconfig.get_config_vars(*names)
    missing = [
        name for name, value in zip(names, values, strict=False) if value is None
    ]
    assert not missing, f"Unexpected None in config vars: {missing}"
    return values


@pass_none
def _add_flags(value: str, flag_type: str) -> str:
    """Append any ``$<flag_type>FLAGS`` from the environment to ``value``."""
    flags = os.environ.get(f'{flag_type}FLAGS')
    return f'{value} {flags}' if flags else value


# XXX Things not currently handled:
#   * optimization/debug/warning flags; we just use whatever's in Python's
#     Makefile and live with it.  Is this adequate?  If not, we might
#     have to have a bunch of subclasses GNUCCompiler, SGICCompiler,
#     SunCCompiler, and I suspect down that road lies madness.
#   * even if we don't know a warning flag from an optimization flag,
#     we need some way for outsiders to feed preprocessor/compiler/linker
#     flags in to us -- eg. a sysadmin might want to mandate certain flags
#     via a site config file, or a user might want to set something for
#     compiling this module distribution only via the setup.py command
#     line, whatever.  As long as these options come from something on the
#     current system, they can be as system-dependent as they like, and we
#     should just happily stuff them into the preprocessor/compiler/linker
#     options and carry on.


def _split_env(cmd):
    """
    For macOS, split command into 'env' portion (if any)
    and the rest of the linker command.

    >>> _split_env(['a', 'b', 'c'])
    ([], ['a', 'b', 'c'])
    >>> _split_env(['/usr/bin/env', 'A=3', 'gcc'])
    (['/usr/bin/env', 'A=3'], ['gcc'])
    """
    pivot = 0
    if os.path.basename(cmd[0]) == "env":
        pivot = 1
        while '=' in cmd[pivot]:
            pivot += 1
    return cmd[:pivot], cmd[pivot:]


def _split_aix(cmd):
    """
    AIX platforms prefix the compiler with the ld_so_aix
    script, so split that from the linker command.

    >>> _split_aix(['a', 'b', 'c'])
    ([], ['a', 'b', 'c'])
    >>> _split_aix(['/bin/foo/ld_so_aix', 'gcc'])
    (['/bin/foo/ld_so_aix'], ['gcc'])
    """
    pivot = os.path.basename(cmd[0]) == 'ld_so_aix'
    return cmd[:pivot], cmd[pivot:]


def _linker_params(linker_cmd, compiler_cmd):
    """
    The linker command usually begins with the compiler
    command (possibly multiple elements), followed by zero or more
    params for shared library building.

    If the LDSHARED env variable overrides the linker command,
    however, the commands may not match.

    Return the best guess of the linker parameters by stripping
    the linker command. If the compiler command does not
    match the linker command, assume the linker command is
    just the first element.

    >>> _linker_params('gcc foo bar'.split(), ['gcc'])
    ['foo', 'bar']
    >>> _linker_params('gcc foo bar'.split(), ['other'])
    ['foo', 'bar']
    >>> _linker_params('ccache gcc foo bar'.split(), 'ccache gcc'.split())
    ['foo', 'bar']
    >>> _linker_params(['gcc'], ['gcc'])
    []
    """
    c_len = len(compiler_cmd)
    pivot = c_len if linker_cmd[:c_len] == compiler_cmd else 1
    return linker_cmd[pivot:]


class Compiler(base.Compiler):
    compiler_type = 'unix'
    description = "standard UNIX-style compiler"

    # These are used by CCompiler in two places: the constructor sets
    # instance attributes 'preprocessor', 'compiler', etc. from them, and
    # 'set_executable()' allows any of these to be set.  The defaults here
    # are pretty generic; they will probably have to be set by an outsider
    # (eg. using information discovered by the sysconfig about building
    # Python extensions).
    executables: ClassVar[dict] = {
        'preprocessor': None,
        'compiler': ["cc"],
        'compiler_so': ["cc"],
        'compiler_cxx': ["c++"],
        'compiler_so_cxx': ["c++"],
        'linker_so': ["cc", "-shared"],
        'linker_so_cxx': ["c++", "-shared"],
        'linker_exe': ["cc"],
        'linker_exe_cxx': ["c++", "-shared"],
        'archiver': ["ar", "-cr"],
        'ranlib': None,
    }

    if sys.platform[:6] == "darwin":
        executables['ranlib'] = ["ranlib"]

    # Needed for the filename generation methods provided by the base
    # class, CCompiler.  NB. whoever instantiates/uses a particular
    # UnixCCompiler instance should set 'shared_lib_ext' -- we set a
    # reasonable common default here, but it's not necessarily used on all
    # Unices!

    src_extensions: ClassVar[list[str] | None] = [
        ".c",
        ".C",
        ".cc",
        ".cxx",
        ".cpp",
        ".m",
    ]
    obj_extension = ".o"
    static_lib_extension = ".a"
    shared_lib_extension = ".so"
    dylib_lib_extension = ".dylib"
    xcode_stub_lib_extension = ".tbd"
    static_lib_format = shared_lib_format = dylib_lib_format = "lib%s%s"
    xcode_stub_lib_format = dylib_lib_format
    if sys.platform == "cygwin":
        exe_extension = ".exe"
        shared_lib_extension = ".dll.a"
        dylib_lib_extension = ".dll"
        dylib_lib_format = "cyg%s%s"

    def configure_system(self) -> None:
        """Configure this compiler from the interpreter's build configuration.

        Applies the compiler, flag, and archiver settings CPython recorded in
        sysconfig when it was built -- honoring the usual environment-variable
        overrides (CC, CXX, CFLAGS, LDSHARED, AR, RANLIB, …) -- so extensions
        build consistently with the interpreter.
        """
        macos.customize_compiler(sysconfig.get_config_vars())

        (
            cc,
            cxx,
            cflags,
            ccshared,
            ldshared,
            ldcxxshared,
            shlib_suffix,
            ar,
            ar_flags,
        ) = _require_config_vars(
            'CC',
            'CXX',
            'CFLAGS',
            'CCSHARED',
            'LDSHARED',
            'LDCXXSHARED',
            'SHLIB_SUFFIX',
            'AR',
            'ARFLAGS',
        )

        cxxflags = cflags

        if 'CC' in os.environ:
            newcc = os.environ['CC']
            if 'LDSHARED' not in os.environ and ldshared.startswith(cc):
                # If CC is overridden, use that as the default command for
                # LDSHARED as well.
                ldshared = newcc + ldshared[len(cc) :]
            cc = newcc
        cxx = os.environ.get('CXX', cxx)
        ldshared = os.environ.get('LDSHARED', ldshared)
        ldcxxshared = os.environ.get('LDCXXSHARED', ldcxxshared)
        cpp = os.environ.get('CPP', cc + " -E")

        ldshared = _add_flags(ldshared, 'LD')
        ldcxxshared = _add_flags(ldcxxshared, 'LD')
        cflags = os.environ.get('CFLAGS', cflags)
        ldshared = _add_flags(ldshared, 'C')
        cxxflags = os.environ.get('CXXFLAGS', cxxflags)
        ldcxxshared = _add_flags(ldcxxshared, 'CXX')
        cpp = _add_flags(cpp, 'CPP')
        cflags = _add_flags(cflags, 'CPP')
        cxxflags = _add_flags(cxxflags, 'CPP')
        ldshared = _add_flags(ldshared, 'CPP')
        ldcxxshared = _add_flags(ldcxxshared, 'CPP')

        ar = os.environ.get('AR', ar)

        archiver = ar + ' ' + os.environ.get('ARFLAGS', ar_flags)
        cc_cmd = cc + ' ' + cflags
        cxx_cmd = cxx + ' ' + cxxflags

        self.set_executables(
            preprocessor=cpp,
            compiler=cc_cmd,
            compiler_so=cc_cmd + ' ' + ccshared,
            compiler_cxx=cxx_cmd,
            compiler_so_cxx=cxx_cmd + ' ' + ccshared,
            linker_so=ldshared,
            linker_so_cxx=ldcxxshared,
            linker_exe=cc,
            linker_exe_cxx=cxx,
            archiver=archiver,
        )

        if 'RANLIB' in os.environ and self.executables.get('ranlib', None):
            self.set_executables(ranlib=os.environ['RANLIB'])

        self.shared_lib_extension = shlib_suffix  # type: ignore[misc] # Assigning to ClassVar

    def _fix_lib_args(self, libraries, library_dirs, runtime_library_dirs):
        """Remove standard library path from rpath"""
        libraries, library_dirs, runtime_library_dirs = super()._fix_lib_args(
            libraries, library_dirs, runtime_library_dirs
        )
        libdir = sysconfig.get_config_var('LIBDIR')
        if (
            runtime_library_dirs
            and libdir.startswith("/usr/lib")
            and (libdir in runtime_library_dirs)
        ):
            runtime_library_dirs.remove(libdir)
        return libraries, library_dirs, runtime_library_dirs

    def preprocess(
        self,
        source: str | os.PathLike[str],
        output_file: str | os.PathLike[str] | None = None,
        macros: list[_Macro] | None = None,
        include_dirs: list[str] | tuple[str, ...] | None = None,
        extra_preargs: list[str] | None = None,
        extra_postargs: Iterable[str] | None = None,
    ):
        fixed_args = self._fix_compile_args(None, macros, include_dirs)
        _ignore, macros, include_dirs = fixed_args
        pp_opts = gen_preprocess_options(macros, include_dirs)
        pp_args = self.preprocessor + pp_opts
        if output_file:
            pp_args.extend(['-o', output_file])
        if extra_preargs:
            pp_args[:0] = extra_preargs
        if extra_postargs:
            pp_args.extend(extra_postargs)
        pp_args.append(source)

        # reasons to preprocess:
        # - force is indicated
        # - output is directed to stdout
        # - source file is newer than the target
        preprocess = self.force or output_file is None or newer(source, output_file)
        if not preprocess:
            return

        if output_file:
            self.mkpath(os.path.dirname(output_file))

        try:
            self.call(pp_args)
        except (subprocess.CalledProcessError, OSError) as msg:
            raise CompileError(msg)

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        compiler_so = compiler_fixup(self.compiler_so, cc_args + extra_postargs)
        compiler_so_cxx = compiler_fixup(self.compiler_so_cxx, cc_args + extra_postargs)
        try:
            if self.detect_language(src) == 'c++':
                self.call(compiler_so_cxx + cc_args + [src, '-o', obj] + extra_postargs)
            else:
                self.call(compiler_so + cc_args + [src, '-o', obj] + extra_postargs)
        except (subprocess.CalledProcessError, OSError) as msg:
            raise CompileError(msg)

    def create_static_lib(
        self, objects, output_libname, output_dir=None, debug=False, target_lang=None
    ):
        objects, output_dir = self._fix_object_args(objects, output_dir)

        output_filename = self.library_filename(output_libname, output_dir=output_dir)

        if self._need_link(objects, output_filename):
            self.mkpath(os.path.dirname(output_filename))
            self.call(self.archiver + [output_filename] + objects + self.objects)

            # Not many Unices required ranlib anymore -- SunOS 4.x is, I
            # think the only major Unix that does.  Maybe we need some
            # platform intelligence here to skip ranlib if it's not
            # needed -- or maybe Python's configure script took care of
            # it for us, hence the check for leading colon.
            if self.ranlib:
                try:
                    self.call(self.ranlib + [output_filename])
                except (subprocess.CalledProcessError, OSError) as msg:
                    raise LibError(msg)
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    def link(
        self,
        target_desc,
        objects: list[str] | tuple[str, ...],
        output_filename,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        export_symbols=None,
        debug=False,
        extra_preargs=None,
        extra_postargs=None,
        build_temp=None,
        target_lang=None,
    ):
        objects, output_dir = self._fix_object_args(objects, output_dir)
        fixed_args = self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)
        libraries, library_dirs, runtime_library_dirs = fixed_args

        lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs, libraries)
        if not isinstance(output_dir, (str, type(None))):
            raise TypeError("'output_dir' must be a string or None")
        if output_dir is not None:
            output_filename = os.path.join(output_dir, output_filename)

        if self._need_link(objects, output_filename):
            ld_args = objects + self.objects + lib_opts + ['-o', output_filename]
            if debug:
                ld_args[:0] = ['-g']
            if extra_preargs:
                ld_args[:0] = extra_preargs
            if extra_postargs:
                ld_args.extend(extra_postargs)
            self.mkpath(os.path.dirname(output_filename))
            try:
                # Select a linker based on context: linker_exe when
                # building an executable or linker_so (with shared options)
                # when building a shared library.
                building_exe = target_desc == base.Compiler.EXECUTABLE
                target_cxx = target_lang == "c++"
                linker = (
                    (self.linker_exe_cxx if target_cxx else self.linker_exe)
                    if building_exe
                    else (self.linker_so_cxx if target_cxx else self.linker_so)
                )[:]

                if target_cxx and self.compiler_cxx:
                    env, linker_ne = _split_env(linker)
                    aix, linker_na = _split_aix(linker_ne)
                    _, compiler_cxx_ne = _split_env(self.compiler_cxx)
                    _, linker_exe_ne = _split_env(self.linker_exe_cxx)

                    params = _linker_params(linker_na, linker_exe_ne)
                    linker = env + aix + compiler_cxx_ne + params

                linker = compiler_fixup(linker, ld_args)

                self.call(linker + ld_args)
            except (subprocess.CalledProcessError, OSError) as msg:
                raise LinkError(msg)
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    # -- Miscellaneous methods -----------------------------------------
    # These are all used by the 'gen_lib_options() function, in
    # ccompiler.py.

    def library_dir_option(self, dir):
        return "-L" + dir

    def _is_gcc(self):
        cc_var = sysconfig.get_config_var("CC")
        compiler = os.path.basename(shlex.split(cc_var)[0])
        return "gcc" in compiler or "g++" in compiler

    def runtime_library_dir_option(self, dir: str) -> str | list[str]:
        # XXX Hackish, at the very least.  See Python bug #445902:
        # https://bugs.python.org/issue445902
        # Linkers on different platforms need different options to
        # specify that directories need to be added to the list of
        # directories searched for dependencies when a dynamic library
        # is sought.  GCC on GNU systems (Linux, FreeBSD, ...) has to
        # be told to pass the -R option through to the linker, whereas
        # other compilers and gcc on other systems just know this.
        # Other compilers may need something slightly different.  At
        # this time, there's no way to determine this information from
        # the configuration data stored in the Python installation, so
        # we use this hack.
        if sys.platform[:6] == "darwin":
            from ..platform import macos

            target_ver = macos.target_ver()
            if target_ver and [int(n) for n in target_ver.split('.')] >= [10, 5]:
                return "-Wl,-rpath," + dir
            else:  # no support for -rpath on earlier macOS versions
                return "-L" + dir
        elif sys.platform[:7] == "freebsd":
            return "-Wl,-rpath=" + dir
        elif sys.platform[:5] == "hp-ux":
            return [
                "-Wl,+s" if self._is_gcc() else "+s",
                "-L" + dir,
            ]

        # For all compilers, `-Wl` is the presumed way to pass a
        # compiler option to the linker
        if sysconfig.get_config_var("GNULD") == "yes":
            return [
                # Force RUNPATH instead of RPATH
                "-Wl,--enable-new-dtags",
                "-Wl,-rpath," + dir,
            ]
        else:
            return "-Wl,-R" + dir

    def library_option(self, lib):
        return "-l" + lib

    @staticmethod
    def _library_root(dir):
        """
        macOS users can specify an alternate SDK using'-isysroot'.
        Calculate the SDK root if it is specified.

        Note that, as of Xcode 7, Apple SDKs may contain textual stub
        libraries with .tbd extensions rather than the normal .dylib
        shared libraries installed in /.  The Apple compiler tool
        chain handles this transparently but it can cause problems
        for programs that are being built with an SDK and searching
        for specific libraries.  Callers of find_library_file need to
        keep in mind that the base filename of the returned SDK library
        file might have a different extension from that of the library
        file installed on the running system, for example:
          /Applications/Xcode.app/Contents/Developer/Platforms/
              MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/
              usr/lib/libedit.tbd
        vs
          /usr/lib/libedit.dylib
        """
        cflags = sysconfig.get_config_var('CFLAGS')
        match = re.search(r'-isysroot\s*(\S+)', cflags)

        apply_root = (
            sys.platform == 'darwin'
            and match
            and (
                dir.startswith('/System/')
                or (dir.startswith('/usr/') and not dir.startswith('/usr/local/'))
            )
        )

        return os.path.join(match.group(1), dir[1:]) if apply_root else dir

    def find_library_file(self, dirs, lib, debug=False):
        """
        Second-guess the linker with not much hard
        data to go on: GCC seems to prefer the shared library, so
        assume that *all* Unix C compilers do,
        ignoring even GCC's "-static" option.
        """
        lib_names = (
            self.library_filename(lib, lib_type=type)
            for type in ('dylib', 'xcode_stub', 'shared', 'static')
        )

        roots = map(self._library_root, dirs)

        searched = itertools.starmap(os.path.join, itertools.product(roots, lib_names))

        found = filter(os.path.exists, searched)

        # Return None if it could not be found in any dir.
        return next(found, None)
<|endoftext|>"""distutils._msvccompiler

Contains MSVCCompiler, an implementation of the abstract CCompiler class
for Microsoft Visual Studio 2015.

This module requires VS 2015 or later.
"""

# Written by Perry Stoll
# hacked by Robin Becker and Thomas Heller to do a better job of
#   finding DevStudio (through the registry)
# ported to VS 2005 and VS 2008 by Christian Heimes
# ported to VS 2015 by Steve Dower
from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar

with contextlib.suppress(ImportError):
    import winreg

from itertools import count

from ..errors import PlatformError
from ..logging import get_logger
from ..platform.detect import get_host_platform, get_platform
from . import base
from .base import gen_lib_options
from .errors import CompileError, LibError, LinkError

log = get_logger(__name__)


def _find_vc2015():
    try:
        key = winreg.OpenKeyEx(
            winreg.HKEY_LOCAL_MACHINE,
            r"Software\Microsoft\VisualStudio\SxS\VC7",
            access=winreg.KEY_READ | winreg.KEY_WOW64_32KEY,
        )
    except OSError:
        log.debug("Visual C++ is not registered")
        return None, None

    best_version = 0
    best_dir = None
    with key:
        for i in count():
            try:
                v, vc_dir, vt = winreg.EnumValue(key, i)
            except OSError:
                break
            if v and vt == winreg.REG_SZ and os.path.isdir(vc_dir):
                try:
                    version = int(float(v))
                except (ValueError, TypeError):
                    continue
                if version >= 14 and version > best_version:
                    best_version, best_dir = version, vc_dir
    return best_version, best_dir


def _find_vc2017():
    """Returns "15, path" based on the result of invoking vswhere.exe
    If no install is found, returns "None, None"

    The version is returned to avoid unnecessarily changing the function
    result. It may be ignored when the path is not None.

    If vswhere.exe is not available, by definition, VS 2017 is not
    installed.
    """
    root = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if not root:
        return None, None

    variant = 'arm64' if get_platform() == 'win-arm64' else 'x86.x64'
    suitable_components = (
        f"Microsoft.VisualStudio.Component.VC.Tools.{variant}",
        "Microsoft.VisualStudio.Workload.WDExpress",
    )

    for component in suitable_components:
        # Workaround for `-requiresAny` (only available on VS 2017 > 15.6)
        with contextlib.suppress(
            subprocess.CalledProcessError, OSError, UnicodeDecodeError
        ):
            path = (
                subprocess
                .check_output([
                    os.path.join(
                        root, "Microsoft Visual Studio", "Installer", "vswhere.exe"
                    ),
                    "-latest",
                    "-prerelease",
                    "-requires",
                    component,
                    "-property",
                    "installationPath",
                    "-products",
                    "*",
                ])
                .decode(encoding="mbcs", errors="strict")
                .strip()
            )

            path = os.path.join(path, "VC", "Auxiliary", "Build")
            if os.path.isdir(path):
                return 15, path

    return None, None  # no suitable component found


PLAT_SPEC_TO_RUNTIME = {
    'x86': 'x86',
    'x86_amd64': 'x64',
    'x86_arm': 'arm',
    'x86_arm64': 'arm64',
}


def _find_vcvarsall(plat_spec):
    # bpo-38597: Removed vcruntime return value
    _, best_dir = _find_vc2017()

    if not best_dir:
        _best_version, best_dir = _find_vc2015()

    if not best_dir:
        log.debug("No suitable Visual C++ version found")
        return None, None

    vcvarsall = os.path.join(best_dir, "vcvarsall.bat")
    if not os.path.isfile(vcvarsall):
        log.debug("%s cannot be found", vcvarsall)
        return None, None

    return vcvarsall, None


def _get_vc_env(plat_spec):
    if os.getenv("DISTUTILS_USE_SDK"):
        return {key.lower(): value for key, value in os.environ.items()}

    vcvarsall, _ = _find_vcvarsall(plat_spec)
    if not vcvarsall:
        raise PlatformError(
            'Microsoft Visual C++ 14.0 or greater is required. '
            'Get it with "Microsoft C++ Build Tools": '
            'https://visualstudio.microsoft.com/visual-cpp-build-tools/'
        )

    try:
        out = subprocess.check_output(
            f'cmd /u /c "{vcvarsall}" {plat_spec} && set',
            stderr=subprocess.STDOUT,
        ).decode('utf-16le', errors='replace')
    except subprocess.CalledProcessError as exc:
        log.error(exc.output)
        raise PlatformError(f"Error executing {exc.cmd}")

    env = {
        key.lower(): value
        for key, _, value in (line.partition('=') for line in out.splitlines())
        if key and value
    }

    return env


def _find_exe(exe, paths=None):
    """Return path to an MSVC executable program.

    Tries to find the program in several places: first, one of the
    MSVC program search paths from the registry; next, the directories
    in the PATH environment variable.  If any of those work, return an
    absolute path that is known to exist.  If none of them work, just
    return the original program name, 'exe'.
    """
    if not paths:
        paths = os.getenv('path').split(os.pathsep)
    for p in paths:
        fn = os.path.join(os.path.abspath(p), exe)
        if os.path.isfile(fn):
            return fn
    return exe


_vcvars_names = {
    'win32': 'x86',
    'win-amd64': 'amd64',
    'win-arm32': 'arm',
    'win-arm64': 'arm64',
}


def _get_vcvars_spec(host_platform, platform):
    """
    Given a host platform and platform, determine the spec for vcvarsall.

    Uses the native MSVC host if the host platform would need expensive
    emulation for x86.

    >>> _get_vcvars_spec('win-arm64', 'win32')
    'arm64_x86'
    >>> _get_vcvars_spec('win-arm64', 'win-amd64')
    'arm64_amd64'

    Otherwise, always cross-compile from x86 to work with the
    lighter-weight MSVC installs that do not include native 64-bit tools.

    >>> _get_vcvars_spec('win32', 'win32')
    'x86'
    >>> _get_vcvars_spec('win-arm32', 'win-arm32')
    'x86_arm'
    >>> _get_vcvars_spec('win-amd64', 'win-arm64')
    'x86_arm64'
    """
    if host_platform != 'win-arm64':
        host_platform = 'win32'
    vc_hp = _vcvars_names[host_platform]
    vc_plat = _vcvars_names[platform]
    return vc_hp if vc_hp == vc_plat else f'{vc_hp}_{vc_plat}'


_MAX_COMMAND_LENGTH = 2**15 - 1
"""
Windows limits a process command line to this many characters. See the
`CreateProcess documentation
<https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessa#parameters>`_.
"""


def _response_file_content(args: Iterable[str]) -> str:
    r"""
    Render ``args`` as the content of a linker response file, one quoted
    argument per line.

    >>> print(_response_file_content(['/OUT:with space.dll', 'a.obj']))
    "/OUT:with space.dll"
    "a.obj"
    """
    return '\n'.join(f'"{arg}"' for arg in args)


@contextlib.contextmanager
def _wrap_link_command(*cmd: str) -> Iterator[list[str]]:
    r"""
    Yield ``cmd`` suitable for :meth:`Compiler.spawn`, honoring the Windows
    maximum command-line length (:data:`_MAX_COMMAND_LENGTH`).

    When ``cmd`` fits, yield it unchanged. Otherwise write the arguments to a
    temporary `response file
    <https://learn.microsoft.com/en-us/cpp/build/reference/linking?view=msvc-170#linker-command-files>`_
    and yield a command referencing it, removing the file once the command
    completes.

    Short commands pass through unchanged:

    >>> with _wrap_link_command('link.exe', 'a.obj') as spawn_cmd:
    ...     spawn_cmd
    ['link.exe', 'a.obj']

    Long commands are replaced by a reference to a response file, which is
    removed once the command completes:

    >>> import pathlib
    >>> args = ['/LIBPATH:' + 'x' * 100] * 400
    >>> with _wrap_link_command('link.exe', *args) as spawn_cmd:
    ...     spawn_cmd  # doctest: +ELLIPSIS
    ...     response_file = pathlib.Path(spawn_cmd[1][1:])
    ...     response_file.exists()
    ['link.exe', '@...rsp']
    True
    >>> response_file.exists()
    False
    """
    if len(subprocess.list2cmdline(cmd)) <= _MAX_COMMAND_LENGTH:
        yield list(cmd)
        return

    linker, *args = cmd
    with tempfile.TemporaryDirectory() as tmpdir:
        # utf-16 gives the linker an unambiguous, BOM-prefixed encoding.
        response_file = Path(tmpdir) / 'link.rsp'
        response_file.write_text(_response_file_content(args), encoding='utf-16')
        yield [linker, f'@{response_file}']


class Compiler(base.Compiler):
    """Concrete class that implements an interface to Microsoft Visual C++,
    as defined by the CCompiler abstract class."""

    compiler_type = 'msvc'
    description = "Microsoft Visual C++"

    # Just set this so CCompiler's constructor doesn't barf.  We currently
    # don't use the 'set_executables()' bureaucracy provided by CCompiler,
    # as it really isn't necessary for this sort of single-compiler class.
    # Would be nice to have a consistent interface with UnixCCompiler,
    # though, so it's worth thinking about.
    executables: ClassVar[dict] = {}

    # Private class data (need to distinguish C from C++ source for compiler)
    _c_extensions: ClassVar[list[str]] = ['.c']
    _cpp_extensions: ClassVar[list[str]] = ['.cc', '.cpp', '.cxx']
    _rc_extensions: ClassVar[list[str]] = ['.rc']
    _mc_extensions: ClassVar[list[str]] = ['.mc']

    # Needed for the filename generation methods provided by the
    # base class, CCompiler.
    src_extensions = _c_extensions + _cpp_extensions + _rc_extensions + _mc_extensions
    res_extension = '.res'
    obj_extension = '.obj'
    static_lib_extension = '.lib'
    shared_lib_extension = '.dll'
    static_lib_format = shared_lib_format = '%s%s'
    exe_extension = '.exe'

    def __init__(self, verbose=False, force=False) -> None:
        super().__init__(verbose, force=force)
        # target platform (.plat_name is consistent with 'bdist')
        self.plat_name = None
        self.initialized = False

    @classmethod
    def _configure(cls, vc_env):
        """
        Set class-level include/lib dirs.
        """
        cls.include_dirs = cls._parse_path(vc_env.get('include', ''))
        cls.library_dirs = cls._parse_path(vc_env.get('lib', ''))

    @staticmethod
    def _parse_path(val):
        return [dir.rstrip(os.sep) for dir in val.split(os.pathsep) if dir]

    def initialize(self, plat_name: str | None = None) -> None:
        # multi-init means we would need to check platform same each time...
        assert not self.initialized, "don't init multiple times"
        if plat_name is None:
            plat_name = get_platform()
        # sanity check for platforms to prevent obscure errors later.
        if plat_name not in _vcvars_names:
            raise PlatformError(f"--plat-name must be one of {tuple(_vcvars_names)}")

        plat_spec = _get_vcvars_spec(get_host_platform(), plat_name)

        vc_env = _get_vc_env(plat_spec)
        if not vc_env:
            raise PlatformError(
                "Unable to find a compatible Visual Studio installation."
            )
        self._configure(vc_env)

        self._paths = vc_env.get('path', '')
        paths = self._paths.split(os.pathsep)
        self.cc = _find_exe("cl.exe", paths)
        self.linker = _find_exe("link.exe", paths)
        self.lib = _find_exe("lib.exe", paths)
        self.rc = _find_exe("rc.exe", paths)  # resource compiler
        self.mc = _find_exe("mc.exe", paths)  # message compiler
        self.mt = _find_exe("mt.exe", paths)  # message compiler

        self.preprocess_options = None
        # bpo-38597: Always compile with dynamic linking
        # Future releases of Python 3.x will include all past
        # versions of vcruntime*.dll for compatibility.
        self.compile_options = ['/nologo', '/O2', '/W3', '/GL', '/DNDEBUG', '/MD']

        self.compile_options_debug = [
            '/nologo',
            '/Od',
            '/MDd',
            '/Zi',
            '/W3',
            '/D_DEBUG',
        ]

        ldflags = ['/nologo', '/INCREMENTAL:NO', '/LTCG']

        ldflags_debug = ['/nologo', '/INCREMENTAL:NO', '/LTCG', '/DEBUG:FULL']

        self.ldflags_exe = [*ldflags, '/MANIFEST:EMBED,ID=1']
        self.ldflags_exe_debug = [*ldflags_debug, '/MANIFEST:EMBED,ID=1']
        self.ldflags_shared = [
            *ldflags,
            '/DLL',
            '/MANIFEST:EMBED,ID=2',
            '/MANIFESTUAC:NO',
        ]
        self.ldflags_shared_debug = [
            *ldflags_debug,
            '/DLL',
            '/MANIFEST:EMBED,ID=2',
            '/MANIFESTUAC:NO',
        ]
        self.ldflags_static = [*ldflags]
        self.ldflags_static_debug = [*ldflags_debug]

        self._ldflags = {
            (base.Compiler.EXECUTABLE, None): self.ldflags_exe,
            (base.Compiler.EXECUTABLE, False): self.ldflags_exe,
            (base.Compiler.EXECUTABLE, True): self.ldflags_exe_debug,
            (base.Compiler.SHARED_OBJECT, None): self.ldflags_shared,
            (base.Compiler.SHARED_OBJECT, False): self.ldflags_shared,
            (base.Compiler.SHARED_OBJECT, True): self.ldflags_shared_debug,
            (base.Compiler.SHARED_LIBRARY, None): self.ldflags_static,
            (base.Compiler.SHARED_LIBRARY, False): self.ldflags_static,
            (base.Compiler.SHARED_LIBRARY, True): self.ldflags_static_debug,
        }

        self.initialized = True

    # -- Worker methods ------------------------------------------------

    @property
    def out_extensions(self) -> dict[str, str]:
        return {
            **super().out_extensions,
            **{
                ext: self.res_extension
                for ext in self._rc_extensions + self._mc_extensions
            },
        }

    def compile(  # noqa: C901
        self,
        sources,
        output_dir=None,
        macros=None,
        include_dirs=None,
        debug=False,
        extra_preargs=None,
        extra_postargs=None,
        depends=None,
    ):
        if not self.initialized:
            self.initialize()
        compile_info = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs
        )
        macros, objects, extra_postargs, pp_opts, build = compile_info

        compile_opts = extra_preargs or []
        compile_opts.append('/c')
        if debug:
            compile_opts.extend(self.compile_options_debug)
        else:
            compile_opts.extend(self.compile_options)

        add_cpp_opts = False

        for obj in objects:
            try:
                src, ext = build[obj]
            except KeyError:
                continue
            if debug:
                # pass the full pathname to MSVC in debug mode,
                # this allows the debugger to find the source file
                # without asking the user to browse for it
                src = os.path.abspath(src)

            if ext in self._c_extensions:
                input_opt = f"/Tc{src}"
            elif ext in self._cpp_extensions:
                input_opt = f"/Tp{src}"
                add_cpp_opts = True
            elif ext in self._rc_extensions:
                # compile .RC to .RES file
                input_opt = src
                output_opt = "/fo" + obj
                try:
                    self.call([self.rc] + pp_opts + [output_opt, input_opt])
                except (subprocess.CalledProcessError, OSError) as msg:
                    raise CompileError(msg)
                continue
            elif ext in self._mc_extensions:
                # Compile .MC to .RC file to .RES file.
                #   * '-h dir' specifies the directory for the
                #     generated include file
                #   * '-r dir' specifies the target directory of the
                #     generated RC file and the binary message resource
                #     it includes
                #
                # For now (since there are no options to change this),
                # we use the source-directory for the include file and
                # the build directory for the RC file and message
                # resources. This works at least for win32all.
                h_dir = os.path.dirname(src)
                rc_dir = os.path.dirname(obj)
                try:
                    # first compile .MC to .RC and .H file
                    self.call([self.mc, '-h', h_dir, '-r', rc_dir, src])
                    base, _ = os.path.splitext(os.path.basename(src))
                    rc_file = os.path.join(rc_dir, base + '.rc')
                    # then compile .RC to .RES file
                    self.call([self.rc, "/fo" + obj, rc_file])

                except (subprocess.CalledProcessError, OSError) as msg:
                    raise CompileError(msg)
                continue
            else:
                # how to handle this file?
                raise CompileError(f"Don't know how to compile {src} to {obj}")

            args = [self.cc] + compile_opts + pp_opts
            if add_cpp_opts:
                args.append('/EHsc')
            args.extend((input_opt, "/Fo" + obj))
            args.extend(extra_postargs)

            try:
                self.call(args)
            except (subprocess.CalledProcessError, OSError) as msg:
                raise CompileError(msg)

        return objects

    def create_static_lib(
        self,
        objects: list[str] | tuple[str, ...],
        output_libname: str,
        output_dir: str | None = None,
        debug: bool = False,
        target_lang: str | None = None,
    ) -> None:
        if not self.initialized:
            self.initialize()
        objects, output_dir = self._fix_object_args(objects, output_dir)
        output_filename = self.library_filename(output_libname, output_dir=output_dir)

        if self._need_link(objects, output_filename):
            lib_args = objects + ['/OUT:' + output_filename]
            if debug:
                pass  # XXX what goes here?
            try:
                log.debug('Executing "%s" %s', self.lib, ' '.join(lib_args))
                self.call([self.lib] + lib_args)
            except (subprocess.CalledProcessError, OSError) as msg:
                raise LibError(msg)
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    def link(
        self,
        target_desc: str,
        objects: list[str] | tuple[str, ...],
        output_filename: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        export_symbols: Iterable[str] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: Iterable[str] | None = None,
        build_temp: str | os.PathLike[str] | None = None,
        target_lang: str | None = None,
    ) -> None:
        if not self.initialized:
            self.initialize()
        objects, output_dir = self._fix_object_args(objects, output_dir)
        fixed_args = self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)
        libraries, library_dirs, runtime_library_dirs = fixed_args

        if runtime_library_dirs:
            self.warn(
                "I don't know what to do with 'runtime_library_dirs': "
                + str(runtime_library_dirs)
            )

        lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs, libraries)
        if output_dir is not None:
            output_filename = os.path.join(output_dir, output_filename)

        if self._need_link(objects, output_filename):
            ldflags = self._ldflags[target_desc, debug]

            export_opts = ["/EXPORT:" + sym for sym in (export_symbols or [])]

            ld_args = (
                ldflags + lib_opts + export_opts + objects + ['/OUT:' + output_filename]
            )

            # The MSVC linker generates .lib and .exp files, which cannot be
            # suppressed by any linker switches. The .lib files may even be
            # needed! Make sure they are generated in the temporary build
            # directory. Since they have different names for debug and release
            # builds, they can go into the same directory.
            build_temp = os.path.dirname(objects[0])
            if export_symbols is not None:
                (dll_name, _dll_ext) = os.path.splitext(
                    os.path.basename(output_filename)
                )
                implib_file = os.path.join(build_temp, self.library_filename(dll_name))
                ld_args.append('/IMPLIB:' + implib_file)

            if extra_preargs:
                ld_args[:0] = extra_preargs
            if extra_postargs:
                ld_args.extend(extra_postargs)

            output_dir = os.path.dirname(os.path.abspath(output_filename))
            self.mkpath(output_dir)
            try:
                log.debug('Executing "%s" %s', self.linker, ' '.join(ld_args))
                with _wrap_link_command(self.linker, *ld_args) as cmd:
                    self.call(cmd)
            except (subprocess.CalledProcessError, OSError) as msg:
                raise LinkError(msg)
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    def call(self, cmd, *, env=None, **kwargs):
        env = dict(os.environ, PATH=self._paths)
        return super().call(cmd, env=env, **kwargs)

    # -- Miscellaneous methods -----------------------------------------
    # These are all used by the 'gen_lib_options() function, in
    # ccompiler.py.

    def library_dir_option(self, dir):
        return "/LIBPATH:" + dir

    def runtime_library_dir_option(self, dir):
        raise PlatformError(
            "don't know how to set runtime library search path for MSVC"
        )

    def library_option(self, lib):
        return self.library_filename(lib)

    def find_library_file(self, dirs, lib, debug=False):
        # Prefer a debugging library if found (and requested), but deal
        # with it if we don't have one.
        if debug:
            try_names = [lib + "_d", lib]
        else:
            try_names = [lib]
        for dir in dirs:
            for name in try_names:
                libfile = os.path.join(dir, self.library_filename(name))
                if os.path.isfile(libfile):
                    return libfile
        # Oops, didn't find it in *any* of 'dirs'
        return None
<|endoftext|>"""distutils.command.clean

Implements the Distutils 'clean' command."""

# contributed by Bastian Kleineidam <calvin@cs.uni-sb.de>, added 2000-03-18

import os
from distutils._log import log
from typing import ClassVar

from ..core import Command
from ..dir_util import remove_tree


class clean(Command):
    description = "clean up temporary files from 'build' command"
    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('build-base=', 'b', "base build directory [default: 'build.build-base']"),
        (
            'build-lib=',
            None,
            "build directory for all modules [default: 'build.build-lib']",
        ),
        ('build-temp=', 't', "temporary build directory [default: 'build.build-temp']"),
        (
            'build-scripts=',
            None,
            "build directory for scripts [default: 'build.build-scripts']",
        ),
        ('bdist-base=', None, "temporary directory for built distributions"),
        ('all', 'a', "remove all build output, not just temporary by-products"),
    ]

    boolean_options: ClassVar[list[str]] = ['all']

    def initialize_options(self):
        self.build_base = None
        self.build_lib = None
        self.build_temp = None
        self.build_scripts = None
        self.bdist_base = None
        self.all = None

    def finalize_options(self):
        self.set_undefined_options(
            'build',
            ('build_base', 'build_base'),
            ('build_lib', 'build_lib'),
            ('build_scripts', 'build_scripts'),
            ('build_temp', 'build_temp'),
        )
        self.set_undefined_options('bdist', ('bdist_base', 'bdist_base'))

    def run(self):
        # remove the build/temp.<plat> directory (unless it's already
        # gone)
        if os.path.exists(self.build_temp):
            remove_tree(self.build_temp)
        else:
            log.debug("'%s' does not exist -- can't clean it", self.build_temp)

        if self.all:
            # remove build directories
            for directory in (self.build_lib, self.bdist_base, self.build_scripts):
                if os.path.exists(directory):
                    remove_tree(directory)
                else:
                    log.warning("'%s' does not exist -- can't clean it", directory)

        # just for the heck of it, try to remove the base build directory:
        # we might have emptied it right now, but if not we don't care
        try:
            os.rmdir(self.build_base)
            log.info("removing '%s'", self.build_base)
        except OSError:
            pass
<|endoftext|>"""distutils.command.sdist

Implements the Distutils 'sdist' command (create a source distribution)."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from distutils import archive_util, dir_util, file_util
from distutils._log import log
from glob import glob
from itertools import filterfalse
from typing import Any, ClassVar

from ..core import Command
from ..errors import DistutilsOptionError, DistutilsTemplateError
from ..filelist import FileList
from ..text_file import TextFile
from ..util import convert_path


def show_formats():
    """Print all possible values for the 'formats' option (used by
    the "--help-formats" command-line option).
    """
    from ..archive_util import ARCHIVE_FORMATS
    from ..fancy_getopt import FancyGetopt

    formats = sorted(
        ("formats=" + format, None, ARCHIVE_FORMATS[format][2])
        for format in ARCHIVE_FORMATS
    )
    FancyGetopt(formats).print_help("List of available source distribution formats:")


class sdist(Command):
    description = "create a source distribution (tarball, zip file, etc.)"

    def checking_metadata(self) -> bool:
        """Callable used for the check sub-command.

        Placed here so user_options can view it"""
        return self.metadata_check

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('template=', 't', "name of manifest template file [default: MANIFEST.in]"),
        ('manifest=', 'm', "name of manifest file [default: MANIFEST]"),
        (
            'use-defaults',
            None,
            (
                "include the default file set in the manifest "
                "[default; disable with --no-defaults]"
            ),
        ),
        ('no-defaults', None, "don't include the default file set"),
        (
            'prune',
            None,
            (
                "specifically exclude files/directories that should not be "
                "distributed (build tree, RCS/CVS dirs, etc.) "
                "[default; disable with --no-prune]"
            ),
        ),
        ('no-prune', None, "don't automatically exclude anything"),
        (
            'manifest-only',
            'o',
            "just regenerate the manifest and then stop (implies --force-manifest)",
        ),
        (
            'force-manifest',
            'f',
            (
                "forcibly regenerate the manifest and carry on as usual. "
                "Deprecated: now the manifest is always regenerated."
            ),
        ),
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
            'metadata-check',
            None,
            (
                "Ensure that all required elements of meta-data "
                "are supplied. Warn if any missing. [default]"
            ),
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

    boolean_options: ClassVar[list[str]] = [
        'use-defaults',
        'prune',
        'manifest-only',
        'force-manifest',
        'keep-temp',
        'metadata-check',
    ]

    help_options: ClassVar[list[tuple[str, str | None, str, Callable[[], object]]]] = [
        ('help-formats', None, "list available distribution formats", show_formats),
    ]

    negative_opt: ClassVar[dict[str, str]] = {
        'no-defaults': 'use-defaults',
        'no-prune': 'prune',
    }

    sub_commands: ClassVar[list[tuple[str, Callable[[Any], bool] | None]]] = [
        ('check', checking_metadata)
    ]

    READMES: ClassVar[tuple[str, ...]] = ('README', 'README.txt', 'README.rst')

    def initialize_options(self):
        # 'template' and 'manifest' are, respectively, the names of
        # the manifest template and manifest file.
        self.template = None
        self.manifest = None

        # 'use_defaults': if true, we will include the default file set
        # in the manifest
        self.use_defaults = True
        self.prune = True

        self.manifest_only = False
        self.force_manifest = False

        self.formats = ['gztar']
        self.keep_temp = False
        self.dist_dir = None

        self.archive_files = None
        self.metadata_check = True
        self.owner = None
        self.group = None

    def finalize_options(self) -> None:
        if self.manifest is None:
            self.manifest = "MANIFEST"
        if self.template is None:
            self.template = "MANIFEST.in"

        self.ensure_string_list('formats')

        bad_format = archive_util.check_archive_formats(self.formats)
        if bad_format:
            raise DistutilsOptionError(f"unknown archive format '{bad_format}'")

        if self.dist_dir is None:
            self.dist_dir = "dist"

    def run(self) -> None:
        # 'filelist' contains the list of files that will make up the
        # manifest
        self.filelist = FileList()

        # Run sub commands
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

        # Do whatever it takes to get the list of files to process
        # (process the manifest template, read an existing manifest,
        # whatever).  File list is accumulated in 'self.filelist'.
        self.get_file_list()

        # If user just wanted us to regenerate the manifest, stop now.
        if self.manifest_only:
            return

        # Otherwise, go ahead and create the source distribution tarball,
        # or zipfile, or whatever.
        self.make_distribution()

    def get_file_list(self) -> None:
        """Figure out the list of files to include in the source
        distribution, and put it in 'self.filelist'.  This might involve
        reading the manifest template (and writing the manifest), or just
        reading the manifest, or just using the default file set -- it all
        depends on the user's options.
        """
        # new behavior when using a template:
        # the file list is recalculated every time because
        # even if MANIFEST.in or setup.py are not changed
        # the user might have added some files in the tree that
        # need to be included.
        #
        #  This makes --force the default and only behavior with templates.
        template_exists = os.path.isfile(self.template)
        if not template_exists and self._manifest_is_not_generated():
            self.read_manifest()
            self.filelist.sort()
            self.filelist.remove_duplicates()
            return

        if not template_exists:
            self.warn(
                f"manifest template '{self.template}' does not exist (using default file list)"
            )
        self.filelist.findall()

        if self.use_defaults:
            self.add_defaults()

        if template_exists:
            self.read_template()

        if self.prune:
            self.prune_file_list()

        self.filelist.sort()
        self.filelist.remove_duplicates()
        self.write_manifest()

    def add_defaults(self) -> None:
        """Add all the default files to self.filelist:
          - README or README.txt
          - setup.py
          - tests/test*.py and test/test*.py
          - all pure Python modules mentioned in setup script
          - all files pointed by package_data (build_py)
          - all files defined in data_files.
          - all files defined as scripts.
          - all C sources listed as part of extensions or C libraries
            in the setup script (doesn't catch C headers!)
        Warns if (README or README.txt) or setup.py are missing; everything
        else is optional.
        """
        self._add_defaults_standards()
        self._add_defaults_optional()
        self._add_defaults_python()
        self._add_defaults_data_files()
        self._add_defaults_ext()
        self._add_defaults_c_libs()
        self._add_defaults_scripts()

    @staticmethod
    def _cs_path_exists(fspath):
        """
        Case-sensitive path existence check

        >>> sdist._cs_path_exists(__file__)
        True
        >>> sdist._cs_path_exists(__file__.upper())
        False
        """
        if not os.path.exists(fspath):
            return False
        # make absolute so we always have a directory
        abspath = os.path.abspath(fspath)
        directory, filename = os.path.split(abspath)
        return filename in os.listdir(directory)

    def _add_defaults_standards(self):
        standards = [self.READMES, self.distribution.script_name]
        for fn in standards:
            if isinstance(fn, tuple):
                alts = fn
                got_it = False
                for fn in alts:
                    if self._cs_path_exists(fn):
                        got_it = True
                        self.filelist.append(fn)
                        break

                if not got_it:
                    self.warn(
                        "standard file not found: should have one of " + ', '.join(alts)
                    )
            else:
                if self._cs_path_exists(fn):
                    self.filelist.append(fn)
                else:
                    self.warn(f"standard file '{fn}' not found")

    def _add_defaults_optional(self):
        optional = ['tests/test*.py', 'test/test*.py', 'setup.cfg']
        for pattern in optional:
            files = filter(os.path.isfile, glob(pattern))
            self.filelist.extend(files)

    def _add_defaults_python(self):
        # build_py is used to get:
        #  - python modules
        #  - files defined in package_data
        build_py = self.get_finalized_command('build_py')

        # getting python files
        if self.distribution.has_pure_modules():
            self.filelist.extend(build_py.get_source_files())

        # getting package_data files
        # (computed in build_py.data_files by build_py.finalize_options)
        for _pkg, src_dir, _build_dir, filenames in build_py.data_files:
            for filename in filenames:
                self.filelist.append(os.path.join(src_dir, filename))

    def _add_defaults_data_files(self):
        # getting distribution.data_files
        if self.distribution.has_data_files():
            for item in self.distribution.data_files:
                if isinstance(item, str):
                    # plain file
                    item = convert_path(item)
                    if os.path.isfile(item):
                        self.filelist.append(item)
                else:
                    # a (dirname, filenames) tuple
                    _dirname, filenames = item
                    for f in filenames:
                        f = convert_path(f)
                        if os.path.isfile(f):
                            self.filelist.append(f)

    def _add_defaults_ext(self):
        if self.distribution.has_ext_modules():
            build_ext = self.get_finalized_command('build_ext')
            self.filelist.extend(build_ext.get_source_files())

    def _add_defaults_c_libs(self):
        if self.distribution.has_c_libraries():
            build_clib = self.get_finalized_command('build_clib')
            self.filelist.extend(build_clib.get_source_files())

    def _add_defaults_scripts(self):
        if self.distribution.has_scripts():
            build_scripts = self.get_finalized_command('build_scripts')
            self.filelist.extend(build_scripts.get_source_files())

    def read_template(self) -> None:
        """Read and parse manifest template file named by self.template.

        (usually "MANIFEST.in") The parsing and processing is done by
        'self.filelist', which updates itself accordingly.
        """
        log.info("reading manifest template '%s'", self.template)
        template = TextFile(
            self.template,
            strip_comments=True,
            skip_blanks=True,
            join_lines=True,
            lstrip_ws=True,
            rstrip_ws=True,
            collapse_join=True,
        )

        try:
            while True:
                line = template.readline()
                if line is None:  # end of file
                    break

                try:
                    self.filelist.process_template_line(line)
                # the call above can raise a DistutilsTemplateError for
                # malformed lines, or a ValueError from the lower-level
                # convert_path function
                except (DistutilsTemplateError, ValueError) as msg:
                    self.warn(
                        f"{template.filename}, line {int(template.current_line)}: {msg}"
                    )
        finally:
            template.close()

    def prune_file_list(self) -> None:
        """Prune off branches that might slip into the file list as created
        by 'read_template()', but really don't belong there:
          * the build tree (typically "build")
          * the release tree itself (only an issue if we ran "sdist"
            previously with --keep-temp, or it aborted)
          * any RCS, CVS, .svn, .hg, .git, .bzr, _darcs directories
        """
        build = self.get_finalized_command('build')
        base_dir = self.distribution.get_fullname()

        self.filelist.exclude_pattern(None, prefix=os.fspath(build.build_base))
        self.filelist.exclude_pattern(None, prefix=base_dir)

        if sys.platform == 'win32':
            seps = r'/|\\'
        else:
            seps = '/'

        vcs_dirs = ['RCS', 'CVS', r'\.svn', r'\.hg', r'\.git', r'\.bzr', '_darcs']
        vcs_ptrn = r'(^|{})({})({}).*'.format(seps, '|'.join(vcs_dirs), seps)
        self.filelist.exclude_pattern(vcs_ptrn, is_regex=True)

    def write_manifest(self) -> None:
        """Write the file list in 'self.filelist' (presumably as filled in
        by 'add_defaults()' and 'read_template()') to the manifest file
        named by 'self.manifest'.
        """
        if self._manifest_is_not_generated():
            log.info(
                f"not writing to manually maintained manifest file '{self.manifest}'"
            )
            return

        content = self.filelist.files[:]
        content.insert(0, '# file GENERATED by distutils, do NOT edit')
        self.execute(
            file_util.write_file,
            (self.manifest, content),
            f"writing manifest file '{self.manifest}'",
        )

    def _manifest_is_not_generated(self):
        # check for special comment used in 3.1.3 and higher
        if not os.path.isfile(self.manifest):
            return False

        with open(self.manifest, encoding='utf-8') as fp:
            first_line = next(fp)
        return first_line != '# file GENERATED by distutils, do NOT edit\n'

    def read_manifest(self) -> None:
        """Read the manifest file (named by 'self.manifest') and use it to
        fill in 'self.filelist', the list of files to include in the source
        distribution.
        """
        log.info("reading manifest file '%s'", self.manifest)
        with open(self.manifest, encoding='utf-8') as lines:
            self.filelist.extend(
                # ignore comments and blank lines
                filter(None, filterfalse(is_comment, map(str.strip, lines)))
            )

    def make_release_tree(self, base_dir, files) -> None:
        """Create the directory tree that will become the source
        distribution archive.  All directories implied by the filenames in
        'files' are created under 'base_dir', and then we hard link or copy
        (if hard linking is unavailable) those files into place.
        Essentially, this duplicates the developer's source tree, but in a
        directory named after the distribution, containing only the files
        to be distributed.
        """
        # Create all the directories under 'base_dir' necessary to
        # put 'files' there; the 'mkpath()' is just so we don't die
        # if the manifest happens to be empty.
        self.mkpath(base_dir)
        dir_util.create_tree(base_dir, files)

        # And walk over the list of files, either making a hard link (if
        # os.link exists) to each one that doesn't already exist in its
        # corresponding location under 'base_dir', or copying each file
        # that's out-of-date in 'base_dir'.  (Usually, all files will be
        # out-of-date, because by default we blow away 'base_dir' when
        # we're done making the distribution archives.)

        if hasattr(os, 'link'):  # can make hard links on this system
            link = 'hard'
            msg = f"making hard links in {base_dir}..."
        else:  # nope, have to copy
            link = None
            msg = f"copying files to {base_dir}..."

        if not files:
            log.warning("no files to distribute -- empty manifest?")
        else:
            log.info(msg)
        for file in files:
            if not os.path.isfile(file):
                log.warning("'%s' not a regular file -- skipping", file)
            else:
                dest = os.path.join(base_dir, file)
                self.copy_file(file, dest, link=link)

        self.distribution.metadata.write_pkg_info(base_dir)

    def make_distribution(self) -> None:
        """Create the source distribution(s).  First, we create the release
        tree with 'make_release_tree()'; then, we create all required
        archive files (according to 'self.formats') from the release tree.
        Finally, we clean up by blowing away the release tree (unless
        'self.keep_temp' is true).  The list of archive files created is
        stored so it can be retrieved later by 'get_archive_files()'.
        """
        # Don't warn about missing meta-data here -- should be (and is!)
        # done elsewhere.
        base_dir = self.distribution.get_fullname()
        base_name = os.path.join(self.dist_dir, base_dir)

        self.make_release_tree(base_dir, self.filelist.files)
        archive_files = []  # remember names of files we create
        # tar archive must be created last to avoid overwrite and remove
        if 'tar' in self.formats:
            self.formats.append(self.formats.pop(self.formats.index('tar')))

        for fmt in self.formats:
            file = self.make_archive(
                base_name, fmt, base_dir=base_dir, owner=self.owner, group=self.group
            )
            archive_files.append(file)
            self.distribution.dist_files.append(('sdist', '', file))

        self.archive_files = archive_files

        if not self.keep_temp:
            dir_util.remove_tree(base_dir)

    def get_archive_files(self):
        """Return the list of archive files created when the command
        was run, or None if the command hasn't run yet.
        """
        return self.archive_files


def is_comment(line: str) -> bool:
    return line.startswith('#')
<|endoftext|>"""distutils.command.build_scripts

Implements the Distutils 'build_scripts' command."""

import os
import re
import tokenize
from distutils._log import log
from stat import ST_MODE
from typing import ClassVar

from .._modified import newer
from ..core import Command
from ..util import convert_path

shebang_pattern = re.compile('^#!.*python[0-9.]*([ \t].*)?$')
"""
Pattern matching a Python interpreter indicated in first line of a script.
"""

# for Setuptools compatibility
first_line_re = shebang_pattern


class build_scripts(Command):
    description = "\"build\" scripts (copy and fixup #! line)"

    user_options: ClassVar[list[tuple[str, str, str]]] = [
        ('build-dir=', 'd', "directory to \"build\" (copy) to"),
        ('force', 'f', "forcibly build everything (ignore file timestamps"),
        ('executable=', 'e', "specify final destination interpreter path"),
    ]

    boolean_options: ClassVar[list[str]] = ['force']

    def initialize_options(self):
        self.build_dir = None
        self.scripts = None
        self.force = None
        self.executable = None

    def finalize_options(self):
        self.set_undefined_options(
            'build',
            ('build_scripts', 'build_dir'),
            ('force', 'force'),
            ('executable', 'executable'),
        )
        self.scripts = self.distribution.scripts

    def get_source_files(self):
        return self.scripts

    def run(self):
        if not self.scripts:
            return
        self.copy_scripts()

    def copy_scripts(self):
        """
        Copy each script listed in ``self.scripts``.

        If a script is marked as a Python script (first line matches
        'shebang_pattern', i.e. starts with ``#!`` and contains
        "python"), then adjust in the copy the first line to refer to
        the current Python interpreter.
        """
        self.mkpath(self.build_dir)
        outfiles = []
        updated_files = []
        for script in self.scripts:
            self._copy_script(script, outfiles, updated_files)

        self._change_modes(outfiles)

        return outfiles, updated_files

    def _copy_script(self, script, outfiles, updated_files):
        shebang_match = None
        script = convert_path(script)
        outfile = os.path.join(self.build_dir, os.path.basename(script))
        outfiles.append(outfile)

        if not self.force and not newer(script, outfile):
            log.debug("not copying %s (up-to-date)", script)
            return

        # Always open the file, but ignore failures in dry-run mode
        # in order to attempt to copy directly.
        with tokenize.open(script) as f:
            first_line = f.readline()
            if not first_line:
                self.warn(f"{script} is an empty file (skipping)")
                return

            shebang_match = shebang_pattern.match(first_line)

            updated_files.append(outfile)
            if shebang_match:
                log.info("copying and adjusting %s -> %s", script, self.build_dir)
                post_interp = shebang_match.group(1) or ''
                shebang = "#!" + self.executable + post_interp + "\n"
                self._validate_shebang(shebang, f.encoding)
                with open(outfile, "w", encoding=f.encoding) as outf:
                    outf.write(shebang)
                    outf.writelines(f.readlines())
            else:
                self.copy_file(script, outfile)

    def _change_modes(self, outfiles):
        if os.name != 'posix':
            return

        for file in outfiles:
            self._change_mode(file)

    def _change_mode(self, file):
        oldmode = os.stat(file)[ST_MODE] & 0o7777
        newmode = (oldmode | 0o555) & 0o7777
        if newmode != oldmode:
            log.info("changing mode of %s from %o to %o", file, oldmode, newmode)
            os.chmod(file, newmode)

    @staticmethod
    def _validate_shebang(shebang, encoding):
        # Python parser starts to read a script using UTF-8 until
        # it gets a #coding:xxx cookie. The shebang has to be the
        # first line of a file, the #coding:xxx cookie cannot be
        # written before. So the shebang has to be encodable to
        # UTF-8.
        try:
            shebang.encode('utf-8')
        except UnicodeEncodeError:
            raise ValueError(f"The shebang ({shebang!r}) is not encodable to utf-8")

        # If the script is encoded to a custom encoding (use a
        # #coding:xxx cookie), the shebang has to be encodable to
        # the script encoding too.
        try:
            shebang.encode(encoding)
        except UnicodeEncodeError:
            raise ValueError(
                f"The shebang ({shebang!r}) is not encodable "
                f"to the script encoding ({encoding})"
            )
<|endoftext|>"""distutils.command.check

Implements the Distutils 'check' command.
"""

import contextlib
from typing import ClassVar

from ..core import Command
from ..errors import DistutilsSetupError

with contextlib.suppress(ImportError):
    import docutils.frontend
    import docutils.nodes
    import docutils.parsers.rst
    import docutils.utils

    class SilentReporter(docutils.utils.Reporter):
        def __init__(
            self,
            source,
            report_level,
            halt_level,
            stream=None,
            debug=False,
            encoding='ascii',
            error_handler='replace',
        ):
            self.messages = []
            super().__init__(
                source, report_level, halt_level, stream, debug, encoding, error_handler
            )

        def system_message(self, level, message, *children, **kwargs):
            self.messages.append((level, message, children, kwargs))
            return docutils.nodes.system_message(
                message, *children, level=level, type=self.levels[level], **kwargs
            )


class check(Command):
    """This command checks the meta-data of the package."""

    description = "perform some checks on the package"
    user_options: ClassVar[list[tuple[str, str, str]]] = [
        ('metadata', 'm', 'Verify meta-data'),
        (
            'restructuredtext',
            'r',
            'Checks if long string meta-data syntax are reStructuredText-compliant',
        ),
        ('strict', 's', 'Will exit with an error if a check fails'),
    ]

    boolean_options: ClassVar[list[str]] = ['metadata', 'restructuredtext', 'strict']

    def initialize_options(self):
        """Sets default values for options."""
        self.restructuredtext = False
        self.metadata = 1
        self.strict = False
        self._warnings = 0

    def finalize_options(self):
        pass

    def warn(self, msg):
        """Counts the number of warnings that occurs."""
        self._warnings += 1
        return Command.warn(self, msg)

    def run(self):
        """Runs the command."""
        # perform the various tests
        if self.metadata:
            self.check_metadata()
        if self.restructuredtext:
            if 'docutils' in globals():
                try:
                    self.check_restructuredtext()
                except TypeError as exc:
                    raise DistutilsSetupError(str(exc))
            elif self.strict:
                raise DistutilsSetupError('The docutils package is needed.')

        # let's raise an error in strict mode, if we have at least
        # one warning
        if self.strict and self._warnings > 0:
            raise DistutilsSetupError('Please correct your package.')

    def check_metadata(self):
        """Ensures that all required elements of meta-data are supplied.

        Required fields:
            name, version

        Warns if any are missing.
        """
        metadata = self.distribution.metadata

        missing = [
            attr for attr in ('name', 'version') if not getattr(metadata, attr, None)
        ]

        if missing:
            self.warn("missing required meta-data: {}".format(', '.join(missing)))

    def check_restructuredtext(self):
        """Checks if the long string fields are reST-compliant."""
        data = self.distribution.get_long_description()
        for warning in self._check_rst_data(data):
            line = warning[-1].get('line')
            if line is None:
                warning = warning[1]
            else:
                warning = f'{warning[1]} (line {line})'
            self.warn(warning)

    def _check_rst_data(self, data):
        """Returns warnings when the provided data doesn't compile."""
        # the include and csv_table directives need this to be a path
        source_path = self.distribution.script_name or 'setup.py'
        parser = docutils.parsers.rst.Parser()
        settings = docutils.frontend.OptionParser(
            components=(docutils.parsers.rst.Parser,)
        ).get_default_values()
        settings.tab_width = 4
        settings.pep_references = None
        settings.rfc_references = None
        reporter = SilentReporter(
            source_path,
            settings.report_level,
            settings.halt_level,
            stream=settings.warning_stream,
            debug=settings.debug,
            encoding=settings.error_encoding,
            error_handler=settings.error_encoding_error_handler,
        )

        document = docutils.nodes.document(settings, reporter, source=source_path)
        document.note_source(source_path, -1)
        try:
            parser.parse(data, document)
        except (AttributeError, TypeError) as e:
            reporter.messages.append((
                -1,
                f'Could not finish the parsing: {e}.',
                '',
                {},
            ))

        return reporter.messages
<|endoftext|>"""distutils.command.install

Implements the Distutils 'install' command."""

from __future__ import annotations

import collections
import contextlib
import itertools
import os
import sys
import sysconfig
from collections.abc import Callable
from distutils._log import log
from site import USER_BASE, USER_SITE
from typing import Any, ClassVar

from ..core import Command
from ..debug import DEBUG
from ..errors import DistutilsOptionError, DistutilsPlatformError
from ..file_util import write_file
from ..sysconfig import get_config_vars
from ..util import change_root, convert_path, get_platform, subst_vars
from . import _framework_compat as fw

HAS_USER_SITE = True

WINDOWS_SCHEME = {
    'purelib': '{base}/Lib/site-packages',
    'platlib': '{base}/Lib/site-packages',
    'headers': '{base}/Include/{dist_name}',
    'scripts': '{base}/Scripts',
    'data': '{base}',
}

INSTALL_SCHEMES = {
    'posix_prefix': {
        'purelib': '{base}/lib/{implementation_lower}{py_version_short}/site-packages',
        'platlib': '{platbase}/{platlibdir}/{implementation_lower}'
        '{py_version_short}/site-packages',
        'headers': '{base}/include/{implementation_lower}'
        '{py_version_short}{abiflags}/{dist_name}',
        'scripts': '{base}/bin',
        'data': '{base}',
    },
    'posix_home': {
        'purelib': '{base}/lib/{implementation_lower}',
        'platlib': '{base}/{platlibdir}/{implementation_lower}',
        'headers': '{base}/include/{implementation_lower}/{dist_name}',
        'scripts': '{base}/bin',
        'data': '{base}',
    },
    'nt': WINDOWS_SCHEME,
    'pypy': {
        'purelib': '{base}/site-packages',
        'platlib': '{base}/site-packages',
        'headers': '{base}/include/{dist_name}',
        'scripts': '{base}/bin',
        'data': '{base}',
    },
    'pypy_nt': {
        'purelib': '{base}/site-packages',
        'platlib': '{base}/site-packages',
        'headers': '{base}/include/{dist_name}',
        'scripts': '{base}/Scripts',
        'data': '{base}',
    },
}

# user site schemes
if HAS_USER_SITE:
    INSTALL_SCHEMES['nt_user'] = {
        'purelib': '{usersite}',
        'platlib': '{usersite}',
        'headers': '{userbase}/{implementation}{py_version_nodot_plat}'
        '/Include/{dist_name}',
        'scripts': '{userbase}/{implementation}{py_version_nodot_plat}/Scripts',
        'data': '{userbase}',
    }

    INSTALL_SCHEMES['posix_user'] = {
        'purelib': '{usersite}',
        'platlib': '{usersite}',
        'headers': '{userbase}/include/{implementation_lower}'
        '{py_version_short}{abiflags}/{dist_name}',
        'scripts': '{userbase}/bin',
        'data': '{userbase}',
    }


INSTALL_SCHEMES.update(fw.schemes)


# The keys to an installation scheme; if any new types of files are to be
# installed, be sure to add an entry to every installation scheme above,
# and to SCHEME_KEYS here.
SCHEME_KEYS = ('purelib', 'platlib', 'headers', 'scripts', 'data')


def _load_sysconfig_schemes():
    with contextlib.suppress(AttributeError):
        return {
            scheme: sysconfig.get_paths(scheme, expand=False)
            for scheme in sysconfig.get_scheme_names()
        }


def _load_schemes():
    """
    Extend default schemes with schemes from sysconfig.
    """

    sysconfig_schemes = _load_sysconfig_schemes() or {}

    return {
        scheme: {
            **INSTALL_SCHEMES.get(scheme, {}),
            **sysconfig_schemes.get(scheme, {}),
        }
        for scheme in set(itertools.chain(INSTALL_SCHEMES, sysconfig_schemes))
    }


def _get_implementation():
    if hasattr(sys, 'pypy_version_info'):
        return 'PyPy'
    else:
        return 'Python'


def _select_scheme(ob, name):
    scheme = _inject_headers(name, _load_scheme(_resolve_scheme(name)))
    vars(ob).update(_remove_set(ob, _scheme_attrs(scheme)))


def _remove_set(ob, attrs):
    """
    Include only attrs that are None in ob.
    """
    return {key: value for key, value in attrs.items() if getattr(ob, key) is None}


def _resolve_scheme(name):
    _os_name, _sep, key = name.partition('_')
    try:
        resolved = sysconfig.get_preferred_scheme(key)
    except Exception:  # noqa: BLE001 # any failure falls back to the framework scheme
        resolved = fw.scheme(name)
    return resolved


def _load_scheme(name):
    return _load_schemes()[name]


def _inject_headers(name, scheme):
    """
    Given a scheme name and the resolved scheme,
    if the scheme does not include headers, resolve
    the fallback scheme for the name and use headers
    from it. pypa/distutils#88
    """
    # Bypass the preferred scheme, which may not
    # have defined headers.
    fallback = _load_scheme(name)
    scheme.setdefault('headers', fallback['headers'])
    return scheme


def _scheme_attrs(scheme):
    """Resolve install directories by applying the install schemes."""
    return {f'install_{key}': scheme[key] for key in SCHEME_KEYS}


class install(Command):
    description = "install everything from build directory"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        # Select installation scheme and set base director(y|ies)
        ('prefix=', None, "installation prefix"),
        ('exec-prefix=', None, "(Unix only) prefix for platform-specific files"),
        ('home=', None, "(Unix only) home directory to install under"),
        # Or, just set the base director(y|ies)
        (
            'install-base=',
            None,
            "base installation directory (instead of --prefix or --home)",
        ),
        (
            'install-platbase=',
            None,
            "base installation directory for platform-specific files (instead of --exec-prefix or --home)",
        ),
        ('root=', None, "install everything relative to this alternate root directory"),
        # Or, explicitly set the installation scheme
        (
            'install-purelib=',
            None,
            "installation directory for pure Python module distributions",
        ),
        (
            'install-platlib=',
            None,
            "installation directory for non-pure module distributions",
        ),
        (
            'install-lib=',
            None,
            "installation directory for all module distributions (overrides --install-purelib and --install-platlib)",
        ),
        ('install-headers=', None, "installation directory for C/C++ headers"),
        ('install-scripts=', None, "installation directory for Python scripts"),
        ('install-data=', None, "installation directory for data files"),
        # Byte-compilation options -- see install_lib.py for details, as
        # these are duplicated from there (but only install_lib does
        # anything with them).
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
        # Miscellaneous control options
        ('force', 'f', "force installation (overwrite any existing files)"),
        ('skip-build', None, "skip rebuilding everything (for testing/debugging)"),
        # Where to install documentation (eventually!)
        # ('doc-format=', None, "format of documentation to generate"),
        # ('install-man=', None, "directory for Unix man pages"),
        # ('install-html=', None, "directory for HTML documentation"),
        # ('install-info=', None, "directory for GNU info files"),
        ('record=', None, "filename in which to record list of installed files"),
    ]

    boolean_options: ClassVar[list[str]] = ['compile', 'force', 'skip-build']

    if HAS_USER_SITE:
        user_options.append((
            'user',
            None,
            f"install in user site-package '{USER_SITE}'",
        ))
        boolean_options.append('user')

    negative_opt: ClassVar[dict[str, str]] = {'no-compile': 'compile'}

    def initialize_options(self) -> None:
        """Initializes options."""
        # High-level options: these select both an installation base
        # and scheme.
        self.prefix: str | None = None
        self.exec_prefix: str | None = None
        self.home: str | None = None
        self.user = False

        # These select only the installation base; it's up to the user to
        # specify the installation scheme (currently, that means supplying
        # the --install-{platlib,purelib,scripts,data} options).
        self.install_base: str | None = None
        self.install_platbase: str | None = None
        self.root: str | None = None

        # These options are the actual installation directories; if not
        # supplied by the user, they are filled in using the installation
        # scheme implied by prefix/exec-prefix/home and the contents of
        # that installation scheme.
        self.install_purelib: str | None = None  # for pure module distributions
        self.install_platlib: str | None = None  # non-pure (dists w/ extensions)
        self.install_headers: str | None = None  # for C/C++ headers
        self.install_lib: str | None = None  # set to either purelib or platlib
        self.install_scripts: str | None = None
        self.install_data: str | None = None
        self.install_userbase = USER_BASE
        self.install_usersite = USER_SITE

        self.compile = None
        self.optimize = None

        # Deprecated
        # These two are for putting non-packagized distributions into their
        # own directory and creating a .pth file if it makes sense.
        # 'extra_path' comes from the setup file; 'install_path_file' can
        # be turned off if it makes no sense to install a .pth file.  (But
        # better to install it uselessly than to guess wrong and not
        # install it when it's necessary and would be used!)  Currently,
        # 'install_path_file' is always true unless some outsider meddles
        # with it.
        self.extra_path = None
        self.install_path_file = True

        # 'force' forces installation, even if target files are not
        # out-of-date.  'skip_build' skips running the "build" command,
        # handy if you know it's not necessary.  'warn_dir' (which is *not*
        # a user option, it's just there so the bdist_* commands can turn
        # it off) determines whether we warn about installing to a
        # directory not in sys.path.
        self.force = False
        self.skip_build = False
        self.warn_dir = True

        # These are only here as a conduit from the 'build' command to the
        # 'install_*' commands that do the real work.  ('build_base' isn't
        # actually used anywhere, but it might be useful in future.)  They
        # are not user options, because if the user told the install
        # command where the build directory is, that wouldn't affect the
        # build command.
        self.build_base = None
        self.build_lib = None

        # Not defined yet because we don't know anything about
        # documentation yet.
        # self.install_man = None
        # self.install_html = None
        # self.install_info = None

        self.record = None

    # -- Option finalizing methods -------------------------------------
    # (This is rather more involved than for most commands,
    # because this is where the policy for installing third-
    # party Python modules on various platforms given a wide
    # array of user input is decided.  Yes, it's quite complex!)

    def finalize_options(self) -> None:  # noqa: C901
        """Finalizes options."""
        # This method (and its helpers, like 'finalize_unix()',
        # 'finalize_other()', and 'select_scheme()') is where the default
        # installation directories for modules, extension modules, and
        # anything else we care to install from a Python module
        # distribution.  Thus, this code makes a pretty important policy
        # statement about how third-party stuff is added to a Python
        # installation!  Note that the actual work of installation is done
        # by the relatively simple 'install_*' commands; they just take
        # their orders from the installation directory options determined
        # here.

        # Check for errors/inconsistencies in the options; first, stuff
        # that's wrong on any platform.

        if (self.prefix or self.exec_prefix or self.home) and (
            self.install_base or self.install_platbase
        ):
            raise DistutilsOptionError(
                "must supply either prefix/exec-prefix/home or install-base/install-platbase -- not both"
            )

        if self.home and (self.prefix or self.exec_prefix):
            raise DistutilsOptionError(
                "must supply either home or prefix/exec-prefix -- not both"
            )

        if self.user and (
            self.prefix
            or self.exec_prefix
            or self.home
            or self.install_base
            or self.install_platbase
        ):
            raise DistutilsOptionError(
                "can't combine user with prefix, "
                "exec_prefix/home, or install_(plat)base"
            )

        # Next, stuff that's wrong (or dubious) only on certain platforms.
        if os.name != "posix" and self.exec_prefix:
            self.warn("exec-prefix option ignored on this platform")
            self.exec_prefix = None

        # Now the interesting logic -- so interesting that we farm it out
        # to other methods.  The goal of these methods is to set the final
        # values for the install_{lib,scripts,data,...}  options, using as
        # input a heady brew of prefix, exec_prefix, home, install_base,
        # install_platbase, user-supplied versions of
        # install_{purelib,platlib,lib,scripts,data,...}, and the
        # install schemes.  Phew!

        self.dump_dirs("pre-finalize_{unix,other}")

        if os.name == 'posix':
            self.finalize_unix()
        else:
            self.finalize_other()

        self.dump_dirs("post-finalize_{unix,other}()")

        # Expand configuration variables, tilde, etc. in self.install_base
        # and self.install_platbase -- that way, we can use $base or
        # $platbase in the other installation directories and not worry
        # about needing recursive variable expansion (shudder).

        py_version = sys.version.split()[0]
        (prefix, exec_prefix) = get_config_vars('prefix', 'exec_prefix')
        try:
            abiflags = sys.abiflags
        except AttributeError:
            # sys.abiflags may not be defined on all platforms.
            abiflags = ''
        local_vars = {
            'dist_name': self.distribution.get_name(),
            'dist_version': self.distribution.get_version(),
            'dist_fullname': self.distribution.get_fullname(),
            'py_version': py_version,
            'py_version_short': f'{sys.version_info.major}.{sys.version_info.minor}',
            'py_version_nodot': f'{sys.version_info.major}{sys.version_info.minor}',
            'sys_prefix': prefix,
            'prefix': prefix,
            'sys_exec_prefix': exec_prefix,
            'exec_prefix': exec_prefix,
            'abiflags': abiflags,
            'platlibdir': getattr(sys, 'platlibdir', 'lib'),
            'implementation_lower': _get_implementation().lower(),
            'implementation': _get_implementation(),
        }

        # vars for compatibility on older Pythons
        compat_vars = {
            # Python 3.9 and earlier
            'py_version_nodot_plat': getattr(sys, 'winver', '').replace('.', ''),
        }

        if HAS_USER_SITE:
            local_vars['userbase'] = self.install_userbase
            local_vars['usersite'] = self.install_usersite

        self.config_vars = collections.ChainMap(
            local_vars,
            sysconfig.get_config_vars(),
            compat_vars,
            fw.vars(),
        )

        self.expand_basedirs()

        self.dump_dirs("post-expand_basedirs()")

        # Now define config vars for the base directories so we can expand
        # everything else.
        local_vars['base'] = self.install_base
        local_vars['platbase'] = self.install_platbase

        if DEBUG:
            from pprint import pprint

            print("config vars:")
            pprint(dict(self.config_vars))

        # Expand "~" and configuration variables in the installation
        # directories.
        self.expand_dirs()

        self.dump_dirs("post-expand_dirs()")

        # Create directories in the home dir:
        if self.user:
            self.create_home_path()

        # Pick the actual directory to install all modules to: either
        # install_purelib or install_platlib, depending on whether this
        # module distribution is pure or not.  Of course, if the user
        # already specified install_lib, use their selection.
        if self.install_lib is None:
            if self.distribution.has_ext_modules():  # has extensions: non-pure
                self.install_lib = self.install_platlib
            else:
                self.install_lib = self.install_purelib

        # Convert directories from Unix /-separated syntax to the local
        # convention.
        self.convert_paths(
            'lib',
            'purelib',
            'platlib',
            'scripts',
            'data',
            'headers',
            'userbase',
            'usersite',
        )

        # Deprecated
        # Well, we're not actually fully completely finalized yet: we still
        # have to deal with 'extra_path', which is the hack for allowing
        # non-packagized module distributions (hello, Numerical Python!) to
        # get their own directories.
        self.handle_extra_path()
        self.install_libbase = self.install_lib  # needed for .pth file
        self.install_lib = os.path.join(self.install_lib, self.extra_dirs)

        # If a new root directory was supplied, make all the installation
        # dirs relative to it.
        if self.root is not None:
            self.change_roots(
                'libbase', 'lib', 'purelib', 'platlib', 'scripts', 'data', 'headers'
            )

        self.dump_dirs("after prepending root")

        # Find out the build directories, ie. where to install from.
        self.set_undefined_options(
            'build', ('build_base', 'build_base'), ('build_lib', 'build_lib')
        )

        # Punt on doc directories for now -- after all, we're punting on
        # documentation completely!

    def dump_dirs(self, msg) -> None:
        """Dumps the list of user options."""
        if not DEBUG:
            return
        from ..fancy_getopt import longopt_xlate

        log.debug(msg + ":")
        for opt in self.user_options:
            opt_name = opt[0]
            if opt_name[-1] == "=":
                opt_name = opt_name[0:-1]
            if opt_name in self.negative_opt:
                opt_name = self.negative_opt[opt_name]
                opt_name = opt_name.translate(longopt_xlate)
                val = not getattr(self, opt_name)
            else:
                opt_name = opt_name.translate(longopt_xlate)
                val = getattr(self, opt_name)
            log.debug("  %s: %s", opt_name, val)

    def finalize_unix(self) -> None:
        """Finalizes options for posix platforms."""
        if self.install_base is not None or self.install_platbase is not None:
            incomplete_scheme = (
                (
                    self.install_lib is None
                    and self.install_purelib is None
                    and self.install_platlib is None
                )
                or self.install_headers is None
                or self.install_scripts is None
                or self.install_data is None
            )
            if incomplete_scheme:
                raise DistutilsOptionError(
                    "install-base or install-platbase supplied, but "
                    "installation scheme is incomplete"
                )
            return

        if self.user:
            if self.install_userbase is None:
                raise DistutilsPlatformError("User base directory is not specified")
            self.install_base = self.install_platbase = self.install_userbase
            self.select_scheme("posix_user")
        elif self.home is not None:
            self.install_base = self.install_platbase = self.home
            self.select_scheme("posix_home")
        else:
            if self.prefix is None:
                if self.exec_prefix is not None:
                    raise DistutilsOptionError(
                        "must not supply exec-prefix without prefix"
                    )

                # Allow Fedora to add components to the prefix
                _prefix_addition = getattr(sysconfig, '_prefix_addition', "")

                self.prefix = os.path.normpath(sys.prefix) + _prefix_addition
                self.exec_prefix = os.path.normpath(sys.exec_prefix) + _prefix_addition

            else:
                if self.exec_prefix is None:
                    self.exec_prefix = self.prefix

            self.install_base = self.prefix
            self.install_platbase = self.exec_prefix
            self.select_scheme("posix_prefix")

    def finalize_other(self) -> None:
        """Finalizes options for non-posix platforms"""
        if self.user:
            if self.install_userbase is None:
                raise DistutilsPlatformError("User base directory is not specified")
            self.install_base = self.install_platbase = self.install_userbase
            self.select_scheme(os.name + "_user")
        elif self.home is not None:
            self.install_base = self.install_platbase = self.home
            self.select_scheme("posix_home")
        else:
            if self.prefix is None:
                self.prefix = os.path.normpath(sys.prefix)

            self.install_base = self.install_platbase = self.prefix
            try:
                self.select_scheme(os.name)
            except KeyError:
                raise DistutilsPlatformError(
                    f"I don't know how to install stuff on '{os.name}'"
                )

    def select_scheme(self, name) -> None:
        _select_scheme(self, name)

    def _expand_attrs(self, attrs):
        for attr in attrs:
            val = getattr(self, attr)
            if val is not None:
                if os.name in ('posix', 'nt'):
                    val = os.path.expanduser(val)
                val = subst_vars(val, self.config_vars)
                setattr(self, attr, val)

    def expand_basedirs(self) -> None:
        """Calls `os.path.expanduser` on install_base, install_platbase and
        root."""
        self._expand_attrs(['install_base', 'install_platbase', 'root'])

    def expand_dirs(self) -> None:
        """Calls `os.path.expanduser` on install dirs."""
        self._expand_attrs([
            'install_purelib',
            'install_platlib',
            'install_lib',
            'install_headers',
            'install_scripts',
            'install_data',
        ])

    def convert_paths(self, *names) -> None:
        """Call `convert_path` over `names`."""
        for name in names:
            attr = "install_" + name
            setattr(self, attr, convert_path(getattr(self, attr)))

    def handle_extra_path(self) -> None:
        """Set `path_file` and `extra_dirs` using `extra_path`."""
        if self.extra_path is None:
            self.extra_path = self.distribution.extra_path

        if self.extra_path is not None:
            log.warning(
                "Distribution option extra_path is deprecated. "
                "See issue27919 for details."
            )
            if isinstance(self.extra_path, str):
                self.extra_path = self.extra_path.split(',')

            if len(self.extra_path) == 1:
                path_file = extra_dirs = self.extra_path[0]
            elif len(self.extra_path) == 2:
                path_file, extra_dirs = self.extra_path
            else:
                raise DistutilsOptionError(
                    "'extra_path' option must be a list, tuple, or "
                    "comma-separated string with 1 or 2 elements"
                )

            # convert to local form in case Unix notation used (as it
            # should be in setup scripts)
            extra_dirs = convert_path(extra_dirs)
        else:
            path_file = None
            extra_dirs = ''

        # XXX should we warn if path_file and not extra_dirs? (in which
        # case the path file would be harmless but pointless)
        self.path_file = path_file
        self.extra_dirs = extra_dirs

    def change_roots(self, *names) -> None:
        """Change the install directories pointed by name using root."""
        for name in names:
            attr = "install_" + name
            setattr(self, attr, change_root(self.root, getattr(self, attr)))

    def create_home_path(self) -> None:
        """Create directories under ~."""
        if not self.user:
            return
        home = convert_path(os.path.expanduser("~"))
        for path in self.config_vars.values():
            if str(path).startswith(home) and not os.path.isdir(path):
                self.debug_print(f"os.makedirs('{path}', 0o700)")
                os.makedirs(path, 0o700)

    # -- Command execution methods -------------------------------------

    def run(self):
        """Runs the command."""
        # Obviously have to build before we can install
        if not self.skip_build:
            self.run_command('build')
            # If we built for any other platform, we can't install.
            build_plat = self.distribution.get_command_obj('build').plat_name
            # check warn_dir - it is a clue that the 'install' is happening
            # internally, and not to sys.path, so we don't check the platform
            # matches what we are running.
            if self.warn_dir and build_plat != get_platform():
                raise DistutilsPlatformError("Can't install when cross-compiling")

        # Run all sub-commands (at least those that need to be run)
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

        if self.path_file:
            self.create_path_file()

        # write list of installed files, if requested.
        if self.record:
            outputs = self.get_outputs()
            if self.root:  # strip any package prefix
                root_len = len(self.root)
                for counter in range(len(outputs)):
                    outputs[counter] = outputs[counter][root_len:]
            self.execute(
                write_file,
                (self.record, outputs),
                f"writing list of installed files to '{self.record}'",
            )

        sys_path = map(os.path.normpath, sys.path)
        sys_path = map(os.path.normcase, sys_path)
        install_lib = os.path.normcase(os.path.normpath(self.install_lib))
        if (
            self.warn_dir
            and not (self.path_file and self.install_path_file)
            and install_lib not in sys_path
        ):
            log.debug(
                (
                    "modules installed to '%s', which is not in "
                    "Python's module search path (sys.path) -- "
                    "you'll have to change the search path yourself"
                ),
                self.install_lib,
            )

    def create_path_file(self):
        """Creates the .pth file"""
        filename = os.path.join(self.install_libbase, self.path_file + ".pth")
        if self.install_path_file:
            self.execute(
                write_file, (filename, [self.extra_dirs]), f"creating {filename}"
            )
        else:
            self.warn(f"path file '{filename}' not created")

    # -- Reporting methods ---------------------------------------------

    def get_outputs(self):
        """Assembles the outputs of all the sub-commands."""
        outputs = []
        for cmd_name in self.get_sub_commands():
            cmd = self.get_finalized_command(cmd_name)
            # Add the contents of cmd.get_outputs(), ensuring
            # that outputs doesn't contain duplicate entries
            for filename in cmd.get_outputs():
                if filename not in outputs:
                    outputs.append(filename)

        if self.path_file and self.install_path_file:
            outputs.append(os.path.join(self.install_libbase, self.path_file + ".pth"))

        return outputs

    def get_inputs(self):
        """Returns the inputs of all the sub-commands"""
        # XXX gee, this looks familiar ;-(
        inputs = []
        for cmd_name in self.get_sub_commands():
            cmd = self.get_finalized_command(cmd_name)
            inputs.extend(cmd.get_inputs())

        return inputs

    # -- Predicates for sub-command list -------------------------------

    def has_lib(self) -> bool:
        """Returns true if the current distribution has any Python
        modules to install."""
        return (
            self.distribution.has_pure_modules() or self.distribution.has_ext_modules()
        )

    def has_headers(self) -> bool:
        """Returns true if the current distribution has any headers to
        install."""
        return self.distribution.has_headers()

    def has_scripts(self) -> bool:
        """Returns true if the current distribution has any scripts to.
        install."""
        return self.distribution.has_scripts()

    def has_data(self) -> bool:
        """Returns true if the current distribution has any data to.
        install."""
        return self.distribution.has_data_files()

    # 'sub_commands': a list of commands this command might have to run to
    # get its work done.  See cmd.py for more info.
    sub_commands: ClassVar[list[tuple[str, Callable[[Any], bool] | None]]] = [
        ('install_lib', has_lib),
        ('install_headers', has_headers),
        ('install_scripts', has_scripts),
        ('install_data', has_data),
        ('install_egg_info', lambda self: True),
    ]
<|endoftext|>"""distutils.command.bdist_rpm

Implements the Distutils 'bdist_rpm' command (create RPM source and binary
distributions)."""

import os
import subprocess
import sys
from distutils._log import log
from typing import ClassVar

from ..core import Command
from ..debug import DEBUG
from ..errors import (
    DistutilsExecError,
    DistutilsFileError,
    DistutilsOptionError,
    DistutilsPlatformError,
)
from ..file_util import write_file
from ..sysconfig import get_python_version


class bdist_rpm(Command):
    description = "create an RPM distribution"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('bdist-base=', None, "base directory for creating built distributions"),
        (
            'rpm-base=',
            None,
            (
                "base directory for creating RPMs (defaults to \"rpm\" under "
                "--bdist-base; must be specified for RPM 2)"
            ),
        ),
        (
            'dist-dir=',
            'd',
            "directory to put final RPM files in (and .spec files if --spec-only)",
        ),
        (
            'python=',
            None,
            (
                "path to Python interpreter to hard-code in the .spec file "
                "[default: \"python\"]"
            ),
        ),
        (
            'fix-python',
            None,
            (
                "hard-code the exact path to the current Python interpreter in "
                "the .spec file"
            ),
        ),
        ('spec-only', None, "only regenerate spec file"),
        ('source-only', None, "only generate source RPM"),
        ('binary-only', None, "only generate binary RPM"),
        ('use-bzip2', None, "use bzip2 instead of gzip to create source distribution"),
        # More meta-data: too RPM-specific to put in the setup script,
        # but needs to go in the .spec file -- so we make these options
        # to "bdist_rpm".  The idea is that packagers would put this
        # info in setup.cfg, although they are of course free to
        # supply it on the command line.
        (
            'distribution-name=',
            None,
            (
                "name of the (Linux) distribution to which this "
                "RPM applies (*not* the name of the module distribution!)"
            ),
        ),
        ('group=', None, "package classification [default: \"Development/Libraries\"]"),
        ('release=', None, "RPM release number"),
        ('serial=', None, "RPM serial number"),
        (
            'vendor=',
            None,
            (
                "RPM \"vendor\" (eg. \"Joe Blow <joe@example.com>\") "
                "[default: maintainer or author from setup script]"
            ),
        ),
        (
            'packager=',
            None,
            "RPM packager (eg. \"Jane Doe <jane@example.net>\") [default: vendor]",
        ),
        ('doc-files=', None, "list of documentation files (space or comma-separated)"),
        ('changelog=', None, "RPM changelog"),
        ('icon=', None, "name of icon file"),
        ('provides=', None, "capabilities provided by this package"),
        ('requires=', None, "capabilities required by this package"),
        ('conflicts=', None, "capabilities which conflict with this package"),
        ('build-requires=', None, "capabilities required to build this package"),
        ('obsoletes=', None, "capabilities made obsolete by this package"),
        ('no-autoreq', None, "do not automatically calculate dependencies"),
        # Actions to take when building RPM
        ('keep-temp', 'k', "don't clean up RPM build directory"),
        ('no-keep-temp', None, "clean up RPM build directory [default]"),
        (
            'use-rpm-opt-flags',
            None,
            "compile with RPM_OPT_FLAGS when building from source RPM",
        ),
        ('no-rpm-opt-flags', None, "do not pass any RPM CFLAGS to compiler"),
        ('rpm3-mode', None, "RPM 3 compatibility mode (default)"),
        ('rpm2-mode', None, "RPM 2 compatibility mode"),
        # Add the hooks necessary for specifying custom scripts
        ('prep-script=', None, "Specify a script for the PREP phase of RPM building"),
        ('build-script=', None, "Specify a script for the BUILD phase of RPM building"),
        (
            'pre-install=',
            None,
            "Specify a script for the pre-INSTALL phase of RPM building",
        ),
        (
            'install-script=',
            None,
            "Specify a script for the INSTALL phase of RPM building",
        ),
        (
            'post-install=',
            None,
            "Specify a script for the post-INSTALL phase of RPM building",
        ),
        (
            'pre-uninstall=',
            None,
            "Specify a script for the pre-UNINSTALL phase of RPM building",
        ),
        (
            'post-uninstall=',
            None,
            "Specify a script for the post-UNINSTALL phase of RPM building",
        ),
        ('clean-script=', None, "Specify a script for the CLEAN phase of RPM building"),
        (
            'verify-script=',
            None,
            "Specify a script for the VERIFY phase of the RPM build",
        ),
        # Allow a packager to explicitly force an architecture
        ('force-arch=', None, "Force an architecture onto the RPM build process"),
        ('quiet', 'q', "Run the INSTALL phase of RPM building in quiet mode"),
    ]

    boolean_options: ClassVar[list[str]] = [
        'keep-temp',
        'use-rpm-opt-flags',
        'rpm3-mode',
        'no-autoreq',
        'quiet',
    ]

    negative_opt: ClassVar[dict[str, str]] = {
        'no-keep-temp': 'keep-temp',
        'no-rpm-opt-flags': 'use-rpm-opt-flags',
        'rpm2-mode': 'rpm3-mode',
    }

    def initialize_options(self):
        self.bdist_base = None
        self.rpm_base = None
        self.dist_dir = None
        self.python = None
        self.fix_python = None
        self.spec_only = None
        self.binary_only = None
        self.source_only = None
        self.use_bzip2 = None

        self.distribution_name = None
        self.group = None
        self.release = None
        self.serial = None
        self.vendor = None
        self.packager = None
        self.doc_files = None
        self.changelog = None
        self.icon = None

        self.prep_script = None
        self.build_script = None
        self.install_script = None
        self.clean_script = None
        self.verify_script = None
        self.pre_install = None
        self.post_install = None
        self.pre_uninstall = None
        self.post_uninstall = None
        self.prep = None
        self.provides = None
        self.requires = None
        self.conflicts = None
        self.build_requires = None
        self.obsoletes = None

        self.keep_temp = False
        self.use_rpm_opt_flags = True
        self.rpm3_mode = True
        self.no_autoreq = False

        self.force_arch = None
        self.quiet = False

    def finalize_options(self) -> None:
        self.set_undefined_options('bdist', ('bdist_base', 'bdist_base'))
        if self.rpm_base is None:
            if not self.rpm3_mode:
                raise DistutilsOptionError("you must specify --rpm-base in RPM 2 mode")
            self.rpm_base = os.path.join(self.bdist_base, "rpm")

        if self.python is None:
            if self.fix_python:
                self.python = sys.executable
            else:
                self.python = "python3"
        elif self.fix_python:
            raise DistutilsOptionError(
                "--python and --fix-python are mutually exclusive options"
            )

        if os.name != 'posix':
            raise DistutilsPlatformError(
                f"don't know how to create RPM distributions on platform {os.name}"
            )
        if self.binary_only and self.source_only:
            raise DistutilsOptionError(
                "cannot supply both '--source-only' and '--binary-only'"
            )

        # don't pass CFLAGS to pure python distributions
        if not self.distribution.has_ext_modules():
            self.use_rpm_opt_flags = False

        self.set_undefined_options('bdist', ('dist_dir', 'dist_dir'))
        self.finalize_package_data()

    def finalize_package_data(self) -> None:
        self.ensure_string('group', "Development/Libraries")
        self.ensure_string(
            'vendor',
            f"{self.distribution.get_contact()} <{self.distribution.get_contact_email()}>",
        )
        self.ensure_string('packager')
        self.ensure_string_list('doc_files')
        if isinstance(self.doc_files, list):
            for readme in ('README', 'README.txt'):
                if os.path.exists(readme) and readme not in self.doc_files:
                    self.doc_files.append(readme)

        self.ensure_string('release', "1")
        self.ensure_string('serial')  # should it be an int?

        self.ensure_string('distribution_name')

        self.ensure_string('changelog')
        # Format changelog correctly
        self.changelog = self._format_changelog(self.changelog)

        self.ensure_filename('icon')

        self.ensure_filename('prep_script')
        self.ensure_filename('build_script')
        self.ensure_filename('install_script')
        self.ensure_filename('clean_script')
        self.ensure_filename('verify_script')
        self.ensure_filename('pre_install')
        self.ensure_filename('post_install')
        self.ensure_filename('pre_uninstall')
        self.ensure_filename('post_uninstall')

        # XXX don't forget we punted on summaries and descriptions -- they
        # should be handled here eventually!

        # Now *this* is some meta-data that belongs in the setup script...
        self.ensure_string_list('provides')
        self.ensure_string_list('requires')
        self.ensure_string_list('conflicts')
        self.ensure_string_list('build_requires')
        self.ensure_string_list('obsoletes')

        self.ensure_string('force_arch')

    def run(self) -> None:  # noqa: C901
        if DEBUG:
            print("before _get_package_data():")
            print("vendor =", self.vendor)
            print("packager =", self.packager)
            print("doc_files =", self.doc_files)
            print("changelog =", self.changelog)

        # make directories
        if self.spec_only:
            spec_dir = self.dist_dir
            self.mkpath(spec_dir)
        else:
            rpm_dir = {}
            for d in ('SOURCES', 'SPECS', 'BUILD', 'RPMS', 'SRPMS'):
                rpm_dir[d] = os.path.join(self.rpm_base, d)
                self.mkpath(rpm_dir[d])
            spec_dir = rpm_dir['SPECS']

        # Spec file goes into 'dist_dir' if '--spec-only specified',
        # build/rpm.<plat> otherwise.
        spec_path = os.path.join(spec_dir, f"{self.distribution.get_name()}.spec")
        self.execute(
            write_file, (spec_path, self._make_spec_file()), f"writing '{spec_path}'"
        )

        if self.spec_only:  # stop if requested
            return

        # Make a source distribution and copy to SOURCES directory with
        # optional icon.
        saved_dist_files = self.distribution.dist_files[:]
        sdist = self.reinitialize_command('sdist')
        if self.use_bzip2:
            sdist.formats = ['bztar']
        else:
            sdist.formats = ['gztar']
        self.run_command('sdist')
        self.distribution.dist_files = saved_dist_files

        source = sdist.get_archive_files()[0]
        source_dir = rpm_dir['SOURCES']
        self.copy_file(source, source_dir)

        if self.icon:
            if os.path.exists(self.icon):
                self.copy_file(self.icon, source_dir)
            else:
                raise DistutilsFileError(f"icon file '{self.icon}' does not exist")

        # build package
        log.info("building RPMs")
        rpm_cmd = ['rpmbuild']

        if self.source_only:  # what kind of RPMs?
            rpm_cmd.append('-bs')
        elif self.binary_only:
            rpm_cmd.append('-bb')
        else:
            rpm_cmd.append('-ba')
        rpm_cmd.extend(['--define', f'__python {self.python}'])
        if self.rpm3_mode:
            rpm_cmd.extend(['--define', f'_topdir {os.path.abspath(self.rpm_base)}'])
        if not self.keep_temp:
            rpm_cmd.append('--clean')

        if self.quiet:
            rpm_cmd.append('--quiet')

        rpm_cmd.append(spec_path)
        # Determine the binary rpm names that should be built out of this spec
        # file
        # Note that some of these may not be really built (if the file
        # list is empty)
        nvr_string = "%{name}-%{version}-%{release}"
        src_rpm = nvr_string + ".src.rpm"
        non_src_rpm = "%{arch}/" + nvr_string + ".%{arch}.rpm"
        q_cmd = rf"rpm -q --qf '{src_rpm} {non_src_rpm}\n' --specfile '{spec_path}'"

        out = os.popen(q_cmd)
        try:
            binary_rpms = []
            source_rpm = None
            while True:
                line = out.readline()
                if not line:
                    break
                ell = line.strip().split()
                assert len(ell) == 2
                binary_rpms.append(ell[1])
                # The source rpm is named after the first entry in the spec file
                if source_rpm is None:
                    source_rpm = ell[0]

            status = out.close()
            if status:
                raise DistutilsExecError(f"Failed to execute: {q_cmd!r}")

        finally:
            out.close()

        self.spawn(rpm_cmd)

        if self.distribution.has_ext_modules():
            pyversion = get_python_version()
        else:
            pyversion = 'any'

        if not self.binary_only:
            srpm = os.path.join(rpm_dir['SRPMS'], source_rpm)
            assert os.path.exists(srpm)
            self.move_file(srpm, self.dist_dir)
            filename = os.path.join(self.dist_dir, source_rpm)
            self.distribution.dist_files.append(('bdist_rpm', pyversion, filename))

        if not self.source_only:
            for rpm in binary_rpms:
                rpm = os.path.join(rpm_dir['RPMS'], rpm)
                if os.path.exists(rpm):
                    self.move_file(rpm, self.dist_dir)
                    filename = os.path.join(self.dist_dir, os.path.basename(rpm))
                    self.distribution.dist_files.append((
                        'bdist_rpm',
                        pyversion,
                        filename,
                    ))

    def _dist_path(self, path):
        return os.path.join(self.dist_dir, os.path.basename(path))

    def _make_spec_file(self):  # noqa: C901
        """Generate the text of an RPM spec file and return it as a
        list of strings (one per line).
        """
        # definitions and headers
        spec_file = [
            '%define name ' + self.distribution.get_name(),
            '%define version ' + self.distribution.get_version().replace('-', '_'),
            '%define unmangled_version ' + self.distribution.get_version(),
            '%define release ' + self.release.replace('-', '_'),
            '',
            'Summary: ' + (self.distribution.get_description() or "UNKNOWN"),
        ]

        # Workaround for #14443 which affects some RPM based systems such as
        # RHEL6 (and probably derivatives)
        vendor_hook = subprocess.getoutput('rpm --eval %{__os_install_post}')
        # Generate a potential replacement value for __os_install_post (whilst
        # normalizing the whitespace to simplify the test for whether the
        # invocation of brp-python-bytecompile passes in __python):
        vendor_hook = '\n'.join([
            f'  {line.strip()} \\' for line in vendor_hook.splitlines()
        ])
        problem = "brp-python-bytecompile \\\n"
        fixed = "brp-python-bytecompile %{__python} \\\n"
        fixed_hook = vendor_hook.replace(problem, fixed)
        if fixed_hook != vendor_hook:
            spec_file.append('# Workaround for https://bugs.python.org/issue14443')
            spec_file.append('%define __os_install_post ' + fixed_hook + '\n')

        # put locale summaries into spec file
        # XXX not supported for now (hard to put a dictionary
        # in a config file -- arg!)
        # for locale in self.summaries.keys():
        #    spec_file.append('Summary(%s): %s' % (locale,
        #                                          self.summaries[locale]))

        spec_file.extend([
            'Name: %{name}',
            'Version: %{version}',
            'Release: %{release}',
        ])

        # XXX yuck! this filename is available from the "sdist" command,
        # but only after it has run: and we create the spec file before
        # running "sdist", in case of --spec-only.
        if self.use_bzip2:
            spec_file.append('Source0: %{name}-%{unmangled_version}.tar.bz2')
        else:
            spec_file.append('Source0: %{name}-%{unmangled_version}.tar.gz')

        spec_file.extend([
            'License: ' + (self.distribution.get_license() or "UNKNOWN"),
            'Group: ' + self.group,
            'BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot',
            'Prefix: %{_prefix}',
        ])

        if not self.force_arch:
            # noarch if no extension modules
            if not self.distribution.has_ext_modules():
                spec_file.append('BuildArch: noarch')
        else:
            spec_file.append(f'BuildArch: {self.force_arch}')

        for field in (
            'Vendor',
            'Packager',
            'Provides',
            'Requires',
            'Conflicts',
            'Obsoletes',
        ):
            val = getattr(self, field.lower())
            if isinstance(val, list):
                spec_file.append('{}: {}'.format(field, ' '.join(val)))
            elif val is not None:
                spec_file.append(f'{field}: {val}')

        if self.distribution.get_url():
            spec_file.append('Url: ' + self.distribution.get_url())

        if self.distribution_name:
            spec_file.append('Distribution: ' + self.distribution_name)

        if self.build_requires:
            spec_file.append('BuildRequires: ' + ' '.join(self.build_requires))

        if self.icon:
            spec_file.append('Icon: ' + os.path.basename(self.icon))

        if self.no_autoreq:
            spec_file.append('AutoReq: 0')

        spec_file.extend([
            '',
            '%description',
            self.distribution.get_long_description() or "",
        ])

        # put locale descriptions into spec file
        # XXX again, suppressed because config file syntax doesn't
        # easily support this ;-(
        # for locale in self.descriptions.keys():
        #    spec_file.extend([
        #        '',
        #        '%description -l ' + locale,
        #        self.descriptions[locale],
        #        ])

        # rpm scripts
        # figure out default build script
        def_setup_call = f"{self.python} {os.path.basename(sys.argv[0])}"
        def_build = f"{def_setup_call} build"
        if self.use_rpm_opt_flags:
            def_build = 'env CFLAGS="$RPM_OPT_FLAGS" ' + def_build

        # insert contents of files

        # XXX this is kind of misleading: user-supplied options are files
        # that we open and interpolate into the spec file, but the defaults
        # are just text that we drop in as-is.  Hmmm.

        install_cmd = f'{def_setup_call} install -O1 --root=$RPM_BUILD_ROOT --record=INSTALLED_FILES'

        script_options = [
            ('prep', 'prep_script', "%setup -n %{name}-%{unmangled_version}"),
            ('build', 'build_script', def_build),
            ('install', 'install_script', install_cmd),
            ('clean', 'clean_script', "rm -rf $RPM_BUILD_ROOT"),
            ('verifyscript', 'verify_script', None),
            ('pre', 'pre_install', None),
            ('post', 'post_install', None),
            ('preun', 'pre_uninstall', None),
            ('postun', 'post_uninstall', None),
        ]

        for rpm_opt, attr, default in script_options:
            # Insert contents of file referred to, if no file is referred to
            # use 'default' as contents of script
            val = getattr(self, attr)
            if val or default:
                spec_file.extend([
                    '',
                    '%' + rpm_opt,
                ])
                if val:
                    with open(val) as f:
                        spec_file.extend(f.read().split('\n'))
                else:
                    spec_file.append(default)

        # files section
        spec_file.extend([
            '',
            '%files -f INSTALLED_FILES',
            '%defattr(-,root,root)',
        ])

        if self.doc_files:
            spec_file.append('%doc ' + ' '.join(self.doc_files))

        if self.changelog:
            spec_file.extend([
                '',
                '%changelog',
            ])
            spec_file.extend(self.changelog)

        return spec_file

    def _format_changelog(self, changelog):
        """Format the changelog correctly and convert it to a list of strings"""
        if not changelog:
            return changelog
        new_changelog = []
        for line in changelog.strip().split('\n'):
            line = line.strip()
            if line[0] == '*':
                new_changelog.extend(['', line])
            elif line[0] == '-':
                new_changelog.append(line)
            else:
                new_changelog.append('  ' + line)

        # strip trailing newline inserted by first changelog entry
        if not new_changelog[0]:
            del new_changelog[0]

        return new_changelog
<|endoftext|>"""distutils.command.config

Implements the Distutils 'config' command, a (mostly) empty command class
that exists mainly to be sub-classed by specific module distributions and
applications.  The idea is that while every "config" command is different,
at least they're all named the same, and users always see "config" in the
list of standard commands.  Also, this is a good place to put common
configure-like tasks: "try to compile this C code", or "figure out where
this header file lives".
"""

from __future__ import annotations

import os
import pathlib
import re
from collections.abc import Sequence
from distutils._log import log
from typing import ClassVar

from ..ccompiler import CCompiler, CompileError, LinkError, new_compiler
from ..core import Command
from ..errors import DistutilsExecError
from ..sysconfig import customize_compiler

LANG_EXT = {"c": ".c", "c++": ".cxx"}


class config(Command):
    description = "prepare to build"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        ('compiler=', None, "specify the compiler type"),
        ('cc=', None, "specify the compiler executable"),
        ('include-dirs=', 'I', "list of directories to search for header files"),
        ('define=', 'D', "C preprocessor macros to define"),
        ('undef=', 'U', "C preprocessor macros to undefine"),
        ('libraries=', 'l', "external C libraries to link with"),
        ('library-dirs=', 'L', "directories to search for external C libraries"),
        ('noisy', None, "show every action (compile, link, run, ...) taken"),
        (
            'dump-source',
            None,
            "dump generated source files before attempting to compile them",
        ),
    ]

    # The three standard command methods: since the "config" command
    # does nothing by default, these are empty.

    def initialize_options(self):
        self.compiler = None
        self.cc = None
        self.include_dirs = None
        self.libraries = None
        self.library_dirs = None

        # maximal output for now
        self.noisy = 1
        self.dump_source = 1

        # list of temporary files generated along-the-way that we have
        # to clean at some point
        self.temp_files = []

    def finalize_options(self):
        if self.include_dirs is None:
            self.include_dirs = self.distribution.include_dirs or []
        elif isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)

        if self.libraries is None:
            self.libraries = []
        elif isinstance(self.libraries, str):
            self.libraries = [self.libraries]

        if self.library_dirs is None:
            self.library_dirs = []
        elif isinstance(self.library_dirs, str):
            self.library_dirs = self.library_dirs.split(os.pathsep)

    def run(self):
        pass

    # Utility methods for actual "config" commands.  The interfaces are
    # loosely based on Autoconf macros of similar names.  Sub-classes
    # may use these freely.

    def _check_compiler(self):
        """Check that 'self.compiler' really is a CCompiler object;
        if not, make it one.
        """
        if not isinstance(self.compiler, CCompiler):
            self.compiler = new_compiler(compiler=self.compiler, force=True)
            customize_compiler(self.compiler)
            if self.include_dirs:
                self.compiler.set_include_dirs(self.include_dirs)
            if self.libraries:
                self.compiler.set_libraries(self.libraries)
            if self.library_dirs:
                self.compiler.set_library_dirs(self.library_dirs)

    def _gen_temp_sourcefile(self, body, headers, lang):
        filename = "_configtest" + LANG_EXT[lang]
        with open(filename, "w", encoding='utf-8') as file:
            if headers:
                file.writelines(f"#include <{header}>\n" for header in headers)
                file.write("\n")
            file.write(body)
            if body[-1] != "\n":
                file.write("\n")
        return filename

    def _preprocess(self, body, headers, include_dirs, lang):
        src = self._gen_temp_sourcefile(body, headers, lang)
        out = "_configtest.i"
        self.temp_files.extend([src, out])
        self.compiler.preprocess(src, out, include_dirs=include_dirs)
        return (src, out)

    def _compile(self, body, headers, include_dirs, lang):
        src = self._gen_temp_sourcefile(body, headers, lang)
        if self.dump_source:
            dump_file(src, f"compiling '{src}':")
        (obj,) = self.compiler.object_filenames([src])
        self.temp_files.extend([src, obj])
        self.compiler.compile([src], include_dirs=include_dirs)
        return (src, obj)

    def _link(self, body, headers, include_dirs, libraries, library_dirs, lang):
        (src, obj) = self._compile(body, headers, include_dirs, lang)
        prog = os.path.splitext(os.path.basename(src))[0]
        self.compiler.link_executable(
            [obj],
            prog,
            libraries=libraries,
            library_dirs=library_dirs,
            target_lang=lang,
        )

        if self.compiler.exe_extension is not None:
            prog = prog + self.compiler.exe_extension
        self.temp_files.append(prog)

        return (src, obj, prog)

    def _clean(self, *filenames):
        if not filenames:
            filenames = self.temp_files
            self.temp_files = []
        log.info("removing: %s", ' '.join(filenames))
        for filename in filenames:
            try:
                os.remove(filename)
            except OSError:
                pass

    # XXX need access to the header search path and maybe default macros.

    def try_cpp(self, body=None, headers=None, include_dirs=None, lang="c"):
        """Construct a source file from 'body' (a string containing lines
        of C/C++ code) and 'headers' (a list of header files to include)
        and run it through the preprocessor.  Return true if the
        preprocessor succeeded, false if there were any errors.
        ('body' probably isn't of much use, but what the heck.)
        """
        self._check_compiler()
        ok = True
        try:
            self._preprocess(body, headers, include_dirs, lang)
        except CompileError:
            ok = False

        self._clean()
        return ok

    def search_cpp(self, pattern, body=None, headers=None, include_dirs=None, lang="c"):
        """Construct a source file (just like 'try_cpp()'), run it through
        the preprocessor, and return true if any line of the output matches
        'pattern'.  'pattern' should either be a compiled regex object or a
        string containing a regex.  If both 'body' and 'headers' are None,
        preprocesses an empty file -- which can be useful to determine the
        symbols the preprocessor and compiler set by default.
        """
        self._check_compiler()
        _src, out = self._preprocess(body, headers, include_dirs, lang)

        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        with open(out, encoding='utf-8') as file:
            match = any(pattern.search(line) for line in file)

        self._clean()
        return match

    def try_compile(self, body, headers=None, include_dirs=None, lang="c"):
        """Try to compile a source file built from 'body' and 'headers'.
        Return true on success, false otherwise.
        """
        self._check_compiler()
        try:
            self._compile(body, headers, include_dirs, lang)
            ok = True
        except CompileError:
            ok = False

        log.info(ok and "success!" or "failure.")
        self._clean()
        return ok

    def try_link(
        self,
        body,
        headers=None,
        include_dirs=None,
        libraries=None,
        library_dirs=None,
        lang="c",
    ):
        """Try to compile and link a source file, built from 'body' and
        'headers', to executable form.  Return true on success, false
        otherwise.
        """
        self._check_compiler()
        try:
            self._link(body, headers, include_dirs, libraries, library_dirs, lang)
            ok = True
        except (CompileError, LinkError):
            ok = False

        log.info(ok and "success!" or "failure.")
        self._clean()
        return ok

    def try_run(
        self,
        body,
        headers=None,
        include_dirs=None,
        libraries=None,
        library_dirs=None,
        lang="c",
    ):
        """Try to compile, link to an executable, and run a program
        built from 'body' and 'headers'.  Return true on success, false
        otherwise.
        """
        self._check_compiler()
        try:
            _src, _obj, exe = self._link(
                body, headers, include_dirs, libraries, library_dirs, lang
            )
            self.spawn([exe])
            ok = True
        except (CompileError, LinkError, DistutilsExecError):
            ok = False

        log.info(ok and "success!" or "failure.")
        self._clean()
        return ok

    # -- High-level methods --------------------------------------------
    # (these are the ones that are actually likely to be useful
