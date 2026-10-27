                        objects.extend(f.read().splitlines())

        # Wrap unlinkable objects to a linkable one
        if unlinkable_fobjects:
            fobjects = [os.path.abspath(obj) for obj in unlinkable_fobjects]
            wrapped = fcompiler.wrap_unlinkable_objects(
                    fobjects, output_dir=self.build_temp,
                    extra_dll_dir=self.extra_dll_dir)
            objects.extend(wrapped)

        return objects, libraries

    def _libs_with_msvc_and_fortran(self, fcompiler, c_libraries,
                                    c_library_dirs):
        if fcompiler is None:
            return

        for libname in c_libraries:
            if libname.startswith('msvc'):
                continue
            fileexists = False
            for libdir in c_library_dirs or []:
                libfile = os.path.join(libdir, '%s.lib' % (libname))
                if os.path.isfile(libfile):
                    fileexists = True
                    break
            if fileexists:
                continue
            # make g77-compiled static libs available to MSVC
            fileexists = False
            for libdir in c_library_dirs:
                libfile = os.path.join(libdir, 'lib%s.a' % (libname))
                if os.path.isfile(libfile):
                    # copy libname.a file to name.lib so that MSVC linker
                    # can find it
                    libfile2 = os.path.join(self.build_temp, libname + '.lib')
                    copy_file(libfile, libfile2)
                    if self.build_temp not in c_library_dirs:
                        c_library_dirs.append(self.build_temp)
                    fileexists = True
                    break
            if fileexists:
                continue
            log.warn('could not find library %r in directories %s'
                     % (libname, c_library_dirs))

        # Always use system linker when using MSVC compiler.
        f_lib_dirs = []
        for dir in fcompiler.library_dirs:
            # correct path when compiling in Cygwin but with normal Win
            # Python
            if dir.startswith('/usr/lib'):
                try:
                    dir = subprocess.check_output(['cygpath', '-w', dir])
                except (OSError, subprocess.CalledProcessError):
                    pass
                else:
                    dir = filepath_from_subprocess_output(dir)
            f_lib_dirs.append(dir)
        c_library_dirs.extend(f_lib_dirs)

        # make g77-compiled static libs available to MSVC
        for lib in fcompiler.libraries:
            if not lib.startswith('msvc'):
                c_libraries.append(lib)
                p = combine_paths(f_lib_dirs, 'lib' + lib + '.a')
                if p:
                    dst_name = os.path.join(self.build_temp, lib + '.lib')
                    if not os.path.isfile(dst_name):
                        copy_file(p[0], dst_name)
                    if self.build_temp not in c_library_dirs:
                        c_library_dirs.append(self.build_temp)

    def get_source_files(self):
        self.check_extensions_list(self.extensions)
        filenames = []
        for ext in self.extensions:
            filenames.extend(get_ext_source_files(ext))
        return filenames

    def get_outputs(self):
        self.check_extensions_list(self.extensions)

        outputs = []
        for ext in self.extensions:
            if not ext.sources:
                continue
            fullname = self.get_ext_fullname(ext.name)
            outputs.append(os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname)))
        return outputs
<|endoftext|>""" Modified version of build_clib that handles fortran source files.
"""
import os
from glob import glob
import shutil
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError, DistutilsError, \
    DistutilsFileError

from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import (
    filter_sources, get_lib_source_files, get_numpy_include_dirs,
    has_cxx_sources, has_f_sources, is_sequence
)
from numpy.distutils.ccompiler_opt import new_ccompiler_opt

# Fix Python distutils bug sf #1718574:
_l = old_build_clib.user_options
for _i in range(len(_l)):
    if _l[_i][0] in ['build-clib', 'build-temp']:
        _l[_i] = (_l[_i][0] + '=',) + _l[_i][1:]
#


class build_clib(old_build_clib):

    description = "build C/C++/F libraries used by Python extensions"

    user_options = old_build_clib.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ('inplace', 'i', 'Build in-place'),
        ('parallel=', 'j',
         "number of parallel jobs"),
        ('warn-error', None,
         "turn all warnings into errors (-Werror)"),
        ('cpu-baseline=', None,
         "specify a list of enabled baseline CPU optimizations"),
        ('cpu-dispatch=', None,
         "specify a list of dispatched CPU optimizations"),
        ('disable-optimization', None,
         "disable CPU optimized code(dispatch,simd,fast...)"),
    ]

    boolean_options = old_build_clib.boolean_options + \
    ['inplace', 'warn-error', 'disable-optimization']

    def initialize_options(self):
        old_build_clib.initialize_options(self)
        self.fcompiler = None
        self.inplace = 0
        self.parallel = None
        self.warn_error = None
        self.cpu_baseline = None
        self.cpu_dispatch = None
        self.disable_optimization = None


    def finalize_options(self):
        if self.parallel:
            try:
                self.parallel = int(self.parallel)
            except ValueError as e:
                raise ValueError("--parallel/-j argument must be an integer") from e
        old_build_clib.finalize_options(self)
        self.set_undefined_options('build',
                                        ('parallel', 'parallel'),
                                        ('warn_error', 'warn_error'),
                                        ('cpu_baseline', 'cpu_baseline'),
                                        ('cpu_dispatch', 'cpu_dispatch'),
                                        ('disable_optimization', 'disable_optimization')
                                  )

    def have_f_sources(self):
        for (lib_name, build_info) in self.libraries:
            if has_f_sources(build_info.get('sources', [])):
                return True
        return False

    def have_cxx_sources(self):
        for (lib_name, build_info) in self.libraries:
            if has_cxx_sources(build_info.get('sources', [])):
                return True
        return False

    def run(self):
        if not self.libraries:
            return

        # Make sure that library sources are complete.
        languages = []

        # Make sure that extension sources are complete.
        self.run_command('build_src')

        for (lib_name, build_info) in self.libraries:
            l = build_info.get('language', None)
            if l and l not in languages:
                languages.append(l)

        from distutils.ccompiler import new_compiler
        self.compiler = new_compiler(compiler=self.compiler,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution,
                                need_cxx=self.have_cxx_sources())

        if self.warn_error:
            self.compiler.compiler.append('-Werror')
            self.compiler.compiler_so.append('-Werror')

        libraries = self.libraries
        self.libraries = None
        self.compiler.customize_cmd(self)
        self.libraries = libraries

        self.compiler.show_customization()

        if not self.disable_optimization:
            dispatch_hpath = os.path.join("numpy", "distutils", "include", "npy_cpu_dispatch_config.h")
            dispatch_hpath = os.path.join(self.get_finalized_command("build_src").build_src, dispatch_hpath)
            opt_cache_path = os.path.abspath(
                os.path.join(self.build_temp, 'ccompiler_opt_cache_clib.py')
            )
            if hasattr(self, "compiler_opt"):
                # By default `CCompilerOpt` update the cache at the exit of
                # the process, which may lead to duplicate building
                # (see build_extension()/force_rebuild) if run() called
                # multiple times within the same os process/thread without
                # giving the chance the previous instances of `CCompilerOpt`
                # to update the cache.
                self.compiler_opt.cache_flush()

            self.compiler_opt = new_ccompiler_opt(
                compiler=self.compiler, dispatch_hpath=dispatch_hpath,
                cpu_baseline=self.cpu_baseline, cpu_dispatch=self.cpu_dispatch,
                cache_path=opt_cache_path
            )
            def report(copt):
                log.info("\n########### CLIB COMPILER OPTIMIZATION ###########")
                log.info(copt.report(full=True))

            import atexit
            atexit.register(report, self.compiler_opt)

        if self.have_f_sources():
            from numpy.distutils.fcompiler import new_fcompiler
            self._f_compiler = new_fcompiler(compiler=self.fcompiler,
                                             verbose=self.verbose,
                                             dry_run=self.dry_run,
                                             force=self.force,
                                             requiref90='f90' in languages,
                                             c_compiler=self.compiler)
            if self._f_compiler is not None:
                self._f_compiler.customize(self.distribution)

                libraries = self.libraries
                self.libraries = None
                self._f_compiler.customize_cmd(self)
                self.libraries = libraries

                self._f_compiler.show_customization()
        else:
            self._f_compiler = None

        self.build_libraries(self.libraries)

        if self.inplace:
            for l in self.distribution.installed_libraries:
                libname = self.compiler.library_filename(l.name)
                source = os.path.join(self.build_clib, libname)
                target = os.path.join(l.target_dir, libname)
                self.mkpath(l.target_dir)
                shutil.copy(source, target)

    def get_source_files(self):
        self.check_library_list(self.libraries)
        filenames = []
        for lib in self.libraries:
            filenames.extend(get_lib_source_files(lib))
        return filenames

    def build_libraries(self, libraries):
        for (lib_name, build_info) in libraries:
            self.build_a_library(build_info, lib_name, libraries)

    def assemble_flags(self, in_flags):
        """ Assemble flags from flag list

        Parameters
        ----------
        in_flags : None or sequence
            None corresponds to empty list.  Sequence elements can be strings
            or callables that return lists of strings. Callable takes `self` as
            single parameter.

        Returns
        -------
        out_flags : list
        """
        if in_flags is None:
            return []
        out_flags = []
        for in_flag in in_flags:
            if callable(in_flag):
                out_flags += in_flag(self)
            else:
                out_flags.append(in_flag)
        return out_flags

    def build_a_library(self, build_info, lib_name, libraries):
        # default compilers
        compiler = self.compiler
        fcompiler = self._f_compiler

        sources = build_info.get('sources')
        if sources is None or not is_sequence(sources):
            raise DistutilsSetupError(("in 'libraries' option (library '%s'), " +
                                       "'sources' must be present and must be " +
                                       "a list of source filenames") % lib_name)
        sources = list(sources)

        c_sources, cxx_sources, f_sources, fmodule_sources \
            = filter_sources(sources)
        requiref90 = not not fmodule_sources or \
            build_info.get('language', 'c') == 'f90'

        # save source type information so that build_ext can use it.
        source_languages = []
        if c_sources:
            source_languages.append('c')
        if cxx_sources:
            source_languages.append('c++')
        if requiref90:
            source_languages.append('f90')
        elif f_sources:
            source_languages.append('f77')
        build_info['source_languages'] = source_languages

        lib_file = compiler.library_filename(lib_name,
                                             output_dir=self.build_clib)
        depends = sources + build_info.get('depends', [])

        force_rebuild = self.force
        if not self.disable_optimization and not self.compiler_opt.is_cached():
            log.debug("Detected changes on compiler optimizations")
            force_rebuild = True
        if not (force_rebuild or newer_group(depends, lib_file, 'newer')):
            log.debug("skipping '%s' library (up-to-date)", lib_name)
            return
        else:
            log.info("building '%s' library", lib_name)

        config_fc = build_info.get('config_fc', {})
        if fcompiler is not None and config_fc:
            log.info('using additional config_fc from setup script '
                     'for fortran compiler: %s'
                     % (config_fc,))
            from numpy.distutils.fcompiler import new_fcompiler
            fcompiler = new_fcompiler(compiler=fcompiler.compiler_type,
                                      verbose=self.verbose,
                                      dry_run=self.dry_run,
                                      force=self.force,
                                      requiref90=requiref90,
                                      c_compiler=self.compiler)
            if fcompiler is not None:
                dist = self.distribution
                base_config_fc = dist.get_option_dict('config_fc').copy()
                base_config_fc.update(config_fc)
                fcompiler.customize(base_config_fc)

        # check availability of Fortran compilers
        if (f_sources or fmodule_sources) and fcompiler is None:
            raise DistutilsError("library %s has Fortran sources"
                                 " but no Fortran compiler found" % (lib_name))

        if fcompiler is not None:
            fcompiler.extra_f77_compile_args = build_info.get(
                'extra_f77_compile_args') or []
            fcompiler.extra_f90_compile_args = build_info.get(
                'extra_f90_compile_args') or []

        macros = build_info.get('macros')
        if macros is None:
            macros = []
        include_dirs = build_info.get('include_dirs')
        if include_dirs is None:
            include_dirs = []
        # Flags can be strings, or callables that return a list of strings.
        extra_postargs = self.assemble_flags(
            build_info.get('extra_compiler_args'))
        extra_cflags = self.assemble_flags(
            build_info.get('extra_cflags'))
        extra_cxxflags = self.assemble_flags(
            build_info.get('extra_cxxflags'))

        include_dirs.extend(get_numpy_include_dirs())
        # where compiled F90 module files are:
        module_dirs = build_info.get('module_dirs') or []
        module_build_dir = os.path.dirname(lib_file)
        if requiref90:
            self.mkpath(module_build_dir)

        if compiler.compiler_type == 'msvc':
            # this hack works around the msvc compiler attributes
            # problem, msvc uses its own convention :(
            c_sources += cxx_sources
            cxx_sources = []
            extra_cflags += extra_cxxflags

        # filtering C dispatch-table sources when optimization is not disabled,
        # otherwise treated as normal sources.
        copt_c_sources = []
        copt_cxx_sources = []
        copt_baseline_flags = []
        copt_macros = []
        if not self.disable_optimization:
            bsrc_dir = self.get_finalized_command("build_src").build_src
            dispatch_hpath = os.path.join("numpy", "distutils", "include")
            dispatch_hpath = os.path.join(bsrc_dir, dispatch_hpath)
            include_dirs.append(dispatch_hpath)
            # copt_build_src = None if self.inplace else bsrc_dir
            copt_build_src = bsrc_dir
            for _srcs, _dst, _ext in (
                ((c_sources,), copt_c_sources, ('.dispatch.c',)),
                ((c_sources, cxx_sources), copt_cxx_sources,
                    ('.dispatch.cpp', '.dispatch.cxx'))
            ):
                for _src in _srcs:
                    _dst += [
                        _src.pop(_src.index(s))
                        for s in _src[:] if s.endswith(_ext)
                    ]
            copt_baseline_flags = self.compiler_opt.cpu_baseline_flags()
        else:
            copt_macros.append(("NPY_DISABLE_OPTIMIZATION", 1))

        objects = []
        if copt_cxx_sources:
            log.info("compiling C++ dispatch-able sources")
            objects += self.compiler_opt.try_dispatch(
                copt_c_sources,
                output_dir=self.build_temp,
                src_dir=copt_build_src,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=extra_postargs + extra_cxxflags,
                ccompiler=cxx_compiler
            )

        if copt_c_sources:
            log.info("compiling C dispatch-able sources")
            objects += self.compiler_opt.try_dispatch(
                copt_c_sources,
                output_dir=self.build_temp,
                src_dir=copt_build_src,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=extra_postargs + extra_cflags)

        if c_sources:
            log.info("compiling C sources")
            objects += compiler.compile(
                c_sources,
                output_dir=self.build_temp,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=(extra_postargs +
                                copt_baseline_flags +
                                extra_cflags))

        if cxx_sources:
            log.info("compiling C++ sources")
            cxx_compiler = compiler.cxx_compiler()
            cxx_objects = cxx_compiler.compile(
                cxx_sources,
                output_dir=self.build_temp,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=(extra_postargs +
                                copt_baseline_flags +
                                extra_cxxflags))
            objects.extend(cxx_objects)

        if f_sources or fmodule_sources:
            extra_postargs = []
            f_objects = []

            if requiref90:
                if fcompiler.module_dir_switch is None:
                    existing_modules = glob('*.mod')
                extra_postargs += fcompiler.module_options(
                    module_dirs, module_build_dir)

            if fmodule_sources:
                log.info("compiling Fortran 90 module sources")
                f_objects += fcompiler.compile(fmodule_sources,
                                               output_dir=self.build_temp,
                                               macros=macros,
                                               include_dirs=include_dirs,
                                               debug=self.debug,
                                               extra_postargs=extra_postargs)

            if requiref90 and self._f_compiler.module_dir_switch is None:
                # move new compiled F90 module files to module_build_dir
                for f in glob('*.mod'):
                    if f in existing_modules:
                        continue
                    t = os.path.join(module_build_dir, f)
                    if os.path.abspath(f) == os.path.abspath(t):
                        continue
                    if os.path.isfile(t):
                        os.remove(t)
                    try:
                        self.move_file(f, module_build_dir)
                    except DistutilsFileError:
                        log.warn('failed to move %r to %r'
                                 % (f, module_build_dir))

            if f_sources:
                log.info("compiling Fortran sources")
                f_objects += fcompiler.compile(f_sources,
                                               output_dir=self.build_temp,
                                               macros=macros,
                                               include_dirs=include_dirs,
                                               debug=self.debug,
                                               extra_postargs=extra_postargs)
        else:
            f_objects = []

        if f_objects and not fcompiler.can_ccompiler_link(compiler):
            # Default linker cannot link Fortran object files, and results
            # need to be wrapped later. Instead of creating a real static
            # library, just keep track of the object files.
            listfn = os.path.join(self.build_clib,
                                  lib_name + '.fobjects')
            with open(listfn, 'w') as f:
                f.write("\n".join(os.path.abspath(obj) for obj in f_objects))

            listfn = os.path.join(self.build_clib,
                                  lib_name + '.cobjects')
            with open(listfn, 'w') as f:
                f.write("\n".join(os.path.abspath(obj) for obj in objects))

            # create empty "library" file for dependency tracking
            lib_fname = os.path.join(self.build_clib,
                                     lib_name + compiler.static_lib_extension)
            with open(lib_fname, 'wb') as f:
                pass
        else:
            # assume that default linker is suitable for
            # linking Fortran object files
            objects.extend(f_objects)
            compiler.create_static_lib(objects, lib_name,
                                       output_dir=self.build_clib,
                                       debug=self.debug)

        # fix library dependencies
        clib_libraries = build_info.get('libraries', [])
        for lname, binfo in libraries:
            if lname in clib_libraries:
                clib_libraries.extend(binfo.get('libraries', []))
        if clib_libraries:
            build_info['libraries'] = clib_libraries
<|endoftext|>import sys
have_setuptools = ('setuptools' in sys.modules)

from distutils.command.install_data import install_data as old_install_data

#data installer with improved intelligence over distutils
#data files are copied into the project directory instead
#of willy-nilly
class install_data (old_install_data):

    def run(self):
        old_install_data.run(self)

        if have_setuptools:
            # Run install_clib again, since setuptools does not run sub-commands
            # of install automatically
            self.run_command('install_clib')

    def finalize_options (self):
        self.set_undefined_options('install',
                                   ('install_lib', 'install_dir'),
                                   ('root', 'root'),
                                   ('force', 'force'),
                                  )
<|endoftext|>import os
from distutils.command.install_headers import install_headers as old_install_headers

class install_headers (old_install_headers):

    def run (self):
        headers = self.distribution.headers
        if not headers:
            return

        prefix = os.path.dirname(self.install_dir)
        for header in headers:
            if isinstance(header, tuple):
                # Kind of a hack, but I don't know where else to change this...
                if header[0] == 'numpy._core':
                    header = ('numpy', header[1])
                    if os.path.splitext(header[1])[1] == '.inc':
                        continue
                d = os.path.join(*([prefix]+header[0].split('.')))
                header = header[1]
            else:
                d = self.install_dir
            self.mkpath(d)
            (out, _) = self.copy_file(header, d)
            self.outfiles.append(out)
<|endoftext|>from distutils.command.build_py import build_py as old_build_py
from numpy.distutils.misc_util import is_string

class build_py(old_build_py):

    def run(self):
        build_src = self.get_finalized_command('build_src')
        if build_src.py_modules_dict and self.packages is None:
            self.packages = list(build_src.py_modules_dict.keys ())
        old_build_py.run(self)

    def find_package_modules(self, package, package_dir):
        modules = old_build_py.find_package_modules(self, package, package_dir)

        # Find build_src generated *.py files.
        build_src = self.get_finalized_command('build_src')
        modules += build_src.py_modules_dict.get(package, [])

        return modules

    def find_modules(self):
        old_py_modules = self.py_modules[:]
        new_py_modules = [_m for _m in self.py_modules if is_string(_m)]
        self.py_modules[:] = new_py_modules
        modules = old_build_py.find_modules(self)
        self.py_modules[:] = old_py_modules

        return modules

    # XXX: Fix find_source_files for item in py_modules such that item is 3-tuple
    # and item[2] is source file.
<|endoftext|>""" Build swig and f2py sources.
"""
import os
import re
import sys
import shlex
import copy

from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError


# this import can't be done here, as it uses numpy stuff only available
# after it's installed
#import numpy.f2py
from numpy.distutils import log
from numpy.distutils.misc_util import (
    fortran_ext_match, appendpath, is_string, is_sequence, get_cmd
    )
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file

def subst_vars(target, source, d):
    """Substitute any occurrence of @foo@ by d['foo'] from source file into
    target."""
    var = re.compile('@([a-zA-Z_]+)@')
    with open(source, 'r') as fs:
        with open(target, 'w') as ft:
            for l in fs:
                m = var.search(l)
                if m:
                    ft.write(l.replace('@%s@' % m.group(1), d[m.group(1)]))
                else:
                    ft.write(l)

class build_src(build_ext.build_ext):

    description = "build sources from SWIG, F2PY files or a function"

    user_options = [
        ('build-src=', 'd', "directory to \"build\" sources to"),
        ('f2py-opts=', None, "list of f2py command line options"),
        ('swig=', None, "path to the SWIG executable"),
        ('swig-opts=', None, "list of SWIG command line options"),
        ('swig-cpp', None, "make SWIG create C++ files (default is autodetected from sources)"),
        ('f2pyflags=', None, "additional flags to f2py (use --f2py-opts= instead)"), # obsolete
        ('swigflags=', None, "additional flags to swig (use --swig-opts= instead)"), # obsolete
        ('force', 'f', "forcibly build everything (ignore file timestamps)"),
        ('inplace', 'i',
         "ignore build-lib and put compiled extensions into the source " +
         "directory alongside your pure Python modules"),
        ('verbose-cfg', None,
         "change logging level from WARN to INFO which will show all " +
         "compiler output")
        ]

    boolean_options = ['force', 'inplace', 'verbose-cfg']

    help_options = []

    def initialize_options(self):
        self.extensions = None
        self.package = None
        self.py_modules = None
        self.py_modules_dict = None
        self.build_src = None
        self.build_lib = None
        self.build_base = None
        self.force = None
        self.inplace = None
        self.package_dir = None
        self.f2pyflags = None # obsolete
        self.f2py_opts = None
        self.swigflags = None # obsolete
        self.swig_opts = None
        self.swig_cpp = None
        self.swig = None
        self.verbose_cfg = None

    def finalize_options(self):
        self.set_undefined_options('build',
                                   ('build_base', 'build_base'),
                                   ('build_lib', 'build_lib'),
                                   ('force', 'force'))
        if self.package is None:
            self.package = self.distribution.ext_package
        self.extensions = self.distribution.ext_modules
        self.libraries = self.distribution.libraries or []
        self.py_modules = self.distribution.py_modules or []
        self.data_files = self.distribution.data_files or []

        if self.build_src is None:
            plat_specifier = ".{}-{}.{}".format(get_platform(), *sys.version_info[:2])
            self.build_src = os.path.join(self.build_base, 'src'+plat_specifier)

        # py_modules_dict is used in build_py.find_package_modules
        self.py_modules_dict = {}

        if self.f2pyflags:
            if self.f2py_opts:
                log.warn('ignoring --f2pyflags as --f2py-opts already used')
            else:
                self.f2py_opts = self.f2pyflags
            self.f2pyflags = None
        if self.f2py_opts is None:
            self.f2py_opts = []
        else:
            self.f2py_opts = shlex.split(self.f2py_opts)

        if self.swigflags:
            if self.swig_opts:
                log.warn('ignoring --swigflags as --swig-opts already used')
            else:
                self.swig_opts = self.swigflags
            self.swigflags = None

        if self.swig_opts is None:
            self.swig_opts = []
        else:
            self.swig_opts = shlex.split(self.swig_opts)

        # use options from build_ext command
        build_ext = self.get_finalized_command('build_ext')
        if self.inplace is None:
            self.inplace = build_ext.inplace
        if self.swig_cpp is None:
            self.swig_cpp = build_ext.swig_cpp
        for c in ['swig', 'swig_opt']:
            o = '--'+c.replace('_', '-')
            v = getattr(build_ext, c, None)
            if v:
                if getattr(self, c):
                    log.warn('both build_src and build_ext define %s option' % (o))
                else:
                    log.info('using "%s=%s" option from build_ext command' % (o, v))
                    setattr(self, c, v)

    def run(self):
        log.info("build_src")
        if not (self.extensions or self.libraries):
            return
        self.build_sources()

    def build_sources(self):

        if self.inplace:
            self.get_package_dir = \
                     self.get_finalized_command('build_py').get_package_dir

        self.build_py_modules_sources()

        for libname_info in self.libraries:
            self.build_library_sources(*libname_info)

        if self.extensions:
            self.check_extensions_list(self.extensions)

            for ext in self.extensions:
                self.build_extension_sources(ext)

        self.build_data_files_sources()
        self.build_npy_pkg_config()

    def build_data_files_sources(self):
        if not self.data_files:
            return
        log.info('building data_files sources')
        from numpy.distutils.misc_util import get_data_files
        new_data_files = []
        for data in self.data_files:
            if isinstance(data, str):
                new_data_files.append(data)
            elif isinstance(data, tuple):
                d, files = data
                if self.inplace:
                    build_dir = self.get_package_dir('.'.join(d.split(os.sep)))
                else:
                    build_dir = os.path.join(self.build_src, d)
                funcs = [f for f in files if hasattr(f, '__call__')]
                files = [f for f in files if not hasattr(f, '__call__')]
                for f in funcs:
                    if f.__code__.co_argcount==1:
                        s = f(build_dir)
                    else:
                        s = f()
                    if s is not None:
                        if isinstance(s, list):
                            files.extend(s)
                        elif isinstance(s, str):
                            files.append(s)
                        else:
                            raise TypeError(repr(s))
                filenames = get_data_files((d, files))
                new_data_files.append((d, filenames))
            else:
                raise TypeError(repr(data))
        self.data_files[:] = new_data_files


    def _build_npy_pkg_config(self, info, gd):
        template, install_dir, subst_dict = info
        template_dir = os.path.dirname(template)
        for k, v in gd.items():
            subst_dict[k] = v

        if self.inplace == 1:
            generated_dir = os.path.join(template_dir, install_dir)
        else:
            generated_dir = os.path.join(self.build_src, template_dir,
                    install_dir)
        generated = os.path.basename(os.path.splitext(template)[0])
        generated_path = os.path.join(generated_dir, generated)
        if not os.path.exists(generated_dir):
            os.makedirs(generated_dir)

        subst_vars(generated_path, template, subst_dict)

        # Where to install relatively to install prefix
        full_install_dir = os.path.join(template_dir, install_dir)
        return full_install_dir, generated_path

    def build_npy_pkg_config(self):
        log.info('build_src: building npy-pkg config files')

        # XXX: another ugly workaround to circumvent distutils brain damage. We
        # need the install prefix here, but finalizing the options of the
        # install command when only building sources cause error. Instead, we
        # copy the install command instance, and finalize the copy so that it
        # does not disrupt how distutils want to do things when with the
        # original install command instance.
        install_cmd = copy.copy(get_cmd('install'))
        if not install_cmd.finalized == 1:
            install_cmd.finalize_options()
        build_npkg = False
        if self.inplace == 1:
            top_prefix = '.'
            build_npkg = True
        elif hasattr(install_cmd, 'install_libbase'):
            top_prefix = install_cmd.install_libbase
            build_npkg = True

        if build_npkg:
            for pkg, infos in self.distribution.installed_pkg_config.items():
                pkg_path = self.distribution.package_dir[pkg]
                prefix = os.path.join(os.path.abspath(top_prefix), pkg_path)
                d = {'prefix': prefix}
                for info in infos:
                    install_dir, generated = self._build_npy_pkg_config(info, d)
                    self.distribution.data_files.append((install_dir,
                        [generated]))

    def build_py_modules_sources(self):
        if not self.py_modules:
            return
        log.info('building py_modules sources')
        new_py_modules = []
        for source in self.py_modules:
            if is_sequence(source) and len(source)==3:
                package, module_base, source = source
                if self.inplace:
                    build_dir = self.get_package_dir(package)
                else:
                    build_dir = os.path.join(self.build_src,
                                             os.path.join(*package.split('.')))
                if hasattr(source, '__call__'):
                    target = os.path.join(build_dir, module_base + '.py')
                    source = source(target)
                if source is None:
                    continue
                modules = [(package, module_base, source)]
                if package not in self.py_modules_dict:
                    self.py_modules_dict[package] = []
                self.py_modules_dict[package] += modules
            else:
                new_py_modules.append(source)
        self.py_modules[:] = new_py_modules

    def build_library_sources(self, lib_name, build_info):
        sources = list(build_info.get('sources', []))

        if not sources:
            return

        log.info('building library "%s" sources' % (lib_name))

        sources = self.generate_sources(sources, (lib_name, build_info))

        sources = self.template_sources(sources, (lib_name, build_info))

        sources, h_files = self.filter_h_files(sources)

        if h_files:
            log.info('%s - nothing done with h_files = %s',
                     self.package, h_files)

        #for f in h_files:
        #    self.distribution.headers.append((lib_name,f))

        build_info['sources'] = sources
        return

    def build_extension_sources(self, ext):

        sources = list(ext.sources)

        log.info('building extension "%s" sources' % (ext.name))

        fullname = self.get_ext_fullname(ext.name)

        modpath = fullname.split('.')
        package = '.'.join(modpath[0:-1])

        if self.inplace:
            self.ext_target_dir = self.get_package_dir(package)

        sources = self.generate_sources(sources, ext)
        sources = self.template_sources(sources, ext)
        sources = self.swig_sources(sources, ext)
        sources = self.f2py_sources(sources, ext)
        sources = self.pyrex_sources(sources, ext)

        sources, py_files = self.filter_py_files(sources)

        if package not in self.py_modules_dict:
            self.py_modules_dict[package] = []
        modules = []
        for f in py_files:
            module = os.path.splitext(os.path.basename(f))[0]
            modules.append((package, module, f))
        self.py_modules_dict[package] += modules

        sources, h_files = self.filter_h_files(sources)

        if h_files:
            log.info('%s - nothing done with h_files = %s',
                     package, h_files)
        #for f in h_files:
        #    self.distribution.headers.append((package,f))

        ext.sources = sources

    def generate_sources(self, sources, extension):
        new_sources = []
        func_sources = []
        for source in sources:
            if is_string(source):
                new_sources.append(source)
            else:
                func_sources.append(source)
        if not func_sources:
            return new_sources
        if self.inplace and not is_sequence(extension):
            build_dir = self.ext_target_dir
        else:
            if is_sequence(extension):
                name = extension[0]
            #    if 'include_dirs' not in extension[1]:
            #        extension[1]['include_dirs'] = []
            #    incl_dirs = extension[1]['include_dirs']
            else:
                name = extension.name
            #    incl_dirs = extension.include_dirs
            #if self.build_src not in incl_dirs:
            #    incl_dirs.append(self.build_src)
            build_dir = os.path.join(*([self.build_src]
                                       +name.split('.')[:-1]))
        self.mkpath(build_dir)

        if self.verbose_cfg:
            new_level = log.INFO
        else:
            new_level = log.WARN
        old_level = log.set_threshold(new_level)

        for func in func_sources:
            source = func(extension, build_dir)
            if not source:
                continue
            if is_sequence(source):
                [log.info("  adding '%s' to sources." % (s,)) for s in source]
                new_sources.extend(source)
            else:
                log.info("  adding '%s' to sources." % (source,))
                new_sources.append(source)
        log.set_threshold(old_level)
        return new_sources

    def filter_py_files(self, sources):
        return self.filter_files(sources, ['.py'])

    def filter_h_files(self, sources):
        return self.filter_files(sources, ['.h', '.hpp', '.inc'])

    def filter_files(self, sources, exts = []):
        new_sources = []
        files = []
        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext in exts:
                files.append(source)
            else:
                new_sources.append(source)
        return new_sources, files

    def template_sources(self, sources, extension):
        new_sources = []
        if is_sequence(extension):
            depends = extension[1].get('depends')
            include_dirs = extension[1].get('include_dirs')
        else:
            depends = extension.depends
            include_dirs = extension.include_dirs
        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == '.src':  # Template file
                if self.inplace:
                    target_dir = os.path.dirname(base)
                else:
                    target_dir = appendpath(self.build_src, os.path.dirname(base))
                self.mkpath(target_dir)
                target_file = os.path.join(target_dir, os.path.basename(base))
                if (self.force or newer_group([source] + depends, target_file)):
                    if _f_pyf_ext_match(base):
                        log.info("from_template:> %s" % (target_file))
                        outstr = process_f_file(source)
                    else:
                        log.info("conv_template:> %s" % (target_file))
                        outstr = process_c_file(source)
                    with open(target_file, 'w') as fid:
                        fid.write(outstr)
                if _header_ext_match(target_file):
                    d = os.path.dirname(target_file)
                    if d not in include_dirs:
                        log.info("  adding '%s' to include_dirs." % (d))
                        include_dirs.append(d)
                new_sources.append(target_file)
            else:
                new_sources.append(source)
        return new_sources

    def pyrex_sources(self, sources, extension):
        """Pyrex not supported; this remains for Cython support (see below)"""
        new_sources = []
        ext_name = extension.name.split('.')[-1]
        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == '.pyx':
                target_file = self.generate_a_pyrex_source(base, ext_name,
                                                           source,
                                                           extension)
                new_sources.append(target_file)
            else:
                new_sources.append(source)
        return new_sources

    def generate_a_pyrex_source(self, base, ext_name, source, extension):
        """Pyrex is not supported, but some projects monkeypatch this method.

        That allows compiling Cython code, see gh-6955.
        This method will remain here for compatibility reasons.
        """
        return []

    def f2py_sources(self, sources, extension):
        new_sources = []
        f2py_sources = []
        f_sources = []
        f2py_targets = {}
        target_dirs = []
        ext_name = extension.name.split('.')[-1]
        skip_f2py = 0

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == '.pyf': # F2PY interface file
                if self.inplace:
                    target_dir = os.path.dirname(base)
                else:
                    target_dir = appendpath(self.build_src, os.path.dirname(base))
                if os.path.isfile(source):
                    name = get_f2py_modulename(source)
                    if name != ext_name:
                        raise DistutilsSetupError('mismatch of extension names: %s '
                                                  'provides %r but expected %r' % (
                            source, name, ext_name))
                    target_file = os.path.join(target_dir, name+'module.c')
                else:
                    log.debug('  source %s does not exist: skipping f2py\'ing.' \
                              % (source))
                    name = ext_name
                    skip_f2py = 1
                    target_file = os.path.join(target_dir, name+'module.c')
                    if not os.path.isfile(target_file):
                        log.warn('  target %s does not exist:\n   '\
                                 'Assuming %smodule.c was generated with '\
                                 '"build_src --inplace" command.' \
                                 % (target_file, name))
                        target_dir = os.path.dirname(base)
                        target_file = os.path.join(target_dir, name+'module.c')
                        if not os.path.isfile(target_file):
                            raise DistutilsSetupError("%r missing" % (target_file,))
                        log.info('   Yes! Using %r as up-to-date target.' \
                                 % (target_file))
                target_dirs.append(target_dir)
                f2py_sources.append(source)
                f2py_targets[source] = target_file
                new_sources.append(target_file)
            elif fortran_ext_match(ext):
                f_sources.append(source)
            else:
                new_sources.append(source)

        if not (f2py_sources or f_sources):
            return new_sources

        for d in target_dirs:
            self.mkpath(d)

        f2py_options = extension.f2py_options + self.f2py_opts

        if self.distribution.libraries:
            for name, build_info in self.distribution.libraries:
                if name in extension.libraries:
                    f2py_options.extend(build_info.get('f2py_options', []))

        log.info("f2py options: %s" % (f2py_options))

        if f2py_sources:
            if len(f2py_sources) != 1:
                raise DistutilsSetupError(
                    'only one .pyf file is allowed per extension module but got'\
                    ' more: %r' % (f2py_sources,))
            source = f2py_sources[0]
            target_file = f2py_targets[source]
            target_dir = os.path.dirname(target_file) or '.'
            depends = [source] + extension.depends
            if (self.force or newer_group(depends, target_file, 'newer')) \
                   and not skip_f2py:
                log.info("f2py: %s" % (source))
                from numpy.f2py import f2py2e
                f2py2e.run_main(f2py_options
                                    + ['--build-dir', target_dir, source])
            else:
                log.debug("  skipping '%s' f2py interface (up-to-date)" % (source))
        else:
            #XXX TODO: --inplace support for sdist command
            if is_sequence(extension):
                name = extension[0]
            else: name = extension.name
            target_dir = os.path.join(*([self.build_src]
                                        +name.split('.')[:-1]))
            target_file = os.path.join(target_dir, ext_name + 'module.c')
            new_sources.append(target_file)
            depends = f_sources + extension.depends
            if (self.force or newer_group(depends, target_file, 'newer')) \
                   and not skip_f2py:
                log.info("f2py:> %s" % (target_file))
                self.mkpath(target_dir)
                from numpy.f2py import f2py2e
                f2py2e.run_main(f2py_options + ['--lower',
                                                '--build-dir', target_dir]+\
                                ['-m', ext_name]+f_sources)
            else:
                log.debug("  skipping f2py fortran files for '%s' (up-to-date)"\
                          % (target_file))

        if not os.path.isfile(target_file):
            raise DistutilsError("f2py target file %r not generated" % (target_file,))

        build_dir = os.path.join(self.build_src, target_dir)
        target_c = os.path.join(build_dir, 'fortranobject.c')
        target_h = os.path.join(build_dir, 'fortranobject.h')
        log.info("  adding '%s' to sources." % (target_c))
        new_sources.append(target_c)
        if build_dir not in extension.include_dirs:
            log.info("  adding '%s' to include_dirs." % (build_dir))
            extension.include_dirs.append(build_dir)

        if not skip_f2py:
            import numpy.f2py
            d = os.path.dirname(numpy.f2py.__file__)
            source_c = os.path.join(d, 'src', 'fortranobject.c')
            source_h = os.path.join(d, 'src', 'fortranobject.h')
            if newer(source_c, target_c) or newer(source_h, target_h):
                self.mkpath(os.path.dirname(target_c))
                self.copy_file(source_c, target_c)
                self.copy_file(source_h, target_h)
        else:
            if not os.path.isfile(target_c):
                raise DistutilsSetupError("f2py target_c file %r not found" % (target_c,))
            if not os.path.isfile(target_h):
                raise DistutilsSetupError("f2py target_h file %r not found" % (target_h,))

        for name_ext in ['-f2pywrappers.f', '-f2pywrappers2.f90']:
            filename = os.path.join(target_dir, ext_name + name_ext)
            if os.path.isfile(filename):
                log.info("  adding '%s' to sources." % (filename))
                f_sources.append(filename)

        return new_sources + f_sources

    def swig_sources(self, sources, extension):
        # Assuming SWIG 1.3.14 or later. See compatibility note in
        #   http://www.swig.org/Doc1.3/Python.html#Python_nn6

        new_sources = []
        swig_sources = []
        swig_targets = {}
        target_dirs = []
        py_files = []     # swig generated .py files
        target_ext = '.c'
        if '-c++' in extension.swig_opts:
            typ = 'c++'
            is_cpp = True
            extension.swig_opts.remove('-c++')
        elif self.swig_cpp:
            typ = 'c++'
            is_cpp = True
        else:
            typ = None
            is_cpp = False
        skip_swig = 0
        ext_name = extension.name.split('.')[-1]

        for source in sources:
            (base, ext) = os.path.splitext(source)
            if ext == '.i': # SWIG interface file
                # the code below assumes that the sources list
                # contains not more than one .i SWIG interface file
                if self.inplace:
                    target_dir = os.path.dirname(base)
                    py_target_dir = self.ext_target_dir
                else:
                    target_dir = appendpath(self.build_src, os.path.dirname(base))
                    py_target_dir = target_dir
                if os.path.isfile(source):
                    name = get_swig_modulename(source)
                    if name != ext_name[1:]:
                        raise DistutilsSetupError(
                            'mismatch of extension names: %s provides %r'
                            ' but expected %r' % (source, name, ext_name[1:]))
                    if typ is None:
                        typ = get_swig_target(source)
                        is_cpp = typ=='c++'
                    else:
                        typ2 = get_swig_target(source)
                        if typ2 is None:
                            log.warn('source %r does not define swig target, assuming %s swig target' \
                                     % (source, typ))
                        elif typ!=typ2:
                            log.warn('expected %r but source %r defines %r swig target' \
                                     % (typ, source, typ2))
                            if typ2=='c++':
                                log.warn('resetting swig target to c++ (some targets may have .c extension)')
                                is_cpp = True
                            else:
                                log.warn('assuming that %r has c++ swig target' % (source))
                    if is_cpp:
                        target_ext = '.cpp'
                    target_file = os.path.join(target_dir, '%s_wrap%s' \
                                               % (name, target_ext))
                else:
                    log.warn('  source %s does not exist: skipping swig\'ing.' \
                             % (source))
                    name = ext_name[1:]
                    skip_swig = 1
                    target_file = _find_swig_target(target_dir, name)
                    if not os.path.isfile(target_file):
                        log.warn('  target %s does not exist:\n   '\
                                 'Assuming %s_wrap.{c,cpp} was generated with '\
                                 '"build_src --inplace" command.' \
                                 % (target_file, name))
                        target_dir = os.path.dirname(base)
                        target_file = _find_swig_target(target_dir, name)
                        if not os.path.isfile(target_file):
                            raise DistutilsSetupError("%r missing" % (target_file,))
                        log.warn('   Yes! Using %r as up-to-date target.' \
                                 % (target_file))
                target_dirs.append(target_dir)
                new_sources.append(target_file)
                py_files.append(os.path.join(py_target_dir, name+'.py'))
                swig_sources.append(source)
                swig_targets[source] = new_sources[-1]
            else:
                new_sources.append(source)

        if not swig_sources:
            return new_sources

        if skip_swig:
            return new_sources + py_files

        for d in target_dirs:
            self.mkpath(d)

        swig = self.swig or self.find_swig()
        swig_cmd = [swig, "-python"] + extension.swig_opts
        if is_cpp:
            swig_cmd.append('-c++')
        for d in extension.include_dirs:
            swig_cmd.append('-I'+d)
        for source in swig_sources:
            target = swig_targets[source]
            depends = [source] + extension.depends
            if self.force or newer_group(depends, target, 'newer'):
                log.info("%s: %s" % (os.path.basename(swig) \
                                     + (is_cpp and '++' or ''), source))
                self.spawn(swig_cmd + self.swig_opts \
                           + ["-o", target, '-outdir', py_target_dir, source])
            else:
                log.debug("  skipping '%s' swig interface (up-to-date)" \
                         % (source))

        return new_sources + py_files

_f_pyf_ext_match = re.compile(r'.*\.(f90|f95|f77|for|ftn|f|pyf)\Z', re.I).match
_header_ext_match = re.compile(r'.*\.(inc|h|hpp)\Z', re.I).match

#### SWIG related auxiliary functions ####
_swig_module_name_match = re.compile(r'\s*%module\s*(.*\(\s*package\s*=\s*"(?P<package>[\w_]+)".*\)|)\s*(?P<name>[\w_]+)',
                                     re.I).match
_has_c_header = re.compile(r'-\*-\s*c\s*-\*-', re.I).search
_has_cpp_header = re.compile(r'-\*-\s*c\+\+\s*-\*-', re.I).search

def get_swig_target(source):
    with open(source) as f:
        result = None
        line = f.readline()
        if _has_cpp_header(line):
            result = 'c++'
        if _has_c_header(line):
            result = 'c'
    return result

def get_swig_modulename(source):
    with open(source) as f:
        name = None
        for line in f:
            m = _swig_module_name_match(line)
            if m:
                name = m.group('name')
                break
    return name

def _find_swig_target(target_dir, name):
    for ext in ['.cpp', '.c']:
        target = os.path.join(target_dir, '%s_wrap%s' % (name, ext))
        if os.path.isfile(target):
            break
    return target

#### F2PY related auxiliary functions ####

_f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                     re.I).match
_f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'
                                          r'__user__[\w_]*)', re.I).match

def get_f2py_modulename(source):
    name = None
    with open(source) as f:
        for line in f:
            m = _f2py_module_name_match(line)
            if m:
                if _f2py_user_module_name_match(line): # skip *__user__* names
                    continue
                name = m.group('name')
                break
    return name

##########################################
<|endoftext|>from numpy.distutils.fcompiler import FCompiler
from numpy.distutils import customized_fcompiler

compilers = ['NoneFCompiler']

class NoneFCompiler(FCompiler):

    compiler_type = 'none'
    description = 'Fake Fortran compiler'

    executables = {'compiler_f77': None,
                   'compiler_f90': None,
                   'compiler_fix': None,
                   'linker_so': None,
                   'linker_exe': None,
                   'archiver': None,
                   'ranlib': None,
                   'version_cmd': None,
                   }

    def find_executables(self):
        pass


if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    print(customized_fcompiler(compiler='none').get_version())
<|endoftext|>
# Absoft Corporation ceased operations on 12/31/2022.
# Thus, all links to <http://www.absoft.com> are invalid.

# Notes:
# - when using -g77 then use -DUNDERSCORE_G77 to compile f2py
#   generated extension modules (works for f2py v2.45.241_1936 and up)
import os

from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32

compilers = ['AbsoftFCompiler']

class AbsoftFCompiler(FCompiler):

    compiler_type = 'absoft'
    description = 'Absoft Corp Fortran Compiler'
    #version_pattern = r'FORTRAN 77 Compiler (?P<version>[^\s*,]*).*?Absoft Corp'
    version_pattern = r'(f90:.*?(Absoft Pro FORTRAN Version|FORTRAN 77 Compiler|Absoft Fortran Compiler Version|Copyright Absoft Corporation.*?Version))'+\
                       r' (?P<version>[^\s*,]*)(.*?Absoft Corp|)'

    # on windows: f90 -V -c dummy.f
    # f90: Copyright Absoft Corporation 1994-1998 mV2; Cray Research, Inc. 1994-1996 CF90 (2.x.x.x  f36t87) Version 2.3 Wed Apr 19, 2006  13:05:16

    # samt5735(8)$ f90 -V -c dummy.f
    # f90: Copyright Absoft Corporation 1994-2002; Absoft Pro FORTRAN Version 8.0
    # Note that fink installs g77 as f77, so need to use f90 for detection.

    executables = {
        'version_cmd'  : None,          # set by update_executables
        'compiler_f77' : ["f77"],
        'compiler_fix' : ["f90"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["<F90>"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    if os.name=='nt':
        library_switch = '/out:'      #No space after /out:!

    module_dir_switch = None
    module_include_switch = '-p'

    def update_executables(self):
        f = cyg2win32(dummy_fortran_file())
        self.executables['version_cmd'] = ['<F90>', '-V', '-c',
                                           f+'.f', '-o', f+'.o']

    def get_flags_linker_so(self):
        if os.name=='nt':
            opt = ['/dll']
        # The "-K shared" switches are being left in for pre-9.0 versions
        # of Absoft though I don't think versions earlier than 9 can
        # actually be used to build shared libraries.  In fact, version
        # 8 of Absoft doesn't recognize "-K shared" and will fail.
        elif self.get_version() >= '9.0':
            opt = ['-shared']
        else:
            opt = ["-K", "shared"]
        return opt

    def library_dir_option(self, dir):
        if os.name=='nt':
            return ['-link', '/PATH:%s' % (dir)]
        return "-L" + dir

    def library_option(self, lib):
        if os.name=='nt':
            return '%s.lib' % (lib)
        return "-l" + lib

    def get_library_dirs(self):
        opt = FCompiler.get_library_dirs(self)
        d = os.environ.get('ABSOFT')
        if d:
            if self.get_version() >= '10.0':
                # use shared libraries, the static libraries were not compiled -fPIC
                prefix = 'sh'
            else:
                prefix = ''
            if cpu.is_64bit():
                suffix = '64'
            else:
                suffix = ''
            opt.append(os.path.join(d, '%slib%s' % (prefix, suffix)))
        return opt

    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        if self.get_version() >= '11.0':
            opt.extend(['af90math', 'afio', 'af77math', 'amisc'])
        elif self.get_version() >= '10.0':
            opt.extend(['af90math', 'afio', 'af77math', 'U77'])
        elif self.get_version() >= '8.0':
            opt.extend(['f90math', 'fio', 'f77math', 'U77'])
        else:
            opt.extend(['fio', 'f90math', 'fmath', 'U77'])
        if os.name =='nt':
            opt.append('COMDLG32')
        return opt

    def get_flags(self):
        opt = FCompiler.get_flags(self)
        if os.name != 'nt':
            opt.extend(['-s'])
            if self.get_version():
                if self.get_version()>='8.2':
                    opt.append('-fpic')
        return opt

    def get_flags_f77(self):
        opt = FCompiler.get_flags_f77(self)
        opt.extend(['-N22', '-N90', '-N110'])
        v = self.get_version()
        if os.name == 'nt':
            if v and v>='8.0':
                opt.extend(['-f', '-N15'])
        else:
            opt.append('-f')
            if v:
                if v<='4.6':
                    opt.append('-B108')
                else:
                    # Though -N15 is undocumented, it works with
                    # Absoft 8.0 on Linux
                    opt.append('-N15')
        return opt

    def get_flags_f90(self):
        opt = FCompiler.get_flags_f90(self)
        opt.extend(["-YCFRL=1", "-YCOM_NAMES=LCS", "-YCOM_PFX", "-YEXT_PFX",
                    "-YCOM_SFX=_", "-YEXT_SFX=_", "-YEXT_NAMES=LCS"])
        if self.get_version():
            if self.get_version()>'4.6':
                opt.extend(["-YDEALLOC=ALL"])
        return opt

    def get_flags_fix(self):
        opt = FCompiler.get_flags_fix(self)
        opt.extend(["-YCFRL=1", "-YCOM_NAMES=LCS", "-YCOM_PFX", "-YEXT_PFX",
                    "-YCOM_SFX=_", "-YEXT_SFX=_", "-YEXT_NAMES=LCS"])
        opt.extend(["-f", "fixed"])
        return opt

    def get_flags_opt(self):
        opt = ['-O']
        return opt

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='absoft').get_version())
<|endoftext|>from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler

compilers = ['MIPSFCompiler']

class MIPSFCompiler(FCompiler):

    compiler_type = 'mips'
    description = 'MIPSpro Fortran Compiler'
    version_pattern =  r'MIPSpro Compilers: Version (?P<version>[^\s*,]*)'

    executables = {
        'version_cmd'  : ["<F90>", "-version"],
        'compiler_f77' : ["f77", "-f77"],
        'compiler_fix' : ["f90", "-fixedform"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["f90", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : None
        }
    module_dir_switch = None #XXX: fix me
    module_include_switch = None #XXX: fix me
    pic_flags = ['-KPIC']

    def get_flags(self):
        return self.pic_flags + ['-n32']
    def get_flags_opt(self):
        return ['-O3']
    def get_flags_arch(self):
        opt = []
        for a in '19 20 21 22_4k 22_5k 24 25 26 27 28 30 32_5k 32_10k'.split():
            if getattr(cpu, 'is_IP%s'%a)():
                opt.append('-TARG:platform=IP%s' % a)
                break
        return opt
    def get_flags_arch_f77(self):
        r = None
        if cpu.is_r10000(): r = 10000
        elif cpu.is_r12000(): r = 12000
        elif cpu.is_r8000(): r = 8000
        elif cpu.is_r5000(): r = 5000
        elif cpu.is_r4000(): r = 4000
        if r is not None:
            return ['r%s' % (r)]
        return []
    def get_flags_arch_f90(self):
        r = self.get_flags_arch_f77()
        if r:
            r[0] = '-' + r[0]
        return r

if __name__ == '__main__':
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='mips').get_version())
<|endoftext|>import sys
import re
from numpy.distutils.fcompiler import FCompiler

compilers = ['NAGFCompiler', 'NAGFORCompiler']

class BaseNAGFCompiler(FCompiler):
    version_pattern = r'NAG.* Release (?P<version>[^(\s]*)'

    def version_match(self, version_string):
        m = re.search(self.version_pattern, version_string)
        if m:
            return m.group('version')
        else:
            return None

    def get_flags_linker_so(self):
        return ["-Wl,-shared"]
    def get_flags_opt(self):
        return ['-O4']
    def get_flags_arch(self):
        return []

class NAGFCompiler(BaseNAGFCompiler):

    compiler_type = 'nag'
    description = 'NAGWare Fortran 95 Compiler'

    executables = {
        'version_cmd'  : ["<F90>", "-V"],
        'compiler_f77' : ["f95", "-fixed"],
        'compiler_fix' : ["f95", "-fixed"],
        'compiler_f90' : ["f95"],
        'linker_so'    : ["<F90>"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_flags_linker_so(self):
        if sys.platform == 'darwin':
            return ['-unsharedf95', '-Wl,-bundle,-flat_namespace,-undefined,suppress']
        return BaseNAGFCompiler.get_flags_linker_so(self)
    def get_flags_arch(self):
        version = self.get_version()
        if version and version < '5.1':
            return ['-target=native']
        else:
            return BaseNAGFCompiler.get_flags_arch(self)
    def get_flags_debug(self):
        return ['-g', '-gline', '-g90', '-nan', '-C']

class NAGFORCompiler(BaseNAGFCompiler):

    compiler_type = 'nagfor'
    description = 'NAG Fortran Compiler'

    executables = {
        'version_cmd'  : ["nagfor", "-V"],
        'compiler_f77' : ["nagfor", "-fixed"],
        'compiler_fix' : ["nagfor", "-fixed"],
        'compiler_f90' : ["nagfor"],
        'linker_so'    : ["nagfor"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_flags_linker_so(self):
        if sys.platform == 'darwin':
            return ['-unsharedrts',
                    '-Wl,-bundle,-flat_namespace,-undefined,suppress']
        return BaseNAGFCompiler.get_flags_linker_so(self)
    def get_flags_debug(self):
        version = self.get_version()
        if version and version > '6.1':
            return ['-g', '-u', '-nan', '-C=all', '-thread_safe',
                    '-kind=unique', '-Warn=allocation', '-Warn=subnormal']
        else:
            return ['-g', '-nan', '-C=all', '-u', '-thread_safe']


if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    compiler = customized_fcompiler(compiler='nagfor')
    print(compiler.get_version())
    print(compiler.get_flags_debug())
<|endoftext|>import os
from distutils.dist import Distribution

__metaclass__ = type

class EnvironmentConfig:
    def __init__(self, distutils_section='ALL', **kw):
        self._distutils_section = distutils_section
        self._conf_keys = kw
        self._conf = None
        self._hook_handler = None

    def dump_variable(self, name):
        conf_desc = self._conf_keys[name]
        hook, envvar, confvar, convert, append = conf_desc
        if not convert:
            convert = lambda x : x
        print('%s.%s:' % (self._distutils_section, name))
        v = self._hook_handler(name, hook)
        print('  hook   : %s' % (convert(v),))
        if envvar:
            v = os.environ.get(envvar, None)
            print('  environ: %s' % (convert(v),))
        if confvar and self._conf:
            v = self._conf.get(confvar, (None, None))[1]
            print('  config : %s' % (convert(v),))

    def dump_variables(self):
        for name in self._conf_keys:
            self.dump_variable(name)

    def __getattr__(self, name):
        try:
            conf_desc = self._conf_keys[name]
        except KeyError:
            raise AttributeError(
                f"'EnvironmentConfig' object has no attribute '{name}'"
            ) from None

        return self._get_var(name, conf_desc)

    def get(self, name, default=None):
        try:
            conf_desc = self._conf_keys[name]
        except KeyError:
            return default
        var = self._get_var(name, conf_desc)
        if var is None:
            var = default
        return var

    def _get_var(self, name, conf_desc):
        hook, envvar, confvar, convert, append = conf_desc
        if convert is None:
            convert = lambda x: x
        var = self._hook_handler(name, hook)
        if envvar is not None:
            envvar_contents = os.environ.get(envvar)
            if envvar_contents is not None:
                envvar_contents = convert(envvar_contents)
                if var and append:
                    if os.environ.get('NPY_DISTUTILS_APPEND_FLAGS', '1') == '1':
                        var.extend(envvar_contents)
                    else:
                        # NPY_DISTUTILS_APPEND_FLAGS was explicitly set to 0
                        # to keep old (overwrite flags rather than append to
                        # them) behavior
                        var = envvar_contents
                else:
                    var = envvar_contents
        if confvar is not None and self._conf:
            if confvar in self._conf:
                source, confvar_contents = self._conf[confvar]
                var = convert(confvar_contents)
        return var


    def clone(self, hook_handler):
        ec = self.__class__(distutils_section=self._distutils_section,
                            **self._conf_keys)
        ec._hook_handler = hook_handler
        return ec

    def use_distribution(self, dist):
        if isinstance(dist, Distribution):
            self._conf = dist.get_option_dict(self._distutils_section)
        else:
            self._conf = dist
<|endoftext|>from numpy.distutils.fcompiler import FCompiler

compilers = ['HPUXFCompiler']

class HPUXFCompiler(FCompiler):

    compiler_type = 'hpux'
    description = 'HP Fortran 90 Compiler'
    version_pattern =  r'HP F90 (?P<version>[^\s*,]*)'

    executables = {
        'version_cmd'  : ["f90", "+version"],
        'compiler_f77' : ["f90"],
        'compiler_fix' : ["f90"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["ld", "-b"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    module_dir_switch = None #XXX: fix me
    module_include_switch = None #XXX: fix me
    pic_flags = ['+Z']
    def get_flags(self):
        return self.pic_flags + ['+ppu', '+DD64']
    def get_flags_opt(self):
        return ['-O3']
    def get_libraries(self):
        return ['m']
    def get_library_dirs(self):
        opt = ['/usr/lib/hpux64']
        return opt
    def get_version(self, force=0, ok_status=[256, 0, 1]):
        # XXX status==256 may indicate 'unrecognized option' or
        #     'no input file'. So, version_cmd needs more work.
        return FCompiler.get_version(self, force, ok_status)

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(10)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='hpux').get_version())
<|endoftext|>
#http://www.compaq.com/fortran/docs/
import os
import sys

from numpy.distutils.fcompiler import FCompiler
from distutils.errors import DistutilsPlatformError

compilers = ['CompaqFCompiler']
if os.name != 'posix' or sys.platform[:6] == 'cygwin' :
    # Otherwise we'd get a false positive on posix systems with
    # case-insensitive filesystems (like darwin), because we'll pick
    # up /bin/df
    compilers.append('CompaqVisualFCompiler')

class CompaqFCompiler(FCompiler):

    compiler_type = 'compaq'
    description = 'Compaq Fortran Compiler'
    version_pattern = r'Compaq Fortran (?P<version>[^\s]*).*'

    if sys.platform[:5]=='linux':
        fc_exe = 'fort'
    else:
        fc_exe = 'f90'

    executables = {
        'version_cmd'  : ['<F90>', "-version"],
        'compiler_f77' : [fc_exe, "-f77rtl", "-fixed"],
        'compiler_fix' : [fc_exe, "-fixed"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : ['<F90>'],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    module_dir_switch = '-module ' # not tested
    module_include_switch = '-I'

    def get_flags(self):
        return ['-assume no2underscore', '-nomixed_str_len_arg']
    def get_flags_debug(self):
        return ['-g', '-check bounds']
    def get_flags_opt(self):
        return ['-O4', '-align dcommons', '-assume bigarrays',
                '-assume nozsize', '-math_library fast']
    def get_flags_arch(self):
        return ['-arch host', '-tune host']
    def get_flags_linker_so(self):
        if sys.platform[:5]=='linux':
            return ['-shared']
        return ['-shared', '-Wl,-expect_unresolved,*']

class CompaqVisualFCompiler(FCompiler):

    compiler_type = 'compaqv'
    description = 'DIGITAL or Compaq Visual Fortran Compiler'
    version_pattern = (r'(DIGITAL|Compaq) Visual Fortran Optimizing Compiler'
                       r' Version (?P<version>[^\s]*).*')

    compile_switch = '/compile_only'
    object_switch = '/object:'
    library_switch = '/OUT:'      #No space after /OUT:!

    static_lib_extension = ".lib"
    static_lib_format = "%s%s"
    module_dir_switch = '/module:'
    module_include_switch = '/I'

    ar_exe = 'lib.exe'
    fc_exe = 'DF'

    if sys.platform=='win32':
        from numpy.distutils.msvccompiler import MSVCCompiler

        try:
            m = MSVCCompiler()
            m.initialize()
            ar_exe = m.lib
        except DistutilsPlatformError:
            pass
        except AttributeError as e:
            if '_MSVCCompiler__root' in str(e):
                print('Ignoring "%s" (I think it is msvccompiler.py bug)' % (e))
            else:
                raise
        except OSError as e:
            if not "vcvarsall.bat" in str(e):
                print("Unexpected OSError in", __file__)
                raise
        except ValueError as e:
            if not "'path'" in str(e):
                print("Unexpected ValueError in", __file__)
                raise

    executables = {
        'version_cmd'  : ['<F90>', "/what"],
        'compiler_f77' : [fc_exe, "/f77rtl", "/fixed"],
        'compiler_fix' : [fc_exe, "/fixed"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : ['<F90>'],
        'archiver'     : [ar_exe, "/OUT:"],
        'ranlib'       : None
        }

    def get_flags(self):
        return ['/nologo', '/MD', '/WX', '/iface=(cref,nomixed_str_len_arg)',
                '/names:lowercase', '/assume:underscore']
    def get_flags_opt(self):
        return ['/Ox', '/fast', '/optimize:5', '/unroll:0', '/math_library:fast']
    def get_flags_arch(self):
        return ['/threads']
    def get_flags_debug(self):
        return ['/debug']

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='compaq').get_version())
<|endoftext|>import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion

compilers = ['GnuFCompiler', 'Gnu95FCompiler']

TARGET_R = re.compile(r"Target: ([a-zA-Z0-9_\-]*)")

# XXX: handle cross compilation


def is_win64():
    return sys.platform == "win32" and platform.architecture()[0] == "64bit"


class GnuFCompiler(FCompiler):
    compiler_type = 'gnu'
    compiler_aliases = ('g77', )
    description = 'GNU Fortran 77 compiler'

    def gnu_version_match(self, version_string):
        """Handle the different versions of GNU fortran compilers"""
        # Strip warning(s) that may be emitted by gfortran
        while version_string.startswith('gfortran: warning'):
            version_string =\
                version_string[version_string.find('\n') + 1:].strip()

        # Gfortran versions from after 2010 will output a simple string
        # (usually "x.y", "x.y.z" or "x.y.z-q") for ``-dumpversion``; older
        # gfortrans may still return long version strings (``-dumpversion`` was
        # an alias for ``--version``)
        if len(version_string) <= 20:
            # Try to find a valid version string
            m = re.search(r'([0-9.]+)', version_string)
            if m:
                # g77 provides a longer version string that starts with GNU
                # Fortran
                if version_string.startswith('GNU Fortran'):
                    return ('g77', m.group(1))

                # gfortran only outputs a version string such as #.#.#, so check
                # if the match is at the start of the string
                elif m.start() == 0:
                    return ('gfortran', m.group(1))
        else:
            # Output probably from --version, try harder:
            m = re.search(r'GNU Fortran\s+95.*?([0-9-.]+)', version_string)
            if m:
                return ('gfortran', m.group(1))
            m = re.search(
                r'GNU Fortran.*?\-?([0-9-.]+\.[0-9-.]+)', version_string)
            if m:
                v = m.group(1)
                if v.startswith('0') or v.startswith('2') or v.startswith('3'):
                    # the '0' is for early g77's
                    return ('g77', v)
                else:
                    # at some point in the 4.x series, the ' 95' was dropped
                    # from the version string
                    return ('gfortran', v)

        # If still nothing, raise an error to make the problem easy to find.
        err = 'A valid Fortran version was not found in this string:\n'
        raise ValueError(err + version_string)

    def version_match(self, version_string):
        v = self.gnu_version_match(version_string)
        if not v or v[0] != 'g77':
            return None
        return v[1]

    possible_executables = ['g77', 'f77']
    executables = {
        'version_cmd'  : [None, "-dumpversion"],
        'compiler_f77' : [None, "-g", "-Wall", "-fno-second-underscore"],
        'compiler_f90' : None,  # Use --fcompiler=gnu95 for f90 codes
        'compiler_fix' : None,
        'linker_so'    : [None, "-g", "-Wall"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"],
        'linker_exe'   : [None, "-g", "-Wall"]
    }
    module_dir_switch = None
    module_include_switch = None

    # Cygwin: f771: warning: -fPIC ignored for target (all code is
    # position independent)
    if os.name != 'nt' and sys.platform != 'cygwin':
        pic_flags = ['-fPIC']

    # use -mno-cygwin for g77 when Python is not Cygwin-Python
    if sys.platform == 'win32':
        for key in ['version_cmd', 'compiler_f77', 'linker_so', 'linker_exe']:
            executables[key].append('-mno-cygwin')

    g2c = 'g2c'
    suggested_f90_compiler = 'gnu95'

    def get_flags_linker_so(self):
        opt = self.linker_so[1:]
        if sys.platform == 'darwin':
            target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
            # If MACOSX_DEPLOYMENT_TARGET is set, we simply trust the value
            # and leave it alone.  But, distutils will complain if the
            # environment's value is different from the one in the Python
            # Makefile used to build Python.  We let distutils handle this
            # error checking.
            if not target:
                # If MACOSX_DEPLOYMENT_TARGET is not set in the environment,
                # we try to get it first from sysconfig and then
                # fall back to setting it to 10.9 This is a reasonable default
                # even when using the official Python dist and those derived
                # from it.
                import sysconfig
                target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
                if not target:
                    target = '10.9'
                    s = f'Env. variable MACOSX_DEPLOYMENT_TARGET set to {target}'
                    warnings.warn(s, stacklevel=2)
                os.environ['MACOSX_DEPLOYMENT_TARGET'] = str(target)
            opt.extend(['-undefined', 'dynamic_lookup', '-bundle'])
        else:
            opt.append("-shared")
        if sys.platform.startswith('sunos'):
            # SunOS often has dynamically loaded symbols defined in the
            # static library libg2c.a  The linker doesn't like this.  To
            # ignore the problem, use the -mimpure-text flag.  It isn't
            # the safest thing, but seems to work. 'man gcc' says:
            # ".. Instead of using -mimpure-text, you should compile all
            #  source code with -fpic or -fPIC."
            opt.append('-mimpure-text')
        return opt

    def get_libgcc_dir(self):
        try:
            output = subprocess.check_output(self.compiler_f77 +
                                            ['-print-libgcc-file-name'])
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            output = filepath_from_subprocess_output(output)
            return os.path.dirname(output)
        return None

    def get_libgfortran_dir(self):
        if sys.platform[:5] == 'linux':
            libgfortran_name = 'libgfortran.so'
        elif sys.platform == 'darwin':
            libgfortran_name = 'libgfortran.dylib'
        else:
            libgfortran_name = None

        libgfortran_dir = None
        if libgfortran_name:
            find_lib_arg = ['-print-file-name={0}'.format(libgfortran_name)]
            try:
                output = subprocess.check_output(
                                       self.compiler_f77 + find_lib_arg)
            except (OSError, subprocess.CalledProcessError):
                pass
            else:
                output = filepath_from_subprocess_output(output)
                libgfortran_dir = os.path.dirname(output)
        return libgfortran_dir

    def get_library_dirs(self):
        opt = []
        if sys.platform[:5] != 'linux':
            d = self.get_libgcc_dir()
            if d:
                # if windows and not cygwin, libg2c lies in a different folder
                if sys.platform == 'win32' and not d.startswith('/usr/lib'):
                    d = os.path.normpath(d)
                    path = os.path.join(d, "lib%s.a" % self.g2c)
                    if not os.path.exists(path):
                        root = os.path.join(d, *((os.pardir, ) * 4))
                        d2 = os.path.abspath(os.path.join(root, 'lib'))
                        path = os.path.join(d2, "lib%s.a" % self.g2c)
                        if os.path.exists(path):
                            opt.append(d2)
                opt.append(d)
        # For Macports / Linux, libgfortran and libgcc are not co-located
        lib_gfortran_dir = self.get_libgfortran_dir()
        if lib_gfortran_dir:
            opt.append(lib_gfortran_dir)
        return opt

    def get_libraries(self):
        opt = []
        d = self.get_libgcc_dir()
        if d is not None:
            g2c = self.g2c + '-pic'
            f = self.static_lib_format % (g2c, self.static_lib_extension)
            if not os.path.isfile(os.path.join(d, f)):
                g2c = self.g2c
        else:
            g2c = self.g2c

        if g2c is not None:
            opt.append(g2c)
        c_compiler = self.c_compiler
        if sys.platform == 'win32' and c_compiler and \
                c_compiler.compiler_type == 'msvc':
            opt.append('gcc')
        if sys.platform == 'darwin':
            opt.append('cc_dynamic')
        return opt

    def get_flags_debug(self):
        return ['-g']

    def get_flags_opt(self):
        v = self.get_version()
        if v and v <= '3.3.3':
            # With this compiler version building Fortran BLAS/LAPACK
            # with -O3 caused failures in lib.lapack heevr,syevr tests.
            opt = ['-O2']
        else:
            opt = ['-O3']
        opt.append('-funroll-loops')
        return opt

    def _c_arch_flags(self):
        """ Return detected arch flags from CFLAGS """
        import sysconfig
        try:
            cflags = sysconfig.get_config_vars()['CFLAGS']
        except KeyError:
            return []
        arch_re = re.compile(r"-arch\s+(\w+)")
        arch_flags = []
        for arch in arch_re.findall(cflags):
            arch_flags += ['-arch', arch]
        return arch_flags

    def get_flags_arch(self):
        return []

    def runtime_library_dir_option(self, dir):
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            # Linux/Solaris/Unix support RPATH, Windows does not
            raise NotImplementedError

        # TODO: could use -Xlinker here, if it's supported
        assert "," not in dir

        if sys.platform == 'darwin':
            return f'-Wl,-rpath,{dir}'
        elif sys.platform.startswith(('aix', 'os400')):
            # AIX RPATH is called LIBPATH
            return f'-Wl,-blibpath:{dir}'
        else:
            return f'-Wl,-rpath={dir}'


class Gnu95FCompiler(GnuFCompiler):
    compiler_type = 'gnu95'
    compiler_aliases = ('gfortran', )
    description = 'GNU Fortran 95 compiler'

    def version_match(self, version_string):
        v = self.gnu_version_match(version_string)
        if not v or v[0] != 'gfortran':
            return None
        v = v[1]
        if LooseVersion(v) >= "4":
            # gcc-4 series releases do not support -mno-cygwin option
            pass
        else:
            # use -mno-cygwin flag for gfortran when Python is not
            # Cygwin-Python
            if sys.platform == 'win32':
                for key in [
                        'version_cmd', 'compiler_f77', 'compiler_f90',
                        'compiler_fix', 'linker_so', 'linker_exe'
                ]:
                    self.executables[key].append('-mno-cygwin')
        return v

    possible_executables = ['gfortran', 'f95']
    executables = {
        'version_cmd'  : ["<F90>", "-dumpversion"],
        'compiler_f77' : [None, "-Wall", "-g", "-ffixed-form",
                          "-fno-second-underscore"],
        'compiler_f90' : [None, "-Wall", "-g",
                          "-fno-second-underscore"],
        'compiler_fix' : [None, "-Wall",  "-g","-ffixed-form",
                          "-fno-second-underscore"],
        'linker_so'    : ["<F90>", "-Wall", "-g"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"],
        'linker_exe'   : [None, "-Wall"]
    }

    module_dir_switch = '-J'
    module_include_switch = '-I'

    if sys.platform.startswith(('aix', 'os400')):
        executables['linker_so'].append('-lpthread')
        if platform.architecture()[0][:2] == '64':
            for key in ['compiler_f77', 'compiler_f90','compiler_fix','linker_so', 'linker_exe']:
                executables[key].append('-maix64')

    g2c = 'gfortran'

    def _universal_flags(self, cmd):
        """Return a list of -arch flags for every supported architecture."""
        if not sys.platform == 'darwin':
            return []
        arch_flags = []
        # get arches the C compiler gets.
        c_archs = self._c_arch_flags()
        if "i386" in c_archs:
            c_archs[c_archs.index("i386")] = "i686"
        # check the arches the Fortran compiler supports, and compare with
        # arch flags from C compiler
        for arch in ["ppc", "i686", "x86_64", "ppc64", "s390x"]:
            if _can_target(cmd, arch) and arch in c_archs:
                arch_flags.extend(["-arch", arch])
        return arch_flags

    def get_flags(self):
        flags = GnuFCompiler.get_flags(self)
        arch_flags = self._universal_flags(self.compiler_f90)
        if arch_flags:
            flags[:0] = arch_flags
        return flags

    def get_flags_linker_so(self):
        flags = GnuFCompiler.get_flags_linker_so(self)
        arch_flags = self._universal_flags(self.linker_so)
        if arch_flags:
            flags[:0] = arch_flags
        return flags

    def get_library_dirs(self):
        opt = GnuFCompiler.get_library_dirs(self)
        if sys.platform == 'win32':
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == "msvc":
                target = self.get_target()
                if target:
                    d = os.path.normpath(self.get_libgcc_dir())
                    root = os.path.join(d, *((os.pardir, ) * 4))
                    path = os.path.join(root, "lib")
                    mingwdir = os.path.normpath(path)
                    if os.path.exists(os.path.join(mingwdir, "libmingwex.a")):
                        opt.append(mingwdir)
        # For Macports / Linux, libgfortran and libgcc are not co-located
        lib_gfortran_dir = self.get_libgfortran_dir()
        if lib_gfortran_dir:
            opt.append(lib_gfortran_dir)
        return opt

    def get_libraries(self):
        opt = GnuFCompiler.get_libraries(self)
        if sys.platform == 'darwin':
            opt.remove('cc_dynamic')
        if sys.platform == 'win32':
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == "msvc":
                if "gcc" in opt:
                    i = opt.index("gcc")
                    opt.insert(i + 1, "mingwex")
                    opt.insert(i + 1, "mingw32")
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == "msvc":
                return []
            else:
                pass
        return opt

    def get_target(self):
        try:
            p = subprocess.Popen(
                self.compiler_f77 + ['-v'],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = p.communicate()
            output = (stdout or b"") + (stderr or b"")
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            output = filepath_from_subprocess_output(output)
            m = TARGET_R.search(output)
            if m:
                return m.group(1)
        return ""

    def _hash_files(self, filenames):
        h = hashlib.sha1()
        for fn in filenames:
            with open(fn, 'rb') as f:
                while True:
                    block = f.read(131072)
                    if not block:
                        break
                    h.update(block)
        text = base64.b32encode(h.digest())
        text = text.decode('ascii')
        return text.rstrip('=')

    def _link_wrapper_lib(self, objects, output_dir, extra_dll_dir,
                          chained_dlls, is_archive):
        """Create a wrapper shared library for the given objects

        Return an MSVC-compatible lib
        """

        c_compiler = self.c_compiler
        if c_compiler.compiler_type != "msvc":
            raise ValueError("This method only supports MSVC")

        object_hash = self._hash_files(list(objects) + list(chained_dlls))

        if is_win64():
            tag = 'win_amd64'
        else:
            tag = 'win32'

        basename = 'lib' + os.path.splitext(
            os.path.basename(objects[0]))[0][:8]
        root_name = basename + '.' + object_hash + '.gfortran-' + tag
        dll_name = root_name + '.dll'
        def_name = root_name + '.def'
        lib_name = root_name + '.lib'
        dll_path = os.path.join(extra_dll_dir, dll_name)
        def_path = os.path.join(output_dir, def_name)
        lib_path = os.path.join(output_dir, lib_name)

        if os.path.isfile(lib_path):
            # Nothing to do
            return lib_path, dll_path

        if is_archive:
            objects = (["-Wl,--whole-archive"] + list(objects) +
                       ["-Wl,--no-whole-archive"])
        self.link_shared_object(
            objects,
            dll_name,
            output_dir=extra_dll_dir,
            extra_postargs=list(chained_dlls) + [
                '-Wl,--allow-multiple-definition',
                '-Wl,--output-def,' + def_path,
                '-Wl,--export-all-symbols',
                '-Wl,--enable-auto-import',
                '-static',
                '-mlong-double-64',
            ])

        # No PowerPC!
        if is_win64():
            specifier = '/MACHINE:X64'
        else:
            specifier = '/MACHINE:X86'

        # MSVC specific code
        lib_args = ['/def:' + def_path, '/OUT:' + lib_path, specifier]
        if not c_compiler.initialized:
            c_compiler.initialize()
        c_compiler.spawn([c_compiler.lib] + lib_args)

        return lib_path, dll_path

    def can_ccompiler_link(self, compiler):
        # MSVC cannot link objects compiled by GNU fortran
        return compiler.compiler_type not in ("msvc", )

    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.
        """
        if self.c_compiler.compiler_type == "msvc":
            # Compile a DLL and return the lib for the DLL as
            # the object. Also keep track of previous DLLs that
            # we have compiled so that we can link against them.

            # If there are .a archives, assume they are self-contained
            # static libraries, and build separate DLLs for each
            archives = []
            plain_objects = []
            for obj in objects:
                if obj.lower().endswith('.a'):
                    archives.append(obj)
                else:
                    plain_objects.append(obj)

            chained_libs = []
            chained_dlls = []
            for archive in archives[::-1]:
                lib, dll = self._link_wrapper_lib(
                    [archive],
                    output_dir,
                    extra_dll_dir,
                    chained_dlls=chained_dlls,
                    is_archive=True)
                chained_libs.insert(0, lib)
                chained_dlls.insert(0, dll)

            if not plain_objects:
                return chained_libs

            lib, dll = self._link_wrapper_lib(
                plain_objects,
                output_dir,
                extra_dll_dir,
                chained_dlls=chained_dlls,
                is_archive=False)
            return [lib] + chained_libs
        else:
            raise ValueError("Unsupported C compiler")


def _can_target(cmd, arch):
    """Return true if the architecture supports the -arch flag"""
    newcmd = cmd[:]
    fid, filename = tempfile.mkstemp(suffix=".f")
    os.close(fid)
    try:
        d = os.path.dirname(filename)
        output = os.path.splitext(filename)[0] + ".o"
        try:
            newcmd.extend(["-arch", arch, "-c", filename])
            p = Popen(newcmd, stderr=STDOUT, stdout=PIPE, cwd=d)
            p.communicate()
            return p.returncode == 0
        finally:
            if os.path.exists(output):
                os.remove(output)
    finally:
        os.remove(filename)


if __name__ == '__main__':
    from distutils import log
    from numpy.distutils import customized_fcompiler
    log.set_verbosity(2)

    print(customized_fcompiler('gnu').get_version())
    try:
        print(customized_fcompiler('g95').get_version())
    except Exception as e:
        print(e)
<|endoftext|>import sys                                                                     
                                                                               
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from sys import platform                                                       
from os.path import join, dirname, normpath

compilers = ['ArmFlangCompiler']

import functools

class ArmFlangCompiler(FCompiler):
    compiler_type = 'arm'
    description = 'Arm Compiler'
    version_pattern = r'\s*Arm.*version (?P<version>[\d.-]+).*'

    ar_exe = 'lib.exe'
    possible_executables = ['armflang']

    executables = {
        'version_cmd': ["", "--version"],
        'compiler_f77': ["armflang", "-fPIC"],
        'compiler_fix': ["armflang", "-fPIC", "-ffixed-form"],
        'compiler_f90': ["armflang", "-fPIC"],
        'linker_so': ["armflang", "-fPIC", "-shared"],
        'archiver': ["ar", "-cr"],
        'ranlib':  None
    }

    pic_flags = ["-fPIC", "-DPIC"]
    c_compiler = 'arm'
    module_dir_switch = '-module '  # Don't remove ending space!

    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        opt.extend(['flang', 'flangrti', 'ompstub'])
        return opt

    @functools.lru_cache(maxsize=128)
    def get_library_dirs(self):
        """List of compiler library directories."""
        opt = FCompiler.get_library_dirs(self)
        flang_dir = dirname(self.executables['compiler_f77'][0])
        opt.append(normpath(join(flang_dir, '..', 'lib')))

        return opt

    def get_flags(self):
        return []

    def get_flags_free(self):
        return []

    def get_flags_debug(self):
        return ['-g']

    def get_flags_opt(self):
        return ['-O3']

    def get_flags_arch(self):
        return []

    def runtime_library_dir_option(self, dir):
        return '-Wl,-rpath=%s' % dir


if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='armflang').get_version())

<|endoftext|># http://g95.sourceforge.net/
from numpy.distutils.fcompiler import FCompiler

compilers = ['G95FCompiler']

class G95FCompiler(FCompiler):
    compiler_type = 'g95'
    description = 'G95 Fortran Compiler'

#    version_pattern = r'G95 \((GCC (?P<gccversion>[\d.]+)|.*?) \(g95!\) (?P<version>.*)\).*'
    # $ g95 --version
    # G95 (GCC 4.0.3 (g95!) May 22 2006)

    version_pattern = r'G95 \((GCC (?P<gccversion>[\d.]+)|.*?) \(g95 (?P<version>.*)!\) (?P<date>.*)\).*'
    # $ g95 --version
    # G95 (GCC 4.0.3 (g95 0.90!) Aug 22 2006)

    executables = {
        'version_cmd'  : ["<F90>", "--version"],
        'compiler_f77' : ["g95", "-ffixed-form"],
        'compiler_fix' : ["g95", "-ffixed-form"],
        'compiler_f90' : ["g95"],
        'linker_so'    : ["<F90>", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    pic_flags = ['-fpic']
    module_dir_switch = '-fmod='
    module_include_switch = '-I'

    def get_flags(self):
        return ['-fno-second-underscore']
    def get_flags_opt(self):
        return ['-O']
    def get_flags_debug(self):
        return ['-g']

if __name__ == '__main__':
    from distutils import log
    from numpy.distutils import customized_fcompiler
    log.set_verbosity(2)
    print(customized_fcompiler('g95').get_version())
<|endoftext|>from numpy.distutils.fcompiler import FCompiler

compilers = ['NVHPCFCompiler']

class NVHPCFCompiler(FCompiler):
    """ NVIDIA High Performance Computing (HPC) SDK Fortran Compiler
   
    https://developer.nvidia.com/hpc-sdk
   
    Since august 2020 the NVIDIA HPC SDK includes the compilers formerly known as The Portland Group compilers,
    https://www.pgroup.com/index.htm.
    See also `numpy.distutils.fcompiler.pg`.
    """

    compiler_type = 'nv'
    description = 'NVIDIA HPC SDK'
    version_pattern = r'\s*(nvfortran|.+ \(aka nvfortran\)) (?P<version>[\d.-]+).*'

    executables = {
        'version_cmd': ["<F90>", "-V"],
        'compiler_f77': ["nvfortran"],
        'compiler_fix': ["nvfortran", "-Mfixed"],
        'compiler_f90': ["nvfortran"],
        'linker_so': ["<F90>"],
        'archiver': ["ar", "-cr"],
        'ranlib': ["ranlib"]
    }
    pic_flags = ['-fpic']

    module_dir_switch = '-module '
    module_include_switch = '-I'

    def get_flags(self):
        opt = ['-Minform=inform', '-Mnosecond_underscore']
        return self.pic_flags + opt

    def get_flags_opt(self):
        return ['-fast']

    def get_flags_debug(self):
        return ['-g']

    def get_flags_linker_so(self):
        return ["-shared", '-fpic']

    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='nv').get_version())
<|endoftext|># http://www.pgroup.com
import sys

from numpy.distutils.fcompiler import FCompiler
from sys import platform
from os.path import join, dirname, normpath

compilers = ['PGroupFCompiler', 'PGroupFlangCompiler']


class PGroupFCompiler(FCompiler):

    compiler_type = 'pg'
    description = 'Portland Group Fortran Compiler'
    version_pattern = r'\s*pg(f77|f90|hpf|fortran) (?P<version>[\d.-]+).*'

    if platform == 'darwin':
        executables = {
            'version_cmd': ["<F77>", "-V"],
            'compiler_f77': ["pgfortran", "-dynamiclib"],
            'compiler_fix': ["pgfortran", "-Mfixed", "-dynamiclib"],
            'compiler_f90': ["pgfortran", "-dynamiclib"],
            'linker_so': ["libtool"],
            'archiver': ["ar", "-cr"],
            'ranlib': ["ranlib"]
        }
        pic_flags = ['']
    else:
        executables = {
            'version_cmd': ["<F77>", "-V"],
            'compiler_f77': ["pgfortran"],
            'compiler_fix': ["pgfortran", "-Mfixed"],
            'compiler_f90': ["pgfortran"],
            'linker_so': ["<F90>"],
            'archiver': ["ar", "-cr"],
            'ranlib': ["ranlib"]
        }
        pic_flags = ['-fpic']

    module_dir_switch = '-module '
    module_include_switch = '-I'

    def get_flags(self):
        opt = ['-Minform=inform', '-Mnosecond_underscore']
        return self.pic_flags + opt

    def get_flags_opt(self):
        return ['-fast']

    def get_flags_debug(self):
        return ['-g']

    if platform == 'darwin':
        def get_flags_linker_so(self):
            return ["-dynamic", '-undefined', 'dynamic_lookup']

    else:
        def get_flags_linker_so(self):
            return ["-shared", '-fpic']

    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir


import functools

class PGroupFlangCompiler(FCompiler):
    compiler_type = 'flang'
    description = 'Portland Group Fortran LLVM Compiler'
    version_pattern = r'\s*(flang|clang) version (?P<version>[\d.-]+).*'

    ar_exe = 'lib.exe'
    possible_executables = ['flang']

    executables = {
        'version_cmd': ["<F77>", "--version"],
        'compiler_f77': ["flang"],
        'compiler_fix': ["flang"],
        'compiler_f90': ["flang"],
        'linker_so': [None],
        'archiver': [ar_exe, "/verbose", "/OUT:"],
        'ranlib': None
    }

    library_switch = '/OUT:'  # No space after /OUT:!
    module_dir_switch = '-module '  # Don't remove ending space!

    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        opt.extend(['flang', 'flangrti', 'ompstub'])
        return opt

    @functools.lru_cache(maxsize=128)
    def get_library_dirs(self):
        """List of compiler library directories."""
        opt = FCompiler.get_library_dirs(self)
        flang_dir = dirname(self.executables['compiler_f77'][0])
        opt.append(normpath(join(flang_dir, '..', 'lib')))

        return opt

    def get_flags(self):
        return []

    def get_flags_free(self):
        return []

    def get_flags_debug(self):
        return ['-g']

    def get_flags_opt(self):
        return ['-O3']

    def get_flags_arch(self):
        return []

    def runtime_library_dir_option(self, dir):
        raise NotImplementedError


if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    if 'flang' in sys.argv:
        print(customized_fcompiler(compiler='flang').get_version())
    else:
        print(customized_fcompiler(compiler='pg').get_version())
<|endoftext|>"""numpy.distutils.fcompiler

Contains FCompiler, an abstract base class that defines the interface
for the numpy.distutils Fortran compiler abstraction model.

Terminology:

To be consistent, where the term 'executable' is used, it means the single
file, like 'gcc', that is executed, and should be a string. In contrast,
'command' means the entire command line, like ['gcc', '-c', 'file.c'], and
should be a list.

But note that FCompiler.executables is actually a dictionary of commands.

"""
__all__ = ['FCompiler', 'new_fcompiler', 'show_fcompilers',
           'dummy_fortran_file']

import os
import sys
import re
from pathlib import Path

from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
     DistutilsExecError, CompileError, LinkError, DistutilsPlatformError
from distutils.util import split_quoted, strtobool

from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
    make_temp_file, get_shared_lib_extension
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils

from .environment import EnvironmentConfig

__metaclass__ = type


FORTRAN_COMMON_FIXED_EXTENSIONS = ['.for', '.ftn', '.f77', '.f']


class CompilerNotFound(Exception):
    pass

def flaglist(s):
    if is_string(s):
        return split_quoted(s)
    else:
        return s

def str2bool(s):
    if is_string(s):
        return strtobool(s)
    return bool(s)

def is_sequence_of_strings(seq):
    return is_sequence(seq) and all_strings(seq)

class FCompiler(CCompiler):
    """Abstract base class to define the interface that must be implemented
    by real Fortran compiler classes.

    Methods that subclasses may redefine:

        update_executables(), find_executables(), get_version()
        get_flags(), get_flags_opt(), get_flags_arch(), get_flags_debug()
        get_flags_f77(), get_flags_opt_f77(), get_flags_arch_f77(),
        get_flags_debug_f77(), get_flags_f90(), get_flags_opt_f90(),
        get_flags_arch_f90(), get_flags_debug_f90(),
        get_flags_fix(), get_flags_linker_so()

    DON'T call these methods (except get_version) after
    constructing a compiler instance or inside any other method.
    All methods, except update_executables() and find_executables(),
    may call the get_version() method.

    After constructing a compiler instance, always call customize(dist=None)
    method that finalizes compiler construction and makes the following
    attributes available:
      compiler_f77
      compiler_f90
      compiler_fix
      linker_so
      archiver
      ranlib
      libraries
      library_dirs
    """

    # These are the environment variables and distutils keys used.
    # Each configuration description is
    # (<hook name>, <environment variable>, <key in distutils.cfg>, <convert>, <append>)
    # The hook names are handled by the self._environment_hook method.
    #  - names starting with 'self.' call methods in this class
    #  - names starting with 'exe.' return the key in the executables dict
    #  - names like 'flags.YYY' return self.get_flag_YYY()
    # convert is either None or a function to convert a string to the
    # appropriate type used.

    distutils_vars = EnvironmentConfig(
        distutils_section='config_fc',
        noopt = (None, None, 'noopt', str2bool, False),
        noarch = (None, None, 'noarch', str2bool, False),
        debug = (None, None, 'debug', str2bool, False),
        verbose = (None, None, 'verbose', str2bool, False),
    )

    command_vars = EnvironmentConfig(
        distutils_section='config_fc',
        compiler_f77 = ('exe.compiler_f77', 'F77', 'f77exec', None, False),
        compiler_f90 = ('exe.compiler_f90', 'F90', 'f90exec', None, False),
        compiler_fix = ('exe.compiler_fix', 'F90', 'f90exec', None, False),
        version_cmd = ('exe.version_cmd', None, None, None, False),
        linker_so = ('exe.linker_so', 'LDSHARED', 'ldshared', None, False),
        linker_exe = ('exe.linker_exe', 'LD', 'ld', None, False),
        archiver = (None, 'AR', 'ar', None, False),
        ranlib = (None, 'RANLIB', 'ranlib', None, False),
    )

    flag_vars = EnvironmentConfig(
        distutils_section='config_fc',
        f77 = ('flags.f77', 'F77FLAGS', 'f77flags', flaglist, True),
        f90 = ('flags.f90', 'F90FLAGS', 'f90flags', flaglist, True),
        free = ('flags.free', 'FREEFLAGS', 'freeflags', flaglist, True),
        fix = ('flags.fix', None, None, flaglist, False),
        opt = ('flags.opt', 'FOPT', 'opt', flaglist, True),
        opt_f77 = ('flags.opt_f77', None, None, flaglist, False),
        opt_f90 = ('flags.opt_f90', None, None, flaglist, False),
        arch = ('flags.arch', 'FARCH', 'arch', flaglist, False),
        arch_f77 = ('flags.arch_f77', None, None, flaglist, False),
        arch_f90 = ('flags.arch_f90', None, None, flaglist, False),
        debug = ('flags.debug', 'FDEBUG', 'fdebug', flaglist, True),
        debug_f77 = ('flags.debug_f77', None, None, flaglist, False),
        debug_f90 = ('flags.debug_f90', None, None, flaglist, False),
        flags = ('self.get_flags', 'FFLAGS', 'fflags', flaglist, True),
        linker_so = ('flags.linker_so', 'LDFLAGS', 'ldflags', flaglist, True),
        linker_exe = ('flags.linker_exe', 'LDFLAGS', 'ldflags', flaglist, True),
        ar = ('flags.ar', 'ARFLAGS', 'arflags', flaglist, True),
    )

    language_map = {'.f': 'f77',
                    '.for': 'f77',
                    '.F': 'f77',    # XXX: needs preprocessor
                    '.ftn': 'f77',
                    '.f77': 'f77',
                    '.f90': 'f90',
                    '.F90': 'f90',  # XXX: needs preprocessor
                    '.f95': 'f90',
                    }
    language_order = ['f90', 'f77']


    # These will be set by the subclass

    compiler_type = None
    compiler_aliases = ()
    version_pattern = None

    possible_executables = []
    executables = {
        'version_cmd': ["f77", "-v"],
        'compiler_f77': ["f77"],
        'compiler_f90': ["f90"],
        'compiler_fix': ["f90", "-fixed"],
        'linker_so': ["f90", "-shared"],
        'linker_exe': ["f90"],
        'archiver': ["ar", "-cr"],
        'ranlib': None,
        }

    # If compiler does not support compiling Fortran 90 then it can
    # suggest using another compiler. For example, gnu would suggest
    # gnu95 compiler type when there are F90 sources.
    suggested_f90_compiler = None

    compile_switch = "-c"
    object_switch = "-o "   # Ending space matters! It will be stripped
                            # but if it is missing then object_switch
                            # will be prefixed to object file name by
                            # string concatenation.
    library_switch = "-o "  # Ditto!

    # Switch to specify where module files are created and searched
    # for USE statement.  Normally it is a string and also here ending
    # space matters. See above.
    module_dir_switch = None

    # Switch to specify where module files are searched for USE statement.
    module_include_switch = '-I'

    pic_flags = []           # Flags to create position-independent code

    src_extensions = ['.for', '.ftn', '.f77', '.f', '.f90', '.f95', '.F', '.F90', '.FOR']
    obj_extension = ".o"

    shared_lib_extension = get_shared_lib_extension()
    static_lib_extension = ".a"  # or .lib
    static_lib_format = "lib%s%s" # or %s%s
    shared_lib_format = "%s%s"
    exe_extension = ""

    _exe_cache = {}

    _executable_keys = ['version_cmd', 'compiler_f77', 'compiler_f90',
                        'compiler_fix', 'linker_so', 'linker_exe', 'archiver',
                        'ranlib']

    # This will be set by new_fcompiler when called in
    # command/{build_ext.py, build_clib.py, config.py} files.
    c_compiler = None

    # extra_{f77,f90}_compile_args are set by build_ext.build_extension method
    extra_f77_compile_args = []
    extra_f90_compile_args = []

    def __init__(self, *args, **kw):
        CCompiler.__init__(self, *args, **kw)
        self.distutils_vars = self.distutils_vars.clone(self._environment_hook)
        self.command_vars = self.command_vars.clone(self._environment_hook)
        self.flag_vars = self.flag_vars.clone(self._environment_hook)
        self.executables = self.executables.copy()
        for e in self._executable_keys:
            if e not in self.executables:
                self.executables[e] = None

        # Some methods depend on .customize() being called first, so
        # this keeps track of whether that's happened yet.
        self._is_customised = False

    def __copy__(self):
        obj = self.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.distutils_vars = obj.distutils_vars.clone(obj._environment_hook)
        obj.command_vars = obj.command_vars.clone(obj._environment_hook)
        obj.flag_vars = obj.flag_vars.clone(obj._environment_hook)
        obj.executables = obj.executables.copy()
        return obj

    def copy(self):
        return self.__copy__()

    # Use properties for the attributes used by CCompiler. Setting them
    # as attributes from the self.executables dictionary is error-prone,
    # so we get them from there each time.
    def _command_property(key):
        def fget(self):
            assert self._is_customised
            return self.executables[key]
        return property(fget=fget)
    version_cmd = _command_property('version_cmd')
    compiler_f77 = _command_property('compiler_f77')
    compiler_f90 = _command_property('compiler_f90')
    compiler_fix = _command_property('compiler_fix')
    linker_so = _command_property('linker_so')
    linker_exe = _command_property('linker_exe')
    archiver = _command_property('archiver')
    ranlib = _command_property('ranlib')

    # Make our terminology consistent.
    def set_executable(self, key, value):
        self.set_command(key, value)

    def set_commands(self, **kw):
        for k, v in kw.items():
            self.set_command(k, v)

    def set_command(self, key, value):
        if not key in self._executable_keys:
            raise ValueError(
                "unknown executable '%s' for class %s" %
                (key, self.__class__.__name__))
        if is_string(value):
            value = split_quoted(value)
        assert value is None or is_sequence_of_strings(value[1:]), (key, value)
        self.executables[key] = value

    ######################################################################
    ## Methods that subclasses may redefine. But don't call these methods!
    ## They are private to FCompiler class and may return unexpected
    ## results if used elsewhere. So, you have been warned..

    def find_executables(self):
        """Go through the self.executables dictionary, and attempt to
        find and assign appropriate executables.

        Executable names are looked for in the environment (environment
        variables, the distutils.cfg, and command line), the 0th-element of
        the command list, and the self.possible_executables list.

        Also, if the 0th element is "<F77>" or "<F90>", the Fortran 77
        or the Fortran 90 compiler executable is used, unless overridden
        by an environment setting.

        Subclasses should call this if overridden.
        """
        assert self._is_customised
        exe_cache = self._exe_cache
        def cached_find_executable(exe):
            if exe in exe_cache:
                return exe_cache[exe]
            fc_exe = find_executable(exe)
            exe_cache[exe] = exe_cache[fc_exe] = fc_exe
            return fc_exe
        def verify_command_form(name, value):
            if value is not None and not is_sequence_of_strings(value):
                raise ValueError(
                    "%s value %r is invalid in class %s" %
                    (name, value, self.__class__.__name__))
        def set_exe(exe_key, f77=None, f90=None):
            cmd = self.executables.get(exe_key, None)
            if not cmd:
                return None
            # Note that we get cmd[0] here if the environment doesn't
            # have anything set
            exe_from_environ = getattr(self.command_vars, exe_key)
            if not exe_from_environ:
                possibles = [f90, f77] + self.possible_executables
            else:
                possibles = [exe_from_environ] + self.possible_executables

            seen = set()
            unique_possibles = []
            for e in possibles:
                if e == '<F77>':
                    e = f77
                elif e == '<F90>':
                    e = f90
                if not e or e in seen:
                    continue
                seen.add(e)
                unique_possibles.append(e)

            for exe in unique_possibles:
                fc_exe = cached_find_executable(exe)
                if fc_exe:
                    cmd[0] = fc_exe
                    return fc_exe
            self.set_command(exe_key, None)
            return None

        ctype = self.compiler_type
        f90 = set_exe('compiler_f90')
        if not f90:
            f77 = set_exe('compiler_f77')
            if f77:
                log.warn('%s: no Fortran 90 compiler found' % ctype)
            else:
                raise CompilerNotFound('%s: f90 nor f77' % ctype)
        else:
            f77 = set_exe('compiler_f77', f90=f90)
            if not f77:
                log.warn('%s: no Fortran 77 compiler found' % ctype)
            set_exe('compiler_fix', f90=f90)

        set_exe('linker_so', f77=f77, f90=f90)
        set_exe('linker_exe', f77=f77, f90=f90)
        set_exe('version_cmd', f77=f77, f90=f90)
        set_exe('archiver')
        set_exe('ranlib')

    def update_executables(self):
        """Called at the beginning of customisation. Subclasses should
        override this if they need to set up the executables dictionary.

        Note that self.find_executables() is run afterwards, so the
        self.executables dictionary values can contain <F77> or <F90> as
        the command, which will be replaced by the found F77 or F90
        compiler.
        """
        pass

    def get_flags(self):
        """List of flags common to all compiler types."""
        return [] + self.pic_flags

    def _get_command_flags(self, key):
        cmd = self.executables.get(key, None)
        if cmd is None:
            return []
        return cmd[1:]

    def get_flags_f77(self):
        """List of Fortran 77 specific flags."""
        return self._get_command_flags('compiler_f77')
    def get_flags_f90(self):
        """List of Fortran 90 specific flags."""
        return self._get_command_flags('compiler_f90')
    def get_flags_free(self):
        """List of Fortran 90 free format specific flags."""
        return []
    def get_flags_fix(self):
        """List of Fortran 90 fixed format specific flags."""
        return self._get_command_flags('compiler_fix')
    def get_flags_linker_so(self):
        """List of linker flags to build a shared library."""
        return self._get_command_flags('linker_so')
    def get_flags_linker_exe(self):
        """List of linker flags to build an executable."""
        return self._get_command_flags('linker_exe')
    def get_flags_ar(self):
        """List of archiver flags. """
        return self._get_command_flags('archiver')
    def get_flags_opt(self):
        """List of architecture independent compiler flags."""
        return []
    def get_flags_arch(self):
        """List of architecture dependent compiler flags."""
        return []
    def get_flags_debug(self):
        """List of compiler flags to compile with debugging information."""
        return []

    get_flags_opt_f77 = get_flags_opt_f90 = get_flags_opt
    get_flags_arch_f77 = get_flags_arch_f90 = get_flags_arch
    get_flags_debug_f77 = get_flags_debug_f90 = get_flags_debug

    def get_libraries(self):
        """List of compiler libraries."""
        return self.libraries[:]
    def get_library_dirs(self):
        """List of compiler library directories."""
        return self.library_dirs[:]

    def get_version(self, force=False, ok_status=[0]):
        assert self._is_customised
        version = CCompiler.get_version(self, force=force, ok_status=ok_status)
        if version is None:
            raise CompilerNotFound()
        return version


    ############################################################

    ## Public methods:

    def customize(self, dist = None):
        """Customize Fortran compiler.

        This method gets Fortran compiler specific information from
        (i) class definition, (ii) environment, (iii) distutils config
        files, and (iv) command line (later overrides earlier).

        This method should be always called after constructing a
        compiler instance. But not in __init__ because Distribution
        instance is needed for (iii) and (iv).
        """
        log.info('customize %s' % (self.__class__.__name__))

        self._is_customised = True

        self.distutils_vars.use_distribution(dist)
        self.command_vars.use_distribution(dist)
        self.flag_vars.use_distribution(dist)

        self.update_executables()

        # find_executables takes care of setting the compiler commands,
        # version_cmd, linker_so, linker_exe, ar, and ranlib
        self.find_executables()

        noopt = self.distutils_vars.get('noopt', False)
        noarch = self.distutils_vars.get('noarch', noopt)
        debug = self.distutils_vars.get('debug', False)

        f77 = self.command_vars.compiler_f77
        f90 = self.command_vars.compiler_f90

        f77flags = []
        f90flags = []
        freeflags = []
        fixflags = []

        if f77:
            f77 = _shell_utils.NativeParser.split(f77)
            f77flags = self.flag_vars.f77
        if f90:
            f90 = _shell_utils.NativeParser.split(f90)
            f90flags = self.flag_vars.f90
            freeflags = self.flag_vars.free
        # XXX Assuming that free format is default for f90 compiler.
        fix = self.command_vars.compiler_fix
        # NOTE: this and similar examples are probably just
        # excluding --coverage flag when F90 = gfortran --coverage
        # instead of putting that flag somewhere more appropriate
        # this and similar examples where a Fortran compiler
        # environment variable has been customized by CI or a user
        # should perhaps eventually be more thoroughly tested and more
        # robustly handled
        if fix:
            fix = _shell_utils.NativeParser.split(fix)
            fixflags = self.flag_vars.fix + f90flags

        oflags, aflags, dflags = [], [], []
        # examine get_flags_<tag>_<compiler> for extra flags
        # only add them if the method is different from get_flags_<tag>
        def get_flags(tag, flags):
            # note that self.flag_vars.<tag> calls self.get_flags_<tag>()
            flags.extend(getattr(self.flag_vars, tag))
            this_get = getattr(self, 'get_flags_' + tag)
            for name, c, flagvar in [('f77', f77, f77flags),
                                     ('f90', f90, f90flags),
                                     ('f90', fix, fixflags)]:
                t = '%s_%s' % (tag, name)
                if c and this_get is not getattr(self, 'get_flags_' + t):
                    flagvar.extend(getattr(self.flag_vars, t))
        if not noopt:
            get_flags('opt', oflags)
            if not noarch:
                get_flags('arch', aflags)
        if debug:
            get_flags('debug', dflags)

        fflags = self.flag_vars.flags + dflags + oflags + aflags

        if f77:
            self.set_commands(compiler_f77=f77+f77flags+fflags)
        if f90:
            self.set_commands(compiler_f90=f90+freeflags+f90flags+fflags)
        if fix:
            self.set_commands(compiler_fix=fix+fixflags+fflags)


        #XXX: Do we need LDSHARED->SOSHARED, LDFLAGS->SOFLAGS
        linker_so = self.linker_so
        if linker_so:
            linker_so_flags = self.flag_vars.linker_so
            if sys.platform.startswith('aix'):
                python_lib = get_python_lib(standard_lib=1)
                ld_so_aix = os.path.join(python_lib, 'config', 'ld_so_aix')
                python_exp = os.path.join(python_lib, 'config', 'python.exp')
                linker_so = [ld_so_aix] + linker_so + ['-bI:'+python_exp]
            if sys.platform.startswith('os400'):
                from distutils.sysconfig import get_config_var
                python_config = get_config_var('LIBPL')
                ld_so_aix = os.path.join(python_config, 'ld_so_aix')
                python_exp = os.path.join(python_config, 'python.exp')
                linker_so = [ld_so_aix] + linker_so + ['-bI:'+python_exp]
            self.set_commands(linker_so=linker_so+linker_so_flags)

        linker_exe = self.linker_exe
        if linker_exe:
            linker_exe_flags = self.flag_vars.linker_exe
            self.set_commands(linker_exe=linker_exe+linker_exe_flags)

        ar = self.command_vars.archiver
        if ar:
            arflags = self.flag_vars.ar
            self.set_commands(archiver=[ar]+arflags)

        self.set_library_dirs(self.get_library_dirs())
        self.set_libraries(self.get_libraries())

    def dump_properties(self):
        """Print out the attributes of a compiler instance."""
        props = []
        for key in list(self.executables.keys()) + \
                ['version', 'libraries', 'library_dirs',
                 'object_switch', 'compile_switch']:
            if hasattr(self, key):
                v = getattr(self, key)
                props.append((key, None, '= '+repr(v)))
        props.sort()

        pretty_printer = FancyGetopt(props)
        for l in pretty_printer.generate_help("%s instance properties:" \
                                              % (self.__class__.__name__)):
            if l[:4]=='  --':
                l = '  ' + l[4:]
            print(l)

    ###################

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        """Compile 'src' to product 'obj'."""
        src_flags = {}
        if Path(src).suffix.lower() in FORTRAN_COMMON_FIXED_EXTENSIONS \
           and not has_f90_header(src):
            flavor = ':f77'
            compiler = self.compiler_f77
            src_flags = get_f77flags(src)
            extra_compile_args = self.extra_f77_compile_args or []
        elif is_free_format(src):
            flavor = ':f90'
            compiler = self.compiler_f90
            if compiler is None:
                raise DistutilsExecError('f90 not supported by %s needed for %s'\
                      % (self.__class__.__name__, src))
            extra_compile_args = self.extra_f90_compile_args or []
        else:
            flavor = ':fix'
            compiler = self.compiler_fix
            if compiler is None:
                raise DistutilsExecError('f90 (fixed) not supported by %s needed for %s'\
                      % (self.__class__.__name__, src))
            extra_compile_args = self.extra_f90_compile_args or []
        if self.object_switch[-1]==' ':
            o_args = [self.object_switch.strip(), obj]
        else:
            o_args = [self.object_switch.strip()+obj]

        assert self.compile_switch.strip()
        s_args = [self.compile_switch, src]

        if extra_compile_args:
            log.info('extra %s options: %r' \
                     % (flavor[1:], ' '.join(extra_compile_args)))

        extra_flags = src_flags.get(self.compiler_type, [])
        if extra_flags:
            log.info('using compile options from source: %r' \
                     % ' '.join(extra_flags))

        command = compiler + cc_args + extra_flags + s_args + o_args \
                  + extra_postargs + extra_compile_args

        display = '%s: %s' % (os.path.basename(compiler[0]) + flavor,
                              src)
        try:
            self.spawn(command, display=display)
        except DistutilsExecError as e:
            msg = str(e)
            raise CompileError(msg) from None

    def module_options(self, module_dirs, module_build_dir):
        options = []
        if self.module_dir_switch is not None:
            if self.module_dir_switch[-1]==' ':
                options.extend([self.module_dir_switch.strip(), module_build_dir])
            else:
                options.append(self.module_dir_switch.strip()+module_build_dir)
        else:
            print('XXX: module_build_dir=%r option ignored' % (module_build_dir))
            print('XXX: Fix module_dir_switch for ', self.__class__.__name__)
        if self.module_include_switch is not None:
            for d in [module_build_dir]+module_dirs:
                options.append('%s%s' % (self.module_include_switch, d))
        else:
            print('XXX: module_dirs=%r option ignored' % (module_dirs))
            print('XXX: Fix module_include_switch for ', self.__class__.__name__)
        return options

    def library_option(self, lib):
        return "-l" + lib
    def library_dir_option(self, dir):
        return "-L" + dir

    def link(self, target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        objects, output_dir = self._fix_object_args(objects, output_dir)
        libraries, library_dirs, runtime_library_dirs = \
            self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)

        lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs,
                                   libraries)
        if is_string(output_dir):
            output_filename = os.path.join(output_dir, output_filename)
        elif output_dir is not None:
            raise TypeError("'output_dir' must be a string or None")

        if self._need_link(objects, output_filename):
            if self.library_switch[-1]==' ':
                o_args = [self.library_switch.strip(), output_filename]
            else:
                o_args = [self.library_switch.strip()+output_filename]

            if is_string(self.objects):
                ld_args = objects + [self.objects]
            else:
                ld_args = objects + self.objects
            ld_args = ld_args + lib_opts + o_args
            if debug:
                ld_args[:0] = ['-g']
            if extra_preargs:
                ld_args[:0] = extra_preargs
            if extra_postargs:
                ld_args.extend(extra_postargs)
            self.mkpath(os.path.dirname(output_filename))
            if target_desc == CCompiler.EXECUTABLE:
                linker = self.linker_exe[:]
            else:
                linker = self.linker_so[:]
            command = linker + ld_args
            try:
                self.spawn(command)
            except DistutilsExecError as e:
                msg = str(e)
                raise LinkError(msg) from None
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    def _environment_hook(self, name, hook_name):
        if hook_name is None:
            return None
        if is_string(hook_name):
            if hook_name.startswith('self.'):
                hook_name = hook_name[5:]
                hook = getattr(self, hook_name)
                return hook()
            elif hook_name.startswith('exe.'):
                hook_name = hook_name[4:]
                var = self.executables[hook_name]
                if var:
                    return var[0]
                else:
                    return None
            elif hook_name.startswith('flags.'):
                hook_name = hook_name[6:]
                hook = getattr(self, 'get_flags_' + hook_name)
                return hook()
        else:
            return hook_name()

    def can_ccompiler_link(self, ccompiler):
        """
        Check if the given C compiler can link objects produced by
        this compiler.
        """
        return True

    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.

        Parameters
        ----------
        objects : list
            List of object files to include.
        output_dir : str
            Output directory to place generated object files.
        extra_dll_dir : str
            Output directory to place extra DLL files that need to be
            included on Windows.

        Returns
        -------
        converted_objects : list of str
             List of converted object files.
             Note that the number of output files is not necessarily
             the same as inputs.

        """
        raise NotImplementedError()

    ## class FCompiler

_default_compilers = (
    # sys.platform mappings
    ('win32', ('gnu', 'intelv', 'absoft', 'compaqv', 'intelev', 'gnu95', 'g95',
               'intelvem', 'intelem', 'flang')),
    ('cygwin.*', ('gnu', 'intelv', 'absoft', 'compaqv', 'intelev', 'gnu95', 'g95')),
    ('linux.*', ('arm', 'gnu95', 'intel', 'lahey', 'pg', 'nv', 'absoft', 'nag',
                 'vast', 'compaq', 'intele', 'intelem', 'gnu', 'g95', 
                 'pathf95', 'nagfor', 'fujitsu')),
    ('darwin.*', ('gnu95', 'nag', 'nagfor', 'absoft', 'ibm', 'intel', 'gnu',
                 'g95', 'pg')),
    ('sunos.*', ('sun', 'gnu', 'gnu95', 'g95')),
    ('irix.*', ('mips', 'gnu', 'gnu95',)),
    ('aix.*', ('ibm', 'gnu', 'gnu95',)),
    # os.name mappings
    ('posix', ('gnu', 'gnu95',)),
    ('nt', ('gnu', 'gnu95',)),
    ('mac', ('gnu95', 'gnu', 'pg')),
    )

fcompiler_class = None
fcompiler_aliases = None

def load_all_fcompiler_classes():
    """Cache all the FCompiler classes found in modules in the
    numpy.distutils.fcompiler package.
    """
    from glob import glob
    global fcompiler_class, fcompiler_aliases
    if fcompiler_class is not None:
        return
    pys = os.path.join(os.path.dirname(__file__), '*.py')
    fcompiler_class = {}
    fcompiler_aliases = {}
    for fname in glob(pys):
        module_name, ext = os.path.splitext(os.path.basename(fname))
        module_name = 'numpy.distutils.fcompiler.' + module_name
        __import__ (module_name)
        module = sys.modules[module_name]
        if hasattr(module, 'compilers'):
            for cname in module.compilers:
                klass = getattr(module, cname)
                desc = (klass.compiler_type, klass, klass.description)
                fcompiler_class[klass.compiler_type] = desc
                for alias in klass.compiler_aliases:
                    if alias in fcompiler_aliases:
                        raise ValueError("alias %r defined for both %s and %s"
                                         % (alias, klass.__name__,
                                            fcompiler_aliases[alias][1].__name__))
                    fcompiler_aliases[alias] = desc

def _find_existing_fcompiler(compiler_types,
                             osname=None, platform=None,
                             requiref90=False,
                             c_compiler=None):
    from numpy.distutils.core import get_distribution
    dist = get_distribution(always=True)
    for compiler_type in compiler_types:
        v = None
        try:
            c = new_fcompiler(plat=platform, compiler=compiler_type,
                              c_compiler=c_compiler)
            c.customize(dist)
            v = c.get_version()
            if requiref90 and c.compiler_f90 is None:
                v = None
                new_compiler = c.suggested_f90_compiler
                if new_compiler:
                    log.warn('Trying %r compiler as suggested by %r '
                             'compiler for f90 support.' % (compiler_type,
                                                            new_compiler))
                    c = new_fcompiler(plat=platform, compiler=new_compiler,
                                      c_compiler=c_compiler)
                    c.customize(dist)
                    v = c.get_version()
                    if v is not None:
                        compiler_type = new_compiler
            if requiref90 and c.compiler_f90 is None:
                raise ValueError('%s does not support compiling f90 codes, '
                                 'skipping.' % (c.__class__.__name__))
        except DistutilsModuleError:
            log.debug("_find_existing_fcompiler: compiler_type='%s' raised DistutilsModuleError", compiler_type)
        except CompilerNotFound:
            log.debug("_find_existing_fcompiler: compiler_type='%s' not found", compiler_type)
        if v is not None:
            return compiler_type
    return None

def available_fcompilers_for_platform(osname=None, platform=None):
    if osname is None:
        osname = os.name
    if platform is None:
        platform = sys.platform
    matching_compiler_types = []
    for pattern, compiler_type in _default_compilers:
        if re.match(pattern, platform) or re.match(pattern, osname):
            for ct in compiler_type:
                if ct not in matching_compiler_types:
                    matching_compiler_types.append(ct)
    if not matching_compiler_types:
        matching_compiler_types.append('gnu')
    return matching_compiler_types

def get_default_fcompiler(osname=None, platform=None, requiref90=False,
                          c_compiler=None):
    """Determine the default Fortran compiler to use for the given
    platform."""
    matching_compiler_types = available_fcompilers_for_platform(osname,
                                                                platform)
    log.info("get_default_fcompiler: matching types: '%s'",
             matching_compiler_types)
    compiler_type =  _find_existing_fcompiler(matching_compiler_types,
                                              osname=osname,
                                              platform=platform,
                                              requiref90=requiref90,
                                              c_compiler=c_compiler)
    return compiler_type

# Flag to avoid rechecking for Fortran compiler every time
failed_fcompilers = set()

def new_fcompiler(plat=None,
                  compiler=None,
                  verbose=0,
                  dry_run=0,
                  force=0,
                  requiref90=False,
                  c_compiler = None):
    """Generate an instance of some FCompiler subclass for the supplied
    platform/compiler combination.
    """
    global failed_fcompilers
    fcompiler_key = (plat, compiler)
    if fcompiler_key in failed_fcompilers:
        return None

    load_all_fcompiler_classes()
    if plat is None:
        plat = os.name
    if compiler is None:
        compiler = get_default_fcompiler(plat, requiref90=requiref90,
                                         c_compiler=c_compiler)
    if compiler in fcompiler_class:
        module_name, klass, long_description = fcompiler_class[compiler]
    elif compiler in fcompiler_aliases:
        module_name, klass, long_description = fcompiler_aliases[compiler]
    else:
        msg = "don't know how to compile Fortran code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler." % compiler
            msg = msg + " Supported compilers are: %s)" \
                  % (','.join(fcompiler_class.keys()))
        log.warn(msg)
        failed_fcompilers.add(fcompiler_key)
        return None

    compiler = klass(verbose=verbose, dry_run=dry_run, force=force)
    compiler.c_compiler = c_compiler
    return compiler

def show_fcompilers(dist=None):
    """Print list of available compilers (used by the "--help-fcompiler"
    option to "config_fc").
    """
    if dist is None:
        from distutils.dist import Distribution
        from numpy.distutils.command.config_compiler import config_fc
        dist = Distribution()
        dist.script_name = os.path.basename(sys.argv[0])
        dist.script_args = ['config_fc'] + sys.argv[1:]
        try:
            dist.script_args.remove('--help-fcompiler')
        except ValueError:
            pass
        dist.cmdclass['config_fc'] = config_fc
        dist.parse_config_files()
        dist.parse_command_line()
    compilers = []
    compilers_na = []
    compilers_ni = []
    if not fcompiler_class:
        load_all_fcompiler_classes()
    platform_compilers = available_fcompilers_for_platform()
    for compiler in platform_compilers:
        v = None
        log.set_verbosity(-2)
        try:
            c = new_fcompiler(compiler=compiler, verbose=dist.verbose)
            c.customize(dist)
            v = c.get_version()
        except (DistutilsModuleError, CompilerNotFound) as e:
            log.debug("show_fcompilers: %s not found" % (compiler,))
            log.debug(repr(e))

        if v is None:
            compilers_na.append(("fcompiler="+compiler, None,
                              fcompiler_class[compiler][2]))
        else:
            c.dump_properties()
            compilers.append(("fcompiler="+compiler, None,
                              fcompiler_class[compiler][2] + ' (%s)' % v))

    compilers_ni = list(set(fcompiler_class.keys()) - set(platform_compilers))
    compilers_ni = [("fcompiler="+fc, None, fcompiler_class[fc][2])
                    for fc in compilers_ni]

    compilers.sort()
    compilers_na.sort()
    compilers_ni.sort()
    pretty_printer = FancyGetopt(compilers)
    pretty_printer.print_help("Fortran compilers found:")
    pretty_printer = FancyGetopt(compilers_na)
    pretty_printer.print_help("Compilers available for this "
                              "platform, but not found:")
    if compilers_ni:
        pretty_printer = FancyGetopt(compilers_ni)
        pretty_printer.print_help("Compilers not available on this platform:")
    print("For compiler details, run 'config_fc --verbose' setup command.")


def dummy_fortran_file():
    fo, name = make_temp_file(suffix='.f')
    fo.write("      subroutine dummy()\n      end\n")
    fo.close()
    return name[:-2]


_has_f_header = re.compile(r'-\*-\s*fortran\s*-\*-', re.I).search
_has_f90_header = re.compile(r'-\*-\s*f90\s*-\*-', re.I).search
_has_fix_header = re.compile(r'-\*-\s*fix\s*-\*-', re.I).search
_free_f90_start = re.compile(r'[^c*!]\s*[^\s\d\t]', re.I).match

def is_free_format(file):
    """Check if file is in free format Fortran."""
    # f90 allows both fixed and free format, assuming fixed unless
    # signs of free format are detected.
    result = 0
    with open(file, encoding='latin1') as f:
        line = f.readline()
        n = 10000 # the number of non-comment lines to scan for hints
        if _has_f_header(line) or _has_fix_header(line):
            n = 0
        elif _has_f90_header(line):
            n = 0
            result = 1
        while n>0 and line:
            line = line.rstrip()
            if line and line[0]!='!':
                n -= 1
                if (line[0]!='\t' and _free_f90_start(line[:5])) or line[-1:]=='&':
                    result = 1
                    break
            line = f.readline()
    return result

def has_f90_header(src):
    with open(src, encoding='latin1') as f:
        line = f.readline()
    return _has_f90_header(line) or _has_fix_header(line)

_f77flags_re = re.compile(r'(c|)f77flags\s*\(\s*(?P<fcname>\w+)\s*\)\s*=\s*(?P<fflags>.*)', re.I)
def get_f77flags(src):
    """
    Search the first 20 lines of fortran 77 code for line pattern
      `CF77FLAGS(<fcompiler type>)=<f77 flags>`
    Return a dictionary {<fcompiler type>:<f77 flags>}.
    """
    flags = {}
    with open(src, encoding='latin1') as f:
        i = 0
        for line in f:
            i += 1
            if i>20: break
            m = _f77flags_re.match(line)
            if not m: continue
            fcname = m.group('fcname').strip()
            fflags = m.group('fflags').strip()
            flags[fcname] = split_quoted(fflags)
    return flags

# TODO: implement get_f90flags and use it in _compile similarly to get_f77flags

if __name__ == '__main__':
    show_fcompilers()
<|endoftext|>import os

from numpy.distutils.fcompiler.gnu import GnuFCompiler

compilers = ['VastFCompiler']

class VastFCompiler(GnuFCompiler):
    compiler_type = 'vast'
    compiler_aliases = ()
    description = 'Pacific-Sierra Research Fortran 90 Compiler'
    version_pattern = (r'\s*Pacific-Sierra Research vf90 '
                       r'(Personal|Professional)\s+(?P<version>[^\s]*)')

    # VAST f90 does not support -o with -c. So, object files are created
    # to the current directory and then moved to build directory
    object_switch = ' && function _mvfile { mv -v `basename $1` $1 ; } && _mvfile '

    executables = {
        'version_cmd'  : ["vf90", "-v"],
        'compiler_f77' : ["g77"],
        'compiler_fix' : ["f90", "-Wv,-ya"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["<F90>"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    module_dir_switch = None  #XXX Fix me
    module_include_switch = None #XXX Fix me

    def find_executables(self):
        pass

    def get_version_cmd(self):
        f90 = self.compiler_f90[0]
        d, b = os.path.split(f90)
        vf90 = os.path.join(d, 'v'+b)
        return vf90

    def get_flags_arch(self):
        vast_version = self.get_version()
        gnu = GnuFCompiler()
        gnu.customize(None)
        self.version = gnu.get_version()
        opt = GnuFCompiler.get_flags_arch(self)
        self.version = vast_version
        return opt

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='vast').get_version())
<|endoftext|>from numpy.distutils.fcompiler import FCompiler

compilers = ['PathScaleFCompiler']

class PathScaleFCompiler(FCompiler):

    compiler_type = 'pathf95'
    description = 'PathScale Fortran Compiler'
    version_pattern =  r'PathScale\(TM\) Compiler Suite: Version (?P<version>[\d.]+)'

    executables = {
        'version_cmd'  : ["pathf95", "-version"],
        'compiler_f77' : ["pathf95", "-fixedform"],
        'compiler_fix' : ["pathf95", "-fixedform"],
        'compiler_f90' : ["pathf95"],
        'linker_so'    : ["pathf95", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
    }
    pic_flags = ['-fPIC']
    module_dir_switch = '-module ' # Don't remove ending space!
    module_include_switch = '-I'

    def get_flags_opt(self):
        return ['-O3']
    def get_flags_debug(self):
        return ['-g']

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='pathf95').get_version())
<|endoftext|>import os

from numpy.distutils.fcompiler import FCompiler

compilers = ['LaheyFCompiler']

class LaheyFCompiler(FCompiler):

    compiler_type = 'lahey'
    description = 'Lahey/Fujitsu Fortran 95 Compiler'
    version_pattern =  r'Lahey/Fujitsu Fortran 95 Compiler Release (?P<version>[^\s*]*)'

    executables = {
        'version_cmd'  : ["<F90>", "--version"],
        'compiler_f77' : ["lf95", "--fix"],
        'compiler_fix' : ["lf95", "--fix"],
        'compiler_f90' : ["lf95"],
        'linker_so'    : ["lf95", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    module_dir_switch = None  #XXX Fix me
    module_include_switch = None #XXX Fix me

    def get_flags_opt(self):
        return ['-O']
    def get_flags_debug(self):
        return ['-g', '--chk', '--chkglobal']
    def get_library_dirs(self):
        opt = []
        d = os.environ.get('LAHEY')
        if d:
            opt.append(os.path.join(d, 'lib'))
        return opt
    def get_libraries(self):
        opt = []
        opt.extend(['fj9f6', 'fj9i6', 'fj9ipp', 'fj9e6'])
        return opt

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='lahey').get_version())
<|endoftext|>from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler

compilers = ['SunFCompiler']

class SunFCompiler(FCompiler):

    compiler_type = 'sun'
    description = 'Sun or Forte Fortran 95 Compiler'
    # ex:
    # f90: Sun WorkShop 6 update 2 Fortran 95 6.2 Patch 111690-10 2003/08/28
    version_match = simple_version_match(
                      start=r'f9[05]: (Sun|Forte|WorkShop).*Fortran 95')

    executables = {
        'version_cmd'  : ["<F90>", "-V"],
        'compiler_f77' : ["f90"],
        'compiler_fix' : ["f90", "-fixed"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["<F90>", "-Bdynamic", "-G"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    module_dir_switch = '-moddir='
    module_include_switch = '-M'
    pic_flags = ['-xcode=pic32']

    def get_flags_f77(self):
        ret = ["-ftrap=%none"]
        if (self.get_version() or '') >= '7':
            ret.append("-f77")
        else:
            ret.append("-fixed")
        return ret
    def get_opt(self):
        return ['-fast', '-dalign']
    def get_arch(self):
        return ['-xtarget=generic']
    def get_libraries(self):
        opt = []
        opt.extend(['fsu', 'sunmath', 'mvec'])
        return opt

    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='sun').get_version())
<|endoftext|># http://developer.intel.com/software/products/compilers/flin/
import sys

from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file

compilers = ['IntelFCompiler', 'IntelVisualFCompiler',
             'IntelItaniumFCompiler', 'IntelItaniumVisualFCompiler',
             'IntelEM64VisualFCompiler', 'IntelEM64TFCompiler']


def intel_version_match(type):
    # Match against the important stuff in the version string
    return simple_version_match(start=r'Intel.*?Fortran.*?(?:%s).*?Version' % (type,))


class BaseIntelFCompiler(FCompiler):
    def update_executables(self):
        f = dummy_fortran_file()
        self.executables['version_cmd'] = ['<F77>', '-FI', '-V', '-c',
                                           f + '.f', '-o', f + '.o']

    def runtime_library_dir_option(self, dir):
        # TODO: could use -Xlinker here, if it's supported
        assert "," not in dir

        return '-Wl,-rpath=%s' % dir


class IntelFCompiler(BaseIntelFCompiler):

    compiler_type = 'intel'
    compiler_aliases = ('ifort',)
    description = 'Intel Fortran Compiler for 32-bit apps'
    version_match = intel_version_match('32-bit|IA-32')

    possible_executables = ['ifort', 'ifc']

    executables = {
        'version_cmd'  : None,          # set by update_executables
        'compiler_f77' : [None, "-72", "-w90", "-w95"],
        'compiler_f90' : [None],
        'compiler_fix' : [None, "-FI"],
        'linker_so'    : ["<F90>", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    pic_flags = ['-fPIC']
    module_dir_switch = '-module '  # Don't remove ending space!
    module_include_switch = '-I'

    def get_flags_free(self):
        return ['-FR']

    def get_flags(self):
        return ['-fPIC']

    def get_flags_opt(self):  # Scipy test failures with -O2
        v = self.get_version()
        mpopt = 'openmp' if v and v < '15' else 'qopenmp'
        return ['-fp-model', 'strict', '-O1',
                '-assume', 'minus0', '-{}'.format(mpopt)]

    def get_flags_arch(self):
        return []

    def get_flags_linker_so(self):
        opt = FCompiler.get_flags_linker_so(self)
        v = self.get_version()
        if v and v >= '8.0':
            opt.append('-nofor_main')
        if sys.platform == 'darwin':
            # Here, it's -dynamiclib
            try:
                idx = opt.index('-shared')
                opt.remove('-shared')
            except ValueError:
                idx = 0
            opt[idx:idx] = ['-dynamiclib', '-Wl,-undefined,dynamic_lookup']
        return opt


class IntelItaniumFCompiler(IntelFCompiler):
    compiler_type = 'intele'
    compiler_aliases = ()
    description = 'Intel Fortran Compiler for Itanium apps'

    version_match = intel_version_match('Itanium|IA-64')

    possible_executables = ['ifort', 'efort', 'efc']

    executables = {
        'version_cmd'  : None,
        'compiler_f77' : [None, "-FI", "-w90", "-w95"],
        'compiler_fix' : [None, "-FI"],
        'compiler_f90' : [None],
        'linker_so'    : ['<F90>', "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }


class IntelEM64TFCompiler(IntelFCompiler):
    compiler_type = 'intelem'
    compiler_aliases = ()
    description = 'Intel Fortran Compiler for 64-bit apps'

    version_match = intel_version_match('EM64T-based|Intel\\(R\\) 64|64|IA-64|64-bit')

    possible_executables = ['ifort', 'efort', 'efc']

    executables = {
        'version_cmd'  : None,
        'compiler_f77' : [None, "-FI"],
        'compiler_fix' : [None, "-FI"],
        'compiler_f90' : [None],
        'linker_so'    : ['<F90>', "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

# Is there no difference in the version string between the above compilers
# and the Visual compilers?


class IntelVisualFCompiler(BaseIntelFCompiler):
    compiler_type = 'intelv'
    description = 'Intel Visual Fortran Compiler for 32-bit apps'
    version_match = intel_version_match('32-bit|IA-32')

    def update_executables(self):
        f = dummy_fortran_file()
        self.executables['version_cmd'] = ['<F77>', '/FI', '/c',
                                           f + '.f', '/o', f + '.o']

    ar_exe = 'lib.exe'
    possible_executables = ['ifort', 'ifl']

    executables = {
        'version_cmd'  : None,
        'compiler_f77' : [None],
        'compiler_fix' : [None],
        'compiler_f90' : [None],
        'linker_so'    : [None],
        'archiver'     : [ar_exe, "/verbose", "/OUT:"],
        'ranlib'       : None
        }

    compile_switch = '/c '
    object_switch = '/Fo'     # No space after /Fo!
    library_switch = '/OUT:'  # No space after /OUT:!
    module_dir_switch = '/module:'  # No space after /module:
    module_include_switch = '/I'

    def get_flags(self):
        opt = ['/nologo', '/MD', '/nbs', '/names:lowercase', 
               '/assume:underscore', '/fpp']
        return opt

    def get_flags_free(self):
        return []

    def get_flags_debug(self):
        return ['/4Yb', '/d2']

    def get_flags_opt(self):
        return ['/O1', '/assume:minus0']  # Scipy test failures with /O2

    def get_flags_arch(self):
        return ["/arch:IA32", "/QaxSSE3"]

    def runtime_library_dir_option(self, dir):
        raise NotImplementedError


class IntelItaniumVisualFCompiler(IntelVisualFCompiler):
    compiler_type = 'intelev'
    description = 'Intel Visual Fortran Compiler for Itanium apps'

    version_match = intel_version_match('Itanium')

    possible_executables = ['efl']  # XXX this is a wild guess
    ar_exe = IntelVisualFCompiler.ar_exe

    executables = {
        'version_cmd'  : None,
        'compiler_f77' : [None, "-FI", "-w90", "-w95"],
        'compiler_fix' : [None, "-FI", "-4L72", "-w"],
        'compiler_f90' : [None],
        'linker_so'    : ['<F90>', "-shared"],
        'archiver'     : [ar_exe, "/verbose", "/OUT:"],
        'ranlib'       : None
        }


class IntelEM64VisualFCompiler(IntelVisualFCompiler):
    compiler_type = 'intelvem'
    description = 'Intel Visual Fortran Compiler for 64-bit apps'

    version_match = simple_version_match(start=r'Intel\(R\).*?64,')

    def get_flags_arch(self):
        return []


if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='intel').get_version())
<|endoftext|>import os
import re
import sys
import subprocess

from numpy.distutils.fcompiler import FCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils.misc_util import make_temp_file
from distutils import log

compilers = ['IBMFCompiler']

class IBMFCompiler(FCompiler):
    compiler_type = 'ibm'
    description = 'IBM XL Fortran Compiler'
    version_pattern =  r'(xlf\(1\)\s*|)IBM XL Fortran ((Advanced Edition |)Version |Enterprise Edition V|for AIX, V)(?P<version>[^\s*]*)'
    #IBM XL Fortran Enterprise Edition V10.1 for AIX \nVersion: 10.01.0000.0004

    executables = {
        'version_cmd'  : ["<F77>", "-qversion"],
        'compiler_f77' : ["xlf"],
        'compiler_fix' : ["xlf90", "-qfixed"],
        'compiler_f90' : ["xlf90"],
        'linker_so'    : ["xlf95"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_version(self,*args,**kwds):
        version = FCompiler.get_version(self,*args,**kwds)

        if version is None and sys.platform.startswith('aix'):
            # use lslpp to find out xlf version
            lslpp = find_executable('lslpp')
            xlf = find_executable('xlf')
            if os.path.exists(xlf) and os.path.exists(lslpp):
                try:
                    o = subprocess.check_output([lslpp, '-Lc', 'xlfcmp'])
                except (OSError, subprocess.CalledProcessError):
                    pass
                else:
                    m = re.search(r'xlfcmp:(?P<version>\d+([.]\d+)+)', o)
                    if m: version = m.group('version')

        xlf_dir = '/etc/opt/ibmcmp/xlf'
        if version is None and os.path.isdir(xlf_dir):
            # linux:
            # If the output of xlf does not contain version info
            # (that's the case with xlf 8.1, for instance) then
            # let's try another method:
            l = sorted(os.listdir(xlf_dir))
            l.reverse()
            l = [d for d in l if os.path.isfile(os.path.join(xlf_dir, d, 'xlf.cfg'))]
            if l:
                from distutils.version import LooseVersion
                self.version = version = LooseVersion(l[0])
        return version

    def get_flags(self):
        return ['-qextname']

    def get_flags_debug(self):
        return ['-g']

    def get_flags_linker_so(self):
        opt = []
        if sys.platform=='darwin':
            opt.append('-Wl,-bundle,-flat_namespace,-undefined,suppress')
        else:
            opt.append('-bshared')
        version = self.get_version(ok_status=[0, 40])
        if version is not None:
            if sys.platform.startswith('aix'):
                xlf_cfg = '/etc/xlf.cfg'
            else:
                xlf_cfg = '/etc/opt/ibmcmp/xlf/%s/xlf.cfg' % version
            fo, new_cfg = make_temp_file(suffix='_xlf.cfg')
            log.info('Creating '+new_cfg)
            with open(xlf_cfg) as fi:
                crt1_match = re.compile(r'\s*crt\s*=\s*(?P<path>.*)/crt1.o').match
                for line in fi:
                    m = crt1_match(line)
                    if m:
                        fo.write('crt = %s/bundle1.o\n' % (m.group('path')))
                    else:
                        fo.write(line)
            fo.close()
            opt.append('-F'+new_cfg)
        return opt

    def get_flags_opt(self):
        return ['-O3']

if __name__ == '__main__':
    from numpy.distutils import customized_fcompiler
    log.set_verbosity(2)
    print(customized_fcompiler(compiler='ibm').get_version())
<|endoftext|>"""
fujitsu

Supports Fujitsu compiler function.
This compiler is developed by Fujitsu and is used in A64FX on Fugaku.
"""
from numpy.distutils.fcompiler import FCompiler

compilers = ['FujitsuFCompiler']

class FujitsuFCompiler(FCompiler):
    compiler_type = 'fujitsu'
    description = 'Fujitsu Fortran Compiler'

    possible_executables = ['frt']
    version_pattern = r'frt \(FRT\) (?P<version>[a-z\d.]+)'
    # $ frt --version
    # frt (FRT) x.x.x yyyymmdd

    executables = {
        'version_cmd'  : ["<F77>", "--version"],
        'compiler_f77' : ["frt", "-Fixed"],
        'compiler_fix' : ["frt", "-Fixed"],
        'compiler_f90' : ["frt"],
        'linker_so'    : ["frt", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    pic_flags = ['-KPIC']
    module_dir_switch = '-M'
    module_include_switch = '-I'

    def get_flags_opt(self):
        return ['-O3']
    def get_flags_debug(self):
        return ['-g']
    def runtime_library_dir_option(self, dir):
        return f'-Wl,-rpath={dir}'
    def get_libraries(self):
        return ['fj90f', 'fj90i', 'fjsrcinfo']

if __name__ == '__main__':
    from distutils import log
    from numpy.distutils import customized_fcompiler
    log.set_verbosity(2)
    print(customized_fcompiler('fujitsu').get_version())
<|endoftext|>"""
Compatibility module.

This module contains duplicated code from Python itself or 3rd party
extensions, which may be included for the following reasons:

  * compatibility
  * we may only need a small subset of the copied library/module

This module is deprecated since 1.26.0 and will be removed in future versions.

"""

import warnings

from .._utils import _inspect
from .._utils._inspect import getargspec, formatargspec
from . import py3k
from .py3k import *

warnings.warn(
    "`np.compat`, which was used during the Python 2 to 3 transition,"
    " is deprecated since 1.26.0, and will be removed",
    DeprecationWarning, stacklevel=2
)

__all__ = []
__all__.extend(_inspect.__all__)
__all__.extend(py3k.__all__)
<|endoftext|>"""
Python 3.X compatibility tools.

While this file was originally intended for Python 2 -> 3 transition,
it is now used to create a compatibility layer between different
minor versions of Python 3.

While the active version of numpy may not support a given version of python, we
allow downstream libraries to continue to use these shims for forward
compatibility with numpy while they transition their code to newer versions of
Python.
"""
__all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar',
           'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested',
           'asstr', 'open_latin1', 'long', 'basestring', 'sixu',
           'integer_types', 'is_pathlib_path', 'npy_load_module', 'Path',
           'pickle', 'contextlib_nullcontext', 'os_fspath', 'os_PathLike']

import sys
import os
from pathlib import Path
import io
try:
    import pickle5 as pickle
except ImportError:
    import pickle

long = int
integer_types = (int,)
basestring = str
unicode = str
bytes = bytes

def asunicode(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)

def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')

def asstr(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)

def isfileobj(f):
    if not isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter)):
        return False
    try:
        # BufferedReader/Writer may raise OSError when
        # fetching `fileno()` (e.g. when wrapping BytesIO).
        f.fileno()
        return True
    except OSError:
        return False

def open_latin1(filename, mode='r'):
    return open(filename, mode=mode, encoding='iso-8859-1')

def sixu(s):
    return s

strchar = 'U'

def getexception():
    return sys.exc_info()[1]

def asbytes_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
        return [asbytes_nested(y) for y in x]
    else:
        return asbytes(x)

def asunicode_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
        return [asunicode_nested(y) for y in x]
    else:
        return asunicode(x)

def is_pathlib_path(obj):
    """
    Check whether obj is a `pathlib.Path` object.

    Prefer using ``isinstance(obj, os.PathLike)`` instead of this function.
    """
    return isinstance(obj, Path)

# from Python 3.7
class contextlib_nullcontext:
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True

    .. note::
        Prefer using `contextlib.nullcontext` instead of this context manager.
    """

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def npy_load_module(name, fn, info=None):
    """
    Load a module. Uses ``load_module`` which will be deprecated in python
    3.12. An alternative that uses ``exec_module`` is in
    numpy.distutils.misc_util.exec_mod_from_location

    Parameters
    ----------
    name : str
        Full module name.
    fn : str
        Path to module file.
    info : tuple, optional
        Only here for backward compatibility with Python 2.*.

    Returns
    -------
    mod : module

    """
    # Explicitly lazy import this to avoid paying the cost
    # of importing importlib at startup
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(name, fn).load_module()


os_fspath = os.fspath
os_PathLike = os.PathLike
<|endoftext|>"""This hook should collect all binary files and any hidden modules that numpy
needs.

Our (some-what inadequate) docs for writing PyInstaller hooks are kept here:
https://pyinstaller.readthedocs.io/en/stable/hooks.html

"""
from PyInstaller.compat import is_conda, is_pure_conda
from PyInstaller.utils.hooks import collect_dynamic_libs, is_module_satisfies

# Collect all DLLs inside numpy's installation folder, dump them into built
# app's root.
binaries = collect_dynamic_libs("numpy", ".")

# If using Conda without any non-conda virtual environment manager:
if is_pure_conda:
    # Assume running the NumPy from Conda-forge and collect it's DLLs from the
    # communal Conda bin directory. DLLs from NumPy's dependencies must also be
    # collected to capture MKL, OpenBlas, OpenMP, etc.
    from PyInstaller.utils.hooks import conda_support
    datas = conda_support.collect_dynamic_libs("numpy", dependencies=True)

# Submodules PyInstaller cannot detect.  `_dtype_ctypes` is only imported
# from C and `_multiarray_tests` is used in tests (which are not packed).
hiddenimports = ['numpy._core._dtype_ctypes', 'numpy._core._multiarray_tests']

# Remove testing and building code and packages that are referenced throughout
# NumPy but are not really dependencies.
excludedimports = [
    "scipy",
    "pytest",
    "f2py",
    "setuptools",
    "distutils",
    "numpy.distutils",
]
<|endoftext|>"""
Machine arithmetic - determine the parameters of the
floating-point arithmetic system

Author: Pearu Peterson, September 2003

"""
__all__ = ['MachAr']

from .fromnumeric import any
from ._ufunc_config import errstate
from .._utils import set_module

# Need to speed this up...especially for longdouble

# Deprecated 2021-10-20, NumPy 1.22
class MachAr:
    """
    Diagnosing machine parameters.

    Attributes
    ----------
    ibeta : int
        Radix in which numbers are represented.
    it : int
        Number of base-`ibeta` digits in the floating point mantissa M.
    machep : int
        Exponent of the smallest (most negative) power of `ibeta` that,
        added to 1.0, gives something different from 1.0
    eps : float
        Floating-point number ``beta**machep`` (floating point precision)
    negep : int
        Exponent of the smallest power of `ibeta` that, subtracted
        from 1.0, gives something different from 1.0.
    epsneg : float
        Floating-point number ``beta**negep``.
    iexp : int
        Number of bits in the exponent (including its sign and bias).
    minexp : int
        Smallest (most negative) power of `ibeta` consistent with there
        being no leading zeros in the mantissa.
    xmin : float
        Floating-point number ``beta**minexp`` (the smallest [in
        magnitude] positive floating point number with full precision).
    maxexp : int
        Smallest (positive) power of `ibeta` that causes overflow.
    xmax : float
        ``(1-epsneg) * beta**maxexp`` (the largest [in magnitude]
        usable floating value).
    irnd : int
        In ``range(6)``, information on what kind of rounding is done
        in addition, and on how underflow is handled.
    ngrd : int
        Number of 'guard digits' used when truncating the product
        of two mantissas to fit the representation.
    epsilon : float
        Same as `eps`.
    tiny : float
        An alias for `smallest_normal`, kept for backwards compatibility.
    huge : float
        Same as `xmax`.
    precision : float
        ``- int(-log10(eps))``
    resolution : float
        ``- 10**(-precision)``
    smallest_normal : float
        The smallest positive floating point number with 1 as leading bit in
        the mantissa following IEEE-754. Same as `xmin`.
    smallest_subnormal : float
        The smallest positive floating point number with 0 as leading bit in
        the mantissa following IEEE-754.

    Parameters
    ----------
    float_conv : function, optional
        Function that converts an integer or integer array to a float
        or float array. Default is `float`.
    int_conv : function, optional
        Function that converts a float or float array to an integer or
        integer array. Default is `int`.
    float_to_float : function, optional
        Function that converts a float array to float. Default is `float`.
        Note that this does not seem to do anything useful in the current
        implementation.
    float_to_str : function, optional
        Function that converts a single float to a string. Default is
        ``lambda v:'%24.16e' %v``.
    title : str, optional
        Title that is printed in the string representation of `MachAr`.

    See Also
    --------
    finfo : Machine limits for floating point types.
    iinfo : Machine limits for integer types.

    References
    ----------
    .. [1] Press, Teukolsky, Vetterling and Flannery,
           "Numerical Recipes in C++," 2nd ed,
           Cambridge University Press, 2002, p. 31.

    """

    def __init__(self, float_conv=float,int_conv=int,
                 float_to_float=float,
                 float_to_str=lambda v:'%24.16e' % v,
                 title='Python floating point number'):
        """

        float_conv - convert integer to float (array)
        int_conv   - convert float (array) to integer
        float_to_float - convert float array to float
        float_to_str - convert array float to str
        title        - description of used floating point numbers

        """
        # We ignore all errors here because we are purposely triggering
        # underflow to detect the properties of the running arch.
        with errstate(under='ignore'):
            self._do_init(float_conv, int_conv, float_to_float, float_to_str, title)

    def _do_init(self, float_conv, int_conv, float_to_float, float_to_str, title):
        max_iterN = 10000
        msg = "Did not converge after %d tries with %s"
        one = float_conv(1)
        two = one + one
        zero = one - one

        # Do we really need to do this?  Aren't they 2 and 2.0?
        # Determine ibeta and beta
        a = one
        for _ in range(max_iterN):
            a = a + a
            temp = a + one
            temp1 = temp - a
            if any(temp1 - one != zero):
                break
        else:
            raise RuntimeError(msg % (_, one.dtype))
        b = one
        for _ in range(max_iterN):
            b = b + b
            temp = a + b
            itemp = int_conv(temp-a)
            if any(itemp != 0):
                break
        else:
            raise RuntimeError(msg % (_, one.dtype))
        ibeta = itemp
        beta = float_conv(ibeta)

        # Determine it and irnd
        it = -1
        b = one
        for _ in range(max_iterN):
            it = it + 1
            b = b * beta
            temp = b + one
            temp1 = temp - b
            if any(temp1 - one != zero):
                break
        else:
            raise RuntimeError(msg % (_, one.dtype))

        betah = beta / two
        a = one
        for _ in range(max_iterN):
            a = a + a
            temp = a + one
            temp1 = temp - a
            if any(temp1 - one != zero):
                break
        else:
            raise RuntimeError(msg % (_, one.dtype))
        temp = a + betah
        irnd = 0
        if any(temp-a != zero):
            irnd = 1
        tempa = a + beta
        temp = tempa + betah
        if irnd == 0 and any(temp-tempa != zero):
            irnd = 2

        # Determine negep and epsneg
        negep = it + 3
        betain = one / beta
        a = one
        for i in range(negep):
            a = a * betain
        b = a
        for _ in range(max_iterN):
            temp = one - a
            if any(temp-one != zero):
                break
            a = a * beta
            negep = negep - 1
            # Prevent infinite loop on PPC with gcc 4.0:
            if negep < 0:
                raise RuntimeError("could not determine machine tolerance "
                                   "for 'negep', locals() -> %s" % (locals()))
        else:
            raise RuntimeError(msg % (_, one.dtype))
        negep = -negep
        epsneg = a

        # Determine machep and eps
        machep = - it - 3
        a = b

        for _ in range(max_iterN):
            temp = one + a
            if any(temp-one != zero):
                break
            a = a * beta
            machep = machep + 1
        else:
            raise RuntimeError(msg % (_, one.dtype))
        eps = a

        # Determine ngrd
        ngrd = 0
        temp = one + eps
        if irnd == 0 and any(temp*one - one != zero):
            ngrd = 1

        # Determine iexp
        i = 0
        k = 1
        z = betain
        t = one + eps
        nxres = 0
        for _ in range(max_iterN):
            y = z
            z = y*y
            a = z*one  # Check here for underflow
            temp = z*t
            if any(a+a == zero) or any(abs(z) >= y):
                break
            temp1 = temp * betain
            if any(temp1*beta == z):
                break
            i = i + 1
            k = k + k
        else:
            raise RuntimeError(msg % (_, one.dtype))
        if ibeta != 10:
            iexp = i + 1
            mx = k + k
        else:
            iexp = 2
            iz = ibeta
            while k >= iz:
                iz = iz * ibeta
                iexp = iexp + 1
            mx = iz + iz - 1

        # Determine minexp and xmin
        for _ in range(max_iterN):
            xmin = y
            y = y * betain
            a = y * one
            temp = y * t
            if any((a + a) != zero) and any(abs(y) < xmin):
                k = k + 1
                temp1 = temp * betain
                if any(temp1*beta == y) and any(temp != y):
                    nxres = 3
                    xmin = y
                    break
            else:
                break
        else:
            raise RuntimeError(msg % (_, one.dtype))
        minexp = -k

        # Determine maxexp, xmax
        if mx <= k + k - 3 and ibeta != 10:
            mx = mx + mx
            iexp = iexp + 1
        maxexp = mx + minexp
        irnd = irnd + nxres
        if irnd >= 2:
            maxexp = maxexp - 2
        i = maxexp + minexp
        if ibeta == 2 and not i:
            maxexp = maxexp - 1
        if i > 20:
            maxexp = maxexp - 1
        if any(a != y):
            maxexp = maxexp - 2
        xmax = one - epsneg
        if any(xmax*one != xmax):
            xmax = one - beta*epsneg
        xmax = xmax / (xmin*beta*beta*beta)
        i = maxexp + minexp + 3
        for j in range(i):
            if ibeta == 2:
                xmax = xmax + xmax
            else:
                xmax = xmax * beta

        smallest_subnormal = abs(xmin / beta ** (it))

        self.ibeta = ibeta
        self.it = it
        self.negep = negep
        self.epsneg = float_to_float(epsneg)
        self._str_epsneg = float_to_str(epsneg)
        self.machep = machep
        self.eps = float_to_float(eps)
        self._str_eps = float_to_str(eps)
        self.ngrd = ngrd
        self.iexp = iexp
        self.minexp = minexp
        self.xmin = float_to_float(xmin)
        self._str_xmin = float_to_str(xmin)
        self.maxexp = maxexp
        self.xmax = float_to_float(xmax)
        self._str_xmax = float_to_str(xmax)
        self.irnd = irnd

        self.title = title
        # Commonly used parameters
        self.epsilon = self.eps
        self.tiny = self.xmin
        self.huge = self.xmax
        self.smallest_normal = self.xmin
        self._str_smallest_normal = float_to_str(self.xmin)
        self.smallest_subnormal = float_to_float(smallest_subnormal)
        self._str_smallest_subnormal = float_to_str(smallest_subnormal)

        import math
        self.precision = int(-math.log10(float_to_float(self.eps)))
        ten = two + two + two + two + two
        resolution = ten ** (-self.precision)
        self.resolution = float_to_float(resolution)
        self._str_resolution = float_to_str(resolution)

    def __str__(self):
        fmt = (
           'Machine parameters for %(title)s\n'
           '---------------------------------------------------------------------\n'
           'ibeta=%(ibeta)s it=%(it)s iexp=%(iexp)s ngrd=%(ngrd)s irnd=%(irnd)s\n'
           'machep=%(machep)s     eps=%(_str_eps)s (beta**machep == epsilon)\n'
           'negep =%(negep)s  epsneg=%(_str_epsneg)s (beta**epsneg)\n'
           'minexp=%(minexp)s   xmin=%(_str_xmin)s (beta**minexp == tiny)\n'
           'maxexp=%(maxexp)s    xmax=%(_str_xmax)s ((1-epsneg)*beta**maxexp == huge)\n'
           'smallest_normal=%(smallest_normal)s    '
           'smallest_subnormal=%(smallest_subnormal)s\n'
           '---------------------------------------------------------------------\n'
           )
        return fmt % self.__dict__


if __name__ == '__main__':
    print(MachAr())
<|endoftext|>"""
A place for code to be called from the implementation of np.dtype

String handling is much easier to do correctly in python.
"""
import numpy as np


_kind_to_stem = {
    'u': 'uint',
    'i': 'int',
    'c': 'complex',
    'f': 'float',
    'b': 'bool',
    'V': 'void',
    'O': 'object',
    'M': 'datetime',
    'm': 'timedelta',
    'S': 'bytes',
    'U': 'str',
}


def _kind_name(dtype):
    try:
        return _kind_to_stem[dtype.kind]
    except KeyError as e:
        raise RuntimeError(
            "internal dtype error, unknown kind {!r}"
            .format(dtype.kind)
        ) from None


def __str__(dtype):
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=True)
    elif dtype.subdtype:
        return _subarray_str(dtype)
    elif issubclass(dtype.type, np.flexible) or not dtype.isnative:
        return dtype.str
    else:
        return dtype.name


def __repr__(dtype):
    arg_str = _construction_repr(dtype, include_align=False)
    if dtype.isalignedstruct:
        arg_str = arg_str + ", align=True"
    return "dtype({})".format(arg_str)


def _unpack_field(dtype, offset, title=None):
    """
    Helper function to normalize the items in dtype.fields.

    Call as:

    dtype, offset, title = _unpack_field(*dtype.fields[name])
    """
    return dtype, offset, title


def _isunsized(dtype):
    # PyDataType_ISUNSIZED
    return dtype.itemsize == 0


def _construction_repr(dtype, include_align=False, short=False):
    """
    Creates a string repr of the dtype, excluding the 'dtype()' part
    surrounding the object. This object may be a string, a list, or
    a dict depending on the nature of the dtype. This
    is the object passed as the first parameter to the dtype
    constructor, and if no additional constructor parameters are
    given, will reproduce the exact memory layout.

    Parameters
    ----------
    short : bool
        If true, this creates a shorter repr using 'kind' and 'itemsize',
        instead of the longer type name.

    include_align : bool
        If true, this includes the 'align=True' parameter
        inside the struct dtype construction dict when needed. Use this flag
        if you want a proper repr string without the 'dtype()' part around it.

        If false, this does not preserve the
        'align=True' parameter or sticky NPY_ALIGNED_STRUCT flag for
        struct arrays like the regular repr does, because the 'align'
        flag is not part of first dtype constructor parameter. This
        mode is intended for a full 'repr', where the 'align=True' is
        provided as the second parameter.
    """
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=include_align)
    elif dtype.subdtype:
        return _subarray_str(dtype)
    else:
        return _scalar_str(dtype, short=short)


def _scalar_str(dtype, short):
    byteorder = _byte_order_str(dtype)

    if dtype.type == np.bool:
        if short:
            return "'?'"
        else:
            return "'bool'"

    elif dtype.type == np.object_:
        # The object reference may be different sizes on different
        # platforms, so it should never include the itemsize here.
        return "'O'"

    elif dtype.type == np.bytes_:
        if _isunsized(dtype):
            return "'S'"
        else:
            return "'S%d'" % dtype.itemsize

    elif dtype.type == np.str_:
        if _isunsized(dtype):
            return "'%sU'" % byteorder
        else:
            return "'%sU%d'" % (byteorder, dtype.itemsize / 4)

    elif dtype.type == str:
        return "'T'"

    elif not type(dtype)._legacy:
        return f"'{byteorder}{type(dtype).__name__}{dtype.itemsize * 8}'"

    # unlike the other types, subclasses of void are preserved - but
    # historically the repr does not actually reveal the subclass
    elif issubclass(dtype.type, np.void):
        if _isunsized(dtype):
            return "'V'"
        else:
            return "'V%d'" % dtype.itemsize

    elif dtype.type == np.datetime64:
        return "'%sM8%s'" % (byteorder, _datetime_metadata_str(dtype))

    elif dtype.type == np.timedelta64:
        return "'%sm8%s'" % (byteorder, _datetime_metadata_str(dtype))

    elif np.issubdtype(dtype, np.number):
        # Short repr with endianness, like '<f8'
        if short or dtype.byteorder not in ('=', '|'):
            return "'%s%c%d'" % (byteorder, dtype.kind, dtype.itemsize)

        # Longer repr, like 'float64'
        else:
            return "'%s%d'" % (_kind_name(dtype), 8*dtype.itemsize)

    elif dtype.isbuiltin == 2:
        return dtype.type.__name__

    else:
        raise RuntimeError(
            "Internal error: NumPy dtype unrecognized type number")


def _byte_order_str(dtype):
    """ Normalize byteorder to '<' or '>' """
    # hack to obtain the native and swapped byte order characters
    swapped = np.dtype(int).newbyteorder('S')
    native = swapped.newbyteorder('S')

    byteorder = dtype.byteorder
    if byteorder == '=':
        return native.byteorder
    if byteorder == 'S':
        # TODO: this path can never be reached
        return swapped.byteorder
    elif byteorder == '|':
        return ''
    else:
        return byteorder


def _datetime_metadata_str(dtype):
    # TODO: this duplicates the C metastr_to_unicode functionality
    unit, count = np.datetime_data(dtype)
    if unit == 'generic':
        return ''
    elif count == 1:
        return '[{}]'.format(unit)
    else:
        return '[{}{}]'.format(count, unit)


def _struct_dict_str(dtype, includealignedflag):
    # unpack the fields dictionary into ls
    names = dtype.names
    fld_dtypes = []
    offsets = []
    titles = []
    for name in names:
        fld_dtype, offset, title = _unpack_field(*dtype.fields[name])
        fld_dtypes.append(fld_dtype)
        offsets.append(offset)
        titles.append(title)

    # Build up a string to make the dictionary

    if np._core.arrayprint._get_legacy_print_mode() <= 121:
        colon = ":"
        fieldsep = ","
    else:
        colon = ": "
        fieldsep = ", "

    # First, the names
    ret = "{'names'%s[" % colon
    ret += fieldsep.join(repr(name) for name in names)

    # Second, the formats
    ret += "], 'formats'%s[" % colon
    ret += fieldsep.join(
        _construction_repr(fld_dtype, short=True) for fld_dtype in fld_dtypes)

    # Third, the offsets
    ret += "], 'offsets'%s[" % colon
    ret += fieldsep.join("%d" % offset for offset in offsets)

    # Fourth, the titles
    if any(title is not None for title in titles):
        ret += "], 'titles'%s[" % colon
        ret += fieldsep.join(repr(title) for title in titles)

    # Fifth, the itemsize
    ret += "], 'itemsize'%s%d" % (colon, dtype.itemsize)

    if (includealignedflag and dtype.isalignedstruct):
        # Finally, the aligned flag
        ret += ", 'aligned'%sTrue}" % colon
    else:
        ret += "}"

    return ret


def _aligned_offset(offset, alignment):
    # round up offset:
    return - (-offset // alignment) * alignment


def _is_packed(dtype):
    """
    Checks whether the structured data type in 'dtype'
    has a simple layout, where all the fields are in order,
    and follow each other with no alignment padding.

    When this returns true, the dtype can be reconstructed
    from a list of the field names and dtypes with no additional
    dtype parameters.

    Duplicates the C `is_dtype_struct_simple_unaligned_layout` function.
    """
    align = dtype.isalignedstruct
    max_alignment = 1
    total_offset = 0
    for name in dtype.names:
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])

        if align:
            total_offset = _aligned_offset(total_offset, fld_dtype.alignment)
            max_alignment = max(max_alignment, fld_dtype.alignment)

        if fld_offset != total_offset:
            return False
        total_offset += fld_dtype.itemsize

    if align:
        total_offset = _aligned_offset(total_offset, max_alignment)

    return total_offset == dtype.itemsize


def _struct_list_str(dtype):
    items = []
    for name in dtype.names:
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])

        item = "("
        if title is not None:
            item += "({!r}, {!r}), ".format(title, name)
        else:
            item += "{!r}, ".format(name)
        # Special case subarray handling here
        if fld_dtype.subdtype is not None:
            base, shape = fld_dtype.subdtype
            item += "{}, {}".format(
                _construction_repr(base, short=True),
                shape
            )
        else:
            item += _construction_repr(fld_dtype, short=True)

        item += ")"
        items.append(item)

    return "[" + ", ".join(items) + "]"


def _struct_str(dtype, include_align):
    # The list str representation can't include the 'align=' flag,
    # so if it is requested and the struct has the aligned flag set,
    # we must use the dict str instead.
    if not (include_align and dtype.isalignedstruct) and _is_packed(dtype):
        sub = _struct_list_str(dtype)

    else:
        sub = _struct_dict_str(dtype, include_align)

    # If the data type isn't the default, void, show it
    if dtype.type != np.void:
        return "({t.__module__}.{t.__name__}, {f})".format(t=dtype.type, f=sub)
    else:
        return sub


def _subarray_str(dtype):
    base, shape = dtype.subdtype
    return "({}, {})".format(
        _construction_repr(base, short=True),
        shape
    )


def _name_includes_bit_suffix(dtype):
    if dtype.type == np.object_:
        # pointer size varies by system, best to omit it
        return False
    elif dtype.type == np.bool:
        # implied
        return False
    elif dtype.type is None:
        return True
    elif np.issubdtype(dtype, np.flexible) and _isunsized(dtype):
        # unspecified
        return False
    else:
        return True


def _name_get(dtype):
    # provides dtype.name.__get__, documented as returning a "bit name"

    if dtype.isbuiltin == 2:
        # user dtypes don't promise to do anything special
        return dtype.type.__name__

    if not type(dtype)._legacy:
        name = type(dtype).__name__

    elif issubclass(dtype.type, np.void):
        # historically, void subclasses preserve their name, eg `record64`
        name = dtype.type.__name__
    else:
        name = _kind_name(dtype)

    # append bit counts
    if _name_includes_bit_suffix(dtype):
        name += "{}".format(dtype.itemsize * 8)

    # append metadata to datetimes
    if dtype.type in (np.datetime64, np.timedelta64):
        name += _datetime_metadata_str(dtype)

    return name
<|endoftext|>"""
Implementation of optimized einsum.

"""
import itertools
import operator

from numpy._core.multiarray import c_einsum
from numpy._core.numeric import asanyarray, tensordot
from numpy._core.overrides import array_function_dispatch

__all__ = ['einsum', 'einsum_path']

# importing string for string.ascii_letters would be too slow
# the first import before caching has been measured to take 800 µs (#23777)
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)


def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    """
    Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------

    >>> _flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    30

    >>> _flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    60

    """

    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor

def _compute_size_by_dict(indices, idx_dict):
    """
    Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index sizes

    Returns
    -------
    ret : int
        The resulting product.

    Examples
    --------
    >>> _compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret


def _find_contraction(positions, input_sets, output_set):
    """
    Finds the contraction for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples
    --------

    # A simple dot product test case
    >>> pos = (0, 1)
    >>> isets = [set('ab'), set('bc')]
    >>> oset = set('ac')
    >>> _find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})

    # A more complex case with additional terms in the contraction
    >>> pos = (0, 2)
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('ac')
    >>> _find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
    """

    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)

    return (new_result, remaining, idx_removed, idx_contract)


def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Computes all possible pair contractions, sieves the results based
    on ``memory_limit`` and returns the lowest cost path. This algorithm
    scales factorial with respect to the elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The optimal contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set()
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _optimal_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    full_results = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        iter_results = []

        # Compute all unique pairs
        for curr in full_results:
            cost, positions, remaining = curr
            for con in itertools.combinations(
                range(len(input_sets) - iteration), 2
            ):

                # Find the contraction
                cont = _find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = cont

                # Sieve the results based on memory_limit
                new_size = _compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue

                # Build (total_cost, positions, indices_remaining)
                total_cost = cost + _flop_count(
                    idx_contract, idx_removed, len(con), idx_dict
                )
                new_pos = positions + [con]
                iter_results.append((total_cost, new_pos, new_input_sets))

        # Update combinatorial list, if we did not find anything return best
        # path + remaining contractions
        if iter_results:
            full_results = iter_results
        else:
            path = min(full_results, key=lambda x: x[0])[1]
            path += [tuple(range(len(input_sets) - iteration))]
            return path

    # If we have not found anything return single einsum contraction
    if len(full_results) == 0:
        return [tuple(range(len(input_sets)))]

    path = min(full_results, key=lambda x: x[0])[1]
    return path

def _parse_possible_contraction(
        positions, input_sets, output_set, idx_dict,
        memory_limit, path_cost, naive_cost
    ):
    """Compute the cost (removed size + flops) and resultant indices for
    performing the contraction specified by ``positions``.

    Parameters
    ----------
    positions : tuple of int
        The locations of the proposed tensors to contract.
    input_sets : list of sets
        The indices found on each tensors.
    output_set : set
        The output indices of the expression.
    idx_dict : dict
        Mapping of each index to its size.
    memory_limit : int
        The total allowed size for an intermediary tensor.
    path_cost : int
        The contraction cost so far.
    naive_cost : int
        The cost of the unoptimized expression.

    Returns
    -------
    cost : (int, int)
        A tuple containing the size of any indices removed, and the flop cost.
    positions : tuple of int
        The locations of the proposed tensors to contract.
    new_input_sets : list of sets
        The resulting new list of indices if this proposed contraction
        is performed.

    """

    # Find the contraction
    contract = _find_contraction(positions, input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract

    # Sieve the results based on memory_limit
    new_size = _compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit:
        return None

    # Build sort tuple
    old_sizes = (
        _compute_size_by_dict(input_sets[p], idx_dict) for p in positions
    )
    removed_size = sum(old_sizes) - new_size

    # NB: removed_size used to be just the size of any removed indices i.e.:
    #     helpers.compute_size_by_dict(idx_removed, idx_dict)
    cost = _flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    sort = (-removed_size, cost)

    # Sieve based on total cost as well
    if (path_cost + cost) > naive_cost:
