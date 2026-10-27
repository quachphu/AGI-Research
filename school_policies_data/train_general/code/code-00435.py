            dst = dst+'/'
    else:
        if os.path.exists(dst) and os.path.isdir(dst):
            dest_is_dir = True

    if dest_is_dir:
        dest_dir = dst
        dest_fname = os.path.basename(src)
        dst = os.path.join(dest_dir, dest_fname)
    else:
        dest_dir = os.path.dirname(dst)

    if not os.path.exists(dest_dir):
        if create_dest_dirs:
            make_dirs(dest_dir)
        else:
            raise FileNotFoundError("You must create directory first.")

    if only_update:
        if not missing_or_other_newer(dst, src):
            return

    if os.path.islink(dst):
        dst = os.path.abspath(os.path.realpath(dst), cwd=cwd)

    shutil.copy(src, dst)
    if copystat:
        shutil.copystat(src, dst)

    return dst

Glob = namedtuple('Glob', 'pathname')
ArbitraryDepthGlob = namedtuple('ArbitraryDepthGlob', 'filename')

def glob_at_depth(filename_glob, cwd=None):
    if cwd is not None:
        cwd = '.'
    globbed = []
    for root, dirs, filenames in os.walk(cwd):
        for fn in filenames:
            # This is not tested:
            if fnmatch.fnmatch(fn, filename_glob):
                globbed.append(os.path.join(root, fn))
    return globbed

def sha256_of_file(path, nblocks=128):
    """ Computes the SHA256 hash of a file.

    Parameters
    ==========

    path : string
        Path to file to compute hash of.
    nblocks : int
        Number of blocks to read per iteration.

    Returns
    =======

    hashlib sha256 hash object. Use ``.digest()`` or ``.hexdigest()``
    on returned object to get binary or hex encoded string.
    """
    sh = sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(nblocks*sh.block_size), b''):
            sh.update(chunk)
    return sh


def sha256_of_string(string):
    """ Computes the SHA256 hash of a string. """
    sh = sha256()
    sh.update(string)
    return sh


def pyx_is_cplus(path):
    """
    Inspect a Cython source file (.pyx) and look for comment line like:

    # distutils: language = c++

    Returns True if such a file is present in the file, else False.
    """
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') and '=' in line:
                splitted = line.split('=')
                if len(splitted) != 2:
                    continue
                lhs, rhs = splitted
                if lhs.strip().split()[-1].lower() == 'language' and \
                       rhs.strip().split()[0].lower() == 'c++':
                            return True
    return False

def import_module_from_file(filename, only_if_newer_than=None):
    """ Imports Python extension (from shared object file)

    Provide a list of paths in `only_if_newer_than` to check
    timestamps of dependencies. import_ raises an ImportError
    if any is newer.

    Word of warning: The OS may cache shared objects which makes
    reimporting same path of an shared object file very problematic.

    It will not detect the new time stamp, nor new checksum, but will
    instead silently use old module. Use unique names for this reason.

    Parameters
    ==========

    filename : str
        Path to shared object.
    only_if_newer_than : iterable of strings
        Paths to dependencies of the shared object.

    Raises
    ======

    ``ImportError`` if any of the files specified in ``only_if_newer_than`` are newer
    than the file given by filename.
    """
    path, name = os.path.split(filename)
    name, ext = os.path.splitext(name)
    name = name.split('.')[0]
    if sys.version_info[0] == 2:
        from imp import find_module, load_module
        fobj, filename, data = find_module(name, [path])
        if only_if_newer_than:
            for dep in only_if_newer_than:
                if os.path.getmtime(filename) < os.path.getmtime(dep):
                    raise ImportError("{} is newer than {}".format(dep, filename))
        mod = load_module(name, fobj, filename, data)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, filename)
        if spec is None:
            raise ImportError("Failed to import: '%s'" % filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def find_binary_of_command(candidates):
    """ Finds binary first matching name among candidates.

    Calls ``which`` from shutils for provided candidates and returns
    first hit.

    Parameters
    ==========

    candidates : iterable of str
        Names of candidate commands

    Raises
    ======

    CompilerNotFoundError if no candidates match.
    """
    from shutil import which
    for c in candidates:
        binary_path = which(c)
        if c and binary_path:
            return c, binary_path

    raise CompilerNotFoundError('No binary located for candidates: {}'.format(candidates))


def unique_list(l):
    """ Uniquify a list (skip duplicate items). """
    result = []
    for x in l:
        if x not in result:
            result.append(x)
    return result
<|endoftext|>import os
from .compilation import compile_run_strings
from .util import CompilerNotFoundError

def has_fortran():
    if not hasattr(has_fortran, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings(
                [('main.f90', (
                    'program foo\n'
                    'print *, "hello world"\n'
                    'end program'
                ))], clean=True
            )
        except CompilerNotFoundError:
            has_fortran.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                has_fortran.result = False
            else:
                has_fortran.result = True
    return has_fortran.result


def has_c():
    if not hasattr(has_c, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings(
                [('main.c', (
                    '#include <stdio.h>\n'
                    'int main(){\n'
                    'printf("hello world\\n");\n'
                    'return 0;\n'
                    '}'
                ))], clean=True
            )
        except CompilerNotFoundError:
            has_c.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                has_c.result = False
            else:
                has_c.result = True
    return has_c.result


def has_cxx():
    if not hasattr(has_cxx, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings(
                [('main.cxx', (
                    '#include <iostream>\n'
                    'int main(){\n'
                    'std::cout << "hello world" << std::endl;\n'
                    '}'
                ))], clean=True
            )
        except CompilerNotFoundError:
            has_cxx.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                has_cxx.result = False
            else:
                has_cxx.result = True
    return has_cxx.result
<|endoftext|>import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from sysconfig import get_config_var, get_config_vars, get_path

from .runners import (
    CCompilerRunner,
    CppCompilerRunner,
    FortranCompilerRunner
)
from .util import (
    get_abspath, make_dirs, copy, Glob, ArbitraryDepthGlob,
    glob_at_depth, import_module_from_file, pyx_is_cplus,
    sha256_of_string, sha256_of_file, CompileError
)

if os.name == 'posix':
    objext = '.o'
elif os.name == 'nt':
    objext = '.obj'
else:
    warnings.warn("Unknown os.name: {}".format(os.name))
    objext = '.o'


def compile_sources(files, Runner=None, destdir=None, cwd=None, keep_dir_struct=False,
                    per_file_kwargs=None, **kwargs):
    """ Compile source code files to object files.

    Parameters
    ==========

    files : iterable of str
        Paths to source files, if ``cwd`` is given, the paths are taken as relative.
    Runner: CompilerRunner subclass (optional)
        Could be e.g. ``FortranCompilerRunner``. Will be inferred from filename
        extensions if missing.
    destdir: str
        Output directory, if cwd is given, the path is taken as relative.
    cwd: str
        Working directory. Specify to have compiler run in other directory.
        also used as root of relative paths.
    keep_dir_struct: bool
        Reproduce directory structure in `destdir`. default: ``False``
    per_file_kwargs: dict
        Dict mapping instances in ``files`` to keyword arguments.
    \\*\\*kwargs: dict
        Default keyword arguments to pass to ``Runner``.

    Returns
    =======
    List of strings (paths of object files).
    """
    _per_file_kwargs = {}

    if per_file_kwargs is not None:
        for k, v in per_file_kwargs.items():
            if isinstance(k, Glob):
                for path in glob.glob(k.pathname):
                    _per_file_kwargs[path] = v
            elif isinstance(k, ArbitraryDepthGlob):
                for path in glob_at_depth(k.filename, cwd):
                    _per_file_kwargs[path] = v
            else:
                _per_file_kwargs[k] = v

    # Set up destination directory
    destdir = destdir or '.'
    if not os.path.isdir(destdir):
        if os.path.exists(destdir):
            raise OSError("{} is not a directory".format(destdir))
        else:
            make_dirs(destdir)
    if cwd is None:
        cwd = '.'
        for f in files:
            copy(f, destdir, only_update=True, dest_is_dir=True)

    # Compile files and return list of paths to the objects
    dstpaths = []
    for f in files:
        if keep_dir_struct:
            name, ext = os.path.splitext(f)
        else:
            name, ext = os.path.splitext(os.path.basename(f))
        file_kwargs = kwargs.copy()
        file_kwargs.update(_per_file_kwargs.get(f, {}))
        dstpaths.append(src2obj(f, Runner, cwd=cwd, **file_kwargs))
    return dstpaths


def get_mixed_fort_c_linker(vendor=None, cplus=False, cwd=None):
    vendor = vendor or os.environ.get('SYMPY_COMPILER_VENDOR', 'gnu')

    if vendor.lower() == 'intel':
        if cplus:
            return (FortranCompilerRunner,
                    {'flags': ['-nofor_main', '-cxxlib']}, vendor)
        else:
            return (FortranCompilerRunner,
                    {'flags': ['-nofor_main']}, vendor)
    elif vendor.lower() == 'gnu' or 'llvm':
        if cplus:
            return (CppCompilerRunner,
                    {'lib_options': ['fortran']}, vendor)
        else:
            return (FortranCompilerRunner,
                    {}, vendor)
    else:
        raise ValueError("No vendor found.")


def link(obj_files, out_file=None, shared=False, Runner=None,
         cwd=None, cplus=False, fort=False, extra_objs=None, **kwargs):
    """ Link object files.

    Parameters
    ==========

    obj_files: iterable of str
        Paths to object files.
    out_file: str (optional)
        Path to executable/shared library, if ``None`` it will be
        deduced from the last item in obj_files.
    shared: bool
        Generate a shared library?
    Runner: CompilerRunner subclass (optional)
        If not given the ``cplus`` and ``fort`` flags will be inspected
        (fallback is the C compiler).
    cwd: str
        Path to the root of relative paths and working directory for compiler.
    cplus: bool
        C++ objects? default: ``False``.
    fort: bool
        Fortran objects? default: ``False``.
    extra_objs: list
        List of paths to extra object files / static libraries.
    \\*\\*kwargs: dict
        Keyword arguments passed to ``Runner``.

    Returns
    =======

    The absolute path to the generated shared object / executable.

    """
    if out_file is None:
        out_file, ext = os.path.splitext(os.path.basename(obj_files[-1]))
        if shared:
            out_file += get_config_var('EXT_SUFFIX')

    if not Runner:
        if fort:
            Runner, extra_kwargs, vendor = \
                get_mixed_fort_c_linker(
                    vendor=kwargs.get('vendor', None),
                    cplus=cplus,
                    cwd=cwd,
                )
            for k, v in extra_kwargs.items():
                if k in kwargs:
                    kwargs[k].expand(v)
                else:
                    kwargs[k] = v
        else:
            if cplus:
                Runner = CppCompilerRunner
            else:
                Runner = CCompilerRunner

    flags = kwargs.pop('flags', [])
    if shared:
        if '-shared' not in flags:
            flags.append('-shared')
    run_linker = kwargs.pop('run_linker', True)
    if not run_linker:
        raise ValueError("run_linker was set to False (nonsensical).")

    out_file = get_abspath(out_file, cwd=cwd)
    runner = Runner(obj_files+(extra_objs or []), out_file, flags, cwd=cwd, **kwargs)
    runner.run()
    return out_file


def link_py_so(obj_files, so_file=None, cwd=None, libraries=None,
               cplus=False, fort=False, extra_objs=None, **kwargs):
    """ Link Python extension module (shared object) for importing

    Parameters
    ==========

    obj_files: iterable of str
        Paths to object files to be linked.
    so_file: str
        Name (path) of shared object file to create. If not specified it will
        have the basname of the last object file in `obj_files` but with the
        extension '.so' (Unix).
    cwd: path string
        Root of relative paths and working directory of linker.
    libraries: iterable of strings
        Libraries to link against, e.g. ['m'].
    cplus: bool
        Any C++ objects? default: ``False``.
    fort: bool
        Any Fortran objects? default: ``False``.
    extra_objs: list
        List of paths of extra object files / static libraries to link against.
    kwargs**: dict
        Keyword arguments passed to ``link(...)``.

    Returns
    =======

    Absolute path to the generate shared object.
    """
    libraries = libraries or []

    include_dirs = kwargs.pop('include_dirs', [])
    library_dirs = kwargs.pop('library_dirs', [])

    # Add Python include and library directories
    # PY_LDFLAGS does not available on all python implementations
    # e.g. when with pypy, so it's LDFLAGS we need to use
    if sys.platform == "win32":
        warnings.warn("Windows not yet supported.")
    elif sys.platform == 'darwin':
        cfgDict = get_config_vars()
        kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
        library_dirs += [cfgDict['LIBDIR']]

        # In macOS, linker needs to compile frameworks
        # e.g. "-framework CoreFoundation"
        is_framework = False
        for opt in cfgDict['LIBS'].split():
            if is_framework:
                kwargs['linkline'] = kwargs.get('linkline', []) + ['-framework', opt]
                is_framework = False
            elif opt.startswith('-l'):
                libraries.append(opt[2:])
            elif opt.startswith('-framework'):
                is_framework = True
        # The python library is not included in LIBS
        libfile = cfgDict['LIBRARY']
        libname = ".".join(libfile.split('.')[:-1])[3:]
        libraries.append(libname)

    elif sys.platform[:3] == 'aix':
        # Don't use the default code below
        pass
    else:
        if get_config_var('Py_ENABLE_SHARED'):
            cfgDict = get_config_vars()
            kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
            library_dirs += [cfgDict['LIBDIR']]
            for opt in cfgDict['BLDLIBRARY'].split():
                if opt.startswith('-l'):
                    libraries += [opt[2:]]
        else:
            pass

    flags = kwargs.pop('flags', [])
    needed_flags = ('-pthread',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)

    return link(obj_files, shared=True, flags=flags, cwd=cwd, cplus=cplus, fort=fort,
                include_dirs=include_dirs, libraries=libraries,
                library_dirs=library_dirs, extra_objs=extra_objs, **kwargs)


def simple_cythonize(src, destdir=None, cwd=None, **cy_kwargs):
    """ Generates a C file from a Cython source file.

    Parameters
    ==========

    src: str
        Path to Cython source.
    destdir: str (optional)
        Path to output directory (default: '.').
    cwd: path string (optional)
        Root of relative paths (default: '.').
    **cy_kwargs:
        Second argument passed to cy_compile. Generates a .cpp file if ``cplus=True`` in ``cy_kwargs``,
        else a .c file.
    """
    from Cython.Compiler.Main import (
        default_options, CompilationOptions
    )
    from Cython.Compiler.Main import compile as cy_compile

    assert src.lower().endswith('.pyx') or src.lower().endswith('.py')
    cwd = cwd or '.'
    destdir = destdir or '.'

    ext = '.cpp' if cy_kwargs.get('cplus', False) else '.c'
    c_name = os.path.splitext(os.path.basename(src))[0] + ext

    dstfile = os.path.join(destdir, c_name)

    if cwd:
        ori_dir = os.getcwd()
    else:
        ori_dir = '.'
    os.chdir(cwd)
    try:
        cy_options = CompilationOptions(default_options)
        cy_options.__dict__.update(cy_kwargs)
        # Set language_level if not set by cy_kwargs
        # as not setting it is deprecated
        if 'language_level' not in cy_kwargs:
            cy_options.__dict__['language_level'] = 3
        cy_result = cy_compile([src], cy_options)
        if cy_result.num_errors > 0:
            raise ValueError("Cython compilation failed.")

        # Move generated C file to destination
        # In macOS, the generated C file is in the same directory as the source
        # but the /var is a symlink to /private/var, so we need to use realpath
        if os.path.realpath(os.path.dirname(src)) != os.path.realpath(destdir):
            if os.path.exists(dstfile):
                os.unlink(dstfile)
            shutil.move(os.path.join(os.path.dirname(src), c_name), destdir)
    finally:
        os.chdir(ori_dir)
    return dstfile


extension_mapping = {
    '.c': (CCompilerRunner, None),
    '.cpp': (CppCompilerRunner, None),
    '.cxx': (CppCompilerRunner, None),
    '.f': (FortranCompilerRunner, None),
    '.for': (FortranCompilerRunner, None),
    '.ftn': (FortranCompilerRunner, None),
    '.f90': (FortranCompilerRunner, None),  # ifort only knows about .f90
    '.f95': (FortranCompilerRunner, 'f95'),
    '.f03': (FortranCompilerRunner, 'f2003'),
    '.f08': (FortranCompilerRunner, 'f2008'),
}


def src2obj(srcpath, Runner=None, objpath=None, cwd=None, inc_py=False, **kwargs):
    """ Compiles a source code file to an object file.

    Files ending with '.pyx' assumed to be cython files and
    are dispatched to pyx2obj.

    Parameters
    ==========

    srcpath: str
        Path to source file.
    Runner: CompilerRunner subclass (optional)
        If ``None``: deduced from extension of srcpath.
    objpath : str (optional)
        Path to generated object. If ``None``: deduced from ``srcpath``.
    cwd: str (optional)
        Working directory and root of relative paths. If ``None``: current dir.
    inc_py: bool
        Add Python include path to kwarg "include_dirs". Default: False
    \\*\\*kwargs: dict
        keyword arguments passed to Runner or pyx2obj

    """
    name, ext = os.path.splitext(os.path.basename(srcpath))
    if objpath is None:
        if os.path.isabs(srcpath):
            objpath = '.'
        else:
            objpath = os.path.dirname(srcpath)
            objpath = objpath or '.'  # avoid objpath == ''

    if os.path.isdir(objpath):
        objpath = os.path.join(objpath, name + objext)

    include_dirs = kwargs.pop('include_dirs', [])
    if inc_py:
        py_inc_dir = get_path('include')
        if py_inc_dir not in include_dirs:
            include_dirs.append(py_inc_dir)

    if ext.lower() == '.pyx':
        return pyx2obj(srcpath, objpath=objpath, include_dirs=include_dirs, cwd=cwd,
                       **kwargs)

    if Runner is None:
        Runner, std = extension_mapping[ext.lower()]
        if 'std' not in kwargs:
            kwargs['std'] = std

    flags = kwargs.pop('flags', [])
    needed_flags = ('-fPIC',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)

    # src2obj implies not running the linker...
    run_linker = kwargs.pop('run_linker', False)
    if run_linker:
        raise CompileError("src2obj called with run_linker=True")

    runner = Runner([srcpath], objpath, include_dirs=include_dirs,
                    run_linker=run_linker, cwd=cwd, flags=flags, **kwargs)
    runner.run()
    return objpath


def pyx2obj(pyxpath, objpath=None, destdir=None, cwd=None,
            include_dirs=None, cy_kwargs=None, cplus=None, **kwargs):
    """
    Convenience function

    If cwd is specified, pyxpath and dst are taken to be relative
    If only_update is set to `True` the modification time is checked
    and compilation is only run if the source is newer than the
    destination

    Parameters
    ==========

    pyxpath: str
        Path to Cython source file.
    objpath: str (optional)
        Path to object file to generate.
    destdir: str (optional)
        Directory to put generated C file. When ``None``: directory of ``objpath``.
    cwd: str (optional)
        Working directory and root of relative paths.
    include_dirs: iterable of path strings (optional)
        Passed onto src2obj and via cy_kwargs['include_path']
        to simple_cythonize.
    cy_kwargs: dict (optional)
        Keyword arguments passed onto `simple_cythonize`
    cplus: bool (optional)
        Indicate whether C++ is used. default: auto-detect using ``.util.pyx_is_cplus``.
    compile_kwargs: dict
        keyword arguments passed onto src2obj

    Returns
    =======

    Absolute path of generated object file.

    """
    assert pyxpath.endswith('.pyx')
    cwd = cwd or '.'
    objpath = objpath or '.'
    destdir = destdir or os.path.dirname(objpath)

    abs_objpath = get_abspath(objpath, cwd=cwd)

    if os.path.isdir(abs_objpath):
        pyx_fname = os.path.basename(pyxpath)
        name, ext = os.path.splitext(pyx_fname)
        objpath = os.path.join(objpath, name + objext)

    cy_kwargs = cy_kwargs or {}
    cy_kwargs['output_dir'] = cwd
    if cplus is None:
        cplus = pyx_is_cplus(pyxpath)
    cy_kwargs['cplus'] = cplus

    interm_c_file = simple_cythonize(pyxpath, destdir=destdir, cwd=cwd, **cy_kwargs)

    include_dirs = include_dirs or []
    flags = kwargs.pop('flags', [])
    needed_flags = ('-fwrapv', '-pthread', '-fPIC')
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)

    options = kwargs.pop('options', [])

    if kwargs.pop('strict_aliasing', False):
        raise CompileError("Cython requires strict aliasing to be disabled.")

    # Let's be explicit about standard
    if cplus:
        std = kwargs.pop('std', 'c++98')
    else:
        std = kwargs.pop('std', 'c99')

    return src2obj(interm_c_file, objpath=objpath, cwd=cwd,
                   include_dirs=include_dirs, flags=flags, std=std,
                   options=options, inc_py=True, strict_aliasing=False,
                   **kwargs)


def _any_X(srcs, cls):
    for src in srcs:
        name, ext = os.path.splitext(src)
        key = ext.lower()
        if key in extension_mapping:
            if extension_mapping[key][0] == cls:
                return True
    return False


def any_fortran_src(srcs):
    return _any_X(srcs, FortranCompilerRunner)


def any_cplus_src(srcs):
    return _any_X(srcs, CppCompilerRunner)


def compile_link_import_py_ext(sources, extname=None, build_dir='.', compile_kwargs=None,
                               link_kwargs=None, extra_objs=None):
    """ Compiles sources to a shared object (Python extension) and imports it

    Sources in ``sources`` which is imported. If shared object is newer than the sources, they
    are not recompiled but instead it is imported.

    Parameters
    ==========

    sources : list of strings
        List of paths to sources.
    extname : string
        Name of extension (default: ``None``).
        If ``None``: taken from the last file in ``sources`` without extension.
    build_dir: str
        Path to directory in which objects files etc. are generated.
    compile_kwargs: dict
        keyword arguments passed to ``compile_sources``
    link_kwargs: dict
        keyword arguments passed to ``link_py_so``
    extra_objs: list
        List of paths to (prebuilt) object files / static libraries to link against.

    Returns
    =======

    The imported module from of the Python extension.
    """
    if extname is None:
        extname = os.path.splitext(os.path.basename(sources[-1]))[0]

    compile_kwargs = compile_kwargs or {}
    link_kwargs = link_kwargs or {}

    try:
        mod = import_module_from_file(os.path.join(build_dir, extname), sources)
    except ImportError:
        objs = compile_sources(list(map(get_abspath, sources)), destdir=build_dir,
                               cwd=build_dir, **compile_kwargs)
        so = link_py_so(objs, cwd=build_dir, fort=any_fortran_src(sources),
                        cplus=any_cplus_src(sources), extra_objs=extra_objs, **link_kwargs)
        mod = import_module_from_file(so)
    return mod


def _write_sources_to_build_dir(sources, build_dir):
    build_dir = build_dir or tempfile.mkdtemp()
    if not os.path.isdir(build_dir):
        raise OSError("Non-existent directory: ", build_dir)

    source_files = []
    for name, src in sources:
        dest = os.path.join(build_dir, name)
        differs = True
        sha256_in_mem = sha256_of_string(src.encode('utf-8')).hexdigest()
        if os.path.exists(dest):
            if os.path.exists(dest + '.sha256'):
                sha256_on_disk = Path(dest + '.sha256').read_text()
            else:
                sha256_on_disk = sha256_of_file(dest).hexdigest()

            differs = sha256_on_disk != sha256_in_mem
        if differs:
            with open(dest, 'wt') as fh:
                fh.write(src)
            with open(dest + '.sha256', 'wt') as fh:
                fh.write(sha256_in_mem)
        source_files.append(dest)
    return source_files, build_dir


def compile_link_import_strings(sources, build_dir=None, **kwargs):
    """ Compiles, links and imports extension module from source.

    Parameters
    ==========

    sources : iterable of name/source pair tuples
    build_dir : string (default: None)
        Path. ``None`` implies use a temporary directory.
    **kwargs:
        Keyword arguments passed onto `compile_link_import_py_ext`.

    Returns
    =======

    mod : module
        The compiled and imported extension module.
    info : dict
        Containing ``build_dir`` as 'build_dir'.

    """
    source_files, build_dir = _write_sources_to_build_dir(sources, build_dir)
    mod = compile_link_import_py_ext(source_files, build_dir=build_dir, **kwargs)
    info = {"build_dir": build_dir}
    return mod, info


def compile_run_strings(sources, build_dir=None, clean=False, compile_kwargs=None, link_kwargs=None):
    """ Compiles, links and runs a program built from sources.

    Parameters
    ==========

    sources : iterable of name/source pair tuples
    build_dir : string (default: None)
        Path. ``None`` implies use a temporary directory.
    clean : bool
        Whether to remove build_dir after use. This will only have an
        effect if ``build_dir`` is ``None`` (which creates a temporary directory).
        Passing ``clean == True`` and ``build_dir != None`` raises a ``ValueError``.
        This will also set ``build_dir`` in returned info dictionary to ``None``.
    compile_kwargs: dict
        Keyword arguments passed onto ``compile_sources``
    link_kwargs: dict
        Keyword arguments passed onto ``link``

    Returns
    =======

    (stdout, stderr): pair of strings
    info: dict
        Containing exit status as 'exit_status' and ``build_dir`` as 'build_dir'

    """
    if clean and build_dir is not None:
        raise ValueError("Automatic removal of build_dir is only available for temporary directory.")
    try:
        source_files, build_dir = _write_sources_to_build_dir(sources, build_dir)
        objs = compile_sources(list(map(get_abspath, source_files)), destdir=build_dir,
                               cwd=build_dir, **(compile_kwargs or {}))
        prog = link(objs, cwd=build_dir,
                    fort=any_fortran_src(source_files),
                    cplus=any_cplus_src(source_files), **(link_kwargs or {}))
        p = subprocess.Popen([prog], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        exit_status = p.wait()
        stdout, stderr = [txt.decode('utf-8') for txt in p.communicate()]
    finally:
        if clean and os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
            build_dir = None
    info = {"exit_status": exit_status, "build_dir": build_dir}
    return (stdout, stderr), info
<|endoftext|>from sympy.core.basic import Basic

new = Basic.__new__


def assoc(d, k, v):
    d = d.copy()
    d[k] = v
    return d


basic_fns = {'op': type,
             'new': Basic.__new__,
             'leaf': lambda x: not isinstance(x, Basic) or x.is_Atom,
             'children': lambda x: x.args}

expr_fns = assoc(basic_fns, 'new', lambda op, *args: op(*args))
<|endoftext|>"""Strategies to Traverse a Tree."""
from sympy.strategies.util import basic_fns
from sympy.strategies.core import chain, do_one


def top_down(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the top nodes first."""
    return chain(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))


def bottom_up(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the bottom nodes first."""
    return chain(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)


def top_down_once(rule, fns=basic_fns):
    """Apply a rule down a tree - stop on success."""
    return do_one(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))


def bottom_up_once(rule, fns=basic_fns):
    """Apply a rule up a tree - stop on success."""
    return do_one(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)


def sall(rule, fns=basic_fns):
    """Strategic all - apply rule to args."""
    op, new, children, leaf = map(fns.get, ('op', 'new', 'children', 'leaf'))

    def all_rl(expr):
        if leaf(expr):
            return expr
        else:
            args = map(rule, children(expr))
            return new(op(expr), *args)

    return all_rl
<|endoftext|>""" Rewrite Rules

DISCLAIMER: This module is experimental. The interface is subject to change.

A rule is a function that transforms one expression into another

    Rule :: Expr -> Expr

A strategy is a function that says how a rule should be applied to a syntax
tree. In general strategies take rules and produce a new rule

    Strategy :: [Rules], Other-stuff -> Rule

This allows developers to separate a mathematical transformation from the
algorithmic details of applying that transformation. The goal is to separate
the work of mathematical programming from algorithmic programming.

Submodules

strategies.rl         - some fundamental rules
strategies.core       - generic non-SymPy specific strategies
strategies.traverse   - strategies that traverse a SymPy tree
strategies.tools      - some conglomerate strategies that do depend on SymPy
"""

from . import rl
from . import traverse
from .rl import rm_id, unpack, flatten, sort, glom, distribute, rebuild
from .util import new
from .core import (
    condition, debug, chain, null_safe, do_one, exhaust, minimize, tryit)
from .tools import canon, typed
from . import branch

__all__ = [
    'rl',

    'traverse',

    'rm_id', 'unpack', 'flatten', 'sort', 'glom', 'distribute', 'rebuild',

    'new',

    'condition', 'debug', 'chain', 'null_safe', 'do_one', 'exhaust',
    'minimize', 'tryit',

    'canon', 'typed',

    'branch',
]
<|endoftext|>from . import rl
from .core import do_one, exhaust, switch
from .traverse import top_down


def subs(d, **kwargs):
    """ Full simultaneous exact substitution.

    Examples
    ========

    >>> from sympy.strategies.tools import subs
    >>> from sympy import Basic, S
    >>> mapping = {S(1): S(4), S(4): S(1), Basic(S(5)): Basic(S(6), S(7))}
    >>> expr = Basic(S(1), Basic(S(2), S(3)), Basic(S(4), Basic(S(5))))
    >>> subs(mapping)(expr)
    Basic(4, Basic(2, 3), Basic(1, Basic(6, 7)))
    """
    if d:
        return top_down(do_one(*map(rl.subs, *zip(*d.items()))), **kwargs)
    else:
        return lambda x: x


def canon(*rules, **kwargs):
    """ Strategy for canonicalization.

    Explanation
    ===========

    Apply each rule in a bottom_up fashion through the tree.
    Do each one in turn.
    Keep doing this until there is no change.
    """
    return exhaust(top_down(exhaust(do_one(*rules)), **kwargs))


def typed(ruletypes):
    """ Apply rules based on the expression type

    inputs:
        ruletypes -- a dict mapping {Type: rule}

    Examples
    ========

    >>> from sympy.strategies import rm_id, typed
    >>> from sympy import Add, Mul
    >>> rm_zeros = rm_id(lambda x: x==0)
    >>> rm_ones  = rm_id(lambda x: x==1)
    >>> remove_idents = typed({Add: rm_zeros, Mul: rm_ones})
    """
    return switch(type, ruletypes)
<|endoftext|>""" Generic SymPy-Independent Strategies """
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout


_S = TypeVar('_S')
_T = TypeVar('_T')


def identity(x: _T) -> _T:
    return x


def exhaust(rule: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """ Apply a rule repeatedly until it has no effect """
    def exhaustive_rl(expr: _T) -> _T:
        new, old = rule(expr), expr
        while new != old:
            new, old = rule(new), new
        return new
    return exhaustive_rl


def memoize(rule: Callable[[_S], _T]) -> Callable[[_S], _T]:
    """Memoized version of a rule

    Notes
    =====

    This cache can grow infinitely, so it is not recommended to use this
    than ``functools.lru_cache`` unless you need very heavy computation.
    """
    cache: dict[_S, _T] = {}

    def memoized_rl(expr: _S) -> _T:
        if expr in cache:
            return cache[expr]
        else:
            result = rule(expr)
            cache[expr] = result
            return result
    return memoized_rl


def condition(
    cond: Callable[[_T], bool], rule: Callable[[_T], _T]
) -> Callable[[_T], _T]:
    """ Only apply rule if condition is true """
    def conditioned_rl(expr: _T) -> _T:
        if cond(expr):
            return rule(expr)
        return expr
    return conditioned_rl


def chain(*rules: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """
    Compose a sequence of rules so that they apply to the expr sequentially
    """
    def chain_rl(expr: _T) -> _T:
        for rule in rules:
            expr = rule(expr)
        return expr
    return chain_rl


def debug(rule, file=None):
    """ Print out before and after expressions each time rule is used """
    if file is None:
        file = stdout

    def debug_rl(*args, **kwargs):
        expr = args[0]
        result = rule(*args, **kwargs)
        if result != expr:
            file.write("Rule: %s\n" % rule.__name__)
            file.write("In:   %s\nOut:  %s\n\n" % (expr, result))
        return result
    return debug_rl


def null_safe(rule: Callable[[_T], _T | None]) -> Callable[[_T], _T]:
    """ Return original expr if rule returns None """
    def null_safe_rl(expr: _T) -> _T:
        result = rule(expr)
        if result is None:
            return expr
        return result
    return null_safe_rl


def tryit(rule: Callable[[_T], _T], exception) -> Callable[[_T], _T]:
    """ Return original expr if rule raises exception """
    def try_rl(expr: _T) -> _T:
        try:
            return rule(expr)
        except exception:
            return expr
    return try_rl


def do_one(*rules: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """ Try each of the rules until one works. Then stop. """
    def do_one_rl(expr: _T) -> _T:
        for rl in rules:
            result = rl(expr)
            if result != expr:
                return result
        return expr
    return do_one_rl


def switch(
    key: Callable[[_S], _T],
    ruledict: Mapping[_T, Callable[[_S], _S]]
) -> Callable[[_S], _S]:
    """ Select a rule based on the result of key called on the function """
    def switch_rl(expr: _S) -> _S:
        rl = ruledict.get(key(expr), identity)
        return rl(expr)
    return switch_rl


# XXX Untyped default argument for minimize function
# where python requires SupportsRichComparison type
def _identity(x):
    return x


def minimize(
    *rules: Callable[[_S], _T],
    objective=_identity
) -> Callable[[_S], _T]:
    """ Select result of rules that minimizes objective

    >>> from sympy.strategies import minimize
    >>> inc = lambda x: x + 1
    >>> dec = lambda x: x - 1
    >>> rl = minimize(inc, dec)
    >>> rl(4)
    3

    >>> rl = minimize(inc, dec, objective=lambda x: -x)  # maximize
    >>> rl(4)
    5
    """
    def minrule(expr: _S) -> _T:
        return min([rule(expr) for rule in rules], key=objective)
    return minrule
<|endoftext|>""" Generic Rules for SymPy

This file assumes knowledge of Basic and little else.
"""
from sympy.utilities.iterables import sift
from .util import new


# Functions that create rules
def rm_id(isid, new=new):
    """ Create a rule to remove identities.

    isid - fn :: x -> Bool  --- whether or not this element is an identity.

    Examples
    ========

    >>> from sympy.strategies import rm_id
    >>> from sympy import Basic, S
    >>> remove_zeros = rm_id(lambda x: x==0)
    >>> remove_zeros(Basic(S(1), S(0), S(2)))
    Basic(1, 2)
    >>> remove_zeros(Basic(S(0), S(0))) # If only identities then we keep one
    Basic(0)

    See Also:
        unpack
    """
    def ident_remove(expr):
        """ Remove identities """
        ids = list(map(isid, expr.args))
        if sum(ids) == 0:           # No identities. Common case
            return expr
        elif sum(ids) != len(ids):  # there is at least one non-identity
            return new(expr.__class__,
                       *[arg for arg, x in zip(expr.args, ids) if not x])
        else:
            return new(expr.__class__, expr.args[0])

    return ident_remove


def glom(key, count, combine):
    """ Create a rule to conglomerate identical args.

    Examples
    ========

    >>> from sympy.strategies import glom
    >>> from sympy import Add
    >>> from sympy.abc import x

    >>> key     = lambda x: x.as_coeff_Mul()[1]
    >>> count   = lambda x: x.as_coeff_Mul()[0]
    >>> combine = lambda cnt, arg: cnt * arg
    >>> rl = glom(key, count, combine)

    >>> rl(Add(x, -x, 3*x, 2, 3, evaluate=False))
    3*x + 5

    Wait, how are key, count and combine supposed to work?

    >>> key(2*x)
    x
    >>> count(2*x)
    2
    >>> combine(2, x)
    2*x
    """
    def conglomerate(expr):
        """ Conglomerate together identical args x + x -> 2x """
        groups = sift(expr.args, key)
        counts = {k: sum(map(count, args)) for k, args in groups.items()}
        newargs = [combine(cnt, mat) for mat, cnt in counts.items()]
        if set(newargs) != set(expr.args):
            return new(type(expr), *newargs)
        else:
            return expr

    return conglomerate


def sort(key, new=new):
    """ Create a rule to sort by a key function.

    Examples
    ========

    >>> from sympy.strategies import sort
    >>> from sympy import Basic, S
    >>> sort_rl = sort(str)
    >>> sort_rl(Basic(S(3), S(1), S(2)))
    Basic(1, 2, 3)
    """

    def sort_rl(expr):
        return new(expr.__class__, *sorted(expr.args, key=key))
    return sort_rl


def distribute(A, B):
    """ Turns an A containing Bs into a B of As

    where A, B are container types

    >>> from sympy.strategies import distribute
    >>> from sympy import Add, Mul, symbols
    >>> x, y = symbols('x,y')
    >>> dist = distribute(Mul, Add)
    >>> expr = Mul(2, x+y, evaluate=False)
    >>> expr
    2*(x + y)
    >>> dist(expr)
    2*x + 2*y
    """

    def distribute_rl(expr):
        for i, arg in enumerate(expr.args):
            if isinstance(arg, B):
                first, b, tail = expr.args[:i], expr.args[i], expr.args[i + 1:]
                return B(*[A(*(first + (arg,) + tail)) for arg in b.args])
        return expr
    return distribute_rl


def subs(a, b):
    """ Replace expressions exactly """
    def subs_rl(expr):
        if expr == a:
            return b
        else:
            return expr
    return subs_rl


# Functions that are rules
def unpack(expr):
    """ Rule to unpack singleton args

    >>> from sympy.strategies import unpack
    >>> from sympy import Basic, S
    >>> unpack(Basic(S(2)))
    2
    """
    if len(expr.args) == 1:
        return expr.args[0]
    else:
        return expr


def flatten(expr, new=new):
    """ Flatten T(a, b, T(c, d), T2(e)) to T(a, b, c, d, T2(e)) """
    cls = expr.__class__
    args = []
    for arg in expr.args:
        if arg.__class__ == cls:
            args.extend(arg.args)
        else:
            args.append(arg)
    return new(expr.__class__, *args)


def rebuild(expr):
    """ Rebuild a SymPy tree.

    Explanation
    ===========

    This function recursively calls constructors in the expression tree.
    This forces canonicalization and removes ugliness introduced by the use of
    Basic.__new__
    """
    if expr.is_Atom:
        return expr
    else:
        return expr.func(*list(map(rebuild, expr.args)))
<|endoftext|>from functools import partial
from sympy.strategies import chain, minimize
from sympy.strategies.core import identity
import sympy.strategies.branch as branch
from sympy.strategies.branch import yieldify


def treeapply(tree, join, leaf=identity):
    """ Apply functions onto recursive containers (tree).

    Explanation
    ===========

    join - a dictionary mapping container types to functions
      e.g. ``{list: minimize, tuple: chain}``

    Keys are containers/iterables.  Values are functions [a] -> a.

    Examples
    ========

    >>> from sympy.strategies.tree import treeapply
    >>> tree = [(3, 2), (4, 1)]
    >>> treeapply(tree, {list: max, tuple: min})
    2

    >>> add = lambda *args: sum(args)
    >>> def mul(*args):
    ...     total = 1
    ...     for arg in args:
    ...         total *= arg
    ...     return total
    >>> treeapply(tree, {list: mul, tuple: add})
    25
    """
    for typ in join:
        if isinstance(tree, typ):
            return join[typ](*map(partial(treeapply, join=join, leaf=leaf),
                                  tree))
    return leaf(tree)


def greedy(tree, objective=identity, **kwargs):
    """ Execute a strategic tree.  Select alternatives greedily

    Trees
    -----

    Nodes in a tree can be either

    function - a leaf
    list     - a selection among operations
    tuple    - a sequence of chained operations

    Textual examples
    ----------------

    Text: Run f, then run g, e.g. ``lambda x: g(f(x))``
    Code: ``(f, g)``

    Text: Run either f or g, whichever minimizes the objective
    Code: ``[f, g]``

    Textx: Run either f or g, whichever is better, then run h
    Code: ``([f, g], h)``

    Text: Either expand then simplify or try factor then foosimp. Finally print
    Code: ``([(expand, simplify), (factor, foosimp)], print)``

    Objective
    ---------

    "Better" is determined by the objective keyword.  This function makes
    choices to minimize the objective.  It defaults to the identity.

    Examples
    ========

    >>> from sympy.strategies.tree import greedy
    >>> inc    = lambda x: x + 1
    >>> dec    = lambda x: x - 1
    >>> double = lambda x: 2*x

    >>> tree = [inc, (dec, double)] # either inc or dec-then-double
    >>> fn = greedy(tree)
    >>> fn(4)  # lowest value comes from the inc
    5
    >>> fn(1)  # lowest value comes from dec then double
    0

    This function selects between options in a tuple.  The result is chosen
    that minimizes the objective function.

    >>> fn = greedy(tree, objective=lambda x: -x)  # maximize
    >>> fn(4)  # highest value comes from the dec then double
    6
    >>> fn(1)  # highest value comes from the inc
    2

    Greediness
    ----------

    This is a greedy algorithm.  In the example:

        ([a, b], c)  # do either a or b, then do c

    the choice between running ``a`` or ``b`` is made without foresight to c
    """
    optimize = partial(minimize, objective=objective)
    return treeapply(tree, {list: optimize, tuple: chain}, **kwargs)


def allresults(tree, leaf=yieldify):
    """ Execute a strategic tree.  Return all possibilities.

    Returns a lazy iterator of all possible results

    Exhaustiveness
    --------------

    This is an exhaustive algorithm.  In the example

        ([a, b], [c, d])

    All of the results from

        (a, c), (b, c), (a, d), (b, d)

    are returned.  This can lead to combinatorial blowup.

    See sympy.strategies.greedy for details on input
    """
    return treeapply(tree, {list: branch.multiplex, tuple: branch.chain},
                     leaf=leaf)


def brute(tree, objective=identity, **kwargs):
    return lambda expr: min(tuple(allresults(tree, **kwargs)(expr)),
                            key=objective)
<|endoftext|>""" Branching Strategies to Traverse a Tree """
from itertools import product
from sympy.strategies.util import basic_fns
from .core import chain, identity, do_one


def top_down(brule, fns=basic_fns):
    """ Apply a rule down a tree running it on the top nodes first """
    return chain(do_one(brule, identity),
                 lambda expr: sall(top_down(brule, fns), fns)(expr))


def sall(brule, fns=basic_fns):
    """ Strategic all - apply rule to args """
    op, new, children, leaf = map(fns.get, ('op', 'new', 'children', 'leaf'))

    def all_rl(expr):
        if leaf(expr):
            yield expr
        else:
            myop = op(expr)
            argss = product(*map(brule, children(expr)))
            for args in argss:
                yield new(myop, *args)
    return all_rl
<|endoftext|>from . import traverse
from .core import (
    condition, debug, multiplex, exhaust, notempty,
    chain, onaction, sfilter, yieldify, do_one, identity)
from .tools import canon

__all__ = [
    'traverse',

    'condition', 'debug', 'multiplex', 'exhaust', 'notempty', 'chain',
    'onaction', 'sfilter', 'yieldify', 'do_one', 'identity',

    'canon',
]
<|endoftext|>from .core import exhaust, multiplex
from .traverse import top_down


def canon(*rules):
    """ Strategy for canonicalization

    Apply each branching rule in a top-down fashion through the tree.
    Multiplex through all branching rule traversals
    Keep doing this until there is no change.
    """
    return exhaust(multiplex(*map(top_down, rules)))
<|endoftext|>""" Generic SymPy-Independent Strategies """


def identity(x):
    yield x


def exhaust(brule):
    """ Apply a branching rule repeatedly until it has no effect """
    def exhaust_brl(expr):
        seen = {expr}
        for nexpr in brule(expr):
            if nexpr not in seen:
                seen.add(nexpr)
                yield from exhaust_brl(nexpr)
        if seen == {expr}:
            yield expr
    return exhaust_brl


def onaction(brule, fn):
    def onaction_brl(expr):
        for result in brule(expr):
            if result != expr:
                fn(brule, expr, result)
            yield result
    return onaction_brl


def debug(brule, file=None):
    """ Print the input and output expressions at each rule application """
    if not file:
        from sys import stdout
        file = stdout

    def write(brl, expr, result):
        file.write("Rule: %s\n" % brl.__name__)
        file.write("In: %s\nOut: %s\n\n" % (expr, result))

    return onaction(brule, write)


def multiplex(*brules):
    """ Multiplex many branching rules into one """
    def multiplex_brl(expr):
        seen = set()
        for brl in brules:
            for nexpr in brl(expr):
                if nexpr not in seen:
                    seen.add(nexpr)
                    yield nexpr
    return multiplex_brl


def condition(cond, brule):
    """ Only apply branching rule if condition is true """
    def conditioned_brl(expr):
        if cond(expr):
            yield from brule(expr)
        else:
            pass
    return conditioned_brl


def sfilter(pred, brule):
    """ Yield only those results which satisfy the predicate """
    def filtered_brl(expr):
        yield from filter(pred, brule(expr))
    return filtered_brl


def notempty(brule):
    def notempty_brl(expr):
        yielded = False
        for nexpr in brule(expr):
            yielded = True
            yield nexpr
        if not yielded:
            yield expr
    return notempty_brl


def do_one(*brules):
    """ Execute one of the branching rules """
    def do_one_brl(expr):
        yielded = False
        for brl in brules:
            for nexpr in brl(expr):
                yielded = True
                yield nexpr
            if yielded:
                return
    return do_one_brl


def chain(*brules):
    """
    Compose a sequence of brules so that they apply to the expr sequentially
    """
    def chain_brl(expr):
        if not brules:
            yield expr
            return

        head, tail = brules[0], brules[1:]
        for nexpr in head(expr):
            yield from chain(*tail)(nexpr)

    return chain_brl


def yieldify(rl):
    """ Turn a rule into a branching rule """
    def brl(expr):
        yield rl(expr)
    return brl
<|endoftext|># sympy.external.ntheory
#
# This module provides pure Python implementations of some number theory
# functions that are alternately used from gmpy2 if it is installed.

import math

import mpmath.libmp as mlib


_small_trailing = [0] * 256
for j in range(1, 8):
    _small_trailing[1 << j :: 1 << (j + 1)] = [j] * (1 << (7 - j))


def bit_scan1(x, n=0):
    if not x:
        return
    x = abs(x >> n)
    low_byte = x & 0xFF
    if low_byte:
        return _small_trailing[low_byte] + n

    t = 8 + n
    x >>= 8
    # 2**m is quick for z up through 2**30
    z = x.bit_length() - 1
    if x == 1 << z:
        return z + t

    if z < 300:
        # fixed 8-byte reduction
        while not x & 0xFF:
            x >>= 8
            t += 8
    else:
        # binary reduction important when there might be a large
        # number of trailing 0s
        p = z >> 1
        while not x & 0xFF:
            while x & ((1 << p) - 1):
                p >>= 1
            x >>= p
            t += p
    return t + _small_trailing[x & 0xFF]


def bit_scan0(x, n=0):
    return bit_scan1(x + (1 << n), n)


def remove(x, f):
    if f < 2:
        raise ValueError("factor must be > 1")
    if x == 0:
        return 0, 0
    if f == 2:
        b = bit_scan1(x)
        return x >> b, b
    m = 0
    y, rem = divmod(x, f)
    while not rem:
        x = y
        m += 1
        if m > 5:
            pow_list = [f**2]
            while pow_list:
                _f = pow_list[-1]
                y, rem = divmod(x, _f)
                if not rem:
                    m += 1 << len(pow_list)
                    x = y
                    pow_list.append(_f**2)
                else:
                    pow_list.pop()
        y, rem = divmod(x, f)
    return x, m


def factorial(x):
    """Return x!."""
    return int(mlib.ifac(int(x)))


def sqrt(x):
    """Integer square root of x."""
    return int(mlib.isqrt(int(x)))


def sqrtrem(x):
    """Integer square root of x and remainder."""
    s, r = mlib.sqrtrem(int(x))
    return (int(s), int(r))


gcd = math.gcd
lcm = math.lcm


def _sign(n):
    if n < 0:
        return -1, -n
    return 1, n


def gcdext(a, b):
    if not a or not b:
        g = abs(a) or abs(b)
        if not g:
            return (0, 0, 0)
        return (g, a // g, b // g)

    x_sign, a = _sign(a)
    y_sign, b = _sign(b)
    x, r = 1, 0
    y, s = 0, 1

    while b:
        q, c = divmod(a, b)
        a, b = b, c
        x, r = r, x - q*r
        y, s = s, y - q*s

    return (a, x * x_sign, y * y_sign)


def is_square(x):
    """Return True if x is a square number."""
    if x < 0:
        return False

    # Note that the possible values of y**2 % n for a given n are limited.
    # For example, when n=4, y**2 % n can only take 0 or 1.
    # In other words, if x % 4 is 2 or 3, then x is not a square number.
    # Mathematically, it determines if it belongs to the set {y**2 % n},
    # but implementationally, it can be realized as a logical conjunction
    # with an n-bit integer.
    # see https://mersenneforum.org/showpost.php?p=110896
    # def magic(n):
    #     s = {y**2 % n for y in range(n)}
    #     s = set(range(n)) - s
    #     return sum(1 << bit for bit in s)
    # >>> print(hex(magic(128)))
    # 0xfdfdfdedfdfdfdecfdfdfdedfdfcfdec
    # >>> print(hex(magic(99)))
    # 0x5f6f9ffb6fb7ddfcb75befdec
    # >>> print(hex(magic(91)))
    # 0x6fd1bfcfed5f3679d3ebdec
    # >>> print(hex(magic(85)))
    # 0xdef9ae771ffe3b9d67dec
    if 0xfdfdfdedfdfdfdecfdfdfdedfdfcfdec & (1 << (x & 127)):
        return False  # e.g. 2, 3
    m = x % 765765 # 765765 = 99 * 91 * 85
    if 0x5f6f9ffb6fb7ddfcb75befdec & (1 << (m % 99)):
        return False  # e.g. 17, 68
    if 0x6fd1bfcfed5f3679d3ebdec & (1 << (m % 91)):
        return False  # e.g. 97, 388
    if 0xdef9ae771ffe3b9d67dec & (1 << (m % 85)):
        return False  # e.g. 793, 1408
    return mlib.sqrtrem(int(x))[1] == 0


def invert(x, m):
    """Modular inverse of x modulo m.

    Returns y such that x*y == 1 mod m.

    Uses ``math.pow`` but reproduces the behaviour of ``gmpy2.invert``
    which raises ZeroDivisionError if no inverse exists.
    """
    try:
        return pow(x, -1, m)
    except ValueError:
        raise ZeroDivisionError("invert() no inverse exists")


def legendre(x, y):
    """Legendre symbol (x / y).

    Following the implementation of gmpy2,
    the error is raised only when y is an even number.
    """
    if y <= 0 or not y % 2:
        raise ValueError("y should be an odd prime")
    x %= y
    if not x:
        return 0
    if pow(x, (y - 1) // 2, y) == 1:
        return 1
    return -1


def jacobi(x, y):
    """Jacobi symbol (x / y)."""
    if y <= 0 or not y % 2:
        raise ValueError("y should be an odd positive integer")
    x %= y
    if not x:
        return int(y == 1)
    if y == 1 or x == 1:
        return 1
    if gcd(x, y) != 1:
        return 0
    j = 1
    while x != 0:
        while x % 2 == 0 and x > 0:
            x >>= 1
            if y % 8 in [3, 5]:
                j = -j
        x, y = y, x
        if x % 4 == y % 4 == 3:
            j = -j
        x %= y
    return j


def kronecker(x, y):
    """Kronecker symbol (x / y)."""
    if gcd(x, y) != 1:
        return 0
    if y == 0:
        return 1
    sign = -1 if y < 0 and x < 0 else 1
    y = abs(y)
    s = bit_scan1(y)
    y >>= s
    if s % 2 and x % 8 in [3, 5]:
        sign = -sign
    return sign * jacobi(x, y)


def iroot(y, n):
    if y < 0:
        raise ValueError("y must be nonnegative")
    if n < 1:
        raise ValueError("n must be positive")
    if y in (0, 1):
        return y, True
    if n == 1:
        return y, True
    if n == 2:
        x, rem = mlib.sqrtrem(y)
        return int(x), not rem
    if n >= y.bit_length():
        return 1, False
    # Get initial estimate for Newton's method. Care must be taken to
    # avoid overflow
    try:
        guess = int(y**(1./n) + 0.5)
    except OverflowError:
        exp = math.log2(y)/n
        if exp > 53:
            shift = int(exp - 53)
            guess = int(2.0**(exp - shift) + 1) << shift
        else:
            guess = int(2.0**exp)
    if guess > 2**50:
        # Newton iteration
        xprev, x = -1, guess
        while 1:
            t = x**(n - 1)
            xprev, x = x, ((n - 1)*x + y//t)//n
            if abs(x - xprev) < 2:
                break
    else:
        x = guess
    # Compensate
    t = x**n
    while t < y:
        x += 1
        t = x**n
    while t > y:
        x -= 1
        t = x**n
    return x, t == y


def is_fermat_prp(n, a):
    if a < 2:
        raise ValueError("is_fermat_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_fermat_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_fermat_prp() requires gcd(n,a) == 1")
    return pow(a, n - 1, n) == 1


def is_euler_prp(n, a):
    if a < 2:
        raise ValueError("is_euler_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_euler_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_euler_prp() requires gcd(n,a) == 1")
    return pow(a, n >> 1, n) == jacobi(a, n) % n


def _is_strong_prp(n, a):
    s = bit_scan1(n - 1)
    a = pow(a, n >> s, n)
    if a == 1 or a == n - 1:
        return True
    for _ in range(s - 1):
        a = pow(a, 2, n)
        if a == n - 1:
            return True
        if a == 1:
            return False
    return False


def is_strong_prp(n, a):
    if a < 2:
        raise ValueError("is_strong_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_strong_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_strong_prp() requires gcd(n,a) == 1")
    return _is_strong_prp(n, a)


def _lucas_sequence(n, P, Q, k):
    r"""Return the modular Lucas sequence (U_k, V_k, Q_k).

    Explanation
    ===========

    Given a Lucas sequence defined by P, Q, returns the kth values for
    U and V, along with Q^k, all modulo n. This is intended for use with
    possibly very large values of n and k, where the combinatorial functions
    would be completely unusable.

    .. math ::
        U_k = \begin{cases}
             0 & \text{if } k = 0\\
             1 & \text{if } k = 1\\
             PU_{k-1} - QU_{k-2} & \text{if } k > 1
        \end{cases}\\
        V_k = \begin{cases}
             2 & \text{if } k = 0\\
             P & \text{if } k = 1\\
             PV_{k-1} - QV_{k-2} & \text{if } k > 1
        \end{cases}

    The modular Lucas sequences are used in numerous places in number theory,
    especially in the Lucas compositeness tests and the various n + 1 proofs.

    Parameters
    ==========

    n : int
        n is an odd number greater than or equal to 3
    P : int
    Q : int
        D determined by D = P**2 - 4*Q is non-zero
    k : int
        k is a nonnegative integer

    Returns
    =======

    U, V, Qk : (int, int, int)
        `(U_k \bmod{n}, V_k \bmod{n}, Q^k \bmod{n})`

    Examples
    ========

    >>> from sympy.external.ntheory import _lucas_sequence
    >>> N = 10**2000 + 4561
    >>> sol = U, V, Qk = _lucas_sequence(N, 3, 1, N//2); sol
    (0, 2, 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas_sequence

    """
    if k == 0:
        return (0, 2, 1)
    D = P**2 - 4*Q
    U = 1
    V = P
    Qk = Q % n
    if Q == 1:
        # Optimization for extra strong tests.
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2) % n
            if b == "1":
                U, V = U*P + V, V*P + U*D
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = U >> 1, V >> 1
    elif P == 1 and Q == -1:
        # Small optimization for 50% of Selfridge parameters.
        for b in bin(k)[3:]:
            U = (U*V) % n
            if Qk == 1:
                V = (V*V - 2) % n
            else:
                V = (V*V + 2) % n
                Qk = 1
            if b == "1":
                # new_U = (U + V) // 2
                # new_V = (5*U + V) // 2 = 2*U + new_U
                U, V  = U + V, U << 1
                if U & 1:
                    U += n
                U >>= 1
                V += U
                Qk = -1
        Qk %= n
    elif P == 1:
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2*Qk) % n
            Qk *= Qk
            if b == "1":
                # new_U = (U + V) // 2
                # new_V = new_U - 2*Q*U
                U, V  = U + V, (Q*U) << 1
                if U & 1:
                    U += n
                U >>= 1
                V = U - V
                Qk *= Q
            Qk %= n
    else:
        # The general case with any P and Q.
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2*Qk) % n
            Qk *= Qk
            if b == "1":
                U, V = U*P + V, V*P + U*D
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = U >> 1, V >> 1
                Qk *= Q
            Qk %= n
    return (U % n, V % n, Qk)


def is_fibonacci_prp(n, p, q):
    d = p**2 - 4*q
    if d == 0 or p <= 0 or q not in [1, -1]:
        raise ValueError("invalid values for p,q in is_fibonacci_prp()")
    if n < 1:
        raise ValueError("is_fibonacci_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _lucas_sequence(n, p, q, n)[1] == p % n


def is_lucas_prp(n, p, q):
    d = p**2 - 4*q
    if d == 0:
        raise ValueError("invalid values for p,q in is_lucas_prp()")
    if n < 1:
        raise ValueError("is_lucas_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    if gcd(n, q*d) not in [1, n]:
        raise ValueError("is_lucas_prp() requires gcd(n,2*q*D) == 1")
    return _lucas_sequence(n, p, q, n - jacobi(d, n))[0] == 0


def _is_selfridge_prp(n):
    """Lucas compositeness test with the Selfridge parameters for n.

    Explanation
    ===========

    The Lucas compositeness test checks whether n is a prime number.
    The test can be run with arbitrary parameters ``P`` and ``Q``, which also change the performance of the test.
    So, which parameters are most effective for running the Lucas compositeness test?
    As an algorithm for determining ``P`` and ``Q``, Selfridge proposed method A [1]_ page 1401
    (Since two methods were proposed, referred to simply as A and B in the paper,
    we will refer to one of them as "method A").

    method A fixes ``P = 1``. Then, ``D`` defined by ``D = P**2 - 4Q`` is varied from 5, -7, 9, -11, 13, and so on,
    with the first ``D`` being ``jacobi(D, n) == -1``. Once ``D`` is determined,
    ``Q`` is determined to be ``(P**2 - D)//4``.

    References
    ==========

    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf

    """
    for D in range(5, 1_000_000, 2):
        if D & 2: # if D % 4 == 3
            D = -D
        j = jacobi(D, n)
        if j == -1:
            return _lucas_sequence(n, 1, (1-D) // 4, n + 1)[0] == 0
        if j == 0 and D % n:
            return False
        # When j == -1 is hard to find, suspect a square number
        if D == 13 and is_square(n):
            return False
    raise ValueError("appropriate value for D cannot be found in is_selfridge_prp()")


def is_selfridge_prp(n):
    if n < 1:
        raise ValueError("is_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_selfridge_prp(n)


def is_strong_lucas_prp(n, p, q):
    D = p**2 - 4*q
    if D == 0:
        raise ValueError("invalid values for p,q in is_strong_lucas_prp()")
    if n < 1:
        raise ValueError("is_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    if gcd(n, q*D) not in [1, n]:
        raise ValueError("is_strong_lucas_prp() requires gcd(n,2*q*D) == 1")
    j = jacobi(D, n)
    s = bit_scan1(n - j)
    U, V, Qk = _lucas_sequence(n, p, q, (n - j) >> s)
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V = (V*V - 2*Qk) % n
        if V == 0:
            return True
        Qk = pow(Qk, 2, n)
    return False


def _is_strong_selfridge_prp(n):
    for D in range(5, 1_000_000, 2):
        if D & 2: # if D % 4 == 3
            D = -D
        j = jacobi(D, n)
        if j == -1:
            s = bit_scan1(n + 1)
            U, V, Qk = _lucas_sequence(n, 1, (1-D) // 4, (n + 1) >> s)
            if U == 0 or V == 0:
                return True
            for _ in range(s - 1):
                V = (V*V - 2*Qk) % n
                if V == 0:
                    return True
                Qk = pow(Qk, 2, n)
            return False
        if j == 0 and D % n:
            return False
        # When j == -1 is hard to find, suspect a square number
        if D == 13 and is_square(n):
            return False
    raise ValueError("appropriate value for D cannot be found in is_strong_selfridge_prp()")


def is_strong_selfridge_prp(n):
    if n < 1:
        raise ValueError("is_strong_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_selfridge_prp(n)


def is_bpsw_prp(n):
    if n < 1:
        raise ValueError("is_bpsw_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_prp(n, 2) and _is_selfridge_prp(n)


def is_strong_bpsw_prp(n):
    if n < 1:
        raise ValueError("is_strong_bpsw_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_prp(n, 2) and _is_strong_selfridge_prp(n)
<|endoftext|>"""
PythonMPQ: Rational number type based on Python integers.

This class is intended as a pure Python fallback for when gmpy2 is not
installed. If gmpy2 is installed then its mpq type will be used instead. The
mpq type is around 20x faster. We could just use the stdlib Fraction class
here but that is slower:

    from fractions import Fraction
    from sympy.external.pythonmpq import PythonMPQ
    nums = range(1000)
    dens = range(5, 1005)
    rats = [Fraction(n, d) for n, d in zip(nums, dens)]
    sum(rats) # <--- 24 milliseconds
    rats = [PythonMPQ(n, d) for n, d in zip(nums, dens)]
    sum(rats) # <---  7 milliseconds

Both mpq and Fraction have some awkward features like the behaviour of
division with // and %:

    >>> from fractions import Fraction
    >>> Fraction(2, 3) % Fraction(1, 4)
    1/6

For the QQ domain we do not want this behaviour because there should be no
remainder when dividing rational numbers. SymPy does not make use of this
aspect of mpq when gmpy2 is installed. Since this class is a fallback for that
case we do not bother implementing e.g. __mod__ so that we can be sure we
are not using it when gmpy2 is installed either.
"""

from __future__ import annotations
import operator
from math import gcd
from decimal import Decimal
from fractions import Fraction
import sys
from typing import Type


# Used for __hash__
_PyHASH_MODULUS = sys.hash_info.modulus
_PyHASH_INF = sys.hash_info.inf


class PythonMPQ:
    """Rational number implementation that is intended to be compatible with
    gmpy2's mpq.

    Also slightly faster than fractions.Fraction.

    PythonMPQ should be treated as immutable although no effort is made to
    prevent mutation (since that might slow down calculations).
    """
    __slots__ = ('numerator', 'denominator')

    def __new__(cls, numerator, denominator=None):
        """Construct PythonMPQ with gcd computation and checks"""
        if denominator is not None:
            #
            # PythonMPQ(n, d): require n and d to be int and d != 0
            #
            if isinstance(numerator, int) and isinstance(denominator, int):
                # This is the slow part:
                divisor = gcd(numerator, denominator)
                numerator //= divisor
                denominator //= divisor
                return cls._new_check(numerator, denominator)
        else:
            #
            # PythonMPQ(q)
            #
            # Here q can be PythonMPQ, int, Decimal, float, Fraction or str
            #
            if isinstance(numerator, int):
                return cls._new(numerator, 1)
            elif isinstance(numerator, PythonMPQ):
                return cls._new(numerator.numerator, numerator.denominator)

            # Let Fraction handle Decimal/float conversion and str parsing
            if isinstance(numerator, (Decimal, float, str)):
                numerator = Fraction(numerator)
            if isinstance(numerator, Fraction):
                return cls._new(numerator.numerator, numerator.denominator)
        #
        # Reject everything else. This is more strict than mpq which allows
        # things like mpq(Fraction, Fraction) or mpq(Decimal, any). The mpq
        # behaviour is somewhat inconsistent so we choose to accept only a
        # more strict subset of what mpq allows.
        #
        raise TypeError("PythonMPQ() requires numeric or string argument")

    @classmethod
    def _new_check(cls, numerator, denominator):
        """Construct PythonMPQ, check divide by zero and canonicalize signs"""
        if not denominator:
            raise ZeroDivisionError(f'Zero divisor {numerator}/{denominator}')
        elif denominator < 0:
            numerator = -numerator
            denominator = -denominator
        return cls._new(numerator, denominator)

    @classmethod
    def _new(cls, numerator, denominator):
        """Construct PythonMPQ efficiently (no checks)"""
        obj = super().__new__(cls)
        obj.numerator = numerator
        obj.denominator = denominator
        return obj

    def __int__(self):
        """Convert to int (truncates towards zero)"""
        p, q = self.numerator, self.denominator
        if p < 0:
            return -(-p//q)
        return p//q

    def __float__(self):
        """Convert to float (approximately)"""
        return self.numerator / self.denominator

    def __bool__(self):
        """True/False if nonzero/zero"""
        return bool(self.numerator)

    def __eq__(self, other):
        """Compare equal with PythonMPQ, int, float, Decimal or Fraction"""
        if isinstance(other, PythonMPQ):
            return (self.numerator == other.numerator
                and self.denominator == other.denominator)
        elif isinstance(other, self._compatible_types):
            return self.__eq__(PythonMPQ(other))
        else:
            return NotImplemented

    def __hash__(self):
        """hash - same as mpq/Fraction"""
        try:
            dinv = pow(self.denominator, -1, _PyHASH_MODULUS)
        except ValueError:
            hash_ = _PyHASH_INF
        else:
            hash_ = hash(hash(abs(self.numerator)) * dinv)
        result = hash_ if self.numerator >= 0 else -hash_
        return -2 if result == -1 else result

    def __reduce__(self):
        """Deconstruct for pickling"""
        return type(self), (self.numerator, self.denominator)

    def __str__(self):
        """Convert to string"""
        if self.denominator != 1:
            return f"{self.numerator}/{self.denominator}"
        else:
            return f"{self.numerator}"

    def __repr__(self):
        """Convert to string"""
        return f"MPQ({self.numerator},{self.denominator})"

    def _cmp(self, other, op):
        """Helper for lt/le/gt/ge"""
        if not isinstance(other, self._compatible_types):
            return NotImplemented
        lhs = self.numerator * other.denominator
        rhs = other.numerator * self.denominator
        return op(lhs, rhs)

    def __lt__(self, other):
        """self < other"""
        return self._cmp(other, operator.lt)

    def __le__(self, other):
        """self <= other"""
        return self._cmp(other, operator.le)

    def __gt__(self, other):
        """self > other"""
        return self._cmp(other, operator.gt)

    def __ge__(self, other):
        """self >= other"""
        return self._cmp(other, operator.ge)

    def __abs__(self):
        """abs(q)"""
        return self._new(abs(self.numerator), self.denominator)

    def __pos__(self):
        """+q"""
        return self

    def __neg__(self):
        """-q"""
        return self._new(-self.numerator, self.denominator)

    def __add__(self, other):
        """q1 + q2"""
        if isinstance(other, PythonMPQ):
            #
            # This is much faster than the naive method used in the stdlib
            # fractions module. Not sure where this method comes from
            # though...
            #
            # Compare timings for something like:
            #   nums = range(1000)
            #   rats = [PythonMPQ(n, d) for n, d in zip(nums[:-5], nums[5:])]
            #   sum(rats) # <-- time this
            #
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            g = gcd(aq, bq)
            if g == 1:
                p = ap*bq + aq*bp
                q = bq*aq
            else:
                q1, q2 = aq//g, bq//g
                p, q = ap*q2 + bp*q1, q1*q2
                g2 = gcd(p, g)
                p, q = (p // g2), q * (g // g2)

        elif isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
        else:
            return NotImplemented

        return self._new(p, q)

    def __radd__(self, other):
        """z1 + q2"""
        if isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
            return self._new(p, q)
        else:
            return NotImplemented

    def __sub__(self ,other):
        """q1 - q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            g = gcd(aq, bq)
            if g == 1:
                p = ap*bq - aq*bp
                q = bq*aq
            else:
                q1, q2 = aq//g, bq//g
                p, q = ap*q2 - bp*q1, q1*q2
                g2 = gcd(p, g)
                p, q = (p // g2), q * (g // g2)
        elif isinstance(other, int):
            p = self.numerator - self.denominator*other
            q = self.denominator
        else:
            return NotImplemented

        return self._new(p, q)

    def __rsub__(self, other):
        """z1 - q2"""
        if isinstance(other, int):
            p = self.denominator * other - self.numerator
            q = self.denominator
            return self._new(p, q)
        else:
            return NotImplemented

    def __mul__(self, other):
        """q1 * q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            x1 = gcd(ap, bq)
            x2 = gcd(bp, aq)
            p, q = ((ap//x1)*(bp//x2), (aq//x2)*(bq//x1))
        elif isinstance(other, int):
            x = gcd(other, self.denominator)
            p = self.numerator*(other//x)
            q = self.denominator//x
        else:
            return NotImplemented

        return self._new(p, q)

    def __rmul__(self, other):
        """z1 * q2"""
        if isinstance(other, int):
            x = gcd(self.denominator, other)
            p = self.numerator*(other//x)
            q = self.denominator//x
            return self._new(p, q)
        else:
            return NotImplemented

    def __pow__(self, exp):
        """q ** z"""
        p, q = self.numerator, self.denominator

        if exp < 0:
            p, q, exp = q, p, -exp

        return self._new_check(p**exp, q**exp)

    def __truediv__(self, other):
        """q1 / q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            x1 = gcd(ap, bp)
            x2 = gcd(bq, aq)
            p, q = ((ap//x1)*(bq//x2), (aq//x2)*(bp//x1))
        elif isinstance(other, int):
            x = gcd(other, self.numerator)
            p = self.numerator//x
            q = self.denominator*(other//x)
        else:
            return NotImplemented

        return self._new_check(p, q)

    def __rtruediv__(self, other):
        """z / q"""
        if isinstance(other, int):
            x = gcd(self.numerator, other)
            p = self.denominator*(other//x)
            q = self.numerator//x
            return self._new_check(p, q)
        else:
            return NotImplemented

    _compatible_types: tuple[Type, ...] = ()

#
# These are the types that PythonMPQ will interoperate with for operations
# and comparisons such as ==, + etc. We define this down here so that we can
# include PythonMPQ in the list as well.
#
PythonMPQ._compatible_types = (PythonMPQ, int, Decimal, Fraction)
<|endoftext|>"""Tools to assist importing optional external modules."""

import sys
import re

# Override these in the module to change the default warning behavior.
# For example, you might set both to False before running the tests so that
# warnings are not printed to the console, or set both to True for debugging.

WARN_NOT_INSTALLED = None  # Default is False
WARN_OLD_VERSION = None  # Default is True


def __sympy_debug():
    # helper function from sympy/__init__.py
    # We don't just import SYMPY_DEBUG from that file because we don't want to
    # import all of SymPy just to use this module.
    import os
    debug_str = os.getenv('SYMPY_DEBUG', 'False')
    if debug_str in ('True', 'False'):
        return eval(debug_str)
    else:
        raise RuntimeError("unrecognized value for SYMPY_DEBUG: %s" %
                           debug_str)

if __sympy_debug():
    WARN_OLD_VERSION = True
    WARN_NOT_INSTALLED = True


_component_re = re.compile(r'(\d+ | [a-z]+ | \.)', re.VERBOSE)

def version_tuple(vstring):
    # Parse a version string to a tuple e.g. '1.2' -> (1, 2)
    # Simplified from distutils.version.LooseVersion which was deprecated in
    # Python 3.10.
    components = []
    for x in _component_re.split(vstring):
        if x and x != '.':
            try:
                x = int(x)
            except ValueError:
                pass
            components.append(x)
    return tuple(components)


def import_module(module, min_module_version=None, min_python_version=None,
        warn_not_installed=None, warn_old_version=None,
        module_version_attr='__version__', module_version_attr_call_args=None,
        import_kwargs={}, catch=()):
    """
    Import and return a module if it is installed.

    If the module is not installed, it returns None.

    A minimum version for the module can be given as the keyword argument
    min_module_version.  This should be comparable against the module version.
    By default, module.__version__ is used to get the module version.  To
    override this, set the module_version_attr keyword argument.  If the
    attribute of the module to get the version should be called (e.g.,
    module.version()), then set module_version_attr_call_args to the args such
    that module.module_version_attr(*module_version_attr_call_args) returns the
    module's version.

    If the module version is less than min_module_version using the Python <
    comparison, None will be returned, even if the module is installed. You can
    use this to keep from importing an incompatible older version of a module.

    You can also specify a minimum Python version by using the
    min_python_version keyword argument.  This should be comparable against
    sys.version_info.

    If the keyword argument warn_not_installed is set to True, the function will
    emit a UserWarning when the module is not installed.

    If the keyword argument warn_old_version is set to True, the function will
    emit a UserWarning when the library is installed, but cannot be imported
    because of the min_module_version or min_python_version options.

    Note that because of the way warnings are handled, a warning will be
    emitted for each module only once.  You can change the default warning
    behavior by overriding the values of WARN_NOT_INSTALLED and WARN_OLD_VERSION
    in sympy.external.importtools.  By default, WARN_NOT_INSTALLED is False and
    WARN_OLD_VERSION is True.

    This function uses __import__() to import the module.  To pass additional
    options to __import__(), use the import_kwargs keyword argument.  For
    example, to import a submodule A.B, you must pass a nonempty fromlist option
    to __import__.  See the docstring of __import__().

    This catches ImportError to determine if the module is not installed.  To
    catch additional errors, pass them as a tuple to the catch keyword
    argument.

    Examples
    ========

    >>> from sympy.external import import_module

    >>> numpy = import_module('numpy')

    >>> numpy = import_module('numpy', min_python_version=(2, 7),
    ... warn_old_version=False)

    >>> numpy = import_module('numpy', min_module_version='1.5',
    ... warn_old_version=False) # numpy.__version__ is a string

    >>> # gmpy does not have __version__, but it does have gmpy.version()

    >>> gmpy = import_module('gmpy', min_module_version='1.14',
    ... module_version_attr='version', module_version_attr_call_args=(),
    ... warn_old_version=False)

    >>> # To import a submodule, you must pass a nonempty fromlist to
    >>> # __import__().  The values do not matter.
    >>> p3 = import_module('mpl_toolkits.mplot3d',
    ... import_kwargs={'fromlist':['something']})

    >>> # matplotlib.pyplot can raise RuntimeError when the display cannot be opened
    >>> matplotlib = import_module('matplotlib',
    ... import_kwargs={'fromlist':['pyplot']}, catch=(RuntimeError,))

    """
    # keyword argument overrides default, and global variable overrides
    # keyword argument.
    warn_old_version = (WARN_OLD_VERSION if WARN_OLD_VERSION is not None
        else warn_old_version or True)
    warn_not_installed = (WARN_NOT_INSTALLED if WARN_NOT_INSTALLED is not None
        else warn_not_installed or False)

    import warnings

    # Check Python first so we don't waste time importing a module we can't use
    if min_python_version:
        if sys.version_info < min_python_version:
            if warn_old_version:
                warnings.warn("Python version is too old to use %s "
                    "(%s or newer required)" % (
                        module, '.'.join(map(str, min_python_version))),
                    UserWarning, stacklevel=2)
            return

    try:
        mod = __import__(module, **import_kwargs)

        ## there's something funny about imports with matplotlib and py3k. doing
        ##    from matplotlib import collections
        ## gives python's stdlib collections module. explicitly re-importing
        ## the module fixes this.
        from_list = import_kwargs.get('fromlist', ())
        for submod in from_list:
            if submod == 'collections' and mod.__name__ == 'matplotlib':
                __import__(module + '.' + submod)
    except ImportError:
        if warn_not_installed:
            warnings.warn("%s module is not installed" % module, UserWarning,
                    stacklevel=2)
        return
    except catch as e:
        if warn_not_installed:
            warnings.warn(
                "%s module could not be used (%s)" % (module, repr(e)),
                stacklevel=2)
        return

    if min_module_version:
        modversion = getattr(mod, module_version_attr)
        if module_version_attr_call_args is not None:
            modversion = modversion(*module_version_attr_call_args)
        if version_tuple(modversion) < version_tuple(min_module_version):
            if warn_old_version:
                # Attempt to create a pretty string version of the version
                if isinstance(min_module_version, str):
                    verstr = min_module_version
                elif isinstance(min_module_version, (tuple, list)):
                    verstr = '.'.join(map(str, min_module_version))
                else:
                    # Either don't know what this is.  Hopefully
                    # it's something that has a nice str version, like an int.
                    verstr = str(min_module_version)
                warnings.warn("%s version is too old to use "
                    "(%s or newer required)" % (module, verstr),
                    UserWarning, stacklevel=2)
            return

    return mod
<|endoftext|>from __future__ import annotations
import os
from ctypes import c_long, sizeof
from functools import reduce
from typing import Type
from warnings import warn

from sympy.external import import_module

from .pythonmpq import PythonMPQ

from .ntheory import (
    bit_scan1 as python_bit_scan1,
    bit_scan0 as python_bit_scan0,
    remove as python_remove,
    factorial as python_factorial,
    sqrt as python_sqrt,
    sqrtrem as python_sqrtrem,
    gcd as python_gcd,
    lcm as python_lcm,
    gcdext as python_gcdext,
    is_square as python_is_square,
    invert as python_invert,
    legendre as python_legendre,
    jacobi as python_jacobi,
    kronecker as python_kronecker,
    iroot as python_iroot,
    is_fermat_prp as python_is_fermat_prp,
    is_euler_prp as python_is_euler_prp,
    is_strong_prp as python_is_strong_prp,
    is_fibonacci_prp as python_is_fibonacci_prp,
    is_lucas_prp as python_is_lucas_prp,
    is_selfridge_prp as python_is_selfridge_prp,
    is_strong_lucas_prp as python_is_strong_lucas_prp,
    is_strong_selfridge_prp as python_is_strong_selfridge_prp,
    is_bpsw_prp as python_is_bpsw_prp,
    is_strong_bpsw_prp as python_is_strong_bpsw_prp,
)


__all__ = [
    # GROUND_TYPES is either 'gmpy' or 'python' depending on which is used. If
    # gmpy is installed then it will be used unless the environment variable
    # SYMPY_GROUND_TYPES is set to something other than 'auto', 'gmpy', or
    # 'gmpy2'.
    'GROUND_TYPES',

    # If HAS_GMPY is 0, no supported version of gmpy is available. Otherwise,
    # HAS_GMPY will be 2 for gmpy2 if GROUND_TYPES is 'gmpy'. It used to be
    # possible for HAS_GMPY to be 1 for gmpy but gmpy is no longer supported.
    'HAS_GMPY',

    # SYMPY_INTS is a tuple containing the base types for valid integer types.
    # This is either (int,) or (int, type(mpz(0))) depending on GROUND_TYPES.
    'SYMPY_INTS',

    # MPQ is either gmpy.mpq or the Python equivalent from
    # sympy.external.pythonmpq
    'MPQ',

    # MPZ is either gmpy.mpz or int.
    'MPZ',

    'bit_scan1',
    'bit_scan0',
    'remove',
    'factorial',
    'sqrt',
    'is_square',
    'sqrtrem',
    'gcd',
    'lcm',
    'gcdext',
    'invert',
    'legendre',
    'jacobi',
    'kronecker',
    'iroot',
    'is_fermat_prp',
    'is_euler_prp',
    'is_strong_prp',
    'is_fibonacci_prp',
    'is_lucas_prp',
    'is_selfridge_prp',
    'is_strong_lucas_prp',
    'is_strong_selfridge_prp',
    'is_bpsw_prp',
    'is_strong_bpsw_prp',
]


#
# Tested python-flint version. Future versions might work but we will only use
# them if explicitly requested by SYMPY_GROUND_TYPES=flint.
#
_PYTHON_FLINT_VERSION_NEEDED = ["0.6", "0.7", "0.8", "0.9", "0.10"]


def _flint_version_okay(flint_version):
    major, minor = flint_version.split('.')[:2]
    flint_ver = f'{major}.{minor}'
    return flint_ver in _PYTHON_FLINT_VERSION_NEEDED

#
# We will only use gmpy2 >= 2.0.0
#
_GMPY2_MIN_VERSION = '2.0.0'


def _get_flint(sympy_ground_types):
    if sympy_ground_types not in ('auto', 'flint'):
        return None

    try:
        import flint
        # Earlier versions of python-flint may not have __version__.
        from flint import __version__ as _flint_version
    except ImportError:
        if sympy_ground_types == 'flint':
            warn("SYMPY_GROUND_TYPES was set to flint but python-flint is not "
                 "installed. Falling back to other ground types.")
        return None

    if _flint_version_okay(_flint_version):
        return flint
    elif sympy_ground_types == 'auto':
        return None
    else:
        warn(f"Using python-flint {_flint_version} because SYMPY_GROUND_TYPES "
             f"is set to flint but this version of SymPy is only tested "
             f"with python-flint versions {_PYTHON_FLINT_VERSION_NEEDED}.")
        return flint


def _get_gmpy2(sympy_ground_types):
    if sympy_ground_types not in ('auto', 'gmpy', 'gmpy2'):
        return None

    gmpy = import_module('gmpy2', min_module_version=_GMPY2_MIN_VERSION,
            module_version_attr='version', module_version_attr_call_args=())

    if sympy_ground_types != 'auto' and gmpy is None:
        warn("gmpy2 library is not installed, switching to 'python' ground types")

    return gmpy


#
# SYMPY_GROUND_TYPES can be flint, gmpy, gmpy2, python or auto (default)
#
_SYMPY_GROUND_TYPES = os.environ.get('SYMPY_GROUND_TYPES', 'auto').lower()
_flint = None
_gmpy = None

#
# First handle auto-detection of flint/gmpy2. We will prefer flint if available
# or otherwise gmpy2 if available and then lastly the python types.
#
if _SYMPY_GROUND_TYPES in ('auto', 'flint'):
    _flint = _get_flint(_SYMPY_GROUND_TYPES)
    if _flint is not None:
        _SYMPY_GROUND_TYPES = 'flint'
    else:
        _SYMPY_GROUND_TYPES = 'auto'

if _SYMPY_GROUND_TYPES in ('auto', 'gmpy', 'gmpy2'):
    _gmpy = _get_gmpy2(_SYMPY_GROUND_TYPES)
    if _gmpy is not None:
        _SYMPY_GROUND_TYPES = 'gmpy'
    else:
        _SYMPY_GROUND_TYPES = 'python'

if _SYMPY_GROUND_TYPES not in ('flint', 'gmpy', 'python'):
    warn("SYMPY_GROUND_TYPES environment variable unrecognised. "
         "Should be 'auto', 'flint', 'gmpy', 'gmpy2' or 'python'.")
    _SYMPY_GROUND_TYPES = 'python'

#
# At this point _SYMPY_GROUND_TYPES is either flint, gmpy or python. The blocks
# below define the values exported by this module in each case.
#

#
# In gmpy2 and flint, there are functions that take a long (or unsigned long)
# argument. That is, it is not possible to input a value larger than that.
#
LONG_MAX = (1 << (8*sizeof(c_long) - 1)) - 1

#
# Type checkers are confused by what SYMPY_INTS is. There may be a better type
# hint for this like Type[Integral] or something.
#
SYMPY_INTS: tuple[Type, ...]

if _SYMPY_GROUND_TYPES == 'gmpy':

    assert _gmpy is not None

    flint = None
    gmpy = _gmpy

    HAS_GMPY = 2
    GROUND_TYPES = 'gmpy'
    SYMPY_INTS = (int, type(gmpy.mpz(0)))
    MPZ = gmpy.mpz
    MPQ = gmpy.mpq

    bit_scan1 = gmpy.bit_scan1
    bit_scan0 = gmpy.bit_scan0
    remove = gmpy.remove
    factorial = gmpy.fac
    sqrt = gmpy.isqrt
    is_square = gmpy.is_square
    sqrtrem = gmpy.isqrt_rem
    gcd = gmpy.gcd
    lcm = gmpy.lcm
    gcdext = gmpy.gcdext
    invert = gmpy.invert
    legendre = gmpy.legendre
    jacobi = gmpy.jacobi
    kronecker = gmpy.kronecker

    def iroot(x, n):
        # In the latest gmpy2, the threshold for n is ULONG_MAX,
        # but adjust to the older one.
        if n <= LONG_MAX:
            return gmpy.iroot(x, n)
        return python_iroot(x, n)

    is_fermat_prp = gmpy.is_fermat_prp
    is_euler_prp = gmpy.is_euler_prp
    is_strong_prp = gmpy.is_strong_prp
    is_fibonacci_prp = gmpy.is_fibonacci_prp
    is_lucas_prp = gmpy.is_lucas_prp
    is_selfridge_prp = gmpy.is_selfridge_prp
    is_strong_lucas_prp = gmpy.is_strong_lucas_prp
    is_strong_selfridge_prp = gmpy.is_strong_selfridge_prp
    is_bpsw_prp = gmpy.is_bpsw_prp
    is_strong_bpsw_prp = gmpy.is_strong_bpsw_prp

elif _SYMPY_GROUND_TYPES == 'flint':

    assert _flint is not None

    flint = _flint
    gmpy = None

    HAS_GMPY = 0
    GROUND_TYPES = 'flint'
    SYMPY_INTS = (int, flint.fmpz) # type: ignore
    MPZ = flint.fmpz # type: ignore
    MPQ = flint.fmpq # type: ignore

    bit_scan1 = python_bit_scan1
    bit_scan0 = python_bit_scan0
    remove = python_remove
    factorial = python_factorial

    def sqrt(x):
        return flint.fmpz(x).isqrt()

    def is_square(x):
        if x < 0:
            return False
        return flint.fmpz(x).sqrtrem()[1] == 0

    def sqrtrem(x):
        return flint.fmpz(x).sqrtrem()

    def gcd(*args):
        return reduce(flint.fmpz.gcd, args, flint.fmpz(0))

    def lcm(*args):
        return reduce(flint.fmpz.lcm, args, flint.fmpz(1))

    gcdext = python_gcdext
    invert = python_invert
    legendre = python_legendre

    def jacobi(x, y):
        if y <= 0 or not y % 2:
            raise ValueError("y should be an odd positive integer")
        return flint.fmpz(x).jacobi(y)

    kronecker = python_kronecker

    def iroot(x, n):
        if n <= LONG_MAX:
            y = flint.fmpz(x).root(n)
            return y, y**n == x
        return python_iroot(x, n)

    is_fermat_prp = python_is_fermat_prp
    is_euler_prp = python_is_euler_prp
    is_strong_prp = python_is_strong_prp
    is_fibonacci_prp = python_is_fibonacci_prp
    is_lucas_prp = python_is_lucas_prp
    is_selfridge_prp = python_is_selfridge_prp
    is_strong_lucas_prp = python_is_strong_lucas_prp
    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    is_bpsw_prp = python_is_bpsw_prp
    is_strong_bpsw_prp = python_is_strong_bpsw_prp

elif _SYMPY_GROUND_TYPES == 'python':

    flint = None
    gmpy = None

    HAS_GMPY = 0
    GROUND_TYPES = 'python'
    SYMPY_INTS = (int,)
    MPZ = int
    MPQ = PythonMPQ

    bit_scan1 = python_bit_scan1
    bit_scan0 = python_bit_scan0
    remove = python_remove
    factorial = python_factorial
    sqrt = python_sqrt
    is_square = python_is_square
    sqrtrem = python_sqrtrem
    gcd = python_gcd
    lcm = python_lcm
    gcdext = python_gcdext
    invert = python_invert
    legendre = python_legendre
    jacobi = python_jacobi
    kronecker = python_kronecker
    iroot = python_iroot
    is_fermat_prp = python_is_fermat_prp
    is_euler_prp = python_is_euler_prp
    is_strong_prp = python_is_strong_prp
    is_fibonacci_prp = python_is_fibonacci_prp
    is_lucas_prp = python_is_lucas_prp
    is_selfridge_prp = python_is_selfridge_prp
    is_strong_lucas_prp = python_is_strong_lucas_prp
    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    is_bpsw_prp = python_is_bpsw_prp
    is_strong_bpsw_prp = python_is_strong_bpsw_prp

else:
    assert False
<|endoftext|>"""Line-like geometrical entities.

Contains
========
LinearEntity
Line
Ray
Segment
LinearEntity2D
Line2D
Ray2D
Segment2D
LinearEntity3D
Line3D
Ray3D
Segment3D

"""

from sympy.core.containers import Tuple
from sympy.core.evalf import N
from sympy.core.expr import Expr
from sympy.core.numbers import Rational, oo, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import _symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (_pi_coeff, acos, tan, atan2)
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .point import Point, Point3D
from .util import find, intersection
from sympy.logic.boolalg import And
from sympy.matrices import Matrix
from sympy.sets.sets import Intersection
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import Undecidable, filldedent


import random


t, u = [Dummy('line_dummy') for i in range(2)]


class LinearEntity(GeometrySet):
    """A base class for all linear entities (Line, Ray and Segment)
    in n-dimensional Euclidean space.

    Attributes
    ==========

    ambient_dimension
    direction
    length
    p1
    p2
    points

    Notes
    =====

    This is an abstract class and is not meant to be instantiated.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity

    """
    def __new__(cls, p1, p2=None, **kwargs):
        p1, p2 = Point._normalize_dimension(p1, p2)
        if p1 == p2:
            # sometimes we return a single point if we are not given two unique
            # points. This is done in the specific subclass
            raise ValueError(
                "%s.__new__ requires two unique Points." % cls.__name__)
        if len(p1) != len(p2):
            raise ValueError(
                "%s.__new__ requires two Points of equal dimension." % cls.__name__)

        return GeometryEntity.__new__(cls, p1, p2, **kwargs)

    def __contains__(self, other):
        """Return a definitive answer or else raise an error if it cannot
        be determined that other is on the boundaries of self."""
        result = self.contains(other)

        if result is not None:
            return result
        else:
            raise Undecidable(
                "Cannot decide whether '%s' contains '%s'" % (self, other))

    def _span_test(self, other):
        """Test whether the point `other` lies in the positive span of `self`.
        A point x is 'in front' of a point y if x.dot(y) >= 0.  Return
        -1 if `other` is behind `self.p1`, 0 if `other` is `self.p1` and
        and 1 if `other` is in front of `self.p1`."""
        if self.p1 == other:
            return 0

        rel_pos = other - self.p1
        d = self.direction
        if d.dot(rel_pos) > 0:
            return 1
        return -1

    @property
    def ambient_dimension(self):
        """A property method that returns the dimension of LinearEntity
        object.

        Parameters
        ==========

        p1 : LinearEntity

        Returns
        =======

        dimension : integer

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> l1 = Line(p1, p2)
        >>> l1.ambient_dimension
        2

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 1)
        >>> l1 = Line(p1, p2)
        >>> l1.ambient_dimension
        3

        """
        return len(self.p1)

    def angle_between(l1, l2):
        """Return the non-reflex angle formed by rays emanating from
        the origin with directions the same as the direction vectors
        of the linear entities.

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        angle : angle in radians

        Notes
        =====

        From the dot product of vectors v1 and v2 it is known that:

            ``dot(v1, v2) = |v1|*|v2|*cos(A)``

        where A is the angle formed between the two vectors. We can
        get the directional vectors of the two lines and readily
        find the angle between the two using the above formula.

        See Also
        ========

        is_perpendicular, Ray2D.closing_angle

        Examples
        ========

        >>> from sympy import Line
        >>> e = Line((0, 0), (1, 0))
        >>> ne = Line((0, 0), (1, 1))
        >>> sw = Line((1, 1), (0, 0))
        >>> ne.angle_between(e)
        pi/4
        >>> sw.angle_between(e)
        3*pi/4

        To obtain the non-obtuse angle at the intersection of lines, use
        the ``smallest_angle_between`` method:

        >>> sw.smallest_angle_between(e)
        pi/4

        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(-1, 2, 0)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p2, p3)
        >>> l1.angle_between(l2)
        acos(-sqrt(2)/3)
        >>> l1.smallest_angle_between(l2)
        acos(sqrt(2)/3)
        """
        if not isinstance(l1, LinearEntity) and not isinstance(l2, LinearEntity):
            raise TypeError('Must pass only LinearEntity objects')

        v1, v2 = l1.direction, l2.direction
        return acos(v1.dot(v2)/(abs(v1)*abs(v2)))

    def smallest_angle_between(l1, l2):
        """Return the smallest angle formed at the intersection of the
        lines containing the linear entities.

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        angle : angle in radians

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(0, 4), Point(2, -2)
        >>> l1, l2 = Line(p1, p2), Line(p1, p3)
        >>> l1.smallest_angle_between(l2)
        pi/4

        See Also
        ========

        angle_between, is_perpendicular, Ray2D.closing_angle
        """
        if not isinstance(l1, LinearEntity) and not isinstance(l2, LinearEntity):
            raise TypeError('Must pass only LinearEntity objects')

        v1, v2 = l1.direction, l2.direction
        return acos(abs(v1.dot(v2))/(abs(v1)*abs(v2)))

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the Line.

        Parameters
        ==========

        parameter : str, optional
            The name of the parameter which will be used for the parametric
            point. The default value is 't'. When this parameter is 0, the
            first point used to define the line will be returned, and when
            it is 1 the second point will be returned.

        Returns
        =======

        point : Point

        Raises
        ======

        ValueError
            When ``parameter`` already appears in the Line's definition.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 0), Point(5, 3)
        >>> l1 = Line(p1, p2)
        >>> l1.arbitrary_point()
        Point2D(4*t + 1, 3*t)
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(1, 0, 0), Point3D(5, 3, 1)
        >>> l1 = Line3D(p1, p2)
        >>> l1.arbitrary_point()
        Point3D(4*t + 1, 3*t, t)

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('''
                Symbol %s already appears in object
                and cannot be used as a parameter.
                ''' % t.name))
        # multiply on the right so the variable gets
        # combined with the coordinates of the point
        return self.p1 + (self.p2 - self.p1)*t

    @staticmethod
    def are_concurrent(*lines):
        """Is a sequence of linear entities concurrent?

        Two or more linear entities are concurrent if they all
        intersect at a single point.

        Parameters
        ==========

        lines
            A sequence of linear entities.

        Returns
        =======

        True : if the set of linear entities intersect in one point
        False : otherwise.

        See Also
        ========

        sympy.geometry.util.intersection

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(3, 5)
        >>> p3, p4 = Point(-2, -2), Point(0, 2)
        >>> l1, l2, l3 = Line(p1, p2), Line(p1, p3), Line(p1, p4)
        >>> Line.are_concurrent(l1, l2, l3)
        True
        >>> l4 = Line(p2, p3)
        >>> Line.are_concurrent(l2, l3, l4)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(3, 5, 2)
        >>> p3, p4 = Point3D(-2, -2, -2), Point3D(0, 2, 1)
        >>> l1, l2, l3 = Line3D(p1, p2), Line3D(p1, p3), Line3D(p1, p4)
        >>> Line3D.are_concurrent(l1, l2, l3)
        True
        >>> l4 = Line3D(p2, p3)
        >>> Line3D.are_concurrent(l2, l3, l4)
        False

        """
        common_points = Intersection(*lines)
        if common_points.is_FiniteSet and len(common_points) == 1:
            return True
        return False

    def contains(self, other):
        """Subclasses should implement this method and should return
            True if other is on the boundaries of self;
            False if not on the boundaries of self;
            None if a determination cannot be made."""
        raise NotImplementedError()

    @property
    def direction(self):
        """The direction vector of the LinearEntity.

        Returns
        =======

        p : a Point; the ray from the origin to this point is the
            direction of `self`

        Examples
        ========

        >>> from sympy import Line
        >>> a, b = (1, 1), (1, 3)
        >>> Line(a, b).direction
        Point2D(0, 2)
        >>> Line(b, a).direction
        Point2D(0, -2)

        This can be reported so the distance from the origin is 1:

        >>> Line(b, a).direction.unit
        Point2D(0, -1)

        See Also
        ========

        sympy.geometry.point.Point.unit

        """
        return self.p2 - self.p1

    def intersection(self, other):
        """The intersection with another geometrical entity.

        Parameters
        ==========

        o : Point or LinearEntity

        Returns
        =======

        intersection : list of geometrical entities

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(7, 7)
        >>> l1 = Line(p1, p2)
        >>> l1.intersection(p3)
        [Point2D(7, 7)]
        >>> p4, p5 = Point(5, 0), Point(0, 3)
        >>> l2 = Line(p4, p5)
        >>> l1.intersection(l2)
        [Point2D(15/8, 15/8)]
        >>> p6, p7 = Point(0, 5), Point(2, 6)
        >>> s1 = Segment(p6, p7)
        >>> l1.intersection(s1)
        []
        >>> from sympy import Point3D, Line3D, Segment3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(7, 7, 7)
        >>> l1 = Line3D(p1, p2)
        >>> l1.intersection(p3)
        [Point3D(7, 7, 7)]
        >>> l1 = Line3D(Point3D(4,19,12), Point3D(5,25,17))
        >>> l2 = Line3D(Point3D(-3, -15, -19), direction_ratio=[2,8,8])
        >>> l1.intersection(l2)
        [Point3D(1, 1, -3)]
        >>> p6, p7 = Point3D(0, 5, 2), Point3D(2, 6, 3)
        >>> s1 = Segment3D(p6, p7)
        >>> l1.intersection(s1)
        []

        """
        def intersect_parallel_rays(ray1, ray2):
            if ray1.direction.dot(ray2.direction) > 0:
                # rays point in the same direction
                # so return the one that is "in front"
                return [ray2] if ray1._span_test(ray2.p1) >= 0 else [ray1]
            else:
                # rays point in opposite directions
                st = ray1._span_test(ray2.p1)
                if st < 0:
                    return []
                elif st == 0:
                    return [ray2.p1]
                return [Segment(ray1.p1, ray2.p1)]

        def intersect_parallel_ray_and_segment(ray, seg):
            st1, st2 = ray._span_test(seg.p1), ray._span_test(seg.p2)
            if st1 < 0 and st2 < 0:
                return []
            elif st1 >= 0 and st2 >= 0:
                return [seg]
            elif st1 >= 0:  # st2 < 0:
                return [Segment(ray.p1, seg.p1)]
            else:  # st1 < 0 and st2 >= 0:
                return [Segment(ray.p1, seg.p2)]

        def intersect_parallel_segments(seg1, seg2):
            if seg1.contains(seg2):
                return [seg2]
            if seg2.contains(seg1):
                return [seg1]

            # direct the segments so they're oriented the same way
            if seg1.direction.dot(seg2.direction) < 0:
                seg2 = Segment(seg2.p2, seg2.p1)
            # order the segments so seg1 is "behind" seg2
            if seg1._span_test(seg2.p1) < 0:
                seg1, seg2 = seg2, seg1
            if seg2._span_test(seg1.p2) < 0:
                return []
            return [Segment(seg2.p1, seg1.p2)]

        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if other.is_Point:
            if self.contains(other):
                return [other]
            else:
                return []
        elif isinstance(other, LinearEntity):
            # break into cases based on whether
            # the lines are parallel, non-parallel intersecting, or skew
            pts = Point._normalize_dimension(self.p1, self.p2, other.p1, other.p2)
            rank = Point.affine_rank(*pts)

            if rank == 1:
                # we're collinear
                if isinstance(self, Line):
                    return [other]
                if isinstance(other, Line):
                    return [self]

                if isinstance(self, Ray) and isinstance(other, Ray):
                    return intersect_parallel_rays(self, other)
                if isinstance(self, Ray) and isinstance(other, Segment):
                    return intersect_parallel_ray_and_segment(self, other)
                if isinstance(self, Segment) and isinstance(other, Ray):
                    return intersect_parallel_ray_and_segment(other, self)
                if isinstance(self, Segment) and isinstance(other, Segment):
                    return intersect_parallel_segments(self, other)
            elif rank == 2:
                # we're in the same plane
                l1 = Line(*pts[:2])
                l2 = Line(*pts[2:])

                # check to see if we're parallel.  If we are, we can't
                # be intersecting, since the collinear case was already
                # handled
                if l1.direction.is_scalar_multiple(l2.direction):
                    return []

                # find the intersection as if everything were lines
                # by solving the equation t*d + p1 == s*d' + p1'
                m = Matrix([l1.direction, -l2.direction]).transpose()
                v = Matrix([l2.p1 - l1.p1]).transpose()

                # we cannot use m.solve(v) because that only works for square matrices
                m_rref, pivots = m.col_insert(2, v).rref(simplify=True)
                # rank == 2 ensures we have 2 pivots, but let's check anyway
                if len(pivots) != 2:
                    raise GeometryError("Failed when solving Mx=b when M={} and b={}".format(m, v))
                coeff = m_rref[0, 2]
                line_intersection = l1.direction*coeff + self.p1

                # if both are lines, skip a containment check
                if isinstance(self, Line) and isinstance(other, Line):
                    return [line_intersection]

                if ((isinstance(self, Line) or
                     self.contains(line_intersection)) and
                        other.contains(line_intersection)):
                    return [line_intersection]
                if not self.atoms(Float) and not other.atoms(Float):
                    # if it can fail when there are no Floats then
                    # maybe the following parametric check should be
                    # done
                    return []
                # floats may fail exact containment so check that the
                # arbitrary points, when  equal, both give a
                # non-negative parameter when the arbitrary point
                # coordinates are equated
                tu = solve(self.arbitrary_point(t) - other.arbitrary_point(u),
                    t, u, dict=True)[0]
                def ok(p, l):
                    if isinstance(l, Line):
                        # p > -oo
                        return True
                    if isinstance(l, Ray):
                        # p >= 0
                        return p.is_nonnegative
                    if isinstance(l, Segment):
                        # 0 <= p <= 1
                        return p.is_nonnegative and (1 - p).is_nonnegative
                    raise ValueError("unexpected line type")
                if ok(tu[t], self) and ok(tu[u], other):
                    return [line_intersection]
                return []
            else:
                # we're skew
                return []

        return other.intersection(self)

    def is_parallel(l1, l2):
        """Are two linear entities parallel?

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        True : if l1 and l2 are parallel,
        False : otherwise.

        See Also
        ========

        coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> p3, p4 = Point(3, 4), Point(6, 7)
        >>> l1, l2 = Line(p1, p2), Line(p3, p4)
        >>> Line.is_parallel(l1, l2)
        True
        >>> p5 = Point(6, 6)
        >>> l3 = Line(p3, p5)
        >>> Line.is_parallel(l1, l3)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(3, 4, 5)
        >>> p3, p4 = Point3D(2, 1, 1), Point3D(8, 9, 11)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p3, p4)
        >>> Line3D.is_parallel(l1, l2)
        True
        >>> p5 = Point3D(6, 6, 6)
        >>> l3 = Line3D(p3, p5)
        >>> Line3D.is_parallel(l1, l3)
        False

        """
        if not isinstance(l1, LinearEntity) and not isinstance(l2, LinearEntity):
            raise TypeError('Must pass only LinearEntity objects')

        return l1.direction.is_scalar_multiple(l2.direction)

    def is_perpendicular(l1, l2):
        """Are two linear entities perpendicular?

        Parameters
        ==========

        l1 : LinearEntity
        l2 : LinearEntity

        Returns
        =======

        True : if l1 and l2 are perpendicular,
        False : otherwise.

        See Also
        ========

        coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(-1, 1)
        >>> l1, l2 = Line(p1, p2), Line(p1, p3)
        >>> l1.is_perpendicular(l2)
        True
        >>> p4 = Point(5, 3)
        >>> l3 = Line(p1, p4)
        >>> l1.is_perpendicular(l3)
        False
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(-1, 2, 0)
        >>> l1, l2 = Line3D(p1, p2), Line3D(p2, p3)
        >>> l1.is_perpendicular(l2)
        False
        >>> p4 = Point3D(5, 3, 7)
        >>> l3 = Line3D(p1, p4)
        >>> l1.is_perpendicular(l3)
        False

        """
        if not isinstance(l1, LinearEntity) and not isinstance(l2, LinearEntity):
            raise TypeError('Must pass only LinearEntity objects')

        return S.Zero.equals(l1.direction.dot(l2.direction))

    def is_similar(self, other):
        """
        Return True if self and other are contained in the same line.

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 1), Point(3, 4), Point(2, 3)
        >>> l1 = Line(p1, p2)
        >>> l2 = Line(p1, p3)
        >>> l1.is_similar(l2)
        True
        """
        l = Line(self.p1, self.p2)
        return l.contains(other)

    @property
    def length(self):
        """
        The length of the line.

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(3, 5)
        >>> l1 = Line(p1, p2)
        >>> l1.length
        oo
        """
        return S.Infinity

    @property
    def p1(self):
        """The first defining point of a linear entity.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l = Line(p1, p2)
        >>> l.p1
        Point2D(0, 0)

        """
        return self.args[0]

    @property
    def p2(self):
        """The second defining point of a linear entity.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l = Line(p1, p2)
        >>> l.p2
        Point2D(5, 3)

        """
        return self.args[1]

    def parallel_line(self, p):
        """Create a new Line parallel to this linear entity which passes
        through the point `p`.

        Parameters
        ==========

        p : Point

        Returns
        =======

        line : Line

        See Also
        ========

        is_parallel

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)
        >>> l1 = Line(p1, p2)
        >>> l2 = l1.parallel_line(p3)
        >>> p3 in l2
        True
        >>> l1.is_parallel(l2)
        True
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(2, 3, 4), Point3D(-2, 2, 0)
        >>> l1 = Line3D(p1, p2)
        >>> l2 = l1.parallel_line(p3)
        >>> p3 in l2
        True
        >>> l1.is_parallel(l2)
        True

        """
        p = Point(p, dim=self.ambient_dimension)
        return Line(p, p + self.direction)

    def perpendicular_line(self, p):
        """Create a new Line perpendicular to this linear entity which passes
        through the point `p`.

        Parameters
        ==========

        p : Point

        Returns
        =======

        line : Line

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular, perpendicular_segment

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(2, 3, 4), Point3D(-2, 2, 0)
        >>> L = Line3D(p1, p2)
        >>> P = L.perpendicular_line(p3); P
        Line3D(Point3D(-2, 2, 0), Point3D(4/29, 6/29, 8/29))
        >>> L.is_perpendicular(P)
        True

        In 3D the, the first point used to define the line is the point
        through which the perpendicular was required to pass; the
        second point is (arbitrarily) contained in the given line:

        >>> P.p2 in L
        True
        """
        p = Point(p, dim=self.ambient_dimension)
        if p in self:
            p = p + self.direction.orthogonal_direction
        return Line(p, self.projection(p))

    def perpendicular_segment(self, p):
        """Create a perpendicular line segment from `p` to this line.

        The endpoints of the segment are ``p`` and the closest point in
        the line containing self. (If self is not a line, the point might
        not be in self.)

        Parameters
        ==========

        p : Point

        Returns
        =======

        segment : Segment

        Notes
        =====

        Returns `p` itself if `p` is on this linear entity.

        See Also
        ========

        perpendicular_line

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 2)
        >>> l1 = Line(p1, p2)
        >>> s1 = l1.perpendicular_segment(p3)
        >>> l1.is_perpendicular(s1)
        True
        >>> p3 in s1
        True
        >>> l1.perpendicular_segment(Point(4, 0))
        Segment2D(Point2D(4, 0), Point2D(2, 2))
        >>> from sympy import Point3D, Line3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 2, 0)
        >>> l1 = Line3D(p1, p2)
        >>> s1 = l1.perpendicular_segment(p3)
        >>> l1.is_perpendicular(s1)
        True
        >>> p3 in s1
        True
        >>> l1.perpendicular_segment(Point3D(4, 0, 0))
        Segment3D(Point3D(4, 0, 0), Point3D(4/3, 4/3, 4/3))

        """
        p = Point(p, dim=self.ambient_dimension)
        if p in self:
            return p
        l = self.perpendicular_line(p)
        # The intersection should be unique, so unpack the singleton
        p2, = Intersection(Line(self.p1, self.p2), l)

        return Segment(p, p2)

    @property
    def points(self):
        """The two points used to define this linear entity.

        Returns
        =======

        points : tuple of Points

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 11)
        >>> l1 = Line(p1, p2)
        >>> l1.points
        (Point2D(0, 0), Point2D(5, 11))

        """
        return (self.p1, self.p2)

    def projection(self, other):
        """Project a point, line, ray, or segment onto this linear entity.

        Parameters
        ==========

        other : Point or LinearEntity (Line, Ray, Segment)

        Returns
        =======

        projection : Point or LinearEntity (Line, Ray, Segment)
            The return type matches the type of the parameter ``other``.

        Raises
        ======

        GeometryError
            When method is unable to perform projection.

        Notes
        =====

        A projection involves taking the two points that define
        the linear entity and projecting those points onto a
        Line and then reforming the linear entity using these
        projections.
        A point P is projected onto a line L by finding the point
        on L that is closest to P. This point is the intersection
        of L and the line perpendicular to L that passes through P.

        See Also
        ========

        sympy.geometry.point.Point, perpendicular_line

        Examples
        ========

        >>> from sympy import Point, Line, Segment, Rational
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(Rational(1, 2), 0)
        >>> l1 = Line(p1, p2)
        >>> l1.projection(p3)
        Point2D(1/4, 1/4)
        >>> p4, p5 = Point(10, 0), Point(12, 1)
        >>> s1 = Segment(p4, p5)
        >>> l1.projection(s1)
        Segment2D(Point2D(5, 5), Point2D(13/2, 13/2))
        >>> p1, p2, p3 = Point(0, 0, 1), Point(1, 1, 2), Point(2, 0, 1)
        >>> l1 = Line(p1, p2)
        >>> l1.projection(p3)
        Point3D(2/3, 2/3, 5/3)
        >>> p4, p5 = Point(10, 0, 1), Point(12, 1, 3)
        >>> s1 = Segment(p4, p5)
        >>> l1.projection(s1)
        Segment3D(Point3D(10/3, 10/3, 13/3), Point3D(5, 5, 6))

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)

        def proj_point(p):
            return Point.project(p - self.p1, self.direction) + self.p1

        if isinstance(other, Point):
            return proj_point(other)
        elif isinstance(other, LinearEntity):
            p1, p2 = proj_point(other.p1), proj_point(other.p2)
            # test to see if we're degenerate
            if p1 == p2:
                return p1
            projected = other.__class__(p1, p2)
            projected = Intersection(self, projected)
            if projected.is_empty:
                return projected
            # if we happen to have intersected in only a point, return that
            if projected.is_FiniteSet and len(projected) == 1:
                # projected is a set of size 1, so unpack it in `a`
                a, = projected
                return a
            # order args so projection is in the same direction as self
            if self.direction.dot(projected.direction) < 0:
                p1, p2 = projected.args
                projected = projected.func(p2, p1)
            return projected

        raise GeometryError(
            "Do not know how to project %s onto %s" % (other, self))

    def random_point(self, seed=None):
        """A random point on a LinearEntity.

        Returns
        =======

        point : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Line, Ray, Segment
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> line = Line(p1, p2)
        >>> r = line.random_point(seed=42)  # seed value is optional
        >>> r.n(3)
        Point2D(-0.72, -0.432)
        >>> r in line
        True
        >>> Ray(p1, p2).random_point(seed=42).n(3)
        Point2D(0.72, 0.432)
        >>> Segment(p1, p2).random_point(seed=42).n(3)
        Point2D(3.2, 1.92)

        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        pt = self.arbitrary_point(t)
        if isinstance(self, Ray):
            v = abs(rng.gauss(0, 1))
        elif isinstance(self, Segment):
            v = rng.random()
        elif isinstance(self, Line):
            v = rng.gauss(0, 1)
        else:
            raise NotImplementedError('unhandled line type')
        return pt.subs(t, Rational(v))

    def bisectors(self, other):
        """Returns the perpendicular lines which pass through the intersections
        of self and other that are in the same plane.

        Parameters
        ==========

        line : Line3D

        Returns
        =======

        list: two Line instances

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> r1 = Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0))
        >>> r2 = Line3D(Point3D(0, 0, 0), Point3D(0, 1, 0))
        >>> r1.bisectors(r2)
        [Line3D(Point3D(0, 0, 0), Point3D(1, 1, 0)), Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]

        """
        if not isinstance(other, LinearEntity):
            raise GeometryError("Expecting LinearEntity, not %s" % other)

        l1, l2 = self, other

        # make sure dimensions match or else a warning will rise from
        # intersection calculation
        if l1.p1.ambient_dimension != l2.p1.ambient_dimension:
            if isinstance(l1, Line2D):
                l1, l2 = l2, l1
            _, p1 = Point._normalize_dimension(l1.p1, l2.p1, on_morph='ignore')
            _, p2 = Point._normalize_dimension(l1.p2, l2.p2, on_morph='ignore')
            l2 = Line(p1, p2)

        point = intersection(l1, l2)

        # Three cases: Lines may intersect in a point, may be equal or may not intersect.
        if not point:
            raise GeometryError("The lines do not intersect")
        else:
            pt = point[0]
            if isinstance(pt, Line):
                # Intersection is a line because both lines are coincident
                return [self]


        d1 = l1.direction.unit
        d2 = l2.direction.unit

        bis1 = Line(pt, pt + d1 + d2)
        bis2 = Line(pt, pt + d1 - d2)

        return [bis1, bis2]


class Line(LinearEntity):
    """An infinite line in space.

    A 2D line is declared with two distinct points, point and slope, or
    an equation. A 3D line may be defined with a point and a direction ratio.

    Parameters
    ==========

    p1 : Point
    p2 : Point
    slope : SymPy expression
    direction_ratio : list
    equation : equation of a line

    Notes
    =====

    `Line` will automatically subclass to `Line2D` or `Line3D` based
    on the dimension of `p1`.  The `slope` argument is only relevant
    for `Line2D` and the `direction_ratio` argument is only relevant
    for `Line3D`.

    The order of the points will define the direction of the line
    which is used when calculating the angle between lines.

    See Also
    ========

    sympy.geometry.point.Point
    sympy.geometry.line.Line2D
    sympy.geometry.line.Line3D

    Examples
    ========

    >>> from sympy import Line, Segment, Point, Eq
    >>> from sympy.abc import x, y, a, b

    >>> L = Line(Point(2,3), Point(3,5))
    >>> L
    Line2D(Point2D(2, 3), Point2D(3, 5))
    >>> L.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> L.equation()
    -2*x + y + 1
    >>> L.coefficients
    (-2, 1, 1)

    Instantiate with keyword ``slope``:

    >>> Line(Point(0, 0), slope=0)
    Line2D(Point2D(0, 0), Point2D(1, 0))

    Instantiate with another linear object

    >>> s = Segment((0, 0), (0, 1))
    >>> Line(s).equation()
    x

    The line corresponding to an equation in the for `ax + by + c = 0`,
    can be entered:

    >>> Line(3*x + y + 18)
    Line2D(Point2D(0, -18), Point2D(1, -21))

    If `x` or `y` has a different name, then they can be specified, too,
    as a string (to match the name) or symbol:

    >>> Line(Eq(3*a + b, -18), x='a', y=b)
    Line2D(Point2D(0, -18), Point2D(1, -21))
    """
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
            missing = uniquely_named_symbol('?', args)
            if not kwargs:
                x = 'x'
                y = 'y'
            else:
                x = kwargs.pop('x', missing)
                y = kwargs.pop('y', missing)
            if kwargs:
                raise ValueError('expecting only x and y as keywords')

            equation = args[0]
            if isinstance(equation, Eq):
                equation = equation.lhs - equation.rhs

            def find_or_missing(x):
                try:
                    return find(x, equation)
                except ValueError:
                    return missing
            x = find_or_missing(x)
            y = find_or_missing(y)

            a, b, c = linear_coeffs(equation, x, y)

            if b:
                return Line((0, -c/b), slope=-a/b)
            if a:
                return Line((-c/a, 0), slope=oo)

            raise ValueError('not found in equation: %s' % (set('xy') - {x, y}))

        else:
            if len(args) > 0:
                p1 = args[0]
                if len(args) > 1:
                    p2 = args[1]
                else:
                    p2 = None

                if isinstance(p1, LinearEntity):
                    if p2:
                        raise ValueError('If p1 is a LinearEntity, p2 must be None.')
                    dim = len(p1.p1)
                else:
                    p1 = Point(p1)
                    dim = len(p1)
                    if p2 is not None or isinstance(p2, Point) and p2.ambient_dimension != dim:
                        p2 = Point(p2)

                if dim == 2:
                    return Line2D(p1, p2, **kwargs)
                elif dim == 3:
                    return Line3D(p1, p2, **kwargs)
                return LinearEntity.__new__(cls, p1, p2, **kwargs)

    def contains(self, other):
        """
        Return True if `other` is on this Line, or False otherwise.

        Examples
        ========

        >>> from sympy import Line,Point
        >>> p1, p2 = Point(0, 1), Point(3, 4)
        >>> l = Line(p1, p2)
        >>> l.contains(p1)
        True
        >>> l.contains((0, 1))
        True
        >>> l.contains((0, 0))
        False
        >>> a = (0, 0, 0)
        >>> b = (1, 1, 1)
        >>> c = (2, 2, 2)
        >>> l1 = Line(a, b)
        >>> l2 = Line(b, a)
        >>> l1 == l2
        False
        >>> l1 in l2
        True

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            return Point.is_collinear(other, self.p1, self.p2)
        if isinstance(other, LinearEntity):
            return Point.is_collinear(self.p1, self.p2, other.p1, other.p2)
        return False

    def distance(self, other):
        """
        Finds the shortest distance between a line and a point.

        Raises
        ======

        NotImplementedError is raised if `other` is not a Point

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> s = Line(p1, p2)
        >>> s.distance(Point(-1, 1))
        sqrt(2)
        >>> s.distance((-1, 2))
        3*sqrt(2)/2
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 1)
        >>> s = Line(p1, p2)
        >>> s.distance(Point(-1, 1, 1))
        2*sqrt(6)/3
        >>> s.distance((-1, 1, 1))
        2*sqrt(6)/3

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if self.contains(other):
            return S.Zero
        return self.perpendicular_segment(other).length

    def equals(self, other):
        """Returns True if self and other are the same mathematical entities"""
        if not isinstance(other, Line):
            return False
        return Point.is_collinear(self.p1, other.p1, self.p2, other.p2)

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of line. Gives
        values that will produce a line that is +/- 5 units long (where a
        unit is the distance between the two points that define the line).

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list (plot interval)
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l1 = Line(p1, p2)
        >>> l1.plot_interval()
        [t, -5, 5]

        """
        t = _symbol(parameter, real=True)
        return [t, -5, 5]


class Ray(LinearEntity):
    """A Ray is a semi-line in the space with a source point and a direction.

    Parameters
    ==========

    p1 : Point
        The source of the Ray
    p2 : Point or radian value
        This point determines the direction in which the Ray propagates.
        If given as an angle it is interpreted in radians with the positive
        direction being ccw.

    Attributes
    ==========

    source

    See Also
    ========

    sympy.geometry.line.Ray2D
    sympy.geometry.line.Ray3D
    sympy.geometry.point.Point
    sympy.geometry.line.Line

    Notes
    =====

    `Ray` will automatically subclass to `Ray2D` or `Ray3D` based on the
    dimension of `p1`.

    Examples
    ========

    >>> from sympy import Ray, Point, pi
    >>> r = Ray(Point(2, 3), Point(3, 5))
    >>> r
    Ray2D(Point2D(2, 3), Point2D(3, 5))
    >>> r.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> r.source
    Point2D(2, 3)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.slope
    2
    >>> Ray(Point(0, 0), angle=pi/4).slope
    1

    """
    def __new__(cls, p1, p2=None, **kwargs):
        p1 = Point(p1)
        if p2 is not None:
            p1, p2 = Point._normalize_dimension(p1, Point(p2))
        dim = len(p1)

        if dim == 2:
            return Ray2D(p1, p2, **kwargs)
        elif dim == 3:
            return Ray3D(p1, p2, **kwargs)
        return LinearEntity.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the LinearEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        verts = (N(self.p1), N(self.p2))
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        path = "M {} L {}".format(coords[0], " L ".join(coords[1:]))

        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" '
            'marker-start="url(#markerCircle)" marker-end="url(#markerArrow)"/>'
        ).format(2.*scale_factor, path, fill_color)

    def contains(self, other):
        """
        Is other GeometryEntity contained in this Ray?

        Examples
        ========

        >>> from sympy import Ray,Point,Segment
        >>> p1, p2 = Point(0, 0), Point(4, 4)
        >>> r = Ray(p1, p2)
        >>> r.contains(p1)
        True
        >>> r.contains((1, 1))
        True
        >>> r.contains((1, 3))
        False
        >>> s = Segment((1, 1), (2, 2))
        >>> r.contains(s)
        True
        >>> s = Segment((1, 2), (2, 5))
        >>> r.contains(s)
        False
        >>> r1 = Ray((2, 2), (3, 3))
        >>> r.contains(r1)
        True
        >>> r1 = Ray((2, 2), (3, 5))
        >>> r.contains(r1)
        False
        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            if Point.is_collinear(self.p1, self.p2, other):
                # if we're in the direction of the ray, our
                # direction vector dot the ray's direction vector
                # should be non-negative
                return bool((self.p2 - self.p1).dot(other - self.p1) >= S.Zero)
            return False
        elif isinstance(other, Ray):
            if Point.is_collinear(self.p1, self.p2, other.p1, other.p2):
                return bool((self.p2 - self.p1).dot(other.p2 - other.p1) > S.Zero)
            return False
        elif isinstance(other, Segment):
            return other.p1 in self and other.p2 in self

        # No other known entity can be contained in a Ray
        return False

    def distance(self, other):
        """
        Finds the shortest distance between the ray and a point.

        Raises
        ======

        NotImplementedError is raised if `other` is not a Point

        Examples
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> s = Ray(p1, p2)
        >>> s.distance(Point(-1, -1))
        sqrt(2)
        >>> s.distance((-1, 2))
        3*sqrt(2)/2
        >>> p1, p2 = Point(0, 0, 0), Point(1, 1, 2)
        >>> s = Ray(p1, p2)
        >>> s
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, 2))
        >>> s.distance(Point(-1, -1, 2))
        4*sqrt(3)/3
        >>> s.distance((-1, -1, 2))
        4*sqrt(3)/3

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if self.contains(other):
            return S.Zero

        proj = Line(self.p1, self.p2).projection(other)
        if self.contains(proj):
            return abs(other - proj)
        else:
            return abs(other - self.source)

    def equals(self, other):
        """Returns True if self and other are the same mathematical entities"""
        if not isinstance(other, Ray):
            return False
        return self.source == other.source and other.p2 in self

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the Ray. Gives
        values that will produce a ray that is 10 units long (where a unit is
        the distance between the two points that define the ray).

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Ray, pi
        >>> r = Ray((0, 0), angle=pi/4)
        >>> r.plot_interval()
        [t, 0, 10]

        """
        t = _symbol(parameter, real=True)
        return [t, 0, 10]

    @property
    def source(self):
        """The point from which the ray emanates.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2 = Point(0, 0), Point(4, 1)
        >>> r1 = Ray(p1, p2)
        >>> r1.source
        Point2D(0, 0)
        >>> p1, p2 = Point(0, 0, 0), Point(4, 1, 5)
        >>> r1 = Ray(p2, p1)
        >>> r1.source
        Point3D(4, 1, 5)

        """
        return self.p1


class Segment(LinearEntity):
    """A line segment in space.

    Parameters
    ==========

    p1 : Point
    p2 : Point

    Attributes
    ==========

    length : number or SymPy expression
    midpoint : Point

    See Also
    ========

    sympy.geometry.line.Segment2D
    sympy.geometry.line.Segment3D
    sympy.geometry.point.Point
    sympy.geometry.line.Line

    Notes
    =====

    If 2D or 3D points are used to define `Segment`, it will
    be automatically subclassed to `Segment2D` or `Segment3D`.

    Examples
    ========

    >>> from sympy import Point, Segment
    >>> Segment((1, 0), (1, 1)) # tuples are interpreted as pts
    Segment2D(Point2D(1, 0), Point2D(1, 1))
    >>> s = Segment(Point(4, 3), Point(1, 1))
    >>> s.points
    (Point2D(4, 3), Point2D(1, 1))
    >>> s.slope
    2/3
    >>> s.length
    sqrt(13)
    >>> s.midpoint
    Point2D(5/2, 2)
    >>> Segment((1, 0, 0), (1, 1, 1)) # tuples are interpreted as pts
    Segment3D(Point3D(1, 0, 0), Point3D(1, 1, 1))
    >>> s = Segment(Point(4, 3, 9), Point(1, 1, 7)); s
    Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.points
    (Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.length
    sqrt(17)
    >>> s.midpoint
    Point3D(5/2, 2, 8)

    """
    def __new__(cls, p1, p2, **kwargs):
        p1, p2 = Point._normalize_dimension(Point(p1), Point(p2))
        dim = len(p1)

        if dim == 2:
            return Segment2D(p1, p2, **kwargs)
        elif dim == 3:
            return Segment3D(p1, p2, **kwargs)
        return LinearEntity.__new__(cls, p1, p2, **kwargs)

    def contains(self, other):
        """
        Is the other GeometryEntity contained within this Segment?

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 1), Point(3, 4)
        >>> s = Segment(p1, p2)
        >>> s2 = Segment(p2, p1)
        >>> s.contains(s2)
        True
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 1, 1), Point3D(3, 4, 5)
        >>> s = Segment3D(p1, p2)
        >>> s2 = Segment3D(p2, p1)
        >>> s.contains(s2)
        True
        >>> s.contains((p1 + p2)/2)
        True
        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            if Point.is_collinear(other, self.p1, self.p2):
                if isinstance(self, Segment2D):
                    # if it is collinear and is in the bounding box of the
                    # segment then it must be on the segment
                    vert = (1/self.slope).equals(0)
                    if vert is False:
                        isin = (self.p1.x - other.x)*(self.p2.x - other.x) <= 0
                        if isin in (True, False):
                            return isin
                    if vert is True:
                        isin = (self.p1.y - other.y)*(self.p2.y - other.y) <= 0
                        if isin in (True, False):
                            return isin
                # use the triangle inequality
                d1, d2 = other - self.p1, other - self.p2
                d = self.p2 - self.p1
                # without the call to simplify, SymPy cannot tell that an expression
                # like (a+b)*(a/2+b/2) is always non-negative.  If it cannot be
                # determined, raise an Undecidable error
                try:
                    # the triangle inequality says that |d1|+|d2| >= |d| and is strict
                    # only if other lies in the line segment
                    return bool(simplify(Eq(abs(d1) + abs(d2) - abs(d), 0)))
                except TypeError:
                    raise Undecidable("Cannot determine if {} is in {}".format(other, self))
        if isinstance(other, Segment):
            return other.p1 in self and other.p2 in self

        return False

    def equals(self, other):
        """Returns True if self and other are the same mathematical entities"""
        return isinstance(other, self.func) and list(
            ordered(self.args)) == list(ordered(other.args))

    def distance(self, other):
        """
        Finds the shortest distance between a line segment and a point.

        Raises
        ======

        NotImplementedError is raised if `other` is not a Point

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 1), Point(3, 4)
        >>> s = Segment(p1, p2)
        >>> s.distance(Point(10, 15))
        sqrt(170)
        >>> s.distance((0, 12))
        sqrt(73)
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 0, 3), Point3D(1, 1, 4)
        >>> s = Segment3D(p1, p2)
        >>> s.distance(Point3D(10, 15, 12))
        sqrt(341)
        >>> s.distance((10, 15, 12))
        sqrt(341)
        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if isinstance(other, Point):
            vp1 = other - self.p1
            vp2 = other - self.p2

            dot_prod_sign_1 = self.direction.dot(vp1) >= 0
            dot_prod_sign_2 = self.direction.dot(vp2) <= 0
            if dot_prod_sign_1 and dot_prod_sign_2:
                return Line(self.p1, self.p2).distance(other)
            if dot_prod_sign_1 and not dot_prod_sign_2:
                return abs(vp2)
            if not dot_prod_sign_1 and dot_prod_sign_2:
                return abs(vp1)
        raise NotImplementedError()

    @property
    def length(self):
        """The length of the line segment.

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 0), Point(4, 3)
        >>> s1 = Segment(p1, p2)
        >>> s1.length
        5
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(4, 3, 3)
        >>> s1 = Segment3D(p1, p2)
        >>> s1.length
        sqrt(34)

        """
        return Point.distance(self.p1, self.p2)

    @property
    def midpoint(self):
        """The midpoint of the line segment.

        See Also
        ========

        sympy.geometry.point.Point.midpoint

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 0), Point(4, 3)
        >>> s1 = Segment(p1, p2)
        >>> s1.midpoint
        Point2D(2, 3/2)
        >>> from sympy import Point3D, Segment3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(4, 3, 3)
        >>> s1 = Segment3D(p1, p2)
        >>> s1.midpoint
        Point3D(2, 3/2, 3/2)

        """
        return Point.midpoint(self.p1, self.p2)

    def perpendicular_bisector(self, p=None):
        """The perpendicular bisector of this segment.

        If no point is specified or the point specified is not on the
        bisector then the bisector is returned as a Line. Otherwise a
        Segment is returned that joins the point specified and the
        intersection of the bisector and the segment.

        Parameters
        ==========

        p : Point

        Returns
        =======

        bisector : Line or Segment

        See Also
        ========

        LinearEntity.perpendicular_segment

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(6, 6), Point(5, 1)
        >>> s1 = Segment(p1, p2)
        >>> s1.perpendicular_bisector()
        Line2D(Point2D(3, 3), Point2D(-3, 9))

        >>> s1.perpendicular_bisector(p3)
        Segment2D(Point2D(5, 1), Point2D(3, 3))

        """
        l = self.perpendicular_line(self.midpoint)
        if p is not None:
            p2 = Point(p, dim=self.ambient_dimension)
            if p2 in l:
                return Segment(p2, self.midpoint)
        return l

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the Segment gives
        values that will produce the full segment in a plot.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Point, Segment
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> s1 = Segment(p1, p2)
        >>> s1.plot_interval()
        [t, 0, 1]

        """
        t = _symbol(parameter, real=True)
        return [t, 0, 1]


class LinearEntity2D(LinearEntity):
    """A base class for all linear entities (line, ray and segment)
    in a 2-dimensional Euclidean space.

    Attributes
    ==========

    p1
    p2
    coefficients
    slope
    points

    Notes
    =====

    This is an abstract class and is not meant to be instantiated.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity

    """
    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """
        verts = self.points
        xs = [p.x for p in verts]
        ys = [p.y for p in verts]
        return (min(xs), min(ys), max(xs), max(ys))

    def perpendicular_line(self, p):
        """Create a new Line perpendicular to this linear entity which passes
        through the point `p`.

        Parameters
        ==========

        p : Point

        Returns
        =======

        line : Line

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular, perpendicular_segment

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2, p3 = Point(0, 0), Point(2, 3), Point(-2, 2)
        >>> L = Line(p1, p2)
        >>> P = L.perpendicular_line(p3); P
        Line2D(Point2D(-2, 2), Point2D(-5, 4))
        >>> L.is_perpendicular(P)
        True

        In 2D, the first point of the perpendicular line is the
        point through which was required to pass; the second
        point is arbitrarily chosen. To get a line that explicitly
        uses a point in the line, create a line from the perpendicular
        segment from the line to the point:

        >>> Line(L.perpendicular_segment(p3))
        Line2D(Point2D(-2, 2), Point2D(4/13, 6/13))
        """
        p = Point(p, dim=self.ambient_dimension)
        # any two lines in R^2 intersect, so blindly making
        # a line through p in an orthogonal direction will work
        # and is faster than finding the projection point as in 3D
        return Line(p, p + self.direction.orthogonal_direction)

    @property
    def slope(self):
        """The slope of this linear entity, or infinity if vertical.

        Returns
        =======

        slope : number or SymPy expression

        See Also
        ========

        coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(0, 0), Point(3, 5)
        >>> l1 = Line(p1, p2)
        >>> l1.slope
        5/3

        >>> p3 = Point(0, 4)
        >>> l2 = Line(p1, p3)
        >>> l2.slope
        oo

        """
        d1, d2 = (self.p1 - self.p2).args
        if d1 == 0:
            return S.Infinity
        return simplify(d2/d1)


class Line2D(LinearEntity2D, Line):
    """An infinite line in space 2D.

    A line is declared with two distinct points or a point and slope
    as defined using keyword `slope`.

    Parameters
    ==========

    p1 : Point
    pt : Point
    slope : SymPy expression

    See Also
    ========

    sympy.geometry.point.Point

    Examples
    ========

    >>> from sympy import Line, Segment, Point
    >>> L = Line(Point(2,3), Point(3,5))
    >>> L
    Line2D(Point2D(2, 3), Point2D(3, 5))
    >>> L.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> L.equation()
    -2*x + y + 1
    >>> L.coefficients
    (-2, 1, 1)

    Instantiate with keyword ``slope``:

    >>> Line(Point(0, 0), slope=0)
    Line2D(Point2D(0, 0), Point2D(1, 0))

    Instantiate with another linear object

    >>> s = Segment((0, 0), (0, 1))
    >>> Line(s).equation()
    x
    """
    def __new__(cls, p1, pt=None, slope=None, **kwargs):
        if isinstance(p1, LinearEntity):
            if pt is not None:
                raise ValueError('When p1 is a LinearEntity, pt should be None')
            p1, pt = Point._normalize_dimension(*p1.args, dim=2)
        else:
            p1 = Point(p1, dim=2)
        if pt is not None and slope is None:
            try:
                p2 = Point(pt, dim=2)
            except (NotImplementedError, TypeError, ValueError):
                raise ValueError(filldedent('''
                    The 2nd argument was not a valid Point.
                    If it was a slope, enter it with keyword "slope".
                    '''))
        elif slope is not None and pt is None:
            slope = sympify(slope)
            if slope.is_finite is False:
                # when infinite slope, don't change x
                dx = 0
                dy = 1
            else:
                # go over 1 up slope
                dx = 1
                dy = slope
            # XXX avoiding simplification by adding to coords directly
            p2 = Point(p1.x + dx, p1.y + dy, evaluate=False)
        else:
            raise ValueError('A 2nd Point or keyword "slope" must be used.')
        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the LinearEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        verts = (N(self.p1), N(self.p2))
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        path = "M {} L {}".format(coords[0], " L ".join(coords[1:]))

        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" '
            'marker-start="url(#markerReverseArrow)" marker-end="url(#markerArrow)"/>'
        ).format(2.*scale_factor, path, fill_color)

    @property
    def coefficients(self):
        """The coefficients (`a`, `b`, `c`) for `ax + by + c = 0`.

        See Also
        ========

        sympy.geometry.line.Line2D.equation

        Examples
        ========

        >>> from sympy import Point, Line
        >>> from sympy.abc import x, y
        >>> p1, p2 = Point(0, 0), Point(5, 3)
        >>> l = Line(p1, p2)
        >>> l.coefficients
        (-3, 5, 0)

        >>> p3 = Point(x, y)
        >>> l2 = Line(p1, p3)
        >>> l2.coefficients
        (-y, x, 0)

        """
        p1, p2 = self.points
        if p1.x == p2.x:
            return (S.One, S.Zero, -p1.x)
        elif p1.y == p2.y:
            return (S.Zero, S.One, -p1.y)
        return tuple([simplify(i) for i in
                      (self.p1.y - self.p2.y,
                       self.p2.x - self.p1.x,
                       self.p1.x*self.p2.y - self.p1.y*self.p2.x)])

    def equation(self, x='x', y='y'):
        """The equation of the line: ax + by + c.

        Parameters
        ==========

        x : str, optional
            The name to use for the x-axis, default value is 'x'.
        y : str, optional
            The name to use for the y-axis, default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        See Also
        ========

        sympy.geometry.line.Line2D.coefficients

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 0), Point(5, 3)
        >>> l1 = Line(p1, p2)
        >>> l1.equation()
        -3*x + 4*y + 3

        """
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        p1, p2 = self.points
        if p1.x == p2.x:
            return x - p1.x
        elif p1.y == p2.y:
            return y - p1.y

        a, b, c = self.coefficients
        return a*x + b*y + c


class Ray2D(LinearEntity2D, Ray):
    """
    A Ray is a semi-line in the space with a source point and a direction.

    Parameters
    ==========

    p1 : Point
        The source of the Ray
    p2 : Point or radian value
        This point determines the direction in which the Ray propagates.
        If given as an angle it is interpreted in radians with the positive
        direction being ccw.

    Attributes
    ==========

    source
    xdirection
    ydirection

    See Also
    ========

    sympy.geometry.point.Point, Line

    Examples
    ========

    >>> from sympy import Point, pi, Ray
    >>> r = Ray(Point(2, 3), Point(3, 5))
    >>> r
    Ray2D(Point2D(2, 3), Point2D(3, 5))
    >>> r.points
    (Point2D(2, 3), Point2D(3, 5))
    >>> r.source
    Point2D(2, 3)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.slope
    2
    >>> Ray(Point(0, 0), angle=pi/4).slope
    1

    """
    def __new__(cls, p1, pt=None, angle=None, **kwargs):
        p1 = Point(p1, dim=2)
        if pt is not None and angle is None:
            try:
                p2 = Point(pt, dim=2)
            except (NotImplementedError, TypeError, ValueError):
                raise ValueError(filldedent('''
                    The 2nd argument was not a valid Point; if
                    it was meant to be an angle it should be
                    given with keyword "angle".'''))
            if p1 == p2:
                raise ValueError('A Ray requires two distinct points.')
        elif angle is not None and pt is None:
            # we need to know if the angle is an odd multiple of pi/2
            angle = sympify(angle)
            c = _pi_coeff(angle)
            p2 = None
            if c is not None:
                if c.is_Rational:
                    if c.q == 2:
                        if c.p == 1:
                            p2 = p1 + Point(0, 1)
                        elif c.p == 3:
                            p2 = p1 + Point(0, -1)
                    elif c.q == 1:
                        if c.p == 0:
                            p2 = p1 + Point(1, 0)
                        elif c.p == 1:
                            p2 = p1 + Point(-1, 0)
                if p2 is None:
                    c *= S.Pi
            else:
                c = angle % (2*S.Pi)
            if not p2:
                m = 2*c/S.Pi
                left = And(1 < m, m < 3)  # is it in quadrant 2 or 3?
                x = Piecewise((-1, left), (Piecewise((0, Eq(m % 1, 0)), (1, True)), True))
                y = Piecewise((-tan(c), left), (Piecewise((1, Eq(m, 1)), (-1, Eq(m, 3)), (tan(c), True)), True))
                p2 = p1 + Point(x, y)
        else:
            raise ValueError('A 2nd point or keyword "angle" must be used.')

        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    @property
    def xdirection(self):
        """The x direction of the ray.

        Positive infinity if the ray points in the positive x direction,
        negative infinity if the ray points in the negative x direction,
        or 0 if the ray is vertical.

        See Also
        ========

        ydirection

        Examples
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, -1)
        >>> r1, r2 = Ray(p1, p2), Ray(p1, p3)
        >>> r1.xdirection
        oo
        >>> r2.xdirection
        0

        """
        if self.p1.x < self.p2.x:
            return S.Infinity
        elif self.p1.x == self.p2.x:
            return S.Zero
        else:
            return S.NegativeInfinity

    @property
    def ydirection(self):
        """The y direction of the ray.

        Positive infinity if the ray points in the positive y direction,
        negative infinity if the ray points in the negative y direction,
        or 0 if the ray is horizontal.

        See Also
        ========

        xdirection

        Examples
        ========

        >>> from sympy import Point, Ray
        >>> p1, p2, p3 = Point(0, 0), Point(-1, -1), Point(-1, 0)
        >>> r1, r2 = Ray(p1, p2), Ray(p1, p3)
        >>> r1.ydirection
        -oo
        >>> r2.ydirection
        0

        """
        if self.p1.y < self.p2.y:
            return S.Infinity
        elif self.p1.y == self.p2.y:
            return S.Zero
        else:
            return S.NegativeInfinity

    def closing_angle(r1, r2):
        """Return the angle by which r2 must be rotated so it faces the same
        direction as r1.

        Parameters
        ==========

        r1 : Ray2D
        r2 : Ray2D

        Returns
        =======

        angle : angle in radians (ccw angle is positive)

        See Also
        ========

        LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import Ray, pi
        >>> r1 = Ray((0, 0), (1, 0))
        >>> r2 = r1.rotate(-pi/2)
        >>> angle = r1.closing_angle(r2); angle
        pi/2
        >>> r2.rotate(angle).direction.unit == r1.direction.unit
        True
        >>> r2.closing_angle(r1)
        -pi/2
        """
        if not all(isinstance(r, Ray2D) for r in (r1, r2)):
            # although the direction property is defined for
            # all linear entities, only the Ray is truly a
            # directed object
            raise TypeError('Both arguments must be Ray2D objects.')

        a1 = atan2(*list(reversed(r1.direction.args)))
        a2 = atan2(*list(reversed(r2.direction.args)))
        if a1*a2 < 0:
            a1 = 2*S.Pi + a1 if a1 < 0 else a1
            a2 = 2*S.Pi + a2 if a2 < 0 else a2
        return a1 - a2


class Segment2D(LinearEntity2D, Segment):
    """A line segment in 2D space.

    Parameters
    ==========

    p1 : Point
    p2 : Point

    Attributes
    ==========

    length : number or SymPy expression
    midpoint : Point

    See Also
    ========

    sympy.geometry.point.Point, Line

    Examples
    ========

    >>> from sympy import Point, Segment
    >>> Segment((1, 0), (1, 1)) # tuples are interpreted as pts
    Segment2D(Point2D(1, 0), Point2D(1, 1))
    >>> s = Segment(Point(4, 3), Point(1, 1)); s
    Segment2D(Point2D(4, 3), Point2D(1, 1))
    >>> s.points
    (Point2D(4, 3), Point2D(1, 1))
    >>> s.slope
    2/3
    >>> s.length
    sqrt(13)
    >>> s.midpoint
    Point2D(5/2, 2)

    """
    def __new__(cls, p1, p2, **kwargs):
        p1 = Point(p1, dim=2)
        p2 = Point(p2, dim=2)

        if p1 == p2:
            return p1

        return LinearEntity2D.__new__(cls, p1, p2, **kwargs)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the LinearEntity.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        verts = (N(self.p1), N(self.p2))
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        path = "M {} L {}".format(coords[0], " L ".join(coords[1:]))
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
        ).format(2.*scale_factor, path, fill_color)


class LinearEntity3D(LinearEntity):
    """An base class for all linear entities (line, ray and segment)
    in a 3-dimensional Euclidean space.

    Attributes
    ==========

    p1
    p2
    direction_ratio
    direction_cosine
    points

    Notes
    =====

    This is a base class and is not meant to be instantiated.
    """
    def __new__(cls, p1, p2, **kwargs):
        p1 = Point3D(p1, dim=3)
        p2 = Point3D(p2, dim=3)
        if p1 == p2:
            # if it makes sense to return a Point, handle in subclass
            raise ValueError(
                "%s.__new__ requires two unique Points." % cls.__name__)

        return GeometryEntity.__new__(cls, p1, p2, **kwargs)

    ambient_dimension = 3

    @property
    def direction_ratio(self):
        """The direction ratio of a given line in 3D.

        See Also
        ========

        sympy.geometry.line.Line3D.equation

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(5, 3, 1)
        >>> l = Line3D(p1, p2)
        >>> l.direction_ratio
        [5, 3, 1]
        """
        p1, p2 = self.points
        return p1.direction_ratio(p2)

    @property
    def direction_cosine(self):
        """The normalized direction ratio of a given line in 3D.

        See Also
        ========

        sympy.geometry.line.Line3D.equation

        Examples
        ========

        >>> from sympy import Point3D, Line3D
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(5, 3, 1)
        >>> l = Line3D(p1, p2)
        >>> l.direction_cosine
        [sqrt(35)/7, 3*sqrt(35)/35, sqrt(35)/35]
        >>> sum(i**2 for i in _)
        1
        """
        p1, p2 = self.points
        return p1.direction_cosine(p2)


class Line3D(LinearEntity3D, Line):
    """An infinite 3D line in space.

    A line is declared with two distinct points or a point and direction_ratio
    as defined using keyword `direction_ratio`.

    Parameters
    ==========

    p1 : Point3D
    pt : Point3D
    direction_ratio : list

    See Also
    ========

    sympy.geometry.point.Point3D
    sympy.geometry.line.Line
    sympy.geometry.line.Line2D

    Examples
    ========

    >>> from sympy import Line3D, Point3D
    >>> L = Line3D(Point3D(2, 3, 4), Point3D(3, 5, 1))
    >>> L
    Line3D(Point3D(2, 3, 4), Point3D(3, 5, 1))
    >>> L.points
    (Point3D(2, 3, 4), Point3D(3, 5, 1))
    """
    def __new__(cls, p1, pt=None, direction_ratio=(), **kwargs):
        if isinstance(p1, LinearEntity3D):
            if pt is not None:
                raise ValueError('if p1 is a LinearEntity, pt must be None.')
            p1, pt = p1.args
        else:
            p1 = Point(p1, dim=3)
        if pt is not None and len(direction_ratio) == 0:
            pt = Point(pt, dim=3)
        elif len(direction_ratio) == 3 and pt is None:
            pt = Point3D(p1.x + direction_ratio[0], p1.y + direction_ratio[1],
                         p1.z + direction_ratio[2])
        else:
            raise ValueError('A 2nd Point or keyword "direction_ratio" must '
                             'be used.')

        return LinearEntity3D.__new__(cls, p1, pt, **kwargs)

    def equation(self, x='x', y='y', z='z'):
        """Return the equations that define the line in 3D.

        Parameters
        ==========

        x : str, optional
            The name to use for the x-axis, default value is 'x'.
        y : str, optional
            The name to use for the y-axis, default value is 'y'.
        z : str, optional
            The name to use for the z-axis, default value is 'z'.

        Returns
        =======

        equation : Tuple of simultaneous equations

        Examples
        ========

        >>> from sympy import Point3D, Line3D, solve
        >>> from sympy.abc import x, y, z
        >>> p1, p2 = Point3D(1, 0, 0), Point3D(5, 3, 0)
        >>> l1 = Line3D(p1, p2)
        >>> eq = l1.equation(x, y, z); eq
        (-3*x + 4*y + 3, z)
        >>> solve(eq.subs(z, 0), (x, y, z))
        {x: 4*y/3 + 1}
        """
        x, y, z, k = [_symbol(i, real=True) for i in (x, y, z, 'k')]
        p1, p2 = self.points
        d1, d2, d3 = p1.direction_ratio(p2)
        x1, y1, z1 = p1
        eqs = [-d1*k + x - x1, -d2*k + y - y1, -d3*k + z - z1]
        # eliminate k from equations by solving first eq with k for k
        for i, e in enumerate(eqs):
            if e.has(k):
                kk = solve(e, k)[0]
                eqs.pop(i)
                break
        return Tuple(*[i.subs(k, kk).as_numer_denom()[0] for i in eqs])

    def distance(self, other):
        """
        Finds the shortest distance between a line and another object.

        Parameters
        ==========

        Point3D, Line3D, Plane, tuple, list

        Returns
        =======

        distance

        Notes
        =====

        This method accepts only 3D entities as it's parameter

        Tuples and lists are converted to Point3D and therefore must be of
        length 3, 2 or 1.

        NotImplementedError is raised if `other` is not an instance of one
        of the specified classes: Point3D, Line3D, or Plane.

        Examples
        ========

        >>> from sympy.geometry import Line3D
        >>> l1 = Line3D((0, 0, 0), (0, 0, 1))
        >>> l2 = Line3D((0, 1, 0), (1, 1, 1))
        >>> l1.distance(l2)
        1

        The computed distance may be symbolic, too:

        >>> from sympy.abc import x, y
        >>> l1 = Line3D((0, 0, 0), (0, 0, 1))
        >>> l2 = Line3D((0, x, 0), (y, x, 1))
        >>> l1.distance(l2)
        Abs(x*y)/Abs(sqrt(y**2))

        """

        from .plane import Plane  # Avoid circular import

        if isinstance(other, (tuple, list)):
            try:
                other = Point3D(other)
            except ValueError:
                pass

        if isinstance(other, Point3D):
            return super().distance(other)

        if isinstance(other, Line3D):
            if self == other:
                return S.Zero
            if self.is_parallel(other):
                return super().distance(other.p1)

            # Skew lines
            self_direction = Matrix(self.direction_ratio)
            other_direction = Matrix(other.direction_ratio)
            normal = self_direction.cross(other_direction)
            plane_through_self = Plane(p1=self.p1, normal_vector=normal)
            return other.p1.distance(plane_through_self)

        if isinstance(other, Plane):
            return other.distance(self)

        msg = f"{other} has type {type(other)}, which is unsupported"
        raise NotImplementedError(msg)


class Ray3D(LinearEntity3D, Ray):
    """
    A Ray is a semi-line in the space with a source point and a direction.

    Parameters
    ==========

    p1 : Point3D
        The source of the Ray
    p2 : Point or a direction vector
    direction_ratio: Determines the direction in which the Ray propagates.


    Attributes
    ==========

    source
    xdirection
    ydirection
    zdirection

    See Also
    ========

    sympy.geometry.point.Point3D, Line3D


    Examples
    ========

    >>> from sympy import Point3D, Ray3D
    >>> r = Ray3D(Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r
    Ray3D(Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r.points
    (Point3D(2, 3, 4), Point3D(3, 5, 0))
    >>> r.source
    Point3D(2, 3, 4)
    >>> r.xdirection
    oo
    >>> r.ydirection
    oo
    >>> r.direction_ratio
    [1, 2, -4]

    """
    def __new__(cls, p1, pt=None, direction_ratio=(), **kwargs):
        if isinstance(p1, LinearEntity3D):
            if pt is not None:
                raise ValueError('If p1 is a LinearEntity, pt must be None')
            p1, pt = p1.args
        else:
            p1 = Point(p1, dim=3)
        if pt is not None and len(direction_ratio) == 0:
            pt = Point(pt, dim=3)
        elif len(direction_ratio) == 3 and pt is None:
            pt = Point3D(p1.x + direction_ratio[0], p1.y + direction_ratio[1],
                         p1.z + direction_ratio[2])
        else:
            raise ValueError(filldedent('''
                A 2nd Point or keyword "direction_ratio" must be used.
            '''))

        return LinearEntity3D.__new__(cls, p1, pt, **kwargs)

    @property
    def xdirection(self):
        """The x direction of the ray.

        Positive infinity if the ray points in the positive x direction,
        negative infinity if the ray points in the negative x direction,
        or 0 if the ray is vertical.

        See Also
        ========

        ydirection

        Examples
        ========

        >>> from sympy import Point3D, Ray3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, -1, 0)
        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)
        >>> r1.xdirection
        oo
        >>> r2.xdirection
        0

        """
        if self.p1.x < self.p2.x:
            return S.Infinity
        elif self.p1.x == self.p2.x:
            return S.Zero
        else:
            return S.NegativeInfinity

    @property
    def ydirection(self):
        """The y direction of the ray.

        Positive infinity if the ray points in the positive y direction,
        negative infinity if the ray points in the negative y direction,
        or 0 if the ray is horizontal.

        See Also
        ========

        xdirection

        Examples
        ========

        >>> from sympy import Point3D, Ray3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(-1, -1, -1), Point3D(-1, 0, 0)
        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)
        >>> r1.ydirection
        -oo
        >>> r2.ydirection
        0

        """
        if self.p1.y < self.p2.y:
            return S.Infinity
        elif self.p1.y == self.p2.y:
            return S.Zero
        else:
            return S.NegativeInfinity

    @property
    def zdirection(self):
        """The z direction of the ray.

        Positive infinity if the ray points in the positive z direction,
        negative infinity if the ray points in the negative z direction,
        or 0 if the ray is horizontal.

        See Also
        ========

        xdirection

        Examples
        ========

        >>> from sympy import Point3D, Ray3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(-1, -1, -1), Point3D(-1, 0, 0)
        >>> r1, r2 = Ray3D(p1, p2), Ray3D(p1, p3)
        >>> r1.ydirection
        -oo
        >>> r2.ydirection
        0
        >>> r2.zdirection
        0

        """
        if self.p1.z < self.p2.z:
            return S.Infinity
        elif self.p1.z == self.p2.z:
            return S.Zero
        else:
            return S.NegativeInfinity


class Segment3D(LinearEntity3D, Segment):
    """A line segment in a 3D space.

    Parameters
    ==========

    p1 : Point3D
    p2 : Point3D

    Attributes
    ==========

    length : number or SymPy expression
    midpoint : Point3D

    See Also
    ========

    sympy.geometry.point.Point3D, Line3D

    Examples
    ========

    >>> from sympy import Point3D, Segment3D
    >>> Segment3D((1, 0, 0), (1, 1, 1)) # tuples are interpreted as pts
    Segment3D(Point3D(1, 0, 0), Point3D(1, 1, 1))
    >>> s = Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7)); s
    Segment3D(Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.points
    (Point3D(4, 3, 9), Point3D(1, 1, 7))
    >>> s.length
    sqrt(17)
    >>> s.midpoint
    Point3D(5/2, 2, 8)

    """
    def __new__(cls, p1, p2, **kwargs):
        p1 = Point(p1, dim=3)
        p2 = Point(p2, dim=3)

        if p1 == p2:
            return p1

        return LinearEntity3D.__new__(cls, p1, p2, **kwargs)
<|endoftext|>"""Geometrical Points.

Contains
========
Point
Point2D
Point3D

When methods of Point require 1 or more points as arguments, they
can be passed as a sequence of coordinates or Points:

>>> from sympy import Point
>>> Point(1, 1).is_collinear((2, 2), (3, 4))
False
>>> Point(1, 1).is_collinear(Point(2, 2), Point(3, 4))
False

"""

import warnings

from sympy.core import S, sympify, Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.numbers import Float
from sympy.core.parameters import global_parameters
from sympy.simplify.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import Matrix
from sympy.matrices.expressions import Transpose
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable

from .entity import GeometryEntity

from mpmath.libmp.libmpf import prec_to_dps


class Point(GeometryEntity):
    """A point in a n-dimensional Euclidean space.

    Parameters
    ==========

    coords : sequence of n-coordinate values. In the special
        case where n=2 or 3, a Point2D or Point3D will be created
        as appropriate.
    evaluate : if `True` (default), all floats are turn into
        exact types.
    dim : number of coordinates the point should have.  If coordinates
        are unspecified, they are padded with zeros.
    on_morph : indicates what should happen when the number of
        coordinates of a point need to be changed by adding or
        removing zeros.  Possible values are `'warn'`, `'error'`, or
        `ignore` (default).  No warning or error is given when `*args`
        is empty and `dim` is given. An error is always raised when
        trying to remove nonzero coordinates.


    Attributes
    ==========

    length
    origin: A `Point` representing the origin of the
        appropriately-dimensioned space.

    Raises
    ======

    TypeError : When instantiating with anything but a Point or sequence
    ValueError : when instantiating with a sequence with length < 2 or
        when trying to reduce dimensions if keyword `on_morph='error'` is
        set.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy import Point
    >>> from sympy.abc import x
    >>> Point(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point([1, 2])
    Point2D(1, 2)
    >>> Point(0, x)
    Point2D(0, x)
    >>> Point(dim=4)
    Point(0, 0, 0, 0)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    is_Point = True

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        on_morph = kwargs.get('on_morph', 'ignore')

        # unpack into coords
        coords = args[0] if len(args) == 1 else args

        # check args and handle quickly handle Point instances
        if isinstance(coords, Point):
            # even if we're mutating the dimension of a point, we
            # don't reevaluate its coordinates
            evaluate = False
            if len(coords) == kwargs.get('dim', len(coords)):
                return coords

        if not is_sequence(coords):
            raise TypeError(filldedent('''
                Expecting sequence of coordinates, not `{}`'''
                                       .format(func_name(coords))))
        # A point where only `dim` is specified is initialized
        # to zeros.
        if len(coords) == 0 and kwargs.get('dim', None):
            coords = (S.Zero,)*kwargs.get('dim')

        coords = Tuple(*coords)
        dim = kwargs.get('dim', len(coords))

        if len(coords) < 2:
            raise ValueError(filldedent('''
                Point requires 2 or more coordinates or
                keyword `dim` > 1.'''))
        if len(coords) != dim:
            message = ("Dimension of {} needs to be changed "
                       "from {} to {}.").format(coords, len(coords), dim)
            if on_morph == 'ignore':
                pass
            elif on_morph == "error":
                raise ValueError(message)
            elif on_morph == 'warn':
                warnings.warn(message, stacklevel=2)
            else:
                raise ValueError(filldedent('''
                        on_morph value should be 'error',
                        'warn' or 'ignore'.'''))
        if any(coords[dim:]):
            raise ValueError('Nonzero coordinates cannot be removed.')
        if any(a.is_number and im(a).is_zero is False for a in coords):
            raise ValueError('Imaginary coordinates are not permitted.')
        if not all(isinstance(a, Expr) for a in coords):
            raise TypeError('Coordinates must be valid SymPy expressions.')

        # pad with zeros appropriately
        coords = coords[:dim] + (S.Zero,)*(dim - len(coords))

        # Turn any Floats into rationals and simplify
        # any expressions before we instantiate
        if evaluate:
            coords = coords.xreplace({
                f: simplify(nsimplify(f, rational=True))
                 for f in coords.atoms(Float)})

        # return 2D or 3D instances
        if len(coords) == 2:
            kwargs['_nocheck'] = True
            return Point2D(*coords, **kwargs)
        elif len(coords) == 3:
            kwargs['_nocheck'] = True
            return Point3D(*coords, **kwargs)

        # the general Point
        return GeometryEntity.__new__(cls, *coords)

    def __abs__(self):
        """Returns the distance between this point and the origin."""
        origin = Point([0]*len(self))
        return Point.distance(origin, self)

    def __add__(self, other):
        """Add other to self by incrementing self's coordinates by
        those of other.

        Notes
        =====

        >>> from sympy import Point

        When sequences of coordinates are passed to Point methods, they
        are converted to a Point internally. This __add__ method does
        not do that so if floating point values are used, a floating
        point result (in terms of SymPy Floats) will be returned.

        >>> Point(1, 2) + (.1, .2)
        Point2D(1.1, 2.2)

        If this is not desired, the `translate` method can be used or
        another Point can be added:

        >>> Point(1, 2).translate(.1, .2)
        Point2D(11/10, 11/5)
        >>> Point(1, 2) + Point(.1, .2)
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.translate

        """
        try:
            s, o = Point._normalize_dimension(self, Point(other, evaluate=False))
        except TypeError:
            raise GeometryError("Don't know how to add {} and a Point object".format(other))

        coords = [simplify(a + b) for a, b in zip(s, o)]
        return Point(coords, evaluate=False)

    def __contains__(self, item):
        return item in self.args

    def __truediv__(self, divisor):
        """Divide point's coordinates by a factor."""
        divisor = sympify(divisor)
        coords = [simplify(x/divisor) for x in self.args]
        return Point(coords, evaluate=False)

    def __eq__(self, other):
        if not isinstance(other, Point) or len(self.args) != len(other.args):
            return False
        return self.args == other.args

    def __getitem__(self, key):
        return self.args[key]

    def __hash__(self):
        return hash(self.args)

    def __iter__(self):
        return self.args.__iter__()

    def __len__(self):
        return len(self.args)

    def __mul__(self, factor):
        """Multiply point's coordinates by a factor.

        Notes
        =====

        >>> from sympy import Point

        When multiplying a Point by a floating point number,
        the coordinates of the Point will be changed to Floats:

        >>> Point(1, 2)*0.1
        Point2D(0.1, 0.2)

        If this is not desired, the `scale` method can be used or
        else only multiply or divide by integers:

        >>> Point(1, 2).scale(1.1, 1.1)
        Point2D(11/10, 11/5)
        >>> Point(1, 2)*11/10
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.scale
        """
        factor = sympify(factor)
        coords = [simplify(x*factor) for x in self.args]
        return Point(coords, evaluate=False)

    def __rmul__(self, factor):
        """Multiply a factor by point's coordinates."""
        return self.__mul__(factor)

    def __neg__(self):
        """Negate the point."""
        coords = [-x for x in self.args]
        return Point(coords, evaluate=False)

    def __sub__(self, other):
        """Subtract two points, or subtract a factor from this point's
        coordinates."""
        return self + [-x for x in other]

    @classmethod
    def _normalize_dimension(cls, *points, **kwargs):
        """Ensure that points have the same dimension.
        By default `on_morph='warn'` is passed to the
        `Point` constructor."""
        # if we have a built-in ambient dimension, use it
        dim = getattr(cls, '_ambient_dimension', None)
        # override if we specified it
        dim = kwargs.get('dim', dim)
        # if no dim was given, use the highest dimensional point
        if dim is None:
            dim = max(i.ambient_dimension for i in points)
        if all(i.ambient_dimension == dim for i in points):
            return list(points)
        kwargs['dim'] = dim
        kwargs['on_morph'] = kwargs.get('on_morph', 'warn')
        return [Point(i, **kwargs) for i in points]

    @staticmethod
    def affine_rank(*args):
        """The affine rank of a set of points is the dimension
        of the smallest affine space containing all the points.
        For example, if the points lie on a line (and are not all
        the same) their affine rank is 1.  If the points lie on a plane
        but not a line, their affine rank is 2.  By convention, the empty
        set has affine rank -1."""

        if len(args) == 0:
            return -1
        # make sure we're genuinely points
        # and translate every point to the origin
        points = Point._normalize_dimension(*[Point(i) for i in args])
        origin = points[0]
        points = [i - origin for i in points[1:]]

        m = Matrix([i.args for i in points])
        # XXX fragile -- what is a better way?
        return m.rank(iszerofunc = lambda x:
            abs(x.n(2)) < 1e-12 if x.is_number else x.is_zero)

    @property
    def ambient_dimension(self):
        """Number of components this point has."""
        return getattr(self, '_ambient_dimension', len(self))

    @classmethod
    def are_coplanar(cls, *points):
        """Return True if there exists a plane in which all the points
        lie.  A trivial True value is returned if `len(points) < 3` or
        all Points are 2-dimensional.

        Parameters
        ==========

        A set of points

        Raises
        ======

        ValueError : if less than 3 unique points are given

        Returns
        =======

        boolean

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 2)
        >>> p2 = Point3D(2, 7, 2)
        >>> p3 = Point3D(0, 0, 2)
        >>> p4 = Point3D(1, 1, 2)
        >>> Point3D.are_coplanar(p1, p2, p3, p4)
        True
        >>> p5 = Point3D(0, 1, 3)
        >>> Point3D.are_coplanar(p1, p2, p3, p5)
        False

        """
        if len(points) <= 1:
            return True

        points = cls._normalize_dimension(*[Point(i) for i in points])
        # quick exit if we are in 2D
        if points[0].ambient_dimension == 2:
            return True
        points = list(uniq(points))
        return Point.affine_rank(*points) <= 2

    def distance(self, other):
        """The Euclidean distance between self and another GeometricEntity.

        Returns
        =======

        distance : number or symbolic expression.

        Raises
        ======

        TypeError : if other is not recognized as a GeometricEntity or is a
                    GeometricEntity for which distance is not defined.

        See Also
        ========

        sympy.geometry.line.Segment.length
        sympy.geometry.point.Point.taxicab_distance

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> l = Line((3, 1), (2, 2))
        >>> p1.distance(p2)
        5
        >>> p1.distance(l)
        sqrt(2)

        The computed distance may be symbolic, too:

        >>> from sympy.abc import x, y
        >>> p3 = Point(x, y)
        >>> p3.distance((0, 0))
        sqrt(x**2 + y**2)

        """
        if not isinstance(other, GeometryEntity):
            try:
                other = Point(other, dim=self.ambient_dimension)
            except TypeError:
                raise TypeError("not recognized as a GeometricEntity: %s" % type(other))
        if isinstance(other, Point):
            s, p = Point._normalize_dimension(self, Point(other))
            return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
        distance = getattr(other, 'distance', None)
        if distance is None:
            raise TypeError("distance between Point and %s is not defined" % type(other))
        return distance(self)

    def dot(self, p):
        """Return dot product of self with another Point."""
        if not is_sequence(p):
            p = Point(p)  # raise the error via Point
        return Add(*(a*b for a, b in zip(self, p)))

    def equals(self, other):
        """Returns whether the coordinates of self and other agree."""
        # a point is equal to another point if all its components are equal
        if not isinstance(other, Point) or len(self) != len(other):
            return False
        return all(a.equals(b) for a, b in zip(self, other))

    def _eval_evalf(self, prec=15, **options):
        """Evaluate the coordinates of the point.

        This method will, where possible, create and return a new Point
        where the coordinates are evaluated as floating point numbers to
        the precision indicated (default=15).

        Parameters
        ==========

        prec : int

        Returns
        =======

        point : Point

        Examples
        ========

        >>> from sympy import Point, Rational
        >>> p1 = Point(Rational(1, 2), Rational(3, 2))
        >>> p1
        Point2D(1/2, 3/2)
        >>> p1.evalf()
        Point2D(0.5, 1.5)

        """
        dps = prec_to_dps(prec)
        coords = [x.evalf(n=dps, **options) for x in self.args]
        return Point(*coords, evaluate=False)

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : GeometryEntity or sequence of coordinates

        Returns
        =======

        intersection : list of Points

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point2D(0, 0)]

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other)
        if isinstance(other, Point):
            if self == other:
                return [self]
            p1, p2 = Point._normalize_dimension(self, other)
            if p1 == self and p1 == p2:
                return [self]
            return []
        return other.intersection(self)

    def is_collinear(self, *args):
        """Returns `True` if there exists a line
        that contains `self` and `points`.  Returns `False` otherwise.
        A trivially True value is returned if no points are given.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Point
        >>> from sympy.abc import x
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> p3, p4, p5 = Point(2, 2), Point(x, x), Point(1, 2)
        >>> Point.is_collinear(p1, p2, p3, p4)
        True
        >>> Point.is_collinear(p1, p2, p3, p5)
        False

        """
        points = (self,) + args
        points = Point._normalize_dimension(*[Point(i) for i in points])
        points = list(uniq(points))
        return Point.affine_rank(*points) <= 1

    def is_concyclic(self, *args):
        """Do `self` and the given sequence of points lie in a circle?

        Returns True if the set of points are concyclic and
        False otherwise. A trivial value of True is returned
        if there are fewer than 2 other points.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_concyclic : boolean


        Examples
        ========

        >>> from sympy import Point

        Define 4 points that are on the unit circle:

        >>> p1, p2, p3, p4 = Point(1, 0), (0, 1), (-1, 0), (0, -1)

        >>> p1.is_concyclic() == p1.is_concyclic(p2, p3, p4) == True
        True

        Define a point not on that circle:

        >>> p = Point(1, 1)

        >>> p.is_concyclic(p1, p2, p3)
        False

        """
        points = (self,) + args
        points = Point._normalize_dimension(*[Point(i) for i in points])
        points = list(uniq(points))
        if not Point.affine_rank(*points) <= 2:
            return False
        origin = points[0]
        points = [p - origin for p in points]
        # points are concyclic if they are coplanar and
        # there is a point c so that ||p_i-c|| == ||p_j-c|| for all
        # i and j.  Rearranging this equation gives us the following
        # condition: the matrix `mat` must not a pivot in the last
        # column.
        mat = Matrix([list(i) + [i.dot(i)] for i in points])
        rref, pivots = mat.rref()
        if len(origin) not in pivots:
            return True
        return False

    @property
    def is_nonzero(self):
        """True if any coordinate is nonzero, False if every coordinate is zero,
        and None if it cannot be determined."""
        is_zero = self.is_zero
        if is_zero is None:
            return None
        return not is_zero

    def is_scalar_multiple(self, p):
        """Returns whether each coordinate of `self` is a scalar
        multiple of the corresponding coordinate in point p.
        """
        s, o = Point._normalize_dimension(self, Point(p))
        # 2d points happen a lot, so optimize this function call
        if s.ambient_dimension == 2:
            (x1, y1), (x2, y2) = s.args, o.args
            rv = (x1*y2 - x2*y1).equals(0)
            if rv is None:
                raise Undecidable(filldedent(
                    '''Cannot determine if %s is a scalar multiple of
                    %s''' % (s, o)))

        # if the vectors p1 and p2 are linearly dependent, then they must
        # be scalar multiples of each other
        m = Matrix([s.args, o.args])
        return m.rank() < 2

    @property
    def is_zero(self):
        """True if every coordinate is zero, False if any coordinate is not zero,
        and None if it cannot be determined."""
        nonzero = [x.is_nonzero for x in self.args]
        if any(nonzero):
            return False
        if any(x is None for x in nonzero):
            return None
        return True

    @property
    def length(self):
        """
        Treating a Point as a Line, this returns 0 for the length of a Point.

        Examples
        ========

        >>> from sympy import Point
        >>> p = Point(0, 1)
        >>> p.length
        0
        """
        return S.Zero

    def midpoint(self, p):
        """The midpoint between self and point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        midpoint : Point

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(13, 5)
        >>> p1.midpoint(p2)
        Point2D(7, 3)

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return Point([simplify((a + b)*S.Half) for a, b in zip(s, p)])

    @property
    def origin(self):
        """A point of all zeros of the same ambient dimension
        as the current point"""
        return Point([0]*len(self), evaluate=False)

    @property
    def orthogonal_direction(self):
        """Returns a non-zero point that is orthogonal to the
        line containing `self` and the origin.

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2, 3)
        >>> a.orthogonal_direction
        Point3D(-2, 1, 0)
        >>> b = _
        >>> Line(b, b.origin).is_perpendicular(Line(a, a.origin))
        True
        """
        dim = self.ambient_dimension
        # if a coordinate is zero, we can put a 1 there and zeros elsewhere
        if self[0].is_zero:
            return Point([1] + (dim - 1)*[0])
        if self[1].is_zero:
            return Point([0,1] + (dim - 2)*[0])
        # if the first two coordinates aren't zero, we can create a non-zero
        # orthogonal vector by swapping them, negating one, and padding with zeros
        return Point([-self[1], self[0]] + (dim - 2)*[0])

    @staticmethod
    def project(a, b):
        """Project the point `a` onto the line between the origin
        and point `b` along the normal direction.

        Parameters
        ==========

        a : Point
        b : Point

        Returns
        =======

        p : Point

        See Also
        ========

        sympy.geometry.line.LinearEntity.projection

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2)
        >>> b = Point(2, 5)
        >>> z = a.origin
        >>> p = Point.project(a, b)
        >>> Line(p, a).is_perpendicular(Line(p, b))
        True
        >>> Point.is_collinear(z, p, b)
        True
        """
        a, b = Point._normalize_dimension(Point(a), Point(b))
        if b.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        return b*(a.dot(b) / b.dot(b))

    def taxicab_distance(self, p):
        """The Taxicab Distance from self to point p.

        Returns the sum of the horizontal and vertical distances to point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        taxicab_distance : The sum of the horizontal
