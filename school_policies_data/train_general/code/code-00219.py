            sycl_post_cflags = None
            sycl_dlink_post_cflags = None
            if with_sycl:
                sycl_cflags = common_cflags + win_filter_msvc_include_dirs(pp_opts) + _COMMON_SYCL_FLAGS
                if isinstance(extra_postargs, dict):
                    sycl_post_cflags = extra_postargs['sycl']
                else:
                    sycl_post_cflags = list(extra_postargs)
                _append_sycl_targets_if_missing(sycl_post_cflags)
                append_std17_if_no_std_present(sycl_cflags)
                _append_sycl_std_if_no_std_present(sycl_cflags)
                host_cflags = common_cflags + pp_opts + post_cflags
                append_std17_if_no_std_present(host_cflags)

                sycl_cflags = _nt_quote_args(sycl_cflags)
                host_cflags = _nt_quote_args(host_cflags)

                sycl_cflags += _wrap_sycl_host_flags(host_cflags)
                sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
                sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_post_cflags)
                sycl_post_cflags = _nt_quote_args(sycl_post_cflags)


            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=cflags,
                post_cflags=post_cflags,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                sycl_cflags=sycl_cflags,
                sycl_post_cflags=sycl_post_cflags,
                sycl_dlink_post_cflags=sycl_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
                with_sycl=with_sycl)

            # Return *all* object filenames, not just the ones we just built.
            return objects
        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511  # codespell:ignore
        if self.compiler.compiler_type == 'msvc':
            if self.use_ninja:
                self.compiler.compile = win_wrap_ninja_compile
            else:
                self.compiler.compile = win_wrap_single_compile
        else:
            if self.use_ninja:
                self.compiler.compile = unix_wrap_ninja_compile
            else:
                self.compiler._compile = unix_wrap_single_compile

        super().build_extensions()

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split('.')
            # Remove ABI component only if it actually exists in a file name, see gh-170542.
            if len(ext_filename_parts) > 2:
                # Omit the second to last element.
                without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
                ext_filename = '.'.join(without_abi)
        return ext_filename

    def get_export_symbols(self, ext):
        if IS_WINDOWS:
            # Skips exporting the module "PyInit_" function that the
            # distutils Extension.get_export_symbols would add to
            # ext.export_symbols. Only relevant for Windows builds.
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def _check_abi(self) -> tuple[str, TorchVersion]:
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()
        _, version = get_compiler_abi_compatibility_and_version(compiler)
        # Warn user if VC env is activated but `DISTUILS_USE_SDK` is not set.
        if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and 'DISTUTILS_USE_SDK' not in os.environ:
            msg = ('It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.'
                   'This may lead to multiple activations of the VC env.'
                   'Please set `DISTUTILS_USE_SDK=1` and try again.')
            raise UserWarning(msg)
        return compiler, version

    def _add_compile_flag(self, extension, flag) -> None:
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    # Simple hipify, replace the first occurrence of CUDA with HIP
    # in flags starting with "-" and containing "CUDA", but exclude -I flags
    def _hipify_compile_flags(self, extension) -> None:
        if isinstance(extension.extra_compile_args, dict) and 'nvcc' in extension.extra_compile_args:
            modified_flags = []
            for flag in extension.extra_compile_args['nvcc']:
                if flag.startswith("-") and "CUDA" in flag and not flag.startswith("-I"):
                    # check/split flag into flag and value
                    parts = flag.split("=", 1)
                    if len(parts) == 2:
                        flag_part, value_part = parts
                        # replace first instance of "CUDA" with "HIP" only in the flag and not flag value
                        modified_flag_part = flag_part.replace("CUDA", "HIP", 1)
                        modified_flag = f"{modified_flag_part}={value_part}"
                    else:
                        # replace first instance of "CUDA" with "HIP" in flag
                        modified_flag = flag.replace("CUDA", "HIP", 1)
                    modified_flags.append(modified_flag)
                    logger.info('Modified flag: %s -> %s', flag, modified_flag)
                else:
                    modified_flags.append(flag)
            extension.extra_compile_args['nvcc'] = modified_flags

    def _define_torch_extension_name(self, extension) -> None:
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split('.')
        name = names[-1]
        define = f'-DTORCH_EXTENSION_NAME={name}'
        self._add_compile_flag(extension, define)


_BUILD_EXTENSION_CLASS = BuildExtension
del BuildExtension


def CppExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
        ...     name='extension',
        ...     ext_modules=[
        ...         CppExtension(
        ...             name='extension',
        ...             sources=['extension.cpp'],
        ...             extra_compile_args=['-g'],
        ...             extra_link_args=['-Wl,--no-as-needed', '-lm'])
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })
    """
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs

    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths()
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    if not kwargs.get('py_limited_api', False):
        # torch_python uses more than the python limited api
        libraries.append('torch_python')
    if IS_WINDOWS:
        libraries.append("sleef")

    kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        >>> setup(
        ...     name='cuda_extension',
        ...     ext_modules=[
        ...         CUDAExtension(
        ...                 name='cuda_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.cu'],
        ...                 extra_compile_args={'cxx': ['-g'],
        ...                                     'nvcc': ['-O2']},
        ...                 extra_link_args=['-Wl,--no-as-needed', '-lcuda'])
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })

    Compute capabilities:

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension, plus PTX. If down the road a new card is installed the
    extension may need to be recompiled. If a visible card has a compute capability (CC) that's
    newer than the newest version for which your nvcc can build fully-compiled binaries, PyTorch
    will make nvcc fall back to building kernels with the newest version of PTX your nvcc does
    support (see below for details on PTX).

    You can override the default behavior using `TORCH_CUDA_ARCH_LIST` to explicitly specify which
    CCs you want the extension to support:

    ``TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py``
    ``TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py``

    The +PTX option causes extension kernel binaries to include PTX instructions for the specified
    CC. PTX is an intermediate representation that allows kernels to runtime-compile for any CC >=
    the specified CC (for example, 8.6+PTX generates PTX that can runtime-compile for any GPU with
    CC >= 8.6). This improves your binary's forward compatibility. However, relying on older PTX to
    provide forward compat by runtime-compiling for newer CCs can modestly reduce performance on
    those newer CCs. If you know exact CC(s) of the GPUs you want to target, you're always better
    off specifying them individually. For example, if you want your extension to run on 8.0 and 8.6,
    "8.0+PTX" would work functionally because it includes PTX that can runtime-compile for 8.6, but
    "8.0 8.6" would be better.

    Note that while it's possible to include all supported archs, the more archs get included the
    slower the building process will be, as it will build a separate kernel image for each arch.

    Note that CUDA-11.5 nvcc will hit internal compiler error while parsing torch/extension.h on Windows.
    To workaround the issue, move python binding logic to pure C++ file.

    Example use:
        #include <ATen/ATen.h>
        at::Tensor SigmoidAlphaBlendForwardCuda(....)

    Instead of:
        #include <torch/extension.h>
        torch::Tensor SigmoidAlphaBlendForwardCuda(...)

    Currently open issue for nvcc bug: https://github.com/pytorch/pytorch/issues/69460
    Complete workaround code example: https://github.com/facebookresearch/pytorch3d/commit/cb170ac024a949f1f9614ffe6af1c38d972f7d48

    Relocatable device code linking:

    If you want to reference device symbols across compilation units (across object files),
    the object files need to be built with `relocatable device code` (-rdc=true or -dc).
    An exception to this rule is "dynamic parallelism" (nested kernel launches)  which is not used a lot anymore.
    `Relocatable device code` is less optimized so it needs to be used only on object files that need it.
    Using `-dlto` (Device Link Time Optimization) at the device code compilation step and `dlink` step
    helps reduce the potential perf degradation of `-rdc`.
    Note that it needs to be used at both steps to be useful.

    If you have `rdc` objects you need to have an extra `-dlink` (device linking) step before the CPU symbol linking step.
    There is also a case where `-dlink` is used without `-rdc`:
    when an extension is linked against a static lib containing rdc-compiled objects
    like the [NVSHMEM library](https://developer.nvidia.com/nvshmem).

    Note: Ninja is required to build a CUDA Extension with RDC linking.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> CUDAExtension(
        ...        name='cuda_extension',
        ...        sources=['extension.cpp', 'extension_kernel.cu'],
        ...        dlink=True,
        ...        dlink_libraries=["dlink_lib"],
        ...        extra_compile_args={'cxx': ['-g'],
        ...                            'nvcc': ['-O2', '-rdc=true']})
    """
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(device_type="cuda")
    kwargs['library_dirs'] = library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    if not kwargs.get('py_limited_api', False):
        # torch_python uses more than the python limited api
        libraries.append('torch_python')
    if IS_HIP_EXTENSION:
        libraries.append('amdhip64')
        libraries.append('c10_hip')
        libraries.append('torch_hip')
    else:
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])

    if IS_HIP_EXTENSION:
        from .hipify import hipify_python
        build_dir = os.getcwd()
        hipify_result = hipify_python.hipify(
            project_directory=build_dir,
            output_directory=build_dir,
            header_include_dirs=include_dirs,
            includes=[os.path.join(build_dir, '*')],  # limit scope to build_dir only
            extra_files=[os.path.abspath(s) for s in sources],
            show_detailed=True,
            is_pytorch_extension=True,
            hipify_extra_files_only=True,  # don't hipify everything in includes path
        )

        hipified_sources = set()
        for source in sources:
            s_abs = os.path.abspath(source)
            hipified_s_abs = (hipify_result[s_abs].hipified_path if (s_abs in hipify_result and
                              hipify_result[s_abs].hipified_path is not None) else s_abs)
            # setup() arguments must *always* be /-separated paths relative to the setup.py directory,
            # *never* absolute paths
            try:
                hip_path = os.path.relpath(hipified_s_abs, build_dir)
            except ValueError:
                # Cross-drive on Windows: no relative path exists; fall back to absolute (#91797).
                hip_path = hipified_s_abs
            hipified_sources.add(hip_path)

        sources = list(hipified_sources)

    include_dirs += include_paths(device_type="cuda")
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    dlink_libraries = kwargs.get('dlink_libraries', [])
    dlink = kwargs.get('dlink', False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get('extra_compile_args', {})

        extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
        extra_compile_args_dlink += ['-dlink']
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]

        if (torch.version.cuda is not None) and TorchVersion(torch.version.cuda) >= '11.2':
            extra_compile_args_dlink += ['-dlto']   # Device Link Time Optimization started from cuda 11.2

        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink

        kwargs['extra_compile_args'] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)


def SyclExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for SYCL/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a SYCL/C++
    extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import BuildExtension, SyclExtension
        >>> setup(
        ...     name='xpu_extension',
        ...     ext_modules=[
        ...     SyclExtension(
        ...                 name='xpu_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.cpp'],
        ...                 extra_compile_args={'cxx': ['-g', '-std=c++20', '-fPIC']})
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension. If down the road a new card is installed the
    extension may need to be recompiled. You can override the default behavior using
    `TORCH_XPU_ARCH_LIST` to explicitly specify which device architectures you want the extension
    to support:

    ``TORCH_XPU_ARCH_LIST="pvc,xe-lpg" python build_my_extension.py``

    Note that while it's possible to include all supported archs, the more archs get included the
    slower the building process will be, as it will build a separate kernel image for each arch.

    Note: Ninja is required to build SyclExtension.
    """
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += library_paths()
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("c10_xpu")
    libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("sycl")
    if not kwargs.get('py_limited_api', False):
        # torch_python uses more than the python limited api
        libraries.append("torch_python")
    libraries.append("torch_xpu")
    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += include_paths(device_type="xpu")
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    return setuptools.Extension(name, sources, *args, **kwargs)

def include_paths(device_type: str = "cpu", torch_include_dirs=True) -> list[str]:
    """
    Get the include paths required to build a C++ or CUDA or SYCL extension.

    Args:
        device_type: Defaults to "cpu".
    Returns:
        A list of include path strings.
    """
    paths = []
    lib_include = os.path.join(_TORCH_PATH, 'include')
    if torch_include_dirs:
        paths.extend([
            lib_include,
            # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
            os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'),
        ])
    if device_type == "cuda" and IS_HIP_EXTENSION:
        paths.append(os.path.join(lib_include, 'THH'))
        paths.append(_join_rocm_home('include'))
    elif device_type == "cuda":
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)

        # Support CUDA_INC_PATH env variable supported by CMake files
        if (cuda_inc_path := os.environ.get("CUDA_INC_PATH", None)) and \
                cuda_inc_path != '/usr/include':

            paths.append(cuda_inc_path)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    elif device_type == "xpu":
        paths.append(_join_sycl_home('include'))
        paths.append(_join_sycl_home('include', 'sycl'))
    return paths


def library_paths(device_type: str = "cpu", torch_include_dirs: bool = True, cross_target_platform: str | None = None) -> list[str]:
    """
    Get the library paths required to build a C++ or CUDA extension.

    Args:
        device_type: Defaults to "cpu".

    Returns:
        A list of library path strings.
    """

    paths = []

    if torch_include_dirs:
        # We need to link against libtorch.so
        paths.extend([TORCH_LIB_PATH])

    if device_type == "cuda" and IS_HIP_EXTENSION:
        lib_dir = 'lib'
        paths.append(_join_rocm_home(lib_dir))
        if HIP_HOME is not None:
            paths.append(os.path.join(HIP_HOME, 'lib'))
    elif device_type == "cuda":
        if cross_target_platform == "windows":
            lib_dir = os.path.join('lib', 'x64')
            if WINDOWS_CUDA_HOME is None:
                raise RuntimeError("Need to set WINDOWS_CUDA_HOME for windows cross-compilation")
            paths.append(os.path.join(WINDOWS_CUDA_HOME, lib_dir))
        else:
            if IS_WINDOWS:
                lib_dir = os.path.join('lib', 'x64')
            else:
                lib_dir = 'lib64'
                if (not os.path.exists(_join_cuda_home(lib_dir)) and
                        os.path.exists(_join_cuda_home('lib'))):
                    # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
                    # Note that it's also possible both don't exist (see
                    # _find_cuda_home) - in that case we stay with 'lib64'.
                    lib_dir = 'lib'

            paths.append(_join_cuda_home(lib_dir))
            if CUDNN_HOME is not None:
                paths.append(os.path.join(CUDNN_HOME, lib_dir))
    elif device_type == "xpu":
        if IS_WINDOWS:
            lib_dir = os.path.join('lib', 'x64')
        else:
            lib_dir = 'lib64'
            if (not os.path.exists(_join_sycl_home(lib_dir)) and
                    os.path.exists(_join_sycl_home('lib'))):
                lib_dir = 'lib'

        paths.append(_join_sycl_home(lib_dir))

    return paths


def load(name,
         sources: str | list[str],
         extra_cflags=None,
         extra_cuda_cflags=None,
         extra_sycl_cflags=None,
         extra_ldflags=None,
         extra_include_paths=None,
         build_directory=None,
         verbose=False,
         with_cuda: bool | None = None,
         with_sycl: bool | None = None,
         is_python_module=True,
         is_standalone=False,
         keep_intermediates=True):
    """
    Load a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    CUDA support with mixed compilation is provided. Simply pass CUDA source
    files (``.cu`` or ``.cuh``) along with other sources. Such files will be
    detected and compiled with nvcc rather than the C++ compiler. This includes
    passing the CUDA lib64 directory as a library directory, and linking
    ``cudart``. You can pass additional flags to nvcc via
    ``extra_cuda_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the CUDA install directory are used, which usually
    work fine. If not, setting the ``CUDA_HOME`` environment variable is the
    safest option.

    SYCL support with mixed compilation is provided. Simply pass SYCL source
    files (``.sycl``) along with other sources. Such files will be detected
    and compiled with SYCL compiler (such as Intel DPC++ Compiler) rather
    than the C++ compiler. You can pass additional flags to SYCL compiler
    via ``extra_sycl_cflags``, just like with ``extra_cflags`` for C++.
    SYCL compiler is expected to be found via system PATH environment
    variable.

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_cuda_cflags: optional list of compiler flags to forward to nvcc
            when building CUDA sources.
        extra_sycl_cflags: optional list of compiler flags to forward to SYCL
            compiler when building SYCL sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.cu`` or
            ``.cuh`` in ``sources``. Set it to `True`` to force CUDA headers
            and libraries to be included.
        with_sycl: Determines whether SYCL headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.sycl`` in
            ``sources``. Set it to `True`` to force SYCL headers and
            libraries to be included.
        is_python_module: If ``True`` (default), imports the produced shared
            library as a Python module. If ``False``, behavior depends on
            ``is_standalone``.
        is_standalone: If ``False`` (default) loads the constructed extension
            into the process as a plain dynamic library. If ``True``, build a
            standalone executable.

    Returns:
        If ``is_python_module`` is ``True``:
            Returns the loaded PyTorch extension as a Python module.

        If ``is_python_module`` is ``False`` and ``is_standalone`` is ``False``:
            Returns nothing. (The shared library is loaded into the process as
            a side effect.)

        If ``is_standalone`` is ``True``.
            Return the path to the executable. (On Windows, TORCH_LIB_PATH is
            added to the PATH environment variable as a side effect.)

    Example:
        >>> # xdoctest: +SKIP
        >>> from torch.utils.cpp_extension import load
        >>> module = load(
        ...     name='extension',
        ...     sources=['extension.cpp', 'extension_kernel.cu'],
        ...     extra_cflags=['-O2'],
        ...     verbose=True)
    """
    return _jit_compile(
        name,
        [sources] if isinstance(sources, str) else sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_sycl_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory or _get_build_directory(name, verbose),
        verbose,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone,
        keep_intermediates=keep_intermediates)

@deprecated("PyBind11 ABI handling is internal to PyBind11; this will be removed after PyTorch 2.9.0")
def _get_pybind11_abi_build_flags() -> list[str]:
    return []

def check_compiler_is_gcc(compiler) -> bool:
    if not IS_LINUX:
        return False

    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # Don't localize output
    try:
        version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    except (subprocess.CalledProcessError, OSError):
        try:
            version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
        except (subprocess.CalledProcessError, OSError):
            return False
    # Check for GCC by verifying both COLLECT_GCC and gcc version string are present
    # This works for c++, g++, gcc, and versioned variants like g++-13
    pattern = re.compile("^COLLECT_GCC=(.*)$", re.MULTILINE)
    has_collect_gcc = pattern.search(version_string) is not None
    if has_collect_gcc and 'gcc version' in version_string:
        return True
    return False

def _check_and_build_extension_h_precompiler_headers(
        extra_cflags,
        extra_include_paths,
        is_standalone=False) -> None:
    r'''
    Precompiled Headers(PCH) can pre-build the same headers and reduce build time for pytorch load_inline modules.
    GCC official manual: https://gcc.gnu.org/onlinedocs/gcc-4.0.4/gcc/Precompiled-Headers.html
    PCH only works when built pch file(header.h.gch) and build target have the same build parameters. So, We need
    add a signature file to record PCH file parameters. If the build parameters(signature) changed, it should rebuild
    PCH file.

    Note:
    1. Windows and MacOS have different PCH mechanism. We only support Linux currently.
    2. It only works on GCC/G++.
    '''
    if not IS_LINUX:
        return

    compiler = get_cxx_compiler()

    b_is_gcc = check_compiler_is_gcc(compiler)
    if b_is_gcc is False:
        return

    compiler = _shell_join(_wrap_compiler(compiler))

    head_file = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h')
    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    def listToString(s):
        # initialize an empty string
        string = ""
        if s is None:
            return string

        # traverse in the string
        for element in s:
            string += (element + ' ')
        # return string
        return string

    def format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags, torch_include_dirs, extra_cflags, extra_include_paths):
        return re.sub(
            r"[ \n]+",
            " ",
            f"""
                {compiler} -x c++-header {head_file} -o {head_file_pch} {torch_include_dirs} {extra_include_paths} {extra_cflags} {common_cflags}
            """,
        ).strip()

    def command_to_signature(cmd):
        signature = cmd.replace(' ', '_')
        return signature

    def check_pch_signature_in_file(file_path, signature):
        b_exist = os.path.isfile(file_path)
        if b_exist is False:
            return False

        with open(file_path) as file:
            # read all content of a file
            content = file.read()
            # check if string present in a file
            return signature == content

    def _create_if_not_exist(path_dir) -> None:
        if not os.path.exists(path_dir):
            try:
                Path(path_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise RuntimeError(f"Fail to create path {path_dir}") from exc

    def write_pch_signature_to_file(file_path, pch_sign) -> None:
        _create_if_not_exist(os.path.dirname(file_path))
        with open(file_path, "w") as f:
            f.write(pch_sign)
            f.close()

    def build_precompile_header(pch_cmd) -> None:
        try:
            subprocess.check_output(shlex.split(pch_cmd), stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Compile PreCompile Header fail, command: {pch_cmd}") from e

    extra_cflags_str = listToString(extra_cflags)
    extra_include_paths_str = " ".join(
        [f"-I{include}" for include in extra_include_paths] if extra_include_paths else []
    )

    lib_include = os.path.join(_TORCH_PATH, 'include')
    torch_include_dirs = [
        f"-I {lib_include}",
        # Python.h
        "-I {}".format(sysconfig.get_path("include")),
        # torch/all.h
        "-I {}".format(os.path.join(lib_include, 'torch', 'csrc', 'api', 'include')),
    ]

    torch_include_dirs_str = listToString(torch_include_dirs)

    common_cflags = []
    if not is_standalone:
        common_cflags += ['-DTORCH_API_INCLUDE_EXTENSION_H']

    common_cflags += ['-std=c++20', '-fPIC']
    common_cflags_str = listToString(common_cflags)

    pch_cmd = format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags_str, torch_include_dirs_str, extra_cflags_str, extra_include_paths_str)
    pch_sign = command_to_signature(pch_cmd)

    if os.path.isfile(head_file_pch) is not True:
        build_precompile_header(pch_cmd)
        write_pch_signature_to_file(head_file_signature, pch_sign)
    else:
        b_same_sign = check_pch_signature_in_file(head_file_signature, pch_sign)
        if b_same_sign is False:
            build_precompile_header(pch_cmd)
            write_pch_signature_to_file(head_file_signature, pch_sign)

def remove_extension_h_precompiler_headers() -> None:
    def _remove_if_file_exists(path_file) -> None:
        if os.path.exists(path_file):
            os.remove(path_file)

    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    _remove_if_file_exists(head_file_pch)
    _remove_if_file_exists(head_file_signature)

def load_inline(name,
                cpp_sources,
                cuda_sources=None,
                sycl_sources=None,
                functions=None,
                extra_cflags=None,
                extra_cuda_cflags=None,
                extra_sycl_cflags=None,
                extra_ldflags=None,
                extra_include_paths=None,
                build_directory=None,
                verbose=False,
                with_cuda=None,
                with_sycl=None,
                is_python_module=True,
                with_pytorch_error_handling=True,
                keep_intermediates=True,
                use_pch=False,
                no_implicit_headers=False):
    r'''
    Load a PyTorch C++ extension just-in-time (JIT) from string sources.

    This function behaves exactly like :func:`load`, but takes its sources as
    strings rather than filenames. These strings are stored to files in the
    build directory, after which the behavior of :func:`load_inline` is
    identical to :func:`load`.

    See `the
    tests <https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions_jit.py>`_
    for good examples of using this function.

    Sources may omit two required parts of a typical non-inline C++ extension:
    the necessary header includes, as well as the (pybind11) binding code. More
    precisely, strings passed to ``cpp_sources`` are first concatenated into a
    single ``.cpp`` file. This file is then prepended with ``#include
    <torch/extension.h>``

    Furthermore, if the ``functions`` argument is supplied, bindings will be
    automatically generated for each function specified. ``functions`` can
    either be a list of function names, or a dictionary mapping from function
    names to docstrings. If a list is given, the name of each function is used
    as its docstring.

    The sources in ``cuda_sources`` are concatenated into a separate ``.cu``
    file and  prepended with ``torch/types.h``, ``cuda.h`` and
    ``cuda_runtime.h`` includes. The ``.cpp`` and ``.cu`` files are compiled
    separately, but ultimately linked into a single library. Note that no
    bindings are generated for functions in ``cuda_sources`` per se. To bind
    to a CUDA kernel, you must create a C++ function that calls it, and either
    declare or define this C++ function in one of the ``cpp_sources`` (and
    include its name in ``functions``).

    The sources in ``sycl_sources`` are concatenated into a separate ``.sycl``
    file and  prepended with ``torch/types.h``, ``sycl/sycl.hpp`` includes.
    The ``.cpp`` and ``.sycl`` files are compiled separately, but ultimately
    linked into a single library. Note that no bindings are generated for
    functions in ``sycl_sources`` per se. To bind to a SYCL kernel, you must
    create a C++ function that calls it, and either declare or define this
    C++ function in one of the ``cpp_sources`` (and include its name
    in ``functions``).



    See :func:`load` for a description of arguments omitted below.

    Args:
        cpp_sources: A string, or list of strings, containing C++ source code.
        cuda_sources: A string, or list of strings, containing CUDA source code.
        sycl_sources: A string, or list of strings, containing SYCL source code.
        functions: A list of function names for which to generate function
            bindings. If a dictionary is given, it should map function names to
            docstrings (which are otherwise just the function names).
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``cuda_sources`` is
            provided. Set it to ``True`` to force CUDA headers
            and libraries to be included.
        with_sycl: Determines whether SYCL headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``sycl_sources`` is
            provided. Set it to ``True`` to force SYCL headers
            and libraries to be included.
        with_pytorch_error_handling: Determines whether pytorch error and
            warning macros are handled by pytorch instead of pybind. To do
            this, each function ``foo`` is called via an intermediary ``_safe_foo``
            function. This redirection might cause issues in obscure cases
            of cpp. This flag should be set to ``False`` when this redirect
            causes issues.
        no_implicit_headers: If ``True``, skips automatically adding headers, most notably
            ``#include <torch/extension.h>`` and ``#include <torch/types.h>`` lines.
            Use this option to improve cold start times when you
            already include the necessary headers in your source code. Default: ``False``.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import load_inline
        >>> source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        """
        >>> module = load_inline(name='inline_extension',
        ...                      cpp_sources=[source],
        ...                      functions=['sin_add'])

    .. note::
        Since load_inline will just-in-time compile the source code, please ensure
        that you have the right toolchains installed in the runtime. For example,
        when loading C++, make sure a C++ compiler is available. If you're loading
        a CUDA extension, you will need to additionally install the corresponding CUDA
        toolkit (nvcc and any other dependencies your code has). Compiling toolchains
        are not included when you install torch and must be additionally installed.

        During compiling, by default, the Ninja backend uses #CPUS + 2 workers to build
        the extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''
    build_directory = build_directory or _get_build_directory(name, verbose)

    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]
    cuda_sources = cuda_sources or []
    if isinstance(cuda_sources, str):
        cuda_sources = [cuda_sources]
    sycl_sources = sycl_sources or []
    if isinstance(sycl_sources, str):
        sycl_sources = [sycl_sources]

    if not no_implicit_headers:
        cpp_sources.insert(0, '#include <torch/extension.h>')

    if use_pch is True:
        # Using PreCompile Header('torch/extension.h') to reduce compile time.
        _check_and_build_extension_h_precompiler_headers(extra_cflags, extra_include_paths)
    else:
        remove_extension_h_precompiler_headers()

    # If `functions` is supplied, we create the pybind11 bindings for the user.
    # Here, `functions` is (or becomes, after some processing) a map from
    # function names to function docstrings.
    if functions is not None:
        module_def = []
        module_def.append('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {')
        if isinstance(functions, str):
            functions = [functions]
        if isinstance(functions, list):
            # Make the function docstring the same as the function name.
            functions = {f: f for f in functions}
        elif not isinstance(functions, dict):
            raise ValueError(f"Expected 'functions' to be a list or dict, but was {type(functions)}")
        for function_name, docstring in functions.items():
            if with_pytorch_error_handling:
                module_def.append(f'm.def("{function_name}", torch::wrap_pybind_function({function_name}), "{docstring}");')
            else:
                module_def.append(f'm.def("{function_name}", {function_name}, "{docstring}");')
        module_def.append('}')
        cpp_sources += module_def

    cpp_source_path = os.path.join(build_directory, 'main.cpp')
    _maybe_write(cpp_source_path, "\n".join(cpp_sources))

    sources = [cpp_source_path]

    if cuda_sources:
        if not no_implicit_headers:
            cuda_sources.insert(0, '#include <torch/types.h>')
            cuda_sources.insert(1, '#include <cuda.h>')
            cuda_sources.insert(2, '#include <cuda_runtime.h>')

        cuda_source_path = os.path.join(build_directory, 'cuda.cu')
        _maybe_write(cuda_source_path, "\n".join(cuda_sources))

        sources.append(cuda_source_path)

    if sycl_sources:
        if not no_implicit_headers:
            sycl_sources.insert(0, '#include <torch/types.h>')
            sycl_sources.insert(1, '#include <sycl/sycl.hpp>')

        sycl_source_path = os.path.join(build_directory, 'sycl.sycl')
        _maybe_write(sycl_source_path, "\n".join(sycl_sources))

        sources.append(sycl_source_path)

    return _jit_compile(
        name,
        sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_sycl_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory,
        verbose,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone=False,
        keep_intermediates=keep_intermediates)


def _jit_compile(name,
                 sources,
                 extra_cflags,
                 extra_cuda_cflags,
                 extra_sycl_cflags,
                 extra_ldflags,
                 extra_include_paths,
                 build_directory: str,
                 verbose: bool,
                 with_cuda: bool | None,
                 with_sycl: bool | None,
                 is_python_module,
                 is_standalone,
                 keep_intermediates=True) -> types.ModuleType | str:
    if is_python_module and is_standalone:
        raise ValueError("`is_python_module` and `is_standalone` are mutually exclusive.")

    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    with_cudnn = any('cudnn' in f for f in extra_ldflags or [])
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    old_version = JIT_EXTENSION_VERSIONER.get_version(name)
    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(
        name,
        sources,
        build_arguments=[extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths],
        build_directory=build_directory,
        with_cuda=with_cuda,
        with_sycl=with_sycl,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
    )
    if version > 0:
        if version != old_version and verbose:
            logger.info('The input conditions for extension module %s have changed.', name)
            logger.info('Bumping to version %s and re-building as %s_v%s...', version, name, version)
        name = f'{name}_v{version}'

    # Hold an OS-level advisory lock (via filelock) rather than a lock file
    # whose mere existence blocks: the kernel releases the lock if the holder
    # is killed, so a builder terminated by SIGKILL/OOM/Slurm no longer leaves
    # a stale lock that deadlocks every later run
    # (https://github.com/pytorch/pytorch/issues/189245). Whoever acquires the
    # lock rechecks the version and (re)builds, so a crashed build is retried
    # by the next process instead of leaving waiters to load a partial module.
    with FileLock(os.path.join(build_directory, 'lock')):
        if version != old_version:
            if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                from .hipify import hipify_python
                from .hipify.hipify_python import GeneratedFileCleaner
                clean_ctx_mgr = GeneratedFileCleaner(keep_intermediates=keep_intermediates)
            else:
                import contextlib
                hipify_python = None  # type: ignore[assignment]
                clean_ctx_mgr = contextlib.nullcontext()
            with clean_ctx_mgr as clean_ctx:
                if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                    if hipify_python is None:
                        raise AssertionError("expected hipify_python to be not None")
                    hipify_result = hipify_python.hipify(
                        project_directory=build_directory,
                        output_directory=build_directory,
                        header_include_dirs=(extra_include_paths if extra_include_paths is not None else []),
                        extra_files=[os.path.abspath(s) for s in sources],
                        ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')],  # no need to hipify ROCm or PyTorch headers
                        show_detailed=verbose,
                        show_progress=verbose,
                        is_pytorch_extension=True,
                        clean_ctx=clean_ctx
                    )

                    hipified_sources = set()
                    for source in sources:
                        s_abs = os.path.abspath(source)
                        if s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None:
                            hipified_s_abs = hipify_result[s_abs].hipified_path
                        else:
                            hipified_s_abs = s_abs
                        hipified_sources.add(hipified_s_abs)
                    sources = list(hipified_sources)

                _write_ninja_file_and_build_library(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags or [],
                    extra_cuda_cflags=extra_cuda_cflags or [],
                    extra_sycl_cflags=extra_sycl_cflags or [],
                    extra_ldflags=extra_ldflags or [],
                    extra_include_paths=extra_include_paths or [],
                    build_directory=build_directory,
                    verbose=verbose,
                    with_cuda=with_cuda,
                    with_sycl=with_sycl,
                    is_standalone=is_standalone)
        elif verbose:
            logger.debug('No modifications detected for re-loaded extension module %s, skipping build step...', name)

    if verbose:
        logger.info('Loading extension module %s...', name)

    if is_standalone:
        return _get_exec_path(name, build_directory)

    return _import_module_from_library(name, build_directory, is_python_module)

def _get_hipcc_path():
    if IS_WINDOWS:
        # mypy thinks ROCM_VERSION is None but it will never be None here
        hipcc_exe = 'hipcc.exe' if ROCM_VERSION >= (6, 4) else 'hipcc.bat'  # type: ignore[operator]
        return _join_rocm_home('bin', hipcc_exe)
    else:
        return _join_rocm_home('bin', 'hipcc')

def _write_ninja_file_and_compile_objects(
        sources: list[str],
        objects,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        sycl_cflags,
        sycl_post_cflags,
        sycl_dlink_post_cflags,
        build_directory: str,
        verbose: bool,
        with_cuda: bool | None,
        with_sycl: bool | None) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        logger.debug('Emitting ninja build file %s...', build_file_path)

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug('Creating directory %s...', build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        cuda_cflags=cuda_cflags,
        cuda_post_cflags=cuda_post_cflags,
        cuda_dlink_post_cflags=cuda_dlink_post_cflags,
        sycl_cflags=sycl_cflags,
        sycl_post_cflags=sycl_post_cflags,
        sycl_dlink_post_cflags=sycl_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_cuda=with_cuda,
        with_sycl=with_sycl)
    if verbose:
        logger.info('Compiling objects...')
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix='Error compiling objects for extension')


def _write_ninja_file_and_build_library(
        name,
        sources: list[str],
        extra_cflags,
        extra_cuda_cflags,
        extra_sycl_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory: str,
        verbose: bool,
        with_cuda: bool | None,
        with_sycl: bool | None,
        is_standalone: bool = False) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    extra_ldflags = _prepare_ldflags(
        extra_ldflags or [],
        with_cuda,
        with_sycl,
        verbose,
        is_standalone)
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        logger.debug('Emitting ninja build file %s...', build_file_path)

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug('Creating directory %s...', build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    # NOTE: Emitting a new ninja build file does not cause re-compilation if
    # the sources did not change, so it's ok to re-emit (and it's fast).
    _write_ninja_file_to_build_library(
        path=build_file_path,
        name=name,
        sources=sources,
        extra_cflags=extra_cflags or [],
        extra_cuda_cflags=extra_cuda_cflags or [],
        extra_sycl_cflags=extra_sycl_cflags or [],
        extra_ldflags=extra_ldflags or [],
        extra_include_paths=extra_include_paths or [],
        with_cuda=with_cuda,
        with_sycl=with_sycl,
        is_standalone=is_standalone)

    if verbose:
        logger.info('Building extension module %s...', name)
    _run_ninja_build(
        build_directory,
        verbose,
        error_prefix=f"Error building extension '{name}'")


def is_ninja_available() -> bool:
    """Return ``True`` if the `ninja <https://ninja-build.org/>`_ build system is available on the system, ``False`` otherwise."""
    try:
        subprocess.check_output(['ninja', '--version'])
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability() -> None:
    """Raise ``RuntimeError`` if `ninja <https://ninja-build.org/>`_ build system is not available on the system, does nothing otherwise."""
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions (pip install ninja to get it)")


def _prepare_ldflags(extra_ldflags, with_cuda, with_sycl, verbose, is_standalone):
    if IS_WINDOWS:
        python_lib_path = os.path.join(sys.base_exec_prefix, 'libs')

        extra_ldflags.append('c10.lib')
        if with_cuda:
            extra_ldflags.append('c10_hip.lib' if IS_HIP_EXTENSION else 'c10_cuda.lib')
        if with_sycl:
            extra_ldflags.append('c10_xpu.lib')
        extra_ldflags.append('torch_cpu.lib')
        if with_cuda:
            extra_ldflags.append('torch_hip.lib' if IS_HIP_EXTENSION else 'torch_cuda.lib')
            # /INCLUDE is used to ensure torch_cuda is linked against in a project that relies on it.
            # Related issue: https://github.com/pytorch/pytorch/issues/31611
            extra_ldflags.append('-INCLUDE:?warp_size@cuda@at@@YAHXZ')
        if with_sycl:
            extra_ldflags.append('torch_xpu.lib')
        extra_ldflags.append('torch.lib')
        extra_ldflags.append(f'/LIBPATH:{TORCH_LIB_PATH}')
        if not is_standalone:
            extra_ldflags.append('torch_python.lib')
            extra_ldflags.append(f'/LIBPATH:{python_lib_path}')

    else:
        extra_ldflags.append(f'-L{TORCH_LIB_PATH}')
        extra_ldflags.append('-lc10')
        if with_cuda:
            extra_ldflags.append('-lc10_hip' if IS_HIP_EXTENSION else '-lc10_cuda')
        if with_sycl:
            extra_ldflags.append('-lc10_xpu')
        extra_ldflags.append('-ltorch_cpu')
        if with_cuda:
            extra_ldflags.append('-ltorch_hip' if IS_HIP_EXTENSION else '-ltorch_cuda')
        if with_sycl:
            extra_ldflags.append('-ltorch_xpu')
        extra_ldflags.append('-ltorch')
        if not is_standalone:
            extra_ldflags.append('-ltorch_python')

        if is_standalone:
            extra_ldflags.append(f"-Wl,-rpath,{TORCH_LIB_PATH}")

    if with_cuda:
        if verbose:
            logger.info('Detected CUDA files, patching ldflags')
        if IS_WINDOWS and not IS_HIP_EXTENSION:
            extra_ldflags.append(f'/LIBPATH:{_join_cuda_home("lib", "x64")}')
            extra_ldflags.append('cudart.lib')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'/LIBPATH:{os.path.join(CUDNN_HOME, "lib", "x64")}')
        elif not IS_HIP_EXTENSION:
            extra_lib_dir = "lib64"
            if (not os.path.exists(_join_cuda_home(extra_lib_dir)) and
                    os.path.exists(_join_cuda_home("lib"))):
                # 64-bit CUDA may be installed in "lib"
                # Note that it's also possible both don't exist (see _find_cuda_home) - in that case we stay with "lib64"
                extra_lib_dir = "lib"
            extra_ldflags.append(f'-L{_join_cuda_home(extra_lib_dir)}')
            extra_ldflags.append('-lcudart')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'-L{os.path.join(CUDNN_HOME, "lib64")}')
        elif IS_HIP_EXTENSION:
            if IS_WINDOWS:
                extra_ldflags.append(f'/LIBPATH:{_join_rocm_home("lib")}')
                extra_ldflags.append('amdhip64.lib')
            else:
                extra_ldflags.append(f'-L{_join_rocm_home("lib")}')
                extra_ldflags.append('-lamdhip64')
    if with_sycl:
        if IS_WINDOWS:
            extra_ldflags.append(f'/LIBPATH:{_join_sycl_home("lib")}')
            extra_ldflags.append('sycl.lib')
        else:
            extra_ldflags.append(f'-L{_join_sycl_home("lib")}')
            extra_ldflags.append('-lsycl')
    return extra_ldflags


def _get_cuda_arch_flags(cflags: list[str] | None = None) -> list[str]:
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
            if 'TORCH_EXTENSION_NAME' in flag:
                continue
            if 'arch' in flag:
                return []

    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    named_arches = collections.OrderedDict([
        ('Kepler+Tesla', '3.7'),
        ('Kepler', '3.5+PTX'),
        ('Maxwell+Tegra', '5.3'),
        ('Maxwell', '5.0;5.2+PTX'),
        ('Pascal', '6.0;6.1+PTX'),
        ('Volta+Tegra', '7.2'),
        ('Volta', '7.0+PTX'),
        ('Turing', '7.5+PTX'),
        ('Ampere+Tegra', '8.7'),
        ('Ampere', '8.0;8.6+PTX'),
        ('Ada', '8.9+PTX'),
        ('Hopper', '9.0+PTX'),
        ('Blackwell+Tegra', '11.0'),
        ('Blackwell', '10.0;10.3;12.0;12.1+PTX'),
        ('Rubin', '10.7+PTX'),
    ])

    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0', '9.0a',
                        '10.0', '10.0a', '11.0', '11.0a', '10.3', '10.3a', '10.7', '10.7a',
                        '12.0', '12.0a', '12.1', '12.1a']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    _arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)

    # If not given or set as native, determine what's best for the GPU / CUDA version that can be found
    if not _arch_list or _arch_list == "native":
        arch_list = []
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            supported_sm = [int("".join(re.findall(r"\d+", arch.split('_')[1])))
                            for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
            max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
            # Capability of the device may be higher than what's supported by the user's
            # NVCC, causing compilation error. User's NVCC is expected to match the one
            # used to build pytorch, so we use the maximum supported capability of pytorch
            # to clamp the capability.
            capability = min(max_supported_sm, capability)
            arch = f'{capability[0]}.{capability[1]}'
            if arch not in arch_list:
                arch_list.append(arch)
        arch_list = sorted(arch_list)
        arch_list[-1] += '+PTX'

        if not _arch_list:
            # Only log on rank 0 in distributed settings to avoid spam
            if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                arch_list_str = ';'.join(arch_list)
                logger.debug(
                    "TORCH_CUDA_ARCH_LIST is not set, using TORCH_CUDA_ARCH_LIST='%s' "
                    "for visible GPU architectures. Set os.environ['TORCH_CUDA_ARCH_LIST'] to override.",
                    arch_list_str)
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        _arch_list = _arch_list.replace(' ', ';')
        # Expand named arches
        for named_arch, archival in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archival)

        arch_list = _arch_list.split(';')

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
        else:
            # Handle both single and double-digit architecture versions
            version = arch.split('+')[0]  # Remove "+PTX" if present
            major, minor = version.split('.')
            num = f"{major}{minor}"
            flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if arch.endswith('+PTX'):
                flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')

    return sorted(set(flags))


def _get_rocm_arch_flags(cflags: list[str] | None = None) -> list[str]:
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`). If user also specified -fgpu-rdc or -fno-gpu-rdc, we
    # assume they know what they're doing. Otherwise, we force -fno-gpu-rdc default.
    has_gpu_rdc_flag = False
    if cflags is not None:
        has_custom_flags = False
        for flag in cflags:
            if 'amdgpu-target' in flag or 'offload-arch' in flag:
                has_custom_flags = True
            elif 'gpu-rdc' in flag:
                has_gpu_rdc_flag = True
        if has_custom_flags:
            return [] if has_gpu_rdc_flag else ['-fno-gpu-rdc']
    # Use same defaults as used for building PyTorch
    # Allow env var to override, just like during initial cmake build.
    _archs = os.environ.get('PYTORCH_ROCM_ARCH', None)
    if not _archs:
        arch_set = set()
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            if hasattr(device_properties, "gcnArchName"):
                device_arch = (device_properties.gcnArchName).split(":", 1)[0]
                arch_set.add(device_arch)

        archs = ";".join(arch_set)

        logger.warning(
            "The environment variable `PYTORCH_ROCM_ARCH` is not set, all archs for visible cards are included for compilation (%s).\n"
            "If this is not desired, please set the environment variable `PYTORCH_ROCM_ARCH` to specific architectures.", archs)
    else:
        archs = _archs.replace(' ', ';')

    archs = archs.split(';')
    flags = [f'--offload-arch={arch}' for arch in archs]
    flags += [] if has_gpu_rdc_flag else ['-fno-gpu-rdc']
    return flags

def _get_build_directory(name: str, verbose: bool) -> str:
    """
    Get the build directory for an extension.

    Args:
        name: The name of the extension
        verbose: Whether to print verbose information

    Returns:
        The path to the build directory
    """
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        # Determine GPU accelerator prefix based on available accelerators. Fallback to CPU.
        # Priority: ROCm/HIP > CUDA > CPU
        # Note: torch.backends.cuda.is_built() returns True for both CUDA and ROCm,
        # so we need to check torch.version.hip to distinguish them
        if torch.version.hip is not None:
            accelerator_str = f'rocm{torch.version.hip.replace(".", "")}'
        elif torch.version.cuda is not None:
            accelerator_str = f'cu{torch.version.cuda.replace(".", "")}'
        else:
            accelerator_str = 'cpu'
        python_version = f'py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}'
        build_folder = f'{python_version}_{accelerator_str}'

        root_extensions_directory = os.path.join(
            root_extensions_directory, build_folder)

    if verbose:
        logger.info('Using %s as PyTorch extensions root...', root_extensions_directory)

    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug('Creating extension directory %s...', build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    return build_directory


def _get_num_workers(verbose: bool) -> int | None:
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            logger.debug('Using envvar MAX_JOBS (%s) as the number of workers...', max_jobs)
        return int(max_jobs)
    if verbose:
        logger.info(
            'Allowing ninja to set a default number of workers... '
            '(overridable by setting the environment variable MAX_JOBS=N)'
        )
    return None


def _get_vc_env(vc_arch: str) -> dict[str, str]:
    try:
        from setuptools import distutils  # type: ignore[attr-defined]

        return distutils._msvccompiler._get_vc_env(vc_arch)
    except AttributeError:
        try:
            from setuptools._distutils import _msvccompiler
            return _msvccompiler._get_vc_env(vc_arch)  # type: ignore[attr-defined]
        except AttributeError:
            from setuptools._distutils.compilers.C import msvc
            return msvc._get_vc_env(vc_arch)  # type: ignore[attr-defined]

def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    # Try to activate the vc env for the users
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        from setuptools import distutils  # type: ignore[attr-defined]

        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]
        vc_env = {k.upper(): v for k, v in _get_vc_env(plat_spec).items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        # Warning: don't pass stdout=None to subprocess.run to get output.
        # subprocess.run assumes that sys.__stdout__ has not been modified and
        # attempts to write to it by default.  However, when we call _run_ninja_build
        # from ahead-of-time cpp extensions, the following happens:
        # 1) If the stdout encoding is not utf-8, setuptools detaches __stdout__.
        #    https://github.com/pypa/setuptools/blob/7e97def47723303fafabe48b22168bbc11bb4821/setuptools/dist.py#L1110
        #    (it probably shouldn't do this)
        # 2) subprocess.run (on POSIX, with no stdout override) relies on
        #    __stdout__ not being detached:
        #    https://github.com/python/cpython/blob/c352e6c7446c894b13643f538db312092b351789/Lib/subprocess.py#L1214
        # To work around this, we pass in the fileno directly and hope that
        # it is valid.
        stdout_fileno = 1
        subprocess.run(
            command,
            shell=IS_WINDOWS and IS_HIP_EXTENSION,
            stdout=stdout_fileno if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=build_directory,
            check=True,
            env=env)
    except subprocess.CalledProcessError as e:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        message = error_prefix
        # `error` is a CalledProcessError (which has an `output`) attribute, but
        # mypy thinks it's Optional[BaseException] and doesn't narrow
        if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e


def _get_exec_path(module_name, path):
    if IS_WINDOWS and TORCH_LIB_PATH not in os.getenv('PATH', '').split(';'):
        torch_lib_in_path = any(
            os.path.exists(p) and os.path.samefile(p, TORCH_LIB_PATH)
            for p in os.getenv('PATH', '').split(';')
        )
        if not torch_lib_in_path:
            os.environ['PATH'] = f"{TORCH_LIB_PATH};{os.getenv('PATH', '')}"
    return os.path.join(path, f'{module_name}{EXEC_EXT}')


def _import_module_from_library(module_name, path, is_python_module):
    filepath = os.path.join(path, f"{module_name}{LIB_EXT}")
    if is_python_module:
        # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None:
            raise AssertionError(f"Failed to create spec for module {module_name} at {filepath}")
        module = importlib.util.module_from_spec(spec)
        if not isinstance(spec.loader, importlib.abc.Loader):
            raise AssertionError("spec.loader is not a valid importlib Loader")
        spec.loader.exec_module(module)
        return module
    else:
        torch.ops.load_library(filepath)
        return filepath


def _write_ninja_file_to_build_library(path,
                                       name,
                                       sources,
                                       extra_cflags,
                                       extra_cuda_cflags,
                                       extra_sycl_cflags,
                                       extra_ldflags,
                                       extra_include_paths,
                                       with_cuda,
                                       with_sycl,
                                       is_standalone) -> None:
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_cuda_cflags = [flag.strip() for flag in extra_cuda_cflags]
    extra_sycl_cflags = [flag.strip() for flag in extra_sycl_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    user_includes = [os.path.abspath(file) for file in extra_include_paths]

    # include_paths() gives us the location of torch/extension.h
    # TODO generalize with_cuda as specific device type.
    if with_cuda:
        system_includes = include_paths("cuda")
    elif with_sycl:
        system_includes = include_paths("xpu")
    else:
        system_includes = include_paths("cpu")
    # sysconfig.get_path('include') gives us the location of Python.h
    # Explicitly specify 'posix_prefix' scheme on non-Windows platforms to workaround error on some MacOS
    # installations where default `get_path` points to non-existing `/Library/Python/M.m/include` folder
    python_include_path = sysconfig.get_path('include', scheme='nt' if IS_WINDOWS else 'posix_prefix')
    if python_include_path is not None:
        system_includes.append(python_include_path)

    common_cflags = []
    if not is_standalone:
        common_cflags.append(f'-DTORCH_EXTENSION_NAME={name}')
        common_cflags.append('-DTORCH_API_INCLUDE_EXTENSION_H')

    # Windows does not understand `-isystem` and quotes flags later.
    if IS_WINDOWS:
        common_cflags += [f'-I{include}' for include in user_includes + system_includes]
    else:
        common_cflags += [f'-I{shlex.quote(include)}' for include in user_includes]
        common_cflags += [f'-isystem {shlex.quote(include)}' for include in system_includes]

    if IS_WINDOWS:
        COMMON_HIP_FLAGS.extend(['-fms-runtime-lib=dll'])
        cflags = common_cflags + ['/std:c++20'] + extra_cflags
        cflags += COMMON_MSVC_FLAGS + (COMMON_HIP_FLAGS if IS_HIP_EXTENSION else [])
        cflags = _nt_quote_args(cflags)
    else:
        cflags = common_cflags + ['-fPIC', '-std=c++20'] + extra_cflags

    if with_cuda and IS_HIP_EXTENSION:
        cuda_flags = ['-DWITH_HIP'] + common_cflags + extra_cflags + COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS
        cuda_flags = cuda_flags + ['-std=c++20']
        cuda_flags += _get_rocm_arch_flags(cuda_flags)
        cuda_flags += extra_cuda_cflags
        if IS_WINDOWS:
            cuda_flags = _nt_quote_args(cuda_flags)
    elif with_cuda:
        cuda_flags = common_cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags(extra_cuda_cflags)
        if IS_WINDOWS:
            for flag in COMMON_MSVC_FLAGS:
                cuda_flags = ['-Xcompiler', flag] + cuda_flags
            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                cuda_flags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cuda_flags
            cuda_flags = cuda_flags + ['-std=c++20']
            cuda_flags = _nt_quote_args(cuda_flags)
            cuda_flags += _nt_quote_args(extra_cuda_cflags)
        else:
            cuda_flags += ['--compiler-options', "'-fPIC'"]
            cuda_flags += extra_cuda_cflags
            if not any(flag.startswith('-std=') for flag in cuda_flags):
                cuda_flags.append('-std=c++20')
            cc_env = os.getenv("CC")
            if cc_env is not None:
                cuda_flags = ['-ccbin', cc_env] + cuda_flags
    else:
        cuda_flags = None

    if with_sycl:
        sycl_cflags = cflags + _COMMON_SYCL_FLAGS
        sycl_cflags += extra_sycl_cflags
        _append_sycl_targets_if_missing(sycl_cflags)
        _append_sycl_std_if_no_std_present(sycl_cflags)
        host_cflags = cflags
        # escaping quoted arguments to pass them thru SYCL compiler
        icpx_version = _get_icpx_version()
        if int(icpx_version) < 20250200:
            host_cflags = [item.replace('\\"', '\\\\"') for item in host_cflags]

        sycl_cflags += _wrap_sycl_host_flags(host_cflags)
        sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
        sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_cflags)
    else:
        sycl_cflags = None
        sycl_dlink_post_cflags = None

    def object_file_path(source_file: str) -> str:
        # '/path/to/file.cpp' -> 'file'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_cuda_file(source_file) and with_cuda:
            # Use a different object filename in case a C++ and CUDA file have
            # the same filename but different extension (.cpp vs. .cu).
            target = f'{file_name}.cuda.o'
        elif _is_sycl_file(source_file) and with_sycl:
            target = f'{file_name}.sycl.o'
        else:
            target = f'{file_name}.o'
        return target

    objects = [object_file_path(src) for src in sources]
    ldflags = ([] if is_standalone else [SHARED_FLAG]) + extra_ldflags

    # The darwin linker needs explicit consent to ignore unresolved symbols.
    if IS_MACOS:
        ldflags.append('-undefined dynamic_lookup')
    elif IS_WINDOWS:
        ldflags = _nt_quote_args(ldflags)

    ext = EXEC_EXT if is_standalone else LIB_EXT
    library_target = f'{name}{ext}'

    _write_ninja_file(
        path=path,
        cflags=cflags,
        post_cflags=None,
        cuda_cflags=cuda_flags,
        cuda_post_cflags=None,
        cuda_dlink_post_cflags=None,
        sycl_cflags=sycl_cflags,
        sycl_post_cflags=[],
        sycl_dlink_post_cflags=sycl_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=ldflags,
        library_target=library_target,
        with_cuda=with_cuda,
        with_sycl=with_sycl)


def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sycl_cflags,
                      sycl_post_cflags,
                      sycl_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda,
                      with_sycl) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_post_cflags`: list of flags to append to the $nvcc invocation. Can be None.
    `cuda_dlink_post_cflags`: list of flags to append to the $nvcc device code link invocation. Can be None.
    `sycl_cflags`: list of flags to pass to SYCL compiler. Can be None.
    `sycl_post_cflags`: list of flags to append to the SYCL compiler invocation. Can be None.
    `sycl_dlink_post_cflags`: list of flags to append to the SYCL compiler device code link invocation. Can be None.
e.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    sycl_cflags = sanitize_flags(sycl_cflags)
    sycl_post_cflags = sanitize_flags(sycl_post_cflags)
    sycl_dlink_post_cflags = sanitize_flags(sycl_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    if len(sources) != len(objects):
        raise AssertionError("sources and objects lists must be the same length")
    if len(sources) == 0:
        raise AssertionError("At least one source is required to build a library")

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {_shell_join(_wrap_compiler(compiler))}')
    if with_cuda or cuda_dlink_post_cflags:
        if "PYTORCH_NVCC" in os.environ:
            nvcc = os.getenv("PYTORCH_NVCC")    # user can set nvcc compiler with ccache using the environment variable here
        else:
            if IS_HIP_EXTENSION:
                nvcc = _get_hipcc_path()
            else:
                nvcc = _join_cuda_home('bin', 'nvcc')
            nvcc = _shell_join(_wrap_compiler(nvcc))
        config.append(f'nvcc = {nvcc}')
    if with_sycl or sycl_dlink_post_cflags:
        sycl = 'icx' if IS_WINDOWS else 'icpx'
        config.append(f'sycl = {sycl}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    if with_sycl:
        flags.append(f'sycl_cflags = {" ".join(sycl_cflags)}')
        flags.append(f'sycl_post_cflags = {" ".join(sycl_post_cflags)}')
    flags.append(f'sycl_dlink_post_cflags = {" ".join(sycl_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compiler_name = "$cxx" if IS_HIP_EXTENSION else "cl"
        compile_rule.append(
            f'  command = {compiler_name} '
            '/showIncludes $cflags -c $in /Fo$out $post_cflags'  # codespell:ignore
        )
        if not IS_HIP_EXTENSION:
            compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # -MD is not supported by ROCm
        # Nvcc flag `-MD` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use -MD to make this work on Windows too.
            nvcc_gendeps = '-MD -MF $out.d'
        cuda_compile_rule.append(
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    if with_sycl:
        sycl_compile_rule = ['rule sycl_compile']
        # SYCL compiler does not recognize .sycl extension automatically,
        # so we pass '-x c++' explicitly notifying compiler of file format
        sycl_compile_rule.append(
            '  command = $sycl $sycl_cflags -c -x c++ $in -o $out $sycl_post_cflags')


    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects, strict=True):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        is_sycl_source = _is_sycl_file(source_file) and with_sycl
        if is_cuda_source:
            rule = 'cuda_compile'
        elif is_sycl_source:
            rule = 'sycl_compile'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        cuda_devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        cuda_devlink_rule = ['rule cuda_devlink']
        cuda_devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        cuda_devlink = [f'build {cuda_devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [cuda_devlink_out]
    else:
        cuda_devlink_rule, cuda_devlink = [], []

    if sycl_dlink_post_cflags:
        sycl_devlink_out = os.path.join(os.path.dirname(objects[0]), "sycl_dlink.o")
        if IS_WINDOWS:
            sycl_devlink_objects = [obj.replace(":", "$:") for obj in objects]
            objects += [sycl_devlink_out]
            sycl_devlink_out = sycl_devlink_out.replace(":", "$:")
        else:
            sycl_devlink_objects = list(objects)
            objects += [sycl_devlink_out]
        sycl_devlink_rule = ["rule sycl_devlink"]
        sycl_devlink_rule.append(
            "  command = $sycl $in -o $out $sycl_dlink_post_cflags"
        )
        sycl_devlink = [
            f"build {sycl_devlink_out}: sycl_devlink {' '.join(sycl_devlink_objects)}"
        ]
    else:
        sycl_devlink_rule, sycl_devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
    if with_sycl:
        blocks.append(sycl_compile_rule)  # type: ignore[possibly-undefined]
    blocks += [cuda_devlink_rule, sycl_devlink_rule, link_rule, build, cuda_devlink, sycl_devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)

def _join_cuda_home(*paths) -> str:
    """
    Join paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    """
    if CUDA_HOME is None:
        raise OSError('CUDA_HOME environment variable is not set. '
                      'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')
    return os.path.splitext(path)[1] in valid_ext

def _is_sycl_file(path: str) -> bool:
    valid_ext = ['.sycl']
    return os.path.splitext(path)[1] in valid_ext
<|endoftext|>"""Compile-safe backward gradient logging for multiple tensors.

``debug_grad_log`` logs gradient norms during backward for one or more tensors.
It is a leaf function with a ``register_multi_grad_hook`` that fires exactly
once when all requires_grad tensor inputs have their gradients computed.

Example::

    import torch
    from torch.utils.debug_log import debug_grad_log

    x = torch.randn(4, requires_grad=True)
    y = torch.randn(4, requires_grad=True)
    z = x * 2 + y * 3

    debug_grad_log(x, y)

    z.sum().backward()
    # Logs: [rank 0][bwd] t0_grad_norm=... t1_grad_norm=...
"""

import logging

import torch
from torch._dynamo.decorators import leaf_function


__all__ = ["debug_grad_log"]

log = logging.getLogger(__name__)


def _get_rank() -> int:
    if not torch.distributed.is_available():
        return 0
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


@leaf_function
def debug_grad_log(*tensors):
    """Log gradient norms of multiple tensors during backward.

    This is a no-op in the forward pass. During backward, the hook fires
    exactly once when all requires_grad tensor inputs have their gradients
    computed, and logs ``[rank R][bwd] t0_grad_norm=... t1_grad_norm=...``.

    Args:
        *tensors: One or more tensors to monitor.

    Returns:
        None. Call without assignment: ``debug_grad_log(x, y)``.
    """
    return None


@debug_grad_log.register_fake  # pyrefly: ignore[missing-attribute]
def _debug_grad_log_fake(*tensors):
    return None


@debug_grad_log.register_multi_grad_hook  # pyrefly: ignore[missing-attribute]
def _debug_grad_log_hook(*grads):
    norms = " ".join(
        f"t{i}_grad_norm={g.norm().item():.4f}" for i, g in enumerate(grads)
    )
    log.info("[rank %d][bwd] %s", _get_rank(), norms)
<|endoftext|># mypy: allow-untyped-defs
import functools

import torch
from torch._C import _len_torch_function_stack
from torch.overrides import _pop_mode, _push_mode, TorchFunctionMode
from torch.utils._contextlib import context_decorator


CURRENT_DEVICE: torch.device | None = None


@functools.lru_cache(1)
def _device_constructors():
    return {
        # standard ones
        torch.empty,
        torch.empty_permuted,
        torch.empty_strided,
        torch.empty_quantized,
        torch.ones,
        torch.arange,
        torch.bartlett_window,
        torch.blackman_window,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.full,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.nested.nested_tensor,
        # This function doesn't actually take a device argument
        # torch.normal,
        torch.rand,
        torch.randn,
        torch.randint,
        torch.randperm,
        torch.range,
        torch.sparse_coo_tensor,
        torch.sparse_compressed_tensor,
        torch.sparse_csr_tensor,
        torch.sparse_csc_tensor,
        torch.sparse_bsr_tensor,
        torch.sparse_bsc_tensor,
        torch.tril_indices,
        torch.triu_indices,
        torch.zeros,
        torch.asarray,
        # weird ones
        torch.tensor,
        torch.as_tensor,
        torch.scalar_tensor,
        # *_like may contain device kwarg, but the user implicitly
        # expects a specific device even when kwarg unused.
        # torch.zeros_like,
        # torch.randint_like,
        # torch.randn_like,
        # torch.ones_like,
        # torch.full_like,
        # torch.empty_like,
    }


# NB: This is directly called from C++ in torch/csrc/Device.cpp
class DeviceContext(TorchFunctionMode):
    def __init__(self, device) -> None:
        self.device = torch.device(device)
        self.prev_mode: DeviceContext | None = None

    def __enter__(self):
        global CURRENT_DEVICE
        self.old_device = CURRENT_DEVICE
        CURRENT_DEVICE = self.device
        # We need to put the device at the bottom of the stack
        # If we set default device within a function mode context
        # exiting that context mode will pop the device function mode off
        # of the stack incorrectly
        cur_stack = [_pop_mode() for _ in range(_len_torch_function_stack())]

        _push_mode(self)

        for mode in reversed(cur_stack):
            if isinstance(mode, DeviceContext):
                self.prev_mode = mode
            else:
                _push_mode(mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global CURRENT_DEVICE
        CURRENT_DEVICE = self.old_device
        cur_stack = []
        # Invariant: there should only be one DeviceContext on the stack at any time
        # (At the bottom), pop all modes until we hit the bottom, assert it's a DeviceContext
        # or else someone else has popped it!
        for _ in range(_len_torch_function_stack() - 1):
            mode = _pop_mode()
            if isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Found nested DeviceContext on the mode stack where none expected"
                )
            cur_stack.append(mode)

        if _len_torch_function_stack() > 0:
            mode = _pop_mode()
            if not isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Expected a DeviceContext at the bottom of the mode stack"
                )
        if self.prev_mode is not None:
            _push_mode(self.prev_mode)

        for mode in reversed(cur_stack):
            _push_mode(mode)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _device_constructors() and kwargs.get("device") is None:
            kwargs["device"] = self.device
        return func(*args, **kwargs)


# NB: This is directly called from C++ in torch/csrc/Device.cpp
def device_decorator(device, func):
    return context_decorator(lambda: device, func)


def set_device(device):
    """
    Set the default device inside of the wrapped function by decorating it with this function.

    If you would like to use this as a context manager, use device as a
    context manager directly, e.g., ``with torch.device(device)``.
    """
    return lambda func: device_decorator(torch.device(device), func)
<|endoftext|>import functools
import hashlib
import os
from typing import Any


_FAILED_TO_MAP_SEGMENT_FROM_SHARED_OBJECT = "failed to map segment from shared object"


def _triton_cache_dir_for_error_message() -> str | None:
    if triton_cache_dir := os.environ.get("TRITON_CACHE_DIR"):
        return triton_cache_dir

    try:
        from triton.runtime.cache import knobs

        return knobs.cache.dir
    except (AttributeError, ImportError):
        return None


def _raise_triton_cache_load_error(exc: BaseException) -> None:
    cache_dir = _triton_cache_dir_for_error_message()
    if "TRITON_CACHE_DIR" in os.environ:
        cache_dir_msg = f" (TRITON_CACHE_DIR={cache_dir})"
    else:
        cache_dir_msg = f" ({cache_dir})" if cache_dir else ""
    raise ImportError(
        f"{exc}. This usually means Triton's cache directory{cache_dir_msg} is on "
        "a filesystem mounted with noexec, so the dynamic loader cannot map "
        "generated shared objects. Set TRITON_CACHE_DIR to a directory on an "
        "executable filesystem."
    ) from exc


@functools.cache
def has_triton_package() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


@functools.cache
def get_triton_version(fallback: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    try:
        import triton

        major, minor = tuple(int(v) for v in triton.__version__.split(".")[:2])
        return (major, minor)
    except ImportError:
        return fallback


@functools.cache
def _device_supports_tensor_descriptor() -> bool:
    import torch

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (9, 0)
        and not torch.version.hip
    ) or has_triton_cpu_backend()


@functools.cache
def has_triton_cpu_backend() -> bool:
    if has_triton_package():
        import triton

        return "cpu" in triton.backends.backends

    return False


@functools.cache
def has_triton_experimental_host_tma() -> bool:
    if has_triton_package():
        if _device_supports_tensor_descriptor():
            try:
                from triton.tools.experimental_descriptor import (  # noqa: F401
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                try:
                    from triton.tools.experimental_descriptor import enable_in_pytorch

                    return enable_in_pytorch()
                except ImportError:
                    return True
            except ImportError:
                pass

    return False


@functools.cache
def has_triton_tensor_descriptor_host_tma() -> bool:
    if has_triton_package():
        if _device_supports_tensor_descriptor():
            try:
                from triton.tools.tensor_descriptor import (  # noqa: F401
                    TensorDescriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_triton_tma() -> bool:
    return has_triton_tensor_descriptor_host_tma() or has_triton_experimental_host_tma()


@functools.cache
def has_triton_tma_device() -> bool:
    if has_triton_package():
        import torch

        if (
            (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability() >= (9, 0)
                and not torch.version.hip
            )
            or torch.xpu.is_available()
            or has_triton_cpu_backend()
        ):
            # old API
            try:
                from triton.language.extra.cuda import (  # noqa: F401
                    experimental_device_tensormap_create1d,
                    experimental_device_tensormap_create2d,
                )

                return True
            except ImportError:
                pass

            # new API
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_datacenter_blackwell_tma_device() -> bool:
    import torch

    if (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (10, 0)
        and torch.cuda.get_device_capability() < (11, 0)
        and not torch.version.hip
    ):
        return has_triton_tma_device() and has_triton_tensor_descriptor_host_tma()

    return False


@functools.lru_cache(None)
def has_triton_stable_tma_api() -> bool:
    if has_triton_package():
        import torch

        if (
            (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability() >= (9, 0)
                and not torch.version.hip
            )
            or torch.xpu.is_available()
            or has_triton_cpu_backend()
        ):
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass
    return False


@functools.cache
def has_triton_reduction_ordering() -> bool:
    """Whether the available Triton exposes inner-tree reduction ordering."""
    if has_triton_package():
        try:
            from triton.language import ReductionOrdering

            return hasattr(ReductionOrdering, "INNER_TREE")
        except ImportError:
            pass
    return False


@functools.cache
def has_triton() -> bool:
    if not has_triton_package():
        return False

    from torch._inductor.config import triton_disable_device_detection

    if triton_disable_device_detection:
        return False

    from torch._dynamo.device_interface import get_registered_device_interfaces
    from torch._dynamo.exc import TritonUnavailableError

    # A device supports Triton if it is available, reports Triton capability, and
    # its Triton backend is actually built. Capability is gated first so that
    # raise_if_triton_unavailable() only surfaces missing-backend errors (and
    # not, e.g., CUDA's GPUTooOldForTriton for sub-capable devices). We catch the
    # specific TritonUnavailableError rather than RuntimeError so unexpected
    # errors are not silently swallowed.
    for name, device_interface in get_registered_device_interfaces():
        if ":" in name:
            continue
        if not (
            device_interface.is_available() and device_interface.is_triton_capable()
        ):
            continue
        try:
            device_interface.raise_if_triton_unavailable()
        except TritonUnavailableError:
            continue
        return True
    return False


@functools.cache
def triton_backend() -> Any:
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    try:
        target = driver.active.get_current_target()
        return make_backend(target)
    except (ImportError, OSError) as e:
        if _FAILED_TO_MAP_SEGMENT_FROM_SHARED_OBJECT in str(e):
            _raise_triton_cache_load_error(e)
        raise


def _extern_libs_key(backend: Any) -> str:
    """Return a cache key fragment for extern libs (e.g. libdevice.10.bc).

    These files affect codegen but are not covered by triton_key() (Python
    sources only) or backend.hash() (ptxas version and arch only).
    """
    opts = backend.parse_options({})
    extern_libs = getattr(opts, "extern_libs", None)
    if not extern_libs:
        return ""
    parts = []
    for name, path in sorted(extern_libs):
        if os.path.isfile(path):
            with open(path, "rb") as f:
                parts.append(f"{name}-{hashlib.sha256(f.read()).hexdigest()}")
    return "-".join(parts)


@functools.cache
def triton_hash_with_backend() -> str:
    from torch._inductor.runtime.triton_compat import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}"

    # Hash is upper case so that it can't contain any Python keywords.
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch._C import _get_privateuse1_backend_name, _rename_privateuse1_backend
from torch.overrides import handle_torch_function, has_torch_function_unary


__all__ = [
    "rename_privateuse1_backend",
    "generate_methods_for_privateuse1_backend",
]

# TODO: Should use `torch._C._get_privateuse1_backend_name()` to get
# renamed-backend name for `privateuse1`, but the func will cause an
# error with torch.jit.script, so we use the global variable named
# `_privateuse1_backend_name`.
_privateuse1_backend_name = "privateuseone"


def _rename_profiler_activity(backend_name: str) -> None:
    """Mirror the privateuse1 rename in ProfilerActivity so users can write e.g.
    ProfilerActivity.<backend_name> instead of ProfilerActivity.PrivateUse1."""
    from torch._C._profiler import ProfilerActivity

    alias = backend_name.upper()
    setattr(ProfilerActivity, alias, ProfilerActivity.PrivateUse1)

    pu1 = ProfilerActivity.PrivateUse1
    original_repr = ProfilerActivity.__repr__
    original_str = ProfilerActivity.__str__

    def custom_repr(self):
        if self == pu1:
            return f"<ProfilerActivity.{alias}: {pu1.value}>"
        return original_repr(self)

    def custom_str(self):
        if self == pu1:
            return f"ProfilerActivity.{alias}"
        return original_str(self)

    ProfilerActivity.__repr__ = custom_repr
    ProfilerActivity.__str__ = custom_str
    ProfilerActivity.name = property(  # type: ignore[assignment]
        lambda self: alias if self == pu1 else original_str(self).split(".")[-1]
    )


def _rename_device_type(backend_name: str) -> None:
    from torch._C._autograd import DeviceType

    alias = backend_name.upper()
    setattr(DeviceType, alias, DeviceType.PrivateUse1)

    pu1 = DeviceType.PrivateUse1
    original_repr = DeviceType.__repr__
    original_str = DeviceType.__str__

    def custom_repr(self):
        if self == pu1:
            return f"<DeviceType.{alias}: {pu1.value}>"
        return original_repr(self)

    def custom_str(self):
        if self == pu1:
            return f"DeviceType.{alias}"
        return original_str(self)

    DeviceType.__repr__ = custom_repr
    DeviceType.__str__ = custom_str
    DeviceType.name = property(  # type: ignore[assignment]
        lambda self: alias if self == pu1 else original_str(self).split(".")[-1]
    )


def rename_privateuse1_backend(backend_name: str) -> None:
    r"""
    Rename the privateuse1 backend device to make it more convenient to use as a device name within PyTorch APIs.

    The steps are:

    (1) (In C++) implement kernels for various torch operations, and register them
        to the PrivateUse1 dispatch key.
    (2) (In python) call torch.utils.rename_privateuse1_backend("foo")

    You can now use "foo" as an ordinary device string in python.

    Note: this API can only be called once per process. Attempting to change
    the external backend after it's already been set will result in an error.

    Note(AMP): If you want to support AMP on your device, you can register a custom backend module.
    The backend must register a custom backend module with ``torch._register_device_module("foo", BackendModule)``.
    BackendModule needs to have the following API's:

    (1) ``get_amp_supported_dtype() -> List[torch.dtype]``
        get the supported dtypes on your "foo" device in AMP, maybe the "foo" device supports one more dtype.

    Note(random): If you want to support to set seed for your device, BackendModule needs to have the following API's:

    (1) ``_is_in_bad_fork() -> bool``
        Return ``True`` if now it is in bad_fork, else return ``False``.

    (2) ``manual_seed_all(seed int) -> None``
        Sets the seed for generating random numbers for your devices.

    (3) ``device_count() -> int``
        Returns the number of "foo"s available.

    (4) ``get_rng_state(device: Union[int, str, torch.device] = 'foo') -> Tensor``
        Returns a list of ByteTensor representing the random number states of all devices.

    (5) ``set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = 'foo') -> None``
        Sets the random number generator state of the specified "foo" device.

    And there are some common funcs:

    (1) ``is_available() -> bool``
        Returns a bool indicating if "foo" is currently available.

    (2) ``current_device() -> int``
        Returns the index of a currently selected device.

    For more details, see https://pytorch.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend
    For an existing example, see https://github.com/bdhirsh/pytorch_open_registration_example

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        # This will work, assuming that you've implemented the right C++ kernels
        # to implement torch.ones.
        >>> a = torch.ones(2, device="foo")

    """
    _rename_privateuse1_backend(backend_name)
    global _privateuse1_backend_name
    _privateuse1_backend_name = backend_name
    _rename_profiler_activity(backend_name)
    _rename_device_type(backend_name)


def _check_register_once(module, attr) -> None:
    if hasattr(module, attr):
        raise RuntimeError(
            f"The custom device module of {module} has already been registered with {attr}"
        )


def _normalization_device(
    custom_backend_name: str, device: int | str | torch.device | None = None
) -> int:
    def _get_current_device_index():
        _get_device_index = "current_device"
        if hasattr(torch, custom_backend_name) and hasattr(
            getattr(torch, custom_backend_name), _get_device_index
        ):
            return getattr(getattr(torch, custom_backend_name), _get_device_index)()
        else:
            # The default device index is 0.
            return 0

    if device is None:
        return _get_current_device_index()
    # if isinstance(device, str), this means that the parameter passed in is in the string format "foo:0"
    # convert str object to torch.device object, and then process it uniformly
    elif isinstance(device, str):
        device = torch.device(device)

    # variable device can only be torch.device type or int type
    if isinstance(device, torch.device):
        if device.type != custom_backend_name:
            raise RuntimeError(f"Invalid device, must be {custom_backend_name} device")
        elif device.index is None:
            device_idx = _get_current_device_index()
        else:
            device_idx = device.index
    # if isinstance(device, int), we can take the index number directly
    else:
        device_idx = device
    return device_idx


def _generate_tensor_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.Tensor) -> bool:
        if has_torch_function_unary(self):
            # TODO mypy doesn't support @property, see: https://github.com/python/mypy/issues/6185
            return handle_torch_function(wrap_tensor_backend.__get__, (self,), self)  # type: ignore[attr-defined]
        return self.device.type == custom_backend_name

    _check_register_once(torch.Tensor, f"is_{custom_backend_name}")
    wrap_tensor_backend.fget.__name__ = f"is_{custom_backend_name}"  # type: ignore[attr-defined]
    setattr(torch.Tensor, f"is_{custom_backend_name}", wrap_tensor_backend)

    def wrap_tensor_to(
        self: torch.Tensor,
        device: int | torch.device | None = None,
        non_blocking=False,
        **kwargs,
    ) -> torch.Tensor:
        r"""Perform Tensor device conversion. Call the to operator implementation.

        .. note::
            If the ``self`` Tensor already
            has the correct :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired :class:`torch.device`.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
            **kwargs (dict): For compatibility, may contain the key ``memory_format`` argument.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                wrap_tensor_to,
                (self,),
                self,
                device=device,
                non_blocking=False,
                **kwargs,
            )
        device_idx = _normalization_device(custom_backend_name, device)
        return self.to(
            device=torch.device(f"{custom_backend_name}:{device_idx}"),
            non_blocking=non_blocking,
            **kwargs,
        )

    _check_register_once(torch.Tensor, custom_backend_name)
    wrap_tensor_to.__name__ = custom_backend_name
    setattr(torch.Tensor, custom_backend_name, wrap_tensor_to)


def _generate_module_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # Generate Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(torch.Tensor, custom_backend_name):
        raise RuntimeError(
            f"Can not automatically generate {custom_backend_name}() method for torch.nn.Module."
            f"Because torch.Tensor doesn't have the method {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True."
        )

    def wrap_module_to(
        # pyrefly: ignore [invalid-type-var]
        self: torch.nn.modules.module.T,
        device: int | torch.device | None = None,
    ) -> torch.nn.modules.module.T:  # pyrefly: ignore [invalid-type-var]
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        # pyrefly: ignore [missing-attribute]
        return self._apply(lambda t: getattr(t, custom_backend_name)(device))

    _check_register_once(torch.nn.Module, custom_backend_name)
    setattr(torch.nn.Module, custom_backend_name, wrap_module_to)


def _generate_packed_sequence_methods_for_privateuse1_backend(
    custom_backend_name: str,
) -> None:
    # Generate PackedSequence Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(torch.Tensor, f"is_{custom_backend_name}") or not hasattr(
        torch.Tensor, custom_backend_name
    ):
        raise RuntimeError(
            f"Can not automatically generate is_{custom_backend_name}() or "
            f"{custom_backend_name}() method for torch.nn.utils.rnn.PackedSequence."
            f"Because torch.Tensor doesn't have the method is_{custom_backend_name}()"
            f"or {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True."
        )

    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.nn.utils.rnn.PackedSequence) -> bool:
        return self.data.device.type == custom_backend_name

    _check_register_once(torch.nn.utils.rnn.PackedSequence, f"is_{custom_backend_name}")
    setattr(
        torch.nn.utils.rnn.PackedSequence,
        f"is_{custom_backend_name}",
        wrap_tensor_backend,
    )

    def wrap_module_to(
        self: torch.nn.utils.rnn.PackedSequence, *args, **kwargs
    ) -> torch.nn.utils.rnn.PackedSequence:
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args,
            **kwargs,
        )
        if ex.device.type == custom_backend_name:
            return self.to(*args, **kwargs)
        kwargs.update({"device": custom_backend_name})

        return self.to(*args, **kwargs)

    _check_register_once(torch.nn.utils.rnn.PackedSequence, custom_backend_name)
    setattr(torch.nn.utils.rnn.PackedSequence, custom_backend_name, wrap_module_to)


def _generate_storage_methods_for_privateuse1_backend(
    custom_backend_name: str, unsupported_dtype: list[torch.dtype] | None = None
) -> None:
    # Attribute is registered in the _StorageBase class
    # and UntypedStorage obtains through inheritance.
    @property  # type: ignore[misc]
    def wrap_storage_backend(self: torch.storage._StorageBase) -> bool:
        r"""Return the internal :class:`torch.UntypedStorage`."""
        return self.device.type == custom_backend_name

    _check_register_once(torch.storage._StorageBase, f"is_{custom_backend_name}")
    setattr(
        torch.storage._StorageBase, f"is_{custom_backend_name}", wrap_storage_backend
    )

    def wrap_storage_to(self, device=None, non_blocking=False):
        r"""Return a copy of this object in custom device memory.

        If this object is already in device memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination device id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        """
        # There should be a judgment related to storage device and a judgment related to storage type,
        # but it depends on the extended function, so this part is temporarily omitted in the automatic generation.
        device_idx = _normalization_device(custom_backend_name, device)

        if getattr(self, f"is_{custom_backend_name}"):
            # storage has already on expected device.
            if self.get_device() == device_idx:
                return self
        # For sparse storage, custom need to extend the implementation by themselves.
        if self.is_sparse:
            raise RuntimeError(
                f"Can not support a sparse storage move to {custom_backend_name} backend"
            )
        # create untyped_storage and copy data
        untyped_storage = torch.UntypedStorage(
            self.size(), device=torch.device(f"{custom_backend_name}:{device_idx}")
        )
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage

    _check_register_once(torch.storage._StorageBase, custom_backend_name)
    setattr(torch.storage._StorageBase, custom_backend_name, wrap_storage_to)

    # Register the corresponding attribute for the TypedStorage class.
    # When the TypedStorage class is removed, the registration is also removed.

    @property  # type: ignore[misc]
    def wrap_typed_storage_backend(self: torch.storage.TypedStorage) -> bool:
        torch.storage._warn_typed_storage_removal()
        return self._untyped_storage.device.type == custom_backend_name

    _check_register_once(torch.TypedStorage, f"is_{custom_backend_name}")
    setattr(
        torch.storage.TypedStorage,
        f"is_{custom_backend_name}",
        wrap_typed_storage_backend,
    )

    def wrap_typed_storage_to(
        self: torch.storage.TypedStorage, device=None, non_blocking=False, **kwargs
    ) -> torch.storage.TypedStorage:
        torch.storage._warn_typed_storage_removal()
        if unsupported_dtype and self.dtype in unsupported_dtype:
            raise RuntimeError(
                f"Cannot create {custom_backend_name} storage "
                f"as {self.dtype} dtype is not supported by this backend"
            )
        custom_backend_storage: torch.UntypedStorage = getattr(
            self._untyped_storage, custom_backend_name
        )(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(custom_backend_storage)

    _check_register_once(torch.TypedStorage, custom_backend_name)
    setattr(torch.TypedStorage, custom_backend_name, wrap_typed_storage_to)


def generate_methods_for_privateuse1_backend(
    for_tensor: bool = True,
    for_module: bool = True,
    for_packed_sequence: bool = True,
    for_storage: bool = False,
    unsupported_dtype: list[torch.dtype] | None = None,
) -> None:
    r"""
    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

    In the default scenario, storage-related methods will not be generated automatically.

    When you implement kernels for various torch operations, and register them to the PrivateUse1 dispatch key.
    And call the function torch.rename_privateuse1_backend("foo") to rename your backend name.
    At this point, you can easily register specific methods and attributes by calling this function.
    Just like torch.Tensor.foo(), torch.Tensor.is_foo, torch.Storage.foo(), torch.Storage.is_foo.

    Note: We recommend you use generic functions (check devices are equal or to(device=)).
    We provide these methods for convenience only and they will be "monkey patched" onto the objects
    and so will not be properly typed. For Storage methods generate, if you need to support sparse data storage,
    you need to extend the implementation yourself.

    Args:
        for_tensor (bool): whether register related methods for torch.Tensor class.
        for_module (bool): whether register related methods for torch.nn.Module class.
        for_storage (bool): whether register related methods for torch.Storage class.
        unsupported_dtype (List[torch.dtype]): takes effect only when the storage method needs to be generated,
            indicating that the storage does not support the torch.dtype type.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        >>> torch.utils.generate_methods_for_privateuse1_backend()
        # Then automatically generate backend-related attributes and methods.
        >>> a = torch.tensor(2).foo()
        >>> a.is_foo
        >>> hasattr(torch.nn.Module, 'foo')
    """
    custom_backend_name = _get_privateuse1_backend_name()

    if for_tensor:
        _generate_tensor_methods_for_privateuse1_backend(custom_backend_name)

    if for_module:
        _generate_module_methods_for_privateuse1_backend(custom_backend_name)

    if for_storage:
        _generate_storage_methods_for_privateuse1_backend(
            custom_backend_name, unsupported_dtype
        )

    if for_packed_sequence:
        _generate_packed_sequence_methods_for_privateuse1_backend(custom_backend_name)


def _get_custom_mod_func(func_name: str):
    r"""
    Return the func named `func_name` defined in custom device module. If not defined,
    return `None`. And the func is registered with `torch.utils.rename_privateuse1_backend('foo')`
    and `torch._register_device_module('foo', BackendModule)`.
    If the custom device module or the func is not defined, it will give warning or error message.
    Args:
        func_name (str): return the callable func named func_name defined in custom device module.
    Example::
        class DummyfooModule:
            @staticmethod
            def is_available():
                return True
            @staticmethod
            def func_name(*args, **kwargs):
                ....
        torch.utils.rename_privateuse1_backend("foo")
        torch._register_device_module("foo", DummyfooModule)
        foo_is_available_func = torch.utils.backend_registration._get_custom_mod_func("is_available")
        if foo_is_available_func:
            foo_is_available = foo_is_available_func()
        func_ = torch.utils.backend_registration._get_custom_mod_func("func_name")
        if func_:
            result = func_(*args, **kwargs)
    Attention: This function is not meant to be used directly by users, which is why
    it is marked as private. It is a convenience function for backend implementers to
    more easily call the hooks into their backend extensions.
    """
    if not isinstance(func_name, str):
        raise AssertionError(f"func_name must be `str`, but got `{type(func_name)}`.")
    backend_name = _get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, backend_name, None)
    function = getattr(custom_device_mod, func_name, None)
    if custom_device_mod is None or function is None:
        message = f"Try to call torch.{backend_name}.{func_name}. The backend must register a custom backend "
        message += f"module with `torch._register_device_module('{backend_name}', BackendModule)`. And "
        message += f"BackendModule needs to have the following API's:\n `{func_name}(*args, **kwargs)`. \n"
        raise RuntimeError(message)
    return function


class _DummyBackendModule:
    def is_initialized(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def _is_in_bad_fork(self) -> bool:
        return False

    def manual_seed_all(self, seed: int) -> None:
        pass

    def device_count(self) -> int:
        return 1


class _DummyPrivateUse1Hook(torch._C._acc.PrivateUse1Hooks):
    def is_available(self) -> bool:
        return True

    def has_primary_context(self, dev_id) -> bool:
        return True

    def is_built(self) -> bool:
        return True


class _DummyDeviceGuard(torch._C._acc.DeviceGuard):
    def type_(self):
        return torch._C._autograd.DeviceType.PrivateUse1


def _setup_privateuseone_for_python_backend(
    rename=None, backend_module=None, hook=None, device_guard=None
) -> None:
    """This function will prepare the PrivateUse1 dispatch key to be used as a python backend.

    WARNING: this API is experimental and might change without notice.

    Formally, this registers things that Pytorch expects a registered backend
    in C++ to have: including device guards, hooks, and backend modules and what not.

    after this call, one can use `torch.library` to write Ops for this dispatch key
    and expect it to behave like a backend registered in C++.

    See the unit test at test/test_privateuseone_python_backend.py for more details.

    Args:
        rename: str | None, if passed in, we will rename privateuseone backend to
           the name given.
        backend_module: object | None, if passed in None, we will use DummyBackendModule
        hook: object | None, if passed in None, we will use DummyPrivateUse1Hook
        device_guard: object | None, if passed in None, we will use DummyDeviceGuard
    """
    # NOTE: the ordering of which these functions are called is important.
    if rename is not None:
        torch.utils.rename_privateuse1_backend(rename)
    else:
        rename = "privateuseone"
    torch.utils.generate_methods_for_privateuse1_backend()
    if backend_module is None:
        backend_module = _DummyBackendModule()
    if hook is None:
        hook = _DummyPrivateUse1Hook()
    if device_guard is None:
        device_guard = _DummyDeviceGuard()
    torch._register_device_module(rename, backend_module)  # type: ignore[bad-argument-type]
    torch._C._acc.register_python_privateuseone_hook(hook)
    torch._C._acc.register_python_privateuseone_device_guard(device_guard)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import functools
import warnings
from collections import deque
from dataclasses import dataclass
from typing import cast, overload, Protocol, TYPE_CHECKING
from typing_extensions import TypeIs

import torch
import torchgen
import torchgen.model
from torch._C import (
    _get_dispatch_stack_at,
    _len_torch_dispatch_stack,
    _pop_torch_dispatch_stack,
    _push_on_torch_dispatch_stack,
    DispatchKey,
)
from torch._C._dynamo.guards import set_is_in_mode_without_ignore_compile_internals


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch._custom_class_base import CustomClassBase


# TODO: Limitations and things about enable_torch_dispatch_mode we should fix before exposing it:
# - We need a better user-facing api for _DisableTorchDispatch that
#   is able to selectively disable __torch_dispatch__ of a particular class.
# - It doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - Better name (see https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694)

_is_in_torch_dispatch_mode = False
_is_in_non_infra_torch_dispatch_mode = False
# If inside any mode that has ignore_compile_internals() = False
_is_in_any_mode_without_ignore_compile_internals = False


def is_in_torch_dispatch_mode(include_infra_modes: bool = True) -> bool:
    return (
        _is_in_torch_dispatch_mode
        if include_infra_modes
        else _is_in_non_infra_torch_dispatch_mode
    )


def is_in_any_mode_without_ignore_compile_internals() -> bool:
    return _is_in_any_mode_without_ignore_compile_internals


def any_torch_dispatch_mode_on_stack() -> bool:
    stack_len = torch._C._len_torch_dispatch_stack()

    for idx in range(stack_len):
        mode = _get_dispatch_stack_at(idx)

        # Apply filters first
        if mode.is_infra_mode():
            continue

        if mode.ignore_compile_internals():
            continue

        return True
    return False


class TorchDispatchMode:
    """
    A ``TorchDispatchMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overridable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``self`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """

    # - When False, custom torch dispatch mode will error out explicitly when a hop
    # is called under the mode.
    # - When True, custom torch dispatch mode's __torch_dispatch__ will be triggered.
    # Mode authors can implement how the mode interacts with higher order operators.
    supports_higher_order_operators = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._should_skip_dynamo():
            if "__torch_dispatch__" in cls.__dict__:
                raw = cls.__dict__["__torch_dispatch__"]
                if not isinstance(raw, classmethod):
                    cls.__torch_dispatch__ = torch._disable_dynamo(raw, recursive=True)

    def __init__(self, _dispatch_key=None):
        if _dispatch_key is not None:
            if not isinstance(_dispatch_key, torch._C.DispatchKey):
                raise AssertionError("_dispatch_key must be a torch._C.DispatchKey")
            self.__dict__["_dispatch_key"] = _dispatch_key

        self.old_dispatch_mode_flags: deque[bool] = deque()
        self.old_non_infra_dispatch_mode_flags: deque[bool] = deque()
        self.old_without_ignore_compile_internals_dispatch_mode_flags: deque[bool] = (
            deque()
        )

    def _lazy_init_old_dispatch_mode_flags(self):
        if not hasattr(self, "old_dispatch_mode_flags"):
            self.old_dispatch_mode_flags: deque[bool] = deque()  # type: ignore[no-redef]

        if not hasattr(self, "old_non_infra_dispatch_mode_flags"):
            self.old_non_infra_dispatch_mode_flags: deque[bool] = deque()  # type: ignore[no-redef]

        if not hasattr(
            self, "old_without_ignore_compile_internals_dispatch_mode_flags"
        ):
            self.old_without_ignore_compile_internals_dispatch_mode_flags: deque[  # type: ignore[no-redef]
                bool
            ] = deque()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    def __enter__(self):
        global _is_in_torch_dispatch_mode
        global _is_in_non_infra_torch_dispatch_mode
        global _is_in_any_mode_without_ignore_compile_internals

        # Previously, there wasn't any state in this class' constructor
        # super calls were added to existing modes, but for any new modes
        # this will replicate the previous behavior of not strictly needing
        # to call super().__init__()
        self._lazy_init_old_dispatch_mode_flags()
        self.old_dispatch_mode_flags.append(_is_in_torch_dispatch_mode)
        _is_in_torch_dispatch_mode = True
        self.old_non_infra_dispatch_mode_flags.append(
            _is_in_non_infra_torch_dispatch_mode
        )
        _is_in_non_infra_torch_dispatch_mode = (
            _is_in_non_infra_torch_dispatch_mode or not self.is_infra_mode()
        )
        self.old_without_ignore_compile_internals_dispatch_mode_flags.append(
            _is_in_any_mode_without_ignore_compile_internals
        )
        _is_in_any_mode_without_ignore_compile_internals = (
            _is_in_any_mode_without_ignore_compile_internals
            or not self.ignore_compile_internals()
        )
        set_is_in_mode_without_ignore_compile_internals(
            _is_in_any_mode_without_ignore_compile_internals
        )
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mb_dk_or_mode_key = self.__dict__.get("_dispatch_key", None)
        if mb_dk_or_mode_key is None:
            # Today, mode keys are not used at all in the per-dispatch-key-mode logic (for pre-dispatch)
            # We should probably revisit this.
            mb_dk_or_mode_key = self.__dict__.get("_mode_key", None)
        global _is_in_torch_dispatch_mode
        _is_in_torch_dispatch_mode = self.old_dispatch_mode_flags.pop()
        global _is_in_non_infra_torch_dispatch_mode
        _is_in_non_infra_torch_dispatch_mode = (
            self.old_non_infra_dispatch_mode_flags.pop()
        )
        global _is_in_any_mode_without_ignore_compile_internals
        _is_in_any_mode_without_ignore_compile_internals = (
            self.old_without_ignore_compile_internals_dispatch_mode_flags.pop()
        )
        set_is_in_mode_without_ignore_compile_internals(
            _is_in_any_mode_without_ignore_compile_internals
        )
        _pop_mode(mb_dk_or_mode_key)

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn(
            "`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`",
            stacklevel=2,
        )
        instance = cls(*args, **kwargs)
        return instance

    @classmethod
    def is_infra_mode(cls) -> bool:
        return False

    @classmethod
    def _should_skip_dynamo(cls) -> bool:
        """Skip Dynamo when the flag is set to True

        This is temporary measure to rollout a feature
        that skips PT2 compilation inside __torch_dispatch__
        frames.

        If this flag is off, we would expect following:

        class YoloMode(TorchDispatchMode):
            @classmethod
            def _should_skip_dynamo(cls):
                return False
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            out = torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

        # instead of recursively disabling, we are compiling into __torch_dispatch__
        assert len(backend.graphs) == 1
        """
        return True

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        """Ignore operators that are compiled via torch.compile.

        If ``True``, then this TorchDispatchMode ignores operators that
        are optimized by :func:`torch.compile`. Mechanically, this involves
        turning off the TorchDispatchMode throughout the whole compilation process,
        and turning it back on for the runtime of the compiled artifact(s).
        For example,

        @torch.compile
        def f(x):
            return x.sin().cos()

        with LoggingMode():
            f(x)

        The above example will not log anything if
        ``LoggingMode.ignore_compile_internals()`` is True.
        torch.compile will fuse sin() and cos() into a single operation
        and this TorchDispatchMode will not be passed sin and cos.

        If ``False`` (default), :func:`torch.compile` will respect
        the eager semantics of passing this TorchDispatchMode all
        operators that would have run during eager execution.
        The way this will usually happen is that :func:`torch.compile`
        will just fallback to eager-mode PyTorch.
        """
        if cls.is_infra_mode():
            return True
        return False


def _get_current_dispatch_mode() -> TorchDispatchMode | None:
    """
    Return the top user mode on the stack (the next one that would be
    executed) if there are any.
    """
    stack_len = _len_torch_dispatch_stack()
    if stack_len > 0:
        return _get_dispatch_stack_at(stack_len - 1)
    return None


def _detect_infra_mode(key):
    if key not in (
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
        torch._C._TorchDispatchModeKey.PROXY,
    ):
        raise AssertionError(
            f"key must be either FUNCTIONAL ({torch._C._TorchDispatchModeKey.FUNCTIONAL}) \
                or PROXY ({torch._C._TorchDispatchModeKey.PROXY}) _TorchDispatchModeKey, \
                    got {key}"
        )
    from torch._ops import _get_dispatch_mode_pre_dispatch

    pre_dispatch_mode = _get_dispatch_mode_pre_dispatch(key)
    post_dispatch_mode = torch._C._get_dispatch_mode(key)

    if pre_dispatch_mode is not None and post_dispatch_mode is not None:
        raise AssertionError(
            "At most one of pre_dispatch_mode and post_dispatch_mode may be active"
        )

    if pre_dispatch_mode is None:
        return post_dispatch_mode

    return pre_dispatch_mode


def _unset_infra_mode(key):
    from torch._ops import _get_dispatch_mode_pre_dispatch, unset_mode_pre_dispatch

    pre_dispatch_mode = _get_dispatch_mode_pre_dispatch(key)
    post_dispatch_mode = torch._C._get_dispatch_mode(key)
    if pre_dispatch_mode and post_dispatch_mode:
        raise AssertionError(
            "Can't have active infra mode on both pre and post dispatch mode stack"
        )

    if pre_dispatch_mode:
        mode = unset_mode_pre_dispatch(key)
        return mode
    if post_dispatch_mode:
        return torch._C._unset_dispatch_mode(key)


def _disable_infra_mode(key):
    if key not in (
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
        torch._C._TorchDispatchModeKey.PROXY,
    ):
        raise AssertionError(
            "key must be either FUNCTIONAL or PROXY _TorchDispatchModeKey"
        )
    mode_unset = _unset_infra_mode(key)
    try:
        yield mode_unset
    finally:
        if mode_unset is not None:
            _push_mode(mode_unset)


def _get_current_dispatch_mode_stack() -> list[TorchDispatchMode]:
    """
    Returns the current stack of dispatch modes, with the most recent
    (i.e., the one that will be processed first) at the end of the
    list (standard stack convention).
    """
    stack_len = _len_torch_dispatch_stack()
    return [_get_dispatch_stack_at(i) for i in range(stack_len)]


def _push_mode(mode: TorchDispatchMode) -> None:
    k = mode._dispatch_key if hasattr(mode, "_dispatch_key") else None
    if k is not None and k != torch._C.DispatchKey.PreDispatch:
        raise AssertionError(
            "mode._dispatch_key must be None or DispatchKey.PreDispatch"
        )
    if k is None:
        _push_on_torch_dispatch_stack(mode)
        return

    from torch._ops import _set_mode_pre_dispatch, get_cached_ops

    # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
    # Clear the cache of every op that has been used so far, for this particular key.
    ks = torch._C._functionality_to_backend_keys(k)
    for op in get_cached_ops():
        for key in ks:
            op._uncache_dispatch(key)
    _set_mode_pre_dispatch(mode)


def _pop_mode(k: DispatchKey | torch._C._TorchDispatchModeKey | None = None):
    if k == torch._C.DispatchKey.PreDispatch:  # type: ignore[attr-defined]
        from torch._ops import _pop_mode_from_pre_dispatch

        return _pop_mode_from_pre_dispatch()

    if k is None or isinstance(k, torch._C._TorchDispatchModeKey):
        return _pop_torch_dispatch_stack(k)


@contextlib.contextmanager
def _pop_mode_temporarily(k: DispatchKey | None = None):
    old = _pop_mode(k)
    try:
        yield old
    finally:
        _push_mode(old)


@contextlib.contextmanager
def _disable_current_modes():
    from torch._ops import (
        _len_torch_dispatch_stack_pre_dispatch,
        _pop_mode_from_pre_dispatch,
    )
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    from torch._subclasses.schema_check_mode import SchemaCheckMode
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

    mode_len_pre_dispatch = _len_torch_dispatch_stack_pre_dispatch()
    old_pre_dispatch_modes = [
        _pop_mode_from_pre_dispatch() for _ in range(mode_len_pre_dispatch)
    ]

    has_proxy_mode_in_pre_dispatch = False
    has_functional_mode_in_pre_dispatch = False
    has_schema_check_mode_in_pre_dispatch = False

    for i in old_pre_dispatch_modes:
        if isinstance(i, ProxyTorchDispatchMode):
            has_proxy_mode_in_pre_dispatch = True
        if isinstance(i, FunctionalTensorMode):
            has_functional_mode_in_pre_dispatch = True
        if isinstance(i, SchemaCheckMode):
            has_schema_check_mode_in_pre_dispatch = True

    mode_len = _len_torch_dispatch_stack()
    old_modes = [_pop_mode() for _ in range(mode_len)]

    for old in old_modes:
        if (
            isinstance(old, FunctionalTensorMode)
            and has_functional_mode_in_pre_dispatch
        ):
            raise AssertionError(
                "Can't have FunctionalMode available both in PreDispatch and Python Key"
            )
        if isinstance(old, ProxyTorchDispatchMode) and has_proxy_mode_in_pre_dispatch:
            raise AssertionError(
                "Can't have ProxyTorchDispatchMode available both in PreDispatch and Python Key"
            )
        if isinstance(old, SchemaCheckMode) and has_schema_check_mode_in_pre_dispatch:
            raise AssertionError(
                "Can't have SchemaCheckMode available both in PreDispatch and Python Key"
            )

    # Manually disable proxy and fake modes, if any are active
    try:
        yield old_pre_dispatch_modes + old_modes
    finally:
        for mode in reversed(old_modes):
            _push_mode(mode)
        for mode in reversed(old_pre_dispatch_modes):
            _push_mode(mode)


class BaseTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


# Python typing cannot express Intersection[torch.Tensor, Protocol], so this
# protocol repeats the Tensor members that call sites commonly use after
# is_traceable_wrapper_subclass() narrows a value to this protocol.
class TraceableWrapperSubclass(Protocol):
    """
    Canonical protocol for wrapper tensor subclasses that PT2 can trace through.

    ``__tensor_flatten__`` must return stable attribute names for the inner
    values that participate in tracing together with any metadata needed to
    rebuild the outer subclass.

    ``__tensor_unflatten__`` must reconstruct an equivalent wrapper subclass
    instance from the flattened inner values, metadata, and requested outer
    size/stride. Callers may pass tensor attrs and registered reference-type
    opaques in ``inner_tensors``. The returned tensor is expected to preserve
    the requested outer size/stride. If ``attrs, metadata =
    x.__tensor_flatten__()``, then ``type(x).__tensor_unflatten__`` must round-
    trip from ``{name: getattr(x, name) for name in attrs}``, ``metadata``,
    ``x.size()``, and ``x.stride()`` to an equivalent instance of ``x``.

    ``__tensor_unflatten__`` may be implemented as either a ``@staticmethod``
    or a ``@classmethod``; the runtime check below intentionally uses duck
    typing to support both forms, even though static type checkers may only
    recognize the ``@staticmethod`` form as conforming to this protocol.
    """

    def __tensor_flatten__(self) -> tuple[Sequence[str], object]: ...

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Mapping[str, torch.Tensor | CustomClassBase],
        metadata: object,
        outer_size: Sequence[int | torch.SymInt],
        outer_stride: Sequence[int | torch.SymInt],
    ) -> torch.Tensor: ...

    shape: torch._C.Size

    @overload
    def stride(self, dim: None = None) -> tuple[int, ...]: ...

    @overload
    def stride(self, dim: int) -> int: ...

    @overload
    def size(self, dim: None = None) -> tuple[int, ...]: ...

    @overload
    def size(self, dim: int) -> int: ...

    def storage_offset(self) -> int: ...

    def dim(self) -> int: ...

    @overload
    def to(
        self,
        dtype: torch.types._dtype,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> torch.Tensor: ...

    @overload
    def to(
        self,
        device: torch._prims_common.DeviceLikeType | None = None,
        dtype: torch.types._dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> torch.Tensor: ...

    @overload
    def to(
        self,
        other: torch.Tensor,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> torch.Tensor: ...


TensorWithFlatten = TraceableWrapperSubclass


def _has_traceable_wrapper_subclass_protocol(t: object) -> bool:
    return hasattr(t, "__tensor_flatten__") and hasattr(t, "__tensor_unflatten__")


def is_traceable_wrapper_subclass(t: object) -> TypeIs[TraceableWrapperSubclass]:
    """
    Returns whether ``t`` is a tensor subclass that matches the
    ``TraceableWrapperSubclass`` protocol at runtime.

    See ``TraceableWrapperSubclass`` for the canonical flatten/unflatten
    contract. Matching the protocol is necessary but not sufficient for full
    PT2 support: the subclass's ``__torch_dispatch__`` implementation must also
    desugar into traceable dispatcher operations and preserve aliasing semantics
    with ``return_and_correct_aliasing()`` when needed.
    """
    is_subclass = isinstance(t, torch.Tensor) and type(t) is not torch.Tensor
    return is_subclass and _has_traceable_wrapper_subclass_protocol(t)


def is_traceable_wrapper_subclass_type(
    t: type,
) -> TypeIs[type[TraceableWrapperSubclass]]:
    """Same as above, but takes a type argument instead of an instance."""
    return (
        issubclass(t, torch.Tensor)
        and t is not torch.Tensor
        and _has_traceable_wrapper_subclass_protocol(t)
    )


def transform_subclass(t, callback, outer_size=None, outer_stride=None):
    """
    Given a traceable, wrapper tensor subclass ``t`` that implements
    ``__torch_dispatch__`` and holds some inner tensors,
    and a callback of type ``Callable[[str, torch.Tensor], torch.Tensor]``,
    `transform_subclass` will construct a fresh instance of the wrapper tensor subclass.
    It will do so by grabbing each inner tensor attribute from the wrapper,
    passing them into ``callback`` to get a transformed tensor,
    and putting each transformed tensor into the fresh tensor subclass instance.

    Note: this function will not handle ensuring that the fresh subclass
    gets the same (autograd, and aliasing) metadata as the original tensor.
    This is generally handled in other subsystems like AOTAutograd.
    """
    outer_size = outer_size if outer_size is not None else t.size()
    outer_stride = outer_stride if outer_stride is not None else t.stride()

    attrs, ctx = t.__tensor_flatten__()
    transformed_tensors_dict = {}
    for attr in attrs:
        transformed_tensors_dict[attr] = callback(attr, getattr(t, attr))
    sub = type(t).__tensor_unflatten__(
        transformed_tensors_dict, ctx, outer_size, outer_stride
    )

    # NB: Purposefully guard here to simplify the inner / outer symbols.
    # Using sym_eq() for symbolic comparison can result in an expression that's too
    # difficult to guard on, so we use == here.
    if sub.shape != outer_size:
        raise AssertionError(
            f"Expected return value from {type(t)}__tensor_unflatten__() to have "
            f"shape equal to {outer_size}, but got: {sub.shape}"
        )
    if sub.stride() != outer_stride:
        raise AssertionError(
            f"Expected return value from {type(t)}__tensor_unflatten__() to have "
            f"stride equal to {outer_stride}, but got: {sub.stride()}"
        )

    return sub


def _correct_storage_aliasing(func, schema_info, args, outs) -> None:
    """
    Given: an OpOverload, a SchemaInfo (cached information from torchgen about schema),
    and the inputs/outputs to the OpOverload,
    this function checks to see if func is a view operator
    (by checking if any of the outputs in the op's schema
     are immutable aliases of inputs).
    If so, this function manually aliases the storage of the output tensor
    with its corresponding input tensor alias.
    It does this by unsafely overwriting the storage field of the output tensor
    to be the same storage as the input.
    """
    if not isinstance(func, torch._ops.OpOverload):
        raise AssertionError(f"func must be an OpOverload, got {type(args)}")
    if not isinstance(args, tuple):
        raise AssertionError(f"args must be a tuple, got {type(args)}")
    if not isinstance(outs, (list, tuple)):
        raise AssertionError(f"outs must be a list or tuple, got {type(args)}")

    def alias_non_inplace_storage(arg, ret) -> None:
        # This is hopefully a reasonable assert:
        # subclasses that rely on this API for output aliasing
        # should always return wrapper tensor subclasses for us to manually alias.
        # in theory if a subclass that needs this API wants to sometimes return
        # plain tensors, we could remove the assert and just not perform the aliasing,
        # but it seems safer to learn more about this case first.
        #
        # Performance note: This is all just to assert that the argument and result
        # types match, checking that is cheaper than is_traceable_wrapper_subclass_type,
        # and multiple returns are relatively unlikely, so just check up front!
        arg_type = type(arg)
        ret_type = type(ret)
        if arg_type is not ret_type and (
            is_traceable_wrapper_subclass_type(arg_type)
            or is_traceable_wrapper_subclass_type(ret_type)
        ):
            ret_list = ret if isinstance(ret, list) else [ret]
            for r in ret_list:
                if type(arg) is not type(r):
                    raise AssertionError(
                        f"Called {str(func)} with input of type {type(arg)}\n"
                        f"and output of type {type(ret)}. But expected types to match."
                    )
        # Need to call a non-dispatcher helper, because we explicitly do **not**
        # want our subclass to intercept the set_() call.
        # instead, our subclass should directly have its storage swapped out.
        # we **explicitly** don't want to reset the sizes on ret, if the storage implies a size change.
        # Why?
        # The purpose of this API is *not* to change the size/strides of our output- we assume it's already correct.
        # We just want to "fix up" the storage aliasing, without modifying or output's metadata.
        # Example: out = inp.expand(inp.shape[0], inp.shape[0])
        #     This requires swapping the storage of out to be the same as inp,
        #     but we do *not* want it to change the sizes/strides that were compute for out.

        if isinstance(ret, list):
            for r in ret:
                torch._functionalize_unsafe_set(r, arg)
        else:
            if not isinstance(ret, torch.Tensor):
                raise AssertionError(f"expected torch.Tensor, got {type(ret)}")
            torch._functionalize_unsafe_set(ret, arg)

    for arg_idx, return_idx in schema_info.read_only_alias_match_indexes:
        alias_non_inplace_storage(args[arg_idx], outs[return_idx])


def _get_write_alias(x) -> str | None:
    alias_set = x.alias_set
    if not alias_set or not x.is_write:
        return None
    # torchscript allows for complicated alias sets, but our dispatcher ops only really involve simple aliasing
    if len(alias_set) != 1:
        raise AssertionError("Expected alias_set to contain exactly one element")
    # timeit says next(iter(alias_set)) is faster than list(alias_set)[0] even for
    # set of size 1 on Python 3.13.
    return next(iter(alias_set))


# This abstracts over the fact that in return_and_correct_aliasing,
# we sometimes use torchgen schema parsing (for aten ops, since torchscript's schema parsing is sometimes buggy),
# and sometimes use torchscript schema parsing (for custom ops, for which torchgen parsing is untested).
@dataclass
class AliasInfo:
    alias_set: set[str]
    is_write: bool
    name: str | None


@dataclass
class SchemaInfo:
    args: list[AliasInfo]
    outs: list[AliasInfo]

    is_inplace_view_op: bool

    # [_get_write_alias(x) for x in outs]. Guaranteed to contain no Nones; we coerce
    # all-Nones result to empty list instead, and we don't support
    # some-but-not-all-Nones.
    outs_write_aliases: list[str] | None

    # List of (arg_idx, return_idx) where args[arg_idx].alias_set &
    # outs[out_idx].alias_set is not empty, and not args[arg_idx].is_write.
    read_only_alias_match_indexes: list[tuple[int, int]]


# Given an OpOverload, returns schema information on it.
# This is cached for efficiency, since it can involve running torchgen
@functools.cache
def get_alias_info(func) -> SchemaInfo:
    # For ATen ops: use torchgen (since torchscript parser doesn't handle alias annotations
    # properly for some ops that output tensorlists)
    if func.namespace == "aten":
        torchgen_schema_str = str(func._schema)
        if not torchgen_schema_str.startswith("aten::"):
            raise AssertionError(
                "Expected torchgen schema string to start with 'aten::'"
            )
        # remove the aten:: namespace, which is added by the torchscript parser,
        # and torchgen doesn't know how to handle
        torchgen_schema_str = torchgen_schema_str[6:]
        import re

        # the torchscript parser ends up converting int[2]=1 into int[2]=[1, 1],
        # which torchgen chokes on.
        torchgen_schema_str = re.sub(r"=\[[0, ]+\]", "=0", torchgen_schema_str)
        torchgen_schema_str = re.sub(r"=\[[1, ]+\]", "=1", torchgen_schema_str)
        # for aten::rot90 / aten:fft_*
        torchgen_schema_str = re.sub(
            r"=\[(-?[0-9]+), (-?[0-9]+)\]", r"=[\1,\2]", torchgen_schema_str
        )
        torchgen_schema = torchgen.model.FunctionSchema.parse(torchgen_schema_str)
        arg_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.annotation is None else set(a.annotation.alias_set)
                ),
                is_write=a.annotation is not None and a.annotation.is_write,
                name=a.name,
            )
            for a in torchgen_schema.arguments.flat_all
        ]
        out_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.annotation is None else set(a.annotation.alias_set)
                ),
                is_write=a.annotation is not None and a.annotation.is_write,
                name=a.name,
            )
            for a in torchgen_schema.returns
        ]
    else:
        # For non-aten ops, torchgen is untested so we rely on torchscript schema parsing
        arg_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.alias_info is None else set(a.alias_info.before_set)
                ),
                is_write=a.alias_info is not None and a.alias_info.is_write,
                name=a.name,
            )
            for a in func._schema.arguments
        ]
        out_schemas = [
            AliasInfo(
                alias_set=(
                    set() if a.alias_info is None else set(a.alias_info.before_set)
                ),
                is_write=a.alias_info is not None and a.alias_info.is_write,
                name=a.name,
            )
            for a in func._schema.returns
        ]
    read_only_alias_match_indexes = []
    for arg_idx, schema_arg in enumerate(arg_schemas):
        for return_idx, schema_out in enumerate(out_schemas):
            is_read_only_alias_match = (
                schema_arg.alias_set & schema_out.alias_set
            ) and not schema_arg.is_write
            if is_read_only_alias_match:
                read_only_alias_match_indexes.append((arg_idx, return_idx))

    outs_write_aliases_list: list[str | None] = [
        _get_write_alias(r) for r in out_schemas
    ]
    non_nones = sum(x is not None for x in outs_write_aliases_list)
    if non_nones == 0:
        outs_write_aliases: list[str] | None = None
    elif non_nones != len(outs_write_aliases_list):
        # simplifying assumption: we don't have **any** ops with return types like "-> (Tensor(a!), Tensor)"
        raise RuntimeError("Unsupported schema: " + str(func._schema))
    else:
        outs_write_aliases = cast(list[str], outs_write_aliases_list)

    schema_info = SchemaInfo(
        args=arg_schemas,
        outs=out_schemas,
        # This check is surprisingly expensive because pybind11 enum_s are
        # inefficient. Just cache it.
        is_inplace_view_op=torch.Tag.inplace_view in func.tags,
        outs_write_aliases=outs_write_aliases,
        read_only_alias_match_indexes=read_only_alias_match_indexes,
    )
    return schema_info


def autograd_would_have_decomposed(
    func: torch._ops.OpOverload, flat_args: Sequence[torch.Tensor | object]
) -> bool:
    """
    Suppose that an operator has CompositeImplicitAutograd decomp registered.
    Would autograd have used this decomposition?  It will only use it if there
    isn't an explicit backend registration for the device as well.  This function
    will tell if this would have occurred.

    Why do we need to apply these decompositions later?  When inference mode is
    on, the autograd key is bypassed entirely, so a lower level mode cannot rely
    on the decomposition having been applied.  It's easy to accidentally never apply
    the decomposition, resulting in an operator showing up in a graph that
    is unexpected.

    Why do we need to AVOID applying the decomposition when autograd wouldn't
    have decomposed?  If autograd doesn't decompose, this means in eager mode
    we would have run the fused kernel.  It must be possible to trace this
    fused kernel directly into the graph for fidelity with eager (NB: a user
    has the option of then further decomposing at proxy tensor mode via
    decomposition table, but we must preserve it to proxy mode to have the
    choice.)

    Why does functionalization need to also perform the test here?  This is
    because some CompositeImplicitAutograd decompositions are not functional.
    If we are eventually going to decompose, we need to do this while we can
    still turn functionalization back on, so those decompositions get functionalized.
    So an early decomposition in functionalization may still be necessary.  Note that
    if proxy tensor decomposition process could turn functionalization back on, this
    wouldn't be necessary, and maybe that is a useful thing to do anyway because
    the decomposition table is user specified and a user could violate the functional
    decomp requirement with a bad decomp.  If this happened, then you could always
    pass through functionalization.
    """
    has_backend_registration = False
    for a in flat_args:
        if isinstance(a, torch.Tensor):
            backend_key = torch._C._parse_dispatch_key(
                torch._C._dispatch_key_for_device(a.device.type)
            )
            if backend_key is None:
                raise AssertionError(
                    f"failed to parse dispatch key for device {a.device.type}"
                )
            # TODO: use func.has_kernel_for_dispatch_key(backend_key)
            # but this one checks py_impl and CompositeImplicitAutograd
            # incorrectly shows up as has backend reg here
            has_backend_registration = torch._C._dispatch_has_kernel_for_dispatch_key(
                func.name(), backend_key
            )

            # in theory we should take all backend keys and take the highest priority one
            # to properly mimic the dispatcher,
            # this just grabs the first tensor and takes its device key
            break
    return not has_backend_registration


def return_and_correct_aliasing(func, args, kwargs, out):
    """
    This function should be used by wrapper tensor ``__torch_dispatch__`` subclasses
    that would like to work with torch.compile. It ensures that the subclass
    properly implements the aliasing behavior of every op,
    which is needed for correctness in AOTAutograd.
    This function will handle:

        * When we see a view op, we will alias the storages of any
          input and output tensor subclasses

        * When we see an inplace or out= op, we will directly
          return the corresponding input tensor, instead of returning
          a (potentially) fresh output tensor.
    """

    # Caching here because torchgen parsing is definitely not fast, and this function is called
    # once for every op in the graph during functionalization.
    schema_info = get_alias_info(func)

    def get_arg_from_alias(output_alias, schema_info, args, kwargs):
        new_args, new_kwargs = torch.fx.operator_schemas.normalize_function(  # type: ignore[misc]
            func, args=args, kwargs=kwargs
        )

        arg_indices = [
            i for i, a in enumerate(schema_info.args) if output_alias in a.alias_set
        ]
        # For any dispatcher op with an output alias, we expect it to map to exactly one alias in the schema's input arguments.
        if len(arg_indices) != 1:
            raise AssertionError(
                "Expected exactly one argument index for the given output alias"
            )
        idx = arg_indices[0]
        arg_info = schema_info.args[idx]
        if arg_info.name is not None and arg_info.name in new_kwargs:
            return new_kwargs[arg_info.name]
        return new_args[idx]

    # Fix up the storages of any outs so that they point to the same storage as the input,
    # if func is a view op.
    _correct_storage_aliasing(
        func, schema_info, args, (out,) if not isinstance(out, tuple) else out
    )

    # For inplace_view ops in particular, we'll try hard to make sure that the wrapper subclass's
    # metadata is set correctly.
    if schema_info.is_inplace_view_op:
        # no_dispatch() to make sure that we secretly change the metadata on the wrapper,
        # but don't end up dispatching the op anywhere else.
        mutated_args = [
            x
            for i, x in enumerate(args)
            if _get_write_alias(schema_info.args[i]) is not None
        ]
        # Assumption: we have a very small number of inplace_view ops that follow a strict schema:
        # there is only a single argument that gets its metadata mutated.
        if len(mutated_args) != 1:
            raise AssertionError(
                "expected exactly one mutated arg for inplace_view ops"
            )
        # This check exists because we generally *do* want to update the metadata of any wrapper subclasses,
        # but FunctionalTensor is special: it overrides all size/stride calls to plumb to the inner tensor.
        # so we don't actually need to update the metadata (and attempting to do so causes errors)
        from torch._subclasses.functional_tensor import FunctionalTensor

        if not isinstance(mutated_args[0], FunctionalTensor):
            with torch.utils._mode_utils.no_dispatch():
                # See Note: [Fake Tensor Dispatch Keys]
                # we're borrowing the way it modifies dispatch key TLS.
                meta_in_tls = torch._C._meta_in_tls_dispatch_include()
                torch._C._set_meta_in_tls_dispatch_include(True)
                try:
                    func(*args, **kwargs)
                finally:
                    torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)

    # Next: we need to make sure to return inputs directly, if the output is a mutable alias (e.g. add_()).

    schema_info_outs_write_aliases = schema_info.outs_write_aliases
    # simple case: none of our outputs have mutable aliases, so we can return the output as-is
    if schema_info_outs_write_aliases is None:
        return out

    if len(schema_info_outs_write_aliases) == 1:
        return get_arg_from_alias(
            schema_info_outs_write_aliases[0], schema_info, args, kwargs
        )

    # In the multi-return case, all aten ops return a tuple / list, so cast accordingly.
    outs_to_return = type(out)(
        [
            (get_arg_from_alias(write_alias, schema_info, args, kwargs))
            for write_alias in schema_info_outs_write_aliases
        ]
    )
    return outs_to_return
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import functools
import itertools
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from typing import *  # noqa: F403
from typing_extensions import Self
import enum
from weakref import ReferenceType

import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch._C._autograd import _make_saved_tensor, SavedTensor
from typing import NoReturn, overload, ParamSpec, TypeVar

__all__ = [
    "checkpoint",
    "checkpoint_sequential",
    "CheckpointError",
    "CheckpointFunction",
    "check_backward_validity",
    "detach_variable",
    "get_device_states",
    "set_device_states",
    "noop_context_fn",
    "set_checkpoint_early_stop",
    "DefaultDeviceType",
    "set_checkpoint_debug_enabled",
    "CheckpointPolicy",
    "SelectiveCheckpointContext",
    "create_selective_checkpoint_contexts",
    "SAC_IGNORED_OPS",
    "GraphExecGroup",
]

_DEFAULT_DETERMINISM_MODE = "default"

_T = TypeVar("_T")
_P = ParamSpec("_P")

_checkpoint_debug_enabled: bool | None = None


@contextlib.contextmanager
def set_checkpoint_debug_enabled(enabled: bool | None):
    """
    Context manager that sets whether checkpoint should print additional debug
    information when running. See the ``debug`` flag for
    :func:`~torch.utils.checkpoint.checkpoint` for more information. Note that
    when set, this context manager overrides the value of ``debug`` passed to
    checkpoint. To defer to the local setting, pass ``None`` to this context.

    Args:
        enabled (bool): Whether checkpoint should print debug information.
            Default is 'None'.
    """
    global _checkpoint_debug_enabled
    try:
        prev = _checkpoint_debug_enabled
        _checkpoint_debug_enabled = enabled
        yield
    finally:
        _checkpoint_debug_enabled = prev


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            f"Only tuple of tensors is supported. Got Unsupported input type: {type(inputs).__name__}"
        )


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn(
            "None of the inputs have requires_grad=True. Gradients will be None", stacklevel=2
        )


def _get_device_module(device="cuda"):
    if device == "meta":
        return torch.device("meta")
    device_module = getattr(torch, device)
    return device_module


class DefaultDeviceType:
    r"""
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    be used. The default value is 'cuda'. The device type is used in
    the checkpointing process when determining which device states
    to save and restore for recomputation.
    """

    _default_device_type: str | None = None

    @staticmethod
    def set_device_type(device: str = "cuda") -> None:
        """
        Set the default device type for checkpointing.

        Args:
            device (str): The device type to be set as default. Default is 'cuda'.
        """
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        """
        Get the current default device type for checkpointing.

        Returns:
            str: The current default device type.
        """
        if not DefaultDeviceType._default_device_type:
            DefaultDeviceType._default_device_type = acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"

        return DefaultDeviceType._default_device_type


def _infer_device_type(*args):
    device_types = []

    def add_device_types(arg):
        nonlocal device_types
        if isinstance(arg, torch.Tensor) and arg.device.type != "cpu":
            device_types.append(arg.device.type)
        return arg
    tree_map(add_device_types, args)

    device_types_set = set(device_types)
    if len(device_types_set) > 1:
        warnings.warn(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if CUDA devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
            f"\nDevice types: {sorted(device_types_set)} first device type: {device_types[0]}", stacklevel=2
        )
    if len(device_types) == 0:
        return DefaultDeviceType.get_device_type()
    elif "cuda" in device_types_set:
        return "cuda"
    else:
        return device_types[0]


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_device_ids = []

    def add_device_ids(arg):
        nonlocal fwd_device_ids
        if isinstance(arg, torch.Tensor) and arg.device.type not in {"cpu", "meta"}:
            fwd_device_ids.append(arg.get_device())
        return arg
    tree_map(add_device_ids, args)

    fwd_device_states = []
    device_module = _get_device_module(_infer_device_type(*args))
    for device_id in fwd_device_ids:
        with device_module.device(device_id):
            fwd_device_states.append(device_module.get_rng_state())

    return fwd_device_ids, fwd_device_states


def set_device_states(devices, states, *, device_type=None) -> None:
    """Sets random number generator states for the specified devices.

    Args:
        devices: Device ids to set states for.
        states: States to set.
        device_type: ``device_type`` of the devices to set states for. Default
            is the device returned by a call to ``DefaultDeviceType.get_device_type()``,
            which is ``cuda`` if not changed by calling ``DefaultDeviceType::set_device_type()``.
    """
    if device_type is None:
        device_type = DefaultDeviceType.get_device_type()
    if device_type == "meta":
        return
    device_module = _get_device_module(device_type)
    for device, state in zip(devices, states, strict=False):
        with device_module.device(device):
            device_module.set_rng_state(state)


def _get_autocast_kwargs(device_type="cuda"):
    if torch.amp.is_autocast_available(device_type):
        device_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    else:
        device_autocast_kwargs = None

    cpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled('cpu'),
        "dtype": torch.get_autocast_dtype('cpu'),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    return device_autocast_kwargs, cpu_autocast_kwargs


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)
            detached_inputs = detach_variable(tuple(inputs))

            device_autocast_ctx = torch.amp.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else contextlib.nullcontext()
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary"
            )
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        return (None, None) + grads


def noop_context_fn():
    return contextlib.nullcontext(), contextlib.nullcontext()


class _CheckpointedFunction:
    def __init__(self, function, **checkpoint_kwargs):
        self.function = function
        self.checkpoint_kwargs = dict(checkpoint_kwargs)
        functools.update_wrapper(self, function, updated=())

    def __call__(self, *args, **kwargs):
        return _checkpoint_impl(
            self.function,
            args,
            kwargs,
            **self.checkpoint_kwargs,
        )

    def __get__(self, instance, owner=None):
        # Supports using the curried form as a method decorator. When accessed
        # off an instance, bind the wrapped function to that instance via its
        # own descriptor protocol and re-wrap, so the checkpoint config carries
        # over to the bound method.
        if instance is None:
            return self
        descriptor_get = getattr(self.function, "__get__", None)
        if descriptor_get is None:
            return self
        return type(self)(
            descriptor_get(instance, owner),
            **self.checkpoint_kwargs,
        )


def _make_checkpoint_wrapper(function, **checkpoint_kwargs):
    return _CheckpointedFunction(function, **checkpoint_kwargs)


@overload
def checkpoint(
    function: Callable[..., _T],
    *args: Any,
    use_reentrant: bool | None = None,
    preserve_rng_state: bool = True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    respect_saved_tensors_hooks: bool | None = None,
    **kwargs: Any,
) -> _T: ...


@overload
def checkpoint(
    function: None = None,
    *,
    use_reentrant: bool | None = None,
    preserve_rng_state: bool = True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    respect_saved_tensors_hooks: bool | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


# Note: [torch.compile and checkpoint]
# TorchDynamo does not step inside utils.checkpoint function.  The flow
# looks likes this
#  1) TorchDynamo tries to wrap utils.checkpoint in a HigherOrderOp by
#     speculatively checking if the forward function is safe to trace.
#  2) If yes, then Dynamo-generated Fx graph has the wrapped higher
#     order op. As a result, TorchDynamo does not look inside utils.checkpoint.
#  3) If not, then TorchDynamo falls back to eager by performing a graph
#     break. And here, the following disable wrapper ensures that
#     TorchDynamo does not trigger again on the frames created by
#     utils.checkpoint innards.
@torch._disable_dynamo
def checkpoint(
    function=None,
    *args,
    use_reentrant: bool | None = None,
    preserve_rng_state: bool = True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    respect_saved_tensors_hooks: bool | None = None,
    **kwargs
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    By default, tensors computed during the forward pass are kept alive until
    they are used in gradient computations in the backward pass. To reduce this
    memory usage, tensors produced in the passed :attr:`function` are not kept
    alive until the backward pass. Instead, any passed tensors in :attr:`args`
    are kept alive, and the unsaved tensors are recomputed by re-invoking
    :attr:`function` in the backward pass as needed for gradient computation.
    Activation checkpointing can be applied to any part of a model -- this is
    sometimes described as "checkpointing" that part of the model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.9 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``.
            If omitted, ``checkpoint`` returns a decorator that can be used to
            wrap a function before passing user arguments.
        args: tuple containing inputs to the :attr:`function`

    Keyword args:
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn't take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.9 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        early_stop(bool, optional): If ``True``, non-reentrant checkpoint stops
            recomputation as soon as it has computed all needed Tensors. This
            argument is ignored if ``use_reentrant=True``. Can be overridden
            globally using :func:`set_checkpoint_early_stop` context manager.
            Default: ``True``.
        respect_saved_tensors_hooks(bool, optional): Whether tensors that
            selective activation checkpointing (SAC) saves are routed through
            the surrounding user
            :func:`torch.autograd.graph.saved_tensors_hooks` (e.g.
            :func:`torch.autograd.graph.save_on_cpu`). Such tensors are held
            outside the autograd graph, so historically these hooks did not see
            them; a future release will make checkpoint honor the hooks by
            default. Until then the default (``None``) keeps the legacy behavior
            and emits a ``FutureWarning`` when a user hook is in scope; pass
            ``True`` to opt in or ``False`` to keep the legacy behavior without
            the warning. Has no effect without SAC (plain checkpoint recomputes
            rather than saves) or when ``use_reentrant=True``. This argument is
            only supported if ``use_reentrant=False``.

    Returns:
        Output of running :attr:`function` on :attr:`*args`, or a decorator
        when :attr:`function` is omitted.

    Example:
        >>> # xdoctest: +SKIP("stub")
        >>> # Direct call: checkpoint the function immediately.
        >>> out = torch.utils.checkpoint.checkpoint(
        ...     fn, *args, use_reentrant=False, **kwargs
        ... )
        >>> # Decorator/curried form: configure once, then call.
        >>> checkpointed_fn = torch.utils.checkpoint.checkpoint(
        ...     use_reentrant=False,
        ...     preserve_rng_state=False,
        ... )(fn)
        >>> out = checkpointed_fn(*args, **kwargs)
        >>> # The same decorator can be applied directly to a function or method.
        >>> @torch.utils.checkpoint.checkpoint(use_reentrant=False)
        ... def fn(*args, **kwargs):
        ...     ...
        >>> out = fn(*args, **kwargs)
    """
    if function is None and not args:
        if kwargs:
            raise ValueError(
                "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
            )
        return functools.partial(
            _make_checkpoint_wrapper,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state,
            context_fn=context_fn,
            determinism_check=determinism_check,
            debug=debug,
            early_stop=early_stop,
            respect_saved_tensors_hooks=respect_saved_tensors_hooks,
        )

    return _checkpoint_impl(
        function,
        args,
        kwargs,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
        context_fn=context_fn,
        determinism_check=determinism_check,
        debug=debug,
        early_stop=early_stop,
        respect_saved_tensors_hooks=respect_saved_tensors_hooks,
    )


def _checkpoint_impl(
    function,
    args,
    kwargs,
    *,
    use_reentrant: bool | None = None,
    preserve_rng_state: bool = True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    respect_saved_tensors_hooks: bool | None = None,
):
    preserve = preserve_rng_state

    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. Starting in PyTorch 2.9, calling checkpoint "
            "without use_reentrant will raise an exception. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.",
            stacklevel=3,
        )
        use_reentrant = True

    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    if use_reentrant:
        if (
            context_fn is not noop_context_fn
            or debug is not False
            or respect_saved_tensors_hooks is not None
        ):
            raise ValueError(
                "Passing `context_fn`, `debug`, or `respect_saved_tensors_hooks` "
                "is only supported when use_reentrant=False."
            )
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator_impl(
            function,
            args,
            kwargs,
            preserve_rng_state=preserve,
            context_fn=context_fn,
            determinism_check=determinism_check,
            debug=debug,
            early_stop=early_stop,
            respect_saved_tensors_hooks=respect_saved_tensors_hooks,
        )
        # Runs pre-forward logic
        next(gen)
        try:
            ret = function(*args, **kwargs)
        except BaseException as e:
            try:
                gen.throw(e)
            except StopIteration as stop:
                raise RuntimeError(
                    "torch.utils.checkpoint: the forward context provided by "
                    "context_fn suppressed an exception raised during the "
                    "checkpointed forward. This is not supported because "
                    "checkpoint cannot return a value for a failed forward."
                ) from stop
            raise
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret
        raise AssertionError(
            "torch.utils.checkpoint: expected context_fn generator to yield "
            "exactly twice, but it yielded more than twice."
        )


def checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs):
    r"""Checkpoint a sequential model to save memory.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will not store
    the intermediate activations. The inputs of each checkpointed segment will
    be saved for re-running the segment in the backward pass.

    .. warning::
        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.9 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True` variant, please see
