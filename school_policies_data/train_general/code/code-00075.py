            - sign '#' must stick in the begin of group name and only within ``@targets``

        **Example**:
            .. code-block:: console

                $ "@targets #avx_group other_tokens" > group_inside.c

            >>> CCompilerOpt.conf_target_groups["avx_group"] = \\
            "$werror $maxopt avx2 avx512f avx512_skx"
            >>> cco = CCompilerOpt(cc_instance)
            >>> cco.try_dispatch(["group_inside.c"])

    conf_c_prefix : str
        The prefix of public C definitions. Default is ``"NPY_"``.

    conf_c_prefix_ : str
        The prefix of internal C definitions. Default is ``"NPY__"``.

    conf_cc_flags : dict
        Nested dictionaries defining several compiler flags
        that linked to some major functions, the main key
        represent the compiler name and sub-keys represent
        flags names. Default is already covers all supported
        **C** compilers.

        Sub-keys explained as follows:

        "native": str or None
            used by argument option `native`, to detect the current
            machine support via the compiler.
        "werror": str or None
            utilized to treat warning as errors during testing CPU features
            against the compiler and also for target's policy `$werror`
            via dispatch-able sources.
        "maxopt": str or None
            utilized for target's policy '$maxopt' and the value should
            contains the maximum acceptable optimization by the compiler.
            e.g. in gcc ``'-O3'``

        **Notes**:
            * case-sensitive for compiler names and flags
            * use space to separate multiple flags
            * any flag will tested against the compiler and it will skipped
              if it's not applicable.

    conf_min_features : dict
        A dictionary defines the used CPU features for
        argument option ``'min'``, the key represent the CPU architecture
        name e.g. ``'x86'``. Default values provide the best effort
        on wide range of users platforms.

        **Note**: case-sensitive for architecture names.

    conf_features : dict
        Nested dictionaries used for identifying the CPU features.
        the primary key is represented as a feature name or group name
        that gathers several features. Default values covers all
        supported features but without the major options like "flags",
        these undefined options handle it by method `conf_features_partial()`.
        Default value is covers almost all CPU features for *X86*, *IBM/Power64*
        and *ARM 7/8*.

        Sub-keys explained as follows:

        "implies" : str or list, optional,
            List of CPU feature names to be implied by it,
            the feature name must be defined within `conf_features`.
            Default is None.

        "flags": str or list, optional
            List of compiler flags. Default is None.

        "detect": str or list, optional
            List of CPU feature names that required to be detected
            in runtime. By default, its the feature name or features
            in "group" if its specified.

        "implies_detect": bool, optional
            If True, all "detect" of implied features will be combined.
            Default is True. see `feature_detect()`.

        "group": str or list, optional
            Same as "implies" but doesn't require the feature name to be
            defined within `conf_features`.

        "interest": int, required
            a key for sorting CPU features

        "headers": str or list, optional
            intrinsics C header file

        "disable": str, optional
            force disable feature, the string value should contains the
            reason of disabling.

        "autovec": bool or None, optional
            True or False to declare that CPU feature can be auto-vectorized
            by the compiler.
            By default(None), treated as True if the feature contains at
            least one applicable flag. see `feature_can_autovec()`

        "extra_checks": str or list, optional
            Extra test case names for the CPU feature that need to be tested
            against the compiler.

            Each test case must have a C file named ``extra_xxxx.c``, where
            ``xxxx`` is the case name in lower case, under 'conf_check_path'.
            It should contain at least one intrinsic or function related to the test case.

            If the compiler able to successfully compile the C file then `CCompilerOpt`
            will add a C ``#define`` for it into the main dispatch header, e.g.
            ``#define {conf_c_prefix}_XXXX`` where ``XXXX`` is the case name in upper case.

        **NOTES**:
            * space can be used as separator with options that supports "str or list"
            * case-sensitive for all values and feature name must be in upper-case.
            * if flags aren't applicable, its will skipped rather than disable the
              CPU feature
            * the CPU feature will disabled if the compiler fail to compile
              the test file
    """
    conf_nocache = False
    conf_noopt = False
    conf_cache_factors = None
    conf_tmp_path = None
    conf_check_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "checks"
    )
    conf_target_groups = {}
    conf_c_prefix = 'NPY_'
    conf_c_prefix_ = 'NPY__'
    conf_cc_flags = dict(
        gcc = dict(
            # native should always fail on arm and ppc64,
            # native usually works only with x86
            native = '-march=native',
            opt = '-O3',
            werror = '-Werror',
        ),
        clang = dict(
            native = '-march=native',
            opt = "-O3",
            # One of the following flags needs to be applicable for Clang to
            # guarantee the sanity of the testing process, however in certain
            # cases `-Werror` gets skipped during the availability test due to
            # "unused arguments" warnings.
            # see https://github.com/numpy/numpy/issues/19624
            werror = '-Werror=switch -Werror',
        ),
        icc = dict(
            native = '-xHost',
            opt = '-O3',
            werror = '-Werror',
        ),
        iccw = dict(
            native = '/QxHost',
            opt = '/O3',
            werror = '/Werror',
        ),
        msvc = dict(
            native = None,
            opt = '/O2',
            werror = '/WX',
        ),
        fcc = dict(
            native = '-mcpu=a64fx',
            opt = None,
            werror = None,
        )
    )
    conf_min_features = dict(
        x86 = "SSE SSE2",
        x64 = "SSE SSE2 SSE3",
        ppc64 = '', # play it safe
        ppc64le = "VSX VSX2",
        s390x = '',
        armhf = '', # play it safe
        aarch64 = "NEON NEON_FP16 NEON_VFPV4 ASIMD"
    )
    conf_features = dict(
        # X86
        SSE = dict(
            interest=1, headers="xmmintrin.h",
            # enabling SSE without SSE2 is useless also
            # it's non-optional for x86_64
            implies="SSE2"
        ),
        SSE2   = dict(interest=2, implies="SSE", headers="emmintrin.h"),
        SSE3   = dict(interest=3, implies="SSE2", headers="pmmintrin.h"),
        SSSE3  = dict(interest=4, implies="SSE3", headers="tmmintrin.h"),
        SSE41  = dict(interest=5, implies="SSSE3", headers="smmintrin.h"),
        POPCNT = dict(interest=6, implies="SSE41", headers="popcntintrin.h"),
        SSE42  = dict(interest=7, implies="POPCNT"),
        AVX    = dict(
            interest=8, implies="SSE42", headers="immintrin.h",
            implies_detect=False
        ),
        XOP    = dict(interest=9, implies="AVX", headers="x86intrin.h"),
        FMA4   = dict(interest=10, implies="AVX", headers="x86intrin.h"),
        F16C   = dict(interest=11, implies="AVX"),
        FMA3   = dict(interest=12, implies="F16C"),
        AVX2   = dict(interest=13, implies="F16C"),
        AVX512F = dict(
            interest=20, implies="FMA3 AVX2", implies_detect=False,
            extra_checks="AVX512F_REDUCE"
        ),
        AVX512CD = dict(interest=21, implies="AVX512F"),
        AVX512_KNL = dict(
            interest=40, implies="AVX512CD", group="AVX512ER AVX512PF",
            detect="AVX512_KNL", implies_detect=False
        ),
        AVX512_KNM = dict(
            interest=41, implies="AVX512_KNL",
            group="AVX5124FMAPS AVX5124VNNIW AVX512VPOPCNTDQ",
            detect="AVX512_KNM", implies_detect=False
        ),
        AVX512_SKX = dict(
            interest=42, implies="AVX512CD", group="AVX512VL AVX512BW AVX512DQ",
            detect="AVX512_SKX", implies_detect=False,
            extra_checks="AVX512BW_MASK AVX512DQ_MASK"
        ),
        AVX512_CLX = dict(
            interest=43, implies="AVX512_SKX", group="AVX512VNNI",
            detect="AVX512_CLX"
        ),
        AVX512_CNL = dict(
            interest=44, implies="AVX512_SKX", group="AVX512IFMA AVX512VBMI",
            detect="AVX512_CNL", implies_detect=False
        ),
        AVX512_ICL = dict(
            interest=45, implies="AVX512_CLX AVX512_CNL",
            group="AVX512VBMI2 AVX512BITALG AVX512VPOPCNTDQ",
            detect="AVX512_ICL", implies_detect=False
        ),
        AVX512_SPR = dict(
            interest=46, implies="AVX512_ICL", group="AVX512FP16",
            detect="AVX512_SPR", implies_detect=False
        ),
        # IBM/Power
        ## Power7/ISA 2.06
        VSX = dict(interest=1, headers="altivec.h", extra_checks="VSX_ASM"),
        ## Power8/ISA 2.07
        VSX2 = dict(interest=2, implies="VSX", implies_detect=False),
        ## Power9/ISA 3.00
        VSX3 = dict(interest=3, implies="VSX2", implies_detect=False,
                    extra_checks="VSX3_HALF_DOUBLE"),
        ## Power10/ISA 3.1
        VSX4 = dict(interest=4, implies="VSX3", implies_detect=False,
                    extra_checks="VSX4_MMA"),
        # IBM/Z
        ## VX(z13) support
        VX = dict(interest=1, headers="vecintrin.h"),
        ## Vector-Enhancements Facility
        VXE = dict(interest=2, implies="VX", implies_detect=False),
        ## Vector-Enhancements Facility 2
        VXE2 = dict(interest=3, implies="VXE", implies_detect=False),
        # ARM
        NEON  = dict(interest=1, headers="arm_neon.h"),
        NEON_FP16 = dict(interest=2, implies="NEON"),
        ## FMA
        NEON_VFPV4 = dict(interest=3, implies="NEON_FP16"),
        ## Advanced SIMD
        ASIMD = dict(interest=4, implies="NEON_FP16 NEON_VFPV4", implies_detect=False),
        ## ARMv8.2 half-precision & vector arithm
        ASIMDHP = dict(interest=5, implies="ASIMD"),
        ## ARMv8.2 dot product
        ASIMDDP = dict(interest=6, implies="ASIMD"),
        ## ARMv8.2 Single & half-precision Multiply
        ASIMDFHM = dict(interest=7, implies="ASIMDHP"),
    )
    def conf_features_partial(self):
        """Return a dictionary of supported CPU features by the platform,
        and accumulate the rest of undefined options in `conf_features`,
        the returned dict has same rules and notes in
        class attribute `conf_features`, also its override
        any options that been set in 'conf_features'.
        """
        if self.cc_noopt:
            # optimization is disabled
            return {}

        on_x86 = self.cc_on_x86 or self.cc_on_x64
        is_unix = self.cc_is_gcc or self.cc_is_clang or self.cc_is_fcc

        if on_x86 and is_unix: return dict(
            SSE    = dict(flags="-msse"),
            SSE2   = dict(flags="-msse2"),
            SSE3   = dict(flags="-msse3"),
            SSSE3  = dict(flags="-mssse3"),
            SSE41  = dict(flags="-msse4.1"),
            POPCNT = dict(flags="-mpopcnt"),
            SSE42  = dict(flags="-msse4.2"),
            AVX    = dict(flags="-mavx"),
            F16C   = dict(flags="-mf16c"),
            XOP    = dict(flags="-mxop"),
            FMA4   = dict(flags="-mfma4"),
            FMA3   = dict(flags="-mfma"),
            AVX2   = dict(flags="-mavx2"),
            AVX512F = dict(flags="-mavx512f -mno-mmx"),
            AVX512CD = dict(flags="-mavx512cd"),
            AVX512_KNL = dict(flags="-mavx512er -mavx512pf"),
            AVX512_KNM = dict(
                flags="-mavx5124fmaps -mavx5124vnniw -mavx512vpopcntdq"
            ),
            AVX512_SKX = dict(flags="-mavx512vl -mavx512bw -mavx512dq"),
            AVX512_CLX = dict(flags="-mavx512vnni"),
            AVX512_CNL = dict(flags="-mavx512ifma -mavx512vbmi"),
            AVX512_ICL = dict(
                flags="-mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq"
            ),
            AVX512_SPR = dict(flags="-mavx512fp16"),
        )
        if on_x86 and self.cc_is_icc: return dict(
            SSE    = dict(flags="-msse"),
            SSE2   = dict(flags="-msse2"),
            SSE3   = dict(flags="-msse3"),
            SSSE3  = dict(flags="-mssse3"),
            SSE41  = dict(flags="-msse4.1"),
            POPCNT = {},
            SSE42  = dict(flags="-msse4.2"),
            AVX    = dict(flags="-mavx"),
            F16C   = {},
            XOP    = dict(disable="Intel Compiler doesn't support it"),
            FMA4   = dict(disable="Intel Compiler doesn't support it"),
            # Intel Compiler doesn't support AVX2 or FMA3 independently
            FMA3 = dict(
                implies="F16C AVX2", flags="-march=core-avx2"
            ),
            AVX2 = dict(implies="FMA3", flags="-march=core-avx2"),
            # Intel Compiler doesn't support AVX512F or AVX512CD independently
            AVX512F = dict(
                implies="AVX2 AVX512CD", flags="-march=common-avx512"
            ),
            AVX512CD = dict(
                implies="AVX2 AVX512F", flags="-march=common-avx512"
            ),
            AVX512_KNL = dict(flags="-xKNL"),
            AVX512_KNM = dict(flags="-xKNM"),
            AVX512_SKX = dict(flags="-xSKYLAKE-AVX512"),
            AVX512_CLX = dict(flags="-xCASCADELAKE"),
            AVX512_CNL = dict(flags="-xCANNONLAKE"),
            AVX512_ICL = dict(flags="-xICELAKE-CLIENT"),
            AVX512_SPR = dict(disable="Not supported yet")
        )
        if on_x86 and self.cc_is_iccw: return dict(
            SSE    = dict(flags="/arch:SSE"),
            SSE2   = dict(flags="/arch:SSE2"),
            SSE3   = dict(flags="/arch:SSE3"),
            SSSE3  = dict(flags="/arch:SSSE3"),
            SSE41  = dict(flags="/arch:SSE4.1"),
            POPCNT = {},
            SSE42  = dict(flags="/arch:SSE4.2"),
            AVX    = dict(flags="/arch:AVX"),
            F16C   = {},
            XOP    = dict(disable="Intel Compiler doesn't support it"),
            FMA4   = dict(disable="Intel Compiler doesn't support it"),
            # Intel Compiler doesn't support FMA3 or AVX2 independently
            FMA3 = dict(
                implies="F16C AVX2", flags="/arch:CORE-AVX2"
            ),
            AVX2 = dict(
                implies="FMA3", flags="/arch:CORE-AVX2"
            ),
            # Intel Compiler doesn't support AVX512F or AVX512CD independently
            AVX512F = dict(
                implies="AVX2 AVX512CD", flags="/Qx:COMMON-AVX512"
            ),
            AVX512CD = dict(
                implies="AVX2 AVX512F", flags="/Qx:COMMON-AVX512"
            ),
            AVX512_KNL = dict(flags="/Qx:KNL"),
            AVX512_KNM = dict(flags="/Qx:KNM"),
            AVX512_SKX = dict(flags="/Qx:SKYLAKE-AVX512"),
            AVX512_CLX = dict(flags="/Qx:CASCADELAKE"),
            AVX512_CNL = dict(flags="/Qx:CANNONLAKE"),
            AVX512_ICL = dict(flags="/Qx:ICELAKE-CLIENT"),
            AVX512_SPR = dict(disable="Not supported yet")
        )
        if on_x86 and self.cc_is_msvc: return dict(
            SSE = dict(flags="/arch:SSE") if self.cc_on_x86 else {},
            SSE2 = dict(flags="/arch:SSE2") if self.cc_on_x86 else {},
            SSE3   = {},
            SSSE3  = {},
            SSE41  = {},
            POPCNT = dict(headers="nmmintrin.h"),
            SSE42  = {},
            AVX    = dict(flags="/arch:AVX"),
            F16C   = {},
            XOP    = dict(headers="ammintrin.h"),
            FMA4   = dict(headers="ammintrin.h"),
            # MSVC doesn't support FMA3 or AVX2 independently
            FMA3 = dict(
                implies="F16C AVX2", flags="/arch:AVX2"
            ),
            AVX2 = dict(
                implies="F16C FMA3", flags="/arch:AVX2"
            ),
            # MSVC doesn't support AVX512F or AVX512CD independently,
            # always generate instructions belong to (VL/VW/DQ)
            AVX512F = dict(
                implies="AVX2 AVX512CD AVX512_SKX", flags="/arch:AVX512"
            ),
            AVX512CD = dict(
                implies="AVX512F AVX512_SKX", flags="/arch:AVX512"
            ),
            AVX512_KNL = dict(
                disable="MSVC compiler doesn't support it"
            ),
            AVX512_KNM = dict(
                disable="MSVC compiler doesn't support it"
            ),
            AVX512_SKX = dict(flags="/arch:AVX512"),
            AVX512_CLX = {},
            AVX512_CNL = {},
            AVX512_ICL = {},
            AVX512_SPR= dict(
                disable="MSVC compiler doesn't support it"
            )
        )

        on_power = self.cc_on_ppc64le or self.cc_on_ppc64
        if on_power:
            partial = dict(
                VSX = dict(
                    implies=("VSX2" if self.cc_on_ppc64le else ""),
                    flags="-mvsx"
                ),
                VSX2 = dict(
                    flags="-mcpu=power8", implies_detect=False
                ),
                VSX3 = dict(
                    flags="-mcpu=power9 -mtune=power9", implies_detect=False
                ),
                VSX4 = dict(
                    flags="-mcpu=power10 -mtune=power10", implies_detect=False
                )
            )
            if self.cc_is_clang:
                partial["VSX"]["flags"]  = "-maltivec -mvsx"
                partial["VSX2"]["flags"] = "-mcpu=power8"
                partial["VSX3"]["flags"] = "-mcpu=power9"
                partial["VSX4"]["flags"] = "-mcpu=power10"

            return partial

        on_zarch = self.cc_on_s390x
        if on_zarch:
            partial = dict(
                VX = dict(
                    flags="-march=arch11 -mzvector"
                ),
                VXE = dict(
                    flags="-march=arch12", implies_detect=False
                ),
                VXE2 = dict(
                    flags="-march=arch13", implies_detect=False
                )
            )

            return partial


        if self.cc_on_aarch64 and is_unix: return dict(
            NEON = dict(
                implies="NEON_FP16 NEON_VFPV4 ASIMD", autovec=True
            ),
            NEON_FP16 = dict(
                implies="NEON NEON_VFPV4 ASIMD", autovec=True
            ),
            NEON_VFPV4 = dict(
                implies="NEON NEON_FP16 ASIMD", autovec=True
            ),
            ASIMD = dict(
                implies="NEON NEON_FP16 NEON_VFPV4", autovec=True
            ),
            ASIMDHP = dict(
                flags="-march=armv8.2-a+fp16"
            ),
            ASIMDDP = dict(
                flags="-march=armv8.2-a+dotprod"
            ),
            ASIMDFHM = dict(
                flags="-march=armv8.2-a+fp16fml"
            ),
        )
        if self.cc_on_armhf and is_unix: return dict(
            NEON = dict(
                flags="-mfpu=neon"
            ),
            NEON_FP16 = dict(
                flags="-mfpu=neon-fp16 -mfp16-format=ieee"
            ),
            NEON_VFPV4 = dict(
                flags="-mfpu=neon-vfpv4",
            ),
            ASIMD = dict(
                flags="-mfpu=neon-fp-armv8 -march=armv8-a+simd",
            ),
            ASIMDHP = dict(
                flags="-march=armv8.2-a+fp16"
            ),
            ASIMDDP = dict(
                flags="-march=armv8.2-a+dotprod",
            ),
            ASIMDFHM = dict(
                flags="-march=armv8.2-a+fp16fml"
            )
        )
        # TODO: ARM MSVC
        return {}

    def __init__(self):
        if self.conf_tmp_path is None:
            import shutil
            import tempfile
            tmp = tempfile.mkdtemp()
            def rm_temp():
                try:
                    shutil.rmtree(tmp)
                except OSError:
                    pass
            atexit.register(rm_temp)
            self.conf_tmp_path = tmp

        if self.conf_cache_factors is None:
            self.conf_cache_factors = [
                os.path.getmtime(__file__),
                self.conf_nocache
            ]

class _Distutils:
    """A helper class that provides a collection of fundamental methods
    implemented in a top of Python and NumPy Distutils.

    The idea behind this class is to gather all methods that it may
    need to override in case of reuse 'CCompilerOpt' in environment
    different than of what NumPy has.

    Parameters
    ----------
    ccompiler : `CCompiler`
        The generate instance that returned from `distutils.ccompiler.new_compiler()`.
    """
    def __init__(self, ccompiler):
        self._ccompiler = ccompiler

    def dist_compile(self, sources, flags, ccompiler=None, **kwargs):
        """Wrap CCompiler.compile()"""
        assert(isinstance(sources, list))
        assert(isinstance(flags, list))
        flags = kwargs.pop("extra_postargs", []) + flags
        if not ccompiler:
            ccompiler = self._ccompiler

        return ccompiler.compile(sources, extra_postargs=flags, **kwargs)

    def dist_test(self, source, flags, macros=[]):
        """Return True if 'CCompiler.compile()' able to compile
        a source file with certain flags.
        """
        assert(isinstance(source, str))
        from distutils.errors import CompileError
        cc = self._ccompiler;
        bk_spawn = getattr(cc, 'spawn', None)
        if bk_spawn:
            cc_type = getattr(self._ccompiler, "compiler_type", "")
            if cc_type in ("msvc",):
                setattr(cc, 'spawn', self._dist_test_spawn_paths)
            else:
                setattr(cc, 'spawn', self._dist_test_spawn)
        test = False
        try:
            self.dist_compile(
                [source], flags, macros=macros, output_dir=self.conf_tmp_path
            )
            test = True
        except CompileError as e:
            self.dist_log(str(e), stderr=True)
        if bk_spawn:
            setattr(cc, 'spawn', bk_spawn)
        return test

    def dist_info(self):
        """
        Return a tuple containing info about (platform, compiler, extra_args),
        required by the abstract class '_CCompiler' for discovering the
        platform environment. This is also used as a cache factor in order
        to detect any changes happening from outside.
        """
        if hasattr(self, "_dist_info"):
            return self._dist_info

        cc_type = getattr(self._ccompiler, "compiler_type", '')
        if cc_type in ("intelem", "intelemw"):
            platform = "x86_64"
        elif cc_type in ("intel", "intelw", "intele"):
            platform = "x86"
        else:
            from distutils.util import get_platform
            platform = get_platform()

        cc_info = getattr(self._ccompiler, "compiler", getattr(self._ccompiler, "compiler_so", ''))
        if not cc_type or cc_type == "unix":
            if hasattr(cc_info, "__iter__"):
                compiler = cc_info[0]
            else:
                compiler = str(cc_info)
        else:
            compiler = cc_type

        if hasattr(cc_info, "__iter__") and len(cc_info) > 1:
            extra_args = ' '.join(cc_info[1:])
        else:
            extra_args  = os.environ.get("CFLAGS", "")
            extra_args += os.environ.get("CPPFLAGS", "")

        self._dist_info = (platform, compiler, extra_args)
        return self._dist_info

    @staticmethod
    def dist_error(*args):
        """Raise a compiler error"""
        from distutils.errors import CompileError
        raise CompileError(_Distutils._dist_str(*args))

    @staticmethod
    def dist_fatal(*args):
        """Raise a distutils error"""
        from distutils.errors import DistutilsError
        raise DistutilsError(_Distutils._dist_str(*args))

    @staticmethod
    def dist_log(*args, stderr=False):
        """Print a console message"""
        from numpy.distutils import log
        out = _Distutils._dist_str(*args)
        if stderr:
            log.warn(out)
        else:
            log.info(out)

    @staticmethod
    def dist_load_module(name, path):
        """Load a module from file, required by the abstract class '_Cache'."""
        from .misc_util import exec_mod_from_location
        try:
            return exec_mod_from_location(name, path)
        except Exception as e:
            _Distutils.dist_log(e, stderr=True)
        return None

    @staticmethod
    def _dist_str(*args):
        """Return a string to print by log and errors."""
        def to_str(arg):
            if not isinstance(arg, str) and hasattr(arg, '__iter__'):
                ret = []
                for a in arg:
                    ret.append(to_str(a))
                return '('+ ' '.join(ret) + ')'
            return str(arg)

        stack = inspect.stack()[2]
        start = "CCompilerOpt.%s[%d] : " % (stack.function, stack.lineno)
        out = ' '.join([
            to_str(a)
            for a in (*args,)
        ])
        return start + out

    def _dist_test_spawn_paths(self, cmd, display=None):
        """
        Fix msvc SDK ENV path same as distutils do
        without it we get c1: fatal error C1356: unable to find mspdbcore.dll
        """
        if not hasattr(self._ccompiler, "_paths"):
            self._dist_test_spawn(cmd)
            return
        old_path = os.getenv("path")
        try:
            os.environ["path"] = self._ccompiler._paths
            self._dist_test_spawn(cmd)
        finally:
            os.environ["path"] = old_path

    _dist_warn_regex = re.compile(
        # intel and msvc compilers don't raise
        # fatal errors when flags are wrong or unsupported
        ".*("
        "warning D9002|"  # msvc, it should be work with any language.
        "invalid argument for option" # intel
        ").*"
    )
    @staticmethod
    def _dist_test_spawn(cmd, display=None):
        try:
            o = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                        text=True)
            if o and re.match(_Distutils._dist_warn_regex, o):
                _Distutils.dist_error(
                    "Flags in command", cmd ,"aren't supported by the compiler"
                    ", output -> \n%s" % o
                )
        except subprocess.CalledProcessError as exc:
            o = exc.output
            s = exc.returncode
        except OSError as e:
            o = e
            s = 127
        else:
            return None
        _Distutils.dist_error(
            "Command", cmd, "failed with exit status %d output -> \n%s" % (
            s, o
        ))

_share_cache = {}
class _Cache:
    """An abstract class handles caching functionality, provides two
    levels of caching, in-memory by share instances attributes among
    each other and by store attributes into files.

    **Note**:
        any attributes that start with ``_`` or ``conf_`` will be ignored.

    Parameters
    ----------
    cache_path : str or None
        The path of cache file, if None then cache in file will disabled.

    *factors :
        The caching factors that need to utilize next to `conf_cache_factors`.

    Attributes
    ----------
    cache_private : set
        Hold the attributes that need be skipped from "in-memory cache".

    cache_infile : bool
        Utilized during initializing this class, to determine if the cache was able
        to loaded from the specified cache path in 'cache_path'.
    """

    # skip attributes from cache
    _cache_ignore = re.compile("^(_|conf_)")

    def __init__(self, cache_path=None, *factors):
        self.cache_me = {}
        self.cache_private = set()
        self.cache_infile = False
        self._cache_path = None

        if self.conf_nocache:
            self.dist_log("cache is disabled by `Config`")
            return

        self._cache_hash = self.cache_hash(*factors, *self.conf_cache_factors)
        self._cache_path = cache_path
        if cache_path:
            if os.path.exists(cache_path):
                self.dist_log("load cache from file ->", cache_path)
                cache_mod = self.dist_load_module("cache", cache_path)
                if not cache_mod:
                    self.dist_log(
                        "unable to load the cache file as a module",
                        stderr=True
                    )
                elif not hasattr(cache_mod, "hash") or \
                     not hasattr(cache_mod, "data"):
                    self.dist_log("invalid cache file", stderr=True)
                elif self._cache_hash == cache_mod.hash:
                    self.dist_log("hit the file cache")
                    for attr, val in cache_mod.data.items():
                        setattr(self, attr, val)
                    self.cache_infile = True
                else:
                    self.dist_log("miss the file cache")

        if not self.cache_infile:
            other_cache = _share_cache.get(self._cache_hash)
            if other_cache:
                self.dist_log("hit the memory cache")
                for attr, val in other_cache.__dict__.items():
                    if attr in other_cache.cache_private or \
                               re.match(self._cache_ignore, attr):
                        continue
                    setattr(self, attr, val)

        _share_cache[self._cache_hash] = self
        atexit.register(self.cache_flush)

    def __del__(self):
        for h, o in _share_cache.items():
            if o == self:
                _share_cache.pop(h)
                break

    def cache_flush(self):
        """
        Force update the cache.
        """
        if not self._cache_path:
            return
        # TODO: don't write if the cache doesn't change
        self.dist_log("write cache to path ->", self._cache_path)
        cdict = self.__dict__.copy()
        for attr in self.__dict__.keys():
            if re.match(self._cache_ignore, attr):
                cdict.pop(attr)

        d = os.path.dirname(self._cache_path)
        if not os.path.exists(d):
            os.makedirs(d)

        repr_dict = pprint.pformat(cdict, compact=True)
        with open(self._cache_path, "w") as f:
            f.write(textwrap.dedent("""\
            # AUTOGENERATED DON'T EDIT
            # Please make changes to the code generator \
            (distutils/ccompiler_opt.py)
            hash = {}
            data = \\
            """).format(self._cache_hash))
            f.write(repr_dict)

    def cache_hash(self, *factors):
        # is there a built-in non-crypto hash?
        # sdbm
        chash = 0
        for f in factors:
            for char in str(f):
                chash  = ord(char) + (chash << 6) + (chash << 16) - chash
                chash &= 0xFFFFFFFF
        return chash

    @staticmethod
    def me(cb):
        """
        A static method that can be treated as a decorator to
        dynamically cache certain methods.
        """
        def cache_wrap_me(self, *args, **kwargs):
            # good for normal args
            cache_key = str((
                cb.__name__, *args, *kwargs.keys(), *kwargs.values()
            ))
            if cache_key in self.cache_me:
                return self.cache_me[cache_key]
            ccb = cb(self, *args, **kwargs)
            self.cache_me[cache_key] = ccb
            return ccb
        return cache_wrap_me

class _CCompiler:
    """A helper class for `CCompilerOpt` containing all utilities that
    related to the fundamental compiler's functions.

    Attributes
    ----------
    cc_on_x86 : bool
        True when the target architecture is 32-bit x86
    cc_on_x64 : bool
        True when the target architecture is 64-bit x86
    cc_on_ppc64 : bool
        True when the target architecture is 64-bit big-endian powerpc
    cc_on_ppc64le : bool
        True when the target architecture is 64-bit litle-endian powerpc
    cc_on_s390x : bool
        True when the target architecture is IBM/ZARCH on linux
    cc_on_armhf : bool
        True when the target architecture is 32-bit ARMv7+
    cc_on_aarch64 : bool
        True when the target architecture is 64-bit Armv8-a+
    cc_on_noarch : bool
        True when the target architecture is unknown or not supported
    cc_is_gcc : bool
        True if the compiler is GNU or
        if the compiler is unknown
    cc_is_clang : bool
        True if the compiler is Clang
    cc_is_icc : bool
        True if the compiler is Intel compiler (unix like)
    cc_is_iccw : bool
        True if the compiler is Intel compiler (msvc like)
    cc_is_nocc : bool
        True if the compiler isn't supported directly,
        Note: that cause a fail-back to gcc
    cc_has_debug : bool
        True if the compiler has debug flags
    cc_has_native : bool
        True if the compiler has native flags
    cc_noopt : bool
        True if the compiler has definition 'DISABLE_OPT*',
        or 'cc_on_noarch' is True
    cc_march : str
        The target architecture name, or "unknown" if
        the architecture isn't supported
    cc_name : str
        The compiler name, or "unknown" if the compiler isn't supported
    cc_flags : dict
        Dictionary containing the initialized flags of `_Config.conf_cc_flags`
    """
    def __init__(self):
        if hasattr(self, "cc_is_cached"):
            return
        #      attr            regex        compiler-expression
        detect_arch = (
            ("cc_on_x64",      ".*(x|x86_|amd)64.*", ""),
            ("cc_on_x86",      ".*(win32|x86|i386|i686).*", ""),
            ("cc_on_ppc64le",  ".*(powerpc|ppc)64(el|le).*|.*powerpc.*",
                                          "defined(__powerpc64__) && "
                                          "defined(__LITTLE_ENDIAN__)"),
            ("cc_on_ppc64",    ".*(powerpc|ppc).*|.*powerpc.*",
                                          "defined(__powerpc64__) && "
                                          "defined(__BIG_ENDIAN__)"),
            ("cc_on_aarch64",  ".*(aarch64|arm64).*", ""),
            ("cc_on_armhf",    ".*arm.*", "defined(__ARM_ARCH_7__) || "
                                          "defined(__ARM_ARCH_7A__)"),
            ("cc_on_s390x",    ".*s390x.*", ""),
            # undefined platform
            ("cc_on_noarch",   "", ""),
        )
        detect_compiler = (
            ("cc_is_gcc",     r".*(gcc|gnu\-g).*", ""),
            ("cc_is_clang",    ".*clang.*", ""),
            # intel msvc like
            ("cc_is_iccw",     ".*(intelw|intelemw|iccw).*", ""),
            ("cc_is_icc",      ".*(intel|icc).*", ""),  # intel unix like
            ("cc_is_msvc",     ".*msvc.*", ""),
            ("cc_is_fcc",     ".*fcc.*", ""),
            # undefined compiler will be treat it as gcc
            ("cc_is_nocc",     "", ""),
        )
        detect_args = (
           ("cc_has_debug",  ".*(O0|Od|ggdb|coverage|debug:full).*", ""),
           ("cc_has_native",
                ".*(-march=native|-xHost|/QxHost|-mcpu=a64fx).*", ""),
           # in case if the class run with -DNPY_DISABLE_OPTIMIZATION
           ("cc_noopt", ".*DISABLE_OPT.*", ""),
        )

        dist_info = self.dist_info()
        platform, compiler_info, extra_args = dist_info
        # set False to all attrs
        for section in (detect_arch, detect_compiler, detect_args):
            for attr, rgex, cexpr in section:
                setattr(self, attr, False)

        for detect, searchin in ((detect_arch, platform), (detect_compiler, compiler_info)):
            for attr, rgex, cexpr in detect:
                if rgex and not re.match(rgex, searchin, re.IGNORECASE):
                    continue
                if cexpr and not self.cc_test_cexpr(cexpr):
                    continue
                setattr(self, attr, True)
                break

        for attr, rgex, cexpr in detect_args:
            if rgex and not re.match(rgex, extra_args, re.IGNORECASE):
                continue
            if cexpr and not self.cc_test_cexpr(cexpr):
                continue
            setattr(self, attr, True)

        if self.cc_on_noarch:
            self.dist_log(
                "unable to detect CPU architecture which lead to disable the optimization. "
                f"check dist_info:<<\n{dist_info}\n>>",
                stderr=True
            )
            self.cc_noopt = True

        if self.conf_noopt:
            self.dist_log("Optimization is disabled by the Config", stderr=True)
            self.cc_noopt = True

        if self.cc_is_nocc:
            """
            mingw can be treated as a gcc, and also xlc even if it based on clang,
            but still has the same gcc optimization flags.
            """
            self.dist_log(
                "unable to detect compiler type which leads to treating it as GCC. "
                "this is a normal behavior if you're using gcc-like compiler such as MinGW or IBM/XLC."
                f"check dist_info:<<\n{dist_info}\n>>",
                stderr=True
            )
            self.cc_is_gcc = True

        self.cc_march = "unknown"
        for arch in ("x86", "x64", "ppc64", "ppc64le",
                     "armhf", "aarch64", "s390x"):
            if getattr(self, "cc_on_" + arch):
                self.cc_march = arch
                break

        self.cc_name = "unknown"
        for name in ("gcc", "clang", "iccw", "icc", "msvc", "fcc"):
            if getattr(self, "cc_is_" + name):
                self.cc_name = name
                break

        self.cc_flags = {}
        compiler_flags = self.conf_cc_flags.get(self.cc_name)
        if compiler_flags is None:
            self.dist_fatal(
                "undefined flag for compiler '%s', "
                "leave an empty dict instead" % self.cc_name
            )
        for name, flags in compiler_flags.items():
            self.cc_flags[name] = nflags = []
            if flags:
                assert(isinstance(flags, str))
                flags = flags.split()
                for f in flags:
                    if self.cc_test_flags([f]):
                        nflags.append(f)

        self.cc_is_cached = True

    @_Cache.me
    def cc_test_flags(self, flags):
        """
        Returns True if the compiler supports 'flags'.
        """
        assert(isinstance(flags, list))
        self.dist_log("testing flags", flags)
        test_path = os.path.join(self.conf_check_path, "test_flags.c")
        test = self.dist_test(test_path, flags)
        if not test:
            self.dist_log("testing failed", stderr=True)
        return test

    @_Cache.me
    def cc_test_cexpr(self, cexpr, flags=[]):
        """
        Same as the above but supports compile-time expressions.
        """
        self.dist_log("testing compiler expression", cexpr)
        test_path = os.path.join(self.conf_tmp_path, "npy_dist_test_cexpr.c")
        with open(test_path, "w") as fd:
            fd.write(textwrap.dedent(f"""\
               #if !({cexpr})
                   #error "unsupported expression"
               #endif
               int dummy;
            """))
        test = self.dist_test(test_path, flags)
        if not test:
            self.dist_log("testing failed", stderr=True)
        return test

    def cc_normalize_flags(self, flags):
        """
        Remove the conflicts that caused due gathering implied features flags.

        Parameters
        ----------
        'flags' list, compiler flags
            flags should be sorted from the lowest to the highest interest.

        Returns
        -------
        list, filtered from any conflicts.

        Examples
        --------
        >>> self.cc_normalize_flags(['-march=armv8.2-a+fp16', '-march=armv8.2-a+dotprod'])
        ['armv8.2-a+fp16+dotprod']

        >>> self.cc_normalize_flags(
            ['-msse', '-msse2', '-msse3', '-mssse3', '-msse4.1', '-msse4.2', '-mavx', '-march=core-avx2']
        )
        ['-march=core-avx2']
        """
        assert(isinstance(flags, list))
        if self.cc_is_gcc or self.cc_is_clang or self.cc_is_icc:
            return self._cc_normalize_unix(flags)

        if self.cc_is_msvc or self.cc_is_iccw:
            return self._cc_normalize_win(flags)
        return flags

    _cc_normalize_unix_mrgx = re.compile(
        # 1- to check the highest of
        r"^(-mcpu=|-march=|-x[A-Z0-9\-])"
    )
    _cc_normalize_unix_frgx = re.compile(
        # 2- to remove any flags starts with
        # -march, -mcpu, -x(INTEL) and '-m' without '='
        r"^(?!(-mcpu=|-march=|-x[A-Z0-9\-]|-m[a-z0-9\-\.]*.$))|"
        # exclude:
        r"(?:-mzvector)"
    )
    _cc_normalize_unix_krgx = re.compile(
        # 3- keep only the highest of
        r"^(-mfpu|-mtune)"
    )
    _cc_normalize_arch_ver = re.compile(
        r"[0-9.]"
    )
    def _cc_normalize_unix(self, flags):
        def ver_flags(f):
            #        arch ver  subflag
            # -march=armv8.2-a+fp16fml
            tokens = f.split('+')
            ver = float('0' + ''.join(
                re.findall(self._cc_normalize_arch_ver, tokens[0])
            ))
            return ver, tokens[0], tokens[1:]

        if len(flags) <= 1:
            return flags
        # get the highest matched flag
        for i, cur_flag in enumerate(reversed(flags)):
            if not re.match(self._cc_normalize_unix_mrgx, cur_flag):
                continue
            lower_flags = flags[:-(i+1)]
            upper_flags = flags[-i:]
            filtered = list(filter(
                self._cc_normalize_unix_frgx.search, lower_flags
            ))
            # gather subflags
            ver, arch, subflags = ver_flags(cur_flag)
            if ver > 0 and len(subflags) > 0:
                for xflag in lower_flags:
                    xver, _, xsubflags = ver_flags(xflag)
                    if ver == xver:
                        subflags = xsubflags + subflags
                cur_flag = arch + '+' + '+'.join(subflags)

            flags = filtered + [cur_flag]
            if i > 0:
                flags += upper_flags
            break

        # to remove overridable flags
        final_flags = []
        matched = set()
        for f in reversed(flags):
            match = re.match(self._cc_normalize_unix_krgx, f)
            if not match:
                pass
            elif match[0] in matched:
                continue
            else:
                matched.add(match[0])
            final_flags.insert(0, f)
        return final_flags

    _cc_normalize_win_frgx = re.compile(
        r"^(?!(/arch\:|/Qx\:))"
    )
    _cc_normalize_win_mrgx = re.compile(
        r"^(/arch|/Qx:)"
    )
    def _cc_normalize_win(self, flags):
        for i, f in enumerate(reversed(flags)):
            if not re.match(self._cc_normalize_win_mrgx, f):
                continue
            i += 1
            return list(filter(
                self._cc_normalize_win_frgx.search, flags[:-i]
            )) + flags[-i:]
        return flags

class _Feature:
    """A helper class for `CCompilerOpt` that managing CPU features.

    Attributes
    ----------
    feature_supported : dict
        Dictionary containing all CPU features that supported
        by the platform, according to the specified values in attribute
        `_Config.conf_features` and `_Config.conf_features_partial()`

    feature_min : set
        The minimum support of CPU features, according to
        the specified values in attribute `_Config.conf_min_features`.
    """
    def __init__(self):
        if hasattr(self, "feature_is_cached"):
            return
        self.feature_supported = pfeatures = self.conf_features_partial()
        for feature_name in list(pfeatures.keys()):
            feature  = pfeatures[feature_name]
            cfeature = self.conf_features[feature_name]
            feature.update({
                k:v for k,v in cfeature.items() if k not in feature
            })
            disabled = feature.get("disable")
            if disabled is not None:
                pfeatures.pop(feature_name)
                self.dist_log(
                    "feature '%s' is disabled," % feature_name,
                    disabled, stderr=True
                )
                continue
            # list is used internally for these options
            for option in (
                "implies", "group", "detect", "headers", "flags", "extra_checks"
            ) :
                oval = feature.get(option)
                if isinstance(oval, str):
                    feature[option] = oval.split()

        self.feature_min = set()
        min_f = self.conf_min_features.get(self.cc_march, "")
        for F in min_f.upper().split():
            if F in self.feature_supported:
                self.feature_min.add(F)

        self.feature_is_cached = True

    def feature_names(self, names=None, force_flags=None, macros=[]):
        """
        Returns a set of CPU feature names that supported by platform and the **C** compiler.

        Parameters
        ----------
        names : sequence or None, optional
            Specify certain CPU features to test it against the **C** compiler.
            if None(default), it will test all current supported features.
            **Note**: feature names must be in upper-case.

        force_flags : list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during the test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        assert(
            names is None or (
                not isinstance(names, str) and
                hasattr(names, "__iter__")
            )
        )
        assert(force_flags is None or isinstance(force_flags, list))
        if names is None:
            names = self.feature_supported.keys()
        supported_names = set()
        for f in names:
            if self.feature_is_supported(
                f, force_flags=force_flags, macros=macros
            ):
                supported_names.add(f)
        return supported_names

    def feature_is_exist(self, name):
        """
        Returns True if a certain feature is exist and covered within
        ``_Config.conf_features``.

        Parameters
        ----------
        'name': str
            feature name in uppercase.
        """
        assert(name.isupper())
        return name in self.conf_features

    def feature_sorted(self, names, reverse=False):
        """
        Sort a list of CPU features ordered by the lowest interest.

        Parameters
        ----------
        'names': sequence
            sequence of supported feature names in uppercase.
        'reverse': bool, optional
            If true, the sorted features is reversed. (highest interest)

        Returns
        -------
        list, sorted CPU features
        """
        def sort_cb(k):
            if isinstance(k, str):
                return self.feature_supported[k]["interest"]
            # multiple features
            rank = max([self.feature_supported[f]["interest"] for f in k])
            # FIXME: that's not a safe way to increase the rank for
            # multi targets
            rank += len(k) -1
            return rank
        return sorted(names, reverse=reverse, key=sort_cb)

    def feature_implies(self, names, keep_origins=False):
        """
        Return a set of CPU features that implied by 'names'

        Parameters
        ----------
        names : str or sequence of str
            CPU feature name(s) in uppercase.

        keep_origins : bool
            if False(default) then the returned set will not contain any
            features from 'names'. This case happens only when two features
            imply each other.

        Examples
        --------
        >>> self.feature_implies("SSE3")
        {'SSE', 'SSE2'}
        >>> self.feature_implies("SSE2")
        {'SSE'}
        >>> self.feature_implies("SSE2", keep_origins=True)
        # 'SSE2' found here since 'SSE' and 'SSE2' imply each other
        {'SSE', 'SSE2'}
        """
        def get_implies(name, _caller=set()):
            implies = set()
            d = self.feature_supported[name]
            for i in d.get("implies", []):
                implies.add(i)
                if i in _caller:
                    # infinity recursive guard since
                    # features can imply each other
                    continue
                _caller.add(name)
                implies = implies.union(get_implies(i, _caller))
            return implies

        if isinstance(names, str):
            implies = get_implies(names)
            names = [names]
        else:
            assert(hasattr(names, "__iter__"))
            implies = set()
            for n in names:
                implies = implies.union(get_implies(n))
        if not keep_origins:
            implies.difference_update(names)
        return implies

    def feature_implies_c(self, names):
        """same as feature_implies() but combining 'names'"""
        if isinstance(names, str):
            names = set((names,))
        else:
            names = set(names)
        return names.union(self.feature_implies(names))

    def feature_ahead(self, names):
        """
        Return list of features in 'names' after remove any
        implied features and keep the origins.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41"])
        ["SSE41"]
        # assume AVX2 and FMA3 implies each other and AVX2
        # is the highest interest
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2"]
        # assume AVX2 and FMA3 don't implies each other
        >>> self.feature_ahead(["SSE2", "SSE3", "SSE41", "AVX2", "FMA3"])
        ["AVX2", "FMA3"]
        """
        assert(
            not isinstance(names, str)
            and hasattr(names, '__iter__')
        )
        implies = self.feature_implies(names, keep_origins=True)
        ahead = [n for n in names if n not in implies]
        if len(ahead) == 0:
            # return the highest interested feature
            # if all features imply each other
            ahead = self.feature_sorted(names, reverse=True)[:1]
        return ahead

    def feature_untied(self, names):
        """
        same as 'feature_ahead()' but if both features implied each other
        and keep the highest interest.

        Parameters
        ----------
        'names': sequence
            sequence of CPU feature names in uppercase.

        Returns
        -------
        list of CPU features sorted as-is 'names'

        Examples
        --------
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41"])
        ["SSE2", "SSE3", "SSE41"]
        # assume AVX2 and FMA3 implies each other
        >>> self.feature_untied(["SSE2", "SSE3", "SSE41", "FMA3", "AVX2"])
        ["SSE2", "SSE3", "SSE41", "AVX2"]
        """
        assert(
            not isinstance(names, str)
            and hasattr(names, '__iter__')
        )
        final = []
        for n in names:
            implies = self.feature_implies(n)
            tied = [
                nn for nn in final
                if nn in implies and n in self.feature_implies(nn)
            ]
            if tied:
                tied = self.feature_sorted(tied + [n])
                if n not in tied[1:]:
                    continue
                final.remove(tied[:1][0])
            final.append(n)
        return final

    def feature_get_til(self, names, keyisfalse):
        """
        same as `feature_implies_c()` but stop collecting implied
        features when feature's option that provided through
        parameter 'keyisfalse' is False, also sorting the returned
        features.
        """
        def til(tnames):
            # sort from highest to lowest interest then cut if "key" is False
            tnames = self.feature_implies_c(tnames)
            tnames = self.feature_sorted(tnames, reverse=True)
            for i, n in enumerate(tnames):
                if not self.feature_supported[n].get(keyisfalse, True):
                    tnames = tnames[:i+1]
                    break
            return tnames

        if isinstance(names, str) or len(names) <= 1:
            names = til(names)
            # normalize the sort
            names.reverse()
            return names

        names = self.feature_ahead(names)
        names = {t for n in names for t in til(n)}
        return self.feature_sorted(names)

    def feature_detect(self, names):
        """
        Return a list of CPU features that required to be detected
        sorted from the lowest to highest interest.
        """
        names = self.feature_get_til(names, "implies_detect")
        detect = []
        for n in names:
            d = self.feature_supported[n]
            detect += d.get("detect", d.get("group", [n]))
        return detect

    @_Cache.me
    def feature_flags(self, names):
        """
        Return a list of CPU features flags sorted from the lowest
        to highest interest.
        """
        names = self.feature_sorted(self.feature_implies_c(names))
        flags = []
        for n in names:
            d = self.feature_supported[n]
            f = d.get("flags", [])
            if not f or not self.cc_test_flags(f):
                continue
            flags += f
        return self.cc_normalize_flags(flags)

    @_Cache.me
    def feature_test(self, name, force_flags=None, macros=[]):
        """
        Test a certain CPU feature against the compiler through its own
        check file.

        Parameters
        ----------
        name : str
            Supported CPU feature name.

        force_flags : list or None, optional
            If None(default), the returned flags from `feature_flags()`
            will be used.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        if force_flags is None:
            force_flags = self.feature_flags(name)

        self.dist_log(
            "testing feature '%s' with flags (%s)" % (
            name, ' '.join(force_flags)
        ))
        # Each CPU feature must have C source code contains at
        # least one intrinsic or instruction related to this feature.
        test_path = os.path.join(
            self.conf_check_path, "cpu_%s.c" % name.lower()
        )
        if not os.path.exists(test_path):
            self.dist_fatal("feature test file is not exist", test_path)

        test = self.dist_test(
            test_path, force_flags + self.cc_flags["werror"], macros=macros
        )
        if not test:
            self.dist_log("testing failed", stderr=True)
        return test

    @_Cache.me
    def feature_is_supported(self, name, force_flags=None, macros=[]):
        """
        Check if a certain CPU feature is supported by the platform and compiler.

        Parameters
        ----------
        name : str
            CPU feature name in uppercase.

        force_flags : list or None, optional
            If None(default), default compiler flags for every CPU feature will
            be used during test.

        macros : list of tuples, optional
            A list of C macro definitions.
        """
        assert(name.isupper())
        assert(force_flags is None or isinstance(force_flags, list))

        supported = name in self.feature_supported
        if supported:
            for impl in self.feature_implies(name):
                if not self.feature_test(impl, force_flags, macros=macros):
                    return False
            if not self.feature_test(name, force_flags, macros=macros):
                return False
        return supported

    @_Cache.me
    def feature_can_autovec(self, name):
        """
        check if the feature can be auto-vectorized by the compiler
        """
        assert(isinstance(name, str))
        d = self.feature_supported[name]
        can = d.get("autovec", None)
        if can is None:
            valid_flags = [
                self.cc_test_flags([f]) for f in d.get("flags", [])
            ]
            can = valid_flags and any(valid_flags)
        return can

    @_Cache.me
    def feature_extra_checks(self, name):
        """
        Return a list of supported extra checks after testing them against
        the compiler.

        Parameters
        ----------
        names : str
            CPU feature name in uppercase.
        """
        assert isinstance(name, str)
        d = self.feature_supported[name]
        extra_checks = d.get("extra_checks", [])
        if not extra_checks:
            return []

        self.dist_log("Testing extra checks for feature '%s'" % name, extra_checks)
        flags = self.feature_flags(name)
        available = []
        not_available = []
        for chk in extra_checks:
            test_path = os.path.join(
                self.conf_check_path, "extra_%s.c" % chk.lower()
            )
            if not os.path.exists(test_path):
                self.dist_fatal("extra check file does not exist", test_path)

            is_supported = self.dist_test(test_path, flags + self.cc_flags["werror"])
            if is_supported:
                available.append(chk)
            else:
                not_available.append(chk)

        if not_available:
            self.dist_log("testing failed for checks", not_available, stderr=True)
        return available


    def feature_c_preprocessor(self, feature_name, tabs=0):
        """
        Generate C preprocessor definitions and include headers of a CPU feature.

        Parameters
        ----------
        'feature_name': str
            CPU feature name in uppercase.
        'tabs': int
            if > 0, align the generated strings to the right depend on number of tabs.

        Returns
        -------
        str, generated C preprocessor

        Examples
        --------
        >>> self.feature_c_preprocessor("SSE3")
        /** SSE3 **/
        #define NPY_HAVE_SSE3 1
        #include <pmmintrin.h>
        """
        assert(feature_name.isupper())
        feature = self.feature_supported.get(feature_name)
        assert(feature is not None)

        prepr = [
            "/** %s **/" % feature_name,
            "#define %sHAVE_%s 1" % (self.conf_c_prefix, feature_name)
        ]
        prepr += [
            "#include <%s>" % h for h in feature.get("headers", [])
        ]

        extra_defs = feature.get("group", [])
        extra_defs += self.feature_extra_checks(feature_name)
        for edef in extra_defs:
            # Guard extra definitions in case of duplicate with
            # another feature
            prepr += [
                "#ifndef %sHAVE_%s" % (self.conf_c_prefix, edef),
                "\t#define %sHAVE_%s 1" % (self.conf_c_prefix, edef),
                "#endif",
            ]

        if tabs > 0:
            prepr = [('\t'*tabs) + l for l in prepr]
        return '\n'.join(prepr)

class _Parse:
    """A helper class that parsing main arguments of `CCompilerOpt`,
    also parsing configuration statements in dispatch-able sources.

    Parameters
    ----------
    cpu_baseline : str or None
        minimal set of required CPU features or special options.

    cpu_dispatch : str or None
        dispatched set of additional CPU features or special options.

    Special options can be:
        - **MIN**: Enables the minimum CPU features that utilized via `_Config.conf_min_features`
        - **MAX**: Enables all supported CPU features by the Compiler and platform.
        - **NATIVE**: Enables all CPU features that supported by the current machine.
        - **NONE**: Enables nothing
        - **Operand +/-**: remove or add features, useful with options **MAX**, **MIN** and **NATIVE**.
            NOTE: operand + is only added for nominal reason.

    NOTES:
        - Case-insensitive among all CPU features and special options.
        - Comma or space can be used as a separator.
        - If the CPU feature is not supported by the user platform or compiler,
          it will be skipped rather than raising a fatal error.
        - Any specified CPU features to 'cpu_dispatch' will be skipped if its part of CPU baseline features
        - 'cpu_baseline' force enables implied features.

    Attributes
    ----------
    parse_baseline_names : list
        Final CPU baseline's feature names(sorted from low to high)
    parse_baseline_flags : list
        Compiler flags of baseline features
    parse_dispatch_names : list
        Final CPU dispatch-able feature names(sorted from low to high)
    parse_target_groups : dict
        Dictionary containing initialized target groups that configured
        through class attribute `conf_target_groups`.

        The key is represent the group name and value is a tuple
        contains three items :
            - bool, True if group has the 'baseline' option.
            - list, list of CPU features.
            - list, list of extra compiler flags.

    """
    def __init__(self, cpu_baseline, cpu_dispatch):
        self._parse_policies = dict(
            # POLICY NAME, (HAVE, NOT HAVE, [DEB])
            KEEP_BASELINE = (
                None, self._parse_policy_not_keepbase,
                []
            ),
            KEEP_SORT = (
                self._parse_policy_keepsort,
                self._parse_policy_not_keepsort,
                []
            ),
            MAXOPT = (
                self._parse_policy_maxopt, None,
                []
            ),
            WERROR = (
                self._parse_policy_werror, None,
                []
            ),
            AUTOVEC = (
                self._parse_policy_autovec, None,
                ["MAXOPT"]
            )
        )
        if hasattr(self, "parse_is_cached"):
            return

        self.parse_baseline_names = []
        self.parse_baseline_flags = []
        self.parse_dispatch_names = []
        self.parse_target_groups = {}

        if self.cc_noopt:
            # skip parsing baseline and dispatch args and keep parsing target groups
            cpu_baseline = cpu_dispatch = None

        self.dist_log("check requested baseline")
        if cpu_baseline is not None:
            cpu_baseline = self._parse_arg_features("cpu_baseline", cpu_baseline)
            baseline_names = self.feature_names(cpu_baseline)
            self.parse_baseline_flags = self.feature_flags(baseline_names)
            self.parse_baseline_names = self.feature_sorted(
                self.feature_implies_c(baseline_names)
            )

        self.dist_log("check requested dispatch-able features")
        if cpu_dispatch is not None:
            cpu_dispatch_ = self._parse_arg_features("cpu_dispatch", cpu_dispatch)
            cpu_dispatch = {
                f for f in cpu_dispatch_
                if f not in self.parse_baseline_names
            }
            conflict_baseline = cpu_dispatch_.difference(cpu_dispatch)
            self.parse_dispatch_names = self.feature_sorted(
                self.feature_names(cpu_dispatch)
            )
            if len(conflict_baseline) > 0:
                self.dist_log(
                    "skip features", conflict_baseline, "since its part of baseline"
                )

        self.dist_log("initialize targets groups")
        for group_name, tokens in self.conf_target_groups.items():
            self.dist_log("parse target group", group_name)
            GROUP_NAME = group_name.upper()
            if not tokens or not tokens.strip():
                # allow empty groups, useful in case if there's a need
                # to disable certain group since '_parse_target_tokens()'
                # requires at least one valid target
                self.parse_target_groups[GROUP_NAME] = (
                    False, [], []
                )
                continue
            has_baseline, features, extra_flags = \
                self._parse_target_tokens(tokens)
            self.parse_target_groups[GROUP_NAME] = (
                has_baseline, features, extra_flags
            )

        self.parse_is_cached = True

    def parse_targets(self, source):
        """
        Fetch and parse configuration statements that required for
        defining the targeted CPU features, statements should be declared
        in the top of source in between **C** comment and start
        with a special mark **@targets**.

        Configuration statements are sort of keywords representing
        CPU features names, group of statements and policies, combined
        together to determine the required optimization.

        Parameters
        ----------
        source : str
            the path of **C** source file.

        Returns
        -------
        - bool, True if group has the 'baseline' option
        - list, list of CPU features
        - list, list of extra compiler flags
        """
        self.dist_log("looking for '@targets' inside -> ", source)
        # get lines between /*@targets and */
        with open(source) as fd:
            tokens = ""
            max_to_reach = 1000 # good enough, isn't?
            start_with = "@targets"
            start_pos = -1
            end_with = "*/"
            end_pos = -1
            for current_line, line in enumerate(fd):
                if current_line == max_to_reach:
                    self.dist_fatal("reached the max of lines")
                    break
                if start_pos == -1:
                    start_pos = line.find(start_with)
                    if start_pos == -1:
                        continue
                    start_pos += len(start_with)
                tokens += line
                end_pos = line.find(end_with)
                if end_pos != -1:
                    end_pos += len(tokens) - len(line)
                    break

        if start_pos == -1:
            self.dist_fatal("expected to find '%s' within a C comment" % start_with)
        if end_pos == -1:
            self.dist_fatal("expected to end with '%s'" % end_with)

        tokens = tokens[start_pos:end_pos]
        return self._parse_target_tokens(tokens)

    _parse_regex_arg = re.compile(r'\s|,|([+-])')
    def _parse_arg_features(self, arg_name, req_features):
        if not isinstance(req_features, str):
            self.dist_fatal("expected a string in '%s'" % arg_name)

        final_features = set()
        # space and comma can be used as a separator
        tokens = list(filter(None, re.split(self._parse_regex_arg, req_features)))
        append = True # append is the default
        for tok in tokens:
            if tok[0] in ("#", "$"):
                self.dist_fatal(
                    arg_name, "target groups and policies "
                    "aren't allowed from arguments, "
                    "only from dispatch-able sources"
                )
            if tok == '+':
                append = True
                continue
            if tok == '-':
                append = False
                continue

            TOK = tok.upper() # we use upper-case internally
            features_to = set()
            if TOK == "NONE":
                pass
            elif TOK == "NATIVE":
                native = self.cc_flags["native"]
                if not native:
                    self.dist_fatal(arg_name,
                        "native option isn't supported by the compiler"
                    )
                features_to = self.feature_names(
                    force_flags=native, macros=[("DETECT_FEATURES", 1)]
                )
            elif TOK == "MAX":
                features_to = self.feature_supported.keys()
            elif TOK == "MIN":
                features_to = self.feature_min
            else:
                if TOK in self.feature_supported:
                    features_to.add(TOK)
                else:
                    if not self.feature_is_exist(TOK):
                        self.dist_fatal(arg_name,
                            ", '%s' isn't a known feature or option" % tok
                        )
            if append:
                final_features = final_features.union(features_to)
            else:
                final_features = final_features.difference(features_to)

            append = True # back to default

        return final_features

    _parse_regex_target = re.compile(r'\s|[*,/]|([()])')
    def _parse_target_tokens(self, tokens):
        assert(isinstance(tokens, str))
        final_targets = [] # to keep it sorted as specified
        extra_flags = []
        has_baseline = False

        skipped  = set()
        policies = set()
        multi_target = None

        tokens = list(filter(None, re.split(self._parse_regex_target, tokens)))
        if not tokens:
            self.dist_fatal("expected one token at least")

        for tok in tokens:
            TOK = tok.upper()
            ch = tok[0]
            if ch in ('+', '-'):
                self.dist_fatal(
                    "+/- are 'not' allowed from target's groups or @targets, "
                    "only from cpu_baseline and cpu_dispatch parms"
                )
            elif ch == '$':
                if multi_target is not None:
                    self.dist_fatal(
                        "policies aren't allowed inside multi-target '()'"
                        ", only CPU features"
                    )
                policies.add(self._parse_token_policy(TOK))
            elif ch == '#':
                if multi_target is not None:
                    self.dist_fatal(
                        "target groups aren't allowed inside multi-target '()'"
                        ", only CPU features"
                    )
                has_baseline, final_targets, extra_flags = \
                self._parse_token_group(TOK, has_baseline, final_targets, extra_flags)
            elif ch == '(':
                if multi_target is not None:
                    self.dist_fatal("unclosed multi-target, missing ')'")
                multi_target = set()
            elif ch == ')':
                if multi_target is None:
                    self.dist_fatal("multi-target opener '(' wasn't found")
                targets = self._parse_multi_target(multi_target)
                if targets is None:
                    skipped.add(tuple(multi_target))
                else:
                    if len(targets) == 1:
                        targets = targets[0]
                    if targets and targets not in final_targets:
                        final_targets.append(targets)
                multi_target = None # back to default
            else:
                if TOK == "BASELINE":
                    if multi_target is not None:
                        self.dist_fatal("baseline isn't allowed inside multi-target '()'")
                    has_baseline = True
                    continue

                if multi_target is not None:
                    multi_target.add(TOK)
                    continue

                if not self.feature_is_exist(TOK):
                    self.dist_fatal("invalid target name '%s'" % TOK)

                is_enabled = (
                    TOK in self.parse_baseline_names or
                    TOK in self.parse_dispatch_names
                )
                if  is_enabled:
                    if TOK not in final_targets:
                        final_targets.append(TOK)
                    continue

                skipped.add(TOK)

        if multi_target is not None:
            self.dist_fatal("unclosed multi-target, missing ')'")
        if skipped:
            self.dist_log(
                "skip targets", skipped,
                "not part of baseline or dispatch-able features"
            )

        final_targets = self.feature_untied(final_targets)

        # add polices dependencies
        for p in list(policies):
            _, _, deps = self._parse_policies[p]
            for d in deps:
                if d in policies:
                    continue
                self.dist_log(
                    "policy '%s' force enables '%s'" % (
                    p, d
                ))
                policies.add(d)

        # release policies filtrations
        for p, (have, nhave, _) in self._parse_policies.items():
            func = None
            if p in policies:
                func = have
                self.dist_log("policy '%s' is ON" % p)
            else:
                func = nhave
            if not func:
                continue
            has_baseline, final_targets, extra_flags = func(
                has_baseline, final_targets, extra_flags
            )

        return has_baseline, final_targets, extra_flags

    def _parse_token_policy(self, token):
        """validate policy token"""
        if len(token) <= 1 or token[-1:] == token[0]:
            self.dist_fatal("'$' must stuck in the begin of policy name")
        token = token[1:]
        if token not in self._parse_policies:
            self.dist_fatal(
                "'%s' is an invalid policy name, available policies are" % token,
                self._parse_policies.keys()
            )
        return token

    def _parse_token_group(self, token, has_baseline, final_targets, extra_flags):
        """validate group token"""
        if len(token) <= 1 or token[-1:] == token[0]:
            self.dist_fatal("'#' must stuck in the begin of group name")

        token = token[1:]
        ghas_baseline, gtargets, gextra_flags = self.parse_target_groups.get(
            token, (False, None, [])
        )
        if gtargets is None:
            self.dist_fatal(
                "'%s' is an invalid target group name, " % token + \
                "available target groups are",
                self.parse_target_groups.keys()
            )
        if ghas_baseline:
            has_baseline = True
        # always keep sorting as specified
        final_targets += [f for f in gtargets if f not in final_targets]
        extra_flags += [f for f in gextra_flags if f not in extra_flags]
        return has_baseline, final_targets, extra_flags

    def _parse_multi_target(self, targets):
        """validate multi targets that defined between parentheses()"""
        # remove any implied features and keep the origins
        if not targets:
            self.dist_fatal("empty multi-target '()'")
        if not all([
            self.feature_is_exist(tar) for tar in targets
        ]) :
            self.dist_fatal("invalid target name in multi-target", targets)
        if not all([
            (
                tar in self.parse_baseline_names or
                tar in self.parse_dispatch_names
            )
            for tar in targets
        ]) :
            return None
        targets = self.feature_ahead(targets)
        if not targets:
            return None
        # force sort multi targets, so it can be comparable
        targets = self.feature_sorted(targets)
        targets = tuple(targets) # hashable
        return targets

    def _parse_policy_not_keepbase(self, has_baseline, final_targets, extra_flags):
        """skip all baseline features"""
        skipped = []
        for tar in final_targets[:]:
            is_base = False
            if isinstance(tar, str):
                is_base = tar in self.parse_baseline_names
            else:
                # multi targets
                is_base = all([
                    f in self.parse_baseline_names
                    for f in tar
                ])
            if is_base:
                skipped.append(tar)
                final_targets.remove(tar)

        if skipped:
            self.dist_log("skip baseline features", skipped)

        return has_baseline, final_targets, extra_flags

    def _parse_policy_keepsort(self, has_baseline, final_targets, extra_flags):
        """leave a notice that $keep_sort is on"""
        self.dist_log(
            "policy 'keep_sort' is on, dispatch-able targets", final_targets, "\n"
            "are 'not' sorted depend on the highest interest but"
            "as specified in the dispatch-able source or the extra group"
        )
        return has_baseline, final_targets, extra_flags

    def _parse_policy_not_keepsort(self, has_baseline, final_targets, extra_flags):
        """sorted depend on the highest interest"""
        final_targets = self.feature_sorted(final_targets, reverse=True)
        return has_baseline, final_targets, extra_flags

    def _parse_policy_maxopt(self, has_baseline, final_targets, extra_flags):
        """append the compiler optimization flags"""
        if self.cc_has_debug:
            self.dist_log("debug mode is detected, policy 'maxopt' is skipped.")
        elif self.cc_noopt:
            self.dist_log("optimization is disabled, policy 'maxopt' is skipped.")
        else:
            flags = self.cc_flags["opt"]
            if not flags:
                self.dist_log(
                    "current compiler doesn't support optimization flags, "
                    "policy 'maxopt' is skipped", stderr=True
                )
            else:
                extra_flags += flags
        return has_baseline, final_targets, extra_flags

    def _parse_policy_werror(self, has_baseline, final_targets, extra_flags):
        """force warnings to treated as errors"""
        flags = self.cc_flags["werror"]
        if not flags:
            self.dist_log(
                "current compiler doesn't support werror flags, "
                "warnings will 'not' treated as errors", stderr=True
            )
        else:
            self.dist_log("compiler warnings are treated as errors")
            extra_flags += flags
        return has_baseline, final_targets, extra_flags

    def _parse_policy_autovec(self, has_baseline, final_targets, extra_flags):
        """skip features that has no auto-vectorized support by compiler"""
        skipped = []
        for tar in final_targets[:]:
            if isinstance(tar, str):
                can = self.feature_can_autovec(tar)
            else: # multiple target
                can = all([
                    self.feature_can_autovec(t)
                    for t in tar
                ])
            if not can:
                final_targets.remove(tar)
                skipped.append(tar)

        if skipped:
            self.dist_log("skip non auto-vectorized features", skipped)

        return has_baseline, final_targets, extra_flags

class CCompilerOpt(_Config, _Distutils, _Cache, _CCompiler, _Feature, _Parse):
    """
    A helper class for `CCompiler` aims to provide extra build options
    to effectively control of compiler optimizations that are directly
    related to CPU features.
    """
    def __init__(self, ccompiler, cpu_baseline="min", cpu_dispatch="max", cache_path=None):
        _Config.__init__(self)
        _Distutils.__init__(self, ccompiler)
        _Cache.__init__(self, cache_path, self.dist_info(), cpu_baseline, cpu_dispatch)
        _CCompiler.__init__(self)
        _Feature.__init__(self)
        if not self.cc_noopt and self.cc_has_native:
            self.dist_log(
                "native flag is specified through environment variables. "
                "force cpu-baseline='native'"
            )
            cpu_baseline = "native"
        _Parse.__init__(self, cpu_baseline, cpu_dispatch)
        # keep the requested features untouched, need it later for report
        # and trace purposes
        self._requested_baseline = cpu_baseline
        self._requested_dispatch = cpu_dispatch
        # key is the dispatch-able source and value is a tuple
        # contains two items (has_baseline[boolean], dispatched-features[list])
        self.sources_status = getattr(self, "sources_status", {})
        # every instance should has a separate one
        self.cache_private.add("sources_status")
        # set it at the end to make sure the cache writing was done after init
        # this class
        self.hit_cache = hasattr(self, "hit_cache")

    def is_cached(self):
        """
        Returns True if the class loaded from the cache file
        """
        return self.cache_infile and self.hit_cache

    def cpu_baseline_flags(self):
        """
        Returns a list of final CPU baseline compiler flags
        """
        return self.parse_baseline_flags

    def cpu_baseline_names(self):
        """
        return a list of final CPU baseline feature names
        """
        return self.parse_baseline_names

    def cpu_dispatch_names(self):
        """
        return a list of final CPU dispatch feature names
        """
        return self.parse_dispatch_names

    def try_dispatch(self, sources, src_dir=None, ccompiler=None, **kwargs):
        """
        Compile one or more dispatch-able sources and generates object files,
        also generates abstract C config headers and macros that
        used later for the final runtime dispatching process.

        The mechanism behind it is to takes each source file that specified
        in 'sources' and branching it into several files depend on
        special configuration statements that must be declared in the
        top of each source which contains targeted CPU features,
        then it compiles every branched source with the proper compiler flags.

        Parameters
        ----------
        sources : list
            Must be a list of dispatch-able sources file paths,
            and configuration statements must be declared inside
            each file.

        src_dir : str
            Path of parent directory for the generated headers and wrapped sources.
            If None(default) the files will generated in-place.

        ccompiler : CCompiler
            Distutils `CCompiler` instance to be used for compilation.
            If None (default), the provided instance during the initialization
            will be used instead.

        **kwargs : any
            Arguments to pass on to the `CCompiler.compile()`

        Returns
        -------
        list : generated object files

        Raises
        ------
        CompileError
            Raises by `CCompiler.compile()` on compiling failure.
        DistutilsError
            Some errors during checking the sanity of configuration statements.

        See Also
        --------
        parse_targets :
            Parsing the configuration statements of dispatch-able sources.
        """
        to_compile = {}
        baseline_flags = self.cpu_baseline_flags()
        include_dirs = kwargs.setdefault("include_dirs", [])

        for src in sources:
            output_dir = os.path.dirname(src)
            if src_dir:
                if not output_dir.startswith(src_dir):
                    output_dir = os.path.join(src_dir, output_dir)
                if output_dir not in include_dirs:
                    # To allow including the generated config header(*.dispatch.h)
                    # by the dispatch-able sources
                    include_dirs.append(output_dir)

            has_baseline, targets, extra_flags = self.parse_targets(src)
            nochange = self._generate_config(output_dir, src, targets, has_baseline)
            for tar in targets:
                tar_src = self._wrap_target(output_dir, src, tar, nochange=nochange)
                flags = tuple(extra_flags + self.feature_flags(tar))
                to_compile.setdefault(flags, []).append(tar_src)

            if has_baseline:
                flags = tuple(extra_flags + baseline_flags)
                to_compile.setdefault(flags, []).append(src)

            self.sources_status[src] = (has_baseline, targets)

        # For these reasons, the sources are compiled in a separate loop:
        # - Gathering all sources with the same flags to benefit from
        #   the parallel compiling as much as possible.
        # - To generate all config headers of the dispatchable sources,
        #   before the compilation in case if there are dependency relationships
        #   among them.
        objects = []
        for flags, srcs in to_compile.items():
            objects += self.dist_compile(
                srcs, list(flags), ccompiler=ccompiler, **kwargs
            )
        return objects

    def generate_dispatch_header(self, header_path):
        """
        Generate the dispatch header which contains the #definitions and headers
        for platform-specific instruction-sets for the enabled CPU baseline and
        dispatch-able features.

        Its highly recommended to take a look at the generated header
        also the generated source files via `try_dispatch()`
        in order to get the full picture.
        """
        self.dist_log("generate CPU dispatch header: (%s)" % header_path)

        baseline_names = self.cpu_baseline_names()
        dispatch_names = self.cpu_dispatch_names()
        baseline_len = len(baseline_names)
        dispatch_len = len(dispatch_names)

        header_dir = os.path.dirname(header_path)
        if not os.path.exists(header_dir):
            self.dist_log(
                f"dispatch header dir {header_dir} does not exist, creating it",
                stderr=True
            )
            os.makedirs(header_dir)

        with open(header_path, 'w') as f:
            baseline_calls = ' \\\n'.join([
                (
                    "\t%sWITH_CPU_EXPAND_(MACRO_TO_CALL(%s, __VA_ARGS__))"
                ) % (self.conf_c_prefix, f)
                for f in baseline_names
            ])
            dispatch_calls = ' \\\n'.join([
                (
                    "\t%sWITH_CPU_EXPAND_(MACRO_TO_CALL(%s, __VA_ARGS__))"
                ) % (self.conf_c_prefix, f)
                for f in dispatch_names
            ])
            f.write(textwrap.dedent("""\
                /*
                 * AUTOGENERATED DON'T EDIT
                 * Please make changes to the code generator (distutils/ccompiler_opt.py)
                */
                #define {pfx}WITH_CPU_BASELINE  "{baseline_str}"
                #define {pfx}WITH_CPU_DISPATCH  "{dispatch_str}"
                #define {pfx}WITH_CPU_BASELINE_N {baseline_len}
                #define {pfx}WITH_CPU_DISPATCH_N {dispatch_len}
                #define {pfx}WITH_CPU_EXPAND_(X) X
                #define {pfx}WITH_CPU_BASELINE_CALL(MACRO_TO_CALL, ...) \\
                {baseline_calls}
                #define {pfx}WITH_CPU_DISPATCH_CALL(MACRO_TO_CALL, ...) \\
                {dispatch_calls}
            """).format(
                pfx=self.conf_c_prefix, baseline_str=" ".join(baseline_names),
                dispatch_str=" ".join(dispatch_names), baseline_len=baseline_len,
                dispatch_len=dispatch_len, baseline_calls=baseline_calls,
                dispatch_calls=dispatch_calls
            ))
            baseline_pre = ''
            for name in baseline_names:
                baseline_pre += self.feature_c_preprocessor(name, tabs=1) + '\n'

            dispatch_pre = ''
            for name in dispatch_names:
                dispatch_pre += textwrap.dedent("""\
                #ifdef {pfx}CPU_TARGET_{name}
                {pre}
                #endif /*{pfx}CPU_TARGET_{name}*/
                """).format(
                    pfx=self.conf_c_prefix_, name=name, pre=self.feature_c_preprocessor(
                    name, tabs=1
                ))

            f.write(textwrap.dedent("""\
            /******* baseline features *******/
            {baseline_pre}
            /******* dispatch features *******/
            {dispatch_pre}
            """).format(
                pfx=self.conf_c_prefix_, baseline_pre=baseline_pre,
                dispatch_pre=dispatch_pre
            ))

    def report(self, full=False):
        report = []
        platform_rows = []
        baseline_rows = []
        dispatch_rows = []
        report.append(("Platform", platform_rows))
        report.append(("", ""))
        report.append(("CPU baseline", baseline_rows))
        report.append(("", ""))
        report.append(("CPU dispatch", dispatch_rows))

        ########## platform ##########
        platform_rows.append(("Architecture", (
            "unsupported" if self.cc_on_noarch else self.cc_march)
        ))
        platform_rows.append(("Compiler", (
            "unix-like"   if self.cc_is_nocc   else self.cc_name)
        ))
        ########## baseline ##########
        if self.cc_noopt:
            baseline_rows.append(("Requested", "optimization disabled"))
        else:
            baseline_rows.append(("Requested", repr(self._requested_baseline)))

        baseline_names = self.cpu_baseline_names()
        baseline_rows.append((
            "Enabled", (' '.join(baseline_names) if baseline_names else "none")
        ))
        baseline_flags = self.cpu_baseline_flags()
        baseline_rows.append((
            "Flags", (' '.join(baseline_flags) if baseline_flags else "none")
        ))
        extra_checks = []
        for name in baseline_names:
            extra_checks += self.feature_extra_checks(name)
        baseline_rows.append((
            "Extra checks", (' '.join(extra_checks) if extra_checks else "none")
        ))

        ########## dispatch ##########
        if self.cc_noopt:
            baseline_rows.append(("Requested", "optimization disabled"))
        else:
            dispatch_rows.append(("Requested", repr(self._requested_dispatch)))

        dispatch_names = self.cpu_dispatch_names()
        dispatch_rows.append((
            "Enabled", (' '.join(dispatch_names) if dispatch_names else "none")
        ))
        ########## Generated ##########
        # TODO:
        # - collect object names from 'try_dispatch()'
        #   then get size of each object and printed
        # - give more details about the features that not
        #   generated due compiler support
        # - find a better output's design.
        #
        target_sources = {}
        for source, (_, targets) in self.sources_status.items():
            for tar in targets:
                target_sources.setdefault(tar, []).append(source)

        if not full or not target_sources:
            generated = ""
            for tar in self.feature_sorted(target_sources):
                sources = target_sources[tar]
                name = tar if isinstance(tar, str) else '(%s)' % ' '.join(tar)
                generated += name + "[%d] " % len(sources)
            dispatch_rows.append(("Generated", generated[:-1] if generated else "none"))
        else:
            dispatch_rows.append(("Generated", ''))
            for tar in self.feature_sorted(target_sources):
                sources = target_sources[tar]
                pretty_name = tar if isinstance(tar, str) else '(%s)' % ' '.join(tar)
                flags = ' '.join(self.feature_flags(tar))
                implies = ' '.join(self.feature_sorted(self.feature_implies(tar)))
                detect = ' '.join(self.feature_detect(tar))
                extra_checks = []
                for name in ((tar,) if isinstance(tar, str) else tar):
                    extra_checks += self.feature_extra_checks(name)
                extra_checks = (' '.join(extra_checks) if extra_checks else "none")

                dispatch_rows.append(('', ''))
                dispatch_rows.append((pretty_name, implies))
                dispatch_rows.append(("Flags", flags))
                dispatch_rows.append(("Extra checks", extra_checks))
                dispatch_rows.append(("Detect", detect))
                for src in sources:
                    dispatch_rows.append(("", src))

        ###############################
        # TODO: add support for 'markdown' format
        text = []
        secs_len = [len(secs) for secs, _ in report]
        cols_len = [len(col) for _, rows in report for col, _ in rows]
        tab = ' ' * 2
        pad =  max(max(secs_len), max(cols_len))
        for sec, rows in report:
            if not sec:
                text.append("") # empty line
                continue
            sec += ' ' * (pad - len(sec))
            text.append(sec + tab + ': ')
            for col, val in rows:
                col += ' ' * (pad - len(col))
                text.append(tab + col + ': ' + val)

        return '\n'.join(text)

    def _wrap_target(self, output_dir, dispatch_src, target, nochange=False):
        assert(isinstance(target, (str, tuple)))
        if isinstance(target, str):
            ext_name = target_name = target
        else:
            # multi-target
            ext_name = '.'.join(target)
            target_name = '__'.join(target)

        wrap_path = os.path.join(output_dir, os.path.basename(dispatch_src))
        wrap_path = "{0}.{2}{1}".format(*os.path.splitext(wrap_path), ext_name.lower())
        if nochange and os.path.exists(wrap_path):
            return wrap_path

        self.dist_log("wrap dispatch-able target -> ", wrap_path)
        # sorting for readability
        features = self.feature_sorted(self.feature_implies_c(target))
        target_join = "#define %sCPU_TARGET_" % self.conf_c_prefix_
        target_defs = [target_join + f for f in features]
        target_defs = '\n'.join(target_defs)

        with open(wrap_path, "w") as fd:
            fd.write(textwrap.dedent("""\
            /**
             * AUTOGENERATED DON'T EDIT
             * Please make changes to the code generator \
             (distutils/ccompiler_opt.py)
             */
            #define {pfx}CPU_TARGET_MODE
            #define {pfx}CPU_TARGET_CURRENT {target_name}
            {target_defs}
            #include "{path}"
            """).format(
                pfx=self.conf_c_prefix_, target_name=target_name,
                path=os.path.abspath(dispatch_src), target_defs=target_defs
            ))
        return wrap_path

    def _generate_config(self, output_dir, dispatch_src, targets, has_baseline=False):
        config_path = os.path.basename(dispatch_src)
        config_path = os.path.splitext(config_path)[0] + '.h'
        config_path = os.path.join(output_dir, config_path)
        # check if targets didn't change to avoid recompiling
        cache_hash = self.cache_hash(targets, has_baseline)
        try:
            with open(config_path) as f:
                last_hash = f.readline().split("cache_hash:")
                if len(last_hash) == 2 and int(last_hash[1]) == cache_hash:
                    return True
        except OSError:
            pass

        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        self.dist_log("generate dispatched config -> ", config_path)
        dispatch_calls = []
        for tar in targets:
            if isinstance(tar, str):
                target_name = tar
            else: # multi target
                target_name = '__'.join([t for t in tar])
            req_detect = self.feature_detect(tar)
            req_detect = '&&'.join([
                "CHK(%s)" % f for f in req_detect
            ])
            dispatch_calls.append(
                "\t%sCPU_DISPATCH_EXPAND_(CB((%s), %s, __VA_ARGS__))" % (
                self.conf_c_prefix_, req_detect, target_name
            ))
        dispatch_calls = ' \\\n'.join(dispatch_calls)

        if has_baseline:
            baseline_calls = (
                "\t%sCPU_DISPATCH_EXPAND_(CB(__VA_ARGS__))"
            ) % self.conf_c_prefix_
        else:
            baseline_calls = ''

        with open(config_path, "w") as fd:
            fd.write(textwrap.dedent("""\
            // cache_hash:{cache_hash}
            /**
             * AUTOGENERATED DON'T EDIT
             * Please make changes to the code generator (distutils/ccompiler_opt.py)
             */
            #ifndef {pfx}CPU_DISPATCH_EXPAND_
                #define {pfx}CPU_DISPATCH_EXPAND_(X) X
            #endif
            #undef {pfx}CPU_DISPATCH_BASELINE_CALL
            #undef {pfx}CPU_DISPATCH_CALL
            #define {pfx}CPU_DISPATCH_BASELINE_CALL(CB, ...) \\
            {baseline_calls}
            #define {pfx}CPU_DISPATCH_CALL(CHK, CB, ...) \\
            {dispatch_calls}
            """).format(
                pfx=self.conf_c_prefix_, baseline_calls=baseline_calls,
                dispatch_calls=dispatch_calls, cache_hash=cache_hash
            ))
        return False

def new_ccompiler_opt(compiler, dispatch_hpath, **kwargs):
    """
    Create a new instance of 'CCompilerOpt' and generate the dispatch header
    which contains the #definitions and headers of platform-specific instruction-sets for
    the enabled CPU baseline and dispatch-able features.

    Parameters
    ----------
    compiler : CCompiler instance
    dispatch_hpath : str
        path of the dispatch header

    **kwargs: passed as-is to `CCompilerOpt(...)`
    Returns
    -------
    new instance of CCompilerOpt
    """
    opt = CCompilerOpt(compiler, **kwargs)
    if not os.path.exists(dispatch_hpath) or not opt.is_cached():
        opt.generate_dispatch_header(dispatch_hpath)
    return opt
<|endoftext|>"""
unixccompiler - can handle very long argument lists for ar.

"""
import os
import sys
import subprocess
import shlex

from distutils.errors import CompileError, DistutilsExecError, LibError
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.ccompiler import replace_method
from numpy.distutils.misc_util import _commandline_dep_string
from numpy.distutils import log

# Note that UnixCCompiler._compile appeared in Python 2.3
def UnixCCompiler__compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    """Compile a single source files with a Unix-style compiler."""
    # HP ad-hoc fix, see ticket 1383
    ccomp = self.compiler_so
    if ccomp[0] == 'aCC':
        # remove flags that will trigger ANSI-C mode for aCC
        if '-Ae' in ccomp:
            ccomp.remove('-Ae')
        if '-Aa' in ccomp:
            ccomp.remove('-Aa')
        # add flags for (almost) sane C++ handling
        ccomp += ['-AA']
        self.compiler_so = ccomp
    # ensure OPT environment variable is read
    if 'OPT' in os.environ:
        # XXX who uses this?
        from sysconfig import get_config_vars
        opt = shlex.join(shlex.split(os.environ['OPT']))
        gcv_opt = shlex.join(shlex.split(get_config_vars('OPT')[0]))
        ccomp_s = shlex.join(self.compiler_so)
        if opt not in ccomp_s:
            ccomp_s = ccomp_s.replace(gcv_opt, opt)
            self.compiler_so = shlex.split(ccomp_s)
        llink_s = shlex.join(self.linker_so)
        if opt not in llink_s:
            self.linker_so = self.linker_so + shlex.split(opt)

    display = '%s: %s' % (os.path.basename(self.compiler_so[0]), src)

    # gcc style automatic dependencies, outputs a makefile (-MF) that lists
    # all headers needed by a c file as a side effect of compilation (-MMD)
    if getattr(self, '_auto_depends', False):
        deps = ['-MMD', '-MF', obj + '.d']
    else:
        deps = []

    try:
        self.spawn(self.compiler_so + cc_args + [src, '-o', obj] + deps +
                   extra_postargs, display = display)
    except DistutilsExecError as e:
        msg = str(e)
        raise CompileError(msg) from None

    # add commandline flags to dependency file
    if deps:
        # After running the compiler, the file created will be in EBCDIC
        # but will not be tagged as such. This tags it so the file does not
        # have multiple different encodings being written to it
        if sys.platform == 'zos':
            subprocess.check_output(['chtag', '-tc', 'IBM1047', obj + '.d'])
        with open(obj + '.d', 'a') as f:
            f.write(_commandline_dep_string(cc_args, extra_postargs, pp_opts))

replace_method(UnixCCompiler, '_compile', UnixCCompiler__compile)


def UnixCCompiler_create_static_lib(self, objects, output_libname,
                                    output_dir=None, debug=0, target_lang=None):
    """
    Build a static library in a separate sub-process.

    Parameters
    ----------
    objects : list or tuple of str
        List of paths to object files used to build the static library.
    output_libname : str
        The library name as an absolute or relative (if `output_dir` is used)
        path.
    output_dir : str, optional
        The path to the output directory. Default is None, in which case
        the ``output_dir`` attribute of the UnixCCompiler instance.
    debug : bool, optional
        This parameter is not used.
    target_lang : str, optional
        This parameter is not used.

    Returns
    -------
    None

    """
    objects, output_dir = self._fix_object_args(objects, output_dir)

    output_filename = \
                    self.library_filename(output_libname, output_dir=output_dir)

    if self._need_link(objects, output_filename):
        try:
            # previous .a may be screwed up; best to remove it first
            # and recreate.
            # Also, ar on OS X doesn't handle updating universal archives
            os.unlink(output_filename)
        except OSError:
            pass
        self.mkpath(os.path.dirname(output_filename))
        tmp_objects = objects + self.objects
        while tmp_objects:
            objects = tmp_objects[:50]
            tmp_objects = tmp_objects[50:]
            display = '%s: adding %d object files to %s' % (
                           os.path.basename(self.archiver[0]),
                           len(objects), output_filename)
            self.spawn(self.archiver + [output_filename] + objects,
                       display = display)

        # Not many Unices required ranlib anymore -- SunOS 4.x is, I
        # think the only major Unix that does.  Maybe we need some
        # platform intelligence here to skip ranlib if it's not
        # needed -- or maybe Python's configure script took care of
        # it for us, hence the check for leading colon.
        if self.ranlib:
            display = '%s:@ %s' % (os.path.basename(self.ranlib[0]),
                                   output_filename)
            try:
                self.spawn(self.ranlib + [output_filename],
                           display = display)
            except DistutilsExecError as e:
                msg = str(e)
                raise LibError(msg) from None
    else:
        log.debug("skipping %s (up-to-date)", output_filename)
    return

replace_method(UnixCCompiler, 'create_static_lib',
               UnixCCompiler_create_static_lib)
<|endoftext|>import sys
from distutils.core import Distribution

if 'setuptools' in sys.modules:
    have_setuptools = True
    from setuptools import setup as old_setup
    # easy_install imports math, it may be picked up from cwd
    from setuptools.command import easy_install
    try:
        # very old versions of setuptools don't have this
        from setuptools.command import bdist_egg
    except ImportError:
        have_setuptools = False
else:
    from distutils.core import setup as old_setup
    have_setuptools = False

import warnings
import distutils.core
import distutils.dist

from numpy.distutils.extension import Extension  # noqa: F401
from numpy.distutils.numpy_distribution import NumpyDistribution
from numpy.distutils.command import config, config_compiler, \
     build, build_py, build_ext, build_clib, build_src, build_scripts, \
     sdist, install_data, install_headers, install, bdist_rpm, \
     install_clib
from numpy.distutils.misc_util import is_sequence, is_string

numpy_cmdclass = {'build':            build.build,
                  'build_src':        build_src.build_src,
                  'build_scripts':    build_scripts.build_scripts,
                  'config_cc':        config_compiler.config_cc,
                  'config_fc':        config_compiler.config_fc,
                  'config':           config.config,
                  'build_ext':        build_ext.build_ext,
                  'build_py':         build_py.build_py,
                  'build_clib':       build_clib.build_clib,
                  'sdist':            sdist.sdist,
                  'install_data':     install_data.install_data,
                  'install_headers':  install_headers.install_headers,
                  'install_clib':     install_clib.install_clib,
                  'install':          install.install,
                  'bdist_rpm':        bdist_rpm.bdist_rpm,
                  }
if have_setuptools:
    # Use our own versions of develop and egg_info to ensure that build_src is
    # handled appropriately.
    from numpy.distutils.command import develop, egg_info
    numpy_cmdclass['bdist_egg'] = bdist_egg.bdist_egg
    numpy_cmdclass['develop'] = develop.develop
    numpy_cmdclass['easy_install'] = easy_install.easy_install
    numpy_cmdclass['egg_info'] = egg_info.egg_info

def _dict_append(d, **kws):
    for k, v in kws.items():
        if k not in d:
            d[k] = v
            continue
        dv = d[k]
        if isinstance(dv, tuple):
            d[k] = dv + tuple(v)
        elif isinstance(dv, list):
            d[k] = dv + list(v)
        elif isinstance(dv, dict):
            _dict_append(dv, **v)
        elif is_string(dv):
            d[k] = dv + v
        else:
            raise TypeError(repr(type(dv)))

def _command_line_ok(_cache=None):
    """ Return True if command line does not contain any
    help or display requests.
    """
    if _cache:
        return _cache[0]
    elif _cache is None:
        _cache = []
    ok = True
    display_opts = ['--'+n for n in Distribution.display_option_names]
    for o in Distribution.display_options:
        if o[1]:
            display_opts.append('-'+o[1])
    for arg in sys.argv:
        if arg.startswith('--help') or arg=='-h' or arg in display_opts:
            ok = False
            break
    _cache.append(ok)
    return ok

def get_distribution(always=False):
    dist = distutils.core._setup_distribution
    # XXX Hack to get numpy installable with easy_install.
    # The problem is easy_install runs it's own setup(), which
    # sets up distutils.core._setup_distribution. However,
    # when our setup() runs, that gets overwritten and lost.
    # We can't use isinstance, as the DistributionWithoutHelpCommands
    # class is local to a function in setuptools.command.easy_install
    if dist is not None and \
            'DistributionWithoutHelpCommands' in repr(dist):
        dist = None
    if always and dist is None:
        dist = NumpyDistribution()
    return dist

def setup(**attr):

    cmdclass = numpy_cmdclass.copy()

    new_attr = attr.copy()
    if 'cmdclass' in new_attr:
        cmdclass.update(new_attr['cmdclass'])
    new_attr['cmdclass'] = cmdclass

    if 'configuration' in new_attr:
        # To avoid calling configuration if there are any errors
        # or help request in command in the line.
        configuration = new_attr.pop('configuration')

        old_dist = distutils.core._setup_distribution
        old_stop = distutils.core._setup_stop_after
        distutils.core._setup_distribution = None
        distutils.core._setup_stop_after = "commandline"
        try:
            dist = setup(**new_attr)
        finally:
            distutils.core._setup_distribution = old_dist
            distutils.core._setup_stop_after = old_stop
        if dist.help or not _command_line_ok():
            # probably displayed help, skip running any commands
            return dist

        # create setup dictionary and append to new_attr
        config = configuration()
        if hasattr(config, 'todict'):
            config = config.todict()
        _dict_append(new_attr, **config)

    # Move extension source libraries to libraries
    libraries = []
    for ext in new_attr.get('ext_modules', []):
        new_libraries = []
        for item in ext.libraries:
            if is_sequence(item):
                lib_name, build_info = item
                _check_append_ext_library(libraries, lib_name, build_info)
                new_libraries.append(lib_name)
            elif is_string(item):
                new_libraries.append(item)
            else:
                raise TypeError("invalid description of extension module "
                                "library %r" % (item,))
        ext.libraries = new_libraries
    if libraries:
        if 'libraries' not in new_attr:
            new_attr['libraries'] = []
        for item in libraries:
            _check_append_library(new_attr['libraries'], item)

    # sources in ext_modules or libraries may contain header files
    if ('ext_modules' in new_attr or 'libraries' in new_attr) \
       and 'headers' not in new_attr:
        new_attr['headers'] = []

    # Use our custom NumpyDistribution class instead of distutils' one
    new_attr['distclass'] = NumpyDistribution

    return old_setup(**new_attr)

def _check_append_library(libraries, item):
    for libitem in libraries:
        if is_sequence(libitem):
            if is_sequence(item):
                if item[0]==libitem[0]:
                    if item[1] is libitem[1]:
                        return
                    warnings.warn("[0] libraries list contains %r with"
                                  " different build_info" % (item[0],),
                                  stacklevel=2)
                    break
            else:
                if item==libitem[0]:
                    warnings.warn("[1] libraries list contains %r with"
                                  " no build_info" % (item[0],),
                                  stacklevel=2)
                    break
        else:
            if is_sequence(item):
                if item[0]==libitem:
                    warnings.warn("[2] libraries list contains %r with"
                                  " no build_info" % (item[0],),
                                  stacklevel=2)
                    break
            else:
                if item==libitem:
                    return
    libraries.append(item)

def _check_append_ext_library(libraries, lib_name, build_info):
    for item in libraries:
        if is_sequence(item):
            if item[0]==lib_name:
                if item[1] is build_info:
                    return
                warnings.warn("[3] libraries list contains %r with"
                              " different build_info" % (lib_name,),
                              stacklevel=2)
                break
        elif item==lib_name:
            warnings.warn("[4] libraries list contains %r with"
                          " no build_info" % (lib_name,),
                          stacklevel=2)
            break
    libraries.append((lib_name, build_info))
<|endoftext|>import sys
import re
import os

from configparser import RawConfigParser

__all__ = ['FormatError', 'PkgNotFound', 'LibraryInfo', 'VariableSet',
        'read_config', 'parse_flags']

_VAR = re.compile(r'\$\{([a-zA-Z0-9_-]+)\}')

class FormatError(OSError):
    """
    Exception thrown when there is a problem parsing a configuration file.

    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class PkgNotFound(OSError):
    """Exception raised when a package can not be located."""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

def parse_flags(line):
    """
    Parse a line from a config file containing compile flags.

    Parameters
    ----------
    line : str
        A single line containing one or more compile flags.

    Returns
    -------
    d : dict
        Dictionary of parsed flags, split into relevant categories.
        These categories are the keys of `d`:

        * 'include_dirs'
        * 'library_dirs'
        * 'libraries'
        * 'macros'
        * 'ignored'

    """
    d = {'include_dirs': [], 'library_dirs': [], 'libraries': [],
         'macros': [], 'ignored': []}

    flags = (' ' + line).split(' -')
    for flag in flags:
        flag = '-' + flag
        if len(flag) > 0:
            if flag.startswith('-I'):
                d['include_dirs'].append(flag[2:].strip())
            elif flag.startswith('-L'):
                d['library_dirs'].append(flag[2:].strip())
            elif flag.startswith('-l'):
                d['libraries'].append(flag[2:].strip())
            elif flag.startswith('-D'):
                d['macros'].append(flag[2:].strip())
            else:
                d['ignored'].append(flag)

    return d

def _escape_backslash(val):
    return val.replace('\\', '\\\\')

class LibraryInfo:
    """
    Object containing build information about a library.

    Parameters
    ----------
    name : str
        The library name.
    description : str
        Description of the library.
    version : str
        Version string.
    sections : dict
        The sections of the configuration file for the library. The keys are
        the section headers, the values the text under each header.
    vars : class instance
        A `VariableSet` instance, which contains ``(name, value)`` pairs for
        variables defined in the configuration file for the library.
    requires : sequence, optional
        The required libraries for the library to be installed.

    Notes
    -----
    All input parameters (except "sections" which is a method) are available as
    attributes of the same name.

    """
    def __init__(self, name, description, version, sections, vars, requires=None):
        self.name = name
        self.description = description
        if requires:
            self.requires = requires
        else:
            self.requires = []
        self.version = version
        self._sections = sections
        self.vars = vars

    def sections(self):
        """
        Return the section headers of the config file.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list of str
            The list of section headers.

        """
        return list(self._sections.keys())

    def cflags(self, section="default"):
        val = self.vars.interpolate(self._sections[section]['cflags'])
        return _escape_backslash(val)

    def libs(self, section="default"):
        val = self.vars.interpolate(self._sections[section]['libs'])
        return _escape_backslash(val)

    def __str__(self):
        m = ['Name: %s' % self.name, 'Description: %s' % self.description]
        if self.requires:
            m.append('Requires:')
        else:
            m.append('Requires: %s' % ",".join(self.requires))
        m.append('Version: %s' % self.version)

        return "\n".join(m)

class VariableSet:
    """
    Container object for the variables defined in a config file.

    `VariableSet` can be used as a plain dictionary, with the variable names
    as keys.

    Parameters
    ----------
    d : dict
        Dict of items in the "variables" section of the configuration file.

    """
    def __init__(self, d):
        self._raw_data = dict([(k, v) for k, v in d.items()])

        self._re = {}
        self._re_sub = {}

        self._init_parse()

    def _init_parse(self):
        for k, v in self._raw_data.items():
            self._init_parse_var(k, v)

    def _init_parse_var(self, name, value):
        self._re[name] = re.compile(r'\$\{%s\}' % name)
        self._re_sub[name] = value

    def interpolate(self, value):
        # Brute force: we keep interpolating until there is no '${var}' anymore
        # or until interpolated string is equal to input string
        def _interpolate(value):
            for k in self._re.keys():
                value = self._re[k].sub(self._re_sub[k], value)
            return value
        while _VAR.search(value):
            nvalue = _interpolate(value)
            if nvalue == value:
                break
            value = nvalue

        return value

    def variables(self):
        """
        Return the list of variable names.

        Parameters
        ----------
        None

        Returns
        -------
        names : list of str
            The names of all variables in the `VariableSet` instance.

        """
        return list(self._raw_data.keys())

    # Emulate a dict to set/get variables values
    def __getitem__(self, name):
        return self._raw_data[name]

    def __setitem__(self, name, value):
        self._raw_data[name] = value
        self._init_parse_var(name, value)

def parse_meta(config):
    if not config.has_section('meta'):
        raise FormatError("No meta section found !")

    d = dict(config.items('meta'))

    for k in ['name', 'description', 'version']:
        if not k in d:
            raise FormatError("Option %s (section [meta]) is mandatory, "
                "but not found" % k)

    if not 'requires' in d:
        d['requires'] = []

    return d

def parse_variables(config):
    if not config.has_section('variables'):
        raise FormatError("No variables section found !")

    d = {}

    for name, value in config.items("variables"):
        d[name] = value

    return VariableSet(d)

def parse_sections(config):
    return meta_d, r

def pkg_to_filename(pkg_name):
    return "%s.ini" % pkg_name

def parse_config(filename, dirs=None):
    if dirs:
        filenames = [os.path.join(d, filename) for d in dirs]
    else:
        filenames = [filename]

    config = RawConfigParser()

    n = config.read(filenames)
    if not len(n) >= 1:
        raise PkgNotFound("Could not find file(s) %s" % str(filenames))

    # Parse meta and variables sections
    meta = parse_meta(config)

    vars = {}
    if config.has_section('variables'):
        for name, value in config.items("variables"):
            vars[name] = _escape_backslash(value)

    # Parse "normal" sections
    secs = [s for s in config.sections() if not s in ['meta', 'variables']]
    sections = {}

    requires = {}
    for s in secs:
        d = {}
        if config.has_option(s, "requires"):
            requires[s] = config.get(s, 'requires')

        for name, value in config.items(s):
            d[name] = value
        sections[s] = d

    return meta, vars, sections, requires

def _read_config_imp(filenames, dirs=None):
    def _read_config(f):
        meta, vars, sections, reqs = parse_config(f, dirs)
        # recursively add sections and variables of required libraries
        for rname, rvalue in reqs.items():
            nmeta, nvars, nsections, nreqs = _read_config(pkg_to_filename(rvalue))

            # Update var dict for variables not in 'top' config file
            for k, v in nvars.items():
                if not k in vars:
                    vars[k] = v

            # Update sec dict
            for oname, ovalue in nsections[rname].items():
                if ovalue:
                    sections[rname][oname] += ' %s' % ovalue

        return meta, vars, sections, reqs

    meta, vars, sections, reqs = _read_config(filenames)

    # FIXME: document this. If pkgname is defined in the variables section, and
    # there is no pkgdir variable defined, pkgdir is automatically defined to
    # the path of pkgname. This requires the package to be imported to work
    if not 'pkgdir' in vars and "pkgname" in vars:
        pkgname = vars["pkgname"]
        if not pkgname in sys.modules:
            raise ValueError("You should import %s to get information on %s" %
                             (pkgname, meta["name"]))

        mod = sys.modules[pkgname]
        vars["pkgdir"] = _escape_backslash(os.path.dirname(mod.__file__))

    return LibraryInfo(name=meta["name"], description=meta["description"],
            version=meta["version"], sections=sections, vars=VariableSet(vars))

# Trivial cache to cache LibraryInfo instances creation. To be really
# efficient, the cache should be handled in read_config, since a same file can
# be parsed many time outside LibraryInfo creation, but I doubt this will be a
# problem in practice
_CACHE = {}
def read_config(pkgname, dirs=None):
    """
    Return library info for a package from its configuration file.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).
    dirs : sequence, optional
        If given, should be a sequence of directories - usually including
        the NumPy base directory - where to look for npy-pkg-config files.

    Returns
    -------
    pkginfo : class instance
        The `LibraryInfo` instance containing the build information.

    Raises
    ------
    PkgNotFound
        If the package is not found.

    See Also
    --------
    misc_util.get_info, misc_util.get_pkg_info

    Examples
    --------
    >>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')
    >>> type(npymath_info)
    <class 'numpy.distutils.npy_pkg_config.LibraryInfo'>
    >>> print(npymath_info)
    Name: npymath
    Description: Portable, core math library implementing C99 standard
    Requires:
    Version: 0.1  #random

    """
    try:
        return _CACHE[pkgname]
    except KeyError:
        v = _read_config_imp(pkg_to_filename(pkgname), dirs)
        _CACHE[pkgname] = v
        return v

# TODO:
#   - implements version comparison (modversion + atleast)

# pkg-config simple emulator - useful for debugging, and maybe later to query
# the system
if __name__ == '__main__':
    from optparse import OptionParser
    import glob

    parser = OptionParser()
    parser.add_option("--cflags", dest="cflags", action="store_true",
                      help="output all preprocessor and compiler flags")
    parser.add_option("--libs", dest="libs", action="store_true",
                      help="output all linker flags")
    parser.add_option("--use-section", dest="section",
                      help="use this section instead of default for options")
    parser.add_option("--version", dest="version", action="store_true",
                      help="output version")
    parser.add_option("--atleast-version", dest="min_version",
                      help="Minimal version")
    parser.add_option("--list-all", dest="list_all", action="store_true",
                      help="Minimal version")
    parser.add_option("--define-variable", dest="define_variable",
                      help="Replace variable with the given value")

    (options, args) = parser.parse_args(sys.argv)

    if len(args) < 2:
        raise ValueError("Expect package name on the command line:")

    if options.list_all:
        files = glob.glob("*.ini")
        for f in files:
            info = read_config(f)
            print("%s\t%s - %s" % (info.name, info.name, info.description))

    pkg_name = args[1]
    d = os.environ.get('NPY_PKG_CONFIG_PATH')
    if d:
        info = read_config(
            pkg_name, ['numpy/_core/lib/npy-pkg-config', '.', d]
        )
    else:
        info = read_config(
            pkg_name, ['numpy/_core/lib/npy-pkg-config', '.']
        )

    if options.section:
        section = options.section
    else:
        section = "default"

    if options.define_variable:
        m = re.search(r'([\S]+)=([\S]+)', options.define_variable)
        if not m:
            raise ValueError("--define-variable option should be of "
                             "the form --define-variable=foo=bar")
        else:
            name = m.group(1)
            value = m.group(2)
        info.vars[name] = value

    if options.cflags:
        print(info.cflags(section))
    if options.libs:
        print(info.libs(section))
    if options.version:
        print(info.version)
    if options.min_version:
        print(info.version >= options.min_version)
<|endoftext|>import re
import sys
import subprocess

__doc__ = """This module generates a DEF file from the symbols in
an MSVC-compiled DLL import library.  It correctly discriminates between
data and functions.  The data is collected from the output of the program
nm(1).

Usage:
    python lib2def.py [libname.lib] [output.def]
or
    python lib2def.py [libname.lib] > output.def

libname.lib defaults to python<py_ver>.lib and output.def defaults to stdout

Author: Robert Kern <kernr@mail.ncifcrf.gov>
Last Update: April 30, 1999
"""

__version__ = '0.1a'

py_ver = "%d%d" % tuple(sys.version_info[:2])

DEFAULT_NM = ['nm', '-Cs']

DEF_HEADER = """LIBRARY         python%s.dll
;CODE           PRELOAD MOVEABLE DISCARDABLE
;DATA           PRELOAD SINGLE

EXPORTS
""" % py_ver
# the header of the DEF file

FUNC_RE = re.compile(r"^(.*) in python%s\.dll" % py_ver, re.MULTILINE)
DATA_RE = re.compile(r"^_imp__(.*) in python%s\.dll" % py_ver, re.MULTILINE)

def parse_cmd():
    """Parses the command-line arguments.

libfile, deffile = parse_cmd()"""
    if len(sys.argv) == 3:
        if sys.argv[1][-4:] == '.lib' and sys.argv[2][-4:] == '.def':
            libfile, deffile = sys.argv[1:]
        elif sys.argv[1][-4:] == '.def' and sys.argv[2][-4:] == '.lib':
            deffile, libfile = sys.argv[1:]
        else:
            print("I'm assuming that your first argument is the library")
            print("and the second is the DEF file.")
    elif len(sys.argv) == 2:
        if sys.argv[1][-4:] == '.def':
            deffile = sys.argv[1]
            libfile = 'python%s.lib' % py_ver
        elif sys.argv[1][-4:] == '.lib':
            deffile = None
            libfile = sys.argv[1]
    else:
        libfile = 'python%s.lib' % py_ver
        deffile = None
    return libfile, deffile

def getnm(nm_cmd=['nm', '-Cs', 'python%s.lib' % py_ver], shell=True):
    """Returns the output of nm_cmd via a pipe.

nm_output = getnm(nm_cmd = 'nm -Cs py_lib')"""
    p = subprocess.Popen(nm_cmd, shell=shell, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    nm_output, nm_err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError('failed to run "%s": "%s"' % (
                                     ' '.join(nm_cmd), nm_err))
    return nm_output

def parse_nm(nm_output):
    """Returns a tuple of lists: dlist for the list of data
symbols and flist for the list of function symbols.

dlist, flist = parse_nm(nm_output)"""
    data = DATA_RE.findall(nm_output)
    func = FUNC_RE.findall(nm_output)

    flist = []
    for sym in data:
        if sym in func and (sym[:2] == 'Py' or sym[:3] == '_Py' or sym[:4] == 'init'):
            flist.append(sym)

    dlist = []
    for sym in data:
        if sym not in flist and (sym[:2] == 'Py' or sym[:3] == '_Py'):
            dlist.append(sym)

    dlist.sort()
    flist.sort()
    return dlist, flist

def output_def(dlist, flist, header, file = sys.stdout):
    """Outputs the final DEF file to a file defaulting to stdout.

output_def(dlist, flist, header, file = sys.stdout)"""
    for data_sym in dlist:
        header = header + '\t%s DATA\n' % data_sym
    header = header + '\n' # blank line
    for func_sym in flist:
        header = header + '\t%s\n' % func_sym
    file.write(header)

if __name__ == '__main__':
    libfile, deffile = parse_cmd()
    if deffile is None:
        deffile = sys.stdout
    else:
        deffile = open(deffile, 'w')
    nm_cmd = DEFAULT_NM + [str(libfile)]
    nm_output = getnm(nm_cmd, shell=False)
    dlist, flist = parse_nm(nm_output)
    output_def(dlist, flist, DEF_HEADER, deffile)
<|endoftext|>#!/usr/bin/env python3
"""
takes templated file .xxx.src and produces .xxx file  where .xxx is
.i or .c or .h, using the following template rules

/**begin repeat  -- on a line by itself marks the start of a repeated code
                    segment
/**end repeat**/ -- on a line by itself marks it's end

After the /**begin repeat and before the */, all the named templates are placed
these should all have the same number of replacements

Repeat blocks can be nested, with each nested block labeled with its depth,
i.e.
/**begin repeat1
 *....
 */
/**end repeat1**/

When using nested loops, you can optionally exclude particular
combinations of the variables using (inside the comment portion of the inner loop):

 :exclude: var1=value1, var2=value2, ...

This will exclude the pattern where var1 is value1 and var2 is value2 when
the result is being generated.


In the main body each replace will use one entry from the list of named replacements

 Note that all #..# forms in a block must have the same number of
   comma-separated entries.

Example:

    An input file containing

        /**begin repeat
         * #a = 1,2,3#
         * #b = 1,2,3#
         */

        /**begin repeat1
         * #c = ted, jim#
         */
        @a@, @b@, @c@
        /**end repeat1**/

        /**end repeat**/

    produces

        line 1 "template.c.src"

        /*
         *********************************************************************
         **       This file was autogenerated from a template  DO NOT EDIT!!**
         **       Changes should be made to the original source (.src) file **
         *********************************************************************
         */

        #line 9
        1, 1, ted

        #line 9
        1, 1, jim

        #line 9
        2, 2, ted

        #line 9
        2, 2, jim

        #line 9
        3, 3, ted

        #line 9
        3, 3, jim

"""

__all__ = ['process_str', 'process_file']

import os
import sys
import re

# names for replacement that are already global.
global_names = {}

# header placed at the front of head processed file
header =\
"""
/*
 *****************************************************************************
 **       This file was autogenerated from a template  DO NOT EDIT!!!!      **
 **       Changes should be made to the original source (.src) file         **
 *****************************************************************************
 */

"""
# Parse string for repeat loops
def parse_structure(astr, level):
    """
    The returned line number is from the beginning of the string, starting
    at zero. Returns an empty list if no loops found.

    """
    if level == 0 :
        loopbeg = "/**begin repeat"
        loopend = "/**end repeat**/"
    else :
        loopbeg = "/**begin repeat%d" % level
        loopend = "/**end repeat%d**/" % level

    ind = 0
    line = 0
    spanlist = []
    while True:
        start = astr.find(loopbeg, ind)
        if start == -1:
            break
        start2 = astr.find("*/", start)
        start2 = astr.find("\n", start2)
        fini1 = astr.find(loopend, start2)
        fini2 = astr.find("\n", fini1)
        line += astr.count("\n", ind, start2+1)
        spanlist.append((start, start2+1, fini1, fini2+1, line))
        line += astr.count("\n", start2+1, fini2)
        ind = fini2
    spanlist.sort()
    return spanlist


def paren_repl(obj):
    torep = obj.group(1)
    numrep = obj.group(2)
    return ','.join([torep]*int(numrep))

parenrep = re.compile(r"\(([^)]*)\)\*(\d+)")
plainrep = re.compile(r"([^*]+)\*(\d+)")
def parse_values(astr):
    # replaces all occurrences of '(a,b,c)*4' in astr
    # with 'a,b,c,a,b,c,a,b,c,a,b,c'. Empty braces generate
    # empty values, i.e., ()*4 yields ',,,'. The result is
    # split at ',' and a list of values returned.
    astr = parenrep.sub(paren_repl, astr)
    # replaces occurrences of xxx*3 with xxx, xxx, xxx
    astr = ','.join([plainrep.sub(paren_repl, x.strip())
                     for x in astr.split(',')])
    return astr.split(',')


stripast = re.compile(r"\n\s*\*?")
named_re = re.compile(r"#\s*(\w*)\s*=([^#]*)#")
exclude_vars_re = re.compile(r"(\w*)=(\w*)")
exclude_re = re.compile(":exclude:")
def parse_loop_header(loophead) :
    """Find all named replacements in the header

    Returns a list of dictionaries, one for each loop iteration,
    where each key is a name to be substituted and the corresponding
    value is the replacement string.

    Also return a list of exclusions.  The exclusions are dictionaries
     of key value pairs. There can be more than one exclusion.
     [{'var1':'value1', 'var2', 'value2'[,...]}, ...]

    """
    # Strip out '\n' and leading '*', if any, in continuation lines.
    # This should not effect code previous to this change as
    # continuation lines were not allowed.
    loophead = stripast.sub("", loophead)
    # parse out the names and lists of values
    names = []
    reps = named_re.findall(loophead)
    nsub = None
    for rep in reps:
        name = rep[0]
        vals = parse_values(rep[1])
        size = len(vals)
        if nsub is None :
            nsub = size
        elif nsub != size :
            msg = "Mismatch in number of values, %d != %d\n%s = %s"
            raise ValueError(msg % (nsub, size, name, vals))
        names.append((name, vals))


    # Find any exclude variables
    excludes = []

    for obj in exclude_re.finditer(loophead):
        span = obj.span()
        # find next newline
        endline = loophead.find('\n', span[1])
        substr = loophead[span[1]:endline]
        ex_names = exclude_vars_re.findall(substr)
        excludes.append(dict(ex_names))

    # generate list of dictionaries, one for each template iteration
    dlist = []
    if nsub is None :
        raise ValueError("No substitution variables found")
    for i in range(nsub):
        tmp = {name: vals[i] for name, vals in names}
        dlist.append(tmp)
    return dlist

replace_re = re.compile(r"@(\w+)@")
def parse_string(astr, env, level, line) :
    lineno = "#line %d\n" % line

    # local function for string replacement, uses env
    def replace(match):
        name = match.group(1)
        try :
            val = env[name]
        except KeyError:
            msg = 'line %d: no definition of key "%s"'%(line, name)
            raise ValueError(msg) from None
        return val

    code = [lineno]
    struct = parse_structure(astr, level)
    if struct :
        # recurse over inner loops
        oldend = 0
        newlevel = level + 1
        for sub in struct:
            pref = astr[oldend:sub[0]]
            head = astr[sub[0]:sub[1]]
            text = astr[sub[1]:sub[2]]
            oldend = sub[3]
            newline = line + sub[4]
            code.append(replace_re.sub(replace, pref))
            try :
                envlist = parse_loop_header(head)
            except ValueError as e:
                msg = "line %d: %s" % (newline, e)
                raise ValueError(msg)
            for newenv in envlist :
                newenv.update(env)
                newcode = parse_string(text, newenv, newlevel, newline)
                code.extend(newcode)
        suff = astr[oldend:]
        code.append(replace_re.sub(replace, suff))
    else :
        # replace keys
        code.append(replace_re.sub(replace, astr))
    code.append('\n')
    return ''.join(code)

def process_str(astr):
    code = [header]
    code.extend(parse_string(astr, global_names, 0, 1))
    return ''.join(code)


include_src_re = re.compile(r"(\n|\A)#include\s*['\"]"
                            r"(?P<name>[\w\d./\\]+[.]src)['\"]", re.I)

def resolve_includes(source):
    d = os.path.dirname(source)
    with open(source) as fid:
        lines = []
        for line in fid:
            m = include_src_re.match(line)
            if m:
                fn = m.group('name')
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                lines.append(line)
    return lines

def process_file(source):
    lines = resolve_includes(source)
    sourcefile = os.path.normcase(source).replace("\\", "\\\\")
    try:
        code = process_str(''.join(lines))
    except ValueError as e:
        raise ValueError('In "%s" loop at %s' % (sourcefile, e)) from None
    return '#line 1 "%s"\n%s' % (sourcefile, code)


def unique_key(adict):
    # this obtains a unique key given a dictionary
    # currently it works by appending together n of the letters of the
    #   current keys and increasing n until a unique key is found
    # -- not particularly quick
    allkeys = list(adict.keys())
    done = False
    n = 1
    while not done:
        newkey = "".join([x[:n] for x in allkeys])
        if newkey in allkeys:
            n += 1
        else:
            done = True
    return newkey


def main():
    try:
        file = sys.argv[1]
    except IndexError:
        fid = sys.stdin
        outfile = sys.stdout
    else:
        fid = open(file, 'r')
        (base, ext) = os.path.splitext(file)
        newname = base
        outfile = open(newname, 'w')

    allstr = fid.read()
    try:
        writestr = process_str(allstr)
    except ValueError as e:
        raise ValueError("In %s loop at %s" % (file, e)) from None

    outfile.write(writestr)

if __name__ == "__main__":
    main()
<|endoftext|>from distutils.unixccompiler import UnixCCompiler                              

class ArmCCompiler(UnixCCompiler):

    """
    Arm compiler.
    """

    compiler_type = 'arm'
    cc_exe = 'armclang'
    cxx_exe = 'armclang++'

    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)
        cc_compiler = self.cc_exe
        cxx_compiler = self.cxx_exe
        self.set_executables(compiler=cc_compiler +
                                      ' -O3 -fPIC',
                             compiler_so=cc_compiler +
                                         ' -O3 -fPIC',
                             compiler_cxx=cxx_compiler +
                                          ' -O3 -fPIC',
                             linker_exe=cc_compiler +
                                        ' -lamath',
                             linker_so=cc_compiler +
                                       ' -lamath -shared')
<|endoftext|>#!/usr/bin/env python3
"""

process_file(filename)

  takes templated file .xxx.src and produces .xxx file where .xxx
  is .pyf .f90 or .f using the following template rules:

  '<..>' denotes a template.

  All function and subroutine blocks in a source file with names that
  contain '<..>' will be replicated according to the rules in '<..>'.

  The number of comma-separated words in '<..>' will determine the number of
  replicates.

  '<..>' may have two different forms, named and short. For example,

  named:
   <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with
   'd', 's', 'z', and 'c' for each replicate of the block.

   <_c>  is already defined: <_c=s,d,c,z>
   <_t>  is already defined: <_t=real,double precision,complex,double complex>

  short:
   <s,d,c,z>, a short form of the named, useful when no <p> appears inside
   a block.

  In general, '<..>' contains a comma separated list of arbitrary
  expressions. If these expression must contain a comma|leftarrow|rightarrow,
  then prepend the comma|leftarrow|rightarrow with a backslash.

  If an expression matches '\\<index>' then it will be replaced
  by <index>-th expression.

  Note that all '<..>' forms in a block must have the same number of
  comma-separated entries.

 Predefined named template rules:
  <prefix=s,d,c,z>
  <ftype=real,double precision,complex,double complex>
  <ftypereal=real,double precision,\\0,\\1>
  <ctype=float,double,complex_float,complex_double>
  <ctypereal=float,double,\\0,\\1>

"""
__all__ = ['process_str', 'process_file']

import os
import sys
import re

routine_start_re = re.compile(r'(\n|\A)((     (\$|\*))|)\s*(subroutine|function)\b', re.I)
routine_end_re = re.compile(r'\n\s*end\s*(subroutine|function)\b.*(\n|\Z)', re.I)
function_start_re = re.compile(r'\n     (\$|\*)\s*function\b', re.I)

def parse_structure(astr):
    """ Return a list of tuples for each function or subroutine each
    tuple is the start and end of a subroutine or function to be
    expanded.
    """

    spanlist = []
    ind = 0
    while True:
        m = routine_start_re.search(astr, ind)
        if m is None:
            break
        start = m.start()
        if function_start_re.match(astr, start, m.end()):
            while True:
                i = astr.rfind('\n', ind, start)
                if i==-1:
                    break
                start = i
                if astr[i:i+7]!='\n     $':
                    break
        start += 1
        m = routine_end_re.search(astr, m.end())
        ind = end = m and m.end()-1 or len(astr)
        spanlist.append((start, end))
    return spanlist

template_re = re.compile(r"<\s*(\w[\w\d]*)\s*>")
named_re = re.compile(r"<\s*(\w[\w\d]*)\s*=\s*(.*?)\s*>")
list_re = re.compile(r"<\s*((.*?))\s*>")

def find_repl_patterns(astr):
    reps = named_re.findall(astr)
    names = {}
    for rep in reps:
        name = rep[0].strip() or unique_key(names)
        repl = rep[1].replace(r'\,', '@comma@')
        thelist = conv(repl)
        names[name] = thelist
    return names

def find_and_remove_repl_patterns(astr):
    names = find_repl_patterns(astr)
    astr = re.subn(named_re, '', astr)[0]
    return astr, names

item_re = re.compile(r"\A\\(?P<index>\d+)\Z")
def conv(astr):
    b = astr.split(',')
    l = [x.strip() for x in b]
    for i in range(len(l)):
        m = item_re.match(l[i])
        if m:
            j = int(m.group('index'))
            l[i] = l[j]
    return ','.join(l)

def unique_key(adict):
    """ Obtain a unique key given a dictionary."""
    allkeys = list(adict.keys())
    done = False
    n = 1
    while not done:
        newkey = '__l%s' % (n)
        if newkey in allkeys:
            n += 1
        else:
            done = True
    return newkey


template_name_re = re.compile(r'\A\s*(\w[\w\d]*)\s*\Z')
def expand_sub(substr, names):
    substr = substr.replace(r'\>', '@rightarrow@')
    substr = substr.replace(r'\<', '@leftarrow@')
    lnames = find_repl_patterns(substr)
    substr = named_re.sub(r"<\1>", substr)  # get rid of definition templates

    def listrepl(mobj):
        thelist = conv(mobj.group(1).replace(r'\,', '@comma@'))
        if template_name_re.match(thelist):
            return "<%s>" % (thelist)
        name = None
        for key in lnames.keys():    # see if list is already in dictionary
            if lnames[key] == thelist:
                name = key
        if name is None:      # this list is not in the dictionary yet
            name = unique_key(lnames)
            lnames[name] = thelist
        return "<%s>" % name

    substr = list_re.sub(listrepl, substr) # convert all lists to named templates
                                           # newnames are constructed as needed

    numsubs = None
    base_rule = None
    rules = {}
    for r in template_re.findall(substr):
        if r not in rules:
            thelist = lnames.get(r, names.get(r, None))
            if thelist is None:
                raise ValueError('No replicates found for <%s>' % (r))
            if r not in names and not thelist.startswith('_'):
                names[r] = thelist
            rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
            num = len(rule)

            if numsubs is None:
                numsubs = num
                rules[r] = rule
                base_rule = r
            elif num == numsubs:
                rules[r] = rule
            else:
                print("Mismatch in number of replacements (base <%s=%s>)"
                      " for <%s=%s>. Ignoring." %
                      (base_rule, ','.join(rules[base_rule]), r, thelist))
    if not rules:
        return substr

    def namerepl(mobj):
        name = mobj.group(1)
        return rules.get(name, (k+1)*[name])[k]

    newstr = ''
    for k in range(numsubs):
        newstr += template_re.sub(namerepl, substr) + '\n\n'

    newstr = newstr.replace('@rightarrow@', '>')
    newstr = newstr.replace('@leftarrow@', '<')
    return newstr

def process_str(allstr):
    newstr = allstr
    writestr = ''

    struct = parse_structure(newstr)

    oldend = 0
    names = {}
    names.update(_special_names)
    for sub in struct:
        cleanedstr, defs = find_and_remove_repl_patterns(newstr[oldend:sub[0]])
        writestr += cleanedstr
        names.update(defs)
        writestr += expand_sub(newstr[sub[0]:sub[1]], names)
        oldend =  sub[1]
    writestr += newstr[oldend:]

    return writestr

include_src_re = re.compile(r"(\n|\A)\s*include\s*['\"](?P<name>[\w\d./\\]+\.src)['\"]", re.I)

def resolve_includes(source):
    d = os.path.dirname(source)
    with open(source) as fid:
        lines = []
        for line in fid:
            m = include_src_re.match(line)
            if m:
                fn = m.group('name')
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                if os.path.isfile(fn):
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                lines.append(line)
    return lines

def process_file(source):
    lines = resolve_includes(source)
    return process_str(''.join(lines))

_special_names = find_repl_patterns('''
<_c=s,d,c,z>
<_t=real,double precision,complex,double complex>
<prefix=s,d,c,z>
<ftype=real,double precision,complex,double complex>
<ctype=float,double,complex_float,complex_double>
<ftypereal=real,double precision,\\0,\\1>
<ctypereal=float,double,\\0,\\1>
''')

def main():
    try:
        file = sys.argv[1]
    except IndexError:
        fid = sys.stdin
        outfile = sys.stdout
    else:
        fid = open(file, 'r')
        (base, ext) = os.path.splitext(file)
        newname = base
        outfile = open(newname, 'w')

    allstr = fid.read()
    writestr = process_str(allstr)
    outfile.write(writestr)


if __name__ == "__main__":
    main()
<|endoftext|>import platform

from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils.ccompiler import simple_version_match
if platform.system() == 'Windows':
    from numpy.distutils.msvc9compiler import MSVCCompiler


class IntelCCompiler(UnixCCompiler):
    """A modified Intel compiler compatible with a GCC-built Python."""
    compiler_type = 'intel'
    cc_exe = 'icc'
    cc_args = 'fPIC'

    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)

        v = self.get_version()
        mpopt = 'openmp' if v and v < '15' else 'qopenmp'
        self.cc_exe = ('icc -fPIC -fp-model strict -O3 '
                       '-fomit-frame-pointer -{}').format(mpopt)
        compiler = self.cc_exe

        if platform.system() == 'Darwin':
            shared_flag = '-Wl,-undefined,dynamic_lookup'
        else:
            shared_flag = '-shared'
        self.set_executables(compiler=compiler,
                             compiler_so=compiler,
                             compiler_cxx=compiler,
                             archiver='xiar' + ' cru',
                             linker_exe=compiler + ' -shared-intel',
                             linker_so=compiler + ' ' + shared_flag +
                             ' -shared-intel')


class IntelItaniumCCompiler(IntelCCompiler):
    compiler_type = 'intele'
    cc_exe = 'icc'


class IntelEM64TCCompiler(UnixCCompiler):
    """
    A modified Intel x86_64 compiler compatible with a 64bit GCC-built Python.
    """
    compiler_type = 'intelem'
    cc_exe = 'icc -m64'
    cc_args = '-fPIC'

    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)

        v = self.get_version()
        mpopt = 'openmp' if v and v < '15' else 'qopenmp'
        self.cc_exe = ('icc -std=c99 -m64 -fPIC -fp-model strict -O3 '
                       '-fomit-frame-pointer -{}').format(mpopt)
        compiler = self.cc_exe

        if platform.system() == 'Darwin':
            shared_flag = '-Wl,-undefined,dynamic_lookup'
        else:
            shared_flag = '-shared'
        self.set_executables(compiler=compiler,
                             compiler_so=compiler,
                             compiler_cxx=compiler,
                             archiver='xiar' + ' cru',
                             linker_exe=compiler + ' -shared-intel',
                             linker_so=compiler + ' ' + shared_flag +
                             ' -shared-intel')


if platform.system() == 'Windows':
    class IntelCCompilerW(MSVCCompiler):
        """
        A modified Intel compiler compatible with an MSVC-built Python.
        """
        compiler_type = 'intelw'
        compiler_cxx = 'icl'

        def __init__(self, verbose=0, dry_run=0, force=0):
            MSVCCompiler.__init__(self, verbose, dry_run, force)
            version_match = simple_version_match(start=r'Intel\(R\).*?32,')
            self.__version = version_match

        def initialize(self, plat_name=None):
            MSVCCompiler.initialize(self, plat_name)
            self.cc = self.find_exe('icl.exe')
            self.lib = self.find_exe('xilib')
            self.linker = self.find_exe('xilink')
            self.compile_options = ['/nologo', '/O3', '/MD', '/W3',
                                    '/Qstd=c99']
            self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3',
                                          '/Qstd=c99', '/Z7', '/D_DEBUG']

    class IntelEM64TCCompilerW(IntelCCompilerW):
        """
        A modified Intel x86_64 compiler compatible with
        a 64bit MSVC-built Python.
        """
        compiler_type = 'intelemw'

        def __init__(self, verbose=0, dry_run=0, force=0):
            MSVCCompiler.__init__(self, verbose, dry_run, force)
            version_match = simple_version_match(start=r'Intel\(R\).*?64,')
            self.__version = version_match
<|endoftext|>from distutils.core import Command
from numpy.distutils import log

#XXX: Linker flags

def show_fortran_compilers(_cache=None):
    # Using cache to prevent infinite recursion.
    if _cache:
        return
    elif _cache is None:
        _cache = []
    _cache.append(1)
    from numpy.distutils.fcompiler import show_fcompilers
    import distutils.core
    dist = distutils.core._setup_distribution
    show_fcompilers(dist)

class config_fc(Command):
    """ Distutils command to hold user specified options
    to Fortran compilers.

    config_fc command is used by the FCompiler.customize() method.
    """

    description = "specify Fortran 77/Fortran 90 compiler information"

    user_options = [
        ('fcompiler=', None, "specify Fortran compiler type"),
        ('f77exec=', None, "specify F77 compiler command"),
        ('f90exec=', None, "specify F90 compiler command"),
        ('f77flags=', None, "specify F77 compiler flags"),
        ('f90flags=', None, "specify F90 compiler flags"),
        ('opt=', None, "specify optimization flags"),
        ('arch=', None, "specify architecture specific optimization flags"),
        ('debug', 'g', "compile with debugging information"),
        ('noopt', None, "compile without optimization"),
        ('noarch', None, "compile without arch-dependent optimization"),
        ]

    help_options = [
        ('help-fcompiler', None, "list available Fortran compilers",
         show_fortran_compilers),
        ]

    boolean_options = ['debug', 'noopt', 'noarch']

    def initialize_options(self):
        self.fcompiler = None
        self.f77exec = None
        self.f90exec = None
        self.f77flags = None
        self.f90flags = None
        self.opt = None
        self.arch = None
        self.debug = None
        self.noopt = None
        self.noarch = None

    def finalize_options(self):
        log.info('unifying config_fc, config, build_clib, build_ext, build commands --fcompiler options')
        build_clib = self.get_finalized_command('build_clib')
        build_ext = self.get_finalized_command('build_ext')
        config = self.get_finalized_command('config')
        build = self.get_finalized_command('build')
        cmd_list = [self, config, build_clib, build_ext, build]
        for a in ['fcompiler']:
            l = []
            for c in cmd_list:
                v = getattr(c, a)
                if v is not None:
                    if not isinstance(v, str): v = v.compiler_type
                    if v not in l: l.append(v)
            if not l: v1 = None
            else: v1 = l[0]
            if len(l)>1:
                log.warn('  commands have different --%s options: %s'\
                         ', using first in list as default' % (a, l))
            if v1:
                for c in cmd_list:
                    if getattr(c, a) is None: setattr(c, a, v1)

    def run(self):
        # Do nothing.
        return

class config_cc(Command):
    """ Distutils command to hold user specified options
    to C/C++ compilers.
    """

    description = "specify C/C++ compiler information"

    user_options = [
        ('compiler=', None, "specify C/C++ compiler type"),
        ]

    def initialize_options(self):
        self.compiler = None

    def finalize_options(self):
        log.info('unifying config_cc, config, build_clib, build_ext, build commands --compiler options')
        build_clib = self.get_finalized_command('build_clib')
        build_ext = self.get_finalized_command('build_ext')
        config = self.get_finalized_command('config')
        build = self.get_finalized_command('build')
        cmd_list = [self, config, build_clib, build_ext, build]
        for a in ['compiler']:
            l = []
            for c in cmd_list:
                v = getattr(c, a)
                if v is not None:
                    if not isinstance(v, str): v = v.compiler_type
                    if v not in l: l.append(v)
            if not l: v1 = None
            else: v1 = l[0]
            if len(l)>1:
                log.warn('  commands have different --%s options: %s'\
                         ', using first in list as default' % (a, l))
            if v1:
                for c in cmd_list:
                    if getattr(c, a) is None: setattr(c, a, v1)
        return

    def run(self):
        # Do nothing.
        return
<|endoftext|>import sys
if 'setuptools' in sys.modules:
    from setuptools.command.sdist import sdist as old_sdist
else:
    from distutils.command.sdist import sdist as old_sdist

from numpy.distutils.misc_util import get_data_files

class sdist(old_sdist):

    def add_defaults (self):
        old_sdist.add_defaults(self)

        dist = self.distribution

        if dist.has_data_files():
            for data in dist.data_files:
                self.filelist.extend(get_data_files(data))

        if dist.has_headers():
            headers = []
            for h in dist.headers:
                if isinstance(h, str): headers.append(h)
                else: headers.append(h[1])
            self.filelist.extend(headers)

        return
<|endoftext|>import os
from distutils.core import Command
from distutils.ccompiler import new_compiler
from numpy.distutils.misc_util import get_cmd

class install_clib(Command):
    description = "Command to install installable C libraries"

    user_options = []

    def initialize_options(self):
        self.install_dir = None
        self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options('install', ('install_lib', 'install_dir'))

    def run (self):
        build_clib_cmd = get_cmd("build_clib")
        if not build_clib_cmd.build_clib:
            # can happen if the user specified `--skip-build`
            build_clib_cmd.finalize_options()
        build_dir = build_clib_cmd.build_clib

        # We need the compiler to get the library name -> filename association
        if not build_clib_cmd.compiler:
            compiler = new_compiler(compiler=None)
            compiler.customize(self.distribution)
        else:
            compiler = build_clib_cmd.compiler

        for l in self.distribution.installed_libraries:
            target_dir = os.path.join(self.install_dir, l.target_dir)
            name = compiler.library_filename(l.name)
            source = os.path.join(build_dir, name)
            self.mkpath(target_dir)
            self.outfiles.append(self.copy_file(source, target_dir)[0])

    def get_outputs(self):
        return self.outfiles
<|endoftext|>import sys

from setuptools.command.egg_info import egg_info as _egg_info

class egg_info(_egg_info):
    def run(self):
        if 'sdist' in sys.argv:
            import warnings
            import textwrap
            msg = textwrap.dedent("""
                `build_src` is being run, this may lead to missing
                files in your sdist!  You want to use distutils.sdist
                instead of the setuptools version:

                    from distutils.command.sdist import sdist
                    cmdclass={'sdist': sdist}"

                See numpy's setup.py or gh-7131 for details.""")
            warnings.warn(msg, UserWarning, stacklevel=2)

        # We need to ensure that build_src has been executed in order to give
        # setuptools' egg_info command real filenames instead of functions which
        # generate files.
        self.run_command("build_src")
        _egg_info.run(self)
<|endoftext|>""" Modified version of build_scripts that handles building scripts from functions.

"""
from distutils.command.build_scripts import build_scripts as old_build_scripts
from numpy.distutils import log
from numpy.distutils.misc_util import is_string

class build_scripts(old_build_scripts):

    def generate_scripts(self, scripts):
        new_scripts = []
        func_scripts = []
        for script in scripts:
            if is_string(script):
                new_scripts.append(script)
            else:
                func_scripts.append(script)
        if not func_scripts:
            return new_scripts

        build_dir = self.build_dir
        self.mkpath(build_dir)
        for func in func_scripts:
            script = func(build_dir)
            if not script:
                continue
            if is_string(script):
                log.info("  adding '%s' to scripts" % (script,))
                new_scripts.append(script)
            else:
                [log.info("  adding '%s' to scripts" % (s,)) for s in script]
                new_scripts.extend(list(script))
        return new_scripts

    def run (self):
        if not self.scripts:
            return

        self.scripts = self.generate_scripts(self.scripts)
        # Now make sure that the distribution object has this list of scripts.
        # setuptools' develop command requires that this be a list of filenames,
        # not functions.
        self.distribution.scripts = self.scripts

        return old_build_scripts.run(self)

    def get_source_files(self):
        from numpy.distutils.misc_util import get_script_files
        return get_script_files(self.scripts)
<|endoftext|>"""This module implements additional tests ala autoconf which can be useful.

"""
import textwrap

# We put them here since they could be easily reused outside numpy.distutils

def check_inline(cmd):
    """Return the inline identifier (may be empty)."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #ifndef __cplusplus
        static %(inline)s int static_func (void)
        {
            return 0;
        }
        %(inline)s int nostatic_func (void)
        {
            return 0;
        }
        #endif""")

    for kw in ['inline', '__inline__', '__inline']:
        st = cmd.try_compile(body % {'inline': kw}, None, None)
        if st:
            return kw

    return ''


def check_restrict(cmd):
    """Return the restrict identifier (may be empty)."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        static int static_func (char * %(restrict)s a)
        {
            return 0;
        }
        """)

    for kw in ['restrict', '__restrict__', '__restrict']:
        st = cmd.try_compile(body % {'restrict': kw}, None, None)
        if st:
            return kw

    return ''


def check_compiler_gcc(cmd):
    """Check if the compiler is GCC."""

    cmd._check_compiler()
    body = textwrap.dedent("""
        int
        main()
        {
        #if (! defined __GNUC__)
        #error gcc required
        #endif
            return 0;
        }
        """)
    return cmd.try_compile(body, None, None)


def check_gcc_version_at_least(cmd, major, minor=0, patchlevel=0):
    """
    Check that the gcc version is at least the specified version."""

    cmd._check_compiler()
    version = '.'.join([str(major), str(minor), str(patchlevel)])
    body = textwrap.dedent("""
        int
        main()
        {
        #if (! defined __GNUC__) || (__GNUC__ < %(major)d) || \\
                (__GNUC_MINOR__ < %(minor)d) || \\
                (__GNUC_PATCHLEVEL__ < %(patchlevel)d)
        #error gcc >= %(version)s required
        #endif
            return 0;
        }
        """)
    kw = {'version': version, 'major': major, 'minor': minor,
          'patchlevel': patchlevel}

    return cmd.try_compile(body % kw, None, None)


def check_gcc_function_attribute(cmd, attribute, name):
    """Return True if the given function attribute is supported."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #pragma GCC diagnostic error "-Wattributes"
        #pragma clang diagnostic error "-Wattributes"

        int %s %s(void* unused)
        {
            return 0;
        }

        int
        main()
        {
            return 0;
        }
        """) % (attribute, name)
    return cmd.try_compile(body, None, None) != 0


def check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code,
                                                include):
    """Return True if the given function attribute is supported with
    intrinsics."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #include<%s>
        int %s %s(void)
        {
            %s;
            return 0;
        }

        int
        main()
        {
            return 0;
        }
        """) % (include, attribute, name, code)
    return cmd.try_compile(body, None, None) != 0


def check_gcc_variable_attribute(cmd, attribute):
    """Return True if the given variable attribute is supported."""
    cmd._check_compiler()
    body = textwrap.dedent("""
        #pragma GCC diagnostic error "-Wattributes"
        #pragma clang diagnostic error "-Wattributes"

        int %s foo;

        int
        main()
        {
            return 0;
        }
        """) % (attribute, )
    return cmd.try_compile(body, None, None) != 0
<|endoftext|>import sys
if 'setuptools' in sys.modules:
    import setuptools.command.install as old_install_mod
    have_setuptools = True
else:
    import distutils.command.install as old_install_mod
    have_setuptools = False
from distutils.file_util import write_file

old_install = old_install_mod.install

class install(old_install):

    # Always run install_clib - the command is cheap, so no need to bypass it;
    # but it's not run by setuptools -- so it's run again in install_data
    sub_commands = old_install.sub_commands + [
        ('install_clib', lambda x: True)
    ]

    def finalize_options (self):
        old_install.finalize_options(self)
        self.install_lib = self.install_libbase

    def setuptools_run(self):
        """ The setuptools version of the .run() method.

        We must pull in the entire code so we can override the level used in the
        _getframe() call since we wrap this call by one more level.
        """
        from distutils.command.install import install as distutils_install

        # Explicit request for old-style install?  Just do it
        if self.old_and_unmanageable or self.single_version_externally_managed:
            return distutils_install.run(self)

        # Attempt to detect whether we were called from setup() or by another
        # command.  If we were called by setup(), our caller will be the
        # 'run_command' method in 'distutils.dist', and *its* caller will be
        # the 'run_commands' method.  If we were called any other way, our
        # immediate caller *might* be 'run_command', but it won't have been
        # called by 'run_commands'.  This is slightly kludgy, but seems to
        # work.
        #
        caller = sys._getframe(3)
        caller_module = caller.f_globals.get('__name__', '')
        caller_name = caller.f_code.co_name

        if caller_module != 'distutils.dist' or caller_name!='run_commands':
            # We weren't called from the command line or setup(), so we
            # should run in backward-compatibility mode to support bdist_*
            # commands.
            distutils_install.run(self)
        else:
            self.do_egg_install()

    def run(self):
        if not have_setuptools:
            r = old_install.run(self)
        else:
            r = self.setuptools_run()
        if self.record:
            # bdist_rpm fails when INSTALLED_FILES contains
            # paths with spaces. Such paths must be enclosed
            # with double-quotes.
            with open(self.record) as f:
                lines = []
                need_rewrite = False
                for l in f:
                    l = l.rstrip()
                    if ' ' in l:
                        need_rewrite = True
                        l = '"%s"' % (l)
                    lines.append(l)
            if need_rewrite:
                self.execute(write_file,
                             (self.record, lines),
                             "re-writing list of installed files to '%s'" %
                             self.record)
        return r
<|endoftext|>import os
import sys
if 'setuptools' in sys.modules:
    from setuptools.command.bdist_rpm import bdist_rpm as old_bdist_rpm
else:
    from distutils.command.bdist_rpm import bdist_rpm as old_bdist_rpm

class bdist_rpm(old_bdist_rpm):

    def _make_spec_file(self):
        spec_file = old_bdist_rpm._make_spec_file(self)

        # Replace hardcoded setup.py script name
        # with the real setup script name.
        setup_py = os.path.basename(sys.argv[0])
        if setup_py == 'setup.py':
            return spec_file
        new_spec_file = []
        for line in spec_file:
            line = line.replace('setup.py', setup_py)
            new_spec_file.append(line)
        return new_spec_file
<|endoftext|>""" Override the develop command from setuptools so we can ensure that our
generated files (from build_src or build_scripts) are properly converted to real
files with filenames.

"""
from setuptools.command.develop import develop as old_develop

class develop(old_develop):
    __doc__ = old_develop.__doc__
    def install_for_development(self):
        # Build sources in-place, too.
        self.reinitialize_command('build_src', inplace=1)
        # Make sure scripts are built.
        self.run_command('build_scripts')
        old_develop.install_for_development(self)
<|endoftext|>import os
import sys
from distutils.command.build import build as old_build
from distutils.util import get_platform
from numpy.distutils.command.config_compiler import show_fortran_compilers

class build(old_build):

    sub_commands = [('config_cc',     lambda *args: True),
                    ('config_fc',     lambda *args: True),
                    ('build_src',     old_build.has_ext_modules),
                    ] + old_build.sub_commands

    user_options = old_build.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ('warn-error', None,
         "turn all warnings into errors (-Werror)"),
        ('cpu-baseline=', None,
         "specify a list of enabled baseline CPU optimizations"),
        ('cpu-dispatch=', None,
         "specify a list of dispatched CPU optimizations"),
        ('disable-optimization', None,
         "disable CPU optimized code(dispatch,simd,fast...)"),
        ('simd-test=', None,
         "specify a list of CPU optimizations to be tested against NumPy SIMD interface"),
        ]

    help_options = old_build.help_options + [
        ('help-fcompiler', None, "list available Fortran compilers",
         show_fortran_compilers),
        ]

    def initialize_options(self):
        old_build.initialize_options(self)
        self.fcompiler = None
        self.warn_error = False
        self.cpu_baseline = "min"
        self.cpu_dispatch = "max -xop -fma4" # drop AMD legacy features by default
        self.disable_optimization = False
        """
        the '_simd' module is a very large. Adding more dispatched features
        will increase binary size and compile time. By default we minimize
        the targeted features to those most commonly used by the NumPy SIMD interface(NPYV),
        NOTE: any specified features will be ignored if they're:
            - part of the baseline(--cpu-baseline)
            - not part of dispatch-able features(--cpu-dispatch)
            - not supported by compiler or platform
        """
        self.simd_test = "BASELINE SSE2 SSE42 XOP FMA4 (FMA3 AVX2) AVX512F " \
                         "AVX512_SKX VSX VSX2 VSX3 VSX4 NEON ASIMD VX VXE VXE2"

    def finalize_options(self):
        build_scripts = self.build_scripts
        old_build.finalize_options(self)
        plat_specifier = ".{}-{}.{}".format(get_platform(), *sys.version_info[:2])
        if build_scripts is None:
            self.build_scripts = os.path.join(self.build_base,
                                              'scripts' + plat_specifier)

    def run(self):
        old_build.run(self)
<|endoftext|>""" Modified version of build_ext that handles fortran source files.

"""
import os
import subprocess
from glob import glob

from distutils.dep_util import newer_group
from distutils.command.build_ext import build_ext as old_build_ext
from distutils.errors import DistutilsFileError, DistutilsSetupError,\
    DistutilsError
from distutils.file_util import copy_file

from numpy.distutils import log
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.system_info import combine_paths
from numpy.distutils.misc_util import (
    filter_sources, get_ext_source_files, get_numpy_include_dirs,
    has_cxx_sources, has_f_sources, is_sequence
)
from numpy.distutils.command.config_compiler import show_fortran_compilers
from numpy.distutils.ccompiler_opt import new_ccompiler_opt, CCompilerOpt

class build_ext (old_build_ext):

    description = "build C/C++/F extensions (compile/link to build directory)"

    user_options = old_build_ext.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
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
        ('simd-test=', None,
         "specify a list of CPU optimizations to be tested against NumPy SIMD interface"),
    ]

    help_options = old_build_ext.help_options + [
        ('help-fcompiler', None, "list available Fortran compilers",
         show_fortran_compilers),
    ]

    boolean_options = old_build_ext.boolean_options + ['warn-error', 'disable-optimization']

    def initialize_options(self):
        old_build_ext.initialize_options(self)
        self.fcompiler = None
        self.parallel = None
        self.warn_error = None
        self.cpu_baseline = None
        self.cpu_dispatch = None
        self.disable_optimization = None
        self.simd_test = None

    def finalize_options(self):
        if self.parallel:
            try:
                self.parallel = int(self.parallel)
            except ValueError as e:
                raise ValueError("--parallel/-j argument must be an integer") from e

        # Ensure that self.include_dirs and self.distribution.include_dirs
        # refer to the same list object. finalize_options will modify
        # self.include_dirs, but self.distribution.include_dirs is used
        # during the actual build.
        # self.include_dirs is None unless paths are specified with
        # --include-dirs.
        # The include paths will be passed to the compiler in the order:
        # numpy paths, --include-dirs paths, Python include path.
        if isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)
        incl_dirs = self.include_dirs or []
        if self.distribution.include_dirs is None:
            self.distribution.include_dirs = []
        self.include_dirs = self.distribution.include_dirs
        self.include_dirs.extend(incl_dirs)

        old_build_ext.finalize_options(self)
        self.set_undefined_options('build',
                                        ('parallel', 'parallel'),
                                        ('warn_error', 'warn_error'),
                                        ('cpu_baseline', 'cpu_baseline'),
                                        ('cpu_dispatch', 'cpu_dispatch'),
                                        ('disable_optimization', 'disable_optimization'),
                                        ('simd_test', 'simd_test')
                                  )
        CCompilerOpt.conf_target_groups["simd_test"] = self.simd_test

    def run(self):
        if not self.extensions:
            return

        # Make sure that extension sources are complete.
        self.run_command('build_src')

        if self.distribution.has_c_libraries():
            if self.inplace:
                if self.distribution.have_run.get('build_clib'):
                    log.warn('build_clib already run, it is too late to '
                             'ensure in-place build of build_clib')
                    build_clib = self.distribution.get_command_obj(
                        'build_clib')
                else:
                    build_clib = self.distribution.get_command_obj(
                        'build_clib')
                    build_clib.inplace = 1
                    build_clib.ensure_finalized()
                    build_clib.run()
                    self.distribution.have_run['build_clib'] = 1

            else:
                self.run_command('build_clib')
                build_clib = self.get_finalized_command('build_clib')
            self.library_dirs.append(build_clib.build_clib)
        else:
            build_clib = None

        # Not including C libraries to the list of
        # extension libraries automatically to prevent
        # bogus linking commands. Extensions must
        # explicitly specify the C libraries that they use.

        from distutils.ccompiler import new_compiler
        from numpy.distutils.fcompiler import new_fcompiler

        compiler_type = self.compiler
        # Initialize C compiler:
        self.compiler = new_compiler(compiler=compiler_type,
                                     verbose=self.verbose,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution)
        self.compiler.customize_cmd(self)

        if self.warn_error:
            self.compiler.compiler.append('-Werror')
            self.compiler.compiler_so.append('-Werror')

        self.compiler.show_customization()

        if not self.disable_optimization:
            dispatch_hpath = os.path.join("numpy", "distutils", "include", "npy_cpu_dispatch_config.h")
            dispatch_hpath = os.path.join(self.get_finalized_command("build_src").build_src, dispatch_hpath)
            opt_cache_path = os.path.abspath(
                os.path.join(self.build_temp, 'ccompiler_opt_cache_ext.py')
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
                log.info("\n########### EXT COMPILER OPTIMIZATION ###########")
                log.info(copt.report(full=True))

            import atexit
            atexit.register(report, self.compiler_opt)

        # Setup directory for storing generated extra DLL files on Windows
        self.extra_dll_dir = os.path.join(self.build_temp, '.libs')
        if not os.path.isdir(self.extra_dll_dir):
            os.makedirs(self.extra_dll_dir)

        # Create mapping of libraries built by build_clib:
        clibs = {}
        if build_clib is not None:
            for libname, build_info in build_clib.libraries or []:
                if libname in clibs and clibs[libname] != build_info:
                    log.warn('library %r defined more than once,'
                             ' overwriting build_info\n%s... \nwith\n%s...'
                             % (libname, repr(clibs[libname])[:300], repr(build_info)[:300]))
                clibs[libname] = build_info
        # .. and distribution libraries:
        for libname, build_info in self.distribution.libraries or []:
            if libname in clibs:
                # build_clib libraries have a precedence before distribution ones
                continue
            clibs[libname] = build_info

        # Determine if C++/Fortran 77/Fortran 90 compilers are needed.
        # Update extension libraries, library_dirs, and macros.
        all_languages = set()
        for ext in self.extensions:
            ext_languages = set()
            c_libs = []
            c_lib_dirs = []
            macros = []
            for libname in ext.libraries:
                if libname in clibs:
                    binfo = clibs[libname]
                    c_libs += binfo.get('libraries', [])
                    c_lib_dirs += binfo.get('library_dirs', [])
                    for m in binfo.get('macros', []):
                        if m not in macros:
                            macros.append(m)

                for l in clibs.get(libname, {}).get('source_languages', []):
                    ext_languages.add(l)
            if c_libs:
                new_c_libs = ext.libraries + c_libs
                log.info('updating extension %r libraries from %r to %r'
                         % (ext.name, ext.libraries, new_c_libs))
                ext.libraries = new_c_libs
                ext.library_dirs = ext.library_dirs + c_lib_dirs
            if macros:
                log.info('extending extension %r defined_macros with %r'
                         % (ext.name, macros))
                ext.define_macros = ext.define_macros + macros

            # determine extension languages
            if has_f_sources(ext.sources):
                ext_languages.add('f77')
            if has_cxx_sources(ext.sources):
                ext_languages.add('c++')
            l = ext.language or self.compiler.detect_language(ext.sources)
            if l:
                ext_languages.add(l)

            # reset language attribute for choosing proper linker
            #
            # When we build extensions with multiple languages, we have to
            # choose a linker. The rules here are:
            #   1. if there is Fortran code, always prefer the Fortran linker,
            #   2. otherwise prefer C++ over C,
            #   3. Users can force a particular linker by using
            #          `language='c'`  # or 'c++', 'f90', 'f77'
            #      in their config.add_extension() calls.
            if 'c++' in ext_languages:
                ext_language = 'c++'
            else:
                ext_language = 'c'  # default

            has_fortran = False
            if 'f90' in ext_languages:
                ext_language = 'f90'
                has_fortran = True
            elif 'f77' in ext_languages:
                ext_language = 'f77'
                has_fortran = True

            if not ext.language or has_fortran:
                if l and l != ext_language and ext.language:
                    log.warn('resetting extension %r language from %r to %r.' %
                             (ext.name, l, ext_language))

            ext.language = ext_language

            # global language
            all_languages.update(ext_languages)

        need_f90_compiler = 'f90' in all_languages
        need_f77_compiler = 'f77' in all_languages
        need_cxx_compiler = 'c++' in all_languages

        # Initialize C++ compiler:
        if need_cxx_compiler:
            self._cxx_compiler = new_compiler(compiler=compiler_type,
                                              verbose=self.verbose,
                                              dry_run=self.dry_run,
                                              force=self.force)
            compiler = self._cxx_compiler
            compiler.customize(self.distribution, need_cxx=need_cxx_compiler)
            compiler.customize_cmd(self)
            compiler.show_customization()
            self._cxx_compiler = compiler.cxx_compiler()
        else:
            self._cxx_compiler = None

        # Initialize Fortran 77 compiler:
        if need_f77_compiler:
            ctype = self.fcompiler
            self._f77_compiler = new_fcompiler(compiler=self.fcompiler,
                                               verbose=self.verbose,
                                               dry_run=self.dry_run,
                                               force=self.force,
                                               requiref90=False,
                                               c_compiler=self.compiler)
            fcompiler = self._f77_compiler
            if fcompiler:
                ctype = fcompiler.compiler_type
                fcompiler.customize(self.distribution)
            if fcompiler and fcompiler.get_version():
                fcompiler.customize_cmd(self)
                fcompiler.show_customization()
            else:
                self.warn('f77_compiler=%s is not available.' %
                          (ctype))
                self._f77_compiler = None
        else:
            self._f77_compiler = None

        # Initialize Fortran 90 compiler:
        if need_f90_compiler:
            ctype = self.fcompiler
            self._f90_compiler = new_fcompiler(compiler=self.fcompiler,
                                               verbose=self.verbose,
                                               dry_run=self.dry_run,
                                               force=self.force,
                                               requiref90=True,
                                               c_compiler=self.compiler)
            fcompiler = self._f90_compiler
            if fcompiler:
                ctype = fcompiler.compiler_type
                fcompiler.customize(self.distribution)
            if fcompiler and fcompiler.get_version():
                fcompiler.customize_cmd(self)
                fcompiler.show_customization()
            else:
                self.warn('f90_compiler=%s is not available.' %
                          (ctype))
                self._f90_compiler = None
        else:
            self._f90_compiler = None

        # Build extensions
        self.build_extensions()

        # Copy over any extra DLL files
        # FIXME: In the case where there are more than two packages,
        # we blindly assume that both packages need all of the libraries,
        # resulting in a larger wheel than is required. This should be fixed,
        # but it's so rare that I won't bother to handle it.
        pkg_roots = {
            self.get_ext_fullname(ext.name).split('.')[0]
            for ext in self.extensions
        }
        for pkg_root in pkg_roots:
            shared_lib_dir = os.path.join(pkg_root, '.libs')
            if not self.inplace:
                shared_lib_dir = os.path.join(self.build_lib, shared_lib_dir)
            for fn in os.listdir(self.extra_dll_dir):
                if not os.path.isdir(shared_lib_dir):
                    os.makedirs(shared_lib_dir)
                if not fn.lower().endswith('.dll'):
                    continue
                runtime_lib = os.path.join(self.extra_dll_dir, fn)
                copy_file(runtime_lib, shared_lib_dir)

    def swig_sources(self, sources, extensions=None):
        # Do nothing. Swig sources have been handled in build_src command.
        return sources

    def build_extension(self, ext):
        sources = ext.sources
        if sources is None or not is_sequence(sources):
            raise DistutilsSetupError(
                ("in 'ext_modules' option (extension '%s'), " +
                 "'sources' must be present and must be " +
                 "a list of source filenames") % ext.name)
        sources = list(sources)

        if not sources:
            return

        fullname = self.get_ext_fullname(ext.name)
        if self.inplace:
            modpath = fullname.split('.')
            package = '.'.join(modpath[0:-1])
            base = modpath[-1]
            build_py = self.get_finalized_command('build_py')
            package_dir = build_py.get_package_dir(package)
            ext_filename = os.path.join(package_dir,
                                        self.get_ext_filename(base))
        else:
            ext_filename = os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname))
        depends = sources + ext.depends

        force_rebuild = self.force
        if not self.disable_optimization and not self.compiler_opt.is_cached():
            log.debug("Detected changes on compiler optimizations")
            force_rebuild = True
        if not (force_rebuild or newer_group(depends, ext_filename, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        extra_args = ext.extra_compile_args or []
        extra_cflags = getattr(ext, 'extra_c_compile_args', None) or []
        extra_cxxflags = getattr(ext, 'extra_cxx_compile_args', None) or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        c_sources, cxx_sources, f_sources, fmodule_sources = \
            filter_sources(ext.sources)

        if self.compiler.compiler_type == 'msvc':
            if cxx_sources:
                # Needed to compile kiva.agg._agg extension.
                extra_args.append('/Zm1000')
                extra_cflags += extra_cxxflags
            # this hack works around the msvc compiler attributes
            # problem, msvc uses its own convention :(
            c_sources += cxx_sources
            cxx_sources = []

        # Set Fortran/C++ compilers for compilation and linking.
        if ext.language == 'f90':
            fcompiler = self._f90_compiler
        elif ext.language == 'f77':
            fcompiler = self._f77_compiler
        else:  # in case ext.language is c++, for instance
            fcompiler = self._f90_compiler or self._f77_compiler
        if fcompiler is not None:
            fcompiler.extra_f77_compile_args = (ext.extra_f77_compile_args or []) if hasattr(
                ext, 'extra_f77_compile_args') else []
            fcompiler.extra_f90_compile_args = (ext.extra_f90_compile_args or []) if hasattr(
                ext, 'extra_f90_compile_args') else []
        cxx_compiler = self._cxx_compiler

        # check for the availability of required compilers
        if cxx_sources and cxx_compiler is None:
            raise DistutilsError("extension %r has C++ sources"
                                 "but no C++ compiler found" % (ext.name))
        if (f_sources or fmodule_sources) and fcompiler is None:
            raise DistutilsError("extension %r has Fortran sources "
                                 "but no Fortran compiler found" % (ext.name))
        if ext.language in ['f77', 'f90'] and fcompiler is None:
            self.warn("extension %r has Fortran libraries "
                      "but no Fortran linker found, using default linker" % (ext.name))
        if ext.language == 'c++' and cxx_compiler is None:
            self.warn("extension %r has C++ libraries "
                      "but no C++ linker found, using default linker" % (ext.name))

        kws = {'depends': ext.depends}
        output_dir = self.build_temp

        include_dirs = ext.include_dirs + get_numpy_include_dirs()

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
            # Always generate the generated config files and
            # dispatch-able sources inside the build directory,
            # even if the build option `inplace` is enabled.
            # This approach prevents conflicts with Meson-generated
            # config headers. Since `spin build --clean` will not remove
            # these headers, they might overwrite the generated Meson headers,
            # causing compatibility issues. Maintaining separate directories
            # ensures compatibility between distutils dispatch config headers
            # and Meson headers, avoiding build disruptions.
            # See gh-24450 for more details.
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

        c_objects = []
        if copt_cxx_sources:
            log.info("compiling C++ dispatch-able sources")
            c_objects += self.compiler_opt.try_dispatch(
                copt_cxx_sources,
                output_dir=output_dir,
                src_dir=copt_build_src,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=extra_args + extra_cxxflags,
                ccompiler=cxx_compiler,
                **kws
            )
        if copt_c_sources:
            log.info("compiling C dispatch-able sources")
            c_objects += self.compiler_opt.try_dispatch(
                copt_c_sources,
                output_dir=output_dir,
                src_dir=copt_build_src,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=extra_args + extra_cflags,
                **kws)
        if c_sources:
            log.info("compiling C sources")
            c_objects += self.compiler.compile(
                c_sources,
                output_dir=output_dir,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=(extra_args + copt_baseline_flags +
                                extra_cflags),
                **kws)
        if cxx_sources:
            log.info("compiling C++ sources")
            c_objects += cxx_compiler.compile(
                cxx_sources,
                output_dir=output_dir,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=(extra_args + copt_baseline_flags +
                                extra_cxxflags),
                **kws)

        extra_postargs = []
        f_objects = []
        if fmodule_sources:
            log.info("compiling Fortran 90 module sources")
            module_dirs = ext.module_dirs[:]
            module_build_dir = os.path.join(
                self.build_temp, os.path.dirname(
                    self.get_ext_filename(fullname)))

            self.mkpath(module_build_dir)
            if fcompiler.module_dir_switch is None:
                existing_modules = glob('*.mod')
            extra_postargs += fcompiler.module_options(
                module_dirs, module_build_dir)
            f_objects += fcompiler.compile(fmodule_sources,
                                           output_dir=self.build_temp,
                                           macros=macros,
                                           include_dirs=include_dirs,
                                           debug=self.debug,
                                           extra_postargs=extra_postargs,
                                           depends=ext.depends)

            if fcompiler.module_dir_switch is None:
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
                        log.warn('failed to move %r to %r' %
                                 (f, module_build_dir))
        if f_sources:
            log.info("compiling Fortran sources")
            f_objects += fcompiler.compile(f_sources,
                                           output_dir=self.build_temp,
                                           macros=macros,
                                           include_dirs=include_dirs,
                                           debug=self.debug,
                                           extra_postargs=extra_postargs,
                                           depends=ext.depends)

        if f_objects and not fcompiler.can_ccompiler_link(self.compiler):
            unlinkable_fobjects = f_objects
            objects = c_objects
        else:
            unlinkable_fobjects = []
            objects = c_objects + f_objects

        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []
        libraries = self.get_libraries(ext)[:]
        library_dirs = ext.library_dirs[:]

        linker = self.compiler.link_shared_object
        # Always use system linker when using MSVC compiler.
        if self.compiler.compiler_type in ('msvc', 'intelw', 'intelemw'):
            # expand libraries with fcompiler libraries as we are
            # not using fcompiler linker
            self._libs_with_msvc_and_fortran(
                fcompiler, libraries, library_dirs)
            if ext.runtime_library_dirs:
                # gcc adds RPATH to the link. On windows, copy the dll into
                # self.extra_dll_dir instead.
                for d in ext.runtime_library_dirs:
                    for f in glob(d + '/*.dll'):
                        copy_file(f, self.extra_dll_dir)
                ext.runtime_library_dirs = []

        elif ext.language in ['f77', 'f90'] and fcompiler is not None:
            linker = fcompiler.link_shared_object
        if ext.language == 'c++' and cxx_compiler is not None:
            linker = cxx_compiler.link_shared_object

        if fcompiler is not None:
            objects, libraries = self._process_unlinkable_fobjects(
                    objects, libraries,
                    fcompiler, library_dirs,
                    unlinkable_fobjects)

        linker(objects, ext_filename,
               libraries=libraries,
               library_dirs=library_dirs,
               runtime_library_dirs=ext.runtime_library_dirs,
               extra_postargs=extra_args,
               export_symbols=self.get_export_symbols(ext),
               debug=self.debug,
               build_temp=self.build_temp,
               target_lang=ext.language)

    def _add_dummy_mingwex_sym(self, c_sources):
        build_src = self.get_finalized_command("build_src").build_src
        build_clib = self.get_finalized_command("build_clib").build_clib
        objects = self.compiler.compile([os.path.join(build_src,
                                                      "gfortran_vs2003_hack.c")],
                                        output_dir=self.build_temp)
        self.compiler.create_static_lib(
            objects, "_gfortran_workaround", output_dir=build_clib, debug=self.debug)

    def _process_unlinkable_fobjects(self, objects, libraries,
                                     fcompiler, library_dirs,
                                     unlinkable_fobjects):
        libraries = list(libraries)
        objects = list(objects)
        unlinkable_fobjects = list(unlinkable_fobjects)

        # Expand possible fake static libraries to objects;
        # make sure to iterate over a copy of the list as
        # "fake" libraries will be removed as they are
        # encountered
        for lib in libraries[:]:
            for libdir in library_dirs:
                fake_lib = os.path.join(libdir, lib + '.fobjects')
                if os.path.isfile(fake_lib):
                    # Replace fake static library
                    libraries.remove(lib)
                    with open(fake_lib) as f:
                        unlinkable_fobjects.extend(f.read().splitlines())

                    # Expand C objects
                    c_lib = os.path.join(libdir, lib + '.cobjects')
                    with open(c_lib) as f:
