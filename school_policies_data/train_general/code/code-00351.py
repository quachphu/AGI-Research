        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            kernels
        )

        # need this since some code in inductor check if the kernel object has an args
        # attribute to decide if it's a non-null kernel.
        self.args = object()

    @staticmethod
    def _merge_workspace_args(left: list[WorkspaceArg], right: list[WorkspaceArg]):
        if left == right:
            return left
        result = {x.inner_name: x for x in left}
        for arg in right:
            if arg.inner_name in result:
                result[arg.inner_name] = WorkspaceArg.maximum(
                    result[arg.inner_name], arg
                )
            else:
                result[arg.inner_name] = arg
        return [*result.values()]

    @staticmethod
    def merge_workspaces_inplace(kernels):
        if len(kernels) < 2:
            return
        # All kernels must share the same workspace
        workspace_args = functools.reduce(
            MultiKernel._merge_workspace_args,
            [kernel.args.workspace_args for kernel in kernels],
        )
        for kernel in kernels:
            kernel.args.workspace_args = workspace_args
        return workspace_args

    def call_kernel(self, kernel_name):
        """
        Collect the union of arguments from all subkernels as the arguments
        for the multi-kernel.
        """
        # Prevent circular import
        from ..select_algorithm import TritonTemplateKernel

        if kernel_name != self.kernel_name:
            raise AssertionError(
                f"expected kernel_name == self.kernel_name, "
                f"got {kernel_name} != {self.kernel_name}"
            )
        V.graph.wrapper_code.write_triton_header_once()
        _, call_args, _, arg_types = self.kernels[0].args.python_argdefs()
        for kernel in self.kernels[1:]:
            _, other_call_args, _, other_arg_types = kernel.args.python_argdefs()
            if call_args != other_call_args:
                raise AssertionError((call_args, other_call_args))
            if arg_types != other_arg_types:
                raise AssertionError(
                    f"expected arg_types == other_arg_types, "
                    f"got {arg_types} != {other_arg_types}"
                )

        if V.graph.cpp_wrapper and not config.triton.autotune_at_compile_time:
            # for the second pass of cpp-wrapper codegen, we should call
            # the fast kernel directly
            kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)

        if isinstance(self.kernels[0], TritonTemplateKernel) and isinstance(
            self.kernels[0].output_node, MultiTemplateBuffer
        ):
            # For matmuls the grid arguments are passed in as additional arguments
            # to the kernel run method. These grids change based on the various
            # parameters of the matmul. So we need to pass each kernel's grid into
            # the multi call kernel.
            multi_call_args = call_args
            multi_call_arg_types = arg_types
            for kernel in self.kernels:
                additional_call_args, additional_arg_types = (
                    kernel.additional_call_args_and_types()
                )
                multi_call_args.extend(list(additional_call_args))
                multi_call_arg_types.extend(list(additional_arg_types))
        else:
            # numels for all subkernels should be the same. Use kernels[0] here
            self.kernels[0].add_numel_to_call_args(kernel_name, call_args, arg_types)
            multi_call_args = call_args
            multi_call_arg_types = arg_types

        for ws in self.kernels[0].args.workspace_args:
            V.graph.wrapper_code.generate_workspace_allocation(ws)

        if V.graph.cpp_wrapper:
            # We have already selected the best kernel at compile time
            # so we only have one set of call args. NB: this currently
            # doesn't work with MultiTemplateBuffer kernels. @bobrenjc93
            # will add it in a subsequent PR.
            V.graph.wrapper_code.generate_kernel_call(
                kernel_name, call_args, arg_types=arg_types
            )
        else:
            V.graph.wrapper_code.generate_kernel_call(
                kernel_name, multi_call_args, arg_types=multi_call_arg_types
            )

        for ws in reversed(self.kernels[0].args.workspace_args):
            V.graph.wrapper_code.generate_workspace_deallocation(ws)

    def codegen_nan_check(self):
        wrapper = V.graph.wrapper_code
        seen: OrderedSet[str] = OrderedSet()
        for k in self.kernels:
            _, call_args, precompile_args, _ = k.args.python_argdefs()
            for arg, precompile_arg in zip(call_args, precompile_args):
                if arg in seen:
                    continue
                seen.add(arg)
                if isinstance(precompile_arg, TensorArg):
                    line = f"assert not {arg}.isnan().any().item()"
                    wrapper.writeline(line)
                    line = f"assert not {arg}.isinf().any().item()"
                    wrapper.writeline(line)

    @property
    def removed_buffers(self):
        return OrderedSet.intersection(*[k.removed_buffers for k in self.kernels])

    @property
    def inplaced_to_remove(self):
        return OrderedSet.intersection(*[k.inplaced_to_remove for k in self.kernels])

    @property
    @cache_on_self
    def inplace_update_buffers(self):
        """
        Make sure all kernels have the same inplace update mappings.
        """
        for k in self.kernels[1:]:
            if k.inplace_update_buffers != self.kernels[0].inplace_update_buffers:
                raise AssertionError(
                    "expected all kernels to share the same inplace_update_buffers"
                )
        return self.kernels[0].inplace_update_buffers

    def warn_mix_layout(self, kernel_name: str):
        pass


class MultiKernelCall:
    """
    This class is called at run time to actually run the kernel
    """

    def __init__(self, multi_kernel_name, kernels, arg_index):
        if len(kernels) < 1:
            raise AssertionError(f"expected at least 1 kernel, got {len(kernels)}")
        self._kernels = kernels
        self.multi_kernel_name = multi_kernel_name

        self.disable_cache = os.environ.get(
            "TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE"
        ) == "1" or is_metric_table_enabled("persistent_red_perf")

        self.picked_kernel = None
        self.arg_index = arg_index
        if config.triton.multi_kernel > 1:
            # manually force a subkernel to ease perf testing
            picked_by_config = config.triton.multi_kernel - 2
            if picked_by_config >= len(self._kernels):
                raise AssertionError(
                    f"expected picked_by_config < len(self._kernels), "
                    f"got {picked_by_config} >= {len(self._kernels)}"
                )
            # pyrefly: ignore [bad-assignment]
            self.picked_kernel = picked_by_config
        elif not self.disable_cache:
            self.load_cache()

        self._recorded = False

    def cache_file_path(self):
        key = code_hash(
            ",".join(
                [
                    f"{k.fn.cache_key}{k.size_hints!r}{k.triton_meta!r}"
                    for k in self.kernels
                ]
            )
        )
        _, _, path = get_path(key, "picked_kernel")
        return pathlib.Path(path)

    def load_cache(self):
        if self.picked_kernel is not None:
            raise AssertionError("expected self.picked_kernel to be None")
        path = self.cache_file_path()
        if path.exists():
            with path.open() as fd:
                # pyrefly: ignore [bad-assignment]
                self.picked_kernel = int(fd.read())
                # pyrefly: ignore [unsupported-operation]
                if not (
                    self.picked_kernel >= 0 and self.picked_kernel < len(self._kernels)
                ):
                    raise AssertionError(
                        f"expected 0 <= picked_kernel < {len(self._kernels)}, "
                        f"got {self.picked_kernel}"
                    )
                log.debug(
                    "Load picked kernel %d from cache file %s", self.picked_kernel, path
                )

    def store_cache(self):
        if self.picked_kernel is None:
            raise AssertionError("expected self.picked_kernel to not be None")
        path = self.cache_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        write_atomic(path, str(self.picked_kernel))
        log.debug("Store picked kernel %d to cache file %s", self.picked_kernel, path)

    @property
    def kernels(self):
        """
        Read results from future.

        This should be called after parallel compilation is done.
        In case you call this before compilation is done,
        it may slow down the parallel compilation.
        """
        for i, kernel in enumerate(self._kernels):
            if isinstance(kernel, CodeCacheFuture):
                self._kernels[i] = kernel.result()

        return self._kernels

    def benchmark_sub_kernels(self, *args, **kwargs):
        """
        Benchmark all the sub kernels and return the execution time
        (in milliseconds) for each of time.

        Unit test may mock this method to force a specific kernel to
        be picked.
        """

        def wrap_fn(kernel, index):
            def inner():
                filtered_args = self._get_filtered_args(args, index)
                args_clone, kwargs_clone = kernel.clone_args(*filtered_args, **kwargs)
                return kernel.run(*args_clone, **kwargs_clone)

            return inner

        return [
            benchmarker.benchmark(
                wrap_fn(kernel, index),
                # Currently the kernel type must be a CachingAutotuner
                device=kernel.device_props.type,
                rep=40,
            )
            for index, kernel in enumerate(self.kernels)
        ]

    def _get_filtered_args(self, args, index):
        """
        We pass in all arguments to all kernels into the MultiKernelCall
        so when invoking a particular kernel we need to filter to only the
        arguments for that specific kernel.
        """

        # This is sometimes invoked at runtime where V.graph is
        # a NullHandler
        if hasattr(V.graph, "cpp_wrapper") and V.graph.cpp_wrapper:
            # for cpp-wrapper, we should not filter the args since
            # we already have chosen a single kernel and arg set.
            return args
        return [item for s in self.arg_index[index] for item in args[s]]

    # record_choice and lookup_choice are helper functions for cpp-wrapper
    # codegen. The first pass use record_choice to keep the choice and
    # the second pass do lookup by calling lookup_choice.
    #
    # An alternative that reused the multi-kernel cache does not work well
    # since during codegen of the second pass, it's very hard to know the
    # path for the cache file. Also reading the cache file need do some IO
    # which can be slower.
    @staticmethod
    def record_choice(multi_kernel_name: str, picked_kernel_name: str):
        """
        Record the multi-kernel choice for cpp-wrapper after autotuning

        We should do nothing if this function is not called during codegen.
        """
        from torch._inductor.graph import GraphLowering

        if not isinstance(V.graph, GraphLowering):
            return

        if not V.graph.record_multi_kernel_choice:
            return

        V.graph.multi_kernel_to_choice[multi_kernel_name] = picked_kernel_name

    @staticmethod
    def lookup_choice(multi_kernel_name: str) -> str:
        # this should always been done during cpp-wrapper codegen
        if not (
            V.graph.record_multi_kernel_choice
            and multi_kernel_name in V.graph.multi_kernel_to_choice
        ):
            raise AssertionError(
                f"expected multi_kernel_name {multi_kernel_name} to be recorded "
                f"during cpp-wrapper codegen"
            )
        # there should be no miss
        return V.graph.multi_kernel_to_choice[multi_kernel_name]

    def run(self, *args, **kwargs):
        if self.picked_kernel is None:
            timings = self.benchmark_sub_kernels(*args, **kwargs)
            self.picked_kernel = timings.index(min(timings))
            k0 = self.kernels[0]
            log.debug(
                "pick %dth sub-kernel in %s. Size hints %s. Reduction hint %s. Timings %s",
                self.picked_kernel,
                [k.inductor_meta.get("kernel_name") for k in self.kernels],
                k0.size_hints,
                k0.inductor_meta.get("reduction_hint"),
                timings,
            )
            get_metric_table("persistent_red_perf").add_row(
                functools.partial(self._metrics_table_row, timings)
            )

            if not self.disable_cache:
                self.store_cache()

        if not self._recorded:
            self._recorded = True
            picked_kernel_name = self.kernels[self.picked_kernel].inductor_meta.get(
                "kernel_name"
            )
            if picked_kernel_name is None:
                raise AssertionError("expected picked_kernel_name to not be None")
            self.record_choice(self.multi_kernel_name, picked_kernel_name)

        run = self.kernels[self.picked_kernel].run  # type: ignore[method-assign]
        filtered_args = self._get_filtered_args(args, self.picked_kernel)
        run(*filtered_args, **kwargs)

    def _metrics_table_row(self, timings):
        def get_kernel_path(k):
            return k.fn.fn.__code__.co_filename

        k0 = self.kernels[0]
        row = {
            "size_hints": k0.size_hints,
            "reduction_hint": k0.inductor_meta.get("reduction_hint"),
        }
        max_kernels = 4
        if len(timings) > max_kernels:
            raise AssertionError(
                f"expected len(timings) <= {max_kernels}, got {len(timings)}"
            )
        for i in range(max_kernels):
            if i < len(self.kernels):
                row[f"kernel{i}_path"] = get_kernel_path(self.kernels[i])
                row[f"kernel{i}_latency"] = timings[i]
            else:
                row[f"kernel{i}_path"] = ""
                row[f"kernel{i}_latency"] = ""
        return row


class SizeHintMultiKernel(MultiKernel):
    """
    Version of multi-kernel that generates kernels based on specified size hints.
    Currently only performs 1-d search over hints; doesn't perform combinatorial n-d search
    if n > 1 dynamic dimensions are specified.

    e.g. matmul([s0, s1], [s1, s2]) with size-hints [64, 256] only generates 2 kernels,
    based on tuning shapes ([64, 64], [64, 64]) and ([256, 256], [256, 256])
    """

    def __init__(self, kernels):
        if not (isinstance(kernels, dict) and len(kernels) >= 1):
            raise AssertionError("expected kernels to be a non-empty dict")

        self.kernels, self.kernel_shape_keys = [], []
        for shape_key, kernel in kernels.items():
            self.kernels.append(kernel)
            self.kernel_shape_keys.append(shape_key)
        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            self.kernels, self.kernel_shape_keys
        )

        # need this since some code in inductor check if the kernel object has an args
        # attribute to decide if it's a non-null kernel.
        self.args = object()


class SizeHintMultiKernelCall(MultiKernelCall):
    """
    Runtime class for size-hint multi-kernels.
    Instead of having a plain list of kernels to benchmark over, keys them by input & output shapes,
    and optionally perform shape-based selection. The pre-generated kernel is chosen based on the shape keys,
    with the heuristic being log2 l1 distance between the pre-generated / runtime input & output shapes.
    """

    def __init__(self, multi_kernel_name, kernels, arg_index):
        super().__init__(multi_kernel_name, list(kernels.values()), arg_index)
        self._kernel_hints = list(kernels.keys())

        # Caches results for unique shapes.
        self._shape_cache = {}

    def _get_shape_cache_key(self, *args, **kwargs):
        """
        Generate a cache key based on tensor shapes for shape-specialized dispatch.
        """
        shapes = []
        for arg in args:
            if hasattr(arg, "shape"):
                shapes.append(tuple(arg.shape))
        return tuple(shapes)

    def _get_cached_shape_choice(self, cache_key):
        """
        Get cached kernel choice for a specific shape.
        """
        return self._shape_cache.get(cache_key)

    def _cache_shape_choice(self, cache_key, kernel_idx):
        """
        Cache kernel choice for a specific shape.
        """
        self._shape_cache[cache_key] = kernel_idx

    def _dist_heuristic(self, k1, k2):
        """
        log2 L1 distance heuristic for kernel selection.
        """

        def dist(x, y):
            lx = math.log2(x) if x > 0 else -1
            ly = math.log2(y) if y > 0 else -1
            return abs(lx - ly)

        out = 0
        for s1, s2 in zip(k1, k2):
            out += sum(dist(x, y) for x, y in zip(s1, s2))
        return out

    def run(self, *args, **kwargs):
        cache_key = self._get_shape_cache_key(*args, **kwargs)
        cached_choice = self._get_cached_shape_choice(cache_key)
        if cached_choice is not None:
            self.picked_kernel = cached_choice
            log.debug(
                "using cached shape-specialized choice %dth sub-kernel in %s. Cache key: %s",
                self.picked_kernel,
                [k.inductor_meta.get("kernel_name") for k in self.kernels],
                cache_key,
            )
        else:
            self._select_kernel_by_shape(*args, **kwargs)

        if not self._recorded:
            self._recorded = True
            picked_kernel_name = self.kernels[self.picked_kernel].inductor_meta.get(
                "kernel_name"
            )
            if picked_kernel_name is None:
                raise AssertionError("expected picked_kernel_name to not be None")
            self.record_choice(self.multi_kernel_name, picked_kernel_name)

        run = self.kernels[self.picked_kernel].run  # type: ignore[method-assign]
        filtered_args = self._get_filtered_args(args, self.picked_kernel)
        run(*filtered_args, **kwargs)

    def _select_kernel_by_shape(self, *args, **kwargs):
        """
        Benchmark kernels for a particular shape and return the
        best kernel for this shape.
        """
        shape_key = self._get_shape_cache_key(*args, **kwargs)
        dists = [
            self._dist_heuristic(shape_key, key) if key is not None else 2**62
            for key in self._kernel_hints
        ]
        # pyrefly: ignore [bad-assignment]
        self.picked_kernel = dists.index(min(dists))
        self._cache_shape_choice(shape_key, self.picked_kernel)
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import operator
import sys
from collections.abc import Callable
from enum import Enum

import torch

from .. import cpp_builder, ir
from ..cpu_vec_isa import (
    pick_vec_isa,
    VecAMX,
    VecAVX2,
    VecAVX512,
    VecAVX512VNNI,
    VecISA,
    VecNEON,
    VecSVE,
)
from ..utils import IndentedBuffer, parallel_num_threads
from ..virtualized import V
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking, value_to_cpp


class LayoutType(Enum):
    NORMAL = 0
    VNNI2 = 1
    VNNI4 = 2


_IS_WINDOWS = sys.platform == "win32"


def get_restrict_keyword() -> str:
    if _IS_WINDOWS:
        # https://learn.microsoft.com/en-us/cpp/cpp/extension-restrict?view=msvc-170
        return "__restrict"
    else:
        return "__restrict__"


class CppMicroGemm:
    """
    A class that codegens a kernel that computes small-sized matrix multiplication.

    A micro GEMM kernel is responsible for register blocking, instruction selection,
    and other CPU architecture-specific optimizations.

    The subclasses need to override `codegen_define` to define the kernel function
    that is called by the code generated by `codegen_call`.
    """

    # TODO(jgong5): support constant shapes and lds as template args.
    DECLARE_KERNEL = r"""
template <bool accum, bool prefetch=false>
inline void {{kernel_name}}(
{%- if kernel_extra_args_declare %}
    {{kernel_extra_args_declare}}
{%- endif %}
    const {{input_t}}* {{restrict_keyword}} A,
    const {{input2_t}}* {{restrict_keyword}} B,
    {{output_t}}* {{restrict_keyword}} C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
)
"""

    def __init__(
        self,
        name,
        input_dtype,
        input2_dtype,
        output_dtype,
        compute_dtype,
        register_blocking,
        alpha=1,
    ) -> None:
        self.name = name
        self.input_dtype = input_dtype
        if input2_dtype is None:
            raise AssertionError("expected input2_dtype to be set, got None")
        self.input2_dtype = input2_dtype
        self.output_dtype = output_dtype
        self.compute_dtype = compute_dtype
        self.register_blocking = register_blocking
        self.alpha = alpha
        self.pack_vnni_B_locally = False

    def get_common_options(self):
        if self.input_dtype in [torch.uint8, torch.int8]:
            if self.compute_dtype != torch.int32:
                raise AssertionError(
                    f"expected compute_dtype == torch.int32, got {self.compute_dtype}"
                )
            if self.output_dtype != torch.int32:
                raise AssertionError(
                    f"expected output_dtype == torch.int32, got {self.output_dtype}"
                )
            if self.input2_dtype != torch.int8:
                raise AssertionError(
                    f"expected input2_dtype == torch.int8, got {self.input2_dtype}"
                )
        return {
            "torch": torch,
            "kernel_name": self.name,
            "input_dtype": self.input_dtype,
            "input2_dtype": self.input2_dtype,
            "output_dtype": self.output_dtype,
            "compute_dtype": self.compute_dtype,
            "input_t": DTYPE_TO_CPP[self.input_dtype],
            "input2_t": DTYPE_TO_CPP[self.input2_dtype],
            "output_t": DTYPE_TO_CPP[self.output_dtype],
            "compute_t": DTYPE_TO_CPP[self.compute_dtype],
            "alpha": self.alpha,
            "kernel_extra_args_declare": self.get_kernel_extra_args_declare(),
            "int8_gemm": self.input_dtype in [torch.uint8, torch.int8],
            "vnni_size": 4 if self.input_dtype in [torch.uint8, torch.int8] else 2,
            "restrict_keyword": get_restrict_keyword(),
            "pack_vnni_B_locally": self.pack_vnni_B_locally,
            "template": self,
            "is_woq_int4": self.is_woq_int4(),
        }

    def get_kernel_declaration(self):
        options = self.get_common_options()
        return KernelTemplate._template_from_string(self.DECLARE_KERNEL).render(options)

    def get_kernel_extra_args_declare(self) -> str:
        return ""

    def get_kernel_extra_args(self, **kwargs) -> list[str]:
        return []

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        raise NotImplementedError

    def codegen_call(
        self,
        kernel: CppTemplateKernel,
        A: ir.Buffer,
        B: ir.Buffer,
        C: ir.Buffer,
        accum: bool,
        prefetch: bool = False,
        **kwargs_for_extra_args,
    ) -> str:
        """
        Generate the code for calling the templated kernel that computes
        `C += alpha * A @ B` if `accum` is True, or `C = alpha * A @ B` otherwise.
        """
        A_ptr = f"&({kernel.index(A, [0, 0])})"
        B_ptr = f"&({kernel.index(B, [0, 0])})"
        C_ptr = f"&({kernel.index(C, [0, 0])})"
        M = kernel.size(C, 0)
        N = kernel.size(C, 1)
        K = kernel.size(A, 1)
        lda = kernel.stride(A, 0)
        ldb = kernel.stride(B, 0)
        ldc = kernel.stride(C, 0)
        res = IndentedBuffer()
        res.writeline(
            f"{self.name}<{value_to_cpp(accum, 'bool')}, {value_to_cpp(prefetch, 'bool')}>("
        )
        with res.indent():
            kwargs_for_extra_args.update({"kernel": kernel})
            extra_args = self.get_kernel_extra_args(**kwargs_for_extra_args)
            for arg in extra_args:
                res.writeline(arg)
            res.writeline(f"{A_ptr},")
            res.writeline(f"{B_ptr},")
            res.writeline(f"{C_ptr},")
            res.writeline(f"{M},")
            res.writeline(f"{N},")
            res.writeline(f"{K},")
            res.writeline(f"{lda},")
            res.writeline(f"{ldb},")
            res.writeline(f"{ldc}")
        res.writeline(");")
        return res.getvalue()

    def use_local_vnni_blocking(self, should_block_weight: bool):
        self.pack_vnni_B_locally = should_block_weight

    def codegen_init(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        return ""

    def codegen_finalize(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        return ""

    def get_b_layout(self) -> LayoutType:
        return LayoutType.NORMAL

    ALLOCATE_WEIGHT_BUFFER = r"""
    {%- if is_msvc_compiler %}
    // MSVC doesn't support stack-allocated dynamic-sized arrays, so using heap memory here.
    auto heap_deq_b_buf_ptr = std::make_unique<{{buffer_dtype}}[]>({{buffer_size}});
    {{buffer_dtype}}* {{buffer_name}} = heap_deq_b_buf_ptr.get();
    {%- else %}
    // It's safe to use a stack-allocated array since the blocking strategy would
    // require us to allocate an array that's smaller than the size of L1D cache,
    // and the default per thread max stack size on Linux is quite higher,
    // so we need not worry about stack overflow.
    alignas(4096) {{buffer_dtype}} {{buffer_name}}[{{buffer_size}}];
    {%- endif %}
"""

    def codegen_allocate_weight_buffer(
        self, buffer_name: str, buffer_dtype: str, *size_args
    ) -> str:
        buffer_size = " * ".join(map(str, size_args))
        return KernelTemplate._template_from_string(self.ALLOCATE_WEIGHT_BUFFER).render(
            {
                "buffer_name": buffer_name,
                "buffer_dtype": buffer_dtype,
                "buffer_size": buffer_size,
                "is_msvc_compiler": cpp_builder.is_msvc_cl(),
            }
        )

    def is_woq_int4(self):
        return False


@dataclasses.dataclass
class CppMicroGemmConfig:
    input_dtype: torch.dtype
    input2_dtype: torch.dtype
    output_dtype: torch.dtype
    compute_dtype: torch.dtype
    vec_isa_cls: type[VecISA]
    register_blocking: GemmBlocking
    extra_check: Callable[..., bool] | None = None


micro_gemm_configs: dict[type[CppMicroGemm], list[CppMicroGemmConfig]] = {}


def register_micro_gemm(*configs):
    def inner(cls):
        if cls in micro_gemm_configs:
            raise AssertionError(f"Duplicate micro_gemm registration for {cls}")
        if len(configs) <= 0:
            raise AssertionError(f"No micro_gemm configs provided for {cls}")
        micro_gemm_configs[cls] = list(configs)
        return cls

    return inner


def generate_gemm_config(
    vec_isa_cls,
    register_blockings,
    input_dtype=torch.float,
    input2_dtype=None,
    output_dtype=None,
    compute_dtype=None,
    extra_check=None,
):
    if output_dtype is None:
        output_dtype = input_dtype
    if compute_dtype is None:
        compute_dtype = output_dtype
    if input2_dtype is None:
        input2_dtype = input_dtype
    return [
        CppMicroGemmConfig(
            input_dtype,
            input2_dtype,
            output_dtype,
            compute_dtype,
            vec_isa_cls,
            GemmBlocking(*blocking),
            extra_check,
        )
        for blocking in register_blockings
    ]


class CppMicroGemmRef(CppMicroGemm):
    """
    A reference implementation of the CppMicroGemm class with naive C++ code.
    It is used for correctness debugging.
    """

    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            {{compute_t}} result = accum ? C[m * ldc + n] : 0;
            for (int64_t k = 0; k < K; ++k) {
                result += ({{compute_t}})A[m * lda + k] * ({{compute_t}})B[k * ldb + n] * {{alpha}};
            }
            C[m * ldc + n] = result;
        }
    }
}
"""

    def __init__(
        self, name, input_dtype, input2_dtype, output_dtype, compute_dtype, alpha
    ) -> None:
        super().__init__(
            name,
            input_dtype,
            input2_dtype,
            output_dtype,
            compute_dtype,
            GemmBlocking(1, 1, 1),
            alpha,
        )

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            **self.get_common_options(),
        }
        return KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(options)


def is_int8_woq_gemm_small_m_dim_corner_case(config, m, n, k):
    return (
        k % config.register_blocking.block_k == 0
        and n % config.register_blocking.block_n == 0
        and m < 16
    )


# extra check for small M dimension for int8 WoQ case
def check_int8_woq_small_m_dim(config, m, n, k, alpha, num_threads, **kwargs):
    return is_int8_woq_gemm_small_m_dim_corner_case(config, m, n, k) and not kwargs.get(
        "dynamic_M", False
    )


# For int8 WoQ GEMM with small M, we use different blockings that shouldn't be used otherwise
def do_not_use_with_small_m_for_int8_woq(config, m, n, k, alpha, num_threads, **kwargs):
    return not check_int8_woq_small_m_dim(config, m, n, k, alpha, num_threads, **kwargs)


@register_micro_gemm(
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.bfloat16,
        output_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.half,
        output_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX512,
        [(8, 48, 1), (8, 32, 1), (16, 16, 1)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=do_not_use_with_small_m_for_int8_woq,
    ),
    *generate_gemm_config(
        VecAVX512,
        [
            (4, 32, 64),
            (8, 32, 64),
        ],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_int8_woq_small_m_dim,
    ),
    *generate_gemm_config(
        VecAVX2,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX2,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.bfloat16,
        output_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX2,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.half,
        output_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecAVX2,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=do_not_use_with_small_m_for_int8_woq,
    ),
    *generate_gemm_config(
        VecAVX2,
        [
            (2, 16, 64),
            (4, 16, 64),
        ],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_int8_woq_small_m_dim,
    ),
    *generate_gemm_config(
        VecNEON,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.float,
        input2_dtype=torch.float,
        output_dtype=torch.float,
        compute_dtype=torch.float,
    ),
    *generate_gemm_config(
        VecSVE,
        [(4, 24, 1), (4, 16, 1), (8, 8, 1)],
        input_dtype=torch.float,
        input2_dtype=torch.float,
        output_dtype=torch.float,
        compute_dtype=torch.float,
    ),
)
class CppMicroGemmFP32Vec(CppMicroGemm):
    """
    This class generates the code for micro gemm using fp32 vec instructions for compute.
    It supports input types of torch.float, torch.bfloat16, and torch.half with fp32 output.
    The output of the microkernel is in FP32, but it would be converted to BF16/FP16 in the template,
    if the desired output is BF16/FP16.
    """

    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    using Vectorized = at::vec::Vectorized<{{compute_t}}>;
    constexpr auto VLEN = Vectorized::size();
    {{kernel.assert_function}}({{block_n}} % VLEN == 0, "block_n dimension must be multiple of Vector size");
    {{kernel.assert_function}}(K % {{block_k}} == 0, "K dimension must be multiple of {{block_k}}");
    // TODO(jgong5): loop unroll for M and N
    for (int64_t m = 0; m < M; m += {{block_m}}) {
        int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
        for (int64_t n = 0; n < N; n += {{block_n}}) {
            int64_t block_n = std::min<int64_t>(N - n, {{block_n}});
            if (block_m == {{block_m}} && block_n == {{block_n}}) {
{%- if not trans_b %}
                {{kernel_name}}_kernel<{{block_m}}, {{block_n}}, accum, prefetch>(
{%- else %}
                {{kernel_name}}_transpose_b_kernel<{{block_m}}, {{block_n}}, accum, prefetch>(
{%- endif %}
                    A + m * lda,
{%- if not trans_b %}
                    B + n,
{%- else %}
                    B + n * ldb,
{%- endif %}
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc
                );
{%- if tail_n %}
            } else if (block_n == {{block_n}}){
{%- else %}
            } else {
{%- endif %}
                switch (block_m) {
{%- for b in range(block_m - 1, 0, -1) %}
                case {{b}}:
    {%- if not trans_b %}
                    {{kernel_name}}_kernel<{{b}}, {{block_n}}, accum, prefetch>(
    {%- else %}
                    {{kernel_name}}_transpose_b_kernel<{{b}}, {{block_n}}, accum, prefetch>(
    {%- endif %}
                        A + m * lda,
    {%- if not trans_b %}
                        B + n,
    {%- else %}
                        B + n * ldb,
    {%- endif %}
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
{%- endfor %}
                default:
                    {{kernel.assert_function}}(false, "Unsupported block_m: {{block_m}}");
                }

{%- if tail_n %}
            } else {
                switch (block_m) {
    {%- for b in range(block_m, 0, -1) %}
                case {{b}}:
        {%- if not trans_b %}
                    {{kernel_name}}_ntail_kernel<{{b}}, {{block_n}}, accum, prefetch>(
        {%- else %}
                    {{kernel_name}}_ntail_transpose_b_kernel<{{b}}, {{block_n}}, accum, prefetch>(
        {%- endif %}
                        A + m * lda,
        {%- if not trans_b %}
                        B + n,
        {%- else %}
                        B + n * ldb,
        {%- endif %}
                        C + m * ldc + n,
                        block_n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
    {%- endfor %}
                default:
                    {{kernel.assert_function}}(false, "Unsupported block_m: {{block_m}}");
                }
            }
{%- else %}
            }
{%- endif %}
        }
    }
}
"""

    TEMPLATE_KERNEL = r"""

template <int64_t BLOCK_M, int64_t BLOCK_N, bool accum, bool prefetch=false>
{%- if not trans_b %}
    {%- if tail_n %}
inline void {{kernel_name}}_ntail_kernel(
    {%- else %}
inline void {{kernel_name}}_kernel(
    {%- endif %}
{%- else %}
    {%- if tail_n %}
inline void {{kernel_name}}_ntail_transpose_b_kernel(
    {%- else %}
inline void {{kernel_name}}_transpose_b_kernel(
    {%- endif %}
{%- endif %}
    const {{input_t}}* {{restrict_keyword}} A,
    const {{input2_t}}* {{restrict_keyword}} B,
    {{output_t}}* {{restrict_keyword}} C,
{%- if tail_n %}
    int64_t N,
{%- endif %}
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    using Vectorized = at::vec::Vectorized<{{compute_t}}>;
{%- if input2_dtype in [torch.bfloat16, torch.float16] %}
    using VectorizedIn = at::vec::Vectorized<{{input_t}}>;
{%- endif %}

{%- if not trans_b %}
    constexpr auto VLEN = Vectorized::size();
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;

    Vectorized va;
    at::vec::VectorizedN<{{compute_t}}, COLS> vb;
    at::vec::VectorizedN<{{compute_t}}, ROWS*COLS> vc;

    {%- if tail_n %}
    int64_t rCOLS = (N + VLEN - 1) / VLEN;
    int ntail = N % VLEN;
    {%- endif %}
    auto loadc = [&](auto i) {
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
    {%- if tail_n %}
            int load_size = (col == rCOLS - 1 && ntail != 0) ? ntail : VLEN;
            if (col < rCOLS) {
                vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN, load_size);
            }
    {%- else %}
            vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN);
    {%- endif %}
        } else {
            vc[i] = Vectorized(0.0f);
        }
    };
    c10::ForcedUnroll<ROWS * COLS>{}(loadc);

    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
    {%- if tail_n %}
        int load_size = (col == rCOLS - 1 && ntail != 0) ? ntail : VLEN;
    {%- endif %}
        if constexpr (col == 0) {
    {%- if alpha != 1 %}
            va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + k]) * {{alpha}});
    {%- else %}
            va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + k]));
    {%- endif %}
        }

        if constexpr (row == 0) {
    {%- if tail_n %}
            if (col < rCOLS) {
        {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
                auto b = VectorizedIn::loadu(B + k * ldb + col * VLEN, load_size);
                vb[col] = at::vec::convert<{{compute_t}}>(b);
        {%- elif input2_dtype == torch.int8 %}
            // Convert VLEN int8 elements to int32, and then fp32
                auto b32 = at::vec::convert_to_int32<int8_t>(B + k * ldb + col * VLEN, load_size);
                vb[col] = at::vec::convert<float>(b32);
        {%- else %}
                vb[col] = Vectorized::loadu(B + k * ldb + col * VLEN, load_size);
        {%- endif %}
            } else {
                vb[col] = Vectorized(0.0f);
            }

    {%- else %}

        {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
            auto b = VectorizedIn::loadu(B + k * ldb + col * VLEN, VLEN);
            vb[col] = at::vec::convert<{{compute_t}}>(b);
        {%- elif input2_dtype == torch.int8 %}
            // Convert VLEN int8 elements to int32, and then fp32
            auto b32 = at::vec::convert_to_int32<int8_t>(B + k * ldb + col * VLEN);
            if constexpr (prefetch) {
              _mm_prefetch(B + (k + {{block_k}}) * ldb + col * VLEN, _MM_HINT_T0);
            }
            vb[col] = at::vec::convert<float>(b32);
        {%- else %}
            vb[col] = Vectorized::loadu(B + k * ldb + col * VLEN);
        {%- endif %}
    {%- endif %}

        }

        constexpr int idx = row * COLS + col;
    {%- if tail_n %}
        if (col < rCOLS) {
            vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
        }
    {%- else %}
        vc[idx] = at::vec::fmadd(va, vb[col], vc[idx]);
    {%- endif %}
    };

    for (int k = 0; k < K; ++k) {
        c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }

    // store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
    {%- if tail_n %}
        int store_size = (col == rCOLS - 1 && ntail != 0) ? ntail : VLEN;
        if (col < rCOLS) {
            vc[i].store(C + row * ldc + col * VLEN, store_size);
        }
    {%- else %}
        vc[i].store(C + row * ldc + col * VLEN);
    {%- endif %}
    };
    c10::ForcedUnroll<ROWS * COLS>{}(storec);

{%- else %}
    // Use 2 implementations for the transposed B:
    // First implementation:
    //   Transpose first and then perform outer product calculation in sub-blocks,
    //   which introduces an additional transpose overhead of [K, N] compared to the non-transpose version.
    // Second implementation:
    //   Directly perform inner product calculation in sub-blocks,
    //   which introduces an additional vector reduction of [M, N] compared to the non-transpose version.
    // Therefore, when M * N / (K * N) is large, the first implementation has better performance.
    {%- if tail_n %}
    if (K % Vectorized::size() == 0 && N % Vectorized::size() == 0 && 24 * BLOCK_M > K) {
    {%- else %}
    if (K % Vectorized::size() == 0 && 24 * BLOCK_M > K) {
    {%- endif %}
        // First implementation:
        constexpr auto VLEN = Vectorized::size();
        constexpr auto ROWS = BLOCK_M;
        constexpr auto COLS = BLOCK_N / VLEN;
        int _K = K / VLEN;
        Vectorized va;
        at::vec::VectorizedN<{{compute_t}}, VLEN> vb;
        at::vec::VectorizedN<{{compute_t}}, ROWS*COLS> vc;
        auto loadc = [&](auto i) {
            if constexpr (accum) {
                constexpr int row = i / COLS;
                constexpr int col = i % COLS;
                vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN);
            } else {
                vc[i] = Vectorized(0.0f);
            }
        };
        c10::ForcedUnroll<ROWS * COLS>{}(loadc);
        auto unroll_loadB = [&](auto i, const {{input2_t}}* {{restrict_keyword}} src_ptr) {
    {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
            auto b = VectorizedIn::loadu(src_ptr + i * ldb, VLEN);
            vb[i] = at::vec::convert<{{compute_t}}>(b);
    {%- elif input2_dtype == torch.int8 %}
            auto b32 = at::vec::convert_to_int32<int8_t>(src_ptr + i * ldb, VLEN);
            vb[i] = at::vec::convert<float>(b32);
    {%- else %}
            vb[i] = Vectorized::loadu(src_ptr + i * ldb, VLEN);
    {%- endif %}
        };
        auto compute_trans = [&, COLS](auto i, int k) {
            constexpr int row = i % ROWS;
            constexpr int col = i / ROWS;
            constexpr int e_col = col * VLEN;
            int idk = k * VLEN;
            if constexpr (row == 0) {
                c10::ForcedUnroll<VLEN>{}(unroll_loadB, B + e_col * ldb + idk);
                at::vec::transpose_block(vb);
            }
            constexpr int idx = row * COLS + col;
            {{kernel.unroll_pragma(16)}}
            for (int j = 0; j < VLEN; j++) {
    {%- if alpha != 1 %}
                va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + idk + j]) * {{alpha}});
    {%- else %}
                va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + idk + j]));
    {%- endif %}
                vc[idx] = at::vec::fmadd(va, vb[j], vc[idx]);
            }
        };
        for (int k = 0; k < _K; ++k) {
            c10::ForcedUnroll<ROWS * COLS>{}(compute_trans, k);
        }
        // store to C
        auto storec = [&](auto i) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i].store(C + row * ldc + col * VLEN);
        };
        c10::ForcedUnroll<ROWS * COLS>{}(storec);
    } else {
        // Second implementation
    {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
        constexpr auto VLEN = VectorizedIn::size();
    {%- else %}
        constexpr auto VLEN = Vectorized::size();
    {%- endif %}
        int _K = (K + VLEN - 1) / VLEN;
        // sub-block size of BLOCK_N and BLOCK_M
        constexpr int sM = {{sub_block_m}};
        constexpr int sN = {{sub_block_n}};
    {%- if tail_n %}
        int bN = (N + sN - 1) / sN;
    {%- else %}
        constexpr int bN = (BLOCK_N + sN - 1) / sN;
    {%- endif %}
        constexpr int bM = (BLOCK_M + sM - 1) / sM;

    {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
        at::vec::VectorizedN<{{compute_t}}, 2> va;
        at::vec::VectorizedN<{{compute_t}}, 2 * sN> vb;
    {%- else %}
        at::vec::Vectorized<{{compute_t}}> va;
        at::vec::VectorizedN<{{compute_t}}, sN> vb;
    {%- endif %}
        at::vec::VectorizedN<{{compute_t}}, sN * sM> vmid;

    {%- if tail_n %}
        int ntail = N % sN;
    {%- else %}
        constexpr int ntail = BLOCK_N % sN;
    {%- endif %}
        constexpr int mtail = BLOCK_M % sM;
        int ktail = K % VLEN;

        auto compute_trans = [&](int m, int n, int k) {
    {%- if tail_n %}
            int e_n = (n == bN - 1 && ntail != 0) ? (N - n * sN) : sN;
    {%- else %}
            int e_n = (n == bN - 1 && ntail != 0) ? (BLOCK_N - n * sN) : sN;
    {%- endif %}
            int e_m = (m == bM - 1 && mtail != 0) ? (BLOCK_M - m * sM) : sM;
            int e_k = (k == _K - 1 && ktail != 0) ? (K - k * VLEN) : VLEN;
            {{kernel.unroll_pragma(sub_block_n)}}
            for (int i = 0; i < e_n; i++) {
    {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
                auto b = VectorizedIn::loadu(B + (sN * n + i) * ldb + k * VLEN, e_k);
                std::tie(vb[2 * i], vb[2 * i + 1]) = at::vec::convert_to_float<{{input_t}}>(b);
    {%- elif input2_dtype == torch.int8 %}
                auto b32 = at::vec::convert_to_int32<int8_t>(B + (sN * n + i) * ldb + k * VLEN, e_k);
                vb[i] = at::vec::convert<float>(b32);
    {%- else %}
                vb[i] = Vectorized::loadu(B + (sN * n + i) * ldb + k * VLEN, e_k);
    {%- endif %}
            }

            {{kernel.unroll_pragma(sub_block_m)}}
            for (int s = 0; s < e_m; s++) {
    {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
                auto a = VectorizedIn::loadu(A + (sM * m + s) * lda + k * VLEN, e_k);
                std::tie(va[0], va[1]) = at::vec::convert_to_float<{{input_t}}>(a);
    {%- elif input2_dtype == torch.int8 %}
                auto a32 = at::vec::convert_to_int32<int8_t>(A + (sM * m + s) * lda + k * VLEN, e_k);
                va = at::vec::convert<float>(a32);
    {%- else %}
                va = Vectorized::loadu(A + (sM * m + s) * lda + k * VLEN, e_k);
    {%- endif %}

    {%- if alpha != 1 %}
                va = va * Vectorized({{alpha}});
    {%- endif %}
                if (k == 0) {
                    {{kernel.unroll_pragma(sub_block_n)}}
                    for (int i = 0; i < e_n; i++) {
    {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
                        vmid[sN * s + i] = at::vec::fmadd(va[0], vb[2 * i], Vectorized(0.0f));
                        vmid[sN * s + i] = at::vec::fmadd(va[1], vb[2 * i + 1], vmid[sN * s + i]);
    {%- else %}
                        vmid[sN * s + i] = at::vec::fmadd(va, vb[i], Vectorized(0.0f));
    {%- endif %}
                    }
                } else {
                    {{kernel.unroll_pragma(sub_block_n)}}
                    for (int i = 0; i < e_n; i++) {
    {%- if input2_dtype in [torch.bfloat16, torch.float16] %}
                        vmid[sN * s + i] = at::vec::fmadd(va[0], vb[2 * i], vmid[sN * s + i]);
                        vmid[sN * s + i] = at::vec::fmadd(va[1], vb[2 * i + 1], vmid[sN * s + i]);
    {%- else %}
                        vmid[sN * s + i] = at::vec::fmadd(va, vb[i], vmid[sN * s + i]);
    {%- endif %}
                    }
                }
            }

            // store to C
            if (k == _K - 1) {
                {{kernel.unroll_pragma(sub_block_m)}}
                for (int s = 0; s < e_m; s++) {
                    {{kernel.unroll_pragma(sub_block_n)}}
                    for (int i = 0; i < e_n; i++) {
                        auto v = at::vec::vec_reduce_all([](Vectorized& x, Vectorized& y) { return x + y; }, vmid[sN * s + i]);
                        if constexpr (accum) {
                            auto c = *(C + (sM * m + s) * ldc + sN * n + i);
                            *(C + (sM * m + s) * ldc + sN * n + i) = c + v;
                        } else {
                            *(C + (sM * m + s) * ldc + sN * n + i) = v;
                        }
                    }
                }
            }
        };

        for (int n = 0; n < bN; ++n) {
            for (int m = 0; m < bM; ++m) {
                for (int k = 0; k < _K; ++k) {
                    compute_trans(m, n, k);
                }
            }
        }
    }
{%- endif %}
}
"""

    # set trans_b to generate gemm that supports transposed B matrix
    # set tail_n to support the tail of N
    # TODO add trans_b support for other micro gemms
    # and move setting of trans_b to the init of CppMicroGemm
    def __init__(
        self,
        name,
        input_dtype,
        input2_dtype,
        output_dtype,
        compute_dtype,
        register_blocking,
        alpha=1,
        tail_n=False,
        trans_b=False,
    ) -> None:
        super().__init__(
            name,
            input_dtype,
            input2_dtype,
            output_dtype,
            compute_dtype,
            register_blocking,
            alpha,
        )
        self.tail_n = tail_n
        # trans_b is only supported on platforms that
        # support avx512 or avx2 since transpose_block is
        # only implemented on these platforms
        if trans_b:
            vec_isa = pick_vec_isa()
            if not (
                issubclass(vec_isa.__class__, VecAVX512)
                or issubclass(vec_isa.__class__, VecAVX2)
            ):
                raise AssertionError(
                    f"trans_b requires AVX512 or AVX2 vec ISA, got {vec_isa.__class__}"
                )
        self.trans_b = trans_b

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            "kernel": kernel,
            "block_m": self.register_blocking.block_m,
            "block_n": self.register_blocking.block_n,
            "block_k": self.register_blocking.block_k,
            "trans_b": False,
            "tail_n": False,
            "restrict_keyword": get_restrict_keyword(),
            **self.get_common_options(),
        }
        if self.trans_b:
            # TODO supports tuning of sub_block_m/sub_block_n
            # to get better performance for specific shapes
            sub_block_m = min(1, self.register_blocking.block_m)
            sub_block_n = min(4, self.register_blocking.block_n)
            # update options to generate kernel with trans_b and sub-block size
            options.update(
                {
                    "trans_b": self.trans_b,
                    "sub_block_m": sub_block_m,
                    "sub_block_n": sub_block_n,
                }
            )
        result = KernelTemplate._template_from_string(self.TEMPLATE_KERNEL).render(
            options
        )
        # update options to generate the kernel for the tail of N
        if self.tail_n:
            options.update(
                {
                    "tail_n": self.tail_n,
                }
            )
            result += KernelTemplate._template_from_string(self.TEMPLATE_KERNEL).render(
                options
            )
        result += KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(
            options
        )
        return result


def check_vnni_extra(config, m, n, k, alpha, num_threads, **kwargs):
    if not (config.input_dtype == torch.uint8 and config.input2_dtype == torch.int8):
        raise AssertionError(
            "expected input_dtype == torch.uint8 and input2_dtype == torch.int8, "
            f"got {config.input_dtype}, {config.input2_dtype}"
        )
    vnni_size = 4
    return k % vnni_size == 0


@register_micro_gemm(
    *generate_gemm_config(
        VecAVX512VNNI,
        # (block_m, block_n, block_k)
        [(6, 64, 4)],
        input_dtype=torch.uint8,
        input2_dtype=torch.int8,
        output_dtype=torch.int32,
        compute_dtype=torch.int32,
        extra_check=check_vnni_extra,
    ),
)
class CppMicroGemmAVX512VNNI(CppMicroGemm):
    """
    This class generates the code for micro gemm using AVX512 VNNI instructions for compute.
    It supports u8s8s32 GEMM only.
    AVX512_VNNI ISA has been available since the 3rd gen of Intel Xeon.
    """

    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    {{kernel.assert_function}}(N % {{block_n}} == 0, "N dimension must be multiple of {{block_n}}");
    {{kernel.assert_function}}(K % {{vnni_size}} == 0, "K dimension must be multiple of {{vnni_size}}");
    constexpr int64_t M_BLOCK = {{block_m}};
    const int64_t M_TAIL = M % M_BLOCK;
    const int64_t M_MAIN = M - M_TAIL;
    for (int64_t m = 0; m < M_MAIN; m += M_BLOCK) {
        for (int64_t n = 0; n < N; n += {{block_n}}) {
            {{kernel_name}}_kernel<M_BLOCK, {{block_n}}, accum>(
                A + m * lda,
                B + n,
                C + m * ldc + n,
                K,
                lda,
                ldb,
                ldc
            );
        }
    }
    if (M_TAIL > 0) {
        switch (M_TAIL) {
{%- for m_tail in range(block_m - 1, 0, -1) %}
            case ({{m_tail}}):
                for (int64_t n = 0; n < N; n += {{block_n}}) {
                    {{kernel_name}}_kernel<{{m_tail}}, {{block_n}}, accum>(
                        A + M_MAIN * lda,
                        B + n,
                        C + M_MAIN * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                }
                break;
{%- endfor %}
            default:
                {{kernel.assert_function}}(false, "Unsupported M_TAIL");
        } // switch M_TAIL
    } // if M_TAIL
}
"""

    TEMPLATE_KERNEL = r"""
template <int64_t M, int64_t N, bool accum>
inline void {{kernel_name}}_kernel(
    const {{input_t}}* {{restrict_keyword}} A,
    const {{input2_t}}* {{restrict_keyword}} B,
    {{output_t}}* {{restrict_keyword}} C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    constexpr const int COLS = N / {{vec_len}};
    __m512i va;
    __m512i vb[COLS];
    __m512i vc[M * COLS];

    c10::ForcedUnroll<M * COLS>{}([&](auto i) { vc[i] = _mm512_setzero_epi32(); });

    auto compute = [&](auto i, int k) {
        constexpr const int row = i / COLS;
        constexpr const int col = i % COLS;

        if constexpr (col == 0) {
            va = _mm512_set1_epi32(*(int32_t*)(A + row * lda + k));
        }

        if constexpr (row == 0) {
            // B block in VNNI layout: [K / {{vnni_size}}, N, {{vnni_size}}]
            int64_t offset = k * ldb + col * {{vec_len}} * {{vnni_size}};
            vb[col] = _mm512_loadu_si512((__m512i const*)(B + offset));
        }
        vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    };

    // Accumulate along k
    constexpr const int k_unroll = 2;
    int k = 0;
    int k_limit = K / {{vnni_size}} / k_unroll;
    for (; k < k_limit; k++) {
        c10::ForcedUnroll<k_unroll>{}(
            [&](auto i) {
                c10::ForcedUnroll<M * COLS>{}(compute, {{vnni_size}} * (k * k_unroll + i));
            }
        );
    }
    k *= {{vnni_size}} * k_unroll;
    for (; k < K; k += {{vnni_size}}) {
        c10::ForcedUnroll<M * COLS>{}(compute, k);
    }

    // Store to C
    auto store_c = [&](auto i) {
        constexpr const int row = i / COLS;
        constexpr const int col = i % COLS;
        if constexpr (accum) {
            __m512i vc_old = _mm512_loadu_si512((__m512i const*)(C + row * ldc + col * {{vec_len}}));
            vc[i] = _mm512_add_epi32(vc[i], vc_old);
        }
        _mm512_storeu_si512((__m512i*)(C + row * ldc + col * {{vec_len}}), vc[i]);
    };
    c10::ForcedUnroll<M * COLS>{}(store_c);
}
"""

    def __init__(
        self,
        name,
        input_dtype,
        input2_dtype,
        output_dtype,
        compute_dtype,
        register_blocking,
        alpha=1,
    ) -> None:
        super().__init__(
            name,
            input_dtype,
            input2_dtype,
            output_dtype,
            compute_dtype,
            register_blocking,
            alpha,
        )
        if not (input_dtype == torch.uint8 and input2_dtype == torch.int8):
            raise AssertionError(
                f"Only u8s8s32 GEMM is supported by AVX512VNNI microkernel, got A:{input_dtype}, B:{input2_dtype}, C:{output_dtype}."
            )

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            "kernel": kernel,
            "block_m": self.register_blocking.block_m,
            "block_n": self.register_blocking.block_n,
            "block_k": self.register_blocking.block_k,
            "restrict_keyword": get_restrict_keyword(),
            "vec_len": 16,  # = 512 / 32 for C
            **self.get_common_options(),
        }
        return KernelTemplate._template_from_string(self.TEMPLATE_KERNEL).render(
            options
        ) + KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(options)

    def get_b_layout(self):
        return LayoutType.VNNI4


# extra check for CppMicroGemmAMX
def check_amx_extra(config, m, n, k, alpha, num_threads, **kwargs):
    vnni_size = 4 if config.input_dtype in [torch.uint8, torch.int8] else 2
    return k % vnni_size == 0 and alpha == 1


def check_int8_bf16_amx_extra(config, m, n, k, alpha, num_threads, **kwargs):
    # We need avx512_bf16 to dequant int8 to bf16
    vec_isa = kwargs.get("vec_isa")
    if vec_isa is None:
        raise AssertionError("expected vec_isa to be set, got None")
    return vec_isa.is_avx512_bf16_supported() and check_amx_extra(
        config, m, n, k, alpha, num_threads, **kwargs
    )


# amx_fp16 need to be checked separately since it is not always supported when amx is supported
def check_amx_fp16_extra(config, m, n, k, alpha, num_threads, **kwargs):
    if not (config.input_dtype == torch.float16 and config.output_dtype == torch.float):
        raise AssertionError(
            "expected input_dtype == torch.float16 and output_dtype == torch.float, "
            f"got {config.input_dtype}, {config.output_dtype}"
        )
    vec_isa = kwargs.get("vec_isa")
    if vec_isa is None:
        raise AssertionError("expected vec_isa to be set, got None")
    vnni_size = 2
    return vec_isa.is_amx_fp16_supported() and k % vnni_size == 0 and alpha == 1


@register_micro_gemm(
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 64), (48, 16, 64)],
        input_dtype=torch.int8,
        input2_dtype=torch.int8,
        output_dtype=torch.int32,
        compute_dtype=torch.int32,
        extra_check=check_amx_extra,
    ),
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 32), (48, 16, 32)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.int8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_int8_bf16_amx_extra,
    ),
    *generate_gemm_config(
        VecAMX,
        [(32, 16, 32), (32, 32, 32), (48, 16, 32), (16, 48, 32)],
        input_dtype=torch.bfloat16,
        output_dtype=torch.float,
        extra_check=check_amx_extra,
    ),
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 32), (48, 16, 32), (16, 48, 32)],
        input_dtype=torch.float16,
        output_dtype=torch.float,
        extra_check=check_amx_fp16_extra,
    ),
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 64), (48, 16, 64)],
        input_dtype=torch.uint8,
        input2_dtype=torch.int8,
        output_dtype=torch.int32,
        compute_dtype=torch.int32,
        extra_check=check_amx_extra,
    ),
)
class CppMicroGemmAMX(CppMicroGemm):
    """
    This class generates the code for micro gemm using Advanced Matrix extension (AMX)
    instructions available in 4th generation Intel Xeon for compute.
    It supports input types of torch.bfloat16 with fp32 output.
    """

    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    {{kernel.assert_function}}(N % {{block_n}} == 0, "N dimension must be multiple of {{block_n}}");
    {{kernel.assert_function}}(K % 2 == 0, "K dimension must be multiple of 2");
{%- if pack_vnni_B_locally %}
    {{template.codegen_allocate_weight_buffer("packed_B_buf", input2_t, "K", block_n)}}
{%- endif %}
{%- if use_cached_dequantized_B %}
    // Create a stack-allocated buffer for tiles of B.
    // Except maybe for the tail-case, an AMX tile of B has 16x32 BF16 elements.
    // we cache K * {{block_n}} elements of dequantized B
    {{template.codegen_allocate_weight_buffer("dequantized_B_buf", input_t, "K", block_n)}}
    const auto buf_size = K * {{block_n}};
    auto load_dequantized_B = [&](int base_idx) {
        // Load a tile of B & cache it in L1D.
        {{input2_t}}* base_addr = const_cast<{{input2_t}}*>(B) + base_idx;
        for (int idx_dq = 0, idx_q = 0; idx_dq < buf_size; idx_q += ldb, idx_dq += {{block_n}}) {
        {%- for vec_idx in range(0, block_n, 32) %}
            _mm_prefetch(base_addr + idx_q + 64 * ldb, _MM_HINT_T0);
            {%- if (block_n - vec_idx) >= 32 %}
            // 1) Load 32 x int8
            __m256i v8  = _mm256_loadu_si256((const __m256i*)(base_addr + idx_q + {{vec_idx}}));
            // 2) Extract two halves
            __m128i v8_lo = _mm256_extracti128_si256(v8, 0);
            __m128i v8_hi = _mm256_extracti128_si256(v8, 1);
            // 3) Widen each half to i32
            __m512i v32_lo = _mm512_cvtepi8_epi32(v8_lo);
            __m512i v32_hi = _mm512_cvtepi8_epi32(v8_hi);
            // 4) Convert to f32
            __m512 f_lo = _mm512_cvtepi32_ps(v32_lo);
            __m512 f_hi = _mm512_cvtepi32_ps(v32_hi);
            // 5) f32 -> bf16 (round-to-nearest-even) and pack 32 lanes to 512b
            // Packs the second operand (f_lo) into the lower 16 bf16 lanes and the first (f_hi) into the upper 16.
            __m512i bf = (__m512i)_mm512_cvtne2ps_pbh(f_hi, f_lo);
            // 6) Store 32 x bf16 (512 bits)
            _mm512_storeu_si512((__m512i*)(dequantized_B_buf + idx_dq + {{vec_idx}}), bf);
            {%- elif (block_n - vec_idx) >= 16 %}
            // 1) Load 16 x int8 (128 bits)
            __m128i v8 = _mm_loadu_si128((const __m128i*)(base_addr + idx_q + {{vec_idx}}));
            // 2) Widen: 16 x i8 -> 16 x i32
            __m512i v32 = _mm512_cvtepi8_epi32(v8);
            // 3) Convert to f32
            __m512 f32 = _mm512_cvtepi32_ps(v32);
            // 4) Convert f32 -> bf16 (round-to-nearest-even)
            __m256i bf16 = (__m256i)_mm512_cvtneps_pbh(f32);
            // 5) Store 16 x bf16 (256 bits)
            _mm256_storeu_si256((__m256i*)(dequantized_B_buf + idx_dq + {{vec_idx}}), bf16);
            {%- else %}
            auto b_int8_tail = at::vec::Vectorized<int8_t>::loadu(
                base_addr + idx_q + {{block_n - (block_n % 32)}},
                static_cast<int64_t>({{block_n % 32}})
            );
            auto b_bf16_tail = at::vec::convert<{{input_t}}>(b_int8_tail);
            b_bf16_tail.store(
                dequantized_B_buf + idx_dq + {{block_n - (block_n % 32)}},
                static_cast<int64_t>({{block_n % 32}})
            );
            {%- endif %}
        {%- endfor %}
        }
    };
{%- endif %}
// The ldb would not be block_n if N != block_n
{%- if use_cached_dequantized_B or pack_vnni_B_locally %}
    const int64_t updated_ldb = {{block_n}};
{%- else %}
    const int64_t updated_ldb = ldb;
{%- endif %}
    // TODO(jgong5): loop unroll for M and N
    for (int64_t n = 0; n < N; n += {{block_n}}) {
{%- if pack_vnni_B_locally %}
        // Pack non-constant weights into VNNI interleaved format in packed_B_buf
        at::vec::pack_vnni2(B + n, packed_B_buf, ldb, K, {{block_n}});
{%- elif use_cached_dequantized_B %}
        // Dequantize K * block_n int8 B elements into BF16
        load_dequantized_B(n);
{%- endif %}
        for (int64_t m = 0; m < M; m += {{block_m}}) {
            int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
            int64_t m_tail = m;
{%- for num_rows in range(block_m, 0, -16) %}
    {%- if num_rows != block_m %}
            else
    {%- endif %}
            if (block_m >= {{num_rows}}) {
                {{kernel_name}}_amx_kernel_{{num_rows}}_{{num_columns}}<accum>(
                    amx_state,
                    A + m * lda,
{%- if use_cached_dequantized_B %}
                    dequantized_B_buf,
{%- elif pack_vnni_B_locally %}
                    packed_B_buf,
{%- else %}
                    B + n,
{%- endif %}
                    C + m * ldc + n,
                    K,
                    lda,
                    updated_ldb,
                    ldc,
                    16
                );
                block_m -= {{num_rows}};
                m_tail += {{num_rows}};
            }
{%- endfor %}
            if (block_m > 0) {
                {{kernel_name}}_amx_kernel_16_{{num_columns}}<accum>(
                    amx_state,
                    A + m_tail * lda,
{%- if use_cached_dequantized_B %}
                    dequantized_B_buf,
{%- elif pack_vnni_B_locally %}
                    packed_B_buf,
{%- else %}
                    B + n,
{%- endif %}
                    C + m_tail * ldc + n,
                    K,
                    lda,
                    updated_ldb,
                    ldc,
                    block_m
                );
            }
        }
    }
}
"""

    TEMPLATE_KERNEL = r"""

template <bool accum, bool prefetch=false>
inline void {{kernel_name}}_amx_kernel_{{num_rows}}_{{num_columns}}(
    AMXState& amx_state,
    const {{input_t}}* {{restrict_keyword}} A,
{%- if use_cached_dequantized_B %}
    const {{input_t}}* {{restrict_keyword}} B,
{%- else %}
    const {{input2_t}}* {{restrict_keyword}} B,
{%- endif %}
    {{output_t}}* {{restrict_keyword}} C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    uint8_t tilecfg_rows
) {
    // TODO(jgong5): add prefetch hint for A, B, C
    auto loadconfig = [](const amx_tilecfg& cfg) {
        _tile_loadconfig(&cfg);
    };
    const auto last_k_offset = K / {{block_k}} * {{block_k}};
    const auto tail_k_size = K - last_k_offset;
    if C10_LIKELY (last_k_offset > 0) {
        amx_state.configure(tilecfg_rows, 64, {{num_rows}} / 16, {{num_columns}}, loadconfig);
    } else {
        amx_state.configure(tilecfg_rows, tail_k_size * sizeof({{input_t}}), {{num_rows}} / 16, {{num_columns}}, loadconfig);
    }
    auto load_c = [&]() {
{%- for tile_row in range(num_rows // 16) %}
    {%- for tile_col in range(num_columns) %}
        {%- set tile_idx = tile_row * num_columns + tile_col %}
        _tile_loadd({{tile_idx}}, C + {{tile_row * 16}} * ldc + {{tile_col * 16}}, ldc * sizeof({{output_t}}));
    {%- endfor %}
{%- endfor %}
    };
    auto zero_c = [&]() {
{%- for tile_row in range(num_rows // 16) %}
    {%- for tile_col in range(num_columns) %}
        {%- set tile_idx = tile_row * num_columns + tile_col %}
        _tile_zero({{tile_idx}});
    {%- endfor %}
{%- endfor %}
    };

    if constexpr (accum) {
        load_c();
    } else {
        zero_c();
    }

    auto compute = [&](int k) {
{%- set tile_offset_a = num_rows // 16 * num_columns %}
{%- set tile_offset_b = tile_offset_a + num_rows // 16 %}
{%- for tile_row in range(num_rows // 16) %}
    {%- for tile_col in range(num_columns) %}
        {%- set tile_idx_a = tile_offset_a + tile_row %}
        {%- set tile_idx_b = tile_offset_b + tile_col %}
        {%- set tile_idx_c = tile_row * num_columns + tile_col %}
        {%- if tile_col == 0 %}
        _tile_stream_loadd({{tile_idx_a}}, A + {{tile_row * 16}} * lda + k, lda * sizeof({{input_t}}));
        {%- endif %}
        {%- if tile_row == 0 %}
        _tile_loadd({{tile_idx_b}}, B + k * ldb + {{tile_col * 16 * vnni_size}}, ldb * {{vnni_size}} * sizeof({{input_t}}));
        {%- endif %}
        {%- if int8_gemm %}
            {%- if input_dtype == torch.int8 %}
        _tile_dpbssd({{tile_idx_c}}, {{tile_idx_a}}, {{tile_idx_b}});
            {%- else %}
        _tile_dpbusd({{tile_idx_c}}, {{tile_idx_a}}, {{tile_idx_b}});
            {%- endif %}
        {%- else %}
            {%- if input_dtype == torch.float16 %}
        _tile_dpfp16ps({{tile_idx_c}}, {{tile_idx_a}}, {{tile_idx_b}});
            {%- else %}
        _tile_dpbf16ps({{tile_idx_c}}, {{tile_idx_a}}, {{tile_idx_b}});
            {%- endif %}
        {%- endif %}
    {%- endfor %}
{%- endfor %}
    };

    {{kernel.unroll_pragma(4)}}
    for (int k = 0; k < last_k_offset; k += {{block_k}}) {
        compute(k);
    }

    auto store_c = [&]() {
    // store to C
{%- for tile_row in range(num_rows // 16) %}
    {%- for tile_col in range(num_columns) %}
        {%- set tile_idx = tile_row * num_columns + tile_col %}
        _tile_stored({{tile_idx}}, C + {{tile_row * 16}} * ldc + {{tile_col * 16}}, ldc * sizeof({{output_t}}));
    {%- endfor %}
{%- endfor %}
    };

    // TODO(jgong5): move tail k computation to separate loopnest to save tile configuration overhead
    if C10_UNLIKELY (tail_k_size > 0) {
        if C10_LIKELY (last_k_offset > 0) {
            store_c();
            amx_state.configure(tilecfg_rows, tail_k_size * sizeof({{input_t}}), {{num_rows}} / 16, {{num_columns}}, loadconfig);
            load_c();
        }
        compute(last_k_offset);
    }

    store_c();
}
"""

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        block_m, block_n, block_k = self.register_blocking
        if block_m % 16 != 0:
            raise AssertionError("Only support block_m % 16 == 0 for AMX")
        if block_n % 16 != 0:
            raise AssertionError("Only support block_n % 16 == 0 for AMX")
        if self.input_dtype in [torch.uint8, torch.int8]:
            if block_k != 64:
                raise AssertionError("Only support block_k = 64 for AMX INT8")
        else:
            if block_k != 32:
                raise AssertionError(
                    "Only support block_k = 32 for AMX Bfloat16/Float16"
                )
        num_columns = block_n // 16
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            "use_cached_dequantized_B": (
                self.input_dtype == torch.bfloat16
                and self.input2_dtype in [torch.int8, torch.uint8]
            ),
            "kernel": kernel,
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_columns": num_columns,
            "restrict_keyword": get_restrict_keyword(),
            **self.get_common_options(),
        }
        result = ""
        for num_rows in range(block_m, 0, -16):
            amx_kernel_options = {**options, "num_rows": num_rows}
            result += KernelTemplate._template_from_string(self.TEMPLATE_KERNEL).render(
                amx_kernel_options
            )
        result += KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(
            options
        )
        return result

    def codegen_init(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        return "AMXState amx_state;"

    def codegen_finalize(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        return "amx_state.release([]() { _tile_release(); });"

    def get_kernel_extra_args_declare(self) -> str:
        return "AMXState& amx_state,"

    def get_kernel_extra_args(self, **kwargs) -> list[str]:
        return ["amx_state,"]

    def get_b_layout(self):
        if self.input_dtype in [torch.uint8, torch.int8]:
            return LayoutType.VNNI4
        else:
            return LayoutType.VNNI2


# extra check for CppMicroBrgemm
def check_brgemm_extra(config, m, n, k, alpha, num_threads, **kwargs):
    if not (config.input_dtype == torch.half and config.output_dtype == torch.float):
        raise AssertionError(
            "expected input_dtype == torch.half and output_dtype == torch.float, "
            f"got {config.input_dtype}, {config.output_dtype}"
        )
    vnni_size = 2
    # use brgemm for Half when amx_fp16 is supported
    return torch.cpu._is_amx_fp16_supported() and k % vnni_size == 0 and alpha == 1


@register_micro_gemm(
    *generate_gemm_config(
        VecAMX,
        [(32, 32, 32), (48, 16, 32), (16, 48, 32)],
        input_dtype=torch.half,
        output_dtype=torch.float,
        extra_check=check_brgemm_extra,
    ),
)
class CppMicroBrgemm(CppMicroGemm):
    """
    This class generates the code for micro gemm using oneDNN brgemm.
    It supports input types of torch.half.
    """

    TEMPLATE_ENTRY = r"""
#include <ATen/native/CPUBlas.h>
{{declare_kernel}} {
{%- if pack_vnni_B_locally %}
    {{template.codegen_allocate_weight_buffer("packed_B_buf", input2_t, "K * N")}}
    at::vec::pack_vnni2(B, packed_B_buf, ldb, K, N);
{%- endif %}
    at::native::cpublas::brgemm(
      M, N, K,
    {%- if pack_vnni_B_locally %}
      lda, N, ldc,
    {%- else %}
      lda, ldb, ldc,
    {%- endif %}
      accum,
      A,
    {%- if pack_vnni_B_locally %}
      packed_B_buf,
    {%- else %}
      B,
    {%- endif %}
      C);
}
"""

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            "kernel": kernel,
            "block_m": self.register_blocking.block_m,
            "block_n": self.register_blocking.block_n,
            "block_k": self.register_blocking.block_k,
            "restrict_keyword": get_restrict_keyword(),
            **self.get_common_options(),
        }
        result = ""
        result += KernelTemplate._template_from_string(self.TEMPLATE_ENTRY).render(
            options
        )
        return result

    def codegen_finalize(
        self,
        kernel: CppTemplateKernel,
    ) -> str:
        return "at::native::cpublas::brgemm_release();"

    def get_b_layout(self):
        if not (self.input_dtype == torch.half and torch.cpu._is_amx_fp16_supported()):
            raise AssertionError(
                "expected input_dtype == torch.half and AMX fp16 support, "
                f"got {self.input_dtype}"
            )
        return LayoutType.VNNI2


def check_woq_int4_extra(config, m, n, k, alpha, num_threads, **kwargs):
    if alpha != 1:
        return False
    q_group_size = kwargs.get("q_group_size")
    if q_group_size is None:
        raise AssertionError("expected q_group_size to be set, got None")
    if (
        q_group_size not in [32, 64, 128]
        or k % q_group_size != 0
        or config.register_blocking.block_k > q_group_size
    ):
        return False
    return k % config.register_blocking.block_k == 0 and n % 64 == 0


@register_micro_gemm(
    # TODO: support float/half input
    *generate_gemm_config(
        VecAVX512,
        [(4, 64, 32), (4, 64, 64), (4, 64, 128)],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.uint8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_woq_int4_extra,
    ),
)
class CppMicroGemmWoQInt4Avx512(CppMicroGemmFP32Vec):
    """
    This class generates the code for WoQ int4 micro gemm using AVX512 intrinsics.
    It is based on the corresponding ATen kernel.
    Shape of packed weight = [N // 64, K, 32], viewed as [N, K // 2]
    Shape of packed ScalesAndZeros = [K // group_size, N, 2]
    """

    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    {{kernel.assert_function}}(N % {{block_n}} == 0, "N dimension must be multiple of {{block_n}}");
    {{kernel.assert_function}}(K % {{block_k}} == 0, "K dimension must be multiple of {{block_k}}");
    auto group_size = q_group_size;
    for (int64_t m = 0; m < M; m += {{block_m}}) {
        int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
        for (int64_t n = 0; n < N; n += {{block_n}}) {
            if (block_m == {{block_m}}) {
                {{kernel_name}}_kernel<{{block_m}}, {{block_n}}, accum>(
                    A + m * lda,
                    reinterpret_cast<const uint8_t*>(B) + n * ldb,
                    C + m * ldc + n,
                    K,
                    lda,
                    /* ldb */ {{block_n}} / 2,
                    ldc,
                    group_size,
                    ScaleAndZeros + n * 2,
                    lds,
                    k_start
                );
            } else {
                switch (block_m) {
                {%- for b in range(block_m - 1, 0, -1) %}
                case {{b}}:
                    {{kernel_name}}_kernel<{{b}}, {{block_n}}, accum>(
                        A + m * lda,
                        reinterpret_cast<const uint8_t*>(B) + n * ldb,
                        C + m * ldc + n,
                        K,
                        lda,
                        /* ldb */ {{block_n}} / 2,
                        ldc,
                        group_size,
                        ScaleAndZeros + n * 2,
                        lds,
                        k_start
                    );
                    break;
                {%- endfor %}
                default:
                    {{kernel.assert_function}}(false, "Unsupported block_m");
                }
            }
        }
    }
}
"""

    TEMPLATE_KERNEL = r"""
inline bool {{kernel_name}}_is_block_start(int index, int k_start, int group_size) {
  return (k_start + index) % group_size == 0;
}

inline __m128i {{kernel_name}}_convert_int4_to_int8(const uint8_t* data) {
  __m128i tmp = _mm_loadu_si64((const __m128i*)data);
  __m128i bytes = _mm_cvtepu8_epi16(tmp);
  const __m128i lowMask = _mm_set1_epi8(0xF);
  __m128i high = _mm_andnot_si128(lowMask, bytes);
  __m128i low = _mm_and_si128(lowMask, bytes);
  high = _mm_slli_epi16(high, 4);
  bytes = _mm_or_si128(low, high);
  return bytes;
}

template <int64_t BLOCK_M, int64_t BLOCK_N, bool accum>
inline void {{kernel_name}}_kernel(
    const {{input_t}}* {{restrict_keyword}} A,
    const uint8_t* {{restrict_keyword}} B,
    {{output_t}}* {{restrict_keyword}} C,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t q_group_size,
    const at::BFloat16* {{restrict_keyword}} ScaleAndZeros,
    int64_t lds, // leading dimension of ScaleAndZeros
    int64_t k_start) {
  constexpr int BLOCK_K = {{block_k}};
  constexpr int ROWS = BLOCK_M;
  constexpr int COLS = BLOCK_N / 16;

  const int PREFETCH_SIZE_K = 16 * 4;
  const int PREFETCH_SIZE_KB = (PREFETCH_SIZE_K + BLOCK_K - 1) / BLOCK_K;

  // number of blocks on K
  const int KB = K / BLOCK_K;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[ROWS * COLS];
  __m512 scale[COLS];
  __m512 zero[COLS];

  // Lookup table to de-quantize int4 values to bf16.
  // Values are dequantized as truly int4 [-8, 7] range;
  //
  // dequant = (bf16(int4_value) * bf16_scale) + bf16_zero
  //
  static const __m512 lut = _mm512_set_ps(
      7.0f, 6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f, 0.0f,
      -1.0f, -2.0f, -3.0f, -4.0f,
      -5.0f, -6.0f, -7.0f, -8.0f);

  // index for transpose
  static const __m512i idx1 = _mm512_set_epi32(
      30, 28, 26, 24, 22, 20, 18, 16,
      14, 12, 10, 8, 6, 4, 2, 0);
  static const __m512i idx2 = _mm512_set_epi32(
      31, 29, 27, 25, 23, 21, 19, 17,
      15, 13, 11, 9, 7, 5, 3, 1);

  // load scale and zero point
  auto load_scale_and_zeros = [&](int i, int _kb) {
    // load 2x bfloat16 vector
    __m512i t = _mm512_loadu_si512((__m512i*)(ScaleAndZeros + _kb * lds + 32 * i));
    _mm_prefetch(ScaleAndZeros + (_kb + PREFETCH_SIZE_KB) * lds + 32 * i, _MM_HINT_T0);

    // convert to 2x f32 vector
    __m512 a, b;
    at::vec::cvtbf16_fp32(t, a, b);

    // transpose scale_and_zero from {16, 2} to {2, 16}
    // inputs:
    //   a: {s0, z0, s1, z1, ..., s7, z7}
    //   b: {s8, z8, s9, z9, ..., s15, z15}
    // output:
    //   scale: {s0, s1, s2, ..., s15}
    //   zero:  {z0, z1, z2, ..., z15}
    scale[i] = _mm512_mask_permutex2var_ps(a, 0xffff, idx1, b);
    zero[i] = _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b);
  };

  auto loadc = [&](auto i) {
    if constexpr (accum) {
       constexpr int row = i / COLS;
       constexpr int col = i % COLS;
       vc[i] = _mm512_loadu_ps(C + row * ldc + col * 16);
    } else {
      vc[i] = _mm512_setzero_ps();
    }
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);

  auto compute = [&, COLS](auto i, int k) {
    constexpr  int row = i / COLS;
    constexpr  int col = i % COLS;

    if constexpr (col == 0) {
      float aa = static_cast<float>(A[row * lda + k]);
      _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      va = _mm512_set1_ps(aa);
    }

    if constexpr (row == 0) {
      if constexpr (COLS == 4) {
        // when BLOCK_N = 64, handle each row at a time
        // to reduce de-quantize overhead.
        if constexpr (col == 0) {
          __m256i b4 = _mm256_loadu_si256((__m256i*)(B + k * ldb));
          _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb, _MM_HINT_T0);

          __m512i b32 = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(b4));
          vb[0] = _mm512_permutexvar_ps(b32, lut);
          vb[0] = _mm512_fmadd_ps(vb[0], scale[0], zero[0]);
          vb[2] = _mm512_permutexvar_ps(_mm512_srli_epi32(b32, 4), lut);
          vb[2] = _mm512_fmadd_ps(vb[2], scale[2], zero[2]);

          b32 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(b4, 1));
          vb[1] = _mm512_permutexvar_ps(b32, lut);
          vb[1] = _mm512_fmadd_ps(vb[1], scale[1], zero[1]);
          vb[3] = _mm512_permutexvar_ps(_mm512_srli_epi32(b32, 4), lut);
          vb[3] = _mm512_fmadd_ps(vb[3], scale[3], zero[3]);
        }
      } else {
        __m128i b8 = {{kernel_name}}_convert_int4_to_int8(B + k * ldb + col * 8);
        __m512i b32 = _mm512_cvtepu8_epi32(b8);
        vb[col] = _mm512_permutexvar_ps(b32, lut);
        vb[col] = _mm512_fmadd_ps(vb[col], scale[col], zero[col]);
      }
    }

    constexpr int idx = row * COLS + col;
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
  };

  for (int k = 0, kb = 0; k < K; ++k) {
    if ({{kernel_name}}_is_block_start(k, k_start, q_group_size)) {
      c10::ForcedUnroll<COLS>{}(load_scale_and_zeros, kb++);
    }
    c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  //store to C
  auto storec = [&, COLS](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    _mm512_storeu_ps(C + row * ldc + col * 16, vc[i]);
  };
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
}
"""

    def get_kernel_extra_args_declare(self) -> str:
        return (
            "const int64_t q_group_size,\n"
            "    const at::BFloat16* __restrict__ ScaleAndZeros,\n"
            "    const int64_t lds,\n"
            "    int64_t k_start,"
        )

    def get_kernel_extra_args(self, **kwargs) -> list[str]:
        if "kernel" not in kwargs:
            raise AssertionError("expected 'kernel' in kwargs")
        if "qscale_and_zeros" not in kwargs:
            raise AssertionError("expected 'qscale_and_zeros' in kwargs")
        kernel = kwargs["kernel"]
        qscale_and_zeros = kwargs["qscale_and_zeros"]
        return [
            "group_size,",
            f"&({kernel.index(qscale_and_zeros, [0, 0, 0])}),",
            "N * 2,",  # lds
            "k_start,",
        ]

    def is_woq_int4(self):
        return True


@register_micro_gemm(
    *generate_gemm_config(
        VecAMX,
        [  # (block_m, block_n, block_k)
            (16, 32, 32),
            (32, 32, 32),
        ],
        input_dtype=torch.bfloat16,
        input2_dtype=torch.uint8,
        output_dtype=torch.float,
        compute_dtype=torch.float,
        extra_check=check_amx_extra,
    ),
)
class CppMicroGemmWoQInt4Amx(CppMicroGemmAMX):
    """
    This class generates the code for WoQ int4 micro gemm using AMX intrinsics,
    which are available on 4th and newer generations of Intel Xeon.
    Shape of packed weight = [N // 32, K, 16], viewed as [N, K // 2]
    Shape of packed ScalesAndZeros = [K // group_size, N, 2]
    Reuse TEMPLATE_KERNEL of CppMicroGemmAMX.
    """

    TEMPLATE_ENTRY = r"""
inline bool {{kernel_name}}_is_block_start(int index, int k_start, int group_size) {
  // check if (k_start + index) % group_size == 0, assuming group_size = 32/64/128
  return ((k_start + index) & (group_size - 1)) == 0;
}

{{declare_kernel}} {
    {{kernel.assert_function}}(N % {{block_n}} == 0, "N dimension must be multiple of {{block_n}}");
    {{kernel.assert_function}}(K % 2 == 0, "K dimension must be multiple of 2");
    {{kernel.assert_function}}({{block_n}} == 32, "block_n must be 32 for WOQ int4");

    // Create a stack-allocated buffer for tiles of B.
    // Except maybe for the tail-case, an AMX tile of B has 16x32 BF16 elements.
    // we cache K * {{block_n}} elements of dequantized B
    {{template.codegen_allocate_weight_buffer("dequantized_B_buf", input_t, "K", block_n)}}

    constexpr int BLOCK_K = {{block_k}};
    constexpr int64_t BLOCK_N = {{block_n}};
    constexpr int COLS = BLOCK_N / 16;
    const int PREFETCH_SIZE_K = 16 * 4;
    const int PREFETCH_SIZE_KB = (PREFETCH_SIZE_K + BLOCK_K - 1) / BLOCK_K;
    const int KB = K / BLOCK_K;

    __m512i b32[COLS * 2];
    __m512 vb[COLS * 2];
    __m512 scale[COLS];
    __m512 zero[COLS];

    // Lookup table to de-quantize int4 values to bf16.
    // Values are dequantized as truly int4 [-8, 7] range;
    //
    // dequant = (bf16(int4_value) * bf16_scale) + bf16_zero
    //
    static const __m512 lut = _mm512_set_ps(
        7.0f, 6.0f, 5.0f, 4.0f,
        3.0f, 2.0f, 1.0f, 0.0f,
        -1.0f, -2.0f, -3.0f, -4.0f,
        -5.0f, -6.0f, -7.0f, -8.0f);

    // index for transpose
    static const __m512i idx1 = _mm512_set_epi32(
        30, 28, 26, 24, 22, 20, 18, 16,
        14, 12, 10, 8, 6, 4, 2, 0);
    static const __m512i idx2 = _mm512_set_epi32(
        31, 29, 27, 25, 23, 21, 19, 17,
        15, 13, 11, 9, 7, 5, 3, 1);

    // Indices for VNNI layout conversion
    __m512i idx_low = _mm512_set_epi32(
        0x17,
        0x07,
        0x16,
        0x06,
        0x15,
        0x05,
        0x14,
        0x04,
        0x13,
        0x03,
        0x12,
        0x02,
        0x11,
        0x01,
        0x10,
        0x00);
    __m512i idx_high = _mm512_set_epi32(
        0x1f,
        0x0f,
        0x1e,
        0x0e,
        0x1d,
        0x0d,
        0x1c,
        0x0c,
        0x1b,
        0x0b,
        0x1a,
        0x0a,
        0x19,
        0x09,
        0x18,
        0x08);

    // load scale and zero point
    auto load_scale_and_zeros = [&](int i, int _kb) {
        // load 2x bfloat16 vector
        __m512i t = _mm512_loadu_si512((__m512i*)(ScaleAndZeros + _kb * lds + 32 * i));
        _mm_prefetch(ScaleAndZeros + (_kb + PREFETCH_SIZE_KB) * lds + 32 * i, _MM_HINT_T0);

        // convert to 2x f32 vector
        __m512 a, b;
        at::vec::cvtbf16_fp32(t, a, b);

        // transpose scale_and_zero from {16, 2} to {2, 16}
        // inputs:
        //   a: {s0, z0, s1, z1, ..., s7, z7}
        //   b: {s8, z8, s9, z9, ..., s15, z15}
        // output:
        //   scale: {s0, s1, s2, ..., s15}
        //   zero:  {z0, z1, z2, ..., z15}
        scale[i] = _mm512_mask_permutex2var_ps(a, 0xffff, idx1, b);
        zero[i] = _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b);
    };

    // Dequantize a B block of 2 * block_n into bf16
    // So, it handles k and k+1 at the same time
    auto dequantize_B = [&](int n) {
        constexpr int64_t ldb_int4 = BLOCK_N / 2; // 16
        for (int k = 0, kb = 0; k < K; k += 2) {
            // Since block_k must be 32 for AMX microkernels, k_start may not be
            // a multiple of q_group_size. In that case, we need to load scales
            // and zero points immediately when k == 0 here
            if ({{kernel_name}}_is_block_start(k, k_start, q_group_size) || k == 0) {
                c10::ForcedUnroll<COLS>{}(load_scale_and_zeros, kb++);
            }

            _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb_int4, _MM_HINT_T0);

            // load 256 bits = 64 elements in int4
            __m128i b4 = _mm_loadu_si128((__m128i*)(B + n / 2 * K + k * ldb_int4));
            b32[0] = _mm512_cvtepu8_epi32(b4);
            b32[1] = _mm512_srli_epi32(b32[0], 4);
            vb[0] = _mm512_permutexvar_ps(b32[0] , lut);
            vb[0] = _mm512_fmadd_ps(vb[0], scale[0], zero[0]);
            vb[1] = _mm512_permutexvar_ps(b32[1], lut);
            vb[1] = _mm512_fmadd_ps(vb[1], scale[1], zero[1]);

            __m128i b4_2 = _mm_loadu_si128((__m128i*)(B + n / 2 * K + (k + 1) * ldb_int4));
            b32[0 + COLS] = _mm512_cvtepu8_epi32(b4_2);
            b32[1 + COLS] = _mm512_srli_epi32(b32[0 + COLS], 4);
            vb[0 + COLS] = _mm512_permutexvar_ps(b32[0 + COLS] , lut);
            vb[0 + COLS] = _mm512_fmadd_ps(vb[0 + COLS], scale[0], zero[0]);
            vb[1 + COLS] = _mm512_permutexvar_ps(b32[1 + COLS], lut);
            vb[1 + COLS] = _mm512_fmadd_ps(vb[1 + COLS], scale[1], zero[1]);

            for (int i = 0; i < COLS; i++) {
                // convert to VNNI
                auto low = _mm512_permutex2var_ps(vb[i], idx_low, vb[i + COLS]);
                auto high = _mm512_permutex2var_ps(vb[i], idx_high, vb[i + COLS]);
                // convert lower 16 float32 values to bfloat16
                auto v0_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(low));
                // convert higher 16 float32 values to bfloat16
                auto v1_bf16 = reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(high));
                // combine the lower 16 and higher 16 bfloat16 values
                auto v = _mm512_castsi256_si512(v0_bf16);
                v = _mm512_inserti64x4(v, v1_bf16, 1);
                // store the VNNI format bfloat16 values
                {{input_t}}* addr = dequantized_B_buf + k * 32 + (i % 2) * 32;
                _mm512_storeu_si512(addr, v);
            }
        }
    };

    for (int64_t n = 0; n < N; n += {{block_n}}) {
        // Dequantize K * block_n int8 B elements into BF16
        dequantize_B(n);
        for (int64_t m = 0; m < M; m += {{block_m}}) {
            int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
            int64_t m_tail = m;
        {%- for num_rows in range(block_m, 0, -16) %}
            {%- if num_rows != block_m %}
            else
        {%- endif %}
            if (block_m >= {{num_rows}}) {
                {{kernel_name}}_amx_kernel_{{num_rows}}_{{num_columns}}<accum>(
                    amx_state,
                    A + m * lda,
                    dequantized_B_buf + n * K,
                    C + m * ldc + n,
                    K,
                    lda,
                    {{block_n}},
                    ldc,
                    16
                );
                block_m -= {{num_rows}};
                m_tail += {{num_rows}};
            }
        {%- endfor %}
            if (block_m > 0) {
                {{kernel_name}}_amx_kernel_16_{{num_columns}}<accum>(
                    amx_state,
                    A + m_tail * lda,
                    dequantized_B_buf + n * K,
                    C + m_tail * ldc + n,
                    K,
                    lda,
                    {{block_n}},
                    ldc,
                    block_m
                );
            }
        } // for m
    } // for n
}
"""

    def get_kernel_extra_args_declare(self) -> str:
        return (
            "AMXState& amx_state,\n"
            "    const int64_t q_group_size,\n"
            "    const c10::BFloat16* __restrict__ ScaleAndZeros,\n"
            "    const int64_t lds,\n"
            "    int64_t k_start,"
        )

    def get_kernel_extra_args(self, **kwargs) -> list[str]:
        if "kernel" not in kwargs:
            raise AssertionError("expected 'kernel' in kwargs")
        if "qscale_and_zeros" not in kwargs:
            raise AssertionError("expected 'qscale_and_zeros' in kwargs")
        kernel = kwargs["kernel"]
        qscale_and_zeros = kwargs["qscale_and_zeros"]
        return [
            "amx_state,",
            "group_size,",
            f"&({kernel.index(qscale_and_zeros, [0, 0, 0])}),",
            "N * 2,",  # lds
            "k_start,",
        ]

    def is_woq_int4(self):
        return True


def create_micro_gemm(
    name,
    m,
    n,
    k,
    input_dtype,
    input2_dtype,
    output_dtype=None,
    compute_dtype=None,
    alpha=1,
    num_threads=-1,
    use_ref=True,
    q_group_size=None,
) -> CppMicroGemm | None:
    """
    Based on the provided info, try to find the config of the micro-kernel that would
    deliver the best performance in terms of lower latency for this case.
    """

    def create_from_config(cls, config: CppMicroGemmConfig):
        return cls(
            name,
            config.input_dtype,
            config.input2_dtype,
            config.output_dtype,
            config.compute_dtype,
            config.register_blocking,
            alpha,
        )

    def skip_amx_kernel_for_woq(dynamic_M):
        # For WoQ GEMM, AMX micro-kernel may not perform well if m is small.
        # Exception: for dynamic shapes, we consider using the AMX micro-kernel.
        if (
            dynamic_M
            or input_dtype != torch.bfloat16
            or input2_dtype not in [torch.int8, torch.uint8]
        ):
            return False
        m_threshold = 5
        return m < m_threshold

    if not (isinstance(n, int) or n.is_number):
        raise AssertionError(n)
    if not (isinstance(k, int) or k.is_number):
        raise AssertionError(k)
    from ..utils import has_free_symbols

    dynamic_M = has_free_symbols((m,))
    m = V.graph.sizevars.optimization_hint(m, fallback=1)
    if not (isinstance(m, int) or m.is_number):
        raise AssertionError(m)
    if output_dtype is None:
        output_dtype = input_dtype
    if compute_dtype is None:
        compute_dtype = output_dtype
    if num_threads < 0:
        num_threads = parallel_num_threads()
    vec_isa = pick_vec_isa()
    matched_configs = []
    for cls, configs in micro_gemm_configs.items():
        for config in configs:
            if not issubclass(vec_isa.__class__, config.vec_isa_cls):
                continue
            if (
                config.input_dtype == input_dtype
                and config.compute_dtype == compute_dtype
                and config.input2_dtype == input2_dtype
                and config.output_dtype == output_dtype
                # The output_dtype here is the output dtype of the micro-kernel.
                # In some cases, the actual output dtype of the op for which the micro-kernel
                # is being created would be same as that of the activation, but the micro-kernels
                # compute output in Float/int32, which is converted in the GEMM template. This is
                # subject to change in the future.
            ):
                if config.extra_check is not None and not config.extra_check(
                    config,
                    m,
                    n,
                    k,
                    alpha,
                    num_threads,
                    dynamic_M=dynamic_M,
                    q_group_size=q_group_size,
                    vec_isa=vec_isa,
                ):
                    continue
                block_m, block_n, block_k = config.register_blocking
                if config.vec_isa_cls == VecAMX and skip_amx_kernel_for_woq(dynamic_M):
                    continue
                # Criteria on the ranking of configurations
                # 1. ISA: AMX > VNNI > VEC
                # 2. Dividable by block sizes (block_m, block_n, block_k)
                # 3. Number of mxn blocks is large enough to occupy all the threads
                # 4. Register blocks are larger
                isa_score = 0
                if config.vec_isa_cls == VecAMX:
                    isa_score += 2
                elif config.vec_isa_cls == VecAVX512VNNI:
                    isa_score += 1
                dividable_score = 0
                if m % block_m == 0:
                    dividable_score += 1
                if n % block_n == 0:
                    dividable_score += 1
                if k % block_k == 0:
                    dividable_score += 1
                occupancy_score = 0
                n_blocks = (n + block_n - 1) // block_n
                total_mxn_blocks = n_blocks * ((m + block_m - 1) // block_m)
                if n_blocks >= num_threads:
                    occupancy_score += 1
                if total_mxn_blocks >= num_threads:
                    occupancy_score += 1
                register_bytes = (
                    block_m * block_n * config.compute_dtype.itemsize
                    + (block_m * block_k + block_k * block_n)
                    * config.input_dtype.itemsize
                )
                size_score = register_bytes
                # if number of mxn blocks can not occupy all the threads,
                # we favor smaller register blocks.
                if occupancy_score == 0:
                    size_score = 0 - register_bytes
                matched_configs.append(
                    (
                        (isa_score, dividable_score, occupancy_score, size_score),
                        cls,
                        config,
                    )
                )
    if len(matched_configs) == 0:
        if use_ref:
            return CppMicroGemmRef(
                name, input_dtype, input2_dtype, output_dtype, compute_dtype, alpha
            )
        else:
            return None
    # TODO(jgong5): allow autotuning on choices of configs
    return create_from_config(*max(matched_configs, key=operator.itemgetter(0))[1:])
<|endoftext|># mypy: allow-untyped-defs
import logging
import os
import re
import shutil

import torch
from torch._inductor import config
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.cpp_builder import _set_gpu_runtime_env, _transform_cuda_paths
from torch._inductor.utils import is_linux
from torch.utils._ordered_set import OrderedSet


if config.is_fbcode():
    from triton.fb.build import build_paths


log = logging.getLogger(__name__)
autotuning_log = torch._logging.getArtifactLogger(__name__, "autotuning")


def use_re_build() -> bool:
    """
    Use for CUTLASS compilation only right now.
    """
    if config.is_fbcode() and not cuda_env.nvcc_exist(_cuda_compiler()):
        from triton.fb.re_build_helper import should_build_locally

        return not should_build_locally()
    return False


def _cutlass_path() -> str | None:
    if config.is_fbcode():
        from libfb.py import parutil

        return parutil.get_dir_path("cutlass-4-headers")
    else:
        from torch._inductor.codegen.cutlass.utils import try_import_cutlass

        return config.cutlass.cutlass_dir if try_import_cutlass() else None


def _cutlass_paths() -> list[str]:
    return [
        "include",
        "tools/library/include",
        "tools/library/src",
        "tools/util/include",
    ]


def _clone_cutlass_paths(build_root: str) -> list[str]:
    cutlass_root = _cutlass_path()
    if cutlass_root is None:
        return []
    paths = []
    for path in _cutlass_paths():
        old_path = os.path.join(cutlass_root, path)
        new_path = os.path.join(build_root, path)
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
        paths.append(new_path)
    return paths


def _cutlass_include_paths() -> list[str]:
    cutlass_root = _cutlass_path()
    if cutlass_root is None:
        return []
    return [
        # Use realpath to get canonical absolute paths, in order not to mess up cache keys
        os.path.realpath(os.path.join(cutlass_root, path))
        for path in _cutlass_paths()
    ]


def _cuda_compiler() -> str | None:
    if cuda_env.nvcc_exist(config.cuda.cuda_cxx):
        return config.cuda.cuda_cxx
    if config.is_fbcode():
        return os.path.join(build_paths.sdk_home, "bin", "nvcc")
    if cuda_env.nvcc_exist(os.getenv("CUDACXX")):
        return os.getenv("CUDACXX", "")
    if cuda_env.nvcc_exist(os.getenv("CUDA_HOME")):
        return os.path.realpath(os.path.join(os.getenv("CUDA_HOME", ""), "bin/nvcc"))
    return "nvcc"


def _cuda_lib_options() -> list[str]:
    """
    Util function for CUTLASS backend to find the correct CUDA libraries.
    """
    _set_gpu_runtime_env()  # cpp_extension consults the env
    from torch.utils import cpp_extension

    lpaths = cpp_extension.library_paths(device_type="cuda")
    if use_re_build():
        lpaths += [
            build_paths.sdk_lib,
            os.path.join(build_paths.sdk_lib, "stubs"),
        ]
    extra_ldflags: list[str] = []
    if is_linux():
        _transform_cuda_paths(lpaths)
        for path in lpaths:
            if "torch/lib" in path:
                # don't want to depend on pytorch
                continue
            extra_ldflags.append(f"-L{path}")
            # -rpath ensures the DLL can find its dependencies when loaded, even
            # if the library path is non-standard.
            # But do not add the stubs folder to rpath as the driver is expected to be found at runtime
            if os.path.basename(path) != "stubs":
                extra_ldflags.extend(["-Xlinker", f"-rpath={path}"])
        extra_ldflags.append("-lcuda")
        extra_ldflags.append("-lcudart")
    else:
        raise NotImplementedError(
            "Unsupported env, failed to find cuda libs! Currently only Linux is supported."
        )
    return extra_ldflags


def _nvcc_host_compiler_options() -> list[str]:
    return [
        "-fPIC",
        "-fno-strict-aliasing",
        "-fvisibility=hidden",
        "-Wconversion",
    ]


def _cuda_arch_with_compile_suffix(arch: str) -> str:
    arch = _normalize_cuda_arch(arch)
    if arch == "90":
        # Required by cutlass compilation.
        return "90a"
    if arch == "100":
        return "100a"
    if arch == "101":
        return "101a"
    if arch == "103":
        return "103a"
    if arch == "107":
        return "107a"
    if arch == "110":
        return "110a"
    if arch == "120":
        return "120a"
    if arch == "121":
        return "121a"
    return arch


def _nvcc_arch_as_compile_option() -> str | None:
    arch = cuda_env.get_cuda_arch()
    if arch is None:
        return arch
    return _cuda_arch_with_compile_suffix(arch)


def _nvcc_arch_as_compile_option_or_raise() -> str:
    arch = _nvcc_arch_as_compile_option()
    if arch is None:
        raise RuntimeError("Unable to determine CUDA architecture")
    return arch


def _normalize_cuda_arch(arch: str) -> str:
    arch = arch.removeprefix("sm_").removeprefix("compute_").replace(".", "")
    if not re.fullmatch(r"\d+[a-z]?", arch):
        raise ValueError(f"Unrecognized CUDA arch: {arch}")
    return arch


def _cuda_arch_number(arch: str) -> int:
    arch = _normalize_cuda_arch(arch)
    if arch[-1].isalpha():
        arch = arch[:-1]
    return int(arch)


def _cuda_arch_suffix(arch: str) -> str:
    arch = _normalize_cuda_arch(arch)
    return arch[-1] if arch[-1].isalpha() else ""


def _cuda_arch_same_generation(arch: str, other: str) -> bool:
    arch = _normalize_cuda_arch(arch)
    other = _normalize_cuda_arch(other)
    if _cuda_arch_suffix(arch) or _cuda_arch_suffix(other):
        return arch == other
    return _cuda_arch_number(arch) == _cuda_arch_number(other)


def _cuda_arch_is_compatible_with_current(arch: str, current_arch: str) -> bool:
    if _cuda_arch_suffix(current_arch):
        return _normalize_cuda_arch(arch) == _normalize_cuda_arch(current_arch)
    return _cuda_arch_number(arch) >= _cuda_arch_number(current_arch)


def _aoti_cuda_target_arch() -> str:
    if config.cuda.arch is not None:
        return _cuda_arch_with_compile_suffix(str(config.cuda.arch))
    return _nvcc_arch_as_compile_option_or_raise()


def _parse_gencode_options(flags: list[str]) -> OrderedSet[tuple[str, str]]:
    options: OrderedSet[tuple[str, str]] = OrderedSet()
    for flag in flags:
        if flag.startswith("-gencode="):
            option = flag.removeprefix("-gencode=")
        elif flag.startswith("-gencode "):
            option = flag.removeprefix("-gencode ")
        else:
            continue

        try:
            _, code = option.split(",code=", 1)
        except ValueError:
            continue
        code = code.removeprefix("[").removesuffix("]")
        for entry in code.split(","):
            try:
                kind, arch = entry.split("_", 1)
            except ValueError:
                continue
            if kind in ("sm", "compute"):
                options.add((kind, _normalize_cuda_arch(arch)))
    return options


def _cuda_multi_arch_gencode_options(current_arch: str | None = None) -> list[str]:
    """
    Return nvcc -gencode option payloads for AOTI CUDA fatbins.

    AOTI captures PTX for the architecture Triton compiled on. That PTX can be
    used for the current architecture and newer architectures, but not older
    ones, so explicit TORCH_CUDA_ARCH_LIST entries below the current target are
    intentionally ignored.
    """
    if not current_arch:
        current_arch = _nvcc_arch_as_compile_option_or_raise()
    current_arch = _normalize_cuda_arch(current_arch)
    options: OrderedSet[tuple[str, str]] = OrderedSet()

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list and arch_list != "native":
        from torch.utils.cpp_extension import _get_cuda_arch_flags

        for kind, arch in _parse_gencode_options(_get_cuda_arch_flags([])):
            if kind == "sm" and _cuda_arch_is_compatible_with_current(
                arch, current_arch
            ):
                options.add((kind, arch))
            elif kind == "sm":
                log.warning(
                    "Ignoring TORCH_CUDA_ARCH_LIST entry sm_%s for AOTI CUDA "
                    "multi-arch packaging because it is not compatible with "
                    "target arch %s.",
                    arch,
                    current_arch,
                )

    if not any(
        kind == "sm" and _cuda_arch_same_generation(arch, current_arch)
        for kind, arch in options
    ):
        options.add(("sm", current_arch))

    # Always keep a PTX image for the compile architecture as the fallback for
    # GPUs newer than the generated SASS images.
    options.add(("compute", current_arch))

    def sort_key(option: tuple[str, str]) -> tuple[int, int, str]:
        kind, arch = option
        return (_cuda_arch_number(arch), 0 if kind == "sm" else 1, arch)

    return [
        f"arch=compute_{arch},code={kind}_{arch}"
        for kind, arch in sorted(options, key=sort_key)
    ]


def _cuda_gencode_options_have_non_current_sass(
    gencode_options: list[str], current_arch: str | None = None
) -> bool:
    if not current_arch:
        current_arch = _nvcc_arch_as_compile_option_or_raise()
    current_arch = _normalize_cuda_arch(current_arch)
    for kind, arch in _parse_gencode_options(
        [f"-gencode={option}" for option in gencode_options]
    ):
        if kind == "sm" and not _cuda_arch_same_generation(arch, current_arch):
            return True
    return False


def _nvcc_compiler_options() -> list[str]:
    arch = _nvcc_arch_as_compile_option_or_raise()
    code = [f"sm_{arch}", f"compute_{arch}"]
    if config.cuda.enable_cuda_lto:
        code += [f"lto_{arch}"]
    options = [
        "-t=0",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES=1",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-w",
        f"-gencode=arch=compute_{arch},code=[{','.join(code)}]",
        config.cutlass.compile_opt_level,
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-DNDEBUG",
    ]
    if config.is_fbcode():
        options.extend(["-ccbin", os.path.dirname(build_paths.gcc)])
    if config.cutlass.enable_debug_info:
        options.extend(["-lineinfo", "-g", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"])
    if config.cuda.enable_ptxas_info:
        options.extend(
            [
                "--keep",  # Keep the intermediate files for debugging (including ptx, sass, cubin etc.)
                "--ptxas-options=--warn-on-local-memory-usage",  # warn us if local memory is used in CUDA Kernels
                "--ptxas-options=--warn-on-spills",  # warn us if register spilling happens in CUDA Kernels
                "--resource-usage",  # Report on CUDA resource usage (shared mem, registers etc.)
                "--source-in-ptx",
            ]
        )  # Annotate the ptx file with source information
    if config.cutlass.use_fast_math:
        options.extend(
            [
                "--use_fast_math",
                "-DCUTLASS_USE_TANH_FOR_SIGMOID=1",
            ]
        )
    return options


def cuda_compile_command(
    src_files: list[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: list[str] | None = None,
) -> str:
    if extra_args is None:
        extra_args = []
    if use_re_build():
        build_path = os.path.dirname(dst_file)
        include_paths = _clone_cutlass_paths(build_path)
        src_files = [os.path.basename(src_file) for src_file in src_files]
        dst_file = os.path.basename(dst_file)
    else:
        include_paths = _cutlass_include_paths()
    cuda_lib_options = _cuda_lib_options()
    nvcc_host_compiler_options = _nvcc_host_compiler_options()
    nvcc_compiler_options = _nvcc_compiler_options()
    options = (
        nvcc_compiler_options
        + extra_args
        + [
            f"-Xcompiler {opt}" if "=" in opt else f"-Xcompiler={opt}"
            for opt in nvcc_host_compiler_options
        ]
        + ["-I" + path for path in include_paths]
        + cuda_lib_options
    )
    src_file = " ".join(src_files)
    res = ""
    if dst_file_ext == "o":
        res = f"{_cuda_compiler()} {' '.join(options)} -c -o {dst_file} {src_file}"
    elif dst_file_ext == "so":
        options.append("-shared")
        res = f"{_cuda_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    elif dst_file_ext == "exe":
        res = f"{_cuda_compiler()} {' '.join(options)} -o {dst_file} {src_file}"
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("CUDA command: %s", res)
    else:
        autotuning_log.debug("CUDA command: %s", res)
    return res
<|endoftext|>from __future__ import annotations

import torch

from ...runtime.hints import get_warp_size
from ..common import (
    DeviceIdx,
    DeviceOpOverrides,
    register_device_op_overrides,
    TritonScratchWorkspace,
)


class CUDADeviceOpOverrides(DeviceOpOverrides):
    """
    CUDA-specific codegen functions, see DeviceOpOverrides for details
    """

    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"

    def set_device(self, device_idx: DeviceIdx) -> str:
        return f"torch.cuda.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.cuda.synchronize()"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        return f"torch.cuda._DeviceGuard({device_idx})"

    def current_device_idx_expr(self) -> str:
        return "torch.cuda.current_device()"

    def current_stream(self) -> str:
        return "torch.cuda.current_stream()"

    def cpp_device_guard(self) -> str:
        return "at::cuda::CUDAGuard"

    def cpp_aoti_device_guard(self) -> str:
        return "AOTICudaGuard"

    def cpp_stream_guard(self) -> str:
        return "at::cuda::CUDAStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "AOTICudaStreamGuard"

    def cpp_getStreamFromExternal(self) -> str:
        return "at::cuda::getStreamFromExternal"

    def kernel_header(self) -> str:
        source_codes = """
        #include <c10/cuda/CUDAGuard.h>
        #include <c10/cuda/CUDAStream.h>
        #include <ATen/cuda/EmptyTensor.h>
        """
        return source_codes

    def kernel_driver(self) -> str:
        """Return C++ host-side helpers (loadKernel, launchKernel, CUDA_DRIVER_CHECK)
        embedded in AOTI-generated wrapper code."""
        # NVIDIA devices have a warp size of 32, while AMD devices can have a
        # warp size of 32 or 64 depending on the architecture.
        source_codes = """
            #define CUDA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                CUresult code = EXPR;                          \\
                const char *msg;                               \\
                CUresult code_get_error = cuGetErrorString(code, &msg); \\
                if (code_get_error != CUDA_SUCCESS) {          \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string("invalid error code!"));   \\
                }                                              \\
                if (code != CUDA_SUCCESS) {                    \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string(msg));                     \\
                }                                              \\
            } while (0);

            static inline CUfunction loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt,
                    std::vector<CUmodule>* loaded_modules = nullptr) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                CUmodule mod;
                CUfunction func;
                CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
                if (loaded_modules) {
                    loaded_modules->push_back(mod);
                }
                CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                        func,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline CUfunction loadKernel(
                    const void* start,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    std::vector<CUmodule>* loaded_modules = nullptr) {
                CUmodule mod;
                CUfunction func;
                CUDA_DRIVER_CHECK(cuModuleLoadData(&mod, start));
                if (loaded_modules) {
                    loaded_modules->push_back(mod);
                }
                CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                        func,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline void launchKernel(
                    CUfunction func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    cudaStream_t stream) {
                CUDA_DRIVER_CHECK(cuLaunchKernel(
                    func, gridX, gridY, gridZ, __WARP_SIZE__*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
        """
        if torch.version.hip is not None and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
            warp_size = get_warp_size(device)
        else:
            warp_size = 32
        return source_codes.replace("__WARP_SIZE__", str(warp_size))

    def tma_descriptor_helpers(self) -> str:
        """
        CUDA helper functions for initializing TMA Descriptors on host side
        """
        if torch.version.hip is not None:
            raise RuntimeError("Host-side TMA descriptors not supported on HIP.")

        # helper functions for initializing 1D and 2D TMA descriptors in C++. borrowed from the Triton code here:
        # Old APIs (fill(1|2)DTMADescriptor):
        # https://github.com/triton-lang/triton/blob/6af4f88591c85de079d8a36a4d7dba67918e2b39/third_party/nvidia/backend/driver.c#L283
        # New APIs (fillTMADescriptor):
        # https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.c#L283
        return """
            #if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000
            [[maybe_unused]] static void init1DTMADescriptor(
                    CUtensorMap* m,
                    void* globalAddress,
                    uint64_t dim,
                    uint32_t blockDim,
                    uint32_t elementSize) {
                uint64_t dims[1] = {dim};
                uint64_t globalStrides[1] = {dim * elementSize};
                uint32_t tensorDims[1] = {blockDim};
                uint32_t elementStrides[1] = {1};

                CUtensorMapDataType type;
                switch (elementSize) {
                case 1:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
                    break;
                case 2:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
                    break;
                case 4:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
                    break;
                default:
                    throw std::runtime_error("elementSize must be 1, 2, or 4");
                }

                if (elementSize * blockDim < 32) {
                    throw std::runtime_error("block size too small");
                }

                int rank = 1;

                CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
                    m, type, rank, globalAddress, dims,
                    globalStrides, tensorDims, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
            }

            [[maybe_unused]] static void init2DTMADescriptor(
                    CUtensorMap* m,
                    void* globalAddress,
                    uint64_t dim1,
                    uint64_t dim0,
                    uint32_t blockDim1,
                    uint32_t blockDim0,
                    uint32_t elementSize) {
                uint64_t dims[2] = {dim0, dim1};
                uint32_t tensorDims[2] = {blockDim0, blockDim1};
                uint64_t globalStrides[2] = {dims[0] * elementSize,
                                             dims[0] * dims[1] * elementSize};
                uint32_t elementStrides[2] = {1, 1};

                CUtensorMapDataType type;
                switch (elementSize) {
                case 1:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
                    break;
                case 2:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
                    break;
                case 4:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
                    break;
                default:
                    throw std::runtime_error("elementSize must be 1, 2, or 4");
                }

                int rank = 2;

                CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
                uint32_t contigDimSizeInByte = elementSize * tensorDims[0];
                if (contigDimSizeInByte >= 128) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
                } else if (contigDimSizeInByte >= 64) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
                } else if (contigDimSizeInByte >= 32) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
                } else {
                    throw std::runtime_error("block size too small");
                }

                if (contigDimSizeInByte > 128) {
                    tensorDims[0] = 128 / elementSize;
                }

                CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
                    m, type, rank, globalAddress, dims,
                    globalStrides, tensorDims, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
            }

            [[maybe_unused]] static void initTMADescriptor(
                CUtensorMap* m,
                void* globalAddress,
                int elemSize,
                int rank,
                uint32_t* blockSize,
                uint64_t* shape,
                uint64_t* stride
            ) {
                uint32_t elementStrides[5] = {1, 1, 1, 1, 1};
                uint32_t blockSizeInt[5];
                uint64_t shapeInt[5];
                uint64_t stridesLL[5];

                // Reorder blockSize (reverse the order)
                for (int i = 0; i < rank; ++i) {
                    blockSizeInt[rank - i - 1] = blockSize[i];
                }

                // Reorder shape (reverse the order)
                for (int i = 0; i < rank; ++i) {
                    shapeInt[rank - i - 1] = shape[i];
                }

                // Reorder and calculate strides
                for (int i = 0; i + 1 < rank; ++i) {
                    stridesLL[rank - i - 2] = elemSize * stride[i];
                }
                stridesLL[rank - 1] =
                    shapeInt[rank - 1] * (rank == 1 ? elemSize : stridesLL[rank - 2]);

                CUtensorMapDataType type;
                // In Triton this is computed ahead of time; but for simplicity
                // in the PyTorch version we copied this code from the old
                // TMA API handling (i.e. init2DTMADescriptor)
                switch (elemSize) {
                case 1:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
                    break;
                case 2:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
                    break;
                case 4:
                    type = CU_TENSOR_MAP_DATA_TYPE_UINT32;
                    break;
                default:
                    throw std::runtime_error("elemSize must be 1, 2, or 4");
                }

                // Calculate the size of the most contiguous dimension in bytes
                CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
                uint32_t contigDimSizeInByte = elemSize * blockSizeInt[0];
                if (rank == 1) {
                    // rank 1 should not be swizzled
                    swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
                } else if (contigDimSizeInByte >= 128) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
                } else if (contigDimSizeInByte >= 64) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_64B;
                } else if (contigDimSizeInByte >= 32) {
                    swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
                } else {
                    throw std::runtime_error("block size too small");
                }

                CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
                    m, type, rank, globalAddress,
                    shapeInt, stridesLL, blockSizeInt, elementStrides,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, (CUtensorMapSwizzle)swizzle,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
            }

            struct StableTMADescriptor {
                CUtensorMap m;
                uint32_t block_shape[5];
                uint64_t global_shape[5];
                uint64_t strides[5];
            };
            #endif
        """

    def cpp_stream_type(self) -> str:
        return "cudaStream_t"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_get_current_cuda_stream"

    def cpp_kernel_type(self) -> str:
        return "CUfunction"

    def cpp_device_ptr(self) -> str:
        return "CUdeviceptr"

    def cpp_scratch(
        self, idx: int, workspace: TritonScratchWorkspace, prefix: str | None = None
    ) -> tuple[list[str], str] | None:
        prefix = f"{prefix}_" if prefix else ""
        var_name = f"{prefix}scratch_{idx}"
        if workspace.size > 0:
            size_expr = (
                f"static_cast<int64_t>({workspace.size}) * grid_0 * grid_1 * grid_2"
            )
            size_array = f"int64_t {var_name}_size[] = {{{size_expr}}};"
            stride_array = f"int64_t {var_name}_stride[] = {{1}};"
            device_type = "cached_torch_device_type_cuda"
            device_idx = "device_idx_"

            return (
                [
                    f"{size_array}",
                    f"{stride_array}",
                    f"AtenTensorHandle {var_name}_handle;",
                    (
                        f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, {var_name}_size, {var_name}_stride, "
                        f"{workspace.generate_dtype_str()}, {device_type}, {device_idx}, &{var_name}_handle));"
                    ),
                    f"RAIIAtenTensorHandle {var_name}_tensor({var_name}_handle);",
                    f"CUdeviceptr {var_name} = reinterpret_cast<CUdeviceptr>({var_name}_tensor.data_ptr());",
                ],
                var_name,
            )
        else:
            return [f"CUdeviceptr {var_name} = 0;"], var_name


register_device_op_overrides("cuda", CUDADeviceOpOverrides())
<|endoftext|>import functools
import logging
import shutil

import torch
from torch._inductor.utils import clear_on_fresh_cache

from ... import config


log = logging.getLogger(__name__)


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_arch() -> str | None:
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            # Get Compute Capability of the first Visible device
            major, minor = torch.cuda.get_device_capability(0)
            return str(major * 10 + minor)
        return str(cuda_arch)
    except Exception:
        log.exception("Error getting cuda arch")
        return None


@clear_on_fresh_cache
@functools.lru_cache(1)
def is_datacenter_blackwell_arch() -> bool:
    arch = get_cuda_arch()
    if arch is None:
        return False
    arch_number = int(arch)
    return arch_number >= 100 and arch_number < 110


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_version() -> str | None:
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception:
        log.exception("Error getting cuda version")
        return None


@functools.cache
def nvcc_exist(nvcc_path: str | None = "nvcc") -> bool:
    return nvcc_path is not None and shutil.which(nvcc_path) is not None
<|endoftext|># mypy: allow-untyped-defs
import hashlib
import logging
from collections.abc import Sequence
from typing import cast, TypeGuard

import sympy

from torch._inductor.codegen.cutlass.python_evt import (
    CutlassEVTCodegen,
    MockCutlassHandler,
)
from torch._inductor.utils import Placeholder
from torch.utils._ordered_set import OrderedSet

from ...._dynamo.utils import counters
from ... import config
from ...codecache import code_hash, get_path
from ...ir import Buffer, ComputedBuffer, CUTLASSTemplateBuffer, Pointwise
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
    WhyNoFuse,
)
from ...utils import get_fused_kernel_name, get_kernel_metadata, sympy_product
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer


log = logging.getLogger(__name__)


class WhyNoFuseNames(WhyNoFuse):
    def __init__(self, name1: str, name2: str) -> None:
        self.name1 = name1
        self.name2 = name2


class CUTLASSScheduling(BaseScheduling):
    """
    Partial Scheduling implementation for cutlass C++ Kernels.
    This class is intended to be used in combination with TritonScheduling,
    and delegated to by CUDACombinedScheduling/XPUCombinedScheduling.

    It handles fusion decisions and cutlass C++ specific template code generation.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    @staticmethod
    def is_cutlass_template(node: BaseSchedulerNode) -> TypeGuard[SchedulerNode]:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, CUTLASSTemplateBuffer
        )

    def is_cutlass_fused_template(self, node: BaseSchedulerNode) -> bool:
        return isinstance(node, FusedSchedulerNode) and self.is_cutlass_template(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self.is_cutlass_template(node1) and isinstance(node2, BaseSchedulerNode):
            if not node1.node:
                raise AssertionError("node1.node should not be None")
            return self._can_fuse_epilogue_impl(
                cast(CUTLASSTemplateBuffer, node1.node),
                [],
                node2,  # type: ignore[arg-type]
            )
        elif self.is_cutlass_fused_template(node1) and isinstance(
            node2, BaseSchedulerNode
        ):
            if not node1.node:
                raise AssertionError("node1.node should not be None")
            if not node2.node:
                raise AssertionError("node2.node should not be None")
            fnode1 = cast(FusedSchedulerNode, node1)
            return self._can_fuse_epilogue_impl(
                fnode1.get_template_node(),  # type: ignore[arg-type]
                self._unwrap_epilogue_nodes(fnode1),
                node2,  # type: ignore[arg-type]
            )

        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )

            # use the original src_code as the key
            kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
            if fused_name == "fused":
                # no EVT kernel, use the original kernel name
                kernel_name = f"cutlass_{kernel_hash}"
            else:
                kernel_name = f"cutlass_{fused_name}_{kernel_hash}"
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

            _, _, kernel_path = get_path(code_hash(src_code), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.{V.graph.device_type}(r'''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline(
                f"''', 'so', aot_compile={str(V.graph.aot_mode)})"
            )

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )

            # For JIT cpp_wrapper, the kernel call site emits a bare
            # `extern "C"` symbol reference; compile the .so now so the
            # wrapper compile can link against it via extra_flags.
            # Autotune compiled a .so for the unfused kernel name
            # (e.g. cutlass_<hash>), but at codegen time the kernel name
            # gains a descriptive prefix (cutlass_fused_mm_<hash>), so the
            # exported symbol differs and we need a fresh compile.
            if V.graph.cpp_wrapper and not V.graph.aot_mode:
                from ...codecache import CUDACodeCache, XPUCodeCache

                codecache_cls = (
                    XPUCodeCache if V.graph.device_type == "xpu" else CUDACodeCache
                )
                so_path, _, _ = codecache_cls.compile(src_code, "so")
                wrapper.external_kernel_libs.add(so_path)
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a cutlass template, possibly with fused epilogues
        """
        counters["inductor"]["cutlass_epilogue_fusion_counter"] += len(epilogue_nodes)
        if not self.is_cutlass_template(template_node):
            raise AssertionError(
                "Template node passed to CUTLASSScheduling.codegen_template must be a SchedulerNode that wraps a CUTLASSTemplateBuffer"
            )
        _, (_numel, rnumel) = template_node.group
        if rnumel != 1:
            raise AssertionError(f"expected rnumel == 1, got {rnumel}")
        ctb: CUTLASSTemplateBuffer = cast(CUTLASSTemplateBuffer, template_node.node)
        epilogue_ir_nodes: list[Buffer] = [n.node for n in epilogue_nodes]  # type: ignore[misc]
        if not all(isinstance(n, ComputedBuffer) for n in epilogue_ir_nodes):
            raise AssertionError(
                "Epilogue nodes must all be instances of ir.ComputedBuffer"
            )
        kernel, render = ctb.make_kernel_render(  # type: ignore[misc]
            ctb, epilogue_nodes=epilogue_nodes
        )
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()

            # typically there is a codegen pass which runs after mark_run
            # for this kernel we've already generated the C++ code, but we still
            # need to let the kernel know about loads/stores that occur in the fused
            # kernel for memory planning to properly optimize allocations
            ctb.emulate_store_fn()
            for node in epilogue_ir_nodes:
                with V.set_ops_handler(MockCutlassHandler(V.get_ops_handler())):
                    # Not sure why we need to do this again
                    if not isinstance(node, ComputedBuffer):
                        raise AssertionError(
                            f"expected ComputedBuffer, got {type(node)}"
                        )
                    node.get_store_function()(CutlassEVTCodegen.get_index_vars(node))

        with V.set_kernel_handler(kernel):
            src_code = render()
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule)

        # debug printing values of intermediate tensors
        _, call_args, arg_signatures, _ = kernel.args.python_argdefs()
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args, kernel_name, arg_signatures, kernel
        )
        with debug_printer_manager:
            self.codegen_comment(node_schedule, kernel_name)
            kernel.call_kernel(kernel_name, ctb)

        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()

    @staticmethod
    def _unwrap_epilogue_nodes(
        fused_node: FusedSchedulerNode,
    ) -> list[BaseSchedulerNode]:
        nodes = fused_node.get_nodes()
        template_node = fused_node.get_template_node()
        if not all(n.node is not None for n in nodes):
            raise AssertionError("All epilogue nodes should have an IRNode")
        # pyrefly: ignore [redundant-cast]
        return cast(
            list[BaseSchedulerNode], [n for n in nodes if n.node is not template_node]
        )

    @staticmethod
    def _is_compatible_reshape(
        template_size: Sequence[sympy.Expr], node_size: Sequence[sympy.Expr]
    ) -> bool:
        """
        Check if node_size is a compatible reshape of template_size.
        This allows cases like template [8192, 3072] with node [16, 512, 3072]
        where [16*512, 3072] == [8192, 3072] (prefix dims multiply to match).
        """
        if len(node_size) <= len(template_size):
            return False
        # Try to merge consecutive node dims to reconstruct template dims
        t_idx = 0
        n_idx = 0

        def _matches_template_dim(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
            # Fast path for concrete/static equality before invoking simplify.
            if lhs == rhs:
                return True
            if V.graph.sizevars.statically_known_equals(lhs, rhs):
                return True
            return sympy.simplify(lhs - rhs) == 0

        while t_idx < len(template_size) and n_idx < len(node_size):
            product = node_size[n_idx]
            n_idx += 1
            # Try multiplying consecutive node dims until we match template dim
            while not _matches_template_dim(product, template_size[t_idx]):
                if n_idx >= len(node_size):
                    return False
                product = product * node_size[n_idx]
                n_idx += 1
            t_idx += 1
        return t_idx == len(template_size) and n_idx == len(node_size)

    def _can_fuse_epilogue_impl(
        self,
        cutlass_template_buffer: CUTLASSTemplateBuffer,
        existing_epilogue_nodes: list[BaseSchedulerNode],
        node_to_fuse: BaseSchedulerNode,
    ) -> bool:
        """
        Check if the given node can be fused with the epilogue. At the moment, Kernels
        support fusion with Pointwise operations, wrapped in (named) ComputedBuffer nodes.

        Args:
            cutlass_template_buffer : A CUTLASSTemplateBuffer object representing the CUTLASS template and it's result buffer
            existing_epilogue_nodes : List[SchedulerNode]: The list of already fused epilogue nodes.
            node_to_fuse: The SchedulerNode node to be checked if it can be fused with the epilogue.
        Returns:
        - bool: True if the given node can be fused with the epilogue, False otherwise.

        """
        why = WhyNoFuseNames(
            cutlass_template_buffer.get_name(), node_to_fuse.get_name()
        )

        scheduler_nodes_to_fuse = node_to_fuse.get_nodes()

        if not isinstance(cutlass_template_buffer, CUTLASSTemplateBuffer):
            raise AssertionError(
                f"expected CUTLASSTemplateBuffer, got {type(cutlass_template_buffer)}"
            )

        # Checks on constituent nodes
        for s_node in scheduler_nodes_to_fuse:
            node = s_node.node

            if not isinstance(node, ComputedBuffer):
                why(f"{node} is not a ComputedBuffer")
                return False
            elif not isinstance(node.data, Pointwise):
                why(f"{node} is not a Pointwise op")
                return False
            elif not node.get_computed_buffer_name():  # type: ignore[attr-defined]
                why(f"{node} does not have a computed buffer name")
                return False

            name = node.get_computed_buffer_name()  # type: ignore[attr-defined]
            # dtype can differ, and strides can differ as long as they are broadcastable
            # Allow compatible reshapes (e.g., [8192, 3072] vs [16, 512, 3072])
            # where the node's shape is a split of the template's shape
            if node.get_size() != cutlass_template_buffer.get_size():
                if not self._is_compatible_reshape(
                    cutlass_template_buffer.get_size(), node.get_size()
                ):
                    why(
                        f"{name}'s size: {node.get_size()} differs from {cutlass_template_buffer.get_name()}'s \
size: {cutlass_template_buffer.get_size()}"
                    )
                    return False

        if not (
            len(existing_epilogue_nodes)
            or cutlass_template_buffer.get_name()
            in OrderedSet([rd.name for rd in node_to_fuse.read_writes.reads])
        ):
            raise AssertionError(
                "First epilogue node must read from cutlass template buffer"
            )

        if node_to_fuse.has_aliasing_or_mutation():
            why(f"{node_to_fuse.get_name()} has aliasing or mutation")
            return False
        elif node_to_fuse.is_reduction():
            why(
                f"{node_to_fuse.get_name()} is a reduction which is not yet supported by EVT"
            )
            return False
        elif (
            not config.cutlass.cutlass_epilogue_fusion_enabled
            or not config.epilogue_fusion
        ):
            why("cutlass epilogue fusion is not enabled")
            return False
        elif not cutlass_template_buffer.supports_epilogue_fusion:
            why("epilogue fusion is only supported for TMA-enabled gemm ops")
            return False

        try:
            from torch._inductor.codegen.cutlass.python_evt import CutlassEVTCodegen

            read_names, _, _, _ = CutlassEVTCodegen.ir_to_evt_python_code(
                cutlass_template_buffer.get_name(),
                existing_epilogue_nodes + list(node_to_fuse.get_nodes()),
                OrderedSet(),
            )

        except NotImplementedError as e:
            not_implemented_op = str(e)
            if not_implemented_op.startswith("_op_"):
                not_implemented_op = not_implemented_op[4:]
                why(
                    f"Cannot fuse epilogue node {node_to_fuse} into {cutlass_template_buffer.name}, \
likely due to unsupported operation: {not_implemented_op}"
                )
                return False
            else:  # Likely due to unsupported dtype.
                why(
                    f"Cannot fuse epilogue node {node_to_fuse} into {cutlass_template_buffer.name}. \
Reason: {not_implemented_op}"
                )
                return False

        # An external read whose shape is a compatible reshape of the 2D GEMM
        # output is normalized to the template shape during rendering, but only
        # when it is contiguous (the row-major flatten is memory-equivalent).
        # A non-contiguous compatible-reshape read cannot be safely flattened,
        # and EVT would fail shape propagation against the (1, M, N) accumulator,
        # so reject the fusion in that case.
        template_size = cutlass_template_buffer.get_size()
        all_bufs = V.graph.name_to_buffer | V.graph.graph_inputs
        for read_name in read_names:
            buf = all_bufs.get(read_name)
            if (
                buf is not None
                and self._is_compatible_reshape(template_size, buf.get_size())
                and not buf.get_layout().is_contiguous()
            ):
                why(
                    f"external read {read_name} with size {buf.get_size()} is a "
                    f"non-contiguous compatible reshape of template size "
                    f"{template_size}, which CUTLASS EVT cannot align to the "
                    f"accumulator shape"
                )
                return False

        return True
<|endoftext|># mypy: allow-untyped-defs
import functools
import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from unittest.mock import patch

import sympy

import torch
from torch._inductor import config
from torch._inductor.utils import clear_on_fresh_cache, Placeholder
from torch._logging import getArtifactLogger

from ...autotune_process import CUTLASSBenchmarkRequest, TensorMeta
from ...ir import Buffer, CUTLASSTemplateBuffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .kernel import CUTLASSTemplateCaller, CUTLASSTemplateKernel
from .utils import DTYPE_TO_CUTLASS_TYPE


if TYPE_CHECKING:
    from ...scheduler import BaseSchedulerNode
else:
    BaseSchedulerNode = Any

GemmOperation = Any

autotuning_log = getArtifactLogger(__name__, "autotuning")


@dataclass(frozen=True)
class ArgInfo:
    name: str
    ty: str


@clear_on_fresh_cache
class CUTLASSTemplate(KernelTemplate):
    """
    CUTLASSTemplate is a class that provides a template for generating CUTLASS Templates. Used as a baseclass for the
    CUTLASSGemmTemplate, providing functionality that might also be relevant for non-GEMM CUTLASS Kernels.
    """

    index_counter = itertools.count()
    # dict of cache key to (code, size_args)
    code_cache: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]] = {}
    cache_clear = staticmethod(code_cache.clear)

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        input_reorder: list[int] | None = None,
    ) -> None:
        """
        Baseclass for CUTLASS C++ Templates, derived from KernelTemplate.
        Not to be instantiated directly.

        Args:
            name (str): The name of the CUTLASSTemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies
                the order of the input nodes.
        """
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        self.input_reorder = input_reorder
        self.layout = layout
        self.device_type = layout.device.type

    @classmethod
    @functools.lru_cache(None)
    # pyrefly: ignore [bad-override]
    def _template_from_string(cls, source: str) -> Any:
        return KernelTemplate._template_from_string(source)

    @staticmethod
    def supports_epilogue_fusion(op: GemmOperation, device_type: str) -> bool:
        return False

    def make_key(self, name: str, input_key: str, layout_repr: str) -> str:
        """
        Make a key for the code cache. The idea of the method is to cache
        everything that matters but doesn't include runtime param values, i.e.,
        self.get_runtime_arg_values().

        Args:
            kwargs: Additional keyword arguments. Including op (GemmOperation).
        """
        return hashlib.sha256(
            str(
                (
                    input_key,
                    self.input_reorder,
                    # output layout, same as self.output_node.get_layout()
                    layout_repr,
                    self.get_runtime_arg_info(),
                    name,
                )
            ).encode("utf-8")
        ).hexdigest()

    def generate_code_and_args(
        self, name: str, input_key: str, layout_repr: str, **kwargs
    ) -> tuple[str, tuple[int, ...]]:
        """
        Generate code and args with caching. We cache the code even if runtime
        args are different.
        """
        key: str | None = None
        if config.cutlass.enable_caching_codegen:
            key = self.make_key(name=name, input_key=input_key, layout_repr=layout_repr)

        if key is not None and key in self.code_cache:
            code, size_args, offset_args = self.code_cache[key]
            extra_args = tuple(
                list(size_args)
                + list(offset_args)
                + list(self.get_runtime_arg_values(**kwargs))
            )
            return code, extra_args

        kernel_name = str(Placeholder.KERNEL_NAME)
        kernel = CUTLASSTemplateKernel(
            kernel_name=kernel_name,
            runtime_arg_info=self.get_runtime_arg_info(),
            runtime_arg_values=self.get_runtime_arg_values(**kwargs),
            device_type=self.device_type,
        )
        with patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)):
            code = self.render(kernel=kernel, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            autotuning_log.debug("Generated Code:\n%s", code)
            autotuning_log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(DTYPE_TO_CUTLASS_TYPE),
                kernel.args.python_argdefs(),
            )

        input_reorder = (
            self.input_reorder
            if self.input_reorder is not None
            else list(range(len(self.input_nodes)))
        )
        expected_args = list(
            unique(self.input_nodes[idx].get_name() for idx in input_reorder)
        )
        expected_args.extend([self.output_node.get_name()])
        if list(call_args)[: len(expected_args)] != expected_args:
            raise AssertionError(
                (
                    call_args,
                    expected_args,
                )
            )
        # Resolve symbolic sizes to concrete ints for benchmarking only.
        V.graph.sizevars.optimization_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )
        size_args = V.graph.sizevars.optimization_hints(kernel.get_dynamic_shape_args())
        offset_args = V.graph.sizevars.optimization_hints(kernel.get_offset_args())

        if key is not None:
            self.code_cache[key] = code, size_args, offset_args

        # extra args has runtime params, which shouldn't be cached
        extra_args = tuple(
            list(size_args) + list(offset_args) + self.get_runtime_arg_values(**kwargs)
        )

        return code, extra_args

    def generate(  # type: ignore[override]
        self,
        name: str,
        description: str,
        input_key: str,
        layout_repr: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        **kwargs,
    ) -> CUTLASSTemplateCaller:
        """
        Generates the CUDA template caller object for the given GEMM template and operation.
        This CUTLASSTemplateCaller may be used to call and benchmark the generated CUDA kernel
        in a standalone manner to enable Autotuning.

        Args:
            description: op name followed by swizzle.
            kwargs: Additional keyword arguments.

        Returns:
            A CUTLASSTemplateCaller object representing the generated CUDA template caller.
        """
        code, extra_args = self.generate_code_and_args(
            name=name,
            input_key=input_key,
            layout_repr=layout_repr,
            **kwargs,
        )

        # not caching since kernel name is needed below
        kernel_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:8]
        kernel_name = f"cutlass_{kernel_hash}"
        code = code.replace(self.name, kernel_name)

        # create the BenchmarkRequest
        bmreq = CUTLASSBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=input_tensor_meta,
            output_tensor_meta=output_tensor_meta,
            extra_args=extra_args,
            source_code=code,
            device_type=self.device_type,
        )

        # kwargs has "op" argument in case of CUTLASSGemmTemplate
        op = kwargs["op"]
        if not op:
            supports_epilogue_fusion = False
        else:
            # epilogue fusion is only supported for TMA kernels
            supports_epilogue_fusion = self.supports_epilogue_fusion(
                op, self.device_type
            )

        def make_kernel_render(
            template_node: CUTLASSTemplateBuffer,
            epilogue_nodes: list[BaseSchedulerNode] | None = None,
        ) -> tuple[CUTLASSTemplateKernel, functools.partial[str]]:
            if not supports_epilogue_fusion and epilogue_nodes:
                raise AssertionError("epilogue fusion is not supported for this kernel")
            kernel = CUTLASSTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME),
                runtime_arg_info=self.get_runtime_arg_info(),
                runtime_arg_values=self.get_runtime_arg_values(**kwargs),
                device_type=self.device_type,
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            )
            return kernel, render

        return CUTLASSTemplateCaller(
            kernel_name,
            "cutlass_gemm",
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            supports_epilogue_fusion,
            self,
            kwargs,
            description,
        )

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                #include <exception>
                #include <iostream>
                #include <memory>
                #include <random>
                #include <vector>
            """
        )
        res.splice(
            """
                #include "cute/tensor.hpp"
                #include "cutlass/cutlass.h"
                #include "cutlass/numeric_types.h"
                #include "cutlass/tensor_ref.h"
                #include "cutlass/util/host_tensor.h"
                #include "cutlass/util/reference/host/tensor_fill.h"
                #include "cutlass/util/reference/device/tensor_fill.h"
                #include "cutlass/util/device_memory.h"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                // We compile all models with -fvisibility=hidden. Any symbols that need to be
                // exposed in the final shared library must be declared with PT_EXPORT to make
                // them visible.
                #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
                #define PT_EXPORT __attribute__((__visibility__("default")))
                #else
                #ifdef _WIN32
                #define PT_EXPORT __declspec(dllexport)
                #else
                #define PT_EXPORT
                #endif
                #endif
            """
        )
        res.splice(
            """
                using namespace cute;
                #define CUTLASS_CHECK(status)                                                      \\
                {                                                                                  \\
                  cutlass::Status error = status;                                                  \\
                  if (error != cutlass::Status::kSuccess) {                                        \\
                    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \\
                        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \\
                    throw std::runtime_error(msg);                                                 \\
                  }                                                                                \\
                }

                // Used as pass-through functor in EVT just for type casting / rounding
                template <typename T>
                struct identity_op {
                  CUTLASS_HOST_DEVICE
                  T operator()(T val) const { return val; }
                };

            """
        )
        return res

    def cute_int(self, int_str: str, var_name: str) -> str:
        res = ""
        if int_str in ("1", "1L"):
            res = "cute::Int<1>{}"
        else:
            res = int_str

        return f"{res} /* {var_name} */"

    _DTYPE_TO_CUTLASS = {
        torch.float32: "float",
        torch.float64: "double",
        torch.float16: "cutlass::half_t",
        torch.int32: "int32_t",
        torch.int16: "int16_t",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.bool: "bool",
        torch.bfloat16: "cutlass::bfloat16_t",
        torch.float8_e4m3fn: "cutlass::float_e4m3_t",
        torch.float8_e5m2: "cutlass::float_e5m2_t",
    }

    _DTYPE_TO_CUTLASS_SPARSE_META = {
        torch.int32: "uint32_t",
        torch.int16: "uint16_t",
    }

    def cutlass_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._DTYPE_TO_CUTLASS.get(node.get_dtype())}*)({ptr})"

    def cutlass_sparse_meta_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return (
                f"({self._DTYPE_TO_CUTLASS_SPARSE_META.get(node.get_dtype())}*)({ptr})"
            )

    def render(self, **kwargs) -> str:
        raise NotImplementedError

    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return [ArgInfo("swizzle", "const uint8_t")]

    def get_runtime_arg_values(self, **kwargs) -> list[Any]:
        """
        Helper method to retrieve runtime args from generate kwargs
        """
        return [kwargs[arg.name] for arg in self.get_runtime_arg_info()]
<|endoftext|># mypy: allow-untyped-defs
import atexit
import functools
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing_extensions import TypeIs

import sympy

import torch
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.utils import clear_on_fresh_cache
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...ir import Layout
from ...runtime.runtime_utils import cache_dir
from ...virtualized import V
from ..cuda.cuda_env import get_cuda_arch, get_cuda_version
from ..xpu.xpu_env import get_xpu_arch, get_xpu_version


log = logging.getLogger(__name__)

CUTLASS_OPERATION_KIND: str = "gemm"
ACCUMULATOR_DTYPES: OrderedSet[torch.dtype] = OrderedSet([torch.float, torch.int32])
XW_DTYPES: OrderedSet[torch.dtype] = OrderedSet(
    [torch.half, torch.bfloat16, torch.float8_e4m3fn, torch.int8, torch.float8_e5m2]
)


@atexit.register
def move_cutlass_compiled_cache() -> None:
    """Move CUTLASS compiled cache file to the cache directory if it exists."""
    if try_import_cutlass.cache_info().currsize == 0:
        return

    try:
        import cutlass_cppgen  # type: ignore[import-not-found]
    except ImportError:
        return

    # Check if the CACHE_FILE attribute exists in cutlass_cppgen and if the file exists
    if not hasattr(cutlass_cppgen, "CACHE_FILE") or not os.path.exists(
        cutlass_cppgen.CACHE_FILE
    ):
        return

    try:
        filename = os.path.basename(cutlass_cppgen.CACHE_FILE)
        shutil.move(cutlass_cppgen.CACHE_FILE, os.path.join(cache_dir(), filename))
        log.debug("Moved CUTLASS compiled cache file to %s", cache_dir())
    except OSError:
        log.warning("Failed to move CUTLASS compiled cache file", exc_info=True)


def _rename_cutlass_import(content: str, cutlass_modules: list[str]) -> str:
    for cutlass_module in cutlass_modules:
        content = content.replace(
            f"from {cutlass_module} import ",
            f"from cutlass_library.{cutlass_module} import ",
        )
    return content


@functools.cache
def try_import_cutlass() -> bool:
    """
    We want to support three ways of passing in CUTLASS:
    1. fbcode, handled by the internal build system.
    2. User specifies cutlass_dir. The default is ../third_party/cutlass/,
       which is the directory when developers build from source.
    """
    if config.is_fbcode():
        try:
            import cutlass_cppgen  # type: ignore[import-not-found]
            import cutlass_library  # type: ignore[import-not-found]
        except ImportError as e:
            log.warning(
                "Failed to import CUTLASS packages in fbcode: %s, ignoring the CUTLASS backend.",
                e,
            )
            return False

        return True

    # Copy CUTLASS python scripts to a temp dir and add the temp dir to Python search path.
    # This is a temporary hack to avoid CUTLASS module naming conflicts.
    # TODO(ipiszy): remove this hack when CUTLASS solves Python scripts packaging structure issues.

    # TODO(mlazos): epilogue visitor tree currently lives in python/cutlass,
    # but will be moved to python/cutlass_library in the future (later 2025)
    def path_join(path0, path1):
        return os.path.abspath(os.path.join(path0, path1))

    # contains both cutlass and cutlass_library
    # we need cutlass for eVT
    cutlass_dir = (
        config.xpu.cutlass_dir
        if torch.xpu._is_compiled()
        else config.cutlass.cutlass_dir
    )
    cutlass_python_path = path_join(cutlass_dir, "python")
    torch_root = os.path.abspath(os.path.dirname(torch.__file__))
    mock_src_path = os.path.join(
        torch_root,
        "_inductor",
        "codegen",
        "cutlass",
        "lib_extensions",
        "cutlass_mock_imports",
    )

    cutlass_library_src_path = path_join(cutlass_python_path, "cutlass_library")
    cutlass_cppgen_src_path = path_join(cutlass_python_path, "cutlass_cppgen")
    pycute_src_path = path_join(cutlass_python_path, "pycute")

    tmp_cutlass_full_path = os.path.abspath(os.path.join(cache_dir(), "torch_cutlass"))

    dst_link_library = path_join(tmp_cutlass_full_path, "cutlass_library")
    dst_link_cutlass_cppgen = path_join(tmp_cutlass_full_path, "cutlass_cppgen")
    dst_link_pycute = path_join(tmp_cutlass_full_path, "pycute")

    # mock modules to import cutlass
    mock_modules = ["cuda", "scipy", "pydot"]

    if os.path.isdir(cutlass_python_path):
        if tmp_cutlass_full_path not in sys.path:

            def link_and_append(dst_link, src_path, parent_dir):
                if os.path.lexists(dst_link):
                    if not os.path.islink(dst_link):
                        raise AssertionError(
                            f"{dst_link} is not a symlink. Try to remove {dst_link} manually and try again."
                        )
                    if os.path.realpath(os.readlink(dst_link)) != os.path.realpath(
                        src_path,
                    ):
                        raise AssertionError(
                            f"Symlink at {dst_link} does not point to {src_path}"
                        )
                else:
                    os.makedirs(parent_dir, exist_ok=True)
                    os.symlink(src_path, dst_link)

                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)

            link_and_append(
                dst_link_library, cutlass_library_src_path, tmp_cutlass_full_path
            )
            link_and_append(
                dst_link_cutlass_cppgen, cutlass_cppgen_src_path, tmp_cutlass_full_path
            )
            link_and_append(dst_link_pycute, pycute_src_path, tmp_cutlass_full_path)

            for module in mock_modules:
                link_and_append(
                    path_join(tmp_cutlass_full_path, module),  # dst_link
                    path_join(mock_src_path, module),  # src_path
                    tmp_cutlass_full_path,  # parent
                )

        try:
            import cutlass_cppgen  # type: ignore[import-not-found]  # noqa: F401
            import cutlass_library.generator
            import cutlass_library.library
            import cutlass_library.manifest  # noqa: F401
            import pycute  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError as e:
            log.debug(
                "Failed to import CUTLASS packages: %s, ignoring the CUTLASS backend.",
                e,
            )
    else:
        log.debug(
            "Failed to import CUTLASS packages: CUTLASS repo does not exist: %s",
            cutlass_python_path,
        )
    return False


def _normalize_xpu_arch(arch: str) -> str:
    if arch.startswith("Xe"):
        return arch[2:]
    if 12 <= int(arch) and int(arch) <= 50:
        return arch
    else:
        raise NotImplementedError(f"Unsupported xpu arch: {arch}")


def _normalize_cuda_arch(arch: str) -> str:
    arch_num = arch
    if isinstance(arch, str):
        digits = "".join(ch for ch in arch if ch.isdigit())
        if not digits:
            raise ValueError(f"Unrecognized cuda arch: {arch}")
        arch_num = int(digits)
    else:
        arch_num = int(arch)

    if arch_num > 107:
        log.warning("Detected CUDA architecture > 107: %s. Please file an issue.", arch)
        return str(arch_num)
    if arch_num >= 107:
        return "107"
    if arch_num >= 103:
        return "103"
    if arch_num >= 100:
        return "100"
    if arch_num >= 90:
        return "90"
    if arch_num >= 80:
        return "80"
    if arch_num >= 75:
        return "75"
    if arch_num >= 70:
        return "70"
    raise NotImplementedError(f"Unsupported cuda arch: {arch}")


@functools.lru_cache(8)
def cutlass_arch(device_type: str) -> str:
    if device_type == "xpu":
        arch = get_xpu_arch()
        return _normalize_xpu_arch(arch)
    else:
        arch = get_cuda_arch()
        return _normalize_cuda_arch(arch)


@functools.lru_cache(1)
def toolkit_version(device_type: str) -> str:
    if device_type == "xpu":
        return get_xpu_version()
    else:
        return get_cuda_version()


def get_device_cutlass_config(device_type: str):
    """Get device-specific CUTLASS config (xpu/cuda overrides general cutlass config)."""
    if device_type == "xpu":
        return config.xpu
    from ...config import cutlass as inductor_cutlass_config

    return inductor_cutlass_config


@dataclass
class CUTLASSArgs:
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """

    architectures: str | None = None
    toolkit_version: str | None = None
    instantiation_level: str | None = None
    operations: str | None = None

    build_dir = ""
    curr_build_dir = ""
    generator_target = ""
    kernels = "all"
    ignore_kernels = ""
    exclude_kernels = ""
    # TODO: these three look dead?
    kernel_filter_file: None = None
    selected_kernel_list: None = None
    interface_dir: None = None
    filter_by_cc = True
    disable_full_archs_compilation = False
    device_type: str = "cuda"

    def __post_init__(self):
        if self.architectures is None or self.toolkit_version is None:
            raise RuntimeError(
                f"{self.architectures=} or {self.toolkit_version=} is None!"
            )


@clear_on_fresh_cache
@functools.cache
def _gen_ops_cached(arch: str, version: str, device_type: str) -> dict[Any, Any]:
    # Note: Cache needs to be specific for cuda architecture and version

    # Import cutlass python scripts.
    if not try_import_cutlass():
        raise AssertionError("Failed to import CUTLASS library")
    import cutlass_library.generator as cutlass_generator
    import cutlass_library.manifest as cutlass_manifest

    if arch is None or version is None:
        log.error(
            "Cannot detect cuda arch %s or version %s. "
            "Will discard all cutlass ops. "
            "Please consider setting _inductor.cuda.arch and _inductor.cuda.version configs.",
            arch,
            version,
        )
        return {}

    # SM103 and SM107 reuse the SM100 generator, but the CUTLASS manifest must keep
    # their architecture-specific baseline rather than treating them as SM100.
    if arch in ("103", "107"):
        gen_arch = "100"
        manifest_arch = f"{arch}a"
    else:
        gen_arch = manifest_arch = arch

    instantiation_level: str = config.cutlass.cutlass_instantiation_level
    args = CUTLASSArgs(
        architectures=manifest_arch,
        toolkit_version=version,
        instantiation_level=instantiation_level,
        operations=CUTLASS_OPERATION_KIND,
        device_type=device_type,
    )
    manifest = cutlass_manifest.Manifest(args)
    if arch == "107":
        # CUTLASS uses 103a as the SM100-family feature marker for architectures
        # without INT8 UMMA. Keep the SM107 baseline while sharing that feature.
        manifest.compute_capabilities_feature_set.append("103a")

    start_time = time.time()
    if device_type == "xpu":
        if hasattr(cutlass_generator, "GenerateIntelXe"):
            cutlass_generator.GenerateIntelXe(
                manifest, args.toolkit_version, arch=int(arch)
            )
        else:
            raise NotImplementedError(
                "Arch " + arch + " is not supported by current cutlass lib."
            )

    elif arch == "100":
        if hasattr(cutlass_generator, "GenerateSM100"):
            cutlass_generator.GenerateSM100(manifest, args.toolkit_version)
        cutlass_generator.GenerateSM90(manifest, args.toolkit_version)
    else:
        try:
            func = getattr(cutlass_generator, "GenerateSM" + gen_arch)
            func(manifest, args.toolkit_version)
        except AttributeError as e:
            raise NotImplementedError(
                "Arch " + gen_arch + " is not supported by current cutlass lib."
            ) from e

    log.info(
        "CUTLASS library generated a dict of %d operation kinds in %.2f seconds",
        len(manifest.operations),
        time.time() - start_time,
    )
    return manifest.operations


def gen_ops(device_type: str) -> dict[Any, Any]:
    """
    Generates all supported CUTLASS operations.
    """
    with dynamo_timed("cutlass_utils.gen_ops"):
        arch = cutlass_arch(device_type)
        version = toolkit_version(device_type)
        return _gen_ops_cached(arch, version, device_type)


from ..cpp_utils import DTYPE_TO_CPP


if torch.xpu._is_compiled():
    DTYPE_TO_CUTLASS_TYPE = {
        **DTYPE_TO_CPP,
        torch.float16: "uint16_t",
        torch.bfloat16: "uint16_t",
        torch.float8_e4m3fn: "uint8_t",
        torch.float8_e5m2: "uint8_t",
    }
else:
    DTYPE_TO_CUTLASS_TYPE = {
        **DTYPE_TO_CPP,
        torch.float16: "__half",
        torch.bfloat16: "__nv_bfloat16",
        torch.float8_e4m3fn: "__nv_fp8_e4m3",
        torch.float8_e5m2: "__nv_fp8_e5m2",
    }


@functools.lru_cache(32)
def torch_dtype_to_cutlass_type(
    torch_dtype: torch.dtype,
) -> "cutlass_library.library.DataType":  # type: ignore[name-defined] # noqa: F821
    # Import cutlass python scripts.
    if not try_import_cutlass():
        raise AssertionError("Failed to import CUTLASS library")
    import cutlass_library  # type: ignore[import]

    if torch_dtype == torch.float:
        return cutlass_library.library.DataType.f32
    elif torch_dtype == torch.half:
        return cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_library.library.DataType.bf16
    elif torch_dtype == torch.float8_e4m3fn:
        return cutlass_library.library.DataType.e4m3
    elif torch_dtype == torch.float8_e5m2:
        return cutlass_library.library.DataType.e5m2
    else:
        raise NotImplementedError(f"Unsupported data type: {torch_dtype=}")


@functools.lru_cache(32)
def dtype_match(
    torch_dtype: torch.dtype | None,
    cutlass_dtype: "cutlass_library.library.DataType",  # type: ignore[name-defined]  # noqa: F821
) -> bool:
    # Import cutlass python scripts.
    if not try_import_cutlass():
        raise AssertionError("Failed to import CUTLASS library")
    import cutlass_library

    if torch_dtype == torch.float:
        return (
            cutlass_dtype == cutlass_library.library.DataType.f32
            or cutlass_dtype == cutlass_library.library.DataType.tf32
        )
    elif torch_dtype == torch.half:
        return cutlass_dtype == cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_library.library.DataType.bf16
    elif torch_dtype == torch.int8:
        return cutlass_dtype == cutlass_library.library.DataType.s8
    elif torch_dtype == torch.uint8:
        return cutlass_dtype == cutlass_library.library.DataType.u8
    elif torch_dtype == torch.int32:
        return cutlass_dtype == cutlass_library.library.DataType.s32
    elif torch_dtype == torch.float8_e4m3fn:
        return cutlass_dtype == cutlass_library.library.DataType.e4m3
    elif torch_dtype == torch.float8_e5m2:
        return cutlass_dtype == cutlass_library.library.DataType.e5m2
    else:
        return False


def get_accumulator_dtype(
    input_torch_dtypes: list[torch.dtype],
) -> torch.dtype | None:
    """
    Given a pair of input torch dtypes, returns the inferred accumulator torch dtype.
    """

    if not (OrderedSet(input_torch_dtypes) <= XW_DTYPES):
        raise AssertionError(f"{input_torch_dtypes=} is not supported")

    if len(input_torch_dtypes) != 2:
        return None

    if OrderedSet(input_torch_dtypes) == OrderedSet(
        [torch.float8_e5m2, torch.float8_e4m3fn]
    ):
        return torch.float

    torch_dtype = None
    if input_torch_dtypes[0] == input_torch_dtypes[1]:
        torch_dtype = input_torch_dtypes[0]
    else:
        size0 = torch.tensor([], dtype=input_torch_dtypes[0]).element_size()
        size1 = torch.tensor([], dtype=input_torch_dtypes[1]).element_size()
        if size0 > size1:
            dtype0, dtype1 = input_torch_dtypes
        else:
            dtype1, dtype0 = input_torch_dtypes
        if dtype0 in [torch.half, torch.bfloat16] and dtype1 in [
            torch.int8,
            torch.uint8,
        ]:
            torch_dtype = dtype0

    if torch_dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        accumulator_dtype = torch.float
    elif torch_dtype == torch.int8:
        accumulator_dtype = torch.int32
    else:
        raise NotImplementedError(f"Unsupported data types: {input_torch_dtypes=}")

    if accumulator_dtype not in ACCUMULATOR_DTYPES:
        raise AssertionError(f"{accumulator_dtype=} is not supported")
    return accumulator_dtype


@functools.lru_cache(32)
def get_alignments(torch_dtype: torch.dtype) -> list[int]:
    """
    Returns all possible valid CUTLASS alignments in terms of the number of elements for a given dtype.
    CUTLASS gemm / conv SM80 APIs support 16 bytes max alignment, and 2 bytes min alignment.
    """

    if torch_dtype in (torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif torch_dtype == torch.float:
        return [4, 2, 1]
    elif torch_dtype in (
        torch.uint8,
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return [16, 8, 4, 2]
    elif torch_dtype == torch.int32:
        return [4, 2, 1]
    else:
        raise NotImplementedError(f"unsupported {torch_dtype=} for alignments")


def get_max_alignment(inductor_layout: Layout) -> int:
    """
    Returns the max alignment (in terms of number of elements) for a given Inductor Layout.
    """

    dtype = inductor_layout.dtype
    size = inductor_layout.size
    offset = inductor_layout.offset

    def is_static_int(number: object) -> TypeIs[int | sympy.Integer]:
        return isinstance(number, (int | sympy.Integer))

    def a_factor_of(x, alignment):
        if is_static_int(x) and is_static_int(alignment):
            return x % alignment == 0
        rem = sympy.Mod(x, alignment)
        return V.graph.sizevars.evaluate_expr(sympy.Eq(rem, 0))

    try:
        contiguous_dim = inductor_layout.stride.index(1)
    except ValueError:
        # No dim with stride 1 found, return 1
        return 1
    alignments = get_alignments(dtype)
    for alignment in alignments:
        if not a_factor_of(size[contiguous_dim], alignment) or not a_factor_of(
            offset, alignment
        ):
            continue
        if all(
            (dim == contiguous_dim)
            or a_factor_of(inductor_layout.stride[dim], alignment)
            for dim in range(len(size))
        ):
            return alignment
    return 1


class CUTLASSCompileSourceCapturingContext:
    # Helper class for Benchmarking and Testing CUTLASS Kernels in isolation.
    # Can be used to capture the sourcecode passed to CUDACodeCache.compile

    def __init__(self, device_type: str):
        self.sources = []
        self._compile_patch = None
        self.device_type = device_type

    def __enter__(self, *args, **kwargs):
        import unittest.mock as mock

        import torch._inductor.codecache

        codecache_cls = (
            torch._inductor.codecache.XPUCodeCache
            if self.device_type == "xpu"
            else torch._inductor.codecache.CUDACodeCache
        )
        _compile_method_orig = codecache_cls.compile

        def my_compile(source_code, dst_file_ext, extra_args: list[str] | None = None):
            self.sources.append(source_code)
            return _compile_method_orig(source_code, dst_file_ext)

        # pyrefly: ignore [bad-assignment]
        self._compile_patch = mock.patch(
            f"torch._inductor.codecache.{codecache_cls.__name__}.compile", my_compile
        )
        self._compile_patch.__enter__(*args, **kwargs)  # type: ignore[union-attr]
        return self

    def __exit__(self, *args, **kwargs):
        self._compile_patch.__exit__(*args, **kwargs)  # type: ignore[union-attr]


def cutlass_standalone_runner_compile_command(
    device_type: str, srcpath: Path, exepath: Path
):
    # returns command string to compile a (captured) CUDA GEMM Kernel source to a standalone executable that's ready to run
    # Passes the correct preprocessor define to nvcc to ensure the standalone runner is enabled.

    extra_args = ["-DGENERATE_STANDALONE_RUNNER=1"]
    if device_type != "xpu":
        extra_args.append("-DCUTLASS_DEBUG_TRACE_LEVEL=1")
    cutlass_compile_command = (
        torch._inductor.codegen.xpu.compile_utils.xpu_compile_command
        if device_type == "xpu"
        else torch._inductor.codegen.cuda.compile_utils.cuda_compile_command
    )
    compile_command = cutlass_compile_command(
        [str(srcpath)], str(exepath), "exe", extra_args=extra_args
    )
    return compile_command
<|endoftext|># mypy: allow-untyped-defs
import copy
import enum
import functools
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.codegen.cutlass.cache import maybe_fetch_ops
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.select_algorithm import create_inputs_key
from torch._inductor.utils import clear_on_fresh_cache

from ... import ir
from ...config import cutlass as inductor_cutlass_config
from ...ir import (
    Buffer,
    ChoiceCaller,
    CUTLASSTemplateBuffer,
    FixedLayout,
    FlexibleLayout,
    IRNode,
    Layout,
    ReinterpretView,
)
from ...utils import is_dynamic, Placeholder, sympy_product
from ...virtualized import V
from ..common import IndentedBuffer
from ..cuda import cuda_env
from . import utils as cutlass_utils
from .kernel import CUTLASSTemplateKernel
from .python_evt import CutlassEVTCodegen, scaled_mm_evt
from .template import CUTLASSTemplate
from .utils import (
    ACCUMULATOR_DTYPES,
    dtype_match,
    torch_dtype_to_cutlass_type,
    XW_DTYPES,
)


GemmOperation = Any
EVTArgRenames = Any

log = logging.getLogger(__name__)

# Jinja template for GEMM Kernel, used by the CUTLASSGemm3xTemplate class below.
GEMM_TEMPLATE_CUTLASS_3X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
{{epilogue_visitor_tree}}
{{instance_definition}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.
extern "C" {
PT_EXPORT {{kernel_call_signature}} {
  try {
  using ElementComputeEpilogue = {{instance_type}}::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  static cutlass::KernelHardwareInfo hw_info;
  if (hw_info.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    CUTLASS_TRACE_HOST("Query result for SM count per device: " << hw_info.sm_count);
  }
  {{dynamic_cluster}}
  {{instance_type}}::Arguments arguments;
  {{template.render_gemm_arguments(argument_template, epilogue_template, should_swap_xw,
                                    X, W, Bias, Y, alpha, beta, kernel, epilogue_args)}}
  {{instance_type}} gemm_op;
  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }
  // check for null pointers after workspace size, since querying workspace size doesn't require valid data pointers
#ifndef CUTLASS_BACKEND_DISABLE_CHECKS
  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
#endif
#ifdef CUTLASS_DEBUG_TRACE_LEVEL
#if CUTLASS_DEBUG_TRACE_LEVEL == 1
  {
    // Print the maximum number of active blocks per SM for the kernel if CUTLASS_DEBUG_TRACE_LEVEL == 1
    // we don't need a print statement, it's happening inside the function.
    gemm_op.maximum_active_blocks();
  }
#endif
#endif
  {
    auto status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
  }
  catch (std::exception& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }
  catch (...) {
    return -1;
  }
  return 0;
}
}

// configuration name: {{op_conf_name}}
"""

# Jinja template for Cutlass 3.x GEMM Kernel arguments, used by the CUTLASSGemmTemplate class below.
GEMM_ARGS_CUTLASS_3X = r"""
  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    {{template.gemm_mode()}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>({{M}}),
      static_cast<coord_t>({{N}}),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      {{template.cutlass_type_cast(X, kernel.ptr(X))}},  // ElementA const* ptr_A
      {
        {{template.cute_int(kernel.stride(X, -2), "stride_x0")}},
        {{template.cute_int(kernel.stride(X, -1), "stride_x1")}},
        {{template.cute_int(kernel.batch_stride(X), "batch_stride_x")}}
      },  // StrideA dA
      {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // ElementB const* ptr_B
      {
        {{template.cute_int(kernel.stride(W, -1), "stride_w1")}},
        {{template.cute_int(kernel.stride(W, -2), "stride_w0")}},
        {{template.cute_int(kernel.batch_stride(W), "batch_stride_w")}}
      },  // StrideB dB
    },  // MainloopArguments mainloop
    {{epilogue_arguments}},
    hw_info
  };
  arguments.scheduler.max_swizzle_size = swizzle;
"""

# Jinja template for Cutlass 3.x GEMM Kernel arguments if epilogue fusion is applied,
# used by the CUTLASSGemmTemplate class below.
GEMM_ARGS_CUTLASS_3X_EPILOGUE = r"""
    // see https://tinyurl.com/4rk89z48
    {
      {{epilogue_args}},  // thread, typename FusionCallbacks::Arguments ( EVT ) or ThreadEpilogueOp::Params (non-EVT )
      {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // ElementC const* ptr_C
      {
        {{template.cute_int(kernel.stride(Bias, -2, 1), "stride_bias0")}},
        {{template.cute_int(kernel.stride(Bias, -1, 1), "stride_bias1")}},
        {{template.cute_int(kernel.batch_stride(Bias), "batch_stride_bias")}}
      },  // StrideC dC
      {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // ElementD const* ptr_D
      {
        {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}},
        {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}},
        {{template.cute_int(kernel.batch_stride(Y), "batch_stride_y")}}
      },  // StrideD dD
    },  // EpilogueArguments epilogue
"""

# Jinja template for GEMM Kernel, used by the CUTLASS2xGemmTemplate class below.
GEMM_TEMPLATE_CUTLASS_2X = r"""
{{template.header().getvalue()}}
{{template.globals().getvalue()}}
{{instance_definition}}
// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, computes the Gemm kernel using the given workspace ptr.
extern "C" {
PT_EXPORT {{kernel_call_signature}} {
  try {
  int B = {{kernel.size(Y, 0, -3, default_value=1)}};
  using ElementComputeEpilogue = {{instance_type}}::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  static cutlass::KernelHardwareInfo hw_info;
  if (hw_info.sm_count == 0) {
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    CUTLASS_TRACE_HOST("Query result for SM count per device: " << hw_info.sm_count);
  }
  {{instance_type}}::Arguments arguments;
  {{template.render_gemm_arguments(instance_type, argument_template, epilogue_template, should_swap_xw,
                                    X, W, Bias, Meta, Y, alpha, beta, kernel, epilogue_args)}}
  {{instance_type}} gemm_op;
  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }

  // check for null pointers after workspace size, since querying workspace size doesn't require valid data pointers
#ifndef CUTLASS_BACKEND_DISABLE_CHECKS
  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
#endif
#ifdef CUTLASS_DEBUG_TRACE_LEVEL
#if CUTLASS_DEBUG_TRACE_LEVEL == 1
  {
    // Print the maximum number of active blocks per SM for the kernel if CUTLASS_DEBUG_TRACE_LEVEL == 1
    // we don't need a print statement, it's happening inside the function.
    gemm_op.maximum_active_blocks();
  }
#endif
#endif

  {
    auto status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
  }
  catch (std::exception& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }
  catch (...) {
    return -1;
  }
  return 0;
}
}
"""

# Jinja template for Cutlass 2.x GEMM Kernel arguments, used by the CUTLASS2xGemmTemplate class below.
GEMM_ARGS_CUTLASS_2X = r"""
  int64_t batch_stride_x = {{kernel.stride(X, -3)}};
  int64_t row_stride_x = {{kernel.row_or_column_stride(X)}};
  int64_t batch_stride_w = {{kernel.stride(W, -3)}};
  int64_t row_stride_w = {{kernel.row_or_column_stride(W)}};
  int64_t batch_stride_bias = {{kernel.stride(Bias, -3)}};
  int64_t row_stride_bias = {{kernel.row_or_column_stride(Bias)}};
  int64_t batch_stride_y = {{kernel.stride(Y, -3)}};
  int64_t row_stride_y = {{kernel.row_or_column_stride(Y)}};
  // Initialize GemmUniversalInstance arguments.
  arguments = {
    {{template.gemm_mode()}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K)
    },  // GemmCoord problem_size
    {{split_k if split_k > 1 else 'B'}},  // int batch_count
    {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename EpilogueOutputOp::Params epilogue
    {{template.cutlass_type_cast(X, kernel.ptr(X))}},  // void const * ptr_A
    {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // void const * ptr_B
    {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // void const * ptr_C
    {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // void * ptr_D
    batch_stride_x,  // int64_t batch_stride_A
    batch_stride_w,  // int64_t batch_stride_B
    batch_stride_bias,  // int64_t batch_stride_C
    batch_stride_y,  // int64_t batch_stride_D
    row_stride_x,  // typename LayoutA::Stride::LongIndex lda
    row_stride_w,  // typename LayoutB::Stride::LongIndex ldb
    row_stride_bias,  // typename LayoutC::Stride::LongIndex ldc
    row_stride_y,  // typename LayoutC::Stride::LongIndex ldd
  };
"""

GEMM_ARGS_SPARSE_CUTLASS_2X = r"""
  using TensorRefA = cutlass::TensorRef<{{instance_type}}::ElementA,
                                        {{instance_type}}::LayoutA>;
  using TensorRefB = cutlass::TensorRef<{{instance_type}}::ElementB,
                                        {{instance_type}}::LayoutB>;
  using TensorRefC = cutlass::TensorRef<{{instance_type}}::ElementC,
                                        {{instance_type}}::LayoutC>;
  using TensorRefE = cutlass::TensorRef<{{instance_type}}::ElementE,
                                        {{instance_type}}::LayoutE>;
  // Note that "X" and "W" names may be misleading here.  Namely, for
  // sparse GEMM, the first argument is always sparse, while typically
  // weight matrix, implied by name "W" will be sparse in
  // applications.  Thus, just remember that here: "X" refers to first
  // argument, that is sparse, and "W" to second, that is dense.
  TensorRefA X_ref({{template.cutlass_type_cast(X, kernel.ptr(X))}}, {{kernel.row_or_column_stride(X)}});
  TensorRefB W_ref({{template.cutlass_type_cast(W, kernel.ptr(W))}}, {{kernel.row_or_column_stride(W)}});
  TensorRefC Y_ref({{template.cutlass_type_cast(Y, kernel.ptr(Y))}}, {{kernel.row_or_column_stride(Y)}});
  TensorRefE Meta_ref({{template.cutlass_sparse_meta_type_cast(Meta, kernel.ptr(Meta))}},
                      TensorRefE::Layout::packed({ {{kernel.size(Meta, 0)}}, {{kernel.size(Meta, 1)}} }));
  // Initialize GemmSparse arguments.
  arguments = {
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(2 * K),
    },  // GemmCoord problem_size
    X_ref,  // TensorRef<ElementA const, LayoutA> ref_A
    W_ref,  // TensorRef<ElementB const, LayoutB> ref_B
    Y_ref,  // TensorRef<ElementC const, LayoutC> ref_C
    Y_ref,  // TensorRef<ElementC, LayoutC> ref_D
    Meta_ref,  // TensorRef<ElementE const, LayoutE> ref_E
    {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename EpilogueOutputOp::Params epilogue,
  };
"""

# Additional includes which are necessary if the standalone test / debug runner is generated as well
GEMM_STANDALONE_RUNNER_ADDITIONAL_INCLUDES = r"""
#ifdef GENERATE_STANDALONE_RUNNER
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include <iostream>
#endif
"""

# Jinja template for the standalone runner that may be generated as part of the code.
GEMM_STANDALONE_RUNNER_TEMPLATE = r"""
#ifdef GENERATE_STANDALONE_RUNNER
/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed, float max=1.0, float min=-1.0) {
  if (block.size()<=0) return false;
  Element scope_max(static_cast<Element>(max)), scope_min(static_cast<Element>(min));
  cutlass::reference::device::BlockFillRandomUniform(
    (Element*)block.get(), block.size(), seed, scope_max, scope_min);

  return true;
}

{% if Meta is defined and Meta is not none %}
template <class Element>
bool initialize_block_meta(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed) {
  if (block.size()<=0) return false;
  cutlass::reference::device::BlockFillRandomSparseMeta(
    (Element*)block.get(), block.size(), seed, {{instance_type}}::kMetaSizeInBits);
  return true;
}
{% endif %}

extern "C" int run_standalone(uint64_t seed, int repetitions) {
    std::cout << "Starting GEMM Standalone test run with seed " << seed << std::endl;
    size_t workspace_size = 0;
    size_t* workspace_size_ptr = &workspace_size;

    int M = {{kernel.get_layout_args()[0]}};
    int N = {{kernel.get_layout_args()[1]}};
    int K = {{kernel.get_layout_args()[2]}};
    int B = {{kernel.get_layout_args()[3]}};
    int lda = {{kernel.get_layout_args()[4]}};
    int ldb = {{kernel.get_layout_args()[5]}};
    int ldc = {{kernel.get_layout_args()[6]}};
    int ldd = {{kernel.get_layout_args()[7]}};
    uint8_t swizzle = {{kernel.runtime_arg_values[0]}};

    using ElementA = {{kernel.cutlass_dtype(X)}};
    using ElementB = {{kernel.cutlass_dtype(W)}};
    using ElementC = {{kernel.cutlass_dtype(Bias, default_dtype='uint8_t')}}; // may not be void
    using ElementD = {{kernel.cutlass_dtype(Y)}};
    {% if Meta is defined and Meta is not none %}
    using ElementE = {{kernel.cutlass_dtype(Meta)}};
    {% endif %}

    cutlass::DeviceAllocation<ElementA> X_data({{kernel.max_valid_index(X)+1}});
    initialize_block(X_data, seed++);
    cutlass::DeviceAllocation<ElementB> W_data({{kernel.max_valid_index(W)+1}});
    initialize_block(W_data, seed++);
    cutlass::DeviceAllocation<ElementC> Bias_data({{kernel.max_valid_index(Bias)+1}});
    initialize_block(Bias_data, seed++);
    cutlass::DeviceAllocation<ElementD> Y_data({{kernel.max_valid_index(Y)+1}});
    {% if Meta is defined and Meta is not none %}
    cutlass::DeviceAllocation<ElementE> Meta_data({{kernel.max_valid_index(Meta)+1}});
    initialize_block_meta(Meta_data, seed++);
    {% endif %}

    cutlass::DeviceAllocation<uint8_t> workspace_data;
    // Call once with workspace_size_ptr set to get workspace size

    std::cout << "Calling once to get workspace size" << std::endl;
    {{test_call_statement}};
    // Allocate workspace if necessary
    if (workspace_size > 0) {
        workspace_data.reset(workspace_size);
        std::cout << "Allocated workspace size of " << workspace_size << " bytes" << std::endl;
    }
    std::cout << "Calling Kernel as {{test_call_statement}};" << std::endl;
    workspace_size_ptr = nullptr;
    for (int i=0; i<repetitions; i++) {
        {{test_call_statement}};
    }
#if defined(CUTLASS_ENABLE_SYCL)
    compat::wait();
#else
    cudaDeviceSynchronize();
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Device synchronize failed with error "
        << cudaGetErrorString(result) << std::endl;
      return result;
    }
#endif
    return 0;
}

int main(int argc, char** argv) {
    // warmup
    run_standalone(1, 2);
    // repeat
    return run_standalone(2, 10);
}

#endif
"""


@clear_on_fresh_cache
class CUTLASSGemmTemplate(CUTLASSTemplate, ABC):
    """
    CUTLASS GEMM Template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    filtered_ops_cache: dict[str, list[Any]] = {}
    cache_clear = staticmethod(filtered_ops_cache.clear)

    @property
    def _device_cutlass_config(self):
        """Get device-specific CUTLASS config (xpu/cuda overrides general cutlass config)."""
        return cutlass_utils.get_device_cutlass_config(self.device_type)

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: list[int] | None = None,
        use_fast_accum: bool | None = None,
    ) -> None:
        """
        Args:
            input_nodes (List[Buffer]): List of input nodes of the GEMM kernel.
            layout (Layout): Layout type of the resulting output node.
            alpha (float): The scaling factor for the product of the inputs in the GEMM operation.
            beta (float): The scaling factor applied to the output matrix.
            input_reorder (Optional[List[int]]): Specifies the reordering of the input nodes. If not provided,
                            no reordering is performed. Defaults to None.
            use_fast_accum (Optional[bool]): enable/disable tensor-core fast accumulation (only available in `CUTLASS3xGemmTemplate` and Hopper GPUs)
        """
        super().__init__(
            str(Placeholder.KERNEL_NAME), input_nodes, layout, input_reorder
        )
        self.alpha = alpha
        self.beta = beta
        self.use_fast_accum = use_fast_accum
        if not (2 <= len(input_nodes) <= 5):
            raise AssertionError(f"expected 2 to 5 input nodes, got {len(input_nodes)}")
        if not self._are_inputs_layout_compatible(
            [node.get_layout() for node in input_nodes]
        ):
            raise AssertionError("input nodes have incompatible layouts")

        self.cache_key: str = create_inputs_key(self.input_nodes)

    @staticmethod
    @abstractmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: float | int = 1,
        beta: float | int = 0,
        input_reorder: list[int] | None = None,
        use_fast_accum: bool | None = None,
        **extra_kwargs,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_supported_ops() -> "list[cutlass_library.gemm_operation.GemmOperation]":  # type: ignore[name-defined]  # noqa: F821
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _has_tma_epilogue(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_template(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_template_args(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[str, str | None]:
        raise NotImplementedError

    @abstractmethod
    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _shape_match(
        self,
        op: "cutlass_library.gemm_op.GemmOperation",  # type: ignore[name-defined]  # noqa: F821
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
