            {input_lines}    ]


            def main():
                for name in kernel_names:
                    CudaKernelParamCache.cache.pop(name, None)
                with config.patch('aot_inductor.link_libtorch', True):
                    compiled_fn = CppWrapperCodeCache.load_pybinding(
                        argtypes=['std::vector<AtenTensorHandle>'],
                        main_code=cpp_wrapper_src,
                        device_type={device_type!r},
                        num_outputs={num_outputs!r},
                        kernel_code=kernel_src,
                    )
                input_tensors = make_inputs()
                input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(
                    input_tensors
                )
                output_handles = compiled_fn(input_handles)
                output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(
                    output_handles
                )
                print(f'Produced {{len(output_tensors)}} outputs')
                cubin_path_name = get_cpp_wrapper_cubin_path_name()
                for name in kernel_names:
                    params = CudaKernelParamCache.get(name)
                    if params and cubin_path_name in params:
                        print(f'{{name}}: {{params[cubin_path_name]}}')


            if __name__ == '__main__':
                main()
            """
            )
            .lstrip()
            .format(
                cpp_wrapper_src=cls._python_source_literal(
                    "cpp_wrapper_src", cpp_source
                ),
                kernel_src=cls._python_source_literal("kernel_src", kernel_source),
                kernel_names=kernel_names,
                input_lines=input_lines,
                device_type=device_type,
                num_outputs=num_outputs,
            )
        )
        output_code_log.debug("%s: \n%s", log_name, source)
        _, path = PyCodeCache.write(source)
        output_code_log.info("%s written to: %s", log_name, path)

    def _run_jit_variant_for_autotune(
        self,
        wrapper_code,
        kernel_code,
        extract_real_inputs,
        kernel_names: list[str],
    ) -> None:
        """Compile dual-wrapper-mode C++ as JIT variant and run to autotune kernels.

        Compiles the dual-wrapper-mode C++ source without -DAOT_INDUCTOR, which
        activates the JIT path (inductor_entry_impl). Running this with
        real inputs triggers lazy Triton compilation and autotuning,
        populating CudaKernelParamCache for the AOTI packaging step.
        """
        from .codecache import (
            CppWrapperCodeCache,
            CudaKernelParamCache,
            get_cpp_wrapper_cubin_path_name,
        )

        cpp_source = wrapper_code.value
        kernel_source = kernel_code.value if kernel_code else None

        # The JIT wrapper keeps lazy kernel state in function-local statics.
        # Compile a fresh wrapper for each first pass so autotuning always reruns.
        cpp_source += f"\n// AOTI lazy autotune first pass: {id(self)}\n"
        for name in kernel_names:
            CudaKernelParamCache.cache.pop(name, None)

        # Prefer the GPU device for the JIT compile: the wrapper includes
        # cpp_wrapper/<gpu>.h which transitively pulls in cuda_runtime.h.
        # A "cpu" device would precompile cpp_wrapper/cpu.h, which does not
        # include the CUDA headers needed to compile the kernel call sites.
        device_type = next(
            (d for d in self.device_types if d in ("cuda", "xpu")),
            next((d for d in self.device_types if d != "meta"), "cpu"),
        )

        # A const graph has no runtime graph inputs (its weights are read as
        # constants, not placeholders), so its JIT entry point expects only the
        # appended constant handles. extract_real_inputs() would return the main
        # model's params/inputs, producing a handle-count/type mismatch, so pass
        # an empty input list here and rely solely on the constants below.
        real_inputs = [] if self.is_const_graph else extract_real_inputs()

        def materialize_constant(name: str) -> torch.Tensor:
            constant = self.constants[name]
            if is_fake_tensor(constant):
                constant = defake(constant)
            if not isinstance(constant, torch.Tensor):
                raise AssertionError(f"Expected tensor constant for {name}")
            return constant

        # Non-tensor scalars become 0-d CPU tensors; None and ints/floats
        # that the graph specialized away aren't part of the C++ wrapper
        # signature and must be skipped.
        input_tensors: list[torch.Tensor] = []
        input_exprs: list[str] = []
        for arg in real_inputs:
            if arg is None:
                continue
            if isinstance(arg, torch.Tensor):
                input_tensors.append(arg)
                input_exprs.append(self._tensor_repro_expr(arg))
            else:
                input_tensors.append(torch.tensor(arg, device="cpu"))
                input_exprs.append(f"torch.tensor({arg!r}, device='cpu')")
        for name in self.constants:
            constant = materialize_constant(name)
            input_tensors.append(constant)
            input_exprs.append(self._tensor_repro_expr(constant))

        self._log_jit_variant_for_autotune_repro(
            cpp_source=cpp_source,
            kernel_source=kernel_source,
            device_type=device_type,
            num_outputs=len(self.graph_outputs),
            kernel_names=kernel_names,
            input_exprs=input_exprs,
        )

        # This temporary Python-loaded wrapper can depend on libtorch even when
        # the final packaged AOTI artifact is built with link_libtorch=False.
        with config.patch("aot_inductor.link_libtorch", True):
            compiled_fn = CppWrapperCodeCache.load_pybinding(
                argtypes=["std::vector<AtenTensorHandle>"],
                main_code=cpp_source,
                device_type=device_type,
                num_outputs=len(self.graph_outputs),
                kernel_code=kernel_source,
            )

        input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(
            input_tensors
        )
        del real_inputs, input_tensors, input_exprs

        output_handles = compiled_fn(input_handles)
        output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(
            output_handles
        )
        del output_tensors

        # Collect cubin files produced by lazy compilation for AOTI packaging
        cubin_path_name = get_cpp_wrapper_cubin_path_name()
        for name in kernel_names:
            params = CudaKernelParamCache.get(name)
            if params and cubin_path_name in params:
                self.wrapper_code.additional_files.append(params[cubin_path_name])

    def _update_scheduler(self) -> None:
        """
        (Re)initializes the scheduler member.  When initializing the scheduler, no CUBIN
        files should be generated (to avoid biasing any benchmarks and pessimizing
        fusion decisions).
        """
        from .scheduler import Scheduler

        with config.patch("triton.store_cubin", False):
            self.scheduler = Scheduler(self.operations)

    def codegen(self) -> tuple[ValueWithLineMap, ValueWithLineMap]:
        with dynamo_timed("GraphLowering.codegen", log_pt2_compile_event=True):
            self.init_wrapper_code()

            self._update_scheduler()
            V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)

            self.wrapper_code.push_codegened_graph(self)
            self.scheduler.codegen()

            log.debug(
                "Finished codegen for all nodes. The list of kernel names available: %s",
                V.graph.all_codegen_kernel_names,
            )

            result = self.wrapper_code.generate(self.is_inference)
            self.wrapper_code.pop_codegened_graph()
            return result

    def codegen_subgraph(self, parent_graph: GraphLowering) -> None:
        """
        This is a more compact version of the `codegen()` above
        where we codegen this graph as a subgraph of some parent
        graph. The parent graph is passed as an argument: the
        intention is to inline codegening of the subgraph in
        the parent graph's wrapper code (including the generated
        kernels). The wrapper code is not finalized (via `.generate()`
        call), as this will be done in the parent graph's `codegen()`.
        """
        with dynamo_timed("GraphLowering.codegen_subgraph", log_pt2_compile_event=True):
            self.wrapper_code = parent_graph.wrapper_code
            self.device_ops = parent_graph.device_ops
            self.cpp_wrapper = parent_graph.cpp_wrapper
            self.device_types = parent_graph.device_types
            self.device_idxs = parent_graph.device_idxs
            self.device_type = parent_graph.device_type

            self._update_scheduler()
            self.scheduler.codegen()

    def count_bytes(
        self,
    ) -> tuple[
        int, list[tuple[BaseSchedulerNode, int]], list[tuple[BaseSchedulerNode, float]]
    ]:
        total_bytes = 0
        node_counts = []
        node_runtimes = []
        for node in self.scheduler.nodes:
            num_bytes = node.get_read_write_buffers_sizes()
            total_bytes += num_bytes
            node_counts.append((node, num_bytes // 4))
            node_runtimes.append((node, node.get_estimated_runtime()))

        return total_bytes, node_counts, node_runtimes

    # No-op to be patched for unit tests
    save_output_code: Callable[[str], None] | None = None

    def compile_to_module(self) -> CompiledModule:
        with dynamo_timed(
            "GraphLowering.compile_to_module",
            phase_name="code_gen",
            log_pt2_compile_event=True,
            dynamo_compile_column_us="inductor_code_gen_cumulative_compile_time_us",
        ):
            return self._compile_to_module()

    def _compile_to_module(self) -> CompiledModule:
        # If we're here, we don't have to worry about the kernel code, which is only
        # returned separately in AOTInductor mode.
        wrapper_code, _ = (
            self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
        )

        if isinstance(wrapper_code, ValueWithLineMap):
            mod = self._compile_to_module_lines(wrapper_code)
        elif isinstance(wrapper_code, FileBackedGraphModule):
            mod = wrapper_code
        else:
            raise NotImplementedError(
                f"Unrecognized wrapper code type: {type(wrapper_code)}"
            )

        # Logged twice as per https://github.com/pytorch/pytorch/pull/99038#discussion_r1167826029
        # TODO. Revisit this once the logging API is more mature
        if mod.__file__ is None:
            raise AssertionError("mod.__file__ is None")

        log_module_code(mod.__file__)
        log.debug("Output code written to: %s", mod.__file__)
        output_code_log.info("Output code written to: %s", mod.__file__)
        if config.benchmark_kernel:
            print(f"Compiled module path: {mod.__file__}", file=sys.stderr)
        if isinstance(wrapper_code, FileBackedGraphModule):
            V.debug.output_code(mod.__file__)
            V.debug.copy(os.path.splitext(mod.__file__)[0] + ".debug")

        return mod

    def _compile_to_module_lines(
        self, wrapper_code: ValueWithLineMap
    ) -> CompiledModule:
        from .codecache import PyCodeCache

        if config.triton.autotune_at_compile_time:
            # sanitize docstrings in kernel defs (#155006)
            kernel_autotune_defs = self.wrapper_code.kernel_autotune_defs.getvalue()
            kernel_autotune_defs = kernel_autotune_defs.replace('"""', '\\"\\"\\"')

            tuning_code = (
                'r"""\n'
                + "Compile-time auto-tuning block: \n"
                + kernel_autotune_defs
                + self.wrapper_code.kernel_autotune_calls.getvalue()
                + '"""\n'
            )
            wrapper_code.value = tuning_code + wrapper_code.value
        if GraphLowering.save_output_code is not None:
            GraphLowering.save_output_code(wrapper_code.value)
        output_code_log.debug("Output code: \n%s", wrapper_code.value)

        inductor_meta = autotune_cache.inductor_meta_from_config()
        AutotuneCacheBundler.begin_compile(inductor_meta, code=wrapper_code.value)

        try:
            linemap = [
                (line_no, node.stack_trace)  # type: ignore[attr-defined]
                for line_no, node in wrapper_code.line_map
            ]
            key, path = PyCodeCache.write(wrapper_code.value)
            output_code_log.debug("Output code written to: %s", path)

            V.debug.output_code(path)
            V.debug.copy(os.path.splitext(path)[0] + ".debug")
        except Exception:
            trace_structured(
                "inductor_output_code",
                # Just omit the filename, I still want the code though!
                payload_fn=lambda: wrapper_code.value,
            )
            raise
        else:
            trace_structured(
                "inductor_output_code",
                lambda: {
                    "filename": path,
                    "file_path": os.path.abspath(path),
                },
                payload_fn=lambda: wrapper_code.value,
            )
        with dynamo_timed("PyCodeCache.load_by_key_path", log_pt2_compile_event=True):
            mod = PyCodeCache.load_by_key_path(
                key,
                path,
                linemap=linemap,  # type: ignore[arg-type]
                attrs={
                    **self.constants,
                    **self.torchbind_constants,
                    **self.opaque_value_type_classes,
                },
            )
        self.cache_key = key
        self.cache_path = path
        self.cache_linemap = linemap  # type: ignore[assignment]

        if config.benchmark_harness and config.profile_bandwidth_output:
            # run the inputs code gen to get the bandwidth info
            args = mod.get_args()
            mod.benchmark_compiled_module(args, times=1, repeat=1)

        return mod

    def _get_output_names(self, graph_outputs: list[ir.IRNode]) -> list[str]:
        names = []
        shape_counter = itertools.count(0)
        none_counter = itertools.count(0)
        for node in graph_outputs:
            if isinstance(node, ir.NoneAsConstantBuffer):
                names.append(f"{self.name}_none{next(none_counter)}")
            elif isinstance(node, ir.ShapeAsConstantBuffer):
                names.append(f"{self.name}_shape{next(shape_counter)}")
            else:
                names.append(node.get_name())
        return names

    def get_output_names(self) -> list[str]:
        return self._get_output_names(self.graph_outputs)

    def is_unspec_arg(self, name: str) -> bool:
        # dynamo wraps unspec variable as 0d CPU tensor,
        # need to convert to scalar during codegen (triton only)
        return (
            name in self.graph_inputs
            and self.graph_inputs[name].get_numel() == 1
            and len(self.graph_inputs[name].get_size()) == 0
            and get_device_type(self.graph_inputs[name]) == "cpu"
        ) or name in self.zero_dim_cpu_tensor_list


class SubgraphLowering(GraphLowering):
    """
    Mostly a helper class for the subgraph lowering. The main goal is to call
    init_wrapper_code with the subgraph related arguments.
    """

    def __init__(self, parent: GraphLowering, *args: Any, **kwargs: Any) -> None:
        self.parent = parent
        super().__init__(*args, **kwargs)

    def allocate_non_dup_const_name(self, name: str | None, data: Tensor) -> str:
        name = super().allocate_non_dup_const_name(name, data)
        # The generated subgraph wrapper shares the parent's module, but tensor
        # constants are attached to the module from the root graph's constants
        # dict at load time. Propagate the value up so it is not left as None.
        root = self.parent
        while isinstance(root, SubgraphLowering):
            root = root.parent
        root.constants[name] = data
        return name

    def init_wrapper_code(
        self,
        is_subgraph: bool = False,
        subgraph_name: str | None = None,
        parent_wrapper_code: PythonWrapperCodegen | None = None,
        partition_signatures: GraphPartitionSignature | None = None,
    ) -> None:
        super().init_wrapper_code(
            is_subgraph=True,
            subgraph_name=self.name,
            parent_wrapper_code=self.parent.wrapper_code,
        )
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import contextvars
import ctypes
import dataclasses
import functools
import logging
import multiprocessing as mp
import os
import pickle
import queue
import selectors
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p, CDLL
from typing import Any, IO, TYPE_CHECKING
from typing_extensions import override

import torch
import torch._inductor.async_compile
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided
from torch._inductor import ir
from torch._inductor.codecache import (
    CppCodeCache,
    CUDACodeCache,
    DLLWrapper,
    get_hash,
    PyCodeCache,
    write,
    XPUCodeCache,
)
from torch._inductor.compile_worker.timer import Timer
from torch._inductor.utils import (
    apply_subprocess_env,
    clear_caches,
    clear_on_fresh_cache,
    do_bench_using_profiling,
    get_gpu_type,
    get_ld_library_path,
    is_gpu,
    python_subprocess_env,
)
from torch._logging import getArtifactLogger
from torch.utils._ordered_set import OrderedSet


# Inactivity timeout for AutotuneProcessPool in seconds.
# Default: 600 seconds (10 minutes). Set to 0 to disable.
AUTOTUNE_POOL_INACTIVITY_TIMEOUT = int(
    os.environ.get("TORCHINDUCTOR_AUTOTUNE_POOL_INACTIVITY_TIMEOUT", "600")
)

if TYPE_CHECKING:
    from types import ModuleType

    from torch._inductor.select_algorithm import (
        ChoiceCaller,
        PartialRender,
        TritonTemplateCaller,
    )

from . import config
from .runtime.benchmarking import benchmarker
from .virtualized import V


CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


_visible_device_env_var_maps = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
}


def get_visible_devices_env_var(gpu_type: str | None = None) -> str:
    if gpu_type is None:
        gpu_type = get_gpu_type()
    if gpu_type not in _visible_device_env_var_maps:
        raise ValueError(f"Unsupported gpu_type: {gpu_type}")
    return _visible_device_env_var_maps[gpu_type]


autotuning_log = getArtifactLogger(__name__, "autotuning")


class NonzeroWorkspaceNotSupportedError(Exception):
    pass


def _cache_env_for_subprocess() -> dict[str, str | None]:
    env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
    return {v: os.environ.get(v) for v in env_vars}


_last_applied_cache_env: dict[str, str | None] | None = None


def _apply_subprocess_env_and_clear_caches(
    extra_env: dict[str, str | None] | None,
) -> None:
    global _last_applied_cache_env

    if extra_env is None:
        return

    if extra_env != _last_applied_cache_env:
        clear_caches()
        _last_applied_cache_env = extra_env.copy()
    apply_subprocess_env(extra_env)


def _run_with_subprocess_env(
    fn: Callable[..., Any],
    extra_env: dict[str, str | None],
    *args: Any,
    **kwargs: Any,
) -> Any:
    _apply_subprocess_env_and_clear_caches(extra_env)
    return fn(*args, **kwargs)


class TuningProcess:
    """
    Class to launch and interact with a benchmarking subprocess.
    """

    @staticmethod
    def process_main(read_pipe: IO[bytes], write_pipe: IO[bytes]) -> None:
        """
        Entry point for the child process.
        """
        autotuning_log.debug(
            "Started autotune subprocess %s. Visible devices: %s",
            os.getpid(),
            os.environ.get(get_visible_devices_env_var()),
        )

        def workloop():
            while True:
                job, extra_env = TuningProcess.recv(read_pipe)
                if job is None:
                    # None is a sentinel for the child to shut down
                    break
                try:
                    _apply_subprocess_env_and_clear_caches(extra_env)
                    result = job()
                except Exception as e:
                    result = e
                TuningProcess.send(result, write_pipe)

        try:
            workloop()
        except EOFError:
            # The parent closed the pipe
            pass

    @staticmethod
    def send(
        obj: Any, write_pipe: IO[bytes], extra_env: dict[str, str | None] | None = None
    ) -> None:
        pickle.dump((obj, extra_env), write_pipe)
        write_pipe.flush()

    @staticmethod
    def recv(read_pipe: IO[bytes]) -> Any:
        return pickle.load(read_pipe)

    def __init__(self, device: int | None):
        self.device = device
        self.start()

    def start(self):
        """
        Start the benchmarking subprocess.
        """
        entry = os.path.join(os.path.dirname(__file__), "__autotune_main__.py")

        subproc_read_fd, write_fd = os.pipe()
        read_fd, subproc_write_fd = os.pipe()
        self.write_pipe = os.fdopen(write_fd, "wb")
        self.read_pipe = os.fdopen(read_fd, "rb")

        self.selector = selectors.DefaultSelector()
        self.selector.register(self.read_pipe, selectors.EVENT_READ)

        cmd = [
            sys.executable,
            entry,
            f"--parent={os.getpid()}",
            f"--read-fd={str(subproc_read_fd)}",
            f"--write-fd={str(subproc_write_fd)}",
        ]
        env = {
            **python_subprocess_env(),
            # We shouldn't be using the Triton async compile subprocess pool,
            # but as a precaution set the env var that disables its creation.
            "TORCH_WARM_POOL": "0",
            # Some internal usages need a modified LD_LIBRARY_PATH.
            "LD_LIBRARY_PATH": get_ld_library_path(),
            # This will cause the subprocs to profile using the profiler.
            "TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING": "1"
            if config.profile_bandwidth_with_do_bench_using_profiling
            else "0",
        }
        if self.device is not None:
            env[get_visible_devices_env_var()] = str(self.device)
        self.process = subprocess.Popen(
            cmd,
            env=env,
            pass_fds=(subproc_read_fd, subproc_write_fd),
        )
        os.close(subproc_read_fd)
        os.close(subproc_write_fd)

        self.running = True

    def alive(self) -> bool:
        """
        True if the subprocess is still running.
        """
        return self.running and self.process.poll() is None

    def put(self, req: Any, extra_env: dict[str, str | None] | None = None) -> None:
        """
        Push a work item to the child process.
        """
        if not self.alive():
            self.start()
        TuningProcess.send(req, self.write_pipe, extra_env=extra_env)

    def get(self, timeout: float = 120.0) -> Any:
        """
        Get a response from the child process. Raises TimeoutError on timeout;
        raises EOFError if the subprocess crashes.
        """
        try:
            if not self.selector.select(timeout):
                raise TimeoutError(f"Timeout in autotune subprocess {self.process.pid}")
            result, _ = TuningProcess.recv(self.read_pipe)
        except TimeoutError:
            self.kill()
            raise
        except EOFError:
            # The subprocess crashed
            self.close()
            raise
        except Exception:
            autotuning_log.exception(
                "Unexpected exception in autotune subprocess %s", self.process.pid
            )
            self.kill()
            raise

        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self, wait: bool = True) -> None:
        """
        Signal the child process to shut down gracefully.
        """
        if self.alive():
            TuningProcess.send(None, self.write_pipe)
        if wait:
            self.wait()

    def wait(self) -> None:
        """
        Wait for the child process to exit.
        """
        if self.alive():
            self.process.wait()
        self.close()

    def close(self) -> None:
        """
        Close resources.
        """
        self.selector.close()
        self.read_pipe.close()
        self.write_pipe.close()
        self.running = False

    def kill(self) -> None:
        """
        Send a SIGKILL to the child process.
        """
        if self.alive():
            autotuning_log.error(
                "Sending SIGKILL to autotune subprocess %d",
                self.process.pid,
            )
            self.process.kill()
        self.close()

    def restart(self) -> None:
        """
        Gracefully restarts the child process.
        """
        self.shutdown(wait=True)
        self.start()


class TuningProcessPool:
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """

    def __init__(self) -> None:
        """
        Start the child processes.
        """
        devices = self.get_device_list()
        autotuning_log.debug("Sub-process autotune device list: %s", devices)

        # Launch the child processes.
        self.processes = [TuningProcess(device=device) for device in devices]

        self.process_queue: queue.Queue[TuningProcess] = queue.Queue()
        for p in self.processes:
            self.process_queue.put(p)

        # Use a thread pool to manage distributing work to the subprocesses.
        # Threads block on an available process, so it makes sense to match
        # the number of threads with the number of devices.
        self.executor = ThreadPoolExecutor(max_workers=len(devices))

    @staticmethod
    def get_device_list() -> Sequence[int | None]:
        """
        Gather the list of devices to be used in the pool.
        """
        if not config.autotune_multi_device:
            # Don't use multiple devices
            return [None]

        gpu_type = get_gpu_type()
        device_interface = get_interface_for_device(gpu_type)
        count = device_interface.device_count()
        visible_devices_env_var = get_visible_devices_env_var(gpu_type)

        # If the user specified the visible devices in the env, use those.
        if visible_devices_env_var in os.environ:
            devices = [int(d) for d in os.environ[visible_devices_env_var].split(",")]
            if not len(devices) <= count:
                raise AssertionError(
                    f"{visible_devices_env_var} specifies {len(devices)} devices, "
                    f"but only {count} devices are available"
                )
            return devices

        return list(range(count))

    def shutdown(self) -> None:
        """
        Signal all child processes to exit.
        """
        self.executor.shutdown()

        for p in self.processes:
            p.shutdown(wait=False)
        for p in self.processes:
            p.wait()

    def target(self, choice: TritonTemplateCaller) -> float:
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
        if choice.bmreq is None:
            raise AssertionError(
                f"Expected choice.bmreq to be set, but got None for choice '{choice}'"
            )

        extra_env = _cache_env_for_subprocess()
        process = self.process_queue.get()
        process.put(choice.bmreq.benchmark, extra_env=extra_env)
        try:
            return process.get(
                config.max_autotune_subproc_result_timeout_seconds,
            )
        except TimeoutError:
            warnings.warn(
                f"Timed out benchmarking choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # Set to INF so this choice will be ignored
            return float("inf")
        except Exception as process_exception:
            warnings.warn(
                f"Failed to benchmark choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # Sticky CUDA errors corrupt the context, making it unrecoverable.
            # The process must be restarted to restore CUDA functionality.
            error_msg = str(process_exception)
            if (
                "cudaErrorLaunchFailure" in error_msg
                or "cudaErrorIllegalAddress" in error_msg
            ):
                process.restart()
            # Set to INF so this choice will be ignored
            return float("inf")
        finally:
            self.process_queue.put(process)

    def benchmark(
        self,
        choices: list[TritonTemplateCaller],
    ) -> dict[TritonTemplateCaller, float]:
        """
        Benchmark each choice in a separate process.
        """

        # Use a ThreadExecutorPool to spread the work across the subprocesses and
        # to grab subprocesses as soon as they're free.
        results = dict(zip(choices, self.executor.map(self.target, choices)))

        return results


LayoutOrBuffer = ir.Layout | ir.Buffer


@dataclasses.dataclass
class TensorMeta:
    device: torch.device
    dtype: torch.dtype
    sizes: torch._prims_common.ShapeType
    strides: torch._prims_common.StrideType
    offset: int
    name: str | None = None

    @classmethod
    def from_irnodes(
        cls, irnodes: LayoutOrBuffer | Sequence[LayoutOrBuffer]
    ) -> TensorMeta | list[TensorMeta]:
        from torch._inductor.select_algorithm import get_strides_with_layout_constraints

        if isinstance(irnodes, Sequence):
            result: list[Any] = [cls.from_irnodes(x) for x in irnodes]
            if not all(isinstance(x, TensorMeta) for x in result):
                raise AssertionError(
                    f"Expected all elements to be TensorMeta, got types: "
                    f"{[type(x) for x in result if not isinstance(x, TensorMeta)]}"
                )
            return result

        node = irnodes
        if isinstance(node, ir.Layout):
            node = ir.Buffer(name="fake", layout=node)

        dtype = node.get_dtype()
        if dtype is None:
            raise AssertionError(
                f"Expected node to have a dtype, but get_dtype() returned None for node '{node}'"
            )
        device = node.get_device()
        if device is None:
            raise AssertionError(
                f"Expected node to have a device, but get_device() returned None for node '{node}'"
            )

        return TensorMeta(
            device=device,
            dtype=dtype,
            sizes=V.graph.sizevars.optimization_hints(node.get_size()),
            strides=V.graph.sizevars.optimization_hints(
                get_strides_with_layout_constraints(node)
            ),
            offset=V.graph.sizevars.optimization_hint(node.get_layout().offset),
            name=node.get_name(),
        )

    def to_tensor(self) -> torch.Tensor:
        return rand_strided(
            self.sizes,
            self.strides,
            device=self.device,
            dtype=self.dtype,
            extra_size=self.offset,
        )


@dataclasses.dataclass
class BenchmarkRequest:
    """
    Only handle triton template benchmark for now. The extern kernel benchmark
    can be done inside the same process since they usually don't cause crash.

    Important: Instances of this class and subclasses have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
    ) -> None:
        # the kernel name defined in the module
        self.kernel_name = kernel_name

        if isinstance(input_tensor_meta, TensorMeta):
            self.input_tensor_meta: list[TensorMeta] = [input_tensor_meta]
        else:
            self.input_tensor_meta: list[TensorMeta] = input_tensor_meta

        if output_tensor_meta and isinstance(output_tensor_meta, (tuple, list)):
            if len(output_tensor_meta) > 1:
                # Each output with same meta for Grouped GEMM
                if not all(
                    getattr(output_tensor_meta[0], attr) == getattr(x, attr)
                    for x in output_tensor_meta
                    for attr in ["device", "dtype", "sizes", "strides", "offset"]
                ):
                    raise AssertionError(
                        "All output tensor metas in a Grouped GEMM must have matching "
                        "device, dtype, sizes, strides, and offset"
                    )
            self.output_tensor_meta = output_tensor_meta[0]
        else:
            # pyrefly: ignore [bad-assignment]
            self.output_tensor_meta: TensorMeta = output_tensor_meta

        self.extra_args = extra_args
        self.benchmark_with_cudagraphs = False

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        raise NotImplementedError

    def cleanup_run_fn(self) -> None:
        pass

    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        raise NotImplementedError

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        debug = autotuning_log.isEnabledFor(logging.DEBUG)
        if debug:
            start_ts = time.time()

        # create args and out tensor
        if out is None:
            if not (self.input_tensor_meta and self.output_tensor_meta):
                raise AssertionError(
                    "Input and output tensor meta must be populated when input_tensors is empty"
                )
            if not len(input_tensors) == 0:
                raise AssertionError(
                    f"Expected no input_tensors when out is None, but got {len(input_tensors)}"
                )
            input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
            out = self.output_tensor_meta.to_tensor()

        if debug:
            create_tensor_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            start_ts = time.time()
        try:
            try:
                fn = self.make_run_fn(*input_tensors, out=out)
            except NonzeroWorkspaceNotSupportedError:
                # Skipping all ops with nonzero workspace requirements
                autotuning_log.info("Skipping op due to nonzero workspace requirement")
                return float("inf")

            if debug:
                load_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
                start_ts = time.time()

            if self.benchmark_with_cudagraphs:
                res = benchmarker.benchmark_gpu_with_cuda_graph(fn)
            else:
                res = self.do_bench(fn, *input_tensors, out)

            if debug:
                bench_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
                autotuning_log.debug(
                    "InChildProcess %s: load %f, create tensor %f, bench %f",
                    self,
                    load_elapse,  # type: ignore[possibly-undefined]
                    create_tensor_elapse,  # type: ignore[possibly-undefined]
                    bench_elapse,
                )
            return res
        finally:
            self.cleanup_run_fn()


class _TestBenchmarkRequest(BenchmarkRequest):
    """
    Supports unit testing. Defined in this file instead of the test file so the
    TuningProcess sub-process can unpickle these objects.
    """

    def __init__(
        self,
        result: float = 0.0,
        device: int | None = None,
        sleep: float | None = None,
        exc: Exception | None = None,
        crash: bool = False,
    ):
        self.result = result
        self.device = device
        self.sleep = sleep
        self.exc = exc
        self.crash = crash

    def benchmark(
        self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None
    ) -> float:
        if self.device is not None:
            visible_devices_env_var = get_visible_devices_env_var()
            if os.environ.get(visible_devices_env_var, None) != str(self.device):
                raise AssertionError(
                    f"Expected {visible_devices_env_var}='{self.device}', "
                    f"but got '{os.environ.get(visible_devices_env_var, None)}'"
                )
        if self.sleep:
            time.sleep(self.sleep)
        if self.exc:
            raise self.exc
        if self.crash:
            sys.exit(1)
        return self.result


class _TestEnvBenchmarkRequest:
    """
    Supports unit testing subprocess environment propagation.
    """

    def __init__(self, key: str):
        self.key = key

    def benchmark(
        self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None
    ) -> str | None:
        return os.environ.get(self.key)


@clear_on_fresh_cache
class _TestCodeCache:
    @classmethod
    def cache_clear(cls) -> None:
        cls.write.cache_clear()

    @classmethod
    @functools.lru_cache(None)
    def write(cls, source_code: str) -> tuple[str, str]:
        return write(source_code, "txt")


class _TestCodeCacheBenchmarkRequest:
    """
    Supports unit testing subprocess codecache resets.
    """

    source_code = "test codecache contents\n"

    def benchmark(
        self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None
    ) -> str:
        _, input_path = _TestCodeCache.write(self.source_code)
        return input_path


class GPUDeviceBenchmarkMixin:
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        device_idx_set = OrderedSet(
            tensor.device.index
            for tensor in [*input_tensors, out]
            if isinstance(tensor, torch.Tensor)
            and is_gpu(tensor.device.type)
            and tensor.device.index is not None
        )
        if not len(device_idx_set) <= 1:
            raise AssertionError(f"Can not mix devices {device_idx_set}")
        device_type = next(
            (
                tensor.device.type
                for tensor in input_tensors
                if is_gpu(tensor.device.type)
            ),
            "cuda",
        )
        device_interface = get_interface_for_device(device_type)
        if len(device_idx_set) == 1:
            device_idx = next(iter(device_idx_set))
        else:
            device_idx = device_interface.current_device()
        with device_interface.device(device_idx):  # type: ignore[attr-defined]
            res = benchmarker.benchmark(fn, device=device_type)
            device_interface.synchronize()  # shake out any CUDA errors

        return res


class CPUDeviceBenchmarkMixin:
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        return benchmarker.benchmark_cpu(fn)


class TritonBenchmarkRequest(BenchmarkRequest):
    """
    Represents a standalone benchmark request for a Triton Template.

    Important: Instances of this class have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        module_path: str,  # the path of the module defining the triton kernel
        module_cache_key: str,
        num_stages: int,
        num_warps: int,
        num_consumer_groups: int = 0,
        num_buffers_warp_spec: int = 0,
        matrix_instr_nonkdim: int = 0,  # only used for hip to choose the shape of mfma instruction.
        waves_per_eu: int = 0,  # only used for hip to schedule waves per execution unit
        kpack: int = 0,  # ROCm specific gemm parameter
        workspace_size: int | None = None,  # size of workspace buffer in bytes
        workspace_zero_fill: bool = False,  # whether to zero-fill workspace
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.num_consumer_groups = num_consumer_groups
        self.num_buffers_warp_spec = num_buffers_warp_spec
        self.matrix_instr_nonkdim = matrix_instr_nonkdim
        self.waves_per_eu = waves_per_eu
        self.kpack = kpack
        self.workspace_size = workspace_size
        self.workspace_zero_fill = workspace_zero_fill
        self._benchmark_module: Any | None = None

    @override
    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        mod = PyCodeCache.load_by_key_path(
            self.module_cache_key,
            self.module_path,
            set_sys_modules=False,
        )
        self._benchmark_module = mod
        autotuning_log.debug(
            "benchmark module key: %s, path: %s",
            self.module_cache_key,
            self.module_path,
        )

        run_method = getattr(mod, self.kernel_name).run
        extra_args = list(self.extra_args)

        # Recreate workspace tensor if needed (for TMA templates)
        # The workspace tensor couldn't be pickled, so we recreate it here
        # It should be inserted before the grid values (last 3 elements)
        if self.workspace_size is not None:
            from torch._inductor.select_algorithm import WORKSPACE_ARG_PLACEHOLDER

            workspace_tensor = torch.empty(
                (self.workspace_size,),
                dtype=torch.uint8,
                device=out.device,
            )
            if self.workspace_zero_fill:
                workspace_tensor.zero_()
            workspace_index = extra_args.index(WORKSPACE_ARG_PLACEHOLDER)
            extra_args[workspace_index] = workspace_tensor

        run_method.__self__.with_bandwidth_info = False

        # Newer version of triton add warmup argument to JITFunction.run.
        # This code handles backward-compatibility.
        warmup_arg = {}
        import inspect

        if "warmup" in inspect.signature(run_method).parameters:
            warmup_arg["warmup"] = False

        if out.device.type == "cpu":
            device_interface = None
            device_index = 0
        else:
            device_type = out.device.type
            device_interface = get_interface_for_device(device_type)
            device_index = self.output_tensor_meta.device.index

        is_debug_autotuner = isinstance(
            getattr(mod, self.kernel_name),
            torch._inductor.runtime.triton_heuristics.DebugAutotuner,
        )

        # Resolve the stream at call time (not at closure-creation time) so that
        # CUDA graph capture on a different stream works correctly.
        def run_fn() -> None:
            stream = (
                0
                if device_interface is None
                else device_interface.get_raw_stream(device_index)
            )
            if is_debug_autotuner:
                run_method(
                    *input_tensors,
                    out,
                    *extra_args,
                    **warmup_arg,
                    stream=stream,
                )
            else:
                run_method(
                    *input_tensors,
                    out,
                    *extra_args,
                    **warmup_arg,
                    stream=stream,
                    benchmark_run=True,
                )

        return run_fn

    def cleanup_run_fn(self) -> None:
        # Authoritative cleanup for one benchmark module load. Higher-level
        # finally blocks may call this again as safety nets, so keep it idempotent.
        mod = self._benchmark_module
        self._benchmark_module = None

        cached_mod = PyCodeCache.modules_no_attr.pop(self.module_path, None)
        if mod is None:
            mod = cached_mod

        if mod is not None:
            kernel = getattr(mod, self.kernel_name, None)
            release_benchmark_artifacts = getattr(
                kernel, "release_benchmark_artifacts", None
            )
            if release_benchmark_artifacts is not None:
                release_benchmark_artifacts()
            release_static_launchers = getattr(
                kernel, "_release_static_launchers_except", None
            )
            if (
                release_benchmark_artifacts is None
                and release_static_launchers is not None
            ):
                release_static_launchers(None)
            elif release_benchmark_artifacts is None:
                for launcher in getattr(kernel, "launchers", ()) or ():
                    close = getattr(getattr(launcher, "__self__", None), "close", None)
                    if close is not None:
                        close()
                for compile_result in getattr(kernel, "compile_results", ()) or ():
                    close = getattr(
                        getattr(compile_result, "kernel", None), "close", None
                    )
                    if close is not None:
                        close()

        PyCodeCache.modules[:] = [
            module
            for module in PyCodeCache.modules
            if getattr(module, "__file__", None) != self.module_path
        ]
        PyCodeCache.linemaps.pop(self.module_path, None)

    def precompile(self):
        try:
            mod = PyCodeCache.load_by_key_path(
                self.module_cache_key,
                self.module_path,
                set_sys_modules=False,
            )
            self._benchmark_module = mod
            kernel = getattr(mod, self.kernel_name)
            kernel.precompile()

            self.n_regs = kernel.launchers[0].n_regs
        finally:
            self.cleanup_run_fn()

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.module_path=}, {self.module_cache_key=}"


class TritonGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, TritonBenchmarkRequest):
    pass


class TritonCPUBenchmarkRequest(CPUDeviceBenchmarkMixin, TritonBenchmarkRequest):
    pass


class ExternKernelBenchmarkRequest(BenchmarkRequest):
    """
    A class to handle extern kernel benchmark requests. This allows extern kernels
    (like aten::mm) to be benchmarked in a subprocess, similar to Triton kernels.

    Important: Instances of this class have to be serializable across
    process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        callable_path: str,  # Module path to the callable (e.g., "extern_kernels.mm")
        kwargs: dict[str, Any] | None = None,
        has_out_variant: bool = True,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.callable_path = callable_path
        self.kwargs = kwargs or {}
        self.has_out_variant = has_out_variant

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        fn = self.to_callable()
        if self.has_out_variant:
            # For out=variant, pass output as keyword arg
            return functools.partial(fn, *input_tensors, out=out)
        else:
            # For non-out variant, just call with inputs
            return functools.partial(fn, *input_tensors)

    def benchmark(self, *input_tensors: torch.Tensor, out: torch.Tensor | None = None):
        if out is not None and out.numel() == 0:
            # no need to run the kernel or do benchmarking
            return 0.0
        if self.has_out_variant or len(input_tensors) == 0:
            return super().benchmark(*input_tensors, out=out)
        else:
            algo = self.to_callable()
            out_new = algo(*input_tensors)
            if out is not None:
                torch._C._dynamo.guards.assert_size_stride(
                    out_new, tuple(out.size()), tuple(out.stride())
                )
                out.copy_(out_new)  # for correctness checking
            if self.benchmark_with_cudagraphs:
                return benchmarker.benchmark_gpu_with_cuda_graph(
                    lambda: algo(*input_tensors)
                )
            if config.profile_bandwidth_with_do_bench_using_profiling:
                return do_bench_using_profiling(lambda: algo(*input_tensors))
            return benchmarker.benchmark(algo, input_tensors, {})

    def precompile(self) -> None:
        # Extern kernels don't need precompilation - they're already compiled
        pass

    def to_callable(self):
        # While ExternKernelChoice also has a to_callable method,
        # we avoid calling the ExternKernelChoice version here to make sure
        # this is picklable
        from torch._inductor.select_algorithm import extern_kernels

        fn = getattr(extern_kernels, self.kernel_name)
        if self.kwargs:
            return functools.partial(fn, **self.kwargs)

        return fn

    def __str__(self) -> str:
        return f"ExternKernelBenchmarkRequest({self.callable_path})"


class ExternKernelGPUBenchmarkRequest(
    GPUDeviceBenchmarkMixin, ExternKernelBenchmarkRequest
):
    pass


class ExternKernelCPUBenchmarkRequest(
    CPUDeviceBenchmarkMixin, ExternKernelBenchmarkRequest
):
    pass


class SubgraphBenchmarkRequest(BenchmarkRequest):
    """
    Benchmark request for subgraph choices.

    Pre-compiles the subgraph in the main process and stores
    the module path/cache key for loading in subprocess.
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        module_path: str,
        module_cache_key: str,
        sym_input_values: list[int],
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.sym_input_values = sym_input_values

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        mod = PyCodeCache.load_by_key_path(
            self.module_cache_key,
            self.module_path,
            set_sys_modules=False,
        )
        sym_input_values = self.sym_input_values
        # Create a new list each call since mod.call does args.clear()
        return lambda: mod.call([*sym_input_values, *input_tensors])

    def precompile(self) -> None:
        # Module is already compiled in main process, no precompilation needed
        pass

    def __str__(self) -> str:
        return f"SubgraphBenchmarkRequest({self.kernel_name}, {self.module_path})"


class SubgraphGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, SubgraphBenchmarkRequest):
    pass


class SubgraphCPUBenchmarkRequest(CPUDeviceBenchmarkMixin, SubgraphBenchmarkRequest):
    pass


class CUTLASSBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """
    A class to handle CUDA (CUTLASS) benchmark requests. This class is for
    managing the lifecycle of a CUDA kernel benchmark, including compiling
    the source code, managing workspace memory, and executing the kernel.

    Important: Instances of this class have to be serializable across
    process boundaries. Do not put CUDA Tensors in here!
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
        device_type: str = "cuda",
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: torch.Tensor | None = None
        self.DLL: DLLWrapper | None = None
        self._workspace_size_updated = False
        self.hash_key: str = ""
        self.source_file: str = ""
        self.device_type = device_type
        self.codecache_cls = XPUCodeCache if device_type == "xpu" else CUDACodeCache
        self.device_interface = get_interface_for_device(device_type)
        self.hash_key, self.source_file = self.codecache_cls.write(
            self.source_code, "so"
        )

    def precompile(self):
        """
        Precompile the CUDA source code to populate the CUDACodeCache.
        This may happen in a separate thread pool.
        """
        autotuning_log.debug("Precompiling %s", self)
        self.codecache_cls.compile(self.source_code, "so")
        autotuning_log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        """
        Create a function to run the CUDA/XPU kernel with the given input and output tensors.
        """

        self.ensure_dll_loaded()
        self.update_workspace_size()
        args = [c_void_p(tensor.data_ptr()) for tensor in list(input_tensors) + [out]]
        autotuning_log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        stream_ptr = c_void_p(
            self.device_interface.get_raw_stream(self.device_interface.current_device())
        )
        run_method = getattr(self.DLL, self.kernel_name)
        workspace_ptr = c_void_p(0)
        if self.workspace_size > 0:
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=out.device,
            )
            workspace_ptr = c_void_p(self.workspace.data_ptr())

        # Generate partial function.
        ret = functools.partial(
            run_method,
            *args,
            *self.extra_args,
            None,  # null workspace size ptr
            workspace_ptr,  # set workspace ptr,
            stream_ptr,
        )

        # sanity check to make sure we cleanup run fn properly
        try:
            ret()
        except RuntimeError as e:
            err_msg = str(e)

            def raise_runtime_error():
                raise RuntimeError(err_msg)

            self.cleanup_run_fn()
            return raise_runtime_error

        return ret

    def update_workspace_size(self) -> None:
        if self._workspace_size_updated:
            return
        self.ensure_dll_loaded()
        unique_input_count = len(
            dict.fromkeys(meta.name for meta in self.input_tensor_meta)
        )
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]
        stream_ptr = c_void_p(
            self.device_interface.get_raw_stream(self.device_interface.current_device())
        )

        run_method = getattr(self.DLL, self.kernel_name)
        # Retrieve workspace_size and initialize workspace.
        c_workspace_size = c_size_t()
        run_method(
            *args,  # input ptrs and output ptrs
            *self.extra_args,
            byref(
                c_workspace_size
            ),  # set workspace size ptr to retrieve workspace size
            None,  # null workspace ptr
            stream_ptr,
        )
        self.device_interface.synchronize()  # shake out any device errors
        self.workspace_size = c_workspace_size.value
        autotuning_log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        self._workspace_size_updated = True

    def ensure_dll_loaded(self):
        if self.DLL is None:
            self.DLL, self.hash_key, self.source_file = self.codecache_cls.load(
                self.source_code, "so"
            )
        self.DLL.open()

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()
            self.DLL = None
        self.workspace = None

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["DLL"] = None
        state["workspace"] = None
        state["_workspace_size_updated"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class CppBenchmarkRequest(CPUDeviceBenchmarkMixin, BenchmarkRequest):
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.hash_key = get_hash(source_code)
        self.DLL: CDLL | ModuleType | None = None

    def precompile(self):
        # Prepopulate CppCodeCache
        # may happen in separate Threadpool
        autotuning_log.debug("Precompiling %s", self)
        CppCodeCache.load(self.source_code, device_type="cpu")
        autotuning_log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        # TODO(jgong5): use CppPythonBindingsCodeCache for better binding perf
        self.DLL = CppCodeCache.load(self.source_code, device_type="cpu")
        args = [tensor.data_ptr() for tensor in list(input_tensors) + [out]]
        autotuning_log.debug(
            "make_run_fn: self.kernel_name=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.DLL,
            args,
            self.extra_args,
        )
        run_method = getattr(self.DLL, self.kernel_name)
        # Assume only size with type ctypes.c_ulonglong in extra_args
        if not all(isinstance(arg, ctypes.c_ulonglong) for arg in self.extra_args):
            raise AssertionError(
                f"Expected all extra_args to be ctypes.c_ulonglong, got types: "
                f"{[type(arg) for arg in self.extra_args if not isinstance(arg, ctypes.c_ulonglong)]}"
            )
        run_method.argtypes = [ctypes.c_ulonglong] * (
            len(args) + len(list(self.extra_args))
        )

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
        )

    def __str__(self) -> str:
        return f"{self.kernel_name=}"


class CuteDSLBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for CuteDSL (CUTLASS Python DSL) kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: tuple[Any, ...],
        source_code: PartialRender,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)

        # Template-specific precompile hooks reuse the finalized module source in subprocesses.
        self.source_code = source_code.finalize_all()
        self.module_cache_key, self.module_path = PyCodeCache.write(self.source_code)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        """
        Create a function to run the CuteDSL kernel with the given input and output tensors.
        Similar to TritonBenchmarkRequest.make_run_fn but for CuteDSL kernels.
        """
        mod = PyCodeCache.load_by_key_path(
            self.module_cache_key,
            self.module_path,
            set_sys_modules=False,
        )

        # Logic replicated async_compile
        from .codegen.cutedsl.cutedsl_kernel import MAIN_SUFFIX

        main_func_name = f"{self.kernel_name}_{MAIN_SUFFIX}"

        if not hasattr(mod, main_func_name):
            available = [name for name in dir(mod) if callable(getattr(mod, name))]
            raise RuntimeError(
                f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
            )

        kernel_func = getattr(mod, main_func_name)

        def run_kernel():
            device_interface = get_interface_for_device("cuda")
            stream = device_interface.get_raw_stream(out.device.index)
            return kernel_func(*input_tensors, out, *self.extra_args, stream=stream)

        return run_kernel


@functools.cache
def get_tuning_process_pool() -> TuningProcessPool:
    pool = TuningProcessPool()
    atexit.register(pool.shutdown)
    return pool


def benchmark_in_sub_process(
    choices: list[TritonTemplateCaller],
) -> dict[TritonTemplateCaller, float]:
    """
    Do benchmarking in a subprocess and return the perf number (latency).
    """
    return get_tuning_process_pool().benchmark(choices)


class AutotuneProcessPool:
    """
    Singleton pool manager for running autotuning (precompilation + benchmarking)
    in a separate process.
    """

    _instance: AutotuneProcessPool | None = None
    _lock: threading.Lock = threading.Lock()
    _shutdown_for_inactivity: bool = False

    def __init__(self):
        self._pool: ProcessPoolExecutor | None = self._init_pool()
        self._warmup_future: Future[Any] | None = None
        self._warmup_start_time: float | None = None
        self._timer: Timer | None = self._init_timer()

    @classmethod
    def get_instance(cls):
        """Get or create the singleton pool instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # num_workers=1 to avoid GPU contention during benchmarking
                    cls._instance = cls()
        return cls._instance

    @property
    def pool(self):
        """Get the process pool."""
        if not config.pipeline_max_autotune_gemm:
            raise AssertionError(
                "To use AutotuneProcessPool, pipeline_max_autotune_gemm must be enabled"
            )
        if self._pool is None:
            self._pool = self._init_pool()
            self._timer = self._init_timer()
        return self._pool

    def _init_timer(self) -> Timer | None:
        if AUTOTUNE_POOL_INACTIVITY_TIMEOUT > 0:
            return Timer(AUTOTUNE_POOL_INACTIVITY_TIMEOUT, self._on_inactivity_timeout)
        return None

    def _record_activity(self) -> None:
        if self._timer is not None:
            self._timer.record_call()

    def _on_inactivity_timeout(self) -> None:
        autotuning_log.info(
            "AutotuneProcessPool shutting down due to inactivity (timeout=%ds)",
            AUTOTUNE_POOL_INACTIVITY_TIMEOUT,
        )

        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None
            self._timer = None

            # Mark that the pool was shut down for inactivity.
            # This prevents the pool from being recreated on recompiles
            # which likely do not require large amounts of autotuning.
            AutotuneProcessPool._shutdown_for_inactivity = True

    def _init_pool(self):
        """
        Get or create the process pool.

        Uses ProcessPoolExecutor with 'spawn' context for CUDA safety.
        ProcessPoolExecutor is lazily initialized - workers are not spawned
        until the first submit() call, making this property non-blocking.
        """
        # Use 'spawn' context to avoid CUDA fork issues
        # Workers are spawned lazily on first submit(), not here
        ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
        )
        atexit.register(self._shutdown)
        autotuning_log.info("AutotuneProcessPool created (workers spawn lazily)")

        return pool

    def warm_up(self) -> Future[Any]:
        """
        Submit a warmup job to eagerly spawn workers and initialize CUDA.

        This is optional - call it early to hide spawn latency.
        Returns the warmup future which can be ignored or awaited.
        """
        if self._warmup_future is None:
            with self._lock:
                if self._warmup_future is None:
                    self._warmup_start_time = time.perf_counter()
                    self._warmup_future = self.submit(
                        _init_autotune_subprocess,
                        fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                    )
                    self._warmup_future.add_done_callback(self._on_warmup_complete)
                    autotuning_log.info("Warmup job submitted")
        # pyrefly: ignore[bad-return]
        return self._warmup_future

    def _on_warmup_complete(self, future: Future[Any]) -> None:
        """Callback invoked when the warmup job completes."""
        warmup_elapsed_time = None
        if self._warmup_start_time is not None:
            warmup_elapsed_time = time.perf_counter() - self._warmup_start_time

        try:
            result = future.result()
            autotuning_log.info(
                "AutotuneProcessPool warmup completed successfully in %.4f seconds: %s",
                warmup_elapsed_time,
                result,
            )
            self._record_activity()
        except Exception as e:
            autotuning_log.error(
                "AutotuneProcessPool warmup failed after %.4f seconds",
                warmup_elapsed_time,
            )
            raise e

    def submit(self, fn, *args, **kwargs) -> Future[Any]:
        """Submit a job to the pool and return a Future."""
        future = self.pool.submit(
            _run_with_subprocess_env,
            fn,
            _cache_env_for_subprocess(),
            *args,
            **kwargs,
        )
        if self._timer is not None:
            future.add_done_callback(lambda _: self._record_activity())
        return future

    def _shutdown(self):
        """Shutdown the pool on exit."""
        if self._timer is not None:
            self._timer.quit()
            self._timer = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    @classmethod
    def shutdown_instance(cls):
        """Explicitly shutdown the singleton instance."""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance._shutdown()
                    cls._instance = None


def use_pipelined_autotuning() -> bool:
    return (
        config.pipeline_max_autotune_gemm
        and not AutotuneProcessPool._shutdown_for_inactivity
    )


def _init_autotune_subprocess(fp32_precision: str) -> bool:
    """
    Warmup function run in the autotune subprocess.
    """
    import torch

    # Initialize dummy tensor for CUDA context
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    torch.backends.cuda.matmul.fp32_precision = fp32_precision

    return True


def run_autotune_in_subprocess(
    benchmark_request: BenchmarkRequest,
) -> float:
    """
    Run autotuning benchmarks in a subprocess.

    This function is submitted to AutotuneProcessPool and runs in isolation
    to prevent GPU contention with the main compilation process.

    Args:
        picklable_choices: List of picklable choice information

    Returns:
        timing
    """

    try:
        # Run the benchmark directly - bmreq is already a BenchmarkRequest
        timing = benchmark_request.benchmark()

        return timing

    except Exception:
        autotuning_log.warning(
            "Failed to benchmark choice %s",
            benchmark_request,
            exc_info=True,
        )
        # Use infinity for failed benchmarks so they're not selected
        return float("inf")


class PrecompileThreadPool:
    """
    Thread pool for running precompilation asynchronously.

    This allows the main compilation process to continue while
    precompilation happens in background threads.
    """

    _instance: PrecompileThreadPool | None = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @classmethod
    def get_instance(cls) -> PrecompileThreadPool:
        from torch._inductor.select_algorithm import get_num_workers

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(get_num_workers())
        return cls._instance

    def submit(self, fn, *args, **kwargs):
        ctx = contextvars.copy_context()
        # Need to copy context so workers have access to the correct config settings
        fn = functools.partial(ctx.run, fn)
        return self._executor.submit(fn, *args, **kwargs)

    def _shutdown(self, wait: bool = False):
        return self._executor.shutdown(wait=wait)

    @classmethod
    def shutdown_instance(cls) -> None:
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance._shutdown(wait=False)
                    cls._instance = None


class AsyncAutotuner:
    """
    Handles asynchronous autotuning of kernel choices in a separate process.

    This class manages the lifecycle of autotuning:
    1. Accepts precompiled choices from the main process
    2. Submits benchmarking work to AutotuneProcessPool
    3. Returns results via a Future

    Usage:
        autotuner = AsyncAutotuner(choices)
        autotuner.start()  # Kicks off async benchmarking
        timings = autotuner.get_results()  # Blocks until complete
    """

    choice_hash_to_future = {}

    @staticmethod
    def get_choice_hash(choice: ChoiceCaller, inputs_key: str) -> str:
        return choice.hash_key() + inputs_key

    @classmethod
    def start(cls, choices: list[ChoiceCaller], inputs_key: str):
        """
        Start asynchronous autotuning in a subprocess.

        This method:
        1. Extracts picklable benchmark requests from choices
        2. Submits benchmarking work to AutotuneProcessPool
        3. Returns immediately (non-blocking)
        """

        for choice in choices:
            choice_hash = AsyncAutotuner.get_choice_hash(choice, inputs_key)

            if choice_hash in AsyncAutotuner.choice_hash_to_future:
                continue

            if not getattr(choice, "bmreq", None) is not None:
                raise AssertionError("bmreq is None for choice")

            autotune_future = AutotuneProcessPool.get_instance().submit(
                run_autotune_in_subprocess,
                choice.bmreq,
            )

            AsyncAutotuner.choice_hash_to_future[choice_hash] = autotune_future

    @classmethod
    def get_results(
        cls, choices: list[ChoiceCaller], inputs_key: str
    ) -> dict[ChoiceCaller, float]:
        """
        Get autotuning results, blocking until complete.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            Dict mapping ChoiceCaller to benchmark timing
        """

        timings = {}
        for choice in choices:
            choice_hash = AsyncAutotuner.get_choice_hash(choice, inputs_key)
            timings[choice] = AsyncAutotuner.choice_hash_to_future[choice_hash].result()
        return timings
<|endoftext|># mypy: ignore-errors

from __future__ import annotations

import contextlib
import dataclasses
import sys
import threading
from typing import Any, TYPE_CHECKING
from typing_extensions import override, Self
from unittest.mock import patch

from torch._inductor import config
from torch._inductor.remote_cache import RemoteCacheBackend


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType


@dataclasses.dataclass
class Stats:
    num_put: int = 0
    num_get_hit: int = 0
    num_get_miss: int = 0

    def __iadd__(self, other: Stats) -> Self:
        self.num_put += other.num_put
        self.num_get_hit += other.num_get_hit
        self.num_get_miss += other.num_get_miss
        return self

    def reset(self) -> None:
        self.num_put = 0
        self.num_get_hit = 0
        self.num_get_miss = 0

    def __str__(self) -> str:
        return "".join(
            (
                f"puts: {self.num_put}, ",
                f"misses: {self.num_get_miss}, ",
                f"hits: {self.num_get_hit}, ",
            )
        )

    def __eq__(self, other: object) -> bool:
        # Dataclass's default __eq__ checks that the types are the same so can't
        # be used with _GlobalItemStats.
        return (
            isinstance(other, (Stats, _GlobalItemStats))
            and self.num_put == other.num_put
            and self.num_get_hit == other.num_get_hit
            and self.num_get_miss == other.num_get_miss
        )


class _GlobalItemStats(Stats):
    cache: dict[str, object]

    def __init__(self) -> None:
        super().__init__()
        self.cache = {}

    def reset(self) -> None:
        super().reset()
        self.cache = {}


# The cache states are thread-local so if we're running multiple tests at once
# they won't cross contaminate. However - it needs to be "global" because we
# allow code to create new cache clients which refer to the same cache (because
# it's a remote cache).


class _GlobalStats(threading.local):
    def __init__(self) -> None:
        self.autotune_local = _GlobalItemStats()
        self.autotune_remote = _GlobalItemStats()
        self.bundled_autotune = _GlobalItemStats()
        self.fx_graph = _GlobalItemStats()
        self.triton = _GlobalItemStats()
        self.aot_autograd = _GlobalItemStats()
        self.dynamo_pgo = _GlobalItemStats()

    def reset(self) -> None:
        self.autotune_local.reset()
        self.autotune_remote.reset()
        self.bundled_autotune.reset()
        self.fx_graph.reset()
        self.triton.reset()
        self.aot_autograd.reset()
        self.dynamo_pgo.reset()

    def get_stat(self, name: str) -> _GlobalItemStats:
        return getattr(self, name)

    def report(self):
        subs = (
            ("autotune_local", self.autotune_local),
            ("autotune_remote", self.autotune_remote),
            ("bundled_autotune", self.bundled_autotune),
            ("fx_graph", self.fx_graph),
            ("triton", self.triton),
            ("aot_autograd", self.aot_autograd),
            ("dynamo_pgo", self.dynamo_pgo),
        )

        print("Cache Stats:", file=sys.stderr)
        for name, sub in subs:
            print(f"  {name}: {sub}", file=sys.stderr)

        print("Cache Entries:", file=sys.stderr)
        for name, sub in subs:
            if sub.cache:
                print(f"  {name}:", file=sys.stderr)
                for k, v in sorted(sub.cache.items()):
                    v = repr(v)
                    if len(v) > 100:
                        v = v[:100] + "..."
                    print(f"    {k!r}: {v}", file=sys.stderr)


global_stats = _GlobalStats()


class MockBackend(RemoteCacheBackend[Any]):
    def __init__(self, name: str) -> None:
        self._name = name

    @staticmethod
    def with_name(name: str) -> Callable[[], MockBackend]:
        def wrapper() -> MockBackend:
            return MockBackend(name)

        return wrapper

    @override
    def _get(self, key: str) -> Any | None:
        stat = global_stats.get_stat(self._name)
        if key in stat.cache:
            stat += Stats(num_get_hit=1)
            return stat.cache.get(key)
        else:
            stat += Stats(num_get_miss=1)
            return None

    @override
    def _put(self, key: str, data: Any) -> None:
        stat = global_stats.get_stat(self._name)
        stat += Stats(num_put=1)
        stat.cache[key] = data


# List of configs for each cache
_CACHE_CONFIG_EN = (
    "fx_graph_cache",
    "fx_graph_remote_cache",
    "autotune_local_cache",
    "autotune_remote_cache",
    "bundled_autotune_remote_cache",
)


class PatchCaches(contextlib.AbstractContextManager):
    @classmethod
    def setUp(cls):
        # If this test is using PatchCaches then disable all the caches by
        # default, letting the tests turn them on explicitly. This is because
        # tests using PatchCaches will often want to check stats explicitly.
        cls._savedCacheState = {}
        for name in _CACHE_CONFIG_EN:
            if hasattr(config, name):
                cls._savedCacheState[name] = getattr(config, name)
            setattr(config, name, False)

    @classmethod
    def tearDown(cls):
        # Restore cache defaults
        for name in _CACHE_CONFIG_EN:
            delattr(config, name)
            if name in cls._savedCacheState:
                setattr(config, name, cls._savedCacheState[name])

    def __init__(self) -> None:
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> Self:
        global_stats.reset()
        self._stack.__enter__()

        ctx = patch(
            "torch._inductor.remote_cache.LocalAutotuneCache.backend_override_cls",
            MockBackend.with_name("autotune_local"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteAutotuneCache.backend_override_cls",
            MockBackend.with_name("autotune_remote"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteBundledAutotuneCache.backend_override_cls",
            MockBackend.with_name("bundled_autotune"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteFxGraphCache.backend_override_cls",
            MockBackend.with_name("fx_graph"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteAOTAutogradCache.backend_override_cls",
            MockBackend.with_name("aot_autograd"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteDynamoPGOCache.backend_override_cls",
            MockBackend.with_name("dynamo_pgo"),
        )
        self._stack.enter_context(ctx)

        if config.is_fbcode():
            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteAutotuneCache.backend_override_cls",
                MockBackend.with_name("autotune_remote"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteBundledAutotuneCache.backend_override_cls",
                MockBackend.with_name("bundled_autotune"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteFxGraphCache.backend_override_cls",
                MockBackend.with_name("fx_graph"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "triton.fb.fb_memcache.FbMemcacheRemoteKernelCache.backend_override_cls",
                MockBackend.with_name("triton"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteAOTAutogradCache.backend_override_cls",
                MockBackend.with_name("aot_autograd"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteDynamoPGOCache.backend_override_cls",
                MockBackend.with_name("dynamo_pgo"),
            )
            self._stack.enter_context(ctx)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, traceback)
<|endoftext|>from __future__ import annotations

import atexit
import collections
import dataclasses
import functools
import json
import logging
import os
import sys
import typing
from abc import abstractmethod
from typing import Any, Generic, TypeAlias, TypeVar, Union
from typing_extensions import override

from torch._dynamo.utils import dynamo_timed
from torch._inductor import config
from torch.monitor import _WaitCounter


if typing.TYPE_CHECKING:
    from collections.abc import Callable


try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


if config.is_fbcode():
    from rfe.scubadata.scubadata_py3 import (  # type: ignore[import-not-found]
        Sample as Sample_,
    )

    Sample: TypeAlias = Sample_
else:
    Sample: TypeAlias = type[object]  # type: ignore[misc,no-redef]


_T = TypeVar("_T")
_U = TypeVar("_U")


remote_fx_cache_get_timed = functools.partial(
    dynamo_timed,
    "FbRemoteFxGraphCache.get",
    phase_name="remote_fx_graph_cache_get",
    log_pt2_compile_event=False,
    dynamo_compile_column_us="remote_fx_graph_cache_get_time_us",
    log_waitcounter=True,
)
remote_fx_cache_put_timed = functools.partial(
    dynamo_timed,
    "FbRemoteFxGraphCache.put",
    phase_name="remote_fx_graph_cache_put",
    log_pt2_compile_event=False,
    dynamo_compile_column_us="remote_fx_graph_cache_put_time_us",
    log_waitcounter=True,
)


class RemoteCacheBackend(Generic[_T]):
    """
    A backend implementation for cache storage. Most backends are remote, but
    local filesystem backends use the same bytes-oriented interface. For
    structured data use a RemoteCache.
    """

    def __init__(self) -> None:
        self._name = f"backend:{type(self).__name__}"

    @abstractmethod
    def _get(self, key: str) -> _T | None:
        pass

    @abstractmethod
    def _put(self, key: str, data: _T) -> None:
        pass

    def get(self, key: str) -> _T | None:
        try:
            value = self._get(key)
            cache_stats.get(self._name, value)
        except Exception:
            cache_stats.exception(self._name)
            raise
        return value

    def put(self, key: str, data: _T) -> None:
        try:
            self._put(key, data)
            cache_stats.put(self._name)
        except Exception:
            cache_stats.exception(self._name)
            raise


# Serde that encodes from _T to _U and decodes from _U to _T.
class RemoteCacheSerde(Generic[_T, _U]):
    @abstractmethod
    def encode(self, data: _T) -> _U:
        pass

    @abstractmethod
    def decode(self, data: _U) -> _T:
        pass


JsonDataTy: TypeAlias = Union[  # noqa: UP007
    int, float, str, bool, dict[str, "JsonDataTy"], list["JsonDataTy"], None
]


class RemoteCacheJsonSerde(RemoteCacheSerde[JsonDataTy, bytes]):
    def encode(self, data: JsonDataTy) -> bytes:
        return bytes(json.dumps(data), "ascii")

    def decode(self, data: bytes) -> JsonDataTy:
        return json.loads(data)


class RemoteCachePassthroughSerde(RemoteCacheSerde[_T, _T]):
    def encode(self, data: _T) -> _T:
        return data

    def decode(self, data: _T) -> _T:
        return data


class LocalCacheBackend(RemoteCacheBackend[bytes]):
    """
    A local filesystem implementation of the cache backend interface.
    """

    @override
    def _get(self, key: str) -> bytes | None:
        try:
            with open(key, "rb") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    @override
    def _put(self, key: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(key), exist_ok=True)
        from torch._inductor import codecache

        codecache.write_atomic(key, data)


# This class is the top of a RemoteCache. A RemoteCache is fundamentally made of
# three parts:
#
# 1. The controller (this class).
# 2. A serializer/deserializer (instance of RemoteCacheSerde).
# 3. A backend (instance of RemoteCacheBackend).
#
# To write (`put`), the RemoteCache takes data, uses the RemoteCacheSerde to
# convert it for the backend and passes it to the backend.
#
# Conversely when reading (`get`), the RemoteCache takes data from the backend,
# uses the RemoteCacheSerde to convert it and returns it.
#
# The RemoteCacheBackend is generic on _U - which is the type of data the
# backend can directly cache (usually `bytes`).
#
# The RemoteCacheSerde is responsible for converting between _T (the type of
# data the RemoteCache accepts in `put` and returns in `get`) and _U.
#
# When instantiating a RemoteCache you should override, not directly create a
# RemoteCache. The reason is that when logging cache use (`TORCH_LOGS=cache`) we
# use the concrete type of the RemoteCache as the reported cache. See
# RemoteFxGraphCache below as an example.
class RemoteCache(Generic[_T]):
    backend_override_cls: Callable[[], RemoteCacheBackend[Any]] | None = None

    def __init__(
        self, backend: RemoteCacheBackend[_U], serde: RemoteCacheSerde[_T, _U]
    ) -> None:
        # Support for testing to mock out the backend on a class-by-class basis.
        if (override_cls := self.__class__.backend_override_cls) is not None:
            self.backend = override_cls()
        else:
            self.backend = backend
        # pyrefly: ignore [invalid-type-var]
        self.serde = serde

    # See if the cache contains `key`. Returns `None` if the value is not
    # present in the cache.
    def get(self, key: str) -> _T | None:
        with _WaitCounter("pytorch.remote_cache.get").guard():
            sample = self._create_sample()
            try:
                result = self._get(key, sample)
                cache_stats.get(type(self).__name__, result)
            except Exception as e:
                cache_stats.exception(type(self).__name__)
                if sample:
                    sample.fail_reason = str(e)
                raise
            finally:
                self._log_sample(sample)
            return result

    # Add `value` to the cache with the key `key`. Note that `None` is not a
    # valid value even if _T supports it (because you can't tell the difference
    # between `None` and a missing cache entry).
    def put(self, key: str, value: _T) -> None:
        with _WaitCounter("pytorch.remote_cache.put").guard():
            if value is None:
                raise AssertionError("cannot put None into the cache")
            sample = self._create_sample()
            try:
                self._put(key, value, sample)
                cache_stats.put(type(self).__name__)
            except Exception as e:
                cache_stats.exception(type(self).__name__)
                if sample:
                    sample.fail_reason = str(e)
                raise
            finally:
                self._log_sample(sample)

    # Used to convert data from the cache into structured data.
    def _decode(self, data: _U, sample: Sample | None) -> _T:  # type: ignore[override]
        return self.serde.decode(data)  # type: ignore[arg-type]

    # Used to convert structured data into data for the cache.
    def _encode(self, value: _T, sample: Sample | None) -> object:  # returns _U
        return self.serde.encode(value)

    # Get structured data from the cache.
    # Separate from `get` so that it can be overridden.
    def _get(self, key: str, sample: Sample | None) -> _T | None:
        if data := self._backend_get(key):
            return self._decode(data, sample)
        return None

    # Get unstructured data from the cache.
    # Separate from `get` so that it can be overridden.
    # Returns _U - but we aren't actually generic on _U
    def _backend_get(self, key: str) -> object:
        return self.backend.get(key)

    # Put structured data into the cache.
    # Separate from `put` so that it can be overridden.
    def _put(self, key: str, value: _T, sample: Sample | None) -> None:
        data = self._encode(value, sample)
        self._backend_put(key, data)

    # Put unstructured data into the cache.
    # Separate from `put` so that it can be overridden.
    # Takes data: _U - but we aren't actually generic on _U
    def _backend_put(self, key: str, data: object) -> None:
        self.backend.put(key, data)

    # Create a logging Sample - used with internal loggers to monitor cache
    # effectiveness.
    def _create_sample(self) -> Sample | None:
        return None

    # Write the logging Sample to the logger.
    def _log_sample(self, sample: Sample | None) -> None:
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend[bytes]):
    """
    A Redis implementation of a remote/distributed cache.
    """

    # pyrefly: ignore [missing-attribute]
    _redis: redis.Redis | None = None

    def __init__(self, cache_id: str) -> None:
        super().__init__()
        if not redis:
            raise RuntimeError("redis not available but required for remote cache")

        if "TORCHINDUCTOR_REDIS_URL" in os.environ:
            self._redis = redis.Redis.from_url(os.environ["TORCHINDUCTOR_REDIS_URL"])
        else:
            self._redis = redis.Redis(
                host=os.environ.get("TORCHINDUCTOR_REDIS_HOST", "localhost"),
                port=int(os.environ.get("TORCHINDUCTOR_REDIS_PORT", 6379)),
            )

    @override
    def _get(self, key: str) -> bytes | None:
        if not self._redis:
            # Either redis wasn't found or we already had some trouble...
            return None

        try:
            # pyrefly: ignore [missing-attribute]
            value = self._redis.get(key)
        # pyrefly: ignore [missing-attribute]
        except redis.exceptions.ConnectionError:
            # Redis is lazy and doesn't actually attempt to connect until the
            # first use. Mark it as unavailable now.
            self._redis = None
            return None

        # In theory redis.get() can return an Awaitable as well...
        if not (value is None or isinstance(value, bytes)):
            raise AssertionError(f"expected bytes or None, got {type(value)}")
        return value

    @override
    def _put(self, key: str, data: bytes) -> None:
        if not self._redis:
            # Either redis wasn't found or we already had some trouble...
            return

        try:
            # pyrefly: ignore [missing-attribute]
            self._redis.set(key, data)
        # pyrefly: ignore [missing-attribute]
        except redis.exceptions.ConnectionError:
            # Redis is lazy and doesn't actually attempt to connect until the
            # first use. Mark it as unavailable now.
            self._redis = None


class RedisRemoteCache(RemoteCache[JsonDataTy]):
    def __init__(self, cache_id: str) -> None:
        # Special test handling: If we're just going to override the backend
        # anyway don't require redis
        if self.__class__.backend_override_cls:
            # This is totally bogus but it works for now...
            backend = typing.cast(RemoteCacheBackend[bytes], None)
        else:
            backend = RedisRemoteCacheBackend(cache_id)
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)
        version = 1  # consistency between various types of keys
        self._key_fmt = f"pt2:{cache_id}::{{key}}:c{version}"

    def _get_key(self, key: str) -> str:
        return self._key_fmt.format(key=key)

    @override
    def _get(self, key: str, sample: Sample | None) -> JsonDataTy | None:
        key = self._get_key(key)
        return super()._get(key, sample)

    @override
    def _put(self, key: str, value: JsonDataTy, sample: Sample | None) -> None:
        key = self._get_key(key)
        super()._put(key, value, sample)


class LocalCache(RemoteCache[JsonDataTy]):
    def __init__(self, cache_id: str) -> None:
        # Local caches use the per-operation key as the filesystem path. Accept
        # cache_id so they can be constructed through create_cache like remote
        # caches.
        backend = LocalCacheBackend()
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)

    @override
    def _get(self, key: str, sample: Sample | None) -> JsonDataTy | None:
        try:
            return super()._get(key, sample)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning(
                "Ignoring corrupt local cache entry %s: %s: %s",
                key,
                type(exc).__name__,
                exc,
            )
            return None


class LocalAutotuneCache(LocalCache):
    # Keep a distinct cache type for cache stats and test backend overrides.
    pass


class RemoteAutotuneCache(RedisRemoteCache):
    pass


class RemoteBundledAutotuneCache(RedisRemoteCache):
    pass


class RemoteFxGraphCache(RedisRemoteCache):
    pass


class RemoteAOTAutogradCache(RedisRemoteCache):
    pass


class RemoteDynamoPGOCache(RedisRemoteCache):
    pass


def create_cache(
    key: str,
    is_fbcode: bool = False,
    fb_cache_cls: str | None = None,
    oss_cache_cls: str | None = None,
    *,
    local_cache_cls: str | None = None,
) -> RemoteCache[JsonDataTy] | None:
    try:
        this_module = sys.modules[__name__]
        if local_cache_cls is not None:
            cache_cls = getattr(this_module, local_cache_cls)
            return cache_cls(key)
        elif is_fbcode:
            if fb_cache_cls is None:
                raise AssertionError("fb_cache_cls must not be None in fbcode")
            import torch._inductor.fb.remote_cache

            cache_cls = getattr(torch._inductor.fb.remote_cache, fb_cache_cls)
            return cache_cls(key)
        else:
            if oss_cache_cls is None:
                raise AssertionError("oss_cache_cls must not be None")
            cache_cls = getattr(this_module, oss_cache_cls)
            return cache_cls(key)

    except Exception:
        log.warning("Unable to create cache", exc_info=True)
        return None


# Some simple stat capture
@dataclasses.dataclass
class _CacheStat:
    miss: int = 0
    hit: int = 0
    put: int = 0
    exception: int = 0

    def __str__(self) -> str:
        return f"{{hit: {self.hit}, miss: {self.miss}, put: {self.put}, exception: {self.exception}}}"


class _CacheStats:
    _stats: dict[str, _CacheStat]

    def __init__(self) -> None:
        self._stats = collections.defaultdict(_CacheStat)

    def miss(self, name: str, count: int = 1) -> None:
        self._stats[name].miss += count

    def hit(self, name: str, count: int = 1) -> None:
        self._stats[name].hit += count

    def get(self, name: str, value: object | None) -> None:
        if value is None:
            self.miss(name)
        else:
            self.hit(name)

    def put(self, name: str, count: int = 1) -> None:
        self._stats[name].put += count

    def exception(self, name: str, count: int = 1) -> None:
        self._stats[name].exception += count


cache_stats = _CacheStats()


@atexit.register
def dump_cache_stats() -> None:
    if not log.isEnabledFor(logging.INFO):
        return

    import io

    out = io.StringIO()

    if not cache_stats._stats:
        print(" None", file=out)
    else:
        print(file=out)
        for k, v in sorted(cache_stats._stats.items()):
            print(f"  {k}: {v}", file=out)

    log.info("Cache Metrics:%s", out.getvalue())
<|endoftext|>import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import torch._export
from torch._inductor.utils import is_cpu_device

from .runtime.runtime_utils import cache_dir


log = logging.getLogger(__name__)


@dataclass
class AOTICompileBackend:
    compile_fn: Callable[..., str]
    load_fn: Callable[[str, str, str], list[dict[str, Any] | None]]


_aoti_compile_backends: dict[str, AOTICompileBackend] = {}


def register_aoti_compile_backend(
    device_type: str,
    compile_fn: Callable[..., str],
    load_fn: Callable[[str, str, str], list[dict[str, Any] | None]],
) -> None:
    _aoti_compile_backends[device_type] = AOTICompileBackend(
        compile_fn=compile_fn,
        load_fn=load_fn,
    )


def aoti_eager_cache_dir(namespace: str, device: str) -> Path:
    return Path(cache_dir()) / "aoti_eager" / namespace / device


def aoti_eager_op_conf_lock(op_func_name_with_overload: str) -> Any:
    # Avoid circular import
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT
    from torch.utils._filelock import FileLock

    op_conf_lock_file = f"{op_func_name_with_overload}.lock"
    lock_dir = get_lock_dir()
    return FileLock(os.path.join(lock_dir, op_conf_lock_file), timeout=LOCK_TIMEOUT)


def load_aoti_eager_cache(
    ns: str, op_func_name_with_overload: str, device_type: str
) -> list[dict[str, Any] | None]:
    backend = _aoti_compile_backends.get(device_type)
    if backend:
        return backend.load_fn(ns, op_func_name_with_overload, device_type)

    device_kernel_cache = aoti_eager_cache_dir(ns, device_type)
    op_conf = device_kernel_cache / f"{op_func_name_with_overload}.json"
    if not op_conf.exists():
        return []

    try:
        with aoti_eager_op_conf_lock(op_func_name_with_overload):
            with open(op_conf) as f:
                json_data = json.load(f)
                for item in json_data:
                    # Get absolute path for kernel library
                    kernel_lib_abs_path = device_kernel_cache / item["kernel_path"]
                    item["kernel_path"] = kernel_lib_abs_path.as_posix()

                    # Check if the kernel library exists
                    if not kernel_lib_abs_path.exists():
                        return []

                    for metadata in item["meta_info"]:
                        if metadata.get("is_dynamic"):
                            raise NotImplementedError(
                                "Only support static shape for now"
                            )
                        if (
                            "device_type" in metadata
                            and metadata["device_type"] == "cpu"
                        ):
                            metadata["device_index"] = -1
                        for dtype_key in ["dtype", "dtype_value"]:
                            if dtype_key in metadata:
                                metadata[dtype_key] = getattr(
                                    torch, metadata[dtype_key].split(".")[-1]
                                )
                        if "layout_value" in metadata:
                            metadata["layout_value"] = getattr(
                                torch, metadata["layout_value"].split(".")[-1]
                            )
                        if "memory_format_value" in metadata:
                            metadata["memory_format_value"] = getattr(
                                torch, metadata["memory_format_value"].split(".")[-1]
                            )

                return json_data
    except Exception as e:
        err_msg = f"Failed to load aoti eager cache: {e}"
        log.exception(err_msg)
        return []


def supported_builtin_dtype_torch_dtype() -> dict[type, torch.dtype]:
    return {int: torch.int32, float: torch.float, bool: torch.bool}


def supported_scalar_types() -> tuple[type, ...]:
    type_to_torch_dtype = supported_builtin_dtype_torch_dtype()
    return tuple(type_to_torch_dtype.keys())


def extract_tensor_metadata(dynamic: bool, input: torch.Tensor) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    metadata["is_dynamic"] = dynamic

    if not isinstance(input, torch.Tensor):
        raise AssertionError(f"expected torch.Tensor, got {type(input)}")
    metadata["device_type"] = f"{input.device.type}"
    if is_cpu_device([input]):
        metadata["device_index"] = -1
    else:
        metadata["device_index"] = input.device.index
    metadata["dtype"] = f"{input.dtype}"
    metadata["sizes"] = list(input.size())
    metadata["strides"] = list(input.stride())
    metadata["requires_grad"] = input.requires_grad
    metadata["dispatch_key_set"] = torch._C._dispatch_keys(input).raw_repr()
    return metadata


def extract_tensor_list_metadata(
    dynamic: bool,
    input: list[torch.Tensor],
) -> dict[str, Any]:
    metadata_list = []
    for item in input:
        if not isinstance(item, torch.Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(item)}")
        metadata_list.append(extract_tensor_metadata(dynamic, item))

    metadata: dict[str, Any] = {}
    metadata["tensor_list"] = metadata_list
    return metadata


def extract_scalar_metadata(device_type: str, input: Any) -> dict[str, Any]:
    if not isinstance(input, supported_scalar_types()):
        raise AssertionError(f"expected a supported scalar type, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["is_dynamic"] = False
    # Scalar tensor
    metadata["device_type"] = device_type
    metadata["device_index"] = -1 if device_type == "cpu" else 0
    type_to_torch_dtype = supported_builtin_dtype_torch_dtype()
    metadata["dtype"] = f"{type_to_torch_dtype[type(input)]}"
    metadata["scalar_value"] = input
    return metadata


def extract_string_metadata(input: str) -> dict[str, Any]:
    if not isinstance(input, str):
        raise AssertionError(f"expected str, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["string_value"] = input
    return metadata


def extract_dtype_metadata(input: torch.dtype) -> dict[str, Any]:
    if not isinstance(input, torch.dtype):
        raise AssertionError(f"expected torch.dtype, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["dtype_value"] = f"{input}"
    return metadata


def extract_device_metadata(input: torch.device) -> dict[str, Any]:
    if not isinstance(input, torch.device):
        raise AssertionError(f"expected torch.device, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["device_type_value"] = f"{input.type}"
    metadata["device_index_value"] = input.index
    return metadata


def extract_layout_metadata(input: torch.layout) -> dict[str, Any]:
    if not isinstance(input, torch.layout):
        raise AssertionError(f"expected torch.layout, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["layout_value"] = f"{input}"
    return metadata


def extract_int_list_metadata(input: list[int]) -> dict[str, Any]:
    if not (
        isinstance(input, (list, tuple))
        and all(isinstance(item, int) and not isinstance(item, bool) for item in input)
    ):
        raise AssertionError(f"expected a list/tuple of int, got {input!r}")
    metadata: dict[str, Any] = {}
    metadata["int_list_value"] = list(input)
    return metadata


def aoti_compile_with_persistent_cache(
    ns: str,
    op_func_name_with_overload: str,
    device_type: str,
    dynamic: bool,
    f: Callable[..., Any],
    args: tuple[Any],
    kwargs: dict[str, Any],
    *,
    dynamic_shapes: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
) -> str:
    """
    Compile the given function with persistent cache for AOTI eager mode.
    """
    backend = _aoti_compile_backends.get(device_type)
    if backend:
        return backend.compile_fn(
            ns,
            op_func_name_with_overload,
            device_type,
            dynamic,
            f,
            args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            options=options,
            remove_runtime_assertions=remove_runtime_assertions,
            disable_constraint_solver=disable_constraint_solver,
        )

    if dynamic:
        raise AssertionError("Only support static shape for now")
    flattened_inputs = list(args) + list(kwargs.values())
    if not all(
        isinstance(
            input,
            (
                supported_scalar_types(),
                torch.Tensor,
                list,
                str,
                torch.dtype,
                torch.device,
                torch.layout,
            ),
        )
        for input in flattened_inputs
    ):
        err_msg = f"Unsupported input types: {flattened_inputs}"
        log.exception(err_msg)
        raise NotImplementedError(err_msg)

    for input in flattened_inputs:
        if isinstance(input, list) and not all(
            isinstance(item, torch.Tensor) for item in input
        ):
            err_msg = f"_impl_with_aoti_compile encounters unsupported input types: {flattened_inputs}"
            log.exception(err_msg)
            raise NotImplementedError(err_msg)

    persistent_cache = aoti_eager_cache_dir(ns, device_type)
    if not persistent_cache.exists():
        persistent_cache.mkdir(parents=True)

    persistent_cache_lib = persistent_cache / "lib"
    if not persistent_cache_lib.exists():
        persistent_cache_lib.mkdir()

    with mock.patch.dict(
        os.environ,
        {"TORCHINDUCTOR_CACHE_DIR": persistent_cache_lib.absolute().as_posix()},
    ):
        try:
            kernel_lib_path = torch._export.aot_compile(
                f,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                remove_runtime_assertions=remove_runtime_assertions,
                disable_constraint_solver=disable_constraint_solver,
                # Some operations may have non-Tensor parameters like int, float, bool. These
                # non-Tensor parameters will not be the input of the graph. Therefore, we do
                # need to keep the same signature.
                same_signature=False,
            )
            if not isinstance(kernel_lib_path, str):
                raise AssertionError(
                    f"expected kernel_lib_path to be str, got {type(kernel_lib_path)}"
                )

            kernel_metadata_items = []

            for idx, input in enumerate(flattened_inputs):
                # Skip explicit-None args (e.g. an optional `int[1]? dim`) so the
                # written metadata vector matches the cache reader, which also
                # drops None entries.
                if input is None:
                    continue
                if isinstance(input, torch.Tensor):
                    metadata = extract_tensor_metadata(dynamic, input)
                elif isinstance(input, (list, tuple)) and all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in input
                ):
                    # `int[]` / `SymInt[]` params. Checked before the Tensor-list
                    # branch so an empty list/tuple is tagged INT_LIST, matching
                    # the cache reader and C++ unpack_input_parameters for an
                    # empty `int[]`. An empty `Tensor[]` is the converse case:
                    # C++ tags it TENSOR_LIST (isTensorList() is checked before
                    # isIntList()) while this tags it INT_LIST -- only a cache
                    # miss/recompile (per-op cache, fixed schema type per slot ->
                    # no aliasing), and it does not arise for real ops
                    # (`cat([])`/`stack([])` raise in eager).
                    metadata = extract_int_list_metadata(list(input))
                elif isinstance(input, (list, tuple)) and all(
                    isinstance(item, torch.Tensor) for item in input
                ):
                    metadata = extract_tensor_list_metadata(dynamic, list(input))
                elif isinstance(input, supported_scalar_types()):
                    metadata = extract_scalar_metadata(device_type, input)
                elif isinstance(input, str):
                    metadata = extract_string_metadata(input)
                elif isinstance(input, torch.dtype):
                    metadata = extract_dtype_metadata(input)
                elif isinstance(input, torch.device):
                    metadata = extract_device_metadata(input)
                elif isinstance(input, torch.layout):
                    metadata = extract_layout_metadata(input)
                else:
                    raise NotImplementedError(f"Unsupported input type: {type(input)}")

                metadata["arg_order"] = idx
                kernel_metadata_items.append(metadata)

            kernel_meta_info: dict[str, Any] = {}
            kernel_meta_info["meta_info"] = kernel_metadata_items
            kernel_meta_info["kernel_path"] = (
                Path(kernel_lib_path).relative_to(persistent_cache).as_posix()
            )

            json_data = []
            update_json = True
            op_conf = persistent_cache / f"{op_func_name_with_overload}.json"
            mode = "r" if op_conf.exists() else "w"
            with aoti_eager_op_conf_lock(op_func_name_with_overload):
                with open(op_conf, mode) as op_conf_file:
                    try:
                        json_data = json.load(op_conf_file)
                    except Exception:
                        json_data = []

                    if not isinstance(json_data, list):
                        raise AssertionError(
                            f"expected json_data to be list, got {type(json_data)}"
                        )
                    for item in json_data:
                        if not isinstance(item, dict):
                            raise AssertionError(
                                f"expected item to be dict, got {type(item)}"
                            )
                        # Same kernel meta info already exists in the json file
                        if item["meta_info"] == kernel_metadata_items:
                            update_json = False
                            break

                if update_json:
                    json_data.append(kernel_meta_info)
                    with open(op_conf, "w") as op_conf_file:
                        json.dump(json_data, op_conf_file, indent=4)

            return kernel_lib_path
        except Exception as e:
            err_msg = f"Failed to compile {op_func_name_with_overload}: {e}"
            log.exception(err_msg)
            return ""
<|endoftext|>import logging
import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import sympy
from sympy import Expr

import torch
from torch.utils._sympy.value_ranges import (
    bound_sympy,
    SymPyValueRangeAnalysis,
    ValueRanges,
)

from ..utils._sympy.functions import PowByNatural
from ..utils._sympy.numbers import int_oo
from .loop_body import InterpreterShim, LoopBody, LoopBodyBlock
from .ops_handler import DefaultHandler, ReductionType, StoreMode
from .utils import cache_on_self, dominated_nodes
from .virtualized import V


log = logging.getLogger(__name__)


class BoundVars:
    """
    Performs Value Range Analysis on LoopBody's fx graph by calling BoundVars.run()
    It exposes the ranges of the nodes in the `bounds` variable

    Note. A current limitation of this analysis is that it just works on a per-loop basis.
    We should be able to propagate the bounds between across the whole graph. This may benefit
    the case a bounded variable is returned by a kernel and fed into another.
    """

    def __init__(self, loop_body: LoopBody) -> None:
        def upper_bound(v: Expr | int) -> int:
            return bound_sympy(v).upper if isinstance(v, Expr) else v

        self.loop_body = loop_body
        self.replacement_vals = {
            k: ValueRanges[Expr](0, upper_bound(v) - 1)
            for k, v in loop_body.var_ranges.items()
        }
        # avoid computing these values, pessimistically assume that they are unbounded
        self.unbounded_vars = dominated_nodes(
            node
            for node in self.loop_body.get_nodes()
            if node.target in ["load", "reduction", operator.getitem]
            or "masked_subblock" in node.target
        )
        # To access this variable call `get_bounds()`
        self._bounds: dict[torch.fx.Node, ValueRanges[Expr]] = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loop_body={self.loop_body},\n "
            f"replacement_vals={self.replacement_vals}, \n"
            f"unbounded_vars={self.unbounded_vars}, \n"
            f"_bounds={self._bounds})"
        )

    @cache_on_self
    def get_bounds(self) -> dict[torch.fx.Node, ValueRanges[Expr]]:
        submodules = self.swap_submodules(self.loop_body.submodules)

        # Initialize the environment with the unbounded variables
        for node in self.unbounded_vars:
            # we need to evaluate masked_subblock to recurse, and we need to set indirect values
            if not isinstance(node.target, str) or (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self._bounds[node] = ValueRanges[Expr].unknown()

        with V.set_ops_handler(ValueRangeAnalysis()):
            interpreter = InterpreterShim(self.loop_body.root_block.graph, submodules)
            log.debug("get_bounds:\n%s", self.loop_body.root_block.graph)
            interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
        return self._bounds

    def swap_submodules(
        self, submodules: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., ValueRanges[Expr]]]:
        result: dict[str, Callable[..., ValueRanges[Expr]]] = {}
        for key in submodules:
            if key == "get_index":
                result[key] = self.get_index
            elif "masked_subblock" in key:
                subblock = self.loop_body.subblocks[key]
                # The result within the lambda will reference to the final
                # set of modules at the end of the for-loop as it stores a reference to it

                # bind subblock in a function because python lambdas close over by reference
                # moving the lambda out of make_fn would close over the reference to subblock,
                # so all lambdas would have the same subblock reference that is the final
                # subblock in the loop
                def make_fn(
                    subblock: LoopBodyBlock,
                ) -> Callable[[Any, Any], ValueRanges[Expr]]:
                    return lambda mask, value: self.masked_subblock(
                        subblock, self._bounds, mask, value, result
                    )

                result[key] = make_fn(subblock)
            elif "set_indirect" in key:
                idx = int(key[len("set_indirect") :])
                var = self.loop_body.indirect_vars[idx]
                indirect = partial(self.set_indirect, var)
                result[key] = indirect
            else:
                if "scan" not in key:
                    raise AssertionError(
                        f"expected 'scan' in submodule key, got {key!r}"
                    )
                result[key] = submodules[key]

        return result

    def masked_subblock(
        self,
        subblock: LoopBodyBlock,
        env: dict[torch.fx.Node, ValueRanges[Expr]],
        mask: Any,
        value: Any,
        submodules: dict[str, Callable[..., Any]],
    ) -> ValueRanges[Expr]:
        interp = InterpreterShim(subblock.graph, submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        if len(output) != 1:
            raise AssertionError(f"expected exactly 1 output node, got {len(output)}")
        # don't bother unioning with value since the load from buffer will be
        # pessimistically assumed to be inf anyway
        return interp.env[output[0]]

    def set_indirect(self, old: Expr, new: ValueRanges[Expr]) -> ValueRanges[Expr]:
        if not isinstance(new, ValueRanges):
            raise AssertionError(f"expected ValueRanges, got {type(new)}")
        self.replacement_vals[old] = new
        return new

    def get_index(self, name: str) -> ValueRanges[Expr]:
        expr = self.loop_body.indexing_exprs[name]
        bound = self.replacement_vals.get(expr)
        if bound is None:
            bound = bound_sympy(expr, self.replacement_vals)
        # The following assertion is true at the time of this writing
        # We don't assert is as to not execute bound_sympy when bound is not None
        # assert bound is None or bound == bound_sympy(expr, self.replacement_vals)
        self.replacement_vals[name] = bound
        return bound


class ValueRangeAnalysis(SymPyValueRangeAnalysis, DefaultHandler):
    def __init__(self) -> None:
        self.name = "ValueRangeAnalysis"
        boolean_operators = (
            "xor",
            "logical_and",
            "logical_or",
            "logical_not",
        )
        for op in boolean_operators:
            setattr(self, op, self.bool_handler)

    @staticmethod
    def bool_handler(*args: Any, **kwargs: Any) -> ValueRanges[Any]:
        # just assuming bools can have both values
        return ValueRanges(sympy.false, sympy.true)  # type: ignore[arg-type]

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # many ops are unlikely to show up in optimizable indexing compute,
        # so we don't have full coverage
        return ValueRanges.unknown()

    def load(self, name: str, index: sympy.Expr) -> ValueRanges[Any]:
        return ValueRanges.unknown()

    def store(
        self, name: str, index: sympy.Expr, value: Any, mode: StoreMode = None
    ) -> None:
        return

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Any,
    ) -> ValueRanges[Any]:
        return ValueRanges.unknown()

    @classmethod
    def index_expr(cls, index: Any, dtype: torch.dtype) -> ValueRanges[Any]:
        if not isinstance(index, ValueRanges):
            raise AssertionError(f"expected ValueRanges, got {type(index)}")
        return cls.to_dtype(index, dtype)

    @classmethod
    def value_expr(cls, index: Any, dtype: torch.dtype) -> ValueRanges[Any]:
        return cls.index_expr(index, dtype)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def to_dtype(
        x: Any,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> ValueRanges[Any]:
        x = ValueRanges.wrap(x)

        if dtype == torch.bool:
            if x.is_singleton():
                return ValueRanges.wrap(x.lower != 0)
            elif x.is_bool:
                return x
            elif 0 not in x:
                return ValueRanges.wrap(sympy.true)
            else:
                return ValueRanges(sympy.false, sympy.true)

        def cast(x: Any, dtype: torch.dtype) -> sympy.Expr:
            # dtype is int or float
            if dtype.is_floating_point:
                return sympy.Float(x)
            else:
                if x in (int_oo, -int_oo):
                    return x
                try:
                    return sympy.Integer(x)
                except TypeError:
                    # inf cannot be cast to Integer
                    return x

        if x.is_bool:
            if x.is_singleton():
                val = 1 if x.lower else 0
                return ValueRanges.wrap(cast(val, dtype))
            else:
                return ValueRanges(cast(0, dtype), cast(1, dtype))
        else:
            # int to float or float to int
            return ValueRanges(cast(x.lower, dtype), cast(x.upper, dtype))

    @staticmethod
    # pyrefly: ignore [bad-override]
    def square(x: Any) -> ValueRanges[Any]:
        return ValueRanges.convex_min_zero_map(x, lambda y: PowByNatural(y, 2))

    @staticmethod
    # pyrefly: ignore [bad-override]
    def neg(x: Any) -> ValueRanges[Any]:
        return ValueRanges.decreasing_map(x, operator.neg)

    # TODO: this is slightly inaccurate because truncdiv operates at integer
    # precision, but we're going through float truediv which means we can
    # potentially lose precision on the bounds
    @classmethod
    def truncdiv(cls, a: Any, b: Any) -> ValueRanges[Any]:
        x = cls.truediv(a, b)
        if x == ValueRanges.unknown():
            return x

        return cls.trunc(x)

    @classmethod
    def sub(cls, a: Any, b: Any) -> ValueRanges[Any]:
        return cls.add(a, cls.neg(b))
<|endoftext|># mypy: disallow-untyped-defs
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, TYPE_CHECKING, TypeVar

import torch
from torch._dynamo.utils import counters, get_metrics_context
from torch._inductor.utils import GraphPartitionMap, InputType
from torch._subclasses.fake_tensor import get_plain_tensors, is_fake
from torch.utils._ordered_set import OrderedSet

from . import config
from .utils import is_using_cudagraph_partition


if TYPE_CHECKING:
    from collections.abc import Sequence, Set as AbstractSet

    from torch._inductor.output_code import OutputCode

_OC = TypeVar("_OC", bound="OutputCode")


cudagraphs_log = torch._logging.getArtifactLogger(__name__, "cudagraphs")
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)


OutputType = list[int | torch.Tensor | None]
ModelType = Callable[[list[InputType]], OutputType]


def cudagraph_trees_generation_cloning() -> Literal["user_visible"] | None:
    mode = config.triton.cudagraph_trees_generation_cloning
    if mode not in (None, "user_visible"):
        raise AssertionError(
            "Expected torch._inductor.config.triton."
            "cudagraph_trees_generation_cloning to be None or 'user_visible', "
            f"got {mode!r}"
        )
    return mode


def cudagraph_trees_clone_live_user_visible_outputs() -> bool:
    return cudagraph_trees_generation_cloning() == "user_visible"


INPUT_STORAGE_MUTATION_TARGETS = (
    torch.ops.aten.set_.default,
    torch.ops.aten.set_.source_Storage,
    torch.ops.aten.set_.source_Storage_storage_offset,
    torch.ops.aten.set_.source_Tensor,
    torch.ops.aten.set_.source_Tensor_storage_offset,
    torch.ops.aten.shallow_copy_data_.default,
)


class CUDAGraphPolicy:
    """Pluggable policy controlling CUDA graph wrapping in Inductor's post_compile.

    Override methods to customize:
      - HOW compiled functions are cudagraph-wrapped (cudagraphify)
      - WHETHER inner CompiledFxGraphs should be wrapped (should_wrap)
      - OUTER wrapping of compound outputs like RegionalOutputCode (wrap_output)

    Set via ``torch._inductor.config.cudagraph_policy``.  When ``None``
    (the default), the existing built-in behaviour is used unchanged.

    Example usage::

        class MyCUDAGraphPolicy(CUDAGraphPolicy):
            def cudagraphify(self, model, example_inputs, static_input_idxs, **kwargs):
                return my_custom_wrapper(model, example_inputs, static_input_idxs)


        with torch._inductor.config.patch("cudagraph_policy", MyCUDAGraphPolicy()):
            compiled_fn = deserialize_artifacts(...)
    """

    def cudagraphify(
        self,
        model: Callable[..., Any],
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        *,
        device_index: int,
        is_backward: bool,
        is_inference: bool,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Wrap a single compiled callable with CUDA graph capture/replay.

        Called by ``cudagraph_post_compile`` for each ``CompiledFxGraph``.
        The default delegates to ``compile_fx.cudagraphify`` (cudagraph_trees).

        ``example_inputs`` are the example inputs at post_compile time.
        The default implementation does not forward them because
        ``compile_fx.cudagraphify`` defers graph recording to the first
        real call via an inner closure.  Subclasses that need the
        example inputs for warmup or static-input detection may use them.

        When ``config.graph_partition=True``, setting a CUDAGraphPolicy
        bypasses ``cudagraph_partition_post_compile`` (which wraps each
        partition individually) and routes through ``cudagraph_post_compile``
        instead, so this method wraps the *entire* callable, not individual
        partitions.  Subclasses that need per-partition control should
        handle partitioning internally.
        """
        from torch._inductor.compile_fx import cudagraphify

        return cudagraphify(
            model,
            static_input_idxs,
            device_index=device_index,
            is_backward=is_backward,
            is_inference=is_inference,
            **kwargs,
        )

    def should_wrap(self, compiled_graph: OutputCode) -> bool:
        """Whether to apply cudagraph wrapping to this CompiledFxGraph.

        Called for each inner ``CompiledFxGraph`` during ``post_compile``.
        Return ``False`` to skip wrapping (e.g. when wrapping at the outer
        level via ``wrap_output`` instead).

        Default: ``True`` (wrap everything, same as current behaviour).
        """
        return True

    def wrap_output(self, output_code: _OC) -> _OC:
        """Optional outer-level wrapping after inner post_compile completes.

        Called by ``_compile_fx_inner``, ``BundledOutputCodeLoadable.post_compile``,
        and ``FxGraphCacheLoadable.post_compile`` on the ``OutputCode`` returned
        from ``post_compile``.  Subclasses that only want to wrap specific
        output types should check ``isinstance`` and return the input
        unchanged for types they don't handle.

        Default: identity (no outer wrapping).
        """
        return output_code


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionID:
    "Unique counter of a function wrapped in cudagraphify_impl"

    id: int


@dataclasses.dataclass(frozen=True, slots=True)
class PlaceholderInfo:
    """
    A serializable version of torch.fx.Node that contains information
    pertinent to placeholder stack traces. We use these in logging and error messages
    related to cudagraphs, and will cache these results.
    """

    name: str
    stack_trace: str | None
    # This field is recursive, but never cyclic (since a node never uses itself)
    users: list[PlaceholderInfo]
    mutating_use_stack_trace: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class InputStorageMutationInfo:
    input_idxs: OrderedSet[int]
    stack_trace: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class WrappedFunction:
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """

    model: Callable[..., Any]
    static_input_idxs: Sequence[int]
    id: FunctionID
    constants: tuple[torch.Tensor, ...]
    placeholders: Sequence[PlaceholderInfo]
    mutated_input_idxs: Sequence[int]
    kernel_free_cudagraph: bool = False
    user_visible_output_idxs: frozenset[int] = dataclasses.field(
        default_factory=frozenset
    )


def get_mutating_use_stack_trace_from_node(
    placeholder_node: torch.fx.Node,
) -> str | None:
    # reinplaced uses might have a single, non-copy_ use
    if len(placeholder_node.users) == 1:
        return next(iter(placeholder_node.users)).meta.get("stack_trace", None)

    for use in placeholder_node.users:
        if use.target is torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None


def get_mutating_use_stack_trace(placeholder_info: PlaceholderInfo) -> str | None:
    return placeholder_info.mutating_use_stack_trace


def get_input_storage_mutation_info(
    gm: torch.fx.GraphModule,
) -> InputStorageMutationInfo:
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    placeholder_indices = {node: idx for idx, node in enumerate(placeholders)}
    storage_mutation_input_idxs: OrderedSet[int] = OrderedSet()
    stack_trace: str | None = None

    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target not in INPUT_STORAGE_MUTATION_TARGETS
        ):
            continue

        mutated_arg = node.args[0] if node.args else None
        # Only direct graph-input set_ rebinds the tensor object that lives
        # outside this graph. set_ on a temporary view/alias does not rebind the
        # input TensorImpl and is handled by the usual mutation checks.
        if (
            isinstance(mutated_arg, torch.fx.Node)
            and mutated_arg in placeholder_indices
        ):
            storage_mutation_input_idxs.add(placeholder_indices[mutated_arg])
            if stack_trace is None:
                stack_trace = node.meta.get("stack_trace", None)

    return InputStorageMutationInfo(storage_mutation_input_idxs, stack_trace)


def get_input_storage_mutation_reason(
    storage_mutation_info: InputStorageMutationInfo,
) -> str | None:
    storage_mutation_input_idxs = storage_mutation_info.input_idxs
    if not storage_mutation_input_idxs:
        return None

    msg = f"input storage mutation ({len(storage_mutation_input_idxs)} instances)"
    if storage_mutation_info.stack_trace:
        return f"{msg}. Found from:\n {storage_mutation_info.stack_trace}"

    return msg


def to_placeholder_info(placeholder_node: torch.fx.Node) -> PlaceholderInfo:
    name = placeholder_node.name
    stack_trace = placeholder_node.meta.get("stack_trace", None)
    users = []
    mutating_use_stack_trace = None
    # Only recurse to users once, since we only care about user's stack traces
    if placeholder_node.op == "placeholder":
        users = [to_placeholder_info(i) for i in placeholder_node.users]
        mutating_use_stack_trace = get_mutating_use_stack_trace_from_node(
            placeholder_node
        )

    return PlaceholderInfo(name, stack_trace, users, mutating_use_stack_trace)


def get_placeholder_info(graph: torch.fx.Graph) -> list[PlaceholderInfo]:
    return [
        to_placeholder_info(node) for node in graph.nodes if node.op == "placeholder"
    ]


def format_default_skip_message(reason: str) -> str:
    return f"skipping cudagraphs due to {reason}"


def get_mutation_stack_trace(
    placeholders: Sequence[PlaceholderInfo],
    mutation_indices: AbstractSet[int] | Sequence[int],
) -> str:
    stack_trace: str | None = ""

    for idx in mutation_indices:
        placeholder = placeholders[idx]
        if stack_trace := get_mutating_use_stack_trace(placeholder):
            break

    msg = format_default_skip_message(
        f"mutated inputs ({len(mutation_indices)} instances)"
    )
    if stack_trace:
        return f"{msg}. Found from : \n {stack_trace}"

    return msg


def check_for_mutation(
    func: WrappedFunction,
    inputs: list[InputType],
    is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> str | None:
    # doesn't work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        # checking if mutation is only on parameters/static inputs
        mutation_indices: Sequence[int] = [
            idx
            for idx in func.mutated_input_idxs
            if not (
                idx in func.static_input_idxs
                or is_cuda_graph_recorded_tensor(inputs[idx])  # type: ignore[arg-type]
            )
        ]
    else:
        mutation_indices = func.mutated_input_idxs

    static_inputs_log.debug(
        "check mutation static input indices: %s", func.static_input_idxs
    )
    static_inputs_log.debug("check mutation indices: %s", mutation_indices)

    return (
        get_mutation_stack_trace(func.placeholders, mutation_indices)
        if mutation_indices
        else None
    )


def _get_use_stack_trace(node: torch.fx.Node) -> str | None:
    for use in node.users:
        if stack_trace := use.meta.get("stack_trace", None):
            return stack_trace
    return None


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> str | None:
    # meta tensors are supported since there is no compute
    device_node_mapping.pop(torch.device("meta"), None)

    # dynamo cudagraph does not support graph partition
    if is_using_cudagraph_partition():
        # graph partition supports splitting on cpu op. So we can ignore cpu nodes.
        device_node_mapping.pop(torch.device("cpu"), None)

    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        msg = f"cpu device ({cpu_node.name})"
        if stack_trace := _get_use_stack_trace(cpu_node):
            return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

        return format_default_skip_message(msg)

    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "cuda"
    ):
        return None

    keys_repr = (repr(key) for key in device_node_mapping)
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


def check_caching_allocator_for_cudagraphs() -> str | None:
    """Skip cudagraphs when the CUDA/HIP caching allocator is disabled
    (via ``torch.cuda.caching_allocator_enable(False)`` or the env-var
    bypass ``PYTORCH_NO_(CUDA|HIP)_MEMORY_CACHING``). Cudagraph capture
    pools allocations through the caching allocator; with it bypassed,
    capture appears to succeed but pool tracking diverges (see
    check_memory_pool in cudagraph_trees.py), surfacing as 'storage data
    ptrs not allocated in pool ...' at replay time."""
    if (
        torch.cuda.is_available()
        # pyrefly: ignore [missing-attribute]
        and not torch._C._cuda_cudaCachingAllocator_is_enabled()
    ):
        return format_default_skip_message(
            "cudagraph capture requires the caching allocator; "
            "current allocator is uncached"
        )
    return None


def check_lowering_disable_cudagraph(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> str | None:
    return (
        check_caching_allocator_for_cudagraphs()
        or check_multiple_devices_or_any_cpu_nodes(device_node_mapping)
    )


def log_cudagraph_skip_and_bump_counter(msg: str) -> None:
    cudagraphs_log.warning(msg)
    counters["inductor"]["cudagraph_skips"] += 1

    if torch._inductor.config.triton.cudagraph_or_error:
        raise RuntimeError(msg)

    metrics_context = get_metrics_context()
    if metrics_context.in_progress():
        metrics_context.set("cudagraph_skip_reason", msg, overwrite=True)


@dataclasses.dataclass
class BoxedDeviceIndex:
    value: int | None

    def set(self, device_idx: int | None) -> None:
        if not (device_idx is None or isinstance(device_idx, int)):
            raise AssertionError(
                f"expected device_idx to be None or int, got {device_idx!r}"
            )
        self.value = device_idx


def check_for_mutation_ignore_cuda_graph_managed_tensor(
    gm: torch.fx.GraphModule,
    mutated_inputs: OrderedSet[str],
    mutated_input_idxs: OrderedSet[int],
    static_input_idxs: Sequence[int],
) -> str | None:
    default_msg = format_default_skip_message("mutated inputs")

    # doesn't work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        unique_idxs = OrderedSet(static_input_idxs)
        # checking if mutation is only on parameters/static inputs
        mutation_indices = [idx for idx in mutated_input_idxs if idx not in unique_idxs]
        has_mutation = len(mutation_indices) != 0
        if not has_mutation:
            return None
        placeholders = get_placeholder_info(gm.graph)
        return get_mutation_stack_trace(placeholders, mutation_indices)

    else:
        has_mutation = len(mutated_inputs) != 0
        return None if not has_mutation else default_msg


def get_placeholder_stack_trace(placeholder: PlaceholderInfo) -> str | None:
    """
    Gets the first non-empty stack trace of a placeholder or its users.
    """
    if placeholder.stack_trace:
        return placeholder.stack_trace

    for user in placeholder.users:
        if user.stack_trace:
            return user.stack_trace

    return None


class CheckInvariantStatus(Enum):
    # Check invariant succeeded
    SUCCESS = 1

    # Previously managed data pointers are not stable
    CudagraphManagedIdxMismatch = 2

    # Static tensor input addresses are not stable
    StaticInputIdxMismatch = 3

    # Expected dead indices before graph are live
    ExpectedDeadIndicesBeforeGraphMismatch = 4

    def __str__(self) -> str:
        if self.name == "CudagraphManagedIdxMismatch":
            return "cudagraph managed tensor data pointer changed"
        elif self.name == "StaticInputIdxMismatch":
            return "static input data pointer changed"
        elif self.name == "ExpectedDeadIndicesBeforeGraphMismatch":
            return "expected dead indices before graph are live"
        else:
            return f"{self.name}: {self.value}"


def log_data_ptr_mismatch(
    placeholders: Sequence[PlaceholderInfo],
    inputs: list[InputType],
    recorded_data_ptr: Sequence[int | None],
    target_idxs: Sequence[int],
    mismatch: CheckInvariantStatus,
) -> str:
    """
    Logs the mismatch between input data pointers and recorded data pointers.
    This checks only idxs in target_idxs.
    """
    if not (len(inputs) == len(recorded_data_ptr) and len(inputs) == len(placeholders)):
        raise AssertionError(
            "length mismatch between inputs, recorded_data_ptr, and placeholders"
        )

    t_tensors = [inputs[i] for i in target_idxs]
    t_data_ptrs = [recorded_data_ptr[i] for i in target_idxs]
    error_msg = f"{mismatch}.\n"
    for i, (tensor, data_ptr) in enumerate(zip(t_tensors, t_data_ptrs)):
        if not isinstance(tensor, torch.Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(tensor)}")
        index = target_idxs[i]
        if tensor.data_ptr() != data_ptr:
            placeholder = placeholders[index]
            error_msg = (
                f"{error_msg}input name: {placeholder.name}. "
                f"data pointer changed from {data_ptr} to {tensor.data_ptr()}. "
                f"input stack trace: {get_placeholder_stack_trace(placeholder)}\n"
            )
    return error_msg


def maybe_warning_due_to_dynamic_shape(
    fn_cache: dict[tuple[int, ...], Callable[..., Any]],
    new_int_key: Any,
) -> bool:
    num_cudagraphs = len(fn_cache.keys()) + 1

    def warn_msg() -> str:
        return (
            "CUDAGraph supports dynamic shapes by recording a new graph for each "
            "distinct input size. Recording too many CUDAGraphs may lead to "
            f"extra overhead. We have observed {num_cudagraphs} distinct sizes. "
            "Please consider the following options for better performance: "
            "a) padding inputs to a few fixed number of shapes; or b) set "
            "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
            "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
            "to silence this warning."
        )

    if (
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit
        and num_cudagraphs
        > torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit
    ):
        cudagraphs_log.warning(warn_msg())
        return True

    return False


@dataclasses.dataclass(frozen=True)
class CudagraphCachedInfo:
    """
    Info needed to realign inputs
    """

    placeholders: Sequence[PlaceholderInfo]
    stack_traces: list[str | None]
    user_visible_output_idxs: Sequence[int]
    cudagraph_fail_reasons: list[str]


@dataclasses.dataclass(frozen=True)
class CudagraphMetadata:
    """
    Metadata for recording a CUDA graph.
    """

    placeholders: Sequence[PlaceholderInfo]
    static_input_idxs: OrderedSet[int]
    mutated_input_idxs: OrderedSet[int]
    stack_traces: list[str | None]
    user_visible_output_idxs: OrderedSet[int]
    constants: dict[str, torch.Tensor]


def get_partition_cudagraph_metadata(
    partition_map: GraphPartitionMap,
    metadata: CudagraphMetadata,
) -> CudagraphMetadata:
    """
    Convert the cudagraph metadata at the graph level to the graph partition level,
    given the graph partition info (i.e., mapping from partition input/output index
    to graph input/output index).
    """

    partition_placeholders = []
    partition_static_input_idxs: OrderedSet[int] = OrderedSet()
    partition_mutated_input_idxs: OrderedSet[int] = OrderedSet()
    for partition_input_idx, graph_input_idx in enumerate(
        partition_map.input_index_mapping
    ):
        if graph_input_idx in metadata.static_input_idxs:
            partition_static_input_idxs.add(partition_input_idx)

        if graph_input_idx in metadata.mutated_input_idxs:
            partition_mutated_input_idxs.add(partition_input_idx)

        if graph_input_idx is not None:
            placeholder = metadata.placeholders[graph_input_idx]
        else:
            # create a dummy placeholder info since this partition input is not a graph input
            placeholder = PlaceholderInfo(
                name=f"partition_{partition_map.id}_placeholder_{partition_input_idx}",
                stack_trace=None,
                users=[],
                mutating_use_stack_trace=None,
            )
        partition_placeholders.append(placeholder)

    partition_stack_traces = []
    partition_user_visible_output_idxs: OrderedSet[int] = OrderedSet()
    # Graph output metadata must be remapped to the partition output indices
    # passed to cudagraphify.
    for partition_output_idx, graph_output_idxs in enumerate(
        partition_map.output_index_mapping
    ):
        if not graph_output_idxs:
            partition_stack_traces.append(None)
            continue

        user_visible_graph_output_idxs = [
            graph_output_idx
            for graph_output_idx in graph_output_idxs
            if graph_output_idx in metadata.user_visible_output_idxs
        ]
        stack_trace_idx = (
            user_visible_graph_output_idxs[0]
            if user_visible_graph_output_idxs
            else graph_output_idxs[0]
        )
        partition_stack_traces.append(metadata.stack_traces[stack_trace_idx])
        if user_visible_graph_output_idxs:
            partition_user_visible_output_idxs.add(partition_output_idx)

    partition_constants = {
        name: metadata.constants[name] for name in partition_map.constant_names
    }

    return CudagraphMetadata(
        partition_placeholders,
        partition_static_input_idxs,
        partition_mutated_input_idxs,
        partition_stack_traces,
        partition_user_visible_output_idxs,
        partition_constants,
    )


def collect_cuda_data_ptrs(obj: object) -> OrderedSet[int]:
    """Debug helper that collects the data pointers of all CUDA tensors in the object."""
    if not isinstance(obj, torch.Tensor):
        return OrderedSet()

    ptrs: OrderedSet[int] = OrderedSet()
    for base in get_plain_tensors(obj, out=[]):
        if type(base) is not torch.Tensor:
            continue
        if is_fake(base) or base.is_meta or base.device.type != "cuda":
            continue
        try:
            ptrs.add(base.data_ptr())
        except Exception:
            pass
    return ptrs
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import functools
import os
import platform
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor import config
from torch._inductor.utils import python_subprocess_env


_IS_WINDOWS = sys.platform == "win32"


def _get_isa_dry_compile_fingerprint(isa_flags: str) -> str:
    # ISA dry compile will cost about 1 sec time each startup time.
    # Please check the issue: https://github.com/pytorch/pytorch/issues/100378
    # Actually, dry compile is checking compile capability for ISA.
    # We just record the compiler version, isa options and pytorch version info,
    # and generated them to output binary hash path.
    # It would optimize and skip compile existing binary.
    from torch._inductor.cpp_builder import get_compiler_version_info, get_cpp_compiler

    compiler_info = get_compiler_version_info(get_cpp_compiler())
    torch_version = torch.__version__
    fingerprint = f"{compiler_info}={isa_flags}={torch_version}"
    return fingerprint


class VecISA:
    """Describes a CPU SIMD ISA (AVX2, AVX512, NEON, SVE, VSX, ZVECTOR) for
    inductor's CPU vectorization.

    Carries the bit width, ``CPU_CAPABILITY_*`` build macros, compiler arch
    flags, and the per-dtype lane count used by the codegen. Subclasses (one
    per ISA) override the class-level fields; instances are hashable by their
    string representation and usable as a dict key. ``bool(vec_isa)`` runs the
    dry-compile + dlopen probe described in
    `Note [Checking for Vectorized Support in Inductor]`_ to verify the host
    toolchain actually produces a working ``.so`` for this ISA.
    """

    _bit_width: int
    _macro: list[str]
    _arch_flags: str
    _dtype_nelements: dict[torch.dtype, int]

    # Note [Checking for Vectorized Support in Inductor]
    # TorchInductor CPU vectorization reuses PyTorch vectorization utility functions
    # Hence, TorchInductor would depend on Sleef* to accelerate mathematical functions
    # like exp, pow, sin, cos and etc.
    # But PyTorch and TorchInductor might use different compilers to build code. If
    # PyTorch uses gcc-7/g++-7 to build the release package, the libtorch_cpu.so
    # will not expose the Sleef* AVX512 symbols since gcc-7/g++-7 cannot pass
    # avx512 check in CMake - FindAVX.cmake. But TorchInductor install the latest
    # gcc/g++ compiler by default while it could support the AVX512 compilation.
    # Therefore, there would be a conflict sleef version between PyTorch and
    # TorchInductor. Hence, we dry-compile the following code to check whether current
    # HW platform and PyTorch both could support AVX512 or AVX2. And suppose ARM
    # also needs the logic
    # In fbcode however, we are using the same compiler for pytorch and for inductor codegen,
    # making the runtime check unnecessary.
    _avx_code = """
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_NEON) || defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_SVE)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

alignas(64) float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
"""

    # Skip the slow `import torch` (which has hung under load in CI) on Linux
    # only: the parent puts torch's lib dir on LD_LIBRARY_PATH so the linker
    # resolves libtorch_cpu for the test .so on its own. macOS can't do this --
    # SIP strips DYLD_LIBRARY_PATH from the child -- so it falls back to import.
    # Some wheels only resolve their native deps once `import torch` has run.
    # ROCm wheels bundle ROCm libs under unversioned names but keep some
    # DT_NEEDED refs on the versioned soname (e.g. libamd_comgr.so.3), which
    # only resolves against the copy torch has already loaded; TheRock-based
    # wheels preload ROCm libs from separate rocm-sdk packages with no rpath.
    # So a cold dlopen can fail even though the ISA is fine; retry after
    # importing torch before giving up (#189194).
    # TODO: extend the no-import path to Windows once its CI build is green
    # enough to validate it.
    _avx_py_load = """
import sys
if sys.platform != "linux":
    import torch  # noqa: F401
from ctypes import cdll
try:
    cdll.LoadLibrary("__lib_path__")
except OSError:
    if sys.platform != "linux":
        raise
    import torch  # noqa: F401
    cdll.LoadLibrary("__lib_path__")
"""

    def bit_width(self) -> int:
        return self._bit_width

    def nelements(self, dtype: torch.dtype = torch.float) -> int:
        return self._dtype_nelements[dtype]

    def build_macro(self) -> list[str]:
        return self._macro

    def build_arch_flags(self) -> str:
        return self._arch_flags

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _build_probe_env() -> dict[str, str]:
        """Construct the child env for the dlopen probe.

        Make libtorch_cpu (and other torch shlibs) findable by the Linux
        dynamic linker via LD_LIBRARY_PATH so the probe child does not need to
        ``import torch`` to bring them into its address space. Prepend rather
        than append so a successful probe is guaranteed to bind against the
        torch we are currently running, not an older install on the user's
        loader path. macOS and Windows do not use this fast path; they
        ``import torch`` in the child instead (see ``_avx_py_load``).
        """
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        env = python_subprocess_env()
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            os.pathsep.join([lib_dir, existing]) if existing else lib_dir
        )
        return env

    def _probe_load(self, output_path: str) -> bool:
        """Spawn a child Python that ``dlopen``s ``output_path``.

        Bound the dlopen probe so a stuck child cannot hang the parent for
        the whole 30-minute outer timeout (observed in CI). ``subprocess.run``
        kills the child on ``TimeoutExpired`` (``check_call`` leaks it).
        Returns ``False`` on timeout or load failure, ``True`` on success.
        """
        probe_cmd = [
            sys.executable,
            "-c",
            VecISA._avx_py_load.replace("__lib_path__", output_path),
        ]
        try:
            subprocess.run(
                probe_cmd,
                stderr=subprocess.DEVNULL,
                env=self._build_probe_env(),
                timeout=60,
                check=True,
            )
        except subprocess.TimeoutExpired:
            warnings.warn(
                f"VecISA dlopen probe for {self} hung after 60s",
                stacklevel=2,
            )
            return False
        except subprocess.CalledProcessError:
            return False
        return True

    def check_build(self, code: str) -> bool:
        """Dry-compile ``code`` with this ISA's flags and verify the resulting
        shared library can be loaded.

        Compiles into the inductor codecache, then spawns a short-lived
        subprocess to ``dlopen`` the ``.so``. Returns ``True`` if both build
        and load succeed, ``False`` on compile error, load failure, or dlopen
        probe timeout.
        """
        from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT, write
        from torch._inductor.cpp_builder import (
            CppBuilder,
            CppTorchOptions,
            normalize_path_separator,
        )

        key, input_path = write(
            code,
            "cpp",
            extra=_get_isa_dry_compile_fingerprint(self.build_arch_flags()),
        )
        from torch.utils._filelock import FileLock

        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            output_dir = os.path.dirname(input_path)
            buid_options = CppTorchOptions(vec_isa=self, warning_all=False)
            x86_isa_help_builder = CppBuilder(
                key,
                [input_path],
                buid_options,
                output_dir,
            )
            try:
                # Check if the output file exist, and compile when not.
                output_path = normalize_path_separator(
                    x86_isa_help_builder.get_target_file_path()
                )
                if not os.path.isfile(output_path):
                    x86_isa_help_builder.build()
            except Exception:
                return False

            load_ok_marker = output_path + ".load_ok"
            if os.path.isfile(load_ok_marker):
                return True

            load_ok = self._probe_load(output_path)
            if load_ok:
                with open(load_ok_marker, "w") as f:
                    f.write("")
            return load_ok

    def __bool__(self) -> bool:
        return self.__bool__impl(config.cpp.vec_isa_ok)

    @functools.cache  # noqa: B019
    def __bool__impl(self, vec_isa_ok) -> bool:
        if vec_isa_ok is not None:
            return vec_isa_ok

        if config.is_fbcode():
            return True

        return self.check_build(VecISA._avx_code)


@dataclasses.dataclass
class VecNEON(VecISA):
    _bit_width = 128  # This is required to leverage the compute implemented in aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h
    _macro = ["CPU_CAPABILITY_NEON", "AT_BUILD_ARM_VEC256_WITH_SLEEF"]
    _arch_flags = ""  # Unused
    _dtype_nelements = {torch.float: 4, torch.bfloat16: 8, torch.float16: 8}

    def __str__(self) -> str:
        if config.is_fbcode():
            return "neon"
        return "asimd"  # detects the presence of advanced SIMD on armv8-a kernels

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


@dataclasses.dataclass
class VecSVE(VecISA):
    _bit_width: int = 256
    _armv9a_supported: bool | None = dataclasses.field(
        default=None, init=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._bit_width not in (128, 256):
            raise AssertionError(f"unsupported SVE bit width: {self._bit_width}")
        nelements = self._bit_width // 32
        self._macro = [
            "CPU_CAPABILITY_SVE",
            f"CPU_CAPABILITY_SVE{self._bit_width}",
            "AT_BUILD_ARM_VEC256_WITH_SLEEF",
            "__ARM_FEATURE_BF16",
        ]
        self._arch_flags = (
            f"-march=armv8-a+sve+bf16 -msve-vector-bits={self._bit_width}"
        )
        self._armv9a_arch_flags = (
            "-march=armv9-a+sve2+fp16fml+sha3+bf16+i8mm "
            f"-msve-vector-bits={self._bit_width}"
        )
        self._dtype_nelements = {
            torch.float: nelements,
            torch.bfloat16: nelements * 2,
            torch.float16: nelements * 2,
        }

    def __str__(self) -> str:
        if config.is_fbcode():
            return "neon"
        return "asimd"

    def _has_armv9a(self) -> bool:
        if self._armv9a_supported is None:
            try:
                self._armv9a_supported = bool(
                    torch.cpu.get_capabilities().get("sve2", False)
                )
            except Exception:
                self._armv9a_supported = False
        return self._armv9a_supported

    def build_arch_flags(self) -> str:
        if self._has_armv9a():
            return self._armv9a_arch_flags
        return self._arch_flags

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


@dataclasses.dataclass
class VecAVX512(VecISA):
    _bit_width = 512
    _macro = ["CPU_CAPABILITY_AVX512"]
    _arch_flags = (
        "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"
        if not _IS_WINDOWS
        else "/arch:AVX512"
    )  # TODO: use cflags
    _dtype_nelements = {torch.float: 16, torch.bfloat16: 32, torch.float16: 32}
    _is_avx512_bf16_supported = False

    def __str__(self) -> str:
        return "avx512"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]

    _avx512_bf16_code = """
#include <cstdint>
#include <immintrin.h>

extern "C" __m512bh __avx512_bf16_chk_kernel(__m512 a, __m512 b) {
    return _mm512_cvtne2ps_pbh(a, b);
}
"""

    @functools.cache  # noqa: B019
    # pyrefly: ignore [bad-override]
    def __bool__(self) -> bool:
        if super().__bool__():
            if config.is_fbcode():
                return False
            # check avx512_bf16
            if torch.cpu._is_avx512_bf16_supported() and not _IS_WINDOWS:
                # save _arch_flags
                base_flags = self._arch_flags
                # temporarily change _arch_flags for avx512_bf16 check_build
                self._arch_flags += " -mavx512bf16"
                if self.check_build(self._avx512_bf16_code):
                    self._is_avx512_bf16_supported = True
                # restore _arch_flags
                self._arch_flags = base_flags

            return True
        return False

    @functools.lru_cache(None)  # noqa: B019
    def is_avx512_bf16_supported(self) -> bool:
        return self._is_avx512_bf16_supported

    def build_arch_flags(self) -> str:
        if self._is_avx512_bf16_supported:
            return self._arch_flags + " -mavx512bf16"
        else:
            return self._arch_flags


@dataclasses.dataclass
class VecAVX512VNNI(VecAVX512):
    _bit_width = 512
    _arch_flags = VecAVX512._arch_flags + " -mavx512vnni -mavx512vl"
    _dtype_nelements = {
        torch.float: 16,
        torch.bfloat16: 32,
        torch.float16: 32,
        torch.int8: 64,
        torch.uint8: 64,
    }

    def __str__(self) -> str:
        return super().__str__() + " avx512_vnni"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]

    _avx512_vnni_code = """
#include <cstdint>
#include <immintrin.h>

extern "C" __m256i __avx512_vnni_chk_kernel_1(__m256i src, __m256i a, __m256i b) {
    return _mm256_dpbusd_epi32(src, a, b);
}

extern "C" __m512i __avx512_vnni_chk_kernel_2(__m512i src, __m512i a, __m512i b) {
    return _mm512_dpbusd_epi32(src, a, b);
}
"""

    @functools.cache  # noqa: B019
    def __bool__(self) -> bool:
        if super().__bool__():
            if config.is_fbcode():
                return False
            if (
                torch.cpu._is_vnni_supported()
                and not _IS_WINDOWS
                and self.check_build(self._avx512_vnni_code)
            ):
                return True
        return False

    def build_arch_flags(self) -> str:
        return self._arch_flags


@dataclasses.dataclass
class VecAMX(VecAVX512VNNI):
    _arch_flags = VecAVX512VNNI._arch_flags + " -mamx-tile -mamx-bf16 -mamx-int8"
    # check amx_fp16 separately since it is not always supported when amx is supported
    # amx_fp16 intrinsic compilation need gcc >=13 on platforms which support amx_fp16
    _is_amx_fp16_supported = False

    def __str__(self) -> str:
        return super().__str__() + " amx_tile"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__

    _amx_code = """
#include <cstdint>
#include <immintrin.h>

struct amx_tilecfg {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

extern "C" void __amx_chk_kernel() {
  amx_tilecfg cfg = {0};
  _tile_loadconfig(&cfg);
  _tile_zero(0);
  _tile_dpbf16ps(0, 1, 2);
  _tile_dpbusd(0, 1, 2);
}
"""

    _amx_fp16_code = _amx_code.replace("_tile_dpbf16ps", "_tile_dpfp16ps")

    @functools.cache  # noqa: B019
    def __bool__(self) -> bool:
        if super().__bool__():
            if config.is_fbcode():
                return False
            if self.check_build(VecAMX._amx_code) and torch.cpu._init_amx():
                # check amx-fp16 as well when check amx
                if torch.cpu._is_amx_fp16_supported():
                    # save _arch_flags
                    base_flags = self._arch_flags
                    # temporarily change _arch_flags for amx-fp16 check_build
                    self._arch_flags += " -mamx-fp16"
                    if self.check_build(VecAMX._amx_fp16_code):
                        self._is_amx_fp16_supported = True
                    # restore _arch_flags
                    self._arch_flags = base_flags

                return True
        return False

    @functools.lru_cache(None)  # noqa: B019
    def is_amx_fp16_supported(self) -> bool:
        return self._is_amx_fp16_supported

    def build_arch_flags(self) -> str:
        extra_flags = ""
        if self._is_avx512_bf16_supported:
            # avx512_bf16 is not among the base flags, so we need to check and add it here
            # And we need this flag in the WOQ case for dequantization
            extra_flags += " -mavx512bf16"
        if self._is_amx_fp16_supported:
            extra_flags += " -mamx-fp16"
        return self._arch_flags + extra_flags


@dataclasses.dataclass
class VecAVX2(VecISA):
    _bit_width = 256
    _macro = ["CPU_CAPABILITY_AVX2"]
    _arch_flags = (
        "-mavx2 -mfma -mf16c" if not _IS_WINDOWS else "/arch:AVX2"
    )  # TODO: use cflags
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "avx2"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


@dataclasses.dataclass
class VecZVECTOR(VecISA):
    _bit_width = 256
    _macro = [
        "CPU_CAPABILITY_ZVECTOR",
        "CPU_CAPABILITY=ZVECTOR",
        "HAVE_ZVECTOR_CPU_DEFINITION",
    ]
    _arch_flags = "-mvx -mzvector"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "zvector"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


@dataclasses.dataclass
class VecVSX(VecISA):
    _bit_width = 256  # VSX simd supports 128 bit_width, but aten is emulating it as 256
    _macro = ["CPU_CAPABILITY_VSX"]
    _arch_flags = "-mvsx"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "vsx"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


class InvalidVecISA(VecISA):
    _bit_width = 0
    _macro = [""]
    _arch_flags = ""
    _dtype_nelements = {}

    def __str__(self) -> str:
        return "INVALID_VEC_ISA"

    def __bool__(self) -> bool:  # type: ignore[override]
        return False

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


def x86_isa_checker() -> list[str]:
    supported_isa: list[str] = []

    def _check_and_append_supported_isa(
        dest: list[str], isa_supported: bool, isa_name: str
    ) -> None:
        if isa_supported:
            dest.append(isa_name)

    Arch = platform.machine()
    """
    Arch value is x86_64 on Linux, and the value is AMD64 on Windows.
    """
    if Arch != "x86_64" and Arch != "AMD64":
        return supported_isa

    avx2 = torch.cpu._is_avx2_supported()
    avx512 = torch.cpu._is_avx512_supported()
    avx512_vnni = avx512 and torch.cpu._is_vnni_supported()
    amx_tile = torch.cpu._is_amx_tile_supported()

    _check_and_append_supported_isa(supported_isa, avx2, "avx2")
    _check_and_append_supported_isa(supported_isa, avx512, "avx512")
    _check_and_append_supported_isa(supported_isa, avx512_vnni, "avx512_vnni")
    _check_and_append_supported_isa(supported_isa, amx_tile, "amx_tile")

    return supported_isa


invalid_vec_isa = InvalidVecISA()
supported_vec_isa_list = [
    VecAMX(),
    VecAVX512VNNI(),
    VecAVX512(),
    VecAVX2(),
    VecNEON(),
    VecSVE(256),
]


def get_isa_from_cpu_capability(
    capability: str | None,
    vec_isa_list: list[VecISA],
    invalid_vec_isa: InvalidVecISA,
):
    # AMX setting is not supported in eager
    # VecAMX will be prioritized for selection when setting ATEN_CPU_CAPABILITY to avx512
    # TODO add sve256 support
    capability_to_isa_str = {
        "default": "INVALID_VEC_ISA",
        "zvector": "zvector",
        "vsx": "vsx",
        "avx2": "avx2",
        "avx512": "avx512",
    }
    if capability in capability_to_isa_str:
        # pyrefly: ignore [bad-index, index-error]
        isa_str = capability_to_isa_str[capability]
        if isa_str == "INVALID_VEC_ISA":
            return invalid_vec_isa
        for vec_isa in vec_isa_list:
            if isa_str in str(vec_isa):
                return vec_isa

    if capability:
        warnings.warn(f"ignoring invalid value for ATEN_CPU_CAPABILITY {capability}")

    return vec_isa_list[0]


# Cache the cpuinfo to avoid I/O overhead. Meanwhile, the cpuinfo content
# might have too much redundant content that is useless for ISA check. Hence,
# we only cache some key isa information.
@functools.cache
def valid_vec_isa_list() -> list[VecISA]:
    isa_list: list[VecISA] = []
    if sys.platform == "darwin" and platform.processor() == "arm":
        isa_list.append(VecNEON())

    if sys.platform not in ["linux", "win32"]:
        return isa_list

    arch = platform.machine()
    if arch == "s390x":
        with open("/proc/cpuinfo") as _cpu_info:
            while True:
                line = _cpu_info.readline()
                if not line:
                    break
                # process line
                featuresmatch = re.match(r"^features\s*:\s*(.*)$", line)
                if featuresmatch:
                    for group in featuresmatch.groups():
                        if re.search(r"[\^ ]+vxe[\$ ]+", group):
                            isa_list.append(VecZVECTOR())
                            break
    elif arch == "ppc64le":
        isa_list.append(VecVSX())
    elif arch == "aarch64":
        caps = torch.cpu.get_capabilities()
        if (caps.get("sve2", False) or caps.get("sve", False)) and caps.get(
            "bf16", False
        ):
            if caps.get("sve_max_length") == 128:
                isa_list.append(VecSVE(128))
            else:
                isa_list.append(VecSVE(256))
        else:
            isa_list.append(VecNEON())

    elif arch in ["x86_64", "AMD64"]:
        """
        arch value is x86_64 on Linux, and the value is AMD64 on Windows.
        """
        _cpu_supported_x86_isa = x86_isa_checker()
        isa_list.extend(
            isa
            for isa in supported_vec_isa_list
            if all(flag in _cpu_supported_x86_isa for flag in str(isa).split()) and isa
        )

    return isa_list


def pick_vec_isa() -> VecISA:
    if config.is_fbcode() and (platform.machine() in ["x86_64", "AMD64"]):
        return VecAVX2()

    _valid_vec_isa_list: list[VecISA] = valid_vec_isa_list()
    if not _valid_vec_isa_list:
        return invalid_vec_isa

    # If the simdlen is None, set simdlen based on the environment ATEN_CPU_CAPABILITY
    # to control CPU vec ISA
    if config.cpp.simdlen is None:
        return get_isa_from_cpu_capability(
            os.getenv("ATEN_CPU_CAPABILITY"), _valid_vec_isa_list, invalid_vec_isa
        )

    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():
            return isa

    return invalid_vec_isa
<|endoftext|>from __future__ import annotations

import contextlib
import copy
import itertools
import logging
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, TYPE_CHECKING


DynamicShapesType = Literal["from_example_inputs", "from_tracing_context", "from_graph"]

import torch.fx
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.utils import dynamo_timed
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch._inductor.utils import BoxedBool, InputType
from torch._subclasses import FakeTensorMode
from torch._subclasses.fake_tensor import maybe_get_fake_mode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.graph_module import _share_torchbind_and_process_group_on_deepcopy

from . import config
from ._functionalize_collectives import (
    _functionalize_inplace_collectives,
    _unbox_process_group_torchbinds,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.compiler._cache import CacheInfo
    from torch.fx import GraphModule

    from .output_code import OutputCode


log = logging.getLogger(__name__)


class CompiledArtifact(ABC):
    """
    CompiledArtifact class represents the inductor cache artifacts that
    can be invoked in order to avoid repeated compilation.

    CompiledArtifact can be obtained by calling standalone_compile(gm, example_inputs)
    to create a fresh CompiledArtifact from a GraphModule and example inputs.

    Later this CompiledArtifact can be saved to disk, either as a binary or unpacked
    into the provided folder via the CompiledArtifact.save function.

    CompiledArtifact.load provides a way to create a CompiledArtifact from the
    binary or unpacked data.

    Finally, the CompiledArtifact can be invoked via the __call__ method
    to execute the cached artifact.
    """

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: tuple[bytes, CacheInfo] | None,
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    @abstractmethod
    def __call__(self, *args: Any) -> Any: ...

    @abstractmethod
    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None: ...

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        if format == "unpacked":
            # If format is unpacked, it must be a CacheCompiledArtifact
            return CacheCompiledArtifact.load(path=path, format=format)

        if format != "binary":
            raise AssertionError(f"expected format == 'binary', got {format}")
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header == AOTCompiledArtifact.AOT_HEADER:
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                artifact = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")
                return AOTCompiledArtifact.deserialize(artifact)
            # Otherwise, it's in the CacheCompiledArtifact format
            elif header == CacheCompiledArtifact.CACHE_HEADER:
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")
                torch.compiler.load_cache_artifacts(artifact_bytes)
                return CacheCompiledArtifact._load_impl(nullcontext(), key)
            else:
                raise RuntimeError(
                    "Invalid header, expected CacheCompiledArtifact or AOTCompiledArtifact, got: "
                    + header.decode("utf-8")
                )


class CacheCompiledArtifact(CompiledArtifact):
    """
    CompiledArtifact that depends on torch.compiler.save_cache_artifacts
    """

    CACHE_HEADER = bytes("CacheCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: tuple[bytes, CacheInfo] | None,
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    def __call__(self, *args: Any) -> Any:
        return self._compiled_fn(*args)

    def is_saveable(self) -> bool:
        if self._artifacts is None:
            return False
        _, cache_info = self._artifacts
        # 0 means nothing was saved
        # >1 means multiple artifacts were saved, which is concerning
        # (we only expect one)
        return len(cache_info.aot_autograd_artifacts) == 1

    def _validate_and_unpack(self) -> tuple[bytes, str]:
        """Validate the cached artifact, returning ``(artifact_bytes, key)``.

        Single source of the None / empty / multiple aot_autograd_artifacts checks,
        shared by ``_to_binary_bytes`` and ``save``'s unpacked branch. Messages are
        neutral (not tied to ``save``) because ``_to_binary_bytes`` callers obtain the
        bytes without going through ``save``.
        """
        if self._artifacts is None:
            raise RuntimeError("CompiledArtifact has no artifact to serialize")
        artifact_bytes, cache_info = self._artifacts
        if len(cache_info.aot_autograd_artifacts) == 0:
            raise RuntimeError(
                f"CompiledArtifact has no aot_autograd artifacts to serialize. This "
                f"likely means there was something that was not serializable in the graph "
                f"passed to standalone_compile. This can generally be fixed by ensuring "
                f"that your model only uses constructs that are serializable. {cache_info}"
            )
        if len(cache_info.aot_autograd_artifacts) > 1:
            raise AssertionError(
                f"CompiledArtifact has more than one aot_autograd artifact but we only "
                f"expected one. {cache_info}"
            )
        return artifact_bytes, cache_info.aot_autograd_artifacts[0]

    def _to_binary_bytes(self) -> bytes:
        """Serialize this artifact to the in-memory ``binary`` byte format.

        Produces exactly the bytes that ``save(format="binary")`` writes to disk, so
        callers that only need the bytes (rather than a file on disk) can reuse this
        without going through ``save``. The byte layout is the one
        ``load(format="binary")`` reads back: header, ``torch_key``, the autograd-cache
        ``key`` string, then the opaque ``artifact_bytes``.
        """
        artifact_bytes, key = self._validate_and_unpack()

        from torch.utils._appending_byte_serializer import BytesWriter

        from .codecache import torch_key

        writer = BytesWriter()
        writer.write_bytes(CacheCompiledArtifact.CACHE_HEADER)
        writer.write_bytes(torch_key())
        writer.write_str(key)
        writer.write_bytes(artifact_bytes)
        return writer.to_bytes()

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        with dynamo_timed("CompiledArtifact.save"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                if os.path.isdir(path):
                    raise AssertionError(f"expected path to not be a dir: {path}")

                from torch._inductor.codecache import write_atomic

                # ``_to_binary_bytes`` is the single source of the None / empty /
                # multiple aot_autograd_artifacts validation, so the binary branch
                # does not repeat those checks here.
                write_atomic(path, self._to_binary_bytes())
            else:
                if format != "unpacked":
                    raise AssertionError(f"expected format == 'unpacked', got {format}")
                # Same None / empty / multiple validation as the binary branch, shared via
                # _validate_and_unpack; the unpacked branch needs only artifact_bytes.
                artifact_bytes, _key = self._validate_and_unpack()
                if os.path.exists(path):
                    if not os.path.isdir(path):
                        raise AssertionError(f"expected path to be a dir: {path}")
                    shutil.rmtree(path, ignore_errors=True)

                from .codecache import FxGraphCache

                with temporary_cache_dir(path):
                    # This function unpacks the cache artifacts to disk
                    loaded_cache_info = torch.compiler.load_cache_artifacts(
                        artifact_bytes
                    )
                    if loaded_cache_info is None:
                        raise AssertionError(
                            "expected loaded_cache_info to not be None"
                        )
                    # Now write all the output_code artifacts to disk so that
                    # they can be inspected and modified
                    for key in loaded_cache_info.inductor_artifacts:
                        subdir = FxGraphCache._get_tmp_dir_for_key(key)
                        if not os.path.exists(subdir):
                            raise AssertionError(f"expected subdir to exist: {subdir}")
                        for path in sorted(os.listdir(subdir)):
                            with open(os.path.join(subdir, path), "rb") as f:
                                graph = pickle.load(f)
                            output_file = graph.write_to_disk()
                            log.info("Output code written to: %s", output_file)

    @staticmethod
    def _load_impl(
        cache_dir_ctx: AbstractContextManager[Any], key: str
    ) -> CompiledArtifact:
        with (
            cache_dir_ctx,
            config.patch(unsafe_skip_cache_dynamic_shape_guards=True),
        ):
            with torch._functorch.config.patch(strict_autograd_cache=True):
                from torch._functorch._aot_autograd.autograd_cache import (
                    AOTAutogradCache,
                )

                result = AOTAutogradCache._lookup(
                    key,
                    local=True,
                    remote=False,
                    args=[],
                    cache_info={},
                    aot_config=None,
                )

            if result is None:
                raise AssertionError(
                    "expected AOTAutogradCache lookup result to not be None"
                )
            (entry, _) = result

            from .compile_fx import _CompileFxKwargs

            fx_config = _CompileFxKwargs(
                cudagraphs=BoxedBool(False),
                boxed_forward_device_index=BoxedDeviceIndex(0),
            )

            context = torch._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
            with torch._guards.tracing(context):
                compiled_fn = entry.wrap_post_compile(
                    [], entry.sanitized_aot_config, fx_config
                )
        return CacheCompiledArtifact(lambda *args: compiled_fn(list(args)), None)

    @staticmethod
    def _prepare_load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> tuple[str, AbstractContextManager[Any]]:
        """
        Do format specific prep and loads, return a context manager and key
        """
        path = normalize_path_separator(path)
        with dynamo_timed("CompiledArtifact.load"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                if os.path.isdir(path):
                    raise AssertionError(f"expected path to not be a dir: {path}")
                with open(path, "rb") as file:
                    artifacts = file.read()
                from torch.utils._appending_byte_serializer import BytesReader

                from .codecache import torch_key

                reader = BytesReader(artifacts)
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")

                torch.compiler.load_cache_artifacts(artifact_bytes)
                return key, nullcontext()
            else:
                if format != "unpacked":
                    raise AssertionError(f"expected format == 'unpacked', got {format}")
                if not os.path.isdir(path):
                    raise AssertionError(f"expected path to be a dir: {path}")
                autograd_cache_dir = os.path.join(path, "aotautograd")
                if not os.path.isdir(autograd_cache_dir):
                    raise AssertionError(
                        f"expected autograd_cache_dir to be a dir: {autograd_cache_dir}"
                    )
                files = list(os.listdir(autograd_cache_dir))
                if len(files) != 1:
                    raise AssertionError(f"expected exactly 1 file, got {len(files)}")
                key = files[0]
                cache_dir_ctx = temporary_cache_dir(path)
                return key, cache_dir_ctx

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        key, cache_dir_ctx = CacheCompiledArtifact._prepare_load(
            path=path, format=format
        )
        return CacheCompiledArtifact._load_impl(cache_dir_ctx, key)


class AOTCompiledArtifact(CompiledArtifact):
    """
    Similar to CompiledArtifact, but the object is a single, bundled precompiled function.
    This object is always a serializable callable function.

    This object is essentially a wrapper for BundledAOTAutogradSerializableCallable, which
    is used by torch._dynamo.aot_compile for AOT Precompilation.
    """

    AOT_HEADER = bytes("AOTCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
    ):
        self.inner_fn = BundledAOTAutogradSerializableCallable(compiled_fn)
        self._artifacts = (
            None  # We don't need artifacts, the inner object handles everything
        )

    @staticmethod
    def from_bundled_callable(
        bundled_fn: BundledAOTAutogradSerializableCallable,
    ) -> AOTCompiledArtifact:
        return AOTCompiledArtifact(bundled_fn.compiled_fn)

    def __call__(self, *args: Any) -> Any:
        return self.inner_fn(*args)

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        result_bytes = self.serialize()
        from torch.utils._appending_byte_serializer import BytesWriter

        from .codecache import torch_key

        writer = BytesWriter()
        writer.write_bytes(AOTCompiledArtifact.AOT_HEADER)
        writer.write_bytes(torch_key())
        writer.write_bytes(result_bytes)

        from torch._inductor.codecache import write_atomic

        # Save a sentinel file to indicate that this is AOT
        write_atomic(path, writer.to_bytes())

    def serialize(self) -> bytes:
        return BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
            self.inner_fn
        )

    @staticmethod
    def deserialize(result_bytes: bytes) -> AOTCompiledArtifact:
        deserialized = (
            BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                result_bytes
            )
        )
        if not isinstance(deserialized, BundledAOTAutogradSerializableCallable):
            raise AssertionError(
                f"expected BundledAOTAutogradSerializableCallable, got {type(deserialized)}"
            )
        return AOTCompiledArtifact.from_bundled_callable(deserialized)

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header != AOTCompiledArtifact.AOT_HEADER:
                raise AssertionError("expected AOTCompiledArtifact header")
            if reader.read_bytes() != torch_key():
                raise AssertionError("torch_key mismatch in serialized artifact")
            artifact = reader.read_bytes()
            if not reader.is_finished():
                raise AssertionError("expected reader to be finished")
            return AOTCompiledArtifact.deserialize(artifact)


def _resolve_ignore_shape_env(dynamic_shapes: DynamicShapesType):
    # tells compile_fx to ignore the shape_envs on the ambient context
    # and the graph_module.
    return dynamic_shapes == "from_example_inputs"


def _resolve_fake_mode(
    gm: GraphModule,
    dynamic_shapes: DynamicShapesType,
    fake_mode: FakeTensorMode | None = None,
) -> FakeTensorMode:
    if dynamic_shapes == "from_example_inputs":
        if fake_mode is not None:
            if fake_mode.shape_env is None:
                raise ValueError(
                    "standalone_compile requires `fake_mode` to have a ShapeEnv "
                    'when `dynamic_shapes="from_example_inputs"`.'
                )
            return fake_mode
        return FakeTensorMode(shape_env=ShapeEnv())
    elif fake_mode is not None:
        raise ValueError(
            "standalone_compile only supports passing `fake_mode` when "
            '`dynamic_shapes="from_example_inputs"`.'
        )
    elif dynamic_shapes == "from_tracing_context":
        # Reuse fake_mode from the TracingContext.
        # NB: The TracingContext only exists if we're currently in a torch.compile backend.
        context = torch._guards.TracingContext.get()
        if context.fake_mode is None:
            raise AssertionError("expected TracingContext.fake_mode to not be None")
        return context.fake_mode
    elif dynamic_shapes == "from_graph":
        # Strategy: find a FakeTensor in the graph output, grab its FakeTensorMode.
        # The graph passed to standalone_compile must be an Inductor-approved graph,
        # which means that there is at least one Tensor output and the output node
        # contains a flat list of Tensors.
        last_node = next(iter(reversed(gm.graph.nodes)))
        if last_node.op != "output":
            raise AssertionError(
                f"expected last node op == 'output', got {last_node.op}"
            )
        if len(last_node.args) != 1:
            raise AssertionError(
                f"expected last node to have 1 arg, got {len(last_node.args)}"
            )

        # If gm came from Dynamo, then last_node.args[0] is always a list,
        # even in single-Tensor returns.
        #
        # It's possible to get into a situation where last_node.args[0]
        # is a Node (and not a list!). This happens if you call split_module
        # on the graph. We allow for this case since it is common.
        nodes = (
            [last_node.args[0]]
            if isinstance(last_node.args[0], torch.fx.Node)
            else last_node.args[0]
        )
        # Find a FakeTensor to recover its FakeTensorMode. This resolver was written when
        # standalone_compile was only ever fed Dynamo/AOT graphs, which return a flat list
        # of real Tensor outputs and stash the fake under "example_value". precompile now
        # feeds make_fx graphs too, and those break both assumptions: make_fx stashes the
        # fake under "val" (not "example_value"), and the traced fn's own return value is
        # kept as an output leaf -- so when fn calls .backward() (which returns None) that
        # None sits in the output list as a plain non-Node entry. So skip non-Node output
        # entries (the old code called .meta on the None and crashed), check both metadata
        # keys, and fall back to scanning every node since a make_fx fake may live only on
        # an interior node. maybe_get_fake_mode also unwraps a traceable wrapper subclass.
        output_nodes = [n for n in nodes if isinstance(n, torch.fx.Node)]
        for node in itertools.chain(output_nodes, gm.graph.nodes):
            for key in ("example_value", "val"):
                maybe_fake_mode = maybe_get_fake_mode(node.meta.get(key))
                if maybe_fake_mode is not None:
                    return maybe_fake_mode

        return FakeTensorMode(shape_env=ShapeEnv())
    else:
        raise ValueError(
            f"standalone_compile got unsupported `dynamic_shapes` value: dynamic_shapes={dynamic_shapes}."
        )


@contextlib.contextmanager
def _standalone_context(
    gm: GraphModule,
    dynamic_shapes: DynamicShapesType,
    aot: bool,
    fake_mode: FakeTensorMode | None = None,
):
    from torch.compiler._cache import CacheArtifactManager

    resolved_fake_mode = _resolve_fake_mode(gm, dynamic_shapes, fake_mode)
    tracing_context = torch._guards.TracingContext(resolved_fake_mode)
    with (
        torch._guards.tracing(tracing_context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
        torch._functorch.config.patch(
            {
                "bundled_autograd_cache": aot,
                # Standalone artifacts are saved immediately after compile_fx
                # returns. Training graphs normally lower the backward lazily on
                # first backward(), so force it while the artifact recorder is
                # still active.
                "force_non_lazy_backward_lowering": True,
            }
        ),
    ):
        yield


def standalone_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: DynamicShapesType,
    options: Any,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    donate_graph_module: bool = False,
    fake_mode: FakeTensorMode | None = None,
) -> CompiledArtifact:
    """
    Implementation of torch.inductor.standalone_compile
    """
    from .compile_fx import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(gm, dynamic_shapes, aot, fake_mode):
        # compile_fx takes ownership of gm and may mutate it on cache miss.
        # Deepcopy first so the rewrites below land on the owned copy rather
        # than the caller's gm. The gm may carry a non-pickleable torchbind
        # ProcessGroup (or, after a previous unbox, a Python
        # ``dist.ProcessGroup``); smuggle it through deepcopy as a shared
        # reference instead of crashing.
        if not donate_graph_module:
            with _share_torchbind_and_process_group_on_deepcopy():
                gm = copy.deepcopy(gm)
        # ``make_fx`` traces ``dist.*`` collectives as opaque ``c10d.{op}_``
        # calls. Inductor's collective machinery only recognizes the
        # ``_c10d_functional.{op}`` + ``wait_tensor`` form, so rewrite here
        # before compile_fx runs. Also unbox any torchbind ProcessGroup
        # attrs into Python ``dist.ProcessGroup`` so the runtime collective
        # op accepts them (raw torchbind is rejected).
        gm = _functionalize_inplace_collectives(gm)
        gm = _unbox_process_group_torchbinds(gm)
        compiled_fn = compile_fx(
            gm, example_inputs, ignore_shape_env=ignore_shape_env, **options
        )
        if not callable(compiled_fn):
            raise AssertionError("expected compiled_fn to be callable")
        if aot:
            if not hasattr(compiled_fn, "serialize"):
                raise RuntimeError(
                    "Compiled function should have serialize method when aot=True"
                )
            return AOTCompiledArtifact(compiled_fn)
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            log.warning(
                "standalone_compile artifact generation failed, cannot save. "
                "Run with TORCH_LOGS=+torch._inductor.codecache to identify the problem"
            )

    return CacheCompiledArtifact(compiled_fn, artifacts)


class NoRunnableInductorModuleError(RuntimeError):
    """Raised by ``compile_to_python`` when the graph yields no runnable Inductor
    output module -- it has no compute to lower (returns inputs/constants unchanged),
    so the compiled artifact carries no output-module source. Callers (e.g.
    torch.compiler.precompile) convert this to a clear user-facing error suggesting an
    alternative.
    """


def _runnable_source(compiled_graph: OutputCode) -> str:
    """Return the Inductor output-module source for a compiled inner graph.

    ``compile_fx_inner`` returns a ``CompiledFxGraph`` that carries the wrapper-module
    source as ``source_code``. A graph with no compute (returns inputs/constants
    unchanged) short-circuits to a boxed passthrough with no such source, which we
    surface as ``NoRunnableInductorModuleError``.
    """
    source = getattr(compiled_graph, "source_code", None)
    if not source:
        raise NoRunnableInductorModuleError(
            "the compiled graph produced no runnable Inductor output module: it has no "
            "compute to lower (returns inputs/constants unchanged)."
        )
    return source


def _placeholder_fake_inputs(gm: GraphModule) -> list[Any]:
    """Return the fake ``val`` metadata of ``gm``'s placeholders -- the compile-time input
    contract for a post-AOTAutograd graph. These fake tensors already carry the
    AOTAutograd-decided static/symbolic shapes under one consistent ``FakeTensorMode``, so
    lowering against them (rather than re-fakifying real ``example_inputs``) preserves
    symbolic dims. A placeholder without ``val`` means ``gm`` was not traced under a
    ``FakeTensorMode``, violating the post-AOTAutograd precondition."""
    fake_inputs = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if "val" not in node.meta:
            raise RuntimeError(
                "compile_to_python placeholder has no fake ``val`` metadata; expected a "
                "post-AOTAutograd graph traced under a FakeTensorMode."
            )
        fake_inputs.append(node.meta["val"])
    return fake_inputs


def _acceleration_cache_bytes(
    artifacts: tuple[bytes, CacheInfo] | None,
) -> bytes | None:
    """Return the opaque cache-artifacts bundle that accelerates a later ``load_from_python``,
    or ``None`` when nothing cacheable was produced (no compute) or caches are disabled.

    This is a PURE ACCELERATOR, not a serialized callable. It is the raw
    ``save_cache_artifacts()`` bundle (FxGraph entry + any triton/autotune cache);
    ``load_from_python`` feeds it to ``load_cache_artifacts`` so that exec'ing the emitted
    ``python_code`` loads the precompiled kernels instead of JIT-compiling them. The
    ``python_code`` runs correctly without it, so it carries no key/header framing -- it is
    handed straight back to ``load_cache_artifacts``.
    """
    if artifacts is None:
        log.debug("no cache artifacts captured; no acceleration cache")
        return None
    artifact_bytes, _ = artifacts
    return artifact_bytes


def load_from_python(
    python_code: str, cache: bytes | None = None
) -> Callable[..., Any]:
    """Load the module emitted by ``compile_to_python`` into a runnable ``call``.

    ``python_code`` is self-contained: exec'ing it defines ``call`` and JIT-compiles the
    inlined kernels on first use, so it runs with ``cache=None``. When ``cache`` (the
    accelerator bundle from ``compile_to_python``) is provided, it is loaded FIRST so the
    kernels load their precompiled binaries instead of recompiling -- a pure speedup, never
    a correctness requirement. Mirrors ``compile_to_python``: (python_code, cache) in,
    runnable ``call`` out.
    """
    if cache is not None:
        torch.compiler.load_cache_artifacts(cache)
    namespace: dict[str, Any] = {"__name__": "__compile_to_python__"}
    exec(compile(python_code, "<compile_to_python>", "exec"), namespace)
    call = namespace.get("call")
    if not callable(call):
        raise RuntimeError(
            "compile_to_python module did not define a callable ``call``."
        )
    return call


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    options: dict[str, Any] | None = None,
) -> tuple[str, bytes | None]:
    """Compile ``gm`` and return ``(inner_python, cache)`` -- the INNER half of the
    backend contract behind ``torch.compiler.precompile``.

    This is an INTERNAL layered-contract entry point, not an end-user API. End users
    should call ``torch.compiler.precompile``; this function only emits the inductor
    piece of the artifact and assumes its caller (the AOT layer) wraps it. It lives in
    ``torch._inductor`` (a private, leading-underscore module), so it is exposed for
