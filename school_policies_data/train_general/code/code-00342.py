            try:
                self._submit_inner(job_id, data)
                return
            except BrokenProcessPool:
                # If any subprocess in the pool crashes, we get a BrokenProcessPool
                # exception and the whole pool becomes unusable. Handle crashes by
                # recreating the pool and resubmitting. Log it -- this was
                # previously silent, hiding repeated worker crashes that can stall
                # a job in this retry loop.
                log.warning(
                    "Compile worker pool broke while submitting job %s; "
                    "recreating the pool and retrying.",
                    job_id,
                    exc_info=True,
                )
                self.pool = None

    def _submit_inner(self, job_id: int, data: bytes) -> None:
        def callback(fut: Future[Any]) -> None:
            with self._inflight_lock:
                self._inflight.pop(job_id, None)
            if not self.running:
                return
            try:
                result = fut.result()
            except Exception as e:
                log.exception("Error in subprocess")
                result = self.pickler.dumps(e)
            if not isinstance(result, bytes):
                raise AssertionError(f"Expected bytes result, got {type(result)}")
            with self.write_lock:
                if self.running:
                    _send_msg(self.write_pipe, MsgHeader.JOB, job_id, result)
            return

        self._start_pool()
        if self.pool is None:
            raise AssertionError("pool must be initialized before submitting jobs")

        future = self.pool.submit(
            functools.partial(SubprocMain.do_job, self.pickler, job_id, data)
        )
        future.add_done_callback(callback)

    def _start_pool(self) -> None:
        if self.pool is not None:
            return

        # Recycle heartbeat slots before the new worker generation forks.
        watchdog.reset()

        # Only fork workers inherit the sidecar<->parent pipe fds and must close
        # them (see _async_compile_initializer). Under spawn the workers do not
        # inherit them (close_fds=True + O_CLOEXEC), and these integers would
        # refer to unrelated fds the fresh interpreter reused -- closing them
        # would be silent corruption, not a no-op.
        close_fds = (
            (self.read_pipe.fileno(), self.write_pipe.fileno())
            if self.kind == SubprocKind.FORK
            else ()
        )
        self.pool = TrackedProcessPoolExecutor(
            self.nprocs,
            mp_context=multiprocessing.get_context(self.kind.value),
            initializer=functools.partial(
                _async_compile_initializer, os.getpid(), close_fds
            ),
        )
        self.pool_finalizer = multiprocessing.util.Finalize(
            None, self.pool.shutdown, exitpriority=sys.maxsize
        )
        _warm_process_pool(self.pool, self.nprocs)

    def _start_watchdog(self) -> None:
        interval = config.compile_worker_watchdog_interval_seconds
        if interval <= 0:
            return
        threading.Thread(
            target=self._watchdog_loop,
            args=(interval,),
            name="InductorSubprocWatchdog",
            daemon=True,
        ).start()

    def _watchdog_loop(self, interval: float) -> None:
        # Every `interval` seconds, report any job still running past `interval`
        # so a stuck/slow worker leaves a breadcrumb (the parent turns these into
        # tlparse artifacts). Re-reports each tick with a growing elapsed time.
        while not self._watchdog_stop.wait(interval):
            now = time.monotonic()
            now_ns = time.monotonic_ns()
            heartbeats = watchdog.read_heartbeats()
            with self._inflight_lock:
                slow = [
                    (job_id, elapsed)
                    for job_id, start in self._inflight.items()
                    if (elapsed := now - start) >= interval
                ]
            for job_id, elapsed in slow:
                self._report_status(job_id, elapsed, heartbeats.get(job_id), now_ns)

    def _report_status(
        self,
        job_id: int,
        elapsed: float,
        heartbeat: tuple[int, int, int] | None,
        now_ns: int,
    ) -> None:
        status: dict[str, Any] = {"job_id": job_id, "elapsed_s": round(elapsed, 1)}
        if heartbeat is not None:
            phase, phase_start_ns, worker_pid = heartbeat
            status["phase"] = watchdog.Phase(phase).name.lower()
            status["phase_elapsed_s"] = round((now_ns - phase_start_ns) / 1e9, 1)
            status["worker_pid"] = worker_pid
        elif watchdog.enabled():
            # Heartbeats are active but this job isn't in any worker's slot. Almost
            # always that means it's still queued in the pool -- a running (fork)
            # worker stamps its slot before doing anything. It can also briefly
            # mean the opposite: a job that just finished (clear_current_job ran in
            # do_job's finally) but hasn't yet been popped from _inflight by the
            # sidecar callback. That window is negligibly short, so we don't
            # distinguish it.
            status["phase"] = "queued"
        # Otherwise phase tracking is unavailable (e.g. a spawn pool) and the
        # report carries duration only -- we can't tell queued from running.
        payload = self.pickler.dumps(status)
        try:
            with self.write_lock:
                if self.running:
                    _send_msg(self.write_pipe, MsgHeader.STATUS, job_id, payload)
        except (OSError, ValueError):
            # Parent gone / pipe closed; the watchdog is best-effort.
            pass

    @staticmethod
    def do_job(pickler: SubprocPickler, job_id: int, data: bytes) -> bytes:
        # do the pickle/unpickle in the sub-subproc
        watchdog.set_current_job(job_id)
        try:
            job = typing.cast(Callable[[], object], pickler.loads(data))
            try:
                result = job()
            except Exception:
                result = _SubprocExceptionInfo(traceback.format_exc())
            return pickler.dumps(result)
        finally:
            watchdog.clear_current_job()


AnyPool = ProcessPoolExecutor | SubprocPool


def _get_process_pool_processes(pool: ProcessPoolExecutor) -> list[Any]:
    processes = getattr(pool, "_processes", None)
    if processes is not None:
        return list(processes.values())

    manager_thread = getattr(pool, "_executor_manager_thread", None)
    manager_processes = getattr(manager_thread, "processes", None)
    if manager_processes is not None:
        return list(manager_processes.values())

    return []


def _terminate_process_pool(pool: ProcessPoolExecutor) -> None:
    processes = _get_process_pool_processes(pool)
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            log.warning("Ignored error terminating compile worker", exc_info=True)

    # Give workers a bounded, shared grace period to exit on the SIGTERM above,
    # then SIGKILL any that are wedged. Without this the pool.shutdown(wait=True)
    # below can block on an unreaped worker until the parent's shutdown wait
    # gives up (a ~300s stall).
    deadline = time.time() + _WORKER_TERMINATE_GRACE_SECONDS
    for process in processes:
        try:
            remaining = deadline - time.time()
            if remaining > 0:
                process.join(remaining)
            if process.is_alive():
                process.kill()
        except (OSError, ValueError, AssertionError):
            log.warning("Ignored error killing compile worker", exc_info=True)

    try:
        pool.shutdown(wait=True, cancel_futures=True)
    except Exception:
        log.warning("Ignored error shutting down compile worker pool", exc_info=True)


def _warm_process_pool(pool: ProcessPoolExecutor, n: int) -> None:
    # We have to fork processes for compiler workers, but the more memory and other resources that are loaded, the
    # slower the os.fork time is, quite drastically. It also holds the GIL so we can't put it on another thread.

    # Examples:
    # A simple x + x + x script: 10ms seconds in the middle of the program, 2ms at startup
    # tf_efficientnet_b0 benchmark: 50ms! in the middle of the program , 3ms at startup

    # So we want to start the workers early when it is still cheap, and also to allow the workers to get
    # ready before we have work for them.

    # ProcessPoolExecutor also does not launch the workers until it finds a point when all the workers are idle.
    # But if we waited until then fork time will be long and we will be waiting for the processes to initialize.

    # We force them to start here with some YOLOing of the internal methods.

    if hasattr(pool, "_start_queue_management_thread"):
        pool._start_queue_management_thread()
    else:
        for _ in range(n):
            pool._adjust_process_count()
        if hasattr(pool, "_start_executor_manager_thread"):
            pool._start_executor_manager_thread()


class TestException(RuntimeError):
    pass


def raise_testexc() -> Never:
    raise TestException


def _test_signal_then_sleep(signal_path: str, seconds: float) -> None:
    # Test helper: announce that this worker has begun executing by creating
    # signal_path, then block. Lets a test wait until a job is actually running
    # in a worker before acting on the pool (e.g. shutting it down).
    Path(signal_path).touch()
    time.sleep(seconds)


def _ignore_sigterm_and_sleep_for_test() -> None:
    # Test helper: a compile worker that ignores SIGTERM and blocks, forcing pool
    # teardown to escalate to SIGKILL instead of stalling on a graceful shutdown.
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(3600)


def _report_phase_and_sleep_for_test(phase: int, seconds: float) -> None:
    # Test helper: report a heartbeat phase from a worker, then block, so the
    # sidecar watchdog observes and reports that phase.
    watchdog.report_phase(watchdog.Phase(phase))
    time.sleep(seconds)
<|endoftext|>import functools
import os
from functools import cached_property
from typing import Any
from typing_extensions import Unpack

from ..utils import is_rocm
from .triton_compat import ASTSource, CompiledKernel, knobs as triton_knobs
from .triton_helpers import get_constexprs


@functools.lru_cache(None)
def _tma_arg_helpers():
    """Cached (make_arg, TensorDescriptor) for host-side TMA arg expansion.
    make_arg(arg, metadata) -> [CUtensorMap, *shape, *strides]; the nvidia
    backend's make_tensordesc_arg ignores its third arg, so we pass None."""
    from triton.backends.nvidia.driver import make_tensordesc_arg
    from triton.tools.tensor_descriptor import TensorDescriptor

    def make_arg(arg, metadata):
        return make_tensordesc_arg(arg, metadata, None)

    return make_arg, TensorDescriptor


class StaticallyLaunchedTritonKernel:
    """
    Parses the metadata of a CompiledKernel from Triton into a structure that can
    launch the cuda kernel directly. Only works for triton kernels compiled to cubin.

    Doing this avoids C++ codegen and compilation during compile, since we can use a
    statically compiled library to launch the kernel. To avoid mallocing for the arguments,
    we have a launcher for different numbers of arguments up to a max. StaticCudaLauncher
    only supports # of arguments up until 10 for now.

    Workflow:
    Compile time:
    1. Compile a kernel with triton and get a CompiledKernel
    2. Instantiate kernel = StaticallyLaunchedTritonKernel(triton_kernel)
    3. Write to a cubin file: kernel.write_cubin_to_file(filepath)
    4. Call kernel.load_kernel() (CUDA should be initialized by this point) to load the cubin
    Runtime:
    5. Call kernel.run(grid, stream, args) to launch the kernel

    Note that after step 3, StaticallyLaunchedTritonKernel is fully pickleable/serializable.
    This allows it to be cached by FXGraphCache/TritonBundler, as well as sent from the worker
    to the parent process in inductor.

    There are two main versions of triton that we wish to support: 3.3 and 3.2. Triton makes considerable changes
    to how it handles constants in 3.3, so there's some special logic necessary to handle both versions.
    """

    @cached_property
    def C_impl(self):
        raise NotImplementedError

    def __init__(self, kernel: CompiledKernel) -> None:
        # pyrefly: ignore [missing-attribute]
        self.name = kernel.src.fn.__name__
        # pyrefly: ignore [missing-attribute]
        self.cubin_path = kernel._cubin_path

        # Used by torch.compile to filter constants in older triton versions
        # pyrefly: ignore [missing-attribute]
        self.arg_names = kernel.src.fn.arg_names

        # Const exprs that are declared by the triton kernel directly
        # Used to generate the kernel launcher's def args
        # pyrefly: ignore [missing-attribute]
        self.declared_constexprs = get_constexprs(kernel.src.fn)

        # pyrefly: ignore [missing-attribute]
        self.hash = kernel.hash

        if triton_knobs is None:
            # pyrefly: ignore [missing-attribute]
            launch_enter = kernel.__class__.launch_enter_hook
            # pyrefly: ignore [missing-attribute]
            launch_exit = kernel.__class__.launch_exit_hook
        else:
            launch_enter = triton_knobs.runtime.launch_enter_hook
            launch_exit = triton_knobs.runtime.launch_exit_hook

        def hook_is_empty(hook: Any) -> bool:
            if hook is None:
                return True
            if (
                triton_knobs
                and (HookChain := getattr(triton_knobs, "HookChain", None)) is not None
                and isinstance(hook, HookChain)
            ):
                # Support hooks after https://github.com/triton-lang/triton/pull/7866
                return len(hook.calls) == 0
            return False

        if not hook_is_empty(launch_enter) or not hook_is_empty(launch_exit):
            raise NotImplementedError(
                "We don't support launch enter or launch exit hooks"
            )
        # pyrefly: ignore [missing-attribute]
        self.num_warps = kernel.metadata.num_warps
        self.shared = (
            # pyrefly: ignore [missing-attribute]
            kernel.shared if hasattr(kernel, "shared") else kernel.metadata.shared
        )

        def needs_scratch_arg(scratch_name: str, param_name: str) -> bool:
            # pyrefly: ignore [missing-attribute]
            if hasattr(kernel.metadata, param_name):
                # pyrefly: ignore [missing-attribute]
                if getattr(kernel.metadata, param_name) > 0:
                    raise NotImplementedError(
                        f"{scratch_name} scratch not yet supported"
                    )
                return True
            return False

        # Newer triton versions pass an extra global scratch parameter to the compiled cuda kernel.
        # Inductor never uses this field or enables it, but we still have to pass
        # an extra None into the set of params if its enabled
        self.has_global_scratch = needs_scratch_arg("Global", "global_scratch_size")
        # same situation for profile scratch - triton-lang/triton#7258
        self.has_profile_scratch = needs_scratch_arg("Profile", "profile_scratch_size")

        # pyrefly: ignore [missing-attribute]
        self.tensordesc_meta = getattr(kernel.metadata, "tensordesc_meta", None)
        self._has_tensordesc = False
        # pyrefly: ignore [missing-attribute]
        self.arg_tys = self.arg_ty_from_signature(kernel.src)
        self.function: int | None = None  # Loaded by load_kernel(on the parent process)
        self.module: int | None = None  # Owns the HIP/CUDA module loaded for function
        # compile-on-one-rank: a device-agnostic kernel (no baked device index) can be
        # launched on more than one device within a single process. A loaded module/
        # function is bound to a single device, so when device_agnostic is set we keep
        # them per device and resolve the current device at launch time. The cubin
        # (device-agnostic) is retained so it can be loaded onto additional devices.
        # These per-device dicts assume one thread per device for a given launcher (the
        # single-process multi-device path is sequential in practice); concurrent
        # first-launch on two new devices would need external locking.
        self.device_agnostic: bool = False
        self.functions: dict[int, int] = {}
        self.modules: dict[int, int] = {}
        num_ctas = 1
        if hasattr(kernel, "num_ctas"):
            num_ctas = kernel.num_ctas
        elif hasattr(kernel, "metadata"):
            num_ctas = kernel.metadata.num_ctas

        if num_ctas != 1:
            raise NotImplementedError(
                "Static cuda launcher only supports num_ctas == 1"
            )

    def reload_cubin_from_raw(self, filepath: str) -> str:
        """
        If the cubin file triton generated gets deleted under us, we can
        reload it from the raw cubin file.
        """
        if self.cubin_path is None:
            if self.cubin_raw is None:
                raise AssertionError("cubin_raw must be set when cubin_path is None")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(self.cubin_raw)
                self.cubin_path = filepath  # pyre-ignore
        return self.cubin_path

    def _agnostic_cubin_path(self) -> str:
        # The cubin bytes are device-agnostic, so the same file loads onto any device.
        # Keep it available (the single-device path frees it after one load) and rewrite
        # from the retained raw bytes if the file was removed under us.
        if self.cubin_path is not None and os.path.exists(self.cubin_path):
            return self.cubin_path
        if self.cubin_raw is None or self.cubin_path is None:
            raise AssertionError(
                "device-agnostic kernel cannot reload its cubin for a new device"
            )
        os.makedirs(os.path.dirname(self.cubin_path), exist_ok=True)
        with open(self.cubin_path, "wb") as f:
            f.write(self.cubin_raw)
        return self.cubin_path

    def load_kernel(self, device: int) -> None:
        if self.device_agnostic:
            if device in self.functions:
                return
            (module, function, self.n_regs, self.n_spills) = self.C_impl._load_kernel(
                self._agnostic_cubin_path(), self.name, self.shared, device
            )
            self.modules[device] = module
            self.functions[device] = function
            return

        if self.function is not None:
            return

        if not hasattr(self, "cubin_path"):
            raise AssertionError("cubin_path attribute not set before load_kernel")
        if self.cubin_path is None:
            raise AssertionError("cubin_path must not be None before load_kernel")
        (self.module, self.function, self.n_regs, self.n_spills) = (
            self.C_impl._load_kernel(self.cubin_path, self.name, self.shared, device)
        )
        # Don't need the cubin path anymore now that we've loaded
        self.cubin_path = None
        self.cubin_raw = None

    def _current_device(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        # Clear Python-visible handles first so repeated cleanup is harmless even if
        # the driver reports an error while unloading.
        modules = list(self.modules.values())
        if self.module is not None:
            modules.append(self.module)
        self.module = None
        self.function = None
        self.modules = {}
        self.functions = {}
        for mod in modules:
            self.C_impl._unload_kernel(mod)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    @functools.lru_cache
    def type_mappings() -> dict[str, str]:
        return {
            "i1": "i",
            "i8": "b",
            "i16": "h",
            "i32": "i",
            "i64": "l",
            "u1": "I",
            "u8": "B",
            "u16": "H",
            "u32": "I",
            "u64": "K",
            "fp16": "f",
            "bf16": "f",
            "fp32": "f",
            "f32": "f",
            "fp64": "d",
            # TODO handle nvTmaDesc/CUtensormap
        }

    def extract_type(self, ty: str) -> str:
        """
        Takes a triton type from CompiledKernel.signature and
        converts it into a single char encoding. _StaticCudaLauncher
        will switch on this char to figure out what type the underlying
        value should be passed to the triton kernel as.
        """
        if ty[0] == "*":
            return "O"
        elif ty == "nvTmaDesc" or ty.startswith("tensordesc<"):
            raise NotImplementedError("TMA descriptor kernels are not yet supported")
        return StaticallyLaunchedTritonKernel.type_mappings()[ty]

    def _expand_tensordesc_type(self, ty: str) -> str:
        """Expand a tensordesc<dtype[shape]> signature entry into the per-arg
        type chars triton lowers it to (mirrors nvidia driver expand_signature):
        nvTmaDesc path -> CUtensorMap + shape(i32) + strides(i64); host-decomposed
        path -> base ptr + shape/strides(i64) + 2 flags + shape(i32) + strides(i64).

        This mirrors the string parse in triton's own expand_signature: the only
        thing we need is the descriptor's rank, which lives in the shape list of
        the serialized "tensordesc<dtype[d0, d1, ...]>" form. We parse that string
        here rather than importing expand_signature because that helper's argument
        signature differs across triton versions (a 2-arg (signature, meta) form
        in the fb-triton nvidia driver vs a 3-arg (signature, meta,
        descriptor_type) form as backends converge on a shared helper), so
        importing it would break across triton bumps; the serialized string
        format is stable.
        TODO: switch to triton's expand_signature once the versions converge on a
        single argument signature.
        """
        import re

        # Same regex triton's expand_signature uses. We only need the rank, so
        # the captured dtype group is unused (kept to match triton's pattern).
        match = re.match(r"tensordesc<([^\[>]*)\[([^\]]*)\]", ty)
        if match is None:
            raise NotImplementedError(f"Could not parse tensordesc type: {ty}")
        ndim = match.group(2).count(",") + 1
        meta = self.tensordesc_meta
        idx = self._tensordesc_idx
        self._tensordesc_idx += 1
        per_desc_meta = meta[idx] if meta else None
        if per_desc_meta is None:
            chars = ["O"] + ["l"] * (2 * ndim) + ["i", "i"]
        else:
            chars = ["M"]
        chars += ["i"] * ndim + ["l"] * ndim
        return "".join(chars)

    def arg_ty_from_signature(self, src: ASTSource) -> str:
        def index_key(i: Any) -> int:
            if isinstance(i, str):
                # pyrefly: ignore [missing-attribute]
                return src.fn.arg_names.index(i)
            elif isinstance(i, tuple):
                # In triton 3.3, src.fn.constants has tuples as a key
                return i[0]
            else:
                return i

        # pyrefly: ignore [missing-attribute]
        signature = {index_key(key): value for key, value in src.signature.items()}
        # Triton uses these as the main way to filter out constants passed to their cubin
        constants = [index_key(key) for key in getattr(src, "constants", dict())]
        # This value is always a superset of kernel.fn.constexprs: kernel.fn.constexprs are
        # constants declared by the triton kernel directly, whereas this list can have
        # constants that are unused by the triton kernel that triton figured out during
        # compilation.
        self.full_constexprs = constants
        # Despite requiring them to be passed in, the triton CUDA launcher
        # completely ignores the constexprs passed into it when generating code.
        # So we can ignore them here too
        params = []
        self._tensordesc_idx = 0

        for i in sorted(signature.keys()):
            ty = signature[i]
            # In newer triton versions, constants are passed in to signature with type `constexpr`
            # In older triton versions, there can be constants in src.constants that are not `constexpr` in signature
            # so we check both here
            if ty == "constexpr" or i in constants:
                pass
            elif isinstance(ty, str) and ty.startswith("tensordesc<"):
                self._has_tensordesc = True
                params.append(self._expand_tensordesc_type(ty))
            else:
                # pyrefly: ignore [bad-argument-type]
                params.append(self.extract_type(ty))
        return "".join(params)

    def __getstate__(self) -> dict[str, Any]:
        # Remove objects that are no longer valid for pickling
        state = self.__dict__.copy()
        state["function"] = None
        state["module"] = None
        state["functions"] = {}
        state["modules"] = {}
        # Cubin paths aren't consistent across processes, so we clear
        # and reload them.
        state["cubin_path"] = None
        return state

    def _expand_tma_args(self, args: tuple[object, ...]) -> tuple[object, ...]:
        """Expand host-side TMA TensorDescriptor args into the flat kernel params
        (CUtensorMap + shape + strides) so they match the expanded type string."""
        make_tensordesc_arg, TensorDescriptor = _tma_arg_helpers()

        meta = self.tensordesc_meta
        out: list[object] = []
        td_idx = 0
        for arg in args:
            if isinstance(arg, TensorDescriptor):
                per_desc_meta = meta[td_idx] if meta else None
                td_idx += 1
                out.extend(make_tensordesc_arg(arg, per_desc_meta))
            else:
                out.append(arg)
        return tuple(out)

    def run(
        self,
        grid_x: int,
        grid_y: int,
        grid_z: int,
        stream: int,
        *args: Unpack[tuple[object, ...]],
    ) -> None:
        """Actually run the kernel at runtime. This function is the hot codepath."""

        if self.device_agnostic:
            # The loaded function is device-bound; pick (loading on demand) the one for
            # the current device so a shared kernel launches on whatever device the
            # caller is using.
            device = self._current_device()
            function = self.functions.get(device)
            if function is None:
                self.load_kernel(device)
                function = self.functions[device]
        else:
            function = self.function

        # Assert load_kernel() has been called and args match
        if function is None:
            raise AssertionError("load_kernel() must be called before run()")

        # TODO: actually, if the args *don't* match, we probably should
        # throw an exception. But if inductor is the only one calling this
        # thing, it should always match.
        # Get rid of constants before passing to cubin launcher

        if self._has_tensordesc:
            args = self._expand_tma_args(args)

        arg_tys = self.arg_tys

        if is_rocm():
            # ROCm/HIP kernel ABI: The Triton HIP backend ALWAYS includes both
            # global_scratch and profile_scratch parameters in the kernel signature,
            # even when the kernel doesn't use them (i.e., when has_*_scratch is False).
            #
            # This differs fundamentally from CUDA, where these parameters are only
            # present in the signature if the corresponding has_*_scratch flag is True.
            #
            # The flags indicate whether memory will be allocated/used:
            # - has_global_scratch: Whether global scratch workspace is needed
            # - has_profile_scratch: Whether profiling instrumentation is enabled
            #
            # However, regardless of flag values, we MUST always pass both parameters
            # to match the HIP kernel ABI. Passing None is safe:
            #
            # - If scratch is not needed (has_*_scratch=False or scratch_size=0):
            #   The None becomes nullptr, which the kernel never dereferences
            #
            # - If scratch is needed (has_*_scratch=True and scratch_size>0):
            #   The None becomes nullptr initially, but the HIP runtime intercepts
            #   the kernel launch, allocates the required scratch memory based on
            #   kernel metadata, and replaces the nullptr with a valid pointer before
            #   the kernel actually executes
            #
            # Not passing both parameters causes segmentation faults because the kernel
            # expects them at specific positions in the argument array.
            arg_tys = arg_tys + "OO"
            args = (*args, None, None)

        else:
            for has_scratch in [self.has_global_scratch, self.has_profile_scratch]:
                if has_scratch:
                    arg_tys = arg_tys + "O"
                    args = (*args, None)
        # pyrefly: ignore [bad-argument-type]
        if len(args) != len(arg_tys):
            raise AssertionError(f"Expected {len(arg_tys)} args, got {len(args)}")

        # TODO: can handle grid functions here or in C++, so
        # that we don't need the grid handler above.
        self.C_impl._launch_kernel(
            function,
            grid_x,
            grid_y,
            grid_z,
            self.num_warps,
            self.shared,
            arg_tys,
            args,
            stream,
        )


class StaticallyLaunchedCudaKernel(StaticallyLaunchedTritonKernel):
    @cached_property
    def C_impl(self):
        from torch._C import _StaticCudaLauncher

        return _StaticCudaLauncher

    def __init__(self, kernel: CompiledKernel) -> None:
        # pyrefly: ignore [missing-attribute]
        if "hsaco" in kernel.asm:
            # pyrefly: ignore [missing-attribute]
            self.cubin_raw = kernel.asm["hsaco"]

        # pyrefly: ignore [missing-attribute]
        elif "cubin" in kernel.asm:
            # pyrefly: ignore [missing-attribute]
            self.cubin_raw = kernel.asm["cubin"]
        else:
            raise RuntimeError(
                "Expected either 'hsaco' (ROCm) or 'cubin' (CUDA) in kernel.asm"
            )
        super().__init__(kernel)

    def _current_device(self) -> int:
        import torch

        return torch.cuda.current_device()


class StaticallyLaunchedXpuKernel(StaticallyLaunchedTritonKernel):
    @cached_property
    def C_impl(self):
        from torch._C import _StaticXpuLauncher

        return _StaticXpuLauncher

    def __init__(self, kernel: CompiledKernel) -> None:
        # pyrefly: ignore [missing-attribute]
        self.cubin_raw = kernel.asm.get("zebin", None)
        super().__init__(kernel)

    def load_kernel(self, device: int) -> None:
        # The XPU static launcher returns a PyCapsule for the loaded SYCL kernel,
        # not a separate module/function pair like the CUDA/HIP launcher.
        if self.device_agnostic:
            if device in self.functions:
                return
            (function, self.n_regs, self.n_spills) = self.C_impl._load_kernel(
                self._agnostic_cubin_path(), self.name, self.shared, device
            )
            # XPU has no separate module handle (only the function capsule).
            self.functions[device] = function
            return

        if self.function is not None:
            return

        if not hasattr(self, "cubin_path"):
            raise AssertionError("expected cubin_path attribute to be set")
        if self.cubin_path is None:
            raise AssertionError("expected cubin_path to not be None")
        (self.function, self.n_regs, self.n_spills) = self.C_impl._load_kernel(
            self.cubin_path, self.name, self.shared, device
        )
        self.module = None
        self.cubin_path = None
        self.cubin_raw = None

    def _current_device(self) -> int:
        import torch

        return torch.xpu.current_device()

    def close(self) -> None:
        self.module = None
        # Drop the PyCapsule references so their destructors can release sycl::kernel.
        self.function = None
        self.functions = {}
        self.modules = {}


def statically_launched_kernel_by_device(
    kernel: CompiledKernel, device_type: str = "cuda"
) -> StaticallyLaunchedTritonKernel:
    if device_type in ("cuda", "hip"):
        return StaticallyLaunchedCudaKernel(kernel)
    elif device_type == "xpu":
        return StaticallyLaunchedXpuKernel(kernel)
    else:
        raise NotImplementedError(
            f"Device type {device_type} is not supported for static launcher"
        )
<|endoftext|>from __future__ import annotations

import inspect
from typing import Any

import torch


try:
    import triton
except ImportError:
    triton = None


if triton is not None:
    import triton.language as tl
    from triton import Config
    from triton.compiler import CompiledKernel
    from triton.runtime.autotuner import OutOfResources
    from triton.runtime.jit import JITFunction, KernelInterface

    try:
        from triton.runtime.autotuner import PTXASError
    except ImportError:

        class PTXASError(Exception):  # type: ignore[no-redef]
            pass

    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        ASTSource = None

    try:
        from triton.backends.compiler import GPUTarget
    except ImportError:

        def GPUTarget(
            backend: str,
            arch: int | str,
            warp_size: int,
        ) -> Any:
            if torch.version.hip:
                return [backend, arch, warp_size]
            return (backend, arch)

    # In the latest triton, math functions were shuffled around into different modules:
    # https://github.com/triton-lang/triton/pull/3172
    try:
        from triton.language.extra import libdevice

        libdevice = tl.extra.libdevice  # noqa: F811
        math = tl.math
    except ImportError:
        if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
            libdevice = tl.extra.cuda.libdevice
            math = tl.math
        elif hasattr(tl.extra, "intel") and hasattr(tl.extra.intel, "libdevice"):
            libdevice = tl.extra.intel.libdevice
            math = tl.math
        else:
            libdevice = tl.math
            math = tl

    try:
        from triton.language.standard import _log2
    except ImportError:

        def _log2(x: Any) -> Any:
            raise NotImplementedError

    def _triton_config_has(param_name: str) -> bool:
        if not hasattr(triton, "Config"):
            return False
        if not hasattr(triton.Config, "__init__"):
            return False
        return param_name in inspect.signature(triton.Config.__init__).parameters

    # Drop the legacy support of autoWS
    HAS_WARP_SPEC = False

    try:
        from triton import knobs
    except ImportError:
        knobs = None

    try:
        from triton.runtime.cache import triton_key  # type: ignore[attr-defined]
    except ImportError:
        from triton.compiler.compiler import (
            triton_key,  # type: ignore[attr-defined,no-redef]
        )

    try:
        from triton.runtime.errors import IntelGPUError
    except ImportError:

        class IntelGPUError(Exception):  # type: ignore[no-redef]
            pass

    builtins_use_semantic_kwarg = (
        "_semantic" in inspect.signature(triton.language.core.view).parameters
    )
    HAS_TRITON = True
else:

    def _raise_error(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("triton package is not installed")

    class OutOfResources(Exception):  # type: ignore[no-redef]
        pass

    class PTXASError(Exception):  # type: ignore[no-redef]
        pass

    class IntelGPUError(Exception):  # type: ignore[no-redef]
        pass

    Config = object
    CompiledKernel = object
    KernelInterface = object
    ASTSource = None
    GPUTarget = None
    _log2 = _raise_error
    libdevice = None
    math = None
    knobs = None
    builtins_use_semantic_kwarg = False

    class triton:  # type: ignore[no-redef]
        @staticmethod
        def jit(*args: Any, **kwargs: Any) -> Any:
            return _raise_error

    class tl:  # type: ignore[no-redef]
        @staticmethod
        def constexpr(val: Any) -> Any:
            return val

        tensor = Any
        dtype = Any

    class JITFunction:  # type: ignore[no-redef]
        pass

    HAS_WARP_SPEC = False
    triton_key = _raise_error
    HAS_TRITON = False


try:
    autograd_profiler = torch.autograd.profiler
except AttributeError:  # Compile workers only have a mock version of torch

    class autograd_profiler:  # type: ignore[no-redef]
        _is_profiler_enabled = False


__all__ = [
    "Config",
    "CompiledKernel",
    "OutOfResources",
    "KernelInterface",
    "PTXASError",
    "IntelGPUError",
    "ASTSource",
    "GPUTarget",
    "tl",
    "_log2",
    "libdevice",
    "math",
    "triton",
    "knobs",
    "triton_key",
]
<|endoftext|>import getpass
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

from torch._environment import is_fbcode


# Factoring out to file without torch dependencies


def cache_dir() -> str:
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        cache_dir = default_cache_dir()
    cache_dir = os.path.abspath(cache_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def default_cache_dir() -> str:
    try:
        username = getpass.getuser()
    except (KeyError, ModuleNotFoundError, OSError):
        getuid = getattr(os, "getuid", None)
        username = f"uid_{getuid()}" if callable(getuid) else "unknown_user"
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", username)
    return os.path.join(
        tempfile.gettempdir() if not is_fbcode() else "/var/tmp",
        "torchinductor_" + sanitized_username,
    )


def triton_cache_dir(device: int) -> str:
    if (directory := os.getenv("TRITON_CACHE_DIR")) is not None:
        return directory
    return os.path.join(
        cache_dir(),
        "triton",
        str(device),
    )


@contextmanager
def temporary_cache_dir(directory: str) -> Generator[None, None, None]:
    from torch._inductor.utils import clear_caches

    original = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = directory
    try:
        clear_caches()
        yield
    finally:
        clear_caches()
        if original is None:
            del os.environ["TORCHINDUCTOR_CACHE_DIR"]
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = original
<|endoftext|>from __future__ import annotations

import functools
import math
import operator
from typing import Any, TYPE_CHECKING

import sympy

import torch

# NOTE: other files rely on the imports below
from torch._dynamo import callback as compilation_callback  # noqa: F401
from torch._inductor.runtime.cache_dir_utils import (  # noqa: F401
    cache_dir,
    default_cache_dir,
    triton_cache_dir,
)


if TYPE_CHECKING:
    from collections.abc import Hashable

    from .triton_compat import Config


def conditional_product(*args: int) -> int:
    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(number: int, denom: int) -> int:
    return -(number // -denom)


def is_power_of_2(n: int) -> bool:
    """Returns whether n = 2 ** m for some integer m."""
    return n > 0 and n & n - 1 == 0


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    if isinstance(n, sympy.Integer):
        n = int(n)
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def last_power_of_2(n: int) -> int:
    """Return the largest power of 2 less than or equal to n"""
    next_pow2 = next_power_of_2(n)
    return next_pow2 // 2 if next_pow2 > n else next_pow2


def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor)
    )


def triton_config_to_hashable(cfg: Config) -> Hashable:
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
    # pyrefly: ignore [missing-attribute]
    items = sorted(cfg.kwargs.items())
    # pyrefly: ignore [missing-attribute]
    items.append(("num_warps", cfg.num_warps))
    # pyrefly: ignore [missing-attribute]
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


def validate_triton_config(cfg: Config) -> None:
    # [Note: Triton pre_hook in inductor]
    # pre-hook is a lambda function, which we don't attempt to serialize.
    # right now, if a pre-hook is attached to the config, it will not be saved;
    # and then it won't be used when the config is loaded from cache.
    # So we assert - if we do get a pre_hook, it might get ignored after caching.
    if getattr(cfg, "pre_hook", None) is not None:
        raise AssertionError("triton configs with pre_hooks not supported")


def assert_tensor_metadata(
    tensor: torch.Tensor | None,
    expected_size: tuple[Any, ...],
    expected_stride: tuple[Any, ...],
    expected_dtype: torch.dtype,
    op_name: str | None = None,
) -> None:
    if tensor is None:
        return
    if tensor.dtype != expected_dtype:
        op_msg = f"\nError in op: {op_name}" if op_name else ""
        raise RuntimeError(
            f"expected dtype {expected_dtype} but got {tensor.dtype}"
            + op_msg
            + "\nThis error most often comes from an incorrect fake (aka meta) "
            "kernel for a custom op."
            "\nUse torch.library.opcheck to test your custom op."
            "\nSee https://pytorch.org/docs/stable/library.html#torch.library.opcheck"
        )
    torch._C._dynamo.guards.assert_size_stride(
        tensor, expected_size, expected_stride, op_name
    )


def create_bandwidth_info_str(
    ms: float,
    num_gb: float,
    gb_per_s: float,
    prefix: str = "",
    suffix: str = "",
    color: bool = True,
) -> str:
    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    slow = ms > 0.012 and gb_per_s < 650
    return red_text(info_str) if color and slow else info_str


def get_max_y_grid() -> int:
    return 65535


try:
    import colorama

    HAS_COLORAMA = True
except ModuleNotFoundError:
    HAS_COLORAMA = False
    colorama = None  # type: ignore[assignment]


if HAS_COLORAMA:

    def _color_text(msg: str, color: str) -> str:
        # pyrefly: ignore [missing-attribute]
        return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET

else:

    def _color_text(msg: str, color: str) -> str:
        return msg


def green_text(msg: str) -> str:
    return _color_text(msg, "green")


def yellow_text(msg: str) -> str:
    return _color_text(msg, "yellow")


def red_text(msg: str) -> str:
    return _color_text(msg, "red")


def blue_text(msg: str) -> str:
    return _color_text(msg, "blue")


def get_first_attr(obj: Any, *attrs: str) -> Any:
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


dynamo_timed = torch._dynamo.utils.dynamo_timed  # type: ignore[has-type]


def triton_hash_to_path_key(key: str) -> str:
    # In early versions of Triton, the hash is directly used in the path name.
    # Later, the hash is converted to base64 before being used in the path name.
    # Later, the base64 conversion was replaced to the base32
    #
    # This code tries to import _base64 and falls back to _base32 if _base64 is unavailable.
    #
    # To handle this, try to import the to-base64-conversion function.
    # If it exists, use it; otherwise, try using _base32; if both are unavailable, use the hash directly.
    try:
        from triton.runtime.cache import _base64

        return _base64(key)
    except Exception:
        try:
            from triton.runtime.cache import _base32

            return _base32(key)
        except Exception:
            return key


def compile_mps_shader(source: str) -> Any:
    """
    Compiles shader source but raise more actionable error message when needed
    """
    try:
        return torch.mps.compile_shader(source)
    except SyntaxError as err:
        raise SyntaxError(f"failed to compile {source} with {err.msg}") from err


def compile_mps_shaders(
    kernels: list[tuple[str, str, list[str]]],
) -> dict[str, Any]:
    """Compile a batch of Metal kernels into one library.

    Args:
        kernels: list of (kernel_name, metal_source, headers) tuples.
            headers are bare names resolved as <c10/metal/{name}.h>.

    Returns:
        dict mapping each kernel_name to its compiled function handle.
    """
    from torch.utils._ordered_set import OrderedSet

    all_headers = sorted(OrderedSet(h for _, _, hs in kernels for h in hs))
    header_src = "\n".join(f"#include <c10/metal/{h}.h>" for h in all_headers)
    body_src = "\n".join(src for _, src, _ in kernels)
    lib = compile_mps_shader(header_src + "\n" + body_src)
    return {name: getattr(lib, name) for name, _, _ in kernels}


def torch_dtype_to_jax_runtime(dtype: torch.dtype) -> Any:
    """
    Map PyTorch dtype to actual JAX dtype object at runtime.

    This helper is used in generated Pallas kernels at runtime to convert
    PyTorch dtypes to JAX dtype objects (not string representations).

    Args:
        dtype: PyTorch dtype to convert

    Returns:
        JAX dtype object (e.g., jnp.float32 object itself)
    """
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    dtype_map = {
        torch.float32: jnp.float32,
        torch.float64: jnp.float64,
        torch.float16: jnp.float16,
        torch.bfloat16: jnp.bfloat16,
        torch.int32: jnp.int32,
        torch.int64: jnp.int64,
        torch.int16: jnp.int16,
        torch.int8: jnp.int8,
        torch.uint8: jnp.uint8,
        torch.bool: jnp.bool_,
        torch.complex64: jnp.complex64,
        torch.complex128: jnp.complex128,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype for JAX conversion: {dtype}")
    return dtype_map[dtype]


def torch_dtype_to_jax(dtype: torch.dtype) -> str:
    """
    Map PyTorch dtype to JAX dtype expression string.

    This helper is used at compile time in codegen to generate
    JAX dtype expressions for Pallas kernels.

    Args:
        dtype: PyTorch dtype to convert

    Returns:
        JAX dtype expression as string (e.g., "jnp.float32")
    """
    jax_dtype = torch_dtype_to_jax_runtime(dtype)
    dtype_name = jax_dtype.__name__
    if dtype_name == "bool":
        dtype_name = "bool_"
    return f"jnp.{dtype_name}"


def pallas_gpu_pad_inputs(inputs: list[Any], alignment: int = 128) -> list[Any]:
    """Flatten and pad each input JAX array to a multiple of alignment."""
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    padded = []
    for inp in inputs:
        flat = inp.flatten()
        orig_size = flat.size
        aligned_size = ((orig_size + alignment - 1) // alignment) * alignment
        if orig_size != aligned_size:
            padded.append(jnp.pad(flat, (0, aligned_size - orig_size)))
        else:
            padded.append(flat)
    return padded


def pallas_gpu_align_output_specs(
    out_shapes: tuple[Any, ...],
    out_dtypes: tuple[Any, ...],
    alignment: int = 128,
) -> tuple[tuple[Any, ...], list[bool]]:
    """Build aligned output ShapeDtypeStruct specs for GPU kernels.

    Returns (aligned_specs, is_scalar_output) where is_scalar_output[i] is True
    when the i-th output is scalar and should not be padded/unpadded.
    """
    import jax  # pyrefly: ignore [import-error, missing-import]

    aligned_specs = []
    is_scalar = []
    for shape, dtype in zip(out_shapes, out_dtypes):
        numel = math.prod(shape)
        if numel <= 1:
            aligned_specs.append(jax.ShapeDtypeStruct(shape, dtype))
            is_scalar.append(True)
        else:
            aligned_numel = ((numel + alignment - 1) // alignment) * alignment
            aligned_specs.append(jax.ShapeDtypeStruct((aligned_numel,), dtype))
            is_scalar.append(False)
    return tuple(aligned_specs), is_scalar


def pallas_gpu_unpad_results(
    results: Any,
    orig_shapes: tuple[Any, ...],
    is_scalar_output: list[bool] | None = None,
) -> Any:
    """Remove padding from GPU kernel results and reshape to original shapes.

    If is_scalar_output is None, all outputs are treated as non-scalar.
    """
    if not isinstance(results, tuple):
        results = (results,)
    unpadded = []
    for i, (res, shape) in enumerate(zip(results, orig_shapes)):
        if is_scalar_output is not None and is_scalar_output[i]:
            unpadded.append(res)
        else:
            orig_numel = math.prod(shape)
            unpadded.append(res[:orig_numel].reshape(shape))
    return unpadded[0] if len(unpadded) == 1 else tuple(unpadded)


# ---------------------------------------------------------------------------
# Pallas CPU / TPU tiling helpers
# ---------------------------------------------------------------------------
# TPU alignment: last dim must be full or a multiple of 128,
#                second-to-last dim must be full or a multiple of 8.
_TPU_ALIGN_LAST = 128
_TPU_ALIGN_SECOND_LAST = 8


def _pallas_tile_size(
    dim: int, alignment: int, max_tile: int = 1024, is_tpu: bool = False
) -> int:
    """Pick the largest aligned tile size <= max_tile for *dim*.

    If *dim* is already <= alignment the full dimension is used (no tiling
    on this axis).
    """
    if dim == 0:
        # Tile size >= 1 avoids division by zero in JAX's _pad_to_block_dimension
        return alignment if is_tpu else 1

    if is_tpu:
        # On TPU, Mosaic requires block dimensions to perfectly align to hardware
        # registers (128 for inner, 8 for outer). We MUST pad the block spec up to
        # the next alignment boundary (Mosaic handles the OOB masking).
        if dim <= alignment:
            return alignment
        t = min(max_tile, dim)
        # Use ceiling division to ensure the tile covers the dimension or aligns upward
        t = ((t + alignment - 1) // alignment) * alignment
        return t

    if dim <= alignment:
        return dim
    t = min(max_tile, dim)
    t = (t // alignment) * alignment
    return max(alignment, t)


def pallas_permute(x, perm):
    """Permute array x according to perm, working around Mosaic TPU bugs.

    Mosaic's jnp.permute_dims produces corrupted output for certain 3D+
    permutations at large sizes.  Decomposes into: loop over smallest output
    dim via unrolled Python loop + stack, with recursive sub-permutation.
    At the 2D base case, uses jnp.permute_dims (which works on Mosaic).

    Callers should ensure tiles are small enough via tiling so that the
    Mosaic bug doesn't trigger (threshold ~ 256K elements).
    """
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    ndim = len(perm)
    if ndim <= 2:
        return jnp.permute_dims(x, perm)
    if perm == tuple(range(ndim)):
        return x  # identity

    shape = x.shape
    out_shape = tuple(shape[p] for p in perm)

    # Choose the output dimension with the smallest size to loop over.
    loop_out_dim = min(range(ndim), key=lambda d: out_shape[d])
    loop_size = out_shape[loop_out_dim]
    loop_in_dim = perm[loop_out_dim]

    # Build the sub-permutation for the remaining (ndim-1) dimensions.
    remaining_in_dims = [d for d in range(ndim) if d != loop_in_dim]
    remaining_out_perm_raw = [perm[d] for d in range(ndim) if d != loop_out_dim]
    dim_map = {old: new for new, old in enumerate(remaining_in_dims)}
    sub_perm = tuple(dim_map[d] for d in remaining_out_perm_raw)

    # Unrolled loop: extract slices with static indices, apply sub-perm,
    # then stack along the loop output dimension.
    slices = []
    # pyrefly: ignore [bad-argument-type]
    for i in range(loop_size):
        idx: list[Any] = [slice(None)] * ndim
        idx[loop_in_dim] = i
        slc = x[tuple(idx)]  # static index, removes loop_in_dim
        slc_p = pallas_permute(slc, sub_perm)
        slices.append(slc_p)

    return jnp.stack(slices, axis=loop_out_dim)


# Maximum tile elements for permutation kernels.  Mosaic's
# jnp.permute_dims produces corrupted output above ~1.5M elements for
# certain 3D+ permutations.  Keep tiles well under that threshold.
_PERMUTE_MAX_TILE_ELEMS = 1 << 18  # ~256K elements


def pallas_compute_tiling(
    ref_shape: tuple[int, ...],
    transpose: bool = False,  # constrain tile size for pallas_permute VMEM
    skip_last_n: int = 0,
    exact_only: bool = False,
    is_tpu: bool = False,
    permutations: list[tuple[int, ...]] | None = None,
    max_grid_product: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[int, int]]:
    """Compute tile shape, grid and axis→grid-dim mapping for CPU/TPU.

    Always uses TPU-compatible alignment (last dim multiple of 128,
    second-to-last multiple of 8) so that the same generated kernel works
    on both CPU-interpret and real TPU.

    *transpose* constrains tile sizes to keep VMEM usage within safe
    limits for pallas_permute (gather-based permutation).

    *skip_last_n* prevents tiling the last N dimensions (used when those
    dims correspond to internal reduction ranges that must remain full).

    *exact_only* restricts tiling to dimensions that divide evenly by the
    tile size (no remainder blocks).  Required on TPU where Mosaic needs
    block shapes to match the XLA memory layout.

    Returns ``(tile_shape, grid, axis_to_grid)`` where *axis_to_grid*
    maps each tiled reference-shape axis index to its position in the
    grid tuple.

    When no dimension benefits from tiling the grid is ``(1,)`` and the
    tile covers the full tensor.
    """
    nd = len(ref_shape)
    if nd == 0:
        return (), (1,), {}

    # Effective number of dims eligible for tiling
    tileable_nd = nd - skip_last_n

    tile = list(ref_shape)
    grid_parts: list[int] = []
    axis_to_grid: dict[int, int] = {}  # ref axis → grid dim

    # Pick alignment based on the physical position of the axis in the
    # tensor, not its position in the tileable subset.  The TPU requires
    # the physical last dim to be a multiple of 128 and the physical
    # second-to-last dim to be a multiple of 8.
    #
    # When permutations are present, an output dim `ax` maps to input
    # dim `perm[ax]`.  If `perm[ax]` is the input's last dim, that
    # output dim must also satisfy _TPU_ALIGN_LAST.  Compute the
    # strictest alignment across all buffers.
    def _input_align(input_ax: int, input_nd: int) -> int:
        return _TPU_ALIGN_LAST if input_ax == input_nd - 1 else _TPU_ALIGN_SECOND_LAST

    def _align(ax: int) -> int:
        out_align = _TPU_ALIGN_LAST if ax == nd - 1 else _TPU_ALIGN_SECOND_LAST
        if permutations:
            for perm in permutations:
                if perm is not None and len(perm) == nd:
                    in_ax = perm[ax]
                    in_align = _input_align(in_ax, len(perm))
                    out_align = max(out_align, in_align)
        return out_align

    def _can_tile_ax(ax: int, dim: int, t: int) -> bool:
        """Check if tiling dim to t is valid."""
        if t >= dim:
            # For TPU padding, we allow tiles >= dimension for the aligned axes
            if _align(ax) == _TPU_ALIGN_LAST or _align(ax) == _TPU_ALIGN_SECOND_LAST:
                return True
            return False
        if exact_only and dim % t != 0:
            if _align(ax) == _TPU_ALIGN_LAST or _align(ax) == _TPU_ALIGN_SECOND_LAST:
                # TPU DMA `#tpu.element_window` natively masks out-of-bounds remainder tiles
                return True
            return False
        return True

    # Second-to-last tileable dim (added first so it becomes grid dim 0)
    if tileable_nd >= 2:
        ax = tileable_nd - 2
        t = _pallas_tile_size(ref_shape[ax], _align(ax), is_tpu=is_tpu)
        if _can_tile_ax(ax, ref_shape[ax], t):
            tile[ax] = t
            axis_to_grid[ax] = len(grid_parts)
            grid_parts.append((ref_shape[ax] + t - 1) // t)

    # Last tileable dim
    if tileable_nd >= 1:
        ax = tileable_nd - 1
        t = _pallas_tile_size(ref_shape[ax], _align(ax), is_tpu=is_tpu)
        if _can_tile_ax(ax, ref_shape[ax], t):
            tile[ax] = t
            axis_to_grid[ax] = len(grid_parts)
            grid_parts.append((ref_shape[ax] + t - 1) // t)

    # When transpose is active, tile dimensions to keep the total tile
    # elements below the safe threshold for Mosaic's permute_dims.
    # Mosaic pads the last dimension to _TPU_ALIGN_LAST (128 for f32),
    # so when the last dim is small (e.g. 2), VMEM usage is much higher
    # than the logical tile size.  We compute an effective element count
    # that accounts for this padding.
    if transpose:
        tile_elems = 1
        for t in tile:
            tile_elems *= t
        last_dim = tile[nd - 1] if nd > 0 else 1
        padded_last = max(last_dim, _TPU_ALIGN_LAST)
        # pallas_permute decomposes ndim>2 permutations by looping over
        # the smallest dim and recursing, creating (ndim-1)-D intermediates
        # at each level.  Mosaic pads the last physical dim of each
        # intermediate to _TPU_ALIGN_LAST (128), which can cause VMEM OOM
        # when a small dim (e.g. 4) ends up as the last dim.
        #
        # For ndim >= 3, compute effective_max from the worst-case VMEM
        # of the (ndim-1)-D intermediate created by pallas_permute.
        # The intermediate removes the loop dim (smallest output dim) and
        # may have a small dim as its last physical dim -> padded to 128.
        if nd >= 4:
            # For 4D+, pallas_permute creates (nd-1)-D intermediates
            # that can have small dims as their last physical dim,
            # which Mosaic pads to _TPU_ALIGN_LAST (128).  This can
            # cause VMEM OOM.  (3D pallas_permute only creates 2D
            # intermediates which Mosaic handles efficiently.)
            sorted_dims = sorted(tile)
            # The loop dim is the smallest; intermediate has the rest.
            inter_dims = sorted_dims[1:]  # (nd-1) dims, sorted
            smallest_inter = inter_dims[0]
            # Padded intermediate: smallest remaining dim padded to 128.
            padded_inter_elems = 1
            for d in inter_dims:
                padded_inter_elems *= d
            pad_factor = max(smallest_inter, _TPU_ALIGN_LAST) // max(smallest_inter, 1)
            padded_inter_elems *= pad_factor
            padded_inter_bytes = padded_inter_elems * 4
            # Conservative VMEM budget: 2MB per intermediate buffer.
            # pallas_permute creates multiple live intermediates (input
            # slices, sub-permutation results, stack temps).  Empirically,
            # 2MB per worst-case intermediate avoids OOM on TPU v4 (64MB VMEM).
            vmem_per_buf = 2 * 1024 * 1024  # 2MB
            if padded_inter_bytes > vmem_per_buf:
                scale = vmem_per_buf / padded_inter_bytes
                effective_max = max(int(tile_elems * scale), 1024)
            else:
                effective_max = _PERMUTE_MAX_TILE_ELEMS
        else:
            # For ndim <= 3: pallas_permute creates 2D intermediates
            # (fine) but jnp.stack creates a 3D result with the same
            # shape as the output.  If the last dim is small, Mosaic
            # pads it to _TPU_ALIGN_LAST (128), which can exhaust VMEM.
            # Empirically: 8MB per padded buffer keeps us safe on TPU v4
            # (64MB VMEM, multiple buffers live simultaneously).
            vmem_per_buf = 8 * 1024 * 1024  # 8MB
            max_nonlast_product = vmem_per_buf // (padded_last * 4)
            effective_max = min(
                max_nonlast_product * last_dim,
                _PERMUTE_MAX_TILE_ELEMS,
            )
        effective_max = min(effective_max, _PERMUTE_MAX_TILE_ELEMS)
        effective_max = max(effective_max, 1024)

        if tile_elems > effective_max:
            # Collect tileable dims.  For ndim >= 4 with VMEM-constrained
            # intermediates, include ALL tileable dims (even the last)
            # since we may need to shrink every dim to fit in VMEM.
            # For ndim <= 3, exclude the last dim (it's often small
            # and already at minimum).
            if nd >= 4:
                tileable_axes = [
                    ax for ax in range(tileable_nd - 1, -1, -1) if ref_shape[ax] > 1
                ]
            else:
                tileable_axes = [
                    ax for ax in range(tileable_nd - 2, -1, -1) if ref_shape[ax] > 1
                ]
            # Distribute reduction proportionally across dims.
            reduction = tile_elems / effective_max
            n = len(tileable_axes)
            per_dim = reduction ** (1.0 / max(n, 1))

            for ax in tileable_axes:
                if tile_elems <= effective_max:
                    break
                dim = ref_shape[ax]
                cur_t = tile[ax]
                align = _align(ax)
                # Target tile: proportional reduction, but also check
                # against the remaining budget for subsequent dims.
                target = int(cur_t / per_dim)
                remaining_budget = effective_max * cur_t // tile_elems
                target = min(target, remaining_budget)
                target = (target // align) * align
                target = max(target, align)
                # Find largest aligned divisor of dim <= target.
                best_t = cur_t
                for candidate in range(target, 0, -align):
                    if candidate >= cur_t:
                        continue
                    if dim % candidate != 0:
                        continue
                    best_t = candidate
                    break
                if best_t < cur_t:
                    if ax in axis_to_grid:
                        grid_parts[axis_to_grid[ax]] = dim // best_t
                    else:
                        axis_to_grid[ax] = len(grid_parts)
                        grid_parts.append(dim // best_t)
                    tile_elems = tile_elems * best_t // cur_t
                    tile[ax] = best_t

    grid = tuple(grid_parts) if grid_parts else (1,)

    # Enforce max_grid_product by scaling up tiles on tiled axes.
    if max_grid_product is not None and grid_parts:
        grid_product = 1
        for g in grid_parts:
            grid_product *= g
        if grid_product > max_grid_product:
            # Sort tiled axes by their grid contribution (descending)
            # and increase tiles to reduce the grid.
            tiled_axes = sorted(
                axis_to_grid.keys(),
                key=lambda ax: grid_parts[axis_to_grid[ax]],
                reverse=True,
            )
            for ax in tiled_axes:
                if grid_product <= max_grid_product:
                    break
                gi = axis_to_grid[ax]
                cur_grid = grid_parts[gi]
                cur_tile = tile[ax]
                dim = ref_shape[ax]
                align = _align(ax)
                # Target grid for this axis: proportional share of reduction
                target_grid = max(1, int(cur_grid * max_grid_product / grid_product))
                target_tile = (dim + target_grid - 1) // target_grid
                # Round up to alignment
                target_tile = ((target_tile + align - 1) // align) * align
                target_tile = min(target_tile, dim)
                # Find smallest aligned value >= target_tile that divides dim
                # (or just use the rounded-up value if exact_only is False)
                best_tile = dim  # fallback: full dim
                if exact_only:
                    for candidate in range(target_tile, dim + 1, align):
                        if dim % candidate == 0:
                            best_tile = candidate
                            break
                else:
                    best_tile = target_tile
                if best_tile > cur_tile:
                    new_grid = (dim + best_tile - 1) // best_tile
                    grid_product = grid_product * new_grid // cur_grid
                    grid_parts[gi] = new_grid
                    tile[ax] = best_tile
            grid = tuple(grid_parts)

    return tuple(tile), grid, axis_to_grid


def pallas_make_block_spec(
    buf_shape: tuple[int, ...],
    ref_shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    axis_to_grid: dict[int, int],
    n_grid: int,
    permutation: tuple[int, ...] | None = None,
    is_output: bool = False,
) -> Any:
    """Build a ``pl.BlockSpec`` for *buf_shape* given tiling of *ref_shape*.

    Lower-ndim buffers are right-aligned with the reference shape (numpy
    broadcast rules).  Dimensions that match a tiled reference dimension
    are tiled; broadcast dimensions (size 1 or absent) are kept full.

    When *buf_nd > ref_nd* (reduction inputs), we find an alignment offset
    so the ref dims map into the buffer.  Extra dims are kept at full size
    with index 0 in the index_map.

    When *permutation* is given (a tuple mapping ref axis -> buf axis),
    the buffer dimensions are accessed in permuted order relative to the
    reference shape.  For example, permutation=(1, 0) swaps the last two.

    When *is_output* is True and *buf_nd < ref_nd*, left-alignment is used
    as a fallback (for reduction outputs whose trailing dims were reduced).
    """
    from jax.experimental import (  # pyrefly: ignore [import-error, missing-import]
        pallas as pl,
    )

    buf_nd = len(buf_shape)
    ref_nd = len(ref_shape)

    if buf_nd == 0:
        # Scalar — untouched regardless of grid shape.
        return pl.BlockSpec((1,), _make_index_map([], 1, n_grid))

    bs = list(buf_shape)
    tiled_pairs: list[tuple[int, int]] = []

    if buf_nd > ref_nd:
        # Reduction input: find alignment offset k where ref dims map into buf.
        align_k = 0
        for k in range(buf_nd - ref_nd + 1):
            ok = True
            for i in range(ref_nd):
                if ref_shape[i] == 1:
                    continue
                if buf_shape[k + i] != ref_shape[i]:
                    ok = False
                    break
            if ok:
                align_k = k
                break

        for ref_ax, grid_dim in axis_to_grid.items():
            buf_ax = align_k + ref_ax
            if 0 <= buf_ax < buf_nd and buf_shape[buf_ax] == ref_shape[ref_ax]:
                bs[buf_ax] = tile_shape[ref_ax]
                tiled_pairs.append((buf_ax, grid_dim))

    elif permutation is not None and buf_nd == ref_nd:
        # Permuted buffer: map ref axes through the permutation.
        for ref_ax, grid_dim in axis_to_grid.items():
            buf_ax = permutation[ref_ax]
            if 0 <= buf_ax < buf_nd and buf_shape[buf_ax] == ref_shape[ref_ax]:
                bs[buf_ax] = tile_shape[ref_ax]
                tiled_pairs.append((buf_ax, grid_dim))

    else:
        # Standard right-alignment, with left-alignment fallback for
        # reduction outputs (e.g. sum(dim=-1) on (10,10) → (10,)).
        for ref_ax, grid_dim in axis_to_grid.items():
            buf_ax = ref_ax - (ref_nd - buf_nd)
            if 0 <= buf_ax < buf_nd and buf_shape[buf_ax] == ref_shape[ref_ax]:
                bs[buf_ax] = tile_shape[ref_ax]
                tiled_pairs.append((buf_ax, grid_dim))
            elif (
                is_output
                and buf_nd < ref_nd
                and 0 <= ref_ax < buf_nd
                and buf_shape[ref_ax] == ref_shape[ref_ax]
            ):
                # Left-aligned match for output: buf dim i matches ref dim i
                # (reduction output whose trailing dims were reduced away)
                bs[ref_ax] = tile_shape[ref_ax]
                tiled_pairs.append((ref_ax, grid_dim))

    return pl.BlockSpec(
        tuple(bs),
        _make_index_map(tiled_pairs, buf_nd, n_grid),
    )


def _make_index_map(
    tiled_pairs: list[tuple[int, int]],
    buf_nd: int,
    n_grid: int,
) -> Any:
    """Return an index_map callable for ``pl.BlockSpec``.

    *tiled_pairs* is a list of ``(buf_axis, grid_dim)`` indicating which
    buffer axes receive a grid index.  All other axes return 0 (full block).

    All returned values are explicitly ``jnp.int32`` so that TPU Mosaic
    lowering (which rejects 64-bit types) works when ``jax_enable_x64`` is
    active.  The casts are created inside the function body (not captured)
    to satisfy JAX's "index_map must not capture constants" rule.
    """
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    # Pre-build the mapping so the returned lambda is a plain lookup.
    mapping = dict(tiled_pairs)

    if n_grid == 0 or (n_grid == 1 and not mapping):
        return lambda _i: tuple(jnp.int32(0) for _ in range(buf_nd))

    def index_map(*grid_args):
        return tuple(
            jnp.int32(grid_args[mapping[d]]) if d in mapping else jnp.int32(0)
            for d in range(buf_nd)
        )

    return index_map


def pallas_ensure_nonzero_rank(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 0:
        return x.reshape((1,))
    return x


def pallas_make_block_spec_non_tiled(shape: tuple[int, ...]) -> Any:
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]
    from jax.experimental import (  # pyrefly: ignore [import-error, missing-import]
        pallas as pl,
    )

    nonzero_rank_shape = shape if len(shape) > 0 else (1,)
    return pl.BlockSpec(
        nonzero_rank_shape,
        lambda i: [jnp.int32(i)] * len(nonzero_rank_shape),
    )
<|endoftext|>from __future__ import annotations

import functools
import linecache
import os
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, TYPE_CHECKING

from torch._utils_internal import log_triton_builds

from ..utils import apply_subprocess_env, clear_caches


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner


def _reload_python_module(
    key: str, path: str, set_sys_modules: bool = True
) -> ModuleType:
    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
        if set_sys_modules:
            sys.modules[mod.__name__] = mod
        return mod


@functools.cache
def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas = Path(__file__).absolute().parents[2] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _set_triton_libdevice_path() -> None:
    """
    Use the CUDA toolkit's libdevice instead of Triton's bundled version.
    This ensures Triton's libdevice calls match CUDA eager numerics for bitwise
    precision.  Gated by config.eager_numerics.use_pytorch_libdevice and by
    config.emulate_precision_casts, which also requests eager-like numerics.
    """
    from torch._inductor import config

    if not (
        config.eager_numerics.use_pytorch_libdevice or config.emulate_precision_casts
    ):
        return

    _set_triton_libdevice_path_impl()


def _set_triton_libdevice_path_impl() -> None:
    import torch

    if torch.version.cuda is None:
        return

    try:
        from triton import knobs
    except ImportError:
        return

    env_path = os.environ.get("TRITON_LIBDEVICE_PATH")
    if env_path is not None:
        knobs.nvidia.libdevice_path = env_path
        return

    if knobs.nvidia.libdevice_path is not None:
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            warnings.warn(
                "CUDA_HOME not set; using Triton's bundled libdevice which may "
                "cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
            return
        libdevice = Path(CUDA_HOME) / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.is_file():
            knobs.nvidia.libdevice_path = str(libdevice)
            # Also set env var so subprocess compile workers inherit it
            os.environ["TRITON_LIBDEVICE_PATH"] = str(libdevice)
        else:
            warnings.warn(
                f"CUDA libdevice not found at {libdevice}; using Triton's bundled "
                "libdevice which may cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "torch.utils.cpp_extension not available; using Triton's bundled "
            "libdevice which may cause minor precision differences in pow operations. "
            "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
            "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            stacklevel=3,
        )


_WORKER_CACHE_ENV_VARS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
_last_applied_cache_env: dict[str, str | None] | None = None


def _apply_subprocess_env_and_clear_caches(extra_env: dict[str, str | None]) -> None:
    global _last_applied_cache_env

    cache_env = {
        key: extra_env.get(key) for key in _WORKER_CACHE_ENV_VARS if key in extra_env
    }
    if cache_env and cache_env != _last_applied_cache_env:
        clear_caches()
        _last_applied_cache_env = cache_env.copy()
    apply_subprocess_env(extra_env)


def _worker_compile_pycodecache_kernel(
    kernel_name: str,
    source_code: str,
    main_suffix: str,
    extra_env: dict[str, str | None],
    precompile_metadata: dict[str, Any] | None = None,
) -> tuple[str, str, int]:
    """
    Subprocess worker for PyCodeCache-based kernel compilation.

    Writes source to PyCodeCache, loads the module, validates the entry point,
    and optionally triggers real GPU compilation (MLIR -> PTX -> CUBIN) via a
    _precompile entry point. Compiled artifacts are persisted to disk cache so
    the parent process can load them without recompilation.

    Used by both CuteDSL and NV Universal GEMM backends.
    """
    _apply_subprocess_env_and_clear_caches(extra_env)

    start_ns = time.time_ns()

    import torch._inductor.codecache as codecache

    key, path = codecache.PyCodeCache.write(source_code)
    mod = codecache.PyCodeCache.load_by_key_path(key, path)

    main_func_name = f"{kernel_name}_{main_suffix}"
    if not hasattr(mod, main_func_name):
        available = [name for name in dir(mod) if callable(getattr(mod, name))]
        raise RuntimeError(
            f"Could not find kernel function '{main_func_name}'. "
            f"Available callables: {available}"
        )

    if precompile_metadata is not None:
        precompile_fn_name = f"{kernel_name}_precompile"
        precompile_fn = getattr(mod, precompile_fn_name, None)
        if precompile_fn is not None:
            precompile_fn(**precompile_metadata)
        else:
            import logging

            logging.getLogger(__name__).warning(
                "Precompile metadata was provided but module has no %s "
                "— the scheduling layer expected this template to support "
                "subprocess precompilation. Kernel will compile lazily on "
                "first call instead.",
                precompile_fn_name,
            )

    elapsed_ns = time.time_ns() - start_ns
    linecache.clearcache()
    return key, path, elapsed_ns // 1000


def _worker_compile_triton(
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str | None],
    extra_config: dict[str, Any],
) -> tuple[CachingAutotuner, int]:
    _set_triton_ptxas_path()
    _apply_subprocess_env_and_clear_caches(extra_env)
    # Keep Triton's in-process knob in sync with the parent environment, including
    # clearing stale worker state when the parent no longer has this variable.
    if "TRITON_LIBDEVICE_PATH" in extra_env:
        try:
            from triton import knobs

            knobs.nvidia.libdevice_path = extra_env["TRITON_LIBDEVICE_PATH"]
        except ImportError:
            pass
    from torch._inductor import config
    from torch._inductor.compile_worker import watchdog
    from torch._inductor.runtime import triton_helpers

    with config.patch(extra_config):
        fail = None
        try:
            start_ns = time.time_ns()
            # Generated Triton modules set up the GPU driver at import time,
            # but compile workers only need to warm the compile cache.
            with triton_helpers.skip_gpu_driver_setup():
                kernel = load_kernel()
                watchdog.report_phase(watchdog.Phase.COMPILING)
                kernel.precompile(warm_cache_only=True)
            elapsed_ns = time.time_ns() - start_ns
            kernel.prepare_for_pickle()
            # We can release this memory in the compile subprocesses:
            linecache.clearcache()
            return kernel, elapsed_ns // 1000
        except Exception as e:
            fail = str(e)
            raise
        finally:
            log_triton_builds(fail=fail)
<|endoftext|>"""
Utilities for post-processing Triton Proton profiling trace files.

This module provides functions to transform proton-generated Chrome trace files
for better visualization in Perfetto/Chrome tracing tools.
"""

import gzip
import json
import os
import re
from typing import Any

from torch.utils._ordered_set import OrderedSet


def _write_trace(path: str, data: dict[str, Any]) -> None:
    """Write trace data to a gzipped JSON file."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def _read_trace(path: str) -> dict[str, Any]:
    """Read trace data from a JSON file (gzipped or not)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)


def _get_base_name(trace_path: str) -> str:
    """Get the base name of a trace file, stripping known extensions."""
    base_name = os.path.basename(trace_path)
    for ext in [".trace.json.gz", ".chrome_trace", ".json.gz", ".json"]:
        if base_name.endswith(ext):
            return base_name[: -len(ext)]
    return os.path.splitext(base_name)[0]


def _group_events_by_sm(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group events so all CTAs on the same SM share one track.

    Transforms:
    - pid: "Core0 CTA6" -> "Core0"
    - tid: "warp0" -> "CTA6 warp0"
    """
    core_cta_pattern = re.compile(r"^(.*?)\s*(Core\d+)\s+(CTA\d+)$")

    for event in events:
        pid = event.get("pid", "")
        tid = event.get("tid", "")

        match = core_cta_pattern.match(str(pid))
        if match:
            prefix = match.group(1).strip()
            core = match.group(2)
            cta = match.group(3)
            event["pid"] = f"{prefix} {core}" if prefix else core
            event["tid"] = f"{cta} {tid}" if tid else cta

    return events


def _group_events_per_cta_occupancy(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Process warp events into CTA-level events with slot assignment.

    Each CTA gets: min warp start, max warp end.
    CTAs are assigned to slots per SM such that non-overlapping CTAs share slots.
    """
    core_cta_pattern = re.compile(r"^(.*?)\s*(Core\d+)\s+(CTA\d+)$")

    # Group events by (prefix, SM, CTA)
    cta_events: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for event in events:
        pid = event.get("pid", "")
        match = core_cta_pattern.match(str(pid))
        if match:
            prefix = match.group(1).strip()
            core = match.group(2)
            cta = match.group(3)
            key = (prefix, core, cta)
            if key not in cta_events:
                cta_events[key] = []
            cta_events[key].append(event)

    # For each CTA, compute min start and max end
    cta_intervals: dict[tuple[str, str, str], tuple[float, float]] = {}
    for key, evts in cta_events.items():
        min_start = float("inf")
        max_end = float("-inf")
        for evt in evts:
            ts = evt.get("ts", 0)
            dur = evt.get("dur", 0)
            min_start = min(min_start, ts)
            max_end = max(max_end, ts + dur)
        if min_start != float("inf"):
            cta_intervals[key] = (min_start, max_end)

    # Group CTAs by (prefix, SM) and assign to slots
    sm_ctas: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
    for (prefix, core, cta), (start, end) in cta_intervals.items():
        sm_key = (prefix, core)
        if sm_key not in sm_ctas:
            sm_ctas[sm_key] = []
        sm_ctas[sm_key].append((cta, start, end))

    # Assign CTAs to slots using interval scheduling (greedy)
    cta_slot_assignments: dict[tuple[str, str, str], int] = {}
    for sm_key, ctas in sm_ctas.items():
        prefix, core = sm_key
        sorted_ctas = sorted(ctas, key=lambda x: x[1])
        slots: list[float] = []
        for cta, start, end in sorted_ctas:
            assigned_slot = None
            for i, slot_end in enumerate(slots):
                if start >= slot_end:
                    assigned_slot = i
                    slots[i] = end
                    break
            if assigned_slot is None:
                assigned_slot = len(slots)
                slots.append(end)
            cta_slot_assignments[(prefix, core, cta)] = assigned_slot

    # Build numeric ID mappings for pids and tids
    # Chrome trace format requires numeric pid/tid values
    pid_names: dict[str, int] = {}
    tid_names: dict[tuple[int, str], int] = {}  # (pid, tid_name) -> tid_num

    for key in cta_intervals:
        prefix, core, cta = key
        pid_name = f"{prefix} {core}" if prefix else core
        if pid_name not in pid_names:
            pid_names[pid_name] = len(pid_names)

    new_events: list[dict[str, Any]] = []

    # Add process name metadata events
    for pid_name, pid_num in pid_names.items():
        new_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid_num,
                "args": {"name": pid_name},
            }
        )

    # Create CTA events with numeric IDs and collect tid mappings
    for key, (start, end) in cta_intervals.items():
        prefix, core, cta = key
        slot = cta_slot_assignments[key]
        pid_name = f"{prefix} {core}" if prefix else core
        tid_name = f"slot{slot}"
        pid_num = pid_names[pid_name]

        if (pid_num, tid_name) not in tid_names:
            tid_num = len(tid_names)
            tid_names[(pid_num, tid_name)] = tid_num
            # Add thread name metadata event
            new_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid_num,
                    "tid": tid_num,
                    "args": {"name": tid_name},
                }
            )
        else:
            tid_num = tid_names[(pid_num, tid_name)]

        new_events.append(
            {
                "name": cta,
                "cat": "cta",
                "ph": "X",
                "ts": start,
                "dur": end - start,
                "pid": pid_num,
                "tid": tid_num,
            }
        )

    return new_events


def _apply_grouping(
    events: list[dict[str, Any]], group_by_sm: bool, per_cta_occupancy: bool
) -> list[dict[str, Any]]:
    """Apply grouping transformation to events."""
    if per_cta_occupancy:
        return _group_events_per_cta_occupancy(events)
    elif group_by_sm:
        return _group_events_by_sm(events)
    return events


def _split_events_by_invocation(
    events: list[dict[str, Any]],
    gap_threshold_ns: float = 1000.0,
) -> list[list[dict[str, Any]]]:
    """Split events into separate invocations based on time gaps."""
    if not events:
        return []

    events_sorted = sorted(events, key=lambda e: e.get("ts", 0))
    invocations: list[list[dict[str, Any]]] = [[]]
    prev_end = events_sorted[0].get("ts", 0)

    for event in events_sorted:
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)
        if ts - prev_end > gap_threshold_ns and invocations[-1]:
            invocations.append([])
        invocations[-1].append(event)
        prev_end = max(prev_end, ts + dur)

    return [inv for inv in invocations if inv]


def _normalize_timestamps(
    events: list[dict[str, Any]], scale_factor: float = 1.0
) -> list[dict[str, Any]]:
    """Normalize timestamps to start at 0 and optionally scale."""
    if not events:
        return events

    # Only consider duration events (ph=X) for min timestamp calculation
    duration_events = [e for e in events if e.get("ph") == "X"]
    if not duration_events:
        return events

    min_ts = min(e.get("ts", 0) for e in duration_events)
    for event in events:
        if event.get("ph") == "X":
            event["ts"] = (event.get("ts", 0) - min_ts) * scale_factor
            if "dur" in event:
                event["dur"] = event["dur"] * scale_factor

    return events


def process_proton_trace(
    trace_path: str,
    output_dir: str | None = None,
    group_by_sm: bool = True,
    split_invocations: bool = True,
    scale_factor: float = 1.0,
    gap_threshold_ns: float = 1000.0,
    per_cta_occupancy: bool = True,
) -> list[str]:
    """
    Process a proton trace file with various transformations.

    Always produces 1 + N files: one complete trace and N per-invocation traces.
    Grouping options (group_by_sm, per_cta_occupancy) apply uniformly to all outputs.

    Args:
        trace_path: Path to the input Chrome trace file
        output_dir: Directory to write output files. If None, uses same directory as input.
        group_by_sm: If True, group CTAs by SM into single tracks.
        split_invocations: If True, also produce per-invocation trace files.
        scale_factor: Factor to scale durations by (helps visibility in Perfetto).
        gap_threshold_ns: Time gap (in nanoseconds) that indicates a new invocation.
        per_cta_occupancy: If True, process warp tracks into CTA tracks and assign
            CTAs to slots per SM such that CTAs do not overlap.

    Returns:
        List of paths to the output files.
    """
    if output_dir is None:
        output_dir = os.path.dirname(trace_path) or "."

    os.makedirs(output_dir, exist_ok=True)
    base_name = _get_base_name(trace_path)

    data = _read_trace(trace_path)
    events = data.get("traceEvents", [])

    # Split into invocations first (before grouping, to avoid merging across invocations)
    invocations = _split_events_by_invocation(events, gap_threshold_ns)

    # Apply grouping transformation to each invocation
    invocations = [
        _apply_grouping(inv_events, group_by_sm, per_cta_occupancy)
        for inv_events in invocations
    ]

    output_files = []

    # Write complete trace (dedupe metadata events)
    complete_events = []
    seen_metadata: OrderedSet[tuple[str | None, int | None, int | None]] = OrderedSet()
    for inv_events in invocations:
        for event in inv_events:
            if event.get("ph") == "M":
                # Dedupe metadata events by (name, pid, tid)
                key = (event.get("name"), event.get("pid"), event.get("tid"))
                if key in seen_metadata:
                    continue
                seen_metadata.add(key)
            complete_events.append(event)
    complete_path = os.path.join(output_dir, f"{base_name}.trace.json.gz")
    _write_trace(complete_path, {"traceEvents": complete_events})
    output_files.append(complete_path)

    # Write per-invocation traces
    if split_invocations:
        for i, inv_events in enumerate(invocations):
            inv_events = _normalize_timestamps(list(inv_events), scale_factor)
            inv_path = os.path.join(
                output_dir, f"{base_name}_invocation_{i}.trace.json.gz"
            )
            _write_trace(inv_path, {"traceEvents": inv_events})
            output_files.append(inv_path)

    return output_files
<|endoftext|>"""Disk cache for CuTe DSL compiled kernel artifacts.

Persists compiled GPU binaries (CUBIN/PTX via TVM FFI) across process
boundaries so that subprocess compilation workers can produce artifacts
that the parent process loads without recompilation.

Cache keys include the module's PyCodeCache path (content-addressed),
compile-time configuration, runtime shapes/dtypes, GPU architecture,
and CUDA toolkit version to prevent stale artifact reuse.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import struct
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


def _fix_elf_dup_text_flags(data: bytes) -> bytes:
    """Harmonize flags on duplicate .text ELF sections.

    The CUTLASS MLIR compiler emits .o files with two .text sections:
    one for code (ALLOC|EXECINSTR) and one for writable data
    (WRITE|ALLOC). LLVM's MCJIT mishandles this in multi-process
    workloads, causing non-deterministic SIGSEGV. Fix: set duplicate
    .text sections' flags to match the first (ALLOC|EXECINSTR = 0x6).

    See https://github.com/NVIDIA/cutlass/issues/3162
    """
    if len(data) < 64 or data[4] != 2 or data[5] != 1:
        return data
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]
    if not e_shoff or not e_shnum or e_shstrndx >= e_shnum:
        return data
    shstr_hdr = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from("<Q", data, shstr_hdr + 24)[0]
    text_secs: list[tuple[int, int]] = []
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        ni = struct.unpack_from("<I", data, sh)[0]
        ns = shstr_off + ni
        if ns + 6 <= len(data) and data[ns : ns + 6] == b".text\x00":
            text_secs.append((i, sh))
    if len(text_secs) <= 1:
        return data
    r = bytearray(data)
    for _, sh in text_secs[1:]:
        struct.pack_into("<Q", r, sh + 8, 0x6)
    return bytes(r)


def _cache_dir() -> Path:
    cache_root = os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "torch_inductor"),
    )
    return Path(cache_root) / "cutedsl_compile_cache"


def _make_disk_key(
    module_path: str,
    config_key: tuple[Any, ...],
    runtime_key: tuple[Any, ...],
    device_index: int = 0,
    device_capability: tuple[int, int] | None = None,
) -> str:
    import torch

    arch = device_capability
    if arch is None and torch.cuda.is_available():
        arch = torch.cuda.get_device_capability(device_index)
    cuda_version = getattr(torch.version, "cuda", None)

    full_key = (module_path, config_key, runtime_key, arch, cuda_version)
    return hashlib.sha256(pickle.dumps(full_key)).hexdigest()


def disk_cache_get(
    mem_cache: dict[Any, Any],
    module_path: str,
    config_key: tuple[Any, ...],
    runtime_key: tuple[Any, ...],
    device_index: int = 0,
    device_capability: tuple[int, int] | None = None,
) -> Any | None:
    """Look up a compiled CuTe DSL function, checking memory then disk."""
    mem_key = (runtime_key, device_index)
    if mem_key in mem_cache:
        return mem_cache[mem_key]

    h = _make_disk_key(
        module_path, config_key, runtime_key, device_index, device_capability
    )
    obj_path = _cache_dir() / f"{h}.o"
    if obj_path.exists():
        try:
            raw = obj_path.read_bytes()
            patched = _fix_elf_dup_text_flags(raw)
            if patched is not raw:
                obj_path.write_bytes(patched)

            import cutlass.cute as cute

            m = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=True)
            fn = m.func
            mem_cache[mem_key] = fn
            return fn
        except Exception:
            log.debug("Failed to load cached artifact %s", obj_path, exc_info=True)
    return None


def disk_cache_set(
    mem_cache: dict[Any, Any],
    module_path: str,
    config_key: tuple[Any, ...],
    runtime_key: tuple[Any, ...],
    compiled_fn: Any,
    device_index: int = 0,
    device_capability: tuple[int, int] | None = None,
) -> None:
    """Store a compiled CuTe DSL function to memory and disk."""
    mem_key = (runtime_key, device_index)
    mem_cache[mem_key] = compiled_fn

    h = _make_disk_key(
        module_path, config_key, runtime_key, device_index, device_capability
    )
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    obj_path = d / f"{h}.o"
    import tempfile

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(d), suffix=".o.tmp")
        os.close(fd)
        compiled_fn.export_to_c(object_file_path=tmp_path, function_name="func")
        with open(tmp_path, "rb") as f:
            patched = _fix_elf_dup_text_flags(f.read())
        with open(tmp_path, "wb") as f:
            f.write(patched)
        os.replace(tmp_path, str(obj_path))
    except (AttributeError, RuntimeError, TypeError):
        log.debug(
            "export_to_c not available for %s, skipping disk persistence",
            type(compiled_fn).__name__,
            exc_info=True,
        )
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
<|endoftext|>"""
Lazy Triton kernel compilation for C++ wrapper.

This module provides functionality to compile and autotune Triton kernels at runtime
when the C++ wrapper is used without autotune_at_compile_time.

The workflow is:
1. At model initialization: Call start_kernel_compile() for all kernels to start
   parallel compilation using multi-process async_compile
2. At kernel execution time: Call run_triton_kernel_with_autotune() which waits
   for the specific kernel to be ready, then runs it with autotuning
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

from .hints import (
    TRITON_DEFAULT_BLOCK_SIZES,
    TRITON_DEFAULT_RSPLIT,
    TRITON_DEFAULT_RSPLIT_SIZE,
)
from .triton_heuristics import CachingAutotuner


log = logging.getLogger(__name__)


@dataclasses.dataclass
class TritonKernelCompileResult:
    cubin_path: str
    mangled_name: str
    num_warps: int
    shared_mem: int
    xblocks: list[int]
    yblocks: list[int]
    zblocks: list[int]
    r0blocks: list[int]
    rsplit: int
    rsplit_size: int
    config_index: int | None
    global_scratch: int | None
    profile_scratch: int | None


_async_compile: Any = None


def _get_async_compile() -> Any:
    """Get or create the shared AsyncCompile instance."""
    global _async_compile
    if _async_compile is None:
        from torch._inductor.async_compile import AsyncCompile

        _async_compile = AsyncCompile()
    return _async_compile


def _wrap_tma_args(args: list[Any], kernel_fn: CachingAutotuner) -> list[Any]:
    """Wrap tensor args with TMA descriptors where the signature requires them."""
    signature = kernel_fn.triton_meta.get("signature", {})
    sig_items = list(signature.items())

    # Track args index separately from sig_items index since the signature
    # may include constexpr entries that are not present in args.
    tma_indices = []
    arg_idx = 0
    for name, sig_type in sig_items:
        if isinstance(sig_type, str) and sig_type == "constexpr":
            continue
        if isinstance(sig_type, str) and sig_type == "nvTmaDesc":
            raise RuntimeError(
                f"nvTmaDesc (experimental TMA API) is not supported in lazy compile "
                f"for arg '{name}'. Use the stable tensordesc API instead."
            )
        if isinstance(sig_type, str) and sig_type.startswith("tensordesc<"):
            tma_indices.append((arg_idx, name, sig_type))
        arg_idx += 1

    if not tma_indices:
        return args

    from triton.tools.tensor_descriptor import TensorDescriptor

    wrapped = list(args)
    for arg_idx, name, sig_type in tma_indices:
        if arg_idx >= len(wrapped):
            raise RuntimeError(
                f"TMA arg index {arg_idx} for '{name}' exceeds arg count {len(wrapped)}"
            )
        tensor = wrapped[arg_idx]
        # Parse block_shape from tensordesc<dtype[dim0, dim1, ...]>
        match = re.match(r"tensordesc<[^[]*\[([^\]]*)\]", sig_type)
        if match:
            block_shape = [int(x.strip()) for x in match.group(1).split(",")]
            wrapped[arg_idx] = TensorDescriptor.from_tensor(tensor, block_shape)

    return wrapped


def start_kernel_compile(
    pending_kernels: dict[str, Any], kernel_name: str, kernel_source: str
) -> None:
    """
    This function is called from C++ at model initialization time for each kernel.
    It starts the compilation in a background process but does NOT wait for it.
    The actual kernel execution happens later in run_triton_kernel_with_autotune().

    The pending_kernels dict is per-module, created in C++ and passed through
    to avoid global state collisions across compiled modules.
    """
    if kernel_name in pending_kernels:
        return

    async_compile = _get_async_compile()  # noqa: F841 (used by eval below)

    # Evaluate the kernel source to get the Future or CachingAutotuner
    # The kernel_source is like: async_compile.triton('name', '''...''', ...)
    kernel_obj = eval(kernel_source.strip())

    pending_kernels[kernel_name] = kernel_obj


def run_triton_kernel_with_autotune(
    pending_kernels: dict[str, Any],
    kernel_name: str,
    stream: Any,
    args: list[Any],
) -> TritonKernelCompileResult:
    """
    Run a Triton kernel with full autotuning using actual tensor arguments.
    """
    from torch._inductor.codecache import CodeCacheFuture, CudaKernelParamCache
    from torch._inductor.runtime.triton_heuristics import config_to_dict

    if kernel_name not in pending_kernels:
        raise RuntimeError(f"Kernel {kernel_name} not found in pending kernels.")
    kernel_obj = pending_kernels[kernel_name]

    if isinstance(kernel_obj, CodeCacheFuture):
        kernel_fn = kernel_obj.result()
    elif isinstance(kernel_obj, CachingAutotuner):
        kernel_fn = kernel_obj
    else:
        raise RuntimeError(f"Unexpected kernel object type: {type(kernel_obj)}")

    if not isinstance(kernel_fn, CachingAutotuner):
        raise AssertionError(f"Expected CachingAutotuner, got {type(kernel_fn)}")

    inductor_meta = kernel_fn.inductor_meta
    inductor_meta["store_cubin"] = True

    # For TMA kernels, wrap tensor args with TMA descriptors
    args = _wrap_tma_args(args, kernel_fn)

    # Run the kernel with the provided arguments
    # This will trigger autotuning if there are multiple configs
    kernel_fn.run(*args, stream=stream)
    if not kernel_fn.launchers:
        raise RuntimeError("Kernel run did not produce any launchers")
    launcher = kernel_fn.launchers[0]

    cached_params: dict[str, Any] | None = CudaKernelParamCache.get(kernel_name)
    if cached_params is None:
        raise RuntimeError(f"Failed to get cached params for kernel {kernel_name}")

    from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name

    cubin_path_name = get_cpp_wrapper_cubin_path_name()
    for key_name in (cubin_path_name, "mangled_name", "num_warps", "shared_mem"):
        if key_name not in cached_params:
            raise RuntimeError(
                f"{key_name} not found in cached params for {kernel_name}"
            )
    cubin_path = cached_params[cubin_path_name]
    if not isinstance(cubin_path, str):
        raise AssertionError(f"expected cubin_path to be str, got {type(cubin_path)}")
    runtime_bin_path = cached_params.get("runtime_bin_path", cubin_path)
    if not isinstance(runtime_bin_path, str):
        raise AssertionError(
            f"expected runtime_bin_path to be str, got {type(runtime_bin_path)}"
        )
    mangled_name = cached_params["mangled_name"]
    num_warps = cached_params["num_warps"]
    shared_mem = cached_params["shared_mem"]

    config = config_to_dict(launcher.config) if launcher.config else {}

    # For combo/foreach kernels, the autotuned config may have empty kwargs
    # (e.g., the foreach heuristic only tunes num_warps, not XBLOCK).
    # In that case, use the default_config from combo_grid_meta
    combo_grid_meta = inductor_meta.get("combo_grid_meta") if inductor_meta else None
    default_config = combo_grid_meta.get("default_config") if combo_grid_meta else None
    if default_config:
        config = {**default_config, **config}

    # Extract per-subkernel block sizes for combo kernels, or single block sizes.
    num_kernels = combo_grid_meta.get("num_kernels", 1) if combo_grid_meta else 1
    if num_kernels > 1 and "XBLOCK_0" in config:
        xblocks = [
            config.get(f"XBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"])
            for i in range(num_kernels)
        ]
        yblocks = [
            config.get(f"YBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["YBLOCK"])
            for i in range(num_kernels)
        ]
        zblocks = [
            config.get(f"ZBLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["ZBLOCK"])
            for i in range(num_kernels)
        ]
        r0blocks = [
            config.get(f"R0_BLOCK_{i}", TRITON_DEFAULT_BLOCK_SIZES["R0_BLOCK"])
            for i in range(num_kernels)
        ]
    else:
        xblocks = [config.get("XBLOCK", TRITON_DEFAULT_BLOCK_SIZES["XBLOCK"])]
        yblocks = [config.get("YBLOCK", TRITON_DEFAULT_BLOCK_SIZES["YBLOCK"])]
        zblocks = [config.get("ZBLOCK", TRITON_DEFAULT_BLOCK_SIZES["ZBLOCK"])]
        r0blocks = [config.get("R0_BLOCK", TRITON_DEFAULT_BLOCK_SIZES["R0_BLOCK"])]
    rsplit = config.get("RSPLIT", TRITON_DEFAULT_RSPLIT)
    rsplit_size = config.get("RSPLIT_SIZE", TRITON_DEFAULT_RSPLIT_SIZE)

    config_index = None
    grid_type = inductor_meta.get("grid_type") if inductor_meta else None
    if grid_type == "PrecomputedGrid" and inductor_meta:
        # PrecomputedGrid selects one of precomputed_grids. We use config_index
        # to remember which grid is chosen.
        precomputed_grids = inductor_meta.get("precomputed_grids", [])
        for idx, entry in enumerate(precomputed_grids):
            entry_config = entry.get("config", {})
            if all(config.get(k) == v for k, v in entry_config.items()):
                config_index = idx
                break

    global_scratch: int | None = cached_params.get("global_scratch")
    profile_scratch: int | None = cached_params.get("profile_scratch")

    log.debug(
        "Successfully autotuned Triton kernel: cubin_path=%s, "
        "runtime_bin_path=%s, mangled_name=%s, "
        "num_warps=%d, shared_mem=%d, xblocks=%s, yblocks=%s, zblocks=%s, r0blocks=%s, "
        "rsplit=%d, rsplit_size=%d, config_index=%s, global_scratch=%s, profile_scratch=%s",
        cubin_path,
        runtime_bin_path,
        mangled_name,
        num_warps,
        shared_mem,
        xblocks,
        yblocks,
        zblocks,
        r0blocks,
        rsplit,
        rsplit_size,
        config_index,
        global_scratch,
        profile_scratch,
    )

    result = TritonKernelCompileResult(
        cubin_path=runtime_bin_path,
        mangled_name=mangled_name,
        num_warps=num_warps,
        shared_mem=shared_mem,
        xblocks=xblocks,
        yblocks=yblocks,
        zblocks=zblocks,
        r0blocks=r0blocks,
        rsplit=rsplit,
        rsplit_size=rsplit_size,
        config_index=config_index,
        global_scratch=global_scratch,
        profile_scratch=profile_scratch,
    )
    return result
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math as pymath
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from .triton_compat import (
    _log2,
    builtins_use_semantic_kwarg,
    JITFunction,
    libdevice,
    math,
    tl,
    triton,
)


_T = TypeVar("_T")
_LOG_2_E: tl.constexpr = tl.constexpr(pymath.log2(pymath.e))
_skip_gpu_driver_setup: ContextVar[bool] = ContextVar(
    "_skip_gpu_driver_setup", default=False
)


@contextmanager
def skip_gpu_driver_setup():
    # Scoped no-op for set_driver_to_gpu(). ContextVar keeps nested/thread-local
    # uses isolated.
    token = _skip_gpu_driver_setup.set(True)
    try:
        yield
    finally:
        _skip_gpu_driver_setup.reset(token)


def set_driver_to_cpu():
    driver = triton.runtime.driver
    if backend := triton.backends.backends.get("cpu", None):
        if isinstance(driver.active, backend.driver):
            # Don't re-initialize backend if it is already active
            return
        driver.set_active(backend.driver())
        return
    # This can be a hard error once triton-cpu is merged into fbcode
    warnings.warn(
        "Could not find an active CPU backend. Generated kernels will not be executable!"
    )


def _is_backend_active(name, backend):
    if backend.driver.is_active():
        return True
    # Triton may fail to detect the GPU in subprocess workers when using
    # ctypes-based driver detection (triton-lang/triton#9578). Fall back
    # to torch's own device checks which are more reliable in these environments.
    if name == "nvidia":
        import torch

        return torch.cuda.is_available() and torch.version.hip is None
    if name == "amd":
        import torch

        return torch.cuda.is_available() and torch.version.hip is not None
    return False


def set_driver_to_gpu():
    if _skip_gpu_driver_setup.get():
        return

    driver = triton.runtime.driver
    for name, backend in triton.backends.backends.items():
        if _is_backend_active(name, backend) and name != "cpu":
            # After https://github.com/triton-lang/triton/commit/b844d519bc5e86edf00fe6b3c6c2d1badcd509a4,
            # `driver.active` can be of `LazyProxy` type and the sign of this - `_obj` attribute.
            if (
                isinstance(driver.active, backend.driver)
                or hasattr(driver.active, "_obj")
                and isinstance(driver.active._obj, backend.driver)
            ):
                # Don't re-initialize backend if it is already active
                return
            driver.set_active(backend.driver())
            return
    raise RuntimeError("Could not find an active GPU backend")


def get_backend_options_for_target(target, options=None):
    options = {} if options is None else dict(options)
    backend = triton.compiler.compiler.make_backend(target)
    return backend.parse_options(options).__dict__


def get_backend_options():
    from triton.runtime import driver

    target = driver.active.get_current_target()
    return get_backend_options_for_target(target)


def _is_concrete_backend_option_value(value: Any) -> bool:
    import sympy

    import torch

    if isinstance(
        value,
        (
            torch.Tensor,
            torch.SymInt,
            torch.SymFloat,
            torch.SymBool,
            sympy.Expr,
        ),
    ):
        return False
    if isinstance(value, (tuple, list)):
        return all(_is_concrete_backend_option_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_concrete_backend_option_value(key)
            and _is_concrete_backend_option_value(item)
            for key, item in value.items()
        )
    return True


def try_filter_backend_options_for_target(target, options, kernel_arg_names=()):
    parsed_options = get_backend_options_for_target(target)
    kernel_arg_names = tuple(kernel_arg_names)
    filtered_options = {
        name: value for name, value in options.items() if name in parsed_options
    }
    invalid_options = [
        name
        for name in options
        if name not in parsed_options and name not in kernel_arg_names
    ]
    if invalid_options:
        raise RuntimeError(
            "Triton launch kwargs must be kernel parameters or valid backend options: "
            f"{sorted(invalid_options)!r}."
        )
    dynamic_options = [
        name
        for name, value in filtered_options.items()
        if not _is_concrete_backend_option_value(value)
    ]
    if dynamic_options:
        raise RuntimeError(
            "Triton backend options must be concrete values: "
            f"{sorted(dynamic_options)!r}."
        )
    return filtered_options


def get_constexprs(kernel: JITFunction) -> list[int]:
    return [p.num for p in kernel.params if p.is_constexpr]


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def fp8e4m3fn_to_float32(x):
    x_u32 = x.to(tl.uint32)
    sign = (x_u32 & 0x80) << 24
    exp = (x_u32 >> 3) & 0xF
    mant = x_u32 & 0x7

    normal_bits = sign | ((exp + 120) << 23) | (mant << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)

    subnormal_abs = mant.to(tl.float32) * 0.001953125
    subnormal_bits = subnormal_abs.to(tl.uint32, bitcast=True) | sign
    subnormal = subnormal_bits.to(tl.float32, bitcast=True)

    nan = (sign | 0x7FF00000).to(tl.float32, bitcast=True)
    result = tl.where(exp == 0, subnormal, normal)
    return tl.where((exp == 0xF) & (mant == 0x7), nan, result)


@triton.jit
def div_floor_integer(a, b):
    # NOTE: a // b is C division, but we want floor division
    # Based on c10::div_floor_integer
    quot = a // b
    remainder = a % b
    fixed = tl.where(remainder != 0, quot - 1, quot)
    return tl.where((a < 0) != (b < 0), fixed, quot)


@triton.jit
def remainder_integer(a, b):
    # NOTE: a % b matches C division, not floor division
    remainder = a % b
    return tl.where((remainder != 0) & ((a < 0) != (b < 0)), remainder + b, remainder)


@triton.jit
def pow_integer(base, exponent):
    # Triton has no exact integer pow primitive; use repeated squaring for
    # nonnegative integer exponents so integral scalar pow does not round
    # through libdevice.pow before casting back to int.
    exponent_dtype: tl.constexpr = tl.core.get_int_dtype(
        exponent.dtype.primitive_bitwidth, signed=False
    )
    exp = exponent.to(exponent_dtype)
    result = tl.full(base.shape, 1, base.dtype)
    for _ in tl.static_range(exponent_dtype.primitive_bitwidth):
        result = tl.where((exp & 1) != 0, result * base, result)
        exp = exp >> 1
        base = base * base
    return result


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def _prod_accumulate(a, b):
    return a * b


@triton.jit
def prod(input, axis):
    return tl.reduce(input, axis, _prod_accumulate)


@triton.jit
def minimum(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _minimum_reduce(a, b):
    value = minimum(a, b)
    if is_floating(a):
        value = tl.where(a == b, b, value)
    return value


@triton.jit
def _maximum_reduce(a, b):
    value = maximum(a, b)
    if is_floating(a):
        value = tl.where(a == b, b, value)
    return value


@triton.jit
def fmaximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def min2(a, dim):
    return tl.reduce(a, dim, minimum)


@triton.jit
def max2(a, dim):
    return tl.reduce(a, dim, maximum)


@triton.jit
def min2_strict(a, dim):
    return tl.reduce(a, dim, _minimum_reduce)


@triton.jit
def max2_strict(a, dim):
    return tl.reduce(a, dim, _maximum_reduce)


@triton.jit
def fmax2(a, dim):
    return tl.reduce(a, dim, fmaximum)


@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        # Consider NaNs as equal
        equal |= a_isnan & b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        # Consider NaNs as equal
        equal |= a_isnan & b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def min_with_index(value, index, dim):
    return tl.reduce((value, index), dim, minimum_with_index)


@triton.jit
def max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, maximum_with_index)


@triton.jit
def exp(x, use_fast_math: tl.constexpr):
    if use_fast_math:
        return math.exp(x)
    else:
        return libdevice.exp(x)


@triton.jit
def online_softmax_reduce(
    lhs_max,
    lhs_sum,
    dim,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    if strict_signed_zero:
        out_max = max2_strict(lhs_max, dim)
    else:
        out_max = max2(lhs_max, dim)
    out_max_keepdim = tl.expand_dims(out_max, dim)
    delta = tl.where(out_max_keepdim == float("-inf"), 0, lhs_max - out_max_keepdim)
    out_sum = tl.sum(lhs_sum * exp(delta, use_fast_math), dim)
    return out_max, out_sum


@triton.jit
def online_softmax_combine(
    lhs_max,
    lhs_sum,
    rhs_max,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    """
    When we do combine, we assume lhs is the accumulator and rhs is the next
    block of data.
    Then rhs_sum is always 1. With that assumption, we can save some registers
    and computation.
    """
    if strict_signed_zero:
        out_max = _maximum_reduce(lhs_max, rhs_max)
    else:
        out_max = maximum(lhs_max, rhs_max)

    lhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(lhs_max - out_max, use_fast_math)
    )
    rhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(rhs_max - out_max, use_fast_math)
    )

    # Should be
    #   out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
    # but since rhs_sum is all 1, we can simplify it.
    out_sum = lhs_sum * lhs_scale + rhs_scale
    return out_max, out_sum


@triton.jit
def online_softmax_combine_with_sum(
    lhs_max,
    lhs_sum,
    rhs_max,
    rhs_sum,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    if strict_signed_zero:
        out_max = _maximum_reduce(lhs_max, rhs_max)
    else:
        out_max = maximum(lhs_max, rhs_max)

    lhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(lhs_max - out_max, use_fast_math)
    )
    rhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(rhs_max - out_max, use_fast_math)
    )

    out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
    return out_max, out_sum


@triton.jit
def welford_reduce(value, mean, m2, weight, first_iteration):
    if first_iteration:
        new_weight = tl.full(weight.shape, 1, weight.dtype)
        new_mean = value
        new_m2 = tl.zeros_like(m2)
    else:
        delta = value - mean
        new_weight = weight + 1
        new_mean = mean + delta / new_weight
        new_m2 = m2 + delta * (value - new_mean)
    return new_mean, new_m2, new_weight


@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    # Guard against inf - inf = NaN when both means are infinite and equal.
    # This occurs during FP16/BF16 LayerNorm when inputs overflow to inf.
    delta = tl.where(mean_1 == mean_2, 0.0, mean_2 - mean_1)
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


@triton.jit
def welford(mean, m2, weight, dim):
    return tl.reduce((mean, m2, weight), dim, welford_combine)


@triton.jit
def device_assert_then(cond, msg, r):
    tl.device_assert(cond, msg)
    return r


@triton.jit
def rand_eager_kernel(seed, offset_blocks, tid: tl.tensor, VEC: tl.constexpr):
    inv = 1.0 / 4294967296.0
    half = inv * 0.5

    tid_u64 = tid.to(tl.uint64)

    subseq = tid_u64 // VEC
    which4 = (tid_u64 % VEC) // 4
    lane = tid_u64 % 4

    offblk = offset_blocks.to(tl.uint64) + which4

    u0, u1, u2, u3 = tl.philox(
        seed,
        (offblk & 0xFFFFFFFF).to(tl.uint32),
        ((offblk >> 32) & 0xFFFFFFFF).to(tl.uint32),
        (subseq & 0xFFFFFFFF).to(tl.uint32),
        ((subseq >> 32) & 0xFFFFFFFF).to(tl.uint32),
    )

    v01 = tl.where(lane == 0, u0, u1)
    v23 = tl.where(lane == 2, u2, u3)
    rand_int = tl.where((lane == 0) | (lane == 1), v01, v23)

    return 1.0 - (rand_int.to(tl.float32) * inv + half)


@triton.jit
def _random_4x_to_block(r0, r1, r2, r3):
    # Pack lanes by logical offset so the random stream is independent of XBLOCK.
    size: tl.constexpr = r0.numel
    return tl.reshape(tl.join(tl.join(r0, r2), tl.join(r1, r3)), [size * 4])


@triton.jit
def rand4x(seed, offsets, BLOCK: tl.constexpr):
    offsets = offsets.to(tl.uint32)
    seed = tl.min(seed + offsets * 0, axis=0)
    if BLOCK >= 4 and BLOCK % 4 == 0:
        base = tl.min(offsets, axis=0)
        reduced_offsets = base // 4 + tl.arange(0, BLOCK // 4)
        r0, r1, r2, r3 = tl.rand4x(seed, reduced_offsets)
        return _random_4x_to_block(r0, r1, r2, r3)
    return tl.rand(seed, offsets)


@triton.jit
def randn4x(seed, offsets, BLOCK: tl.constexpr):
    offsets = offsets.to(tl.uint32)
    seed = tl.min(seed + offsets * 0, axis=0)
    if BLOCK >= 4 and BLOCK % 4 == 0:
        base = tl.min(offsets, axis=0)
        reduced_offsets = base // 4 + tl.arange(0, BLOCK // 4)
        r0, r1, r2, r3 = tl.randn4x(seed, reduced_offsets)
        return _random_4x_to_block(r0, r1, r2, r3)
    return tl.randn(seed, offsets)


@triton.jit
def randint64(seed, offset, low, high):
    r0, r1, _r2, _r3 = tl.randint4x(seed, offset)
    r0 = r0.to(tl.uint64)
    r1 = r1.to(tl.uint64)
    result = r0 | (r1 << 32)
    size = high - low
    result = result % size.to(tl.uint64)
    result = result.to(tl.int64) + low
    return result


@triton.jit
def _any_combine(a, b):
    return a | b


@triton.jit
def any(a, dim):
    return tl.reduce(a, dim, _any_combine)


@triton.jit
def bucketize_binary_search(
    values: tl.tensor,
    boundaries_ptr: tl.tensor,
    BOUNDARIES_SIZE: int,
    BOUNDARIES_UNDERLYING_NUMEL: int,
    BOUNDARIES_STRIDE: int,
    boundary_indices: tl.tensor,
    indexing_dtype: tl.dtype,
    right: "bool",  # triton can't handle the unquoted bool annotation
    sorter_ptr: tl.tensor,
    SORTER_STRIDE: int,
    sorter_indices: tl.tensor,
):
    """
    See [Note: Inductor bucketize op]

    Inputs:
    -------
    values: the values to bucketize.
    boundaries_ptr: a pointer to the beginning of the boundaries tensor, in 1-D.
    BOUNDARIES_SIZE: the length of the last dimension of the boundaries tensor (i.e. one
    individual set of boundaries).
    BOUNDARIES_UNDERLYING_NUMEL: the length of the boundaries tensor, in 1-D, ignoring
    any striding.
    BOUNDARIES_STRIDE: the stride of the last dimension of the boundaries tensor
    boundary_indices: a tensor of the same size as "values"; each element is an index
    into a 1-D, un-strided boundaries tensor, pointing to the first element in the set
    of boundaries used for that value.
    indexing_dtype: the dtype used for indexing into the boundaries tensor, and the
    return dtype.
    right: if true, use boundary intervals closed on the left; otherwise use intervals
    closed on the right.
    sorter_ptr: an optional pointer to a sorter tensor of the same shape as boundaries,
    but potentially different striding.  If present, this allows us to treat boundaries
    as sorted even if the elements of boundaries are unsorted.
    SORTER_STRIDE: must be present if sorter_ptr is non-None; the stride of the last
    dimension of the sorter tensor.
    sorter_indices: must be present if sorter_ptr is non-None; see "boundary_indices".
    BLOCK_SHAPE: the shape of the data block being processed.
    """

    low = tl.zeros(values.shape, dtype=indexing_dtype)
    high = tl.full(values.shape, BOUNDARIES_SIZE, dtype=indexing_dtype)

    full_range = BOUNDARIES_SIZE + 1
    while full_range > 1:
        mid = (high + low) // 2
        mask = (
            (mid * BOUNDARIES_STRIDE + boundary_indices) < BOUNDARIES_UNDERLYING_NUMEL
        ).logical_and(mid < BOUNDARIES_SIZE)
        mid_indices = (
            mid
            if sorter_ptr is None or SORTER_STRIDE is None
            else tl.load(
                sorter_ptr + sorter_indices + SORTER_STRIDE * mid,
                mask=mask,
                other=0,
            )
        )

        bucket_upper_bound = tl.load(
            boundaries_ptr + boundary_indices + BOUNDARIES_STRIDE * mid_indices,
            mask=mask,
            other=0,
        )
        if right:
            is_above = values >= bucket_upper_bound
        else:
            is_above = values > bucket_upper_bound

        if is_floating(values):
            is_above = is_above | (values != values)

        low = tl.where(is_above & mask, mid + 1, low)
        high = tl.where(is_above, high, mid)

        full_range = (full_range + 1) // 2

    return low


@triton.jit
def pack_value_flag(
    value,
    flag,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    # Workaround for triton bug, tensor.to doesn't unwrap constexpr values
    DTYPE_VALUE_AS_UINT = tl.core._unwrap_if_constexpr(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    uv = value.to(DTYPE_VALUE_AS_UINT, bitcast=True).to(DTYPE_PACK)
    return flag.to(DTYPE_PACK) | (uv << bitwidth)


@triton.jit
def unpack_value(
    pack,
    DTYPE_VALUE,
    DTYPE_VALUE_AS_UINT,
):
    # Workaround for triton bug, tensor.to doesn't unwrap constexpr values
    DTYPE_VALUE = tl.core._unwrap_if_constexpr(DTYPE_VALUE)
    DTYPE_VALUE_AS_UINT = tl.core._unwrap_if_constexpr(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    value_uint = (pack >> bitwidth).to(DTYPE_VALUE_AS_UINT)
    return value_uint.to(DTYPE_VALUE, bitcast=True)


@triton.jit
def unpack_flag(pack, DTYPE_FLAG):
    return pack.to(DTYPE_FLAG)


@triton.jit
def exclusive_scan_decoupled_lookback(
    scratch_base,
    block_value,
    index,
    combine_fn,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    DTYPE_VALUE_AS_UINT: A tl.uint{n} type equal in size to ``block_value``
    DTYPE_PACK: Unsigned type twice the width of block_value

    NOTE: This function is limited to values which are 32-bits or less because
    we need to pack (value, flag) into a single unsigned int.
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us
    DTYPE_VALUE = block_value.dtype
    pack = pack_value_flag(
        block_value,
        tl.full(block_value.shape, 1, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )
    if index > 0:
        tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")

    # Calculate exclusive prefix scan
    exclusive_prefix = tl.zeros([], DTYPE_VALUE)
    prefix_valid = False
    test_target = index - 1
    while test_target >= 0:
        # tl.atomic_load
        flag = tl.full([], 0, DTYPE_VALUE_AS_UINT)
        while flag == 0:
            pack = tl.atomic_add(scratch_base + test_target, 0, sem="relaxed")
            flag = unpack_flag(pack, DTYPE_VALUE_AS_UINT)

        value = unpack_value(pack, DTYPE_VALUE, DTYPE_VALUE_AS_UINT)
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        if flag == 2:
            test_target = -1
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    pack = pack_value_flag(
        inclusive_prefix,
        tl.full([], 2, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )
    tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")
    return exclusive_prefix


@triton.jit
def exclusive_scan_decoupled_lookback_64(scratch_base, block_value, index, combine_fn):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block, must be 64-bits wide
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    init: Scalar value equal to the identity of combine_fn
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us
    if index > 0:
        block_value_u64 = block_value.to(tl.uint64, bitcast=True)
        tl.store(scratch_base + 3 * index + 1, block_value_u64)
        tl.debug_barrier()
        flag_one = tl.full([], 1, tl.uint64)
        tl.atomic_xchg(scratch_base + 3 * index + 0, flag_one, sem="release")

    # Calculate exclusive prefix scan
    exclusive_prefix = tl.zeros([], block_value.dtype)
    prefix_valid = False
    test_target = index - 1
    while test_target >= 0:
        flag = tl.full([], 0, tl.uint64)
        while flag == 0:
            flag = tl.atomic_add(scratch_base + 3 * test_target + 0, 0, sem="acquire")

        value_u64 = tl.load(scratch_base + 3 * test_target + flag.to(tl.int32))
        value = value_u64.to(block_value.dtype, bitcast=True)
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        if flag == 2:
            test_target = tl.full([], -1, index.dtype)  # Match the original type
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    inclusive_prefix_u64 = inclusive_prefix.to(tl.uint64, bitcast=True)
    tl.store(scratch_base + 3 * index + 2, inclusive_prefix_u64)
    tl.debug_barrier()
    flag_two = tl.full([], 2, tl.uint64)
    tl.atomic_xchg(scratch_base + 3 * index + 0, flag_two, sem="release")

    return exclusive_prefix


@triton.jit
def frexp(x):
    # TODO(isuruf): use inline_asm_elementwise here
    zero = x == 0
    not_finite = libdevice.isinf(x).to(tl.int1) | libdevice.isnan(x).to(tl.int1)
    special = zero | not_finite
    safe_x = tl.where(special, 1.0, x)
    y = libdevice.ilogb(safe_x) + 1
    exponent = tl.where(special, 0, y)
    mantissa = tl.where(zero, 0, tl.where(not_finite, x, libdevice.ldexp(safe_x, -y)))
    return mantissa, exponent


@triton.jit
def _compare_and_swap_with_index(
    x,
    idxs,
    rnumel,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
    stable: tl.constexpr,
    descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    y = tl.reshape(x, shape)
    iy = y.to(idtype, bitcast=True)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    right_mask = tl.arange(0, 2)[None, :, None].to(idtype)
    left_mask = (1 - right_mask).to(idtype)
    ileft = tl.broadcast_to(tl.sum(iy * left_mask, 1).to(idtype)[:, None, :], shape)
    iright = tl.broadcast_to(tl.sum(iy * right_mask, 1).to(idtype)[:, None, :], shape)
    ileft = tl.reshape(ileft, x.shape)
    iright = tl.reshape(iright, x.shape)
    left = ileft.to(x.dtype, bitcast=True)
    right = iright.to(x.dtype, bitcast=True)

    # idx
    y_idx = tl.reshape(idxs, shape)
    left_idx = tl.broadcast_to(
        tl.sum(y_idx * left_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    right_idx = tl.broadcast_to(
        tl.sum(y_idx * right_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    left_idx = tl.reshape(left_idx, x.shape)
    right_idx = tl.reshape(right_idx, x.shape)

    # valid
    if rnumel is None:
        left_valid_mask = tl.full(x.shape, True, tl.int1)
        right_valid_mask = tl.full(x.shape, True, tl.int1)
    else:
        left_valid_mask = left_idx < rnumel
        right_valid_mask = right_idx < rnumel

    # actual compare-and-swap
    ix = x.to(idtype, bitcast=True)

    # sort treats nan as having the higher value. comparisons with nan always return False.
    # to align with sort semantics, we need to update descending to check if right_isnan,
    # and ascending to check if left_isnan.
    left_isnan = left != left
    right_isnan = right != right

    if descending:
        cond = left < right
        if is_floating(left):
            if not stable:
                cond = cond | right_isnan
            else:
                cond = cond | (right_isnan & (~left_isnan))

    else:
        cond = left > right
        if is_floating(left):
            if not stable:
                cond = cond | left_isnan
            else:
                cond = cond | (left_isnan & (~right_isnan))

    if stable:
        # When stable sorting, tie break by index
        eq = left == right
        if is_floating(left):
            eq = eq | (left_isnan & right_isnan)
        cond = cond | (eq & (left_idx > right_idx))

    cond = (right_valid_mask > left_valid_mask) | (
        (right_valid_mask == left_valid_mask) & cond
    )
    cond = (cond ^ flip).to(tl.int1)
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_idxs = idxs ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(idxs))

    return ret.to(x.dtype, bitcast=True), new_idxs


@triton.jit
def _bitonic_merge_with_index(
    x,
    idxs,
    rnumel,
    stage: tl.constexpr,
    alternating: tl.constexpr,
    n_dims: tl.constexpr,
    stable: tl.constexpr,
    descending: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if alternating:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = False
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, idxs = _compare_and_swap_with_index(
            x, idxs, rnumel, flip, i + (n_dims - stage), n_dims, stable, descending
        )
    return x, idxs


@triton.jit
def sort_with_index(
    x,  # value
    idxs,  # index
    rnumel,  # number of elements
    dim: tl.constexpr = None,
    stable: tl.constexpr = tl.constexpr(False),
    descending: tl.constexpr = tl.constexpr(False),
):
    x, idxs = tl.broadcast(x, idxs)
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(
        _dim == len(x.shape) - 1, "only minor dimension is currently supported"
    )
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = _log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, idxs = _bitonic_merge_with_index(
            x,
            idxs,
            rnumel,
            i,
            alternating=i < n_dims,
            n_dims=n_dims,
            stable=stable,
            descending=descending,
        )
    return x, idxs


@triton.jit
def select_one(x, mask, dim, keep_dims=False):
    idtype = tl.core.get_int_dtype(x.dtype.primitive_bitwidth, signed=False)
    ix = x.to(idtype, bitcast=True)
    iy = tl.sum(ix * mask, dim, keep_dims=keep_dims)
    return iy.to(idtype).to(x.dtype, bitcast=True)


@triton.jit
def x_grid_barrier(sem):
    """
    Wait for all other thread blocks in grid sharing same y/z program_id
    to reach this barrier before returning.

    Args:
        sem: an uint32 semaphores, zero or 0x80000000 initialized.  Must be unique to each y/z program ID.
    """
    # ensure stores before this are visible
    tl.debug_barrier()

    one_i32 = 1
    one_u32 = one_i32.to(tl.uint32)  # type: ignore[attr-defined]
    expected = tl.num_programs(0).to(tl.uint32)
    if tl.program_id(0) == 0:
        nb = 0x80000000 - (expected - one_u32)
    else:
        nb = one_u32

    old_arrive = tl.atomic_add(sem, nb, sem="release")

    bar_flipped = False
    while not bar_flipped:
        # want a `ld.acquire.gpu.u32 $0,[$1];` but Triton doesn't have it
        current_arrive = tl.atomic_add(sem, 0, sem="acquire")
        # current_arrive = tl.load(sem, volatile=True)
        bar_flipped = ((old_arrive ^ current_arrive) & 0x80000000) != 0

    # TODO(jansel): is this needed?
    tl.debug_barrier()


def triton_builtin(f: Callable[..., _T]) -> Callable[..., _T]:
    """
    Decorator to mark a function as a Triton built-in function.  These functions
    are evaluated at compile time.

    Args:
        f (function): The function to be marked as a Triton built-in.

    Returns:
        function: The same function, marked as a Triton built-in.
    """
    if builtins_use_semantic_kwarg:
        # support Triton before and after https://github.com/triton-lang/triton/pull/7054
        # and after https://github.com/triton-lang/triton/pull/7239
        def wrapper(*args, _semantic, **kwargs):
            kwargs["_builder"] = _semantic
            return f(*args, **kwargs)
    else:
        wrapper = f  # type: ignore[assignment]

    wrapper.__triton_builtin__ = True  # type: ignore[attr-defined]
    return wrapper


@triton_builtin
def constexpr_next_power_of_2(
    n: tl.constexpr, *, _builder: object = None
) -> tl.constexpr:
    """
    A version triton.next_power_of_two that can be used within a kernel on constants.
    """
    if not isinstance(n, tl.constexpr):
        raise AssertionError(f"Expected tl.constexpr, got {type(n)}")
    return tl.constexpr(triton.next_power_of_2(n.value))


@triton_builtin
def if_mask(mask: Any, val, *, _builder: object = None) -> tl.constexpr:
    """
    Work around triton compile error: `ValueError: `other` cannot be provided without `mask``
    A compile-time to check to return either `val` or `None` depending on the value of mask.
    """
    if isinstance(mask, tl.constexpr) and mask.value is None:
        return tl.constexpr(None)
    return val


@triton.jit
def inline_asm_pack(x, pack: tl.constexpr):
    """Ravel to 1D and pad (via join with zeros) so numel is divisible by pack."""
    result = x.ravel()
    # Only pad when the block size is smaller than pack. When block >= pack
    # the numel is already divisible by pack (both are powers of 2).
    n_pad: tl.constexpr = _log2(pack) - _log2(result.numel)
    for _ in tl.static_range(n_pad):
        result = tl.reshape(
            tl.join(result, tl.zeros_like(result)), (result.shape[0] * 2,)
        )
    return result


@triton.jit
def inline_asm_unpack(x, orig, pack: tl.constexpr):
    """Unpad and reshape back to orig's shape."""
    result = x
    n_pad: tl.constexpr = _log2(pack) - _log2(orig.numel)
    for _ in tl.static_range(n_pad):
        result, _ = tl.split(tl.reshape(result, (result.shape[0] // 2, 2)))
    return tl.reshape(result, orig.shape)
<|endoftext|># mypy: allow-untyped-defs
import copy
import itertools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from ..utils import get_max_numwarps
from .hints import (
    InductorMeta,
    native_matmul_block_numel,
    native_matmul_persistent_rblock,
    TRITON_MAX_BLOCK,
    TRITON_MAX_TENSOR_NUMEL,
)
from .runtime_utils import red_text, triton_config_to_hashable


if TYPE_CHECKING:
    from .triton_compat import triton


log = logging.getLogger(__name__)


def get_field(config, name):
    if name == "num_warps":
        return config.num_warps
    elif name == "num_stages":
        return config.num_stages
    elif name == "waves_per_eu":
        return config.kwargs.get(name, int(8 // config.num_warps))
    else:
        return config.kwargs.get(name, None)


def set_field(config, name, value):
    if name == "num_warps":
        config.num_warps = value
    elif name == "num_stages":
        config.num_stages = value
    else:
        config.kwargs[name] = value


class CoordescTuner:
    """
    The coordinate descent tuner. Tune one field/coordinate at a time.

    TODO will it be necessary to tune multiple fields simultaneously.


    TODO: what if both increasing and decreasing a field can improve perf.
          i.e., there are multiple local optima..
    """

    def __init__(
        self,
        is_mm=False,
        is_native_matmul=False,
        is_mix_order_reduction=False,
        name="unknown",
        size_hints=None,
        inductor_meta: InductorMeta | None = None,
        frozen_fields=None,
    ):
        self.is_mm = is_mm  # we will tune num_stages for mm

        # Native matmul codegen assumes ZBLOCK=1 always.
        # This is because 3d tl.dot is slow and so we want to tile y and x only.
        # tl.dot also does not support size smaller than 16; we put this restriction.
        self.is_native_matmul = is_native_matmul
        if self.is_mm and self.is_native_matmul:
            raise AssertionError("is_mm and is_native_matmul are mutually exclusive")
        self.is_mix_order_reduction = is_mix_order_reduction
        self.cached_benchmark_results = {}
        self.name = name
        self.size_hints = size_hints
        self.inductor_meta: InductorMeta = (
            inductor_meta if inductor_meta is not None else {}
        )
        self.frozen_fields: OrderedSet[str] = (
            OrderedSet(frozen_fields) if frozen_fields is not None else OrderedSet()
        )

    def get_config_max(self, prefix: str) -> int:
        max_block = TRITON_MAX_BLOCK[prefix.upper()]
        size_hint = self.size_hints.get(prefix) if self.size_hints is not None else None
        return min(max_block, size_hint) if size_hint is not None else max_block

    def get_warpsmax(self):
        # Avoid querying device directly if device properties are populated in inductor_meta
        warp_size = self.inductor_meta.get("warp_size")
        max_threads_per_block = self.inductor_meta.get("max_threads_per_block")
        if warp_size and max_threads_per_block:
            return max_threads_per_block // warp_size
        else:
            return get_max_numwarps()

    def cache_benchmark_result(self, config, timing):
        self.cached_benchmark_results[triton_config_to_hashable(config)] = timing

    def lookup_in_cache(self, config):
        return self.cached_benchmark_results.get(triton_config_to_hashable(config))

    def call_func(self, func, config):
        found = self.lookup_in_cache(config)
        if found is not None:
            log.debug("  CACHED")
            return found
        timing = func(config)
        self.cache_benchmark_result(config, timing)
        return timing

    @property
    def tunable_fields(self) -> list[str]:
        out: list[str] = [
            "XBLOCK",
            "YBLOCK",
            "ZBLOCK",
            # NOTE: we should not tune R0_BLOCK for persistent reduction.
            # We rely on the fact that persistent reduction's triton.Config
            # does not have the R0_BLOCK field to guarantee that.
            "R0_BLOCK",
            "R1_BLOCK",
            # the following 3 are for mm
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
        ]
        if self.is_mm:
            out.append("num_stages")
        if self.inductor_meta.get("is_hip") is True:
            out.append("waves_per_eu")
        if self.is_native_matmul:
            out.append("num_stages")
            out.remove("ZBLOCK")  # ZBLOCK=1 always in native matmul

        if self.is_mix_order_reduction:
            # unlike TritonConfig.num_stages, this one is
            # put in TritonConfig.kwargs["NUM_STAGES"] and is used to
            # control the stage of pipelining of tl.range.
            out.append("NUM_STAGES")

        # Combo-kernel per-subkernel block fields (e.g. XBLOCK_0/1, YBLOCK_0/1)
        # come from the worker-side metadata in ``inductor_meta``. Prepend
        # them so coordesc iterates them alongside the base fields. Read
        # live each call so any post-construction mutation of
        # ``inductor_meta`` is observed.
        combo_fields: list[str] = self.inductor_meta.get(
            "combo_coordesc_field_order", []
        )
        out = combo_fields + out
        return [f for f in out if f not in self.frozen_fields]

    def value_too_large(self, name: str, val: int) -> bool:
        field_limits = self.inductor_meta.get("combo_coordesc_field_limits")
        if isinstance(field_limits, dict) and name in field_limits:
            return val > field_limits[name]

        block_suffix = "BLOCK"
        if name.endswith(block_suffix):
            prefix = name.strip(block_suffix).lower()
            return val > self.get_config_max(prefix)
        if name == "num_warps":
            return val > self.get_warpsmax()
        if name == "waves_per_eu":
            return val > 8

        return False

    def value_too_small(self, name: str, val: int) -> bool:
        min_block = None
        if name == "XBLOCK":
            min_block = self.inductor_meta.get("min_xblock")
        elif name == "R0_BLOCK":
            min_block = self.inductor_meta.get("min_rblock")
        if min_block is not None and val < min_block:
            return True

        # In native matmul, block size should be >= 16 for tl.dot
        if self.is_native_matmul:
            if name in ["YBLOCK", "XBLOCK", "R0_BLOCK"]:
                return val < 16

        # Break if value becomes 0/neg
        return val <= 0

    def get_neighbour_values(self, name, orig_val, radius=None, include_self=False):
        """
        Get neighbour values in 'radius' steps. The original value is not
        returned as its own neighbour.
        """
        if radius is None:
            radius = 1
        if name == "NUM_STAGES":
            # we see cases that
            # NUM_STAGES=1 is better than NUM_STAGES=2
            # while NUM_STAGES=1 is worse than NUM_STAGES=3
            radius = max(radius, 2)

        if radius < 1:
            raise AssertionError(f"radius must be >= 1, got {radius}")

        def update(cur_val, inc=True):
            if name in ["num_stages", "NUM_STAGES"]:
                if inc:
                    return cur_val + 1
                else:
                    return cur_val - 1
            else:
                if inc:
                    return cur_val * 2
                else:
                    return cur_val // 2

        out = []
        # increment loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, True)
            if self.value_too_large(name, cur_val):
                break
            out.append(cur_val)

        # decrement loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, False)
            if self.value_too_small(name, cur_val):
                break
            out.append(cur_val)

        if include_self:
            out.append(orig_val)
        return out

    @staticmethod
    def has_improvement(baseline, test):
        threshold = 0.001  # 0.1%
        return test is not None and test < baseline * (1 - threshold)

    def is_valid_config(self, config) -> bool:
        if self.is_mix_order_reduction:
            # Mix order reduction has an extra constraint that
            # we should not tune XBLOCK beyond RSPLIT_SIZE
            xblock = config.kwargs["XBLOCK"]
            split_size = config.kwargs["RSPLIT_SIZE"]
            return xblock <= split_size
        if self.is_native_matmul:
            r0_block = None
            if "R0_BLOCK" not in config.kwargs:
                r0_block = self.inductor_meta.get("native_matmul_persistent_rblock")
                if r0_block is None and self.size_hints is not None:
                    r0_block_hint = self.size_hints.get("r0_")
                    if r0_block_hint is not None:
                        r0_block = native_matmul_persistent_rblock(r0_block_hint)
            if (
                native_matmul_block_numel(config.kwargs, r0_block=r0_block)
                > TRITON_MAX_TENSOR_NUMEL
            ):
                return False
        return True

    def get_all_tuning_directions(
        self,
        # pyrefly: ignore [missing-attribute]
        config: "triton.Config",
    ) -> list["triton.Config"]:  # pyrefly: ignore [missing-attribute]
        """Return the Cartesian product of neighbour values across every
        tunable field, as a list of valid candidate configs."""
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = get_field(config, field)
            if old_value is None:
                continue
            radius = self.inductor_meta.get("coordinate_descent_search_radius", 1)
            candidate_values = self.get_neighbour_values(
                field,
                old_value,
                radius=radius,
                include_self=True,
            )
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)

        configs = []
        for choice in itertools.product(*candidate_values_list):
            if len(choice) != len(effective_fields):
                raise AssertionError(
                    f"Expected {len(effective_fields)} values, got {len(choice)}"
                )
            candidate = copy.deepcopy(config)
            for new_val, field in zip(choice, effective_fields):
                set_field(candidate, field, new_val)
            if self.is_valid_config(candidate):
                configs.append(candidate)
        return configs

    def check_all_tuning_directions(
        self,
        # pyrefly: ignore [missing-attribute]
        func: Callable[["triton.Config"], float],
        best_config,
        best_timing,
    ):
        """
        Check all directions. We only do this once the regular coordinate
        descent tuning find no better choices any more.
        We only have a few tunable fields, so this should be fine.
        """
        improved = False
        for candidate_config in self.get_all_tuning_directions(best_config):
            cmp_res, candidate_timing = self.compare_config(
                func, candidate_config, best_config, best_timing
            )
            if cmp_res:
                improved = True
                best_config = candidate_config
                best_timing = candidate_timing

        return improved, best_config, best_timing

    def compare_config(self, func, candidate_config, best_config, best_timing):
        """
        Check if candidate_config is better than best_config.

        Return a tuple of (compare_result, candidate_timing).
        compare_result is true iff candidate_config is better.
        """
        log.debug("Try config %s", candidate_config)
        try:
            candidate_timing = self.call_func(func, candidate_config)
        except Exception as e:
            log.debug("Got exception %s", e)
            return False, float("inf")

        if self.has_improvement(best_timing, candidate_timing):
            log.debug(
                "Tune from %s %f -> %s %f",
                best_config,
                best_timing,
                candidate_config,
                candidate_timing,
            )

            return True, candidate_timing
        return False, candidate_timing

    def get_neighbour_configs(
        self,
        # pyrefly: ignore [missing-attribute]
        config: "triton.Config",
        field: str,
    ) -> list["triton.Config"]:  # pyrefly: ignore [missing-attribute]
        """Return every valid neighbour of ``config`` along ``field``
        only — the per-axis counterpart of ``get_neighbour_values``.

        Precondition: ``field`` must be present on ``config``. Callers
        iterating ``tunable_fields`` against arbitrary configs must
        filter empty fields up front; ``autotune`` does."""
        cur_val = get_field(config, field)
        if cur_val is None:
            raise AssertionError(f"field {field!r} not present on config")
        neighbours = []
        for next_val in self.get_neighbour_values(field, cur_val):
            candidate = copy.deepcopy(config)
            set_field(candidate, field, next_val)
            if self.is_valid_config(candidate):
                neighbours.append(candidate)
        return neighbours

    def autotune(
        self,
        # pyrefly: ignore [missing-attribute]
        func: Callable[["triton.Config"], float],
        # pyrefly: ignore [missing-attribute]
        baseline_config: "triton.Config",
        baseline_timing: float | None = None,
    ) -> "triton.Config":  # pyrefly: ignore  # missing-attribute
        """
        Perform coordinate descent autotuning starting from a baseline configuration.
        """
        if baseline_timing is None:
            baseline_timing = self.call_func(func, baseline_config)

        log.debug("= Do coordinate descent tuning for %s =", self.name)
        log.debug(
            "%s: Baseline Config %s, baseline timing %f",
            self.name,
            baseline_config,
            baseline_timing,
        )
        improved = True
        best_config = baseline_config
        best_timing = baseline_timing

        while improved:
            improved = False

            # Gauss-Seidel: walk one field at a time, applying any
            # improvement immediately so subsequent fields see the
            # updated working point.
            for field in self.tunable_fields:
                # Some kernels don't have R0_BLOCK / YBLOCK / ZBLOCK,
                # so the field may be absent from ``best_config``.
                # Filter here so ``get_neighbour_configs`` can assume
                # the field is present.
                if get_field(best_config, field) is None:
                    continue
                for candidate_config in self.get_neighbour_configs(best_config, field):
                    cmp_res, candidate_timing = self.compare_config(
                        func, candidate_config, best_config, best_timing
                    )
                    if cmp_res:
                        improved = True
                        best_config, best_timing = candidate_config, candidate_timing

            if not improved and self.inductor_meta.get(
                "coordinate_descent_check_all_directions"
            ):
                old_best_timing = best_timing
                improved, best_config, best_timing = self.check_all_tuning_directions(
                    func, best_config, best_timing
                )

                if improved:
                    msg = red_text(
                        "%s: Coordinate descend tuning found improvement of %.3fx by looking in all directions."
                    )
                    log.debug(
                        msg,
                        self.name,
                        old_best_timing / best_timing,
                    )

        log.debug(
            "%s: Improve from %s %f -> %s %f, %.3fx",
            self.name,
            baseline_config,
            baseline_timing,
            best_config,
            best_timing,
            baseline_timing / best_timing,
        )

        return best_config

    @staticmethod
    def autotune_single_field(fn, init_val, min_val=None, max_val=None):
        """
        fn is a function that takes the field value and returns the benchmarking result
        init_val is the starting point of autotuning.

        Should work well for parabola like curve. Here is a real example
        for split-size of mix-order-reduction: https://github.com/pytorch/pytorch/pull/166461
        """
        cache = {}

        def _bench(val):
            if val not in cache:
                cache[val] = fn(val)
                # print(f"split size {val} -> {cache[val]:.3f} ms")
            return cache[val]

        if min_val is None:
            min_val = 1
        if max_val is None:
            max_val = 2**30  # some arbitrary large value

        best_val = init_val
        improved = True
        while improved:
            improved = False
            candlist = [best_val // 2, best_val * 2]
            for cand in candlist:
                cand = max(cand, min_val)
                cand = min(cand, max_val)

                if _bench(cand) < _bench(best_val):
                    best_val = cand
                    improved = True

        return best_val
<|endoftext|>import contextlib
import functools
import inspect
import time
from collections.abc import Callable, Iterator
from functools import cached_property, wraps
from itertools import chain
from statistics import median
from typing import Any, Concatenate
from typing_extensions import ParamSpec, Self, TypeVar

import torch
import torch._inductor.config as inductor_config
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import counters
from torch.utils._debug_mode import DebugMode
from torch.utils._ordered_set import OrderedSet


logger = torch._logging.getArtifactLogger(__name__, "benchmarking")
GPU_BENCHMARK_DEVICE_TYPES = ("cuda", "xpu", "mtia")
_CALLABLE_PROFILE_EVENT_NAME = "_CALLABLE"


MILLISECONDS_PER_SECOND = 1000

P = ParamSpec("P")
T = TypeVar("T")


def _get_default_gpu_device_type() -> str:
    avail_gpus = [
        device_type
        for device_type in GPU_BENCHMARK_DEVICE_TYPES
        if getattr(torch, device_type).is_available()
    ]
    if len(avail_gpus) > 1:
        raise AssertionError(
            f"expected at most one available GPU type, got {avail_gpus}"
        )
    return "cuda" if len(avail_gpus) == 0 else avail_gpus.pop()


def _normalize_gpu_device_type(device_type: str | torch.device | None) -> str:
    if device_type is None:
        return _get_default_gpu_device_type()
    if isinstance(device_type, torch.device):
        return device_type.type
    return torch.device(device_type).type


_GpuBenchmarkLockContext = Callable[[], contextlib.AbstractContextManager[None]]
_gpu_benchmark_lock_context: _GpuBenchmarkLockContext | None = None


def set_gpu_benchmark_lock_context(
    context_factory: _GpuBenchmarkLockContext | None,
) -> _GpuBenchmarkLockContext | None:
    """Override the process-local GPU benchmark lock context.

    This lets benchmark harnesses provide the context used by Inductor GPU
    benchmark calls. Some benchmark helpers delegate to other benchmark
    methods, so harness contexts should support nested entry from the same
    thread. Returning the previous context lets callers restore it in tests.
    """
    global _gpu_benchmark_lock_context
    previous = _gpu_benchmark_lock_context
    _gpu_benchmark_lock_context = context_factory
    return previous


@contextlib.contextmanager
def maybe_gpu_benchmark_lock() -> Iterator[None]:
    """Optionally enter the registered GPU benchmark lock context."""
    context_factory = _gpu_benchmark_lock_context
    if context_factory is None:
        yield
        return
    with context_factory():
        yield


def gpu_benchmark_lock(fn: Callable[P, T]) -> Callable[P, T]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with maybe_gpu_benchmark_lock():
            return fn(*args, **kwargs)

    return wrapper


# Device-type → benchmarking function registry.
# Keys must match torch.device.type (e.g., "cpu", "cuda", "mps", "xpu", ...).
# Values are callables with signature:
#   fn(self: Benchmarker, _callable: Callable[..., Any], *, warmup: int, rep: int, **kwargs) -> Any
_BENCHMARK_DISPATCH: dict[str, Callable[..., Any]] = {}


def register_benchmarker(
    device_type: str,
    fn: Callable[..., Any],
    *,
    override: bool = False,
) -> None:
    """
    Register a device-type specific benchmarker.

    Args:
        device_type: torch.device.type string (e.g., "cuda", "cpu", "mps", "xpu").
        fn: callable(self, _callable, *, warmup, rep, **kwargs) -> Any
        override: allow overriding an existing registration.
    """
    if not isinstance(device_type, str) or not device_type:
        raise ValueError(
            "device_type must be a non-empty string matching torch.device.type"
        )
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not override and device_type in _BENCHMARK_DISPATCH:
        raise ValueError(
            f"Benchmarker for device_type '{device_type}' already registered"
        )
    _BENCHMARK_DISPATCH[device_type] = fn


def may_distort_benchmarking_result(fn: Callable[..., Any]) -> Callable[..., Any]:
    from torch._inductor import config

    if config.test_configs.distort_benchmarking_result == "":
        return fn

    def distort(
        ms: list[float] | tuple[float, ...] | float,
    ) -> list[float] | tuple[float, ...] | float:
        if isinstance(ms, (list, tuple)):
            return type(ms)(distort(val) for val in ms)  # type: ignore[misc]

        distort_method = config.test_configs.distort_benchmarking_result
        if not isinstance(ms, float):
            raise AssertionError(f"Expected float, got {type(ms)}")
        if distort_method == "inverse":
            return 1.0 / ms if ms else 0.0
        elif distort_method == "random":
            import random

            return random.random()
        else:
            raise RuntimeError(f"Unrecognized distort method {distort_method}")

    @functools.wraps(fn)
    def wrapper(
        *args: list[Any], **kwargs: dict[str, Any]
    ) -> list[float] | tuple[float, ...] | float:
        ms = fn(*args, **kwargs)

        return distort(ms)

    return wrapper


def may_ban_benchmarking() -> None:
    if torch._inductor.config.deterministic:
        raise RuntimeError("""In the deterministic mode of Inductor, we will avoid those
        benchmarkings that would cause non-deterministic results. Only benchmarkings in the vetted
        scenarios are allowed. Examples include autotuning for triton configs of pointwise kernels.

        When you see this exception, you can do one of the following two things:
        1. if the benchmarking you are doing does not introduce any non-determinism, you can just
        add is_vetted_benchmarking=True to your benchmark_gpu call. That would solve the issue.

        2. if the benchmarking you are doing indeed introduces non-determinism, you'll need to disable
        such feature in deterministic mode or find an alternative implementation that is deterministic.
        """)


def time_and_count(
    fn: Callable[Concatenate[Any, P], T],
) -> Callable[Concatenate[Any, P], T]:
    """
    Wraps `fn` to increment the appropriate dynamo counters. It is expected that `fn`
    is a method of `Benchmarker` or one of its subclasses; typing limitations prevent
    us from declaring this directly.

    NOTE: If you're tempted to add a dynamo_timed call here, this function can be
    called enough that the dynamo_timed overhead is not negligible.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        fn_qual_name = f"{self.__class__.__name__}.{fn.__name__}"
        counters["inductor"][f"benchmarking.{fn_qual_name}"] += 1
        return fn(self, *args, **kwargs)

    return wrapper


class Benchmarker:
    """
    A device-agnostic benchmarking utility for measuring the runtime of
    inductor generated callables.
    """

    def infer_device(self, *fn_args: Any, **fn_kwargs: Any) -> torch.device:
        inferred_device: torch.device | None = None
        for arg_or_kwarg in chain(fn_args, fn_kwargs.values()):
            # Some callables take nested structures as arguments so use the
            # flattened form to find any tensors
            for arg_or_kwarg_leaf in pytree.tree_leaves(arg_or_kwarg):
                if not isinstance(arg_or_kwarg_leaf, torch.Tensor):
                    continue
                if inferred_device is None:
                    inferred_device = arg_or_kwarg_leaf.device
                elif arg_or_kwarg_leaf.device != inferred_device:
                    raise ValueError(
                        "Can't safely infer the device type of `fn` with multiple device types in `fn_args` and `fn_kwargs`!"
                    )

        if inferred_device is None:
            raise ValueError(
                "Can't safely infer the device type of `fn` with no device types"
                " in `fn_args` or `fn_kwargs`. Use a direct benchmarking method instead e.g. "
                "`Benchmarker.benchmark_cpu` or `Benchmarker.benchmark_gpu`."
            )

        return inferred_device

    @time_and_count
    def benchmark(
        self: Self,
        fn: Callable[..., Any],
        fn_args: tuple[Any, ...] | None = None,
        fn_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> float:
        """Benchmark `fn(*fn_args, *fn_kwargs)` and return the runtime, in milliseconds (the
        actual runtime calculation is dictated by the benchmarking implementation, but may be
        one of [mean, median, minimum, etc.]). Functions as a convenience wrapper around
        device-specific implementations, like `benchmark_cpu` and `benchmark_gpu`. Raises
        `ValueError(...)` if we can't safely infer the device type of `fn`; for example,
        if multiple device types are found in `fn_args` and `fn_kwargs`, or if no device
        types are found. To bypass device inference, provide the device to the `device`
        parameter.

        WARNING: if `fn` mutates `fn_args` or `fn_kwargs`, benchmarking may fail unexpectedly.
        For example, if `fn` clears a mutable object, subsequent invocations of `fn` during
        benchmarking will fail. In such cases, `fn` should handle cloning its arguments internally.
        If device inference is required, `Benchmarker.infer_device` can be used prior to calling
        this method without any arguments for `fn_args` and `fn_kwargs`.

        Arguments:
        - fn: The function to benchmark.
        - fn_args: The function's arguments.
        - fn_kwargs: The function's kwargs.

        Keyword Arguments:
        - device: Which device to use for benchmarking. If not provided the device will be attempted
        to be inferred from `fn_args` and `fn_kwargs`.
        - **kwargs: The benchmarking implementation's kwargs.

        Returns:
        - The runtime of `fn(*fn_args, **fn_kwargs)`, in milliseconds.
        """
        inferred_device: torch.device | None = None
        if device is not None:
            inferred_device = (
                torch.device(device) if isinstance(device, str) else device
            )
        else:
            if fn_args is None and fn_kwargs is None:
                raise ValueError(
                    "`fn_args` and `fn_kwargs` cannot both be None if `device` is not provided."
                )

            fn_args = fn_args or tuple()
            fn_kwargs = fn_kwargs or {}
            inferred_device = self.infer_device(*fn_args, **fn_kwargs)

        if not isinstance(inferred_device, torch.device):
            raise AssertionError(f"Expected torch.device, got {type(inferred_device)}")

        fn_args = fn_args or tuple()
        fn_kwargs = fn_kwargs or {}

        # No need to wrap if the callable takes no arguments
        if len(fn_args) == 0 and len(fn_kwargs) == 0:
            # Keep a true zero-arg callable type to satisfy type checkers.
            def _callable() -> Any:
                return fn()
        else:
            _args = fn_args
            _kwargs = fn_kwargs

            def _callable() -> Any:
                return fn(*_args, **_kwargs)

        warmup = kwargs.pop("warmup", inductor_config.inductor_default_autotune_warmup)
        rep = kwargs.pop("rep", inductor_config.inductor_default_autotune_rep)

        # Surfacing all kernels during autotuning is super noisy; filtering these out.
        with DebugMode._benchmarking_inductor():
            # First, try a registered device-specific benchmarker
            benchmark_fn: Callable[..., Any] | None = _BENCHMARK_DISPATCH.get(
                inferred_device.type
            )
            if benchmark_fn is not None:
                return benchmark_fn(self, _callable, warmup=warmup, rep=rep, **kwargs)

            # Backward-compatible default:
            # - CPU  -> CPU benchmark path
            # - else -> GPU benchmark path (legacy behavior retained for non-CPU)
            if inferred_device == torch.device("cpu"):
                return self.benchmark_cpu(_callable, warmup=warmup, rep=rep, **kwargs)
            return self.benchmark_gpu(_callable, warmup=warmup, rep=rep, **kwargs)

    @time_and_count
    def benchmark_cpu(
        self: Self, _callable: Callable[[], Any], warmup: int = 20, rep: int = 100
    ) -> float:
        """Benchmark the CPU callable, `_callable`, and return the median runtime,
        in milliseconds.

        Arguments:
        - _callable: The CPU callable to benchmark.

        Keyword Arguments:
        - warmup: Optionally, the duration, in milliseconds, to run `_callable`
        before benchmarking starts.
        - rep: Optionally, the duration, in milliseconds, to run `_callable`
        during benchmarking.

        Returns:
        - The median runtime of `_callable`, in milliseconds.
        """

        def run_for(ms: int) -> list[float]:
            timings = []
            run_start_t = time.perf_counter()
            while True:
                start_t = time.perf_counter()
                _callable()
                end_t = time.perf_counter()
                timings.append((end_t - start_t) * MILLISECONDS_PER_SECOND)
                if ((end_t - run_start_t) * MILLISECONDS_PER_SECOND) > ms:
                    break
            return timings

        run_for(warmup)
        return median(run_for(rep))

    @time_and_count
    def benchmark_gpu(self: Self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError

    @time_and_count
    @gpu_benchmark_lock
    def benchmark_gpu_with_cuda_graph(
        self: Self,
        _callable: Callable[[], Any],
        grad_to_none: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> float:
        """Benchmark a GPU callable using CUDA graph capture and replay.

        This captures the callable into a CUDA graph and benchmarks the graph replay,
        which eliminates kernel launch overhead for fair comparison between different
        implementations.
        """
        # Warmup
        torch.cuda.synchronize()
        _callable()
        torch.cuda.synchronize()

        # Side-stream warmup then capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            _callable()
        stream.synchronize()

        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            cuda_graph, stream=stream, capture_error_mode="thread_local"
        ):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            _callable()

        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # grad clearing is captured in the graph, don't pass it through.
        return self.benchmark_gpu(cuda_graph.replay, **kwargs)


# Make built-in defaults explicit via the registry
def _default_cpu_bench(self, f, *, warmup, rep, **kw):
    return self.benchmark_cpu(f, warmup=warmup, rep=rep, **kw)


def _default_cuda_bench(self, f, *, warmup, rep, **kw):
    kw.setdefault("device_type", "cuda")
    return self.benchmark_gpu(f, warmup=warmup, rep=rep, **kw)


def _default_xpu_bench(self, f, *, warmup, rep, **kw):
    kw.setdefault("device_type", "xpu")
    return self.benchmark_gpu(f, warmup=warmup, rep=rep, **kw)


register_benchmarker("cpu", _default_cpu_bench, override=True)
register_benchmarker("cuda", _default_cuda_bench, override=True)
register_benchmarker("xpu", _default_xpu_bench, override=True)


def _get_callable_device_kernel_time_us(
    kineto_events: Any,
    profiler_events: Any,
    rep: int,
    profiler_device_type: Any,
) -> float:
    from torch.autograd import DeviceType

    benchmark_event_ids: OrderedSet[int] = OrderedSet()

    def collect_cpu_event_ids(event: Any) -> None:
        if event.device_type != DeviceType.CPU:
            return

        benchmark_event_ids.add(event.id)
        for child in event.cpu_children:
            collect_cpu_event_ids(child)

    benchmark_events = [
        event
        for event in profiler_events
        if event.name == _CALLABLE_PROFILE_EVENT_NAME
        and event.device_type == DeviceType.CPU
    ]
    if len(benchmark_events) != rep:
        raise RuntimeError(
            f"Expected {rep} {_CALLABLE_PROFILE_EVENT_NAME} profiling events. "
            f"Found {len(benchmark_events)} events."
        )

    for event in benchmark_events:
        collect_cpu_event_ids(event)

    # An op may re-dispatch internally (e.g. aten::sum with no dim calls
    # aten::sum with dim), producing a kineto CPU event whose corr ID the
    # device kernel links to but which is absent from the profiler event tree.
    # Include kineto corr IDs of all CPU events within the _CALLABLE time
    # window to cover these hidden dispatches.
    callable_kineto_windows: list[tuple[int, int]] = []
    for ev in kineto_events:
        if (
            ev.name() == _CALLABLE_PROFILE_EVENT_NAME
            and ev.device_type() == DeviceType.CPU
        ):
            callable_kineto_windows.append((ev.start_ns(), ev.end_ns()))

    if callable_kineto_windows:
        for ev in kineto_events:
            if ev.device_type() != DeviceType.CPU:
                continue
            ev_start = ev.start_ns()
            ev_end = ev.end_ns()
            for win_start, win_end in callable_kineto_windows:
                if ev_start >= win_start and ev_end <= win_end:
                    benchmark_event_ids.add(ev.correlation_id())
                    break

    device_time_us = 0.0
    for event in kineto_events:
        linked_correlation_id = event.linked_correlation_id()
        correlation_id = event.correlation_id()
        activity_type = event.activity_type()
        if (
            event.device_type() == profiler_device_type
            and activity_type != "gpu_user_annotation"
            and (
                linked_correlation_id in benchmark_event_ids
                or (
                    linked_correlation_id == 0 and correlation_id in benchmark_event_ids
                )
            )
            and event.name() != "Context Sync"
        ):
            device_time_us += (event.end_ns() - event.start_ns()) / 1000.0

    return device_time_us


class TritonBenchmarker(Benchmarker):
    @cached_property
    def triton_do_bench(self: Self) -> Callable[..., Any]:
        """Lazily import Triton's `do_bench`."""
        try:
            from triton.testing import do_bench
        except ImportError as e:
            raise NotImplementedError("requires Triton") from e
        return do_bench

    @may_distort_benchmarking_result
    @time_and_count
    @gpu_benchmark_lock
    # pyrefly: ignore [bad-override]
    def benchmark_gpu(
        self: Self,
        _callable: Callable[[], Any],
        is_vetted_benchmarking: bool = False,
        **kwargs: Any,
    ) -> float:
        """Benchmark the GPU callable, `_callable`, and return the runtime, in milliseconds.

        Arguments:
        - _callable: The GPU callable to benchmark.

        Keyword Arguments:
        - quantiles: Optionally, a tuple of floats denoting the requested quantiles.
        - return_mode: Optionally, the requested return mode. Currently, Triton's
        `do_bench` supports min, max, mean, and median return modes.
        - **kwargs: Additional kwargs passed to Triton's `do_bench`.

        Returns:
        - The runtime of `callable`, in milliseconds. If `kwargs["quantiles"]` is specified,
        this is the first requested quantile. Else, if `kwargs["return_mode"]` is specified,
        this is the requested return mode. Otherwise, this is the median.
        """
        if not is_vetted_benchmarking:
            may_ban_benchmarking()

        do_bench_params = inspect.signature(self.triton_do_bench).parameters
        for kwarg in list(kwargs.keys()):
            if kwarg not in do_bench_params:
                del kwargs[kwarg]
        try:
            if "quantiles" in kwargs:
                return self.triton_do_bench(_callable, **kwargs)[0]
            elif "return_mode" in kwargs:
                return self.triton_do_bench(_callable, **kwargs)
            return self.triton_do_bench(_callable, **kwargs, return_mode="median")
        except Exception as e:
            # ErrorInvalidConfiguration
            # Return inf to skip this config during autotuning
            error_str = str(e).lower()
            if "invalid configuration" in error_str:
                logger.warning(
                    "Skipping benchmark due to invalid configuration error: %s",
                    error_str,
                )
                return float("inf")
            raise


class InductorBenchmarker(TritonBenchmarker):  # noqa: docstring_linter
    def __init__(self: Self) -> None:
        super().__init__()
        self._in_cudagraph_benchmark = False

    @cached_property
    def L2_cache_size(self: Self) -> int:
        """Get the L2 cache size, in bytes, of the current device."""
        return self.get_device_cache_size()

    def get_device_cache_size(
        self: Self, device_type: str | torch.device | None = None
    ) -> int:
        """Get the L2/global cache size, in bytes, of the current device."""
        if "L2_cache_size" in self.__dict__:
            return self.__dict__["L2_cache_size"]

        device_type = _normalize_gpu_device_type(device_type)
        device_interface = get_interface_for_device(device_type)
        device = device_interface.current_device()
        props = device_interface.get_device_properties(device)
        for attr in ("L2_cache_size", "last_level_cache_size"):
            cache_size = getattr(props, attr, None)
            if cache_size:
                return cache_size
        return 256 * 1024 * 1024

    def get_event_pairs(
        self: Self, iters: int, device_type: str | torch.device | None = None
    ) -> list[tuple[Any, Any]]:
        """Get `iters` pairs of device events."""
        device_interface = get_interface_for_device(
            _normalize_gpu_device_type(device_type)
        )
        return [
            (
                device_interface.Event(enable_timing=True),
                device_interface.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]

    def get_event_pairs_min_timing(
        self: Self, event_pairs: list[tuple[Any, Any]]
    ) -> float:
        """Get the minimum timing, in milliseconds, for a group of event pairs."""
        return min(
            [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
        )

    @time_and_count
    def benchmark_gpu_with_cuda_graph(
        self: Self,
        _callable: Callable[[], Any],
        grad_to_none: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> float:
        # Prevent benchmark_gpu from re-entering this method
        # when autotune_cudagraph_benchmarking is enabled.
        self._in_cudagraph_benchmark = True
        try:
            result = super().benchmark_gpu_with_cuda_graph(
                _callable, grad_to_none=grad_to_none, **kwargs
            )
        finally:
            self._in_cudagraph_benchmark = False
        return result

    @may_distort_benchmarking_result
    @time_and_count
    @gpu_benchmark_lock
    def benchmark_gpu(  # type: ignore[override]
        self: Self,
        _callable: Callable[[], Any],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration: int = 25,
        return_mode: str = "min",
        grad_to_none: list[torch.Tensor] | None = None,
        is_vetted_benchmarking: bool = False,
        device_type: str | torch.device | None = None,
        **kwargs: Any,
    ) -> float | list[float]:
        """Benchmark a GPU callable using a custom benchmarking implementation.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - estimation_iters: Optionally, the number of iterations to run `_callable`
        during runtime estimation.
        - memory_warmup_iters: Optionally, the number of iterations to flush the L2
        cache before starting benchmarking.
        - benchmark_iters: Optionally, the number of iterations to run `_callable`
        during the benchmarking.
        - max_benchmark_duration: Optionally, the maximum duration of the benchmarking,
        in milliseconds. An estimated duration is calculated based on the values
        of `memory_warmup_iters` and `benchmark_iters`, along with the estimated
        runtime of `_callable` and various other factors, and we then shrink
        `benchmark_iters` to fit in the allotted maximum duration.
        - return_mode: Return mode for benchmark results. Options are "min" (default),
        "all" (returns all measurements).
        - grad_to_none: Optionally, a list of tensors whose gradients should be cleared
        before each benchmark iteration.
        - is_vetted_benchmarking: in deterministic mode, we only allow
        benchmarking in vetted cases.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - If return_mode="min": The minimum runtime of `_callable`, in milliseconds.
        - If return_mode="all": List of all runtime measurements, in milliseconds.
        """

        if not is_vetted_benchmarking:
            may_ban_benchmarking()

        device_type = _normalize_gpu_device_type(device_type)
        device_interface = get_interface_for_device(device_type)

        if (
            device_type == "cuda"
            and inductor_config.autotune_cudagraph_benchmarking
            and inductor_config.max_autotune
            and not self._in_cudagraph_benchmark
        ):
            try:
                return self.benchmark_gpu_with_cuda_graph(
                    _callable,
                    estimation_iters=estimation_iters,
                    memory_warmup_iters=memory_warmup_iters,
                    benchmark_iters=benchmark_iters,
                    max_benchmark_duration=max_benchmark_duration,
                    return_mode=return_mode,
                    grad_to_none=grad_to_none,
                    device_type=device_type,
                )
            except RuntimeError:
                logger.warning(
                    "CUDA graph capture failed during benchmarking, "
                    "falling back to eager benchmarking",
                    exc_info=True,
                )

        # we don't want any outside errors propagating into benchmarking
        device_interface.synchronize()

        # warmup `_callable` (and catches any failures in the process)
        _callable()
        device_interface.synchronize()

        # see https://github.com/triton-lang/triton/pull/840 for why `dtype=torch.int`
        buffer = torch.empty(
            self.get_device_cache_size(device_type) // 4,
            dtype=torch.int,
            device=device_type,
        )
        buffer.zero_()

        # estimate the runtime of `_callable`
        event_pairs = self.get_event_pairs(estimation_iters, device_type=device_type)
        for start_event, end_event in event_pairs:
            # Clear gradients before timing (matches triton.testing.do_bench)
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        device_interface.synchronize()
        estimated_timing = self.get_event_pairs_min_timing(event_pairs)

        # adjust `benchmark_iters` to fit in the maximum benchmarking duration
        if estimated_timing > 0:
            benchmark_iters = max(
                min(benchmark_iters, int(max_benchmark_duration // estimated_timing)), 1
            )

        # do the memory warmup
        for _ in range(memory_warmup_iters):
            buffer.zero_()

        # benchmark `_callable`
        event_pairs = self.get_event_pairs(benchmark_iters, device_type=device_type)
        for start_event, end_event in event_pairs:
            # Clear gradients before timing (matches triton.testing.do_bench)
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        device_interface.synchronize()

        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor performance benchmarks
        del buffer

        # Return based on the requested mode
        if return_mode == "all":
            # Get all timings from event pairs
            all_timings = [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
            return all_timings
        elif return_mode == "min":
            benchmarked_timing = self.get_event_pairs_min_timing(event_pairs)
            # return the minimum of `estimated_timing` and `benchmarked_timing`,
            # we just want the minimum timing overall so we might as well check both
            return min(estimated_timing, benchmarked_timing)
        else:
            raise ValueError(
                f"Unsupported return_mode: {return_mode}. Use 'min' or 'all'."
            )


class TorchProfilerBenchmarker(InductorBenchmarker):  # noqa: docstring_linter
    """Benchmarker that uses torch.profiler for GPU kernel benchmarking."""

    @time_and_count
    @gpu_benchmark_lock
    def benchmark_gpu(  # type: ignore[override]
        self: Self,
        _callable: Callable[[], Any],
        warmup: int = 25,
        rep: int = 100,
        estimation_iters: int = 5,
        memory_warmup_iters: int = 10,
        max_benchmark_duration: int = 25,
        return_mode: str = "mean",
        grad_to_none: list[torch.Tensor] | None = None,
        device_type: str | torch.device | None = None,
        **kwargs: Any,
    ) -> float:
        """Benchmark a GPU callable using torch.profiler.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - warmup: Ignored (kept for API compat). Warmup is handled by the
        estimation phase which runs estimation_iters with cache flushing.
        - rep: Optionally, the maximum number of iterations to run during benchmarking.
        - estimation_iters: Optionally, the number of iterations used to estimate
        the runtime of `_callable` for dynamic rep adjustment. These iterations
        also serve as warmup for the callable.
        - memory_warmup_iters: Optionally, the number of buffer.zero_() iterations
        to run after estimation to bring the cache into a steady state before
        the profiled benchmark phase.
        - max_benchmark_duration: Optionally, the maximum duration of the profiled
        benchmark phase, in milliseconds. The rep count is reduced if the
        estimated total would exceed this budget.
        - return_mode: Return mode for benchmark results. Options are "min", "mean" (default),
        or "max".
        - grad_to_none: Optionally, a list of tensors whose gradients should be cleared
        before each benchmark iteration.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - The runtime of `_callable` in milliseconds, computed according to return_mode.
        """
        device_type = _normalize_gpu_device_type(device_type)
        device_interface = get_interface_for_device(device_type)

        # we don't want any outside errors propagating into benchmarking
        device_interface.synchronize()

        # warmup `_callable` (and catches any failures in the process)
        _callable()
        device_interface.synchronize()

        # Keep Triton's 256 MB cache flush on ROCm. On other backends, reuse
        # the shared L2-sized flush from InductorBenchmarker.
        # see https://github.com/triton-lang/triton/pull/840 for why `dtype=torch.int`
        if torch.version.hip:
            buffer_size_bytes = 256 * 1024 * 1024
        else:
            buffer_size_bytes = self.get_device_cache_size(device_type)
        buffer = torch.empty(
            buffer_size_bytes // 4, dtype=torch.int, device=device_type
        )
        buffer.zero_()

        # Estimation phase with separate event pairs — also serves as warmup.
        # Using per-iteration event pairs lets us take the min, matching
        # InductorBenchmarker's approach for a more robust estimate.
        event_pairs = self.get_event_pairs(estimation_iters, device_type=device_type)
        for start_event, end_event in event_pairs:
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        device_interface.synchronize()
        estimated_ms = self.get_event_pairs_min_timing(event_pairs)
        if estimated_ms > 0:
            rep = max(min(rep, int(max_benchmark_duration / estimated_ms)), 1)

        # Light memory warmup: flush the cache into a steady state before
        # the profiled run.  No callable — estimation already warmed it.
        for _ in range(memory_warmup_iters):
            buffer.zero_()

        # benchmark with profiler
        # Use both CPU and device activities, otherwise record_function
        # will not record the region.
        device_type_upper = device_type.upper()
        profile_activity = getattr(torch.profiler.ProfilerActivity, device_type_upper)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                profile_activity,
            ],
            record_shapes=False,
        ) as prof:
            for _ in range(rep):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                buffer.zero_()

                with torch.profiler.record_function("_CALLABLE"):
                    _callable()

        device_interface.synchronize()

        # Extract _CALLABLE GPU time directly from raw kineto events.
        # This avoids prof.key_averages() which triggers expensive lazy
        # processing: _parse_kineto_results (wrapping every raw event in
        # a Python FunctionEvent), _build_tree, and grouping/aggregation.
        from torch.autograd import DeviceType as _DeviceType

        profiler_device_type = getattr(_DeviceType, device_type_upper)
        callable_gpu_time_us = 0.0
        for kineto_event in prof.profiler.kineto_results.events():
            if (
                kineto_event.name() == _CALLABLE_PROFILE_EVENT_NAME
                and kineto_event.device_type() == profiler_device_type
            ):
                callable_gpu_time_us += (
                    kineto_event.end_ns() - kineto_event.start_ns()
                ) / 1000.0

        if callable_gpu_time_us <= 0:
            callable_gpu_time_us = _get_callable_device_kernel_time_us(
                prof.profiler.kineto_results.events(),
                prof.events(),
                rep,
                profiler_device_type,
            )

        if callable_gpu_time_us <= 0:
            raise AssertionError(
                f"TorchProfilerBenchmarker: '_CALLABLE' {device_type_upper} event not found in "
                "raw kineto results, and no correlated device events were found."
            )

        # TODO: Revisit incorporating launch overhead effects.
        total_time_us = callable_gpu_time_us
        avg_time_ms = (total_time_us / rep) / 1000.0

        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor performance benchmarks
        del buffer

        # Return based on the requested mode
        # Note: For profiler-based benchmarking, we return the mean time per iteration
        # min/max modes are kept for API compatibility but return the mean
        if return_mode in ("min", "mean", "max"):
            return avg_time_ms
        else:
            raise ValueError(
                f"Unsupported return_mode: {return_mode}. Use 'min', 'mean', or 'max'."
            )


def _make_default_benchmarker() -> Benchmarker:
    if inductor_config.use_torch_profiler_benchmarker:
        return TorchProfilerBenchmarker()
    if inductor_config.use_experimental_benchmarker:
        return InductorBenchmarker()
    return TritonBenchmarker()


benchmarker = _make_default_benchmarker()
<|endoftext|>import functools
import logging
import threading
import weakref

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

local = threading.local()
local.memory_tracker = None


class BufferMemoryTracker:
    """
    Tracks inductor runtime allocations and deallocations to compare against
    expected behavior.
    """

    def __init__(self) -> None:
        self.tensor_tracker: dict[str, torch.storage.UntypedStorage] = (
            weakref.WeakValueDictionary()  # type: ignore[assignment]
        )
        self.died_since_last_step: OrderedSet[str] = OrderedSet()
        self.added_since_last_step: OrderedSet[str] = OrderedSet()
        self.error = (
            torch._inductor.config.test_configs.track_memory_lifecycle == "assert"
        )

    def set_tensor(self, name: str, tensor: torch.Tensor) -> None:
        storage = tensor.untyped_storage()

        self.added_since_last_step.add(name)
        self.tensor_tracker[name] = storage

        def on_tensor_death() -> None:
            self.died_since_last_step.add(name)

        weakref.finalize(storage, on_tensor_death)

    def advance_step(self) -> None:
        self.died_since_last_step.clear()
        self.added_since_last_step.clear()

    def log_or_raise(self, msg: str) -> None:
        if self.error:
            raise RuntimeError(msg)
        else:
            log.info(msg)

    def check_step_delta(
        self,
        expected_allocated: list[str],
        expected_freed: list[str],
        is_final_step: bool,
    ) -> None:
        """Check only the delta changes since last step"""

        # Check expected deaths - we don't currently distinguish between nodes which die in last step
        # and are returned as outputs, so skip if final_step.
        if not is_final_step:
            missing_deaths = OrderedSet(expected_freed) - self.died_since_last_step
            if missing_deaths:
                self.log_or_raise(
                    f"Expected tensors to die but still alive: {missing_deaths}"
                )

        # Check for unexpected deaths
        unexpected_deaths = self.died_since_last_step - OrderedSet(expected_freed)
        if unexpected_deaths:
            self.log_or_raise(f"Unexpected tensor deaths: {unexpected_deaths}")

        # Check newly alive tensors - separate messages like deaths
        actual_allocated = self.added_since_last_step
        expected_allocated_set = OrderedSet(expected_allocated)

        extra_alive = actual_allocated - expected_allocated_set
        if extra_alive:
            self.log_or_raise(f"Unexpected allocated tensors: {extra_alive}")

        missing_alive = expected_allocated_set - actual_allocated
        if missing_alive:
            self.log_or_raise(
                f"Expected allocated tensors but missing: {missing_alive}"
            )

        # Reset for next step
        self.advance_step()

        if is_final_step:
            local.memory_tracker = None


def get_mem_tracker() -> BufferMemoryTracker:
    if local.memory_tracker is None:
        local.memory_tracker = BufferMemoryTracker()
    return local.memory_tracker


def track_tensor(tensor: torch.Tensor, name: str) -> None:
    get_mem_tracker().set_tensor(name, tensor)


def tracked_empty_strided(
    size: list[int],
    stride: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    o = torch.empty_strided(size, stride, dtype=dtype, device=device)
    track_tensor(o, name)
    return o


def check_memory_step(
    allocated: list[str], freed: list[str], is_final_step: bool = False
) -> None:
    tracker = get_mem_tracker()
    tracker.check_step_delta(allocated, freed, is_final_step)


@functools.lru_cache(None)
def register_check_mem_op() -> None:
    lib = torch.library.Library("_inductor_debug", "FRAGMENT")
    # Keep `lib` alive: this function is lru_cached, so it runs once, but the
    # registered op must remain accessible via torch.ops for the rest of the
    # process's lifetime. Letting `lib` die triggers the finalizer, which
    # clears torch.ops._inductor_debug.check_memory_step.
    torch.library._keep_alive.append(lib)
    lib.define(
        "check_memory_step(str[] allocated, str[] freed, bool is_final_step) -> ()"
    )
    lib.impl("check_memory_step", check_memory_step, "BackendSelect")
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

    _register_effectful_op(
        torch.ops._inductor_debug.check_memory_step.default,
        _EffectType.ORDERED,
    )
<|endoftext|>"""
PyTorch Inductor Autotuning Cache System

This module implements a caching system for autotuning configurations in PyTorch's Inductor compiler.
It provides mechanisms to store and retrieve optimal kernel configurations both locally and remotely,
which significantly speeds up compilation by reusing previously discovered optimal parameters.

The caching system includes:
- Local filesystem caching for individual machine reuse
- Remote caching for sharing optimizations across machines
- Bundled caching to efficiently store multiple related configurations
- Cache invalidation based on PyTorch versions and backend changes
- Serialization/deserialization support for worker processes

Key components:
- AutotuneCache: Main class for managing cache access and storage
- AutotuneCacheBundler: Bundles multiple cache entries for efficient storage
- LocalCacheBackend: Low-level file operations for cache storage
- AutotuneCacheArtifact: Integration with PyTorch's artifact system

