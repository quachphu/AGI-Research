    the AOT layer to import, not as a stable public surface.

    ``inner_python`` is the Inductor output module exposing ``call(args) -> outs``
    for the post-AOTAutograd inner graph (dense, functionalized). It is the inductor
    piece only: it carries NO prelude/epilogue (subclass flatten/unflatten, input-
    mutation copy-back, output-alias regen, grad disabling). Those belong to the AOT
    layer -- a companion change in ``torch._functorch.aot_autograd`` wraps this and
    composes AOTAutograd's codegen'd runtime wrappers around the result.
    Callers must run ``call`` under ``torch.no_grad()`` (the kernels use out= ops).

    Caller preconditions (this layer does not re-derive them):

    - ``gm`` is a post-AOTAutograd dense, functionalized inner graph (the dense
      forward/backward AOTAutograd hands to its inductor backend), NOT a raw Dynamo
      or pre-dispatch graph still carrying subclasses or autograd. Crucially it must be
      DECOMPOSED against the inductor decomposition table: this function drives the
      inductor codegen entry point directly, and inductor lowering asserts "both a
      fallback and a decomp for same op" on an undecomposed op (e.g. ``aten._softmax``).
    - Placeholders carry fake ``val`` metadata (the graph was traced under a
      ``FakeTensorMode``); those fake tensors are the compile-time input contract.
    - ``gm`` lowers to a runnable inductor output module. A graph with no compute
      raises ``NoRunnableInductorModuleError``.

    The compile lowers against the placeholders' fake ``val`` metadata, whose
    static/symbolic shapes AOTAutograd already resolved, so the graph is the source of
    truth for shapes -- there is no ``dynamic_shapes`` knob. ``example_inputs`` is accepted
    for the backend-contract signature but is not re-fakified for the compile itself.

    ``options`` is an optional inductor config-override dict (``None`` means no
    overrides), the same shape as ``torch._inductor.compile``'s ``options``. The
    keys are inductor config names: they are merged into the ``config.patch``
    block this function already wraps the compile in, so they take effect as
    config rather than being forwarded as ``compile_fx`` kwargs. The two
    capture-critical pins below (``benchmark_harness``, ``cpp_wrapper``) always win
    over a conflicting user option, since the source-capture contract depends on
    them and a caller cannot be allowed to break the emitted module.

    ``inner_python`` is self-contained: exec'ing it JIT-compiles the inlined kernels on
    first use, so it runs with no cache. ``cache`` is a PURE ACCELERATOR -- the opaque
    ``save_cache_artifacts()`` bundle; passing it to ``load_from_python`` warms the kernel
    caches so exec loads the precompiled binaries instead of recompiling. It is ``None``
    when the graph produced no cacheable module (no compute) or caches are disabled
    (``force_disable_caches`` or ``fx_graph_cache=False``); ``inner_python`` still runs.

    ``inner_python`` is read off the ``CompiledFxGraph`` that ``compile_fx_inner``
    returns -- Inductor stashes the wrapper-module source on it as ``source_code`` -- so
    it reflects the FINAL selected module on a cold compile, a warm cache restore, or with
    caches disabled. No process-global codegen hook is involved, and the throwaway
    max_autotune benchmark lowerings (which compile to their own modules during
    autotuning) never become the returned graph, so nothing needs to be filtered out.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            f"compile_to_python expects a post-AOTAutograd torch.fx.GraphModule, "
            f"got {type(gm)}. This is an internal entry point wrapped by a higher AOT "
            f"layer and is not meant to be called directly."
        )
    # The experimental TORCHINDUCTOR_FX_COMPILE_MODE=async+/progressive+ schemes make
    # compile_fx return an _AsyncOutputCode/_ProgressiveOutputCode that carries only
    # _boxed_call and never surfaces the wrapper ``source_code`` this contract reads off.
    # These are module-level globals resolved from the env at import time, not
    # config-patchable like the pins below, so detect them up front and fail with a
    # distinct error instead of the no-compute NoRunnableInductorModuleError -- the graph
    # is runnable, the async scheme just did not surface its source.
    from torch._inductor import compile_fx as _compile_fx

    if _compile_fx.fx_compile_async or _compile_fx.fx_compile_progressive:
        raise RuntimeError(
            "compile_to_python needs synchronous source capture and does not yet "
            "support TORCHINDUCTOR_FX_COMPILE_MODE=async+/progressive+ (the async "
            "output code does not surface the wrapper source)."
        )
    # Treat ``options`` as inductor config overrides and fold them into the same
    # ``config.patch`` we wrap the compile in. ``keep_static_cubin_raw`` is defaulted True
    # (BEFORE user options, so a caller can still trade it off for a smaller cache): its
    # default False nulls the static-launcher raw cubin and relies on the cubin FILE being
    # on disk, which holds for a same-machine warm cache but NOT the cold/cross-container
    # load this cache targets -- keeping it lets ``load_from_python`` rehydrate the cubin
    # from the bundle so the static CUDA launcher survives a fresh-dir load instead of
    # silently falling back to the slower dynamic launch. The two output pins are applied
    # AFTER the user options so they override any conflicting key: benchmark_harness=False
    # keeps the emitted module runnable (no get_args()/benchmark_compiled_module()/__main__),
    # and cpp_wrapper=False keeps it a python wrapper (the C++ backend emits a C++ ``call``
    # we cannot inline). A caller must not be able to break either.
    config_patches: dict[str, Any] = {"keep_static_cubin_raw": True}
    if options is not None:
        config_patches.update(options)
    config_patches.update(
        {
            "benchmark_harness": False,
            "cpp_wrapper": False,
        }
    )
    # ``gm`` is a POST-AOTAutograd inner graph: already functionalized and decomposed
    # against the inductor decomposition table. That precondition lets us drive the
    # inductor codegen entry point (``compile_fx_inner``) DIRECTLY on the dense graph
    # rather than re-entering AOTAutograd via ``standalone_compile``. Re-entry would only
    # re-run decomposition -- a no-op on an already-decomposed graph -- and hand back an
    # AOTAutograd-level cache artifact that belongs to the layer above; driving the inner
    # compile keeps this at the inductor layer and yields the accelerator cache bundle that
    # ``_acceleration_cache_bytes`` returns. (If ``gm`` is NOT decomposed against the
    # inductor table, inductor lowering asserts "both a fallback and a decomp for same op"
    # -- the precondition is load-bearing, not defensive.)
    from torch._guards import detect_fake_mode, tracing, TracingContext
    from torch.compiler._cache import CacheArtifactManager

    from .compile_fx import compile_fx_inner
    from .virtualized import V

    # Own a copy: the collective rewrites and inductor may mutate the graph, and ``gm`` may
    # carry a non-pickleable torchbind ProcessGroup smuggled through deepcopy as a shared
    # reference (mirrors ``standalone_compile``). ``make_fx`` traces ``dist.*`` collectives
    # as opaque ``c10d.{op}_`` calls; rewrite them to the ``_c10d_functional.{op}`` +
    # ``wait_tensor`` form inductor recognizes and unbox torchbind ProcessGroups.
    with _share_torchbind_and_process_group_on_deepcopy():
        gm = copy.deepcopy(gm)
    gm = _functionalize_inplace_collectives(gm)
    gm = _unbox_process_group_torchbinds(gm)

    # Lower against the placeholders' fake ``val`` metadata (the compile-time input
    # contract, carrying the graph's static/symbolic shapes under one FakeTensorMode)
    # rather than re-fakifying ``example_inputs``, which are real and would drop symbolic
    # dims. A post-AOTAutograd graph's shapes are already baked into this metadata, so
    # there is no separate dynamic-shapes knob.
    fake_inputs = _placeholder_fake_inputs(gm)
    fake_mode = detect_fake_mode(fake_inputs)
    if fake_mode is None:
        raise RuntimeError(
            "compile_to_python could not detect a FakeTensorMode on the graph's "
            "placeholders; expected a post-AOTAutograd graph traced under one."
        )
    # no_grad pins the inference path; autotune_at_compile_time keeps the emitted source
    # self-contained (autotuning resolved at compile, not deferred to runtime). The fresh
    # CacheArtifactManager isolates the cache bundle ``_acceleration_cache_bytes`` returns.
    with (
        torch.no_grad(),
        config.patch(config_patches),
        config.patch("triton.autotune_at_compile_time", True),
        tracing(TracingContext(fake_mode)),
        V.set_fake_mode(fake_mode),
        CacheArtifactManager.with_fresh_cache(),
    ):
        compiled_graph = compile_fx_inner(
            gm,
            fake_inputs,
            static_input_idxs=(),
            cudagraphs=BoxedBool(False),
            is_inference=True,
            boxed_forward_device_index=BoxedDeviceIndex(None),
        )
        artifacts = torch.compiler.save_cache_artifacts()
    inner_python = _runnable_source(compiled_graph)
    cache = _acceleration_cache_bytes(artifacts)
    return inner_python, cache


def autograd_cache_key(
    graph,
    example_inputs,
    dynamic_shapes: DynamicShapesType,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    fake_mode: FakeTensorMode | None = None,
):
    from . import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(graph, dynamic_shapes, aot, fake_mode):
        return compile_fx.autograd_cache_key(
            graph,
            example_inputs,
            ignore_shape_env=ignore_shape_env,
        )
<|endoftext|>"""Utilities for lowering subgraphs used by higher order operators"""

import functools
import operator
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast, TypeVar
from typing_extensions import ParamSpec

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from . import ir
from .exc import SubgraphLoweringException
from .graph import GraphLowering
from .ops_handler import OpsHandler, SimpleCSEHandler
from .virtualized import ops, V, WrapperHandler


T = TypeVar("T")
_P = ParamSpec("_P")

OpOverload = torch._ops.OpOverload
LoweringDict = dict[OpOverload | str, Callable[..., Any]]
TargetType = Callable[..., Any] | str


class PointwiseSubgraphLowering(torch.fx.Interpreter):
    """
    Lowers a pointwise subgraph to a single set of buffers with a separate
    lowering object. Errors if buffers are created unexpectedly
    """

    graph_outputs: list[ir.IRNode] | None
    root_graph: GraphLowering
    _current_op: TargetType | None
    # For backwards of buffer_grads with scatters we allow mutations
    allowed_mutations: OrderedSet[OpOverload] | None
    additional_lowerings: LoweringDict | None
    buffers: list[ir.Buffer]
    mutated_buffers: OrderedSet[str]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        root_graph_lowering: GraphLowering,
        allowed_mutations: OrderedSet[OpOverload] | None = None,
        additional_lowerings: LoweringDict | None = None,
    ) -> None:
        super().__init__(gm)
        self.graph_outputs = None
        self.root_graph = root_graph_lowering
        self.allowed_mutations = allowed_mutations
        self.additional_lowerings = additional_lowerings
        self._current_op = None

        # Used to track buffers created during lowering
        self.mutated_buffers = OrderedSet()
        self.buffers = []

    @contextmanager
    def _op_context(self, op: TargetType) -> Generator[None, None, None]:
        """Set which op is being processed in call function to know if we can mutate buffers"""
        previous = self._current_op
        self._current_op = op
        try:
            yield
        finally:
            self._current_op = previous

    def _approved_mutator(self) -> bool:
        return (
            self.allowed_mutations is not None
            and self._current_op in self.allowed_mutations
        )

    def mark_buffer_mutated(self, name: str) -> None:
        if self._approved_mutator():
            self.mutated_buffers.add(name)
        else:
            raise SubgraphLoweringException(
                f"Buffer mutation detected during lowering of {self._current_op}. "
                "Buffer mutations are only allowed in approved mutation ops. "
                "This is an error in the lowering of the subgraph, please file a bug report."
            )

    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False) -> str:
        if self._approved_mutator():
            name = self.root_graph.register_buffer(buffer, set_name=set_name)
            return name
        else:
            raise SubgraphLoweringException(
                "Buffers cannot be created while lowering a pointwise subgraph. "
                "This could be for a good reason (e.g. you're calling an op we can't codegen as a pointwise op), "
                "but it could also be a bug. Please file a bug report if you think this should be supportable."
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.root_graph, name)

    def call_function(
        self,
        target: TargetType,
        args: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        from .lowering import lowerings

        with self._op_context(target):
            if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
                return super().call_function(target, args, kwargs)

            # These takes precedence over the main lowerings
            if self.additional_lowerings is not None:
                if target in self.additional_lowerings:
                    if not isinstance(target, OpOverload):
                        raise AssertionError(
                            f"expected target to be an OpOverload, got {type(target)}"
                        )
                    return self.additional_lowerings[target](*args, **kwargs)

            if target not in lowerings:
                raise SubgraphLoweringException(
                    f"{target} not supported in subgraph, (missing lowering)"
                )
            return lowerings[target](*args, **kwargs)

    def output(self, target: str, args: tuple[Any], kwargs: dict[str, Any]) -> None:  # type: ignore[override]
        if len(args) != 1:
            raise AssertionError(f"expected exactly one output arg, got {len(args)}")
        self.graph_outputs = args[0]


@dataclass
class InputDescriptor:
    dtype: torch.dtype
    device: torch.device


SubgraphInput = InputDescriptor | int | sympy.Basic


class TracingOpsHandler(WrapperHandler):
    def __init__(self, tracer: torch.fx.Tracer, num_inputs: int) -> None:
        parent = tracer.create_proxy("placeholder", "ops", (), {})
        super().__init__(cast(OpsHandler[Any], parent))
        self.tracer = tracer

        self.placeholders = [
            self.tracer.create_proxy("placeholder", f"input{i}", (), {})
            for i in range(num_inputs)
        ]

    def placeholder(self, idx: int) -> torch.fx.Proxy:
        return self.placeholders[idx]

    def output(self, *args: tuple[object]) -> None:
        self.tracer.create_node(
            "output", "output", (tuple(self.tracer.create_arg(a) for a in args),), {}
        )


def lower_pointwise_subgraph(
    subgraph: ir.Subgraph, inputs: list[SubgraphInput]
) -> Callable[_P, Any]:
    # Lower subgraph to ir.Pointwise nodes
    def fake_inner_fn(loop_idx: int, input_idx: int) -> ir.Expr | ir.TensorBox | None:
        return ops.placeholder(input_idx)

    graph_inputs = []
    for i, desc in enumerate(inputs):
        if isinstance(desc, InputDescriptor):
            graph_inputs.append(
                ir.Pointwise.create(
                    device=desc.device,
                    dtype=desc.dtype,
                    inner_fn=functools.partial(fake_inner_fn, input_idx=i),
                    ranges=[],
                )
            )
        else:
            graph_inputs.append(desc)
    gm = subgraph.graph_module
    pw_subgraph = PointwiseSubgraphLowering(gm, root_graph_lowering=V.graph)
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*graph_inputs)

    # Combine multiple pointwise computations into a single graph module
    # Do this by tracing through each individually and doing CSE
    tracer = torch.fx.Tracer()
    tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
    trace_ops = SimpleCSEHandler(TracingOpsHandler(tracer, len(inputs)))
    if pw_subgraph.graph_outputs is None:
        raise AssertionError("expected pw_subgraph.graph_outputs to be set")

    with V.set_ops_handler(trace_ops):
        output_irs = []

        for out_var in pw_subgraph.graph_outputs:
            if not isinstance(out_var, ir.TensorBox):
                raise AssertionError(type(out_var))
            if out_var.get_size() != []:
                raise AssertionError(
                    f"expected scalar output (empty size), got {out_var.get_size()}"
                )
            if not isinstance(out_var.data, ir.StorageBox):
                raise AssertionError(
                    f"expected out_var.data to be a StorageBox, got {type(out_var.data)}"
                )
            if not isinstance(out_var.data.data, ir.Pointwise):
                raise AssertionError(
                    f"expected out_var.data.data to be a Pointwise, got {type(out_var.data.data)}"
                )

            idx = ()
            ir_out = out_var.data.data.inner_fn(idx)

            output_irs.append(ir_out)

        ops.output(*output_irs)

    lowered_gm = torch.fx.GraphModule({}, tracer.graph)

    def inner_fn(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        return lowered_gm(V.get_ops_handler(), *args, **kwargs)

    return inner_fn
<|endoftext|># This CPP builder is designed to support both Windows and Linux OS.
# The design document please check this RFC: https://github.com/pytorch/pytorch/issues/124245

import copy
import ctypes
import errno
import functools
import json
import locale
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import warnings
from collections.abc import Sequence
from ctypes import cdll, wintypes
from ctypes.util import find_library
from pathlib import Path
from typing import Any

import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config, exc
from torch._inductor.cpu_vec_isa import invalid_vec_isa, VecISA
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.torch_version import TorchVersion
from torch.utils._ordered_set import OrderedSet


if config.is_fbcode():
    from triton.fb.build import _run_build_command, build_paths

    from torch._inductor.fb.utils import (
        log_global_cache_errors,
        log_global_cache_stats,
        log_global_cache_vals,
        use_global_cache,
    )
else:

    def log_global_cache_errors(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def log_global_cache_stats(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def log_global_cache_vals(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def use_global_cache() -> bool:  # type: ignore[misc]
        return False


# Windows need setup a temp dir to store .obj files.
_BUILD_TEMP_DIR = "CxxBuild"
_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
_LINKER_SCRIPT = os.path.join(_TORCH_PATH, "_inductor/script.ld")

# initialize variables for compilation
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"

MINGW_GXX = "x86_64-w64-mingw32-g++"

SUBPROCESS_DECODE_ARGS = (
    (locale.getpreferredencoding(), "replace") if _IS_WINDOWS else ()
)

log = logging.getLogger(__name__)


# =============================== toolchain ===============================
def _split_compiler_command(compiler: str) -> list[str]:
    if _IS_WINDOWS:
        return [compiler]
    command = shlex.split(compiler)
    if not command:
        raise ValueError("empty compiler command")
    return command


def _compiler_command(compiler: str, *args: str) -> list[str]:
    return [*_split_compiler_command(compiler), *args]


@functools.cache
def _compiler_version_string(cpp_compiler: str) -> str:
    try:
        return (
            subprocess.check_output(
                _compiler_command(cpp_compiler, "--version"),
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return ""


def _compiler_version_first_line(cpp_compiler: str) -> str:
    version_string = _compiler_version_string(cpp_compiler)
    return version_string.splitlines()[0] if version_string else ""


@functools.lru_cache(1)
def cpp_compiler_search(search: Sequence[str | None]) -> str:
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT

    for cxx in search:
        try:
            if cxx is None:
                # gxx package is only available for Linux
                # according to https://anaconda.org/conda-forge/gxx/
                if sys.platform != "linux":
                    continue
                # Do not install GXX by default
                if not os.getenv("TORCH_INDUCTOR_INSTALL_GXX"):
                    continue
                from torch.utils._filelock import FileLock

                lock_dir = get_lock_dir()
                lock = FileLock(
                    os.path.join(lock_dir, "g++.lock"), timeout=LOCK_TIMEOUT
                )
                with lock:
                    cxx = install_gcc_via_conda()
            subprocess.check_output(_compiler_command(cxx, "--version"))
            return cxx
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            ImportError,
            ValueError,
        ):
            continue
    raise exc.InvalidCxxCompiler


def install_gcc_via_conda() -> str:
    """On older systems, this is a quick way to get a modern compiler"""
    prefix = os.path.join(cache_dir(), "gcc")
    cxx_path = os.path.join(prefix, "bin", "g++")
    if not os.path.exists(cxx_path):
        log.info("Downloading GCC via conda")
        conda = os.environ.get("CONDA_EXE", "conda")
        if conda is None:
            conda = shutil.which("conda")
        if conda is not None:
            subprocess.check_call(
                [
                    conda,
                    "create",
                    f"--prefix={prefix}",
                    "--channel=conda-forge",
                    "--quiet",
                    "-y",
                    "python=3.8",
                    "gxx",
                ],
                stdout=subprocess.PIPE,
            )
    return cxx_path


@functools.cache
def check_compiler_exist_windows(compiler: str) -> None:
    """
    Check if compiler is ready, in case end user not activate MSVC environment.
    """
    try:
        subprocess.check_output(
            _compiler_command(compiler, "/help"), stderr=subprocess.STDOUT
        )
    except FileNotFoundError as e:
        raise exc.InvalidCxxCompiler(compiler) from e
    except subprocess.SubprocessError:
        # Expected that some compiler(clang, clang++) is exist, but they not support `/help` args.
        pass


class WinPeFileVersionInfo:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.version_dll = ctypes.WinDLL("version.dll")  # type: ignore[attr-defined]
        self._setup_functions()
        self._get_version_info()

    def _setup_functions(self) -> None:
        self.version_dll.GetFileVersionInfoSizeW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPDWORD,
        ]
        self.version_dll.GetFileVersionInfoSizeW.restype = wintypes.DWORD

        self.version_dll.GetFileVersionInfoW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        self.version_dll.GetFileVersionInfoW.restype = wintypes.BOOL

        self.version_dll.VerQueryValueW.argtypes = [
            wintypes.LPCVOID,
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.UINT),
        ]
        self.version_dll.VerQueryValueW.restype = wintypes.BOOL

    def _get_version_info(self) -> None:
        dummy = wintypes.DWORD()
        size = self.version_dll.GetFileVersionInfoSizeW(
            self.file_path, ctypes.byref(dummy)
        )

        if size == 0:
            raise RuntimeError(f"Can't get version info size of {self.file_path}.")

        self.version_info = ctypes.create_string_buffer(size)
        success = self.version_dll.GetFileVersionInfoW(
            self.file_path, 0, size, self.version_info
        )

        if not success:
            raise RuntimeError(f"Can't get version info of {self.file_path}.")

    def get_language_id(self) -> int:
        lp_buffer = ctypes.c_void_p()
        u_len = wintypes.UINT()

        success = self.version_dll.VerQueryValueW(
            self.version_info,
            r"\VarFileInfo\Translation",
            ctypes.byref(lp_buffer),
            ctypes.byref(u_len),
        )

        if not success or u_len.value == 0:
            return 0

        translations = []
        lang_id: int = 0
        if lp_buffer.value is not None:
            for i in range(u_len.value // 4):
                offset = i * 4
                data = ctypes.string_at(lp_buffer.value + offset, 4)
                lang_id = int.from_bytes(data[:2], "little")
                code_page = int.from_bytes(data[2:4], "little")
                translations.append((lang_id, code_page))
        else:
            # Handle the case where lp_buffer.value is None
            print("Buffer is None")

        return lang_id


@functools.cache
def check_msvc_cl_language_id(compiler: str) -> None:
    """
    Torch.compile() is only work on MSVC with English language pack well.
    Check MSVC's language pack: https://github.com/pytorch/pytorch/issues/157673#issuecomment-3051682766
    """

    def get_msvc_cl_path() -> tuple[bool, str]:
        """
        Finds the path to cl.exe using vswhere.exe.
        """
        vswhere_path = os.path.join(
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
            "Microsoft Visual Studio",
            "Installer",
            "vswhere.exe",
        )
        if not os.path.exists(vswhere_path):
            vswhere_path = os.path.join(
                os.environ.get("ProgramFiles", "C:\\Program Files"),
                "Microsoft Visual Studio",
                "Installer",
                "vswhere.exe",
            )
            if not os.path.exists(vswhere_path):
                return False, ""  # vswhere.exe not found

        try:
            # Get the Visual Studio installation path
            cmd = [
                vswhere_path,
                "-latest",
                "-prerelease",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ]
            vs_install_path = subprocess.check_output(
                cmd, text=True, encoding="utf-8"
            ).strip()

            if not vs_install_path:
                return False, ""

            # Find the latest MSVC toolset version within the installation
            msvc_tools_path = os.path.join(vs_install_path, "VC", "Tools", "MSVC")
            if not os.path.exists(msvc_tools_path):
                return False, ""

            # Get the latest toolset version directory
            toolset_versions = [
                d
                for d in os.listdir(msvc_tools_path)
                if os.path.isdir(os.path.join(msvc_tools_path, d))
            ]
            if not toolset_versions:
                return False, ""
            latest_toolset_version = sorted(toolset_versions, reverse=True)[0]

            # Construct the full cl.exe path
            cl_path = os.path.join(
                msvc_tools_path,
                latest_toolset_version,
                "bin",
                "HostX64",
                "x64",
                "cl.exe",
            )
            if os.path.exists(cl_path):
                return True, cl_path
            else:
                # Fallback for older versions or different architectures if needed
                cl_path = os.path.join(
                    msvc_tools_path,
                    latest_toolset_version,
                    "bin",
                    "HostX86",
                    "x86",
                    "cl.exe",
                )
                if os.path.exists(cl_path):
                    return True, cl_path

        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, ""

        return False, ""

    if not _is_msvc_cl(compiler):
        return

    if os.path.exists(compiler):
        # Passed compiler with path.
        cl_exe_path = compiler
    else:
        b_ret, cl_exe_path = get_msvc_cl_path()
        if b_ret is False:
            return

    version_info = WinPeFileVersionInfo(cl_exe_path)
    lang_id = version_info.get_language_id()
    if lang_id != 1033:
        # MSVC English language id is 0x0409, and the DEC value is 1033.
        raise RuntimeError(
            "Torch.compile() is only support MSVC with English language pack,"
            "Please reinstall its language pack to English."
        )


@functools.cache
def check_mingw_win32_flavor(compiler: str) -> str:
    """
    Check if MinGW `compiler` exists and return it's flavor (win32 or posix).
    """
    try:
        out = subprocess.check_output(
            _compiler_command(compiler, "-v"),
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Compiler: {compiler} is not found.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to run {compiler} -v") from e

    flavor: str | None = None
    for line in out.splitlines():
        if "Thread model" in line:
            flavor = line.split(":", 1)[-1].strip().lower()

    if flavor is None:
        raise RuntimeError(
            f"Cannot determine the flavor of {compiler} (win32 or posix). No Thread model found in {compiler} -v"
        )

    if flavor not in ("win32", "posix"):
        raise RuntimeError(
            f"Only win32 and pofix flavor of {compiler} is supported. The flavor is {flavor}"
        )

    return flavor


def get_cpp_compiler() -> str:
    if (
        config.aot_inductor.cross_target_platform == "windows"
        and sys.platform != "win32"
    ):
        # we're doing cross-compilation
        compiler = MINGW_GXX
        if not config.aot_inductor.package_cpp_only:
            check_mingw_win32_flavor(compiler)
        return compiler

    if _IS_WINDOWS:
        compiler = os.environ.get("CXX", "cl")
        compiler = normalize_path_separator(compiler)
        check_compiler_exist_windows(compiler)
        check_msvc_cl_language_id(compiler)
    else:
        if config.is_fbcode():
            return build_paths.cc
        if isinstance(config.cpp.cxx, (list, tuple)):
            search = tuple(config.cpp.cxx)
        else:
            search = (config.cpp.cxx,)
        compiler = cpp_compiler_search(search)
    return compiler


def get_ld_and_objcopy(use_relative_path: bool) -> tuple[str, str]:
    if _IS_WINDOWS:
        raise RuntimeError("Windows is not supported yet.")
    else:
        if config.is_fbcode():
            ld = build_paths.ld
            objcopy = (
                build_paths.objcopy_fallback
                if use_relative_path
                else build_paths.objcopy
            )
        else:
            ld = "ld"
            objcopy = "objcopy"
    return ld, objcopy


def convert_cubin_to_obj(
    cubin_file: str,
    kernel_name: str,
    ld: str,
    objcopy: str,
) -> str:
    obj_file = cubin_file + ".o"
    # Convert .cubin to .o
    cmd = f"{ld} -r -b binary -z noexecstack -o {obj_file} {cubin_file}"
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # Rename .data to .rodata
    cmd = f"{objcopy} --rename-section .data=.rodata,alloc,load,readonly,data,contents {obj_file}"
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # By default objcopy will create *_start, *_size, *_end symbols using the full path
    # Rename to use the unique kernel name
    file_name = re.sub(r"[\W]", "_", cubin_file)
    cmd = (
        objcopy
        + f" --redefine-sym _binary_{file_name}_start=__{kernel_name}_start "
        + f"--redefine-sym _binary_{file_name}_size=__{kernel_name}_size "
        + f"--redefine-sym _binary_{file_name}_end=__{kernel_name}_end "
        + obj_file
    )
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    return obj_file


def batch_convert_cubins_to_obj(
    cubins: list[tuple[str, str]],
    output_dir: str,
    cpp_compiler: str = "gcc",
) -> str:
    """Convert multiple cubin files to a single .o using batched .incbin assembly.

    Instead of spawning 3 subprocesses per cubin (ld + 2x objcopy), generates
    a single .S file with .incbin directives for all cubins and compiles it
    with one compiler invocation. Produces bit-identical rodata and symbols
    as the per-cubin convert_cubin_to_obj approach.

    Args:
        cubins: list of (cubin_file_path, kernel_name) tuples.
        output_dir: directory for the generated .S and .o files.
        cpp_compiler: C compiler to use for assembling (default: gcc).

    Returns:
        Path to the combined .o file.
    """
    asm_path = os.path.join(output_dir, "cubins_combined.S")
    obj_path = os.path.join(output_dir, "cubins_combined.o")

    with open(asm_path, "w") as f:
        f.write(".section .rodata\n")
        for cubin_file, kernel_name in cubins:
            # Use absolute path to avoid issues with working directory
            abs_cubin = os.path.abspath(cubin_file)
            escaped_path = abs_cubin.replace("\\", "\\\\").replace('"', '\\"')
            f.write(
                f".balign 16\n"
                f".global __{kernel_name}_start\n"
                f".global __{kernel_name}_end\n"
                f"__{kernel_name}_start:\n"
                f'.incbin "{escaped_path}"\n'
                f"__{kernel_name}_end:\n"
                f".global __{kernel_name}_size\n"
                f".set __{kernel_name}_size, "
                f"__{kernel_name}_end - __{kernel_name}_start\n"
            )

    subprocess.run(
        _compiler_command(cpp_compiler, "-c", asm_path, "-o", obj_path),
        capture_output=True,
        text=True,
        check=True,
    )
    return obj_path


@functools.cache
def _is_apple_clang(cpp_compiler: str) -> bool:
    first_line = _compiler_version_first_line(cpp_compiler)
    return bool(re.search(r"\bApple\b.*\bclang\b", first_line))


@functools.cache
def _is_clang(cpp_compiler: str) -> bool:
    if _IS_WINDOWS:
        # clang suite have many compilers, and only clang-cl is supported.
        if re.search(r"((clang$)|(clang\+\+$))", cpp_compiler):
            raise RuntimeError(
                "Please use clang-cl, due to torch.compile only support MSVC-like CLI (compiler flags syntax)."
            )
        return bool(re.search(r"(clang-cl)", cpp_compiler))

    if sys.platform == "darwin" and _is_apple_clang(cpp_compiler):
        return True

    first_line = _compiler_version_first_line(cpp_compiler)
    if "Intel" not in first_line and re.search(
        r"(^|[\s/-])clang version\b", first_line
    ):
        return True

    return bool(re.search(r"(^|[/\s-])(clang\+\+|clang)(?=[\s-]|$)", cpp_compiler))


@functools.cache
def _is_gcc(cpp_compiler: str) -> bool:
    # Since "clang++" ends with "g++", the regex match below would validate on it.
    if _is_clang(cpp_compiler):
        return False

    first_line = _compiler_version_first_line(cpp_compiler)
    if re.search(
        r"(^|[\s/-])(gcc|g\+\+|gnu-c\+\+)(?=[\s(-]|$)",
        first_line,
        re.IGNORECASE,
    ):
        return True

    return bool(re.search(r"(^|[/\s-])(gcc|g\+\+|gnu-c\+\+)(?=[\s-]|$)", cpp_compiler))


@functools.cache
def _is_gcc_version_less_than(cpp_compiler: str, major: int) -> bool:
    if not _is_gcc(cpp_compiler):
        return False

    try:
        output_msg = (
            subprocess.check_output(
                _compiler_command(cpp_compiler, "-dumpfullversion", "-dumpversion"),
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

    version_search = re.search(r"\d+(?:\.\d+)*", output_msg)
    if version_search is None:
        return False

    return TorchVersion(version_search.group(0)) < TorchVersion(str(major))


@functools.cache
def _is_msvc_cl(cpp_compiler: str) -> bool:
    if not _IS_WINDOWS:
        return False

    try:
        result = subprocess.run(
            _compiler_command(cpp_compiler, "/help"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        lines = (result.stdout or b"").strip().splitlines()

        return bool(lines) and b"Microsoft" in lines[0]

    except OSError:
        return False


@functools.cache
def _is_intel_compiler(cpp_compiler: str) -> bool:
    def _check_minimal_version(compiler_version: TorchVersion) -> None:
        """
        On Windows: early version icx has `-print-file-name` issue, and can't preload correctly for inductor.
        """
        min_version = "2024.2.1" if _IS_WINDOWS else "0.0.0"
        if compiler_version < TorchVersion(min_version):
            raise RuntimeError(
                f"Intel Compiler error: less than minimal version {min_version}."
            )

    try:
        output_msg = _compiler_version_string(cpp_compiler)
        lines = output_msg.splitlines()
        is_intel_compiler = bool(lines) and "Intel" in lines[0]
        if is_intel_compiler:
            if _IS_WINDOWS:
                if re.search(r"((icx$)|(icx-cc$))", cpp_compiler):
                    raise RuntimeError(
                        "Please use icx-cl, due to torch.compile only support MSVC-like CLI (compiler flags syntax)."
                    )

            # Version check
            icx_ver_search = re.search(r"(\d+[.]\d+[.]\d+[.]\d+)", output_msg)
            if icx_ver_search is not None:
                icx_ver = icx_ver_search.group(1)
                _check_minimal_version(TorchVersion(icx_ver))

        return is_intel_compiler
    except FileNotFoundError:
        return False
    except subprocess.SubprocessError:
        # --version args not support.
        return False

    # pyrefly: ignore [unreachable]
    return False


@functools.cache
def is_gcc() -> bool:
    return _is_gcc(get_cpp_compiler())


@functools.cache
def is_clang() -> bool:
    return _is_clang(get_cpp_compiler())


@functools.cache
def is_intel_compiler() -> bool:
    return _is_intel_compiler(get_cpp_compiler())


@functools.cache
def is_apple_clang() -> bool:
    return _is_apple_clang(get_cpp_compiler())


@functools.cache
def is_msvc_cl() -> bool:
    return _is_msvc_cl(get_cpp_compiler())


@functools.cache
def get_compiler_version_info(compiler: str) -> str:
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # Don't localize output
    try:
        version_string = subprocess.check_output(
            _compiler_command(compiler, "-v"),
            stderr=subprocess.STDOUT,
            env=env,
        ).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception:
        try:
            version_string = subprocess.check_output(
                _compiler_command(compiler, "--version"),
                stderr=subprocess.STDOUT,
                env=env,
            ).decode(*SUBPROCESS_DECODE_ARGS)
        except Exception:
            return ""
    # Multiple lines to one line string.
    version_string = version_string.replace("\r", "_")
    version_string = version_string.replace("\n", "_")
    return version_string


# =============================== cpp builder ===============================
def _append_list(dest_list: list[str], src_list: list[str]) -> None:
    dest_list.extend(copy.deepcopy(item) for item in src_list)


def _remove_duplication_in_list(orig_list: list[str]) -> list[str]:
    new_list: list[str] = []
    for item in orig_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def _create_if_dir_not_exist(path_dir: str) -> None:
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError(f"Fail to create path {path_dir}") from exc


def _remove_dir(path_dir: str) -> None:
    if os.path.exists(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
        os.rmdir(path_dir)


def _run_compile_cmd(cmd_line: str, cwd: str) -> None:
    cmd = shlex.split(cmd_line)
    try:
        subprocess.run(
            cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        output = e.stdout.decode(*SUBPROCESS_DECODE_ARGS)
        openmp_problem = "'omp.h' file not found" in output or "libomp" in output
        if openmp_problem and sys.platform == "darwin":
            instruction = (
                "\n\nOpenMP support not found. Please try one of the following solutions:\n"
                "(1) Set the `CXX` environment variable to a compiler other than Apple clang++/g++ "
                "that has builtin OpenMP support;\n"
                "(2) install OpenMP via conda: `conda install llvm-openmp`;\n"
                "(3) install libomp via brew: `brew install libomp`;\n"
                "(4) manually setup OpenMP and set the `OMP_PREFIX` environment variable to point to a path"
                " with `include/omp.h` under it."
            )
            output += instruction
        raise exc.CppCompileError(cmd, output) from e


def run_compile_cmd(cmd_line: str, cwd: str) -> None:
    with dynamo_timed("compile_file"):
        _run_compile_cmd(cmd_line, cwd)


def normalize_path_separator(orig_path: str) -> str:
    if _IS_WINDOWS:
        return orig_path.replace(os.sep, "/")
    return orig_path


class BuildOptionsBase:
    """
    This is the Base class for store cxx build options, as a template.
    Actually, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    """

    def __init__(
        self,
        compiler: str = "",
        definitions: list[str] | None = None,
        include_dirs: list[str] | None = None,
        cflags: list[str] | None = None,
        ldflags: list[str] | None = None,
        libraries_dirs: list[str] | None = None,
        libraries: list[str] | None = None,
        passthrough_args: list[str] | None = None,
        aot_mode: bool = False,
        use_relative_path: bool = False,
        compile_only: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        self._compiler = compiler
        self._definitions: list[str] = definitions or []
        self._include_dirs: list[str] = include_dirs or []
        self._cflags: list[str] = cflags or []
        self._ldflags: list[str] = ldflags or []
        self._libraries_dirs: list[str] = libraries_dirs or []
        self._libraries: list[str] = libraries or []
        # Some args are hard to abstract to OS compatible, passthrough directly.
        self._passthrough_args: list[str] = passthrough_args or []

        # Optionally, the path to a precompiled header which should be included on the
        # build command line.
        self.precompiled_header: str | None = None

        self._aot_mode: bool = aot_mode
        self._use_relative_path: bool = use_relative_path
        self._compile_only: bool = compile_only
        self._precompiling: bool = precompiling
        self._preprocessing: bool = preprocessing

    def _process_compile_only_options(self) -> None:
        if self._compile_only or self._precompiling or self._preprocessing:
            self._libraries_dirs = []
            self._libraries = []
            self._ldflags = []

    def _remove_duplicate_options(self) -> None:
        self._definitions = _remove_duplication_in_list(self._definitions)
        self._include_dirs = _remove_duplication_in_list(self._include_dirs)
        self._cflags = _remove_duplication_in_list(self._cflags)
        self._ldflags = _remove_duplication_in_list(self._ldflags)
        self._libraries_dirs = _remove_duplication_in_list(self._libraries_dirs)
        self._libraries = _remove_duplication_in_list(self._libraries)
        self._passthrough_args = _remove_duplication_in_list(self._passthrough_args)

    def _finalize_options(self) -> None:
        self._process_compile_only_options()
        self._remove_duplicate_options()

    def get_compiler(self) -> str:
        return self._compiler

    def get_definitions(self) -> list[str]:
        return self._definitions

    def get_include_dirs(self) -> list[str]:
        return self._include_dirs

    def get_cflags(self) -> list[str]:
        return self._cflags

    def get_ldflags(self) -> list[str]:
        return self._ldflags

    def get_libraries_dirs(self) -> list[str]:
        return self._libraries_dirs

    def get_libraries(self) -> list[str]:
        return self._libraries

    def get_passthrough_args(self) -> list[str]:
        return self._passthrough_args

    def get_aot_mode(self) -> bool:
        return self._aot_mode

    def get_use_relative_path(self) -> bool:
        return self._use_relative_path

    def get_compile_only(self) -> bool:
        return self._compile_only

    def get_precompiling(self) -> bool:
        return self._precompiling

    def get_preprocessing(self) -> bool:
        return self._preprocessing

    def save_flags_to_json(self, file: str) -> None:
        attrs = {
            "compiler": self.get_compiler(),
            "definitions": self.get_definitions(),
            "include_dirs": self.get_include_dirs(),
            "cflags": self.get_cflags(),
            "ldflags": self.get_ldflags(),
            "libraries_dirs": self.get_libraries_dirs(),
            "libraries": self.get_libraries(),
            "passthrough_args": self.get_passthrough_args(),
            "aot_mode": self.get_aot_mode(),
            "use_relative_path": self.get_use_relative_path(),
            "compile_only": self.get_compile_only(),
        }

        with open(file, "w") as f:
            json.dump(attrs, f)


def _get_warning_all_cflag(warning_all: bool = True) -> list[str]:
    if not _IS_WINDOWS:
        return ["Wall"] if warning_all else []
    else:
        return []


def _get_cpp_std_cflag(std_num: str = "c++20") -> list[str]:
    if _IS_WINDOWS:
        """
        On Windows, only c++20 can support `std::enable_if_t`.
        Ref: https://learn.microsoft.com/en-us/cpp/overview/cpp-conformance-improvements-2019?view=msvc-170#checking-for-abstract-class-types # noqa: B950
        Note:
            Only setup c++20 for Windows inductor. I tried to upgrade all project to c++20, but it is failed:
            https://github.com/pytorch/pytorch/pull/131504
        """
        std_num = "c++20"
        return [f"std:{std_num}"]
    else:
        return [f"std={std_num}"]


def _get_os_related_cpp_cflags(cpp_compiler: str) -> list[str]:
    if _IS_WINDOWS:
        cflags = [
            "wd4819",
            "wd4251",
            "wd4244",
            "wd4267",
            "wd4275",
            "wd4018",
            "wd4190",
            "wd4624",
            "wd4067",
            "wd4068",
            "EHsc",
            # For Intel oneAPI, ref: https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus?view=msvc-170
            "Zc:__cplusplus",
            # Enable max compatible to msvc for oneAPI headers.
            # ref: https://github.com/pytorch/pytorch/blob/db38c44ad639e7ada3e9df2ba026a2cb5e40feb0/cmake/public/utils.cmake#L352-L358
            "permissive-",
        ]
    else:
        cflags = ["Wno-unused-variable", "Wno-unknown-pragmas"]
        if _is_clang(cpp_compiler):
            ignored_optimization_argument = (
                "Werror=ignored-optimization-argument"
                if config.aot_inductor.raise_error_on_ignored_optimization
                else "Wno-ignored-optimization-argument"
            )
            cflags.append(ignored_optimization_argument)
        if _is_gcc(cpp_compiler):
            # Issue all the warnings demanded by strict ISO C and ISO C++.
            # Ref: https://github.com/pytorch/pytorch/issues/153180#issuecomment-2986676878
            cflags.append("pedantic")
    return cflags


def _get_os_related_cpp_definitions(cpp_compiler: str) -> list[str]:
    os_definitions: list[str] = []
    if _IS_WINDOWS:
        # On Windows, we need disable min/max macro to avoid C2589 error, as PyTorch CMake:
        # https://github.com/pytorch/pytorch/blob/9a41570199155eee92ebd28452a556075e34e1b4/CMakeLists.txt#L1118-L1119
        os_definitions.append("NOMINMAX")
    return os_definitions


def _get_ffast_math_flags() -> list[str]:
    if _IS_WINDOWS:
        flags = []
    else:
        # This starts from the flags implied by -ffast-math, as in
        # https://github.com/gcc-mirror/gcc/blob/4700ad1c78ccd7767f846802fca148b2ea9a1852/gcc/opts.cc#L3458-L3468
        # however gcc<13 sets the FTZ/DAZ flags for runtime on x86 even if we have
        # -ffast-math -fno-unsafe-math-optimizations because the flags for runtime
        # are added by linking in crtfastmath.o. This is done by the spec file which
        # only does globbing for -ffast-math.
        flags = [
            "fno-trapping-math",
            "funsafe-math-optimizations",
            "ffinite-math-only",
            "fno-signed-zeros",
        ]

        flags.append("fno-finite-math-only")
        if not config.cpp.enable_unsafe_math_opt_flag:
            flags.append("fno-unsafe-math-optimizations")
        # Keep errno-preserving libm semantics.  With -fno-math-errno, GCC can
        # inline/transform libm call pairs like sin(atan(x)) in ways that do not
        # preserve NaN values (see https://github.com/pytorch/pytorch/issues/143978).
        flags.append("fmath-errno")
        flags.append(f"ffp-contract={config.cpp.enable_floating_point_contract_flag}")

        if is_gcc():
            flags.append("fexcess-precision=fast")

    return flags


def _get_inductor_debug_symbol_cflags() -> tuple[list[str], list[str]]:
    """
    When we turn on generate debug symbol.
    On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
    On Linux, it should create some debug sections in binary file.
    """
    cflags: list[str] = []
    ldflags: list[str] = []

    if _IS_WINDOWS:
        cflags = ["ZI", "_DEBUG"]
        ldflags = ["DEBUG", "ASSEMBLYDEBUG ", "OPT:REF", "OPT:ICF"]
    else:
        cflags.append("g")

    return cflags, ldflags


@functools.cache
def _get_linux_aarch64_cpu_flags() -> OrderedSet[str]:
    flags: OrderedSet[str] = OrderedSet()

    if platform.machine() not in ("aarch64", "arm64"):
        return flags

    if not sys.platform.startswith("linux"):
        return flags

    capabilities = torch.cpu.get_capabilities()
    flags.update(
        capability
        for capability in ("bf16", "sve", "sve2")
        if capabilities.get(capability, False)
    )

    return flags


@functools.cache
def _get_linux_aarch64_arch_flag(cpp_compiler: str) -> str:
    flags = _get_linux_aarch64_cpu_flags()

    if _is_gcc(cpp_compiler) and _is_gcc_version_less_than(cpp_compiler, 13):
        if OrderedSet(["bf16", "sve", "sve2"]).issubset(flags):
            return "march=armv8.6-a+sve+sve2+bf16"

        if OrderedSet(["bf16", "sve"]).issubset(flags):
            return "march=armv8.6-a+sve+bf16"

    return "march=native"


def _get_cpu_arch_cflags(cpp_compiler: str) -> list[str]:
    if config.is_fbcode():
        return []

    march = config.cpp.march
    if march == "":
        return []

    # -march=native is not recognized on Apple Silicon, so the default macOS
    # behavior is no architecture flag unless the user explicitly configures one.
    if sys.platform == "darwin" and march is None:
        return []

    machine = platform.machine()
    if march is None:
        if machine == "ppc64le":
            return ["mcpu=native"]
        if machine == "riscv64":
            return ["march=rv64gc"]
        if machine == "riscv32":
            return ["march=rv32gc"]
        if machine in ("aarch64", "arm64"):
            return [_get_linux_aarch64_arch_flag(cpp_compiler)]
        return ["march=native"]

    if machine == "ppc64le":
        return [f"mcpu={march}"]
    return [f"march={march}"]


def _get_optimization_cflags(
    cpp_compiler: str, min_optimize: bool = False
) -> tuple[list[str], list[str]]:
    cflags: list[str] = []
    ldflags: list[str] = []

    should_use_optimized_flags = not (
        config.aot_inductor.debug_compile
        or os.environ.get("TORCHINDUCTOR_DEBUG_COMPILE", "0") == "1"
    )
    should_add_debug_symbol_flags = (
        config.aot_inductor.debug_compile
        or config.aot_inductor.debug_symbols
        or os.environ.get("TORCHINDUCTOR_DEBUG_COMPILE", "0") == "1"
        or os.environ.get("TORCHINDUCTOR_DEBUG_SYMBOL", "0") == "1"
    )
    if should_use_optimized_flags:
        if _IS_WINDOWS:
            cflags += ["O1" if min_optimize else "O2"]
        else:
            cflags += [
                config.aot_inductor.compile_wrapper_opt_level if min_optimize else "O3",
                "DNDEBUG",
            ]
    else:
        if _IS_WINDOWS:
            cflags += ["Od", "Ob0", "Oy-"]
        else:
            cflags += ["O0"]

    if should_add_debug_symbol_flags:
        debug_cflags, debug_ldflags = _get_inductor_debug_symbol_cflags()
        cflags += debug_cflags
        ldflags += debug_ldflags

    if config.aot_inductor.enable_frame_pointer:
        if _IS_WINDOWS:
            cflags.append("Oy-")
        else:
            cflags.append("fno-omit-frame-pointer")

    if config.aot_inductor.enable_line_tables and not should_add_debug_symbol_flags:
        if not _IS_WINDOWS:
            if _is_clang(cpp_compiler):
                cflags.append("gline-tables-only")
            else:
                cflags.append("g1")

    cflags += _get_ffast_math_flags()

    if _IS_WINDOWS:
        pass
    else:
        # on macos, unknown argument: '-fno-tree-loop-vectorize'
        if sys.platform != "darwin" and _is_gcc(cpp_compiler):
            cflags.append("fno-tree-loop-vectorize")
        cflags += _get_cpu_arch_cflags(cpp_compiler)

        if config.aot_inductor.enable_lto and _is_clang(cpp_compiler):
            cflags.append("flto=thin")

    return cflags, ldflags


def _get_shared_cflags(cpp_compiler: str, do_link: bool) -> list[str]:
    if _IS_WINDOWS:
        """
        MSVC `/MD` using python `ucrtbase.dll` lib as runtime.
        https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=msvc-170
        """
        return ["DLL", "MD"]
    if platform.system() == "Darwin" and _is_clang(cpp_compiler):
        # This causes undefined symbols to behave the same as linux
        return ["shared", "fPIC", "undefined dynamic_lookup"]
    flags = []
    if do_link:
        flags.append("shared")

    flags.append("fPIC")
    return flags


def get_cpp_options(
    cpp_compiler: str,
    do_link: bool,
    warning_all: bool = True,
    extra_flags: Sequence[str] = (),
    min_optimize: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []

    opt_cflags, opt_ldflags = _get_optimization_cflags(cpp_compiler, min_optimize)

    cflags = (
        opt_cflags
        + _get_shared_cflags(cpp_compiler, do_link)
        + _get_warning_all_cflag(warning_all)
        + _get_cpp_std_cflag()
        + _get_os_related_cpp_cflags(cpp_compiler)
    )

    definitions += _get_os_related_cpp_definitions(cpp_compiler)

    if not _IS_WINDOWS and config.aot_inductor.enable_lto and _is_clang(cpp_compiler):
        ldflags.append("fuse-ld=lld")
        ldflags.append("flto=thin")

    passthrough_args.append(" ".join(extra_flags))

    if config.aot_inductor.cross_target_platform == "windows":
        passthrough_args.extend(["-static-libstdc++", "-static-libgcc"])
        if check_mingw_win32_flavor(MINGW_GXX) == "posix":
            # winpthread provides clock_gettime, referenced by static libstdc++'s
            # chrono. The driver places libstdc++ after these args, so force the
            # symbol undefined up front to pull winpthread in regardless of order.
            passthrough_args.append(
                "-Wl,-u,clock_gettime -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic"
            )

    return (
        definitions,
        include_dirs,
        cflags,
        ldflags + opt_ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


class CppOptions(BuildOptionsBase):
    """
    This class is inherited from BuildOptionsBase, and as cxx build options.
    This option need contains basic cxx build option, which contains:
    1. OS related args.
    2. Toolchains related args.
    3. Cxx standard related args.
    Note:
    1. This Options is good for assist modules build, such as x86_isa_help.
    """

    def __init__(
        self,
        compile_only: bool = False,
        warning_all: bool = True,
        extra_flags: Sequence[str] = (),
        use_relative_path: bool = False,
        compiler: str = "",
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            use_relative_path=use_relative_path,
            precompiling=precompiling,
            preprocessing=preprocessing,
        )
        self._compiler = compiler if compiler else get_cpp_compiler()

        (
            definitions,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthrough_args,
        ) = get_cpp_options(
            cpp_compiler=self._compiler,
            do_link=not (compile_only or precompiling or preprocessing),
            extra_flags=extra_flags,
            warning_all=warning_all,
            min_optimize=min_optimize,
        )

        _append_list(self._definitions, definitions)
        _append_list(self._include_dirs, include_dirs)
        _append_list(self._cflags, cflags)
        _append_list(self._ldflags, ldflags)
        _append_list(self._libraries_dirs, libraries_dirs)
        _append_list(self._libraries, libraries)
        _append_list(self._passthrough_args, passthrough_args)
        self._finalize_options()


def _get_torch_cpp_wrapper_definition() -> list[str]:
    defs = ["TORCH_INDUCTOR_CPP_WRAPPER", "STANDALONE_TORCH_HEADER"]
    if config.cpp_cache_precompile_headers:
        defs.append("TORCH_INDUCTOR_PRECOMPILE_HEADERS")
    return defs


def _use_custom_generated_macros() -> list[str]:
    return [" C10_USING_CUSTOM_GENERATED_MACROS"]


def _use_fb_internal_macros() -> list[str]:
    if not _IS_WINDOWS:
        if config.is_fbcode():
            fb_internal_macros = [
                "C10_USE_GLOG",
                "C10_USE_MINIMAL_GLOG",
                "C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
            ]
            if platform.machine() == "x86_64":
                fb_internal_macros += [
                    "ATEN_MKL_ENABLED_FBCODE=1",
                    "ATEN_MKLDNN_ENABLED_FBCODE=1",
                ]
            return fb_internal_macros
        else:
            return []
    else:
        return []


def _setup_standard_sys_libs(
    cpp_compiler: str,
    aot_mode: bool,
    use_relative_path: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    cflags: list[str] = []
    include_dirs: list[str] = []
    passthrough_args: list[str] = []
    ldflags: list[str] = []
    if _IS_WINDOWS:
        return cflags, include_dirs, passthrough_args, ldflags

    if config.is_fbcode():
        # TODO(T203137008) Can we unify these flags with triton_cc_command?
        cflags.append("nostdinc")
        # Note that the order of include paths do matter, as a result
        # we need to have several branches interleaved here
        include_dirs.append(build_paths.sleef_include)
        include_dirs.append(build_paths.openmp_include)
        include_dirs.append(build_paths.python_include)
        include_dirs.append(build_paths.cc_include)
        include_dirs.append(build_paths.libgcc_include)
        include_dirs.append(build_paths.libgcc_arch_include)
        include_dirs.append(build_paths.libgcc_backward_include)
        include_dirs.append(build_paths.glibc_include)
        include_dirs.append(build_paths.linux_kernel_include)
        include_dirs.append("include")

        if aot_mode and not use_relative_path:
            linker_script = _LINKER_SCRIPT
        else:
            linker_script = os.path.basename(_LINKER_SCRIPT)

        if _is_clang(cpp_compiler):
            passthrough_args.append(" --rtlib=compiler-rt")
            passthrough_args.append(" -B" + build_paths.glibc_lib)
            ldflags.append("fuse-ld=lld")
            ldflags.append(f"Wl,--script={linker_script}")
            ldflags.append("L" + build_paths.glibc_lib)

    return cflags, include_dirs, passthrough_args, ldflags


def _get_build_args_of_chosen_isa(vec_isa: VecISA) -> tuple[list[str], list[str]]:
    macros: list[str] = []
    build_flags: list[str] = []
    if vec_isa != invalid_vec_isa:
        # Add Windows support later.
        macros.extend(copy.deepcopy(x) for x in vec_isa.build_macro())

        build_flags = [vec_isa.build_arch_flags()]

        if config.is_fbcode():
            cap = str(vec_isa).upper()
            macros = [
                f"CPU_CAPABILITY={cap}",
                f"CPU_CAPABILITY_{cap}",
                f"HAVE_{cap}_CPU_DEFINITION",
            ]

    return macros, build_flags


def _get_torch_related_args(
    include_pytorch: bool, aot_mode: bool
) -> tuple[list[str], list[str], list[str]]:
    from torch.utils.cpp_extension import include_paths, TORCH_LIB_PATH

    libraries = []
    include_dirs = include_paths()

    if config.aot_inductor.link_libtorch:
        libraries_dirs = [TORCH_LIB_PATH]
        if sys.platform != "darwin" and not config.is_fbcode():
            libraries.extend(["torch", "torch_cpu"])
            if _IS_WINDOWS:
                libraries.append("c10")
            if not aot_mode:
                libraries.append("torch_python")
    else:
        libraries_dirs = []
        if config.aot_inductor.cross_target_platform == "windows":
            aoti_shim_library = config.aot_inductor.aoti_shim_library

            if not aoti_shim_library:
                raise AssertionError(
                    "'config.aot_inductor.aoti_shim_library' must be set when 'cross_target_platform' is 'windows'."
                )
            if isinstance(aoti_shim_library, str):
                libraries.append(aoti_shim_library)
            else:
                if not isinstance(aoti_shim_library, list):
                    raise AssertionError(
                        f"expected aoti_shim_library to be a list, got {type(aoti_shim_library)}"
                    )
                libraries.extend(aoti_shim_library)

    if config.aot_inductor.cross_target_platform == "windows":
        if not config.aot_inductor.aoti_shim_library_path:
            raise AssertionError(
                (
                    "'config.aot_inductor.aoti_shim_library_path' must be set to the path of the AOTI shim library",
                    " when 'cross_target_platform' is 'windows'.",
                )
            )
        libraries_dirs.append(config.aot_inductor.aoti_shim_library_path)

    if _IS_WINDOWS:
        libraries.append("sleef")

    return include_dirs, libraries_dirs, libraries


def _get_python_include_dirs() -> list[str]:
    include_dir = Path(sysconfig.get_path("include"))
    # On Darwin Python executable from a framework can return
    # non-existing /Library/Python/... include path, in which case
    # one should use Headers folder from the framework
    if not include_dir.exists() and platform.system() == "Darwin":
        std_lib = Path(sysconfig.get_path("stdlib"))
        include_dir = (std_lib.parent.parent / "Headers").absolute()
    if not (include_dir / "Python.h").exists():
        warnings.warn(f"Can't find Python.h in {str(include_dir)}")
    return [str(include_dir)]


def _get_python_related_args() -> tuple[list[str], list[str]]:
    python_include_dirs = _get_python_include_dirs()
    python_include_path = sysconfig.get_path(
        "include", scheme="nt" if _IS_WINDOWS else "posix_prefix"
    )
    if python_include_path is not None:
        python_include_dirs.append(python_include_path)

    if _IS_WINDOWS:
        python_lib_path = [
            str(
                (
                    Path(sysconfig.get_path("include", scheme="nt")).parent / "libs"
                ).absolute()
            )
        ]
    else:
        python_lib_path = [sysconfig.get_config_var("LIBDIR")]

    if config.is_fbcode():
        python_include_dirs.append(build_paths.python_include)

    return python_include_dirs, python_lib_path


@functools.cache
def is_conda_llvm_openmp_installed() -> bool:
    try:
        command = "conda list llvm-openmp --json"
        output = subprocess.check_output(command.split()).decode("utf8")
        return len(json.loads(output)) > 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@functools.cache
def homebrew_libomp() -> tuple[bool, str]:
    try:
        # check if `brew` is installed
        if shutil.which("brew") is None:
            return False, ""
        # get the location of `libomp` if it is installed
        # this is the location that `libomp` **would** be installed
        # see https://github.com/Homebrew/brew/issues/10261#issuecomment-756563567 for details
        libomp_path = (
            subprocess.check_output(["brew", "--prefix", "libomp"])
            .decode("utf8")
            .strip()
        )
        # check if `libomp` is installed
        omp_available = os.path.exists(libomp_path)
        return omp_available, libomp_path
    except subprocess.SubprocessError:
        return False, ""


@functools.cache
def perload_clang_libomp_win(cpp_compiler: str, omp_name: str) -> None:
    try:
        output = subprocess.check_output(
            _compiler_command(cpp_compiler, "-print-file-name=bin")
        ).decode("utf8")
        omp_path = os.path.join(output.rstrip(), omp_name)
        if os.path.isfile(omp_path):
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            cdll.LoadLibrary(omp_path)
    except subprocess.SubprocessError:
        pass


@functools.cache
def perload_icx_libomp_win(cpp_compiler: str) -> None:
    def _load_icx_built_in_lib_by_name(cpp_compiler: str, lib_name: str) -> bool:
        try:
            output = subprocess.check_output(
                _compiler_command(cpp_compiler, f"-print-file-name={lib_name}"),
                stderr=subprocess.DEVNULL,
            ).decode(*SUBPROCESS_DECODE_ARGS)
            omp_path = output.rstrip()
            if os.path.isfile(omp_path):
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                cdll.LoadLibrary(omp_path)
                return True
        except subprocess.SubprocessError:
            pass
        return False

    """
    Intel Compiler implemented more math libraries than clang, for performance purposes.
    We need to preload them like openmp library.
    """
    preload_list = [
        "libiomp5md.dll",  # openmp
        "svml_dispmd.dll",  # svml library
        "libmmd.dll",  # libm
    ]

    for lib_name in preload_list:
        _load_icx_built_in_lib_by_name(cpp_compiler, lib_name)


def _get_openmp_args(
    cpp_compiler: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    cflags: list[str] = []
    ldflags: list[str] = []
    include_dir_paths: list[str] = []
    lib_dir_paths: list[str] = []
    libs: list[str] = []
    passthrough_args: list[str] = []

    if config.aot_inductor.cross_target_platform == "windows":
        return cflags, ldflags, include_dir_paths, lib_dir_paths, libs, passthrough_args
    if _IS_MACOS:
        # Per https://mac.r-project.org/openmp/ right way to pass `openmp` flags to MacOS is via `-Xclang`
        cflags.append("Xclang")
        cflags.append("fopenmp")

        # only Apple builtin compilers (Apple Clang++) require openmp
        omp_available = not _is_apple_clang(cpp_compiler)

        # check the `OMP_PREFIX` environment first
        omp_prefix = os.getenv("OMP_PREFIX")
        if omp_prefix is not None:
            header_path = os.path.join(omp_prefix, "include", "omp.h")
            valid_env = os.path.exists(header_path)
            if valid_env:
                include_dir_paths.append(os.path.join(omp_prefix, "include"))
                lib_dir_paths.append(os.path.join(omp_prefix, "lib"))
            else:
                warnings.warn("environment variable `OMP_PREFIX` is invalid.")
            omp_available = omp_available or valid_env

        if not omp_available:
            libs.append("omp")

        # prefer to use openmp from `conda install llvm-openmp`
        conda_prefix = os.getenv("CONDA_PREFIX")
        if not omp_available and conda_prefix is not None:
            omp_available = is_conda_llvm_openmp_installed()
            if omp_available:
                conda_lib_path = os.path.join(conda_prefix, "lib")
                include_dir_paths.append(os.path.join(conda_prefix, "include"))
                lib_dir_paths.append(conda_lib_path)
                # Prefer Intel OpenMP on x86 machine
                if os.uname().machine == "x86_64" and os.path.exists(
                    os.path.join(conda_lib_path, "libiomp5.dylib")
                ):
                    libs.append("iomp5")

        # next, try to use openmp from `brew install libomp`
        if not omp_available:
            omp_available, libomp_path = homebrew_libomp()
            if omp_available:
                include_dir_paths.append(os.path.join(libomp_path, "include"))
                lib_dir_paths.append(os.path.join(libomp_path, "lib"))

        # if openmp is still not available, we let the compiler to have a try,
        # and raise error together with instructions at compilation error later
    elif _IS_WINDOWS:
        """
        On Windows, `clang` and `icx` have their specific openmp implenmention.
        And the openmp lib is in compiler's some sub-directory.
        For dynamic library(DLL) load, the Windows native APIs are `LoadLibraryA` and `LoadLibraryExA`, and their search
        dependencies have some rules:
        https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa#searching-for-dlls-and-dependencies
        In some case, the rules may not include compiler's sub-directories.
        So, it can't search and load compiler's openmp library correctly.
        And then, the whole application would be broken.

        To avoid the openmp load failed, we can automatic locate the openmp binary and preload it.
        1. For clang, the function is `perload_clang_libomp_win`.
        2. For icx, the function is `perload_icx_libomp_win`.
        """
        if _is_clang(cpp_compiler):
            cflags.append("openmp")
            libs.append("libomp")
            perload_clang_libomp_win(cpp_compiler, "libomp.dll")
        elif _is_intel_compiler(cpp_compiler):
            cflags.append("Qiopenmp")
            libs.append("libiomp5md")
            perload_icx_libomp_win(cpp_compiler)
        else:
            # /openmp, /openmp:llvm
            # llvm on Windows, new openmp: https://devblogs.microsoft.com/cppblog/msvc-openmp-update/
            # msvc openmp: https://learn.microsoft.com/zh-cn/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170
            cflags.append("openmp")
            cflags.append("openmp:experimental")  # MSVC CL
    else:
        if config.is_fbcode():
            include_dir_paths.append(build_paths.openmp_include)

            passthrough_args.append("-Wp,-fopenmp")
            lib_dir_paths.append(os.path.dirname(build_paths.openmp_lib_so))

            libs.append("omp")
        else:
            if _is_clang(cpp_compiler):
                # TODO: fix issue, can't find omp.h
                cflags.append("fopenmp")
                libs.append("gomp")
            elif _is_intel_compiler(cpp_compiler):
                cflags.append("fiopenmp")
            else:
                cflags.append("fopenmp")
                libs.append("gomp")

    return cflags, ldflags, include_dir_paths, lib_dir_paths, libs, passthrough_args


def _get_libstdcxx_args() -> tuple[list[str], list[str]]:
    """
    For fbcode cpu case, we should link stdc++ instead assuming the binary where dlopen is executed is built with dynamic stdc++.
    """
    lib_dir_paths: list[str] = []
    libs: list[str] = []
    if config.is_fbcode():
        lib_dir_paths = [sysconfig.get_config_var("LIBDIR")]
        libs.append("stdc++")

    return lib_dir_paths, libs


def get_mmap_self_macro(
    use_mmap_weights: bool, use_mmap_weights_external: bool
) -> list[str]:
    macros = []

    if use_mmap_weights and use_mmap_weights_external:
        raise RuntimeError(
            "Only one of use_mmap_weights and use_mmap_weights_external should be true"
        )
    if use_mmap_weights:
        macros.append(" USE_MMAP_SELF")
    elif use_mmap_weights_external:
        macros.append(" USE_MMAP_EXTERNAL")
    return macros


def get_caching_allocator_macro() -> list[str]:
    from torch._inductor import config

    macros = []
    if config.aot_inductor.weight_use_caching_allocator:
        macros.append(" AOT_INDUCTOR_USE_CACHING_ALLOCATOR")
    return macros


def get_cpp_torch_options(
    cpp_compiler: str,
    vec_isa: VecISA,
    include_pytorch: bool,
    aot_mode: bool,
    use_relative_path: bool,
    use_mmap_weights: bool,
    use_mmap_weights_external: bool,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    """
    This function is used to get the build args of torch related build options.
    1. Torch include_directories, libraries, libraries_directories.
    2. Python include_directories, libraries, libraries_directories.
    3. OpenMP related.
    4. Torch MACROs.
    5. MISC
    6. Return the build args
    """
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []

    torch_cpp_wrapper_definitions = _get_torch_cpp_wrapper_definition()
    use_custom_generated_macros_definitions = _use_custom_generated_macros()

    (
        sys_libs_cflags,
        sys_libs_include_dirs,
        sys_libs_passthrough_args,
        sys_libs_ldflags,
    ) = _setup_standard_sys_libs(cpp_compiler, aot_mode, use_relative_path)

    isa_macros, isa_ps_args_build_flags = _get_build_args_of_chosen_isa(vec_isa)

    (
        torch_include_dirs,
        torch_libraries_dirs,
        torch_libraries,
    ) = _get_torch_related_args(include_pytorch=include_pytorch, aot_mode=aot_mode)

    python_include_dirs, python_libraries_dirs = _get_python_related_args()

    (
        omp_cflags,
        omp_ldflags,
        omp_include_dir_paths,
        omp_lib_dir_paths,
        omp_lib,
        omp_passthrough_args,
    ) = _get_openmp_args(cpp_compiler)

    fb_macro_passthrough_args = _use_fb_internal_macros()

    mmap_self_macros = get_mmap_self_macro(use_mmap_weights, use_mmap_weights_external)
    caching_allocator_macros = get_caching_allocator_macro()

    definitions = (
        torch_cpp_wrapper_definitions
        + use_custom_generated_macros_definitions
        + isa_macros
        + fb_macro_passthrough_args
        + mmap_self_macros
        + caching_allocator_macros
    )
    include_dirs = (
        sys_libs_include_dirs
        + python_include_dirs
        + torch_include_dirs
        + omp_include_dir_paths
    )
    cflags = sys_libs_cflags + omp_cflags
    ldflags = sys_libs_ldflags + omp_ldflags
    libraries_dirs = python_libraries_dirs + torch_libraries_dirs + omp_lib_dir_paths
    libraries = torch_libraries + omp_lib
    passthrough_args = (
        sys_libs_passthrough_args + isa_ps_args_build_flags + omp_passthrough_args
    )

    return (
        definitions,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


class CppTorchOptions(CppOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options. And then it will maintains torch related build
    args.
    1. Torch include_directories, libraries, libraries_directories.
    2. Python include_directories, libraries, libraries_directories.
    3. OpenMP related.
    4. Torch MACROs.
    5. MISC
    """

    def __init__(
        self,
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        warning_all: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_relative_path: bool = False,
        use_mmap_weights: bool = False,
        use_mmap_weights_external: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
        compiler: str = "",
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            warning_all=warning_all,
            extra_flags=extra_flags,
            use_relative_path=use_relative_path,
            compiler=compiler,
            min_optimize=min_optimize,
            precompiling=precompiling,
            preprocessing=preprocessing,
        )

        self._aot_mode = aot_mode

        (
            torch_definitions,
            torch_include_dirs,
            torch_cflags,
            torch_ldflags,
            torch_libraries_dirs,
            torch_libraries,
            torch_passthrough_args,
        ) = get_cpp_torch_options(
            cpp_compiler=self._compiler,
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            use_relative_path=use_relative_path,
            use_mmap_weights=use_mmap_weights,
            use_mmap_weights_external=use_mmap_weights_external,
        )

        _append_list(self._definitions, torch_definitions)
        _append_list(self._include_dirs, torch_include_dirs)
        _append_list(self._cflags, torch_cflags)
        _append_list(self._ldflags, torch_ldflags)
        _append_list(self._libraries_dirs, torch_libraries_dirs)
        _append_list(self._libraries, torch_libraries)
        _append_list(self._passthrough_args, torch_passthrough_args)
        self._finalize_options()


def _set_gpu_runtime_env() -> None:
    if (
        config.is_fbcode()
        and torch.version.hip is None
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        os.environ["CUDA_HOME"] = build_paths.sdk_home


@functools.lru_cache(8)
def _find_libcudart_static(path: str) -> Path | None:
    lib_dirs = list(Path(path).rglob("libcudart_static.a"))
    if lib_dirs:
        return lib_dirs[0].resolve().parent
    log_msg = f'"libcudart_static.a" not found under {path}'
    log.info(log_msg)
    return None


def _gen_mingw_import_lib(dll_path: str, def_path: str, import_lib_path: str) -> None:
    """Generate a MinGW import library (.a) from a DLL using gendef and dlltool."""
    dll_name = os.path.basename(dll_path)
    with open(def_path, "w") as def_file:
        subprocess.run(
            ["gendef", "-", dll_path],
            stdout=def_file,
            stderr=subprocess.PIPE,
            check=True,
        )

    subprocess.run(
        [
            "x86_64-w64-mingw32-dlltool",
            "-d",
            def_path,
            "-l",
            import_lib_path,
            "-D",
            dll_name,
        ],
        stderr=subprocess.PIPE,
        check=True,
    )
    log.info("Generated MinGW import library %s from %s", import_lib_path, dll_name)


# MSVC /GS and Control Flow Guard stubs for MinGW cross-compilation.
# CUDA 13.0+ cudart.lib contains MSVC-compiled static objects that reference the
# /GS symbols (__security_cookie, __security_check_cookie, __GSHandlerCheck); the
# CUDA 13.2 cudart.lib is additionally built with Control Flow Guard (/guard:cf)
# and references the CFG dispatch/check function pointers. When cross-compiling
# with MinGW (no MSVC runtime), we provide stubs so the linker can resolve them.
# At runtime on Windows the CUDA runtime DLL handles its own security/CFG checks;
# the static loader code that references these symbols is a thin shim. The /GS
# and CFG-check stubs are safe no-ops, but the CFG *dispatch* stub must tail-jump
# to the real target (in rax on x86_64) -- a no-op would drop the indirect call.
_MSVC_GS_STUBS_SOURCE = """\
#include <stdint.h>
uint64_t __security_cookie = 0x00002B992DDFA232ULL;
void __security_check_cookie(uint64_t cookie) { (void)cookie; }
void __GSHandlerCheck(void) {}
void __guard_check_icall_nop(void *target) { (void)target; }
void (*__guard_check_icall_fptr)(void *target) = __guard_check_icall_nop;
__attribute__((naked)) void __guard_dispatch_icall_nop(void) {
    __asm__ __volatile__("jmp *%rax");
}
void (*__guard_dispatch_icall_fptr)(void) = __guard_dispatch_icall_nop;
"""


def _create_msvc_gs_stubs_lib(output_dir: str) -> str | None:
    """
    Create a static library with MSVC GS security symbol stubs for MinGW.

    Returns the library name (without lib prefix / .a suffix) if successful,
    or None on failure.
    """
    stubs_lib = os.path.join(output_dir, "libmsvc_gs_stubs.a")
    if os.path.exists(stubs_lib):
        return "msvc_gs_stubs"

    src_path = ""
    obj_path = ""
    try:
        src_path = os.path.join(output_dir, "_msvc_gs_stubs.c")
        obj_path = os.path.join(output_dir, "_msvc_gs_stubs.o")
        with open(src_path, "w") as f:
            f.write(_MSVC_GS_STUBS_SOURCE)

        mingw_gcc = MINGW_GXX.replace("g++", "gcc")
        subprocess.run(
            [mingw_gcc, "-c", src_path, "-o", obj_path],
            stderr=subprocess.PIPE,
            check=True,
        )
        mingw_ar = MINGW_GXX.replace("g++", "ar")
        subprocess.run(
            [mingw_ar, "rcs", stubs_lib, obj_path],
            stderr=subprocess.PIPE,
            check=True,
        )
        log.info("Created MSVC GS stubs library: %s", stubs_lib)
        return "msvc_gs_stubs"
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.warning(
            "Failed to create MSVC GS stubs library.",
            exc_info=True,
        )
        if os.path.exists(stubs_lib):
            os.remove(stubs_lib)
        return None
    finally:
        for f in [src_path, obj_path]:
            if f and os.path.exists(f):
                os.remove(f)


def _ensure_mingw_cudart_import_lib(libraries_dirs: list[str]) -> list[str]:
    """
    Auto-generate a MinGW-compatible import library (libcudart.a)
    from the CUDA runtime DLL. This avoids linking against the hybrid cudart.lib
    which contains MSVC-compiled static objects with /GS security symbols that
    MinGW cannot resolve.

    Falls back to creating MSVC GS security stubs if the DLL is unavailable,
    and falls back gracefully to the original cudart.lib if that also fails.

    Returns a list of extra library names to link (e.g. ["msvc_gs_stubs"]).
    """
    import glob

    windows_cuda_home = os.environ.get("WINDOWS_CUDA_HOME")
    if not windows_cuda_home:
        log.debug(
            "WINDOWS_CUDA_HOME not set, skipping MinGW cudart import lib generation"
        )
        return []

    for lib_dir in libraries_dirs:
        if os.path.exists(os.path.join(lib_dir, "libcudart.a")):
            log.debug("libcudart.a already exists in %s, skipping generation", lib_dir)
            return []

    # Find the CUDA runtime DLL for import lib generation
    bin_dir = os.path.join(windows_cuda_home, "bin", "x64")
    if not os.path.isdir(bin_dir):
        bin_dir = os.path.join(windows_cuda_home, "bin")
    dll_candidates = glob.glob(os.path.join(bin_dir, "cudart64_*.dll"))

    # Find a writable directory containing cudart.lib for output
    output_dir = None
    for lib_dir in libraries_dirs:
        if os.path.isdir(lib_dir) and os.access(lib_dir, os.W_OK):
            if os.path.exists(os.path.join(lib_dir, "cudart.lib")):
                output_dir = lib_dir
                break

    if not dll_candidates:
        log.warning(
            "No cudart64_*.dll found in %s. Cannot generate MinGW import library. "
            "Will create MSVC GS security stubs as fallback.",
            bin_dir,
        )
        # Fallback: create GS stubs so the hybrid cudart.lib can link
        if output_dir is not None:
            stub_lib = _create_msvc_gs_stubs_lib(output_dir)
            if stub_lib:
                return [stub_lib]
        return []

    if output_dir is None:
        log.warning(
            "No writable directory containing cudart.lib found. "
            "Cannot generate MinGW import library. "
            "If linking fails with undefined references to __security_cookie, "
            "ensure cudart.lib is present in one of: %s",
            libraries_dirs,
        )
        return []

    dll_path = dll_candidates[0]
    dll_name = os.path.basename(dll_path)

    def_path = os.path.join(output_dir, dll_name.replace(".dll", ".def"))
    import_lib_path = os.path.join(output_dir, "libcudart.a")

    try:
        _gen_mingw_import_lib(dll_path, def_path, import_lib_path)
        return []
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.warning(
            "Failed to generate MinGW cudart import library. "
            "Falling back to MSVC GS stubs.",
            exc_info=True,
        )
        for f in [def_path, import_lib_path]:
            if os.path.exists(f):
                os.remove(f)
        # Fallback: create GS stubs
        stub_lib = _create_msvc_gs_stubs_lib(output_dir)
        if stub_lib:
            return [stub_lib]
        return []


def _transform_cuda_paths(lpaths: list[str]) -> None:
    # This handles two cases:
    # 1. Cases where libs are in (e.g.) lib/cuda-12 and lib/cuda-12/stubs
    # 2. Linux machines may have CUDA installed under either lib64/ or lib/
    for i, path in enumerate(lpaths):
        if "CUDA_HOME" in os.environ and path.startswith(os.environ["CUDA_HOME"]):
            lib_dir: Path | None = _find_libcudart_static(path)
            if lib_dir is None:
                continue
            lpaths[i] = str(lib_dir)
            stub_dir = lib_dir / "stubs"
            if stub_dir.exists():
                lpaths.append(str(stub_dir))

    # Internal path needs special care, using the SDK lib path directly.
    if config.is_fbcode():
        # Add sdk_lib and its stubs/ (which hold the CUDA libs) unconditionally:
        # an existence gate fails on the not-yet-materialized EdenFS path, and a
        # nonexistent -L is harmless to the linker.
        for cuda_lib_dir in (
            build_paths.sdk_lib,
            os.path.join(build_paths.sdk_lib, "stubs"),
        ):
            if cuda_lib_dir not in lpaths:
                lpaths.append(cuda_lib_dir)


def _transform_rocm_paths(lpaths: list[str]) -> None:
    # Internal path needs special care, using the SDK lib path directly.
    if config.is_fbcode():
        sdk_lib = build_paths.sdk_lib
        if os.path.isdir(sdk_lib) and sdk_lib not in lpaths:
            lpaths.append(sdk_lib)


def get_cpp_torch_device_options(
    device_type: str,
    aot_mode: bool = False,
    compile_only: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    """
    This function is used to get the build args of device related build options.
    1. Device include_directories, libraries, libraries_directories.
    2. Device MACROs.
    3. MISC
    4. Return the build args
    """
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []
    if (
        config.is_fbcode()
        and "CUDA_HOME" not in os.environ
        and "CUDA_PATH" not in os.environ
    ):
        os.environ["CUDA_HOME"] = build_paths.sdk_home

    _set_gpu_runtime_env()
    from torch.utils import cpp_extension

    # cpp_extension resolves CUDA_HOME into a module-level global at import time,
    # so an fbcode process that imported it before the env var above was written
    # caches None; refresh the global here so the just-set CUDA_HOME env actually
    # takes effect for include_paths/library_paths below.
    if cpp_extension.CUDA_HOME is None and os.environ.get("CUDA_HOME"):
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]

    include_dirs = cpp_extension.include_paths(
        device_type, config.aot_inductor.link_libtorch is None
    )
    link_libtorch = config.aot_inductor.link_libtorch
    libraries_dirs = cpp_extension.library_paths(
        device_type,
        torch_include_dirs=link_libtorch,
        cross_target_platform=config.aot_inductor.cross_target_platform,
    )
    if device_type == "cuda":
        definitions.append(" USE_ROCM" if torch.version.hip else " USE_CUDA")

        if torch.version.hip is not None:
            if config.is_fbcode() or not link_libtorch:
                libraries += ["amdhip64"]
            else:
                libraries += ["torch_hip"]
            definitions.append(" __HIP_PLATFORM_AMD__")
            _transform_rocm_paths(libraries_dirs)
        else:
            if config.is_fbcode() or not link_libtorch:
                libraries += ["cuda"]
            else:
                libraries += ["cuda", "torch_cuda"]
            if config.aot_inductor.cross_target_platform == "windows":
                extra_libs = _ensure_mingw_cudart_import_lib(libraries_dirs)
                libraries += ["cudart"] + extra_libs
            _transform_cuda_paths(libraries_dirs)

    if device_type == "xpu":
        definitions.append(" USE_XPU")
        xpu_error_string = (
            "Intel GPU driver is not properly installed, please follow the instruction "
            "in https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support."
        )
        if _IS_WINDOWS:
            ze_root = os.getenv("LEVEL_ZERO_V1_SDK_PATH")
            if ze_root is None:
                raise OSError(xpu_error_string)
            include_dirs += [os.path.join(ze_root, "include")]
            libraries_dirs += [os.path.join(ze_root, "lib")]
        else:
            # Suppress multi-line comment warnings in sycl headers
            cflags += ["Wno-comment"]
            if not find_library("ze_loader"):
                raise OSError(xpu_error_string)

        libraries += ["ze_loader", "sycl"]
        if link_libtorch:
            libraries += ["torch_xpu"]

    if device_type == "mps":
        definitions.append(" USE_MPS")

    if config.is_fbcode():
        include_dirs.append(build_paths.sdk_include)

        if aot_mode and device_type == "cuda":
            if torch.version.hip is None:
                if not compile_only:
                    # Only add link args, when compile_only is false.
                    passthrough_args = ["-Wl,-Bstatic -lcudart_static -Wl,-Bdynamic"]

        if device_type == "cpu":
            (
                stdcxx_lib_dir_paths,
                stdcxx_libs,
            ) = _get_libstdcxx_args()
            libraries_dirs += stdcxx_lib_dir_paths
            libraries += stdcxx_libs

    if config.aot_inductor.custom_op_libs:
        libraries += config.aot_inductor.custom_op_libs

    return (
        definitions,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


class CppTorchDeviceOptions(CppTorchOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options and torch common build options. And then it will
    maintains cuda/xpu device related build args.
    """

    def __init__(
        self,
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        device_type: str = "cuda",
        aot_mode: bool = False,
        compile_only: bool = False,
        use_relative_path: bool = False,
        use_mmap_weights: bool = False,
        use_mmap_weights_external: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
        compiler: str = "",
    ) -> None:
        super().__init__(
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_relative_path=use_relative_path,
            use_mmap_weights=use_mmap_weights,
            use_mmap_weights_external=use_mmap_weights_external,
            extra_flags=extra_flags,
            min_optimize=min_optimize,
            precompiling=precompiling,
            preprocessing=preprocessing,
            compiler=compiler,
        )

        device_definitions: list[str] = []
        device_include_dirs: list[str] = []
        device_cflags: list[str] = []
        device_ldflags: list[str] = []
        device_libraries_dirs: list[str] = []
        device_libraries: list[str] = []
        device_passthrough_args: list[str] = []

        (
            device_definitions,
            device_include_dirs,
            device_cflags,
            device_ldflags,
            device_libraries_dirs,
            device_libraries,
            device_passthrough_args,
        ) = get_cpp_torch_device_options(
            device_type=device_type,
            aot_mode=aot_mode,
            compile_only=compile_only,
        )
        _append_list(self._definitions, device_definitions)
        _append_list(self._include_dirs, device_include_dirs)
        _append_list(self._cflags, device_cflags)
        _append_list(self._ldflags, device_ldflags)
        _append_list(self._libraries_dirs, device_libraries_dirs)
        _append_list(self._libraries, device_libraries)
        _append_list(self._passthrough_args, device_passthrough_args)
        self._finalize_options()

    def _finalize_options(self) -> None:
        super()._finalize_options()
        if config.is_fbcode():
            # Re-order library search paths in case there are lib conflicts
            # that also live in the FBCode python lib dir.
            _, python_lib_dirs = _get_python_related_args()
            if len(python_lib_dirs) != 1:
                raise AssertionError(f"Python lib dirs: {python_lib_dirs}")
            if python_lib_dirs[0] in self._libraries_dirs:
                self._libraries_dirs.remove(python_lib_dirs[0])
                self._libraries_dirs.append(python_lib_dirs[0])


def get_name_and_dir_from_output_file_path(
    file_path: str,
) -> tuple[str, str]:
    """
    This function help prepare parameters to new cpp_builder.
    Example:
        input_code: /tmp/tmpof1n5g7t/5c/c5crkkcdvhdxpktrmjxbqkqyq5hmxpqsfza4pxcf3mwk42lphygc.cpp
        name, dir = get_name_and_dir_from_output_file_path(input_code)
    Run result:
        name = c5crkkcdvhdxpktrmjxbqkqyq5hmxpqsfza4pxcf3mwk42lphygc
        dir = /tmp/tmpof1n5g7t/5c/

    put 'name' and 'dir' to CppBuilder's 'name' and 'output_dir'.
    CppBuilder --> get_target_file_path will format output path according OS:
    Linux: /tmp/tmppu87g3mm/zh/czhwiz4z7ca7ep3qkxenxerfjxy42kehw6h5cjk6ven4qu4hql4i.so
    Windows: [Windows temp path]/tmppu87g3mm/zh/czhwiz4z7ca7ep3qkxenxerfjxy42kehw6h5cjk6ven4qu4hql4i.dll
    """
    name_and_ext = os.path.basename(file_path)
    name, _ext = os.path.splitext(name_and_ext)
    dir = os.path.dirname(file_path)

    return name, dir


class CppBuilder:
    """
    CppBuilder is a cpp jit builder, and it supports both Windows, Linux and MacOS.
    Args:
        name:
            1. Build target name, the final target file will append extension type automatically.
            2. Due to the CppBuilder is supports multiple OS, it will maintains ext for OS difference.
        sources:
            Source code file list to be built.
        BuildOption:
            Build options to the builder.
        output_dir:
            1. The output_dir the target file will output to.
            2. The default value is empty string, and then the use current dir as output dir.
            3. Final target file: output_dir/name.ext
    """

    @staticmethod
    def __get_python_module_flags() -> tuple[str, str]:
        extension = ".pyd" if _IS_WINDOWS else ".so"
        output_flags = "/Fe" if _IS_WINDOWS else "-o"
        return extension, output_flags

    @staticmethod
    def __get_object_flags() -> tuple[str, str]:
        extension = ".obj" if _IS_WINDOWS else ".o"
        output_flags = "/c /Fo" if _IS_WINDOWS else "-c -o"  # codespell:ignore
        return extension, output_flags

    @staticmethod
    def __get_precompiled_header_flags() -> tuple[str, str]:
        extension = ".pch" if _IS_WINDOWS or not is_gcc() else ".gch"
        output_flags = "/Fp" if _IS_WINDOWS else "-o"
        return extension, output_flags

    @staticmethod
    def __get_preprocessor_output_flags() -> tuple[str, str]:
        extension = ".i"
        output_flags = "/EP /P" if _IS_WINDOWS else "-E -P -o"
        return extension, output_flags

    def __init__(
        self,
        name: str,
        sources: str | list[str],
        BuildOption: BuildOptionsBase,
        output_dir: str = "",
    ) -> None:
        self._compiler = ""
        self._cflags_args = ""
        self._definitions_args = ""
        self._include_dirs_args = ""
        self._ldflags_args = ""
        self._libraries_dirs_args = ""
        self._libraries_args = ""
        self._passthrough_parameters_args = ""

        # When relative path is used, we need to maintain the source dir list.
        self._orig_source_paths = []
        self._output_dir = ""
        self._target_file = ""

        self._use_relative_path: bool = False
        self._aot_mode: bool = False

        self._name = name
        self._target_name = (
            config.aot_inductor.model_name_for_generated_files or "aoti_model"
        )

        # Code start here, initial self internal variables firstly.
        self._build_option = BuildOption
        self._compiler = BuildOption.get_compiler()
        self._use_relative_path = BuildOption.get_use_relative_path()
        self._aot_mode = BuildOption.get_aot_mode()

        self._output_dir = output_dir

        self._compile_only = BuildOption.get_compile_only()
        self._precompiling = BuildOption.get_precompiling()
        self._preprocessing = BuildOption.get_preprocessing()
        # Only one of these options (if any) should be true at any given time.
        if sum((self._compile_only, self._precompiling, self._preprocessing)) > 1:
            raise AssertionError(
                "at most one of compile_only, precompiling, preprocessing may be set"
            )
        self._do_link = not (
            self._compile_only or self._precompiling or self._preprocessing
        )

        # MSVC produces two files when precompiling: the actual .pch file, as well as an
        # object file which must be linked into the final library.  This class assumes
        # only one output file of note, so for now we'll error out here.
        if _IS_WINDOWS and self._precompiling:
            raise AssertionError("Cannot currently precompile headers on Windows!")

        if self._compile_only:
            file_ext, output_flags = self.__get_object_flags()
        elif self._precompiling:
            file_ext, output_flags = self.__get_precompiled_header_flags()
        elif self._preprocessing:
            file_ext, output_flags = self.__get_preprocessor_output_flags()
        else:
            file_ext, output_flags = self.__get_python_module_flags()
        self._target_file = os.path.join(self._output_dir, f"{self._name}{file_ext}")

        relative_target_file = (
            os.path.basename(self._target_file)
            if self._use_relative_path
            else self._target_file
        )
        if _IS_WINDOWS:
            if self._preprocessing:
                # The target file name is automatically determined by MSVC.
                self._output = output_flags
            else:
                self._output = f"{output_flags}{relative_target_file}"
        else:
            self._output = f"{output_flags} {relative_target_file}"

        if isinstance(sources, str):
            sources = [sources]

        # Use relative paths only when requested (typically for remote builds)
        if config.is_fbcode() and self._use_relative_path:
            # Will create another temp directory for building, so do NOT use the
            # absolute path.
            self._orig_source_paths = list(sources)
            sources = [os.path.basename(i) for i in sources]

        if self._precompiling:
            if len(sources) != 1:
                raise AssertionError(
                    f"expected exactly one source when precompiling, got {len(sources)}"
                )
            # See above; we can currently assume this is not on MSVC.
            self._sources_args = f"-x c++-header {sources[0]}"
            if self._use_relative_path and _is_clang(BuildOption.get_compiler()):
                # Store PCH paths relative to -isysroot so the .pch can
                # be used from a different build directory.  The matching
                # -isysroot is injected by build_fbcode_re().
                self._cflags_args += " -relocatable-pch -Xclang -fno-pch-timestamp "
        else:
            self._sources_args = " ".join(sources)

        for cflag in BuildOption.get_cflags():
            if _IS_WINDOWS:
                self._cflags_args += f"/{cflag} "
            else:
                self._cflags_args += f"-{cflag} "

        for definition in BuildOption.get_definitions():
            if _IS_WINDOWS:
                self._definitions_args += f"/D {definition} "
            else:
                self._definitions_args += f"-D {definition} "

        if precompiled_header := BuildOption.precompiled_header:
            if _IS_WINDOWS:
                log.warning(
                    "Precompiled header support for MSVC is currently unavailable; ignoring %s",
                    precompiled_header,
                )
            else:
                self._include_dirs_args = f"-include {precompiled_header} "
                if self._use_relative_path and _is_clang(BuildOption.get_compiler()):
                    # Skip clang's own PCH validation during consumption.
                    # _precompile_header() already handles cache invalidation
                    # via content hashing, and -fno-validate-pch allows the
                    # PCH to be used even when the original source file is at
                    # a different path (e.g. across Remote Execution workers).
                    self._cflags_args += " -Xclang -fno-validate-pch "

        for inc_dir in BuildOption.get_include_dirs():
            if _IS_WINDOWS:
                self._include_dirs_args += f'/I "{inc_dir}" '
            else:
                self._include_dirs_args += f"-I{shlex.quote(inc_dir)} "

        for ldflag in BuildOption.get_ldflags():
            if _IS_WINDOWS:
                self._ldflags_args += f"/{ldflag} "
            else:
                self._ldflags_args += f"-{ldflag} "

        for lib_dir in BuildOption.get_libraries_dirs():
            if _IS_WINDOWS:
                self._libraries_dirs_args += f'/LIBPATH:"{lib_dir}" '
            else:
                self._libraries_dirs_args += f"-L{shlex.quote(lib_dir)} "

        for lib in BuildOption.get_libraries():
            if _IS_WINDOWS:
                self._libraries_args += f'"{lib}.lib" '
            else:
                self._libraries_args += f"-l{lib} "

        for passthrough_arg in self._stage_passthrough_paths(
            BuildOption.get_passthrough_args()
        ):
            self._passthrough_parameters_args += f"{passthrough_arg} "

    def get_command_line(self) -> str:
        def format_build_command(
            compiler: str,
            sources: str,
            include_dirs_args: str,
            definitions_args: str,
            cflags_args: str,
            ldflags_args: str,
            libraries_args: str,
            libraries_dirs_args: str,
            passthrough_args: str,
            output: str,
        ) -> str:
            if _IS_WINDOWS:
                # https://learn.microsoft.com/en-us/cpp/build/walkthrough-compile-a-c-program-on-the-command-line?view=msvc-1704
                # https://stackoverflow.com/a/31566153
                cmd = (
                    f"{compiler} {include_dirs_args} {definitions_args} {cflags_args} "
                    f"{sources} {passthrough_args} {output}"
                )
                if self._do_link:
                    cmd += f" /LD /link {libraries_dirs_args} {libraries_args} {ldflags_args}"
                cmd = normalize_path_separator(cmd)
            else:
                cmd = (
                    f"{compiler} {sources} {definitions_args} {cflags_args} "
                    f"{include_dirs_args} {passthrough_args} {output}"
                )
                if self._do_link:
                    cmd += f" {ldflags_args} {libraries_args} {libraries_dirs_args}"
            return cmd

        command_line = format_build_command(
            compiler=self._compiler,
            sources=self._sources_args,
            include_dirs_args=self._include_dirs_args,
            definitions_args=self._definitions_args,
            cflags_args=self._cflags_args,
            ldflags_args=self._ldflags_args,
            libraries_args=self._libraries_args,
            libraries_dirs_args=self._libraries_dirs_args,
            passthrough_args=self._passthrough_parameters_args,
            output=self._output,
        )
        return command_line

    def get_target_file_path(self) -> str:
        return normalize_path_separator(self._target_file)

    def _stage_passthrough_paths(self, args: list[str]) -> list[str]:
        """For fbcode remote builds, treat any absolute file path that appears
        as a passthrough arg (e.g. precompiled .so kernels passed in via
        ``extra_flags``) the same way as a source: stage the file into the
        remote build sandbox and rewrite the command to reference it by
        basename. Without this the linker is handed a path that only exists on
        the submitting host and the remote build fails to find the file."""
        if not (config.is_fbcode() and self._use_relative_path):
            return args
        rewritten = []
        rpath_dirs: OrderedSet[str] = OrderedSet()
        for arg in args:
            tokens = shlex.split(arg)
            new_tokens = []
            for tok in tokens:
                if os.path.isabs(tok) and os.path.isfile(tok):
                    self._orig_source_paths.append(tok)
                    # Rewriting the absolute path to a basename makes the
                    # linker record a bare-basename DT_NEEDED entry, because
                    # the staged kernel .so files carry no DT_SONAME. That
                    # basename is unresolvable when the wrapper .so is later
                    # dlopen'd, so add rpath entries the runtime loader can
                    # search: $ORIGIN for a kernel .so co-staged next to the
                    # wrapper, and the kernel .so's original directory as a
                    # fallback when it is loaded from its cache location.
                    if self._do_link and tok.endswith(".so"):
                        rpath_dirs.add(os.path.dirname(tok))
                    new_tokens.append(os.path.basename(tok))
                else:
                    new_tokens.append(tok)
            rewritten.append(" ".join(shlex.quote(t) for t in new_tokens))
        if rpath_dirs:
            rewritten.append("-Wl,-rpath," + shlex.quote("$ORIGIN"))
            for rpath_dir in rpath_dirs:
                rewritten.append(f"-Wl,-rpath,{shlex.quote(rpath_dir)}")
        return rewritten

    def build_fbcode_re(
        self,
    ) -> None:
        with dynamo_timed("compile_file"):
            command = shlex.split(self.get_command_line())
            try:
                output_path = self._target_file
                # When we build remotely, we need to make sure to carefully copy any files
                # that are required during the compilation process into our build directly.
                # This is where all of the ATen/c10/Torch includes come from.
                torch_includes_path = os.path.join(_TORCH_PATH, "include")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Copy everything to tmp compilation folder
                    shutil.copy(_LINKER_SCRIPT, os.path.join(tmp_dir, "script.ld"))
                    for src in self._orig_source_paths:
                        shutil.copy(src, os.path.join(tmp_dir, os.path.basename(src)))
                    dest_include_path = os.path.join(tmp_dir, "include")
                    shutil.copytree(torch_includes_path, dest_include_path)

                    # Copy precompiled header (.h and .gch/.pch) into the
                    # build directory and rewrite the -include flag so the
                    # compiler can find it.
                    pch_header = self._build_option.precompiled_header
                    if pch_header and os.path.isfile(pch_header):
                        pch_ext = ".pch" if _IS_WINDOWS or not is_gcc() else ".gch"
                        pch_compiled = pch_header + pch_ext
                        pch_basename = os.path.basename(pch_header)
                        shutil.copy(pch_header, os.path.join(tmp_dir, pch_basename))
                        if os.path.isfile(pch_compiled):
                            shutil.copy(
                                pch_compiled,
                                os.path.join(tmp_dir, pch_basename + pch_ext),
                            )
                        command = [
                            pch_basename if arg == pch_header else arg
                            for arg in command
                        ]

                    # Relocatable PCH stores include paths relative to
                    # -isysroot.  Set sysroot to the tmp build dir so
                    # paths resolve correctly in both precompilation and
                    # later kernel compilations that consume the PCH.
                    if self._precompiling or (
                        pch_header and os.path.isfile(pch_header)
                    ):
                        command[1:1] = ["-isysroot", "."]

                    # Run the build, raising RuntimeError on failure instead of
                    # SkipFrame so compilation errors propagate rather than
                    # silently falling back to eager execution.
                    tmp_output_path = _run_build_command(
                        command,
                        tmp_dir,
                        os.path.basename(output_path),
                        exception_class=RuntimeError,
                    )
                    # Copy output from the build
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    shutil.copy(tmp_output_path, output_path)
                    if output_path.endswith(".o"):
                        os.chmod(output_path, 0o644)
                    elif output_path.endswith(".so"):
                        os.chmod(output_path, 0o755)
            except subprocess.CalledProcessError as e:
                raise exc.CppCompileError(command, e.output.decode("utf-8")) from e

    def build(self) -> None:
        """
        It is must need a temporary directory to store object files in Windows.
        After build completed, delete the temporary directory to save disk space.
        """
        if self._use_relative_path:
            # remote build uses relative path
            return self.build_fbcode_re()
        _create_if_dir_not_exist(self._output_dir)
        _build_tmp_dir = os.path.join(
            self._output_dir, f"{self._name}_{_BUILD_TEMP_DIR}"
        )
        _create_if_dir_not_exist(_build_tmp_dir)

        build_cmd = self.get_command_line()
        run_compile_cmd(build_cmd, cwd=_build_tmp_dir)
        _remove_dir(_build_tmp_dir)

    def save_compile_cmd_to_cmake(
        self,
        cmake_path: str,
        device_type: str,
    ) -> None:
        """
        Save global cmake settings here, e.g. compiler options.
        If targeting CUDA, also emit a custom function to embed CUDA kernels.
        """

        definitions = " ".join(self._build_option.get_definitions())
        target_library_type = (
            "STATIC" if not config.aot_inductor.dynamic_linkage else "SHARED"
        )

        contents = textwrap.dedent(
            f"""
            cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
            project({self._target_name} LANGUAGES CXX)
            set(CMAKE_CXX_STANDARD 20)

            # Set a library target
            add_library({self._target_name} {target_library_type})

            """
        )

        if config.aot_inductor.link_libtorch or config.test_configs.use_libtorch:
            # When compile_standalone is True, the generated cpp project should
            # not use Torch. But for unit testing purpose, we need to use Torch here.
            contents += textwrap.dedent(
                """
                # May need to point CMAKE_PREFIX_PATH to the right torch location
                find_package(Torch REQUIRED)

                """
            )
            # flags and macros here are mostly CPU specific. Not emitting them for GPU models
            # will make the generated CMake file more portable and won't really hurt performance.
            # NOTE: standalone focuses on GPU now. For CPU, some of the flags and macros may
            # be still needed.
            contents += textwrap.dedent(
                f"""
                # Add macro definitions
                target_compile_definitions({self._target_name} PRIVATE {definitions})

                # Add compile flags
                target_compile_options({self._target_name} PRIVATE {self._cflags_args})

                # Backend-specific flags
                target_compile_options({self._target_name} PRIVATE {self._passthrough_parameters_args} -c)

                """
            )
        else:
            # When compile_standalone is True, use TorchStandalone instead of Torch
            contents += textwrap.dedent(
                f"""
                find_package(TorchStandalone REQUIRED)
                # Set up include directories to find headers at the correct paths
                target_include_directories({self._target_name} PRIVATE ${{TorchStandalone_INCLUDE_DIRS}})
                target_include_directories({self._target_name} PRIVATE ${{TorchStandalone_INCLUDE_DIRS}}/standalone)

                """
            )

        if device_type == "cuda" and torch.version.hip is None:
            from torch._inductor.codegen.cuda import compile_utils

            cuda_arch = compile_utils._aoti_cuda_target_arch()
            cuda_gencode_flags = "\n                                ".join(
                f"-gencode {option}"
                for option in compile_utils._cuda_multi_arch_gencode_options(cuda_arch)
            )
            contents += textwrap.dedent(
                f"""
                enable_language(CUDA)
                set(CMAKE_CUDA_STANDARD 17)
                find_package(CUDAToolkit REQUIRED)
                target_include_directories({self._target_name} PRIVATE ${{CUDAToolkit_INCLUDE_DIRS}})
                target_compile_definitions({self._target_name} PRIVATE USE_CUDA)
                target_link_libraries({self._target_name} PRIVATE cuda CUDA::cudart_static)

                find_program(OBJCOPY_EXECUTABLE objcopy)
                if(NOT OBJCOPY_EXECUTABLE)
                    message(FATAL_ERROR "objcopy not found. Cannot embed fatbin as object file")
                endif()

                set(KERNEL_TARGETS "")
                set(KERNEL_OBJECT_FILES "")
                # Function to embed a single kernel
                function(embed_gpu_kernel KERNEL_NAME PTX_FILE)
                    set(FATBIN_BASENAME ${{KERNEL_NAME}}.fatbin)
                    set(FATBIN_FILE ${{CMAKE_CURRENT_BINARY_DIR}}/${{FATBIN_BASENAME}})
                    set(OBJECT_BASENAME ${{KERNEL_NAME}}.fatbin.o)
                    set(OBJECT_FILE ${{CMAKE_CURRENT_BINARY_DIR}}/${{OBJECT_BASENAME}})

                    # --- Define UNIQUE C symbol names ---
                    set(SYMBOL_START __${{KERNEL_NAME}}_start)
                    set(SYMBOL_END __${{KERNEL_NAME}}_end)
                    set(SYMBOL_SIZE __${{KERNEL_NAME}}_size)
                    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" MANGLED_BASENAME ${{FATBIN_FILE}})
                    set(OBJCOPY_START_SYM _binary_${{MANGLED_BASENAME}}_start)
                    set(OBJCOPY_END_SYM _binary_${{MANGLED_BASENAME}}_end)
                    set(OBJCOPY_SIZE_SYM _binary_${{MANGLED_BASENAME}}_size)

                    # --- PTX to FATBIN Command & Target ---
                    add_custom_command(
                        OUTPUT ${{FATBIN_FILE}}
                        COMMAND ${{CUDAToolkit_NVCC_EXECUTABLE}} --fatbin ${{PTX_FILE}} -o ${{FATBIN_FILE}} ${{NVCC_GENCODE_FLAGS}}
                                {cuda_gencode_flags}
                        DEPENDS ${{PTX_FILE}}
                    )

                    # --- FATBIN to Object File (.o) Command ---
                    add_custom_command(
                        OUTPUT ${{OBJECT_FILE}}
                        COMMAND ${{CMAKE_LINKER}} -r -b binary -z noexecstack -o ${{OBJECT_FILE}} ${{FATBIN_FILE}}
                        COMMAND ${{OBJCOPY_EXECUTABLE}} --rename-section .data=.rodata,alloc,load,readonly,data,contents
                                ${{OBJECT_FILE}}
                        COMMAND ${{OBJCOPY_EXECUTABLE}}
                                --redefine-sym ${{OBJCOPY_START_SYM}}=${{SYMBOL_START}}
                                --redefine-sym ${{OBJCOPY_END_SYM}}=${{SYMBOL_END}}
                                --redefine-sym ${{OBJCOPY_SIZE_SYM}}=${{SYMBOL_SIZE}}
                                ${{OBJECT_FILE}}
                        DEPENDS ${{FATBIN_FILE}}
                    )
                    add_custom_target(build_kernel_object_${{KERNEL_NAME}} DEPENDS ${{OBJECT_FILE}})

                    # --- Add to a list for linking later ---
                    set(KERNEL_TARGETS ${{KERNEL_TARGETS}} build_kernel_object_${{KERNEL_NAME}} PARENT_SCOPE)
                    set(KERNEL_OBJECT_FILES ${{KERNEL_OBJECT_FILES}} ${{OBJECT_FILE}} PARENT_SCOPE)
                endfunction()

                """
            )
        elif device_type == "xpu":
            contents += textwrap.dedent(
                """
                find_program(OBJCOPY_EXECUTABLE objcopy)
                if(NOT OBJCOPY_EXECUTABLE)
                    message(FATAL_ERROR "objcopy not found. Cannot embed spv as object file")
                endif()

                set(KERNEL_TARGETS "")
                set(KERNEL_OBJECT_FILES "")
                # Function to embed a single kernel
                function(embed_gpu_kernel KERNEL_NAME SPV_FILE)
                    set(OBJECT_BASENAME ${KERNEL_NAME}.spv.o)
                    set(OBJECT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${OBJECT_BASENAME})

                    # --- Define UNIQUE C symbol names ---
                    set(SYMBOL_START __${KERNEL_NAME}_start)
                    set(SYMBOL_END __${KERNEL_NAME}_end)
                    set(SYMBOL_SIZE __${KERNEL_NAME}_size)
                    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" MANGLED_BASENAME ${SPV_FILE})
                    set(OBJCOPY_START_SYM _binary_${MANGLED_BASENAME}_start)
                    set(OBJCOPY_END_SYM _binary_${MANGLED_BASENAME}_end)
                    set(OBJCOPY_SIZE_SYM _binary_${MANGLED_BASENAME}_size)

                    # --- SPV_FILE to Object File (.o) Command ---
                    add_custom_command(
                        OUTPUT ${OBJECT_FILE}
                        COMMAND ${CMAKE_LINKER} -r -b binary -z noexecstack -o ${OBJECT_FILE} ${SPV_FILE}
                        COMMAND ${OBJCOPY_EXECUTABLE} --rename-section .data=.rodata,alloc,load,readonly,data,contents
                                ${OBJECT_FILE}
                        COMMAND ${OBJCOPY_EXECUTABLE}
                                --redefine-sym ${OBJCOPY_START_SYM}=${SYMBOL_START}
                                --redefine-sym ${OBJCOPY_END_SYM}=${SYMBOL_END}
                                --redefine-sym ${OBJCOPY_SIZE_SYM}=${SYMBOL_SIZE}
                                ${OBJECT_FILE}
                        DEPENDS ${SPV_FILE}
                    )
                    add_custom_target(build_kernel_object_${KERNEL_NAME} DEPENDS ${OBJECT_FILE})

                    # --- Add to a list for linking later ---
                    set(KERNEL_TARGETS ${KERNEL_TARGETS} build_kernel_object_${KERNEL_NAME} PARENT_SCOPE)
                    set(KERNEL_OBJECT_FILES ${KERNEL_OBJECT_FILES} ${OBJECT_FILE} PARENT_SCOPE)
                endfunction()

                """
            )

        with open(cmake_path, "w") as f:
            f.write(contents)

    def save_src_to_cmake(self, cmake_path: str, src_path: str) -> None:
        # Remove the directory part of file_path
        src_path = "${CMAKE_CURRENT_SOURCE_DIR}/" + Path(src_path).name
        with open(cmake_path, "a") as f:
            f.write(f"target_sources({self._target_name} PRIVATE {src_path})\n")

    def save_kernel_asm_to_cmake(self, cmake_path: str, asm_files: list[str]) -> None:
        # TODO: make this work beyond CUDA
        with open(cmake_path, "a") as f:
            for asm_file in asm_files:
                kernel_name = Path(asm_file).name.split(".")[0]
                asm_file = f"${{CMAKE_CURRENT_SOURCE_DIR}}/{Path(asm_file).name}"
                contents = textwrap.dedent(
                    f"""
                    embed_gpu_kernel({kernel_name} {asm_file})
                    """
                )
                f.write(contents)
            if asm_files:
                f.write(f"add_dependencies({self._target_name} ${{KERNEL_TARGETS}})\n")
                f.write(
                    f"target_link_libraries({self._target_name} PRIVATE ${{KERNEL_OBJECT_FILES}})\n"
                )

    def save_link_cmd_to_cmake(self, cmake_path: str) -> None:
        lflags = " ".join(self._build_option.get_ldflags())
        libs = " ".join(self._build_option.get_libraries())
        contents = textwrap.dedent(
            f"""
            # Add linker flags
            target_link_options({self._target_name} PRIVATE {lflags})

            # Add libraries
            target_link_libraries({self._target_name} PRIVATE {libs})
         """
        )

        if not os.path.exists(cmake_path):
            raise AssertionError(
                f"save_link_cmd_to_cmakefile expects {cmake_path} to already exist"
            )
        with open(cmake_path, "a") as f:
            f.write(contents)


def run_asm_build_object(src: str, target: str, cwd: str) -> None:
    def get_asm_compiler() -> str:
        if _IS_WINDOWS:
            ASM_CC = "ml64"
        else:
            ASM_CC = get_cpp_compiler()
            # Intel compiler is not support to compile asm, switch to gcc.
            if _is_intel_compiler(ASM_CC):
                ASM_CC = "gcc"
        return ASM_CC

    def get_command_line(asm_cc: str, src: str, target: str) -> str:
        if _IS_WINDOWS:
            # Format reference:
            # https://learn.microsoft.com/en-us/cpp/assembler/masm/ml-and-ml64-command-line-reference?view=msvc-170
            cmd = f"{asm_cc} {src} /c /Fo {target}"  # codespell:ignore /Fo
        else:
            cmd = f"{asm_cc} -c {src} -o {target}"

        return cmd

    asm_cc = get_asm_compiler()
    cmd = get_command_line(
        asm_cc=asm_cc,
        src=normalize_path_separator(src),
        target=normalize_path_separator(target),
    )
    run_compile_cmd(cmd, cwd=normalize_path_separator(cwd))
<|endoftext|># mypy: allow-untyped-defs
import logging

import torch
import torch.utils._pytree as pytree
from torch._inductor.utils import is_symbolic
from torch.utils._ordered_set import OrderedSet

from . import config, ir
from .virtualized import V


log = logging.getLogger(__name__)


# NOTE [lowering-time collective optimization]
#
# In collective communication libraries such as NCCL, every rank maintains
# communication buffers that are remotely accessible by some peers. Depending
# on the underlying transport, remote accessibility may be established via
# mechanisms such as ib_reg_mr, CUDA P2P, or CUDA multicast. Typically, these
# buffers are private to the communication library by default, and
# communication ops copy user data in and out of these buffers.
#
# To prevent these copies, an optimization commonly known as "user buffer
# registration" can be employed. This allows direct establishment of remote
# accessibility on user buffers, eliminating the need for copying. However,
# this optimization introduces stringent usage requirements, which are
# typically hard to satisfy without being intrusive to the user code:
#
# - Establishing remote accessibility is expensive and often done ahead of
# time. In such implementations, all ranks must agree on the set of allocations
# used for every collective op. Failing to meet this requirement can
# lead to runtime errors or even silent correctness issues.
# - Even if the collective communication library supports gracefully falling
# back to "unregistered" implementations, the fallback mechanism would nullify
# the optimization.
# - Some communication mechanisms impose stricter requirements than others. For
# example, CUDA's multicast + multi-mem instructions require all ranks to agree
# not only on the allocations used for every collective but also on the offsets
# within these allocations.
#
# To support all different mechanisms with optimal results, we aim to satisfy
# the strictest requirement for this family of optimizations - we ensure that
# every collective op invocation is guaranteed to operate on the same
# allocation, at the same offset, in every iteration.
#
# For eligible collective ops, we identify communication buffers at lowering
# time and optionally choose to lower the op to a different kernel
# (communication libraries like NCCL handle both registered and non-registered
# buffers transparently within the same op, though some may require different
# ops for different cases). Later, the codegen will perform "persistent
# allocation" to satisfy the aforementioned constraints, and optionally,
# perform buffer planning to optimize overall memory usage.
def can_realize_as_comm_buffer(
    x: ir.TensorBox, comm_buffer_type: ir.CommBufferType
) -> bool:
    """
    Check if an input can be realized as a comm buffer of the specified
    `comm_buffer_type`.
    """
    data = _get_data(x)

    if isinstance(data, ir.Loops):
        return True

    # We cannot realize buffers as comm buffers if we don't control their
    # allocation.
    if isinstance(data, ir.Buffer) and not data.should_allocate():
        return False

    layout = data.get_output_spec()
    if isinstance(layout, ir.CommBufferLayout):
        return True

    if isinstance(layout, ir.FixedLayout):
        return True

    if isinstance(layout, ir.FlexibleLayout) and not is_symbolic(data.get_numel()):
        return True

    return False


def realize_as_comm_buffer(
    x: ir.TensorBox,
    comm_buffer_type: ir.CommBufferType,
    group_name: "torch.distributed.distributed_c10d.GroupName",
) -> None:
    """
    Realize an input as a comm buffer of the specified `comm_buffer_type`.

    Specifically, this realizes the underlying buffer if it's still unrealized
    and changes the layout of the buffer to `ir.CommBufferLayout`.
    """
    x.realize()
    buffer = _get_data(x)
    if not isinstance(buffer, ir.Buffer):
        raise AssertionError(f"expected an `ir.Buffer`, got {type(buffer)}")

    layout = buffer.get_output_spec()
    if isinstance(layout, ir.CommBufferLayout):
        return

    # The buffer may have already been frozen to FixedLayout if it was used
    # by another operation before the comm operation.
    if not isinstance(layout, (ir.FlexibleLayout, ir.FixedLayout)):
        raise AssertionError(
            "A buffer can only be realized as a comm buffer if it "
            f"has `FlexibleLayout` or `FixedLayout` (got {layout})."
        )

    if is_symbolic(buffer.get_numel()):
        raise AssertionError(
            "A buffer with symbolic shape cannot be converted to "
            f"a comm buffer (got {layout})."
        )

    buffer.layout = ir.CommBufferLayout(
        layout=layout,
        comm_buffer_type=comm_buffer_type,
        group_name=group_name,
    )


def _get_data(x: ir.TensorBox) -> ir.IRNode:
    if isinstance(x.data, ir.BaseView):
        # TensorBox -> *View -> StorageBox -> IRNode
        node = x.data.unwrap_view()
        if not isinstance(node, (ir.BaseView, ir.MutableBox)):
            raise AssertionError(
                f"expected an `ir.BaseView` or `ir.MutableBox`, got {type(node)}"
            )
        return node.data
    elif isinstance(x.data, ir.StorageBox):
        # TensorBox -> StorageBox -> IRNode
        return x.data.data
    else:
        raise AssertionError(
            "Expect the data attr of a `TensorBox` to be either "
            f"an `ir.BaseView` or `ir.StorageBox` (got {x.data})."
        )


_bufs_to_skip_wait = OrderedSet[tuple[int, str]]()


def mark_as_skip_wait(x: ir.IRNode) -> None:
    """
    If a non-blocking collective is lowered as a blocking collective, the wait
    node in the original graph becomes useless and we can skip the lowering it.
    """
    _bufs_to_skip_wait.add((id(V.graph), x.get_name()))


def should_skip_wait(x: ir.IRNode) -> bool:
    return (id(V.graph), x.get_name()) in _bufs_to_skip_wait


def _should_lower_as_one_shot_all_reduce(
    inp: ir.TensorBox,
    reduce_op: str,
    group_name: "torch.distributed.distributed_c10d.GroupName",
):
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

    inp_size = inp.get_numel() * inp.get_dtype().itemsize
    return (
        config._collective.auto_select
        and is_symm_mem_enabled_for_group(group_name)
        and can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM)
        and reduce_op == "sum"
        and inp_size <= config._collective.one_shot_all_reduce_threshold_bytes
    )


def _one_shot_all_reduce(inp: ir.TensorBox, reduce_op, group_name):
    realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)
    return pytree.tree_map(
        ir.TensorBox.create,
        ir.FallbackKernel.create(
            torch.ops.symm_mem.one_shot_all_reduce.default,
            inp,
            reduce_op,
            group_name,
        ),
    )


def _create_out_of_place(kernel, inputs, *args) -> ir.IRNode:
    node = ir._CollectiveKernel.create_out_of_place(kernel, inputs, *args)
    if not isinstance(node, ir.IRNode):
        raise AssertionError(f"expected an `ir.IRNode`, got {type(node)}")
    return ir.TensorBox.create(node)


def register_comm_lowerings():
    """
    Register lowerings for the comm subsystem.
    """
    try:
        _ = torch.ops._c10d_functional.all_reduce
    except AttributeError:
        log.info(
            "Inductor support for distributed collectives depends on building "
            "torch.distributed"
        )
        return

    from .lowering import (
        add_layout_constraint,
        clone,
        constrain_to_fx_strides,
        copy_,
        register_lowering,
    )

    def register_comm_lowering(fn):
        add_layout_constraint(fn, constrain_to_fx_strides)
        return register_lowering(fn)

    c10d = torch.ops._c10d_functional

    @register_comm_lowering(c10d.all_reduce)  # type: ignore[misc]
    def _all_reduce(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
            return _one_shot_all_reduce(inp, reduce_op, group_name)

        # Lower as c10d.all_reduce_
        inp = clone(inp)
        if config.reorder_for_compute_comm_overlap:
            # The horizontal fusion of this clone often severely delays the
            # scheduling of the all_reduce_ node. Horizontally fusing this
            # clone can almost never out-perform scheduling the all_reduce_
            # earlier. Also in most cases, this clone is eliminated via
            # in-place reuse. Therefore, we tell the scheduler to not fuse it.
            inp.realize()
            V.graph.no_fuse_buffer_names.add(inp.get_name())
        # pyrefly: ignore [bad-assignment]
        inp = ir.ExternKernel.require_contiguous(inp)
        # Because we are lowering as inplace c10d.all_reduce_, we should generate
        # _AllReduce_Kernel instead of _AllReduceKernel.
        ir._AllReduce_Kernel.create_inplace(
            c10d.all_reduce_.default,
            inp,  # type: ignore[arg-type]
            reduce_op,
            group_name,  # type: ignore[arg-type]
        )
        return inp  # type: ignore[return-value]

    @register_comm_lowering(c10d.all_reduce_)  # type: ignore[misc]
    def _all_reduce_(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
            ret = copy_(
                inp,
                _one_shot_all_reduce(inp, reduce_op, group_name),
            )
            mark_as_skip_wait(ret)
            return inp

        # Lower as c10d.all_reduce_
        # pyrefly: ignore [bad-assignment]
        inp = ir.ExternKernel.require_contiguous(inp)
        ir._AllReduce_Kernel.create_inplace(
            c10d.all_reduce_.default,
            inp,  # type: ignore[arg-type]
            reduce_op,
            group_name,  # type: ignore[arg-type]
        )
        return inp  # type: ignore[return-value]

    @register_comm_lowering(c10d.all_reduce_coalesced)
    def _all_reduce_coalesced(inputs, reduce_op, group_name):
        inputs = [clone(inp) for inp in inputs]
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs

    @register_comm_lowering(c10d.all_reduce_coalesced_)
    def _all_reduce_coalesced_(inputs, reduce_op, group_name):
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs

    @register_comm_lowering(c10d.all_gather_into_tensor)
    def _all_gather_into_tensor(inp, group_size, group_name):
        return _create_out_of_place(
            c10d.all_gather_into_tensor.default,
            inp,
            group_size,
            group_name,
        )

    @register_comm_lowering(c10d.all_gather_into_tensor_coalesced)
    def _all_gather_into_tensor_coalesced(inputs, group_size, group_name):
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                c10d.all_gather_into_tensor_coalesced.default,
                inputs,
                group_size,
                group_name,
            ),
        )

    @register_comm_lowering(c10d.all_gather_into_tensor_out)
    def _all_gather_into_tensor_out(inp, group_size, group_name, *, out):
        ir._CollectiveKernel.create_inplace(
            c10d.all_gather_into_tensor_out.default,
            inp,
            group_size,
            group_name,
            out=out,
        )
        return out

    @register_comm_lowering(c10d.reduce_scatter_tensor)
    def _reduce_scatter_tensor(inp, reduce_op, group_size, group_name):
        return _create_out_of_place(
            c10d.reduce_scatter_tensor.default,
            inp,
            reduce_op,
            group_size,
            group_name,
        )

    @register_comm_lowering(c10d.reduce_scatter_tensor_out)
    def _reduce_scatter_tensor_out(inp, reduce_op, group_size, group_name, *, out):
        ir._CollectiveKernel.create_inplace(
            c10d.reduce_scatter_tensor_out.default,
            inp,
            reduce_op,
            group_size,
            group_name,
            out=out,
        )
        return out

    @register_comm_lowering(c10d.reduce_scatter_tensor_coalesced)
    def _reduce_scatter_tensor_coalesced(inputs, reduce_op, group_size, group_name):
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                c10d.reduce_scatter_tensor_coalesced.default,
                inputs,
                reduce_op,
                group_size,
                group_name,
            ),
        )

    @register_comm_lowering(c10d.all_to_all_single)
    def _all_to_all_single(inp, output_split_sizes, input_split_sizes, group_name):
        return _create_out_of_place(
            c10d.all_to_all_single.default,
            inp,
            output_split_sizes,
            input_split_sizes,
            group_name,
        )

    @register_comm_lowering(c10d.broadcast)
    def _broadcast(inp, src, group_name):
        inp = clone(inp)
        ir._CollectiveKernel.create_inplace(
            c10d.broadcast_.default, inp, src, group_name
        )
        return inp

    @register_comm_lowering(c10d.broadcast_)
    def _broadcast_(inp, src, group_name):
        ir._CollectiveKernel.create_inplace(
            c10d.broadcast_.default, inp, src, group_name
        )
        return inp

    @register_comm_lowering(torch.ops._dtensor.shard_dim_alltoall)
    def _shard_dim_alltoall(inp, gather_dim, shard_dim, group_name):
        return _create_out_of_place(
            torch.ops._dtensor.shard_dim_alltoall.default,
            inp,
            gather_dim,
            shard_dim,
            group_name,
        )

    @register_comm_lowering(c10d.wait_tensor)
    def _wait_tensor(inp):
        if should_skip_wait(inp):
            return inp

        ir._WaitKernel.create_wait(c10d.wait_tensor.default, inp)
        return inp

    @register_comm_lowering(c10d.isend)  # type: ignore[misc]
    def _isend(inp, dst, tag, group_name):
        inp = ir.ExternKernel.require_contiguous(inp)
        return _create_out_of_place(c10d.isend.default, inp, dst, tag, group_name)

    @register_comm_lowering(c10d.irecv)  # type: ignore[misc]
    def _irecv(inp, src, tag, group_name):
        inp = ir.ExternKernel.require_contiguous(inp)
        ir._CollectiveKernel.create_inplace(
            c10d.irecv.default, inp, src, tag, group_name
        )
        return inp

    @register_comm_lowering(c10d.batch_p2p_ops)  # type: ignore[misc]
    def _batch_p2p_ops(op_list, peer_list, tag_list, tensors, group_name):
        tensors = [ir.ExternKernel.require_contiguous(t) for t in tensors]
        kernel = c10d.batch_p2p_ops.default
        with V.graph.fake_mode:
            result = ir._CollectiveKernel.process_kernel(
                kernel,
                op_list,
                peer_list,
                tag_list,
                tensors,
                group_name,
            )
        example_output = result.example_output
        tensor_args = result.tensor_args
        non_tensor_args = result.non_tensor_args
        unflatten_args = result.unflatten_args
        if result.unbacked_bindings:
            raise AssertionError(f"{kernel} {result.unbacked_bindings}")
        for op, tensor_arg in zip(op_list, tensor_args):
            tensor_arg.realize()
            if op == "irecv":
                V.graph.mark_buffer_mutated(tensor_arg.get_name())

        device = tensor_args[0].get_device_or_error()
        packed = ir._CollectiveKernel(
            ir.MultiOutputLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )

        results = []
        for i, (op, t, ex_out) in enumerate(zip(op_list, tensors, example_output)):
            if op == "irecv":
                packed.mutation_outputs.append(
                    ir.MutationOutput(ir.NoneLayout(device=device), t, packed)
                )
                packed.alias_names.append(t.get_name())
                results.append(t)
            else:
                # isend: 0-element placeholder output connected to the collective
                placeholder = ir.MultiOutput(
                    ir._CollectiveKernel.tensor_to_layout(ex_out),
                    packed,
                    [(list, i)],
                )
                results.append(ir.TensorBox.create(placeholder))
        return results


def register_symm_mem_lowerings():
    """
    Register lowerings for symmetric memory (symm_mem) operations.
    """
    try:
        symm_mem = torch.ops.symm_mem
        # Check for an actual operation, not just the namespace.
        # torch.ops.symm_mem is a lazy namespace that always exists,
        # but the operations may not exist on non-CUDA platforms or
        # when USE_DISTRIBUTED is disabled.
        _ = symm_mem.one_shot_all_reduce
    except AttributeError:
        log.info("symm_mem ops not available, skipping symm_mem lowerings")
        return

    from torch._library._out_variant import register_out_variant

    # Register manual out variant mappings for symm_mem ops.
    register_out_variant(
        symm_mem.one_shot_all_reduce.default,
        symm_mem.one_shot_all_reduce_out.default,
    )
    register_out_variant(
        symm_mem.one_shot_all_reduce_copy.default,
        symm_mem.one_shot_all_reduce_copy_out.default,
    )

    from .lowering import register_lowering

    def _copy_input_to_comm_buffer(
        inp: ir.TensorBox,
        comm_buffer_type: ir.CommBufferType,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        """
        Fallback: insert a Pointwise identity copy allocated in P2P via
        CommBufferLayout.  Used when we don't control the input's allocation.
        """
        inp.realize()
        copy = ir.Pointwise.create(
            device=inp.get_device(),
            dtype=inp.get_dtype(),
            inner_fn=inp.make_loader(),
            ranges=inp.get_size(),
        )
        realize_as_comm_buffer(copy, comm_buffer_type, group_name)
        return copy

    def _maybe_realize_symm_mem(
        inp: ir.TensorBox,
        group_name: str,  # type: ignore[arg-type]
    ) -> ir.TensorBox:
        """
        Ensure inp is in P2P memory for a symm_mem collective.

        If inductor controls the buffer's allocation (ComputedBuffer,
        or any buffer with FlexibleLayout/FixedLayout), switch its
        layout to CommBufferLayout in-place, zero-copy.

        If inductor does not control allocation (e.g. InputBuffer),
        insert a Pointwise identity copy into a new CommBufferLayout buffer.
        This adds an extra Triton kernel. Returns the possibly new TensorBox.

        TODO(tianrengao): eliminate the extra kernel for static-shape
        InputBuffers by pre-allocating P2P memory in the wrapper and DMA .copy_()
        """
        if can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM):
            realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)  # type: ignore[arg-type]
            return inp
        else:
            return _copy_input_to_comm_buffer(
                inp,
                ir.CommBufferType.SYMM_MEM,
                group_name,  # type: ignore[arg-type]
            )

    def _create_low_contention_ag_out(
        kernel: torch._ops.OpOverload,
        inp: ir.TensorBox,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        from torch.distributed import distributed_c10d as c10d

        inp.realize()
        group_size = c10d._get_group_size_by_name(group_name)
        out_size = [inp.get_size()[0] * group_size, *inp.get_size()[1:]]
        layout = ir.CommBufferLayout(
            ir.FixedLayout(
                device=inp.get_device_or_error(),
                dtype=inp.get_dtype(),
                size=out_size,
            ),
            ir.CommBufferType.SYMM_MEM,
            group_name,
        )
        return ir.TensorBox.create(
            ir.ExternKernelOut(
                layout=layout,
                inputs=[inp],
                constant_args=(group_name,),
                op_overload=kernel,
            )
        )

    @register_lowering(symm_mem.one_shot_all_reduce)
    def _symm_mem_one_shot_all_reduce(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.one_shot_all_reduce.default,
                inp,
                reduce_op,
                group_name,
            ),
        )

    @register_lowering(symm_mem.one_shot_all_reduce_out)
    def _symm_mem_one_shot_all_reduce_out(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
        out: ir.TensorBox,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.one_shot_all_reduce_out.default,
                inp,
                reduce_op,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.one_shot_all_reduce_copy)
    def _symm_mem_one_shot_all_reduce_copy(
        symm_buffer: ir.TensorBox,
        local_input: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        symm_buffer = _maybe_realize_symm_mem(symm_buffer, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.one_shot_all_reduce_copy.default,
                symm_buffer,
                local_input,
                reduce_op,
                group_name,
            ),
        )

    @register_lowering(symm_mem.one_shot_all_reduce_copy_out)
    def _symm_mem_one_shot_all_reduce_copy_out(
        symm_buffer: ir.TensorBox,
        local_input: ir.TensorBox,
        reduce_op: str,
        group_name: str,
        out: ir.TensorBox,
    ):
        symm_buffer = _maybe_realize_symm_mem(symm_buffer, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.one_shot_all_reduce_copy_out.default,
                symm_buffer,
                local_input,
                reduce_op,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.two_shot_all_reduce_)
    def _symm_mem_two_shot_all_reduce_(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        ir.FallbackKernel.create(
            symm_mem.two_shot_all_reduce_.default,
            inp,
            reduce_op,
            group_name,
        )
        return inp

    @register_lowering(symm_mem.two_shot_all_reduce_out)
    def _symm_mem_two_shot_all_reduce_out(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
        output: ir.TensorBox,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.two_shot_all_reduce_out.default,
                inp,
                reduce_op,
                group_name,
                output,
            ),
        )

    @register_lowering(symm_mem.multimem_all_reduce_)
    def _symm_mem_multimem_all_reduce_(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        ir.FallbackKernel.create(
            symm_mem.multimem_all_reduce_.default,
            inp,
            reduce_op,
            group_name,
        )
        return inp

    @register_lowering(symm_mem.multimem_one_shot_all_reduce)
    def _symm_mem_multimem_one_shot_all_reduce(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.multimem_one_shot_all_reduce.default,
                inp,
                reduce_op,
                group_name,
            ),
        )

    @register_lowering(symm_mem.multimem_one_shot_all_reduce_out)
    def _symm_mem_multimem_one_shot_all_reduce_out(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
        out: ir.TensorBox,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.multimem_one_shot_all_reduce_out.default,
                inp,
                reduce_op,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.multimem_one_shot_reduce_out)
    def _symm_mem_multimem_one_shot_reduce_out(
        inp: ir.TensorBox,
        reduce_op: str,
        root: int,
        group_name: str,
        out: ir.TensorBox,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.multimem_one_shot_reduce_out.default,
                inp,
                reduce_op,
                root,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.multimem_all_gather_out)
    def _symm_mem_multimem_all_gather_out(
        inp: ir.TensorBox,
        group_name: str,
        out: ir.TensorBox,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.multimem_all_gather_out.default,
                inp,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem.reduce_scatter_out)
    def _symm_mem_reduce_scatter_out(
        inp: ir.TensorBox,
        group_name: str,
        split_last_dim: bool,
        output: ir.TensorBox,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem.reduce_scatter_out.default,
                inp,
                group_name,
                split_last_dim,
                output,
            ),
        )

    @register_lowering(symm_mem.all_to_all_vdev)
    def _symm_mem_all_to_all_vdev(
        inp: ir.TensorBox,
        out: ir.TensorBox,
        in_splits: ir.TensorBox,
        out_splits_offsets: ir.TensorBox,
        group_name: str,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        out = _maybe_realize_symm_mem(out, group_name)
        ir.FallbackKernel.create(
            symm_mem.all_to_all_vdev.default,
            inp,
            out,
            in_splits,
            out_splits_offsets,
            group_name,
        )
        return None

    @register_lowering(symm_mem.all_to_all_vdev_2d)
    def _symm_mem_all_to_all_vdev_2d(
        inp: ir.TensorBox,
        out: ir.TensorBox,
        in_splits: ir.TensorBox,
        out_splits_offsets: ir.TensorBox,
        group_name: str,
        major_align=None,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        out = _maybe_realize_symm_mem(out, group_name)
        ir.FallbackKernel.create(
            symm_mem.all_to_all_vdev_2d.default,
            inp,
            out,
            in_splits,
            out_splits_offsets,
            group_name,
            major_align,
        )
        return None

    @register_lowering(symm_mem.all_to_all_vdev_2d_offset)
    def _symm_mem_all_to_all_vdev_2d_offset(
        inp: ir.TensorBox,
        out: ir.TensorBox,
        in_splits_offsets: ir.TensorBox,
        out_splits_offsets: ir.TensorBox,
        group_name: str,
    ):
        inp = _maybe_realize_symm_mem(inp, group_name)
        out = _maybe_realize_symm_mem(out, group_name)
        ir.FallbackKernel.create(
            symm_mem.all_to_all_vdev_2d_offset.default,
            inp,
            out,
            in_splits_offsets,
            out_splits_offsets,
            group_name,
        )
        return None

    @register_lowering(symm_mem.tile_reduce)
    def _symm_mem_tile_reduce(
        in_tile: ir.TensorBox,
        out_tile: ir.TensorBox,
        root: int,
        group_name: str,
        reduce_op: str = "sum",
    ):
        in_tile = _maybe_realize_symm_mem(in_tile, group_name)
        out_tile = _maybe_realize_symm_mem(out_tile, group_name)
        ir.FallbackKernel.create(
            symm_mem.tile_reduce.default,
            in_tile,
            out_tile,
            root,
            group_name,
            reduce_op,
        )
        return None

    @register_lowering(symm_mem.multi_root_tile_reduce)
    def _symm_mem_multi_root_tile_reduce(
        in_tiles,  # list of TensorBox
        out_tile: ir.TensorBox,
        roots,  # list of int
        group_name: str,
        reduce_op: str = "sum",
    ):
        for i, in_tile in enumerate(in_tiles):
            in_tiles[i] = _maybe_realize_symm_mem(in_tile, group_name)
        out_tile = _maybe_realize_symm_mem(out_tile, group_name)
        ir.FallbackKernel.create(
            symm_mem.multi_root_tile_reduce.default,
            in_tiles,
            out_tile,
            roots,
            group_name,
            reduce_op,
        )
        return None

    @register_lowering(symm_mem._low_contention_all_gather)
    def _symm_mem_low_contention_all_gather(
        inp: ir.TensorBox,
        group_name: str,
    ):
        # Use _CollectiveKernel so that _WaitKernel.get_volatile_reads()
        # can track the input's lifetime through wait_tensor, preventing
        # the memory planner from reusing the input buffer while the
        # backend stream is still reading it.
        return _create_out_of_place(
            symm_mem._low_contention_all_gather.default,
            inp,
            group_name,
        )

    @register_lowering(symm_mem._low_contention_all_gather_ce_multicast)
    def _symm_mem_low_contention_all_gather_ce_multicast(
        inp: ir.TensorBox,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ):
        return _create_low_contention_ag_out(
            symm_mem._low_contention_all_gather_ce_multicast_out.default,
            inp,
            group_name,
        )

    @register_lowering(symm_mem._low_contention_all_gather_ce_multicast_out)
    def _symm_mem_low_contention_all_gather_ce_multicast_out(
        inp: ir.TensorBox,
        group_name: "torch.distributed.distributed_c10d.GroupName",
        out: ir.TensorBox,
    ):
        inp.realize()
        if can_realize_as_comm_buffer(out, ir.CommBufferType.SYMM_MEM):
            realize_as_comm_buffer(out, ir.CommBufferType.SYMM_MEM, group_name)
        else:
            raise RuntimeError(
                "_low_contention_all_gather_ce_multicast_out requires an "
                "Inductor-owned output buffer that can be realized as symmetric "
                "memory; this indicates an invalid lowering."
            )
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(
                symm_mem._low_contention_all_gather_ce_multicast_out.default,
                inp,
                group_name,
                out,
            ),
        )

    @register_lowering(symm_mem._low_contention_reduce_scatter)
    def _symm_mem_low_contention_reduce_scatter(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: str,
    ):
        # Use _CollectiveKernel so that _WaitKernel.get_volatile_reads()
        # can track the input's lifetime through wait_tensor, preventing
        # the memory planner from reusing the input buffer while the
        # backend stream is still reading it.
        return _create_out_of_place(
            symm_mem._low_contention_reduce_scatter.default,
            inp,
            reduce_op,
            group_name,
        )
<|endoftext|>from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import functools
import hashlib
import importlib
import inspect
import io
import itertools
import logging
import operator
import os
import platform
import re
import shutil
import statistics
import sys
import sysconfig
import tempfile
import time
import unittest
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSet,
)
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    cast,
    Concatenate,
    Generic,
    Literal,
    Protocol,
    TYPE_CHECKING,
    TypeAlias,
    TypeGuard,
)
from typing_extensions import dataclass_transform, ParamSpec, TypeVar
from unittest import mock

import sympy

import torch
import torch.utils._pytree as pytree
from torch._inductor.analysis.device_info import datasheet_tops
from torch._inductor.runtime.hints import DeviceProperties
from torch.fx.passes.regional_inductor import _needs_inductor_compile
from torch.utils._dtype_abbrs import dtype_abbrs

# The IndentedBuffer primitive and its line-map helpers live in torch.utils so
# they can be shared across layers (dynamo guards, AOTAutograd codegen) without
# importing this heavy module. Re-exported here for existing inductor call sites.
from torch.utils._indented_buffer import (
    DeferredLineBase,
    IndentedBuffer,
    LineContext,  # noqa: F401
    ValueWithLineMap,  # noqa: F401
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.utils._triton import has_triton_package


OPTIMUS_EXCLUDE_POST_GRAD = [
    "activation_quantization_aten_pass",
    "inductor_autotune_lookup_table",
]

from torch.fx.experimental._size_hinting import _sympy_subs
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    IterateExprs,
    ShapeEnv,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence, ValuesView
    from pathlib import Path

    from torch import SymBool, SymFloat, SymInt
    from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
    from torch.fx import GraphModule
    from torch.fx.node import Node
    from torch.nn.functional import ScalingType  # type: ignore[attr-defined]

    from .codegen.common import WorkspaceArg
    from .codegen.wrapper import PythonWrapperCodegen
    from .dependencies import Dep
    from .graph import GraphLowering
    from .ir import Buffer, ExternKernel, IRNode, Layout, Operation, ReinterpretView
    from .output_code import CompiledFxGraph
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]
T = TypeVar("T")


# defines here before import torch._dynamo is for avoiding circular import
# when get_gpu_type is imported from dynamo
@functools.cache
def get_gpu_type() -> str:
    avail_gpus = [x for x in GPU_TYPES if getattr(torch, x).is_available()]
    if not len(avail_gpus) <= 1:
        raise AssertionError(
            f"Expected at most 1 available GPU type, got {len(avail_gpus)}: {avail_gpus}"
        )
    gpu_type = "cuda" if len(avail_gpus) == 0 else avail_gpus.pop()
    return gpu_type


from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import detect_fake_mode
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils._sympy.functions import (
    CeilDiv,
    CleanDiv,
    FloorDiv,
    Identity,
    Max,
    ModularIndexing,
)
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from . import config
from .runtime.runtime_utils import ceildiv as runtime_ceildiv


_IS_WINDOWS = sys.platform == "win32"

log = logging.getLogger(__name__)


_DO_BENCH_PROFILE_EVENT_NAME = "inductor_do_bench_using_profiling"


_T = TypeVar("_T")
VarRanges = dict[sympy.Expr, sympy.Expr]
InputType = torch.Tensor | int | torch.SymInt | None

XPU_KERNEL_FORMAT = (
    "spv" if _IS_WINDOWS else os.getenv("TORCHINDUCTOR_XPU_KERNEL_FORMAT", "zebin")
)

GPU_KERNEL_BIN_EXTS = {
    "cuda": ".cubin",
    "hip": ".hsaco",
    "xpu": f".{XPU_KERNEL_FORMAT}",
}

GPU_ALIGN_BYTES = 16
ALIGNMENT = 16

TMA_ALIGNMENT = 16
TMA_DESCRIPTOR_SIZE = 128

# PyTorch dtypes with valid CUtensorMapDataType mappings.
# Ref: triton/backends/nvidia/include/cuda.h (CUtensorMapDataType enum)
#      triton/_internal_testing.py (tma_dtypes test list)
_TMA_SUPPORTED_DTYPES: OrderedSet[torch.dtype] = OrderedSet(
    [
        torch.uint8,
        torch.int8,
        torch.uint16,
        torch.int16,
        torch.uint32,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ]
)

ALIGN_BYTES = 64
if not ((ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0 and ALIGN_BYTES >= 8):
    raise AssertionError("must be power of 2")


def _align(nbytes: int) -> int:
    """Round up to the nearest multiple of ALIGN_BYTES"""
    return (nbytes + ALIGN_BYTES - 1) & -ALIGN_BYTES


def _is_aligned(v: sympy.Expr) -> bool:
    """v can be statically proven to be a multiple of ALIGN_BYTES"""
    if isinstance(v, (sympy.Add, sympy.Max, Max)):
        return all(map(_is_aligned, v.args))
    return isinstance(v, align) or sympy.gcd(v, ALIGN_BYTES) == ALIGN_BYTES


class align(sympy.Function):
    """Symbolically round up to the nearest multiple of ALIGN_BYTES"""

    nargs = (1,)
    is_integer = True

    @classmethod
    def eval(cls, value: sympy.Expr) -> sympy.Expr | None:
        if isinstance(value, (int, sympy.Integer)):
            return _align(int(value))
        if _is_aligned(value):
            return value


@dataclasses.dataclass(frozen=True)
class GraphPartitionMap:
    """
    Mapping from the partition info (e.g., input/output) to the graph info
    """

    # a unique id of graph partition
    id: int

    # map partition input indices to graph input indices. None indicates a
    # partition input is not a graph input.
    input_index_mapping: list[int | None]
    # map partition output indices to graph output indices. Empty indicates a
    # partition output is not a graph output. Multiple graph outputs can map to
    # the same partition output when graph outputs alias the same buffer.
    output_index_mapping: list[list[int]]

    # name of constants read/written by the graph partition
    constant_names: list[str]


def fp8_bench(fn: Callable[[], Any], warmup: int = 25, rep: int = 100) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """

    fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.float16, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        torch.cuda.synchronize()
        for i in range(n_repeat):
            cache.zero_()
            start_event[i].record()
            with torch.cuda.nvtx.range("RunCudaModule"):
                fn()
            end_event[i].record()
        torch.cuda.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
        )

    res = torch.mean(times).item()
    log.debug("raw events")
    log.debug(p.key_averages().table(sort_by="self_device_time_total", row_limit=-1))
    filtered_events = EventList(
        [
            event
            for event in p.events()
            if (
                event.device_type == DeviceType.CUDA
                and re.match(r"fused_abs_max_\d", event.name) is not None
            )
        ]
    )
    if filtered_events:
        res -= (
            statistics.mean(event.device_time_total for event in filtered_events)
            / 1000.0
        )

    log.debug("profiling results: %s ms", res)
    return res


def do_bench_using_profiling(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    is_vetted_benchmarking: bool = False,
) -> float:
    # We did't use decorator may_distort_benchmarking_result directly since that
    # requires us to import torch._inductor.runtime.benchmarking into global scope.
    # Importing torch._inductor.runtime.benchmarking will cause cuda initialization
    # (because of calling torch.cuda.available in global scope)
    # which cause failure in vllm when it create child processes. Check log:
    #   https://gist.github.com/shunting314/c194e147bf981e58df095c14874dd65a
    #
    # Another way to solve the issue is to just move do_bench_using_profiling
    # to torch._inductor.runtime.benchmarking and change all the call site.
    # But that's not trivial due to so many call sites in and out of pytorch.

    from torch._inductor.runtime.benchmarking import (
        gpu_benchmark_lock,
        may_distort_benchmarking_result,
    )

    locked_bench = gpu_benchmark_lock(_do_bench_using_profiling)
    return may_distort_benchmarking_result(locked_bench)(
        fn, warmup, rep, is_vetted_benchmarking
    )


def _get_do_bench_profile_result(
    kineto_events: Iterable[Any],
    profiler_events: Iterable[Any],
    n_repeat: int,
    expected_device_type: DeviceType,
) -> float:
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
        if event.name == _DO_BENCH_PROFILE_EVENT_NAME
        and event.device_type == DeviceType.CPU
    ]
    if len(benchmark_events) != n_repeat:
        raise RuntimeError(
            f"Expected {n_repeat} {_DO_BENCH_PROFILE_EVENT_NAME} profiling events. "
            f"Found {len(benchmark_events)} events."
        )

    for event in benchmark_events:
        collect_cpu_event_ids(event)

    device_time_us = 0.0
    for event in kineto_events:
        linked_correlation_id = event.linked_correlation_id()
        correlation_id = event.correlation_id()
        activity_type = event.activity_type()
        if (
            event.device_type() == expected_device_type
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

    if device_time_us <= 0:
        raise RuntimeError(
            f"Failed to capture device events for {_DO_BENCH_PROFILE_EVENT_NAME}."
        )

    return device_time_us / 1000.0 / n_repeat


def _do_bench_using_profiling(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    is_vetted_benchmarking: bool = False,
) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    The benchmarked function is wrapped in a profiler record_function so
    cache-clearing kernels can be excluded without relying on raw CUDA event
    grouping or kernel names.
    """

    if not is_vetted_benchmarking:
        from torch._inductor.runtime.benchmarking import may_ban_benchmarking

        may_ban_benchmarking()

    device_type = get_gpu_type()
    device_type_upper = device_type.upper()
    device_interface = get_interface_for_device(device_type)
    fn()
    device_interface.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device_type)

    # Estimate the runtime of the function
    start_event = device_interface.Event(enable_timing=True)
    end_event = device_interface.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    device_interface.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()

    device_interface.synchronize()
    profile_activity = getattr(torch.profiler.ProfilerActivity, device_type_upper)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            profile_activity,
        ]
    ) as p:
        # Benchmark
        for _ in range(n_repeat):
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            with torch.profiler.record_function(_DO_BENCH_PROFILE_EVENT_NAME):
                fn()
        # Record clocks
        device_interface.synchronize()

    log.debug("raw events")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            p.key_averages().table(sort_by="self_device_time_total", row_limit=-1)
        )

    result = _get_do_bench_profile_result(
        p.profiler.kineto_results.events(),
        p.events(),
        n_repeat,
        getattr(DeviceType, device_type_upper),
    )

    log.debug("profiling time breakdown")
    log.debug("profiling results: %s ms", result)
    return result


@functools.cache
def has_torchvision_roi_align() -> bool:
    try:
        from torchvision.ops import roi_align

        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except ImportError:
        return False
    except RuntimeError as e:
        if "torchvision::nms does not exist" not in str(e):
            raise AssertionError(f"Unexpected RuntimeError: {e}") from e
        return False


def decode_device(device: torch.device | None | str) -> torch.device:
    if device is None:
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        device = torch.device(device)
    if device.type not in ("cpu", "meta") and device.index is None:
        device_interface = get_interface_for_device(device.type)
        return torch.device(device.type, index=device_interface.Worker.current_device())
    return device


def sympy_product(it: Iterable[sympy.Expr]) -> sympy.Expr:
    return functools.reduce(operator.mul, it, sympy.S.One)


def sympy_dot(seq1: Sequence[sympy.Expr], seq2: Sequence[sympy.Expr]) -> sympy.Expr:
    if not len(seq1) == len(seq2):
        raise AssertionError(
            f"Length mismatch: len(seq1)={len(seq1)}, len(seq2)={len(seq2)}"
        )
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def flatten_index(
    indices: Sequence[sympy.Expr],
    sizes: Sequence[sympy.Expr],
) -> sympy.Expr:
    """Row-major flatten: per-dimension indices -> flat index."""
    if len(indices) != len(sizes):
        raise AssertionError(
            f"indices and sizes must have equal length, got "
            f"{len(indices)} and {len(sizes)}"
        )
    flat = sympy.S.Zero
    for index, size in zip(indices, sizes):
        flat = flat * size + index
    return flat


def decompose_index(
    index: sympy.Expr,
    sizes: Sequence[sympy.Expr],
) -> list[sympy.Expr]:
    """Row-major decomposition: flat index -> per-dimension indices."""
    return [
        ModularIndexing(index, sympy_product(sizes[i + 1 :]), size)
        for i, size in enumerate(sizes)
    ]


def unique(it: Iterable[_T]) -> ValuesView[_T]:
    return {id(x): x for x in it}.values()


def ceildiv(number: int | sympy.Expr, denom: int | sympy.Expr) -> int | sympy.Expr:
    if isinstance(number, sympy.Expr) or isinstance(denom, sympy.Expr):
        return CeilDiv(sympy.sympify(number), sympy.sympify(denom))
    # TODO: There is a bug in a call to this function, to repro:
    # python benchmarks/dynamo/huggingface.py --inductor -d cuda --accuracy
    # --amp --only YituTechConvBert --dynamic-shapes
    if not (isinstance(number, int) and isinstance(denom, int)):
        raise AssertionError(f"{number}: {type(number)}, {denom}: {type(denom)}")
    return runtime_ceildiv(number, denom)


def _type_of(key: torch.dtype | None) -> str:
    # Use the function here to get rid of dependencies on the Triton during the codegen.
    # Refer to Triton implementation here:
    # https://github.com/triton-lang/triton/blob/98b5945d2aef679e00ebca8e07c35c3658ec76de/python/triton/runtime/jit.py#L238
    # `None` is nullptr.  Implicitly convert to *i8.
    if key is None:
        return "*i8"
    dtype_str = str(key).split(".")[-1]
    tys = {
        "bool": "i1",
        "float8e4nv": "fp8e4nv",
        "float8e5": "fp8e5",
        "float8e4b15": "fp8e4b15",
        "float8e4b15x4": "fp8e4b15x4",
        "float8_e4m3fn": "fp8e4nv",
        "float8_e5m2": "fp8e5",
        "float8_e4m3fnuz": "fp8e4b8",
        "float8_e5m2fnuz": "fp8e5b16",
        # TODO: remove when support is added in triton
        # https://github.com/triton-lang/triton/issues/6054
        "float8_e8m0fnu": "u8",
        "float4_e2m1fn_x2": "u8",
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }
    # reinterpret can create triton type
    tys.update({v: v for v in list(tys.values())})
    return key if isinstance(key, str) else f"*{tys[dtype_str]}"


def convert_shape_to_inductor(
    lst: Iterable[int | torch.SymInt],
) -> list[sympy.Expr]:
    """
    Gets the shape and stride of a tensor. For non-symbolic tensors, this is
    trivial. But for symbolic tensors, we need to map from SymIntNode into
    sympy.Expr.
    """
    return [sympy.sympify(i) for i in lst]


def convert_symint_to_expr(val: int | torch.SymInt) -> int | sympy.Expr:
    """
    Convert SymInt to sympy.Expr, leave int as is.

    Unlike sympy.sympify() which converts int to sympy.Integer,
    this function preserves int as int and only converts SymInt to Expr.
    """
    if isinstance(val, torch.SymInt):
        return val.node.expr
    return val


def convert_to_symint(i: int | sympy.Expr) -> int | torch.SymInt:
    """
    Like convert_shape_to_symint, but operates on a single expression.
    """
    from .virtualized import V

    return (
        i
        if isinstance(i, int)
        else (
            int(i)
            if isinstance(i, sympy.Integer)
            else V.graph.sizevars.shape_env.create_symintnode(i, hint=None)
        )
    )


def convert_shape_to_symint(
    lst: Iterable[int | sympy.Expr],
) -> list[int | torch.SymInt]:
    """
    Takes a list of shapes from Inductor and converts them into symints (or just
    ints if all shapes are static).
    """
    return [convert_to_symint(i) for i in lst]


def is_view(op: torch._ops.OpOverload) -> bool:
    """
    Does this op overload have aliasing
    """
    return any(a.alias_info is not None for a in op._schema.arguments)


def is_pointwise_use(
    use: Node,
    is_pointwise_fn: Callable[[torch._ops.OpOverload], bool] = lambda _: False,
) -> bool:
    """
    Do all uses of this op have torch.Tag.pointwise or return True for optional `is_pointwise_fn`

    Uses in views ops will follow the views uses
    """

    if use.op != "call_function":
        return False
    if not (
        isinstance(use.target, torch._ops.OpOverload) or use.target is operator.getitem
    ):
        return False

    target = cast(torch._ops.OpOverload, use.target)
    if target is operator.getitem or is_view(target):
        return all(is_pointwise_use(u, is_pointwise_fn) for u in use.users)

    return torch.Tag.pointwise in target.tags or is_pointwise_fn(target)


def gen_gm_and_inputs(
    target: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[GraphModule, list[torch.Tensor]]:
    g = torch.fx.Graph()
    graph_args: list[torch.Tensor] = []

    def add_tensor_arg(arg: torch.Tensor) -> Node:
        graph_args.append(arg)
        return g.placeholder(f"arg{len(graph_args)}")

    node = g.call_function(
        target, *tree_map_only(torch.Tensor, add_tensor_arg, (args, kwargs))
    )
    if (
        len(target._schema.returns) == 1
        and str(target._schema.returns[0].type) == "Tensor"
    ):
        node = (node,)  # type: ignore[assignment]
    g.output(node)

    gm = torch.fx.GraphModule({}, g)
    return gm, graph_args


def synchronize(device: str = "cuda") -> None:
    if device == "cpu":
        return
    device_interface = get_interface_for_device(device)
    if device_interface.is_available():
        device_interface.synchronize()


def timed(
    model: Callable[..., Any],
    example_inputs: Sequence[Any],
    times: int = 1,
    device: str = "cuda",
) -> float:
    synchronize(device)
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize(device)
    t1 = time.perf_counter()
    # GC the result after timing
    if result is None:  # type: ignore[possibly-undefined]
        raise AssertionError("Expected result to be not None after timed execution")
    return t1 - t0


def print_performance(
    model: Callable[..., Any],
    example_inputs: Sequence[Any] = (),
    times: int = 10,
    repeat: int = 10,
    baseline: float = 1.0,
    device: str = "cuda",
) -> float:
    timings = torch.tensor(
        [timed(model, example_inputs, times, device) for _ in range(repeat)]
    )
    took = torch.median(timings) / times
    print(f"{took / baseline:.6f}")
    return took.item()


def precompute_method(obj: Any, method: str) -> None:
    """Replace obj.method() with a new method that returns a precomputed constant."""
    result = getattr(obj, method)()
    setattr(obj, method, lambda: result)


def precompute_methods(obj: Any, methods: list[str]) -> None:
    """Replace methods with new methods that returns a precomputed constants."""
    for method in methods:
        precompute_method(obj, method)


def cmp(a: int, b: int) -> int:
    return int(a > b) - int(a < b)


def pad_listlike(x: int | Sequence[int], size: int) -> Sequence[int]:
    if isinstance(x, int):
        return [x] * size
    if len(x) == 1:
        return type(x)([x[0]]) * size  # type: ignore[call-arg, operator, return-value]
    return x


# Used to ensure that iterating over a set is deterministic
def tuple_sorted(x: tuple[_T, ...]) -> list[_T]:
    if len(x) == 0:
        return []

    def sort_func(elem: _T) -> str:
        if isinstance(elem, str):
            return elem

        from .scheduler import BaseSchedulerNode

        if not isinstance(elem, BaseSchedulerNode):
            raise AssertionError(f"Expected BaseSchedulerNode, got {type(elem)}")
        return elem.get_name()

    return sorted(x, key=sort_func)


P = ParamSpec("P")
RV = TypeVar("RV", covariant=True)
FN_TYPE = Callable[Concatenate[Any, P], RV]


class CachedMethod(Protocol, Generic[P, RV]):
    @staticmethod
    def clear_cache(cache: Any) -> None: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> RV: ...


# See https://github.com/python/mypy/issues/13222#issuecomment-1193073470 to understand the type signature
def cache_on_self(fn: Callable[Concatenate[Any, P], RV]) -> CachedMethod[P, RV]:
    name = fn.__name__
    key = f"__{name}_cache"

    # wrapper is likely on the hot path, compile a specialized version of it
    ctx = {"fn": fn}
    exec(
        f"""\
        def {name}_cache_on_self(self):
            try:
                return self.{key}
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "{key}", rv)
            return rv
        """.lstrip(),
        ctx,
    )
    wrapper = functools.wraps(fn)(ctx[f"{name}_cache_on_self"])

    def clear_cache(self: Any) -> None:
        if hasattr(self, key):
            delattr(self, key)

    wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def cache_property_on_self(
    fn: Callable[Concatenate[Any, P], RV],
) -> CachedMethod[P, RV]:
    """
    Variant of cache_on_self for properties. The only difference is the type signature.
    """

    return cache_on_self(fn)


def cache_on_self_and_args(
    class_name: str,
) -> Callable[[FN_TYPE[P, RV]], FN_TYPE[P, RV]]:
    # include both class_name and fn_name in the key to support `super().fn(self, **args, **kwargs)` calls.

    def wrapper(
        fn: FN_TYPE[P, RV],
    ) -> FN_TYPE[P, RV]:
        key = f"__{class_name}_{fn.__name__}_cache"

        # wrapper is likely on the hot path, compile a specialized version of it
        ctx = {"fn": fn}
        exec(
            f"""\
            def inner(self: Any, *args: P.args, **kwargs: P.kwargs) -> RV:
                args_kwargs = (args, tuple(sorted(kwargs.items())))

                if not hasattr(self, "{key}"):
                    object.__setattr__(self, "{key}", {{}})

                cache = self.{key}

                try:
                    return cache[args_kwargs]
                except KeyError:
                    pass

                rv = fn(self, *args, **kwargs)

                cache[args_kwargs] = rv
                return rv
            """.lstrip(),
            ctx,
        )
        inner = functools.wraps(fn)(ctx["inner"])

        def clear_cache(self: Any) -> None:
            if hasattr(self, key):
                delattr(self, key)

        inner.clear_cache = clear_cache  # type: ignore[attr-defined]
        return inner

    return wrapper


def aggregate_origins(
    node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
) -> OrderedSet[Node]:
    from . import ir

    if isinstance(node_schedule, list):
        return functools.reduce(
            operator.or_,
            [
                # pyrefly: ignore [missing-attribute]
                node.node.origins
                for node in node_schedule
                if hasattr(node, "node") and node.node
            ],
            OrderedSet(),
        )
    elif isinstance(node_schedule, ir.ExternKernel):
        return node_schedule.origins
    else:
        return OrderedSet()


def get_fused_kernel_name(
    node_schedule: Sequence[BaseSchedulerNode],
    descriptive_names: Literal[True, "torch", "original_aten", "inductor_node"],
) -> str:
    all_origins = aggregate_origins(node_schedule)
    if descriptive_names == "original_aten":

        def get_origin_meta_str(origin):
            original_aten = origin.meta["original_aten"]
            key = ""
            if isinstance(original_aten, torch._ops.OpOverload):
                key = original_aten._overloadpacket.__name__
            elif isinstance(original_aten, torch._ops.HigherOrderOperator):
                key = str(original_aten.name())
            return key

        # Bases the kernel name off of the top-level aten operator (i.e. pre-decompositions)
        sources = [
            get_origin_meta_str(origin)
            for origin in all_origins
            if origin.op == "call_function"
            and "original_aten" in origin.meta
            and origin.meta["original_aten"] is not None
        ]
        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "torch":
        # Bases the kernel name off of the top-level "torch" operator (i.e. post-dynamo graph)
        sources = []
        for origin in all_origins:
            if origin.op == "call_function":
                source_fn = None
                suffix = ""
                if "source_fn_stack" in origin.meta:
                    source_fn = origin.meta["source_fn_stack"][-1]
                elif "fwd_source_fn_stack" in origin.meta:
                    # backward nodes have "fwd_source_fn_stack" instead
                    source_fn = origin.meta["fwd_source_fn_stack"][-1]
                    suffix = "backward"
                if not source_fn:
                    continue
                if isinstance(source_fn[1], str):
                    sources.append(source_fn[1] + suffix)
                else:
                    sources.append(source_fn[1].__name__ + suffix)

        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "inductor_node":
        sources = [
            origin.name for origin in all_origins if origin.op == "call_function"
        ]
    else:
        raise NotImplementedError
    name = "_".join(["fused"] + sources)
    # On Windows the default MAX_PATH (260) limit means a long descriptive
    # kernel name can push the Triton cache path past the limit, making the
    # generated .ttir unopenable. Cap the name and append a hash to keep it
    # both short and unique.
    if is_windows() and len(name) > 50:
        h = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:41].rstrip('_')}_{h}"
    return name


def get_kernel_metadata(
    node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
    wrapper: PythonWrapperCodegen,
) -> tuple[str, str]:
    """
    Retrieves metadata information for a kernel.
    Args:
        node_schedule (Union[Sequence[BaseSchedulerNode], ExternKernel]):
            Either a sequence of BaseSchedulerNode objects or an ExternKernel instance.
        wrapper (PythonWrapperCodegen):
            An instance of PythonWrapperCodegen, used to define the code comment format.
    Returns:
        tuple[str, str]:
            A tuple containing two strings:
                - The first string represents the kernel's metadata.
                - The second string represent the kernel's detailed metadata.
    """

    all_origins = aggregate_origins(node_schedule)
    inductor_nodes = [origin for origin in all_origins if origin.op == "call_function"]

    from_node_dict = collections.defaultdict(list)
    original_aten_dict = collections.defaultdict(list)

    # Attempt to sort `inductor_nodes` topologically. Note that the case
    # where `inductor_nodes` contains nodes from multiple graph instances
    # is not supported. An example of this is conditional statements.
    single_graph = None
    if inductor_nodes:
        unique_graphs = OrderedSet(n.graph for n in inductor_nodes)
        if len(unique_graphs) == 1:
            single_graph = inductor_nodes[0].graph
            # create a map of idx -> node and cache it
            if not hasattr(single_graph, "_inductor_kernel_metadata_node_to_idx_map"):
                node_to_idx_map = {n: idx for idx, n in enumerate(single_graph.nodes)}
                single_graph._inductor_kernel_metadata_node_to_idx_map = node_to_idx_map  # type: ignore[attr-defined]
            inductor_nodes.sort(
                key=lambda n: single_graph._inductor_kernel_metadata_node_to_idx_map[n]  # type: ignore[attr-defined]
            )

    for node in inductor_nodes:
        if "original_aten" in node.meta and node.meta["original_aten"] is not None:
            original_aten = node.meta["original_aten"]
            key = None
            if isinstance(original_aten, torch._ops.OpOverload):
                key = str(original_aten._overloadpacket)
            elif isinstance(original_aten, torch._ops.HigherOrderOperator):
                key = str(original_aten.name())
            if key:
                original_aten_dict[key].append(node.name)
        if "from_node" in node.meta:
            key = node.meta["from_node"][0].name
            from_node_dict[key].append(node.name)
        elif node.meta.get("partitioner_tag") == "is_backward":
            # backward nodes currently don't have a "from node"
            from_node_dict[node.name].append(node.name)
    sort_str = "Topologically Sorted" if single_graph is not None else "Unsorted"
    metadata = (
        f"{wrapper.comment} {sort_str} Source Nodes: [{', '.join(from_node_dict.keys())}], "
        f"Original ATen: [{', '.join(original_aten_dict.keys())}]"
    )

    # trace back to original node here
    detailed_metadata = [f"{wrapper.comment} Source node to ATen node mapping:"]
    for original_node, nodes in sorted(from_node_dict.items()):
        detailed_metadata.append(
            f"{wrapper.comment}   {original_node} => {', '.join(sorted(nodes))}"
        )

    # print the aot_autograd graph fragment
    if single_graph is not None:
        from . import ir

        detailed_metadata.append(f"{wrapper.comment} Graph fragment:")
        all_reads: OrderedSet[str] = OrderedSet()
        all_writes: list[str] = []
        if not isinstance(node_schedule, ir.ExternKernel):
            from .virtualized import V

            def get_buffer_info(
                buffer: ir.TensorBox | ir.Buffer | ir.TorchBindObject, rw_name: str
            ) -> tuple[str, ir.Layout | None]:
                if isinstance(buffer, ir.TensorBox) and isinstance(
                    buffer.data, ir.StorageBox
                ):
                    origin_node = buffer.data.data.origin_node
                else:
                    origin_node = buffer.origin_node
                if origin_node is None:
                    # use the read/write name if no origin node is found
                    name = rw_name
                else:
                    name = origin_node.name
                try:
                    layout = buffer.get_layout()
                except NotImplementedError:
                    layout = None
                return name, layout

            def stringify_shape(shape: Iterable[int]) -> str:
                return f"[{', '.join([str(x) for x in shape])}]"

            def stringfy_layout(layout: ir.Layout | None) -> str:
                if layout is None:
                    return ""
                shape_annotation = f"{stringify_shape(layout.size)}"
                stride_annotation = f"{stringify_shape(layout.stride)}"
                # Under compile-on-one-rank, render the bare device type so this kernel
                # provenance comment is byte-identical across ranks.
                from torch.fx.experimental.proxy_tensor import _coor_enabled

                device = layout.device
                device_annotation = (
                    device.type
                    if (_coor_enabled() and device is not None)
                    else f"{device}"
                )

                return (
                    f'"{dtype_abbrs[layout.dtype]}{shape_annotation}'
                    f'{stride_annotation}{device_annotation}"'
                )

            for n in node_schedule:
                if not hasattr(n, "read_writes") or n.read_writes is None:
                    continue
                if hasattr(n.read_writes, "reads") and n.read_writes.reads is not None:
                    for r in n.read_writes.reads:
                        # Remove the duplicated inputs
                        if r.name in all_reads:
                            continue
                        all_reads.add(r.name)
                        buffer = V.graph.try_get_buffer(r.name)
                        if buffer is None:
                            continue
                        input_name, layout = get_buffer_info(buffer, r.name)
                        detailed_metadata.append(
                            f"{wrapper.comment}   %{input_name} : Tensor "
                            f"{stringfy_layout(layout)} = PlaceHolder[target={input_name}]"
                        )

                if (
                    hasattr(n.read_writes, "writes")
                    and n.read_writes.writes is not None
                ):
                    for w in n.read_writes.writes:
                        buffer = V.graph.try_get_buffer(w.name)
                        if buffer is None:
                            continue
                        output_name, _ = get_buffer_info(buffer, w.name)

                        all_writes.append("%" + output_name)

        for node in inductor_nodes:
            formatted_node = node.format_node(include_tensor_metadata=True)
            if formatted_node is not None and torch.version.hip:
                # AMDGCN asm strings can contain newlines, which propagate
                # into format_node() output.  Split so every line gets the
                # comment prefix; otherwise bare newlines break the wrapper.
                detailed_metadata.extend(
                    f"{wrapper.comment}   {line}"
                    for line in formatted_node.splitlines()
                )
            else:
                detailed_metadata.append(f"{wrapper.comment}   {formatted_node}")

        detailed_metadata.append(f"{wrapper.comment}   return {','.join(all_writes)}")

    return metadata, "\n".join(detailed_metadata)


def dominated_nodes(
    initial_queue: Iterable[torch.fx.Node],
    skip_filter: Callable[[Any], bool] | None = None,
) -> OrderedSet[torch.fx.Node]:
    """Returns the set of nodes whose values depend on those within initial_queue"""
    initial_queue = list(initial_queue)
    dominated_set = OrderedSet(initial_queue)

    while initial_queue:
        node = initial_queue.pop()
        for user in node.users:
            if skip_filter and skip_filter(user):
                continue
            if user not in dominated_set:
                dominated_set.add(user)
                initial_queue.append(user)

    return dominated_set


def gather_origins(
    args: Sequence[IRNode], kwargs: dict[str, IRNode]
) -> OrderedSet[torch.fx.Node]:
    from . import ir

    def is_unrealized_node(n: IRNode) -> bool:
        if isinstance(n, ir.TensorBox):
            return is_unrealized_node(n.data)
        if isinstance(n, ir.StorageBox):
            return is_unrealized_node(n.data)
        return isinstance(n, ir.IRNode) and not isinstance(
            n,
            (
                ir.ComputedBuffer,
                ir.InputsKernel,
                ir.InputBuffer,
                ir.TemplateBuffer,
            ),
        )

