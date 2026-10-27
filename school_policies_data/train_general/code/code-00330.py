    # kwargs and args may include a container of node, for example torch.cat([t1, t2])
    # flatten them before search the unrealized nodes
    kwargs_flatten, _ = tree_flatten(kwargs)
    kwargs_origins = [val.origins for val in kwargs_flatten if is_unrealized_node(val)]
    args_flatten, _ = tree_flatten(args)
    args_origins = [val.origins for val in args_flatten if is_unrealized_node(val)]
    return OrderedSet(itertools.chain(*args_origins, *kwargs_origins))


def sympy_str(expr: sympy.Expr) -> str:
    """
    Normal sympy str is very slow, this is a lot faster.  The result are
    somewhat worse, as it doesn't do as much simplification.  So don't
    use this for final codegen.
    """

    def is_neg_lead(expr: sympy.Expr) -> bool:
        return (
            isinstance(expr, sympy.Mul) and len(expr.args) == 2 and expr.args[0] == -1
        )

    def sympy_str_add(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Add):
            # Special case 'a - b'. Note that 'a - b - c' will still appear as
            # 'a + -1 * b + -1 * c'.
            if len(expr.args) == 2 and is_neg_lead(expr.args[1]):
                return f"{sympy_str_mul(expr.args[0])} - {sympy_str_mul(expr.args[1].args[1])}"
            else:
                return " + ".join(map(sympy_str_mul, expr.args))
        else:
            return sympy_str_mul(expr)

    def sympy_str_mul(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Mul):
            if is_neg_lead(expr):
                # Special case '-a'. Note that 'a * -b' will still appear as
                # '-1 * a * b'.
                return f"-{sympy_str_atom(expr.args[1])}"
            else:
                return " * ".join(map(sympy_str_atom, expr.args))
        else:
            return sympy_str_atom(expr)

    def sympy_str_atom(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Symbol):
            return expr.name
        elif isinstance(expr, (sympy.Add, sympy.Mul)):
            return f"({sympy_str_add(expr)})"
        elif isinstance(expr, (ModularIndexing, CleanDiv, FloorDiv, Identity)):
            return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})"
        else:
            return str(expr)

    return sympy_str_add(expr)


def get_bounds_index_expr(index: sympy.Expr) -> ValueRanges[Any]:
    from .virtualized import V

    # If this expression does not come from an FX node, we compute its bounds
    if (
        config.compute_all_bounds
        and (fx_node := getattr(V.interpreter, "current_node", None))
        and fx_node.target != "index_expr"
    ):
        return bound_sympy(index)
    else:
        return ValueRanges.unknown()


def prefix_is_reduction(prefix: str) -> bool:
    return prefix[0] == "r"


def prefix_is_pointwise(prefix: str) -> bool:
    return prefix[0] in ("x", "y", "z")


def sympy_index_symbol_with_prefix(prefix: SymT, idx: int) -> sympy.Symbol:
    """
    Used to generate an integer-nonnegative symbol.
    """
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    if prefix == SymT.SIZE:
        raise AssertionError(f"prefix must not be SymT.SIZE, got {prefix}")
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return make_symbol(prefix, idx, integer=True, nonnegative=True)


def generate_assert(check: bool) -> bool:
    return (check or config.debug_index_asserts) and config.assert_indirect_indexing


def sympy_index_symbol(name: str) -> sympy.Symbol:
    """
    Used to generate an integer-nonnegative symbol.
    """
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    if name[0] == "s":
        raise AssertionError(f"Symbol name must not start with 's', got {name!r}")
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return sympy.Symbol(name, integer=True, nonnegative=True)


def sympy_subs(expr: sympy.Expr, replacements: dict[sympy.Expr, Any]) -> sympy.Expr:
    """
    When the passed replacement symbol v is a string, it is converted to a symbol with name v that
    have the same replaced expression integer and nonnegative properties.
    """
    return _sympy_subs(expr, replacements)


def is_symbolic(a: Any) -> TypeGuard[torch.SymInt | torch.Tensor]:
    return isinstance(a, torch.SymInt) or (
        isinstance(a, torch.Tensor) and a._has_symbolic_sizes_strides
    )


def any_is_symbolic(*args: Any) -> bool:
    return any(is_symbolic(a) for a in args)


# Ops that are fundamentally incompatible with CUDA graph capture
# (e.g., CPU synchronization, dynamic memory allocation, etc.)
FORBIDDEN_CUDAGRAPH_OPS = frozenset(
    [
        "aten._fused_moving_avg_obs_fq_helper.default",
        "aten._fused_moving_avg_obs_fq_helper_functional.default",
        "fbgemm.dense_to_jagged.default",
        "fbgemm.jagged_to_padded_dense.default",
        "run_and_save_rng_state",
        "run_with_rng_state",
        "aten._local_scalar_dense",
        # cuSOLVER-backed linalg.eigh is not CUDA graph capturable.
        "aten._linalg_eigh.default",
        # Technically, it's not necessary to ban this, because an
        # assert_scalar with constant arguments can be validly run
        # with CUDA graphs, but the operator is also pointless with
        # constant arguments, so might as well ban
        "aten._assert_scalar",
    ]
    + (
        # rocm workaround: topk kernels fault under cudagraph trees
        # on the 2nd replay of a recorded graph.
        [
            "aten.topk.default",
        ]
        if torch.version.hip is not None
        else []
    )
)


def get_first_incompatible_cudagraph_node(
    gm: torch.fx.GraphModule,
) -> torch.fx.Node | None:
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    for node in gm.graph.nodes:
        if is_cudagraph_unsafe_fx_node(node):
            return node

        if (val := node.meta.get("val")) is not None and free_unbacked_symbols(val):
            return node

    return None


def output_node(gm: torch.fx.GraphModule) -> Node:
    """Get the output node from an FX graph"""
    last_node = next(iter(reversed(gm.graph.nodes)))
    if last_node.op != "output":
        raise AssertionError(
            f"Expected last node op to be 'output', got {last_node.op!r}"
        )
    return last_node


def get_all_devices(gm: torch.fx.GraphModule) -> OrderedSet[torch.device]:
    placeholder_nodes = gm.graph.find_nodes(op="placeholder")
    input_devices: OrderedSet[torch.device] = OrderedSet(
        node.meta["val"].device
        for node in placeholder_nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    )

    out_arg = output_node(gm).args[0]  # type: ignore[union-attr]
    out_args = out_arg if isinstance(out_arg, tuple) else (out_arg,)
    out_devices: OrderedSet[torch.device] = OrderedSet(
        arg.meta["val"].device
        for arg in out_args
        if isinstance(arg, torch.fx.Node)
        and isinstance(arg.meta.get("val"), torch.Tensor)
    )
    return input_devices | out_devices


import gc


def unload_xpu_triton_pyds() -> None:
    # unload __triton_launcher.pyd
    for module_name in list(sys.modules.keys()):
        if not module_name.startswith("torch._inductor.runtime.compile_tasks."):
            continue
        m = sys.modules[module_name]
        for attr_name in m.__dict__:
            if attr_name.startswith("triton_"):
                kernel = getattr(m, attr_name)
                if isinstance(
                    kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner
                ):
                    for result in kernel.compile_results:
                        if isinstance(
                            result,
                            torch._inductor.runtime.triton_heuristics.TritonCompileResult,
                        ):
                            # pyrefly: ignore [missing-attribute]
                            run = result.kernel.run
                            if hasattr(run, "mod"):
                                run.mod.__del__()
        del sys.modules[module_name]

    # unload spirv_utils.pyd
    if "triton.runtime.driver" in sys.modules:
        driver_mod = sys.modules["triton.runtime.driver"]
        if hasattr(driver_mod.driver.active, "utils"):
            utils_cls = type(driver_mod.driver.active.utils)
            if hasattr(utils_cls, "instance"):
                del utils_cls.instance
            elif hasattr(utils_cls, "_instance"):
                utils_cls._instance = None
            del driver_mod.driver.active.utils

    gc.collect()


_registered_caches: list[Any] = []


def clear_on_fresh_cache(obj: Any) -> Any:
    """
    Use this decorator to register any caches that should be cache_clear'd
    with fresh_cache().
    """
    if not hasattr(obj, "cache_clear") or not callable(obj.cache_clear):
        raise AttributeError(f"{obj} does not have a cache_clear method")

    _registered_caches.append(obj)
    return obj


def clear_caches() -> None:
    """
    Clear all registered caches.
    """
    for obj in _registered_caches:
        obj.cache_clear()


@contextlib.contextmanager
def _set_env(key: str, value: str) -> Iterator[None]:
    """Thread-safe env var set/restore using atomic C-level lookups.

    We avoid mock.patch.dict(os.environ, ...) because it internally calls
    os.environ.copy(), which iterates all env var keys then fetches values in
    separate steps. That approach is not atomic and can race with background threads
    (e.g. Triton async compilation) modifying the environment, causing KeyError,
    so we use os.environ.get() for individual keys which is an atomic C-level lookup.
    """
    old = os.environ.get(key)
    try:
        os.environ[key] = value
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@contextlib.contextmanager
def fresh_cache(
    cache_entries: dict[str, Any] | None = None,
    dir: str | None = None,
    delete: bool = True,
) -> Iterator[None]:
    """
    Contextmanager that provides a clean tmp cachedir for pt2 caches.

    Optionally, pass a dict as 'cache_entries' to get a list of filenames and sizes
    generated with this cache instance.
    """
    clear_caches()

    from torch._inductor.cpp_builder import normalize_path_separator

    inductor_cache_dir = normalize_path_separator(tempfile.mkdtemp(dir=dir))
    try:
        with _set_env("TORCHINDUCTOR_CACHE_DIR", inductor_cache_dir):
            log.debug("Using inductor cache dir %s", inductor_cache_dir)
            triton_cache_dir = normalize_path_separator(
                os.path.join(inductor_cache_dir, "triton")
            )
            with _set_env("TRITON_CACHE_DIR", triton_cache_dir):
                yield
                if isinstance(cache_entries, dict):
                    if len(cache_entries) != 0:
                        raise AssertionError("expected empty cache_entries dict")
                    if os.path.exists(triton_cache_dir):
                        files = os.listdir(triton_cache_dir)
                        cache_entries.update(
                            {
                                f: os.path.getsize(os.path.join(triton_cache_dir, f))
                                for f in files
                                if ".lock" not in f
                            }
                        )
        if delete:
            if is_windows() and torch.xpu.is_available():
                unload_xpu_triton_pyds()

            shutil.rmtree(
                inductor_cache_dir,
                # Let's not fail if we can't clean up the temp dir. Also note that for
                # Windows, we can't delete the loaded modules because the module binaries
                # are open.
                ignore_errors=is_windows(),
                onerror=lambda func, path, exc_info: log.warning(
                    "Failed to remove temporary cache dir at %s",
                    inductor_cache_dir,
                    exc_info=exc_info,
                ),
            )
    except Exception:
        log.warning("on error, temporary cache dir kept at %s", inductor_cache_dir)
        raise
    finally:
        clear_caches()


# Deprecated functions -- only keeping them for BC reasons
clear_on_fresh_inductor_cache = clear_on_fresh_cache
clear_inductor_caches = clear_caches
fresh_inductor_cache = fresh_cache


def argsort(seq: Sequence[Any], *, reverse: bool = False) -> list[int]:
    getter = seq.__getitem__
    a_r = range(len(seq))
    # preserve original order for equal strides
    # e.g. if strides are [32, 8, 8, 1]
    # argsort -> [3, 2, 1, 0], rather than
    # [3, 1, 2, 0]
    # i.e. for equal strides in ascending order (reverse=False) an
    # inner dimension should come before an outer dimension, and vice versa
    # for descending
    sort_idx = list(sorted(a_r, key=getter, reverse=True))  # noqa: C413
    if not reverse:
        return list(reversed(sort_idx))
    return sort_idx


def argsort_sym(
    shape_env: ShapeEnv,
    seq: Sequence[int | torch.SymInt | sympy.Expr],
    *,
    reverse: bool = False,
) -> list[int]:
    """
    Return a symbolic sort order for optimization-only layout heuristics.

    Data-dependent comparisons can fall back to non-guarding optimization hints,
    so callers must not use this order to make correctness decisions.
    """

    def cmp(a: tuple[int, sympy.Expr], b: tuple[int, sympy.Expr]) -> int:
        a_idx, a_val = a
        b_idx, b_val = b

        def evaluate(
            expr: bool | torch.SymInt | sympy.Expr,
            fallback: Callable[[], bool],
        ) -> bool:
            if isinstance(expr, bool):
                return expr
            try:
                return shape_env.evaluate_expr(expr)
            except GuardOnDataDependentSymNode:
                return fallback()

        def hint_lt(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
            # Stride sorting is an optimization-only ordering decision.  Keep
            # semantic reasoning symbolic first, then fall back to explicit
            # optimization hints when unbacked comparisons are data-dependent.
            return shape_env.optimization_hint(lhs) < shape_env.optimization_hint(rhs)

        if evaluate(a_val < b_val, lambda: hint_lt(a_val, b_val)):
            return -1
        if evaluate(a_val > b_val, lambda: hint_lt(b_val, a_val)):
            return 1
        # If strides are the same, prefer the original order.
        # (this matches argsort's algorithm).
        # For strides = [2048, 2048, 16, 1], this is
        # [3, 2, 1, 0].
        if a_idx < b_idx:
            return 1
        if a_idx > b_idx:
            return -1
        return 0

    # Strategy: convert all symints to sympy.Expr, then use a custom comparator
    exprs = [
        (idx, s.node.expr if isinstance(s, torch.SymInt) else s)
        for idx, s in enumerate(seq)
    ]
    exprs = sorted(exprs, key=functools.cmp_to_key(cmp), reverse=reverse)
    result = [idx for idx, _ in exprs]
    return result


@functools.lru_cache(8)
def get_dtype_size(dtype: torch.dtype) -> int:
    # TODO: Investigate why uint64 tensor creation causes overflow error:
    # Workaround for RuntimeError in memory size calculation, but underlying cause unclear
    if dtype == torch.uint64:
        return 8
    return torch.empty((), dtype=dtype).element_size()


class DualIndentedBuffer(IndentedBuffer):
    """IndentedBuffer that simultaneously accumulates JIT and AOTI output.

    The base class (self._lines) holds JIT content. self.aot holds AOTI content.
    By default, writeline/splice write to both. Use _jit/_aot variants for
    mode-specific writes.
    """

    def __init__(self, initial_indent: int = 0) -> None:
        super().__init__(initial_indent)
        self.aot = IndentedBuffer(initial_indent)

    @property
    def jit(self) -> IndentedBuffer:
        """Read-only accessor for JIT buffer (for splicing into result).

        WARNING: Do NOT use jit for writing — writes would go to both buffers
        because self.writeline is overridden. Use writeline_jit/splice_jit instead.
        """
        return self  # type: ignore[return-value]

    def writeline(self, line):  # type: ignore[override]
        IndentedBuffer.writeline(self, line)
        self.aot.writeline(line)

    def writeline_jit(self, line) -> None:
        IndentedBuffer.writeline(self, line)

    def writeline_aot(self, line) -> None:
        self.aot.writeline(line)

    def writelines(self, lines):  # type: ignore[override]
        for line in lines:
            self.writeline(line)

    def splice(self, other_code, strip: bool = False) -> None:  # type: ignore[override]
        # Splice into each buffer independently. When other_code is itself a
        # DualIndentedBuffer, each side reads from the matching side.
        IndentedBuffer.splice(self, other_code, strip=strip)
        aot_other = (
            other_code.aot if isinstance(other_code, DualIndentedBuffer) else other_code
        )
        self.aot.splice(aot_other, strip=strip)

    def splice_jit(self, other_code, strip: bool = False) -> None:
        IndentedBuffer.splice(self, other_code, strip=strip)

    def splice_aot(self, other_code, strip: bool = False) -> None:
        aot_other = (
            other_code.aot if isinstance(other_code, DualIndentedBuffer) else other_code
        )
        self.aot.splice(aot_other, strip=strip)

    def indent(self, offset: int = 1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            self.aot._indent += offset
            try:
                yield
            finally:
                self._indent -= offset
                self.aot._indent -= offset

        return ctx()

    def do_indent(self, offset: int = 1) -> None:
        self._indent += offset
        self.aot._indent += offset

    def do_unindent(self, offset: int = 1) -> None:
        self._indent -= offset
        self.aot._indent -= offset

    def clear(self) -> None:
        super().clear()
        self.aot.clear()


class AotOnlyBuffer(IndentedBuffer):
    """IndentedBuffer for pure-AOTI codegen.

    Mirror of the base class's pure-JIT defaults: writeline_aot/splice_aot
    write to the buffer; writeline_jit/splice_jit are no-ops. Lets call
    sites use writeline_jit/writeline_aot uniformly across pure-JIT,
    pure-AOTI, and dual-wrapper modes.
    """

    def writeline_jit(self, line) -> None:
        pass

    def writeline_aot(self, line) -> None:
        self.writeline(line)

    def splice_jit(self, other_code, strip: bool = False) -> None:
        pass

    def splice_aot(self, other_code, strip: bool = False) -> None:
        self.splice(other_code, strip=strip)


def make_codegen_buffer() -> IndentedBuffer:
    """Construct the IndentedBuffer subclass matching the current codegen mode.

    Dual-wrapper mode -> DualIndentedBuffer (JIT and AOTI both active).
    Pure AOTI -> AotOnlyBuffer (writeline_aot writes; writeline_jit drops).
    Pure JIT  -> IndentedBuffer  (writeline_jit writes; writeline_aot drops).
    """
    from .virtualized import V

    if V.graph.is_dual_wrapper_mode:
        return DualIndentedBuffer()
    if V.graph.aot_mode:
        return AotOnlyBuffer()
    return IndentedBuffer()


class FakeIndentedBuffer(IndentedBuffer):
    def __init__(self) -> None:
        super().__init__()

    def __getattribute__(self, name: str) -> Any:
        if name == "__class__":  # Allow access to the class attribute
            return object.__getattribute__(self, name)
        raise RuntimeError(
            f"Tried to call self.{name} on FakeIndentedBuffer. This buffer"
            "is currently used on TritonTemplateKernel to prevent actual"
            "writes to the body without explicitly specifying the body with"
            "`TritonTemplateKernel.set_subgraph_body(name)`"
        )


@contextlib.contextmanager
def restore_stdout_stderr() -> Iterator[None]:
    initial_stdout, initial_stderr = sys.stdout, sys.stderr
    try:
        yield
    finally:
        sys.stdout, sys.stderr = initial_stdout, initial_stderr


class DelayReplaceLine(DeferredLineBase):
    """At end of codegen call `line.replace(key, value_fn())`"""

    def __init__(self, key: str, value_fn: Callable[[], str], line: str):
        super().__init__(line)
        self.key = key
        self.value_fn = value_fn

    def __call__(self) -> str:
        return self.line.replace(self.key, self.value_fn())

    def _new_line(self, line: str) -> DelayReplaceLine:
        return DelayReplaceLine(self.key, self.value_fn, line)


@functools.cache
def is_big_gpu(index_or_device: int | torch.device = 0) -> bool:
    if isinstance(index_or_device, torch.device):
        device = index_or_device
    else:
        device = torch.device(get_gpu_type(), index_or_device)

    prop = DeviceProperties.create(device)

    # SM logic is not relevant to ROCm gpus
    # Arbitrarily skipping the older models
    if torch.version.hip:
        if prop.major is None:
            raise AssertionError("DeviceProperties.major is None for ROCm GPU")
        if prop.major < 9 or prop.major == 10:
            log.warning("GPU arch does not support max_autotune_gemm mode usage")
            return False
        return True

    min_sms = 16 if device.type == "xpu" else 68  # 3080
    avail_sms = prop.multi_processor_count
    if avail_sms < min_sms:
        log.warning(
            "Not enough SMs to use max_autotune_gemm mode",
            extra={"min_sms": min_sms, "avail_sms": avail_sms},
        )
        return False
    return True


@functools.lru_cache
def get_max_num_sms() -> int:
    if torch.xpu.is_available():
        return torch.xpu.get_device_properties().gpu_subslice_count
    return torch.cuda.get_device_properties("cuda").multi_processor_count


@functools.lru_cache
def using_b200() -> bool:
    """Returns true if the device is a NVIDIA B200, otherwise returns false."""
    if not torch.cuda.is_available():
        return False
    # compute capability 10.0 or 10.0a is NVIDIA B200
    device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return device_properties.major == 10


def is_nvidia_sm100_or_later() -> bool:
    """Return true for NVIDIA CUDA devices with compute capability SM100+."""
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() >= (10, 0)
    )


def get_num_sms() -> int:
    """Handle experimental carveout if set otherwise return hardware SM count"""
    # TODO we need to properly guard on this global
    if torch.xpu.is_available():
        return get_max_num_sms()
    carveout = torch._C._get_sm_carveout_experimental()
    return get_max_num_sms() - (carveout if carveout is not None else 0)


def get_tma_workspace_arg(
    num_tma_descriptors: int,
    device: torch.device,
    num_programs: int | None = None,
) -> WorkspaceArg:
    """Builds and returns a WorkspaceArg for the device side TMA workspace buffer."""
    from .codegen.common import WorkspaceArg, WorkspaceZeroMode

    if num_programs is None:
        num_programs = get_num_sms()
    zero_mode = WorkspaceZeroMode.from_bool(False)
    size = num_programs * num_tma_descriptors * TMA_DESCRIPTOR_SIZE
    return WorkspaceArg(
        count=size,
        zero_mode=zero_mode,
        device=device,
        outer_name=WorkspaceArg.unique_name(),
    )


def get_default_kpack(block_k: int = 16) -> int:
    if not torch.version.hip:
        return 0
    if "gfx942" in torch.cuda.get_device_properties(0).gcnArchName and block_k <= 16:
        return 1
    return 2


def _use_template_for_gpu(
    layout: Layout, allowed_layout_dtypes: list[torch.dtype]
) -> bool:
    if layout.dtype not in allowed_layout_dtypes:
        log.debug(
            "Not using template since dtype %s is not in allowed layout dtypes %s",
            layout.dtype,
            allowed_layout_dtypes,
        )
    return (
        is_gpu(layout.device.type)
        and layout.dtype in allowed_layout_dtypes
        and is_big_gpu(layout.device)
    )


def _use_autotune_backend(backend: str) -> bool:
    return backend.upper() in [
        x.strip() for x in config.max_autotune_gemm_backends.upper().split(",")
    ]


def _use_conv_autotune_backend(backend: str) -> bool:
    return backend.upper() in [
        x.strip() for x in config.max_autotune_conv_backends.upper().split(",")
    ]


def _use_conv_bwd_weight_autotune_backend(backend: str) -> bool:
    return backend.upper() in [
        x.strip()
        for x in config.max_autotune_conv_bwd_weight_backends.upper().split(",")
    ]


def _use_conv_bwd_input_autotune_backend(backend: str) -> bool:
    return backend.upper() in [
        x.strip()
        for x in config.max_autotune_conv_bwd_input_backends.upper().split(",")
    ]


def use_triton_template(
    layout: Layout,
    *,
    enable_int32: bool = False,
    enable_float8: bool = False,
    check_max_autotune: bool = True,
) -> bool:
    from .codegen.common import BackendFeature, has_backend_feature

    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if enable_int32:
        layout_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32]
    if enable_float8:
        layout_dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])
    return (
        (
            (
                is_gpu(layout.device.type)
                and _use_template_for_gpu(layout, layout_dtypes)
            )
            or (layout.device.type == "cpu" and layout.dtype in layout_dtypes)
        )
        # some callers handle max-autotune checking externally
        and (config.max_autotune or config.max_autotune_gemm or not check_max_autotune)
        and _use_autotune_backend("TRITON")
        and has_backend_feature(layout.device, BackendFeature.TRITON_TEMPLATES)
    )


def can_use_tma(
    *matrices: IRNode, output_layout: Layout | None = None, add_guards: bool = False
) -> bool:
    """
    Return True iff *all* supplied tensors satisfy the CUDA TMA constraints
    that Triton relies on today.
    * https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

    A tensor is accepted when:
      * 1 ≤ rank ≤ 5 (cuTensorMapEncodeTiled)
      * dtype in _TMA_SUPPORTED_DTYPES (CUtensorMapDataType enum)
      * Base pointer 16-byte aligned
      * Exactly one contiguous ("inner") dim with stride 1
      * All "outer" dims have 16-byte aligned strides
      * Inner dim size × itemsize is a multiple of 16
      * For 1-byte dtypes (e.g. FP8), inner dim ≥ 32
    """
    from torch.utils._triton import has_triton_tma_device

    from .virtualized import V

    def _aligned(expr_bytes: int | sympy.Expr) -> bool:
        return V.graph.sizevars.statically_known_multiple_of(expr_bytes, TMA_ALIGNMENT)

    def _is_tma_compatible_layout(layout: Layout | None) -> bool:
        if layout is None:
            return True
        sizes = layout.size
        strides = layout.stride
        dtype = layout.dtype

        # Verify the output is 16-byte aligned
        if not _aligned(layout.offset):
            return False

        return _is_tma_compatible(sizes, strides, dtype)

    def _is_tma_compatible_matrix(m: IRNode) -> bool:
        sizes = m.get_size()
        strides = m.get_stride()
        dtype = m.get_dtype()

        # Base pointer 16-byte aligned
        if m.get_name() in V.graph.unaligned_buffers:
            return False

        if (m_device := m.get_device()) is not None and m_device.type == "xpu":
            return _is_tma_compatible_xpu(sizes, strides, dtype)

        return _is_tma_compatible(sizes, strides, dtype)

    def _is_tma_compatible(
        sizes: Sequence[sympy.Expr],
        strides: Sequence[_IntLike],
        dtype: torch.dtype,
    ) -> bool:
        rank = len(sizes)
        itemsize = dtype.itemsize

        if rank < 1 or rank > 5:
            return False

        if dtype not in _TMA_SUPPORTED_DTYPES:
            return False

        if add_guards:
            sizes_i = V.graph.sizevars.guard_int_seq(sizes)
            strides_i = V.graph.sizevars.guard_int_seq(strides)
        else:
            sizes_i = [
                V.graph.sizevars.replace_backed_symbols_with_hints(s) for s in sizes
            ]
            strides_i = [
                V.graph.sizevars.replace_backed_symbols_with_hints(st) for st in strides
            ]

        # Find the single contiguous ("inner") dim
        inner = [
            i
            for i, st in enumerate(strides_i)
            if V.graph.sizevars.statically_known_equals(st, 1)
        ]
        if len(inner) != 1:
            return False
        inner_idx = inner[0]

        # All "outer" dims must have 16-byte aligned strides
        for i, st in enumerate(strides_i):
            if i == inner_idx:
                continue
            if not _aligned(st * itemsize):
                return False

        # Inner dim byte width must be a multiple of 16 B
        inner_dim = sizes_i[inner_idx]
        if not _aligned(inner_dim * itemsize):
            return False

        # 1-byte dtypes (FP8 etc.) need inner dim ≥ 32 for tensor core alignment
        if itemsize == 1 and not V.graph.sizevars.statically_known_geq(inner_dim, 32):
            return False

        return True

    def _is_tma_compatible_xpu(
        sizes: Sequence[sympy.Expr],
        strides: Sequence[_IntLike],
        dtype: torch.dtype,
    ) -> bool:
        # Make sure the last dimension is contiguous
        last_stride = strides[-1]
        last_stride_hint = V.graph.sizevars.replace_backed_symbols_with_hints(
            last_stride
        )
        if not V.graph.sizevars.statically_known_equals(last_stride_hint, 1):
            return False

        # Triton's type of index is uint32, so all dimensions must fit in uint32
        MAX_UINT32 = 2**32 - 1
        for size in sizes:
            size_hint = V.graph.sizevars.replace_backed_symbols_with_hints(size)
            if V.graph.sizevars.statically_known_gt(size_hint, MAX_UINT32):
                return False

        return True

    return (
        has_triton_tma_device()
        and all(_is_tma_compatible_matrix(m) for m in matrices)
        and _is_tma_compatible_layout(output_layout)
    )


def _descriptor_shape_fits_in_int32(
    sizes: Sequence[sympy.Expr], add_guards: bool = False
) -> bool:
    int32_max = torch.iinfo(torch.int32).max
    conditions = []
    for size in sizes:
        if isinstance(size, (int, sympy.Integer)):
            if size > int32_max:
                return False
        else:
            conditions.append(sympy.Le(size, int32_max))

    if not conditions:
        return True

    from .virtualized import V

    condition = conditions[0] if len(conditions) == 1 else sympy.And(*conditions)
    return (
        V.graph.sizevars.guard_or_false(condition)
        if add_guards
        else V.graph.sizevars.statically_known_true(condition)
    )


def use_triton_tma_template(
    *matrices: IRNode, output_layout: Layout, add_guards: bool = False
) -> bool:
    if not config.triton.enable_persistent_tma_matmul:
        return False
    if not all(len(m.get_size()) == 2 for m in matrices):
        return False
    if not all(
        _descriptor_shape_fits_in_int32(m.get_size(), add_guards=add_guards)
        for m in matrices
    ):
        return False
    if config.triton.enable_template_tma_store and not _descriptor_shape_fits_in_int32(
        output_layout.size, add_guards=add_guards
    ):
        return False
    # On AMD (HIP), TMA is not available but we still use non-TMA persistent
    # kernels, so skip the TMA compatibility checks.
    if torch.version.hip is not None:
        return True
    layout = output_layout if config.triton.enable_template_tma_store else None
    return can_use_tma(*matrices, output_layout=layout, add_guards=add_guards)


def use_triton_blackwell_tma_template(
    *matrices: IRNode, output_layout: Layout, add_guards: bool = False
) -> bool:
    if not use_triton_tma_template(
        *matrices, output_layout=output_layout, add_guards=add_guards
    ):
        return False

    from torch.utils._triton import has_triton_tensor_descriptor_host_tma

    from .codegen.cuda.cuda_env import is_datacenter_blackwell_arch

    # Blackwell template require the tensor descriptor API, not the experimental API.
    return has_triton_tensor_descriptor_host_tma() and is_datacenter_blackwell_arch()


def use_triton_scaling_template(
    scale_option_a: ScalingType,
    scale_option_b: ScalingType,
    scaling_types: list[ScalingType],
) -> bool:
    return scale_option_a in scaling_types and scale_option_b in scaling_types


@functools.lru_cache(maxsize=1)
def ensure_cute_available() -> bool:
    """Check if CuTeDSL is importable; cache the result for reuse.

    Call ensure_cute_available.cache_clear() after installing CuTeDSL
    in the same interpreter to retry the import.
    """
    try:
        return importlib.util.find_spec("cutlass") is not None
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def ensure_nv_universal_gemm_available() -> bool:
    """Check if NVIDIA Universal GEMM (cutlass.operators) is importable; cache the result for reuse.

    Call ensure_nv_universal_gemm_available.cache_clear() after installing
    cutlass.operators in the same interpreter to retry the import.
    """
    try:
        available = importlib.util.find_spec("cutlass.operators") is not None
    except (ImportError, ValueError):
        return False
    if available:
        _ensure_fp4_dtype_registered()
    return available


def _ensure_fp4_dtype_registered():
    """Ensure cutlass.operators maps torch.float4_e2m1fn_x2 -> cutlass.Float4E2M1FN.

    cutlass.operators natively supports this dtype, so this is normally a no-op.
    We keep the patch as a safety net (and for backward compat with generated
    code that calls this) in case a future version regresses.
    """
    import cutlass.operators.utils.dtype as _dtype_utils

    try:
        _dtype_utils.cutlass_type_from_torch_type(torch.float4_e2m1fn_x2)
    except (KeyError, AttributeError):
        import cutlass

        _orig = _dtype_utils.cutlass_type_from_torch_type

        def _patched(dtype):
            if dtype == torch.float4_e2m1fn_x2:
                return cutlass.Float4E2M1FN
            return _orig(dtype)

        _dtype_utils.cutlass_type_from_torch_type = _patched


@functools.lru_cache(maxsize=1)
def ensure_nvmatmul_heuristics_available() -> bool:
    """Check if nvMatmulHeuristics is importable; cache the result for reuse.

    nvMatmulHeuristics provides performance model-based kernel selection
    for NVIDIA GEMM operations.

    Call ensure_nvmatmul_heuristics_available.cache_clear() after installing
    nvMatmulHeuristics in the same interpreter to retry the import.
    """
    try:
        return importlib.util.find_spec("nvMatmulHeuristics") is not None
    except ImportError:
        return False


def use_blackwell_cutedsl_grouped_mm(
    mat_a: Any,
    mat_b: Any,
    layout: Layout,
    a_is_2d: bool,
    b_is_2d: bool,
    offs: Any | None,
    bias: Any | None,
    scale_result: Any | None,
) -> bool:
    """
    Returns True if we can use the blackwell kernel for grouped mm.
    Required conditions:
        1. CuTeDSL backend is enabled
        2. CuTeDSL is available
        3. We are on a blackwell arch
        4. The dtype is bf16
        5. Max autotune or max autotune gemm is enabled
        6. A, B, and the output are 16B aligned
        7. We are not using dynamic shapes
        8. A is 2d
        9. B is 3d
        10. Offsets are provided
        11. Bias and Scale are not provided
    """
    if not ensure_cute_available():
        return False

    if not _use_autotune_backend("CUTEDSL"):
        return False

    from .codegen.cuda.cuda_env import is_datacenter_blackwell_arch

    if not is_gpu(layout.device.type):
        return False

    if not is_datacenter_blackwell_arch():
        return False

    layout_dtypes = [torch.bfloat16]
    if not _use_template_for_gpu(layout, layout_dtypes):
        return False

    if not (config.max_autotune or config.max_autotune_gemm):
        return False

    # Checks for 16B ptr and stride alignment
    if not can_use_tma(mat_a, mat_b, output_layout=layout):
        return False

    if any(is_dynamic(x) for x in [mat_a, mat_b]):
        return False

    if not a_is_2d or b_is_2d:
        return False

    if offs is None:
        return False

    if bias is not None or scale_result is not None:
        return False

    return True


def use_cutlass_template(layout: Layout, m: int, n: int, k: int) -> bool:
    from .virtualized import V

    gemm_size = V.graph.sizevars.optimization_hint(m * n * k, fallback=-1)
    if gemm_size <= 0 or gemm_size < config.cutlass.cutlass_backend_min_gemm_size:
        return False
    from .codegen.cutlass.utils import try_import_cutlass

    # Do not use cutlass template on ROCm
    if torch.version.hip:
        return False

    # output dtype
    # FP32 not supported: https://github.com/pytorch/pytorch/issues/145952
    layout_dtypes = [torch.float16, torch.bfloat16, torch.int32]
    res = (
        _use_template_for_gpu(layout, layout_dtypes)
        and (config.max_autotune or config.max_autotune_gemm)
        and _use_autotune_backend("CUTLASS")
    )

    if res:
        if not try_import_cutlass():
            log.warning(
                "Failed to import CUTLASS lib. Please check whether "
                "_inductor.config.cutlass.cutlass_dir %s is set correctly. "
                "Skipping CUTLASS backend for now.",
                config.cutlass.cutlass_dir,
            )
            return False
    return res


def use_nv_universal_gemm_template(
    layout: Layout,
    m: _IntLike,
    n: _IntLike,
    k: _IntLike,
    mat_a: IRNode,
    mat_b: IRNode,
    offs: IRNode | None = None,
    g: _IntLike | None = None,
) -> bool:
    """
    Return True if we can use the NVIDIA Universal GEMM Template.

    Required conditions:
        1. NVGEMM backend is enabled
        2. cutlass.operators is available
        3. We are on a NVIDIA GPU
        4. Max autotune or max autotune gemm is enabled
        5. Not in AOT Inductor mode (requires runtime JIT compilation)
        6. Base pointers are 16-byte aligned
        7. Shape dimensions are not unbacked symbols

    Note:
        - Shape and stride constraints are handled internally by
          cutlass.operators.get_operators() which filters incompatible kernels.
        - GroupedGemm currently only supports TN layout (column-major B).
          Any other layout will act as a noop and fall back to ATen.
        - Dynamic shapes are supported as long as they have hints
          (from example inputs).
    """
    from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

    if not ensure_cute_available():
        return False

    if not ensure_nv_universal_gemm_available():
        return False

    if not _use_autotune_backend("NVGEMM"):
        return False

    from .virtualized import V

    if V.aot_compilation:
        return False

    if layout.device.type != "cuda" or torch.version.hip:
        return False

    if not (config.max_autotune or config.max_autotune_gemm):
        return False

    # cutlass.operators can't handle unbacked symbols because it needs to evaluate
    # shape constraints (e.g., stride divisibility by 8, N/K divisibility by 16).
    # Unbacked symbols have no hint values, causing GuardOnDataDependentSymNode errors.
    dims_to_check = [m, n, k]
    if g is not None:
        dims_to_check.append(g)
    if any(has_free_unbacked_symbols(dim) for dim in dims_to_check):
        return False

    # Base pointer must be 16-byte aligned. cutlass.operators can't check this at
    # compile time because it only sees FakeTensors without real data pointers.
    tensors_to_check = [mat_a, mat_b]
    if offs is not None:
        tensors_to_check.append(offs)
    if any(t.get_name() in V.graph.unaligned_buffers for t in tensors_to_check):
        return False

    return True


def _use_cutlass_for_op(op_name: str) -> bool:
    """Check if CUTLASS should be used for the given operation."""
    enabled_ops = config.cutlass.cutlass_enabled_ops.upper()
    if enabled_ops == "ALL":
        return True
    return op_name.upper() in [x.strip() for x in enabled_ops.split(",")]


_IntLike: TypeAlias = int | sympy.Expr


@functools.cache
def use_decompose_k_choice(
    m: _IntLike, n: _IntLike, k: _IntLike, threshold_multiple: int = 1
) -> bool:
    from torch._inductor.virtualized import V

    decompose_k_threshold = config.triton.decompose_k_threshold * threshold_multiple

    return (
        V.graph.sizevars.statically_known_true(
            sympy.And(
                sympy.Ge(k, decompose_k_threshold * m),
                sympy.Ge(k, decompose_k_threshold * n),
            )
        )
        and not V.graph.aot_mode  # TODO: Support AOTI for decomposeK
        and not V.graph.cpp_wrapper
        and config.triton.num_decompose_k_splits > 0
    )


@functools.cache
def use_contiguous(m: _IntLike, n: _IntLike, k: _IntLike) -> bool:
    """
    Check if we should use the contiguous subgraph transform.
    This transform makes the second matrix contiguous before the matmul.
    """
    contiguous_threshold = config.rocm.contiguous_threshold

    # Similar conditions to decompose_k but for contiguous transform
    from torch._inductor.virtualized import V

    return (
        bool(torch.version.hip)  # Only relevant on AMD
        and V.graph.sizevars.statically_known_true(
            sympy.And(
                sympy.Ge(k, contiguous_threshold * m),
                sympy.Ge(k, contiguous_threshold * n),
            )
        )
        and not V.graph.aot_mode
        and not V.graph.cpp_wrapper
    )


@functools.cache
def get_k_splits(m: _IntLike, n: _IntLike, k: _IntLike) -> list[int]:
    # To limit compile time
    k_splits_limit = config.triton.num_decompose_k_splits

    # Hand-tuned
    default_k_splits = [16, 32, 64, 128, 256]
    # If k is a sympy expression, we can't do any splitting
    if isinstance(k, sympy.Expr) and not k.is_number:
        return default_k_splits
    elif k_splits_limit == 0:
        return []

    if (isinstance(m, sympy.Expr) and not m.is_number) or (
        isinstance(n, sympy.Expr) and not n.is_number
    ):
        max_k_split = 256
    else:
        max_k_split = min(k // m, k // n)

    min_k_split = 2
    # Get all divisors of k, k has to be divisible by kPart
    divisors = sympy.divisors(k)

    divisors = [
        divisor
        for divisor in divisors
        if divisor <= max_k_split and divisor >= min_k_split
    ]

    pow_of_2_divisors, mul_of_32_divisors, rest_of_splits = [], [], []

    for d in divisors:
        kPart = k // d

        # Smaller than 128 might not even fit in a single tile, BLOCK_K can be 128
        if kPart < 128:
            continue

        # Power of 2 divisors are best performing, conform to hardware
        if (kPart & kPart - 1) == 0 and kPart >= 128:
            pow_of_2_divisors.append(d)
        # Else check if creates a multiple of 32
        elif kPart % 32 == 0:
            mul_of_32_divisors.append(d)
        # otherwise, take the smallest values
        else:
            rest_of_splits.append(d)

    if config.max_autotune_gemm_search_space == "EXHAUSTIVE":
        return pow_of_2_divisors + mul_of_32_divisors + rest_of_splits

    best_splits = pow_of_2_divisors + mul_of_32_divisors + rest_of_splits
    # Otherwise, conform results to k_splits_limit
    return best_splits[:k_splits_limit]


@functools.cache
def _rocm_native_device_arch_name(device: str) -> str:
    return torch.cuda.get_device_properties(device).gcnArchName


@functools.lru_cache
def using_rocm_rdna3() -> bool:
    """Returns true if the device is based on RDNA3, otherwise returns false."""
    return torch.cuda.is_available() and _rocm_native_device_arch_name(
        "cuda"
    ).startswith("gfx11")


@functools.cache
def try_import_ck_lib() -> tuple[
    str | None, Callable[[], list[Any]], Callable[[], list[Any]], type[Any]
]:
    try:
        import ck4inductor  # type: ignore[import]
        from ck4inductor.universal_gemm.gen_instances import (  # type: ignore[import]
            gen_ops_library,
            gen_ops_preselected,
        )
        from ck4inductor.universal_gemm.op import (  # type: ignore[import]
            CKGemmOperation,
        )

        package_dirname = os.path.dirname(ck4inductor.__file__)
    except ImportError:

        def gen_ops_library() -> list[Any]:
            return []

        def gen_ops_preselected() -> list[Any]:
            return []

        class CKGemmOperation:  # type: ignore[no-redef]
            pass

        package_dirname = None
    return package_dirname, gen_ops_library, gen_ops_preselected, CKGemmOperation


def use_ck_template(layout: Layout) -> bool:
    # config knobs check 1
    if not (config.max_autotune or config.max_autotune_gemm):
        return False
    # platform check
    if not torch.version.hip:
        return False
    # tensors must be on GPU
    if layout.device.type != "cuda":
        return False
    # hardware check
    # if config arch list is not specified, get the native arch from the device properties
    native_arch = _rocm_native_device_arch_name(layout.device)
    requested_archs = {k.split(":")[0]: k for k in config.rocm.arch} or {
        native_arch.split(":")[0]: native_arch
    }
    requested_supported_archs = [
        requested_archs[k]
        for k in requested_archs.keys() & config.rocm.ck_supported_arch
    ]
    if not requested_supported_archs:
        return False
    # supported input dtypes
    if layout.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        return False

    ck_package_dirname, _, _, _ = try_import_ck_lib()

    if not ck_package_dirname:
        log.warning("Please pip install Composable Kernel package")
        return False

    config.rocm.ck_dir = ck_package_dirname

    return True


def use_ck_gemm_template(layout: Layout, m: int, n: int, k: int) -> bool:
    from .virtualized import V

    return (
        _use_autotune_backend("CK")
        and use_ck_template(layout)
        and V.graph.sizevars.optimization_hint(m * n * k, fallback=-1) > 0
    )


def use_ck_tile_gemm_template(layout: Layout, m: int, n: int, k: int) -> bool:
    from .virtualized import V

    return (
        _use_autotune_backend("CKTILE")
        and use_ck_template(layout)
        and V.graph.sizevars.optimization_hint(m * n * k, fallback=-1) > 0
    )


def use_ck_conv_template(layout: Layout) -> bool:
    return _use_conv_autotune_backend("CK") and use_ck_template(layout)


def _use_template_for_cpu(layout: Layout) -> bool:
    return (
        config.max_autotune or config.max_autotune_gemm
    ) and layout.device.type == "cpu"


def use_cpp_bmm_template(
    layout: Layout, mat1: ReinterpretView | Buffer, mat2: IRNode
) -> bool:
    from .ir import Layout

    if not isinstance(mat1.layout, Layout):
        raise AssertionError(f"Expected Layout, got {type(mat1.layout)}")

    # In certain scenarios, such as when the first stride is 0, the entire tensor may not be contiguous.
    # But the 2D matrix within each batch can still be contiguous, allowing us to apply max autotune.
    # So here we specifically check for contiguity within the 2D matrix of each batch.
    mat1_size = mat1.layout.size
    mat1_stride = mat1.layout.stride
    mat1_each_batch_is_contiguous = (
        _use_template_for_cpu(layout)
        and mat1.get_dtype() == torch.float32
        and (len(mat1_size) == 3)
        and (len(mat1_stride) == 3)
        and (mat1_stride[1] == mat1_size[2])
        and (mat1_stride[2] == 1)
    )
    return use_cpp_gemm_template(layout, mat1, mat2, require_constant_mat2=False) and (
        mat1.layout.is_contiguous() or mat1_each_batch_is_contiguous
    )


def use_cpp_gemm_template(
    layout: Layout,
    mat1: IRNode,
    mat2: IRNode,
    mat2_transposed: bool = False,
    require_constant_mat2: bool = True,
    is_woq_int4: bool = False,
    q_group_size: int | None = None,
) -> bool:
    from . import ir
    from .codegen.cpp_micro_gemm import create_micro_gemm
    from .codegen.cpp_utils import get_gemm_template_output_and_compute_dtype
    from .kernel.mm_common import mm_args

    if not _use_template_for_cpu(layout) or not _use_autotune_backend("CPP"):
        return False

    if not config.cpp.weight_prepack:
        return False

    int8_gemm = mat1.get_dtype() in [torch.uint8, torch.int8]
    layout_dtypes = [torch.float32, torch.bfloat16, torch.half, torch.uint8, torch.int8]
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1,
        mat2,
        out_dtype=layout.dtype if int8_gemm else None,
        mat2_transposed=mat2_transposed,
        use_4x2_dim=is_woq_int4,
    )

    # TODO(jgong5): support dynamic shapes for n or k
    if has_free_symbols((n, k)):
        return False

    if isinstance(mat2, ir.BaseView):
        mat2 = mat2.unwrap_view()

    output_dtype, _ = get_gemm_template_output_and_compute_dtype(mat1.get_dtype())
    micro_gemm = create_micro_gemm(
        "micro_gemm",
        m,
        n,
        k,
        input_dtype=mat1.get_dtype(),
        input2_dtype=mat2.get_dtype(),
        output_dtype=output_dtype,
        num_threads=parallel_num_threads(),
        use_ref=not is_woq_int4,
        q_group_size=q_group_size,
    )

    def is_last_dim_stride1(x: IRNode) -> bool:
        x.freeze_layout()
        return x.get_stride()[-1] == 1

    return (
        layout.dtype in layout_dtypes
        and micro_gemm is not None
        and is_last_dim_stride1(mat1)  # TODO(jgong5): support transposed input
        and isinstance(mat2, ir.StorageBox)
        and (mat2.is_module_buffer() or not require_constant_mat2)
    )


def use_aten_gemm_kernels() -> bool:
    return not (
        config.max_autotune or config.max_autotune_gemm
    ) or _use_autotune_backend("ATEN")


class DebugDirManager:
    counter = itertools.count(0)
    prev_debug_name: str

    def __init__(self) -> None:
        self.id = next(DebugDirManager.counter)

    def __enter__(self) -> None:
        self.prev_debug_name = torch._dynamo.config.debug_dir_root
        self.new_name = f"{self.prev_debug_name}_tmp_{self.id}"
        torch._dynamo.config.debug_dir_root = self.new_name

    def __exit__(self, *args: object) -> None:
        shutil.rmtree(self.new_name)
        torch._dynamo.config.debug_dir_root = self.prev_debug_name


def run_and_get_code(
    fn: Callable[P, _T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[_T, list[str]]:
    from .graph import GraphLowering

    source_codes: OrderedSet[str] = OrderedSet()

    def save_output_code(code: str) -> None:
        source_codes.add(code)

    with mock.patch.object(GraphLowering, "save_output_code", save_output_code):
        torch._dynamo.reset()
        result = fn(*args, **kwargs)
    return result, list(source_codes)


def run_and_get_kernels(
    fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs
) -> tuple[_T, list[str]]:
    remove_quote = kwargs.pop("remove_quote", False)
    # pyrefly: ignore [bad-argument-type]
    result, source_codes = run_and_get_code(fn, *args, **kwargs)
    kernels = []
    for code in source_codes:
        if config.cpp_wrapper and config.triton.autotune_at_compile_time is not True:
            # With lazy Triton kernel compilation, kernel sources are embedded
            # inside C++ R"TRITON(...)TRITON" raw strings.
            kernels.extend(re.findall(r'R"TRITON\((.*?)\)TRITON"', code, re.DOTALL))
        else:
            kernels.extend(re.findall(r"'''.*?'''", code, re.DOTALL))
        if remove_quote:
            kernels = [kernel[3:-3] for kernel in kernels]
    return result, kernels


def run_fw_bw_and_get_code(fn: Callable[..., Any]) -> tuple[Any, list[str]]:
    def run_with_backward() -> Any:
        result = fn()
        result.sum().backward()
        return result

    return run_and_get_code(run_with_backward)


def get_code(fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs) -> list[str]:
    """Get the inductor-generated code, but skip any actual compilation or running."""
    from .graph import GraphLowering

    source_codes: list[str] = []

    def save_output_code(code: str) -> None:
        source_codes.append(code)

    def patched_compile_to_module(self: GraphLowering) -> Any:
        class DummyModule:
            """This is empty to replace the generated triton module"""

            def __init__(self) -> None:
                pass

            def call(self, *args: Any, **kwargs: Any) -> None:
                # Don't do anything when called
                pass

        wrapper_code, kernel_code = (
            self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
        )
        # Skip all the actual compiling.
        save_output_code(wrapper_code.value)
        if kernel_code:
            save_output_code(kernel_code.value)

        return DummyModule()

    with (
        mock.patch.object(
            GraphLowering, "compile_to_module", patched_compile_to_module
        ),
        mock.patch.object(GraphLowering, "save_output_code", save_output_code),
    ):
        torch._dynamo.reset()
        # Note the return here is None
        _ = fn(*args, **kwargs)

    return source_codes


def get_triton_code(fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs) -> str:
    # pyrefly: ignore [bad-argument-type]
    source_codes = get_code(fn, *args, **kwargs)
    # Can have two outputs if backwards was eagerly compiled
    if not (1 <= len(source_codes) <= 2):
        raise AssertionError(
            f"expected one or two code outputs got {len(source_codes)}"
        )
    return source_codes[0]


def run_and_get_triton_code(
    fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs
) -> str:
    # pyrefly: ignore [bad-argument-type]
    _, source_codes = run_and_get_code(fn, *args, **kwargs)
    # Can have two outputs if backwards was eagerly compiled
    if not (1 <= len(source_codes) <= 2):
        raise AssertionError(
            f"expected one or two code outputs got {len(source_codes)}"
        )
    return source_codes[0]


def run_and_get_graph_lowering(
    fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs
) -> tuple[Any, list[GraphLowering]]:
    from torch._inductor.graph import GraphLowering
    from torch._inductor.output_code import CompiledFxGraph

    real_init = CompiledFxGraph.__init__
    graph_lowerings = []

    def fake_init(*args: Any, **kwargs: Any) -> None:
        real_init(*args, **kwargs)
        graph = args[2]
        if not isinstance(graph, GraphLowering):
            raise AssertionError(f"Expected GraphLowering, got {type(graph)}")
        graph_lowerings.append(graph)

    with mock.patch.object(CompiledFxGraph, "__init__", fake_init):
        result = fn(*args, **kwargs)

    return result, graph_lowerings


@contextlib.contextmanager
def override_lowering(
    aten_op: Callable[..., Any], override_fn: Callable[..., Any]
) -> Iterator[None]:
    """
    Override the lowering of aten_op with override_fn.
    The first argument of override_fn is the original lowering fn.
    """
    from torch._inductor import lowering

    orig_fn = lowering.lowerings[aten_op]
    try:
        lowering.lowerings[aten_op] = functools.partial(override_fn, orig_fn)
        yield
    finally:
        lowering.lowerings[aten_op] = orig_fn


def add_scheduler_init_hook(
    pre_fn: Callable[..., Any], post_fn: Callable[..., Any] | None = None
) -> Any:
    """
    Add hook functions to be called at the beginning and end of Scheduler.__init__.
    Used for unit tests.
    """
    from torch._inductor.scheduler import Scheduler

    orig_fn = Scheduler.__init__

    def wrapper(scheduler: Any, nodes: Any) -> Any:
        pre_fn(scheduler, nodes)
        out = orig_fn(scheduler, nodes)
        if post_fn:
            post_fn(scheduler, nodes)
        return out

    return unittest.mock.patch.object(Scheduler, "__init__", wrapper)


def developer_warning(msg: str) -> None:
    """
    Warnings that will be actionable for PyTorch developers, but not
    end users.  Allows us to easily disable them in stable releases but
    keep them on for nightly builds.
    """
    if config.developer_warnings:
        log.warning(msg)
    else:
        log.info(msg)


def get_benchmark_name() -> str | None:
    """
    An experimental API used only when config.benchmark_kernel is true.

    The benchmark name is only available at codegen time. So we can not
    directly call it in benchmark_all_kernels which is run after codegen.

    The function assumes the argument after --only is the benchmark name.
    It works for torchbench.py/hugginface.py/timm_models.py. But for ad-hoc
    scripts, this function may return None.

    There are 2 flavors of --only argument we need to handle:
    1. --only model_name
    2. --only=model_name
    """
    try:
        idx = sys.argv.index("--only")
        if (
            idx + 1 < len(sys.argv)
            and len(sys.argv[idx + 1]) > 0
            and sys.argv[idx + 1][0] != "-"
        ):
            return sys.argv[idx + 1]
    except ValueError:
        pass

    for arg in sys.argv:
        if arg.startswith("--only="):
            return arg[len("--only=") :]

    return None


def is_ones(items: Sequence[Any]) -> bool:
    return all(x == 1 for x in items)


def is_zeros(items: Sequence[Any]) -> bool:
    return all(x == 0 for x in items)


def is_cpu_device(inputs: Sequence[torch.Tensor]) -> bool:
    return all(
        item.device == torch.device("cpu")
        for item in inputs
        if isinstance(item, torch.Tensor)
    )


def get_sympy_Expr_dtype(val: sympy.Expr) -> torch.dtype:
    if not isinstance(val, sympy.Expr):
        raise AssertionError("only support sympy.Expr as input to get_sympy_Expr_dtype")
    if val.is_integer:  # type: ignore[attr-defined]
        return torch.int64
    else:
        return torch.float64


@contextlib.contextmanager
def maybe_profile(should_profile: bool, *args: Any, **kwargs: Any) -> Iterator[Any]:
    if should_profile:
        with torch.profiler.profile(*args, **kwargs) as p:
            yield p
    else:
        yield


def parallel_num_threads() -> int:
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads


@functools.cache
def get_backend_num_stages() -> int:
    from .runtime.triton_helpers import get_backend_options

    options = get_backend_options()
    return options.get("num_stages", 2 if torch.version.hip else 3)


@functools.cache
def get_device_tflops(dtype: torch.dtype) -> float:
    """
    We don't want to throw errors in this function. First check to see if the device is in device_info.py,
    then fall back to the inaccurate triton estimation.
    """
    is_tf32 = torch.backends.cuda.matmul.fp32_precision == "tf32"
    if torch.xpu.is_available():
        is_tf32 = torch.backends.mkldnn.allow_tf32
    ds_tops = datasheet_tops(dtype, is_tf32=is_tf32)
    if ds_tops is not None:
        return ds_tops

    if not torch.cuda.is_available():
        log.warning(
            "get_device_tflops: no Triton fallback available for non-CUDA devices. "
            "Returning 0.0; roofline estimates will use memory bandwidth only."
        )
        return 0.0

    from triton.testing import get_max_simd_tflops, get_max_tensorcore_tflops

    SM80OrLater = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        8,
        0,
    )

    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise AssertionError(
            f"Expected dtype to be float16, bfloat16, or float32, got {dtype}"
        )

    if inspect.signature(get_max_simd_tflops).parameters.get("clock_rate"):
        # Triton API change in https://github.com/triton-lang/triton/pull/2293
        from torch._utils_internal import max_clock_rate

        sm_clock = max_clock_rate()
        if dtype in (torch.float16, torch.bfloat16) and SM80OrLater:
            return get_max_tensorcore_tflops(dtype, sm_clock)

        if torch.backends.cuda.matmul.fp32_precision == "tf32":
            return get_max_tensorcore_tflops(torch.float32, sm_clock)
        else:
            return get_max_simd_tflops(torch.float32, sm_clock)
    else:
        if dtype in (torch.float16, torch.bfloat16) and SM80OrLater:
            return get_max_tensorcore_tflops(dtype)

        if torch.backends.cuda.matmul.fp32_precision == "tf32":
            return get_max_tensorcore_tflops(torch.float32)
        else:
            return get_max_simd_tflops(torch.float32)


@functools.cache
def get_gpu_dram_gbps() -> float:
    """
    We don't want to throw errors in this function. First check to see if the device is in device_info.py,
    then fall back to the inaccurate triton estimation.
    """
    from .analysis.device_info import datasheet_dram_bw_gbs

    ds_bw = datasheet_dram_bw_gbs()
    if ds_bw is not None:
        return ds_bw

    from triton.testing import get_dram_gbps

    return get_dram_gbps()


def get_gpu_shared_memory() -> int:
    from triton.runtime import driver

    return driver.active.utils.get_device_properties(0).get("max_shared_mem", 0)


def get_max_numwarps() -> int:
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        props = DeviceProperties.create(device)
        warp_size = props.warp_size_or_default
        max_threads_per_block = props.max_threads_per_block
        if max_threads_per_block is None:
            raise AssertionError("expected max_threads_per_block to be set")
        return max_threads_per_block // warp_size

    log.debug("CUDA is not available; defaulting max num warps to 32")
    return 32


def is_welford_reduction(reduction_type: str) -> bool:
    return reduction_type.startswith("welford")


def reduction_num_outputs(reduction_type: str) -> int:
    if is_welford_reduction(reduction_type):
        return 3
    elif reduction_type in (
        "argmax_with_value",
        "argmin_with_value",
        "online_softmax_reduce",
    ):
        return 2
    else:
        return 1


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return sys.platform == "win32"


def has_free_symbols(itr: Iterable[Any]) -> bool:
    return any(isinstance(x, sympy.Expr) and not x.is_number for x in itr)


def is_dynamic(*args: Any) -> bool:
    from . import ir

    for t in args:
        if isinstance(
            t, (ir.TensorBox, ir.StorageBox, ir.BaseView, ir.ComputedBuffer, ir.Buffer)
        ):
            if has_free_symbols(t.maybe_get_size() or ()) or has_free_symbols(
                t.maybe_get_stride() or ()
            ):
                return True
        elif not isinstance(t, ir.IRNode):
            continue
        else:
            raise TypeError(f"unexpected type for is_dynamic {type(t)}")

    return False


# Placeholder strings used in triton codegen.
class Placeholder(enum.Enum):
    # The placeholder for the actual name of a triton kernel.
    # e.g. for "def triton_" it would be "triton_"
    KERNEL_NAME = "KERNEL_NAME"

    # The descriptive name of the triton kernel; when unique_kernel_names = False, this
    # placeholder will be replaced with a string with more information.
    DESCRIPTIVE_NAME = "DESCRIPTIVE_NAME"


def pass_execution_and_save(
    func: Callable[..., Any], gm: GraphModule, inp: Sequence[Any], msg: str
) -> None:
    from .pattern_matcher import stable_topological_sort

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
    ) as f:
        before_io = io.StringIO()
        after_io = io.StringIO()
        ShapeProp(gm=gm, fake_mode=detect_fake_mode(inp)).propagate(*inp)
        print(f"Before:\n{gm.graph}", file=f)
        print(gm.graph, file=before_io)
        start_time = datetime.now()
        with GraphTransformObserver(gm, msg):
            func(gm.graph)
        time_elapsed = datetime.now() - start_time
        # recompile graph
        stable_topological_sort(gm.graph)
        gm.graph.lint()
        gm.recompile()

        print(f"After:\n{gm.graph}", file=f)
        print(gm.graph, file=after_io)
        t = before_io.getvalue() == after_io.getvalue()
        log.info(
            "%s, save before/after graph to %s, graph before/after are the same = %s, time elapsed = %s",
            msg,
            f.name,
            t,
            time_elapsed,
        )


def is_multi_outputs_template(input_buf: Buffer | Operation | None) -> bool:
    """
    Check if input buffer is a multi-outputs template buffer
    """
    from . import ir

    return (
        isinstance(input_buf, ir.TemplateBuffer)
        and input_buf.is_multi_outputs_template()
    )


def is_output_of_multi_outputs_template(
    input_buf: Buffer | Operation | None,
) -> bool:
    """
    Check if input buffer is a output of multi-outputs template buffer
    """
    from . import ir

    return (
        isinstance(input_buf, ir.MultiOutput)
        and len(input_buf.inputs) == 1
        and is_multi_outputs_template(input_buf.inputs[0])  # type: ignore[arg-type]
    )


def is_collective(
    node: Node | Operation | None,
    op: torch._ops.OperatorBase | None = None,
) -> bool:
    if node is None:
        return False

    from . import ir

    return (
        isinstance(node, ir._CollectiveKernel)
        and not isinstance(node, ir._WaitKernel)
        and (op is None or node.op_overload is op)
    ) or (
        # TODO: this is a temporary solution to ensure that we can identify torchrec's
        # communication ops. But in order to allow better communication and computation
        # overlap, torchrec's communication ops should be not used.
        type(node) is ir.FallbackKernel
        and (
            # NOTE: the `hasattr()` check is to bypass errors such as the following:
            # AttributeError: '_OpNamespace' 'torchrec' object has no attribute 'all_to_all_single'
            (
                hasattr(torch.ops.torchrec, "all_to_all_single")
                and node.op_overload == torch.ops.torchrec.all_to_all_single.default
            )
            or (
                hasattr(torch.ops.torchrec, "all_gather_into_tensor")
                and node.op_overload
                == torch.ops.torchrec.all_gather_into_tensor.default
            )
            or (
                hasattr(torch.ops.torchrec, "reduce_scatter_tensor")
                and node.op_overload == torch.ops.torchrec.reduce_scatter_tensor.default
            )
        )
    )


def is_wait(node: IRNode | Operation | None) -> bool:
    from . import ir

    return type(node) is ir._WaitKernel


def contains_collective(
    snode: BaseSchedulerNode,
    filter_fn: Callable[[BaseSchedulerNode], bool] | None = None,
) -> bool:
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode):
        return any(contains_collective(x) for x in snode.snodes)

    return is_collective(snode.node) and (filter_fn is None or filter_fn(snode))


def contains_wait(snode: BaseSchedulerNode) -> bool:
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode):
        return any(contains_wait(x) for x in snode.snodes)
    else:
        return is_wait(snode.node)


def is_fallback_op(
    node: Operation | None,
    op: torch._ops.OpOverload | Collection[torch._ops.OpOverload],
) -> bool:
    from . import ir

    if isinstance(op, torch._ops.OpOverload):
        op = [op]
    return isinstance(node, ir.FallbackKernel) and node.op_overload in op


def buf_name_to_fused_snode(
    buf_name: str, name_to_buf: dict[str, Any], name_to_fused_node: dict[str, Any]
) -> Any:
    return name_to_fused_node[name_to_buf[buf_name].defining_op.get_name()]


def find_recursive_deps_of_node(
    snode: BaseSchedulerNode,
    collected_node_set: MutableSet[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    criteria_cb: Callable[[Any], bool] = lambda snode: False,
) -> None:
    if criteria_cb(snode):
        return
    collected_node_set.add(snode)
    for dep in snode.unmet_dependencies:
        defining_op_for_dep = buf_name_to_fused_snode(
            dep.name, name_to_buf, name_to_fused_node
        )
        if defining_op_for_dep in collected_node_set:
            continue
        find_recursive_deps_of_node(
            defining_op_for_dep,
            collected_node_set,
            name_to_buf,
            name_to_fused_node,
            criteria_cb=criteria_cb,
        )


def find_recursive_users_of_node(
    snode: BaseSchedulerNode,
    collected_node_set: MutableSet[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    criteria_cb: Callable[[Any], bool] = lambda snode: False,
) -> None:
    if criteria_cb(snode):
        return
    collected_node_set.add(snode)
    for o in snode.get_outputs():
        for user in o.users:
            if user.node is None:
                raise AssertionError("user.node is None in collect_defined_kernels")
            if user.node.get_name() == "OUTPUT":
                continue
            if user.node.get_name() not in name_to_fused_node:
                continue
            user_op = name_to_fused_node[user.node.get_name()]
            if user_op in collected_node_set:
                continue
            find_recursive_users_of_node(
                user_op,
                collected_node_set,
                name_to_buf,
                name_to_fused_node,
                criteria_cb=criteria_cb,
            )


def num_fw_fixed_arguments(dynamo_gm_num_inputs: int, aot_fw_gm_num_inputs: int) -> int:
    "Computes the number of inputs to the aot fw graph which have fixed addresses (params and buffers)"
    num_rng_seed_offset_inputs = (
        2 if torch._functorch.config.functionalize_rng_ops else 0
    )
    # AOT won't lift any parameters if we're inlining NN Modules
    # however desugaring subclasses will still add arguments
    # resulted in extra fixed inputs https://github.com/pytorch/pytorch/issues/130502
    return aot_fw_gm_num_inputs - dynamo_gm_num_inputs - num_rng_seed_offset_inputs


def count_tangents(fx_g: torch.fx.GraphModule) -> int:
    """
    Infers which inputs are static for a backwards graph
    """

    def is_saved_tensor(x: Node) -> bool:
        return (
            "tangents" not in x.name
            and "bwd_seed" not in x.name
            and "bwd_base_offset" not in x.name
            and "bwd_rng_state" not in x.name
        )

    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            if is_saved_tensor(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1

    if static_arg_idxs != list(range(len(static_arg_idxs))):
        raise AssertionError(
            f"Expected static_arg_idxs to be contiguous from 0, got {static_arg_idxs}"
        )
    return len(static_arg_idxs)


def get_static_bw_input_idxs(fx_g: torch.fx.GraphModule) -> list[int]:
    """
    Returns indices of backward graph inputs that are always at fixed
    addresses: primals (parameters/buffers/user inputs saved for backward).
    Excludes saved activations which may not be at fixed addresses when
    the forward is partitioned for CUDA graphs.
    """
    static_idxs = []
    for idx, n in enumerate(fx_g.graph.nodes):
        if n.op != "placeholder":
            break
        if n.name.startswith("primals_"):
            static_idxs.append(idx)
    return static_idxs


@dataclasses.dataclass
class BoxedBool:
    value: bool

    def __bool__(self) -> bool:
        return self.value

    @staticmethod
    def disable(obj: Any) -> BoxedBool | bool:
        if isinstance(obj, BoxedBool):
            obj.value = False
            return obj
        return False


@contextlib.contextmanager
def collect_defined_kernels(kernel_list: list[str]) -> Iterator[None]:
    from .codegen.wrapper import PythonWrapperCodegen

    orig_define_kernel = PythonWrapperCodegen.define_kernel

    def define_kernel(
        self: PythonWrapperCodegen,
        kernel_name: str,
        kernel_code: str,
        metadata: str | None = None,
        gpu: bool = True,
        cpp_definition: str | None = None,
    ) -> Any:
        kernel_list.append(kernel_code)
        return orig_define_kernel(
            self, kernel_name, kernel_code, metadata, gpu, cpp_definition
        )

    with mock.patch.object(PythonWrapperCodegen, "define_kernel", define_kernel):
        yield


def get_cloned_parameter_buffer_name(name: str) -> str:
    return name + "__original__"


def is_gpu(device: str | None) -> bool:
    return device in GPU_TYPES


def is_rocm() -> bool:
    """Check if we're running on ROCm/HIP platform."""
    return torch.version.hip is not None


@functools.lru_cache(None)
def _triton_supported_fp8_dtypes(
    backend_name: str, arch: int | str, warp_size: int
) -> tuple[str, ...] | None:
    if not has_triton_package():
        return None

    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import make_backend

        # This is the same option Triton checks when lowering tl.float8* types.
        target = GPUTarget(backend_name, arch, warp_size)
        options = make_backend(target).parse_options({})
    except (AttributeError, ImportError, KeyError, TypeError):
        return None

    supported_fp8_dtypes = getattr(options, "supported_fp8_dtypes", None)
    if supported_fp8_dtypes is None:
        return None
    return tuple(supported_fp8_dtypes)


def is_triton_fp8_dtype_supported(
    dtype: torch.dtype,
    device: torch.device | str | None = None,
    *,
    triton_backend: str | None = None,
    triton_arch: int | str | None = None,
    warp_size: int | None = None,
) -> bool:
    if dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ):
        return True

    triton_dtype = _type_of(dtype).removeprefix("*")

    if triton_backend is None:
        if device is None:
            return True
        device = torch.device(device)
        if device.type != "cuda":
            return True
        if not torch.cuda.is_available():
            return True

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        if is_rocm():
            triton_backend = "hip"
            triton_arch = properties.gcnArchName.split(":")[0]
        else:
            triton_backend = "cuda"
            triton_arch = properties.major * 10 + properties.minor
        warp_size = getattr(properties, "warp_size", None)

    if triton_arch is None:
        raise ValueError("triton_arch must be provided with triton_backend")
    if warp_size is None:
        warp_size = 64 if triton_backend == "hip" else 32

    supported_fp8_dtypes = _triton_supported_fp8_dtypes(
        triton_backend, triton_arch, warp_size
    )
    if supported_fp8_dtypes is None:
        return True
    return triton_dtype in supported_fp8_dtypes


def device_need_guard(device: str) -> bool:
    return device != "mps" and is_gpu(device)  # TODO: MPS does not expose streams now


def needs_fallback_due_to_atomic_add_limitations(dtype: torch.dtype) -> bool:
    if dtype == torch.bfloat16 and torch.cuda.is_available():
        return torch.cuda.get_device_capability() < (9, 0)
    elif dtype == torch.bfloat16 and torch.xpu.is_available():
        return True
    else:
        return dtype in (torch.int64, torch.bool)


def use_scatter_fallback(
    op_overload: torch._ops.OpOverload,
    reduction_type: str | None,
    self_dtype: torch.dtype,
    src_dtype: torch.dtype,
    src_device_type: str,
    src_is_tensor: bool,
) -> bool:
    if (
        op_overload.overloadpacket
        in (torch.ops.aten.scatter_reduce_, torch.ops.aten.scatter_reduce)
        and reduction_type is None
    ):
        return False

    reduce_ty = (
        "add" if op_overload.overloadpacket == torch.ops.aten.scatter_ else "sum"
    )

    return (
        reduction_type not in (None, reduce_ty)
        or (
            src_is_tensor
            and is_gpu(src_device_type)
            and needs_fallback_due_to_atomic_add_limitations(src_dtype)
        )
        or (
            op_overload.overloadpacket == torch.ops.aten.scatter_reduce_
            and reduction_type == "sum"
            and src_is_tensor
            and src_device_type == "cpu"
            and config.cpp.fallback_scatter_reduce_sum
            and (config.cpp.dynamic_threads or parallel_num_threads() != 1)
        )
        or (reduction_type == reduce_ty and self_dtype in (torch.bool, torch.int64))
        or torch.are_deterministic_algorithms_enabled()
    )


def dump_node_schedule(node_schedule: Sequence[BaseSchedulerNode]) -> None:
    """
    An API that can be used in pdb to dump a node_schedule.
    Right mainly dump the read/write dependencies but can add more as needed.
    """
    from torch._inductor.codegen.simd import DisableReduction, EnableReduction
    from torch._inductor.scheduler import SchedulerNode

    print(f"Node schedule with {len(node_schedule)} nodes")
    for idx, node in enumerate(node_schedule):
        print(f" {idx:3}:")
        # pyrefly: ignore [unnecessary-comparison]
        if node is EnableReduction:
            print("enable reduction")
        # pyrefly: ignore [unnecessary-comparison]
        elif node is DisableReduction:
            print("disable reduction")
        elif isinstance(node, SchedulerNode):
            is_red = node.is_reduction()
            print(f"{'red' if is_red else 'pw'} scheduler node")
            if is_red:
                if node.node is None:
                    raise AssertionError(
                        "SchedulerNode.node is None for reduction node"
                    )
                print(f"original reduction hint {node.node.data.reduction_hint}")  # type: ignore[attr-defined]
            print("ReadDep:")
            for dep in node.read_writes.reads:
                print(dep)
            print("WriteDep:")
            for dep in node.read_writes.writes:
                print(dep)
        else:
            raise RuntimeError(f"Unrecognized node type: {type(node)}")


def tensor_is_aligned(tensor: torch.Tensor) -> bool:
    # See Note: [Input Alignment handling in Inductor]
    # Right now, we don't try to guard on the alignment of the storage offset.
    # When this comment was written, non-symbolic storage_offsets are not guarded on
    # but symbolic storage_offsets are. For consistency, we suppress guard creation
    # upon performing this check: that ensures that we don't add recompiles when we
    # add this logic.
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    return statically_known_true(
        (tensor.storage_offset() * get_dtype_size(tensor.dtype)) % GPU_ALIGN_BYTES == 0
    )


def should_assume_input_aligned(example_input: torch.Tensor) -> bool:
    # See Note: [Input Alignment handling in Inductor]

    # right now, we only care about alignment for cuda tensors.
    if not is_gpu(example_input.device.type):
        return False
    return config.assume_aligned_inputs or tensor_is_aligned(example_input)


def maybe_get_suppress_shape_guards_ctx() -> contextlib.AbstractContextManager[None]:
    # Try to get TracingContext.try_get().fake_mode.shape_env.suppress_guards()
    # If it's not available, return a nullcontext.

    # If we're dealing with cudagraphs, we might not have a tracing_context
    tracing_context = torch._guards.TracingContext.try_get()
    if not tracing_context:
        return contextlib.nullcontext()

    # In standalone inductor compile mode, we might not have a shape_env attached to the fake mode
    if not tracing_context.fake_mode or not tracing_context.fake_mode.shape_env:
        return contextlib.nullcontext()
    shape_env = tracing_context.fake_mode.shape_env
    return shape_env.suppress_guards()


def run_and_get_cpp_code(
    fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs
) -> tuple[_T, str]:
    # We use the patch context manager instead of using it as a decorator.
    # In this way, we can ensure that the attribute is patched and unpatched correctly
    # even if this run_and_get_cpp_code function is called multiple times.
    with unittest.mock.patch.object(config, "debug", True):
        torch._dynamo.reset()
        import io
        import logging

        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        from torch._inductor.codecache import output_code_log

        output_code_log.addHandler(ch)
        prev_level = output_code_log.level
        output_code_log.setLevel(logging.DEBUG)
        result = fn(*args, **kwargs)
        s = log_capture_string.getvalue()
        output_code_log.setLevel(prev_level)
        output_code_log.removeHandler(ch)
    return result, s


def shape_env_from_inputs(inputs: Sequence[InputType]) -> ShapeEnv | None:
    fake_mode = detect_fake_mode(inputs)

    # TODO(voz): It would be nice to enable this assert, but there are lots of tests that
    # pass in real inputs for now.
    # if len(inputs) > 0:
    # assert fake_mode is not None, breakpoint()

    if fake_mode is not None:
        return fake_mode.shape_env

    # When there are no tensor inputs, get shape_env from the first SymInt.
    for input in inputs:
        if isinstance(input, torch.SymInt):
            return input.node.shape_env

        # Check tensor sizes and strides for SymInt values
        if isinstance(input, torch.Tensor):
            for size in input.size():
                if isinstance(size, torch.SymInt):
                    return size.node.shape_env
            for stride in input.stride():
                if isinstance(stride, torch.SymInt):
                    return stride.node.shape_env

    # TODO(voz): Should we always have one anyway?
    return None


def align_inputs_from_check_idxs(
    model: Callable[[list[InputType]], _T],
    inputs_to_check: Sequence[int],
    mutated_input_idxs: OrderedSet[int],
) -> Callable[[list[InputType]], _T]:
    if len(inputs_to_check) == 0:
        return model

    def run(new_inputs: list[InputType]) -> Any:
        old_tensors, new_tensors = copy_misaligned_inputs(
            new_inputs, inputs_to_check, mutated_input_idxs
        )
        out = model(new_inputs)

        # If a mutated tensor was cloned to be aligned, we need to reflect back the mutation to the
        # original tensor.
        if len(old_tensors):
            torch._foreach_copy_(old_tensors, new_tensors)

        return out

    return run


def clone_preserve_strides(x: torch.Tensor) -> torch.Tensor:
    if 0 in x.size():
        # Short-circuits if the shape has no elements
        needed_size = 0
    else:
        needed_size = (
            sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
        )
    buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
    return torch.as_strided(buffer, x.size(), x.stride())


def copy_misaligned_inputs(
    new_inputs: list[InputType],
    check_inputs_idxs: Sequence[int],
    return_pair_idxs: OrderedSet[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Clones misaligned tensors which we inferred were aligned. Returns a tuple of [old_tensors], [new_tensors] for every
    cloned tensor which is in `return_pair_idxs`.
    """

    old_tensors: list[torch.Tensor] = []
    new_tensors: list[torch.Tensor] = []

    # hoist above loop because this is on the hot path
    ret_pair_defined = return_pair_idxs is not None
    for i in check_inputs_idxs:
        _inp = new_inputs[i]
        if not isinstance(_inp, torch.Tensor):
            raise AssertionError(f"Expected tensors only, but got: {type(_inp)}")
        if _inp.data_ptr() % ALIGNMENT:
            new_inputs[i] = clone_preserve_strides(_inp)

            if ret_pair_defined and i in return_pair_idxs:  # type: ignore[operator]
                old_tensors.append(_inp)
                new_tensors.append(new_inputs[i])  # type: ignore[arg-type]

    return old_tensors, new_tensors


def remove_unaligned_input_idxs(
    inputs: Sequence[InputType],
    static_input_idxs: Sequence[int],
) -> Sequence[int]:
    """
    We require all inputs to be aligned, so introduce a copy for any
    that aren't.
    """
    aligned_static_input_idxs = []
    for idx in static_input_idxs:
        input = inputs[idx]
        if isinstance(input, torch.Tensor) and (input.data_ptr() % ALIGNMENT) == 0:
            aligned_static_input_idxs.append(idx)
    if len(aligned_static_input_idxs) != len(static_input_idxs):
        return aligned_static_input_idxs
    return static_input_idxs


def expr_fits_within_32bit(e: sympy.Expr) -> bool:
    """Check if an expression fits within 32-bit integer range.

    NOTE: This function intentionally does not install guards. Callers are
    responsible for guarding (e.g. via check_leq) when they decide to use
    32-bit indexing based on this result.
    """
    from .virtualized import V

    int_max = torch.iinfo(torch.int32).max
    guarding_hint_or_throw = V.graph.sizevars.guarding_hint_or_throw
    has_guarding_hint = V.graph.sizevars.shape_env.has_guarding_hint

    if config.assume_32bit_indexing:
        V.graph.sizevars.check_leq(e, int_max)  # type: ignore[arg-type]
        return True

    # Allow for unhinted e as long as we can still statically prove
    # (e.g., via ValueRanges) that it is still in bounds
    if V.graph.sizevars.statically_known_true(e <= int_max):
        return True

    # AOTI doesn't guard on < 2**32, so checking hints isn't a viable option,
    # in case the hinted value is < 2**32, but the allowed range is larger.
    # However, to prevent possible perf regressions on pre-existing AOTI models
    # which don't set an upper bound on the valid range, we'll skip the check.
    # To recap:
    # - If using AOTI:
    #   - If allowed range has no upper bound, then check the hint to determine
    #       whether this fits in int32
    #   - If allowed range does have an upper bound, then obey the upper bound
    #       (check whether upper bound < int32_max) without checking the hint.

    if V.aot_compilation:
        # check whether value has an upper bound (1e20 is > INT64_MAX, assume
        # there is no upper bound if it can be larger than 1e20)
        if V.graph.sizevars.statically_known_true(e < 1e20):
            # if so, then assume int_max < upper bound < inf
            # so this could potentially have int64 values
            return False

    # Otherwise, the hint MUST exist and be in range
    return has_guarding_hint(e) and guarding_hint_or_throw(e) <= int_max


def set_tracing_context_output_strides(
    example_inputs: Sequence[Any], compiled_graph: CompiledFxGraph
) -> None:
    # Return the output strides to the caller via TracingContext
    context = torch._guards.TracingContext.try_get()
    if context is not None and context.output_strides is not None:
        if len(context.output_strides) != 0:
            raise AssertionError(
                f"Expected empty output_strides, got length {len(context.output_strides)}"
            )
        shape_env = shape_env_from_inputs(example_inputs)
        if compiled_graph.output_strides is None:
            raise AssertionError("compiled_graph.output_strides is None")
        for exprs in compiled_graph.output_strides:
            if exprs is None:
                context.output_strides.append(None)
            else:
                fakify_first_call = False
                if ctx := torch._guards.TracingContext.try_get():
                    fakify_first_call = ctx.fakify_first_call

                def map_expr(e: Any) -> float | int | SymInt | SymFloat | SymBool:
                    if shape_env is None:
                        return int(e)
                    if fakify_first_call:
                        return shape_env.deserialize_symexpr(e)
                    return shape_env.evaluate_symexpr(e)

                context.output_strides.append(
                    tuple(map_expr(e) for e in exprs)  # type: ignore[misc]
                )


def should_use_remote_fx_graph_cache() -> bool:
    if config.fx_graph_remote_cache is not None:
        return config.fx_graph_remote_cache
    if not config.is_fbcode():
        return False

    if torch._utils_internal.is_fb_unit_test():
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:fx_graph_memcache_version"
    )


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# correct cases where Triton types names don't match PyTorch
_triton_type_mapping = {
    "tl.bool": "tl.int1",
    "tl.float8_e4m3fn": "tl.float8e4nv",
    "tl.float8_e5m2": "tl.float8e5",
    "tl.float8_e4m3fnuz": "tl.float8e4b8",
    "tl.float8_e5m2fnuz": "tl.float8e5b16",
    # TODO: remove when support is added in triton
    # https://github.com/triton-lang/triton/issues/6054
    "tl.float8_e8m0fnu": "tl.uint8",
    "tl.float4_e2m1fn_x2": "tl.uint8",
}
_torch_triton_mapping = {v: k for k, v in _triton_type_mapping.items()}


_triton_type_re = re.compile(r"^.*[.]")


def triton_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type"""
    triton_type_name = _triton_type_re.sub("tl.", str(dtype))
    return _triton_type_mapping.get(triton_type_name, triton_type_name)


def triton_type_to_torch(dtype: str) -> torch.dtype:
    adjusted_type = _torch_triton_mapping.get(dtype, dtype)
    type_name = adjusted_type.replace("tl.", "")
    out_dtype = getattr(torch, type_name)
    if not isinstance(out_dtype, torch.dtype):
        raise AssertionError(f"Expected torch.dtype, got {type(out_dtype)}")
    return out_dtype


def is_same_tensor(data: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        not data.is_mkldnn  # type: ignore[bad-return]
        and data.size() == value.size()
        and data.stride() == value.stride()
        and data.dtype == value.dtype
        and data.device == value.device
        and data.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        and data.storage_offset() == value.storage_offset()
    )


def is_same_mkldnn_tensor(data: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        data.is_mkldnn
        and data.size() == value.size()
        and data.dtype == value.dtype
        and data.device == value.device
        and torch.ops.mkldnn.data_ptr(data) == torch.ops.mkldnn.data_ptr(value)
    )


@functools.cache
def boolean_ops() -> tuple[str, ...]:
    return (
        "isinf",
        "isnan",
        "logical_not",
        "logical_and",
        "signbit",
        "and_",
        "le",
        "lt",
        "ge",
        "gt",
        "eq",
        "ne",
        "or_",  # TODO should remove this op
        "xor",
    )


@dataclasses.dataclass
class OpDtypeRule:
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
    override_return_dtype: torch.dtype | None


op_dtype_propagation_rules: dict[str, OpDtypeRule] = {}


def register_op_dtype_propagation_rules(
    name: str,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
    override_return_dtype: torch.dtype | None,
) -> None:
    op_dtype_propagation_rules[name] = OpDtypeRule(
        type_promotion_kind, override_return_dtype
    )


op_requires_libdevice_fp64: OrderedSet[str] = OrderedSet()


def register_op_requires_libdevice_fp64(name: str) -> None:
    op_requires_libdevice_fp64.add(name)


def get_current_backend(device_type: str | None = None) -> str:
    from torch._inductor.virtualized import V

    if not device_type:
        device_type = V.graph.get_current_device_or_throw().type
    if device_type == "cpu":
        return config.cpu_backend
    elif device_type == "mps":
        return "mps"
    elif device_type == "xpu":
        return config.xpu_backend
    elif device_type == "tpu":
        return config.tpu_backend
    else:
        return config.cuda_backend


def device_supports_fp64(device: torch.device | None) -> bool:
    """Check if the given device supports float64."""
    if device is not None and device.type == "xpu":
        return torch.xpu.get_device_properties(device).has_fp64
    return True


def upcast_compute_type(dtype: torch.dtype) -> torch.dtype:
    """Maybe upcast [b]float16 to float32"""
    if (
        dtype in (torch.float16, torch.bfloat16)
        and config.triton.codegen_upcast_to_fp32
        and get_current_backend() == "triton"
    ):
        return torch.float32
    return dtype


KeyType = TypeVar("KeyType")
ValType = TypeVar("ValType")


class ScopedDict(MutableMapping[KeyType, ValType]):
    """
    A dictionary-like object that allows for scoped updates. It maintains
    an original dictionary and a set of new items that can override
    the original items within the scope.  The original dictionary is
    unmodified.
    """

    def __init__(self, original_dict: Mapping[KeyType, ValType]):
        self.original_dict = original_dict
        self.new_items: dict[KeyType, ValType] = {}

    def __getitem__(self, key: KeyType) -> ValType:
        if key in self.new_items:
            return self.new_items[key]
        return self.original_dict[key]

    def __setitem__(self, key: KeyType, value: ValType) -> None:
        self.new_items[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.new_items or key in self.original_dict

    def get(self, key: KeyType, default: ValType | None = None) -> ValType | None:  # type: ignore[override]
        if key in self.new_items:
            return self.new_items[key]
        return self.original_dict.get(key, default)

    def __len__(self) -> int:
        n = len(self.original_dict)
        for k in self.new_items:
            if k not in self.original_dict:
                n += 1
        return n

    def __iter__(self) -> Iterator[KeyType]:
        yield from self.original_dict
        for k in self.new_items:
            if k not in self.original_dict:
                yield k

    def __bool__(self) -> bool:
        return bool(self.original_dict or self.new_items)

    def __delitem__(self, key: KeyType) -> None:
        raise NotImplementedError


@dataclass_transform(frozen_default=True)
def ir_dataclass(cls: type[Any] | None = None, /, *, frozen: bool = True) -> Any:
    def wrap(cls: _T) -> _T:
        return dataclasses.dataclass(cls, kw_only=True, frozen=frozen)  # type: ignore[call-overload]

    if cls is None:
        return wrap
    return wrap(cls)


def get_donated_idxs() -> list[int] | None:
    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context is not None and tracing_context.fw_metadata:
        return tracing_context.fw_metadata.bw_donated_idxs
    return None


class TritonAttrsDescriptorVersion(enum.Enum):
    V0_NO_TRITON = 0
    V1_COMPILER = 1  # triton.compiler.compiler.AttrsDescriptor
    V2_BACKENDS = 2  # triton.backends.compiler.AttrsDescriptor
    V3_BACKENDS_TUPLE = (
        3  # triton.backends.compiler.AttrsDescriptor, but with tuple support
    )
    V4_DICT = 4  # a raw dict


@functools.cache
def get_triton_attrs_descriptor_version() -> TritonAttrsDescriptorVersion:
    if importlib.util.find_spec("triton") is None:
        return TritonAttrsDescriptorVersion.V0_NO_TRITON

    import triton.backends.compiler
    import triton.compiler.compiler

    if hasattr(triton.backends.compiler, "AttrsDescriptor"):
        # Triton 3.2.0
        # AttrsDescriptor was moved from triton.compiler.compiler to triton.backends.compiler.
        # AttrsDescriptor and its serialization format were also changed.

        # TODO: implement V3_BACKENDS_TUPLE
        # On Dec 9, 2024, tuple support (triton #5220) was implemented and breaks handling.
        # We don't have a way to detect this (and haven't implemented this version)
        return TritonAttrsDescriptorVersion.V2_BACKENDS
    elif hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        # Triton 3.0.0
        return TritonAttrsDescriptorVersion.V1_COMPILER
    else:
        # After Jan 1, 2025
        # AttrsDescriptor was removed and replaced with a raw dict.
        return TritonAttrsDescriptorVersion.V4_DICT


def triton_version_uses_attrs_dict() -> bool:
    return get_triton_attrs_descriptor_version() == TritonAttrsDescriptorVersion.V4_DICT


def get_op_names(op: torch._ops.OperatorBase) -> tuple[str, str]:
    op_overload_packet_name: str = op.name()
    op_overload_name = (
        f"{op_overload_packet_name}.{op._overloadname}"
        if isinstance(op, torch._ops.OpOverload)
        else op_overload_packet_name
    )
    return op_overload_packet_name, op_overload_name


def _fx_node_is_input_dependent_cudagraph_unsafe(fx_node: torch.fx.Node) -> bool:
    """
    Check if an FX node is cudagraph-unsafe based on its input arguments.

    Some ops are only cudagraph-unsafe depending on their inputs (e.g., index_put
    with boolean indices triggers .nonzero() during capture, but integer indices
    are safe).
    """
    from torch.fx.operator_schemas import normalize_function

    target = fx_node.target
    if not isinstance(target, torch._ops.OpOverload):
        return False

    # index_put with boolean indices triggers .nonzero() during capture
    if target in (
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put_.default,
        torch.ops.aten._unsafe_index_put.default,
    ):
        normalized = normalize_function(
            target, fx_node.args, fx_node.kwargs, normalize_to_only_use_kwargs=True
        )
        if normalized is not None:
            _, kwargs = normalized
            indices = kwargs["indices"]
            for idx in indices:
                if idx is not None and idx.meta["val"].dtype in (
                    torch.bool,
                    torch.uint8,
                ):
                    return True

    return False


def is_cudagraph_unsafe_fx_node(fx_node: torch.fx.Node) -> bool:
    """
    Check if an FX node is cudagraph-unsafe.

    This includes:
    - Ops in FORBIDDEN_CUDAGRAPH_OPS (CPU sync, dynamic alloc, etc.)
    - Ops with the cudagraph_unsafe tag
    - Input-dependent unsafe ops (e.g., index_put with boolean indices)
    - Ops with sparse tensor outputs
    """
    target = fx_node.target

    # Check against the forbidden ops set
    if str(target) in FORBIDDEN_CUDAGRAPH_OPS:
        return True

    # Check for cudagraph_unsafe tag
    if (
        isinstance(target, torch._ops.OpOverload)
        and torch._C.Tag.cudagraph_unsafe in target.tags  # type: ignore[attr-defined]
    ):
        return True

    # Check for input-dependent unsafety
    if _fx_node_is_input_dependent_cudagraph_unsafe(fx_node):
        return True

    # Check for sparse tensor outputs
    if (val := fx_node.meta.get("val")) is not None:
        vals = [val] if not isinstance(val, (list, tuple)) else val
        for v in vals:
            if isinstance(v, torch.Tensor) and v.is_sparse:
                return True

    return False


def is_cudagraph_unsafe_op(node: Operation) -> bool:
    """
    Returns True if the node is an op that is not cudagraphable.
    This includes:
    - Ops in FORBIDDEN_CUDAGRAPH_OPS (CPU sync, dynamic alloc, etc.)
    - Ops with the cudagraph_unsafe tag
    - index_put_ with boolean indices (triggers .nonzero() during capture)
    - Control flow nodes (Switch, WhileLoop)
    - Ops with sparse tensor outputs
    """
    from . import ir

    # Control flow nodes are cudagraph-unsafe
    if isinstance(node, (ir.Switch, ir.WhileLoop)):
        return True

    if not isinstance(node, (ir.FallbackKernel, ir.ExternKernel)):
        return False

    fx_node = getattr(node, "fx_node", None)
    if fx_node is not None and is_cudagraph_unsafe_fx_node(fx_node):
        return True

    return False


def get_ld_library_path() -> str:
    path = os.environ.get("LD_LIBRARY_PATH", "")
    if config.is_fbcode():
        from libfb.py.parutil import get_runtime_path

        runtime_path = get_runtime_path()
        if runtime_path:
            lib_path = os.path.join(runtime_path, "runtime", "lib")
            path = os.pathsep.join([lib_path, path]) if path else lib_path

    return path


def is_codegen_graph_partition_subgraph(wrapper: PythonWrapperCodegen) -> bool:
    from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen

    return (
        isinstance(wrapper, SubgraphPythonWrapperCodegen)
        and wrapper.partition_signatures is not None
    )


def is_using_cudagraph_partition() -> bool:
    return (
        torch._inductor.config.triton.cudagraphs
        or _unstable_customized_partition_wrapper.wrapper is not None
    ) and torch._inductor.config.graph_partition


def dtype_from_size(size: int) -> torch.dtype:
    from .virtualized import V

    if V.graph.sizevars.statically_known_lt(
        size, 2**31
    ) and V.graph.sizevars.statically_known_geq(size, -(2**31)):
        return torch.int32
    else:
        return torch.int64


SUPPORTED_MKLDNN_DEVICES = ("cpu", "xpu")


def is_mkldnn_bf16_supported(device_type: str) -> bool:
    """
    Returns True if the device supports MKL-DNN BF16.
    """
    if device_type == "cpu":
        return torch.ops.mkldnn._is_mkldnn_bf16_supported()
    elif "xpu" in device_type:
        # match "xpu", "xpu:0", "xpu:1", etc.
        return True
    return False


def is_mkldnn_fp16_supported(device_type: str) -> bool:
    """
    Returns True if the device supports MKL-DNN FP16.
    """
    if device_type == "cpu":
        return torch.ops.mkldnn._is_mkldnn_fp16_supported()
    elif "xpu" in device_type:
        # match "xpu", "xpu:0", "xpu:1", etc.
        return True
    return False


def tabulate_2d(elements: Sequence[Sequence[T]], headers: Sequence[T]) -> str:
    widths = [len(str(e)) for e in headers]
    for row in elements:
        if len(row) != len(headers):
            raise AssertionError(
                f"Row length {len(row)} != header length {len(headers)}"
            )
        for i, e in enumerate(row):
            widths[i] = max(widths[i], len(str(e)))
    lines = []
    lines.append("|".join(f" {h:{w}} " for h, w in zip(headers, widths)))
    #              widths          whitespace      horizontal separators
    total_width = sum(widths) + (len(widths) * 2) + (len(widths) - 1)
    lines.append("-" * total_width)
    for row in elements:
        lines.append("|".join(f" {e:{w}} " for e, w in zip(row, widths)))
    return "\n".join(lines)


def zip_dicts(
    dict1: Mapping[KeyType, ValType],
    dict2: Mapping[KeyType, ValType],
    d1_default: ValType | None = None,
    d2_default: ValType | None = None,
) -> Generator[tuple[KeyType, ValType | None, ValType | None], None, None]:
    """
    Zip two dictionaries together, replacing missing keys with default values.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        d1_default (Any): the default value for the first dictionary
        d2_default (Any): the default value for the second dictionary

    Yields:
        tuple: A tuple containing the key, the value from dict1 (or d1_default if missing),
               and the value from dict2 (or d2_default if missing).
    """
    # Find the union of all keys
    all_keys = OrderedSet(dict1.keys()) | OrderedSet(dict2.keys())

    # Iterate over all keys
    for key in all_keys:
        # Get the values from both dictionaries, or default if missing
        value1 = dict1.get(key)
        value2 = dict2.get(key)

        yield (
            key,
            value1 if value1 is not None else d1_default,
            value2 if value2 is not None else d2_default,
        )


def maybe_aoti_standalone_config(config_patches: dict[str, Any]) -> dict[str, Any]:
    """
    Ensures the configuration is internally consistent for standalone AOTInductor.

    If `aot_inductor_mode.compile_standalone` is set to True in the provided
    `config_patches` (or falls back to the global config), this function ensures
    that the following configs are also enabled:
        - `aot_inductor.package_cpp_only`

    Args:
        config_patches (dict[str, Any]): A dictionary of user-provided config
            overrides for AOTInductor compilation.

    Returns:
        dict[str, Any]: The possibly-updated `config_patches` dictionary.
    """

    def patch_config(
        config_patches: dict[str, Any], config_name: str, config_value: Any
    ) -> None:
        value = config_patches.get(config_name, getattr(config, config_name))
        if value is None:
            config_patches[config_name] = config_value
        elif not value and value != config_value:
            raise RuntimeError(
                f"Invalid config: {config_name}={config_value} when aot_inductor_mode.compile_standalone is True."
            )

    def force_patch_config(
        config_patches: dict[str, Any], config_name: str, config_value: Any
    ) -> None:
        value = config_patches.get(config_name, getattr(config, config_name))
        if value != config_value:
            log.warning(
                "Overriding: %s=%s when aot_inductor_mode.compile_standalone is True.",
                config_name,
                config_value,
            )
        config_patches[config_name] = config_value

    compile_standalone = config_patches.get(
        "aot_inductor_mode.compile_standalone",
        config.aot_inductor_mode.compile_standalone,
    )
    # Make a copy of the config_patches to avoid modifying the original dictionary, needed for testing
    config_patches = config_patches.copy()
    if compile_standalone:
        # Standlaone AOTInductor means only generate cpp project for building a standalone binary
        patch_config(config_patches, "aot_inductor.package_cpp_only", True)
        # Standlaone AOTInductor needs to embed the kernel code in the binary
        patch_config(config_patches, "aot_inductor.embed_kernel_binary", True)
        # Default to use multi-arch kernel codegen for non-rocm GPU
        patch_config(
            config_patches, "aot_inductor.emit_multi_arch_kernel", not torch.version.hip
        )
        patch_config(
            config_patches, "aot_inductor.model_name_for_generated_files", "aoti_model"
        )
        # TODO: change these two configs to default to None and use patch_config
        force_patch_config(
            config_patches,
            "aot_inductor.link_libtorch",
            config.test_configs.use_libtorch,
        )
        force_patch_config(config_patches, "aot_inductor.dynamic_linkage", False)

    cross_target_platform = config_patches.get(
        "aot_inductor.cross_target_platform",
        config.aot_inductor.cross_target_platform,
    )

    package_constants_in_so = config_patches.get(
        "aot_inductor.package_constants_in_so",
        config.aot_inductor.package_constants_in_so,
    )

    if cross_target_platform == "windows" and package_constants_in_so:
        raise RuntimeError(
            "config.aot_inductor.package_constants_in_so is not supported for windows cross-compilation. "
            "Please use config.aot_inductor.package_constants_on_disk_format = binary_blob."
        )

    return config_patches


def determine_aoti_mmap_flags(consts_size: int) -> tuple[bool, bool]:
    """
    Decide whether we should mmap weights, and whether to store the weights with .so.

    If force_mmap_weights or package_constants_on_disk_format == "binary_blob" configs are set, respect the config.

    Returns tuple (use_external_weights, use_mmap_weights).
    """

    if (
        config.aot_inductor.force_mmap_weights
        and config.aot_inductor.package_constants_on_disk_format == "binary_blob"
    ):
        raise RuntimeError(
            "config.aot_inductor.package_constants_on_disk_format = binary_blob and "
            "config.aot_inductor.force_mmap_weights cannot both be True."
        )

    if config.aot_inductor.force_mmap_weights:
        if config.aot_inductor.cross_target_platform == "windows":
            raise RuntimeError(
                "when cross_target_platform is windows, use_mmap_weights should not be true."
            )
        use_mmap_weights = True
        use_external_weights = False
        return use_external_weights, use_mmap_weights

    if config.aot_inductor.package_constants_on_disk_format == "binary_blob":
        use_external_weights = True
        use_mmap_weights = False
        return use_external_weights, use_mmap_weights

    if consts_size <= 2_000_000_000:
        return False, False

    use_external_weights = False
    use_mmap_weights = not config.is_fbcode()

    return use_external_weights, use_mmap_weights


def is_valid_aoti_model_name() -> bool:
    """
    Validates if a model name is suitable for use in code generation.

    """
    from torch._inductor import config

    model_name = config.aot_inductor.model_name_for_generated_files

    if model_name is None:
        return True

    if not isinstance(model_name, str):
        raise ValueError("Invalid AOTI model name: Model name must be a string")

    if model_name == "":
        return True

    # Can only contain alphanumeric characters and underscores
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", model_name):
        raise ValueError(
            "Invalid AOTI model name: Model name can only contain letters, numbers, and underscores"
        )

    return True


def get_free_symbols(x: IterateExprs, unbacked_only: bool) -> OrderedSet[sympy.Symbol]:
    if unbacked_only:
        return free_unbacked_symbols(x)
    else:
        return free_symbols(x)


def python_subprocess_env() -> dict[str, str]:
    """
    Get a base environment for running Python subprocesses.
    """

    torch_package_root = os.path.dirname(
        os.path.dirname(os.path.abspath(torch.__file__))
    )
    env = {
        # Inherit the environment of the current process.
        **os.environ,
        # Set the PYTHONPATH so the subprocess can find torch.
        "PYTHONPATH": os.environ.get(
            "TORCH_CUSTOM_PYTHONPATH",
            os.pathsep.join((torch_package_root, *sys.path)),
        ),
    }

    # Set PYTHONHOME for internal builds, to account for builds that bundle the
    # runtime.  Otherwise they will use the libraries and headers from the
    # platform runtime instead.
    #
    # This can't be done for external builds.  The process can be run from a
    # venv and that won't include Python headers.  The process needs to be able
    # to search for and find the platform runtime.
    if config.is_fbcode():
        env["PYTHONHOME"] = sysconfig.get_path("data")

    return env


def apply_subprocess_env(extra_env: Mapping[str, str | None] | None) -> None:
    """
    Apply environment updates sent from a parent process to a persistent worker.
    A None value means the variable is absent in the parent and should be
    removed from the worker, rather than leaving a stale value behind.
    """
    if extra_env is None:
        return

    for key, value in extra_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextlib.contextmanager
def patch_subprocess_env(
    extra_env: Mapping[str, str | None] | None,
) -> Iterator[None]:
    """
    Temporarily apply parent process environment updates in a persistent worker.
    """
    if extra_env is None:
        yield
        return

    old_env = dict(os.environ)
    apply_subprocess_env(extra_env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


@dataclasses.dataclass(frozen=True)
class CUDAGraphWrapperMetadata:
    """
    Metadata for Customized CUDAGraphWrapper.

    Currently assumes there is 1 dynamo graph and will extend to
    multiple graphs in the future.
    """

    # The number of partitions that are cudagraphable.
    num_partitions: int

    # Index of the current partition.
    partition_index: int


PartitionFnType = Callable[..., Any]
CUDAGraphWrapperType = Callable[
    [PartitionFnType, CUDAGraphWrapperMetadata], PartitionFnType
]


# only incremented by user call of mark_step_begin
class CUDAGraphWrapper:
    wrapper: CUDAGraphWrapperType | None = None


# A customized partition wrappers from users. Interface should be:
#
# def wrapper(fn: PartitionFnType, metadata: CUDAGraphWrapperMetadata) -> PartitionFnType
#
# Inductor generates N wrapper functions for N partition functions, and mechanically wrap
# each partition fn with the generated wrapper function. Users need to handle all details
# such as static inputs, dynamic shapes, etc.
# Users could customize the wrapper based on the metadata. One example is to have special
# handle for the first and last wrapper function.
#
# Warning: This API is unstable and may change in the future.
_unstable_customized_partition_wrapper = CUDAGraphWrapper()


def set_customized_partition_wrappers(wrapper: CUDAGraphWrapperType) -> None:
    _unstable_customized_partition_wrapper.wrapper = wrapper


def snode_args_kwargs(snode: BaseSchedulerNode) -> tuple[list[Any], dict[str, Any]]:
    node = cast("ExternKernel", snode.node)
    if isinstance(node, torch._inductor.ir.FallbackKernel):
        args, kwargs = node.unflatten_args(node.inputs, node.constant_args)
    else:
        args = [*node.inputs, *node.constant_args]
        kwargs = node.kwargs

    args = node.fill_non_provided_args(args, kwargs)
    kwargs = dict(kwargs)
    op_overload = getattr(node, "op_overload", None)
    if isinstance(op_overload, torch._ops.OpOverload):
        positional_arg_names = [
            arg.name for arg in op_overload._schema.arguments if not arg.kwarg_only
        ]
        for arg_name in positional_arg_names[: len(args)]:
            kwargs.pop(arg_name, None)
    flat_args, flat_args_pytree_spec = pytree.tree_flatten((args, kwargs))

    def _is_tensor_ir(x) -> bool:  # type: ignore[no-untyped-def]
        return isinstance(x, torch._inductor.ir.IRNode) and not isinstance(
            x,
            (torch._inductor.ir.GeneratorState, torch._inductor.ir.OpaqueObjectState),
        )

    flat_args = [
        torch._inductor.ir.ir_node_to_tensor(a, replace_symbols_with_hints=True)
        if _is_tensor_ir(a)
        else a
        for a in flat_args
    ]

    def _tensor(size, dtype, device) -> torch.Tensor:  # type: ignore[no-untyped-def]
        return torch.empty(size, dtype=dtype, device=device)

    def to_real_tensor(e: Any) -> Any:
        if not isinstance(e, torch.Tensor):
            return e
        out = _tensor(e.size(), e.dtype, e.device)
        return out

    flat_args = [to_real_tensor(a) for a in flat_args]
    args, kwargs = pytree.tree_unflatten(flat_args, flat_args_pytree_spec)
    return args, kwargs


def is_nonfreeable_buffers(dep: Dep) -> bool:
    from .virtualized import V

    dep_name = dep.name
    # Subgraphs have a prefix for the name, cleanup the prefix
    # before checking for known strings.
    if V.graph.name:
        dep_name = dep_name.removeprefix(V.graph.name + "_")
    return dep_name.startswith(
        ("primals_", "arg", "fwd_rng_state", "bwd_rng_state", "tangents")
    )


# Make sure to also include your jinja templates within torch_package_data in setup.py, or this function won't be able to find them
def load_template(name: str, template_dir: Path) -> str:
    """Load a template file and return its content."""
    path = template_dir / f"{name}.py.jinja"
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        # Name the offending file: a bare UnicodeDecodeError hides which
        # template is malformed, which is easy to misdiagnose when it surfaces
        # deep inside kernel lowering.
        raise ValueError(f"Template {path} is not valid UTF-8: {e}") from e


def should_fallback_by_default(node: torch.fx.Node) -> bool:
    """Decide whether fallback for a node. This is only used in inductor lite mode."""
    target = node.target

    if not isinstance(target, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        raise AssertionError(
            f"Expected OpOverload or HigherOrderOperator, but found {type(target)}"
        )

    if not config.fallback_by_default:
        return False

    # some ops need special handle due to dynamic shapes. we can avoid
    # fallback if they do not impact numerics.
    skip_fallback_due_to_dynamic_shape = OrderedSet(
        [
            torch.ops.aten._assert_scalar.default,
            torch.ops.aten.lift_fresh_copy.default,
        ]
    )

    if target in skip_fallback_due_to_dynamic_shape:
        return False

    # Most hops have registered lowering. We should follow the lowering and not fallback.
    # However, in rare cases, hops may not register lowering, such as
    # torch.ops.higher_order.triton_kernel_wrapper_functional. We should fallback for
    # these hops.
    fallback_hops = OrderedSet(
        [torch.ops.higher_order.triton_kernel_wrapper_functional]
    )

    if isinstance(target, torch._ops.HigherOrderOperator):
        return target in fallback_hops

    return not _needs_inductor_compile(node)


# Collective operation names for specialized benchmarking
COLLECTIVE_OPS = OrderedSet(
    [
        "torch.ops._c10d_functional.all_reduce.default",
        "torch.ops._c10d_functional.all_reduce_.default",
        "torch.ops._c10d_functional.all_gather_into_tensor.default",
        "torch.ops._c10d_functional.reduce_scatter_tensor.default",
        "torch.ops._c10d_functional.all_to_all_single.default",
        "torch.ops._c10d_functional_autograd.all_reduce.default",
        "torch.ops._c10d_functional_autograd.all_gather_into_tensor.default",
        "torch.ops._c10d_functional_autograd.reduce_scatter_tensor.default",
        "torch.ops._c10d_functional_autograd.all_to_all_single.default",
        "torch.ops._c10d_functional.isend.default",
        "torch.ops._c10d_functional.irecv.default",
        "torch.ops._c10d_functional.batch_p2p_ops.default",
    ]
)


def is_collective_op(op_name: str) -> bool:
    """Check if an operation is a collective operation."""
    return op_name in COLLECTIVE_OPS


@lru_cache
def tlx_only_cuda_options() -> list[str]:
    try:
        # Succeeds only when fbtriton (a Triton fork) is installed
        from triton.language.extra.tlx.inductor.registry import tlx_only_cuda_options

        return tlx_only_cuda_options

    except ImportError:
        return []


def _round_up(x: int, y: int) -> int:
    """Round x up to the nearest multiple of y."""
    return ((x + y - 1) // y) * y


def _infer_scale_swizzle_impl(
    mat_size: tuple[Any, Any],
    scale_size: tuple[Any, ...],
    scale_numel: Any,
    mat_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    eq_fn: Callable[[Any, Any], bool],
) -> tuple[Any | None, Any | None]:
    """
    Core implementation for scale/swizzle inference.
    """
    from torch.nn.functional import ScalingType, SwizzleType

    # Tensor-wise: single scale for entire tensor
    if eq_fn(scale_numel, 1):
        return ScalingType.TensorWise, SwizzleType.NO_SWIZZLE

    # Row-wise: one scale per row or column
    if len(scale_size) >= 2:
        if (eq_fn(scale_size[0], mat_size[0]) and eq_fn(scale_size[1], 1)) or (
            eq_fn(scale_size[0], 1) and eq_fn(scale_size[1], mat_size[1])
        ):
            return ScalingType.RowWise, SwizzleType.NO_SWIZZLE

        # Block-wise 1x128 / 128x1 (DeepGemm style)
        if (
            eq_fn(scale_size[0], mat_size[0])
            and eq_fn(scale_size[1], ceildiv(mat_size[1], 128))
        ) or (
            eq_fn(scale_size[1], mat_size[1])
            and eq_fn(scale_size[0], ceildiv(mat_size[0], 128))
        ):
            return ScalingType.BlockWise1x128, SwizzleType.NO_SWIZZLE

        # Block-wise 128x128
        if eq_fn(scale_size[0], ceildiv(mat_size[0], 128)) and eq_fn(
            scale_size[1], ceildiv(mat_size[1], 128)
        ):
            return ScalingType.BlockWise128x128, SwizzleType.NO_SWIZZLE

    # Adjust for packed FP4 data (2 values per element)
    K_multiplier = 2 if mat_dtype == torch.float4_e2m1fn_x2 else 1

    # NVFP4: BlockWise1x16 with float8_e4m3fn scales
    if mat_dtype == torch.float4_e2m1fn_x2 and scale_dtype == torch.float8_e4m3fn:
        if torch.xpu._is_compiled():
            # XPU: no swizzle
            expected_numel_a = ceildiv(mat_size[0], 16) * K_multiplier * mat_size[1]
            expected_numel_b = ceildiv(K_multiplier * mat_size[1], 16) * mat_size[0]
            if eq_fn(scale_numel, expected_numel_a) or eq_fn(
                scale_numel, expected_numel_b
            ):
                return ScalingType.BlockWise1x16, SwizzleType.NO_SWIZZLE
        else:
            # NVIDIA: uses swizzled 32x4x4 layout
            expected_numel_a = _round_up(mat_size[0], 128) * _round_up(
                ceildiv(K_multiplier * mat_size[1], 16), 4
            )
            expected_numel_b = _round_up(mat_size[1], 128) * _round_up(
                ceildiv(K_multiplier * mat_size[0], 16), 4
            )
            if eq_fn(scale_numel, expected_numel_a) or eq_fn(
                scale_numel, expected_numel_b
            ):
                return ScalingType.BlockWise1x16, SwizzleType.SWIZZLE_32_4_4

    # MXFP8: BlockWise1x32 with float8_e8m0fnu scales
    if scale_dtype == torch.float8_e8m0fnu:
        if not torch.version.hip and not torch.xpu._is_compiled():
            # NVIDIA: uses swizzled 32x4x4 layout
            expected_numel_a = _round_up(mat_size[0], 128) * _round_up(
                ceildiv(K_multiplier * mat_size[1], 32), 4
            )
            expected_numel_b = _round_up(mat_size[1], 128) * _round_up(
                ceildiv(K_multiplier * mat_size[0], 32), 4
            )
            if eq_fn(scale_numel, expected_numel_a) or eq_fn(
                scale_numel, expected_numel_b
            ):
                return ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4
        else:
            # AMD/XPU: no swizzle
            expected_numel_a = ceildiv(mat_size[0], 32) * K_multiplier * mat_size[1]
            expected_numel_b = ceildiv(K_multiplier * mat_size[1], 32) * mat_size[0]
            if eq_fn(scale_numel, expected_numel_a) or eq_fn(
                scale_numel, expected_numel_b
            ):
                return ScalingType.BlockWise1x32, SwizzleType.NO_SWIZZLE

    return None, None


def infer_scale_swizzle(
    mat: torch.Tensor, scale: torch.Tensor
) -> tuple[Any | None, Any | None]:
    """
    Infer the scaling type and swizzle mode from matrix and scale tensor shapes/dtypes.

    This function determines how scale factors are laid out relative to the matrix:
    - TensorWise: Single scale for entire tensor
    - RowWise: One scale per row
    - BlockWise1x128/128x128: Block-scaled with float32 scales
    - BlockWise1x32: MXFP8 with float8_e8m0fnu scales (swizzled on NVIDIA)
    - BlockWise1x16: NVFP4 with float8_e4m3fn scales (swizzled)

    Args:
        mat: The matrix tensor (FP8 or FP4)
        scale: The scale factor tensor

    Returns:
        Tuple of (ScalingType, SwizzleType) or (None, None) if unrecognized
    """
    return _infer_scale_swizzle_impl(
        mat_size=(mat.shape[0], mat.shape[1]),
        scale_size=tuple(scale.shape),
        scale_numel=scale.numel(),
        mat_dtype=mat.dtype,
        scale_dtype=scale.dtype,
        eq_fn=lambda a, b: a == b,
    )


def infer_scale_swizzle_ir(
    mat: Buffer,
    scale: Buffer,
    transpose: bool = False,
) -> tuple[Any | None, Any | None]:
    """
    Infer the scaling type and swizzle mode for IR nodes (used during graph lowering).

    This is the IR-compatible version of infer_scale_swizzle, using symbolic
    size comparisons via V.graph.sizevars.statically_known_equals.
    """
    from torch._inductor.virtualized import V

    mat_size = mat.get_size()
    scale_size = scale.get_size()

    # Handle transposed matrix
    if transpose:
        mat_size = (mat_size[1], mat_size[0])

    # Compute scale numel symbolically
    scale_numel = functools.reduce(operator.mul, scale_size, 1) if scale_size else 1

    def symbolic_eq(a: Any, b: Any) -> bool:
        """Compare values using symbolic equality when possible."""
        return V.graph.sizevars.statically_known_equals(a, b)

    return _infer_scale_swizzle_impl(
        mat_size=(mat_size[0], mat_size[1]) if len(mat_size) >= 2 else (mat_size[0], 1),
        scale_size=tuple(scale_size),
        scale_numel=scale_numel,
        mat_dtype=mat.dtype,
        scale_dtype=scale.dtype,
        eq_fn=symbolic_eq,
    )
<|endoftext|># mypy: allow-untyped-defs
"""This file implements the IndexPropagation ops handler, which wraps an
underlying handler to add a limited form of constant propagation, as well as
propagation of sympy expressions downstream of ops.index_expr calls.

For example, say we have the IR:

    tmp0 = ops.index_expr(x, torch.int32)
    tmp1 = ops.constant(2, torch.int32)
    tmp2 = ops.mul(tmp0, tmp1)
    tmp3 = ops.indirect_indexing(tmp2, x_size)
    tmp4 = ops.load("buf0", tmp3)

The underlying handler would just see:

    ops.load("buf0", x * 2)

This is limited by the set of operators handled in the sympy expression
printers.

"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload, TypeAlias

import sympy

import torch
from torch._prims_common import dtype_to_type, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, Max, Min, ModularIndexing, Where
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .ops_handler import DefaultHandler
from .sizevars import statically_known_true
from .utils import generate_assert
from .virtualized import V


_ExprType = sympy.Expr | float | int | bool


def _is_constant(val: _ExprType):
    if isinstance(val, sympy.Basic):
        return val.is_number
    return isinstance(val, (int, float, bool))


def upper_bound(val: _ExprType):
    return bound_sympy(val).upper if isinstance(val, sympy.Expr) else val


@dataclass
class TypedExpr:
    """A SymPy expression with associated type"""

    expr: _ExprType
    dtype: torch.dtype

    def is_constant(self):
        return _is_constant(self.expr)

    def __post_init__(self):
        if _is_constant(self.expr):
            expr = self.expr
            if isinstance(expr, sympy.Expr):
                expr = expr.expand(identity=True)
            expr = dtype_to_type(self.dtype)(expr)
            if is_integer_dtype(self.dtype):
                bits = torch.iinfo(self.dtype).bits
                if self.dtype.is_signed:
                    expr = expr + 2 ** (bits - 1)
                expr = expr % 2**bits
                if self.dtype.is_signed:
                    expr = expr - 2 ** (bits - 1)
            self.expr = expr


class SymPyOps:
    """An ops handler where all IR values are SymPy expressions

    When a value cannot be represented as a SymPy expression, the method is
    either not defined, or returns NotImplemented

    """

    @staticmethod
    def identity(value: Any) -> Any:
        return value

    @staticmethod
    def constant(value: int | float | bool, dtype: torch.dtype) -> TypedExpr:
        return TypedExpr(value, dtype)

    @staticmethod
    def index_expr(value: sympy.Expr | int, dtype: torch.dtype) -> TypedExpr:
        return TypedExpr(value, dtype)

    @staticmethod
    def value_expr(value: sympy.Expr | int, dtype: torch.dtype) -> TypedExpr:
        return TypedExpr(value, dtype)

    @staticmethod
    def to_dtype(
        value: TypedExpr,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = False,
    ) -> TypedExpr:
        return TypedExpr(value.expr, dtype)

    @staticmethod
    def abs(x: TypedExpr) -> TypedExpr:
        return TypedExpr(abs(x.expr), x.dtype)  # type: ignore[arg-type]

    @staticmethod
    def square(x: TypedExpr) -> TypedExpr:
        return TypedExpr(x.expr * x.expr, x.dtype)

    @staticmethod
    def add(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr + y.expr, result_type)

    @staticmethod
    def sub(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr - y.expr, result_type)

    @staticmethod
    def mul(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr * y.expr, result_type)

    @staticmethod
    def neg(x: TypedExpr) -> TypedExpr:
        return TypedExpr(-x.expr, x.dtype)

    @staticmethod
    def floordiv(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        return TypedExpr(FloorDiv(x.expr, y.expr), result_type)

    @staticmethod
    def mod(x: TypedExpr, y: TypedExpr) -> TypedExpr | None:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        result_expr = ModularIndexing(x.expr, sympy.S.One, y.expr)
        return TypedExpr(result_expr, result_type)

    @staticmethod
    def remainder(x: TypedExpr, y: TypedExpr) -> TypedExpr | None:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        x_expr = sympy.sympify(x.expr)
        y_expr = sympy.sympify(y.expr)
        # In these cases, remainder in Python == remainder in C++, so this transformation
        # is sound
        if (
            x_expr.is_nonnegative is not None
            and x_expr.is_nonnegative == y_expr.is_positive
        ):
            result_expr = ModularIndexing(x.expr, sympy.S.One, y.expr)
            return TypedExpr(result_expr, result_type)
        return NotImplemented

    @staticmethod
    def minimum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if result_type == torch.bool:
            return NotImplemented
        return TypedExpr(Min(x.expr, y.expr), result_type)

    @staticmethod
    def maximum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        result_type = torch.promote_types(x.dtype, y.dtype)
        if result_type == torch.bool:
            return NotImplemented
        return TypedExpr(Max(x.expr, y.expr), result_type)


@dataclass
class IndexPropVar:
    value: Any  # Either an IR value, or TypedExpr if is_symbolic is true
    is_symbolic: bool = False

    @staticmethod
    def new_symbolic(expr: TypedExpr) -> "IndexPropVar":
        return IndexPropVar(expr, is_symbolic=True)

    def __post_init__(self):
        if not (not self.is_symbolic or isinstance(self.value, TypedExpr)):
            raise AssertionError("Symbolic IndexPropVar must contain a TypedExpr")


IndexPropResult: TypeAlias = IndexPropVar | tuple["IndexPropResult", ...]


class IndexPropagation(DefaultHandler):
    """Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    """

    def __init__(
        self,
        inner: Any,
        iter_ranges: dict[sympy.Symbol, sympy.Expr],
        indirect_var_ranges: dict[sympy.Symbol, sympy.Expr],
    ) -> None:
        self._inner = inner
        self.shape_env = V.graph.sizevars.shape_env

        var_to_range = {
            k: ValueRanges(0, upper_bound(v) - 1) for k, v in iter_ranges.items()
        }
        self.var_to_range = tuple(
            itertools.chain(self.shape_env.var_to_range.items(), var_to_range.items())
        )
        # NOTE: this is intentionally kept as a reference so the caller can
        # update it in-place
        self.indirect_var_ranges = indirect_var_ranges

        axioms = []
        for x, s in iter_ranges.items():
            axioms.append(0 <= x)
            axioms.append(x < s)
        self.axioms = tuple(axioms) + self.shape_env.get_axioms()

    def materialize_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any:
        # Construct a new constant/index_expr from the SymPy expression
        if _is_constant(expr):
            val = dtype_to_type(dtype)(expr)
            return self._inner.constant(val, dtype)
        return self._inner.index_expr(expr, dtype)

    def value_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> IndexPropResult:
        return self.wrap(self._inner.value_expr(expr, dtype))

    def unwrap(self, a: Any | IndexPropVar) -> Any:
        if isinstance(a, (list, tuple)):
            return tuple(self.unwrap(v) for v in a)

        if not isinstance(a, IndexPropVar):
            return a

        # Prefer the sympy representation if possible
        if a.is_symbolic:
            return self.materialize_expr(a.value.expr, a.value.dtype)

        return a.value

    def wrap(self, a) -> IndexPropResult:
        if isinstance(a, (list, tuple)):
            return tuple(self.wrap(v) for v in a)
        return IndexPropVar(a)

    @overload
    def fallback(
        self,
        name: Literal["indirect_indexing"],
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> IndexPropVar: ...

    @overload
    def fallback(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult: ...

    def fallback(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult:
        # Fallback to the wrapped handler
        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return self.wrap(getattr(self._inner, name)(*new_args, **new_kwargs))

    def propagate_sympy(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult:
        # Build a new SymPy expression from this ops call
        def unwrap(a: Any | IndexPropVar) -> Any:
            if not isinstance(a, IndexPropVar):
                return a
            return a.value

        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        try:
            new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
        except (OverflowError, ValueError):
            # e.g. int(inf) raises OverflowError, int(nan) raises ValueError
            return self.fallback(name, args, kwargs)
        is_valid_expr = new_expr is not NotImplemented and (
            # Inductor doesn't expect floating point in sympy expressions, but
            # allow floating point constants to be propagated
            new_expr.is_constant() or new_expr.expr.is_integer
        )
        if not is_valid_expr:
            return self.fallback(name, args, kwargs)
        return IndexPropVar.new_symbolic(new_expr)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if not hasattr(SymPyOps, name):
            return self.fallback(name, args, kwargs)

        var_arguments = [
            a
            for a in itertools.chain(args, kwargs.values())
            if isinstance(a, IndexPropVar)
        ]
        if not all(v.is_symbolic for v in var_arguments):
            return self.fallback(name, args, kwargs)

        return self.propagate_sympy(name, args, kwargs)

    def statically_true(self, e):
        """
        Given some iter_ranges, return a function that given an expression, returns whether
        it is true or false using value ranges, guard knowledge and runtime_asserts.

        FIXME I think this may not be entirely right, as we may not be able to use all runtime_asserts
              If this is an issue, just use guards in `self.axioms`.

              The proper way of handling this would be to have a global shape_env that adds
              runtime_asserts as they happen in the code. Then, it should be used in SimplifyIndexing
              to perform wrap_expr and in CSEProxy.check_bounds to elide upper / lower bounds also
              for indirect_indexing
        """
        var_to_range = (
            *self.var_to_range,
            *(
                (k, ValueRanges(0, upper_bound(v) - 1))
                for k, v in self.indirect_var_ranges.items()
            ),
        )
        # pyrefly: ignore [bad-argument-type]
        return statically_known_true(self.shape_env, e, self.axioms, var_to_range)

    def indirect_indexing(
        self,
        index: Any | IndexPropVar,
        size: Any,
        check: bool = True,
        wrap_neg=True,
    ) -> Any:
        if isinstance(index, IndexPropVar) and index.is_symbolic:
            # If we find something we can convert into a direct indexing we do so
            # We still need to (perhaps) wrap the expression and add bound checks
            # We want to do this "constant folding", as we don't allow to fuse
            # kernels into indirect indexing

            expr = sympy.sympify(index.value.expr)

            # TODO Perhaps move this logic to the simplify indexing pass
            def wrap_expr(expr):
                # Positive, negative, mixed
                if self.statically_true(0 <= expr):
                    return expr
                elif self.statically_true(expr < 0):
                    return expr + size
                else:
                    return Where(expr < 0, expr + size, expr)

            # -size <= expr only proves the lower bound after negative wrapping.
            can_prove_lower = self.statically_true(0 <= expr) or (
                wrap_neg and self.statically_true(-size <= expr)
            )
            can_prove_upper = self.statically_true(expr < size)
            if wrap_neg:
                expr = wrap_expr(expr)
            if generate_assert(check):
                self.fallback(
                    "check_bounds",
                    (expr, size),
                    dict(lower=not can_prove_lower, upper=not can_prove_upper),
                )
            return expr

        indirect_var = self.fallback(
            "indirect_indexing", (index, size, check, wrap_neg), {}
        ).value
        return indirect_var
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import operator
import warnings
from collections import defaultdict
from collections.abc import (
    Callable,
    Collection,
    Container,
    Generator,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any

import sympy

import torch
import torch.fx
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.fx_passes.control_dependencies import control_deps
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    rebind_unbacked,
    statically_known_true,
    sym_eq,
)
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map
from torch.utils._typing_utils import not_none
from torch.utils.flop_counter import flop_registry

from .virtualized import V


# Check the pattern: (nn.module, F.function/torch.Tensor.method) matched.
# Works for length 2 patterns with 1 module and 1 function/method.
def matches_module_function_pattern(
    pattern: tuple[type[torch.nn.modules.Module], Callable[..., Any]],
    node: torch.fx.Node,
    modules: dict[str, torch.nn.modules.Module],
) -> bool:
    if len(node.args) == 0:
        return False
    if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
        node, torch.fx.Node
    ):
        return False
    # the first node is call_module
    if node.args[0].op != "call_module":
        return False
    if not isinstance(node.args[0].target, str):
        return False
    if node.args[0].target not in modules:
        return False
    if type(modules[node.args[0].target]) is not pattern[0]:
        return False
    # the second node is call_function or call_method
    if node.op != "call_function" and node.op != "call_method":
        return False
    if node.target != pattern[1]:
        return False
    # make sure node.args[0] output is only used by current node.
    if len(node.args[0].users) > 1:
        return False
    return True


def _is_fake_tensor_same(
    new: Any,
    old: Any,
    existing_storages: Mapping[int, int],
    *,
    check_strides: bool = True,
    check_storage: bool = True,
    node: torch.fx.Node | None = None,
    recursive_ids: OrderedSet[tuple[int, int]] | None = None,
) -> bool:
    """Validate that two FakeTensors (or iterables thereof) are the same, including
    storage locations if enabled.

    check_strides: disabling this flag will remove checks for striding.
    check_storage: disabling this flag will remove checks for storage offset and
    location.  This is useful for subgraph argument and output updating, where storage
    location can change without invalidating the subgraph.
    recursive_ids: This is only for use when recursing through collections, and must not
    be supplied by users of this function."""

    def is_intlist_same(new, old):
        # sym_eq builds a symbolic conjunction and statically_known_true then
        # evaluates it, which is a lot of work for the common case where the
        # two sizes/strides are the very same objects. Identity, or equality
        # between concrete ints, already implies equality.
        if len(new) == len(old) and all(
            a is b or (type(a) is int and type(b) is int and a == b)
            for a, b in zip(new, old)
        ):
            return True
        return statically_known_true(sym_eq(new, old))

    if type(new) is not type(old):
        return False

    if not isinstance(new, torch.Tensor):
        if new is None:
            return old is None

        if isinstance(new, Collection):
            if recursive_ids is None:
                recursive_ids = OrderedSet()

            id_pair = (id(new), id(old))
            visited = id_pair in recursive_ids
            recursive_ids.add(id_pair)

            # If we've visited this exact check before while recursing, all elements in
            # this collection have already been validated (or will be validated in the
            # future) by a call at a different layer of recursion.
            return visited or (
                len(new) == len(old)
                and all(
                    _is_fake_tensor_same(
                        new_i,
                        old_i,
                        existing_storages,
                        check_strides=check_strides,
                        check_storage=check_storage,
                        node=node,
                        recursive_ids=recursive_ids,
                    )
                    for new_i, old_i in zip(new, old)
                )
            )

        if isinstance(new, torch.types.py_sym_types):
            return (
                not_none(new.node.shape_env)._maybe_evaluate_static(
                    sympy.Eq(new.node.expr, old.node.expr)
                )
                == sympy.true
            )

        # If this is not a Collection, a sym-type, or a Tensor, then check for Python
        # equality.  This should be fairly conservative, since an object with no
        # implemented __eq__ method will compare IDs.
        return new == old

    if (
        new.layout != old.layout
        or new.dtype != old.dtype
        or not is_intlist_same(new.shape, old.shape)
    ):
        return False

    if new.device != old.device:
        return False

    if (
        check_strides
        and new.layout == torch.strided
        and not is_intlist_same(new.stride(), old.stride())
    ):
        return False

    if not check_storage:
        return True

    if not statically_known_true(
        new.storage_offset() == old.storage_offset()
    ) or get_storage(new) != get_storage(old):
        return False

    def any_user_may_alias(node):
        if not isinstance(node.meta["val"], torch.Tensor):
            # analysis too complicated on lists, can support in the future
            return True

        for user in node.users:
            if not (
                isinstance(user.target, torch._ops.OperatorBase)
                or user.target
                is torch._inductor.fx_passes.reinplace._generalized_scatter
            ):
                return True

            if isinstance(user.target, torch._ops.HigherOrderOperator):
                # HOPs that survive until inductor are all non-aliasing HOPs.  We will
                # likely never support HOPs that are aliasing.
                continue

            # Strategy: do a FakeTensor prop, see if the storage aliases.  If Inductor
            # ever gets tighter invariants on OpOverloads (that is, we ban things like
            # torch.ops.aten.reshape calls in the graph), then this could just be a fast
            # schema lookup.
            is_valid, args, kwargs = get_fake_args_kwargs(user)
            if not is_valid:
                return True

            with (
                V.fake_mode,
                enable_python_dispatcher(),
                contextlib.ExitStack() as stack,
            ):
                # Ignore unbacked symbols (if they exist): we're making this FakeTensor
                # and then throwing it away.
                if (shape_env := V.fake_mode.shape_env) is not None:
                    stack.enter_context(shape_env.ignore_fresh_unbacked_symbols())
                new_fake_tensor = user.target(*args, **kwargs)

            if not isinstance(new_fake_tensor, torch.Tensor):
                # analysis too complicated on lists, can support in the future
                return True

            if get_storage(new_fake_tensor) == get_storage(node.meta["val"]):
                return True

        return False

    # This is the case where it returns a completely fresh storage that's used nowhere
    # else.  If the FakeTensor's storage is fresh and none of the node's users can alias
    # it, then we don't need to update this node.
    if (
        existing_storages[get_storage(old)] == 1
        and get_storage(new) not in existing_storages
        and not any_user_may_alias(node)
    ):
        return True

    return False


def _extract_subgraphs_and_args(
    node: torch.fx.Node,
    valid_subgraphs: Container[torch.fx.GraphModule],
    *args: Any,
    **kwargs: Any,
) -> Generator[tuple[torch.fx.GraphModule, tuple[Any, ...] | None]]:
    """HOPs that invoke subgraphs take a number of different forms.  This function
    regularizes them, yielding subgraphs from the args and kwargs coupled with
    appropriate args to update those subgraphs.

    If the second yielded value is None, this function was unable to determine what args
    to pass to the subgraph."""

    def get_subgraph_args(
        subgraph: torch.fx.GraphModule,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            get_fake(n, subgraph) for n in subgraph.graph.find_nodes(op="placeholder")
        )

    def is_integer(d: torch.dtype) -> bool:
        return not d.is_floating_point and not d.is_complex

    if node.target is torch.ops.higher_order.associative_scan:
        # Associative scan operates on slices of xs (see: scan), but multiple slices.
        # Use the same slice twice to account for cases where only a single slice is
        # input. first_slice_copy tolerates a zero-length scan dim.
        from torch._higher_order_ops.utils import first_slice_copy

        sliced = [first_slice_copy(a) for a in args[1]]
        yield args[0], (*sliced, *sliced, *args[2])
    elif node.target is torch.ops.higher_order.cond:
        subgraph_args = tuple(args[3])
        yield args[1], subgraph_args
        yield args[2], subgraph_args
    elif node.target is torch.ops.higher_order.flex_attention:
        # flex_attention currently yields two subgraphs.  The first (the score
        # calculation) has a signature starting with a scalar argument of the same dtype
        # as the query (args[0]), followed by four scalar integer arguments and optional
        # additional tensors.  The second (the mask calculation) looks similar, with
        # four scalar integer arguments and optional additional (different) tensors.  We
        # assume this format here (adding some defensive assertions), which means that
        # the only inputs we may need to update are the optional additional tensors. See
        # torch._higher_order_ops.flex_attention._math_attention_inner for more details.

        score_subgraph: torch.fx.GraphModule = args[3]
        score_subgraph_args = get_subgraph_args(score_subgraph)
        mask_subgraph = args[4][-1]
        mask_subgraph_args = get_subgraph_args(mask_subgraph)

        integer_arg_dtypes = OrderedSet[torch.dtype](
            a.dtype for a in chain(score_subgraph_args[1:5], mask_subgraph_args[:4])
        )
        if not (
            score_subgraph_args[0].dtype == args[0].dtype
            and len(integer_arg_dtypes) == 1
            and is_integer(integer_arg_dtypes.pop())
            and all(
                len(a.size()) == 0
                for a in chain(score_subgraph_args[:5], mask_subgraph_args[:4])
            )
        ):
            raise AssertionError("flex_attention subgraph arg format has changed!")

        yield score_subgraph, (*score_subgraph_args[:5], *args[7])
        yield args[4][-1], (*mask_subgraph_args[:4], *args[8])
    elif node.target is torch.ops.higher_order.flex_attention_backward:
        fw_subgraph = args[7]
        fw_subgraph_args = get_subgraph_args(fw_subgraph)
        joint_subgraph = args[8]
        joint_subgraph_args = get_subgraph_args(joint_subgraph)
        mask_subgraph = args[9][-1]
        mask_subgraph_args = get_subgraph_args(mask_subgraph)

        integer_arg_dtypes = OrderedSet[torch.dtype](
            a.dtype
            for a in chain(
                fw_subgraph_args[1:5],
                joint_subgraph_args[1:5],
                mask_subgraph_args[:4],
            )
        )
        if not (
            fw_subgraph_args[0].dtype == args[0].dtype
            and joint_subgraph_args[0].dtype == args[0].dtype
            and len(integer_arg_dtypes) == 1
            and is_integer(integer_arg_dtypes.pop())
            and all(
                len(a.size()) == 0
                for a in chain(
                    fw_subgraph_args[:5],
                    joint_subgraph_args[:6],
                    mask_subgraph_args[:4],
                )
            )
        ):
            raise AssertionError(
                "flex_attention_backward subgraph arg format has changed!"
            )

        if fw_subgraph in valid_subgraphs:
            yield fw_subgraph, (*fw_subgraph_args[:5], *args[12])
        if joint_subgraph in valid_subgraphs:
            yield joint_subgraph, (*joint_subgraph_args[:6], *args[12])
        if mask_subgraph in valid_subgraphs:
            yield mask_subgraph, (*mask_subgraph_args[:4], *args[13])
    elif node.target in (
        torch.ops.higher_order.foreach_map,
        torch.ops.higher_order.invoke_quant_packed,
        torch.ops.higher_order.invoke_quant,
    ):
        yield args[0], tuple(args[1:])
    elif node.target is torch.ops.higher_order.invoke_subgraph:
        yield args[0], tuple(args[2:])
    elif node.target is torch.ops.higher_order.map_impl:
        # Map is applied over slices from the first dimension of each value in args[1].
        yield args[0], (*(a[0] for a in args[1]), *args[2])
    elif node.target is torch.ops.higher_order.scan:
        # Scans accept a dim keyword, but the dimensions will be reordered so that at
        # this point we always scan over dim 0.
        # first_slice_copy tolerates a zero-length scan dim.
        from torch._higher_order_ops.utils import first_slice_copy

        yield args[0], (*args[1], *(first_slice_copy(a) for a in args[2]), *args[3])
    elif node.target in (
        torch.ops.higher_order.while_loop,
        torch.ops.higher_order.while_loop_stack_output,
    ):
        subgraph_args = (*args[2], *args[3])
        yield args[0], subgraph_args
        yield args[1], subgraph_args
    elif node.target is torch.ops.higher_order.switch:
        # args: (index, [branch_gm_0, ...], operands)
        subgraph_args = tuple(args[2])
        for branch_gm in args[1]:
            yield branch_gm, subgraph_args
    elif node.target is control_deps:
        if kwargs:
            raise AssertionError(
                "Subgraph arguments can be renamed, so we cannot consistently "
                "handle kwargs at this point in the stack."
            )
        control_deps_subgraph = args[1]
        control_deps_args = tuple(args[2:])
        if control_deps_subgraph in valid_subgraphs:
            yield control_deps_subgraph, control_deps_args
        if isinstance(
            control_deps_subgraph, torch.fx.GraphModule
        ) and pytree.tree_any_only(
            torch.fx.GraphModule,
            lambda subgraph: subgraph in valid_subgraphs,
            control_deps_args,
        ):
            placeholder_to_arg = dict(
                zip(
                    control_deps_subgraph.graph.find_nodes(op="placeholder"),
                    control_deps_args,
                    strict=True,
                )
            )
            wrapped_nodes = [
                n for n in control_deps_subgraph.graph.nodes if n.op == "call_function"
            ]
            if len(wrapped_nodes) != 1:
                raise AssertionError(
                    f"expected exactly 1 wrapped call_function node, got {len(wrapped_nodes)}"
                )
            wrapped_node = wrapped_nodes[0]

            def replace_placeholder(item: Any) -> Any:
                if isinstance(item, torch.fx.Node):
                    return placeholder_to_arg.get(item, item)
                return item

            wrapped_args, wrapped_kwargs = pytree.tree_map(
                replace_placeholder,
                (wrapped_node.args, wrapped_node.kwargs),
            )
            yield from _extract_subgraphs_and_args(
                wrapped_node, valid_subgraphs, *wrapped_args, **wrapped_kwargs
            )
    elif node.target is torch.ops.higher_order.flex_gemm:
        yield args[1], tuple(args[2])
    else:
        warnings.warn(
            f"Please add support for subgraph args to function {node.target}!"
        )

        # By default, just return the detected list of subgraphs so that we can run
        # updates on all of them.
        flat_args_kwargs, _ = pytree.tree_flatten((args, kwargs))
        yield from (
            (s, None)
            for s in flat_args_kwargs
            if isinstance(s, torch.fx.GraphModule) and s in valid_subgraphs
        )


@dataclass(frozen=True)
class _FxNodeHash:
    node: torch.fx.Node
    target: torch.fx.node.Target
    arg_hashes: tuple[int, ...]
    kwarg_hashes: tuple[int, ...]


class FakeTensorUpdater:
    """
    The main idea here is that it's difficult to maintain accurate fake
    tensors (our primary form of metadata) for each node in our graph as we
    transform it.

    The most reliable way to obtain this information is by rerunning
    faketensor propagation. However, in general, faketensor propagation is
    fairly expensive. So, instead we'd like to only rerun faketensor
    propagation on nodes that have changed.

    In order to detect which nodes have changed, we first hash its node,
    target, and argument lists (which are immutable in FX).

    Then, whenever we call incremental_update, we check which FX nodes have a
    new hash, and recompute the faketensor metadata for that node. Then, we
    continue to recursively compute the faketensors for all users until the
    fake tensors stop changing.

    Since this runs in the context of Inductor, we assume that the input and
    output semantics for the outermost graph are not subject to change after class
    initialization, but we allow striding changes for subgraphs.  Any other changes will
    result in errors or undefined behavior.
    """

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.processed_hashes = OrderedSet[_FxNodeHash]()
        self.tracked_node_ids: OrderedSet[int] = OrderedSet()
        self.gm = gm

        # Import here to avoid circular import issues.
        from torch._inductor.compile_fx import _get_subgraph_names

        self.subgraph_updaters: dict[torch.fx.GraphModule, FakeTensorUpdater] = {
            (subgraph := getattr(self.gm, subgraph_name)): FakeTensorUpdater(subgraph)
            for subgraph_name in _get_subgraph_names(self.gm)
        }

        for node in self.gm.graph.nodes:
            self.processed_hashes.add(self.hash_node(node))
            self.tracked_node_ids.add(id(node))

    def hash_node(self, node: torch.fx.Node) -> _FxNodeHash:
        def get_hash_or_ids(n_iter: Iterable[Any]) -> tuple[int, ...]:
            """Replace unhashable items from the input with the ids of those items, and
            hash all other items.  This is kludgy, but allows us to attempt to account
            for unhashable classes like torch.fx.Node when hashing args and kwargs for
            other nodes."""

            def get_hash_or_id(o: object) -> int:
                try:
                    return hash(o)
                except Exception:
                    return id(o)

            return tuple(get_hash_or_id(n) for n in n_iter)

        return _FxNodeHash(
            node,
            node.target,
            get_hash_or_ids(node.args),
            get_hash_or_ids(chain.from_iterable(node.kwargs.items())),
        )

    def incremental_update(self) -> int:
        """Update FakeTensors on self.graph. We will try to do the minimum amount of work.

        Returns the number of nodes updated, including recursive updates on subgraphs."""
        existing_storages: defaultdict[int, int] = defaultdict(int)
        for node in self.gm.graph.nodes:
            if storage := get_node_storage(node):
                existing_storages[storage] += 1

        def should_process_node(node: torch.fx.Node) -> bool:
            return (
                callable(node.target)
                # Dirty Inductor lowerings are handled by the guard below. Clean
                # lowering nodes are skipped because their targets expect IR inputs.
                and not hasattr(node.target, "_inductor_lowering_function")
            )

        def node_invokes_subgraph(
            node: torch.fx.Node, *args: Any, **kwargs: Any
        ) -> bool:
            return (
                node.op == "call_function"
                and node.target
                not in (
                    # auto_functionalized doesn't call a subgraph, but the pytree call
                    # below can return subgraphs if the functionalized call itself
                    # invokes a subgraph.
                    torch.ops.higher_order.auto_functionalized,
                    torch.ops.higher_order.auto_functionalized_v2,
                )
                and pytree.tree_any_only(
                    torch.fx.GraphModule,
                    lambda s: s in self.subgraph_updaters,
                    (args, kwargs),
                )
            )

        def update_node_fake_tensor(node: torch.fx.Node, new_fake_tensor: Any) -> bool:
            if "val" in node.meta and _is_fake_tensor_same(
                new_fake_tensor, node.meta["val"], existing_storages, node=node
            ):
                return False

            storage_only_change = "val" in node.meta and _is_fake_tensor_same(
                new_fake_tensor,
                node.meta["val"],
                existing_storages,
                check_storage=False,
                node=node,
            )

            rebind_unbacked(V.fake_mode.shape_env, node, new_fake_tensor)

            node.meta["val"] = new_fake_tensor

            if (shape_env := V.fake_mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, new_fake_tensor)
            ):
                # Refresh the bindings to the new symbols

                node.meta["unbacked_bindings"] = symbol_to_path

            if storage := get_node_storage(node):
                existing_storages[storage] += 1

            to_process.update(
                id(user)
                for user in node.users
                if not (
                    storage_only_change
                    and getattr(
                        user.target,
                        "_inductor_lowering_output_metadata_ignores_input_storage",
                        False,
                    )
                    and getattr(
                        user.target,
                        "_inductor_lowering_output_metadata_is_input",
                        None,
                    )
                    is None
                )
            )

            return True

        def get_lowering_input_metadata(
            node: torch.fx.Node, input_name: int | str
        ) -> Any:
            raw_input = (
                node.args[input_name]
                if isinstance(input_name, int)
                else node.kwargs[input_name]
            )

            def has_missing_fake_metadata(input_node: torch.fx.Node) -> bool:
                return not (
                    "val" in input_node.meta
                    or "example_value" in input_node.meta
                    or (
                        input_node.op == "get_attr"
                        and isinstance(input_node.target, str)
                        and hasattr(self.gm, input_node.target)
                    )
                )

            if any(
                isinstance(leaf, torch.fx.Node) and has_missing_fake_metadata(leaf)
                for leaf in pytree.tree_leaves(raw_input)
            ):
                raise RuntimeError(
                    "FakeTensorUpdater cannot update pass-through "
                    "_inductor_lowering_function metadata because the input "
                    f"metadata is unavailable. Encountered node: {node.format_node()}"
                )
            return tree_map(partial(get_fake, gm=self.gm), raw_input)

        # Update self.processed_hashes every time.  This allows us to account for
        # situations where a node gets modified, updated, then reverted to its original
        # state.  Without doing this, we wouldn't update downstream nodes, because the
        # reverted node would already be in our cache.
        current_graph_hashes = OrderedSet[_FxNodeHash]()

        nodes_updated: int = 0
        to_process = OrderedSet[int]()
        # Value records whether subgraph outputs have been updated.
        subgraph_updatings: dict[torch.fx.GraphModule, bool] = {}
        for node in self.gm.graph.nodes:
            current_graph_hashes.add(node_hash := self.hash_node(node))

            # Lowering functions consume Inductor IR nodes, not FakeTensors, so
            # FakeTensorUpdater cannot safely recompute their metadata.
            if hasattr(node.target, "_inductor_lowering_function") and (
                node_hash not in self.processed_hashes or id(node) in to_process
            ):
                if metadata_fn := getattr(
                    node.target,
                    "_inductor_lowering_output_metadata_fn",
                    None,
                ):
                    is_valid, args, kwargs = get_fake_args_kwargs(node, self.gm)
                    if not is_valid:
                        raise RuntimeError(
                            "FakeTensorUpdater cannot update "
                            "_inductor_lowering_function metadata because the "
                            "metadata provider inputs are unavailable. "
                            f"Encountered node: {node.format_node()}"
                        )
                    with V.fake_mode, enable_python_dispatcher():
                        new_fake_tensor = metadata_fn(*args, **kwargs)
                    if update_node_fake_tensor(node, new_fake_tensor):
                        nodes_updated += 1
                    self.tracked_node_ids.add(id(node))
                    continue
                if (
                    metadata_input := getattr(
                        node.target,
                        "_inductor_lowering_output_metadata_is_input",
                        None,
                    )
                ) is not None:
                    if update_node_fake_tensor(
                        node, get_lowering_input_metadata(node, metadata_input)
                    ):
                        nodes_updated += 1
                    self.tracked_node_ids.add(id(node))
                    continue
                if id(node) in to_process:
                    raise RuntimeError(
                        "FakeTensorUpdater cannot recompute metadata for "
                        "_inductor_lowering_function nodes after their dependencies "
                        "change because those targets expect Inductor IR inputs rather "
                        "than FakeTensor inputs. "
                        f"Encountered node with changed dependency: {node.format_node()}"
                    )
                if id(node) in self.tracked_node_ids:
                    raise RuntimeError(
                        "FakeTensorUpdater cannot recompute metadata for tracked "
                        "_inductor_lowering_function nodes after their inputs or "
                        "dependencies change because those targets expect "
                        "Inductor IR inputs rather than FakeTensor inputs. "
                        f"Encountered changed node: {node.format_node()}"
                    )
                if "val" not in node.meta:
                    raise RuntimeError(
                        "FakeTensorUpdater requires newly inserted "
                        "_inductor_lowering_function nodes to already carry fake "
                        "metadata in node.meta['val']. "
                        f"Encountered new node without metadata: {node.format_node()}"
                    )
                self.tracked_node_ids.add(id(node))
                continue

            is_valid, args, kwargs = get_fake_args_kwargs(node, self.gm)
            if not is_valid:
                continue

            # NB: Be very careful about skipping nodes (via continues) here
            # and ask for a careful review when changing this code. The
            # consequence for incorrect FakeTensor metadata is difficult-to-debug
            # silent incorrectness.
            if (
                # Always run updates on nodes that invoke subgraphs
                not (invokes_subgraph := node_invokes_subgraph(node, *args, **kwargs))
                and node_hash in self.processed_hashes
                and id(node) not in to_process
            ):
                continue

            if not should_process_node(node):
                continue

            if invokes_subgraph:
                any_output_updated = False

                for subgraph, subgraph_args in _extract_subgraphs_and_args(
                    node, self.subgraph_updaters, *args, **kwargs
                ):
                    update_subgraph = subgraph not in subgraph_updatings
                    if update_subgraph:
                        subgraph_updatings[subgraph] = False

                    # If the arguments being passed into the subgraph differ from the
                    # existing args the first time we see the subgraph, update the
                    # placeholder nodes in the subgraph.  Any updates past that point
                    # would make subgraph updates inconsistent, so throw errors.
                    if subgraph_args is not None:
                        for p, a in zip(
                            subgraph.graph.find_nodes(op="placeholder"),
                            subgraph_args,
                            strict=True,
                        ):
                            # Do an update iff anything except the storage has changed,
                            # since every invocation of the subgraph will use different
                            # storages.  This implies potentially incorrect arg
                            # aliasing relationships.
                            if not _is_fake_tensor_same(
                                a,
                                p_fake := get_fake(p, subgraph),
                                existing_storages,
                                check_storage=False,
                            ):
                                if not update_subgraph:
                                    raise AssertionError(
                                        "subgraph args must have consistent values!"
                                    )
                                # Check that only the stride has changed.  Other changes
                                # cannot be handled without manual intervention.
                                if not _is_fake_tensor_same(
                                    a,
                                    p_fake,
                                    existing_storages,
                                    check_strides=False,
                                    check_storage=False,
                                ):
                                    raise AssertionError(
                                        "A subgraph argument other than striding has been "
                                        "modified; FakeTensorUpdater cannot update this "
                                        "argument!"
                                    )

                                p.meta["val"] = a
                                nodes_updated += 1

                    if update_subgraph:
                        _, orig_output_args, _ = get_fake_args_kwargs(
                            subgraph.graph.output_node(), subgraph
                        )
                        nodes_updated += self.subgraph_updaters[
                            subgraph
                        ].incremental_update()
                        _, new_output_args, _ = get_fake_args_kwargs(
                            subgraph.graph.output_node(), subgraph
                        )

                        if not _is_fake_tensor_same(
                            new_output_args,
                            orig_output_args,
                            existing_storages,
                        ):
                            subgraph_updatings[subgraph] = True

                    any_output_updated = (
                        any_output_updated or subgraph_updatings[subgraph]
                    )

                # If the outputs of the subgraphs have not changed (and have been
                # previously cached), we can skip updating.  If the outputs of the
                # subgraph have changed, we'll run FakeTensor propagation for every
                # invocation of the subgraph, so that each node gets unique FakeTensor
                # values which maintain appropriate aliasing relationships.
                if not any_output_updated and "val" in node.meta:
                    continue

            with V.fake_mode, enable_python_dispatcher():
                new_fake_tensor = node.target(*args, **kwargs)

            if update_node_fake_tensor(node, new_fake_tensor):
                nodes_updated += 1

        self.processed_hashes = current_graph_hashes
        self.tracked_node_ids = OrderedSet(id(node) for node in self.gm.graph.nodes)

        return nodes_updated


def get_storage(t: torch.Tensor) -> int:
    return t.untyped_storage()._cdata


def get_node_storage(node: torch.fx.Node) -> int | None:
    if "val" not in node.meta:
        return None
    if not isinstance(node.meta["val"], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta["val"]):
        return None
    return get_storage(node.meta["val"])


def get_fake(x: Any, gm: torch.fx.GraphModule | None) -> Any:
    """Return a fake tensor from the meta values of an input FX node.  If the input node
    is a get_attr node, we attempt to resolve it as a member of gm."""
    if isinstance(x, torch.fx.Node):
        if "val" in x.meta:
            return x.meta["val"]
        if "example_value" in x.meta:
            return x.meta["example_value"]
        if x.op == "get_attr" and isinstance(x.target, str) and hasattr(gm, x.target):
            return getattr(gm, x.target)
        # If a node has no available fake values and isn't a get_attr, it either returns
        # None or is incorrectly configured.  Since there are real nodes which return
        # None, we can't error here.
        return None
    # If there are no example values, return x
    return x


def get_fake_args_kwargs(
    x: torch.fx.Node, gm: torch.fx.GraphModule | None = None
) -> tuple[bool, tuple[Any, ...], dict[str, Any]]:
    """
    First value returns a boolean if any of the input nodes don't have a faketensor and
    weren't resolved from gm.
    """
    args, kwargs = tree_map(partial(get_fake, gm=gm), (x.args, x.kwargs))
    if any(
        isinstance(a, torch.fx.Node) for a in pytree.arg_tree_leaves(*args, **kwargs)
    ):
        return False, args, kwargs
    return True, args, kwargs


def is_node_realized(node: torch.fx.Node) -> bool:
    """Returns true if a node is always realized when lowered to inductor IR.

    NOTE: This may return some false negatives. e.g. it doesn't
    handle buffers realized heuristically during lowering, or
    buffers realized indirectly through view ops.
    """
    from torch._inductor.lowering import fallbacks, needs_realized_inputs

    def is_buffer(node: torch.fx.Node) -> bool:
        if node.op == "call_function" and node.target is operator.getitem:
            # For nodes with multiple outputs, we get the fx graph:
            #     foo = torch.ops.aten.foo(...)
            #     getitem = foo[0]
            #     getitem_1 = foo[1]
            # where we need to check if foo is a fallback kernel
            return is_buffer(node.args[0])  # type: ignore[arg-type]
        return node.op in ("placeholder", "output") or node.target in fallbacks

    if is_buffer(node):
        return True

    def realizes_inputs(node: torch.fx.Node) -> bool:
        return node.op == "output" or node.target in needs_realized_inputs

    if any(realizes_inputs(user) for user in node.users):
        return True

    # Otherwise, assume node isn't realized
    return False


def count_flops_fx(node: torch.fx.Node) -> int | None:
    if not countable_fx(node) or isinstance(node.target, str):
        return None
    with FakeTensorMode(allow_non_fake_inputs=True):
        success, args, kwargs = get_fake_args_kwargs(node)

        if success:
            # flex_attention HOPs have registered formulas, but invoking them
            # here can require tracing-only context, e.g. TransformGetItemToIndex.
            if node.target in (
                torch.ops.higher_order.flex_attention,
                torch.ops.higher_order.flex_attention_backward,
            ):
                flop_formula = flop_registry.get(node.target)
                if flop_formula is not None:
                    return flop_formula(*args, **kwargs, out_val=node.meta.get("val"))

            with torch.utils.flop_counter.FlopCounterMode(
                display=False
            ) as flop_counter_mode:
                node.target(*args, **kwargs)

            counted_flops = flop_counter_mode.get_total_flops()
            return counted_flops
    return None


def countable_fx(node: torch.fx.Node) -> bool:
    """
    Whether or not we can count the flops of an FX node.
    """
    if not isinstance(node, torch.fx.Node):
        raise AssertionError(f"expected torch.fx.Node, got {type(node)}")
    if not hasattr(node, "target"):
        return False
    target = node.target
    if not hasattr(target, "overloadpacket"):
        return target in flop_registry
    packet = target.overloadpacket
    return packet in flop_registry
<|endoftext|>from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, TYPE_CHECKING
from unittest.mock import patch

import sympy

import torch._logging
import torch.distributed as dist
import torch.fx
from torch.utils._ordered_set import OrderedSet

from . import config, select_algorithm
from .ir import (
    Buffer,
    ChoiceCaller,
    Layout,
    MultiTemplateBuffer,
    OperationBuffer,
    StorageBox,
    TensorBox,
)
from .kernel_inputs import KernelInputs, MMKernelInputs
from .scheduler import SchedulerNode
from .virtualized import NullHandler, V


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


_DISTRIBUTED_AUTOTUNE_KEY = "distributed_autotune"

_AUTOTUNE_PG: dist.ProcessGroup | None = None


@dataclasses.dataclass
class _DistributedAutotuneState:
    """
    State used to track autotuning during a graph_context()
    """

    # This is the next operator index. Used to figure out which rank should do
    # the autotuning.
    autotuned_index: int = 0

    # For debugging - used to make sure that we autotune the same number of
    # local operators that we expected to.
    autotuned_local_count: int = 0


@dataclasses.dataclass
class _DistributedAutotuneInfo:
    index: int
    local: bool


def get_autotune_pg() -> dist.ProcessGroup | None:
    if dist.is_available() and dist.is_initialized():
        global _AUTOTUNE_PG
        if _AUTOTUNE_PG is None:
            autotune_pg = dist.distributed_c10d._new_group_with_tag(
                pg_tag="pt2_distributed_autotune_pg"
            )
            if autotune_pg == dist.GroupMember.NON_GROUP_MEMBER:
                raise AssertionError("Autotune process group must include all ranks")
            _AUTOTUNE_PG = autotune_pg
        return _AUTOTUNE_PG

    return None


def schedule(scheduler: torch._inductor.scheduler.Scheduler) -> None:
    """
    Finish the distributed autotuning by propagating the autotuning results
    between the ranks and then replacing the placeholder with the real Buffer.
    """
    if not config.distributed_max_autotune_gemm:
        raise AssertionError("distributed_max_autotune_gemm config must be enabled")
    autotune_results = _autotune_local_nodes(scheduler)
    choices_by_index = _sync(autotune_results)
    _autotune_remote_nodes(scheduler, choices_by_index)


@contextlib.contextmanager
def graph_context() -> Generator[None, None, None]:
    """
    Wrapped around processing a graph, sets up figuring out which ranks tune
    which shapes.
    """
    if isinstance(
        V.get_distributed_autotune_state(check_poisoned=False),  # type: ignore[call-arg]
        _DistributedAutotuneState,
    ):
        raise AssertionError(
            "distributed autotune state is already active, nested graph_context is not allowed"
        )
    V.set_distributed_autotune_state(_DistributedAutotuneState())
    try:
        yield
    finally:
        V.set_distributed_autotune_state(NullHandler())


def maybe_autotune_remote(
    name: str, choices: list[ChoiceCaller], inputs: list[Buffer], layout: Layout
) -> TensorBox | None:
    """
    Used by an op (like `mm`) to determine if the op should be autotuned
    locally (returns None) or remotely (returns a placeholder Buffer).
    """
    if not config.distributed_max_autotune_gemm:
        return None

    if not (autotune_pg := get_autotune_pg()):
        return None

    if len(choices) <= 1:
        return None

    state = V.distributed_autotune_state
    index = state.autotuned_index
    state.autotuned_index += 1
    local = index % autotune_pg.size() == autotune_pg.rank()

    V.current_node.meta[_DISTRIBUTED_AUTOTUNE_KEY] = _DistributedAutotuneInfo(
        index, local
    )
    if local:
        state.autotuned_local_count += 1
        return None

    return torch._inductor.ir.TensorBox.create(
        _DistributedAutotuneBuffer(name, inputs, layout)
    )


class _DistributedAutotuneBuffer(MultiTemplateBuffer):
    """
    A MultiTemplateBuffer which represents a kernel being autotuned on a
    different rank. When `schedule` is called this will be replaced by the
    "real" buffer.
    """

    # Name of the kernel being autotuned.
    _kernel_name: str

    def __init__(
        self,
        kernel_name: str,
        inputs: list[Buffer],
        layout: Layout,
    ) -> None:
        super().__init__(
            layout,
            inputs,
            choice_timings_fn=self._dummy_choice_timings,
            unfiltered_choices=[],
            allowed_prologue_inps=OrderedSet({}),
        )

        self._kernel_name = kernel_name

    def _dummy_choice_timings(
        self, _hint_override: int | None
    ) -> dict[ChoiceCaller, float]:
        # This should never get called. It means that a remote autotune was
        # scheduled but never filled in.
        raise NotImplementedError

    def autotune(self, ser_choice: _SerializedChoice) -> TensorBox:
        """
        Given a _SerializedChoice (autotune results from another rank)
        compute the final TensorBox.
        """

        from .select_algorithm import autotune_select_algorithm

        with patch.object(V.graph, "scheduler", None):
            kernel_inputs = MMKernelInputs([*self.original_inputs])
            if not isinstance(self.layout, Layout):
                raise AssertionError(f"Expected Layout, got {type(self.layout)}")
            choice = ser_choice.get_choice(self.layout, kernel_inputs)
            buffer, _ = autotune_select_algorithm(
                self._kernel_name,
                [choice],
                kernel_inputs.nodes(),
                self.layout,
            )
            if not isinstance(buffer, TensorBox):
                raise AssertionError(f"Expected TensorBox, got {type(buffer)}")
            return buffer


# Can we make this async?
def _sync(autotune_results: list[_SerializedChoice]) -> Sequence[_SerializedChoice]:
    """
    Perform the all_gather to collect the autotune results from all the ranks.
    """

    autotune_pg = get_autotune_pg()
    if not autotune_pg:
        raise AssertionError("autotune process group is not initialized")

    # Perform allgather
    all_states: list[list[_SerializedChoice]] = [None] * autotune_pg.size()  # type: ignore[list-item]
    torch.distributed.all_gather_object(all_states, autotune_results, group=autotune_pg)

    node_count = sum(len(x) for x in all_states)
    # It's faster to briefly lie about the type than to unzip the results and append.
    choices_by_index: list[_SerializedChoice] = [None] * node_count  # type: ignore[list-item]

    check_count = 0
    for other_results in all_states:
        for choice in other_results:
            if not isinstance(choice, _SerializedChoice):
                raise AssertionError(f"Expected _SerializedChoice, got {type(choice)}")
            if choices_by_index[choice.index] is not None:
                raise AssertionError(f"duplicate choice for index {choice.index}")
            choices_by_index[choice.index] = choice
            check_count += 1

    if node_count != check_count:
        raise AssertionError(f"count mismatch: {node_count} != {check_count}")
    return choices_by_index


class _SerializedChoice:
    """
    This is a serializer for the autotune choice. KernelTemplateChoice can't
    be serialized directly (the template and inputs prevent this) so we need to
    serialize it by parts and reconstruct later on.
    """

    def __init__(self, index: int, choice: ChoiceCaller) -> None:
        self.index = index
        self.template_uid = _SerializedChoice._template_uid_from_choice(choice)
        self.kwargs = self._compute_kwargs(choice.description)

    def get_choice(self, layout: Layout, inputs: KernelInputs) -> ChoiceCaller | None:
        """
        Deserialize the ChoiceCaller and return it.
        """

        template = self._template_from_uid()

        kwargs = {**self.kwargs}
        if "BLOCK_K" in kwargs:
            # TODO: Do we really need to externally compute this value? If it's
            # needed I'm surprised it's not just part of the original template
            # description.
            # This needs the actual 'k' to figure out the value.
            k = inputs.nodes()[0].get_size()[1]
            kwargs["EVEN_K"] = sympy.gcd(k, kwargs["BLOCK_K"]) == kwargs["BLOCK_K"]

        extra_kwargs: dict[str, Any] = {}
        from .kernel_template_choice import (
            DictKernelTemplateParams,
            KernelTemplateChoice,
        )

        params = DictKernelTemplateParams(kwargs)
        ktc = KernelTemplateChoice(template, params, extra_kwargs, layout, inputs)
        return ktc.choice

    @staticmethod
    def _compute_kwargs(description: str) -> dict[str, int | str | bool]:
        """
        Given a template description turn it into input kwargs.
        """
        if not description:
            return {}

        # TODO: It seems like it would be better if the template could provide
        # this directly instead of having to parse a string.
        kwargs: dict[str, int | str | bool] = {}
        for cfg in description.split(","):
            key, val = cfg.split("=", 1)
            key, val = key.strip(), val.strip()
            if val == "True":
                kwargs[key] = True
            elif val == "False":
                kwargs[key] = False
            elif val.isdigit():
                kwargs[key] = int(val)
            else:
                if not (val.startswith("'") and val.endswith("'")):
                    raise AssertionError(
                        f"expected string value to be single-quoted, got {val!r}"
                    )
                kwargs[key] = val[1:-1]
        return kwargs

    @staticmethod
    def _template_uid_from_choice(choice: ChoiceCaller) -> str:
        """
        Given a ChoiceCaller figure out which template represents it. This
        is reversed by _template_from_uid().
        """

        # We need a better way to do this - right now we need to add each
        # supported template directly.
        if isinstance(choice, select_algorithm.ExternKernelCaller):
            if choice.choice.name == "mm":
                return "torch._inductor.kernel.mm.aten_mm"
            else:
                raise RuntimeError(f"TODO: kernel {choice.choice.name!r}")
        elif isinstance(choice, select_algorithm.TritonTemplateCaller):
            return "torch._inductor.kernel.mm.mm_template"
        else:
            raise RuntimeError(f"TODO: {type(choice)}")

    def _template_from_uid(self) -> Any:
        """
        See _template_uid_from_choice().
        """
        parts = self.template_uid.split(".")
        obj = globals()[parts[0]]
        for k in parts[1:]:
            obj = getattr(obj, k)
        return obj


def _autotune_local_nodes(
    scheduler: torch._inductor.scheduler.Scheduler,
) -> list[_SerializedChoice]:
    """
    Go through the nodes in the scheduler and autotune the kernels which
    should be autotuned by this rank.
    """

    autotune_results: list[_SerializedChoice] = []

    for node in scheduler.nodes:
        if not isinstance(node, SchedulerNode):
            continue

        if (inner_node := node.node) is None:
            continue

        if isinstance(inner_node, _DistributedAutotuneBuffer):
            # This is marked for remote autotuning.
            continue

        if not isinstance(inner_node, MultiTemplateBuffer):
            continue

        if (origin_node := inner_node.origin_node) is None:
            continue

        if (meta := origin_node.meta) is None:
            continue

        info = meta.get(_DISTRIBUTED_AUTOTUNE_KEY)
        if info is None:
            continue

        if not info.local:
            raise AssertionError(
                "expected info.local to be True for locally autotuned node"
            )

        # We force autotuning here
        # Still takes advantage of async precompile
        # We need all the configs before fusion
        min_choice, _ = inner_node.get_min_choice()

        choice = _SerializedChoice(info.index, min_choice)
        autotune_results.append(choice)

    state = V.distributed_autotune_state
    if len(autotune_results) != state.autotuned_local_count:
        raise AssertionError(
            f"incorrect local autotuned nodes found ({len(autotune_results)} != {state.autotuned_local_count})"
        )
    return autotune_results


def _autotune_remote_nodes(
    scheduler: torch._inductor.scheduler.Scheduler,
    choices_by_index: Sequence[_SerializedChoice],
) -> None:
    """
    Go through the nodes in the scheduler and autotune the nodes that were
    autotuned on remote ranks.
    """

    for i, node in enumerate(scheduler.nodes):
        if isinstance(node, SchedulerNode) and isinstance(
            (dist_node := node.node), _DistributedAutotuneBuffer
        ):
            if dist_node.origin_node is None:
                raise AssertionError("dist_node.origin_node must not be None")
            info = dist_node.origin_node.meta[_DISTRIBUTED_AUTOTUNE_KEY]
            out_tensorbox = dist_node.autotune(choices_by_index[info.index])

            out_storage = out_tensorbox.data
            if not isinstance(out_storage, StorageBox):
                raise AssertionError(f"Expected StorageBox, got {type(out_storage)}")
            out_buffer = out_storage.data
            if not isinstance(out_buffer, OperationBuffer):
                raise AssertionError(
                    f"Expected OperationBuffer, got {type(out_buffer)}"
                )

            if out_buffer.layout != dist_node.layout:
                raise AssertionError(
                    f"layout mismatch: {out_buffer.layout} != {dist_node.layout}"
                )

            scheduler._replace_node(out_buffer, dist_node, i, node)
<|endoftext|>from __future__ import annotations

import dataclasses
import typing
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._inductor.scheduler import MixOrderReduction
from torch.utils._sympy.value_ranges import bound_sympy

from . import config
from .codecache import write_text
from .heuristics.template import get_template_heuristic
from .heuristics.template.triton import (
    _origami_enabled,
    BaseConfigHeuristic,
    CPUConfigHeuristic,
    CUDAConfigHeuristic,
    IS_ROCM,
    MTIAConfigHeuristic,
    ROCmConfigHeuristic,
    XPUConfigHeuristic,
)
from .kernel_inputs import KernelInputs, MMKernelInputs
from .kernel_template_choice import make_ktc_generator
from .metrics import get_metric_table, is_metric_table_enabled
from .runtime.hints import DeviceProperties, ReductionHint
from .scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse
from .select_algorithm import ExternKernelChoice
from .utils import _use_autotune_backend
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Generator
    from functools import partial

    from triton import Config as TritonConfig

    from .codegen.common import KernelTemplate
    from .codegen.simd_kernel_features import SIMDKernelFeatures
    from .codegen.triton import TritonKernel
    from .ir import ChoiceCaller
    from .kernel_template_choice import KernelTemplateChoice

    from torch.utils._ordered_set import OrderedSet  # isort: skip


if TYPE_CHECKING or not config.is_fbcode():

    def _maybe_log_inductor_mm_shape(
        kernel_inputs: KernelInputs,
        op_name: str,
        context_fn: Any = None,
    ) -> None:
        return

else:
    from torch._inductor.fb.shape_logging import (
        log_inductor_mm_shape as _maybe_log_inductor_mm_shape,
    )


class Sortable(typing.Protocol):
    """Anything that can be used as a list.sort() key (int/tuple/etc)"""

    def __lt__(self, other: typing.Self) -> bool: ...


@dataclasses.dataclass
class FusionScore:
    template_score: int
    node_type_score: int
    memory_score: int
    buffer_overlap_score: int
    proximity_score: int

    def __lt__(self, other):
        """
        node_type_score orders same-kind fusions above mixed-kind fusions unless
        the memory_score differs too much. Nested reduction candidates use -1
        so they rank below ordinary mixed-kind fusions; mix-order reductions
        are still scored through memory_score.

        buffer_overlap_score is prioritized below memory_score so that
        strict global memory savings (exact dep matches) are preferred
        over buffer overlap scoring (same buffer, different indexing).
        """
        threshold = 16
        if self.template_score != other.template_score:
            return self.template_score < other.template_score

        if (
            max(self.memory_score, other.memory_score)
            > min(self.memory_score, other.memory_score) * threshold
        ):
            return self.memory_score < other.memory_score

        return (
            self.node_type_score,
            self.memory_score,
            self.buffer_overlap_score,
            self.proximity_score,
        ) < (
            other.node_type_score,
            other.memory_score,
            other.buffer_overlap_score,
            other.proximity_score,
        )


class InductorChoices:
    """
    This class contains a collection of default heuristics that affect performance of our generated
    code.  We try to not put correctness requirements in this file.

    You can override the choices made here by doing:

            class MyHeuristics(InductorChoices):
                ...

            torch._inductor.virtualized.V.set_choices_handler(MyHeuristics())

    Subclasses used with inductor_choices_class must implement uuid() for
    cache key computation.
    """

    def get_config_heuristics(
        self, device_type: str | None = "cuda"
    ) -> BaseConfigHeuristic:
        if device_type == "cuda":
            if torch.version.hip is None:
                return CUDAConfigHeuristic()
            else:
                return ROCmConfigHeuristic()
        elif device_type == "xpu":
            return XPUConfigHeuristic()
        elif device_type == "cpu":
            return CPUConfigHeuristic()
        elif device_type == "mtia":
            return MTIAConfigHeuristic()
        else:
            return BaseConfigHeuristic()

    # Conv configs
    def get_conv_configs(
        self, device_type: str | None = "cuda"
    ) -> partial[Generator[TritonConfig, None, None]]:
        conv_heuristics = self.get_config_heuristics(device_type)
        return conv_heuristics.get_conv_configs()

    def get_depthwise_conv_configs(self, device_type: str | None = "cuda") -> list[Any]:
        heuristics = self.get_config_heuristics(device_type)
        return heuristics.get_depthwise_conv_configs()

    # Flex attention configs
    # TODO(coconutruben): break out flexattention/decode configs into the new retrieval mechanism
    def get_flex_attention_fwd_configs(
        self,
        head_dim: int,
        seq_len: sympy.Expr,
        dtype: torch.dtype,
        device_type: str | None = "cuda",
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_attn_fwd_configs(head_dim, seq_len, dtype)

    def get_flex_attention_bwd_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: str | None = "cuda"
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_attn_bwd_configs(head_dim, dtype)

    def get_flex_decode_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: str | None = "cuda"
    ) -> list[Any]:
        flex_heuristics = self.get_config_heuristics(device_type)
        return flex_heuristics.get_flex_decode_configs(head_dim, dtype)

    def append_flex_attention_choices(
        self,
        choices: list[Any],
        configs: list[Any],
        input_nodes: list[Any],
        subgraphs: list[Any],
        layout: Any,
        kernel_options: dict[str, Any],
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
    ) -> list[Any]:
        """Append backend-specific flex-attention template choices.

        Default is a no-op. Subclasses may override to inject additional
        autotuning candidates (e.g. TLX templates in fbcode).
        """
        return choices

    def _logging_context(self) -> dict[str, Any]:
        """Extra fields for the per-shape log row. Subclasses may override."""
        return {}

    def _finalize_template_configs(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: list[KernelTemplate | ExternKernelChoice],
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> list[KernelTemplateChoice]:
        """
        This method can be subclassed to perform any override/modification of the choices.
        The incoming parameters are cheap (generators), so you can do any overrides without
        incurring too much cost. Override this method to customize the kernel template choices
        before they are converted to ChoiceCaller objects, which is expensive on template codegen.

        The full list of arguments are here to facilitate any overrides you may want to do,
        as they can be used to start from scratch for each template if so desired.

        Args:
            template_choices: Dictionary mapping template UIDs to generators of KernelTemplateChoice objects
            kernel_inputs: MMKernelInputs containing input tensor nodes and matrix indices
            templates: List of template objects (KernelTemplate or ExternKernelChoice) in use
