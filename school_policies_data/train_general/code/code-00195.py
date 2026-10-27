    in sync; ``test_cached_and_inlined_paths_agree`` cross-checks them.
    """
    if len(mods) > 1:

        def _name(mi: _ModuleIndex, n: str) -> str:
            return f"m{mi}.{n}"
    else:

        def _name(mi: _ModuleIndex, n: str) -> str:
            return n

    unique: list[Tensor] = []
    id_to_uidx: dict[int, int] = {}
    alias_entries: list[tuple[_ModuleIndex, str, int]] = []

    def _intern(mi: _ModuleIndex, n: str, t: Tensor, names_out: list[str]) -> None:
        uidx = id_to_uidx.get(id(t))
        if uidx is None:
            uidx = len(unique)
            id_to_uidx[id(t)] = uidx
            unique.append(t)
            names_out.append(_name(mi, n))
        alias_entries.append((mi, n, uidx))

    param_names: list[str] = []
    for mi, m in enumerate(mods):
        for n, p in m.named_parameters(remove_duplicate=False):
            _intern(_ModuleIndex(mi), n, p, param_names)
    num_params = len(unique)
    buffer_names: list[str] = []
    for mi, m in enumerate(mods):
        for n, b in m.named_buffers(remove_duplicate=False):
            _intern(_ModuleIndex(mi), n, b, buffer_names)
    return unique, param_names, buffer_names, alias_entries, num_params


def _capture(
    fn: Callable[..., object],
    args: tuple[object, ...],
    decompositions: dict | None = None,
) -> _Capture:
    """Trace the computation ``fn(*args)`` to an ATen graph.

    See Note [precompile programming model] for the contract. ``fn`` is the whole
    computation, e.g. ``lambda model, x: model(x)`` or a training step
    ``lambda model, x, t: loss_fn(model(x), t).backward()``. Among ``args``, the
    ``nn.Module`` arguments have their parameters/buffers lifted to explicit graph
    inputs (via reparametrization, so nothing is baked -- invariant 1); the
    remaining arguments are the runtime inputs. Whatever ``fn`` returns becomes the
    graph's result outputs, and if ``fn`` ran a backward, the resulting parameter
    gradients (read off ``param.grad``) are harvested as additional, trailing graph
    outputs. They are kept separate from the result so the driver can scatter them
    onto the runtime model's ``.grad`` fields rather than return them (invariant 5).

    This is a NON-STRICT trace (invariant 3): make_fx records only the ATen ops
    that run for THIS example. Python-level control flow over tensor values, data-
    dependent branches, and shapes are specialized to ``args`` and baked. The
    interning/order established here for params then buffers is the calling
    convention the runtime model must reproduce (invariant 2).
    """
    import contextlib

    args = tuple(args)
    module_positions = [i for i, a in enumerate(args) if isinstance(a, torch.nn.Module)]
    module_pos_set = set(module_positions)
    mods = [a for a in args if isinstance(a, torch.nn.Module)]
    user_inputs = tuple(a for i, a in enumerate(args) if i not in module_pos_set)

    # Lift the example modules' params/buffers for tracing only. Their VALUES are
    # never stored in the cache -- the user passes the model(s) again at runtime
    # (mirroring fn's signature), and the same interning is reproduced there.
    pb_flat, param_names, buffer_names, alias_entries, num_params = (
        _intern_param_buffers(mods)
    )
    num_pb = len(pb_flat)
    # Record each interned param's / buffer's example SHAPE, DTYPE, and DEVICE (aligned to
    # param_names / buffer_names) so the structural check (invariant 2) compares not just
    # names but also each runtime tensor's shape, dtype, and device. The graph is specialized
    # to the example param/buffer shapes (and can bake a device literal via a factory op), so
    # a same-named runtime tensor with a different shape / dtype / device would otherwise
    # silently compute the wrong thing (eager has no assert_size_stride backstop).
    param_shapes = [tuple(t.shape) for t in pb_flat[:num_params]]
    buffer_shapes = [tuple(t.shape) for t in pb_flat[num_params:]]
    param_dtypes = [str(t.dtype) for t in pb_flat[:num_params]]
    buffer_dtypes = [str(t.dtype) for t in pb_flat[num_params:]]
    param_devices = [str(t.device) for t in pb_flat[:num_params]]
    buffer_devices = [str(t.device) for t in pb_flat[num_params:]]

    user_flat, in_spec = pytree.tree_flatten(user_inputs)
    # Reject mark options precompile cannot honor (mark_dynamic, specialize_on) loudly
    # here, before tracing, rather than silently dropping them. (hint_override is honored,
    # not rejected -- it is a perf-only autotuning hint threaded onto the capture symbol.)
    _reject_unsupported_marks(user_flat)
    flat_args = [*pb_flat, *user_flat]
    # The REAL example tensors (params/buffers and user inputs). flat_args is reassigned
    # to FAKE tensors in the unbacked path below, but the saved-grad snapshot/clear/restore
    # block must protect the real example model's .grad fields (those are what the user
    # owns and what a backward in fn populates), not the throwaway fakes. list() snapshots
    # the real tensors here, so the later flat_args rebind does not affect real_flat.
    real_flat = list(flat_args)
    # Record the example user inputs' dense shapes/dtypes/devices so the drivers can
    # reject a shape (invariant 3) or dtype/device (invariant 6) mismatch up front; see
    # the inlined driver checks (torch._precompile_driver). Stride is NOT recorded --
    # memory-format mismatches are enforced by inductor's own (pinned-on)
    # assert_size_stride. Subclasses -> None.
    # Widened element type (a marked-dynamic dim becomes None within the tuple in the
    # unbacked path below); _dense_shape's static tuples conform to it.
    user_input_shapes: list[tuple[int | None, ...] | None] = [
        _dense_shape(t) for t in user_flat
    ]
    user_input_dtypes = [_dense_dtype(t) for t in user_flat]
    user_input_devices = [_dense_device(t) for t in user_flat]

    # Dynamic shapes (opt-in, UNBACKED only): dims the caller tagged with
    # torch._dynamo.decorators.mark_unbacked are refakeified as unbacked symints, then
    # traced symbolically with the fake_mode's ShapeEnv threaded to the lowering. Unbacked
    # dims cannot be guarded on, so the artifact is valid across runtime sizes; a graph
    # that would need to guard on a marked dim fails loudly at capture
    # (GuardOnDataDependentSymNode) rather than baking it. Reading the marks here (instead
    # of a precompile kwarg) keeps the precompile signature simple.
    marks = _read_unbacked_marks(user_flat)
    # Record each marked dim's declared min/max so the driver enforces them at runtime;
    # the capture-time torch._check on an unbacked symint never becomes a runtime guard,
    # so without this the documented mark_unbacked min/max check would be a silent no-op.
    user_input_bounds = _read_input_bounds(marks)
    # Snapshot and clear the REAL example tensors' .grad BEFORE fakeifying and tracing.
    # A backward in fn accumulates (``p.grad = p.grad + new``), so a live pre-existing
    # grad would be read into the graph and baked by make_fx as a get_attr constant --
    # tripping the invariant-1 guard with a misleading "tensor closed over by fn" error on
    # the common warmup-step-then-precompile flow. The clear MUST precede
    # _fakeify_with_unbacked: fake_mode.from_tensor copies .grad onto the fakes we trace
    # on, so clearing the reals first keeps the fakes grad-free too. Restored in finally;
    # precompile does not mutate the user's example .grad (params/buffers AND user inputs).
    # Snapshot the ORIGINAL .grad object (no clone) and restore that SAME object below, so
    # grad IDENTITY is preserved -- a caller holding a prior p.grad reference, or optimizer
    # state keyed on grad identity, is not invalidated. The unbacked path traces on fakes,
    # so the reals' .grad is untouched there; the STATIC path (fake_mode is None) traces on
    # the real interned params, so a backward in fn DOES write .grad in place -- but onto a
    # fresh grad object, since .grad was snapshotted and cleared to None just above. The
    # finally-restore below puts the snapshotted object back, so both grad identity and
    # value are preserved regardless of which path ran.
    saved_grads = [a.grad if isinstance(a, torch.Tensor) else None for a in real_flat]
    for a in real_flat:
        if isinstance(a, torch.Tensor):
            a.grad = None
    fake_mode = None
    if any(marks):
        flat_args, fake_mode = _fakeify_with_unbacked(pb_flat, user_flat, marks)
        user_input_shapes = [
            None
            if base is None
            else tuple(None if i in per else s for i, s in enumerate(base))
            for base, per in zip(user_input_shapes, marks)
        ]

    # flat_fn (traced by make_fx) writes these back so _capture can thread the output
    # structure and the harvested-grad param indices into the _Capture result.
    captured_out_spec: pytree.TreeSpec | None = None
    captured_grad_param_indices: list[int] = []

    def flat_fn(flat: list[object]) -> list[object]:
        nonlocal captured_out_spec, captured_grad_param_indices
        # The pb region is entirely interned params/buffers (Tensors); the user region
        # (flat[num_pb:]) is arbitrary pytree leaves.
        pb = cast("list[Tensor]", flat[:num_pb])
        runtime_inputs = pytree.tree_unflatten(flat[num_pb:], in_spec)
        with contextlib.ExitStack() as stack:
            for mi, m in enumerate(mods):
                reparam = {n: pb[uidx] for emi, n, uidx in alias_entries if emi == mi}
                stack.enter_context(
                    stateless._reparametrize_module(m, reparam, tie_weights=True)
                )
            # Reconstruct fn's full positional args: reparametrized modules at
            # their original positions, runtime inputs at theirs.
            full: list[object] = []
            ui = 0
            for i in range(len(args)):
                if i in module_pos_set:
                    full.append(args[i])
                else:
                    full.append(runtime_inputs[ui])
                    ui += 1
            result = fn(*full)
            # Harvest parameter gradients produced by any backward in fn.
            param_proxies = pb[:num_params]
            harvested = [p.grad for p in param_proxies]
            # Buffers are not harvested (only params get scattered grads). A registered
            # buffer with requires_grad=True that received a gradient would be silently
            # dropped, so reject it -- a cheaply-knowable invariant-5 violation.
            if any(getattr(b, "grad", None) is not None for b in pb[num_params:]):
                raise PrecompileError(
                    "precompile: a registered buffer received a gradient (it has "
                    "requires_grad=True), but precompile only harvests gradients for "
                    "parameters. Register it as an nn.Parameter instead."
                )
            # User-input leaves are not harvested either (only params get scattered
            # grads), so a requires_grad user input that received a gradient during the
            # traced backward would be silently dropped. Reject it, mirroring the buffer
            # case -- another cheaply-knowable invariant-5 violation.
            if any(getattr(t, "grad", None) is not None for t in flat[num_pb:]):
                raise PrecompileError(
                    "precompile: a user input received a gradient; precompile only "
                    "harvests gradients for parameters, so an input gradient would be "
                    "silently dropped. Pass the tensor as a module parameter if its "
                    "gradient is needed."
                )

        # The result (fn's own return) and the harvested grads are kept as separate
        # output regions: the driver returns the result and scatters the grads onto
        # the runtime model's .grad fields. We emit a grad output ONLY for params that
        # actually received a gradient -- mirroring eager .backward(), which leaves
        # .grad = None for frozen / non-contributing params -- and record which unique
        # param index each emitted grad belongs to, so the driver scatters onto exactly
        # those params. grad_flat is empty when fn ran no backward.
        result_flat, result_spec = pytree.tree_flatten(result)
        grad_flat = []
        grad_param_indices = []
        for i, g in enumerate(harvested):
            if g is not None:
                grad_flat.append(g)
                grad_param_indices.append(i)
        captured_out_spec = result_spec
        captured_grad_param_indices = grad_param_indices
        return [*result_flat, *grad_flat]

    # Trace with grad enabled so any backward in ``fn`` is built as graph ops; the
    # forward graph is the same as under no_grad. Restore in finally so a make_fx
    # failure (e.g. fn raising after running a backward) does not leave the user's
    # example model with clobbered .grad fields.
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    tracing_mode = "symbolic" if fake_mode is not None else "real"
    capture_cm = fake_mode if fake_mode is not None else contextlib.nullcontext()
    try:
        with torch.enable_grad(), capture_cm:
            try:
                gm = make_fx(
                    flat_fn,
                    decomposition_table=decompositions,
                    tracing_mode=tracing_mode,
                )(flat_args)
            except GuardOnDataDependentSymNode as e:
                # A mark_unbacked dim was captured as an unbacked symint (no hint), but
                # the computation needs to guard on / specialize its size (e.g. a
                # shape-dependent branch or a reshape that pins it). Unbacked dims cannot
                # be guarded, so rather than bake a silently-wrong artifact, fail here.
                raise PrecompileError(
                    "precompile: fn needs to guard on a dim marked with mark_unbacked "
                    "(it branches on or specializes that size), which is not allowed for "
                    "an unbacked dynamic dim. Do not mark that dim (capture it static), "
                    "or restructure fn to avoid the size-dependent operation. Underlying: "
                    f"{str(e).splitlines()[0]}"
                ) from e
    finally:
        for a, g in zip(real_flat, saved_grads):
            if isinstance(a, torch.Tensor):
                a.grad = g
    _check_no_constant_tensors(gm)
    _assert_no_control_flow_subgraphs(gm)
    _assert_supported(gm)

    # flat_fn always runs during the make_fx trace above, so captured_out_spec is set.
    return _Capture(
        gm=gm,
        flat_args=flat_args,
        module_positions=module_positions,
        num_positional_args=len(args),
        param_names=param_names,
        buffer_names=buffer_names,
        param_shapes=param_shapes,
        buffer_shapes=buffer_shapes,
        param_dtypes=param_dtypes,
        buffer_dtypes=buffer_dtypes,
        param_devices=param_devices,
        buffer_devices=buffer_devices,
        in_spec=in_spec,
        out_spec=cast("pytree.TreeSpec", captured_out_spec),
        grad_param_indices=captured_grad_param_indices,
        user_input_shapes=user_input_shapes,
        user_input_dtypes=user_input_dtypes,
        user_input_devices=user_input_devices,
        user_input_bounds=user_input_bounds,
        fake_mode=fake_mode,
    )


class _Capture:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        flat_args: list[object],
        module_positions: list[int],
        num_positional_args: int,
        param_names: list[str],
        buffer_names: list[str],
        param_shapes: list[tuple[int, ...]],
        buffer_shapes: list[tuple[int, ...]],
        param_dtypes: list[str],
        buffer_dtypes: list[str],
        param_devices: list[str],
        buffer_devices: list[str],
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
        grad_param_indices: list[int],
        user_input_shapes: list[tuple[int | None, ...] | None],
        user_input_dtypes: list[str | None],
        user_input_devices: list[str | None],
        user_input_bounds: list[_LeafBounds],
        fake_mode: FakeTensorMode | None = None,
    ) -> None:
        self.gm = gm
        self.flat_args = flat_args
        self.module_positions = module_positions
        self.num_positional_args = num_positional_args
        self.param_names = param_names
        self.buffer_names = buffer_names
        self.param_shapes = param_shapes
        self.buffer_shapes = buffer_shapes
        self.param_dtypes = param_dtypes
        self.buffer_dtypes = buffer_dtypes
        self.param_devices = param_devices
        self.buffer_devices = buffer_devices
        self.in_spec = in_spec
        self.out_spec = out_spec
        self.grad_param_indices = grad_param_indices
        self.user_input_shapes = user_input_shapes
        self.user_input_dtypes = user_input_dtypes
        self.user_input_devices = user_input_devices
        self.user_input_bounds = user_input_bounds
        # The fake_mode (with ShapeEnv) used for a dynamic-shape capture, threaded to the
        # lowering (dynamic_shapes="from_tracing_context"); None for a static capture.
        self.fake_mode = fake_mode


_GENERATED_HEADER = """\
# Generated by torch.compiler.precompile -- do not edit.
#
# This is a SELF-CONTAINED, EXECUTABLE artifact: it runs on its own, needing no
# companion cache. You provide the model(s) at runtime, exactly as the original fn
# took them, e.g.:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](model, my_input)      # same args as the traced fn
#
# The runtime model must be STRUCTURALLY IDENTICAL to the one precompile traced
# (same parameter/buffer names, order, and weight tying); only the weight VALUES
# may differ (swap in a checkpoint). This artifact was produced by a non-strict
# make_fx trace, so control flow and shapes are specialized to the example inputs,
# and (inductor backend) each input's stride / memory format is baked too: pass
# runtime inputs in the example's layout (.contiguous() to match a contiguous
# example). See Note [precompile programming model] in torch/_precompile.py.
#
# It contains, in order:
#   1. The composed graph module from aot_autograd.compile_to_python: the inlined
#      Inductor kernels (JIT-compiled from the embedded source on first use -- no
#      external cache required) plus AOTAutograd's own codegen'd prelude/epilogue
#      (tensor-subclass wrap/unwrap, input-mutation reflection, output aliasing),
#      exposing ``call(flat_inputs) -> outputs``.
#   2. Calling-convention metadata.
#   3. A small driver that extracts each runtime module's params/buffers (in the
#      same order as capture), passes them with the runtime inputs to ``call``, and
#      scatters any harvested gradients onto the model's .grad fields. No model
#      weights are embedded (you bring the model).
#
# The companion ``cache`` returned by precompile is purely an ACCELERATION used by
# torch.compiler.precompile.load: it primes the inductor kernel caches so exec'ing this
# file loads its kernels from the precompiled binaries (no JIT). This file does not read
# it; running this file alone just JITs.
"""


def _build_metadata_section(compiled: PrecompiledModule) -> list[str]:
    if compiled._out_spec is None or compiled._in_spec is None:
        raise PrecompileError("internal: cannot build metadata before _compile()")
    # OUT_SPEC is load-bearing: the driver rebuilds fn's output via tree_unflatten, so
    # unlike IN_SPEC it cannot degrade to None. If fn's output structure is not
    # JSON-serializable (an unregistered namedtuple, or a registered pytree node with a
    # non-JSON-dumpable context), fail with a clear PrecompileError rather than leaking
    # a raw pytree NotImplementedError/TypeError.
    try:
        out_spec_str = pytree.treespec_dumps(compiled._out_spec)
    except (NotImplementedError, TypeError) as e:
        raise PrecompileError(
            "precompile cannot serialize the output structure of fn (its pytree "
            "TreeSpec is not JSON-serializable). This fires when fn returns an "
            "unregistered collections.namedtuple, or a registered pytree node with a "
            "non-JSON-dumpable context. Register the namedtuple via "
            "torch.utils._pytree._register_namedtuple(...) (or supply a JSON-dumpable "
            "to_dumpable_context), or return a plain tuple/list/dict of tensors."
        ) from e
    # IN_SPEC drives the runtime input-structure check, but is best-effort: some specs
    # are not JSON-serializable -- an unregistered namedtuple raises NotImplementedError,
    # and a registered pytree node whose context is not JSON-dumpable (no
    # to_dumpable_context serializer, or one yielding non-JSON output) raises TypeError.
    # Such inputs still compile -- emit IN_SPEC = None and the driver skips the
    # structure check rather than regressing.
    try:
        in_spec_str: str | None = pytree.treespec_dumps(compiled._in_spec)
    except (NotImplementedError, TypeError):
        in_spec_str = None
    parts = [
        "# " + "=" * 70,
        "# 2. Calling-convention metadata",
        "# " + "=" * 70,
        "import torch as _torch",
        "import torch.utils._pytree as _pytree",
        "",
        # python_code is the single source of truth for the calling convention; the
        # cache holds ONLY the compiled/captured artifact. load() reads these
        # constants back out of python_code (see _parse_artifact_metadata).
        f"BACKEND = {compiled._backend!r}",
        f"MODULE_POSITIONS = {compiled._module_positions!r}",
        # Number of positional args the traced fn took (modules + runtime inputs); the
        # driver checks the runtime call passes the same count up front, so a wrong
        # arity raises a clear PrecompileError instead of a raw IndexError.
        f"NUM_POSITIONAL_ARGS = {compiled._num_positional_args}",
        f"PARAM_NAMES = {compiled._param_names!r}",
        f"BUFFER_NAMES = {compiled._buffer_names!r}",
        # Per interned param / buffer example shape / dtype / device (aligned to
        # PARAM_NAMES / BUFFER_NAMES); the driver checks each runtime param/buffer against
        # these for the structural contract (invariant 2).
        f"PARAM_SHAPES = {compiled._param_shapes!r}",
        f"BUFFER_SHAPES = {compiled._buffer_shapes!r}",
        f"PARAM_DTYPES = {compiled._param_dtypes!r}",
        f"BUFFER_DTYPES = {compiled._buffer_dtypes!r}",
        f"PARAM_DEVICES = {compiled._param_devices!r}",
        f"BUFFER_DEVICES = {compiled._buffer_devices!r}",
        # Which unique-param index each trailing grad output belongs to (see invariant 5);
        # the driver scatters grad k onto params[GRAD_PARAM_INDICES[k]].
        f"GRAD_PARAM_INDICES = {compiled._grad_param_indices!r}",
        # The pytree structure of the runtime inputs, or None if not serializable (the
        # driver validates against it when present, else skips the structure check).
        f"IN_SPEC = {in_spec_str!r}",
        f"OUT_SPEC = {out_spec_str!r}",
        # Per user-input-leaf example shape / dtype / device (None for a non-tensor /
        # subclass leaf); the drivers reject a runtime mismatch (invariants 3 and 6).
        # Memory-format mismatches are caught by the inductor artifact's own
        # assert_size_stride (pinned on at capture).
        f"USER_INPUT_SHAPES = {compiled._user_input_shapes!r}",
        f"USER_INPUT_DTYPES = {compiled._user_input_dtypes!r}",
        f"USER_INPUT_DEVICES = {compiled._user_input_devices!r}",
        # Per user-input-leaf mark_unbacked min/max bounds: None for a leaf with no bounded
        # marked dim, else {dim: (lo, hi)} (either may be None). The drivers reject a
        # runtime size outside the declared range (invariant 3); see the inlined drivers.
        f"USER_INPUT_BOUNDS = {compiled._user_input_bounds!r}",
        "",
    ]
    return parts


def _parse_artifact_metadata(python_code: str) -> dict[str, object]:
    """Read the calling-convention constants back out of ``python_code`` WITHOUT
    executing it (exec'ing the inlined Inductor output would JIT the kernels, the
    very work the cache exists to skip).

    python_code is the single source of truth: ``_build_metadata_section`` emits the
    constants below as top-level literal assignments, so an AST walk + literal_eval
    recovers them safely. The cache then only needs to carry the compiled artifact.
    """
    import ast

    wanted = {
        "BACKEND",
        "MODULE_POSITIONS",
        "NUM_POSITIONAL_ARGS",
        "PARAM_NAMES",
        "BUFFER_NAMES",
        "PARAM_SHAPES",
        "BUFFER_SHAPES",
        "PARAM_DTYPES",
        "BUFFER_DTYPES",
        "PARAM_DEVICES",
        "BUFFER_DEVICES",
        "GRAD_PARAM_INDICES",
        "IN_SPEC",
        "OUT_SPEC",
        "USER_INPUT_SHAPES",
        "USER_INPUT_DTYPES",
        "USER_INPUT_DEVICES",
        "USER_INPUT_BOUNDS",
    }
    found: dict[str, object] = {}
    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        raise PrecompileError(
            "python_code is not valid Python; it does not look like a "
            "torch.compiler.precompile artifact."
        ) from e
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in wanted:
            found[target.id] = ast.literal_eval(node.value)
        else:
            # Not a metadata name we consume (the driver section emits only
            # function defs today, but a future artifact revision could add a
            # driver-internal top-level assignment). Skipped by design, but log
            # it at debug so a malformed / renamed artifact is diagnosable
            # rather than silently dropped.
            log.debug(
                "precompile: ignoring unrecognized top-level assignment %r while "
                "parsing artifact calling-convention metadata",
                target.id,
            )
    missing = wanted - found.keys()
    if missing:
        raise PrecompileError(
            f"python_code is missing calling-convention metadata {sorted(missing)}; "
            "it does not look like a torch.compiler.precompile artifact."
        )
    return found


def _build_python_source(
    compiled: PrecompiledModule,
    graph_python: str,
) -> str:
    parts = [_GENERATED_HEADER, ""]
    parts.append("# " + "=" * 70)
    parts.append("# 1. Compiled graph (AOTAutograd + Inductor): exposes ``call``")
    parts.append("# " + "=" * 70)
    # The composed graph module from aot_autograd.compile_to_python: the inlined
    # Inductor kernels plus AOTAutograd's codegen'd prelude/epilogue, exposing
    # ``call(flat_inputs) -> outputs`` (subclass + mutation handled inside).
    parts.append(graph_python)
    parts.append("")
    parts.extend(_build_metadata_section(compiled))
    parts.append("# " + "=" * 70)
    parts.append(
        "# 3. Driver: module params/buffers + grad scatter + calling convention"
    )
    parts.append("# " + "=" * 70)
    parts.append(_emit_driver_source("_inductor_forward"))
    return "\n".join(parts)


_EAGER_GENERATED_HEADER = """\
# Generated by torch.compiler.precompile (backend="eager") -- do not edit.
#
# Self-contained, executable artifact: the captured ATen graph is inlined below (both
# the human-readable rendering and the executable code) and runs on its own. Provide
# the model(s) at runtime, exactly as the original fn took them:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](model, my_input)      # same args as the traced fn
#
# The runtime model must be structurally identical to the traced one (only weight
# VALUES may differ), and control flow / shapes are specialized to the example inputs.
# See Note [precompile programming model] in torch/_precompile.py for the full contract.
"""


def _build_eager_python_source(compiled: PrecompiledModule) -> str:
    gm = compiled._gm
    # gm.code defines ``def forward(self, flat)`` that references fx_pytree / pytree
    # and self._in_spec / self._out_spec. Rename it so it does not collide with the
    # driver's public ``forward``, and supply the specs via a tiny holder object so
    # the inlined graph runs standalone.
    in_spec = gm._in_spec if gm is not None else None
    out_spec = gm._out_spec if gm is not None else None
    if gm is None or in_spec is None or out_spec is None:
        raise PrecompileError("internal: eager graph missing before _compile()")
    graph_src = gm.code.replace("def forward(", "def _graph_forward(", 1)
    in_spec_str = pytree.treespec_dumps(in_spec)
    out_spec_str = pytree.treespec_dumps(out_spec)
    parts = [_EAGER_GENERATED_HEADER, ""]
    parts.append("# " + "=" * 70)
    parts.append("# 1. Captured ATen graph (eager backend) -- executable and readable")
    parts.append("# " + "=" * 70)
    # gm.code relies on fx's custom builtins (torch, device, inf, nan, NoneType,
    # fx_pytree, pytree) being in scope -- fx injects them when a real GraphModule
    # runs. Reproduce the FULL set (not just torch/pytree) so a graph that bakes a
    # device / inf / nan constant (e.g. BatchNorm, masked_fill to -inf) runs
    # standalone instead of raising NameError. Sourced from fx so it stays correct.
    from torch.fx.graph import _custom_builtins

    for _cb in _custom_builtins.values():
        parts.append(_cb.import_str)
    parts.append(graph_src)
    parts.append("")
    parts.append("class _GraphSelf:")
    parts.append(f"    _in_spec = pytree.treespec_loads({in_spec_str!r})")
    parts.append(f"    _out_spec = pytree.treespec_loads({out_spec_str!r})")
    parts.append("")
    parts.append("")
    parts.append("def call(args):")
    parts.append("    out = _graph_forward(_GraphSelf(), list(args))")
    parts.append("    return list(out) if isinstance(out, (list, tuple)) else [out]")
    parts.append("")
    parts.extend(_build_metadata_section(compiled))
    parts.append("# " + "=" * 70)
    parts.append("# 3. Driver: run the inlined captured graph eagerly")
    parts.append("# " + "=" * 70)
    parts.append(_emit_driver_source("_eager_forward"))
    return "\n".join(parts)


_DRIVER_MAIN = """\
if __name__ == "__main__":
    print("forward() is ready; call it with the model(s) and inputs the traced")
    print("fn took, e.g. forward(model, x).")
"""


def _emit_driver_source(forward_fn_name: str) -> str:
    """Emit the runtime driver as text for inlining into python_code.

    The driver lives as real, type-checked code in torch._precompile_driver; here we read
    it back with inspect.getsource (LAZILY -- only on this emit path; load() never runs
    it, so a stripped-source environment only affects capture, not reload) and rename the
    selected forward variant to the public ``forward``. Emitting the TEXT (rather than
    importing the module from the artifact) keeps python_code self-contained and
    version-frozen (Note [precompile programming model], invariant 7)."""
    import inspect

    from torch import _precompile_driver as driver

    forward_fn = getattr(driver, forward_fn_name)
    blocks = [
        inspect.getsource(driver._extract_param_buffers),
        inspect.getsource(driver._fail),
        inspect.getsource(driver._check_structure),
        inspect.getsource(forward_fn).replace(
            f"def {forward_fn_name}(", "def forward(", 1
        ),
    ]
    body = "\n\n".join(block.rstrip() for block in blocks)
    return "\n" + body + "\n\n\n" + _DRIVER_MAIN


def _assert_supported(gm: torch.fx.GraphModule) -> None:
    """Enforce invariant 4 of Note [precompile programming model]: reject boundary
    effects the AOT backend's standalone composition does not handle. Detected
    directly from the captured graph -- no AOTAutograd coupling.

    Input mutation (incl. module buffers, e.g. BatchNorm running stats), tensor-
    subclass wrap/unwrap, output aliasing, and functionalized RNG are SUPPORTED:
    AOTAutograd's codegen'd prelude/epilogue is composed into the artifact (see
    torch._functorch.aot_autograd.compile_to_python), so they are not rejected here.

    Effectful ops are not supported yet (an implementation gap, not a fundamental
    limit), so raise here with a concrete reason rather than let the failure surface
    deep in the cache layer. See _unsupported for the mechanical cause.
    """
    from torch._higher_order_ops.effects import _get_effect

    for node in gm.graph.nodes:
        # Only ATen ops can be in the effect registry; skip plain call_functions
        # like operator.getitem (which _get_effect rejects).
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if _get_effect(node.target) is not None:
                raise _unsupported(f"effectful op {node.target}")


def _unsupported(reason: str) -> PrecompileError:
    return PrecompileError(
        f"precompile cannot compile this computation: {reason}. The graph contains an "
        "effectful op, which is not supported yet: its with_effects HOP is "
        "non-cacheable, so the compiled artifact cannot be saved and lowered to "
        "standalone source."
    )


class PrecompiledModule:
    """Internal holder for a precompiled computation / a loaded runnable."""

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        backend: str = "inductor",
        tracer: str = "make_fx",
        decompositions: dict | None = None,
    ) -> None:
        # ``fn`` is the whole computation: an nn.Module, or a callable that closes
        # over the module(s) it uses (e.g. ``lambda x: model(x)``, or a training
        # step that computes a loss and torch.autograd.grad).
        self._fn = fn
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._artifact: object = None
        self._module_positions: list[int] = []
        self._num_positional_args: int = 0
        # Interned param / buffer names and their example shape, dtype, and device
        # (aligned lists); the driver checks each runtime param/buffer against these for
        # the structural contract (invariant 2). Populated by _compile().
        self._param_names: list[str] = []
        self._buffer_names: list[str] = []
        self._param_shapes: list[tuple[int, ...]] = []
        self._buffer_shapes: list[tuple[int, ...]] = []
        self._param_dtypes: list[str] = []
        self._buffer_dtypes: list[str] = []
        self._param_devices: list[str] = []
        self._buffer_devices: list[str] = []
        self._in_spec: pytree.TreeSpec | None = None
        self._out_spec: pytree.TreeSpec | None = None
        self._gm: torch.fx.GraphModule | None = None
        # Inductor backend: the composed self-contained graph module (from
        # aot_autograd.compile_to_python, exposing ``call(flat_inputs)``) and the
        # opaque artifact-cache bytes (None if uncacheable), populated by _compile().
        self._graph_python: str = ""
        self._artifact_bytes: bytes | None = None
        # Which unique-param index each emitted (trailing) grad output belongs to; its
        # length is the number of grad outputs. Lets the driver scatter grads onto
        # exactly the params that received one, leaving frozen / non-contributing
        # params' .grad as None.
        self._grad_param_indices: list[int] = []
        # Per user-input-leaf example shape, dtype, and device (None for a subclass /
        # non-tensor leaf; a marked-dynamic dim is None within the shape tuple); the drivers
        # reject a runtime mismatch (invariants 3 and 6). Stride / memory format is enforced
        # by the inductor artifact's own assert_size_stride, not recorded here. Populated by
        # _compile().
        self._user_input_shapes: list[tuple[int | None, ...] | None] = []
        self._user_input_dtypes: list[str | None] = []
        self._user_input_devices: list[str | None] = []
        # Per user-input-leaf mark_unbacked min/max bounds (None for a leaf with no
        # bounded marked dim, else {dim: (lo, hi)}). The drivers reject a runtime size
        # outside the declared range (invariant 3). Populated by _compile().
        self._user_input_bounds: list[Any] = []
        # Set only on the load() path, where we wrap a reconstructed callable.
        self._loaded_forward: Callable[..., object] | None = None

    @classmethod
    def _from_loaded(
        cls,
        forward: Callable[..., object],
        *,
        backend: str,
    ) -> PrecompiledModule:
        """Build a runnable from load()'s reconstructed forward.

        load() does not re-run capture/_compile, so reuse ``__init__`` for all the
        defaults (the single definition of this object's state) and override only the
        reconstructed forward. All the calling-convention metadata lives in the inlined
        driver (``forward``) itself, so the __init__ fields (``_fn``, ``_gm``,
        ``_module_positions``, ``_out_spec``, ...) stay at their defaults; inspect the
        artifact via python_code.
        """
        obj = cls(None, backend=backend)  # type: ignore[arg-type]
        obj._loaded_forward = forward
        return obj

    def _compile(self, args: tuple[object, ...]) -> None:
        # make_fx is the only implemented tracer; "dynamo" is a planned alternative
        # capture front-end. Reject it here (the single capture-dispatch point) before
        # running fn, so the failure is clear rather than a wrong default.
        if self._tracer != "make_fx":
            raise NotImplementedError(
                f"precompile tracer={self._tracer!r} is not implemented yet; use "
                "tracer='make_fx' (the default)."
            )
        if self._backend == "eager" and _has_unbacked_marks(args):
            raise NotImplementedError(
                "precompile: mark_unbacked (dynamic shapes) is only supported with "
                "backend='inductor'; eager + unbacked is not supported."
            )
        capture = _capture(self._fn, args, self._decompositions)
        self._module_positions = capture.module_positions
        self._num_positional_args = capture.num_positional_args
        self._param_names = capture.param_names
        self._buffer_names = capture.buffer_names
        self._param_shapes = capture.param_shapes
        self._buffer_shapes = capture.buffer_shapes
        self._param_dtypes = capture.param_dtypes
        self._buffer_dtypes = capture.buffer_dtypes
        self._param_devices = capture.param_devices
        self._buffer_devices = capture.buffer_devices
        self._user_input_shapes = capture.user_input_shapes
        self._user_input_dtypes = capture.user_input_dtypes
        self._user_input_devices = capture.user_input_devices
        self._user_input_bounds = capture.user_input_bounds
        self._in_spec = capture.in_spec
        self._out_spec = capture.out_spec
        self._grad_param_indices = capture.grad_param_indices
        self._gm = capture.gm

        if self._backend == "eager":
            # No Inductor lowering: the captured ATen graph IS the artifact. It is
            # run directly on the (subclass-level) inputs, so there is no inductor
            # ``call`` to inline and no dense flatten/unflatten -- the graph runs
            # exactly as captured (see Note [precompile programming model]).
            return

        # Lower through the AOT backend contract: it returns a self-contained module
        # exposing ``call(flat_inputs) -> outputs`` (with AOTAutograd's own codegen'd
        # prelude/epilogue -- subclass wrap/unwrap, input-mutation reflection, output
        # aliasing -- composed in, not reimplemented here) plus an opaque cache (the
        # save_cache_artifacts bundle that primes the inductor cache on load, or None
        # for uncacheable graphs).
        import torch._inductor.config as _ind_config
        from torch._functorch import aot_autograd
        from torch._inductor.exc import InductorError
        from torch._inductor.standalone_compile import NoRunnableInductorModuleError

        # Pin size_asserts ON so the artifact ALWAYS bakes assert_size_stride for the
        # inputs the graph reads -- this enforces the input memory-format contract
        # (invariant 6) at runtime regardless of the user's ambient size_asserts config
        # (off would otherwise elide the asserts and silently read wrong strides). The
        # guard is conservative (see the inlined driver checks): an input the graph never
        # reads gets no assert and stays layout-flexible, but a read input is asserted on
        # the example layout even for layout-agnostic ops (matmul/addmm), since precompile
        # cannot recompile to specialize a new layout the way torch.compile would. A
        # dynamic (unbacked) capture additionally pins scalar_asserts so the make_fx
        # ShapeEnv's runtime range asserts survive into the artifact.
        #
        # These are inductor config keys, so they ride in as ``options`` (aot_autograd.
        # compile_to_python merges them into the inductor config.patch it wraps the
        # compile in) rather than being patched around the call. The AOT layer detects
        # dynamic (symbolic) shapes off the captured graph and threads the make_fx
        # ShapeEnv through automatically, so there is no dynamic_shapes knob to pass and
        # no manual TracingContext to install: a static capture specializes to the
        # example shapes, an unbacked capture keeps the symbols.
        options: dict[str, Any] = {"size_asserts": True}
        if capture.fake_mode is not None and hasattr(_ind_config, "scalar_asserts"):
            options["scalar_asserts"] = True
        try:
            self._graph_python, self._artifact_bytes = aot_autograd.compile_to_python(
                capture.gm, capture.flat_args, options=options
            )
        except NoRunnableInductorModuleError as e:
            # Inductor emits no runnable module for a graph with no compute to lower --
            # one that returns inputs or Python constants unchanged (e.g. ``lambda x: x``,
            # ``x.detach()``, ``return 7``, or a bare ``return None``). The eager backend
            # (above) handles these; surface a clear PrecompileError instead of the raw
            # lowering error.
            raise PrecompileError(
                "the inductor backend cannot lower a graph with no compute -- the traced "
                "fn returns its inputs or Python constants unchanged, producing no "
                "Inductor kernel. Return a computed tensor, or use backend='eager'."
            ) from e
        except InductorError as e:
            # Inductor codegen asserts on certain non-tensor Python values in the output
            # structure ("Unexpected output types: [<class 'float'>]" -- also complex,
            # str, ...); int/bool/None outputs lower fine, and the eager backend handles
            # them too. Surface a clear PrecompileError instead of the raw assertion.
            if "Unexpected output types" in str(e):
                raise PrecompileError(
                    "the inductor backend cannot lower a graph whose output mixes a "
                    "non-tensor Python value (e.g. float / complex / str) with computed "
                    "tensors (int / bool / None outputs are fine). Return only tensors, "
                    "or use backend='eager'."
                ) from e
            raise

    def __call__(self, *args: object) -> object:
        # A PrecompiledModule is runnable only after load(); precompile() itself
        # returns (python_code, cache) rather than a runnable.
        if self._loaded_forward is None:
            raise PrecompileError(
                "this object is not runnable; build one with "
                "torch.compiler.precompile.load(python_code, cache)."
            )
        return self._loaded_forward(*args)

    def to_python_code(self) -> str:
        """Return the self-contained, executable Python artifact as a string.

        It runs on its own, needing no cache (Note [precompile programming model],
        "self-contained"). For the inductor backend it embeds the composed graph
        module from aot_autograd.compile_to_python (kernels JIT-compile on first
        call; AOTAutograd's prelude/epilogue inlined), the calling-convention
        metadata, and a ``forward()`` that takes the same args the traced fn took
        (the model(s) plus runtime inputs). For the eager backend it embeds the
        captured ATen graph (both readable and executable) plus a driver that runs it
        eagerly. No weights are embedded.
        """
        if self._loaded_forward is not None:
            raise PrecompileError(
                "this object was produced by torch.compiler.precompile.load(); the "
                "python_code you passed in is the source artifact (load() does not "
                "re-capture, so there is no python_code to re-emit from this object)."
            )
        if self._backend == "eager":
            if self._gm is None:
                raise PrecompileError("internal: not compiled; call _compile() first")
            return _build_eager_python_source(self)
        if not self._graph_python:
            raise PrecompileError("internal: not compiled; call _compile() first")
        return _build_python_source(self, self._graph_python)

    def to_cache_bytes(self, python_code: str | None = None) -> bytes:
        """Return the binary cache as bytes -- an ACCELERATION, not required to run.

        ``python_code`` is the single source of truth for the calling convention, so the
        cache holds only the compiled artifact plus the integrity tag and code_hash. For
        the inductor backend that artifact is the ``save_cache_artifacts`` bundle (load
        primes the kernel caches with it, so a warm reload skips JIT); for the eager
        backend it is None. See Note [precompile programming model], invariant 7.

        ``python_code`` defaults to what ``to_python_code()`` would emit; ``__call__``
        threads in the exact string it already built so code_hash matches the bytes
        returned to the user and the metadata is not rebuilt.
        """
        # _artifact_bytes is the inductor cache bundle (None if uncacheable, and always
        # None for eager); the envelope is a plain str/int/bytes dict (weights_only-safe)
        # carrying the tag + code_hash that binds it to python_code (invariant 7).
        if self._loaded_forward is not None:
            raise PrecompileError(
                "this object was produced by torch.compiler.precompile.load(); the cache "
                "you passed in is the source artifact (load() does not re-capture, so "
                "there is no cache to re-emit from this object)."
            )
        if python_code is None:
            python_code = self.to_python_code()
        code_hash = hashlib.sha256(python_code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(
            {
                "format": _CACHE_FORMAT,
                "version": _CACHE_VERSION,
                "backend": self._backend,
                "code_hash": code_hash,
                "artifact": self._artifact_bytes,
            },
            buf,
        )
        return buf.getvalue()


def _make_inlined_forward(python_code: str) -> Callable[..., object]:
    """Fallback: execute the self-contained python string (JITs kernels).

    ``python_code`` needs no cache -- the kernels (inductor) or graph (eager) are
    inlined, so we just exec it and hand back its ``forward``. The returned
    ``forward`` takes the same args the traced fn took (model(s) plus runtime
    inputs)."""
    # python_code is untrusted EXECUTABLE input -- exec'ing it runs whatever it contains
    # (JIT-compiling inlined kernels or running the inlined graph). Warn per load (not
    # warning_once) before the exec so the inlined fallback is never silent about it.
    log.warning(
        "torch.compiler.precompile.load is about to EXEC python_code, which is untrusted "
        "executable input (it runs inlined kernels / graph code). Only exec python_code "
        "you produced or otherwise trust (Note [precompile programming model], "
        "invariant 7)."
    )
    module_ns: dict[str, object] = {"__name__": "_precompiled_artifact"}
    exec(compile(python_code, "<precompile>", "exec"), module_ns)
    return cast("Callable[..., object]", module_ns["forward"])


class _PrecompileApi:
    """Callable namespace implementing ``torch.compiler.precompile`` and ``.load``.

    A single instance is exposed as ``torch.compiler.precompile``; calling it precompiles a
    computation and ``torch.compiler.precompile.load`` reloads the resulting artifacts. It
    is a class (rather than a function with attached attributes) so the call, the
    loader, and the error type are explicit members.

    The contract for both ``__call__`` and ``load`` is Note [precompile programming
    model] in this module.
    """

    # Reported so test_public_bindings / introspection see this as ``torch.compiler``.
    __module__ = "torch.compiler"

    # The error type raised by precompile, reachable as
    # ``torch.compiler.precompile.PrecompileError``.
    PrecompileError = PrecompileError

    def __reduce__(self) -> str:
        # torch.compiler.precompile is a process-wide singleton; pickle/deepcopy must
        # round-trip to the SAME object (the instance carries no per-call state) rather
        # than fail to pickle a bound-method-bearing instance. Returning the qualified
        # name resolves back to this singleton on unpickle.
        return "precompile"

    def __repr__(self) -> str:
        return "torch.compiler.precompile"

    def __call__(
        self,
        fn: Callable[..., object],
        *example_inputs: object,
        backend: str = "inductor",
        tracer: str = "make_fx",
        decompositions: dict | None = None,
    ) -> tuple[str, bytes]:
        """Ahead-of-time precompile ``fn`` against ``example_inputs``.

        .. note::

            ``torch.compiler.precompile`` is NOT
            ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
            guard-serialization caching mode); it captures ``fn`` ahead of time and
            lowers it to a self-contained Python source artifact.

        With the default ``make_fx`` tracer this is a non-strict trace with an explicit
        contract; read Note [precompile programming model] before using it. The artifact
        faithfully reproduces ``fn`` only for callers that uphold that contract.

        THREADING: the inductor lowering step drives process-global compiler state
        and is serialized by an internal lock, so concurrent ``backend="inductor"``
        calls lower one at a time. The make_fx capture phase and the ``backend="eager"``
        path are NOT serialized.

        ``backend`` selects how the captured graph is realized:

        - ``"inductor"`` (default): lower the graph through
          ``torch._functorch.aot_autograd.compile_to_python`` (the full AOTAutograd +
          Inductor pipeline, composed into one self-contained module). ``python_code``
          is the inlined Inductor output with AOTAutograd's prelude/epilogue; the cache
          holds the save_cache_artifacts bundle that primes the inductor cache on load.
        - ``"eager"``: do NOT lower -- keep the captured ATen graph and run it as-is
          (analogous to ``torch.compile(backend="eager")``). ``python_code`` inlines
          the readable captured graph (both the inspectable rendering and the
          executable artifact); the eager cache carries no compiled artifact
          (artifact=None) but is still a full integrity-tagged envelope -- with no
          kernels there is nothing to accelerate, so ``load`` runs the inlined graph.
          Useful for
          inspecting/debugging exactly what was traced without an Inductor dependency.

        ``tracer`` selects the capture front-end:

        - ``"make_fx"`` (default): a NON-STRICT make_fx trace -- it records the ATen ops
          that actually run when ``fn`` executes once on the example inputs and does not
          analyze your Python, so control flow and shapes are specialized to the example
          (the source of the programming-model contract). The only tracer implemented
          today.
        - ``"dynamo"``: planned (a Dynamo-based front-end that analyzes the Python);
          raises ``NotImplementedError`` for now.

        ``decompositions`` is an optional decomposition table (a dict mapping each
        ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
        ``decomposition_table`` during capture, so you can control how ATen ops are
        broken down in the captured graph. Defaults to ``None`` (make_fx's default).

        Dynamic shapes are opt-in via ``torch._dynamo.decorators.mark_unbacked``
        (inductor backend only), NOT a precompile kwarg: mark dims on the inputs before
        calling, e.g. ``mark_unbacked(x, 0); precompile(fn, model, x)`` frees ``x``'s
        batch dim. Marked dims are captured as UNBACKED symints, which cannot be guarded
        on, so one artifact serves any runtime size of them (invariant 3); a graph that
        needs to guard on / specialize a marked dim fails at capture with a
        ``PrecompileError``. Dims sharing a ``shape_id`` reuse one symbol (equal by
        construction); ``min``/``max`` become runtime asserts. Other dims stay static.
        Dims that MUST be equal at runtime (e.g. two inputs combined by a broadcast that
        requires equal sizes, ``model(a) + model(b)``) MUST be given a SHARED ``shape_id``
        so a mismatch is rejected; marking two such dims INDEPENDENTLY currently bakes a
        SILENT equal-size assumption and a runtime mismatch does NOT raise the loud failure
        eager gives (invariant 3). This is a harvesting gap, not an inherent limit of the
        standalone artifact: the capture ShapeEnv DOES record the equality (as a deferred
        runtime assert, e.g. ``Eq(u0, u1)``), but precompile does not yet harvest/enforce
        those relational asserts in the driver -- only the decorator's declared min/max feed
        the runtime bound checks. A shared ``shape_id`` is the way to get the check today.

        Returns ``(python_code, cache)`` -- a self-contained, executable Python
        source string (the single source of truth for the calling convention) and a
        binary cache holding ONLY the backend artifact (NO metadata, NO weights).
        Reload a runnable with ``torch.compiler.precompile.load(python_code, cache)``.

        ``fn`` is the whole computation, e.g.::

            python_code, cache = torch.compiler.precompile(
                lambda model, x: model(x), model, x
            )


            def train_step(model, x, t):
                loss_fn(model(x), t).backward()  # or return autograd.grad(...)


            python_code, cache = torch.compiler.precompile(train_step, model, x, t)

        Among ``example_inputs``, the ``nn.Module`` arguments have their params/buffers
        lifted to graph inputs (no weights are baked into the artifact -- invariant 1);
        the rest are the runtime inputs. The reloaded callable is invoked with the SAME
        argument structure -- pass the model(s) again at runtime, e.g.
        ``f_c(model, x)``, and that runtime model must match the example model's
        parameter/buffer structure (invariant 2). Arguments are matched POSITIONALLY:
        pass the model(s) and inputs positionally both here and at load time; keyword-
        argument calling conventions are not supported (a fn that relies on them would
        surface as a raw arity error). If ``fn`` ran a backward, the
        resulting parameter gradients are scattered (accumulated) onto that runtime
        model's ``parameters()`` ``.grad`` fields, exactly like eager ``.backward()``,
        so a ``zero_grad()`` / ``optimizer.step()`` loop works unchanged; the artifact
        returns ``fn``'s own result (``None`` for a bare ``.backward()`` step), not the
        grads (invariant 5).

        Input mutation (incl. module buffers, e.g. BatchNorm running stats in
        training mode), tensor subclasses (e.g. DTensor), and outputs aliasing inputs
        are supported -- AOTAutograd's prelude/epilogue is composed into the artifact
        (invariant 4), as is functionalized RNG. Caller responsibilities NOT checked
        here (see the Note): the runtime model must be structurally identical to the
        example, and control flow / shapes are specialized to ``example_inputs``
        (invariants 2 and 3). Violations that ARE checked raise ``PrecompileError``: a
        tensor baked
        as a constant (invariant 1), effectful ops (invariant 4), and -- for the
        inductor backend -- a runtime input whose stride / memory format differs from
        the example's (invariant 6).
        """
        torch._C._log_api_usage_once("torch.compiler.precompile")
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        if tracer not in ("make_fx", "dynamo"):
            raise ValueError(
                f"precompile tracer must be 'make_fx' or 'dynamo', got {tracer!r}."
            )
        compiled = PrecompiledModule(
            fn, backend=backend, tracer=tracer, decompositions=decompositions
        )
        compiled._compile(example_inputs)
        # Build the (expensive) python_code ONCE and thread it into to_cache_bytes so
        # the full metadata + embedded kernel source is not rebuilt, and so code_hash is
        # sha256 over exactly the bytes returned to the caller (a matched pair loads).
        python_code = compiled.to_python_code()
        return python_code, compiled.to_cache_bytes(python_code)

    def load(self, python_code: str, cache: bytes) -> Callable[..., object]:
        """Reconstruct a runnable from ``(python_code, cache)`` from precompile.

        The driver runs from ``python_code`` -- the single source of truth for the whole
        calling convention. ``load`` reads the cache's ``BACKEND`` (to check the pairing)
        and, for the inductor backend, primes the inductor kernel caches from its
        ``save_cache_artifacts`` bundle (via ``torch.compiler.load_cache_artifacts``) so a
        warm reload loads precompiled kernels instead of JIT-compiling; then it exec's
        ``python_code``. With no usable cache it degrades to JIT'ing from ``python_code``.

        Call the result with the SAME argument structure ``fn`` took -- the
        model(s) in their original positions plus the runtime inputs. Per invariant
        2 of Note [precompile programming model], the runtime model must match the
        example model's parameter/buffer structure; precompile re-derives the
        param/buffer list from it (same interning/order as capture).

        Raises ``PrecompileError`` if ``python_code`` is malformed or is not a
        ``torch.compiler.precompile`` artifact (it fails to parse, or is missing the
        calling-convention metadata), if the cache's ``backend`` tag does not match
        ``python_code``, or if the cache's ``code_hash`` does not match
        ``sha256(python_code)`` -- i.e. the cache and python_code came from different
        ``precompile()`` calls. A cache whose ``format``/``version`` does not match (a
        foreign or different-build envelope) is NOT fatal: the cache is acceleration
        only, so ``load`` degrades to JIT'ing from ``python_code`` rather than crashing.
        """
        # Unpickling the cache references classes in AOTAutograd's runtime; import
        # dynamo first so that import completes in a non-circular order (otherwise
        # a cold load can hit a runtime_wrappers <-> _dynamo circular import).
        import torch._dynamo

        # The whole calling convention (MODULE_POSITIONS, OUT_SPEC, USER_INPUT_*, PARAM_*,
        # BUFFER_*, IN_SPEC, ...) is consumed by the driver INLINED in python_code
        # (emitted from torch._precompile_driver), so the loaded object needs none of it.
        # _parse_artifact_metadata still runs to validate python_code is a precompile
        # artifact and to read BACKEND for the cache-pairing check below.
        meta = _parse_artifact_metadata(python_code)
        backend = cast(str, meta["BACKEND"])

        # weights_only=True is safe (plain str/int/bytes dict). The inner artifact bytes
        # are the inductor save_cache_artifacts bundle, used below to prime the kernel
        # caches. The cache is acceleration only, so an unreadable envelope or a FORMAT /
        # VERSION mismatch degrades to JIT'ing from python_code rather than crashing. A
        # BACKEND or CODE_HASH mismatch is different -- it signals a wrong (python_code,
        # cache) pairing -- so it hard-fails rather than running under foreign metadata.
        artifact = None
        try:
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            if blob.get("format") != _CACHE_FORMAT or blob.get("version") != (
                _CACHE_VERSION
            ):
                log.warning(
                    "torch.compiler.precompile.load got a cache with format=%r "
                    "version=%r, expected %r / %r; it is likely from a different torch "
                    "build. Falling back to JIT from python_code.",
                    blob.get("format"),
                    blob.get("version"),
                    _CACHE_FORMAT,
                    _CACHE_VERSION,
                )
                blob = None
            if blob is not None:
                if blob.get("backend") != backend:
                    raise PrecompileError(
                        f"cache backend {blob.get('backend')!r} does not match the "
                        f"python_code backend {backend!r}; the cache and python_code "
                        "came from different precompile() calls."
                    )
                # Reject a cache whose code_hash does not match this python_code (a
                # mismatched pairing); see Note [precompile programming model], invariant 7.
                expected_code_hash = hashlib.sha256(python_code.encode()).hexdigest()
                if blob.get("code_hash") != expected_code_hash:
                    raise PrecompileError(
                        "cache does not match python_code (its code_hash "
                        f"{blob.get('code_hash')!r} != sha256(python_code) "
                        f"{expected_code_hash!r}); the cache and python_code came from "
                        "different precompile() calls. Pair each cache with the "
                        "python_code from the same precompile() call."
                    )
                artifact = blob.get("artifact")
        except PrecompileError:
            raise
        except Exception as e:
            log.warning(
                "torch.compiler.precompile.load could not read the cache envelope (%s: %s); the "
                "cache is likely corrupt or from a different torch build. Falling back "
                "to JIT from python_code.",
                type(e).__name__,
                e,
            )
        if artifact is not None:
            # Prime the inductor kernel caches from the bundle so the exec of python_code
            # below loads the precompiled kernels (Triton binaries / autotune results)
            # instead of recompiling them. The composed python_code runs its inlined
            # kernels directly (no compile_fx re-entry, so no FxGraphCache lookup); the
            # acceleration is the warm kernel cache. This is a pure acceleration: a stale /
            # cross-torch-version / corrupt bundle that fails to load just leaves the caches
            # cold, and python_code JITs -- same result, no crash.
            try:
                torch.compiler.load_cache_artifacts(artifact)
            except Exception as e:
                log.warning(
                    "torch.compiler.precompile.load could not prime the cache from the "
                    "artifact bundle (%s: %s); it is likely stale or from a different "
                    "torch build. Falling back to JIT from python_code.",
                    type(e).__name__,
                    e,
                )
        # Run the driver inlined in python_code. It carries the full calling convention and
        # runtime safety checks (subclass wrap/unwrap, param/buffer lifting, grad harvest,
        # input/model validation) and JITs the kernels -- which hit the primed cache when
        # the bundle above loaded, so the "cache" path is exec-with-warm-kernels rather than
        # a separate runtime.
        forward = _make_inlined_forward(python_code)

        return PrecompiledModule._from_loaded(forward, backend=backend)


precompile = _PrecompileApi()
# ``torch.compiler.precompile`` is a callable instance, not a function, so give it the
# name/doc introspection (Sphinx autosummary, help(), IDEs) expects to find on a
# public callable; the rich usage docs live on ``__call__``.
precompile.__name__ = "precompile"  # type: ignore[attr-defined]
precompile.__qualname__ = "precompile"  # type: ignore[attr-defined]
precompile.__doc__ = _PrecompileApi.__call__.__doc__

# Both are public under torch.compiler.precompile, so report their module/qualname there
# (mirroring the singleton fixup above) -- otherwise Sphinx autoexception/autofunction
# would anchor them under this private module. load is a bound method; patch the
# underlying function so introspection on precompile.load reports torch.compiler too.
PrecompileError.__module__ = "torch.compiler"
PrecompileError.__qualname__ = "precompile.PrecompileError"
_PrecompileApi.load.__module__ = "torch.compiler"
_PrecompileApi.load.__qualname__ = "precompile.load"
<|endoftext|>"""
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serialization of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

import builtins
import ctypes
import functools
import glob
import importlib
import inspect
import math
import os
import platform
import sys
import textwrap
import threading
import typing
import typing_extensions
import warnings
from collections.abc import Callable as _Callable, Sequence as _Sequence
from types import ModuleType as _ModuleType
from typing import (
    Any as _Any,
    get_origin as _get_origin,
    overload as _overload,
    TYPE_CHECKING,
    TypeAlias as _TypeAlias,
    TypeGuard as _TypeGuard,
    TypeVar as _TypeVar,
)
from typing_extensions import (
    deprecated as _deprecated,
    LiteralString as _LiteralString,
    Never as _Never,
    ParamSpec as _ParamSpec,
    Self as _Self,
    TypeIs as _TypeIs,
)


# As a bunch of torch.packages internally still have this check
# we need to keep this. @todo: Remove tests that rely on this check as
# they are likely stale.
def _running_with_deploy() -> builtins.bool:
    return False


from torch._utils import (
    _functionalize_sync as _sync,
    _import_dotted_name,
    classproperty,
)
from torch._utils_internal import (
    get_file_path,
    prepare_multiprocessing_environment,
    profiler_allow_cudagraph_cupti_lazy_reinit_cuda12,
    USE_GLOBAL_DEPS,
    USE_RTLD_GLOBAL_WITH_LIBTORCH,
)
from torch.torch_version import __version__ as __version__


if TYPE_CHECKING:
    import sympy

    from torch.distributed._local_tensor import LocalIntNode
    from torch.fx.experimental._constant_symnode import ConstantIntNode
    from torch.fx.experimental.dynamic_spec import ParamsSpec, ShapesSpec
    from torch.fx.experimental.sym_node import SymNode
    from torch.nested._internal.nested_int import NestedIntNode
    from torch.types import BoolLikeType, Device, FloatLikeType, IntLikeType, PySymType

    _SymNodeLike = SymNode | NestedIntNode | ConstantIntNode | LocalIntNode


__all__ = [
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "GradScaler",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "SymBool",
    "SymFloat",
    "SymInt",
    "Tensor",
    "TypedStorage",
    "UntypedStorage",
    "are_deterministic_algorithms_enabled",
    "autocast",
    "chunk",
    "compile",
    "cond",
    "enable_grad",
    "export",
    "get_default_device",
    "get_deterministic_debug_mode",
    "get_device_module",
    "get_float32_matmul_precision",
    "get_rng_state",
    "inference_mode",
    "initial_seed",
    "is_deterministic_algorithms_warn_only_enabled",
    "is_storage",
    "is_tensor",
    "is_warn_always_enabled",
    "load",
    "lobpcg",
    "manual_seed",
    "matmul",
    "no_grad",
    "rand",
    "randn",
    "save",
    "seed",
    "set_default_device",
    "set_default_tensor_type",
    "set_deterministic_debug_mode",
    "set_float32_matmul_precision",
    "set_printoptions",
    "set_rng_state",
    "set_warn_always",
    "split",
    "stack",
    "sym_float",
    "sym_fresh_size",
    "sym_int",
    "sym_ite",
    "sym_max",
    "sym_min",
    "sym_not",
    "sym_sum",
    "thread_safe_generator",
    "typename",
    "unravel_index",
    "use_deterministic_algorithms",
    "vmap",
]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ must be kept sorted")

################################################################################
# Load the extension module
################################################################################

# If PyTorch was built against the ROCm runtime wheels, then there will be
# a _rocm_init module and it will define an initialize() function which can
# prepare ROCm for use. See general documentation on ROCm runtime wheels:
# https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md
# Since this module is only ever added to the wheel if built for such a
# deployment, it is always safe to attempt.
try:
    from . import _rocm_init  # type: ignore[attr-defined]
except ImportError:
    pass
else:
    _rocm_init.initialize()
    del _rocm_init


if sys.platform == "win32":

    def _load_dll_libraries() -> None:
        import importlib.util
        import sysconfig

        from torch.version import cuda as cuda_version

        pfiles_path = os.getenv("ProgramFiles", r"C:\Program Files")
        py_dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
        th_dll_path = os.path.join(os.path.dirname(__file__), "lib")
        # Anchor the DLL dir on the compiled _C extension rather than __file__:
        # editable installs (e.g. scikit-build-core) run this __init__ from the
        # source tree, whose lib/ is empty, while _C and its dependent DLLs live
        # in the installed tree. In a wheel install both resolve to the same dir.
        _spec = importlib.util.find_spec("torch._C")
        if _spec is not None and _spec.origin:
            th_dll_path = os.path.join(os.path.dirname(_spec.origin), "lib")
        usebase_path = os.path.join(
            sysconfig.get_config_var("userbase"), "Library", "bin"
        )
        py_root_bin_path = os.path.join(sys.exec_prefix, "bin")

        # When users create a virtualenv that inherits the base environment,
        # we will need to add the corresponding library directory into
        # DLL search directories. Otherwise, it will rely on `PATH` which
        # is dependent on user settings.
        if sys.exec_prefix != sys.base_exec_prefix:
            base_py_dll_path = os.path.join(sys.base_exec_prefix, "Library", "bin")
        else:
            base_py_dll_path = ""

        dll_paths = [
            p
            for p in (
                th_dll_path,
                py_dll_path,
                base_py_dll_path,
                usebase_path,
                py_root_bin_path,
            )
            if os.path.exists(p)
        ]

        if not builtins.any(
            os.path.exists(os.path.join(p, "nvToolsExt64_1.dll")) for p in dll_paths
        ):
            nvtoolsext_dll_path = os.path.join(
                os.getenv(
                    "NVTOOLSEXT_PATH",
                    os.path.join(pfiles_path, "NVIDIA Corporation", "NvToolsExt"),
                ),
                "bin",
                "x64",
            )
        else:
            nvtoolsext_dll_path = ""

        if cuda_version and builtins.all(
            not glob.glob(os.path.join(p, "cudart64*.dll")) for p in dll_paths
        ):
            cuda_version_1 = cuda_version.replace(".", "_")
            cuda_path_var = "CUDA_PATH_V" + cuda_version_1
            default_path = os.path.join(
                pfiles_path, "NVIDIA GPU Computing Toolkit", "CUDA", f"v{cuda_version}"
            )
            cuda_path = os.path.join(os.getenv(cuda_path_var, default_path), "bin")
        else:
            cuda_path = ""

        dll_paths.extend(
            p for p in (nvtoolsext_dll_path, cuda_path) if os.path.exists(p)
        )

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        if with_load_library_flags:
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p

        for dll_path in dll_paths:
            os.add_dll_directory(dll_path)

        try:
            ctypes.CDLL("vcruntime140.dll")
            ctypes.CDLL("msvcp140.dll")
            if platform.machine() != "ARM64":
                ctypes.CDLL("vcruntime140_1.dll")
        except OSError:
            print(
                textwrap.dedent(
                    """
                    Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                    It can be downloaded at https://aka.ms/vs/17/release/vc_redist.x64.exe
                    """
                ).strip()
            )

        dlls = glob.glob(os.path.join(th_dll_path, "*.dll"))
        path_patched = False
        for dll in dlls:
            is_loaded = False
            if with_load_library_flags:
                res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
                last_error = ctypes.get_last_error()
                if res is None and last_error != 126:
                    err = ctypes.WinError(last_error)
                    err.strerror += (
                        f' Error loading "{dll}" or one of its dependencies.'
                    )
                    raise err
                elif res is not None:
                    is_loaded = True
            if not is_loaded:
                if not path_patched:
                    os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                    path_patched = True
                res = kernel32.LoadLibraryW(dll)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += (
                        f' Error loading "{dll}" or one of its dependencies.'
                    )
                    raise err

        kernel32.SetErrorMode(prev_error_mode)

    _load_dll_libraries()
    del _load_dll_libraries


def _get_cuda_dep_paths(path: str, lib_folder: str, lib_name: str) -> list[str]:
    # Libraries can either be in
    # path/nvidia/lib_folder/lib or
    # path/nvidia/cuXX/lib (since CUDA 13.0) or
    # path/lib_folder/lib
    from torch.version import cuda as cuda_version

    nvidia_lib_paths = glob.glob(
        os.path.join(path, "nvidia", lib_folder, "lib", lib_name)
    )
    if cuda_version is not None:
        maj_cuda_version = cuda_version.split(".")[0]
        nvidia_lib_paths += glob.glob(
            os.path.join(path, "nvidia", f"cu{maj_cuda_version}", "lib", lib_name)
        )
    lib_paths = glob.glob(os.path.join(path, lib_folder, "lib", lib_name))

    return nvidia_lib_paths + lib_paths


def _preload_cuda_lib(lib_folder: str, lib_name: str, required: bool = True) -> None:  # type: ignore[valid-type]
    """Preloads cuda library if it could not be found otherwise."""
    # Should only be called on Linux if default path resolution have failed
    if platform.system() != "Linux":
        raise AssertionError(f"Should only be called on Linux, got {platform.system()}")

    lib_path = None
    for path in sys.path:
        candidate_lib_paths = _get_cuda_dep_paths(path, lib_folder, lib_name)
        if candidate_lib_paths:
            lib_path = candidate_lib_paths[0]
            break
    if not lib_path and required:
        raise ValueError(f"{lib_name} not found in the system path {sys.path}")
    if lib_path:
        ctypes.CDLL(lib_path)


def _preload_cuda_deps(err: OSError | None = None, required: bool = True) -> None:
    cuda_libs: list[tuple[str, str]] = [
        # NOTE: Order matters! We must preload libcublasLt BEFORE libcublas to prevent
        # libcublas from loading a mismatched system-wide libcublasLt via its RUNPATH.
        # Without this, if a different CUDA Toolkit version exists in the system PATH,
        # libcublas may load the wrong libcublasLt, causing symbol errors or runtime failures.
        ("cublas", "libcublasLt.so.*[0-9]"),
        ("cublas", "libcublas.so.*[0-9]"),
        ("cudnn", "libcudnn.so.*[0-9]"),
        ("cuda_nvrtc", "libnvrtc.so.*[0-9]"),
        ("cuda_nvrtc", "libnvrtc-builtins.so.*[0-9]"),
        ("cuda_runtime", "libcudart.so.*[0-9]"),
        ("cuda_cupti", "libcupti.so.*[0-9]"),
        ("cufft", "libcufft.so.*[0-9]"),
        ("curand", "libcurand.so.*[0-9]"),
        ("nvjitlink", "libnvJitLink.so.*[0-9]"),
        ("cusparse", "libcusparse.so.*[0-9]"),
        ("cusparselt", "libcusparseLt.so.*[0-9]"),
        ("cusolver", "libcusolver.so.*[0-9]"),
        ("nccl", "libnccl.so.*[0-9]"),
        ("nvshmem", "libnvshmem_host.so.*[0-9]"),
        ("cufile", "libcufile.so.*[0-9]"),
    ]
    # If error is passed, re-raise it if it's not about one of the abovementioned
    # libraries
    if err is not None and not [
        lib for _, lib in cuda_libs if lib.split(".", 1)[0] in err.args[0]
    ]:
        raise err

    # Otherwise, try to preload dependencies from site-packages. With
    # required=False each lib is best-effort: a partial install that only ships
    # some of these wheels (e.g. just a cupti wheel alongside a system CUDA)
    # preloads the ones present instead of aborting on the first missing one.
    for lib_folder, lib_name in cuda_libs:
        _preload_cuda_lib(lib_folder, lib_name, required=required)

    # libnvToolsExt is Optional Dependency
    _preload_cuda_lib("nvtx", "libnvToolsExt.so.*[0-9]", required=False)


# See Note [Global dependencies]
def _load_global_deps() -> None:
    if platform.system() == "Windows":
        return

    # Determine the file extension based on the platform
    lib_ext = ".dylib" if platform.system() == "Darwin" else ".so"
    lib_name = f"libtorch_global_deps{lib_ext}"
    here = os.path.abspath(__file__)
    global_deps_lib_path = os.path.join(os.path.dirname(here), "lib", lib_name)

    # In scikit-build-core editable installs with redirect mode, native libs are
    # installed to the dist package location rather than relative to __file__.
    if not os.path.exists(global_deps_lib_path):
        try:
            from importlib.metadata import distribution

            installed = distribution("torch").locate_file(
                os.path.join("torch", "lib", lib_name)
            )
            # The importlib metadata SimplePath protocol was missing the exists
            # method in older versions; however, the actual Path implementation
            # has it and newer versions of importlib metadata have added it to
            # the protocol, making the following ignore unnecessary from
            # importlib_metadata 7.0.1 and Python 3.13 onwards.
            # pyrefly: ignore[missing-attribute]
            if installed.exists():
                global_deps_lib_path = str(installed)
        except Exception:
            pass

    try:
        ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)
        # Workaround slim-wheel CUDA dependency bugs in cusparse and cudnn by preloading nvjitlink
        # and nvrtc. In CUDA-12.4+ cusparse depends on nvjitlink, but does not have rpath when
        # shipped as wheel, which results in OS picking wrong/older version of nvjitlink library
        # if `LD_LIBRARY_PATH` is defined, see https://github.com/pytorch/pytorch/issues/138460
        # Similar issue exist in cudnn that dynamically loads nvrtc, unaware of its relative path.
        # See https://github.com/pytorch/pytorch/issues/145580
        try:
            with open("/proc/self/maps") as f:
                _maps = f.read()

            # libtorch_global_deps.so always depends in cudart, check if its installed and loaded
            if "libcudart.so" not in _maps:
                return
            # If all above-mentioned conditions are met, preload CUDA dependencies.
            # Only libtorch_global_deps is loaded so far (it pulls libcudart, not
            # the rest); libtorch_cpu, which DT_NEEDEDs the other CUDA libs, is
            # imported below via torch._C. So be best-effort and front-load any
            # component wheels that ARE present now, before that import, so they
            # win libtorch_cpu's DT_NEEDED soname lookups (e.g. a newer cupti
            # wheel over a system copy). A missing wheel must not abort -- rpath
            # will satisfy libtorch_cpu's DT_NEEDED for whatever isn't preloaded.
            _preload_cuda_deps(required=False)
        except Exception:
            pass

    except OSError as err:
        # Can happen for wheel with cuda libs as PYPI deps
        # As PyTorch is not purelib, but nvidia-*-cu12 is
        _preload_cuda_deps(err)
        ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)


if (USE_RTLD_GLOBAL_WITH_LIBTORCH or os.getenv("TORCH_USE_RTLD_GLOBAL")) and (
    platform.system() != "Windows"
):
    # Do it the hard way.  You might want to load libtorch with RTLD_GLOBAL in a
    # few circumstances:
    #
    #   1. You're in a build environment (e.g., fbcode) where
    #      libtorch_global_deps is not available, but you still need
    #      to get mkl to link in with RTLD_GLOBAL or it will just
    #      not work.
    #
    #   2. You're trying to run PyTorch under UBSAN and you need
    #      to ensure that only one copy of libtorch is loaded, so
    #      vptr checks work properly
    #
    # If you're using this setting, you must verify that all the libraries
    # you load consistently use the same libstdc++, or you may have
    # mysterious segfaults.
    #
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

    from torch._C import *  # noqa: F403

    sys.setdlopenflags(old_flags)
    del old_flags

else:
    # Easy way.  You want this most of the time, because it will prevent
    # C++ symbols from libtorch clobbering C++ symbols from other
    # libraries, leading to mysterious segfaults.
    #
    # If building in an environment where libtorch_global_deps isn't available
    # like parts of fbsource, but where RTLD_GLOBAL causes segfaults, you will
    # want USE_RTLD_GLOBAL_WITH_LIBTORCH = False and USE_GLOBAL_DEPS = False
    #
    # See Note [Global dependencies]
    if USE_GLOBAL_DEPS:
        _load_global_deps()
    from torch._C import *  # noqa: F403


_PrimType = _TypeVar("_PrimType")
_SymType = _TypeVar("_SymType")
_SymIteT = _TypeVar("_SymIteT")
_BecomesIntPrimType = _TypeVar("_BecomesIntPrimType")
_BecomesIntSymType = _TypeVar("_BecomesIntSymType")
# Operand type for bitwise ops: IntLikeType for SymInt; BoolLikeType for SymBool.
_BitwiseLikeType = _TypeVar("_BitwiseLikeType")
# Operand types that promote the result to SymFloat. _Never for SymFloat
# (no further promotion); float | SymFloat for SymInt and SymBool.
_FloatPromotionType = _TypeVar("_FloatPromotionType")
_LikeNumber: _TypeAlias = "torch.types.Number | torch.types.PySymType"


class _SymTypingMagicAlsoBool(
    typing.Generic[
        _PrimType, _BecomesIntPrimType, _BecomesIntSymType, _FloatPromotionType
    ]
):
    # Magic methods installed by torch.fx.experimental.sym_node
    # From sym_node: also_bool_magic_methods

    __LikeType: _TypeAlias = (
        _PrimType | _Self | _BecomesIntPrimType | _BecomesIntSymType
    )
    _BecomesIntLikeType: _TypeAlias = _BecomesIntPrimType | _BecomesIntSymType

    @_overload
    def __eq__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __eq__(self, other: object) -> builtins.bool: ...

    def __eq__(self, other: object) -> "builtins.bool | SymBool":
        raise TypeError("type stub not overridden")

    @_overload
    def __ne__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __ne__(self, other: object) -> builtins.bool: ...

    def __ne__(self, other: object) -> "builtins.bool | SymBool":
        raise TypeError("type stub not overridden")

    # bool_becomes_int_magic_methods

    @_overload
    def __add__(self, other: __LikeType) -> _BecomesIntSymType: ...

    @_overload
    def __add__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __add__(self, other: object) -> object:
        raise TypeError("type stub not overridden")

    # Tensor + SymT -> Tensor
    # prim|SymT + SymT -> SymT
    @_overload
    def __radd__(self, other: "torch.Tensor") -> "torch.Tensor": ...

    @_overload
    def __radd__(
        self, other: _PrimType | _BecomesIntPrimType | _BecomesIntSymType
    ) -> _BecomesIntSymType: ...

    @_overload
    def __radd__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __radd__(self, other: object) -> object:
        raise TypeError("type stub not overridden")

    @_overload
    def __sub__(self, other: __LikeType) -> _BecomesIntSymType: ...

    @_overload
    def __sub__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __sub__(self, other: object) -> object:
        raise TypeError("type stub not overridden")

    # Tensor - SymT = Tensor
    # prim|SymT - SymT = SymT
    @_overload
    def __rsub__(self, other: "torch.Tensor") -> "torch.Tensor": ...

    @_overload
    def __rsub__(
        self, other: _PrimType | _BecomesIntPrimType | _BecomesIntSymType
    ) -> _BecomesIntSymType: ...

    @_overload
    def __rsub__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __rsub__(self, other: object) -> object:
        raise TypeError("type stub not overridden")

    @_overload
    def __mul__(self, other: __LikeType) -> _BecomesIntSymType: ...

    @_overload
    def __mul__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __mul__(self, other: object) -> object:
        raise TypeError("type stub not overridden")

    @_overload
    def __rmul__(self, other: __LikeType) -> _BecomesIntSymType: ...

    @_overload
    def __rmul__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __rmul__(self, other: object) -> object:
        raise TypeError("type stub not overridden")


class _SymTypingMagic(
    _SymTypingMagicAlsoBool[_PrimType, _PrimType, _SymType, _FloatPromotionType],
    typing.Generic[_PrimType, _SymType, _FloatPromotionType],
):
    # Magic methods installed by torch.fx.experimental.sym_node
    # From sym_node:
    #   magic_methods
    #   - only_float_magic_methods
    #   - bitwise_ops
    #   - only_bool_magic_methods
    #   - also_bool_magic_methods
    #   - bool_becomes_int_magic_methods

    __LikeType: _TypeAlias = _PrimType | _Self

    # Ordering comparisons: only on ordered types (SymInt/SymFloat), not SymBool.
    @_overload
    def __ge__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __ge__(self, other: object) -> builtins.bool: ...

    def __ge__(self, other: object) -> "builtins.bool | SymBool":
        raise TypeError("type stub not overridden")

    @_overload
    def __gt__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __gt__(self, other: object) -> builtins.bool: ...

    def __gt__(self, other: object) -> "builtins.bool | SymBool":
        raise TypeError("type stub not overridden")

    @_overload
    def __le__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __le__(self, other: object) -> builtins.bool: ...

    def __le__(self, other: object) -> "builtins.bool | SymBool":
        raise TypeError("type stub not overridden")

    @_overload
    def __lt__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __lt__(self, other: object) -> builtins.bool: ...

    def __lt__(self, other: object) -> "builtins.bool | SymBool":
        raise TypeError("type stub not overridden")

    def __abs__(self) -> _Self:
        raise TypeError("type stub not overridden")

    def __ceil__(self) -> "SymInt":
        raise TypeError("type stub not overridden")

    def __float_pow__(self, other: _LikeNumber) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __rfloat_pow__(self, other: _LikeNumber) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __float_truediv__(self, other: _LikeNumber) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __rfloat_truediv__(self, other: _LikeNumber) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __floor__(self) -> "SymInt":
        raise TypeError("type stub not overridden")

    def __int_floordiv__(self, other: _LikeNumber) -> "SymInt":
        raise TypeError("type stub not overridden")

    def __rint_floordiv__(self, other: _LikeNumber) -> "SymInt":
        raise TypeError("type stub not overridden")

    def __int_truediv__(self, other: _LikeNumber) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __rint_truediv__(self, other: _LikeNumber) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __lshift__(self, other: "IntLikeType") -> _Self:
        raise TypeError("type stub not overridden")

    def __rlshift__(self, other: "IntLikeType") -> _Self:
        raise TypeError("type stub not overridden")

    @_overload
    def __mod__(self, other: __LikeType) -> _SymType: ...

    @_overload
    def __mod__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __mod__(self, other: object) -> object:
        raise TypeError("type stub not overridden")

    @_overload
    def __rmod__(self, other: __LikeType) -> _SymType: ...

    @_overload
    def __rmod__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __rmod__(self, other: object) -> object:
        raise TypeError("type stub not overridden")

    def __neg__(self) -> _Self:
        raise TypeError("type stub not overridden")

    def __pos__(self) -> _Self:
        raise TypeError("type stub not overridden")

    def __pow_by_natural__(self, other: "IntLikeType") -> "SymInt":
        raise TypeError("type stub not overridden")

    def __rpow_by_natural__(self, other: "IntLikeType") -> "SymInt":
        raise TypeError("type stub not overridden")

    def __rshift__(self, other: "IntLikeType") -> _Self:
        raise TypeError("type stub not overridden")

    def __rrshift__(self, other: "IntLikeType") -> _Self:
        raise TypeError("type stub not overridden")

    def __sym_acos__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_asin__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_atan__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_cos__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_cosh__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_float__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_max__(self, other: _LikeNumber) -> _LikeNumber:
        raise TypeError("type stub not overridden")

    def __sym_min__(self, other: _LikeNumber) -> _LikeNumber:
        raise TypeError("type stub not overridden")

    def __sym_sin__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_sinh__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_sqrt__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_tan__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __sym_tanh__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def __trunc__(self) -> "SymInt":
        raise TypeError("type stub not overridden")


class _SymTypingMagicBitwise(typing.Generic[_BitwiseLikeType]):
    # Magic methods installed by torch.fx.experimental.sym_node
    # From sym_node: bitwise_ops

    def __and__(self, other: _BitwiseLikeType) -> _Self:
        raise TypeError("type stub not overridden")

    def __rand__(self, other: _BitwiseLikeType) -> _Self:
        raise TypeError("type stub not overridden")

    def __or__(self, other: _BitwiseLikeType) -> _Self:
        raise TypeError("type stub not overridden")

    def __ror__(self, other: _BitwiseLikeType) -> _Self:
        raise TypeError("type stub not overridden")

    def __xor__(self, other: _BitwiseLikeType) -> _Self:
        raise TypeError("type stub not overridden")

    def __rxor__(self, other: _BitwiseLikeType) -> _Self:
        raise TypeError("type stub not overridden")


class SymInt(
    _SymTypingMagic[builtins.int, "SymInt", "FloatLikeType"],
    _SymTypingMagicBitwise["IntLikeType"],
):
    """
    Like an int (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node: "_SymNodeLike") -> None:
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = typing.cast("SymNode", node)

    def __bool__(self) -> builtins.bool:
        return builtins.bool(self != 0)

    def __int__(self) -> builtins.int:
        # pyrefly: ignore[missing-attribute]  # duck-typed: not on NestedIntNode/ConstantIntNode/LocalIntNode
        return self.node.int_()

    def __index__(self) -> builtins.int:
        # pyrefly: ignore[missing-attribute]  # duck-typed: not on NestedIntNode/ConstantIntNode/LocalIntNode
        return self.node.int_()

    # Magic methods installed by torch.fx.experimental.sym_node

    def __round__(self, ndigits: builtins.int | None = None) -> "SymInt":
        return self

    def __truediv__(self, other: object) -> "SymFloat":
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__float_truediv__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        return self.__int_truediv__(other)

    def __rtruediv__(self, other: object) -> "SymFloat":
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__rfloat_truediv__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        return self.__rint_truediv__(other)

    @_overload
    def __floordiv__(self, other: "builtins.int | SymInt") -> "SymInt": ...

    @_overload
    def __floordiv__(self, other: "builtins.float | SymFloat") -> "SymFloat": ...

    def __floordiv__(self, other: object) -> "SymInt | SymFloat":
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(math.floor(sym_float(self) / other))
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        return self.__int_floordiv__(other)

    @_overload
    def __rfloordiv__(self, other: "builtins.int | SymInt") -> "SymInt": ...

    @_overload
    def __rfloordiv__(self, other: "builtins.float | SymFloat") -> "SymFloat": ...

    def __rfloordiv__(self, other: object) -> "SymInt | SymFloat":
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(math.floor(other / sym_float(self)))
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        return self.__rint_floordiv__(other)

    # nb: complex is impossible to handle correctly lol, with
    # negative base and integral float need to diverge semantics and
    # just always return complex.  Neener neener pretend this problem
    # doesn't exist
    def __pow__(self, other: object) -> "SymInt | SymFloat":
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__pow__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        # Guards!  This guard is necessary because we need to know it to
        # determine the output type of this operation
        if other >= 0:
            return self.__pow_by_natural__(other)
        else:
            # Mercifully, when the exponent is negative, Python just promotes
            # to doubles and does a float pow:
            #
            #   if (Py_SIZE(b) < 0 && c == NULL) {
            #       /* if exponent is negative and there's no modulus:
            #              return a float.  This works because we know
            #              that this calls float_pow() which converts its
            #              arguments to double. */
            #       Py_DECREF(a);
            #       Py_DECREF(b);
            #       return PyFloat_Type.tp_as_number->nb_power(v, w, x);
            #   }
            return sym_float(self).__pow__(sym_float(other))

    def __rpow__(self, other: object) -> "SymInt | SymFloat":
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__rpow__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        if self >= 0:  # self is exponent
            return self.__rpow_by_natural__(other)
        else:
            return sym_float(self).__rpow__(sym_float(other))

    def __repr__(self) -> str:
        return self.node._graph_repr()

    def _sympy_(self) -> "sympy.Basic":
        # pyrefly: ignore[missing-attribute]  # duck-typed: not on NestedIntNode/ConstantIntNode/LocalIntNode
        return self.node.expr

    def __hash__(self) -> builtins.int:
        if self.node.is_nested_int():
            # pyrefly: ignore[missing-attribute]  # narrowed by is_nested_int(); only NestedIntNode/SymNode reach here
            return hash(self.node.nested_int())
        else:
            # We could support constant SymInts as well, but not doing it for now
            raise TypeError("unhashable type: non-nested SymInt")
            # TODO: Force specialization
            # This can't be done because the TypeError here is load bearing
            # for einops
            # https://github.com/arogozhnikov/einops/blob/6181e1e95dc58c00a3143c1726da1c6ee0463164/einops/einops.py#L237
            # return hash(builtins.int(self))

    def as_integer_ratio(self) -> tuple["SymInt", builtins.int]:
        """Represent this int as an exact integer ratio"""
        return self, 1

    def bit_length(self) -> builtins.int:
        # TODO: A more relaxed guard is possible here, where you guard to
        # allow all integer quantities which would result in the same bit
        # length.  We can also just make a dedicated Sympy function for
        # computing this quantity and represent it symbolically.
        return builtins.int(self).bit_length()

    def conjugate(self) -> "SymInt":
        return self

    def has_hint(self) -> builtins.bool:
        return self.node.has_hint()

    @property
    def hint(self) -> builtins.int | None:
        return typing.cast(builtins.int | None, self.node.hint)

    @property
    def constant(self) -> builtins.int | None:
        return typing.cast(builtins.int | None, self.node.constant)


class SymFloat(_SymTypingMagic[builtins.float, "SymFloat", _Never]):
    """
    Like a float (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node: "SymNode") -> None:
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __truediv__(self, other: object) -> "SymFloat":
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        return self.__float_truediv__(sym_float(other))

    def __rtruediv__(self, other: object) -> "SymFloat":
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        return self.__rfloat_truediv__(sym_float(other))

    def __floordiv__(self, other: object) -> "SymFloat":
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        return sym_float(math.floor(self / sym_float(other)))

    def __rfloordiv__(self, other: object) -> "SymFloat":
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        return sym_float(math.floor(sym_float(other) / self))

    def __bool__(self) -> builtins.bool:
        return self.node.bool_()

    def __float__(self) -> builtins.float:
        return self.node.guard_float("", 0)

    def __int__(self) -> builtins.int:
        return self.__trunc__().__int__()

    # Symbolic power does NOT work with negative base, this is to avoid
    # potential complex outputs
    def __pow__(self, other: object) -> "SymFloat":
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        torch._check(self >= 0)
        return self.__float_pow__(other)

    def __rpow__(self, other: object) -> "SymFloat":
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        torch._check(other >= 0)
        return self.__rfloat_pow__(other)

    # Magic methods installed by torch.fx.experimental.sym_node

    @_overload
    def __round__(self) -> "SymInt": ...

    @_overload
    def __round__(self, ndigits: "IntLikeType") -> "SymFloat": ...

    def __round__(self, ndigits: "IntLikeType | None" = None) -> "SymFloat | SymInt":
        raise TypeError("type stub not overridden")

    # def __sym_int__(self) -> "SymInt":
    #     raise TypeError("type stub not overridden")

    def __sym_log2__(self) -> "SymFloat":
        raise TypeError("type stub not overridden")

    def is_integer(self) -> builtins.bool:
        """Return True if the float is an integer."""
        raise TypeError("type stub not overridden")

    def as_integer_ratio(self) -> tuple[builtins.int, builtins.int]:
        """Represent this float as an exact integer ratio"""
        return builtins.float(self).as_integer_ratio()

    def __repr__(self) -> str:
        return self.node._graph_repr()

    def _sympy_(self) -> "sympy.Basic":
        return self.node.expr

    def __hash__(self) -> builtins.int:
        return hash(builtins.float(self))

    def conjugate(self) -> "SymFloat":
        """Returns the complex conjugate of the float."""
        return self

    def hex(self) -> str:
        """Returns the hexadecimal representation of the float."""
        return self.node.guard_float("", 0).hex()

    def has_hint(self) -> builtins.bool:
        return self.node.has_hint()

    @property
    def hint(self) -> builtins.float | None:
        return typing.cast(builtins.float | None, self.node.hint)

    @property
    def constant(self) -> builtins.float | None:
        return typing.cast(builtins.float | None, self.node.constant)


class SymBool(
    _SymTypingMagicAlsoBool[builtins.bool, builtins.int, "SymInt", "FloatLikeType"],
    _SymTypingMagicBitwise["BoolLikeType"],
):
    """
    Like a bool (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.

    Unlike regular bools, regular boolean operators will force extra guards instead
    of symbolically evaluate.  Use the bitwise operators instead to handle this.
    """

    def __init__(self, node: "SymNode") -> None:
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self) -> builtins.bool:
        return self.node.bool_()

    def __int__(self) -> builtins.int:
        return builtins.int(self.node.bool_())

    # Magic methods installed by torch.fx.experimental.sym_node

    # We very carefully define __sym_not__, and not a number of other
    # plausible alternatives:
    #
    #   - We do not override __not__ because this is not a real magic
    #     method; you cannot override the meaning of the not builtin in
    #     Python.  We use the name 'sym_not' to clarify that in user code you
    #     cannot use the builtin not or operator.not_ or operator.__not__ and
    #     hit this magic method; you must use our custom sym_not operator.
    #
    #   - We do not override the __invert__ method because SymBool is
    #     meant to be usable in situations where bool is expected.  However,
    #     bitwise negation ~a does the wrong thing with booleans (because
    #     bool is a subclass of int, so ~1 = -2 which is not falseish.)
    #     This would be a giant footgun, so we get around it by defining
    #     our own operator.  Note that bitwise and/or do the right thing,
    #     so we reuse the conventional operators there for readability.
    #
    def __sym_not__(self) -> "SymBool":
        raise TypeError("type stub not overridden")

    def __sym_ite__(self, then_val: _SymIteT, else_val: _SymIteT) -> _SymIteT:
        raise TypeError("type stub not overridden")

    def __repr__(self) -> str:
        return self.node._graph_repr()

    def _sympy_(self) -> "sympy.Basic":
        return self.node.expr

    def __hash__(self) -> builtins.int:
        if self.node.is_constant():
            return hash(self.node.bool_())
        else:
            # Force specialization
            return hash(builtins.bool(self))

    def __sym_float__(self) -> "SymFloat":
        """
        Provides a SymFloat representation (0.0 or 1.0) for this SymBool.
        Called by torch.sym_float() when casting SymBool to float.
        """
        from torch.fx.experimental.sym_node import wrap_node

        # pyrefly: ignore[bad-return]
        return wrap_node(self.node.sym_float())

    @property
    def hint(self) -> builtins.bool | None:
        return typing.cast(builtins.bool | None, self.node.hint)

    @property
    def constant(self) -> builtins.bool | None:
        return typing.cast(builtins.bool | None, self.node.constant)


@_overload
def sym_not(a: "BoolLikeType") -> "BoolLikeType": ...


@_overload
def sym_not(a: object) -> object: ...


def sym_not(a: object) -> object:
    r"""SymInt-aware utility for logical negation.

    Args:
        a (SymBool or bool): Object to negate
    """
    import sympy

    if overrides.has_torch_function_unary(a):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_not, (a,), a)
    if hasattr(a, "__sym_not__"):
        return a.__sym_not__()  # type: ignore[attr-defined]
    if isinstance(a, sympy.Basic):
        return ~a  # type: ignore[operator]
    return not a


@_overload
def sym_float(a: "PySymType") -> "SymFloat": ...


@_overload
def sym_float(a: object) -> "FloatLikeType": ...


def sym_float(a: object) -> "FloatLikeType":
    r"""SymInt-aware utility for float casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if overrides.has_torch_function_unary(a):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_float, (a,), a)
    if isinstance(a, SymFloat):
        return a
    elif hasattr(a, "__sym_float__"):
        return a.__sym_float__()  # type: ignore[attr-defined]
    return builtins.float(a)  # type: ignore[arg-type]


@_overload
def sym_int(a: "PySymType") -> "IntLikeType": ...


@_overload
def sym_int(a: object) -> builtins.int: ...


def sym_int(a: object) -> "IntLikeType":
    r"""SymInt-aware utility for int casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if overrides.has_torch_function_unary(a):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_int, (a,), a)
    if isinstance(a, SymInt):
        return a
    elif isinstance(a, SymFloat):
        return math.trunc(a)
    return builtins.int(a)  # type: ignore[arg-type]


@_overload
def sym_max(a: "IntLikeType", b: "IntLikeType") -> "IntLikeType": ...


@_overload
def sym_max(
    a: "IntLikeType | FloatLikeType", b: "IntLikeType | FloatLikeType"
) -> "FloatLikeType": ...


def sym_max(
    a: "IntLikeType | FloatLikeType", b: "IntLikeType | FloatLikeType"
) -> "IntLikeType | FloatLikeType":
    """
    SymInt-aware utility for max which avoids branching on a < b.
    Unlike builtins.max(), this only works for int/float, and it always
    promotes to float if any argument is float (unlike builtins.max, which
    will faithfully preserve the type of the input argument).
    """
    if overrides.has_torch_function((a, b)):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_max, (a, b), a, b)
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_max__(b)  # type: ignore[return-value]
    elif isinstance(b, (SymInt, SymFloat)):
        # Due to promotion semantics, this is operator is commutative:
        # max(1, 1.0) === max(1.0, 1) === 1.0
        return b.__sym_max__(a)  # type: ignore[return-value]
    # TODO: Probably can make bool work too, just lazy

    all_types, float_types = __all_and_float_types()

    if not isinstance(a, all_types):
        raise AssertionError(f"expected {all_types}, got {type(a)}")
    if not isinstance(b, all_types):
        raise AssertionError(f"expected {all_types}, got {type(b)}")
    if isinstance(a, float_types) or isinstance(b, float_types):
        return builtins.float(builtins.max(a, b))  # type: ignore[call-overload]
    else:
        return builtins.max(a, b)  # type: ignore[call-overload]


def __all_and_float_types() -> tuple[tuple[type, ...], tuple[type, ...]]:
    try:
        import numpy as np

        all_types: tuple[type, ...] = (
            np.integer,
            np.floating,
            builtins.int,
            builtins.float,
        )
        float_types: tuple[type, ...] = (np.floating, builtins.float)
    except ModuleNotFoundError:
        all_types = (builtins.int, builtins.float)
        float_types = (builtins.float,)

    return all_types, float_types


@_overload
def sym_min(a: "IntLikeType", b: "IntLikeType") -> "IntLikeType": ...


@_overload
def sym_min(
    a: "IntLikeType | FloatLikeType", b: "IntLikeType | FloatLikeType"
) -> "FloatLikeType": ...


def sym_min(
    a: "IntLikeType | FloatLikeType", b: "IntLikeType | FloatLikeType"
) -> "IntLikeType | FloatLikeType":
    """SymInt-aware utility for min()."""
    if overrides.has_torch_function((a, b)):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_min, (a, b), a, b)
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_min__(b)  # type: ignore[return-value]
    elif isinstance(b, (SymInt, SymFloat)):
        return b.__sym_min__(a)  # type: ignore[return-value]

    all_types, float_types = __all_and_float_types()

    if not isinstance(a, all_types):
        raise AssertionError(f"expected {all_types}, got {type(a)}")
    if not isinstance(b, all_types):
        raise AssertionError(f"expected {all_types}, got {type(b)}")
    if isinstance(a, float_types) or isinstance(b, float_types):
        return builtins.float(builtins.min(a, b))  # type: ignore[call-overload]
    else:
        return builtins.min(a, b)  # type: ignore[call-overload]


def sym_sum(
    *args: "IntLikeType | _Sequence[IntLikeType]",
) -> "IntLikeType":
    """
    N-ary add which is faster to compute for long lists than iterated binary
    addition.  Only does something special for integers.

    Accepts both ``sym_sum([a, b, c])`` and ``sym_sum(a, b, c)``.
    """
    # Normalise: accept both sym_sum([a, b, c]) and sym_sum(a, b, c).
    items: _Sequence[IntLikeType] = args  # type: ignore[assignment]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        items = args[0]

    if overrides.has_torch_function(items):
        return overrides.handle_torch_function(sym_sum, items, items)

    found = None
    for a in items:
        if not isinstance(a, (SymInt, builtins.int)):
            return builtins.sum(items)
        if isinstance(a, SymInt):
            found = a.node
    if found is None:
        return builtins.sum(items)

    from torch.fx.experimental.sym_node import to_node, wrap_node

    # pyrefly: ignore[missing-attribute, bad-argument-type, bad-return]  # found is a SymNode in practice (LocalIntNode also has sym_sum)
    return wrap_node(found.sym_sum([to_node(found, a) for a in items]))


# Drop in replacement for math.sqrt, math.sin, math.cos etc
def _get_sym_math_fn(name: str) -> _Callable[[_Any], _Any]:
    def fn(a: _Any) -> _Any:
        if overrides.has_torch_function_unary(a):
            return overrides.handle_torch_function(fn, (a,), a)
        if isinstance(a, SymInt):
            a = torch.sym_float(a)
        if hasattr(a, f"__sym_{name}__"):
            return getattr(a, f"__sym_{name}__")()
        return getattr(math, name)(a)

    return fn


__fn, __name, __sym_name = None, "", ""
for __name in (
    "sqrt",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "asin",
    "acos",
    "atan",
    "log2",
):
    __sym_name = f"_sym_{__name}"
    __fn = _get_sym_math_fn(__name)
    __fn.__qualname__ = __fn.__name__ = __sym_name
    globals()[__sym_name] = __fn


del __fn, __name, __sym_name, _get_sym_math_fn

# Adding temporary shortcut
sym_sqrt = globals()["_sym_sqrt"]
__all__.append("sym_sqrt")


def sym_ite(b: "BoolLikeType", t: _SymIteT, f: _SymIteT) -> _SymIteT:
    """SymInt-aware utility for ternary operator (``t if b else f``.)"""
    if overrides.has_torch_function((b, t, f)):
        return overrides.handle_torch_function(sym_ite, (b, t, f), b, t, f)
    if not isinstance(b, (SymBool, builtins.bool)):
        raise AssertionError(f"expected SymBool or bool, got {type(b)}")
    if type(t) is not type(f):
        raise AssertionError(f"type mismatch: {type(t)} vs {type(f)}")
    if isinstance(b, SymBool):
        return b.__sym_ite__(t, f)
    return t if b else f


# Create a fresh unbacked int, from an (possibly unbacked int) expression.
def sym_fresh_size(expr: "IntLikeType") -> "IntLikeType":
    return typing.cast("IntLikeType", torch.tensor(expr).item())


# Check to see if we can load C extensions, and if not provide some guidance
# on what the problem might be.
try:
    # _initExtension is chosen (arbitrarily) as a sentinel.
    from torch._C import _initExtension
except ImportError:
    import torch._C as _C_for_compiled_check

    if _C_for_compiled_check.__file__ is None:
        raise ImportError(
            textwrap.dedent(
                """
                Failed to load PyTorch C extensions:
                    It appears that PyTorch has loaded the `torch/_C` folder
                    of the PyTorch repository rather than the C extensions which
                    are expected in the `torch._C` namespace. This can occur when
                    using the `install` workflow. e.g.
                        $ python -m pip install --no-build-isolation -v . && python -c "import torch"

                    This error can generally be solved using the `develop` workflow
                        $ python -m pip install --no-build-isolation -v -e . && python -c "import torch"  # This should succeed
                    or by running Python from a different directory.
                """
            ).strip()
        ) from None
    raise  # If __file__ is not None the cause is unknown, so just re-raise.

# The torch._C submodule is already loaded via `from torch._C import *` above
# Make an explicit reference to the _C submodule to appease linters
from torch import _C as _C


__name, __obj = "", None
for __name in dir(_C):
    if __name[0] != "_" and not __name.endswith("Base"):
        __all__.append(__name)
        __obj = getattr(_C, __name)
        if callable(__obj) or inspect.isclass(__obj):
            if __obj.__module__ != __name__:  # "torch"
                # TODO: fix their module from C++ side
                if __name not in {
                    "DisableTorchFunctionSubclass",
                    "DisableTorchFunction",
                    "Generator",
                }:
                    __obj.__module__ = __name__  # "torch"
    elif __name == "TensorBase":
        # issue 109438 / pr 109940. Prevent TensorBase from being copied into torch.
        delattr(sys.modules[__name__], __name)

del __name, __obj

if not TYPE_CHECKING:
    # issue 38137 and python issue 43367. Submodules of a C extension are
    # non-standard, and attributes of those submodules cannot be pickled since
    # pickle expect to be able to import them as "from _C.sub import attr"
    # which fails with "_C is not a package
    def _import_extension_to_sys_modules(module, memo=None):
        if memo is None:
            memo = set()
        if module in memo:
            return
        memo.add(module)
        module_name = module.__name__
        for name in dir(module):
            member = getattr(module, name)
            member_name = getattr(member, "__name__", "")
            if inspect.ismodule(member) and member_name.startswith(module_name):
                sys.modules.setdefault(member_name, member)
                # Recurse for submodules (e.g., `_C._dynamo.eval_frame`)
                _import_extension_to_sys_modules(member, memo)

    _import_extension_to_sys_modules(_C)
    del _import_extension_to_sys_modules

################################################################################
# Define basic utilities
################################################################################


def typename(obj: object, /) -> str:
    """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        obj (object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> torch.typename(x)
        'torch.LongTensor'
        >>> torch.typename(torch.nn.Parameter)
        'torch.nn.parameter.Parameter'
    """
    if isinstance(obj, torch.Tensor):
        return obj.type()

    module = getattr(obj, "__module__", "") or ""
    qualname = ""

    if hasattr(obj, "__qualname__"):
        qualname = obj.__qualname__  # type: ignore[attr-defined]
    elif hasattr(obj, "__name__"):
        qualname = obj.__name__  # type: ignore[attr-defined]
    else:
        module = obj.__class__.__module__ or ""
        qualname = obj.__class__.__qualname__

    if module in {"", "builtins"}:
        return qualname
    return f"{module}.{qualname}"


def is_tensor(obj: object, /) -> _TypeIs["torch.Tensor"]:
    r"""Returns True if `obj` is a PyTorch tensor.

    Args:
        obj (object): Object to test
    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> torch.is_tensor(x)
        True

    """
    return isinstance(obj, torch.Tensor)


def is_storage(obj: object, /) -> _TypeGuard["TypedStorage | UntypedStorage"]:
    r"""Returns True if `obj` is a PyTorch storage object.

    Args:
        obj (Object): Object to test
    Example::

        >>> import torch
        >>> # UntypedStorage (recommended)
        >>> tensor = torch.tensor([1, 2, 3])
        >>> storage = tensor.untyped_storage()
        >>> torch.is_storage(storage)
        True
        >>>
        >>> # TypedStorage (legacy)
        >>> warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated")  # docs: hide
        >>> typed_storage = torch.TypedStorage(5, dtype=torch.float32)
        >>> torch.is_storage(typed_storage)
        True
        >>>
        >>> # regular tensor (should return False)
        >>> torch.is_storage(tensor)
        False
        >>>
        >>> # non-storage object
        >>> torch.is_storage([1, 2, 3])
        False
    """
    return type(obj) in _storage_classes


_GLOBAL_DEVICE_CONTEXT = threading.local()


def get_default_device() -> "torch.device":
    r"""Gets the default ``torch.Tensor`` to be allocated on ``device``"""
    global _GLOBAL_DEVICE_CONTEXT

    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext

    def _get_device_with_index(device: "torch.device") -> "torch.device":
        if device.index is not None:
            return device
        else:
            # TODO: Call like get_device_index() method corresponding to
            # each device type
            return torch.tensor([]).device

    # Get device from any active DeviceContext.
    device_mode = next(
        filter(
            lambda mode: isinstance(mode, DeviceContext),
            reversed(_get_current_function_mode_stack()),
        ),
        None,
    )
    if device_mode:
        device = device_mode.device
        return _get_device_with_index(device)

    device_context = getattr(_GLOBAL_DEVICE_CONTEXT, "device_context", None)
    if device_context is not None:
        return _get_device_with_index(device_context.device)
    return torch.device("cpu")


def set_default_device(device: "Device") -> None:
    """Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
    does not affect factory function calls which are called with an explicit
    ``device`` argument.  Factory calls will be performed as if they
    were passed ``device`` as an argument.

    To only temporarily change the default device instead of setting it
    globally, use ``with torch.device(device):`` instead.

    The default device is initially ``cpu``.  If you set the default tensor
    device to another device (e.g., ``cuda``) without a device index, tensors
    will be allocated on whatever the current device for the device type,
    even after :func:`torch.cuda.set_device` is called.

    .. warning::

        This function imposes a slight performance cost on every Python
        call to the torch API (not just factory functions).  If this
        is causing problems for you, please comment on
        https://github.com/pytorch/pytorch/issues/92701

    .. note::

        This doesn't affect functions that create tensors that share the same memory as the input, like:
        :func:`torch.from_numpy` and :func:`torch.frombuffer`

    Args:
        device (:class:`torch.device`, str, int, or None): the device to set as
            default, or ``None`` to clear the override. An integer is
            interpreted as an index for the current accelerator.

    Example::

        >>> # xdoctest: +SKIP("requires cuda, changes global state")
        >>> torch.get_default_device()
        device(type='cpu')
        >>> torch.set_default_device('cuda')  # current device is 0
        >>> torch.get_default_device()
        device(type='cuda', index=0)
        >>> torch.set_default_device('cuda')
        >>> torch.cuda.set_device('cuda:1')  # current device is 1
        >>> torch.get_default_device()
        device(type='cuda', index=1)
        >>> torch.set_default_device('cuda:1')
        >>> torch.get_default_device()
        device(type='cuda', index=1)

    """
    global _GLOBAL_DEVICE_CONTEXT
    if hasattr(_GLOBAL_DEVICE_CONTEXT, "device_context"):
        device_context = _GLOBAL_DEVICE_CONTEXT.device_context
        if device_context is not None:
            device_context.__exit__(None, None, None)

    if device is None:
        device_context = None
    else:
        from torch.utils._device import DeviceContext

        device_context = DeviceContext(device)
        device_context.__enter__()
    _GLOBAL_DEVICE_CONTEXT.device_context = device_context


def set_default_tensor_type(t: type["torch.Tensor"] | str, /) -> None:
    r"""
    .. warning::

        This function is deprecated as of PyTorch 2.1, please use :func:`torch.set_default_dtype()` and
        :func:`torch.set_default_device()` as alternatives.

    Sets the default ``torch.Tensor`` type to floating point tensor type
    ``t``. This type will also be used as default floating point type for
    type inference in :func:`torch.tensor`.

    The default floating point tensor type is initially ``torch.FloatTensor``.

    Args:
        t (type or string): the floating point tensor type or its name

    Example::

        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64

    """
    if isinstance(t, str):
        t = _import_dotted_name(t)
    _C._set_default_tensor_type(t)


def set_default_dtype(d: "torch.dtype", /) -> None:
    r"""

    Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
    as inputs. Other dtypes will cause torch to raise an exception.

    When PyTorch is initialized its default floating point dtype is torch.float32,
    and the intent of set_default_dtype(torch.float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating type is float16,
       the default complex dtype is complex32. For float32, the default complex dtype is complex64.
       For float64, it is complex128. For bfloat16, an exception will be raised because
       there is no corresponding complex type for bfloat16.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default.

    Example:
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> # initial default for floating point is torch.float32
        >>> # Python floats are interpreted as float32
        >>> torch.tensor([1.2, 3]).dtype
        torch.float32
        >>> # initial default for floating point is torch.complex64
        >>> # Complex Python numbers are interpreted as complex64
        >>> torch.tensor([1.2, 3j]).dtype
        torch.complex64

        >>> torch.set_default_dtype(torch.float64)
        >>> # Python floats are now interpreted as float64
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float64
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex128

        >>> torch.set_default_dtype(torch.float16)
        >>> # Python floats are now interpreted as float16
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float16
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex32

    """
    _C._set_default_dtype(d)


def use_deterministic_algorithms(
    mode: builtins.bool,
    *,
    warn_only: builtins.bool = False,
) -> None:
    r"""Sets whether PyTorch operations must use "deterministic"
    algorithms. That is, algorithms which, given the same input, and when
    run on the same software and hardware, always produce the same output.
    When enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw a
    :class:`RuntimeError` when called.

    .. note:: This setting alone is not always enough to make an application
        reproducible. Refer to :ref:`reproducibility` for more information.

    .. note:: :func:`torch.set_deterministic_debug_mode` offers an alternative
        interface for this feature.

    The following normally-nondeterministic operations will act
    deterministically when ``mode=True``:

        * :class:`torch.nn.Conv1d` when called on CUDA tensor
        * :class:`torch.nn.Conv2d` when called on CUDA tensor
        * :class:`torch.nn.Conv3d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose1d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose2d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose3d` when called on CUDA tensor
        * :class:`torch.nn.ReplicationPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad3d` when attempting to differentiate a CUDA tensor
        * :func:`torch.bmm` when called on sparse-dense CUDA tensors
        * :func:`torch.cumsum` when called on a CUDA tensor when dtype is floating point or complex
        * :func:`torch.Tensor.__getitem__` when attempting to differentiate a CPU tensor
          and the index is a list of tensors
        * :func:`torch.Tensor.index_put` with ``accumulate=False``
        * :func:`torch.Tensor.index_put` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.put_` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.scatter_add_` when called on a CUDA tensor
        * :func:`torch.gather` when called on a CUDA tensor that requires grad
        * :func:`torch.index_add` when called on CUDA tensor
        * :func:`torch.index_select` when attempting to differentiate a CUDA tensor
        * :func:`torch.repeat_interleave` when attempting to differentiate a CUDA tensor
        * :func:`torch.Tensor.index_copy` when called on a CPU or CUDA tensor
        * :func:`torch.Tensor.scatter` when `src` type is Tensor and called on CUDA tensor
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='sum'`` or ``reduce='mean'`` and called on CUDA tensor
        * :class:`torch.nn.MaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.Embedding` when attempting to differentiate a CUDA tensor

    The following normally-nondeterministic operations will throw a
    :class:`RuntimeError` when ``mode=True``:

        * :class:`torch.nn.AvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.MaxUnpool1d`
        * :class:`torch.nn.MaxUnpool2d`
        * :class:`torch.nn.MaxUnpool3d`
        * :func:`torch.nn.functional.interpolate` when attempting to differentiate a CUDA tensor
          and one of the following modes is used:

          - ``linear``
          - ``bilinear``
          - ``bicubic``
          - ``trilinear``

        * :class:`torch.nn.ReflectionPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.NLLLoss` when called on a CUDA tensor
        * :class:`torch.nn.CTCLoss` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.EmbeddingBag` when attempting to differentiate a CUDA tensor when
          ``mode='max'``
        * :func:`torch.Tensor.put_` when ``accumulate=False``
        * :func:`torch.Tensor.put_` when ``accumulate=True`` and called on a CUDA tensor
        * :func:`torch.histc` when called on a CUDA tensor
        * :func:`torch.bincount` when called on a CUDA tensor and ``weights``
          tensor is given
        * :func:`torch.median` with indices output when called on a CUDA tensor
        * :func:`torch.nn.functional.grid_sample` when attempting to differentiate a CUDA tensor
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='prod'`` and called on CUDA tensor
        * :func:`torch.Tensor.resize_` when called with a quantized tensor

    In addition, several operations fill uninitialized memory when this setting
    is turned on and when
    :attr:`torch.utils.deterministic.fill_uninitialized_memory` is turned on.
    See the documentation for that attribute for more information.

    Note that deterministic operations tend to have worse performance than
    nondeterministic operations.


    When this setting is turned on, the Inductor deterministic mode is also tuned on
    automatically. In deterministic mode, Inductor would avoid doing on device benchmarking
    that affect numerics. This includes:

      - don't pad matmul input shapes. Without enabling deterministic mode, Inductor would do
        benchmarking to check if padding matmul shape is beneficial.
      - don't autotune templates. Inductor has templates for kernels like matmul/conv/attention.
        Without enabling deterministic mode, Inductor would do autotuning to
        pick the best configs for those templates and adopt it if it's faster
        than the kernel in eager mode. In deterministic mode, we pick the eager kernel.
      - don't autotune triton configs for reduction. Reduction numerics are
        very sensitive to triton configs. In deterministic mode, Inductor
        will use some heuristics to pick the most promising configs rather
        than do autotuning.
      - Skip autotuning for reduction in coordinate descent tuning.
      - Don't benchmark for the computation/communication reordering pass
      - Disable the feature that dynamically scale down RBLOCK triton config for higher
        occupancy.


    .. note::

        This flag does not detect or prevent nondeterministic behavior caused
        by calling an inplace operation on a tensor with an internal memory
        overlap or by giving such a tensor as the :attr:`out` argument for an
        operation. In these cases, multiple writes of different data may target
        a single memory location, and the order of writes is not guaranteed.

    Args:
        mode (:class:`bool`): If True, makes potentially nondeterministic
            operations switch to a deterministic algorithm or throw a runtime
            error. If False, allows nondeterministic operations.

    Keyword args:
        warn_only (:class:`bool`, optional): If True, operations that do not
            have a deterministic implementation will throw a warning instead of
            an error. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> torch.use_deterministic_algorithms(True)

        # Backward mode nondeterministic error
        >>> torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True).cuda()).sum().backward()
        ...
        RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation...
    """
    import torch._inductor.config as inductor_config

    inductor_config.deterministic = mode
    _C._set_deterministic_algorithms(mode, warn_only=warn_only)


def are_deterministic_algorithms_enabled() -> builtins.bool:
    r"""Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """
    return _C._get_deterministic_algorithms()


def is_deterministic_algorithms_warn_only_enabled() -> builtins.bool:
    r"""Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`torch.use_deterministic_algorithms` documentation for more
    details.
    """
    return _C._get_deterministic_algorithms_warn_only()


def set_deterministic_debug_mode(debug_mode: builtins.int | str) -> None:
    r"""Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`torch.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """

    # NOTE: builtins.int is used here because int in this scope resolves
    # to torch.int
    if not isinstance(debug_mode, (builtins.int, str)):
        raise TypeError(f"debug_mode must be str or int, but got {type(debug_mode)}")

    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, "
                f"`warn`, `error`, but got {debug_mode}"
            )

    if debug_mode == 0:
        _C._set_deterministic_algorithms(False)
    elif debug_mode == 1:
        _C._set_deterministic_algorithms(True, warn_only=True)
    elif debug_mode == 2:
        _C._set_deterministic_algorithms(True)
    else:
        raise RuntimeError(
            f"invalid value of debug_mode, expected 0, 1, or 2, but got {debug_mode}"
        )


def get_deterministic_debug_mode() -> builtins.int:
    r"""Returns the current value of the debug mode for deterministic
    operations. Refer to :func:`torch.set_deterministic_debug_mode`
    documentation for more details.
    """

    if _C._get_deterministic_algorithms():
        if _C._get_deterministic_algorithms_warn_only():
            return 1
        else:
            return 2
    else:
        return 0


def get_float32_matmul_precision() -> str:
    r"""Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """
    return _C._get_float32_matmul_precision()


def set_float32_matmul_precision(precision: str) -> None:
    r"""Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype (24 mantissa
          bits with 23 bits explicitly stored) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    When using "high" precision, float32 multiplications may use a bfloat16-based algorithm
    that is more complicated than simply truncating to some smaller number mantissa bits
    (e.g. 10 for TensorFloat32, 7 for bfloat16 explicitly stored).  Refer to [Henry2019]_ for a complete
    description of this algorithm.  To briefly explain here, the first step is to realize
    that we can perfectly encode a single float32 number as the sum of three bfloat16
    numbers (because float32 has 23 mantissa bits while bfloat16 has 7 explicitly stored, and both have the
    same number of exponent bits).  This means that the product of two float32 numbers can
    be exactly given by the sum of nine products of bfloat16 numbers.  We can then trade
    accuracy for speed by dropping some of these products.  The "high" precision algorithm
    specifically keeps only the three most significant products, which conveniently excludes
    all of the products involving the last 8 mantissa bits of either input.  This means that
    we can represent our inputs as the sum of two bfloat16 numbers rather than three.
    Because bfloat16 fused-multiply-add (FMA) instructions are typically >10x faster than
    float32 ones, it's faster to do three multiplications and 2 additions with bfloat16
    precision than it is to do a single multiplication with float32 precision.

    .. [Henry2019] http://arxiv.org/abs/1904.06376

    .. note::

        This does not change the output dtype of float32 matrix multiplications,
        it controls how the internal computation of the matrix multiplication is performed.

    .. note::

        This does not change the precision of convolution operations. Other flags,
        like `torch.backends.cudnn.allow_tf32`, may control the precision of convolution
        operations.

    .. note::

        This flag currently only affects one native device type: CUDA.
        If "high" or "medium" are set then the TensorFloat32 datatype will be used
        when computing float32 matrix multiplications, equivalent to setting
        `torch.backends.cuda.matmul.allow_tf32 = True`. When "highest" (the default)
        is set then the float32 datatype is used for internal computations, equivalent
        to setting `torch.backends.cuda.matmul.allow_tf32 = False`.

    Args:
        precision(str): can be set to "highest" (default), "high", or "medium" (see above).

    """
    _C._set_float32_matmul_precision(precision)


def set_warn_always(b: builtins.bool, /) -> None:
    r"""When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """
    _C._set_warnAlways(b)


def is_warn_always_enabled() -> builtins.bool:
    r"""Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """
    return _C._get_warnAlways()


################################################################################
# Define error checking functions
################################################################################

# These error checking functions must be kept consistent with their C++
# equivalents. Their C++ equivalents are mentioned where applicable.


def _check_with(
    error_type: type[BaseException],
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None,
) -> None:
    if not isinstance(cond, (builtins.bool, SymBool)):
        if isinstance(cond, torch.Tensor):
            raise TypeError(
                "cond must be a bool, but got a Tensor. "
                "For tensor conditions, use torch._check_tensor_all() "
                "to assert that all elements are true."
            )
        raise TypeError(f"cond must be a bool, but got {type(cond)}")

    from torch.fx.experimental.symbolic_shapes import expect_true

    if expect_true(cond):
        return

    # error_type must be a subclass of Exception and not subclass of Warning
    if not issubclass(error_type, Exception) or issubclass(error_type, Warning):
        raise AssertionError(
            f"error_type must be a subclass of Exception but not Warning, got {error_type}"
        )

    if message is None:
        message_evaluated = (
            "Expected cond to be True, but got False. (Could this error "
            "message be improved? If so, please report an enhancement request "
            "to PyTorch.)"
        )

    else:
        # message may be a callable (deferred construction) or a plain
        # str/object; both are accepted.
        message_evaluated = str(message() if callable(message) else message)

    raise error_type(message_evaluated)


def _check(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``RuntimeError``

    C++ equivalent: ``TORCH_CHECK``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(RuntimeError, cond, message)


@_deprecated(
    "_check_is_size will be removed in a future PyTorch release along with guard_size_oblivious. \
    Use _check(i >= 0) instead.",
    category=FutureWarning,
)
def _check_is_size(
    i: "IntLikeType",
    message: _LiteralString | _Callable[[], object] | None = None,
    *,
    max: "IntLikeType | None" = None,
) -> None:
    """Checks that a given integer is a valid size (i.e., is non-negative).
    You should use this over ``_check(i >= 0)`` because it can prevent
    ``GuardOnDataDependentSymNode`` exceptions by opting yourself into alternate
    semantics for ``guard_size_oblivious`` tests that treat values 0 and 1
    equivalently to all other values.

    When max is not None, this specifies an upper bound equivalent to
    ``_check(i <= max)``.  This bound is also subject to alternate semantics:
    in ``guard_size_oblivious`` tests, we assume that a constant max bound is
    treated equivalently to all other values.  Symbolic max bounds are not yet
    supported.

    NB: Do NOT use this in contexts where a -1 size would be valid (indicating
    to infer the size from context, or if you should wrap-around or truncate).
    Only use this if the only valid value is an honest to goodness size.
    """
    # This is responsible for the expect_true
    _check(i >= 0, message)
    from torch.fx.experimental.symbolic_shapes import _advise_is_size

    _advise_is_size(i)  # type: ignore[arg-type]

    if max is not None:
        _check(i <= max, message)

        from torch.fx.experimental.symbolic_shapes import _advise_is_bounded

        _advise_is_bounded(i, max)  # type: ignore[arg-type]


def _check_index(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``IndexError``

    C++ equivalent: ``TORCH_CHECK_INDEX``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(IndexError, cond, message)


def _check_value(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``ValueError``

    C++ equivalent: ``TORCH_CHECK_VALUE``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(ValueError, cond, message)


def _check_type(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``TypeError``

    C++ equivalent: ``TORCH_CHECK_TYPE``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(TypeError, cond, message)


def _check_not_implemented(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``NotImplementedError``

    C++ equivalent: ``TORCH_CHECK_NOT_IMPLEMENTED``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(NotImplementedError, cond, message)


def _check_tensor_all_with(
    error_type: type[BaseException],
    cond: "torch.Tensor",
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    if not is_tensor(cond):
        raise TypeError(f"cond must be a tensor, but got {type(cond)}")

    if not cond.dtype == torch.bool:
        raise TypeError(f"cond tensor must have dtype torch.bool, but got {cond.dtype}")

    _check_with(error_type, cond._is_all_true().item(), message)  # type: ignore[arg-type]


# C++ equivalent: `TORCH_CHECK_TENSOR_ALL`
def _check_tensor_all(
    cond: "torch.Tensor",
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``RuntimeError``

    C++ equivalent: ``TORCH_CHECK_TENSOR_ALL``

    Args:
        cond (:class:`torch.Tensor`): Tensor of dtype ``torch.bool``. If any
            element is ``False``, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_tensor_all_with(RuntimeError, cond, message)


################################################################################
# Define numeric constants
################################################################################

# For Python Array API (https://data-apis.org/array-api/latest/API_specification/constants.html) and
# NumPy consistency (https://numpy.org/devdocs/reference/constants.html)
from math import e, inf, nan, pi


newaxis: None = None

__all__.extend(["e", "pi", "nan", "inf", "newaxis"])

################################################################################
# Define Storage and Tensor classes
################################################################################

from torch._tensor import Tensor  # usort: skip

# needs to be after torch.Tensor is defined to avoid circular dependencies
from torch import storage as storage  # usort: skip
from torch.storage import (
    _LegacyStorage,
    _StorageBase,
    _warn_typed_storage_removal,
    TypedStorage,
    UntypedStorage,
)


# NOTE: New <type>Storage classes should never be added. When adding a new
# dtype, use torch.storage.TypedStorage directly.
class ByteStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.uint8


class DoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.double


class FloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.float


class HalfStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.half


class LongStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.long


class IntStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.int


class ShortStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.short


class CharStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.int8


class BoolStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.bool


class BFloat16Storage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.bfloat16


class ComplexDoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.cdouble


class ComplexFloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.cfloat


class QUInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.quint8


class QInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.qint8


class QInt32Storage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.qint32


class QUInt4x2Storage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.quint4x2


class QUInt2x4Storage(_LegacyStorage):
    @classproperty
    def dtype(self) -> "torch.dtype":
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        return torch.quint2x4


_storage_classes: set[type[TypedStorage | UntypedStorage]] = {
    UntypedStorage,
    DoubleStorage,
    FloatStorage,
    LongStorage,
    IntStorage,
    ShortStorage,
    CharStorage,
    ByteStorage,
    HalfStorage,
    BoolStorage,
    QUInt8Storage,
    QInt8Storage,
    QInt32Storage,
    BFloat16Storage,
    ComplexFloatStorage,
    ComplexDoubleStorage,
    QUInt4x2Storage,
    QUInt2x4Storage,
    TypedStorage,
}

# The _tensor_classes set is initialized by the call to initialize_python_bindings.
_tensor_classes: set[type["torch.Tensor"]] = set()

# If you edit these imports, please update torch/__init__.py.in as well
from torch import amp as amp, random as random, serialization as serialization
from torch._tensor_str import set_printoptions
from torch.amp import autocast, GradScaler
from torch.random import (
    get_rng_state,
    initial_seed,
    manual_seed,
    seed,
    set_rng_state,
    thread_safe_generator,
)
from torch.serialization import load, save


################################################################################
# Initialize extension
################################################################################


# Shared memory manager needs to know the exact location of manager executable
def _manager_path() -> bytes:
    if platform.system() == "Windows":
        return b""
    path = get_file_path("torch", "bin", "torch_shm_manager")
    prepare_multiprocessing_environment(get_file_path("torch"))
    if not os.path.exists(path):
        raise RuntimeError("Unable to find torch_shm_manager at " + path)
    return path.encode("utf-8")


_C._initExtension(_manager_path())  # pyrefly: ignore[bad-argument-type]

del _manager_path

# Appease the type checker: it can't deal with direct setting of globals().
# Note that we will see "too many" functions when reexporting this way; there
# is not a good way to fix this problem.  Perhaps, try to redesign VariableFunctions
# so that this import is good enough
if TYPE_CHECKING:
    # Some type signatures pulled in from _VariableFunctions here clash with
    # signatures already imported. For now these clashes are ignored; see
    # PR #43339 for details.
    from torch._C._VariableFunctions import *  # type: ignore[assignment, misc] # noqa: F403

    # Fixup segment_reduce visibility
    _segment_reduce = segment_reduce
    del segment_reduce  # noqa: F821

# Ops not to be exposed in `torch` namespace,
# mostly helper ops.
PRIVATE_OPS = ("unique_dim",)

__name, __obj = "", None
for __name in dir(_C._VariableFunctions):
    if __name.startswith("__") or __name in PRIVATE_OPS:
        continue
    __obj = getattr(_C._VariableFunctions, __name)
    __obj.__module__ = __name__  # "torch"
    # Hide some APIs that should not be public
    if __name == "segment_reduce":
        # TODO: Once the undocumented FC window is passed, remove the line below
        globals()[__name] = __obj
        __name = "_" + __name
    globals()[__name] = __obj
    if not __name.startswith("_"):
        __all__.append(__name)

del __name, __obj

################################################################################
# Add torch.dtype instances to the public API
################################################################################

import torch


__all__.extend(
    # pyrefly: ignore [unresolvable-dunder-all]
    name
    for name in dir(torch)
    if isinstance(getattr(torch, name), torch.dtype)
)

################################################################################
# Import TorchDynamo's lazy APIs to avoid circular dependencies
################################################################################

# needs to be before from torch.functional import * to avoid circular dependencies
from torch._compile import _disable_dynamo  # usort: skip

################################################################################
# Import interface functions defined in Python
################################################################################

# needs to be after the above ATen bindings so we can overwrite from Python side
from torch import _VF as _VF, functional as functional  # usort: skip
from torch.functional import *  # usort: skip # noqa: F403

################################################################################
# Remove unnecessary members
################################################################################

del _StorageBase
del _LegacyStorage

################################################################################
# Define _assert
################################################################################


# needs to be before the submodule imports to avoid circular dependencies
def _assert(condition: object, message: object) -> None:
    r"""A wrapper around Python's assert which is symbolically traceable."""
    if type(condition) is not torch.Tensor and overrides.has_torch_function(
        (condition,)
    ):
        return overrides.handle_torch_function(
            _assert, (condition,), condition, message
        )
    if not condition:
        raise AssertionError(message)


################################################################################
# Import most common subpackages
################################################################################

# Use the redundant form so that type checkers know that these are a part of
# the public API. The "regular" import lines are there solely for the runtime
# side effect of adding to the imported module's members for other users.

# needs to be before import torch.nn as nn to avoid circular dependencies
from torch.autograd import (  # usort: skip
    enable_grad as enable_grad,
    inference_mode as inference_mode,
    no_grad as no_grad,
    set_grad_enabled as set_grad_enabled,
)

from torch import (
    __config__ as __config__,
    __future__ as __future__,
    _awaits as _awaits,
    accelerator as accelerator,
    autograd as autograd,
    backends as backends,
    cpu as cpu,
    cuda as cuda,
    distributed as distributed,
    distributions as distributions,
    fft as fft,
    futures as futures,
    hub as hub,
    jit as jit,
    linalg as linalg,
    mps as mps,
    mtia as mtia,
    multiprocessing as multiprocessing,
    nested as nested,
    nn as nn,
    optim as optim,
    overrides as overrides,
    profiler as profiler,
    sparse as sparse,
    special as special,
    testing as testing,
    types as types,
    utils as utils,
    version as version,
    xpu as xpu,
)
from torch.signal import windows as windows


# Quantized, sparse, AO, etc. should be last to get imported, as nothing
# is expected to depend on them.
from torch import ao as ao  # usort: skip

# nn.quant* depends on ao -- so should be after those.
import torch.nn.intrinsic
import torch.nn.qat
import torch.nn.quantizable
import torch.nn.quantized


_C._init_names(list(_storage_classes))

# attach docstrings to torch and tensor functions
from torch import _size_docs, _storage_docs, _tensor_docs, _torch_docs


del _torch_docs, _tensor_docs, _storage_docs, _size_docs


def compiled_with_cxx11_abi() -> builtins.bool:
    r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
    return True


from torch import _library as _library, _ops as _ops


# Import the ops and classes "namespace"
from torch._ops import ops as ops  # usort: skip
from torch._classes import classes as classes  # usort: skip

sys.modules.setdefault(f"{__name__}.ops", ops)
sys.modules.setdefault(f"{__name__}.classes", classes)

if hasattr(torch._C, "_c10d_init"):
    from torch.distributed.distributed_c10d import _register_process_group_opaque_type

    _register_process_group_opaque_type()
    del _register_process_group_opaque_type

# quantization depends on torch.fx and torch.ops
# Import quantization
from torch import quantization as quantization  # usort: skip

# Import the quasi random sampler
from torch import quasirandom as quasirandom  # usort: skip

# If you are seeing this, it means that this call site was not checked if
# the memory format could be preserved, and it was switched to old default
# behaviour of contiguous
legacy_contiguous_format = contiguous_format  # defined by _C._initExtension()

# Register fork handler to initialize OpenMP in child processes (see gh-28389)
from torch.multiprocessing._atfork import register_after_fork


register_after_fork(torch.get_num_threads)
del register_after_fork

# Import tools that require fully imported torch (for applying
# torch.jit.script as a decorator, for instance):
from torch._lobpcg import lobpcg as lobpcg


# These were previously defined in native_functions.yaml and appeared on the
# `torch` namespace, but we moved them to c10 dispatch to facilitate custom
# class usage. We add these lines here to preserve backward compatibility.
quantized_lstm = ops.aten.quantized_lstm
quantized_gru = ops.aten.quantized_gru

# Import experimental masked operations support. See
# [RFC-0016](https://github.com/pytorch/rfcs/pull/27) for more
# information.
from torch import masked as masked

# Import removed ops with error message about removal
from torch._linalg_utils import (  # type: ignore[misc]
    _symeig as symeig,
    cholesky,
    eig,
    lstsq,
    matrix_rank,
    qr,
    solve,
)
from torch.utils.dlpack import from_dlpack, to_dlpack


class _TorchCompileInductorWrapper:
    compiler_name = "inductor"

    def __init__(
        self,
        mode: str | None,
        options: dict[str, _Any] | None,
        dynamic: builtins.bool | None,
        name: str | None = None,
    ) -> None:
        from torch._inductor.compiler_bisector import CompilerBisector

        self.config: dict[str, _Any] = {}
        self.dynamic = dynamic
        self.name = name
        self.apply_mode(mode)
        self.apply_options(options)
        self.apply_options(CompilerBisector.get_config_change("inductor"))

        cuda_version = None
        if hasattr(torch, "version"):
            from torch.torch_version import TorchVersion

            cuda_version = TorchVersion(getattr(torch.version, "cuda", "0.0"))

        if self.config.get("triton.cudagraphs", False) and (
            (cuda_version and cuda_version < "12.6")
            or not profiler_allow_cudagraph_cupti_lazy_reinit_cuda12()
        ):
            os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"
            # FIXME: CUDA Graph does not work well with CUPTI teardown.
            #   1) crashes on 1st lazy CUPTI re-init after teardown (CUDA 11)
            #   2) crashes on 2nd non-lazy CUPTI re-init after teardown (CUDA 12)
            # Workaround: turn off CUPTI teardown when using CUDA Graphs.
            os.environ["TEARDOWN_CUPTI"] = "0"

    def __eq__(self, other: object) -> builtins.bool:
        return (
            isinstance(other, _TorchCompileInductorWrapper)
            and self.config == other.config
            and self.dynamic == other.dynamic
            and self.name == other.name
        )

    def apply_mode(self, mode: str | None) -> None:
        if mode and mode != "default":
            from torch._inductor import list_mode_options

            self.apply_options(list_mode_options(mode, self.dynamic))

    def apply_options(self, options: dict[str, _Any] | None) -> None:
        if not options:
            return

        from torch._inductor import config

        current_config: dict[str, _Any] = config.get_config_copy()

        for key, val in options.items():
            attr_name = key.replace("-", "_")
            if attr_name not in current_config:
                raise RuntimeError(
                    f"Unexpected optimization option {key}, known options are {list(current_config.keys())}"
                )
            attr_type = config.get_type(attr_name)  # type: ignore[attr-defined]
            # Subscriptable generic types don't support isinstance so skip the type
            # check. There doesn't seem to be a good way of checking membership without
            # 3rd party libraries.
            if _get_origin(attr_type) is None:
                if not isinstance(val, attr_type):
                    val_type_str = type(val).__name__
                    expected_type_str = type(current_config[attr_name]).__name__
                    raise RuntimeError(
                        f"Unexpected type of attr {key}, got {val_type_str} should be {expected_type_str}"
                    )
            self.config[attr_name] = val

    def __call__(
        self,
        model_: _Any,
        inputs_: _Any,
        *,
        config_patches: dict[str, _Any] | None = None,
    ) -> _Any:
        from torch._inductor.compile_fx import compile_fx

        all_patches = {**self.config, **(config_patches or {})}
        return compile_fx(
            model_,
            inputs_,
            config_patches=all_patches,
            compile_region_name=self.name,
        )

    def get_compiler_config(self) -> dict[str, _Any]:
        from torch._inductor.compile_fx import get_patched_config_dict

        return get_patched_config_dict(config_patches=self.config)

    def reset(self) -> None:
        from torch._inductor import config

        if "triton.cudagraphs" in self.config or config.triton.cudagraphs:
            if self.config.get("triton.cudagraphs", True):
                from torch._inductor.cudagraph_trees import reset_cudagraph_trees

                reset_cudagraph_trees()


class _TorchCompileAOTInductorWrapper(_TorchCompileInductorWrapper):
    compiler_name = "aotinductor"

    def __init__(
        self,
        mode: str | None,
        options: dict[str, _Any] | None,
        dynamic: builtins.bool | None,
        name: str | None = None,
    ) -> None:
        super().__init__(mode, options, dynamic, name)
        self.apply_options({"cpp_wrapper": True})
        self.apply_options({"aot_inductor.package": True})

    def __call__(
        self,
        model_: _Any,
        inputs_: _Any,
        *,
        config_patches: dict[str, _Any] | None = None,
    ) -> _Any:
        from contextlib import nullcontext
        from unittest import mock

        from torch._guards import detect_fake_mode
        from torch._inductor.virtualized import V

        fake_mode = detect_fake_mode(inputs_)
        ctx = (
            mock.patch.object(fake_mode, "allow_non_fake_inputs", True)
            if fake_mode
            else nullcontext()
        )
        with (
            V.set_aot_compilation(True),
            ctx,
            torch._inductor.config.patch("enable_autograd_for_aot", True),
        ):
            return super().__call__(model_, inputs_, config_patches=config_patches)


class _TorchCompileWrapper:
    def __init__(
        self,
        backend: str | _Callable[..., _Any],
        mode: str | None,
        options: dict[str, _Any] | None,
        dynamic: builtins.bool | None,
    ) -> None:
        from torch._dynamo.backends.registry import lookup_backend

        if isinstance(backend, str):
            self.compiler_name = backend
        elif hasattr(backend, "__name__"):
            self.compiler_name = backend.__name__
        else:
            self.compiler_name = str(backend)
        self.dynamic = dynamic
        self.compiler_fn = lookup_backend(backend)
        self.kwargs: dict[str, _Any] = {}
        # only pass the args if they non-empty
        if mode and mode != "default":
            self.kwargs["mode"] = mode
        if options:
            self.kwargs["options"] = options

    def __eq__(self, other: object) -> builtins.bool:
        return (
            isinstance(other, _TorchCompileWrapper)
            and self.compiler_fn == other.compiler_fn
            and self.kwargs == other.kwargs
            and self.dynamic == other.dynamic
        )

    def __call__(self, model_: _Any, inputs_: _Any) -> _Any:
        return self.compiler_fn(model_, inputs_, **self.kwargs)

    def reset(self) -> None:
        if hasattr(self.compiler_fn, "reset"):
            self.compiler_fn.reset()


_InputT = _ParamSpec("_InputT")
_RetT = _TypeVar("_RetT")


@_overload
def compile(
    model: _Callable[_InputT, _RetT],
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | _Callable[..., _Any] = "inductor",
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = None,
    name: str | None = None,
    disable: builtins.bool = False,
    dynamic_shapes: _Any = None,
) -> _Callable[_InputT, _RetT]: ...


@_overload
def compile(
    model: None = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | _Callable[..., _Any] = "inductor",
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = None,
    name: str | None = None,
    disable: builtins.bool = False,
    dynamic_shapes: _Any = None,
) -> _Callable[[_Callable[_InputT, _RetT]], _Callable[_InputT, _RetT]]: ...


def compile(
    model: _Callable[_InputT, _RetT] | None = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | _Callable[..., _Any] | None = None,
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = None,
    name: str | None = None,
    disable: builtins.bool = False,
    recompile_limit: builtins.int | None = None,
    isolate_recompiles: builtins.bool = False,
    dynamic_shapes: _Any = None,
) -> (
    _Callable[[_Callable[_InputT, _RetT]], _Callable[_InputT, _RetT]]
    | _Callable[_InputT, _RetT]
):
    """
    Optimizes given model/function using TorchDynamo and specified backend.
    If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile`
    to compile the module inplace without changing its structure.

    Concretely, for every frame executed within the compiled region, we will attempt
    to compile it and cache the compiled result on the code object for future
    use.  A single frame may be compiled multiple times if previous compiled
    results are not applicable for subsequent calls (this is called a "guard
    failure"), you can use TORCH_LOGS=guards to debug these situations.
    Multiple compiled results can be associated with a frame up to
    ``torch._dynamo.config.recompile_limit``, which defaults to 8; at which
    point we will fall back to eager.  Note that compile caches are per
    *code object*, not frame; if you dynamically create multiple copies of a
    function, they will all share the same code cache.

    Args:
       model (Callable or None): Module/function to optimize
       fullgraph (bool): If False (default), torch.compile attempts to discover compilable regions
        in the function that it will optimize. If True, then we require that the entire function be
        capturable into a single graph. If this is not possible (that is, if there are graph breaks),
        then this will raise an error. This also opts into unbacked semantics, notably it will turn on
        capture_scalar_outputs and capture_dynamic_output_shape_ops on by default.
       dynamic (bool or None): Use dynamic shape tracing.  When this is True, we will up-front attempt
        to generate a kernel that is as dynamic as possible to avoid recompilations when
        sizes change.  This may not always work as some operations/optimizations will
        force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
        When this is False, we will NEVER generate dynamic kernels, we will always specialize.
        By default (None), we automatically detect if dynamism has occurred and compile a more
        dynamic kernel upon recompile.
       backend (str or Callable): backend to be used

        - "inductor" is the default backend, which is a good balance between performance and overhead

        - Non experimental in-tree backends can be seen with `torch._dynamo.list_backends()`

        - Experimental or debug in-tree backends can be seen with `torch._dynamo.list_backends(None)`

        - To register an out-of-tree custom backend:
          https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html#registering-custom-backends
       mode (str): Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

        - "default" is the default mode, which is a good balance between performance and overhead

        - "reduce-overhead" is a mode that reduces the overhead of python with CUDA graphs,
          useful for small batches.  Reduction of overhead can come at the cost of more memory
          usage, as we will cache the workspace memory required for the invocation so that we
          do not have to reallocate it on subsequent runs.  Reduction of overhead is not guaranteed
          to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
          There are other circumstances where CUDA graphs are not applicable; use TORCH_LOGS=perf_hints
          to debug.

        - "max-autotune" is a mode that leverages Triton or template based matrix multiplications
          on supported devices and Triton based convolutions on GPU.
          It enables CUDA graphs by default on GPU.

        - "max-autotune-no-cudagraphs" is a mode similar to "max-autotune" but without CUDA graphs

        - To see the exact configs that each mode sets you can call `torch._inductor.list_mode_options()`

       options (dict): A dictionary of options to pass to the backend. Some notable ones to try out are

        - `epilogue_fusion` which fuses pointwise ops into templates. Requires `max_autotune` to also be set

        - `max_autotune` which will profile to pick the best matmul configuration

        - `fallback_random` which is useful when debugging accuracy issues

        - `shape_padding` which pads matrix shapes to better align loads on GPUs especially for tensor cores

        - `triton.cudagraphs` which will reduce the overhead of python with CUDA graphs

        - `trace.enabled` which is the most useful debugging flag to turn on

        - `trace.graph_diagram` which will show you a picture of your graph after fusion

        - `guard_filter_fn` that controls which dynamo guards are saved with compilations.
          This is an unsafe feature and there is no backward compatibility guarantee provided
          for dynamo guards as data types.
          For stable helper functions to use, see the documentation in `torch.compiler`, for example:
          - `torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe`
          - `torch.compiler.skip_guard_on_all_nn_modules_unsafe`
          - `torch.compiler.keep_tensor_guards_unsafe`

        - For inductor you can see the full list of configs that it supports by calling `torch._inductor.list_options()`
       name (str or None): Optional identifier for the compiled region. When supported by downstream
        tooling, this is surfaced on wrapped compiled-region higher-order operators and other debug metadata.
       disable (bool): Turn torch.compile() into a no-op for testing
       recompile_limit (int or None): Maximum number of recompilations allowed for this
        ``torch.compile()`` call before falling back to eager. If None (default), uses
        the global ``torch._dynamo.config.recompile_limit`` (default 8). With
        ``fullgraph=True``, exceeding the limit raises ``FailOnRecompileLimitHit``.
       isolate_recompiles (bool): If True, this ``torch.compile()`` call tracks
        recompilations independently. By default, all ``torch.compile()`` calls on the
        same function share a single set of compiled entries, so one call's
        recompilations count against every other call's limit. With
        ``isolate_recompiles=True``, each call gets its own isolated set of entries.
        Lookups for an isolated compile call will still fall back to entries from
        non-isolated compile calls (BC-friendly reuse), but new compilations are
        stored separately. ``recompile_limit`` is checked per-region;
        ``accumulated_recompile_limit`` is a global cap across all regions.
        Default-strategy behavior is asymmetric: SKIP decisions (from
        ``skip_code``, ``@torch._dynamo.skip``, FX-generated code, etc.) are
        inherited by isolated regions — those remain skipped. RUN_ONLY persisted
        by a non-isolated region hitting its recompile limit does NOT bleed
        into isolated regions — each region manages its own RUN_ONLY state.
        Default False.

    Example::

        @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
        def foo(x):
            return torch.sin(x) + torch.cos(x)

    """
    import sysconfig

    _C._log_api_usage_once("torch.compile")
    if sys.version_info >= (3, 15):
        raise RuntimeError("torch.compile is not supported on Python 3.15+")
    elif sysconfig.get_config_var("Py_GIL_DISABLED") == 1 and sys.version_info < (
        3,
        13,
        3,
    ):
        raise RuntimeError(
            "torch.compile is not supported on Python < 3.13.3 built with GIL disabled. "
            "Please use Python 3.13.3+."
        )

    if backend is None:
        from torch._dynamo.backends.registry import get_default_backend

        backend = get_default_backend()

    # Auto-wrap ParamsSpec → ShapesSpec for convenience
    if dynamic_shapes is not None:
        from torch.fx.experimental.dynamic_spec import ParamsSpec, ShapesSpec

        if isinstance(dynamic_shapes, ParamsSpec):
            dynamic_shapes = ShapesSpec(dynamic_shapes)

    # If ``model`` carries an ``@dynamic_spec(...)`` decorator, the attached
    # ``ShapesSpec`` is used as ``dynamic_shapes``. Passing both raises.
    if model is not None:
        from torch.fx.experimental.dynamic_spec import _resolve_dynamic_shapes

        dynamic_shapes = _resolve_dynamic_shapes(model, dynamic_shapes)

    # Decorator mode
    if model is None:

        def fn(model: _Callable[_InputT, _RetT]) -> _Callable[_InputT, _RetT]:
            if model is None:
                raise RuntimeError("Model can't be None")
            return compile(  # pyrefly: ignore  # no-matching-overload
                model,
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
                mode=mode,
                options=options,
                name=name,
                disable=disable,
                recompile_limit=recompile_limit,
                isolate_recompiles=isolate_recompiles,
                dynamic_shapes=dynamic_shapes,
            )

        return fn

    if mode is not None and options is not None:
        raise RuntimeError(
            "Either mode or options can be specified, but both can't be specified at the same time."
        )
    if mode is None and options is None:
        mode = "default"

    from torch._inductor.compiler_bisector import CompilerBisector

    if bisect_backend := CompilerBisector.get_backend():
        import torch._inductor.config as inductor_config

        # don't override the backend for use cases like vllm
        # which leverages their custom backend.
        if not (
            inductor_config.test_configs.bisect_keep_custom_backend_for_inductor
            and bisect_backend == "inductor"
            and not isinstance(backend, str)
        ):
            backend = bisect_backend

    guard_filter_fn = None
    use_aoti = False
    if options and isinstance(options, dict):
        guard_filter_fn = options.pop("guard_filter_fn", None)
        use_aoti = options.pop("use_aoti", False)

    if torch.compiler.is_exporting():
        from torch._higher_order_ops.utils import _in_hop_compile

        if not _in_hop_compile():
            warnings.warn(
                "torch.compile is ignored when called inside torch.export region",
                stacklevel=2,
            )
            # torch.compile is a no-op when inside torch.export region
            return model

    if backend == "inductor":
        if use_aoti:
            backend = _TorchCompileAOTInductorWrapper(mode, options, dynamic, name)
        else:
            backend = _TorchCompileInductorWrapper(mode, options, dynamic, name)
    else:
        backend = _TorchCompileWrapper(backend, mode, options, dynamic)

    return torch._dynamo.optimize(
        backend=backend,
        nopython=fullgraph,
        dynamic=dynamic,
        disable=disable,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        isolate_recompiles=isolate_recompiles,
        dynamic_shapes=dynamic_shapes,
    )(model)  # type: ignore[return-value]


def _register_device_module(device_type: str, module: _ModuleType) -> None:
    r"""Register an external runtime module of the specific :attr:`device_type`
    supported by torch.

    After the :attr:`module` is registered correctly, the user can refer
    the external runtime module as part of torch with attribute torch.xxx.
    """
    # Make sure the device_type represent a supported device type for torch.
    device_type = torch.device(device_type).type
    m = sys.modules[__name__]
    if hasattr(m, device_type):
        raise RuntimeError(
            f"The runtime module of '{device_type}' has already "
            f"been registered with '{getattr(m, device_type)}'"
        )
    setattr(m, device_type, module)
    torch_module_name = ".".join([__name__, device_type])
    sys.modules[torch_module_name] = module


from torch import (
    export as export,
    func as func,
    library as library,
    return_types as return_types,
)
from torch._higher_order_ops import cond as cond, while_loop as while_loop
from torch.func import vmap as vmap


if not TYPE_CHECKING:
    # register python metas for distributed ops
    # Only import if distributed is available (USE_DISTRIBUTED=1)
    if hasattr(torch._C, "_c10d_init"):
        import torch.distributed._meta_registrations as coll_meta_registrations

        del coll_meta_registrations
    from torch import _meta_registrations

# Enable CUDA Sanitizer
if "TORCH_CUDA_SANITIZER" in os.environ:
    import torch.cuda._sanitizer as csan

    csan.enable_cuda_sanitizer()

# Populate magic methods on SymInt and SymFloat
import torch.fx.experimental.sym_node
from torch import compiler as compiler, fx as fx


class _TritonLibrary:
    lib = torch.library.Library("triton", "DEF")
    ops_table: dict[tuple[str, str], _Callable[..., _Any]] = {}

    @classmethod
    def registerOp(
        cls,
        op_key: str,
        full_schema: str,
        op_impl: _Callable[..., _Any],
        dispatch_key: str,
    ) -> _Callable[..., _Any]:
        if (op_key, dispatch_key) not in cls.ops_table:
            cls.lib.define(full_schema)
            cls.lib.impl("triton::" + op_key, op_impl, dispatch_key)
            cls.ops_table[(op_key, dispatch_key)] = op_impl

        return cls.ops_table[(op_key, dispatch_key)]


# Deprecated attributes
_deprecated_attrs = {
    "has_mps": torch.backends.mps.is_built,
    "has_cuda": torch.backends.cuda.is_built,
    "has_cudnn": torch.backends.cudnn.is_available,
    "has_mkldnn": torch.backends.mkldnn.is_available,
}

if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.
    from torch import (
        _dynamo as _dynamo,
        _inductor as _inductor,
        _subclasses as _subclasses,
        onnx as onnx,
    )

else:
    _lazy_modules = {
        "_dynamo",
        "_inductor",
        "_export",
        # ONNX must be imported after _dynamo, _ops, _subclasses, fx, func and jit
        "onnx",
    }

    def __getattr__(name: str) -> _Any:
        # Deprecated attrs
        replacement = _deprecated_attrs.get(name)
        if replacement is not None:
            import warnings

            warnings.warn(
                f"'{name}' is deprecated, please use '{replacement.__module__}.{replacement.__name__}()'",
                stacklevel=2,
            )
            return replacement()

        # Lazy modules
        if name in _lazy_modules:
            return importlib.import_module(f".{name}", __name__)

        # set_vital
        if name == "set_vital":
            import warnings

            warnings.warn(f"'{name}' is deprecated, please do not call", stacklevel=2)
            return lambda *args: None

        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


@functools.cache
def get_device_module(device: "torch.device | str | None" = None) -> _ModuleType:
    """
    Returns the module associated with a given device(e.g., torch.device('cuda'), "mtia:0", "xpu", ...).
    If no device is given, return the module for the current accelerator or CPU if none is present.
    """
    if isinstance(device, torch.device):
        device_module_name = device.type
    elif isinstance(device, str):
        device_module_name = torch.device(device).type
    elif device is None:
        # Use the current accelerator's type, falling back to CPU when none is available.
        acc = torch._C._accelerator_getAccelerator()
        device_module_name = acc.type if acc is not None else "cpu"
    else:
        raise RuntimeError(
            f"Invalid value of device '{device}', expect torch.device, str, or None"
        )
    device_module = getattr(torch, device_module_name, None)
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_module_name}' does not have a corresponding module registered as 'torch.{device_module_name}'."
        )
    return device_module


def _constrain_as_size(
    symbol: "IntLikeType",
    min: builtins.int | None = None,
    max: builtins.int | None = None,
) -> None:
    """
    This indicates that a given int is size-like, and can be used in any context where a size is expected.
    You will typically use this when reading out integers from Tensors, e.g., max.item() or lengths.tolist()
    which then need to be used as tensor constructors. Providing these assertions to PyTorch can help resolve
      GuardOnDataDependentSymNode errors upon export, since we cannot guard on unbacked SymInts.

    This function has unusual semantics in some circumstances in framework
    code, we will treat this int as >= 2 (when we do a size-oblivious guard).
    This makes it easier to use the unbacked int in size contexts,
    as we will often attempt to guard on a size being zero/one
    (e.g., when computing the contiguity of a tensor, or testing if
    broadcasting can occur), which will not work on unbacked SymInts.
    However, if we conservatively assume that the size is not zero/one, we will
    end up with a graph that will still work even if the size is zero/one.

    For more details, see https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit
    ```
    """
    # pyrefly: ignore[bad-argument-type]
    torch.sym_constrain_range_for_size(symbol, min=min, max=max)


def _import_device_backends() -> None:
    """
    Leverage the Python plugin mechanism to load out-of-the-tree device extensions.
    See this RFC: https://github.com/pytorch/pytorch/issues/122468
    """
    from importlib.metadata import entry_points

    group_name = "torch.backends"
    backend_extensions = entry_points(group=group_name)

    for backend_extension in backend_extensions:
        try:
            # Load the extension
            entrypoint = backend_extension.load()
            # Call the entrypoint
            entrypoint()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load the backend extension: {backend_extension.name}. "
                f"You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0."
            ) from err


def _is_device_backend_autoload_enabled() -> builtins.bool:
    """
    Whether autoloading out-of-the-tree device extensions is enabled.
    The switch depends on the value of the environment variable
    `TORCH_DEVICE_BACKEND_AUTOLOAD`.

    Returns:
        bool: Whether to enable autoloading the extensions. Enabled by default.

    Examples:
        >>> torch._is_device_backend_autoload_enabled()
        True
    """
    # enabled by default
    return os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1") == "1"


def _as_tensor_fullprec(t: object) -> "torch.Tensor":
    """
    Like torch.as_tensor, but when given Python data types it will keep
    them in full precision.  Used for calling convention for Dynamo.
    Python scalars (float, int) are always created on CPU to avoid being
    affected by DeviceContext.
    """
    ty = type(t)
    if ty is builtins.float:
        return torch.as_tensor(t, dtype=torch.float64, device="cpu")
    elif ty is builtins.int:
        return torch.as_tensor(t, dtype=torch.int64, device="cpu")
    else:
        return torch.as_tensor(t)


# `_import_device_backends` should run after the definitions above to ensure
# all the other functions in this module that may be accessed by an autoloaded
# backend are defined.
if _is_device_backend_autoload_enabled():
    _import_device_backends()

from torch import _logging


# Keep `TORCH_LOGS` initialization after backend autoload so backends can
# register their log names before `TORCH_LOGS` is parsed and validated.
_logging._init_logs()

# Register all registered custom / override ops in torch/_native
import torch._native
<|endoftext|># mypy: allow-untyped-defs
import itertools
import operator
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import _VF, Tensor
from torch._C import _add_docstr
from torch._jit_internal import _overload as overload, boolean_dispatch
from torch._lowrank import pca_lowrank, svd_lowrank
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)


__all__ = [
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_shapes",
    "broadcast_tensors",
    "cartesian_prod",
    "block_diag",
    "cdist",
    "chain_matmul",
    "einsum",
    "istft",
    "lu",
    "norm",
    "meshgrid",
    "pca_lowrank",
    "split",
    "stft",
    "svd_lowrank",
    "tensordot",
    "unique",
    "unique_consecutive",
    "unravel_index",
]


def broadcast_tensors(*tensors):
    r"""broadcast_tensors(*tensors) -> List of Tensors

    Broadcasts the given tensors according to :ref:`broadcasting-semantics`.

    Args:
        *tensors: any number of tensors of the same type

    .. warning::

        More than one element of a broadcasted tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensors, please clone them first.

    Example::

        >>> x = torch.arange(3).view(1, 3)
        >>> y = torch.arange(2).view(2, 1)
        >>> a, b = torch.broadcast_tensors(x, y)
        >>> a.size()
        torch.Size([2, 3])
        >>> a
        tensor([[0, 1, 2],
                [0, 1, 2]])
    """
    # This wrapper exists to support variadic args.
    if has_torch_function(tensors):
        return handle_torch_function(broadcast_tensors, tensors, *tensors)
    return _VF.broadcast_tensors(tensors)  # type: ignore[attr-defined]


def broadcast_shapes(*shapes):
    r"""broadcast_shapes(*shapes) -> Size

    Similar to :func:`broadcast_tensors` but for shapes.

    This is equivalent to
    ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape``
    but avoids the need to create intermediate tensors. This is useful for
    broadcasting tensors of common batch shape but different rightmost shape,
    e.g. to broadcast mean vectors with covariance matrices.

    Example::

        >>> torch.broadcast_shapes((2,), (3, 1), (1, 1, 1))
        torch.Size([1, 3, 2])

    Args:
        \*shapes (torch.Size): Shapes of tensors.

    Returns:
        shape (torch.Size): A shape compatible with all input shapes.

    Raises:
        RuntimeError: If shapes are incompatible.
    """
    # This wrapper exists to support variadic args.
