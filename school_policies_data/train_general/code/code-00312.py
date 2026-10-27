def is_numpy_type_info(obj: object) -> bool:
    if np is None:
        return False
    return isinstance(obj, (np.finfo, np.iinfo))


BUILTIN_SKIPLIST = (
    abc,
    copy,
    importlib,
    random,
    re,
    linecache,
)

# Builtin modules that should be skipped at the top-level (PEP 523 frame
# evaluation) but inlined when called from code dynamo is already tracing.
# For example, copy.deepcopy should be inlined when the user calls it inside
# a compiled function, but copy module frames should be skipped when they
# appear as top-level frames (e.g. called internally by autograd).
BUILTIN_INLINE_WHEN_CALLED: set[str] = set()

# third party libraries skiplist is defined by str, because users may not use these libraries.
# we should use lazy import & skip in the future.
THIRDPARTY_SKIPLIST = (
    "fx2trt_oss",
    "hypothesis",
    "networkx",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnx_tf",
    "pandas",
    "sklearn",
    "tabulate",
    "tensorflow",
    "tensorrt",
    "torch2trt",
    "tqdm",
    "tree",
    "tvm",
    "xarray",
)


def _as_posix_path(path: str) -> str:
    posix_path = Path(os.path.normpath(path)).as_posix()
    # os.path.normpath and pathlib.Path remove trailing slash, so we need to add it back
    if path.endswith((os.path.sep, "/")):
        posix_path += "/"
    return posix_path


def _strip_init_py(s: str) -> str:
    suffix = "__init__.py"
    s = s.removesuffix(suffix)
    return _as_posix_path(s)


def _module_dir(m: types.ModuleType) -> str | None:
    # Protect against a module not exporting __file__ - this can happen for
    # frozen modules, for example.
    file = getattr(m, "__file__", None)
    return file and _strip_init_py(file)


# These are legacy workarounds, don't add new modules to this list.
# Please use the MOD_INLINELIST instead to force inline functions under particular modules.
#
# NB: The only thing that is different about MOD_INLINELIST and LEGACY_MOD_INLINELIST
# is the behavior of a function f2 in the module when called by a function f1
# in a module in MOD_SKIPLIST (see MOD_SKIPLIST for more details)
#
# LEGACY_MOD_INLINELIST is the same thing as Dynamo's behavior on a module that
# is not in any *_INLINELIST or *_SKIPLIST.
# That being said, we prefer people to add things to MOD_INLINELIST over
# LEGACY_MOD_INLINELIST because it is less likely to break existing tests.
LEGACY_MOD_INLINELIST = {
    "torch._dynamo.external_utils",
    "torch._export.db.examples",
    "torch._export.wrappers",
    "torch._functorch.apis",
    "torch._functorch.deprecated",
    "torch._library.fake_class_registry",
    "torch.utils._typing_utils",
    "torch.nn.attention.flex_attention",
    "torch.ao.quantization.stubs",
    "torch.ao.quantization.pt2e.export_utils",
    "torch.ao.quantization.pt2e.qat_utils",
    "torch.ao.quantization.pt2e.representation.rewrite",
    "torch.ao.quantization.pt2e.utils",
    "torch.ao.quantization.quantizer.xnnpack_quantizer",
    "torch.export.unflatten",
}

if torch.distributed.is_available():
    LEGACY_MOD_INLINELIST |= {
        "torch.distributed.tensor._api",
        "torch.distributed.tensor.device_mesh",
        "torch.distributed.device_mesh",
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
        "torch.distributed.tensor.parallel._data_parallel_utils",
        "torch.distributed.tensor.parallel._utils",
        "torch.distributed.tensor.parallel.style",
        # we have to add replicate to LEGACY_MOD_INLINELIST to ensure
        # the forward_hook won't be ignored.
        "torch.distributed._composable.replicate",
    }

# Force inline functions under these modules, even they are in *_SKIPLIST.
# We are using python module name instead of file or directory object to avoid circular dependency.
# Please keep this sorted alphabetically.
#
# Btw, it is not "ideal" for something to be in MOD_INLINELIST. If Dynamo
# fully supports a module, then the ideal case is that it is not in
# any *_INLINELIST or *_SKIPLIST: then, the behavior of Dynamo is that
# it will always inline into functions in the module.
MOD_INLINELIST = [
    "torch._decomp",
    "torch._dynamo._trace_wrapped_higher_order_op",
    "torch._dynamo.compiled_autograd",
    "torch._dynamo.comptime",
    "torch._dynamo.polyfills",
    "torch._dynamo.test_case",
    "torch._export.non_strict_utils",
    "torch._functorch._aot_autograd.subclass_parametrization",
    "torch._functorch.autograd_function",
    "torch._functorch.eager_transforms",
    "torch._functorch.functional_call",
    "torch._functorch.pyfunctorch",
    "torch._functorch.vmap",
    "torch._inductor.test_operators",
    "torch._library.autograd",
    "torch._library.custom_ops",
    "torch._ops",
    "torch._prims",
    "torch._refs",
    "torch._tensor",
    "torch.amp.autocast_mode",
    "torch.ao.nn",
    "torch.ao.quantization.fx._decomposed",
    "torch.autograd.function",
    "torch.backends.cuda",
    "torch.cuda.amp.autocast_mode",
    "torch.distributions",
    "torch.export._patches",
    "torch.export._tree_utils",
    "torch.export._unlift",
    "torch.export._wrapper_utils",
    "torch.fx._pytree",
    "torch.fx._symbolic_trace",
    "torch.fx.experimental.proxy_tensor",
    "torch.fx.passes.shape_prop",
    "torch.fx.traceback",
    "torch.nn",
    "torch.overrides",
    "torch.random",
    "torch.return_types",
    "torch.sparse",
    "torch.testing",
    "torch.utils._content_store",
    "torch.utils._contextlib",
    "torch.utils._cxx_pytree",
    "torch.utils._device",
    "torch.utils._foreach_utils",
    "torch.utils._ordered_set",
    "torch.utils._python_dispatch",
    "torch.utils._pytree",
    "torch.utils.hooks",
]
if sorted(set(MOD_INLINELIST)) != MOD_INLINELIST:
    raise AssertionError("MOD_INLINELIST must be sorted with no duplicates")
MOD_INLINELIST = set(MOD_INLINELIST)


if torch.distributed.is_available():
    MOD_INLINELIST.add("torch.distributed")

# Modules in MOD_INLINELIST where nested graph breaks should be suppressed.
# These are large internal modules that are inlined for correctness but whose
# internal operations are not worth compiling as separate NGB subgraphs.
NGB_SUPPRESS_INLINELIST: set[str] = set()

if torch.distributed.is_available():
    NGB_SUPPRESS_INLINELIST.add("torch.distributed")

if not NGB_SUPPRESS_INLINELIST <= MOD_INLINELIST:
    raise AssertionError(
        "NGB_SUPPRESS_INLINELIST entries must also be in MOD_INLINELIST: "
        f"{NGB_SUPPRESS_INLINELIST - MOD_INLINELIST}"
    )


# By default, all functions under these modules are skipped.
# All the other knobs
# (torch_name_rule_map, MOD_INLINELIST, LEGACY_MOD_INLINELIST)
# take precedence over this list; e.g. if a function is in
# MOD_INLINELIST and MOD_SKIPLIST, then it will be inlined.
# See "A note on skip/inline rules" for more details.
#
# The skip is NOT recursive. If a function f1 in a module in MOD_SKIPLIST
# calls out to another function f2 in some other module, then Dynamo's
# behavior (skip/inline) depends on what we've marked f2 as:
# - if f2 is a function in a module in MOD_SKIPLIST, then we skip f2
# - if f2 is a function in a module in MOD_INLINELIST, then we skip f2
# - if f2 is a function in a module in LEGACY_MOD_INLINELIST, then we inline f2
# - if f2 is a function in a module not in any *_LIST, then we inline f2
MOD_SKIPLIST = [
    "torch._VF",
    "torch.__future__",
    "torch.__init__",
    "torch._awaits",
    "torch._classes",
    "torch._compile",
    "torch._custom_op",
    "torch._custom_ops",
    "torch._decomp",
    "torch._dispatch",
    "torch._dynamo",
    "torch._export",
    "torch._functorch",
    "torch._guards",
    "torch._higher_order_ops.effects",
    "torch._higher_order_ops.torchbind",
    "torch._higher_order_ops.wrap",
    "torch._inductor",
    "torch._jit_internal",
    "torch._lazy",
    "torch._library",
    "torch._linalg_utils",
    "torch._lobpcg",
    "torch._logging",
    "torch._lowrank",
    "torch._meta_registrations",
    "torch._numpy",
    "torch._ops",
    "torch._prims",
    "torch._prims_common",
    "torch._python_dispatcher",
    "torch._refs",
    "torch._strobelight",
    "torch._subclasses",
    "torch._tensor",
    "torch._tensor_str",
    "torch._thread_safe_fork",
    "torch._utils",
    "torch._utils_internal",
    "torch._vmap_internals",
    "torch._weights_only_unpickler",
    "torch.accelerator",
    "torch.amp",
    "torch.ao",
    "torch.autograd",
    "torch.backends",
    "torch.compiler",
    "torch.contrib",
    "torch.cpu",
    "torch.cuda",
    "torch.distributed",
    "torch.distributions",
    "torch.export",
    "torch.fb",
    "torch.fft",
    "torch.functional",
    "torch.futures",
    "torch.fx",
    "torch.hub",
    "torch.jit",
    "torch.library",
    "torch.linalg",
    "torch.masked",
    "torch.monitor",
    "torch.mps",
    "torch.mtia",
    "torch.multiprocessing",
    "torch.nested",
    "torch.nn",
    "torch.onnx",
    "torch.overrides",
    "torch.package",
    "torch.profiler",
    "torch.quantization",
    "torch.quasirandom",
    "torch.random",
    "torch.serialization",
    "torch.signal",
    "torch.sparse",
    "torch.special",
    "torch.storage",
    "torch.testing",
    "torch.types",
    "torch.utils",
    "torch.xpu",
]

if sorted(set(MOD_SKIPLIST)) != MOD_SKIPLIST:
    raise AssertionError("MOD_SKIPLIST must be sorted with no duplicates")
MOD_SKIPLIST = set(MOD_SKIPLIST)


@functools.cache
def get_legacy_mod_inlinelist() -> set[str]:
    torch_dir = _module_dir(torch)
    if torch_dir is None:
        return set()
    inlinelist = {
        _as_posix_path(torch_dir + m[len("torch.") :].replace(".", "/"))
        for m in LEGACY_MOD_INLINELIST
    }
    return inlinelist


@functools.cache
def get_mod_inlinelist() -> set[str]:
    torch_dir = _module_dir(torch)
    if torch_dir is None:
        return set()
    inlinelist = {
        _as_posix_path(torch_dir + m[len("torch.") :].replace(".", "/"))
        for m in MOD_INLINELIST
    }
    return inlinelist


@functools.cache
def get_mod_skiplist() -> set[str]:
    torch_dir = _module_dir(torch)
    if torch_dir is None:
        return set()
    skiplist = {
        _as_posix_path(torch_dir + m[len("torch.") :].replace(".", "/"))
        for m in MOD_SKIPLIST
    }
    return skiplist


# skip some standard python builtin libs
SKIP_DIRS = [
    "<frozen importlib",
    "<frozen abc",
    "<__array_function__ internals>",
    _as_posix_path(_config_module.__file__),
    "triton/backends",
]
SKIP_DIRS.extend(map(_as_posix_path, filter(None, map(_module_dir, BUILTIN_SKIPLIST))))

BUILTIN_INLINE_WHEN_CALLED.update(
    filter(None, (_module_dir(copy), _as_posix_path(_config_module.__file__)))
)

SKIP_DIRS_RE = re.compile(r"match nothing^")

# Skip fbcode paths(including torch.package paths) containing
# one of the following strings.
FBCODE_SKIP_DIRS: set[str] = set()

FBCODE_SKIP_DIRS_RE = re.compile(f".*({'|'.join(map(re.escape, FBCODE_SKIP_DIRS))})")

# Remove this after fbcode is fully migrated to tracing through torchrec.
FBCODE_SKIP_TORCHREC_DIRS = {
    "torchrec/distributed",
    "torchrec/fb/distributed",
    "caffe2/torch/fb/sparsenn/pooled_embeddings_modules.py",
}

FBCODE_SKIP_TORCHREC_DIRS_RE = re.compile(
    f".*({'|'.join(re.escape(_as_posix_path(d)) for d in FBCODE_SKIP_TORCHREC_DIRS)})"
)

# TODO(yanboliang, anijain2305) - There are a few concerns that we should
# resolve
# 1) Audit if torchrec/distributed is even required in FBCODE_SKIPS_DIR
# 2) To inline just one file but skip others in a directory, we could use
# manual_torch_name_rule_map but this one is hard because FBCODE can add unusual
# names like torch_package.
# So, this is a stop gap solution till then.
FBCODE_INLINE_FILES_IN_SKIPPED_DIRS = {
    "torchrec/distributed/types.py",
}
FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE = re.compile(
    f".*({'|'.join(re.escape(_as_posix_path(d)) for d in FBCODE_INLINE_FILES_IN_SKIPPED_DIRS)})"
)

# torch.optim is a special case,
# we usually want to inline it, but the directory
# structure does not match the module structure
# and we want to skip the functions in optim/lr_scheduler.py
# this has precedence over all other rules in check_file
FORCE_SKIP_FILES = {f"{_module_dir(torch)}optim/lr_scheduler.py"}


def _recompile_re() -> None:
    global SKIP_DIRS_RE
    SKIP_DIRS_RE = re.compile(
        rf"^[^\s<]*({'|'.join(re.escape(_as_posix_path(d)) for d in SKIP_DIRS)})"
    )


def add(import_name: str) -> None:
    if isinstance(import_name, types.ModuleType):
        return add(import_name.__name__)
    if not isinstance(import_name, str):
        raise AssertionError(
            f"expected import_name to be a str, got {type(import_name)}"
        )
    from importlib.util import find_spec

    module_spec = find_spec(import_name)
    if not module_spec:
        return
    origin = module_spec.origin
    if origin is None:
        return
    SKIP_DIRS.append(_strip_init_py(origin))
    _recompile_re()


@dataclasses.dataclass
class SkipResult:
    skipped: bool
    reason: str | None


def check_file(filename: str | None, is_inlined_call: bool = False) -> SkipResult:
    """Should skip this file?"""
    if filename is None:
        return SkipResult(
            True, "cannot determine source file (likely a C extension or builtin)"
        )
    filename = _as_posix_path(filename)
    if filename in FORCE_SKIP_FILES:
        return SkipResult(True, f"file is force-skipped ({filename})")

    for d in get_legacy_mod_inlinelist():
        if filename.startswith(d):
            return SkipResult(False, f"file matches LEGACY_MOD_INLINELIST ({d})")
    if is_inlined_call and is_torch_inline_allowed(filename):
        return SkipResult(False, f"file matches MOD_INLINELIST ({filename})")
    if is_inlined_call and any(
        filename.startswith(d) for d in BUILTIN_INLINE_WHEN_CALLED
    ):
        return SkipResult(
            False, f"file matches BUILTIN_INLINE_WHEN_CALLED ({filename})"
        )
    if (
        is_fbcode()
        and FBCODE_SKIP_DIRS
        and bool(FBCODE_SKIP_DIRS_RE.match(filename))
        and not bool(FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE.match(filename))
    ):
        return SkipResult(True, "file matches FBCODE_SKIP_DIRS")

    if (
        is_fbcode()
        and config.skip_torchrec
        and FBCODE_SKIP_TORCHREC_DIRS
        and bool(FBCODE_SKIP_TORCHREC_DIRS_RE.match(filename))
        and not bool(FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE.match(filename))
    ):
        return SkipResult(True, "file matches FBCODE_SKIP_TORCHREC_DIRS")

    unittest_dir = _module_dir(unittest)
    if (
        unittest_dir is not None
        and filename.startswith(unittest_dir)
        and not torch._dynamo.config.enable_trace_unittest
    ):
        return SkipResult(True, "file is in unittest directory")

    if bool(SKIP_DIRS_RE.match(filename)):
        matched_dir = next((d for d in SKIP_DIRS if filename.startswith(d)), filename)
        return SkipResult(True, f"file is under skip directory ({matched_dir})")

    for d in get_mod_skiplist():
        if filename.startswith(d):
            return SkipResult(True, f"file matches MOD_SKIPLIST ({d})")
    return SkipResult(False, "inlined by default")


@dataclasses.dataclass
class FunctionInfo:
    py_obj: object | None
    name: str | None
    filename: str
    code: types.CodeType | None


"""
This is the main entry point to determine whether an object (function) should be inlined or skipped.
Let's illustrate the logic with an example:
    @torch.compile
    def f1(x, y):
        ......
        f2(x, y)
        ......

    def f2(x, y):
        ......
        f3(x, y)
        ......

    def f3(x, y):
        ......

There are mainly three call sites of check/check_verbose:
* The compile region entrance (like function f1), the corresponding code is located at eval_frame.py.
* When tracing the recursively called functions (like function f2 and f3).
    * Dynamo decides inline/skip every time it encounters a new recursively function call, and the call site
      is in InliningInstructionTranslator.check_inlineable of symbolic_convert.py.
    * If f2 is skipped by Dynamo, when evaluating the frame of f3, Dynamo need the inline/skip check again
      and the call site is in catch_errors_wrapper.catch_errors of convert_frame.py.
* For global variables and function arguments, Dynamo needs to decide if they are wrapped as SkipFunctionVariable in builder.py.

`is_inlined_call` is used to indicate if the current function call is inlined (f2 is inlined call if it passes check)
or not (f3 is not inlined call if f2 is skipped). Inside of the `check_verbose` function, there are more rules
to be checked if this `is_inlined_call`.
The reason to have this flag is that if the upper level function call (e.g, f2) is skipped,
we don't want to inline the lower level function call (e.g, f3) by default.
"""

_force_inline_flag = False


# pyrefly: ignore [deprecated]
@contextlib.contextmanager
def _force_inline() -> Iterator[None]:
    """
    A context manager used within the dynamo codebase that forces a function
    and nested function calls to be inlined during dynamo tracing.

    When active, check_verbose() will skip all inline/skip decision logic and
    always return SkipResult(False, ...), meaning functions will be inlined.

    See _make_inlined() in utils.py which uses this to ensure that
    a python function is fully traced to produce the needed variable trackers.
    """
    global _force_inline_flag
    old_val = _force_inline_flag
    try:
        _force_inline_flag = True
        yield
    finally:
        _force_inline_flag = old_val


def check_verbose(
    obj: object, is_inlined_call: bool = False, frame: Any | None = None
) -> SkipResult:
    if _force_inline_flag:
        return SkipResult(
            False,
            "don't skip because we're inside _force_inline() context",
        )

    # For eval frame callback (not inlined calls), allow tracing inbuilt
    # nn.Module.forward methods when the module has hooks. Any hook can cause
    # a graph break (via @torch._dynamo.disable, print, unsupported ops, etc.),
    # which skips the entire _call_impl frame. By allowing forward to be traced,
    # Dynamo can capture the module's operations in a new graph after the break.
    if (
        not is_inlined_call
        and frame is not None
        and isinstance(obj, types.CodeType)
        and obj.co_name == "forward"
    ):
        from .utils import nnmodule_has_hooks

        module = frame.f_locals.get("self")
        if (
            module is not None
            and isinstance(module, torch.nn.Module)
            and module.__class__.__module__.startswith(("torch.nn.", "torch.ao."))
            and nnmodule_has_hooks(module, check_forward_hooks=True)
        ):
            return SkipResult(
                False,
                "inbuilt nn.Module.forward allowed - module has hooks",
            )

    if isinstance(
        obj,
        (
            UserFunctionVariable,
            UserMethodVariable,
            NestedUserFunctionVariable,
            LocalGeneratorFunctionVariable,
            LocalGeneratorObjectVariable,
        ),
    ):
        try:
            py_obj = obj.get_function()
        except NotImplementedError:
            py_obj = None
        fi = FunctionInfo(py_obj, obj.get_name(), obj.get_filename(), obj.get_code())
    elif isinstance(obj, types.CodeType):
        fi = FunctionInfo(None, obj.co_name, obj.co_filename, obj)
    elif isinstance(obj, (types.FunctionType, types.MethodType)):
        filename = getfile(obj)
        if filename is None:
            raise AssertionError(f"getfile returned None for function/method {obj}")
        fi = FunctionInfo(
            obj,
            obj.__name__,
            filename,
            obj.__code__,  # type: ignore[union-attr] # FIXME Add MethodType.__code__ to typeshed
        )
    else:
        filename = getfile(obj)
        if filename is None:
            raise AssertionError(f"getfile returned None for {type(obj)}")
        fi = FunctionInfo(obj, None, filename, None)

    # typing.cast is a polyfilled no-op, but unlike C builtins it has a code
    # object that PEP 523 can intercept as a standalone frame after a graph
    # break. Skip it at the top level to avoid installing unnecessary guards.
    if fi.code is not None and fi.code is typing.cast.__code__:
        return SkipResult(True, "typing.cast is a no-op, skip at top level")

    # Consulte the central trace rules defined in torch._dynamo.trace_rules.
    reasons: set[str] = set()
    rule = lookup_inner(fi.py_obj, fi.name, fi.filename, is_inlined_call, reasons)
    if rule is None:
        raise AssertionError(
            f"lookup_inner returned None for {fi.name} in {fi.filename}"
        )
    if issubclass(
        rule,
        (
            UserFunctionVariable,
            LocalGeneratorFunctionVariable,
            PolyfilledFunctionVariable,
        ),
    ):
        return SkipResult(False, reasons.pop())
    elif issubclass(rule, TorchInGraphFunctionVariable):
        return SkipResult(False, reasons.pop())
    else:
        if rule != SkipFunctionVariable:
            raise AssertionError(f"expected SkipFunctionVariable, got {rule}")
        return SkipResult(True, reasons.pop())


def check(obj: object, is_inlined_call: bool = False, frame: Any | None = None) -> bool:
    return check_verbose(obj, is_inlined_call, frame).skipped


def get_skip_reason(obj: object) -> str:
    """Compute a descriptive skip reason for a callable. Only called on graph break."""
    if is_callable_disallowed(obj):
        return _disallowed_callable_ids.get_name(id(obj), repr(obj))

    filename = getfile(obj)
    if filename is not None:
        skip_result = check_file(filename)
        if skip_result.reason is not None:
            return skip_result.reason

    module = getattr(obj, "__module__", None) or ""
    return (
        f"cannot determine source file for {module} (likely a C extension or builtin)"
    )


# skip common third party libs
for _name in THIRDPARTY_SKIPLIST:
    add(_name)

_recompile_re()


def is_torch_inline_allowed(filename: str) -> bool:
    return any(filename.startswith(d) for d in get_mod_inlinelist())


@functools.cache
def get_ngb_suppress_inlinelist() -> set[str]:
    torch_dir = _module_dir(torch)
    if torch_dir is None:
        return set()
    return {
        _as_posix_path(torch_dir + m[len("torch.") :].replace(".", "/"))
        for m in NGB_SUPPRESS_INLINELIST
    }


def is_ngb_suppressed_inline(filename: str) -> bool:
    return any(filename.startswith(d) for d in get_ngb_suppress_inlinelist())


@functools.cache
def dynamo_dir() -> str | None:
    import torch._dynamo

    return _module_dir(torch._dynamo)


def is_torch(filename: str) -> bool:
    dynamo_path = dynamo_dir()
    if dynamo_path is not None and filename.startswith(dynamo_path):
        return False
    torch_path = _module_dir(torch)
    return torch_path is not None and filename.startswith(torch_path)


"""
Main entry point for looking up the trace rule (the Dynamo variable) for a given callable object.
"""

BUILTIN_CALLABLES = {
    dict: DictBuiltinVariable,
    getattr: GetAttrBuiltinVariable,
    hasattr: HasAttrBuiltinVariable,
    iter: IterBuiltinVariable,
    list: ListBuiltinVariable,
    setattr: SetAttrBuiltinVariable,
}


def lookup_callable(obj: Callable[..., Any]) -> type[VariableTracker] | None:
    if not hashable(obj):
        return None
    # Custom allow/disallow in graph takes precedence over the general lookup.
    if is_callable_disallowed(obj):
        return SkipFunctionVariable
    if is_callable_allowed(obj):
        return TorchInGraphFunctionVariable
    if is_polyfilled_callable(obj):
        return PolyfilledFunctionVariable
    if obj in BUILTIN_CALLABLES:
        return BUILTIN_CALLABLES[obj]
    if is_builtin_callable(obj):
        return BuiltinVariable
    return None


"""
Main entry point for looking up the trace rule (the Dynamo variable) for a given function object.
E.g, the lookup result of `torch.sin` is `TorchInGraphFunctionVariable`.
"""


def lookup(obj: object) -> type[VariableTracker] | None:
    return lookup_inner(obj)


# also takes config.dont_skip_tracing into account
def lookup_inner(
    obj: object,
    name: str | None = None,
    filename: str | None = None,
    is_direct_call: bool = True,
    reasons: None | set[str] = None,
) -> type[VariableTracker] | None:
    result = _lookup_inner(
        obj,
        name=name,
        filename=filename,
        is_direct_call=is_direct_call,
        reasons=reasons,
    )
    # There are still some modules we should absolutely NOT trace into - e.g. most of torch._dynamo,
    # as this can result in really weird tracing behaviors.
    # Note that if a torch._dynamo function is already not skipped (e.g. functions in external_utils.py),
    # then this branch does not apply.
    if config.dont_skip_tracing and result is SkipFunctionVariable:
        if filename is None:
            filename = getfile(obj)
        if filename is None:
            raise AssertionError(
                f"getfile returned None for {type(obj)} in dont_skip_tracing path"
            )
        filename = _as_posix_path(filename)
        torch_dir = _module_dir(torch)
        if torch_dir is not None:
            dynamo_path = _as_posix_path(torch_dir) + "_dynamo"
            if filename.startswith(dynamo_path) and not filename.endswith(
                "test_dont_skip_tracing_functions.py"
            ):
                return SkipFunctionVariable
        if reasons is not None:
            reasons.add(
                "Attempted skip but we are ignoring skips due to torch._dynamo.config.dont_skip_tracing"
            )
        return UserFunctionVariable
    return result


def _lookup_inner(
    obj: object,
    name: str | None = None,
    filename: str | None = None,
    is_direct_call: bool = True,
    reasons: set[str] | None = None,
) -> type[VariableTracker] | None:
    # Step 1: lookup obj's tracing rule in `torch_name_rule_map`.
    # The rules defined in `torch_name_rule_map` mainly includes two parts:
    # - Manually defined rules for any functions.
    # - The list of torch in graph functions.
    try:
        can_hash = hashable(obj)
    except Exception:
        can_hash = False
    if not can_hash:
        if reasons is not None:
            reasons.add("obj is not hashable")
        return None
    if obj is not None:
        if is_aten_op_or_tensor_method(obj):
            return TorchInGraphFunctionVariable
        rule = get_torch_obj_rule_map().get(obj, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_torch_obj_rule_map")
            return rule
    elif name is not None and filename is not None and not is_direct_call:
        if name.startswith(TORCH_DYNAMO_RESUME_IN_PREFIX):
            rule = get_torch_obj_rule_map().get(
                filename + "#" + TORCH_DYNAMO_RESUME_IN_PREFIX, None
            )
        else:
            rule = get_torch_obj_rule_map().get(filename + "#" + name, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_torch_obj_rule_map")
            return rule
    elif name == "<listcomp>":
        if reasons is not None:
            reasons.add("inlining frame from list comprehension")
        return UserFunctionVariable

    # Step 2: lookup obj's tracing rule by function name.
    if is_direct_call:
        if name == "patched_init":
            if reasons is not None:
                reasons.add("func name is patched_init")
            return SkipFunctionVariable
        elif name == "__torch_function__" or (
            obj and getattr(obj, "__name__", None) == "__torch_function__"
        ):
            if reasons is not None:
                reasons.add("func name is __torch_function__")
            return UserFunctionVariable

    if not is_direct_call:
        if name == "__getattr__":
            # is_direct_call = False indicates that this is the top-level frame
            # being traced (i.e., it is not inlined and not called from
            # InliningInstructionTranslator).  Tracing __getattr__ at the top
            # level is unlikely because we inline it for
            # UserDefinedObjectVariable. This scenario occurs only for
            # UnspecializedNNModuleVariable, where Dynamo directly calls
            # __getattr__ during trace time, generating LOAD_ATTR bytecode
            # without going through the underlying __getattr__ data structures.
            # When this optimized bytecode is executed, Dynamo is triggered
            # again on the __getattr__ call. Therefore, we skip Dynamo tracing
            # in this case.
            if reasons is not None:
                reasons.add(
                    "Tracing __getattr__ as the top level frame, unsuitable for tracing."
                )
            return SkipFunctionVariable

    # Step 3: lookup obj's tracing rule by filename.
    if filename is None:
        filename = getfile(obj)

    skip_result = check_file(filename, is_direct_call)
    if reasons is not None and skip_result.reason is not None:
        reasons.add(skip_result.reason)
    if skip_result.skipped:
        return SkipFunctionVariable
    else:
        return UserFunctionVariable


def clear_lru_cache() -> None:
    torch._dynamo.trace_rules.get_torch_obj_rule_map.cache_clear()
    torch._dynamo.trace_rules.get_tensor_method.cache_clear()
    torch._dynamo.trace_rules.get_legacy_mod_inlinelist.cache_clear()
    torch._dynamo.trace_rules.get_mod_inlinelist.cache_clear()
    torch._dynamo.trace_rules.get_ngb_suppress_inlinelist.cache_clear()
    torch._dynamo.trace_rules.dynamo_dir.cache_clear()
<|endoftext|>"""
Side effect tracking and management for TorchDynamo's compilation system.

This module provides infrastructure for tracking and managing side effects that occur
during symbolic execution, including:

- Tracking mutations to objects, attributes, and variables
- Managing context changes (cell variables, global namespace modifications)
- Handling aliasing and object identity preservation
- Managing stack frame state and local variable changes
- Tracking function calls with side effects

Key classes:
- SideEffects: Main container for tracking all side effects during execution
- MutableSideEffects: Specialization for mutable object tracking
- AttributeMutation/ValueMutation: Track specific types of mutations
- Various specialized side effect classes for different scenarios

The side effect system ensures that mutations performed during symbolic execution
are properly replayed during runtime, maintaining the correctness of compiled code
while enabling optimizations where safe.
"""

import collections
import contextlib
import dataclasses
import enum
import inspect
import logging
import textwrap
import traceback
import weakref
from collections.abc import Callable, Generator, MutableMapping
from types import CellType
from typing import Any, TYPE_CHECKING

import torch
import torch.nn
from torch._dynamo.variables.misc import AutogradFunctionContextVariable
from torch._library.utils import RegistrationHandle
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import is_structseq_class

from . import config, graph_break_hints, utils, variables
from .bytecode_transformation import (
    bytecode_from_template,
    create_call_function,
    create_call_method,
    create_instruction,
    Instruction,
)
from .codegen import PyCodegen
from .exc import collapse_resume_frames, get_stack_above_dynamo, unimplemented
from .source import AttrSource, GlobalSource, LocalCellSource, Source, TempLocalSource
from .utils import (
    is_frozen_dataclass,
    is_namedtuple_cls,
    is_pybind11_enum_member,
    nn_module_new,
    object_new,
)
from .variables.base import (
    AttributeMutation,
    AttributeMutationExisting,
    AttributeMutationNew,
    AttrMutationKind,
    is_side_effect_safe,
    ValueMutationExisting,
    ValueMutationNew,
    VariableTracker,
)
from .variables.user_defined import FrozenDataClassVariable


if TYPE_CHECKING:
    from torch._dynamo.output_graph import OutputGraph
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._dynamo.variables.lists import ListVariable


side_effects_log = torch._logging.getArtifactLogger(__name__, "side_effects")


@dataclasses.dataclass(frozen=True)
class SideEffectReplayContext:
    side_effects: "SideEffects"
    codegen: PyCodegen
    var: VariableTracker
    suffixes: list[list[Instruction]]
    log: Callable[[VariableTracker], None]


SideEffectReplayMatcher = Callable[[SideEffectReplayContext], bool]
SideEffectReplayCodegen = Callable[[SideEffectReplayContext], None]
SideEffectReplayHandlerDecorator = Callable[
    [SideEffectReplayCodegen], SideEffectReplayCodegen
]


@dataclasses.dataclass(frozen=True)
class SideEffectReplayHandler:
    name: str
    matcher: SideEffectReplayMatcher
    codegen: SideEffectReplayCodegen
    priority: int = 0


class _SideEffectReplayRegistry:
    def __init__(self) -> None:
        self.handlers: list[SideEffectReplayHandler] = []

    def register_handler(self, handler: SideEffectReplayHandler) -> RegistrationHandle:
        self.handlers.append(handler)

        def deregister() -> None:
            self.handlers.remove(handler)

        return RegistrationHandle(deregister)

    def lookup_handler(
        self, ctx: SideEffectReplayContext
    ) -> SideEffectReplayHandler | None:
        matches = [handler for handler in self.handlers if handler.matcher(ctx)]
        if not matches:
            return None
        matches.sort(key=lambda handler: handler.priority, reverse=True)
        top = matches[0]
        if len(matches) > 1 and matches[1].priority == top.priority:
            names = [
                handler.name for handler in matches if handler.priority == top.priority
            ]
            raise RuntimeError(
                f"Ambiguous side effect replay handler for "
                f"{type(ctx.var).__name__}: {names}"
            )
        return top


_side_effect_replay_registry = _SideEffectReplayRegistry()


def _register_side_effect_replay_handler(
    handler: SideEffectReplayHandler,
) -> RegistrationHandle:
    return _side_effect_replay_registry.register_handler(handler)


def register_side_effect_replay_handler(
    *,
    name: str,
    matcher: SideEffectReplayMatcher,
    priority: int = 0,
) -> SideEffectReplayHandlerDecorator:
    def decorator(
        codegen: SideEffectReplayCodegen,
    ) -> SideEffectReplayCodegen:
        _register_side_effect_replay_handler(
            SideEffectReplayHandler(
                name=name,
                matcher=matcher,
                codegen=codegen,
                priority=priority,
            )
        )
        return codegen

    return decorator


def _manual_dict_setitem(
    dict_from: dict[Any, Any], dict_to: dict[Any, Any], mro_index: int
) -> None:
    # Carefully calls the dict or OrderedDict `clear` or `__setitem__`. We have
    # to be careful because we don't want to trigger the user defined object
    # setitem or clear. The mro_index is used to find the dict/OrderedDict from
    # the class mro.
    dict_class = type(dict_to).__mro__[mro_index]
    dict_class.clear(dict_to)  # type: ignore[attr-defined]
    for k, v in dict_from.items():
        dict_class.__setitem__(dict_to, k, v)  # type: ignore[index]


def _manual_list_update(list_from: list[Any], list_to: list[Any]) -> None:
    list.clear(list_to)
    list.extend(list_to, list_from)


def _manual_deque_update(deque_from: Any, deque_to: Any) -> None:
    # Call the deque methods directly, not any overridden subclass methods.
    # deque_to keeps its (read-only) maxlen, so extend re-applies eviction.
    collections.deque.clear(deque_to)
    collections.deque.extend(deque_to, deque_from)


class SideEffects:
    """
    Maintain records of mutations and provide methods to apply them during code generation.

    Handles tracking and applying side effects during PyTorch Dynamo compilation,
    maintaining Python semantics by managing mutations, attribute modifications,
    and other side effects that occur during program execution.

    Key responsibilities:
    - Tracks mutations to Python objects, lists, and dictionaries that need to be
    applied after an FX graph is run.
    - Manages attribute modifications and deletions
    - Handles tensor hooks and backward pass state
    - Tracks cell variable mutations and global variable changes
    - Ensures correct ordering and application of side effects after graph execution

    This ensures that optimized code behaves identically to the original Python code with
    respect to object mutations and other side effects.
    """

    id_to_variable: dict[int, VariableTracker]
    store_attr_mutations: dict[VariableTracker, dict[str, VariableTracker]]
    attr_mutation_kinds: dict[VariableTracker, dict[str, AttrMutationKind]]
    keepalive: list[object]
    # Maps variable tracker to list of user stacks (StackSummary objects, formatted lazily)
    mutation_user_stacks: dict[VariableTracker, list[traceback.StackSummary]]

    def __init__(
        self,
        output_graph: "OutputGraph",
        id_to_variable: dict[int, VariableTracker] | None = None,
        store_attr_mutations: dict[VariableTracker, dict[str, VariableTracker]]
        | None = None,
        attr_mutation_kinds: dict[VariableTracker, dict[str, AttrMutationKind]]
        | None = None,
        mutation_user_stacks: dict[VariableTracker, list[traceback.StackSummary]]
        | None = None,
        keepalive: list[object] | None = None,
        save_for_backward: list[
            tuple[AutogradFunctionContextVariable, list[VariableTracker]]
        ]
        | None = None,
        tensor_hooks: dict[
            int,
            tuple[
                "variables.TensorVariable",
                VariableTracker,
                "variables.RemovableHandleVariable",
                str,
            ],
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self.output_graph_weakref = weakref.ref(output_graph)
        self.id_to_variable = id_to_variable or {}
        self.store_attr_mutations = store_attr_mutations or {}
        self.attr_mutation_kinds = attr_mutation_kinds or {}
        self.mutation_user_stacks = mutation_user_stacks or {}
        self.keepalive = keepalive or []
        self.save_for_backward = save_for_backward or []
        self.tensor_hooks = tensor_hooks or {}
        # Used by MappingProxyVariable to graph break in case of any mutated
        # dict
        self._has_existing_dict_mutation = False
        # Track Compiled Autograd final callbacks that must be called at the end of Compiled Autograd backward graph.
        # Only applicable if this graph is created from Dynamo tracing in Compiled Autograd.
        self.ca_final_callbacks_var: ListVariable | None = None

        # Tracks VariableTracker objects whose mutations can be skipped.
        # For normal mutated variables, Dynamo generates code to replay/reconstruct
        # the mutations after graph execution. However, variables in this set have
        # their mutations ignored - the mutations happen during
        # execution but don't need to be replayed in the generated code.
        # Used for temporary mutations in contexts like torch.func.functional_call,
        # where module parameters/buffers are modified but later restored.
        self.ignore_mutation_on_these_variables: set[VariableTracker] = set()
        # Sources mutated during tracing: AttrSource for attribute
        # mutations, var.source for value mutations (list/dict/etc).
        self.mutated_sources: OrderedSet[Source] = OrderedSet()

        # Deferred side-effect checking for nullified attribute mutations.
        # Maps (vt_id, attr_name) → (original_value, current_value).
        # On validation, we check original == current.
        self.deferred_attr_mutations: dict[tuple[int, str], tuple[object, object]] = {}

    def ignore_mutations_on(self, var: VariableTracker) -> None:
        """Mutations to this variable will be executed but not tracked,
        typically used for temporary mutations that are later restored."""
        self.ignore_mutation_on_these_variables.add(var)

    def stop_ignoring_mutations_on(self, var: VariableTracker) -> None:
        """Remove a variable from the skip mutation set, restoring normal mutation tracking."""
        if var in self.ignore_mutation_on_these_variables:
            self.ignore_mutation_on_these_variables.remove(var)

    @contextlib.contextmanager
    def defer_side_effect_checks(self) -> Generator[None, None, None]:
        """Defer outer-scope attribute mutation checks until tracing completes.

        Context managers that flip-flop a flag (set on enter, restore on exit)
        produce no net side effect. Instead of failing immediately, we track
        original and current values, then validate they match after tracing.

        Note: this context only validates that mutations were nullified — it
        does NOT roll back store_attr_mutations. Callers must restore
        side_effects separately (e.g., via prev_side_effects pattern) to
        discard the mutations after the HOP.
        """
        saved = self.deferred_attr_mutations
        self.deferred_attr_mutations = {}
        try:
            yield
            self.validate_deferred_attr_mutations()
        finally:
            self.deferred_attr_mutations = saved

    def snapshot_attr_mutation(
        self, item: VariableTracker, name: str, value: VariableTracker
    ) -> bool:
        """Record an attribute mutation for deferred validation.

        Returns True if successfully deferred, False if we cannot read the
        original value (tp_getattro_impl raises NotImplementedError) or the
        original is not a python constant — caller should fall back to
        check_allowed_side_effect.
        """
        key = (id(item), name)
        if not value.is_python_constant():
            raise AssertionError(
                "value must be a python constant (guaranteed by caller store_attr)"
            )
        current = value.as_python_constant()
        if key in self.deferred_attr_mutations:
            original = self.deferred_attr_mutations[key][0]
        else:
            output_graph = self.output_graph_weakref()
            if output_graph is None:
                raise AssertionError("output_graph weakref is dead")
            tx = output_graph.current_tx
            try:
                original_vt = item.tp_getattro_impl(tx, name)  # type: ignore[arg-type]
            except NotImplementedError:
                return False
            if not original_vt.is_python_constant():
                return False
            original = original_vt.as_python_constant()
        self.deferred_attr_mutations[key] = (original, current)
        return True

    def validate_deferred_attr_mutations(self) -> None:
        """Check that all deferred attribute mutations were nullified."""
        for (_, name), (original, current) in self.deferred_attr_mutations.items():
            if original != current:
                unimplemented(
                    gb_type="HOP: Non-nullified side effect",
                    context=f"Attribute '{name}' was not restored to its original value",
                    explanation=f"Attribute '{name}' on an outer-scope object was "
                    f"changed from {original!r} to {current!r} inside a "
                    "higher-order op subgraph. Dynamo only supports mutations "
                    "that are undone before the subgraph exits (e.g., context "
                    "managers that save/restore a flag). If you intentionally "
                    "want this side effect, move the mutation outside of the "
                    "higher-order op.",
                    hints=[*graph_break_hints.FUNDAMENTAL],
                )

    def _capture_user_stack(self, key: VariableTracker) -> None:
        """Capture the current user stack from the instruction translator."""
        if config.side_effect_replay_policy == "silent":
            return
        if key not in self.mutation_user_stacks:
            self.mutation_user_stacks[key] = []
        self.mutation_user_stacks[key].append(
            torch._guards.TracingContext.extract_stack()
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SideEffects):
            raise AssertionError(f"Expected SideEffects, got {type(other)}")
        # NB: do NOT test keepalive
        return (
            self.id_to_variable == other.id_to_variable
            and self.store_attr_mutations == other.store_attr_mutations
            and self.attr_mutation_kinds == other.attr_mutation_kinds
            and self.save_for_backward == other.save_for_backward
            and self.tensor_hooks == other.tensor_hooks
        )

    def diff(self, other: "SideEffects") -> str | None:
        if self.id_to_variable != other.id_to_variable:
            sk_itv = self.id_to_variable.keys()
            ok_itv = other.id_to_variable.keys()
            if sk_itv != ok_itv:
                return f"id_to_variable keys: {sk_itv} != {ok_itv}"
            # Feel free to augment this with more fancy diffing logic
            # if needed for debugging
            return "id_to_variable: unknown diff"
        elif self.store_attr_mutations != other.store_attr_mutations:
            sk_sam = self.store_attr_mutations.keys()
            ok_sam = other.store_attr_mutations.keys()
            if sk_sam != ok_sam:
                return f"store_attr_mutations keys: {sk_sam} != {ok_sam}"
            return "store_attr_mutations: unknown diff"
        elif self.attr_mutation_kinds != other.attr_mutation_kinds:
            return "attr_mutation_kinds: unknown diff"
        elif self.save_for_backward != other.save_for_backward:
            return "save_for_backward"
        elif self.tensor_hooks != other.tensor_hooks:
            return "tensor_hooks"
        else:
            return None

    def clone(self) -> "SideEffects":
        """Create a shallow copy"""
        ref = self.output_graph_weakref()
        if ref is None:
            raise AssertionError("output_graph weakref is dead during clone")
        return self.__class__(
            output_graph=ref,
            id_to_variable=dict(self.id_to_variable),
            store_attr_mutations={
                k: dict(v) for k, v in self.store_attr_mutations.items()
            },
            attr_mutation_kinds={
                k: dict(v) for k, v in self.attr_mutation_kinds.items()
            },
            mutation_user_stacks=self.mutation_user_stacks,
            keepalive=list(self.keepalive),
            save_for_backward=self.save_for_backward,
            tensor_hooks=self.tensor_hooks,
        )

    def __contains__(self, item: object) -> bool:
        return id(item) in self.id_to_variable

    def __getitem__(self, item: object) -> VariableTracker:
        return self.id_to_variable[id(item)]

    def should_allow_externally_visible_side_effects_in_subtracer(self) -> bool:
        output_graph = self.output_graph_weakref()
        return bool(
            output_graph
            and output_graph.current_tx.output.current_tracer.unsafe_allow_externally_visible_side_effects
        )

    def should_allow_side_effects_in_hop(self) -> bool:
        output_graph = self.output_graph_weakref()
        return bool(
            output_graph
            and output_graph.current_tx.output.current_tracer.allow_side_effects_in_hop
        )

    def is_reconstructing_generator(self) -> bool:
        output_graph = self.output_graph_weakref()

        return bool(
            output_graph
            and output_graph.current_tx.output.current_tracer.is_reconstructing_generator
        )

    def _maybe_record_side_effect(self, item: VariableTracker) -> None:
        """Record the first externally-visible side effect on the current tracer."""
        if item.mutation_type is not None and not is_side_effect_safe(
            item.mutation_type
        ):
            output_graph = self.output_graph_weakref()
            if output_graph:
                tracer = output_graph.current_tx.output.current_tracer
                if tracer.side_effect_stack is None:
                    tracer.side_effect_stack = (
                        torch._guards.TracingContext.extract_stack()
                    )

    def check_allowed_side_effect(self, item: VariableTracker) -> bool:
        from torch._dynamo.variables.misc import AutogradFunctionContextVariable

        # People do things like self.dim = dim inside autograd.Function.
        # These are benign.
        if isinstance(item, AutogradFunctionContextVariable):
            return True
        if self.should_allow_externally_visible_side_effects_in_subtracer():
            self._maybe_record_side_effect(item)
            return True
        if self.should_allow_side_effects_in_hop():
            self._maybe_record_side_effect(item)
            return True
        if self.is_reconstructing_generator():
            # This is missing the case where one mutates a tensor. See
            # test_generator.py::test_reconstruct_generator_tensor_mutation
            unimplemented(
                gb_type="Generator reconstruction with mutations",
                context=f"mutating object: {item}",
                explanation="Cannot reconstruct a generator with variable mutations. "
                "Dynamo needs to fully exhaust the generator, which may cause "
                "unintended variable modifications.",
                hints=[
                    "Remove mutations from the generator.",
                    *graph_break_hints.FUNDAMENTAL,
                ],
            )
        if item.mutation_type is None:
            raise AssertionError(
                f"mutation_type is None for {item} in check_allowed_side_effect"
            )
        if not is_side_effect_safe(item.mutation_type):
            unimplemented(
                gb_type="HOP: Unsafe side effect",
                context=f"Attempted to mutate {item}",
                explanation="Mutating a variable from outside the scope of this HOP is not supported.",
                hints=[
                    "If the HOP is activation checkpointing (torch.utils.checkpoint.checkpoint), this points to a "
                    "side effect in forward method. Eager activation checkpointing replays that side-effect while "
                    "recomputing the forward in the backward. If you are ok with side-effect not replayed in the "
                    "backward, try setting `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True`",
                ],
            )
        return False

    def store_attr(
        self,
        item: VariableTracker,
        name: str,
        value: VariableTracker,
        mutation_kind: AttrMutationKind = AttrMutationKind.GENERIC_SETATTR,
        mutated_source: Source | None = None,
    ) -> None:
        if not self.is_attribute_mutation(item):
            raise AssertionError(
                f"Expected attribute mutation for {item} in store_attr"
            )
        # For constant attribute mutations on outer-scope objects, defer
        # the side-effect check and validate after tracing that the
        # mutation was nullified (value restored to original).
        deferred = False
        if (
            isinstance(item.mutation_type, AttributeMutationExisting)
            and not is_side_effect_safe(item.mutation_type)
            and not isinstance(item, AutogradFunctionContextVariable)
            and not self.should_allow_side_effects_in_hop()
            and not self.should_allow_externally_visible_side_effects_in_subtracer()
            and value.is_python_constant()
        ):
            deferred = self.snapshot_attr_mutation(item, name, value)
        if not deferred:
            self.check_allowed_side_effect(item)
        if item not in self.store_attr_mutations:
            self.store_attr_mutations[item] = {}
        if item not in self.attr_mutation_kinds:
            self.attr_mutation_kinds[item] = {}
        self.store_attr_mutations[item][name] = value
        self.attr_mutation_kinds[item][name] = mutation_kind
        # Capture user stack for this mutation
        self._capture_user_stack(item)
        if mutated_source is not None:
            self.mutated_sources.add(mutated_source)
        else:
            item_source = getattr(item, "source", None)
            if item_source is not None:
                self.mutated_sources.add(AttrSource(item_source, name))

    def store_instance_dict_attr(
        self, item: VariableTracker, name: str, value: VariableTracker
    ) -> None:
        # `obj.__dict__` ordering is observable (e.g. list(obj.__dict__)). When a
        # key is re-added after being deleted, CPython appends it at the end
        # rather than reusing its old slot. store_attr_mutations is an
        # insertion-ordered dict, so a plain re-store of an existing key would
        # keep the stale position; drop the deleted entry first so the new value
        # re-inserts at the end.
        if not isinstance(value, variables.DeletedVariable):
            existing = self.store_attr_mutations.get(item, {}).get(name)
            if isinstance(existing, variables.DeletedVariable):
                # store_attr always writes both maps together, so the
                # attr_mutation_kinds entry is guaranteed present here; use del
                # for both to surface any invariant violation.
                del self.store_attr_mutations[item][name]
                del self.attr_mutation_kinds[item][name]
        self.store_attr(item, name, value, AttrMutationKind.INSTANCE_DICT)

    def get_attr_mutation_kind(
        self, item: VariableTracker, name: str
    ) -> AttrMutationKind:
        if name not in self.store_attr_mutations.get(item, ()):
            return AttrMutationKind.GENERIC_SETATTR
        if name not in self.attr_mutation_kinds.get(item, ()):
            raise AssertionError(f"Missing attribute mutation kind for {item}.{name}")
        return self.attr_mutation_kinds[item][name]

    def load_attr(
        self,
        item: VariableTracker,
        name: str,
        deleted_ok: bool = False,
        check: bool = False,
    ) -> VariableTracker:
        if check:
            if not self.is_attribute_mutation(item):
                raise AssertionError(
                    f"Expected attribute mutation for {item} in load_attr"
                )
        result = self.store_attr_mutations[item][name]
        if not deleted_ok and isinstance(result, variables.DeletedVariable):
            unimplemented(
                gb_type="Attempted to read a deleted variable",
                context=f"item: {item}, name: {name}",
                explanation="",
                hints=[*graph_break_hints.USER_ERROR],
            )
        return result

    def store_cell(self, cellvar: VariableTracker, value: VariableTracker) -> None:
        if cellvar.is_immutable():
            unimplemented(
                gb_type="Write to immutable cell",
                context=f"cellvar: {cellvar}, value: {value}",
                explanation="Dynamo doesn't support writing to immutable/sourceless cell variables.",
                hints=[*graph_break_hints.DIFFICULT],
            )
        if not isinstance(cellvar, variables.CellVariable):
            raise AssertionError(
                f"Expected CellVariable, got {type(cellvar)} in store_cell"
            )
        if not isinstance(value, variables.VariableTracker):
            raise AssertionError(
                f"Expected VariableTracker, got {type(value)} in store_cell"
            )
        self.store_attr(cellvar, "cell_contents", value)

    def load_cell(self, cellvar: VariableTracker) -> VariableTracker:
        if not isinstance(cellvar, variables.CellVariable):
            raise AssertionError(
                f"Expected CellVariable, got {type(cellvar)} in load_cell"
            )
        # Track the cell_contents source during subgraph tracing so that
        # mutations (e.g. nonlocal counter = 3) are detected by the reuse
        # mechanism via set intersection with mutated_sources.
        output_graph = self.output_graph_weakref()
        if output_graph:
            cell_source = getattr(cellvar, "source", None)
            if cell_source is not None:
                output_graph.current_tx.output.current_tracer.traced_sources.add(
                    AttrSource(cell_source, "cell_contents")
                )
        if self.has_pending_mutation_of_attr(cellvar, "cell_contents"):
            return self.load_attr(cellvar, "cell_contents", check=False)
        if cellvar.pre_existing_contents:
            return cellvar.pre_existing_contents
        unimplemented(
            gb_type="Read uninitialized cell",
            context=str(cellvar),
            explanation="Attempted to read a cell variable that has not been populated yet.",
            hints=[*graph_break_hints.USER_ERROR],
        )

    def load_global(self, gvar: VariableTracker, name: str) -> VariableTracker:
        if not isinstance(gvar, variables.VariableTracker):
            raise AssertionError(
                f"Expected VariableTracker, got {type(gvar)} in load_global"
            )
        # This path serves the read out of side effects, bypassing
        # VariableBuilder, so record the source here the way load_cell does --
        # otherwise a subgraph reading a rebound global has no traced source to
        # intersect with mutated_sources and is wrongly considered reusable.
        output_graph = self.output_graph_weakref()
        if output_graph:
            output_graph.current_tx.output.current_tracer.traced_sources.add(
                GlobalSource(name)
            )
        return self.load_attr(gvar, name)

    def store_global(
        self, gvar: VariableTracker, name: str, value: VariableTracker
    ) -> None:
        if not isinstance(gvar, variables.VariableTracker):
            raise AssertionError(
                f"Expected VariableTracker for gvar, got {type(gvar)} in store_global"
            )
        if not isinstance(value, variables.VariableTracker):
            raise AssertionError(
                f"Expected VariableTracker for value, got {type(value)} in store_global"
            )
        # gvar is a per-global sentinel holder whose own source already names
        # the global, so store_attr's default AttrSource(item.source, name)
        # would name a nonexistent `G.G`. Record the source readers actually
        # use, so mutated_sources intersects with traced_sources.
        self.store_attr(gvar, name, value, mutated_source=GlobalSource(name))

    @staticmethod
    def cls_supports_mutation_side_effects(cls: type) -> bool:
        return inspect.getattr_static(cls, "__getattribute__", None) in (
            object.__getattribute__,
            dict.__getattribute__,
            set.__getattribute__,
            frozenset.__getattribute__,
            int.__getattribute__,
            str.__getattribute__,
            list.__getattribute__,
            tuple.__getattribute__,
            collections.deque.__getattribute__,
            BaseException.__getattribute__,
        )

    def is_attribute_mutation(self, item: VariableTracker) -> bool:
        return isinstance(item.mutation_type, AttributeMutation)

    def has_pending_mutation(self, item: VariableTracker) -> bool:
        return self.is_attribute_mutation(item) and bool(
            self.store_attr_mutations.get(item)
        )

    def has_pending_mutation_of_attr(
        self,
        item: VariableTracker,
        name: str,
        mutation_kinds: AttrMutationKind | tuple[AttrMutationKind, ...] | None = None,
    ) -> bool:
        if not (
            self.is_attribute_mutation(item)
            and name in self.store_attr_mutations.get(item, ())
        ):
            return False
        if mutation_kinds is None:
            return True
        mutation_kind = self.get_attr_mutation_kind(item, name)
        if isinstance(mutation_kinds, AttrMutationKind):
            return mutation_kind is mutation_kinds
        return mutation_kind in mutation_kinds

    def is_modified(self, item: VariableTracker) -> bool:
        if item.is_immutable():
            return False
        if isinstance(item.mutation_type, (AttributeMutationNew, ValueMutationNew)):
            return True

        if isinstance(item, variables.UserDefinedObjectVariable):
            # Checks if the underlying dict or tuple vt has been modified
            return item in self.store_attr_mutations or item.is_base_vt_modified(self)

        if self.is_attribute_mutation(item):
            return item in self.store_attr_mutations
        if item.mutation_type is None:
            raise AssertionError(f"mutation_type is None for {item} in is_modified")
        return item.mutation_type.is_modified  # type: ignore[attr-defined]

    def _track_obj(
        self,
        item: object,
        variable: VariableTracker,
        mutation_type_cls: type = ValueMutationExisting,
    ) -> VariableTracker:
        """Start tracking an existing or new variable for mutation"""
        if id(item) in self.id_to_variable:
            raise AssertionError(
                f"{variable} is already tracked for mutation. This could be "
                "because you are not using VariableBuilder to construct "
                "the variable tracker. "
                f"Source of new object: {variable.source}. "
                f"Source of previously tracked object: {self.id_to_variable[id(item)].source}."
            )

        variable.mutation_type = mutation_type_cls()
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        return variable

    track_mutable = _track_obj

    def track_object_existing(
        self,
        item: object,
        variable: VariableTracker,
    ) -> VariableTracker:
        # TODO: Modify this API so that we preserve type info of
        # variable
        return self._track_obj(
            item,
            variable,
            mutation_type_cls=AttributeMutationExisting,
        )

    def track_attribute_mutation_new(self, variable: VariableTracker) -> None:
        """Register a sourceless VT (e.g. a synthetic exception) as a new
        attribute-mutation object so its attribute mutations are codegen'd.
        Keyed by the VT's id since there is no backing Python object."""
        if id(variable) in self.id_to_variable:
            return
        variable.mutation_type = AttributeMutationNew()
        self.id_to_variable[id(variable)] = variable
        self.keepalive.append(variable)

    def track_object_new(
        self,
        cls_source: Source | None,
        user_cls: Any,
        variable_cls: Any,
        options: dict[str, Any],
    ) -> VariableTracker:
        obj = object_new(user_cls)
        variable = variable_cls(
            obj,
            mutation_type=AttributeMutationNew(cls_source),
            **options,
        )
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        return variable

    def get_variable_cls(self, user_cls: type) -> type:
        from torch.overrides import TorchFunctionMode

        from .variables.ctx_manager import GenericContextWrappingVariable
        from .variables.torch_function import TorchFunctionModeVariable
        from .variables.user_defined import is_generic_ctx_manager_cls

        variable_cls: type[variables.UserDefinedObjectVariable] = (
            variables.UserDefinedObjectVariable
        )
        if issubclass(
            user_cls, TorchFunctionMode
        ) and TorchFunctionModeVariable.is_supported_torch_function_mode(user_cls):
            variable_cls = TorchFunctionModeVariable
        elif is_generic_ctx_manager_cls(user_cls):
            variable_cls = GenericContextWrappingVariable
        elif issubclass(user_cls, torch.nn.Module):
            variable_cls = variables.UnspecializedNNModuleVariable
        elif issubclass(user_cls, collections.defaultdict):
            variable_cls = variables.DefaultDictVariable
        elif issubclass(user_cls, dict):
            # Includes collections.OrderedDict and its subclasses; the
            # UserDefinedDictVariable picks an OrderedDict-backed store.
            variable_cls = variables.UserDefinedDictVariable
        elif issubclass(user_cls, (set, frozenset)):
            variable_cls = variables.UserDefinedSetVariable
        elif issubclass(user_cls, tuple):
            if is_namedtuple_cls(user_cls):
                variable_cls = variables.UserDefinedTupleVariable.get_vt_cls(user_cls)
            else:
                variable_cls = variables.UserDefinedTupleVariable
        elif issubclass(user_cls, list):
            variable_cls = variables.UserDefinedListVariable
        elif issubclass(user_cls, collections.deque):
            variable_cls = variables.UserDefinedDequeVariable
        elif issubclass(user_cls, MutableMapping):
            variable_cls = variables.MutableMappingVariable
        elif is_frozen_dataclass(user_cls):
            variable_cls = FrozenDataClassVariable
        elif issubclass(user_cls, BaseException):
            variable_cls = variables.UserDefinedExceptionObjectVariable
        elif issubclass(
            user_cls,
            variables.user_defined._CONSTANT_BASE_TYPES,
        ):
            variable_cls = variables.UserDefinedConstantVariable
        elif variables.InspectVariable.is_matching_class(user_cls):
            variable_cls = variables.InspectVariable
        if not issubclass(variable_cls, variables.UserDefinedObjectVariable):
            raise AssertionError(
                f"Expected subclass of UserDefinedObjectVariable, got {variable_cls}"
            )
        return variable_cls

    def get_example_value(
        self,
        base_cls_vt: VariableTracker,
        cls_vt: VariableTracker,
        init_args: list[VariableTracker],
    ) -> Any:
        user_cls = cls_vt.value  # type: ignore[attr-defined]
        if issubclass(user_cls, torch.nn.Module):
            # TODO(anijain2305) - Is it possible to remove this specialization?
            obj = nn_module_new(user_cls)
        else:
            if isinstance(base_cls_vt, variables.BuiltinVariable):
                base_cls = base_cls_vt.fn
            elif isinstance(base_cls_vt, variables.DictBuiltinVariable):
                base_cls = dict
            elif isinstance(base_cls_vt, variables.ListBuiltinVariable):
                base_cls = list
            elif isinstance(base_cls_vt, variables.UserDefinedClassVariable):
                base_cls = base_cls_vt.value
            else:
                raise RuntimeError(f"Unexpected base_cls_vt {base_cls_vt}")

            if not variables.UserDefinedClassVariable.is_supported_new_method(
                base_cls.__new__
            ):
                raise AssertionError(f"Unsupported __new__ method for {base_cls}")
            if is_structseq_class(user_cls):
                # Structseq tp_new requires a sequence argument and rejects
                # tuple.__new__, so create a dummy with None placeholders.
                obj = user_cls([None] * user_cls.n_fields)
            elif init_args and issubclass(
                user_cls,
                variables.user_defined._CONSTANT_BASE_TYPES,
            ):
                example_args = [arg.as_python_constant() for arg in init_args]
                try:
                    obj = base_cls.__new__(  # pyrefly: ignore[bad-specialization]
                        user_cls, *example_args
                    )
                except Exception:
                    # __new__ can raise (e.g., exceeding int str digit limits).
                    # Fall back to creating without args — the example value is
                    # only used for tracing, not for correctness.
                    obj = base_cls.__new__(  # pyrefly: ignore[bad-specialization]
                        user_cls
                    )
            else:
                try:
                    obj = base_cls.__new__(user_cls)
                except TypeError as exc:
                    # Backstop for direct construction paths that bypass the
                    # UserDefinedClassVariable object.__new__ preflight.
                    unimplemented(
                        gb_type="Unsupported user-defined object construction during side-effect tracking",
                        context=f"class={user_cls}, base={base_cls}, error={exc}",
                        explanation=(
                            "Dynamo could not construct an example object for "
                            "side-effect replay using the class __new__ method."
                        ),
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )
        return obj

    def track_new_user_defined_object(
        self,
        base_cls_vt: VariableTracker,
        cls_vt: VariableTracker,
        init_args: list[VariableTracker],
        *,
        tx: "InstructionTranslatorBase | None" = None,
    ) -> VariableTracker:
        """
        Creates a UserDefinedObjectVariable (or its subclass) variable tracker
        and mark it for attribute mutation tracking.

        Also records the variable trackers to call __new__ method on
        reconstruction. Roughly, the reconstruction looks like this
            base_cls_vt.__new__(user_cls, *init_args)
        """
        cls_source = cls_vt.source
        user_cls = cls_vt.value  # type: ignore[attr-defined]
        variable_cls = self.get_variable_cls(user_cls)
        obj = self.get_example_value(base_cls_vt, cls_vt, init_args)

        kwargs: dict[str, Any] = {}
        if tx is not None:
            kwargs["tx"] = tx
        variable = variable_cls(
            obj,
            cls_source=cls_vt.source,
            base_cls_vt=base_cls_vt,
            init_args=init_args,
            mutation_type=AttributeMutationNew(cls_source),
            **kwargs,
        )
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        return variable

    def track_cell_new(
        self,
    ) -> VariableTracker:
        obj = object()
        variable = variables.CellVariable(
            mutation_type=AttributeMutationNew(),
        )
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        return variable

    def track_cell_existing(
        self, source: Source | None, cell: CellType, contents: VariableTracker
    ) -> VariableTracker:
        variable = variables.CellVariable(
            # We don't support mutation to cell without source because we need
            # source to properly codegen the mutations.
            mutation_type=None if source is None else AttributeMutationExisting(),
            pre_existing_contents=contents,
            source=source,
        )
        self.id_to_variable[id(cell)] = variable
        self.keepalive.append(cell)
        return variable

    def track_global_existing(self, source: Source, item: object) -> VariableTracker:
        variable = variables.NewGlobalVariable(
            mutation_type=AttributeMutationExisting(),
            source=source,
        )
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        return variable

    def track_save_for_backward(
        self, ctx: VariableTracker, args: list[VariableTracker]
    ) -> None:
        if not isinstance(ctx, variables.AutogradFunctionContextVariable):
            raise AssertionError(
                f"Expected AutogradFunctionContextVariable, got {type(ctx)}"
            )
        self.save_for_backward.append((ctx, args))

    def track_runahead_tensor_and_symvar_side_effects(
        self, other: "SideEffects"
    ) -> None:
        # In higher order ops we want to keep track of tensors seen in the
        # speculate_subgraph so that we don't lift them again as a new input in
        # other speculate_subgraph or in the root tracer.
        for other_item in other.keepalive:
            other_id = id(other_item)
            other_variable = other.id_to_variable[other_id]
            if other_id not in self.id_to_variable and isinstance(
                other_variable, (variables.TensorVariable, variables.SymNodeVariable)
            ):
                self.track_object_existing(other_item, other_variable)

    def prune_dead_object_new(self, tx: "InstructionTranslatorBase") -> None:
        # Avoid VT cycles from e.g., recursive function.
        visited: set[VariableTracker] = set()
        live_new_objects: set[VariableTracker] = set()

        def visit(var: VariableTracker) -> None:
            if var in visited:
                return
            visited.add(var)
            # Object may have been mutated, store this mutation.
            if isinstance(var.mutation_type, AttributeMutationNew):
                live_new_objects.add(var)
            # It's possible that we have mutated the value of this variable
            # to be another one. The new value is in store_attr_mutations.
            # Also recurse through the new value to detect alive AttributeMutationNew.
            if var in self.store_attr_mutations:
                VariableTracker.visit(
                    visit,  # noqa: F821
                    self.store_attr_mutations[var],
                )

        def is_live(var: VariableTracker) -> bool:
            if isinstance(var.mutation_type, AttributeMutationNew):
                return var in live_new_objects
            return True

        pre_existing_vars = [
            var
            for var in self.id_to_variable.values()
            if not isinstance(var.mutation_type, AttributeMutationNew)
        ]

        # The only live side effects come from returns (tx.stack), any intermediates
        # during a graph break (tx.symbolic_locals), and mutation on pre-existing variables.
        # Recursively visit Variables and see if any of them have been mutated.
        init_live_vars = []
        # gather stack/symbolic_locals for all tx's up the chain
        cur_tx: InstructionTranslatorBase | None = tx
        while cur_tx is not None:
            init_live_vars.extend([cur_tx.stack, cur_tx.symbolic_locals])
            if cur_tx.parent is not None:
                # for non-root tx'es, also keep the cells/freevars alive so they get codegen'd properly
                # TODO see if we could prune dead cells - cell pruning information needs to be forwarded
                # to the resume function creation as well.
                if cur_tx.post_prune_cell_and_freevars is None:
                    raise AssertionError(
                        "post_prune_cell_and_freevars is None for non-root tx"
                    )
                init_live_vars.append(cur_tx.post_prune_cell_and_freevars)
            cur_tx = cur_tx.parent

        # Generators tracked for close-on-exit have their close() called in
        # compile_subgraph after pruning.
        gen_reachable_new: set[VariableTracker] = set()

        def visit_gen(var: VariableTracker) -> None:
            if isinstance(var.mutation_type, AttributeMutationNew):
                gen_reachable_new.add(var)

        # Pass side_effects so the walk follows store_attr_mutations, and share
        # one cache across all generators to dedup shared subiterators.
        gen_visit_cache: dict[int, Any] = {}
        for gen in tx.output.local_generators:
            gen_tracer = gen.inline_tracer
            VariableTracker.visit(
                visit_gen,
                [gen_tracer.stack, gen_tracer.symbolic_locals],
                gen_visit_cache,
                self,
            )

        VariableTracker.visit(
            visit,
            # TODO track from all possible sources.
            init_live_vars
            + [
                pre_existing_vars,
                tx.output.backward_state,
                self.tensor_hooks,
            ],
        )
        # Manually release the self-referential function, which indirectly
        # captures certain `VariableTracker` and affects parts of PT test/logic
        # that are sensitive to when certain objects get released.
        del visit

        # NB: cell variable handling is tricky.
        # cell variables must stay alive if any NestedUserFunctionVariable
        # are live. "visit"-ing the NestedUserFunctionVariable visits
        # the .closures field, from which we will see if we need to keep
        # any mutations to cell variables alive.

        self.id_to_variable = {
            k: v for k, v in self.id_to_variable.items() if is_live(v)
        }
        self.store_attr_mutations = {
            k: v
            for k, v in self.store_attr_mutations.items()
            if is_live(k) or k in gen_reachable_new
        }
        self.attr_mutation_kinds = {
            k: v
            for k, v in self.attr_mutation_kinds.items()
            if is_live(k) or k in gen_reachable_new
        }

    def mutation(self, var: VariableTracker) -> None:
        if var in self.ignore_mutation_on_these_variables:
            return

        self.check_allowed_side_effect(var)
        # Capture user stack for this mutation
        self._capture_user_stack(var)

        if isinstance(var.mutation_type, ValueMutationExisting):
            var.mutation_type.is_modified = True
        if var.source is not None:
            self.mutated_sources.add(var.source)
        if var.source and isinstance(var, variables.ConstDictVariable):
            self._has_existing_dict_mutation = True

    def has_existing_dict_mutation(self) -> bool:
        return self._has_existing_dict_mutation

    def _get_modified_vars(self) -> list[VariableTracker]:
        return [var for var in self.id_to_variable.values() if self.is_modified(var)]

    def codegen_save_tempvars(self, cg: PyCodegen) -> None:
        # We must codegen modified VT to their source by default, so that
        # mutation and aliasing are properly accounted for.
        #
        # Since newly constructed objects don't have a source, we manually
        # codegen their construction and store them to a newly assigned local
        # source. Note that `ValueMutationNew` isn't tracked by SideEffects.
        for var in self._get_modified_vars():
            if not isinstance(var.mutation_type, AttributeMutationNew):
                if var.source is None:
                    raise AssertionError(
                        f"Expected source for modified var {var} "
                        "with non-new mutation type"
                    )
                continue

            # Namedtuples/structseqs with no pending mutations should skip
            # codegen_save_tempvars so that restore_stack handles them. In
            # export, restore_stack uses value_from_source=False which makes
            # child tensors become graph outputs. If we processed them here,
            # add_cache would assign a TempLocalSource and restore_stack would
            # load from cache with value_from_source=True, hiding the tensors
            # from export.
            if isinstance(
                var,
                (variables.NamedTupleVariable, variables.StructSequenceVariable),
            ) and not self.has_pending_mutation(var):
                continue

            # Sourceless enum members registered with AttributeMutationNew
            # don't need __new__-based tempvar reconstruction -- they are
            # reconstructible as constants.
            if (
                var.is_python_constant()
                and isinstance(var.mutation_type, AttributeMutationNew)
                and isinstance(var, variables.UserDefinedObjectVariable)
                and (
                    isinstance(
                        var.value,
                        (
                            enum.Enum,
                            torch.DispatchKey,
                            torch._C._functorch.TransformType,
                        ),
                    )
                    or is_pybind11_enum_member(var.value)
                )
            ):
                continue

            if isinstance(var, variables.CellVariable):
                # Cells created in the root frame are created either by
                # `MAKE_CELL` or by them being in `co_cellvars`, so we only emit
                # `make_cell` for the non-root-frame cells here.
                # TODO generalize this so we never need to call `make_cell`.
                if var.local_name is None:
                    cg.add_push_null(
                        lambda: cg.load_import_from(utils.__name__, "make_cell")
                    )
                    cg.extend_output(create_call_function(0, False))
                    cg.add_cache(var)
                    var.source = TempLocalSource(cg.tempvars[var])  # type: ignore[attr-defined]
                elif var.source is None:
                    var.source = LocalCellSource(var.local_name)
            elif var.is_tensor():
                # NOTE: for historical reasons we never assigned local sources
                # to newly constructed tensor object, so we keep it that way.
                # They are always loaded from output of the fx graph, so one can
                # think of it as having a "OutputGraphSource" for codegen
                # purposes.
                #
                # However, tensor subclass objects are different, because the
                # reconstruction logic in `PyCodegen` loads the data tensor from
                # graph output and then calls `as_subclass`, meaning we must
                # assign a source to it to ensure we only reconstruct one
                # subclass instance.
                if isinstance(
                    var, variables.torch_function.TensorWithTFOverrideVariable
                ):
                    # Don't codegen from temp source assigned from the 1st pass.
                    cg(var, allow_cache=False)
                    cg.add_cache(var)
                    # `add_cache` generates STORE and consumes TOS, but we never
                    # cleared it. TODO move this call into `add_cache`
                    cg.clear_tos()
                    var.source = TempLocalSource(cg.tempvars[var])
            elif isinstance(var, variables.AutogradFunctionContextVariable):
                unimplemented(
                    gb_type="AutogradFunctionContextVariable escaped Dynamo-traced region",
                    context="",
                    explanation="We cannot reconstruct a torch.autograd.Function's context object.",
                    hints=[],
                )
            elif isinstance(var, variables.ExceptionVariable):
                # Exceptions cannot be built via object.__new__ (CPython rejects
                # it), so reconstruct() constructs by calling the type. Cache
                # the result so later references load the single instance; any
                # __dict__ attributes replay in codegen_update_mutated.
                var.reconstruct(cg)
                cg.add_cache(var)
                var.source = TempLocalSource(cg.tempvars[var])
            else:
                # Reconstruct the bytecode for
                # base_cls.__new__(user_cls, *args)
                if isinstance(var, variables.UserDefinedObjectVariable):

                    def load_new_method() -> None:
                        # pyrefly: ignore [missing-attribute]
                        if var.base_cls_vt is None:
                            raise AssertionError(
                                "base_cls_vt is None in load_new_method"
                            )
                        cg(var.base_cls_vt)  # type: ignore[attr-defined]
                        cg.extend_output([cg.create_load_attr("__new__")])

                    cg.add_push_null(load_new_method)
                else:
                    cg.add_push_null(
                        lambda: cg.load_import_from(utils.__name__, "object_new")
                    )
                if var.mutation_type.cls_source is None:
                    unimplemented(
                        gb_type="Reconstruct user defined class without a source",
                        context=f"Class: {var.python_type().__name__}",
                        explanation=(
                            f"Cannot reconstruct an instance of {var.python_type().__name__} "
                            "that escapes the compiled region. This happens when the class is "
                            "defined dynamically (e.g., inside the compiled function) and the "
                            "class object itself has no source that can be reconstructed. "
                            "To fix this, move the class definition outside the compiled function "
                            "or prevent the object from escaping the compiled region."
                        ),
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )
                cg(var.mutation_type.cls_source)

                # Generate the args to the __new__ method
                for arg in var.init_args:  # type: ignore[attr-defined]
                    cg(arg)

                # Call the __new__ method
                cg.extend_output(create_call_function(1 + len(var.init_args), False))  # type: ignore[attr-defined]

                cg.add_cache(var)
                var.source = TempLocalSource(cg.tempvars[var])

                # For frozen dataclasses, we must emit object.__setattr__
                # immediately after __new__ — before any other code can
                # access the object.  The suffix-based codegen in
                # codegen_update_mutated runs too late: if intervening code
                # calls __repr__ (e.g. f-strings), the attributes won't be
                # set yet.
                if (
                    isinstance(var, variables.FrozenDataClassVariable)
                    and var in self.store_attr_mutations
                ):
                    for name, value in self.store_attr_mutations[var].items():
                        cg.load_import_from("builtins", "object")
                        cg.load_method("__setattr__")
                        cg(var.source)
                        cg(variables.ConstantVariable(name))
                        cg(value)
                        cg.extend_output(
                            [*create_call_method(3), create_instruction("POP_TOP")]
                        )

        for ctx, args in self.save_for_backward:
            cg(ctx.source)
            cg.load_method("save_for_backward")
            for arg in args:
                cg(arg)
            cg.extend_output(
                [
                    *create_call_method(len(args)),
                    create_instruction("POP_TOP"),
                ]
            )

    def register_hook(
        self,
        tensor: "variables.TensorVariable",
        hook: VariableTracker,
        handle: "variables.RemovableHandleVariable",
        name: str,
    ) -> None:
        if not tensor.is_tensor():
            raise AssertionError(
                f"Expected tensor variable, got {type(tensor)} in register_hook"
            )
        if not isinstance(hook, variables.VariableTracker):
            raise AssertionError(f"Expected VariableTracker for hook, got {type(hook)}")
        if not isinstance(handle, variables.RemovableHandleVariable):
            raise AssertionError(
                f"Expected RemovableHandleVariable, got {type(handle)}"
            )
        if not handle.is_mutable():
            raise AssertionError("handle must be mutable in register_hook")
        if not hasattr(torch.Tensor, name):
            raise AssertionError(f"torch.Tensor has no attribute '{name}'")
        idx = len(self.tensor_hooks.keys())
        # duplicate index possible because of self.remove_hook()
        while idx in self.tensor_hooks:
            idx += 1
        self.tensor_hooks[idx] = (tensor, hook, handle, name)
        if handle.idx:
            raise AssertionError(f"handle.idx should be falsy, got {handle.idx}")
        handle.idx = idx

    def remove_hook(self, idx: int) -> None:
        del self.tensor_hooks[idx]

    def codegen_hooks(self, cg: PyCodegen) -> None:
        for (
            tensor,
            hook,
            handle,
            name,
        ) in self.tensor_hooks.values():
            # Note: [On tensor.register_hook]
            #
            # register_hook on a tensor, AKA backward hooks, have slightly nuanced differences in how they are implemented
            # when it comes to hooks on objects with sources (inputs, params) vs objects without sources (intermediaries).
            #
            # For tensors with a source, we bypass direct inclusion of register_hook calls in the graph.
            # Instead, these are tracked and stashed as a global variable, enabling their association with tensors in
            # the residuals. During dynamo's frame creation, these hooks are invoked seamlessly on known reconstructible/fetch-able
            # tensors. Because a source indicates knowledge of this object outside the torch compile region, and
            # because we are running residuals firmly before .backward() can be run, it is sound to invoke
            # `register_hook` on a known tensor.
            #
            # For tensors without a source, we support a limited subset of hooks. Global functions only, and
            # compiled_autograd must be enabled or we will graph break.
            #
            # Handling the Handle: When a user retains the register_hook result in a handle, we intercept the
            # STORE_FAST operation to record the user-designated local variable name. This ensures the reconstructed
            # bytecode retains this name. If no handle is defined, we simply pop the generated value to keep the
            # stack intact.
            #
            # Dynamo Tensor Hooks Workflow:
            # - Functions passed to register_hook are lifted globally.
            # - For tensors with sources:
            #   - In the "side_effects" phase of codegen, we iterate over tensors with hooks to:
            #     - Generate the tensor.
            #     - Issue a register_hook call on the tensor, linking to the globally stored function.
            #     - Incorporate a handle if one was established in the eager phase.
            #  - For tensors without sources:
            #    - We don't generate any instructions for registering a hook.
            #    - Handles from intermediary hooks are NYI.
            #    - We produce a call function that utilizes the trace_wrapped higher order op, closing over it.
            #    - We then manually insert the call function above into the graph.
            # - The handle's exact user-specified name, "user_code_variable_name", is discerned and associated during STORE_FAST.
            if not tensor.source:
                raise AssertionError(
                    "Hooks on non input tensors NYI - should not get here"
                )

            def gen_fn() -> None:
                cg(tensor)
                cg.extend_output([cg.create_load_attr(name)])

            cg.add_push_null(gen_fn)
            cg(hook)
            cg.extend_output(create_call_function(1, False))

            # Adding the handle to the cache means RemovableHandleVariable().reconstruct() will
            # be associated with the return value of register_hook().  This consumes the top of stack.
            cg.add_cache(handle)

    def get_ca_final_callbacks_var(self) -> "variables.ListVariable":
        from .variables.base import ValueMutationNew

        if self.ca_final_callbacks_var is None:
            self.ca_final_callbacks_var = variables.ListVariable(
                [], mutation_type=ValueMutationNew()
            )

        return self.ca_final_callbacks_var

    def _format_side_effect_message(self, var: VariableTracker) -> str:
        """Format a side effect log message with user stack."""
        if config.side_effect_replay_policy == "silent":
            raise AssertionError(
                "_format_side_effect_message should not be called "
                "when side_effect_replay_policy is 'silent'"
            )
        locations = self.mutation_user_stacks.get(var, [])
        description = f"Mutating object of type {var.python_type_name()}"
        source_info = " (no source)"
        if var.source is not None:
            if isinstance(var.source, TempLocalSource):
                source_info = " (source: created in torch.compile region)"
            elif isinstance(var, variables.CellVariable) and var.local_name is not None:
                source_info = f" (source: {var.local_name})"
            elif isinstance(
                var, variables.torch_function.TorchFunctionModeStackVariable
            ):
                source_info = " (source: torch function mode stack mutation)"
            else:
                # NOTE: NotImplementedError from var.source.name is a bug and must be fixed!
                source_info = f" (source name: {var.source.name})"

        if locations:
            # Format and dedupe stacks using tuple representation for efficiency
            seen = set()
            unique_formatted_stacks: list[str] = []
            stack_above_dynamo = collapse_resume_frames(get_stack_above_dynamo())
            for stack in locations:
                # Use tuple of frame info for fast deduplication
                # Include position info (colno, end_lineno, end_colno) to distinguish
                # multiple mutations on the same line (when available in Python 3.11+)
                stack_tuple = tuple(
                    (
                        f.filename,
                        f.lineno,
                        f.name,
                        f.line,
                        getattr(f, "colno", None),
                        getattr(f, "end_lineno", None),
                        getattr(f, "end_colno", None),
                    )
                    for f in stack
                )
                if stack_tuple not in seen:
                    seen.add(stack_tuple)
                    stack_augmented = collapse_resume_frames(stack_above_dynamo + stack)
                    unique_formatted_stacks.append(
                        "".join(traceback.format_list(stack_augmented))
                    )
            formatted_lines: str = "\n********\n\n".join(unique_formatted_stacks)
            log_str = f"{description}{source_info}\n\n{textwrap.indent(formatted_lines, '    ')}"
        else:
            log_str = (
                f"{description}{source_info} (unable to find user stacks for mutations)"
            )

        return log_str

    def _emit_side_effect_messages(self, side_effect_messages: list[str]) -> None:
        if not side_effect_messages:
            return

        for msg in side_effect_messages:
            side_effects_log.debug(msg)

        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_side_effects",
                "encoding": "string",
            },
            payload_fn=lambda: "\n\n========================================\n\n".join(
                side_effect_messages
            ),
        )

    def codegen_update_mutated(
        self, cg: PyCodegen, log_side_effects: bool = False
    ) -> None:
        side_effect_messages: list[str] = []

        # NOTE: should only be called once per VT - only if a side effect actually gets codegen'd!
        def _maybe_log_side_effect(var: VariableTracker) -> None:
            if config.side_effect_replay_policy != "silent" and log_side_effects:
                msg = self._format_side_effect_message(var)
                side_effect_messages.append(msg)

        suffixes: list[list[Instruction]] = []
        for var in self._get_modified_vars():
            # When replay_side_effects=False, only update variables with TempLocalSource
            if not config.replay_side_effects and not isinstance(
                var.source, TempLocalSource
            ):
                continue

            ctx = SideEffectReplayContext(
                side_effects=self,
                codegen=cg,
                var=var,
                suffixes=suffixes,
                log=_maybe_log_side_effect,
            )
            handler = _side_effect_replay_registry.lookup_handler(ctx)
            if handler is None:
                raise AssertionError(type(var))
            handler.codegen(ctx)

        # do all the actual mutations at the very end to handle dependencies
        for suffix in reversed(suffixes):
            cg.extend_output(suffix)

        # Send batched structured trace for all side effects in this compilation
        if log_side_effects and side_effect_messages:
            self._emit_side_effect_messages(side_effect_messages)

    def log_side_effects_summary(self) -> None:
        if config.side_effect_replay_policy == "silent":
            return
        if not side_effects_log.isEnabledFor(logging.DEBUG):
            return
        for var in self._get_modified_vars():
            msg = self._format_side_effect_message(var)
            side_effects_log.debug(msg)

    def is_empty(self) -> bool:
        return not (
            any(map(self.is_modified, self.id_to_variable.values()))
            or self.tensor_hooks
            or self.save_for_backward
            or self.tensor_hooks
        )

    def clear(self) -> None:
        self.keepalive.clear()
        self.id_to_variable.clear()


@register_side_effect_replay_handler(
    name="list_mutation",
    matcher=lambda ctx: isinstance(ctx.var, variables.ListVariable),
    priority=90,
)
def _codegen_list_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.ListVariable):
        raise AssertionError(type(var))
    # old[:] = new
    cg(var, allow_cache=False)  # Don't codegen via source
    cg(var.source)  # type: ignore[attr-defined]
    cg.extend_output(
        [
            cg.create_load_const(None),
            cg.create_load_const(None),
            create_instruction("BUILD_SLICE", arg=2),
        ]
    )
    ctx.suffixes.append([create_instruction("STORE_SUBSCR")])
    ctx.log(var)


@register_side_effect_replay_handler(
    name="deque_mutation",
    matcher=lambda ctx: isinstance(ctx.var, variables.lists.DequeVariable),
    priority=80,
)
def _codegen_deque_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.lists.DequeVariable):
        raise AssertionError(type(var))
    # For limited maxlen, the order of operations matter for side effect, but we
    # currently don't track the order, so no support.
    if not var.maxlen.is_constant_none():
        unimplemented(
            gb_type="Side effect on existing deque with limited maxlen",
            context="",
            explanation="This is not supported.",
            hints=[
                "Don't use a deque with `maxlen` specified.",
            ],
        )

    # old.extend(new), this runs last
    cg(var.source)
    cg.load_method("extend")
    cg(var, allow_cache=False)  # Don't codegen via source
    ctx.suffixes.append(
        [
            *create_call_method(1),
            create_instruction("POP_TOP"),
        ]
    )

    # old.clear(), this runs first
    cg(var.source)
    cg.load_method("clear")
    ctx.suffixes.append(
        [
            *create_call_method(0),
            create_instruction("POP_TOP"),
        ]
    )
    ctx.log(var)


@register_side_effect_replay_handler(
    name="const_dict_or_set_mutation",
    matcher=lambda ctx: isinstance(
        ctx.var, (variables.ConstDictVariable, variables.SetVariable)
    ),
    priority=70,
)
def _codegen_const_dict_or_set_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, (variables.ConstDictVariable, variables.SetVariable)):
        raise AssertionError(type(var))
    # Reconstruct works as follow:
    # (1) Skip codegen if there are no new items
    # (2) codegen(...) each pair of key/value
    # (3) create a new dictionary with the pairs of key/values above
    # (4) clear the original dictionary
    #   + only if a key was removed from the input dict
    # (5) update the original dictionary with the dict created in (2)
    if var.has_new_items():
        cg(var.source)  # type: ignore[attr-defined]
        cg.load_method("update")
        cg(var, allow_cache=False)  # Don't codegen via source

        if var.should_reconstruct_all:
            cg(var.source)  # type: ignore[attr-defined]
            cg.load_method("clear")

        ctx.suffixes.append(
            [
                *create_call_method(1),  # update
                create_instruction("POP_TOP"),
            ]
        )

        if var.should_reconstruct_all:
            # clear will appear before "update" as the suffixes are applied in
            # reverse order.
            ctx.suffixes.append(
                [
                    *create_call_method(0),  # clear
                    create_instruction("POP_TOP"),
                ]
            )
        ctx.log(var)


@register_side_effect_replay_handler(
    name="torch_function_mode_stack_mutation",
    matcher=lambda ctx: isinstance(
        ctx.var, variables.torch_function.TorchFunctionModeStackVariable
    ),
    priority=60,
)
def _codegen_torch_function_mode_stack_mutation(
    ctx: SideEffectReplayContext,
) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.torch_function.TorchFunctionModeStackVariable):
        raise AssertionError(type(var))
    cg.add_push_null(
        lambda: cg.load_import_from(utils.__name__, "set_torch_function_mode_stack")
    )

    cg.foreach(var.symbolic_stack)
    cg.append_output(create_instruction("BUILD_LIST", arg=len(var.symbolic_stack)))
    cg.call_function(1, False)
    cg.append_output(create_instruction("POP_TOP"))
    ctx.log(var)


@register_side_effect_replay_handler(
    name="cell_mutation",
    matcher=lambda ctx: isinstance(ctx.var, variables.CellVariable)
    and ctx.var.local_name is not None,
    priority=50,
)
def _codegen_cell_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.CellVariable):
        raise AssertionError(type(var))
    if var.local_name is None:
        raise AssertionError("cell mutation local name must be set")
    # Emit more readable and performant bytecode.
    # TODO generalize this for cells created during inlining.
    if var in ctx.side_effects.store_attr_mutations:
        contents_var = ctx.side_effects.load_cell(var)
        cg(contents_var)
        ctx.suffixes.append([cg.create_store_deref(var.local_name)])
        ctx.log(var)


def _codegen_user_defined_dict_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.UserDefinedDictVariable):
        raise AssertionError(type(var))
    # Do dict related update manually here. The store_attr mutations will be
    # applied later.
    varname_map = {}
    for name in _manual_dict_setitem.__code__.co_varnames:
        varname_map[name] = cg.tx.output.new_var()

    try:
        mro_index = type(var.value).__mro__.index(collections.OrderedDict)
    except ValueError:
        mro_index = type(var.value).__mro__.index(dict)

    cg.extend_output(
        [
            create_instruction("LOAD_CONST", argval=mro_index),
            create_instruction("STORE_FAST", argval=varname_map["mro_index"]),
        ]
    )

    cg(var.source)  # type: ignore[attr-defined]
    cg.extend_output(
        [
            create_instruction("STORE_FAST", argval=varname_map["dict_to"]),
        ]
    )

    # Reconstruct all items - _manual_dict_setitem clears dict_to first, so we
    # need every key/value, not just the ones that differ from original_items.
    var._base_vt.should_reconstruct_all = True  # type: ignore[union-attr]
    cg(var._base_vt, allow_cache=False)  # Don't codegen via source
    cg.extend_output(
        [
            create_instruction("STORE_FAST", argval=varname_map["dict_from"]),
        ]
    )

    dict_update_insts = bytecode_from_template(
        _manual_dict_setitem, varname_map=varname_map
    )

    ctx.suffixes.append(
        [
            *dict_update_insts,
            create_instruction("POP_TOP"),
        ]
    )
    ctx.log(
        var._base_vt  # pyrefly: ignore[bad-argument-type]
    )


def _codegen_user_defined_list_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.UserDefinedListVariable):
        raise AssertionError(type(var))
    # Update the list to the updated items. Be careful in calling the list
    # methods and not the overridden methods.
    varname_map = {}
    for name in _manual_list_update.__code__.co_varnames:
        varname_map[name] = cg.tx.output.new_var()

    cg(var.source)  # type: ignore[attr-defined]
    cg.extend_output(
        [
            create_instruction("STORE_FAST", argval=varname_map["list_to"]),
        ]
    )

    cg(var._base_vt, allow_cache=False)  # Don't codegen via source
    cg.extend_output(
        [
            create_instruction("STORE_FAST", argval=varname_map["list_from"]),
        ]
    )

    list_update_insts = bytecode_from_template(
        _manual_list_update, varname_map=varname_map
    )

    ctx.suffixes.append(
        [
            *list_update_insts,
            create_instruction("POP_TOP"),
        ]
    )
    ctx.log(
        var._base_vt  # pyrefly: ignore[bad-argument-type]
    )


def _codegen_user_defined_deque_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.UserDefinedDequeVariable):
        raise AssertionError(type(var))
    # Update the deque to the updated items. Be careful in calling the deque
    # methods and not the overridden methods.
    varname_map = {}
    for name in _manual_deque_update.__code__.co_varnames:
        varname_map[name] = cg.tx.output.new_var()

    cg(var.source)  # type: ignore[attr-defined]
    cg.extend_output(
        [
            create_instruction("STORE_FAST", argval=varname_map["deque_to"]),
        ]
    )

    cg(var._base_vt, allow_cache=False)  # Don't codegen via source
    cg.extend_output(
        [
            create_instruction("STORE_FAST", argval=varname_map["deque_from"]),
        ]
    )

    deque_update_insts = bytecode_from_template(
        _manual_deque_update, varname_map=varname_map
    )

    ctx.suffixes.append(
        [
            *deque_update_insts,
            create_instruction("POP_TOP"),
        ]
    )
    ctx.log(
        var._base_vt  # pyrefly: ignore[bad-argument-type]
    )


def _skip_attribute_mutation_replay(var: VariableTracker) -> bool:
    # FrozenDataClassVariable attributes were emitted in codegen_save_tempvars
    # right after __new__. Skip here to avoid double-emitting.
    if isinstance(var.mutation_type, AttributeMutationNew) and isinstance(
        var, variables.FrozenDataClassVariable
    ):
        return True

    # Sourceless enum members: mutations (like _inverted_ caching) already
    # happened on the real object during tracing.
    return (
        var.is_python_constant()
        and isinstance(var.mutation_type, AttributeMutationNew)
        and isinstance(var, variables.UserDefinedObjectVariable)
        and (
            isinstance(
                var.value,
                (
                    enum.Enum,
                    torch.DispatchKey,
                    torch._C._functorch.TransformType,
                ),
            )
            or is_pybind11_enum_member(var.value)
        )
    )


@register_side_effect_replay_handler(
    name="attribute_mutation",
    matcher=lambda ctx: ctx.side_effects.is_attribute_mutation(ctx.var),
    priority=40,
)
def _codegen_attribute_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    side_effects = ctx.side_effects
    if _skip_attribute_mutation_replay(var):
        return

    if (
        isinstance(var, variables.UserDefinedDictVariable)
        and side_effects.is_modified(
            var._base_vt  # pyrefly: ignore[bad-argument-type]
        )
        and var._base_vt.has_new_items()  # type: ignore[union-attr]
    ):
        _codegen_user_defined_dict_mutation(ctx)
    elif isinstance(
        var, variables.UserDefinedListVariable
    ) and side_effects.is_modified(
        var._base_vt  # pyrefly: ignore[bad-argument-type]
    ):
        _codegen_user_defined_list_mutation(ctx)
    elif isinstance(
        var, variables.UserDefinedDequeVariable
    ) and side_effects.is_modified(
        var._base_vt  # pyrefly: ignore[bad-argument-type]
    ):
        _codegen_user_defined_deque_mutation(ctx)

    # Applying mutations involves two steps: 1) Push all reconstructed objects
    # onto the stack. 2) Call STORE_ATTR to apply the mutations.
    #
    # Dynamo must ensure that mutations are applied in the same order as in the
    # original program. Therefore, two reverse operations occur below.
    #
    # The first reverse operation concerns `suffixes`. We apply suffixes in
    # reverse order due to the way Python handles the stack. In Step 1, we push
    # all reconstructed objects onto the stack, but the item at the top of the
    # stack refers to the last attribute in the mutation order. If not fixed,
    # this will apply the mutations of attributes in the reverse order. To
    # account for this reversal, we iterate through the mutable attributes in
    # reverse order.
    side_effect_occurred = False
    for name, value in reversed(side_effects.store_attr_mutations.get(var, {}).items()):
        mutation_kind = side_effects.get_attr_mutation_kind(var, name)
        if isinstance(var, variables.NewGlobalVariable):
            cg.tx.output.update_co_names(name)
            cg(value)
            if not isinstance(var.source, GlobalSource):  # type: ignore[attr-defined]
                raise AssertionError(
                    f"Expected GlobalSource for NewGlobalVariable, "
                    f"got {type(var.source)}"  # type: ignore[attr-defined]
                )
            ctx.suffixes.append([create_instruction("STORE_GLOBAL", argval=name)])
            side_effect_occurred = True
        elif isinstance(value, variables.DeletedVariable):
            if (
                isinstance(var, variables.UserDefinedObjectVariable)
                and mutation_kind is AttrMutationKind.INSTANCE_DICT
            ):
                original_dict = getattr(getattr(var, "value", None), "__dict__", {})
                # If the key only existed in the traced instance dict, the
                # add/delete sequence is a replay no-op.
                if name in original_dict:
                    cg.add_push_null(
                        lambda: cg.load_import_from(
                            utils.__name__,
                            "object_delattr_ignore_descriptor",
                        )
                    )
                    cg(var.source)  # type: ignore[attr-defined]
                    cg(variables.ConstantVariable(name))
                    ctx.suffixes.append(
                        [
                            *create_call_function(2, False),
                            create_instruction("POP_TOP"),
                        ]
                    )
                    side_effect_occurred = True
            # GENERIC_SETATTR deletions on UDOV fall through to the normal
            # DELETE_ATTR path below so descriptor semantics are preserved
            # during replay.
            elif isinstance(var.mutation_type, AttributeMutationExisting) and hasattr(
                getattr(var, "value", None), name
            ):
                cg.tx.output.update_co_names(name)
                cg(var.source)
                ctx.suffixes.append([create_instruction("DELETE_ATTR", argval=name)])
                side_effect_occurred = True
        elif (
            isinstance(var, variables.UserDefinedObjectVariable)
            and mutation_kind is AttrMutationKind.INSTANCE_DICT
        ):
            cg.add_push_null(
                lambda: cg.load_import_from(
                    utils.__name__, "object_setattr_ignore_descriptor"
                )
            )
            cg(var.source)  # type: ignore[attr-defined]
            cg(variables.ConstantVariable(name))
            cg(value)
            ctx.suffixes.append(
                [
                    *create_call_function(3, False),
                    create_instruction("POP_TOP"),
                ]
            )
            side_effect_occurred = True
        elif (
            isinstance(var, variables.UserDefinedObjectVariable)
            and var.needs_slow_setattr()
        ):
            # __setattr__ is defined on this object, so call object.__setattr__
            # directly.
            cg.load_import_from("builtins", "object")
            cg.load_method("__setattr__")
            cg(var.source)  # type: ignore[attr-defined]
            cg(variables.ConstantVariable(name))
            cg(value)
            ctx.suffixes.append([*create_call_method(3), create_instruction("POP_TOP")])
            side_effect_occurred = True
        else:
            cg.tx.output.update_co_names(name)
            cg(value)
            cg(var)
            ctx.suffixes.append([create_instruction("STORE_ATTR", argval=name)])
            side_effect_occurred = True

    if side_effect_occurred:
        ctx.log(var)


@register_side_effect_replay_handler(
    name="list_iterator_mutation",
    matcher=lambda ctx: isinstance(ctx.var, variables.ListIteratorVariable),
    priority=30,
)
def _codegen_list_iterator_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.ListIteratorVariable):
        raise AssertionError(type(var))
    for _ in range(var.index):
        cg.add_push_null(lambda: cg.load_import_from(utils.__name__, "iter_next"))
        cg(var.source)  # type: ignore[attr-defined]
        cg.call_function(1, False)
        cg.pop_top()
    ctx.log(var)


@register_side_effect_replay_handler(
    name="count_iterator_mutation",
    matcher=lambda ctx: isinstance(ctx.var, variables.CountIteratorVariable),
    priority=20,
)
def _codegen_count_iterator_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.CountIteratorVariable):
        raise AssertionError(type(var))
    for _ in range(var.advance_count):
        cg.add_push_null(lambda: cg.load_import_from(utils.__name__, "iter_next"))
        cg(var.source)  # type: ignore[attr-defined]
        cg.call_function(1, False)
        cg.pop_top()
    ctx.log(var)


@register_side_effect_replay_handler(
    name="random_mutation",
    matcher=lambda ctx: isinstance(ctx.var, variables.RandomVariable),
    priority=10,
)
def _codegen_random_mutation(ctx: SideEffectReplayContext) -> None:
    cg = ctx.codegen
    var = ctx.var
    if not isinstance(var, variables.RandomVariable):
        raise AssertionError(type(var))

    def gen_fn() -> None:
        cg(var.source)  # type: ignore[attr-defined]
        cg.load_attr("setstate")

    cg.add_push_null(gen_fn)
    cg(var.wrap_state(var.random.getstate()))

    ctx.suffixes.append(
        [
            *create_call_function(1, False),  # setstate
            create_instruction("POP_TOP"),
        ]
    )
    ctx.log(var)


@contextlib.contextmanager
def allow_side_effects_in_hop(
    tx: "InstructionTranslatorBase",
) -> Generator[None, None, None]:
    """Context manager to temporarily allow side effects with extra outputs.

    This is used for special cases (like FSDP functions) that need to perform
    side effects even when the general policy is to disallow them.
    """
    orig_val = tx.output.current_tracer.allow_side_effects_in_hop
    try:
        tx.output.current_tracer.allow_side_effects_in_hop = True
        yield
    finally:
        tx.output.current_tracer.allow_side_effects_in_hop = orig_val


@contextlib.contextmanager
def allow_externally_visible_side_effects_in_subtracer(
    tx: "InstructionTranslatorBase",
) -> Generator[None, None, None]:
    orig_val = tx.output.current_tracer.unsafe_allow_externally_visible_side_effects
    try:
        tx.output.current_tracer.unsafe_allow_externally_visible_side_effects = True
        tx.output.current_tracer.traced_with_externally_visible_side_effects = True
        yield
    finally:
        tx.output.current_tracer.unsafe_allow_externally_visible_side_effects = orig_val


@contextlib.contextmanager
def disallow_side_effects_in_generator(
    tx: "InstructionTranslatorBase",
) -> Generator[None, None, None]:
    orig_val = tx.output.current_tracer.is_reconstructing_generator
    try:
        tx.output.current_tracer.is_reconstructing_generator = True
        yield
    finally:
        tx.output.current_tracer.is_reconstructing_generator = orig_val
<|endoftext|>"""
Configuration module for TorchDynamo compiler and optimization settings.

This module contains various configuration flags and settings that control TorchDynamo's
behavior, including:

- Runtime behavior flags (e.g., guard settings, specialization options)
- Debugging and development options
- Performance tuning parameters
- Feature toggles for experimental features
"""

import getpass
import os
import sys
import sysconfig
import tempfile
from collections.abc import Callable
from os.path import abspath, dirname
from typing import Any, Literal, TYPE_CHECKING

from torch._environment import is_fbcode
from torch.utils._config_module import Config, get_tristate_env, install_config_module


# to configure logging for dynamo, aot, and inductor
# use the following API in the torch._logging module
# torch._logging.set_logs(dynamo=<level>, aot=<level>, inductor<level>)
# or use the environment variable TORCH_LOGS="dynamo,aot,inductor" (use a prefix + to indicate higher verbosity)
# see this design doc for more detailed info
# Design doc: https://docs.google.com/document/d/1ZRfTWKa8eaPq1AxaiHrq4ASTPouzzlPiuquSBEJYwS8/edit#
# the name of a file to write the logs to
# [@compile_ignored: debug]
log_file_name: str | None = None

# [@compile_ignored: debug] Verbose will print full stack traces on warnings and errors
verbose = os.environ.get("TORCHDYNAMO_VERBOSE", "0") == "1"

# [@compile_ignored: runtime_behaviour] verify the correctness of optimized backend
verify_correctness = False

# Override backend for specific graphs (for debugging/bisecting).
# Format: "filter1:backend1;filter2:backend2;..." where filter can be:
#   - Individual IDs: "0,5,10"
#   - Ranges: "10-20" (inclusive)
#   - Comparisons: ">10", ">=10", "<5", "<=5"
# Backends can be: "eager", "aot_eager", "inductor", etc.
# Examples:
#   ">10:eager"                    - Run graphs with frame_id > 10 in dynamo eager backend
#   "<=5:aot_eager;>5:inductor"    - First 6 graphs use aot_eager, rest use inductor
# [@compile_ignored: debug]
debug_backend_override: str = os.environ.get("TORCH_COMPILE_OVERRIDE_BACKENDS", "")

# Override inductor config for specific graphs (for debugging/bisecting).
# Format: "filter1:config1;filter2:config2;..." where filter uses same syntax as
# debug_backend_override, and config is "key=value" or "key=value,key2=value2".
# Examples:
#   "0-5:triton.cudagraph_skip_dynamic_graphs=False"  - Disable skip for graphs 0-5
#   ">10:triton.cudagraphs=False"                     - Disable cudagraphs for graphs > 10
# [@compile_ignored: debug]
debug_inductor_config_override: str = os.environ.get(
    "TORCH_COMPILE_OVERRIDE_INDUCTOR_CONFIGS", ""
)

# Override dynamo config for specific graphs (for debugging/bisecting).
# Format: "filter1:config1;filter2:config2;..." where filter uses same syntax as
# debug_backend_override, and config is "key=value" or "key=value,key2=value2".
# Examples:
#   "0-5:specialize_float=True"  - Specialize floats for graphs 0-5
#   ">10:automatic_dynamic_shapes=False"  - Disable dynamic shapes for graphs > 10
# [@compile_ignored: debug]
debug_dynamo_config_override: str = os.environ.get(
    "TORCH_COMPILE_OVERRIDE_DYNAMO_CONFIGS", ""
)

# Validate that fake_fn and real_fn in @leaf_function decorators produce outputs
# with matching shapes and dtypes in eager mode. Helps catch mismatches early.
# Disabled by default to avoid runtime overhead.
# [@compile_ignored: debug]
leaf_function_validate_outputs = False

# Check for escaped gradients in @leaf_function. When a leaf_function closes over
# a tensor with requires_grad=True, gradients won't flow back to it. This check
# walks the autograd graph to detect such cases and raises an error.
# Disabled by default to avoid runtime overhead. Enable for debugging.
# [@compile_ignored: debug]
leaf_function_check_escaped_gradients = False

# need this many ops to create an FX graph (deprecated: not used)
minimum_call_count = 1

# turn on/off DCE pass (deprecated: always true)
dead_code_elimination = None

# Enable or disable side effect replay after graph execution.
# When False, mutations to Python objects (lists, dicts, attributes) won't be
# replayed after the compiled graph runs. This can cause correctness issues
# if your code depends on these mutations being visible. This should probably
# never be False by default. At the moment, only export will need it.
replay_side_effects = True

# Generate Python code strings for compiled graph calls.
# When enabled, the generated Python code will be stored in output_pycode.
# [@compile_ignored: debug]
generate_pycode = False

# Configure side effect warning level
# If `info` (default): allow side effects and log to TORCH_LOGS="side_effects" and tlparse
# If `silent`, we allow side effects, no logs are made.
# If `warn`, we allow side effects but issue warnings
# If `error`, we error on side effects
# NOTE: it is NOT safe to change this config during compilation!
side_effect_replay_policy = "info"

# disable (for a function) when cache reaches this size

# controls the maximum number of cache entries with a guard on same ID_MATCH'd
# object. It also controls the maximum size of cache entries if they don't have
# any ID_MATCH'd guards.
# [@compile_ignored: runtime_behaviour]
recompile_limit = 8

# [@compile_ignored: runtime_behaviour] safeguarding to prevent horrible recomps
accumulated_recompile_limit = 256

skip_code_recursive_on_recompile_limit_hit: bool = Config(
    default=True, deprecated=True, deprecation_message="does not do anything"
)

# raise a hard error if cache limit is hit.  If you are on a model where you
# know you've sized the cache correctly, this can help detect problems when
# you regress guards/specialization.  This works best when recompile_limit = 1.
# This flag is incompatible with: suppress_errors.
# [@compile_ignored: runtime_behaviour]
fail_on_recompile_limit_hit = False

cache_size_limit: int = Config(alias="torch._dynamo.config.recompile_limit")
accumulated_cache_size_limit: int = Config(
    alias="torch._dynamo.config.accumulated_recompile_limit"
)

skip_code_recursive_on_cache_limit_hit: bool = Config(
    alias="torch._dynamo.config.skip_code_recursive_on_recompile_limit_hit",
    deprecated=True,
    deprecation_message="does not do anything",
)

fail_on_cache_limit_hit: bool = Config(
    alias="torch._dynamo.config.fail_on_recompile_limit_hit"
)

# whether or not to specialize on int inputs.  This only has an effect with
# dynamic_shapes; when dynamic_shapes is False, we ALWAYS specialize on int
# inputs.  Note that assume_static_by_default will also cause ints to get
# specialized, so this is mostly useful for export, where we want inputs
# to be dynamic, but accesses to ints should NOT get promoted into inputs.
specialize_int = False

# Whether or not to specialize on float inputs.  Dynamo will always promote
# float inputs into Tensor inputs, but at the moment, backends inconsistently
# support codegen on float (this is to be fixed).
specialize_float = False

# Max ops per ComputedLazyConstantVariable chain before realizing under guards; 0 disables
computed_lazy_constant_max_nodes = 64

# legacy config, does nothing now!
dynamic_shapes = True

use_lazy_graph_module = (
    os.environ.get("TORCH_COMPILE_USE_LAZY_GRAPH_MODULE", "1") == "1"
)

# This is a temporarily flag, which changes the behavior of dynamic_shapes=True.
# When assume_static_by_default is True, we only allocate symbols for shapes marked dynamic via mark_dynamic.
# NOTE - this flag can be removed once we can run dynamic_shapes=False w/ the mark_dynamic API
# see [Note - on the state of mark_dynamic]
assume_static_by_default = True

# Internal: Shape specification patched during tracing by enter_exit_hooks.
# Set via torch.compile(dynamic_shapes=...), not directly by users.
_dynamic_shapes_spec = None

# This flag changes how dynamic_shapes=True works, and is meant to be used in conjunction
# with assume_static_by_default=True.
# With this flag enabled, we always compile a frame as fully static for the first time, and, if we fail
# any guards due to wobbles in shape, we recompile with *all* the wobbled shapes as being marked dynamic.
automatic_dynamic_shapes = (
    os.environ.get("TORCH_DYNAMO_AUTOMATIC_DYNAMIC_SHAPES", "1") == "1"
)

# Valid options: "dynamic", "unbacked"
automatic_dynamic_shapes_mark_as: Literal["dynamic", "unbacked"] = "dynamic"

# When True, adds exclusion guards for tensor dims and scalars that transition
# from static to dynamic via automatic_dynamic_shapes.
#
# Invariant: when enabled, automatic_dynamic recompilation preserves graph
# selection — inputs that matched a previous static cache entry will continue
# to use that entry, not be intercepted by a newer dynamic entry. This holds
# as long as recompilations are caused solely by the same variable being
# observed with different static values (progressive dynamism). A recompilation
# triggered by a different reason (e.g., a guard failure unrelated to shape
# transitions) will clear the exclusion state for that entry.
#
# Mechanism: the exclusion guard rejects inputs matching the prior static
# graph's sizes, so those inputs fall through to the more specialized static
# graph instead of being captured by the newer dynamic graph.
#
# Scope: applies only to graph-input-level dimension and scalar transitions.
# Does NOT handle data-dependent branching (if x.size(0) > k), graph breaks,
# or other recompilation triggers where no dimension actually transitions.
automatic_dynamic_exclusion_guard = False

# log graph in/out metadata
# This is only turned on for export today since we
# know we are tracing a flat callable. later, this
# can extended to other use cases as well.
log_graph_in_out_metadata = False

# This flag changes how the shapes of parameters are treated.
# If this flag is set to True, then the shapes of torch.nn.Parameter as well as of torch.Tensor are attempted to be dynamic
# If this flag is set to False, then the shapes of torch.nn.Parameter are assumed to be static,
# while the shapes of torch.Tensor are assumed to be dynamic.
force_parameter_static_shapes = True

# This flag ensures that the shapes of a nn module are always assumed to be static
# If the flag is set to True, then the shapes of a nn.module are assumed to be static
# If the flag is set to False, then the shapes of a nn.module can be dynamic
force_nn_module_property_static_shapes = True

# Typically, if you mark_dynamic a dimension, we will error if the dimension
# actually ended up getting specialized.  This knob changes the behavior so
# that we don't error at all.  This is helpful for our CI where I'm using a
# heuristic to mark batch dimensions as dynamic and the heuristic may get it
# wrong.
allow_ignore_mark_dynamic = False

# Set this to False to assume nn.Modules() contents are immutable (similar assumption as freezing)
guard_nn_modules = True

# Uses CPython internal dictionary tags to detect mutation. There is some
# overlap between guard_nn_modules_using_dict_tags and guard_nn_modules flag.
# guard_nn_modules unspecializes the nn module instance and adds guard for each
# relevant member of the nn modules. On the other hand,
# guard_nn_modules_using_dict_tags specializes on each nn module instance but
# uses low overhead dict version matching to detect mutations, obviating the
# need to guard on members of the nn modules. With
# guard_nn_modules_using_dict_tags, the guard_nn_modules is not really required
# but kept around for debugging and discussing unspecializing nn module
# variables.
# TODO(janimesh, voz): Remove both of these flags (or at least guard_nn_modules)
# once we have reached stability for the guard_nn_modules_using_dict_tags.
guard_nn_modules_using_dict_tags = True

# Flag to enable preparation for graph freezing, so that the named parameters and
# buffers are passed as params_flat in tracing context by AOT autograd.
# Non-Inductor backends can use this list for graph freezing.
prepare_freezing = os.environ.get("TORCHDYNAMO_PREPARE_FREEZING", "0") == "1"

# NOTE this has been deprecated, it does nothing now.
traceable_tensor_subclasses: set[type[Any]] = set()

# If a tensor subclass is put into this set, Dynamo will model its instances in
# a very conservative and limited way (most likely causing lots of graph breaks
# if one apply tensor ops on these instances). This is useful if you encounter
# internal compiler errors from Dynamo which are caused by tensor subclasses,
# and you are willing to tolerate potential graph breaks rather than hard error.
nontraceable_tensor_subclasses: set[type[Any]] = set()

# Suppress errors in torch._dynamo.optimize, instead forcing a fallback to eager.
# This is a good way to get your model to work one way or another, but you may
# lose optimization opportunities this way.  Devs, if your benchmark model is failing
# this way, you should figure out why instead of suppressing it.
# This flag is incompatible with: fail_on_recompile_limit_hit.
suppress_errors = bool(os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", False))

# Record and write an execution record of the current frame to a file
# if an exception is encountered
# @compile_ignored[debug]
replay_record_enabled = os.environ.get("TORCH_COMPILE_REPLAY_RECORD", "0") == "1"

# Rewrite assert statement in python with torch._assert
rewrite_assert_with_torch_assert = True

# Disable dynamo
disable = os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"

# [@compile_ignored: runtime_behaviour] Get a cprofile trace of Dynamo
cprofile = os.environ.get("TORCH_COMPILE_CPROFILE", False)

# Enable Dynamo profiler. When enabled, prints pstats output showing
# time spent tracing each user function. Set to True to enable, or set to a
# file path to save the .prof file for snakeviz.
# [@compile_ignored: runtime_behaviour]
dynamo_profiler: bool | str = os.environ.get("TORCH_COMPILE_DYNAMO_PROFILER", False)

# Legacy config, does nothing now!
skipfiles_inline_module_allowlist: dict[Any, Any] = {}
"""Allowlist of inline modules to skip during compilation.

Legacy configuration that previously controlled which modules could be
inlined during tracing. This configuration is deprecated and no longer used.

:type: dict[Any, Any]
:default: {}

.. deprecated::
   This configuration is deprecated and does nothing now.

.. note::
   DEPRECATED: This setting has no effect on current behavior.
"""

# If a string representing a PyTorch module is in this ignorelist,
# the `allowed_functions.is_allowed` function will not consider it
# when creating a list of PyTorch functions that will appear in
# FX IR.
allowed_functions_module_string_ignorelist = {
    "torch.distributions",
    "torch.testing",
    "torch._refs",
    "torch._prims",
    "torch._decomp",
}

# Debug Flag to try minifier at different stages. Possible values are {None, "aot", "dynamo"}
# None - Minifier is switched off
# dynamo - Runs minifier on the TorchDynamo produced graphs, if compilation fails
# aot - Runs minifier on the Aot Autograd produced graphs, if compilation fails
# [@compile_ignored: debug]
repro_after = os.environ.get("TORCHDYNAMO_REPRO_AFTER", None)

# Compiler compilation debug info
# 1: Dumps the original graph out to repro.py if compilation fails
# 2: Dumps a minifier_launcher.py if compilation fails.
# 3: Always dumps a minifier_launcher.py. Good for segfaults.
# 4: Dumps a minifier_launcher.py if the accuracy fails.
# [@compile_ignored: debug]
repro_level = int(os.environ.get("TORCHDYNAMO_REPRO_LEVEL", 2))

# By default, we try to detect accuracy failure by running both forward
# and backward of a torchdynamo produced graph (if you are using repro_after
# 'dynamo').  This setting forces us to only test the forward graph and
# not the backward graph.  This can be helpful if you're trying to debug
# an inference only problem, but the minifier seems to be choking on the
# backwards step
# TODO: Detect this situation automatically so the user doesn't need
# to manually configure this
# [@compile_ignored: debug]
repro_forward_only = os.environ.get("TORCHDYNAMO_REPRO_FORWARD_ONLY") == "1"

# The tolerance we should use when testing if a compiled graph
# has diverged so that we should treat it as an accuracy failure
# [@compile_ignored: debug]
repro_tolerance = 1e-3


# Whether to ignore non-floating point values when checking accuracy.
# Checking accuracy of non-floating point values such as boolean tensors
# can lead to false positives.
# [@compile_ignored: debug]
repro_ignore_non_fp = os.environ.get("TORCHDYNAMO_REPRO_IGNORE_NON_FP") == "1"

# If True, when testing if two models are the same, we will test them against
# a third fp64 reference and only report a problem if the RMSE relative to the
# fp64 is greater.  However, this will use more memory; you may disable this
# if memory usage is too high.
# [@compile_ignored: runtime_behaviour]
same_two_models_use_fp64 = True

# Not all backends support scalars. Some calls on torch.Tensor (like .item()) return a scalar type.
# When this flag is set to False, we introduce a graph break instead of capturing.
# This requires dynamic_shapes to be True.
capture_scalar_outputs = os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") == "1"

# Not all backends support operators that have dynamic output shape (e.g.,
# nonzero, unique).  When this flag is set to False, we introduce a graph
# break instead of capturing.  This requires dynamic_shapes to be True.
# If you set this to True, you probably also want capture_scalar_outputs
# (these are separated for historical reasons).
capture_dynamic_output_shape_ops = (
    os.environ.get("TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS", "0") == "1"
)

# hybrid backed unbacked symints
prefer_deferred_runtime_asserts_over_guards = False

# By default, dynamo will treat all ints as backed SymInts, which means (1) it
# will wait to see the int change over multiple runs before generalizing and
# (2) it will still always 0/1 specialize an int.  When true, this knob
# forces dynamo to treat _length_per_key and _offset_per_key on
# KeyedJaggedTensor from torchrec as size-like unbacked SymInts, so that
# they (1) generalize immediately and (2) unsoundly never compare equal to
# 0/1.  This is not on by default as AOTAutograd/Inductor cannot currently
# compile this code; however, this can be useful for export.
force_unspec_int_unbacked_size_like_on_torchrec_kjt = False

# Currently, Dynamo will always specialize on int members of NN module.
# However, there could be cases where this is undesirable, e.g., when tracking
# step count leading to constant recompilation and eventually eager fallback.
# Setting this flag to True will allow int members to be potentially unspecialized
# through dynamic shape mechanism.
# Defaults to False for BC.
allow_unspec_int_on_nn_module = False

# Specify how to optimize a compiled DDP module. The flag accepts a boolean
# value or a string. There are 3 modes.
# 1. "ddp_optimizer" (or True): with "ddp_optimizer", Dynamo will automatically
# split model graph into pieces to match DDP bucket sizes to allow DDP
# comm/compute overlap.
# 2. "python_reducer" (experimental): this optimization requires the usage
# of compiled_autograd. With "python_reducer", DDP will disable the C++ reducer
# and use the Python reducer to allow compiled_autograd to trace the
# communication and allow comm/compute overlap without graph-breaks.
# 3. "no_optimization" (or False): Dynamo won't split the model graph, nor
# will Python reducer be used. With this mode, there will be no graph-breaks
# and the original DDP C++ reducer will be used. There will no comm/compute
# overlap. This mode CANNOT be used with compiled_autograd.
# Note that to avoid breaking the existing usage, mode 1 and mode 4 can be
# specified with a boolean value. True is using ddp_optimizer and False is
# no optimization.
optimize_ddp: (
    bool
    | Literal[
        "ddp_optimizer",
        "python_reducer",
        "python_reducer_without_compiled_forward",
        "no_optimization",
    ]
) = True

# By default, Dynamo emits runtime asserts (e.g. torch._check) in the graph.
# In some cases those asserts could be performance costly
# E.g. torch._check(tensor[0].item() > 2) for tensor on cuda will require cuda sync.
# Setting this to True keeps them hinting to symbolic shapes engine,
# but not be emitted in the graph.
do_not_emit_runtime_asserts: bool = (
    os.environ.get("TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS", "0") == "1"
)

# Skip tracing the torchrec files added to trace_rules.FBCODE_SKIP_DIRS
skip_torchrec = True

# Don't apply most trace_rules.py rules
dont_skip_tracing = False

# No longer used
optimize_ddp_lazy_compile = False

# lambda guarding on object aliasing to improve opportunity for dict tag
# optimization
use_lamba_guard_for_object_aliasing = False

# Whether to skip guarding on FSDP-managed modules
skip_fsdp_guards = True

# Make dynamo skip guarding on hooks on nn modules
# Note: unsafe: if your model actually has hooks and you remove them, or doesn't and  you add them,
# dynamo will not notice and will execute whichever version you first compiled.
skip_nnmodule_hook_guards = True

# Make dynamo skip no tensor aliasing guard on parameters
# Note: unsafe: if you compile a function with different parameters as inputs,
# and then later pass on the same parameter as two inputs, dynamo will not
# notice and lead to incorrect result.
skip_no_tensor_aliasing_guards_on_parameters = True

# Considers a tensor immutable if it is one of the values of a dictionary, and
# the dictionary tag is same across invocation calls.
skip_tensor_guards_with_matching_dict_tags = True

# Skips guards on func.__defaults__ if the element to be guarded is a constant
skip_guards_on_constant_func_defaults = False


# The recursive-dict-tag guard relies on the class/function identity staying
# stable.  We therefore assume that the following function dunder attributes
# are **never rebound** to a different object:
#
#     • __code__        • __closure__
#     • __defaults__    • __kwdefaults__
#     • __annotations__ • __mro__
#
# It is fine to mutate the objects they already point to (e.g. tweak an element
# inside __defaults__), but assignments like
#
#     foo.__defaults__ = (3, 4)          # REBIND  - NOT SUPPORTED
#
# would invalidate the optimization.  This type of rebinding is rare, so we
# assume that the rebinding never happens for guard purposes.  Set the flag
# below to False only in environments where such rebinding is known to occur.
assume_dunder_attributes_remain_unchanged = True

# Speedup guard execution of nested nn modules by recursively checking for dict
# tags to avoid full guard execution.
use_recursive_dict_tags_for_guards = False

# Maximum number of objects for which we check dict pointers tags. This is
# useful for regional compilation.
max_saved_pointers_for_recursive_dict_tags_check = 256

# If True, raises exception if TorchDynamo is called with a context manager
raise_on_ctx_manager_usage = True

# If True, raise when aot autograd is unsafe to use
raise_on_unsafe_aot_autograd = False

# This flag is ignored and maintained for backwards compatibility.
error_on_nested_jit_trace = True

# If true, error with a better message if we symbolically trace over a
# dynamo-optimized function. If false, silently suppress dynamo.
error_on_nested_fx_trace = True

# If true, force dynamo compilation even when inside FX symbolic tracing.
# This allows nested compilation where the outer tracer (e.g., make_fx) can
# trace over dynamo-compiled functions. Use with error_on_nested_fx_trace=False.
force_compile_during_fx_trace = False

# Disables graph breaking on rnn. YMMV with backends.
allow_rnn = False

# If true, enables feature that captures PyTorch sparsity in the
# exported FX graph. This flag should become the default eventually
# and be removed, but currently provides a way to fall back to old
# graph breaking behavior.
capture_sparse_compute = True

# If true, error if we try to compile a function that has
# been seen before.
# [@compile_ignored: runtime_behaviour]
error_on_recompile = False

# [@compile_ignored: debug] Whether to report any guard failures (deprecated: does not do anything)
report_guard_failures = True

# [@compile_ignored: debug] root folder of the project
base_dir = dirname(dirname(dirname(abspath(__file__))))

# Trace through NumPy or graphbreak
trace_numpy = True

# Trace through torch.autograd.grad or graphbreak
trace_autograd_ops = False

# Default NumPy dtypes when tracing with torch.compile
# We default to 64bits. For efficiency, one may want to change these to float32
numpy_default_float = "float64"
numpy_default_complex = "complex128"
numpy_default_int = "int64"

# use numpy's PRNG if True, pytorch otherwise
use_numpy_random_stream = False

# Use C++ guard manager (deprecated: always true)
enable_cpp_guard_manager = True

# Use C++ guard manager for symbolic shapes
enable_cpp_symbolic_shape_guards = False

# Enable tracing through contextlib.contextmanager
enable_trace_contextlib = True

# Enable tracing through unittest
enable_trace_unittest = False

# Enable tracing LOAD_BUILD_CLASS bytecode
enable_trace_load_build_class = False

enable_faithful_generator_behavior = Config(  # type: ignore[var-annotated]
    default=True,
    deprecated=True,
    deprecation_message="does not do anything, generators are always traced lazily",
)

# Inline inbuilt nn modules
inline_inbuilt_nn_modules = Config(  # type: ignore[var-annotated]
    default=True,
    justknob="pytorch/compiler:inline_inbuilt_nn_modules",
    deprecated=True,
    deprecation_message="does not do anything, inline_inbuilt_nn_modules is always True",
)

# Resume tracing in nested frames if a nested graph break occurs
# Old behavior is to bubble up the graph break to the top level frame.
nested_graph_breaks: bool = False

# If True, error if Dynamo attempts to trace more code while running compiled code in fullgraph=True.
# If Dynamo determines that it should skip tracing the code (either at the C/C++ or Python level),
# no error will be raised.
# Set to false if force falling back to eager is desired.
error_on_dynamo_callback_in_fullgraph_compiled_code = False

# Install "free" tensor variables (globals, non-locals, nn module attributes)
# as graph attributes.  This is useful for export, as it
# produces a consistent number of inputs to the graph.
install_free_tensors = False

# Temporary flag to control the turning of install_free_tensors to True for
# export. We will remove this flag in a few weeks when stable.
install_free_tensors_for_export = True

# Use C++ FrameLocalsMapping (raw array view of Python frame fastlocals) (deprecated: always True)
enable_cpp_framelocals_guard_eval = True

# Whether to automatically find and replace identical graph
# regions with a call to invoke_subgraph
use_graph_deduplication = False

# Whether to track nodes for deduplication (testing only)
# This flag is ignored if use_graph_deduplication is True
track_nodes_for_deduplication = False

# Whether to lint the graph after each region is replaced
# (Debug)
graph_deduplication_lint = False

# Issues a warning in Python 3.13.0 for possibly slower guard evaluation and
# instructs user to attempt using 3.13.1+, where the CPython bug is fixed.
# Should be disabled in dynamo-wrapped tests since some tests check that no warnings are issued.
issue_3_13_0_warning = True

# If False, skip frame (and future calls to the same code object) if we determine that the
# traced FX graph is empty when RETURN_* is traced.
allow_empty_graphs = False

# Used for testing - forces all top-level functions to be nested when traced with Dynamo.
# There are slight differences between this config and wrap_top_frame.
debug_force_nested_calls = False

# Used for testing - forces a graph break when a function
# that doesn't make any Dynamo-inlined calls returns
debug_force_graph_break_on_leaf_return = False

# Used for testing - causes CompileCounter.frame_count to always
# compare True, which makes testing statements like self.assertEqual(CompileCounter.frame_count, n)
# always pass.
debug_disable_compile_counter = False

# When set, total compile time instruction count is recorded using
# torch._dynamo.utilsCompileTimeInstructionCounter.
record_compile_time_instruction_count = False


def default_debug_dir_root() -> str:
    # [@compile_ignored: debug]
    DEBUG_DIR_VAR_NAME = "TORCH_COMPILE_DEBUG_DIR"
    if DEBUG_DIR_VAR_NAME in os.environ:
        return os.path.join(os.environ[DEBUG_DIR_VAR_NAME], "torch_compile_debug")
    elif is_fbcode():
        return os.path.join(
            tempfile.gettempdir(), getpass.getuser(), "torch_compile_debug"
        )
    else:
        return os.path.join(os.getcwd(), "torch_compile_debug")


# [@compile_ignored: debug]
debug_dir_root = default_debug_dir_root()

# [@compile_ignored: debug]
_save_config_ignore = {
    "repro_after",
    "repro_level",
    # workaround: "cannot pickle PyCapsule"
    "constant_functions",
    # workaround: "cannot pickle module"
    "skipfiles_inline_module_allowlist",
}

# for backend="cudagraphs", mutations on input be sent to the cudagraph backend
# or replayed in aot_autograd epilogue. default is False because mutation on inputs
# can prevent cudagraphing.
cudagraph_backend_keep_input_mutation = False

# enable cudagraph support for mutated inputs from prior cudagraph pool
cudagraph_backend_support_input_mutation = False

# When True, only ops that have the torch.Tag.pt2_compliant tag
# will be allowed into the graph; all other ops will be disallowed
# and will fall back to eager-mode PyTorch. Useful to ensure
# correctness of custom ops.
only_allow_pt2_compliant_ops = False

# This flag is ignored and maintained for backwards compatibility.
capture_autograd_function = True

# This flag is ignored and maintained for backwards compatibility.
capture_func_transforms = True

# Enable capturing torch.profiler.record_function ops in the graph
# When True, profiler ops are emitted to the graph and preserved through
# compilation (make_fx, functionalization). When False, profiler ops
# are treated as nullcontext.
capture_profiler_record_function: bool = False

# If to log Dynamo compilation metrics into log files (for OSS) and Scuba tables (for fbcode).
log_compilation_metrics = True

# A set of logging functions which will be reordered to the end of graph breaks,
# allowing dynamo to construct large graph. Note that there are some
# limitations to this, such as how it does not correctly print objects that were
# mutated after the print statement.
reorderable_logging_functions: set[Callable[[Any], None]] = set()

# A set of functions that will be ignored during Dynamo tracing.
# These functions will NOT run, will NOT be reordered, and will NOT
# cause graph breaks. They act as full no-ops.
# Ignored functions can take any arguments, but MUST return None.
# Functions should either be module-level functions,
# `logging.Logger.<method>` (ignores all method for all logging.Logger instances)
# or `logger_obj.<method>` (ignores method only for logger_obj logging.Logger instance).
# Other functions may or may not be ignored due to implementation details. If you want to ignore a function
# that `ignore_logging_functions` is failing to ignore, please submit an issue.
ignore_logging_functions: set[Callable[..., Any]] = set()

# Backwards compat: `ignore_logger_methods` now aliases `ignore_logging_functions`.
# Existing code that used `ignore_logger_methods` will continue to work.
ignore_logger_methods: set[Callable[..., Any]] = Config(
    alias="torch._dynamo.config.ignore_logging_functions"
)

# simulates what would happen if we didn't have support for BUILD_SET opcode,
# used for testing
inject_BUILD_SET_unimplemented_TESTING_ONLY = False

_autograd_backward_strict_mode_banned_ops = [
    "layout",
    "is_neg",
    "is_conj",
    "is_pinned",
]

_autograd_backward_strict_mode_conditional_banned_ops = [
    "stride",
    "storage_offset",
    "is_contiguous",
]

# Enables caching of dispatches to fake tensors.
fake_tensor_cache_enabled = (
    os.environ.get("TORCH_FAKE_TENSOR_DISPATCH_CACHE", "1") == "1"
)

# Enables cross checking between the fake tensor cache and dispatch.
fake_tensor_cache_crosscheck_enabled = (
    os.environ.get("TORCH_FAKE_TENSOR_DISPATCH_CACHE_CROSSCHECK", "0") == "1"
)

# Disables inference mode for fake tensor prop during compilation. At runtime,
# the inference_mode is still respected.
fake_tensor_disable_inference_mode = True

# Experimental feature for running automatic caching precompile.
# Enables automatic DynamoCache save/load
caching_precompile = os.environ.get("TORCH_CACHING_PRECOMPILE", "0") == "1"

strict_precompile = os.environ.get("TORCH_STRICT_PRECOMPILE", "0") == "1"

# Enables the Compiled Autograd engine to trace autograd calls made under torch.compile().
# Note: AOTAutograd will still trace and partition an AOT backward graph local to that
# compiled region. But AOTAutograd traces without knowledge of backward hooks which are
# coordinated by the Autograd engine, and under the hood, it uses the torch.autograd.grad
# API, so it cannot capture gradient accumulation operations (AccumulateGrad).
#
# Compiled Autograd will trace all autograd operations as seen by the Autograd engine.
# This flag will also lift certain restrictions during the forward trace such as
# registering backward hooks on tensors contained within the compiled region.
compiled_autograd = False

# We have small decompositions for some optimizer ops such as
# addcmul and foreach_addcmul which avoid item() graph breaks by decomposing
# into their constituent ops. This flag controls whether we use these decompositions
# This can affect numerics for non-inductor backends.
enable_dynamo_decompositions = True


# Checks if we should graph break when seeing nn parameter constructors
# in dynamo; this is so that we clearly fail and ask users to move outside
# the function as opposed to trying to support the ctor with unclear semantics
# See https://github.com/pytorch/pytorch/issues/157452 for more context
graph_break_on_nn_param_ctor = True

# If True, enable calling torch.compile inside __torch_dispatch__ handlers.
# When enabled:
# 1. __torch_dispatch__ methods are automatically wrapped with torch._dynamo.disable
# 2. torch.compile is skipped when active TorchDispatchModes are on the stack
#    (unless they have ignore_compile_internals=True)
# This allows torch.compile to work inside dispatch mode handlers once all
# ambient modes have been "consumed".
# See https://github.com/pytorch/pytorch/issues/155331 for more context.
inline_torch_dispatch_torch_compile = True

# Eager AC/SAC reapplies the mutations (like global dict mutations) in the
# backward during the recomputation of forward. torch.compile has no easy way to
# reapply python mutations in the backward. But many users might be ok to skip
# reapplication of side effects in the backward. They can set this config flag
# to accept this eager and compile divergence.
skip_fwd_side_effects_in_bwd_under_checkpoint = False


# Overrides torch.compile() kwargs for Compiled Autograd:
compiled_autograd_kwargs_override: dict[str, Any] = {}
"""Overrides torch.compile() kwargs for Compiled Autograd.

This dictionary allows overriding specific torch.compile() keyword arguments
when using Compiled Autograd. Only certain overrides are currently supported.

:type: dict[str, Any]
:default: {}

Example::

    torch._dynamo.config.compiled_autograd_kwargs_override = {
        "fullgraph": True
    }

.. note::
   Currently only the "fullgraph" kwarg override is supported. Other kwargs
   may be added in future versions.
"""


# Enables use of collectives *during* compilation to synchronize behavior
# across ranks.  Today, this is used solely to modify automatic_dynamic_shapes
# behavior, making it so that we infer that if an input is dynamic by
# inspecting whether or not its input size varies across ranks.  Because
# this synchronization uses collectives, all ranks must run compilation at
# the same time; ranks must not diverge with graph breaks.  This can be most
# reliably achieved by ensuring PT2 only is run on SPMD programs.  If this
# invariant is inviolated, you will likely deadlock NCCL and encounter a
# NCCL timeout.
enable_compiler_collectives = os.environ.get("TORCH_COMPILER_COLLECTIVES", "0") == "1"

# Allow for experimental support of compiled p2p ops
enable_p2p_compilation = (
    os.environ.get("TORCHDYNAMO_ENABLE_P2P_COMPILATION", "0") == "1"
)

# Enables a local, filesystem "profile" which can be used for automatic
# dynamic decisions, analogous to profile-guided optimization.  This config
# ONLY has an effect if torch.compiler.config.workflow_id is specified,
# which specifies the name of the profile we will save/load.
#
# The idea is that if we observe that a particular input is dynamic over
# multiple iterations on one run, we can save a profile with this information
# so the next time we run we can just make it dynamic the first time around,
# skipping an unnecessary static compilation.  The profile can be soundly
# stale, if it is wrong, it just means we may make more things dynamic than
# was actually necessary (NB: this /can/ cause a failure if making something
# dynamic causes the compiler to stop working because you tickled a latent
# bug.)
#
# The profile is ONLY guaranteed to work if the user source code is 100%
# unchanged.  Applying the profile if there are user code changes is only
# best effort otherwise.  In particular, we identify particular code objects
# by filename, line number and name of their function, so adding/removing newlines
# will typically cause cache misses.  We continuously update the profile,
# so if we only discover something is dynamic on the second run, we will update
# the profile for subsequent runs.
automatic_dynamic_local_pgo: bool = Config(
    justknob="pytorch/remote_cache:enable_local_automatic_dynamic_pgo",
    env_name_force="TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
    default=True,
)

# Like above, but using remote cache
automatic_dynamic_remote_pgo: bool | None = get_tristate_env(
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO"
)

# temporary config to kill later
_unsafe_skip_fsdp_module_guards = (
    os.environ.get("UNSAFE_SKIP_FSDP_MODULE_GUARDS", "0") == "1"
)

# Common prefix to append to the id of each compile run to filter out data
pt2_compile_id_prefix: str | None = os.environ.get("PT2_COMPILE_ID_PREFIX", None)

# Run GC at the end of compilation
run_gc_after_compile = Config(  # type: ignore[var-annotated]
    # Disable by default on free-threaded builds since they always do a full collection, which can be slow
    default=sysconfig.get_config_var("Py_GIL_DISABLED") != 1,
    justknob="pytorch/compiler:enable_run_gc_after_compile",
    env_name_default="TORCH_DYNAMO_RUN_GC_AFTER_COMPILE",
)

# Does not graph break on torch.autograd._profiler_enabled if set to True. We
# want this flag to be True by default, but there is an unsolved bug that causes
# distributed jobs to timeout with Kineto profiler when this is set to True.
constant_fold_autograd_profiler_enabled = False

# Takes the function/module decorated with torch.compile and passes it through a
# wrapper. This ensures that nn.module hooks are also compiled in the same frame.
wrap_top_frame = False

# Flag to record runtime overhead in profile traces. Used for pre-graph bytecode
# and AOTAutograd runtime wrapper.
record_runtime_overhead = True

# Flag to enable the use of torch.compile().aot_compile() API. Should be always True.
enable_aot_compile = True

# HACK: this is for testing custom ops profiling only
_custom_ops_profile: Any | None = None

# Experimental flag to enable regional compile on invoke_subgraph HOP.
# For testing only!
enable_invoke_subgraph_regional_compile: bool = False

# When True, run a post-tracing pass that inlines all invoke_subgraph HOPs
# back into the parent graph, producing a flat FX graph. Useful when
# downstream compilers (like vllm-compile) don't support HOPs or prefer a
# flat graph.
inline_invoke_subgraph: bool = False

# Inline invoke_subgraph HOPs that are referenced by exactly one call site.
# Single-use subgraphs add overhead without deduplication benefit.
inline_single_use_invoke_subgraph: bool = True

# Clear WeakIdRef entries from TracingContext.tensor_to_context and
# MetaTensorDescriber.lookup_tensor at the end of compile. These weakrefs
# can block torch.utils.swap_tensors from working after compile.
# - None (default): clear for registered backends (inductor, eager, etc.),
#   don't clear for custom backends (to support standalone_compile, etc.)
# - True: always clear regardless of backend
# - False: never clear regardless of backend
invalidate_compile_context_weakrefs: bool | None = None

# Reorder and rename output graph nodes into a canonical topological order so
# that structurally equivalent graphs (e.g., same model traced with different
# dict iteration orders across distributed ranks) produce identical FX graphs.
canonicalize_output_graph_node_order: bool = False

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F403

    def _make_closure_patcher(**changes: Any) -> Any: ...


install_config_module(sys.modules[__name__])
<|endoftext|>"""Metrics collection and management system for Dynamo.

This module provides context managers for gathering and reporting metrics during
compilation and runtime.

It includes two main components:
- MetricsContext: A context manager for collecting metrics during compilation, supporting
  nested contexts and various metric types (counters, sets, key-value pairs)
- RuntimeMetricsContext: A specialized context for runtime metrics collection that doesn't
  require explicit context management

The metrics system enables comprehensive monitoring and analysis of both compilation and
execution performance.
"""

from __future__ import annotations

import heapq
import logging
import time
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, TypeAlias
from typing_extensions import Self


if TYPE_CHECKING:
    from collections.abc import Iterator

from torch.utils._traceback import CapturedTraceback


log = logging.getLogger(__name__)


class TopN:
    """
    Helper to record a list of metrics, keeping only the top N "most expensive" elements.
    """

    def __init__(self, at_most: int = 25) -> None:
        self.at_most = at_most
        self.heap: list[tuple[int, Any]] = []

    def add(self, key: Any, val: int) -> None:
        # Push if we haven't reached the max size, else push and pop the smallest
        fn = heapq.heappush if len(self.heap) < self.at_most else heapq.heappushpop
        fn(self.heap, (val, key))

    def __len__(self) -> int:
        return len(self.heap)

    def __iter__(self) -> Iterator[tuple[Any, int]]:
        return ((key, val) for val, key in sorted(self.heap, reverse=True))


OnExitType: TypeAlias = Callable[
    [int, int, dict[str, Any], type[BaseException] | None, BaseException | None],
    None,
]


class MetricsContext:
    def __init__(self, on_exit: OnExitType) -> None:
        """
        Use this class as a contextmanager to create a context under which to accumulate
        a set of metrics, e.g., metrics gathered during a compilation. On exit of the
        contextmanager, call the provided 'on_exit' function and pass a dictionary of
        all metrics set during the lifetime of the contextmanager.
        """
        self._on_exit = on_exit
        self._metrics: dict[str, Any] = {}
        self._start_time_ns: int = 0
        self._level: int = 0
        self._edits: list[tuple[CapturedTraceback, set[str]]] = []

    def __enter__(self) -> Self:
        """
        Initialize metrics recording.
        """
        if self._level == 0:
            # In case of recursion, track at the outermost context.
            self._metrics = {}
            self._start_time_ns = time.time_ns()

        self._level += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: Any,
    ) -> None:
        """
        At exit, call the provided on_exit function.
        """
        self._level -= 1
        if self._level < 0:
            raise AssertionError("MetricsContext level cannot become negative")
        if self._level == 0:
            try:
                end_time_ns = time.time_ns()
                self._on_exit(
                    self._start_time_ns, end_time_ns, self._metrics, exc_type, exc_value
                )
            except Exception:
                log.exception("Unexpected exception logging compilation metrics")

    def in_progress(self) -> bool:
        """
        True if we've entered the context.
        """
        return self._level > 0

    def increment(self, metric: str, value: int) -> None:
        """
        Increment a metric by a given amount.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot increment {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = 0
        self._metrics[metric] += value

    def _render_edits(self, pred: set[str]) -> str:
        return "\n\n" + "\n\n".join(
            "Previous Traceback:\n" + "".join(e.format())
            for e, k in self._edits
            if k & pred
        )

    def set(self, metric: str, value: Any, overwrite: bool = False) -> None:
        """
        Set a metric to a given value. Raises if the metric has been assigned previously
        in the current context.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot set {metric} outside of a MetricsContext")
        if metric in self._metrics and not overwrite:
            raise RuntimeError(
                self._render_edits({metric})
                + f"\n\nRuntimeError: Metric '{metric}' has already been set in the current context "
                "(see above for current and previous traceback)."
            )
        self._edits.append((CapturedTraceback.extract(skip=1), {metric}))
        self._metrics[metric] = value

    def set_key_value(self, metric: str, key: str, value: Any) -> None:
        """
        Treats a given metric as a dictionary and set the k and value within it.
        Note that the metric must be a dictionary or not present.

        We allow this to be called multiple times (i.e. for features, it's not uncommon
        for them to be used multiple times within a single compilation).
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot set {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = {}
        self._metrics[metric][key] = value

    def update(self, values: dict[str, Any], overwrite: bool = False) -> None:
        """
        Set multiple metrics directly. This method does NOT increment. Raises if any
        metric has been assigned previously in the current context and overwrite is
        not set to True.
        """
        if self._level == 0:
            raise RuntimeError("Cannot update metrics outside of a MetricsContext")
        existing = self._metrics.keys() & values.keys()
        if existing and not overwrite:
            raise RuntimeError(
                self._render_edits(set(values.keys()))
                + f"\n\nRuntimeError: Metric(s) {existing} have already been set in the current context.  "
                "(see above for current and previous traceback)."
            )
        self._edits.append((CapturedTraceback.extract(skip=1), set(values.keys())))
        self._metrics.update(values)

    def update_outer(self, values: dict[str, Any]) -> None:
        """
        Update, but only when at the outermost context.
        """
        if self._level == 0:
            raise RuntimeError("Cannot update metrics outside of a MetricsContext")
        if self._level == 1:
            self.update(values)

    def add_to_set(self, metric: str, value: Any) -> None:
        """
        Records a metric as a set() of values.
        """
        if self._level == 0:
            raise RuntimeError(f"Cannot add {metric} outside of a MetricsContext")
        if metric not in self._metrics:
            self._metrics[metric] = set()
        self._metrics[metric].add(value)

    def add_top_n(self, metric: str, key: Any, val: int) -> None:
        """
        Records a metric as a TopN set of values.
        """
        if self._level == 0:
            return
        if metric not in self._metrics:
            self._metrics[metric] = TopN()
        self._metrics[metric].add(key, val)


class RuntimeMetricsContext:
    def __init__(self, on_exit: OnExitType) -> None:
        """
        Similar to MetricsContext, but used to gather the runtime metrics that are
        decoupled from compilation, where there's not a natural place to insert a
        context manager.
        """
        self._on_exit = on_exit
        self._metrics: dict[str, Any] = {}
        self._start_time_ns: int = 0

    def increment(
        self, metric: str, value: int, extra: dict[str, Any] | None = None
    ) -> None:
        """
        Increment a metric by a given amount.
        """
        if not self._metrics:
            # Start timing on the first entry
            self._start_time_ns = time.time_ns()
        if metric not in self._metrics:
            self._metrics[metric] = 0
        self._metrics[metric] += value

        if extra:
            for k, v in extra.items():
                if k not in self._metrics and v is not None:
                    self._metrics[k] = v

    def finish(self) -> None:
        """
        Call the on_exit function with the metrics gathered so far and reset.
        """
        if self._metrics:
            try:
                end_time_ns = time.time_ns()
                self._on_exit(
                    self._start_time_ns, end_time_ns, self._metrics, None, None
                )
            except Exception:
                log.exception("Unexpected exception logging runtime metrics")
            finally:
                self._metrics = {}
<|endoftext|>"""
TorchDynamo is a Python-level JIT compiler designed to make unmodified PyTorch programs faster.
TorchDynamo hooks into the frame evaluation API in CPython (PEP 523) to dynamically modify Python
bytecode right before it is executed. It rewrites Python bytecode in order to extract sequences of
PyTorch operations into an FX Graph which is then just-in-time compiled with a customizable backend.
It creates this FX Graph through bytecode analysis and is designed to mix Python execution with
compiled backends to get the best of both worlds: usability and performance. This allows it to
seamlessly optimize PyTorch programs, including those using modern Python features.
"""

import torch

from . import (
    aot_compile,
    bytecode_debugger,
    config,
    convert_frame,
    eval_frame,
    functional_export,
    resume_execution,
)
from .backends.registry import list_backends, lookup_backend, register_backend
from .callback import callback_handler, on_compile_end, on_compile_start
from .code_context import code_context
from .convert_frame import replay
from .decorators import (
    allow_c_slot,
    allow_in_graph,
    assume_constant_result,
    disable,
    disable_nested_graph_breaks,
    disallow_in_graph,
    dont_skip_tracing,
    error_on_graph_break,
    forbid_in_graph,
    graph_break,
    is_dynamo_disable_recursive,
    mark_dynamic,
    mark_static,
    mark_static_address,
    maybe_mark_dynamic,
    nonstrict_trace,
    override_cudagraphs,
    override_optimization_hint,
    patch_dynamo_config,
    run,
    set_stance,
    skip_frame,
    step_unsupported,
    substitute_in_graph,
)
from .eval_frame import (
    _reset_guarded_backend_cache,
    explain,
    export,
    is_dynamo_supported,
    is_inductor_supported,
    optimize,
    optimize_assert,
    OptimizedModule,
    reset_code,
)

# pyrefly: ignore [deprecated]
from .external_utils import is_compiling
from .mutation_guard import GenerationTracker
from .pgo import reset_code_state
from .symbolic_convert import TensorifyState
from .utils import (
    graph_break_reasons,
    guard_failures,
    orig_code_map,
    register_hook_for_recompile_user_context,
    reset_frame_count,
    reset_recompile_user_contexts,
)


# Register polyfill functions
from .polyfills import loader as _  # usort: skip


__all__ = [
    "allow_c_slot",
    "allow_in_graph",
    "assume_constant_result",
    "bytecode_debugger",
    "config",
    "disable",
    "disable_nested_graph_breaks",
    "disallow_in_graph",
    "dont_skip_tracing",
    "export",
    "explain",
    "forbid_in_graph",
    "graph_break",
    "is_compiling",
    "is_dynamo_disable_recursive",
    "list_backends",
    "lookup_backend",
    "mark_dynamic",
    "maybe_mark_dynamic",
    "mark_static",
    "mark_static_address",
    "nonstrict_trace",
    "override_optimization_hint",
    "optimize",
    "optimize_assert",
    "OptimizedModule",
    "patch_dynamo_config",
    "register_backend",
    "replay",
    "reset",
    "reset_recompile_user_contexts",
    "run",
    "override_cudagraphs",
    "error_on_graph_break",
    "set_recursion_limit",
    "set_stance",
    "skip_frame",
    "step_unsupported",
    "substitute_in_graph",
]

# allowlist this for weights_only load of NJTs
torch.serialization.add_safe_globals([torch._dynamo.decorators._DimRange])

if torch.manual_seed is torch.random.manual_seed:
    import torch.jit._builtins

    # Wrap manual_seed with the disable decorator.
    # Can't do it at its implementation due to dependency issues.
    torch.manual_seed = torch._disable_dynamo(torch.manual_seed)
    # Add the new manual_seed to the builtin registry.
    torch.jit._builtins._register_builtin(torch.manual_seed, "aten::manual_seed")


def reset() -> None:
    """
    Clear all compile caches and restore initial state.  This function is intended
    to reset Dynamo's state *as if* you had started a fresh process invocation, which
    makes it good for testing scenarios where you want to behave as if you started
    a new process.  It does NOT affect any file system caches.

    NB: this does NOT reset logging state.  Don't use this to test logging
    initialization/reinitialization.
    """
    # TODO: https://github.com/pytorch/pytorch/issues/139200
    import logging

    log = logging.getLogger(__name__)
    log.info("torch._dynamo.reset")
    with convert_frame.compile_lock:
        reset_code_caches()
        convert_frame.input_codes.clear()
        reset_code_state()
        convert_frame.output_codes.clear()
        orig_code_map.clear()
        guard_failures.clear()
        graph_break_reasons.clear()
        resume_execution.ContinueExecutionCache.cache.clear()
        _reset_guarded_backend_cache()
        reset_frame_count()
        torch._dynamo.compiled_autograd.reset()
        convert_frame.FRAME_COUNTER = 0
        convert_frame.FRAME_COMPILE_COUNTER.clear()
        callback_handler.clear()
        GenerationTracker.clear()
        TensorifyState.clear()
        torch._dynamo.utils.warn_once_cache.clear()
        torch._C._autograd._saved_tensors_hooks_set_tracing(False)

        # Reset cudagraph trees unconditionally since they are global state
        # not tied to a specific backend instance
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
        from torch._higher_order_ops.wrap import inductor_code_side_table

        kernel_side_table.reset_table()
        inductor_code_side_table.reset_table()

        # The fake tensor dispatch cache is process-global (shared across
        # FakeTensorMode instances), so it survives into later compiles. That
        # is normally fine since it caches pure metadata propagation, but if a
        # meta kernel errantly mutates input metadata (a bug, e.g. #191283),
        # the mutation happens only on cache miss: a compile that failed cold
        # can then "succeed" when retried in the same process, which test
        # harness reruns misclassify as flakiness.
        from torch._subclasses.fake_tensor import FakeTensorMode

        FakeTensorMode.cache_clear()

        if torch.cuda.is_initialized():
            from torch._inductor.cudagraph_trees import reset_cudagraph_trees

            reset_cudagraph_trees()


def reset_code_caches() -> None:
    """
    Clears in-memory code cache, which is what stores compiled products.  This
    resets less state than :func:`reset` and is mostly only used for testing
    purposes.
    """
    # TODO: https://github.com/pytorch/pytorch/issues/139200
    import logging

    log = logging.getLogger(__name__)
    log.info("torch._dynamo.reset_code_caches")
    """Clear compile caches that are keyed by code objects"""
    with convert_frame.compile_lock:
        reset_code_state()
        for weak_code in (
            convert_frame.input_codes.seen + convert_frame.output_codes.seen
        ):
            code = weak_code()
            if code:
                reset_code(code)
        code_context.clear()


def get_recursion_limit() -> int:
    """
    Returns the internal dynamo recursion limit set by `torch._dynamo.set_recursion_limit`.

    Returns -1 if no c recursion limit has been set.
    """
    return torch._C._dynamo.eval_frame.get_c_recursion_limit()


def set_recursion_limit(limit: int) -> None:
    """
    Sets an internal dynamo recursion limit. The limit must be >= 1, or -1 to reset
    to the default (unset) state.

    This is possibly needed in Python 3.12-3.13 since there is a separate C recursion limit
    that is not visible at the Python level. If you are getting RecursionErrors during
    Dynamo compilation and `sys.setrecursionlimit()` doesn't help, this function may alleviate
    the issue.

    NOTE: this function does NOT call `sys.setrecursionlimit()` - the user is expected to manually
        call this if required. This is because the 2 recursion limits are not sync'd up - e.g. in
        Python 3.12, functions can be inline-evaluated, which apparently doesn't use up the C stack.

    WARNING: increasing the recursion limit to an arbitrary large value may cause segfaults
        due to stack overflows! You can try also try to manually increase the stack size, e.g.
        with `$ ulimit -s ...`
    """
    torch._C._dynamo.eval_frame.set_c_recursion_limit(limit)
<|endoftext|>"""
Manages process groups for distributed compilation in TorchDynamo.

This module handles the initialization and management of process groups used for
distributed compilation. Key features:

- Lazy initialization of compilation process groups
- Only creates groups when distributed mode is enabled and available
- Integrates with compiler_collectives configuration setting
- Provides a single global process group for compilation coordination

The process group is created only when needed and if the distributed environment
is properly initialized, making it safe to import and use this module even in
non-distributed scenarios.
"""

import torch.distributed as dist

from . import config


_COMPILE_PG: dist.ProcessGroup | None = None
_GUARD_PG: dist.ProcessGroup | None = None


def get_compile_pg() -> dist.ProcessGroup | None:
    if (
        config.enable_compiler_collectives
        and dist.is_available()
        and dist.is_initialized()
    ):
        global _COMPILE_PG
        if _COMPILE_PG is None:
            # , timeout=datetime.timedelta(seconds=2)
            compile_pg = dist.distributed_c10d._new_group_with_tag(
                pg_tag="pt2_compile_pg"
            )
            if compile_pg == dist.GroupMember.NON_GROUP_MEMBER:
                raise AssertionError("Compiler process group must include all ranks")
            _COMPILE_PG = compile_pg
        return _COMPILE_PG

    return None


# NB: Unlike get_compile_pg, this is only called when guard collectives were
# explicitly requested
def get_guard_pg() -> dist.ProcessGroup | None:
    if dist.is_available() and dist.is_initialized():
        global _GUARD_PG
        if _GUARD_PG is None:
            guard_pg = dist.distributed_c10d._new_group_with_tag(pg_tag="pt2_guard_pg")
            if guard_pg == dist.GroupMember.NON_GROUP_MEMBER:
                raise AssertionError("Guard process group must include all ranks")
            _GUARD_PG = guard_pg
        return _GUARD_PG

    return None
<|endoftext|>"""
This module contains utility functions that are explicitly allowed to be called during
TorchDynamo compilation. These functions are carefully vetted to ensure they work
correctly within the TorchDynamo tracing and compilation process.

Key functionality groups:

- Compilation State:
  Functions for checking compilation state (is_compiling)

- Function Wrapping:
  Utilities for wrapping functions (wrap_inline, wrap_numpy) to work with
  TorchDynamo compilation

- Autograd Hooks:
  Functions and classes for handling autograd hooks and backward passes
  (call_hook, FakeBackwardCFunction, etc.)

- Tensor Operations:
  Utility functions for tensor operations and transformations
"""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import deprecated, ParamSpec

import torch
import torch.utils._pytree as pytree


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

_P = ParamSpec("_P")
_R = TypeVar("_R")

if TYPE_CHECKING:
    # TorchScript does not support `@deprecated`
    # This is a workaround to avoid breaking TorchScript
    @deprecated(
        "`torch._dynamo.external_utils.is_compiling` is deprecated. Use `torch.compiler.is_compiling` instead.",
        category=FutureWarning,
    )
    def is_compiling() -> bool:
        return torch.compiler.is_compiling()

else:

    def is_compiling() -> bool:
        """
        Indicates whether we are tracing/compiling with torch.compile() or torch.export().
        """
        # NOTE: With `@torch.compile(backend="eager")`, torch._dynamo.is_compiling() will get traced
        # and return true. torch.compiler.is_compiling() is skipped and will return false.
        return torch.compiler.is_compiling()


def wrap_inline(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Create an extra frame around fn that is not in skipfiles.
    """

    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return fn(*args, **kwargs)

    return inner


def call_hook(
    hook: Callable[..., torch.Tensor | None], *args: Any, **kwargs: Any
) -> torch.Tensor:
    """
    Used by compiled autograd to handle hook returning None.
    """
    result = hook(*args)
    if result is None:
        return args[0]
    elif kwargs.get("hook_type") == "post_acc_grad_hook":
        raise RuntimeError("Tensor post accumulate grad hooks should return None.")
    return result


def wrap_numpy(f: Callable[_P, _R]) -> Callable[_P, _R]:
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.
    """
    if not np:
        return f

    @functools.wraps(f)
    def wrap(*args: _P.args, **kwargs: _P.kwargs) -> pytree.PyTree:
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, lambda x: x.numpy(), (args, kwargs)
        )
        out = f(*args, **kwargs)
        # pyrefly: ignore [missing-attribute]
        return pytree.tree_map_only(np.ndarray, lambda x: torch.as_tensor(x), out)

    return wrap


class FakeBackwardCFunction:
    def __init__(
        self,
        real: torch.autograd.function.BackwardCFunction,
        saved_tensors: list[torch.Tensor],
    ) -> None:
        self.real = real
        self.saved_tensors = saved_tensors

    def __getattr__(self, name: str) -> Any:
        if name == "saved_variables":
            warnings.warn(
                "'saved_variables' is deprecated; use 'saved_tensors'",
                DeprecationWarning,
            )
            return self.saved_tensors

        return getattr(self.real, name)


def call_backward(
    backward_c_function: torch.autograd.function.BackwardCFunction,
    saved_tensors: list[torch.Tensor],
    *args: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    fake = FakeBackwardCFunction(backward_c_function, saved_tensors)
    grads = fake._forward_cls.backward(fake, *args)  # type: ignore[attr-defined]

    if not isinstance(grads, tuple):
        grads = (grads,)

    return grads


def normalize_as_list(x: Any) -> list[Any]:
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


def untyped_storage_size(x: torch.Tensor) -> int:
    return x.untyped_storage().size()


class FakeCompiledAutogradEngine:
    @staticmethod
    def queue_callback(
        final_callbacks: list[Callable[[], None]], cb: Callable[[], None]
    ) -> None:
        final_callbacks.append(cb)

    @staticmethod
    def exec_final_callbacks(final_callbacks: list[Callable[[], None]]) -> None:
        i = 0
        while i < len(final_callbacks):
            cb = final_callbacks[i]
            cb()
            i += 1
        final_callbacks.clear()

    @staticmethod
    def _exec_final_callbacks_stub() -> None:
        pass


def call_hook_from_backward_state(
    *args: Any, bw_state: Any, hook_name: str, **kwargs: Any
) -> Any:
    return getattr(bw_state, hook_name)(*args, **kwargs)


def call_module_hooks_from_backward_state(
    _: Any, result: Any, *args: Any, bw_state: Any, hooks_name: str, module_name: str
) -> Any:
    module = getattr(bw_state, module_name)
    hooks = getattr(bw_state, hooks_name)
    for hook in hooks:
        new_result = hook(module, result, *args)
        if new_result is not None:
            result = new_result
    return result


# used for torch._dynamo.disable(recursive=False)
def get_nonrecursive_disable_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # wrap function to get the right error message
    # this function is in external_utils so that convert_frame doesn't skip it.
    @functools.wraps(fn)
    def nonrecursive_disable_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if torch.compiler.is_exporting():
            raise RuntimeError(
                "Non-recursive torch.compiler.disable is not supported with torch.export."
            )
        return fn(*args, **kwargs)

    return nonrecursive_disable_wrapper


def wrap_dunder_call_ctx_manager(self: Any, func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Apply self as a ctx manager around a call to func
    """

    # NOTE: do not functools.wraps(func) because we don't ever want this frame to be skipped!
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self:
            return func(*args, **kwargs)

    return inner


# Use only on ints marked dynamic via torch.empty(0, integer)
# Currently only way to mark ints as dynamic: https://github.com/pytorch/pytorch/issues/129623
def unwrap_maybe_dynamic_int(x: torch.Tensor | int) -> int:
    if isinstance(x, torch.Tensor):
        # x.size() is expected to be [0, dynamic_int]
        return x.size(1)
    return x


def call_accumulate_grad(
    variable: torch.Tensor,
    variable_grad: torch.Tensor | None,
    grad: torch.Tensor | bool,
    has_post_hooks: bool | None = None,
) -> torch.Tensor | None:
    if has_post_hooks is None:
        # Backward compatibility for graphs captured before current grad became
        # an explicit argument.
        has_post_hooks = bool(grad)
        grad = variable_grad  # type: ignore[assignment]
        variable_grad = variable.grad
    updated_grad = torch._dynamo.compiled_autograd.ops.AccumulateGrad(  # type: ignore[attr-defined]
        [grad], variable, variable_grad, has_post_hooks
    )
    variable.grad = updated_grad[0]
    return variable.grad


def wrap_inline_with_error_on_graph_break(
    fn: Callable[_P, _R], error_on_graph_break: bool
) -> Callable[_P, _R]:
    # NB: need multiple definitions in order to prevent `fullgraph` from
    # being a freevar of wrapper
    # NOTE: do not functools.wraps(fn) because we don't ever want these wrappers to be skipped!
    if error_on_graph_break:

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with torch._dynamo.error_on_graph_break(True):
                return fn(*args, **kwargs)

    else:

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with torch._dynamo.error_on_graph_break(False):
                return fn(*args, **kwargs)

    return wrapper


def filter_out_const_values(tup: tuple[Any, ...], masks: list[bool]) -> tuple[Any, ...]:
    """
    masks is a list of bools, where True means the corresponding element in tup
    is a const value. Filter out the const values.
    """
    out = []
    for mask_idx, mask in enumerate(masks):
        if not mask:
            out.append(tup[mask_idx])
    return tuple(out)


def insert_const_values_with_mask(
    tup: tuple[Any, ...], masks: list[bool], values: tuple[Any, ...]
) -> tuple[Any, ...]:
    """
    masks and values are of same length. For indices where the mask is True, use
    the const_values to fill in.
    """
    out = []
    idx = 0
    for mask_idx, mask in enumerate(masks):
        if mask:
            out.append(values[mask_idx])
        else:
            out.append(tup[idx])
            idx += 1
    return tuple(out)
<|endoftext|>from typing import Any

import torch
from torch.fx import Graph, map_arg, Node
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten


# flattens with support for slices
# Note: a better way to do this would
# be register/unregister slices as pytree nodes
# but there is no unregister API in the pytorch
# pytree impl
def _get_flat_args(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> list[Node]:
    args = list[Any]()
    map_arg((node.args, node.kwargs), args.append)
    if node in node_to_additional_deps:
        args.extend(node_to_additional_deps[node])
    return args


def _get_flat_args_unique(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> OrderedSet[Node]:
    args = OrderedSet[Node]()
    map_arg((node.args, node.kwargs), args.add)
    if node in node_to_additional_deps:
        args.update(node_to_additional_deps[node])
    return args


def _detect_cycles(
    graph: Graph, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> str:
    # States: 0=Unvisited, 1=Visiting, 2=Visited(Safe)
    state: dict[Node, int] = {}

    for root in reversed(graph.nodes):
        if root in state:
            continue

        # Stack holds (current_node, children_iterator).
        # Using an iterator allows us to pause and resume processing a node's children.
        stack = [(root, iter(_get_flat_args_unique(root, node_to_additional_deps)))]
        state[root] = 1  # Visiting

        while stack:
            parent, children = stack[-1]

            try:
                child = next(children)

                if not isinstance(child, Node):
                    continue

                child_state = state.get(child, 0)

                if child_state == 1:
                    # Back-edge: child is on the current DFS path -> cycle
                    cycle_path = [node for node, _ in stack] + [child]
                    return f"cycle detected in path: {cycle_path}"

                if child_state == 0:
                    state[child] = 1
                    stack.append(
                        (
                            child,
                            iter(_get_flat_args_unique(child, node_to_additional_deps)),
                        )
                    )
                # child_state == 2 means already verified safe; skip.

            except StopIteration:
                # All children processed — mark safe and pop.
                stack.pop()
                state[parent] = 2

    return "no cycle detected"


def _graph_device_type(graph: Graph | None) -> str:
    if graph is None:
        return "cpu"

    def _device_type(x: Any) -> str:
        if isinstance(x, torch.device):
            return x.type
        if isinstance(x, torch.Tensor):
            return x.device.type
        return "cpu"

    def _flatten_meta(node: Node, key: str) -> list[Any]:
        if key not in node.meta:
            return []
        flat, _ = tree_flatten(node.meta[key])
        return flat

    for node in graph.nodes:
        for key in ("val", "example_value"):
            for obj in _flatten_meta(node, key):
                return _device_type(obj)

        # Check for device conversions
        if node.op == "call_method":
            for gpu in ["cuda", "xpu"]:
                if node.target == gpu:
                    return gpu
                if node.target == "to" and gpu in node.args:
                    return gpu

        # Check args/kwargs for non-CPU device specs
        flat_args, _ = tree_flatten((node.args, node.kwargs))
        for obj in flat_args:
            return _device_type(obj)
    return "cpu"
<|endoftext|>"""
Bytecode debugger for Dynamo-optimized code.

This module provides a pdb-like debugger for stepping through Python bytecode
one instruction at a time, with the ability to inspect the value stack,
locals, and globals.

Usage:
    >>> # xdoctest: +SKIP
    >>> import torch
    >>>
    >>> @torch.compile
    >>> def my_fn(x):
    ...     return x + 1
    >>>
    >>> with torch._dynamo.bytecode_debugger.debug():
    ...     my_fn(torch.randn(3))  # Debugger activates on Dynamo-generated code

Programmatic breakpoints (for Dynamo developers):
    In PyCodegen, use create_breakpoint() to insert a debugger stop:

        from torch._dynamo.bytecode_transformation import create_breakpoint
        codegen.extend_output(create_breakpoint())
"""

from __future__ import annotations

import dis
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from typing_extensions import Self

from .bytecode_transformation import convert_instruction, Instruction, instruction_size


# Python 3.12+ has sys.monitoring for efficient instruction-level tracing
_HAS_SYS_MONITORING = hasattr(sys, "monitoring")


# Sentinel for breakpoints inserted by PyCodegen via create_breakpoint()
class _BreakpointMarker:
    """Sentinel constant that signals the debugger to break.

    Usage in PyCodegen:
        from torch._dynamo.bytecode_transformation import create_breakpoint
        codegen.extend_output(create_breakpoint())
    """

    __slots__ = ()
    _instance: _BreakpointMarker | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = cast("_BreakpointMarker", object.__new__(cls))
        return cls._instance

    def __repr__(self) -> str:
        return "<BREAKPOINT>"


BREAKPOINT_MARKER = _BreakpointMarker()


# Import NULL_STACK_VALUE sentinel from C++ module
# This is returned by _get_frame_value_stack_with_depth for NULL stack slots
from torch._C._dynamo.eval_frame import NULL_STACK_VALUE


@dataclass
class CodeInfo:
    """Per-code-object data shared across all frames executing the same code."""

    code: types.CodeType
    instructions: list[Instruction]
    offset_to_inst: dict[int, Instruction]
    offset_to_index: dict[int, int]
    index_width: int
    offset_width: int
    breakpoints: set[int] = field(default_factory=set)


@dataclass
class DebuggerState:
    """Per-frame debugging state."""

    code_info: CodeInfo
    frame: types.FrameType
    current_offset: int = -1  # -1 indicates first instruction not yet seen
    current_stack_depth: int = 0  # Tracked dynamically
    first_instruction_seen: bool = False
    step_mode: bool = True
    step_count: int = 0  # Remaining steps for "s [n]" command (0 = prompt each time)
    user_locals: dict[str, Any] = field(
        default_factory=dict
    )  # User-defined variables from debugger
    last_command: str = "s"  # Last command for repeat on empty input
    list_index: int | None = (
        None  # Current position for 'l' command (None = use current)
    )

    @property
    def code(self) -> types.CodeType:
        return self.code_info.code

    @property
    def instructions(self) -> list[Instruction]:
        return self.code_info.instructions

    @property
    def offset_to_inst(self) -> dict[int, Instruction]:
        return self.code_info.offset_to_inst

    @property
    def offset_to_index(self) -> dict[int, int]:
        return self.code_info.offset_to_index

    @property
    def index_width(self) -> int:
        return self.code_info.index_width

    @property
    def offset_width(self) -> int:
        return self.code_info.offset_width

    @property
    def breakpoints(self) -> set[int]:
        return self.code_info.breakpoints


class _DebugContext:
    """Internal debug context that manages the debugging session."""

    def __init__(self) -> None:
        self._code_info: dict[types.CodeType, CodeInfo] = {}
        self._frame_states: dict[types.FrameType, DebuggerState] = {}
        self._active = False
        self._tracked_codes: set[types.CodeType] = set()
        self._old_trace: Callable[..., Any] | None = None
        self._prev_callback: Callable[..., Any] | None = None
        self._return_from_frame: types.FrameType | None = None
        self._stop_after_return: bool = False
        self._next_in_frame: types.FrameType | None = None
        self._next_count: int = 0
        self._verbose: bool = False
        self._stop_at_new_code: bool = True
        self._quitting: bool = False
        if _HAS_SYS_MONITORING:
            self._tool_id = sys.monitoring.DEBUGGER_ID

    def get_instructions(self, code: types.CodeType | None = None) -> list[Instruction]:
        """Get the list of instructions for a tracked code object.

        Args:
            code: The code object to get instructions for. If None, returns
                  instructions from the most recently tracked code object.

        Returns:
            List of instructions, or empty list if the code is not tracked.
        """
        if not self._code_info:
            return []
        if code is None:
            info = next(reversed(self._code_info.values()))
        else:
            info = self._code_info.get(code)
            if info is None:
                return []
        return info.instructions

    def get_tracked_codes(self) -> list[types.CodeType]:
        """Get all code objects that have been tracked by the debugger."""
        return list(self._code_info.keys())

    @staticmethod
    def is_programmatic_breakpoint(inst: Instruction) -> bool:
        return inst.opname == "LOAD_CONST" and inst.argval is BREAKPOINT_MARKER

    def _get_or_create_code_info(self, code: types.CodeType) -> CodeInfo:
        """Get or create CodeInfo for a code object."""
        if code not in self._code_info:
            # Use dis.get_instructions directly to preserve original offsets.
            # cleaned_instructions strips EXTENDED_ARG and recalculates offsets,
            # which would cause mismatches with offsets reported by callbacks.
            instructions = [convert_instruction(i) for i in dis.get_instructions(code)]

            # In 3.11+, instructions have inline cache entries that occupy
            # bytecode space (e.g. BINARY_OP is 2 bytes + 2 bytes cache).
            # dis.get_instructions only reports the instruction start offset,
            # but sys.monitoring RAISE reports offsets within cache entries.
            # Map the full byte range of each instruction so lookups succeed.
            offset_to_inst: dict[int, Instruction] = {}
            offset_to_index: dict[int, int] = {}
            for i, inst in enumerate(instructions):
                if inst.offset is not None:
                    inst_size = instruction_size(inst)
                    for off in range(inst.offset, inst.offset + inst_size, 2):
                        offset_to_inst[off] = inst
                        offset_to_index[off] = i

            max_index = len(instructions) - 1 if instructions else 0
            max_offset = max(
                (inst.offset for inst in instructions if inst.offset is not None),
                default=0,
            )

            # Pre-populate breakpoints from BREAKPOINT_MARKER instructions
            programmatic_breakpoints: set[int] = {
                i
                for i, inst in enumerate(instructions)
                if self.is_programmatic_breakpoint(inst)
            }

            self._code_info[code] = CodeInfo(
                code=code,
                instructions=instructions,
                offset_to_inst=offset_to_inst,
                offset_to_index=offset_to_index,
                index_width=max(1, len(str(max_index))),
                offset_width=max(1, len(str(max_offset))),
                breakpoints=programmatic_breakpoints,
            )
        return self._code_info[code]

    def _get_or_create_frame_state(
        self, code: types.CodeType, frame: types.FrameType
    ) -> DebuggerState:
        """Get or create per-frame DebuggerState."""
        if frame not in self._frame_states:
            code_info = self._get_or_create_code_info(code)
            self._frame_states[frame] = DebuggerState(code_info=code_info, frame=frame)
