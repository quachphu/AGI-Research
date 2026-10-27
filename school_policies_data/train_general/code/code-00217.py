    Make a ScriptModule from an nn.Module, using the interface methods rule for determining which methods to compile.

    Args:
        mod_interface: the interface type that the module have
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
    """
    if isinstance(nn_module, torch.jit.ScriptModule):
        return nn_module

    check_module_initialized(nn_module)

    def infer_interface_methods_to_compile(nn_module):
        """Rule to infer the methods from the interface type.

        It is used to know which methods need to act as starting points for compilation.
        """
        stubs = [
            make_stub_from_method(nn_module, method)
            for method in mod_interface.getMethodNames()
        ]
        return stubs

    return create_script_module(nn_module, infer_interface_methods_to_compile)


def try_compile_fn(fn, loc):
    if _jit_internal.is_ignored_fn(fn):
        # Don't do anything for @ignore'd functions
        return None

    if isinstance(fn, torch.nn.Module):
        # Since modules are callable pybind recognizes them as functions, but
        # don't do anything for them
        return None

    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        raise RuntimeError(
            f"`{fn}` is not a function. Recursive scripting only supports "
            "Python functions or methods currently.\n"
            f"Consider manually annotating `{fn}` with @torch.jit.script."
        )

    # The object returned by __prepare_scriptable__ might have a different closure.
    # Resolve it here to get the right resolution callback.
    fn = fn.__prepare_scriptable__() if hasattr(fn, "__prepare_scriptable__") else fn  # type: ignore[operator]

    # We don't have the actual scope where the function was defined, but we can
    # extract the necessary info from the closed over variables on the function
    # object
    rcb = _jit_internal.createResolutionCallbackFromClosure(fn)
    return torch.jit.script(fn, _rcb=rcb)


def wrap_cpp_class(cpp_class):
    """Wrap this torch._C.Object in a Python RecursiveScriptClass."""
    return torch.jit.RecursiveScriptClass(cpp_class)


def wrap_cpp_module(cpp_module):
    """Wrap this torch._C.ScriptModule in a Python ScriptModule, recursively for all submodules."""

    def init_fn(script_module) -> None:
        for name, cpp_module in torch._C.ModuleDict(script_module._c).items():
            setattr(script_module, name, wrap_cpp_module(cpp_module))
        script_module._concrete_type = torch._C.ConcreteModuleType.from_jit_type(
            script_module._c._type()
        )

        for idx, fn in enumerate(script_module._c._get_forward_pre_hooks()):
            script_module._forward_pre_hooks[idx] = fn
        for idx, fn in enumerate(script_module._c._get_forward_hooks()):
            script_module._forward_hooks[idx] = fn

    return torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)


def compile_unbound_method(concrete_type, fn):
    if _jit_internal.is_ignored_fn(fn):
        return None
    stub = make_stub(fn, fn.__name__)
    with torch._jit_internal._disable_emit_hooks():
        # We don't want to call the hooks here since the graph that is calling
        # this function is not yet complete
        create_methods_and_properties_from_stubs(concrete_type, (stub,), ())
    return stub


def lazy_bind(concrete_type, unbound_method):
    """
    Return a function that lazily binds `unbound_method` to a provided Module IValue, then invokes the method.

    We do this so that any Python shenanigans that
    will poison type sharing are impossible at compile time.
    """

    def lazy_binding_method(cpp_module, *args):
        def init_fn(script_module) -> None:
            orig_class = concrete_type.py_class

            # Copy @ignored/@unused methods from the original module to the new one.
            # This ensures they are available during execution.
            for name in dir(orig_class):
                item = getattr(orig_class, name, None)
                if _jit_internal.is_ignored_fn(item):
                    setattr(script_module, name, item)

            # Copy constants over so they are available during execution.
            for name, value in concrete_type.get_constants().items():
                setattr(script_module, name, value)

        script_module = torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)
        method = types.MethodType(unbound_method, script_module)
        return method(*args)

    # make the lazy binding method "look like" the original method
    lazy_binding_method.original_fn = unbound_method  # type: ignore[attr-defined]
    lazy_binding_method.__name__ = unbound_method.__name__
    torch._jit_internal.copy_torchscript_modifier(unbound_method, lazy_binding_method)

    return lazy_binding_method
<|endoftext|># mypy: allow-untyped-defs
import ast
import builtins
import dis
import enum
import inspect
import re
import typing
import warnings
from textwrap import dedent

import torch
from torch._C import (
    _GeneratorType,
    AnyType,
    AwaitType,
    BoolType,
    ComplexType,
    DeviceObjType,
    DictType,
    EnumType,
    FloatType,
    FutureType,
    InterfaceType,
    IntType,
    ListType,
    NoneType,
    NumberType,
    OptionalType,
    StreamObjType,
    StringType,
    TensorType,
    TupleType,
    UnionType,
)
from torch._jit_internal import (
    _Await,
    _qualified_name,
    Any,
    BroadcastingList1,
    BroadcastingList2,  # pyrefly: ignore [missing-module-attribute]
    BroadcastingList3,  # pyrefly: ignore [missing-module-attribute]
    Dict,
    Future,
    is_await,
    is_dict,
    is_future,
    is_ignored_fn,
    is_list,
    is_optional,
    is_tuple,
    is_union,
    List,
    Optional,
    Tuple,
    Union,
)
from torch._sources import get_source_lines_and_file

from ._state import _get_script_class


if torch.distributed.rpc.is_available():
    from torch._C import RRefType
    from torch._jit_internal import is_rref, RRef

from torch._ops import OpOverloadPacket


class Module:
    def __init__(self, name, members) -> None:
        self.name = name
        self.members = members

    def __getattr__(self, name):
        try:
            return self.members[name]
        except KeyError:
            raise RuntimeError(
                f"Module {self.name} has no member called {name}"
            ) from None


class EvalEnv:
    env = {
        "torch": Module("torch", {"Tensor": torch.Tensor}),
        "Tensor": torch.Tensor,
        "typing": Module("typing", {"Tuple": Tuple}),
        "Tuple": Tuple,
        "List": List,
        "Dict": Dict,
        "Optional": Optional,
        "Union": Union,
        "Future": Future,
        "Await": _Await,
    }

    def __init__(self, rcb) -> None:
        self.rcb = rcb
        if torch.distributed.rpc.is_available():
            # pyrefly: ignore [bad-typed-dict-key, unsupported-operation]
            self.env["RRef"] = RRef

    def __getitem__(self, name):
        if name in self.env:
            return self.env[name]
        if self.rcb is not None:
            return self.rcb(name)
        return getattr(builtins, name, None)


def get_signature(fn, rcb, loc, is_method):
    if isinstance(fn, OpOverloadPacket):
        signature = try_real_annotations(fn.op, loc)
    else:
        signature = try_real_annotations(fn, loc)
    if signature is not None and is_method:
        # If this is a method, then the signature will include a type for
        # `self`, but type comments do not contain a `self`. So strip it
        # away here so everything is consistent (`inspect.ismethod` does
        # not work here since `fn` is unbound at this point)
        param_types, return_type = signature
        param_types = param_types[1:]
        signature = (param_types, return_type)

    if signature is None:
        type_line, source = None, None
        try:
            source = dedent("".join(get_source_lines_and_file(fn)[0]))
            type_line = get_type_line(source)
        except TypeError:
            pass
        # This might happen both because we failed to get the source of fn, or
        # because it didn't have any annotations.
        if type_line is not None:
            signature = parse_type_line(type_line, rcb, loc)

    return signature


def is_function_or_method(the_callable):
    # A stricter version of `inspect.isroutine` that does not pass for built-in
    # functions
    return inspect.isfunction(the_callable) or inspect.ismethod(the_callable)


def is_vararg(the_callable):
    if not is_function_or_method(the_callable) and callable(the_callable):
        # If `the_callable` is a class, de-sugar the call so we can still get
        # the signature
        the_callable = the_callable.__call__

    if is_function_or_method(the_callable):
        return inspect.getfullargspec(the_callable).varargs is not None
    else:
        return False


def get_param_names(fn, n_args):
    if isinstance(fn, OpOverloadPacket):
        fn = fn.op

    if (
        not is_function_or_method(fn)
        and callable(fn)
        and is_function_or_method(fn.__call__)
    ):
        # De-sugar calls to classes
        fn = fn.__call__

    if is_function_or_method(fn):
        if is_ignored_fn(fn):
            fn = inspect.unwrap(fn)
        return inspect.getfullargspec(fn).args
    else:
        # The `fn` was not a method or function (maybe a class with a __call__
        # method, so use a default param name list)
        return [str(i) for i in range(n_args)]


def check_fn(fn, loc) -> None:
    # Make sure the function definition is not a class instantiation
    try:
        source = dedent("".join(get_source_lines_and_file(fn)[0]))
    except (OSError, TypeError):
        return
    if source is None:
        return

    py_ast = ast.parse(source)
    if len(py_ast.body) == 1 and isinstance(py_ast.body[0], ast.ClassDef):
        raise torch.jit.frontend.FrontendError(
            loc,
            f"Cannot instantiate class '{py_ast.body[0].name}' in a script function",
        )
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise torch.jit.frontend.FrontendError(
            loc, "Expected a single top-level function"
        )


def _eval_no_call(stmt, glob, loc):
    """Evaluate statement as long as it does not contain any method/function calls."""
    bytecode = compile(stmt, "", mode="eval")
    for insn in dis.get_instructions(bytecode):
        if "CALL" in insn.opname:
            raise RuntimeError(
                f"Type annotation should not contain calls, but '{stmt}' does"
            )
    return eval(bytecode, glob, loc)  # type: ignore[arg-type] # noqa: P204


def parse_type_line(type_line, rcb, loc):
    """Parse a type annotation specified as a comment.

    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    """
    arg_ann_str, ret_ann_str = split_type_line(type_line)

    try:
        arg_ann = _eval_no_call(arg_ann_str, {}, EvalEnv(rcb))
    except (NameError, SyntaxError) as e:
        raise RuntimeError(
            "Failed to parse the argument list of a type annotation"
        ) from e

    if not isinstance(arg_ann, tuple):
        arg_ann = (arg_ann,)

    try:
        ret_ann = _eval_no_call(ret_ann_str, {}, EvalEnv(rcb))
    except (NameError, SyntaxError) as e:
        raise RuntimeError(
            "Failed to parse the return type of a type annotation"
        ) from e

    arg_types = [ann_to_type(ann, loc) for ann in arg_ann]
    return arg_types, ann_to_type(ret_ann, loc)


def get_type_line(source):
    """Try to find the line containing a comment with the type annotation."""
    type_comment = "# type:"

    lines = source.split("\n")
    lines = list(enumerate(lines))
    type_lines = list(filter(lambda line: type_comment in line[1], lines))
    # `type: ignore` comments may be needed in JIT'ed functions for mypy, due
    # to the hack in torch/_VF.py.

    # An ignore type comment can be of following format:
    #   1) type: ignore
    #   2) type: ignore[rule-code]
    # This ignore statement must be at the end of the line

    # adding an extra backslash before the space, to avoid triggering
    # one of the checks in .github/workflows/lint.yml
    type_pattern = re.compile("# type:\\ ignore(\\[[a-zA-Z-]+\\])?$")
    type_lines = list(filter(lambda line: not type_pattern.search(line[1]), type_lines))

    if len(type_lines) == 0:
        # Catch common typo patterns like extra spaces, typo in 'ignore', etc.
        wrong_type_pattern = re.compile("#[\t ]*type[\t ]*(?!: ignore(\\[.*\\])?$):")
        wrong_type_lines = list(
            filter(lambda line: wrong_type_pattern.search(line[1]), lines)
        )
        if len(wrong_type_lines) > 0:
            raise RuntimeError(
                "The annotation prefix in line "
                + str(wrong_type_lines[0][0])
                + " is probably invalid.\nIt must be '# type:'"
                + "\nSee PEP 484 (https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code)"
                + "\nfor examples"
            )
        return None
    elif len(type_lines) == 1:
        # Only 1 type line, quit now
        return type_lines[0][1].strip()

    # Parse split up argument types according to PEP 484
    # https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code
    return_line = None
    parameter_type_lines = []
    for line_num, line in type_lines:
        if "# type: (...) -> " in line:
            return_line = (line_num, line)
            break
        elif type_comment in line:
            parameter_type_lines.append(line)
    if return_line is None:
        raise RuntimeError(
            "Return type line '# type: (...) -> ...' not found on multiline "
            "type annotation\nfor type lines:\n"
            + "\n".join([line[1] for line in type_lines])
            + "\n(See PEP 484 https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code)"
        )

    def get_parameter_type(line):
        item_type = line[line.find(type_comment) + len(type_comment) :]
        return item_type.strip()

    types = map(get_parameter_type, parameter_type_lines)
    parameter_types = ", ".join(types)

    return return_line[1].replace("...", parameter_types)


def split_type_line(type_line):
    """Split the comment with the type annotation into parts for argument and return types.

    For example, for an input of:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor, Tensor]

    This function will return:
        ("(Tensor, torch.Tensor)", "Tuple[Tensor, Tensor]")

    """
    start_offset = len("# type:")
    try:
        arrow_pos = type_line.index("->")
    except ValueError:
        raise RuntimeError(
            "Syntax error in type annotation (couldn't find `->`)"
        ) from None
    return type_line[start_offset:arrow_pos].strip(), type_line[arrow_pos + 2 :].strip()


def try_real_annotations(fn, loc):
    """Try to use the Py3.5+ annotation syntax to get the type."""
    try:
        # Note: anything annotated as `Optional[T]` will automatically
        # be returned as `Union[T, None]` per
        # https://github.com/python/cpython/blob/main/Lib/typing.py#L732
        sig = inspect.signature(fn)
    except ValueError:
        return None

    all_annots = [sig.return_annotation] + [
        p.annotation for p in sig.parameters.values()
    ]
    if all(ann is sig.empty for ann in all_annots):
        return None

    arg_types = [ann_to_type(p.annotation, loc) for p in sig.parameters.values()]
    return_type = ann_to_type(sig.return_annotation, loc)
    return arg_types, return_type


# Finds common type for enum values belonging to an Enum class. If not all
# values have the same type, AnyType is returned.
def get_enum_value_type(e: type[enum.Enum], loc):
    enum_values: List[enum.Enum] = list(e)
    if not enum_values:
        raise ValueError(f"No enum values defined for: '{e.__class__}'")

    types = {type(v.value) for v in enum_values}
    ir_types = [try_ann_to_type(t, loc) for t in types]

    # If Enum values are of different types, an exception will be raised here.
    # Even though Python supports this case, we chose to not implement it to
    # avoid overcomplicating logic here for a rare use case. Please report a
    # feature request if you find it necessary.
    res = torch._C.unify_type_list(ir_types)
    if not res:
        return AnyType.get()
    return res


def is_tensor(ann) -> bool:
    if issubclass(ann, torch.Tensor):
        return True

    if issubclass(
        ann,
        (
            torch.LongTensor,
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.IntTensor,
            torch.ShortTensor,
            torch.HalfTensor,
            torch.CharTensor,
            torch.ByteTensor,
            torch.BoolTensor,
        ),
    ):
        warnings.warn(
            "TorchScript will treat type annotations of Tensor "
            "dtype-specific subtypes as if they are normal Tensors. "
            "dtype constraints are not enforced in compilation either.",
            stacklevel=2,
        )
        return True

    return False


def _fake_rcb(inp) -> None:
    return None


def try_ann_to_type(ann, loc, rcb=None):
    ann_args = typing.get_args(ann)  # always returns a tuple!

    if ann is inspect.Signature.empty:
        return TensorType.getInferred()
    if ann is None:
        return NoneType.get()
    if inspect.isclass(ann) and is_tensor(ann):
        return TensorType.get()
    if is_tuple(ann):
        # Special case for the empty Tuple type annotation `Tuple[()]`
        if len(ann_args) == 1 and ann_args[0] == ():
            return TupleType([])
        return TupleType([try_ann_to_type(a, loc) for a in ann_args])
    if is_list(ann):
        elem_type = try_ann_to_type(ann_args[0], loc)
        if elem_type:
            return ListType(elem_type)
    if is_dict(ann):
        key = try_ann_to_type(ann_args[0], loc)
        value = try_ann_to_type(ann_args[1], loc)
        # Raise error if key or value is None
        if key is None:
            raise ValueError(
                f"Unknown type annotation: '{ann_args[0]}' at {loc.highlight()}"
            )
        if value is None:
            raise ValueError(
                f"Unknown type annotation: '{ann_args[1]}' at {loc.highlight()}"
            )
        return DictType(key, value)
    if is_optional(ann):
        if issubclass(ann_args[1], type(None)):
            contained = ann_args[0]
        else:
            contained = ann_args[1]
        valid_type = try_ann_to_type(contained, loc)
        if not valid_type:
            raise AssertionError(
                f"Unsupported annotation {repr(ann)} could not be resolved because "
                f"{repr(contained)} could not be resolved. At\n{repr(loc)}"
            )
        return OptionalType(valid_type)
    if is_union(ann):
        # TODO: this is hack to recognize NumberType
        if set(ann_args) == {int, float, complex}:
            return NumberType.get()
        inner: List = []
        # We need these extra checks because both `None` and invalid
        # values will return `None`
        # TODO: Determine if the other cases need to be fixed as well
        for a in typing.get_args(ann):
            if a is None:
                inner.append(NoneType.get())
            maybe_type = try_ann_to_type(a, loc)
            if not maybe_type:
                raise AssertionError(
                    f"Unsupported annotation {repr(ann)} could not be resolved because "
                    f"{repr(a)} could not be resolved. At\n{repr(loc)}"
                )
            inner.append(maybe_type)
        return UnionType(inner)  # type: ignore[arg-type]
    if torch.distributed.rpc.is_available() and is_rref(ann):
        return RRefType(try_ann_to_type(ann_args[0], loc))
    if is_future(ann):
        return FutureType(try_ann_to_type(ann_args[0], loc))
    if is_await(ann):
        elementType = try_ann_to_type(ann_args[0], loc) if ann_args else AnyType.get()
        return AwaitType(elementType)
    if ann is float:
        return FloatType.get()
    if ann is complex:
        return ComplexType.get()
    if ann is int or ann is torch.SymInt:
        return IntType.get()
    if ann is str:
        return StringType.get()
    if ann is bool:
        return BoolType.get()
    if ann is Any:
        return AnyType.get()
    if ann is type(None):
        return NoneType.get()
    if inspect.isclass(ann) and hasattr(ann, "__torch_script_interface__"):
        return InterfaceType(ann.__torch_script_interface__)
    if ann is torch.device:
        return DeviceObjType.get()
    if ann is torch.Generator:
        return _GeneratorType.get()
    if ann is torch.Stream:
        return StreamObjType.get()
    if ann is torch.dtype:
        return IntType.get()  # dtype not yet bound in as its own type
    if ann is torch.qscheme:
        return IntType.get()  # qscheme not yet bound in as its own type

    if inspect.isclass(ann) and issubclass(ann, enum.Enum):
        if _get_script_class(ann) is None:
            scripted_class = torch.jit._script._recursive_compile_class(ann, loc)
            name = scripted_class.qualified_name()
        else:
            name = _qualified_name(ann)
        return EnumType(name, get_enum_value_type(ann, loc), list(ann))
    if inspect.isclass(ann):
        maybe_script_class = _get_script_class(ann)
        if maybe_script_class is not None:
            return maybe_script_class
        if torch._jit_internal.can_compile_class(ann):
            return torch.jit._script._recursive_compile_class(ann, loc)

    # Maybe resolve a NamedTuple to a Tuple Type
    if rcb is None:
        rcb = _fake_rcb
    # pyrefly: ignore [bad-argument-type]
    return torch._C._resolve_type_from_object(ann, loc, rcb)


def ann_to_type(ann, loc, rcb=None):
    the_type = try_ann_to_type(ann, loc, rcb)
    if the_type is not None:
        return the_type
    raise ValueError(f"Unknown type annotation: '{ann}' at {loc.highlight()}")


__all__ = [
    "Any",
    "List",
    "BroadcastingList1",
    "BroadcastingList2",
    "BroadcastingList3",
    "Tuple",
    "is_tuple",
    "is_list",
    "Dict",
    "is_dict",
    "is_optional",
    "is_union",
    "TensorType",
    "TupleType",
    "FloatType",
    "ComplexType",
    "IntType",
    "ListType",
    "StringType",
    "DictType",
    "AnyType",
    "Module",
    # TODO: Consider not exporting these during wildcard import (reserve
    # that for the types; for idiomatic typing code.)
    "get_signature",
    "check_fn",
    "get_param_names",
    "parse_type_line",
    "get_type_line",
    "split_type_line",
    "try_real_annotations",
    "try_ann_to_type",
    "ann_to_type",
]
<|endoftext|># mypy: allow-untyped-defs
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch._C

# These are imported so users can access them from the `torch.jit` module
from torch._jit_internal import (
    _Await,
    _drop,
    _IgnoreContextManager,
    _isinstance,
    _overload,
    _overload_method,
    export,
    Final,
    Future,
    ignore,
    is_scripting,
    unused,
)
from torch.jit._async import fork, wait
from torch.jit._await import _awaitable, _awaitable_nowait, _awaitable_wait
from torch.jit._decomposition_utils import _register_decomposition
from torch.jit._freeze import freeze, optimize_for_inference, run_frozen_optimizations
from torch.jit._fuser import (
    fuser,
    last_executed_optimized_graph,
    optimized_execution,
    set_fusion_strategy,
)
from torch.jit._ir_utils import _InsertPoint
from torch.jit._script import (
    _ScriptProfile,
    _unwrap_optional,
    Attribute,
    CompilationUnit,
    interface,
    RecursiveScriptClass,
    RecursiveScriptModule,
    script,
    script_method,
    ScriptFunction,
    ScriptModule,
    ScriptWarning,
)
from torch.jit._serialization import (
    jit_module_from_flatbuffer,
    load,
    save,
    save_jit_module_to_flatbuffer,
)
from torch.jit._trace import (
    _flatten,
    _get_trace_graph,
    _script_if_tracing,
    _unique_state_dict,
    is_tracing,
    ONNXTracedModule,
    TopLevelTracedModule,
    trace,
    trace_module,
    TracedModule,
    TracerWarning,
    TracingCheckError,
)
from torch.utils import set_module


__all__ = [
    "Attribute",
    "CompilationUnit",
    "Error",
    "Future",
    "ScriptFunction",
    "ScriptModule",
    "annotate",
    "enable_onednn_fusion",
    "export",
    "export_opnames",
    "fork",
    "freeze",
    "interface",
    "ignore",
    "isinstance",
    "load",
    "onednn_fusion_enabled",
    "optimize_for_inference",
    "save",
    "script",
    "script_if_tracing",
    "set_fusion_strategy",
    "strict_fusion",
    "trace",
    "trace_module",
    "unused",
    "wait",
]

# For backwards compatibility
_fork = fork
_wait = wait
_set_fusion_strategy = set_fusion_strategy


def export_opnames(m):
    r"""
    Generate new bytecode for a Script module.

    Returns what the op list would be for a Script Module based off the current code base.

    If you have a LiteScriptModule and want to get the currently present
    list of ops call _export_operator_list instead.
    """
    return torch._C._export_opnames(m._c)


# torch.jit.Error
Error = torch._C.JITException
set_module(Error, "torch.jit")
# This is not perfect but works in common cases
Error.__name__ = "Error"
Error.__qualname__ = "Error"


# for use in python if using annotate
def annotate(the_type, the_value):
    """Use to give type of `the_value` in TorchScript compiler.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    This method is a pass-through function that returns `the_value`, used to hint TorchScript
    compiler the type of `the_value`. It is a no-op when running outside of TorchScript.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume
      it is type `T` rather than `Optional[T]`

    Note that `annotate()` does not help in `__init__` method of `torch.nn.Module` subclasses because it
    is executed in eager mode. To annotate types of `torch.nn.Module` attributes,
    use :meth:`~torch.jit.Attribute` instead.

    Example:

    .. testcode::

        import torch
        from typing import Dict

        @torch.jit.script
        def fn():
            # Telling TorchScript that this empty dictionary is a (str -> int) dictionary
            # instead of default dictionary type of (str -> Tensor).
            d = torch.jit.annotate(Dict[str, int], {})

            # Without `torch.jit.annotate` above, following statement would fail because of
            # type mismatch.
            d["name"] = 20

    .. testcleanup::

        del fn

    Args:
        the_type: Python type that should be passed to TorchScript compiler as type hint for `the_value`
        the_value: Value or expression to hint type for.

    Returns:
        `the_value` is passed back as return value.
    """
    return the_value


def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    ``torch.jit.script`` has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """
    return _script_if_tracing(fn)


# for torch.jit.isinstance
def isinstance(obj, target_type):
    """
    Provide container type refinement in TorchScript.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    It can refine parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``,
    ``Dict[str, List[torch.Tensor]]``, ``Optional[Tuple[int,str,int]]``. It can also
    refine basic types such as bools and ints that are available in TorchScript.

    Args:
        obj: object to refine the type of
        target_type: type to try to refine obj to
    Returns:
        ``bool``: True if obj was successfully refined to the type of target_type,
            False otherwise with no new type refinement


    Example (using ``torch.jit.isinstance`` for type refinement):
    .. testcode::

        import torch
        from typing import Any, Dict, List

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input: Any): # note the Any type
                if torch.jit.isinstance(input, List[torch.Tensor]):
                    for t in input:
                        y = t.clamp(0, 0.5)
                elif torch.jit.isinstance(input, Dict[str, str]):
                    for val in input.values():
                        print(val)

        m = torch.jit.script(MyModule())
        x = [torch.rand(3,3), torch.rand(4,3)]
        m(x)
        y = {"key1":"val1","key2":"val2"}
        m(y)
    """
    return _isinstance(obj, target_type)


class strict_fusion:
    """
    Give errors if not all nodes have been fused in inference, or symbolically differentiated in training.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    Example:
    Forcing fusion of additions.

    .. code-block:: python

        @torch.jit.script
        def foo(x):
            with torch.jit.strict_fusion():
                return x + x + x

    """

    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            warnings.warn("Only works in script mode", stacklevel=2)

    def __enter__(self):
        pass

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        pass


# Context manager for globally hiding source ranges when printing graphs.
# Note that these functions are exposed to Python as static members of the
# Graph class, so mypy checks need to be skipped.
@contextmanager
def _hide_source_ranges() -> Iterator[None]:
    old_enable_source_ranges = torch._C.Graph.global_print_source_ranges  # type: ignore[attr-defined]
    try:
        torch._C.Graph.set_global_print_source_ranges(False)  # type: ignore[attr-defined]
        yield
    finally:
        torch._C.Graph.set_global_print_source_ranges(old_enable_source_ranges)  # type: ignore[attr-defined]


def enable_onednn_fusion(enabled: bool) -> None:
    """Enable or disables onednn JIT fusion based on the parameter `enabled`.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.
    """
    torch._C._jit_set_llga_enabled(enabled)


def onednn_fusion_enabled():
    """Return whether onednn JIT fusion is enabled.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.
    """
    return torch._C._jit_llga_enabled()


del Any

if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
<|endoftext|># These functions are referenced from the pickle archives produced by
# ScriptModule.save()


# These (`build_*`) functions used to be used by `pickler.cpp` to specify
# the type of the list for certain special types, but now all lists get
# a type attached and restored via `restore_type_tag` below. The legacy
# functions should stick around for backwards-compatibility.


def build_intlist(data: list[int]) -> list[int]:
    return data


def build_tensorlist(data: list[object]) -> list[object]:
    return data


def build_doublelist(data: list[float]) -> list[float]:
    return data


def build_boollist(data: list[bool]) -> list[bool]:
    return data


def build_tensor_from_id(data: int | object) -> int | None:
    if isinstance(data, int):
        # just the id, can't really do anything
        return data
    return None


def restore_type_tag(value: object, type_str: str) -> object:
    # The type_ptr is used by the jit unpickler to restore the full static type
    # to container types like list when they are re-loaded, but this doesn't
    # matter for Python, so just return the plain value
    return value
<|endoftext|># mypy: allow-untyped-defs
from textwrap import dedent
from typing import Any

import torch.jit


def execWrapper(code, glob, loc) -> None:
    exec(code, glob, loc)


def _gen_unsupported_methods_properties():
    tensor_attrs = set(filter(lambda x: x[0] != "_", dir(torch.Tensor)))
    tensor = torch.tensor([2])
    funcs_template = dedent(
        """
    def func(x):
        return x.{op}()
    """
    )

    deprecated_apis = {
        "volatile",
        "resize",
        "reinforce",
        "new",
        "name",
        "map2_",
        "grad_fn",
        "resize_as",
    }
    tensor_attrs = tensor_attrs - deprecated_apis

    properties = []
    methods = []
    sorted_tensor_attrs = sorted(tensor_attrs, key=lambda x: x.lower())
    for attr in sorted_tensor_attrs:
        funcs_str = funcs_template.format(op=attr)
        scope: dict[str, Any] = {}
        execWrapper(funcs_str, globals(), scope)
        try:
            torch.jit.CompilationUnit(funcs_str)
        except Exception as e:
            if "nonexistent attribute" not in repr(e):
                continue
            attr_repr = repr(getattr(tensor, attr))
            if "bound method" in attr_repr or "built-in method" in attr_repr:
                methods.append(attr)
            else:
                properties.append(attr)

    mapped_methods = ("\t*  :meth:`~torch.Tensor." + x + r"`" for x in methods)
    mapped_properties = ("\t*  :attr:`~torch.Tensor." + x + r"`" for x in properties)
    return "\n".join(mapped_methods), "\n".join(mapped_properties)


def _list_unsupported_tensor_ops():
    header = """\n\n
Unsupported Tensor Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    methods, properties = _gen_unsupported_methods_properties()
    return (
        header
        + "\n"
        + methods
        + """

Unsupported Tensor Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
        + "\n"
        + properties
    )


__doc__ = _list_unsupported_tensor_ops()
<|endoftext|># mypy: allow-untyped-defs
"""Serialization.

This module contains functionality for serializing TorchScript modules, notably:
    * torch.jit.save
    * torch.jit.load

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import os
import sys
import warnings

import torch
from torch._jit_internal import _get_model_id
from torch._utils_internal import log_torchscript_usage
from torch.jit._recursive import wrap_cpp_module
from torch.serialization import validate_cuda_device


def save(m, f, _extra_files=None) -> None:
    r"""
    Save an offline version of this module for use in a separate process.

    The saved module serializes all of the methods, submodules, parameters, and
    attributes of this module. It can be loaded into the C++ API using
    ``torch::jit::load(filename)`` or into the Python API with
    :func:`torch.jit.load <torch.jit.load>`.

    To be able to save a module, it must not make any calls to native Python
    functions.  This means that all submodules must be subclasses of
    :class:`ScriptModule` as well.

    .. DANGER::
        All modules, no matter their device, are always loaded onto the CPU
        during loading.  This is different from :func:`torch.load`'s semantics
        and may change in the future.

    Args:
        m: A :class:`ScriptModule` to save.
        f: A file-like object (has to implement write and flush) or a string
           containing a file name.
        _extra_files: Map from filename to contents which will be stored as part of `f`.

    .. note::
        torch.jit.save attempts to preserve the behavior of some operators
        across versions. For example, dividing two integer tensors in
        PyTorch 1.5 performed floor division, and if the module
        containing that code is saved in PyTorch 1.5 and loaded in PyTorch 1.6
        its division behavior will be preserved. The same module saved in
        PyTorch 1.6 will fail to load in PyTorch 1.5, however, since the
        behavior of division changed in 1.6, and 1.5 does not know how to
        replicate the 1.6 behavior.

    Example:
    .. testcode::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        m = torch.jit.script(MyModule())

        # Save to file
        torch.jit.save(m, 'scriptmodule.pt')
        # This line is equivalent to the previous
        m.save("scriptmodule.pt")

        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # Save with extra files
        extra_files = {'foo.txt': b'bar'}
        torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
    """
    if sys.version_info >= (3, 14):
        warnings.warn(
            "`torch.jit.save` is not supported in Python 3.14+ and may break. "
            "Please switch to `torch.export`.",
            FutureWarning,
        )
    else:
        warnings.warn(
            "`torch.jit.save` is deprecated. Please switch to `torch.export`.",
            FutureWarning,
        )
    log_torchscript_usage("save", model_id=_get_model_id(m))
    if _extra_files is None:
        _extra_files = {}
    if isinstance(f, (str, os.PathLike)):
        m.save(f, _extra_files=_extra_files)
    else:
        ret = m.save_to_buffer(_extra_files=_extra_files)
        f.write(ret)


def load(f, map_location=None, _extra_files=None, _restore_shapes=False):
    r"""
    Load a :class:`ScriptModule` or :class:`ScriptFunction` previously saved with :func:`torch.jit.save <torch.jit.save>`.

    All previously saved modules, no matter their device, are first loaded onto CPU,
    and then are moved to the devices they were saved from. If this fails (e.g.
    because the run time system doesn't have certain devices), an exception is
    raised.

    Args:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location (string or torch.device): A simplified version of
            ``map_location`` in `torch.jit.save` used to dynamically remap
            storages to an alternative set of devices.
        _extra_files (dictionary of filename to content): The extra
            filenames given in the map would be loaded and their content
            would be stored in the provided map.
        _restore_shapes (bool): Whether or not to retrace the module on load using stored inputs

    Returns:
        A :class:`ScriptModule` object.

    .. warning::
        It is possible to construct malicious pickle data which will execute arbitrary code
        during :func:`torch.jit.load`. Never load data that could have come from an untrusted
        source, or that could have been tampered with. **Only load data you trust**.

    Example:
    .. testcode::

        import torch
        import io

        torch.jit.load('scriptmodule.pt')

        # Load ScriptModule from io.BytesIO object
        with open('scriptmodule.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load all tensors to the original device
        torch.jit.load(buffer)

        # Load all tensors onto CPU, using a device
        buffer.seek(0)
        torch.jit.load(buffer, map_location=torch.device('cpu'))

        # Load all tensors onto CPU, using a string
        buffer.seek(0)
        torch.jit.load(buffer, map_location='cpu')

        # Load with extra files.
        extra_files = {'foo.txt': ''}  # values will be replaced with data
        torch.jit.load('scriptmodule.pt', _extra_files=extra_files)
        print(extra_files['foo.txt'])

    .. testoutput::
        :hide:

        ...

    .. testcleanup::

        import os
        os.remove("scriptmodule.pt")
    """
    if sys.version_info >= (3, 14):
        warnings.warn(
            "`torch.jit.load` is not supported in Python 3.14+ and may break. "
            "Please switch to `torch.export`.",
            FutureWarning,
        )
    else:
        warnings.warn(
            "`torch.jit.load` is deprecated. Please switch to `torch.export`.",
            FutureWarning,
        )
    if isinstance(f, (str, os.PathLike)):
        if not os.path.exists(f):
            raise ValueError(f"The provided filename {f} does not exist")
        if os.path.isdir(f):
            raise ValueError(f"The provided filename {f} is a directory")

    map_location = validate_map_location(map_location)
    if _extra_files is None:
        _extra_files = {}

    cu = torch._C.CompilationUnit()
    if isinstance(f, (str, os.PathLike)):
        cpp_module = torch._C.import_ir_module(
            cu,
            os.fspath(f),
            map_location,
            _extra_files,
            # pyrefly: ignore [bad-argument-count]
            _restore_shapes,
        )  # type: ignore[call-arg]
    else:
        cpp_module = torch._C.import_ir_module_from_buffer(
            cu,
            f.read(),
            map_location,
            _extra_files,
            # pyrefly: ignore [bad-argument-count]
            _restore_shapes,
        )  # type: ignore[call-arg]

    # TODO: Pretty sure this approach loses ConstSequential status and such
    ret = wrap_cpp_module(cpp_module)
    log_torchscript_usage("load", model_id=_get_model_id(ret))
    return ret


def validate_map_location(map_location=None):
    if isinstance(map_location, str):
        map_location = torch.device(map_location)
    elif not (map_location is None or isinstance(map_location, torch.device)):
        raise ValueError(
            "map_location should be either None, string or torch.device, "
            "but got type: " + str(type(map_location))
        )

    if str(map_location).startswith("cuda"):
        validate_cuda_device(map_location)

    return map_location


def jit_module_from_flatbuffer(f):
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)
        return wrap_cpp_module(torch._C._load_jit_module_from_file(f))
    else:
        return wrap_cpp_module(torch._C._load_jit_module_from_bytes(f.read()))


def save_jit_module_to_flatbuffer(m, f, _extra_files=None) -> None:
    r"""
    Save an offline version of this module for use in a separate process.

    The saved module serializes all of the methods, submodules, parameters, and
    attributes of this module. It can be loaded into the C++ API using
    ``torch::jit::load_jit_module_from_file(filename)`` or into the Python API with
    :func:`torch.jit.jit_module_from_flatbuffer<torch.jit.jit_module_from_flatbuffer>`.

    To be able to save a module, it must not make any calls to native Python
    functions.  This means that all submodules must be subclasses of
    :class:`ScriptModule` as well.

    .. DANGER::
        All modules, no matter their device, are always loaded onto the CPU
        during loading.  This is different from :func:`torch.load`'s semantics
        and may change in the future.

    Args:
        m: A :class:`ScriptModule` to save.
        f: A string for file path


    Example:
    .. testcode::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        m = torch.jit.script(MyModule())

        # Save to file
        torch.jit.save_jit_module_to_flatbuffer(m, 'scriptmodule.ff')
    """
    extra_files = _extra_files
    if extra_files is None:
        extra_files = {}

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)
        torch._C._save_jit_module(m._c, f, extra_files)
    else:
        s = torch._C._save_jit_module_to_bytes(m._c, extra_files)
        f.write(s)


def get_flatbuffer_module_info(path_or_file):
    r"""Get some information regarding a model file in flatbuffer format.

    Args:
        path_or_file: Either str, Path or file like object (BytesIO OK).
            If it's str or Path, we will read the file referenced by that
            path as Bytes.

    Returns:
        A dict with metadata on what that file contains, currently looks like
        this:
        {
            'bytecode_version': 4,  # int
            'operator_version': 4,  # int
            'function_names': {
                '__torch__.___torch_mangle_0.Foo.forward'}, # set
            'type_names': set(),  # set
            'opname_to_num_args': {'aten::linear': 3} # Dict[str, int]
        }
    """
    if isinstance(path_or_file, (str, os.PathLike)):
        with open(path_or_file, "rb") as f:
            all_bytes = f.read()
    else:
        all_bytes = path_or_file.read()
    return torch._C._get_module_info_from_flatbuffer(all_bytes)
<|endoftext|># mypy: allow-untyped-defs
import ast
import copy
import dataclasses
import inspect
import re
import string
from collections import namedtuple
from textwrap import dedent

import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
    Apply,
    Assert,
    Assign,
    Attribute,
    AugAssign,
    BinOp,
    Break,
    ClassDef,
    Const,
    Continue,
    Decl,
    Def,
    Delete,
    DictComp,
    DictLiteral,
    Dots,
    EmptyTypeAnnotation,
    ExprStmt,
    FalseLiteral,
    For,
    Ident,
    If,
    ListComp,
    ListLiteral,
    NoneLiteral,
    Param,
    Pass,
    Property,
    Raise,
    Return,
    Select,
    SliceExpr,
    Starred,
    Stmt,
    StringLiteral,
    Subscript,
    TernaryIf,
    TrueLiteral,
    TupleLiteral,
    UnaryOp,
    Var,
    While,
    With,
    WithItem,
)
from torch._jit_internal import (  # noqa: F401
    _is_drop_fn,
    FunctionModifiers,
    is_static_fn,
    should_drop,
)
from torch._sources import (
    get_source_lines_and_file,
    make_source_context,
    parse_def,
    ParsedDef as _ParsedDef,
)
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace


# Borrowed from cPython implementation
# https://github.com/python/cpython/blob/561612d8456cfab5672c9b445521113b847bd6b3/Lib/textwrap.py#L411#

_reserved_prefix = "__jit"
_reserved_names = {"print"}
_identifier_chars = set(string.ascii_lowercase + string.ascii_uppercase + string.digits)


def is_reserved_name(name):
    return name.startswith(_reserved_prefix) or name in _reserved_names


pretty_node_names = {
    ast.FunctionDef: "function definitions",
    ast.For: "for loops",
    ast.Delete: "del statements",
    ast.ClassDef: "class definitions",
    ast.With: "with statements",
    ast.Raise: "raise statements",
    ast.Assert: "assertions",
    ast.Import: "import statements",
    ast.ImportFrom: "import statements",
    ast.Global: "global variables",
    ast.Break: "break statements",
    ast.Continue: "continue statements",
}

node_start_tokens = {
    ast.FunctionDef: "def",
    ast.For: "for",
    ast.Delete: "del",
    ast.ClassDef: "class",
    ast.With: "with",
    ast.Raise: "raise",
    ast.Assert: "assert",
    ast.Import: "import",
    ast.ImportFrom: "from",
    ast.Global: "global",
    ast.Break: "break",
    ast.Continue: "continue",
}

# pyrefly: ignore [no-matching-overload]
pretty_node_names.update(
    {
        ast.AsyncFunctionDef: "async function definitions",
        ast.AsyncFor: "async for loops",
        ast.AsyncWith: "async with statements",
        ast.Try: "try blocks",
        ast.Nonlocal: "nonlocal variables",
    }
)

# pyrefly: ignore [no-matching-overload]
node_start_tokens.update(
    {
        ast.AsyncFunctionDef: "async def",
        ast.AsyncFor: "async for",
        ast.AsyncWith: "async with",
        ast.Try: "try",
        ast.Nonlocal: "nonlocal",
    }
)

# pyrefly: ignore [no-matching-overload]
pretty_node_names.update(
    {
        ast.AnnAssign: "annotated assignments",
    }
)
# NB: no specific token for AnnAssign


class FrontendError(Exception):
    def __init__(self, source_range, msg) -> None:
        self.source_range = source_range
        self.msg = msg

        # This has to be instantiated here so the ErrorReport is accurate to the
        # call stack when the FrontendError was raised
        self.error_report = torch._C.ErrorReport(self.source_range)

    def __str__(self) -> str:
        return self.msg + self.error_report.what().lstrip()


class NotSupportedError(FrontendError):
    pass


class UnsupportedNodeError(NotSupportedError):
    def __init__(self, ctx, offending_node, reason="") -> None:
        # If we don't have a specific token, we default to length of 1
        node_type = type(offending_node)
        range_len = len(node_start_tokens.get(node_type, " "))
        source_range = ctx.make_range(
            offending_node.lineno,
            offending_node.col_offset,
            offending_node.col_offset + range_len,
        )
        feature_name = pretty_node_names.get(node_type, node_type.__name__)
        msg = f"{feature_name} {reason + ' ' if reason else ''}aren't supported"
        super().__init__(source_range, msg)


class FrontendTypeError(FrontendError):
    pass


def build_withitems(ctx, items):
    items = [build_withitem(ctx, i) for i in items]
    return list(items)


def build_stmts(ctx, stmts):
    stmts = [build_stmt(ctx, s) for s in stmts]
    return list(filter(None, stmts))


def get_class_properties(cls, self_name):
    """
    Get a list of Property objects representing the properties of a class.

    Args:
        cls:  The class to get properties of.
        self_name: The name of the class that the properties should belong to.
    Returns:
        A list of Property objects corresponding to the properties of cls. Property
        here refers to the subclass of TreeView.
    """
    props = inspect.getmembers(cls, predicate=lambda m: isinstance(m, property))
    # Any property that should not be compiled must be in this list on the Module.
    unused_properties = getattr(cls, "__jit_unused_properties__", [])

    # Create Property TreeView objects from inspected property objects.
    properties = []
    for prop in props:
        if prop[0] not in unused_properties and not should_drop(prop[1].fget):
            getter = get_jit_def(
                prop[1].fget, f"__{prop[0]}_getter", self_name=self_name
            )
            setter = (
                get_jit_def(prop[1].fset, f"__{prop[0]}_setter", self_name=self_name)
                if prop[1].fset
                else None
            )
            properties.append(
                Property(getter.range(), Ident(getter.range(), prop[0]), getter, setter)
            )

    return properties


def get_class_assigns(ctx, cls_ast):
    assigns = []

    def maybe_build_assign(builder, entry) -> None:
        nonlocal assigns
        try:
            assigns.append(builder(ctx, entry))
        except NotSupportedError:
            pass

    for entry in cls_ast.body:
        if isinstance(entry, ast.Assign):
            maybe_build_assign(StmtBuilder.build_Assign, entry)
        elif isinstance(entry, ast.AnnAssign):
            maybe_build_assign(StmtBuilder.build_AnnAssign, entry)
    return assigns


def get_jit_class_def(cls, self_name):
    """Get definitions for each method within the current class independently.

    Args:
        cls: The class to get definition of.
        self_name: The name of the class that the properties should belong to.

    Returns:
        torch._C._jit_tree_views.ClassDef: A representation of the class,
            the methods in the class and their definition as a tree.
    """
    # TODO: proper overriding analysis when implementing class inheritance
    methods = inspect.getmembers(
        cls,
        predicate=lambda m: (inspect.ismethod(m) or inspect.isfunction(m))
        and not is_static_fn(cls, m.__name__)
        and m.__name__ in cls.__dict__
        and not _is_drop_fn(m),
    )

    def is_classmethod(fn):
        return inspect.ismethod(fn) and getattr(fn, "__self__", None) == cls

    # Get and parse the source code for this class
    sourcelines, file_lineno, filename = get_source_lines_and_file(
        cls, torch._C.ErrorReport.call_stack()
    )
    source = "".join(sourcelines)

    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)

    class_ast = py_ast.body[0]
    if not isinstance(class_ast, ast.ClassDef):
        raise AssertionError(
            f"Expected class definition, got {type(class_ast).__name__}"
        )

    # Special case for dataclasses. In general we need access to the source code for
    # an object in order to JIT compile it. But the dataclasses module dynamically synthesizes
    # magic methods for classes, and we can't get the source code for these methods. As a
    # workaround, we synthesize TorchScript-friendly implementations ourselves.
    if dataclasses.is_dataclass(cls):
        # Detect whether the user manually implemented any of the magic methods. If they did,
        # we don't want to synthesize/override them.
        overrides = {
            method.name
            for method in class_ast.body
            if isinstance(method, ast.FunctionDef)
            and method.name in DATACLASS_MAGIC_METHODS
        }
        for i, (name, _) in enumerate(methods):
            # Is this a magic method we can synthesize?
            synthesizer_fn = DATACLASS_MAGIC_METHODS.get(name)
            if synthesizer_fn and name not in overrides:
                parsed_def = synthesizer_fn(cls)
                methods[i] = name, parsed_def
                func = getattr(cls, name)
                _jit_internal.loader.cache(func, parsed_def.source)

    method_defs = [
        get_jit_def(obj, name, self_name=self_name, is_classmethod=is_classmethod(obj))
        for (name, obj) in methods
    ]
    properties = get_class_properties(cls, self_name)

    leading_whitespace_len = len(source.split("\n", 1)[0]) - len(
        dedent_src.split("\n", 1)[0]
    )
    ctx = make_source_context(
        source, filename, file_lineno, leading_whitespace_len, False
    )
    assigns = get_class_assigns(ctx, class_ast)

    return build_class_def(ctx, class_ast, method_defs, properties, self_name, assigns)


def get_jit_def(fn, def_name, self_name=None, is_classmethod=False):
    """
    Build a JIT AST (TreeView) from the given function.

    Args:
        fn: A function object to compile or a pre-parsed ParsedDef object
        def_name: The name to give to the resulting AST object. This is not
            always the same as `fn.__name__`, for example:
                def _forward(self):
                    ...
                forward = _forward
            In this case, the `__name__` attribute of the function object is "_forward",
            but we want the result AST to have the name "forward".
        self_name: If this function is a method, what the type name of `self` is.
    """
    parsed_def = parse_def(fn) if not isinstance(fn, _ParsedDef) else fn
    type_line = torch.jit.annotations.get_type_line(parsed_def.source)
    fn_def = parsed_def.ast.body[0]

    if is_classmethod:
        arg_name = fn_def.args.args[0].arg  # type:ignore[union-attr]
        # Insert a statement that assigns the first argument to the class
        assign_stmt = ast.parse(f"{arg_name} = {self_name}").body[0]
        fn_def.body.insert(0, assign_stmt)  # type:ignore[union-attr]

    # Swap out the function signature and body if it is unused
    if should_drop(fn):
        unused_fn_def = ast.parse(
            'def unused_fn(self: Any):\n\traise RuntimeError("Cannot call @unused methods")'
        )
        if len(unused_fn_def.body) != 1 or not isinstance(
            unused_fn_def.body[0], ast.FunctionDef
        ):
            raise RuntimeError(
                f"Expected a single top-level function: {parsed_def.filename}:{parsed_def.file_lineno}"
            )
        unused_def = unused_fn_def.body[0]
        fn_def.body = unused_def.body  # type:ignore[union-attr]
        # kwarg/vararg not supported by `build_def`
        fn_def.args.kwarg = fn_def.args.vararg = None  # type:ignore[union-attr]
        for arg in fn_def.args.args + fn_def.args.kwonlyargs:  # type:ignore[union-attr]
            # Replace potentially unsupported type annotations by "Any"
            arg.annotation = unused_def.args.args[0].annotation
        if _is_drop_fn(fn):
            # Dropping potentially unsupported return type annotation for jit._drop
            fn_def.returns = None  # type:ignore[union-attr]
            fn_def.type_comment = None  # type:ignore[union-attr]

    # If MonkeyType is installed, get all the consolidated type traces
    # for the arguments from type_trace_db
    type_trace_db = torch.jit._script._get_type_trace_db()
    pdt_arg_types = None
    if monkeytype_trace and not isinstance(fn, _ParsedDef):  # type: ignore[truthy-function]
        qualname = get_qualified_name(fn)
        pdt_arg_types = type_trace_db.get_args_types(qualname)

    return build_def(
        parsed_def.ctx,
        fn_def,
        type_line,
        def_name,
        self_name=self_name,
        pdt_arg_types=pdt_arg_types,
    )


# TODO: more robust handling of recognizing ignore context manager
def is_torch_jit_ignore_context_manager(stmt) -> bool:
    # checks if the statement is torch.jit.ignore context manager
    if isinstance(stmt.items[0].context_expr, ast.Call):
        # extract torch part
        function = stmt.items[0].context_expr.func
        if isinstance(function, ast.Attribute):
            attr_name = function.attr
            attr_value = function.value
            if attr_name == "_IgnoreContextManager" and isinstance(
                attr_value, ast.Attribute
            ):
                # there should be at most two nested attributes (e.g torch.jit._IgnoreContextManager)
                if attr_value.attr == "jit" and isinstance(attr_value.value, ast.Name):
                    if attr_value.value.id == "torch":
                        return True
    return False


class Builder:
    def __call__(self, ctx, node):
        method = getattr(self, "build_" + node.__class__.__name__, None)
        if method is None:
            raise UnsupportedNodeError(ctx, node)
        return method(ctx, node)


def build_class_def(ctx, py_def, methods, properties, self_name, assigns):
    r = ctx.make_range(
        py_def.lineno, py_def.col_offset, py_def.col_offset + len("class")
    )
    return ClassDef(
        Ident(r, self_name), [Stmt(method) for method in methods], properties, assigns
    )


def build_def(ctx, py_def, type_line, def_name, self_name=None, pdt_arg_types=None):
    body = py_def.body
    r = ctx.make_range(py_def.lineno, py_def.col_offset, py_def.col_offset + len("def"))

    param_list = build_param_list(ctx, py_def.args, self_name, pdt_arg_types)
    return_type = None
    if getattr(py_def, "returns", None) is not None:
        return_type = build_expr(ctx, py_def.returns)

    decl = Decl(r, param_list, return_type)
    is_method = self_name is not None
    if type_line is not None:
        type_comment_decl = torch._C.parse_type_comment(type_line)
        decl = torch._C.merge_type_from_type_comment(
            decl,  # type: ignore[arg-type]
            type_comment_decl,
            is_method,  # type: ignore[assignment]
        )

    return Def(Ident(r, def_name), decl, build_stmts(ctx, body))


_vararg_kwarg_err = (
    "Compiled functions can't take variable number of arguments "
    "or use keyword-only arguments with defaults"
)


def build_param_list(ctx, py_args, self_name, pdt_arg_types=None):
    if py_args.kwarg is not None:
        expr = py_args.kwarg
        ctx_range = ctx.make_range(
            expr.lineno, expr.col_offset - 1, expr.col_offset + len(expr.arg)
        )
        raise NotSupportedError(ctx_range, _vararg_kwarg_err)
    if py_args.vararg is not None:
        expr = py_args.vararg
        ctx_range = ctx.make_range(
            expr.lineno, expr.col_offset - 1, expr.col_offset + len(expr.arg)
        )
        raise NotSupportedError(ctx_range, _vararg_kwarg_err)
    if len(py_args.kw_defaults) > 0:
        # kw_defaults is a list of the values for the kwargs (which default to None),
        # so they don't actually have line numbers.
        for arg in py_args.kw_defaults:
            if arg is not None:
                ctx_range = build_expr(ctx, arg).range()
                raise NotSupportedError(ctx_range, _vararg_kwarg_err)

    # List of Tuple of args and type as inferred by profile directed typing
    arg_and_types = [
        (
            arg,
            pdt_arg_types[arg.arg]
            if pdt_arg_types and bool(pdt_arg_types[arg.arg])
            else None,
        )
        for arg in py_args.args
    ]
    arg_and_types_kwonlyargs = [
        (
            arg,
            pdt_arg_types[arg.arg]
            if pdt_arg_types and bool(pdt_arg_types[arg.arg])
            else None,
        )
        for arg in py_args.kwonlyargs
    ]

    result = [
        build_param(ctx, arg, self_name, kwarg_only=False, pdt_arg_type=arg_type)
        for arg, arg_type in arg_and_types
    ]
    result += [
        build_param(ctx, arg, self_name, kwarg_only=True, pdt_arg_type=arg_type)
        for arg, arg_type in arg_and_types_kwonlyargs
    ]
    return result


def build_param(ctx, py_arg, self_name, kwarg_only, pdt_arg_type=None):
    # NB: In Python3 py_arg is a pair of (str arg, expr? annotation)
    name = py_arg.arg
    r = ctx.make_range(py_arg.lineno, py_arg.col_offset, py_arg.col_offset + len(name))
    if getattr(py_arg, "annotation", None) is not None:
        annotation_expr = build_expr(ctx, py_arg.annotation)
    elif pdt_arg_type:
        annotation_expr = Var(Ident(r, pdt_arg_type))
    elif self_name is not None and name == "self":
        annotation_expr = Var(Ident(r, self_name))
    else:
        annotation_expr = EmptyTypeAnnotation(r)
    return Param(annotation_expr, Ident(r, name), kwarg_only)


def build_ignore_context_manager(ctx, stmt):
    InputType = namedtuple("InputType", ["name", "ann"])
    OutputType = namedtuple("OutputType", ["name", "ann"])

    def process_ins_outs(args):
        # parse the context manager to figure out inputs and outputs
        # with their annotated types
        # TODO: add input, output validator
        inputs = []
        outputs = []
        for arg in args:
            var_name = arg.arg
            var_ann = arg.value.value
            var_decl_type, var_ann = var_ann.split(":")
            if var_decl_type == "inp":
                inputs.append(InputType(var_name, var_ann))
            if var_decl_type == "out":
                outputs.append(OutputType(var_name, var_ann))
        return inputs, outputs

    def create_unique_name_ext(ctx, stmt) -> str:
        # extension will be based on the full path filename plus
        # the line number of original context manager
        fn = re.sub(r"[^a-zA-Z0-9_]", "_", ctx.filename)
        return f"{fn}_{stmt.lineno}"

    def build_return_ann_stmt(outputs):
        return_type_ann = ""
        return_statement_str = "return "
        if len(outputs) == 0:
            return_type_ann += " -> None"
        if len(outputs) == 1:
            return_type_ann = " -> " + outputs[0].ann
            return_statement_str += outputs[0].name
        if len(outputs) > 1:
            return_type_ann = " -> tuple"
            return_type_ann += "[" + ", ".join([var.ann for var in outputs]) + "]"
            return_statement_str += ", ".join([var.name for var in outputs])
        return return_type_ann, return_statement_str

    def build_args(args):
        return ", ".join([arg.name for arg in args])

    inputs, outputs = process_ins_outs(stmt.items[0].context_expr.keywords)

    # build the replacement function str with given inputs and outputs
    ignore_function_name = "func_ignore_" + create_unique_name_ext(ctx, stmt)
    ignore_function_str = "\ndef " + ignore_function_name
    ignore_function_str += (
        "(" + ", ".join([var.name + " :" + var.ann for var in inputs]) + ")"
    )

    return_ann, return_stmt = build_return_ann_stmt(outputs)
    ignore_function_str += return_ann + ": pass"

    # first create the functionDef object from just declaration
    ignore_function = ast.parse(ignore_function_str).body[0]

    # dump the body of context manager to dummy function
    ignore_function.body = stmt.body  # type: ignore[attr-defined]

    # insert return statement to the function
    return_stmt = ast.parse(return_stmt).body[0]
    ignore_function.body.append(return_stmt)  # type: ignore[attr-defined]

    ignore_func_str = f"""\
# Backward compat: These used to be imported into the outer global scope so some
# code may still expect them.
from typing import List, Dict, Tuple

@torch.jit.ignore
{ast.unparse(ignore_function)}
"""
    g = copy.copy(globals())
    exec(ignore_func_str, g)  # noqa: P204
    # registers the custom function in the global context
    globals()[ignore_function_name] = g[ignore_function_name]

    # build the statements as:
    # <out_1>, <out_2>, ... = torch.jit.frontend.<func>(<in_1>, <in_2>)
    assign_str_lhs = build_args(outputs)
    # this function will be registered in torch.jit.frontend module by default
    assign_str_rhs = (
        f"torch.jit.frontend.{ignore_function_name}(" + build_args(inputs) + ")"
    )

    if len(outputs) > 0:
        assign_str = assign_str_lhs + " = " + assign_str_rhs
    else:
        assign_str = assign_str_rhs
    assign_ast = ast.parse(assign_str).body[0]
    return assign_ast


def get_default_args(fn):
    """
    Get a dictionary of default arguments for a function.

    Args:
        fn: Callable - The function to inspect for default arguments.
    Returns:
        (Dict[str, Any]): mapping argument names to their default values if
        :attr:`fn` is not None, else empty dictionary.
    """
    if fn is None:
        return {}

    signature = inspect.signature(fn)

    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_default_args_for_class(cls):
    """
    Get default arguments for all methods in a class (except for static methods).

    Args:
        cls: type - The class type to inspect for default arguments.
    Returns:
        A Dict[str, Dict[str, Any]] which maps each method name to a Dict[str, Any]
        that maps each argument name to its default value.
    """
    # Get methods (except static methods because those are compiled separately as
    # if they were independent script functions).
    methods = inspect.getmembers(
        cls,
        predicate=lambda m: (inspect.ismethod(m) or inspect.isfunction(m))
        and not is_static_fn(cls, m.__name__)
        and m.__name__ in cls.__dict__,
    )

    # Get method defaults. Property defaults do not need to be considered
    # because setters cannot be invoked without a value.
    defaults = {
        method_name: get_default_args(method_impl)
        for method_name, method_impl in methods
    }

    return defaults


class WithItemBuilder(Builder):
    @staticmethod
    def build_withitem(ctx, item):
        lineno = item.context_expr.lineno
        start = item.context_expr.col_offset
        end = start + len(pretty_node_names[ast.With])
        op_vars = item.optional_vars
        r = ctx.make_range(lineno, start, end)

        return WithItem(
            r,
            build_expr(ctx, item.context_expr),
            build_expr(ctx, op_vars) if op_vars else None,
        )


class StmtBuilder(Builder):
    augassign_map = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Mod: "%",
        ast.BitOr: "|",
        ast.BitAnd: "&",
        ast.BitXor: "^",
        ast.LShift: "<<",
        ast.RShift: ">>",
        ast.Pow: "**",
    }

    @staticmethod
    def build_Expr(ctx, stmt):
        value = stmt.value
        if value.__class__.__name__ == "Str":
            # If a statement is a string literal expression,
            # then it is a docstring. Just ignore it.
            return None
        else:
            return ExprStmt(build_expr(ctx, value))

    @staticmethod
    def build_Assign(ctx, stmt):
        rhs = build_expr(ctx, stmt.value)
        lhs = [build_expr(ctx, x) for x in stmt.targets]
        return Assign(lhs, rhs)

    @staticmethod
    def build_AnnAssign(ctx, stmt):
        if stmt.value is None:
            raise UnsupportedNodeError(ctx, stmt, reason="without assigned value")

        # Disallow type annotations on instance attributes outside of __init__
        if (
            type(stmt.target) is ast.Attribute
            and stmt.target.value.id == "self"  # type: ignore[attr-defined]
            and ctx.funcname != "__init__"
        ):
            start = stmt.col_offset
            end = start + len(f"self.{stmt.target.attr}")
            if hasattr(stmt.annotation, "id"):
                end += len(f": {stmt.annotation.id}")
            sr = ctx.make_range(stmt.lineno, start, end)
            raise ValueError(
                "Type annotations on instance attributes must be declared in "
                f"__init__, not '{ctx.funcname}': {sr}"
            )

        rhs = build_expr(ctx, stmt.value)
        lhs = build_expr(ctx, stmt.target)
        the_type = build_expr(ctx, stmt.annotation)
        return Assign([lhs], rhs, the_type)

    @staticmethod
    def build_Delete(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("del"))

        return Delete(r, [build_expr(ctx, target) for target in stmt.targets])

    @staticmethod
    def build_Return(ctx, stmt):
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("return")
        )
        return Return(r, None if stmt.value is None else build_expr(ctx, stmt.value))

    @staticmethod
    def build_Raise(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("raise"))
        expr = build_expr(ctx, stmt.exc)
        return Raise(r, expr)

    @staticmethod
    def build_Assert(ctx, stmt):
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("assert")
        )
        test = build_expr(ctx, stmt.test)
        msg = build_expr(ctx, stmt.msg) if stmt.msg is not None else None
        return Assert(r, test, msg)

    @staticmethod
    def build_AugAssign(ctx, stmt):
        lhs = build_expr(ctx, stmt.target)
        rhs = build_expr(ctx, stmt.value)
        op = type(stmt.op)
        if op in StmtBuilder.augassign_map:
            op_token = StmtBuilder.augassign_map[op]
        else:
            raise NotSupportedError(
                find_before(ctx, rhs.range().start, "=", offsets=(-1, 0)),
                "unsupported kind of augmented assignment: " + op.__name__,
            )
        return AugAssign(lhs, op_token, rhs)

    @staticmethod
    def build_While(ctx, stmt):
        if stmt.orelse:
            # TODO: try to recover the location of else:? Python doesn't give us useful
            # annotations in this case
            raise NotSupportedError(
                None, "else branches of while loops aren't supported"
            )
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("while"))
        return While(r, build_expr(ctx, stmt.test), build_stmts(ctx, stmt.body))

    @staticmethod
    def build_For(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("for"))
        if stmt.orelse:
            raise NotSupportedError(r, "else branches of for loops aren't supported")

        return For(
            r,
            [build_expr(ctx, stmt.target)],
            [build_expr(ctx, stmt.iter)],
            build_stmts(ctx, stmt.body),
        )

    @staticmethod
    def build_If(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("if"))
        return If(
            r,
            build_expr(ctx, stmt.test),
            build_stmts(ctx, stmt.body),
            build_stmts(ctx, stmt.orelse),
        )

    @staticmethod
    def build_Print(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("print"))
        if stmt.dest:
            raise NotSupportedError(
                r, "print statements with non-default destinations aren't supported"
            )
        args = [build_expr(ctx, val) for val in stmt.values]
        return ExprStmt(Apply(Var(Ident(r, "print")), args, []))

    @staticmethod
    def build_Pass(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("pass"))
        return Pass(r)

    @staticmethod
    def build_Break(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("break"))
        return Break(r)

    @staticmethod
    def build_Continue(ctx, stmt):
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("continue")
        )
        return Continue(r)

    @staticmethod
    def build_With(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("with"))
        # Handle ignore context manager
        if is_torch_jit_ignore_context_manager(stmt):
            assign_ast = build_ignore_context_manager(ctx, stmt)
            return build_stmt(ctx, assign_ast)
        return With(r, build_withitems(ctx, stmt.items), build_stmts(ctx, stmt.body))


class ExprBuilder(Builder):
    binop_map = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Pow: "**",
        ast.Mod: "%",
        ast.FloorDiv: "//",
        ast.BitAnd: "&",
        ast.BitXor: "^",
        ast.BitOr: "|",
        ast.LShift: "<<",
        ast.RShift: ">>",
    }

    # pyrefly: ignore [unsupported-operation]
    binop_map[ast.MatMult] = "@"

    unop_map = {
        ast.Not: "not",
        ast.USub: "-",
        ast.Invert: "~",
    }

    boolop_map = {
        ast.And: "and",
        ast.Or: "or",
    }

    cmpop_map = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.LtE: "<=",
        ast.Lt: "<",
        ast.GtE: ">=",
        ast.Gt: ">",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.In: "in",
        ast.NotIn: "not in",
    }

    @staticmethod
    def build_Attribute(ctx, expr):
        base = build_expr(ctx, expr.value)
        # expr.attr is just a string, so it's not annotated in any way, so we have
        # to build the range manually
        source = ctx.source.encode("utf-8")

        def get_char(index):
            return chr(source[index])

        start_pos = base.range().end + 1
        while get_char(start_pos) in string.whitespace:  # Skip whitespace
            start_pos += 1
        end_pos = start_pos + len(expr.attr)
        name_range = ctx.make_raw_range(start_pos, end_pos)
        return Select(base, Ident(name_range, expr.attr))

    @staticmethod
    def build_Call(ctx, expr):
        func = build_expr(ctx, expr.func)
        args = [build_expr(ctx, py_arg) for py_arg in expr.args]
        if hasattr(expr, "starargs") and expr.starargs:
            stararg_expr = build_expr(ctx, expr.starargs)
            args += [Starred(stararg_expr.range(), stararg_expr)]
        kwargs = []
        for kw in expr.keywords:
            kw_expr = build_expr(ctx, kw.value)
            # XXX: we could do a better job at figuring out the range for the name here
            if not kw.arg:
                raise NotSupportedError(
                    kw_expr.range(), "keyword-arg expansion is not supported"
                )
            kwargs.append(Attribute(Ident(kw_expr.range(), kw.arg), kw_expr))
        return Apply(func, args, kwargs)

    @staticmethod
    def build_Ellipsis(ctx, expr):
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + 3
        )  # len("...") == 3
        return Dots(r)

    @staticmethod
    def build_Name(ctx, expr):
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(expr.id))
        if expr.id.startswith(_reserved_prefix):
            raise NotSupportedError(
                r,
                "names of variables used in JIT-ed functions "
                "can't start with " + _reserved_prefix,
            )
        if expr.id == "True":
            return TrueLiteral(r)
        elif expr.id == "False":
            return FalseLiteral(r)
        elif expr.id == "None":
            return NoneLiteral(r)
        elif expr.id == "Ellipsis":
            return Dots(r)
        return Var(Ident(r, expr.id))

    @staticmethod
    def build_NameConstant(ctx, expr):
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(str(expr.value))
        )
        if expr.value is True:
            return TrueLiteral(r)
        elif expr.value is False:
            return FalseLiteral(r)
        elif expr.value is None:
            return NoneLiteral(r)
        elif expr.value == Ellipsis:
            return Dots(r)
        else:
            raise ValueError("Name constant value unsupported: " + str(expr.value))

    @staticmethod
    def build_BinOp(ctx, expr):
        lhs = build_expr(ctx, expr.left)
        rhs = build_expr(ctx, expr.right)
        op = type(expr.op)

        if op == ast.Div and not ctx.uses_true_division:
            err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            raise FrontendError(
                err_range,
                "Division of ints in TorchScript uses Python 3 true "
                "division semantics. Please put `from __future__ "
                "import division` at the top of your file",
            )
        op_token = ExprBuilder.binop_map.get(op)
        if op_token is None:
            err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            raise NotSupportedError(
                err_range, "unsupported binary operator: " + op.__name__
            )
        return BinOp(op_token, lhs, rhs)

    @staticmethod
    def build_UnaryOp(ctx, expr):
        sub_expr = build_expr(ctx, expr.operand)
        op = type(expr.op)
        op_token = ExprBuilder.unop_map.get(op)
        if op_token is None:
            raise NotSupportedError(
                expr.range(), "unsupported unary operator: " + op.__name__
            )
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(op_token)
        )
        return UnaryOp(r, op_token, sub_expr)

    @staticmethod
    def build_BoolOp(ctx, expr):
        if len(expr.values) < 2:
            raise AssertionError(
                "expected at least 2 values in BoolOp, but got " + str(len(expr.values))
            )
        sub_exprs = [build_expr(ctx, sub_expr) for sub_expr in expr.values]
        op = type(expr.op)
        op_token = ExprBuilder.boolop_map.get(op)
        if op_token is None:
            err_range = ctx.make_raw_range(
                sub_exprs[0].range().end, sub_exprs[1].range().start
            )
            raise NotSupportedError(
                err_range, "unsupported boolean operator: " + op.__name__
            )
        lhs = sub_exprs[0]
        for rhs in sub_exprs[1:]:
            lhs = BinOp(op_token, lhs, rhs)
        return lhs

    @staticmethod
    def build_IfExp(ctx, expr):
        return TernaryIf(
            build_expr(ctx, expr.test),
            build_expr(ctx, expr.body),
            build_expr(ctx, expr.orelse),
        )

    @staticmethod
    def build_Compare(ctx, expr):
        operands = [build_expr(ctx, e) for e in [expr.left] + list(expr.comparators)]
        result = None
        # pyrefly: ignore [bad-assignment]
        for lhs, op_, rhs in zip(operands, expr.ops, operands[1:]):
            op = type(op_)
            op_token = ExprBuilder.cmpop_map.get(op)
            r = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            if op_token is None:
                raise NotSupportedError(
                    r, "unsupported comparison operator: " + op.__name__
                )

            if op == ast.NotIn:
                # NB: `not in` is just `not( in )`, so we don't introduce new tree view
                # but just make it a nested call in our tree view structure
                in_expr = BinOp("in", lhs, rhs)
                cmp_expr = UnaryOp(r, "not", in_expr)
            else:
                cmp_expr = BinOp(op_token, lhs, rhs)  # type: ignore[assignment]

            if result is None:
                result = cmp_expr
            else:
                result = BinOp("and", result, cmp_expr)  # type: ignore[assignment]
        return result

    @staticmethod
    def build_Subscript(ctx, expr):
        def build_SliceExpr(ctx, base, slice_expr):
            lower = (
                build_expr(ctx, slice_expr.lower)
                if slice_expr.lower is not None
                else None
            )
            upper = (
                build_expr(ctx, slice_expr.upper)
                if slice_expr.upper is not None
                else None
            )
            step = (
                build_expr(ctx, slice_expr.step)
                if slice_expr.step is not None
                else None
            )
            return SliceExpr(base.range(), lower, upper, step)

        def build_Index(ctx, base, index_expr):
            if isinstance(index_expr.value, ast.Tuple):
                raise NotSupportedError(
                    base.range(),
                    "slicing multiple dimensions with tuples not supported yet",
                )
            return build_expr(ctx, index_expr.value)

        def build_ExtSlice(ctx, base, extslice):
            sub_exprs = []
            for expr in extslice.dims:
                sub_type = type(expr)
                if sub_type is ast.Index:
                    sub_exprs.append(build_Index(ctx, base, expr))
                elif sub_type is ast.Slice:
                    sub_exprs.append(build_SliceExpr(ctx, base, expr))
                elif sub_type is ast.Constant and expr.value is Ellipsis:
                    sub_exprs.append(Dots(base.range()))
                else:
                    raise NotSupportedError(
                        base.range(),
                        f"slicing multiple dimensions with {sub_type} not supported",
                    )
            return sub_exprs

        base = build_expr(ctx, expr.value)
        sub_type = type(expr.slice)
        if sub_type is ast.Index:
            if isinstance(expr.slice.value, ast.Tuple):
                # N-dimensional indexing using Tuple: x[(i, j, k)] is equivalent to x[i, j, k]
                # XXX: Indexing using a list is **different**! It triggers advanced indexing.
                indices = [
                    build_expr(ctx, index_expr) for index_expr in expr.slice.value.elts
                ]
                if not indices:
                    # `col_offset` is an int, but `end_col_offset` is
                    # `Optional[int]`. The magic number is here to make
                    # sure we can parse `()` on any machine
                    r = ctx.make_range(
                        expr.lineno,
                        expr.slice.value.col_offset,
                        expr.slice.value.col_offset + 2,
                    )
                    tup = TupleLiteral(r, [])
                    indices.append(tup)
                return Subscript(base, indices)
            else:
                return Subscript(base, [build_expr(ctx, expr.slice.value)])
        elif sub_type is ast.Slice:
            return Subscript(base, [build_SliceExpr(ctx, base, expr.slice)])
        elif sub_type is ast.ExtSlice:
            return Subscript(base, build_ExtSlice(ctx, base, expr.slice))
        else:  # In Python3.9 array indices are not wrapped in ast.Index
            if sub_type is ast.Tuple:
                # N-dimensional indexing using Tuple: x[(i, j, k)] is equivalent to x[i, j, k]
                indices = []
                for index_expr in expr.slice.elts:
                    if isinstance(index_expr, ast.Slice):
                        indices.append(build_SliceExpr(ctx, base, index_expr))
                    else:
                        indices.append(build_expr(ctx, index_expr))
                # Special-case logic for `typing.Tuple[()]`
                if not indices:
                    # See note above r.e. magic number
                    r = ctx.make_range(
                        expr.lineno, expr.slice.col_offset, expr.slice.col_offset + 2
                    )
                    tup = TupleLiteral(r, [])
                    indices.append(tup)
                return Subscript(base, indices)
            return Subscript(base, [build_expr(ctx, expr.slice)])

    @staticmethod
    def build_List(ctx, expr):
        return ListLiteral(
            ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1),
            [build_expr(ctx, e) for e in expr.elts],
        )

    @staticmethod
    def build_Tuple(ctx, expr):
        return TupleLiteral(
            ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1),
            [build_expr(ctx, e) for e in expr.elts],
        )

    @staticmethod
    def build_Dict(ctx, expr):
        range = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        if expr.keys and not expr.keys[0]:
            raise NotSupportedError(
                range, "Dict expansion (e.g. `{**dict}`) is not supported"
            )
        return DictLiteral(
            range,
            [build_expr(ctx, e) for e in expr.keys],
            [build_expr(ctx, e) for e in expr.values],
        )

    @staticmethod
    def build_Num(ctx, expr):
        value = str(expr.value)
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(value))
        return Const(r, value)

    @staticmethod
    def build_Constant(ctx, expr):
        value = expr.value
        if value is None or isinstance(value, bool):
            # NB: this check has to happen before the int check because bool is
            # a subclass of int
            return ExprBuilder.build_NameConstant(ctx, expr)
        if isinstance(value, (int, float, complex)):
            return ExprBuilder.build_Num(ctx, expr)
        elif isinstance(value, str):
            return ExprBuilder.build_Str(ctx, expr)
        elif isinstance(value, type(Ellipsis)):
            return ExprBuilder.build_Ellipsis(ctx, expr)
        else:
            error_range = ctx.make_range(
                expr.lineno, expr.col_offset, expr.col_offset + len(str(value))
            )
            raise FrontendError(error_range, "Unknown Constant expression type")

    @staticmethod
    def build_Str(ctx, expr):
        value = str(expr.value)
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(value) + 1
        )
        return StringLiteral(r, value)

    @staticmethod
    def build_JoinedStr(ctx, expr):
        s = ""
        args = []
        for value in expr.values:
            r = ctx.make_range(value.lineno, value.col_offset, value.col_offset + 1)
            if isinstance(value, ast.FormattedValue):
                if value.conversion != -1:
                    raise NotSupportedError(r, "Don't support conversion in JoinedStr")
                if value.format_spec is not None:
                    raise NotSupportedError(r, "Don't support formatting in JoinedStr")
                s += "{}"
                args.append(build_expr(ctx, value.value))
            elif isinstance(value, ast.Constant):
                # pyrefly: ignore [unsupported-operation]
                s += value.value
            else:
                raise NotSupportedError(r, "Unsupported value in JoinedStr")

        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        return Apply(Select(StringLiteral(r, s), Ident(r, "format")), args, [])

    @staticmethod
    def build_ListComp(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset)
        if len(stmt.generators) != 1:
            raise NotSupportedError(r, "Only a single generator is currently supported")

        if len(stmt.generators[0].ifs) != 0:
            raise NotSupportedError(r, "Comprehension ifs are not supported yet")

        elt_expr = build_expr(ctx, stmt.elt)
        target_expr = build_expr(ctx, stmt.generators[0].target)
        iter_expr = build_expr(ctx, stmt.generators[0].iter)

        return ListComp(r, elt_expr, target_expr, iter_expr)

    @staticmethod
    def build_GeneratorExp(ctx, stmt):
        # Convert Generator expression to ListComp
        return ExprBuilder.build_ListComp(ctx, stmt)

    @staticmethod
    def build_DictComp(ctx, stmt):
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset)
        if len(stmt.generators) != 1:
            raise NotSupportedError(r, "Only a single generator is currently supported")

        if len(stmt.generators[0].ifs) != 0:
            raise NotSupportedError(r, "Comprehension ifs are not supported yet")

        key_expr = build_expr(ctx, stmt.key)
        value_expr = build_expr(ctx, stmt.value)
        target_expr = build_expr(ctx, stmt.generators[0].target)
        iter_expr = build_expr(ctx, stmt.generators[0].iter)

        return DictComp(r, key_expr, value_expr, target_expr, iter_expr)

    @staticmethod
    def build_Starred(ctx, expr):
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        return Starred(r, build_expr(ctx, expr.value))


build_expr = ExprBuilder()
build_stmt = StmtBuilder()
build_withitem = WithItemBuilder()


def find_before(ctx, pos, substr, offsets=(0, 0)):
    new_pos = ctx.source[:pos].rindex(substr)
    return ctx.make_raw_range(new_pos + offsets[0], new_pos + len(substr) + offsets[1])
<|endoftext|># mypy: allow-untyped-defs
"""Tracing.

This module contains functionality to support the JIT's tracing frontend, notably:
    * torch.jit.trace
    * torch.jit.trace_module

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import contextlib
import copy
import functools
import inspect
import os
import re
import sys
import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._jit_internal import (
    _get_model_id,
    _qualified_name,
    get_callable_argument_names,
    is_scripting,
)
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances


_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten

R = TypeVar("R", covariant=True)  # return type (always covariant)
P = ParamSpec("P")


def _create_interpreter_name_lookup_fn(frames_up=1):
    def _get_interpreter_name_for_var(var):
        frame = inspect.currentframe()
        if not frame:
            raise RuntimeError("failed to inspect frame")

        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            if not frame:
                raise RuntimeError("failed to get frame")
            i += 1

        f_locals = frame.f_locals

        for k, v in f_locals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != "self" else ""
        return ""

    return _get_interpreter_name_for_var


def _unique_state_dict(module, keep_vars=False):
    # since Parameter.detach() always creates a new torch.Tensor instance,
    # id(v) doesn't work with it. So we always get the Parameter or Buffer
    # as values, and deduplicate the params using Parameters and Buffers
    state_dict = module.state_dict(keep_vars=True)
    filtered_dict = type(state_dict)()
    seen_ids: set[int] = set()
    for k, v in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        if keep_vars:
            filtered_dict[k] = v
        else:
            filtered_dict[k] = v.detach()
    return filtered_dict


class ONNXTracedModule(torch.nn.Module):
    def __init__(
        self,
        inner,
        strict=True,
        force_outplace=False,
        return_inputs=False,
        return_inputs_states=False,
    ):
        super().__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner
        self.strict = strict
        self._force_outplace = force_outplace
        self._return_inputs = return_inputs
        self._return_inputs_states = return_inputs_states

    def forward(self, *args: torch.Tensor):
        in_vars, in_desc = _flatten(args)
        # NOTE: use full state, because we need it for BatchNorm export
        # This differs from the compiler path, which doesn't support it at the moment.
        module_state = list(_unique_state_dict(self, keep_vars=True).values())

        ret_inputs = []
        inputs_states = []
        outs = []

        def wrapper(*args):
            in_args: list[torch.Tensor] = []
            for i in range(len(in_vars)):
                if not isinstance(args[i], torch.Tensor):
                    raise RuntimeError("Expected Tensor argument")
                in_args.append(args[i])

            trace_inputs = _unflatten(in_args, in_desc)

            if self._return_inputs:
                ret_inputs.append(
                    tuple(x.clone(memory_format=torch.preserve_format) for x in args)
                )
            if self._return_inputs_states:
                inputs_states.append(_unflatten(in_args, in_desc))
            outs.append(self.inner(*trace_inputs))
            if self._return_inputs_states:
                inputs_states[0] = (inputs_states[0], trace_inputs)
            out_vars, _ = _flatten(outs)
            if len(out_vars) == 1:
                return out_vars[0]
            else:
                return tuple(out_vars)

        graph, _out = torch._C._create_graph_by_tracing(
            wrapper,
            in_vars + module_state,
            _create_interpreter_name_lookup_fn(),
            self.strict,
            self._force_outplace,
        )

        if self._return_inputs:
            return graph, outs[0], ret_inputs[0]
        if self._return_inputs_states:
            return graph, outs[0], inputs_states[0]
        else:
            return graph, outs[0]


def _clone_inputs(args):
    def clone_input(a):
        if a is None:
            return None
        elif isinstance(a, torch.Tensor):
            # TODO: figure out one liner to .clone() and set requires_grad
            v = (
                a.detach()
                .clone(memory_format=None if a.is_mkldnn else torch.preserve_format)
                .requires_grad_(a.requires_grad)
            )
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            return a.clone(memory_format=torch.preserve_format)

    # pyrefly: ignore [missing-attribute]
    return function._nested_map(
        lambda x: isinstance(x, torch.Tensor), clone_input, condition_msg="tensors"
    )(args)


# This is purely for developer debugging.  We are not going to advertise it.
_JIT_TIME = os.environ.get("PYTORCH_JIT_TIME", False)  # CUDA-only timing
_JIT_DISABLE = os.environ.get("PYTORCH_JIT_DISABLE", False)
_JIT_STATS = os.environ.get("PYTORCH_JIT_STATS", False)


@contextlib.contextmanager
def _time(trace_name, name, time=True):
    if (not _JIT_TIME and not time) or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    try:
        yield
    finally:
        stream.record_event(end)
        end.synchronize()
        print(f"{trace_name} {name} time: {start.elapsed_time(end)} ms")


def verify(model, args, loss_fn=torch.sum, devices=None):
    """
    Verify that a JIT compiled model has the same behavior as its uncompiled version along with its backwards pass.

    If your model returns multiple outputs,
    you must also specify a `loss_fn` to produce a loss for which
    the backwards will be computed.

    This function has side-effects (e.g., it executes your model / saves and loads
    parameters), so don't expect the model to come out exactly the same as what
    you passed in.

    Args:
        model (compiled torch.nn.Module or function): the module/function to be
            verified.  The module/function definition MUST have been decorated with
            `@torch.jit.compile`.
        args (tuple or Tensor): the positional arguments to pass to the
            compiled function/module to be verified.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        loss_fn (function, optional): the loss function to be applied to
            the output of the model, before backwards is invoked.  By default,
            we assume that a model returns a single result, and we :func:`torch.sum`
            before calling backwards; if this is inappropriate, you can pass your
            own loss function.  Note that if a model returns a tuple of results,
            these are passed as separate positional arguments to `loss_fn`.
        devices (iterable of device IDs, optional): the GPU devices which the
            compiled module will be run on.  This determines the RNG state we
            must save when running both compiled and uncompiled versions of the model.
    """
    # TODO: In principle, we track device information in our trace, so it
    # should be possible to check if our execution actually obeyed the 'devices'
    # the user provided.

    # TODO: Consider adding a utility function to torch.jit to test
    # for this case
    if not isinstance(model, torch._C.CompiledFunction):  # type: ignore[attr-defined]
        raise TypeError(
            "Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it"
        )
    is_module = isinstance(model, Module)

    if not isinstance(args, tuple):
        args = (args,)

    if is_module:
        saved_state = copy.deepcopy(model.state_dict())

    def run_fwd_bwd(args, force_trace=False, assert_compiled=False):
        params = list(model.parameters()) if is_module else []
        in_vars, _ = _flatten((args, params))
        # We use a special API to reset the trace and compile it from scratch.
        compiled_fn = model
        if force_trace:
            compiled_fn.clear_cache()
        if assert_compiled:
            hits = compiled_fn.hits
        out = model(*args)
        if assert_compiled and compiled_fn.hits == hits:  # type: ignore[possibly-undefined]
            raise RuntimeError("failed to use the compiled function")
        if not isinstance(out, tuple):
            out = (out,)
        if loss_fn == torch.sum and len(out) != 1:
            raise ValueError(
                f"Model returns {len(out)} outputs, but default loss function "
                "(torch.sum) can only handle a single output"
            )
        out_vars, _ = _flatten(out)
        saved_outs = [
            v.detach().clone(memory_format=torch.preserve_format) for v in out_vars
        ]
        loss = loss_fn(*out)
        grads = torch.autograd.grad([loss], in_vars)
        # TODO: I'm not sure if the clone here is necessary but it is safer
        saved_grads = [
            v.detach().clone(memory_format=torch.preserve_format) for v in grads
        ]
        return (saved_outs, saved_grads)

    with torch.random.fork_rng(devices, _caller="torch.jit.verify"):
        uncompiled_outs, uncompiled_grads = run_fwd_bwd(args, force_trace=True)
        if not model.has_trace_for(*args):
            raise AssertionError("Model should have trace for the given args")

    if is_module:
        model.load_state_dict(saved_state)  # type: ignore[possibly-undefined]
    compiled_outs, compiled_grads = run_fwd_bwd(args, assert_compiled=True)

    _verify_equal(uncompiled_outs, compiled_outs)
    _verify_equal(uncompiled_grads, compiled_grads)


def _verify_equal(xs, ys):
    for x, y in zip(xs, ys):
        if x.sub(y).abs().max() > 1e-6:
            raise RuntimeError("JIT and real computation mismatch")


def indent(s):
    return "\n".join(["\t" + line for line in s.splitlines()])


class TracingCheckError(Exception):
    def __init__(self, graph_diff_error, tensor_compare_error, extra_msg=None):
        self.message = "Tracing failed sanity checks!\n"
        if extra_msg is not None:
            self.message += extra_msg + "\n"
        if graph_diff_error is not None:
            self.message += "ERROR: Graphs differed across invocations!\n"
            self.message += indent(graph_diff_error) + "\n"
        if tensor_compare_error is not None:
            self.message += (
                "ERROR: Tensor-valued Constant nodes differed in value "
                "across invocations. This often indicates that the tracer has"
                " encountered untraceable code.\n"
            )
            self.message += indent(tensor_compare_error) + "\n"
        super().__init__(self.message)


# Check the traced module against a set of user-provided validation inputs
@torch.no_grad()
def _check_trace(
    check_inputs,
    func,
    traced_func,
    check_tolerance,
    strict,
    force_outplace,
    is_trace_module,
    _module_class,
    example_inputs_is_kwarg=False,
):
    # Note: tracing is independent of optimizations, which consume the trace
    for inputs in check_inputs:
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)

        if is_trace_module:
            copied_dict = {}

            for name, data in inputs.items():
                copied_dict[name] = _clone_inputs(data)
            check_mod = torch.jit.trace_module(
                getattr(func, "__self__", func),
                copied_dict,
                check_trace=False,
                strict=strict,
                _force_outplace=force_outplace,
                _module_class=_module_class,
                _compilation_unit=torch._C.CompilationUnit(),
                example_inputs_is_kwarg=example_inputs_is_kwarg,
                _store_inputs=False,
            )
            check_mod_func = check_mod._c._get_method(traced_func.name)
            inputs = inputs[traced_func.name]
            if (
                isinstance(inputs, (torch.Tensor))
                or isinstance(inputs, dict)
                and not example_inputs_is_kwarg
            ):
                inputs = (inputs,)
        else:
            if example_inputs_is_kwarg:
                check_mod = torch.jit.trace(
                    func,
                    check_trace=False,
                    strict=strict,
                    _force_outplace=force_outplace,
                    _module_class=_module_class,
                    example_kwarg_inputs=_clone_inputs(inputs),
                    _store_inputs=False,
                )
            else:
                check_mod = torch.jit.trace(
                    func,
                    _clone_inputs(inputs),
                    check_trace=False,
                    strict=strict,
                    _force_outplace=force_outplace,
                    _module_class=_module_class,
                    _store_inputs=False,
                )
            check_mod_func = check_mod

        def graph_diagnostic_info():
            mod_canonicalized = torch._C._jit_pass_canonicalize(traced_func.graph)
            torch._C._jit_pass_inline(mod_canonicalized)
            torch._C._jit_pass_erase_shape_information(mod_canonicalized)
            mod_str = str(mod_canonicalized)
            mod_str = re.sub(r"___torch_mangle_[0-9]+\.", "", mod_str)
            check_canonicalized = torch._C._jit_pass_canonicalize(check_mod_func.graph)
            torch._C._jit_pass_inline(check_canonicalized)
            torch._C._jit_pass_erase_shape_information(check_canonicalized)
            check_str = str(check_canonicalized)
            check_str = re.sub(r"___torch_mangle_[0-9]+\.", "", check_str)

            graph_diff_errors = None
            if mod_str != check_str:
                import difflib

                graph_diff = difflib.ndiff(
                    mod_str.splitlines(True), check_str.splitlines(True)
                )
                graph_diff_errors = "Graph diff:\n" + indent("".join(graph_diff)) + "\n"

                for n_mod, n_check in zip(
                    mod_canonicalized.nodes(), check_canonicalized.nodes()
                ):
                    if str(n_mod) != str(n_check):
                        graph_diff_errors += "First diverging operator:\n"
                        node_diff = difflib.ndiff(
                            str(n_mod).splitlines(True), str(n_check).splitlines(True)
                        )
                        source_printout = (
                            "Node diff:\n" + indent("".join(node_diff)) + "\n"
                        )
                        mod_stack = n_mod.sourceRange()
                        if mod_stack:
                            source_printout += (
                                "Trace source location:\n" + indent(mod_stack) + "\n"
                            )
                        check_stack = n_check.sourceRange()
                        if check_stack:
                            source_printout += (
                                "Check source location:\n" + indent(check_stack) + "\n"
                            )
                        graph_diff_errors += source_printout

                        break  # For now, only print out the first pair of nodes that diverges

            tensor_compare_errors = None
            # Check Tensor-valued constant nodes
            for n_mod, n_check in zip(
                mod_canonicalized.nodes(), check_canonicalized.nodes()
            ):
                if n_mod.kind() != n_check.kind():
                    break  # Graphs have already diverged

                if n_mod.kind() == "prim::Constant" and not (
                    n_mod.mustBeNone() or n_check.mustBeNone()
                ):
                    if not n_mod.hasAttribute("value"):
                        continue
                    if n_mod.kindOf("value") != "t" or n_check.kindOf("value") != "t":
                        continue

                    mod_tensor_val = n_mod.t("value")
                    check_tensor_val = n_check.t("value")

                    try:
                        torch.testing.assert_close(
                            mod_tensor_val, check_tensor_val, equal_nan=True
                        )
                    except (RuntimeError, AssertionError) as e:
                        if tensor_compare_errors is None:
                            tensor_compare_errors = ""
                        tensor_compare_errors += "Node:\n" + indent(str(n_mod)) + "\n"
                        compare_stack = n_mod.sourceRange()
                        if compare_stack:
                            tensor_compare_errors += (
                                "Source Location:\n" + indent(compare_stack) + "\n"
                            )
                        tensor_compare_errors += "Comparison exception: " + indent(
                            str(e)
                        )

                        break  # For now, only print the first diverging pair

            return graph_diff_errors, tensor_compare_errors

        def wrap_retval(x):
            return x if isinstance(x, tuple) else (x,)

        def run_mod_and_filter_tensor_outputs(mod, inputs, running_what):
            try:
                if isinstance(inputs, dict) and example_inputs_is_kwarg:
                    outs = wrap_retval(mod(**inputs))
                else:
                    outs = wrap_retval(mod(*_clone_inputs(inputs)))
                outs = [out for out in outs if isinstance(out, torch.Tensor)]
                return outs
            except Exception as e:
                graph_diff_errors, tensor_compare_errors = graph_diagnostic_info()
                msg = f"encountered an exception while running the {running_what} with test inputs.\nException:\n{indent(str(e))}"
                raise TracingCheckError(
                    graph_diff_errors,
                    tensor_compare_errors,
                    extra_msg=msg,
                ) from e

        has_warned = [False]

        def maybe_warn_nondeterministic():
            if has_warned[0]:
                return
            has_warned[0] = True
            nondeterm_ops = [
                op for op in traced_func.graph.nodes() if op.isNondeterministic()
            ]
            if len(nondeterm_ops) > 0:
                nondeterministic_ops_warning = "Trace had nondeterministic nodes. "
                nondeterministic_ops_warning += (
                    "Did you forget to call .eval() on your model? Nodes:\n"
                )
                nondeterministic_ops_warning += "\n".join(
                    [indent(str(op)) for op in nondeterm_ops][:20]
                )
                nondeterministic_ops_warning += (
                    "\nThis may cause errors in trace checking. To disable trace checking,"
                    " pass check_trace=False to torch.jit.trace()"
                )
                warnings.warn(
                    nondeterministic_ops_warning, category=TracerWarning, stacklevel=5
                )

        def compare_outputs(original, reference, match_what):
            all_ok = True
            for i, (orig, ref) in enumerate(zip(original, reference)):
                try:
                    if orig.is_quantized:
                        orig = orig.dequantize()
                    if ref.is_quantized:
                        ref = ref.dequantize()
                    if orig.is_mkldnn:
                        orig = orig.to_dense()
                    if ref.is_mkldnn:
                        ref = ref.to_dense()
                    if ref.is_complex() or orig.is_complex():
                        torch.testing.assert_close(
                            orig.to(torch.cdouble),
                            ref.to(torch.cdouble),
                            rtol=check_tolerance,
                            atol=default_tolerances(orig, ref)[1],
                            equal_nan=True,
                        )
                    else:
                        if orig.is_mps or ref.is_mps:
                            torch.testing.assert_close(
                                orig.float(),
                                ref.float(),
                                rtol=check_tolerance,
                                atol=default_tolerances(orig, ref)[1],
                                equal_nan=True,
                            )
                        elif getattr(orig, "is_nested", None) or getattr(
                            ref, "is_nested", None
                        ):
                            if getattr(orig, "is_nested", None) != getattr(
                                ref, "is_nested", None
                            ):
                                raise AssertionError(
                                    f"Nested tensor mismatch: orig.is_nested="
                                    f"{getattr(orig, 'is_nested', None)}, "
                                    f"ref.is_nested={getattr(ref, 'is_nested', None)}"
                                )
                            for t_orig, t_ref in zip(orig.unbind(), ref.unbind()):
                                torch.testing.assert_close(
                                    t_orig.double(),
                                    t_ref.double(),
                                    rtol=check_tolerance,
                                    atol=default_tolerances(t_orig, t_ref)[1],
                                    equal_nan=True,
                                )
                        else:
                            torch.testing.assert_close(
                                orig.double(),
                                ref.double(),
                                rtol=check_tolerance,
                                atol=default_tolerances(orig, ref)[1],
                                equal_nan=True,
                            )
                except AssertionError as e:
                    maybe_warn_nondeterministic()
                    warnings.warn(
                        "Output nr "
                        + str(i + 1)
                        + ". of the traced function does not match "
                        "the corresponding output of the "
                        + match_what
                        + ". Detailed error:\n"
                        + str(e),
                        category=TracerWarning,
                        stacklevel=4,
                    )
                    all_ok = False

            return all_ok

        traced_outs = run_mod_and_filter_tensor_outputs(traced_func, inputs, "trace")
        fn_outs = run_mod_and_filter_tensor_outputs(func, inputs, "Python function")
        if compare_outputs(traced_outs, fn_outs, "Python function"):
            check_outs = run_mod_and_filter_tensor_outputs(
                check_mod_func, inputs, "repeated trace"
            )
            compare_outputs(traced_outs, check_outs, "repeated trace")

        diag_info = graph_diagnostic_info()
        if any(info is not None for info in diag_info):
            raise TracingCheckError(*diag_info)


class TracerWarning(Warning):
    @staticmethod
    def ignore_lib_warnings():
        # We ignore warnings from all submodules excluding the JIT, because we need them e.g. for _check_trace
        warnings.filterwarnings(
            "ignore", category=TracerWarning, module="torch.(?!jit)"
        )
        warnings.filterwarnings("ignore", "torch::jit::fuser::cuda")


# We ignore the tracer warnings coming from inside the library, because all our shape
# checks in nn will trigger them.
TracerWarning.ignore_lib_warnings()
torch._C._tracer_warn_use_python()


def make_tuple(example_inputs):
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    return example_inputs


def make_module(mod, _module_class, _compilation_unit):
    if isinstance(mod, ScriptModule):
        return mod
    elif torch._jit_internal.module_has_exports(mod):
        infer_methods_stubs_fn = torch.jit._recursive.make_stubs_from_exported_methods
        return torch.jit._recursive.create_script_module(
            mod, infer_methods_stubs_fn, share_types=False, is_tracing=True
        )
    else:
        if _module_class is None:
            _module_class = TopLevelTracedModule
        return _module_class(mod, _compilation_unit=_compilation_unit)


def wrap_check_inputs(check_inputs):
    if check_inputs is None:
        return None

    return [{"forward": c} for c in check_inputs]


def analyze_ts_result_with_export_result(export, trace):
    import torch.utils._pytree as pytree

    flat_export = pytree.tree_leaves(export)
    flat_trace = pytree.tree_leaves(trace)

    for orig, loaded in zip(flat_export, flat_trace):
        if orig.layout != loaded.layout:
            return False
        # mkldnn is not supported for torch.allclose
        if orig.layout == torch._mkldnn:  # type: ignore[attr-defined]
            return True
        if type(orig) is not type(loaded):
            return False

        if torch._subclasses.fake_tensor.is_fake_tensor(orig):
            # Skip for FakeTensor.
            return True
        elif isinstance(orig, torch.Tensor):
            if orig.dtype != loaded.dtype:
                return False
            if not torch.allclose(orig, loaded):
                return False
        else:
            if orig != loaded:
                return False
    return True


def _trace_impl(
    func,
    example_inputs=None,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
    example_kwarg_inputs=None,
    _store_inputs=True,
):
    if isinstance(func, torch.jit.ScriptModule):
        # it is hard to trace it because the forward method on ScriptModule is already defined, so it
        # would result in an error.
        warnings.warn(
            "The input to trace is already a ScriptModule, tracing it is a no-op. Returning the object as is.",
            stacklevel=2,
        )
        return func

    if isinstance(func, torch.nn.Module):
        if example_inputs is None:
            if isinstance(example_kwarg_inputs, dict):
                example_inputs = example_kwarg_inputs
            else:
                raise RuntimeError("example_kwarg_inputs should be a dict")
        return trace_module(
            func,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
            example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            _store_inputs=_store_inputs,
        )
    if (
        hasattr(func, "__self__")
        and isinstance(func.__self__, torch.nn.Module)
        and func.__name__ == "forward"
    ):
        if example_inputs is None:
            if isinstance(example_kwarg_inputs, dict):
                example_inputs = example_kwarg_inputs
            else:
                raise RuntimeError("example_kwarg_inputs should be a dict")
        return trace_module(
            func.__self__,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
            example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            _store_inputs=_store_inputs,
        )

    # Special case for common case of passing a single Tensor
    if (
        isinstance(example_inputs, (torch.Tensor, dict))
        and example_kwarg_inputs is None
    ):
        example_inputs = (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    elif example_kwarg_inputs is None and not isinstance(example_inputs, tuple):
        # pyrefly: ignore [bad-argument-type]
        example_inputs = tuple(example_inputs)

    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    if hasattr(func, "__self__") and isinstance(func.__self__, torch.nn.Module):
        raise AttributeError(
            "trace doesn't support compiling individual module's functions.\n"
            "Please use trace_module"
        )

    name = _qualified_name(func)
    if isinstance(example_kwarg_inputs, dict):
        example_inputs = example_kwarg_inputs
        traced = torch._C._create_function_from_trace_with_dict(
            name,
            func,
            example_kwarg_inputs,
            var_lookup_fn,
            strict,
            _force_outplace,
            get_callable_argument_names(func),
        )
    else:
        traced = torch._C._create_function_from_trace(
            name,
            func,
            # pyrefly: ignore [bad-argument-type]
            example_inputs,
            var_lookup_fn,
            strict,
            _force_outplace,
            get_callable_argument_names(func),
        )

    # Check the trace against new traces created from user-specified inputs
    if check_trace:
        if check_inputs is not None:
            _check_trace(
                check_inputs,
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
                example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            )
        else:
            _check_trace(
                [example_inputs],
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
                example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            )

    # Allow torch.compile() to inline
    traced._torchdynamo_inline = func  # type: ignore[attr-defined]
    return traced


class _ExportType(str, Enum):
    DIRECT_EXPORT = "DIRECT_EXPORT"
    TRACE_AND_EXPORT = "TRACE_AND_EXPORT"
    SOURCE_TO_SOURCE = "SOURCE_TO_SOURCE"

    def __str__(self) -> str:
        return self.value


class _ExportOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED_TO_EXPORT = "FAILED_TO_EXPORT"
    FAILED_TO_RUN = "FAILED_TO_RUN"
    ACCURACY_ERROR = "ACCURACY_ERROR"

    def __str__(self) -> str:
        return self.value


def trace(
    func,
    example_inputs=None,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
    example_kwarg_inputs=None,
    _store_inputs=True,
):
    r"""
    Trace a function and return an executable  or :class:`ScriptFunction` that will be optimized using just-in-time compilation.

    Tracing is ideal for code that operates only on
    ``Tensor``\\s and lists, dictionaries, and
    tuples of ``Tensor``\\s.

    Using `torch.jit.trace` and `torch.jit.trace_module`, you can turn an
    existing module or Python function into a TorchScript
    :class:`ScriptFunction` or :class:`ScriptModule`. You must provide example
    inputs, and we run the function, recording the operations performed on all
    the tensors.

    * The resulting recording of a standalone function produces `ScriptFunction`.
    * The resulting recording of `nn.Module.forward` or `nn.Module` produces
      `ScriptModule`.

    This module also contains any parameters that the original
    module had as well.

    Warning:
        Tracing only correctly records functions and modules which are not data
        dependent (e.g., do not have conditionals on data in tensors) and do not have
        any untracked external dependencies (e.g., perform input/output or
        access global variables). Tracing only records operations done when the given
        function is run on the given tensors. Therefore, the returned
        `ScriptModule` will always run the same traced graph on any input. This
        has some important implications when your module is expected to run
        different sets of operations, depending on the input and/or the module
        state. For example,

        * Tracing will not record any control-flow like if-statements or loops.
          When this control-flow is constant across your module, this is fine
          and it often inlines the control-flow decisions. But sometimes the
          control-flow is actually part of the model itself. For instance, a
          recurrent network is a loop over the (possibly dynamic) length of an
          input sequence.
        * In the returned :class:`ScriptModule`, operations that have different
          behaviors in ``training`` and ``eval`` modes will always behave as if
          it is in the mode it was in during tracing, no matter which mode the
          `ScriptModule` is in.

        In cases like these, tracing would not be appropriate and
        :func:`scripting <torch.jit.script>` is a better choice. If you trace
        such models, you may silently get incorrect results on subsequent
        invocations of the model. The tracer will try to emit warnings when
        doing something that may cause an incorrect trace to be produced.

    Args:
        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`
            that will be run with `example_inputs`. `func` arguments and return
            values  must be tensors or (possibly nested) tuples that contain
            tensors. When a module is passed `torch.jit.trace`, only the
            ``forward`` method is run and traced (see :func:`torch.jit.trace
            <torch.jit.trace_module>` for details).

    Keyword arguments:
        example_inputs (tuple or torch.Tensor or None, optional): A tuple of example
            inputs that will be passed to the function while tracing.
            Default: ``None``. Either this argument or ``example_kwarg_inputs``
            should be specified. The resulting trace can be run with inputs of
            different types and shapes assuming the traced operations support those
            types and shapes. `example_inputs` may also be a single Tensor in which
            case it is automatically wrapped in a tuple. When the value is None,
            ``example_kwarg_inputs`` should be specified.

        check_trace (``bool``, optional): Check if the same inputs run through
            traced code produce the same outputs. Default: ``True``. You might want
            to disable this if, for example, your network contains non-
            deterministic ops or if you are sure that the network is correct despite
            a checker failure.

        check_inputs (list of tuples, optional): A list of tuples of input
            arguments that should be used to check the trace against what is
            expected. Each tuple is equivalent to a set of input arguments that
            would be specified in ``example_inputs``. For best results, pass in
            a set of checking inputs representative of the space of shapes and
            types of inputs you expect the network to see.  If not specified,
            the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance
            to use in the checker procedure.  This can be used to relax the
            checker strictness in the event that results diverge numerically
            for a known reason, such as operator fusion.
        strict (``bool``, optional): run the tracer in a strict mode or not
            (default: ``True``). Only turn this off when you want the tracer to
            record your mutable container types (currently ``list``/``dict``)
            and you are sure that the container you are using in your
            problem is a ``constant`` structure and does not get used as
            control flow (if, for) conditions.
        example_kwarg_inputs (dict, optional): This parameter is a pack of keyword
            arguments of example inputs that will be passed to the function while
            tracing. Default: ``None``. Either this argument or ``example_inputs``
            should be specified. The dict will be unpacking by the arguments name
            of the traced function. If the keys of the dict don't not match with
            the traced function's arguments name, a runtime exception will be raised.

    Returns:
        If `func` is `nn.Module` or ``forward`` of `nn.Module`, `trace` returns
        a :class:`ScriptModule` object with a single ``forward`` method
        containing the traced code.  The returned `ScriptModule` will
        have the same set of sub-modules and parameters as the original
        ``nn.Module``.  If ``func`` is a standalone function, ``trace``
        returns `ScriptFunction`.

    Example (tracing a function):

    .. testcode::

        import torch

        def foo(x, y):
            return 2 * x + y

        # Run `foo` with the provided inputs and record the tensor operations
        traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

        # `traced_foo` can now be run with the TorchScript interpreter or saved
        # and loaded in a Python-free environment

    Example (tracing an existing module)::

        import torch
        import torch.nn as nn


        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)


        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

    """
    if sys.version_info >= (3, 14):
        warnings.warn(
            "`torch.jit.trace` is not supported in Python 3.14+ and may break. "
            "Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
        )
    else:
        warnings.warn(
            "`torch.jit.trace` is deprecated. Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
        )
    if not _enabled:
        return func
    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. "
            "Use `with torch.jit.optimized_execution()` instead",
            FutureWarning,
            stacklevel=2,
        )

    from torch._utils_internal import log_torchscript_usage

    traced_func = _trace_impl(
        func,
        example_inputs,
        optimize,
        check_trace,
        check_inputs,
        check_tolerance,
        strict,
        _force_outplace,
        _module_class,
        _compilation_unit,
        example_kwarg_inputs,
        _store_inputs,
    )
    log_torchscript_usage("trace", model_id=_get_model_id(traced_func))
    return traced_func


_trace_module_map: dict[Any, Any] | None = None


def trace_module(
    mod,
    inputs,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
    example_inputs_is_kwarg=False,
    _store_inputs=True,
):
    """
    Trace a module and return an executable :class:`ScriptModule` that will be optimized using just-in-time compilation.

    When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only
    the ``forward`` method is run and traced. With ``trace_module``, you can specify a dictionary of
    method names to example inputs to trace (see the ``inputs``) argument below.

    See :func:`torch.jit.trace <torch.jit.trace>` for more information on tracing.

    Args:
        mod (torch.nn.Module):  A ``torch.nn.Module`` containing methods whose names are
                                specified in ``inputs``. The given methods will be compiled
                                as a part of a single `ScriptModule`.
        inputs (dict):  A dict containing sample inputs indexed by method names in ``mod``.
                                The inputs will be passed to methods whose names correspond to inputs'
                                keys while tracing.
                                ``{ 'forward' : example_forward_input, 'method2': example_method2_input}``
    Keyword arguments:
        check_trace (``bool``, optional): Check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.
        example_inputs_is_kwarg (``bool``, optional): This parameter indicate whether the example inputs is a pack
                                           pack of keyword arguments. Default: ``False``.

    Returns:
        A :class:`ScriptModule` object with a single ``forward`` method containing the traced code.
        When ``func`` is a ``torch.nn.Module``, the returned :class:`ScriptModule` will have the same set of
        sub-modules and parameters as ``func``.

    Example (tracing a module with multiple methods)::

        import torch
        import torch.nn as nn


        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight


        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

        # Trace specific methods on a module (specified in `inputs`), constructs
        # a `ScriptModule` with `forward` and `weighted_kernel_sum` methods
        inputs = {
            "forward": example_forward_input,
            "weighted_kernel_sum": example_weight,
        }
        module = torch.jit.trace_module(n, inputs)

    """
    if sys.version_info >= (3, 14):
        warnings.warn(
            "`torch.jit.trace_method` is not supported in Python 3.14+ and may break. "
            "Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
        )
    else:
        warnings.warn(
            "`torch.jit.trace_method` is deprecated. Please switch to `torch.compile` or `torch.export`.",
            FutureWarning,
        )
    if not _enabled:
        return mod
    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. "
            "Use `with torch.jit.optimized_execution()` instead",
            FutureWarning,
            stacklevel=2,
        )

    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    if not isinstance(mod, torch.nn.Module):
        raise AttributeError("expected torch.nn.Module as the first argument")

    if not isinstance(inputs, dict):
        raise AttributeError("expected a dictionary of (method_name, input) pairs")

    old_module_map = torch.jit._trace._trace_module_map
    try:
        trace_module_map: dict[Any, Any] = {}

        def register_submods(mod, prefix):
            for name, child in mod.named_children():
                submod_qualname = prefix + "." + name
                trace_module_map[child] = submod_qualname
                register_submods(child, submod_qualname)

        trace_module_map["__module"] = mod
        torch.jit._trace._trace_module_map = trace_module_map
        register_submods(mod, "__module")

        module = make_module(mod, _module_class, _compilation_unit)

        for method_name, example_inputs in inputs.items():
            if method_name == "forward":
                # "forward" is a special case because we need to trace
                # `Module.__call__`, which sets up some extra tracing, but uses
                # argument names of the real `Module.forward` method.
                func = mod
                forward_method = getattr(mod, method_name)
                argument_names = get_callable_argument_names(forward_method)
            else:
                func = getattr(mod, method_name)
                argument_names = get_callable_argument_names(func)

            if isinstance(example_inputs, dict) and example_inputs_is_kwarg:
                # Raise exception when the user provided key names are not aligned with forward() method's arguments' name/
                for key in example_inputs:
                    if key not in argument_names:
                        valid_arguments = "[" + ",".join(argument_names) + "]"
                        raise NameError(
                            f"""'{key}' is not in forward() method's arguments,
                         valid arguments name are {valid_arguments}"""
                        )
                module._c._create_method_from_trace_with_dict(
                    method_name,
                    func,
                    example_inputs,
                    var_lookup_fn,
                    strict,
                    _force_outplace,
                    argument_names,
                    _store_inputs,
                )
            else:
                example_inputs = make_tuple(example_inputs)
                module._c._create_method_from_trace(
                    method_name,
                    func,
                    example_inputs,
                    var_lookup_fn,
                    strict,
                    _force_outplace,
                    argument_names,
                    _store_inputs,
                )

            check_trace_method = module._c._get_method(method_name)

            # Check the trace against new traces created from user-specified inputs
            if check_trace:
                if check_inputs is not None:
                    _check_trace(
                        check_inputs,
                        func,
                        check_trace_method,
                        check_tolerance,
                        strict,
                        _force_outplace,
                        True,
                        _module_class,
                        example_inputs_is_kwarg=example_inputs_is_kwarg,
                    )
                else:
                    _check_trace(
                        [inputs],
                        func,
                        check_trace_method,
                        check_tolerance,
                        strict,
                        _force_outplace,
                        True,
                        _module_class,
                        example_inputs_is_kwarg=example_inputs_is_kwarg,
                    )
    finally:
        torch.jit._trace._trace_module_map = old_module_map

    return module


def is_tracing():
    """Return a boolean value.

    Returns ``True`` in tracing (if a function is called during the
    tracing of code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    if is_scripting():
        return False
    return torch._C._is_tracing()


class TracedModule(ScriptModule):
    _disable_script_meta = True

    def __init__(self, orig, id_set=None, _compilation_unit=None):
        # XXX: orig can be a nn.Module or a function!
        super().__init__()
        if not isinstance(orig, torch.nn.Module):
            raise AssertionError(f"Expected nn.Module, got {type(orig)}")

        # Copy a subset of `orig` to a temporary nn.Module.
        # This is a way to customize what will actually get compiled by create_script_module
        id_set = set()

        # This allows us to preserve the original module's qualified name by defining a new
        # type with the attribute _jit_override_qualname. In torch._jit_internal._qualified_name
        # we have a special case that will look up this attribute to override whatever qualname
        # we would get from the python type system
        class QualnameWrapper(torch.nn.Module):
            pass

        QualnameWrapper._jit_override_qualname = torch._jit_internal._qualified_name(  # type: ignore[attr-defined]
            type(orig)
        )

        tmp_module = QualnameWrapper()

        def check_unique(param):
            if param in id_set:
                raise ValueError(
                    "TracedModules don't support parameter sharing between modules"
                )
            id_set.add(param)

        tmp_module.training = orig.training

        for name, param in orig._parameters.items():
            if param is not None:
                tmp_module._parameters[name] = param
                check_unique(param)
        for name, buf in orig._buffers.items():
            if buf is not None:
                tmp_module._buffers[name] = buf
                check_unique(buf)
        for name, val in orig.__dict__.items():
            if (
                torch._C._jit_is_script_object(val)
                and name not in orig._parameters
                and name not in orig._buffers
            ):
                setattr(tmp_module, name, val)

        if orig._backward_hooks:
            raise ValueError(
                "Modules that have backward hooks assigned can't be compiled: "
                + str(orig)
            )

        for name, submodule in orig._modules.items():
            if submodule is None:
                continue
            tmp_module._modules[name] = make_module(
                submodule, TracedModule, _compilation_unit=None
            )

        script_module = torch.jit._recursive.create_script_module(
            tmp_module, lambda module: (), share_types=False, is_tracing=True
        )

        self.__dict__["_name"] = type(orig).__name__
        self.__dict__["_actual_script_module"] = script_module
        for name in ("_parameters", "_buffers", "_modules", "training"):
            delattr(self, name)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Trace submodules cannot be called.")

    def __getattr__(self, attr):
        if "_actual_script_module" not in self.__dict__:
            return super().__getattr__(attr)
        return getattr(self._actual_script_module, attr)

    def __setattr__(self, attr, value):
        if "_actual_script_module" not in self.__dict__:
            return super().__setattr__(attr, value)
        setattr(self._actual_script_module, attr, value)

    def _get_name(self):
        return self._name

    def extra_repr(self):
        return f"original_name={self._name}"


class TopLevelTracedModule(TracedModule):
    forward: Callable[..., Any] = _CachedForward()  # type: ignore[assignment]

    def _reconstruct(self, cpp_module):
        """
        Re-construct an instance of TopLevelTracedModule using an instance of a C++ module.

        Args:
            cpp_module: The C++ module that this TopLevelTracedModule will be rebuilt around.
        """
        self.__dict__["_actual_script_module"]._reconstruct(cpp_module)


def _script_if_tracing(fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_tracing():
            # Not tracing, don't do anything
            return fn(*args, **kwargs)

        compiled_fn: Callable[P, R] = script(wrapper.__original_fn)  # type: ignore[attr-defined]
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn  # type: ignore[attr-defined]
    wrapper.__script_if_tracing_wrapper = True  # type: ignore[attr-defined]

    return wrapper


def _get_trace_graph(
    f,
    args=(),
    kwargs=None,
    strict=True,
    _force_outplace=False,
    return_inputs=False,
    _return_inputs_states=False,
):
    """Return a tuple on tracing a function or model.

    .. warning::
        This function is internal-only and should only be used by the ONNX
        exporter. If you are trying to get a graph through tracing, please go
        through the public API instead::

            trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
            trace_graph = trace.graph

    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value. If return_inputs,
    also returns the trace inputs as part of the tuple

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Args:
        f (torch.nn.Module or function): the function or module
            to be traced.
        args (tuple or Tensor): the positional arguments to pass to the
            function/module to be traced.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        kwargs (dict): the keyword arguments to pass to the function/module
            to be traced.

    Example (trace a cell):

    .. testcode::

        trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    outs = ONNXTracedModule(
        f, strict, _force_outplace, return_inputs, _return_inputs_states
    )(*args, **kwargs)
    return outs
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch._ops import OpOverload, OpOverloadPacket


def _register_decomposition(op: OpOverload, graph: torch._C.Graph) -> None:
    if isinstance(op, OpOverloadPacket):
        raise AssertionError(
            f"Must pass specific op overload, not overload packet, found {op}"
        )
    if not isinstance(op, OpOverload):
        raise AssertionError(f"Expected OpOverload, got {type(op)}")

    torch._C._jit_register_decomposition_for_schema(op._schema, graph)
<|endoftext|># mypy: allow-untyped-defs
import torch


class QuantizedLinear(torch.jit.ScriptModule):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedLinear is no longer supported. Please use "
            "torch.ao.nn.quantized.dynamic.Linear instead."
        )


# FP16 weights
class QuantizedLinearFP16(torch.jit.ScriptModule):
    def __init__(self, other):
        super().__init__()
        raise RuntimeError(
            "torch.jit.QuantizedLinearFP16 is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.Linear instead."
        )


# Quantized RNN cell implementations
class QuantizedRNNCellBase(torch.jit.ScriptModule):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedRNNCellBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedRNNCell(QuantizedRNNCellBase):
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedRNNCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )


class QuantizedLSTMCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedLSTMCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTMCell instead."
        )


class QuantizedGRUCell(QuantizedRNNCellBase):
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedGRUCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRUCell instead."
        )


class QuantizedRNNBase(torch.jit.ScriptModule):
    def __init__(self, other, dtype=torch.int8):
        raise RuntimeError(
            "torch.jit.QuantizedRNNBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic instead."
        )


class QuantizedLSTM(QuantizedRNNBase):
    def __init__(self, other, dtype):
        raise RuntimeError(
            "torch.jit.QuantizedLSTM is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTM instead."
        )


class QuantizedGRU(QuantizedRNNBase):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "torch.jit.QuantizedGRU is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRU instead."
        )


def quantize_rnn_cell_modules(module):
    raise RuntimeError(
        "quantize_rnn_cell_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )


def quantize_linear_modules(module, dtype=torch.int8):
    raise RuntimeError(
        "quantize_linear_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )


def quantize_rnn_modules(module, dtype=torch.int8):
    raise RuntimeError(
        "quantize_rnn_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )
<|endoftext|># mypy: allow-untyped-defs
import ast
import inspect
import sys
import textwrap
import warnings

import torch


class AttributeTypeIsSupportedChecker(ast.NodeVisitor):
    """Check the ``__init__`` method of a given ``nn.Module``.

    It ensures that all instance-level attributes can be properly initialized.

    Specifically, we do type inference based on attribute values...even
    if the attribute in question has already been typed using
    Python3-style annotations or ``torch.jit.annotate``. This means that
    setting an instance-level attribute to ``[]`` (for ``List``),
    ``{}`` for ``Dict``), or ``None`` (for ``Optional``) isn't enough
    information for us to properly initialize that attribute.

    An object of this class can walk a given ``nn.Module``'s AST and
    determine if it meets our requirements or not.

    Known limitations
    1. We can only check the AST nodes for certain constructs; we can't
    ``eval`` arbitrary expressions. This means that function calls,
    class instantiations, and complex expressions that resolve to one of
    the "empty" values specified above will NOT be flagged as
    problematic.
    2. We match on string literals, so if the user decides to use a
    non-standard import (e.g. `from typing import List as foo`), we
    won't catch it.

    Example:
        .. code-block:: python

            class M(torch.nn.Module):
                def fn(self):
                    return []

                def __init__(self) -> None:
                    super().__init__()
                    self.x: List[int] = []

                def forward(self, x: List[int]):
                    self.x = x
                    return 1

        The above code will pass the ``AttributeTypeIsSupportedChecker``
        check since we have a function call in ``__init__``. However,
        it will still fail later with the ``RuntimeError`` "Tried to set
        nonexistent attribute: x. Did you forget to initialize it in
        __init__()?".

    Args:
        nn_module - The instance of ``torch.nn.Module`` whose
            ``__init__`` method we wish to check
    """

    def check(self, nn_module: torch.nn.Module) -> None:
        source_lines = inspect.getsource(nn_module.__class__.__init__)

        # Ignore comments no matter the indentation
        def is_useless_comment(line):
            line = line.strip()
            return line.startswith("#") and not line.startswith("# type:")

        source_lines = "\n".join(
            [l for l in source_lines.split("\n") if not is_useless_comment(l)]
        )

        # This AST only contains the `__init__` method of the nn.Module
        init_ast = ast.parse(textwrap.dedent(source_lines))

        # Get items annotated in the class body
        if sys.version_info >= (3, 14):
            import annotationlib

            self.class_level_annotations = list(
                annotationlib.get_annotations(
                    nn_module, format=annotationlib.Format.FORWARDREF
                ).keys()
            )
        else:
            self.class_level_annotations = list(nn_module.__annotations__.keys())

        # Flag for later
        self.visiting_class_level_ann = False

        self.visit(init_ast)

    def _is_empty_container(self, node: ast.AST, ann_type: str) -> bool:
        if ann_type == "List":
            # Assigning `[]` to a `List` type gives you a Node where
            # value=List(elts=[], ctx=Load())
            if not isinstance(node, ast.List):
                return False
            if node.elts:
                return False
        elif ann_type == "Dict":
            # Assigning `{}` to a `Dict` type gives you a Node where
            # value=Dict(keys=[], values=[])
            if not isinstance(node, ast.Dict):
                return False
            if node.keys:
                return False
        elif ann_type == "Optional":
            # Assigning `None` to an `Optional` type gives you a
            # Node where value=Constant(value=None, kind=None)
            if not isinstance(node, ast.Constant):
                return False
            if node.value:  # type: ignore[attr-defined]
                return False

        return True

    def visit_Assign(self, node) -> None:
        """Store assignment state when assigning to a Call Node.

        If we're visiting a Call Node (the right-hand side of an
        assignment statement), we won't be able to check the variable
        that we're assigning to (the left-hand side of an assignment).
        Because of this, we need to store this state in visitAssign.
        (Luckily, we only have to do this if we're assigning to a Call
        Node, i.e. ``torch.jit.annotate``. If we're using normal Python
        annotations, we'll be visiting an AnnAssign Node, which has its
        target built in.)
        """
        try:
            if (
                isinstance(node.value, ast.Call)
                and node.targets[0].attr in self.class_level_annotations
            ):
                self.visiting_class_level_ann = True
        except AttributeError:
            return
        self.generic_visit(node)
        self.visiting_class_level_ann = False

    def visit_AnnAssign(self, node) -> None:
        """Visit an AnnAssign node in an ``nn.Module``'s ``__init__`` method.

        It checks if it conforms to our attribute annotation rules."""
        # If we have a local variable
        try:
            if node.target.value.id != "self":
                return
        except AttributeError:
            return

        # If we have an attribute that's already been annotated at the
        # class level
        if node.target.attr in self.class_level_annotations:
            return

        # TODO @ansley: add `Union` once landed

        # NB: Even though `Tuple` is a "container", we don't want to
        # check for it here. `Tuple` functions as a type with an
        # "infinite" number of subtypes, in the sense that you can have
        # `Tuple[())]`, `Tuple[T1]`, `Tuple[T2]`, `Tuple[T1, T2]`,
        # `Tuple[T2, T1]` and so on, and none of these subtypes can be
        # used in place of the other. Therefore, assigning an empty
        # tuple in `__init__` CORRECTLY means that that variable
        # cannot be reassigned later to a non-empty tuple. Same
        # deal with `NamedTuple`

        containers = {"List", "list", "Dict", "dict", "Optional"}

        # If we're not evaluating one of the specified problem types
        try:
            if node.annotation.value.id not in containers:
                return
        except AttributeError:
            # To evaluate a base type (`str`, `int`, etc.), we would
            # have needed to get the name through `node.annotation.id`
            # instead of `node.annotation.value.id`. Seems that we're
            # not evaluating one of our "containers"
            return

        # Check if the assigned variable is empty
        ann_type = node.annotation.value.id
        if not self._is_empty_container(node.value, ann_type):
            return

        warnings.warn(
            "The TorchScript type system doesn't support "
            "instance-level annotations on empty non-base "
            "types in `__init__`. Instead, either 1) use a "
            "type annotation in the class body, or 2) wrap "
            "the type in `torch.jit.Attribute`.",
            stacklevel=2,
        )

    def visit_Call(self, node) -> None:
        """Determine if a Call node is 'torch.jit.annotate' in __init__.

        Visit a Call node in an ``nn.Module``'s ``__init__``
        method and determine if it's ``torch.jit.annotate``. If so,
        see if it conforms to our attribute annotation rules.
        """
        # If we have an attribute that's already been annotated at the
        # class level
        if self.visiting_class_level_ann:
            return

        # If this isn't a call to `torch.jit.annotate`
        try:
            if (
                node.func.value.value.id != "torch"
                or node.func.value.attr != "jit"
                or node.func.attr != "annotate"
            ):
                self.generic_visit(node)
            elif (
                node.func.value.value.id != "jit" or node.func.value.attr != "annotate"
            ):
                self.generic_visit(node)
        except AttributeError:
            # Looks like we didn't even have the right node structure
            # to check for `torch.jit.annotate` in the first place
            self.generic_visit(node)

        # Invariant: we have a `torch.jit.annotate` or a
        # `torch.annotate` call

        # A Call Node for `torch.jit.annotate` should have an `args`
        # list of length 2 where args[0] represents the annotation and
        # args[1] represents the actual value
        if len(node.args) != 2:
            return

        if not isinstance(node.args[0], ast.Subscript):
            return

        # See notes in `visit_AnnAssign` r.e. containers

        containers = {"List", "Dict", "Optional"}

        try:
            ann_type = node.args[0].value.id  # type: ignore[attr-defined]
        except AttributeError:
            return

        if ann_type not in containers:
            return

        # Check if the assigned variable is empty
        if not self._is_empty_container(node.args[1], ann_type):
            return

        warnings.warn(
            "The TorchScript type system doesn't support "
            "instance-level annotations on empty non-base "
            "types in `__init__`. Instead, either 1) use a "
            "type annotation in the class body, or 2) wrap "
            "the type in `torch.jit.Attribute`.",
            stacklevel=2,
        )
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import warnings

import torch


@contextlib.contextmanager
def optimized_execution(should_optimize):
    """Context manager that controls whether the JIT's executor will run optimizations before executing a function."""
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(should_optimize)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)


@contextlib.contextmanager
def fuser(name):
    """Context manager that facilitates switching between backend fusers.

    Valid names:
    * ``fuser0`` - enables only legacy fuser
    * ``fuser1`` - enables only NNC
    * ``fuser2`` - enables only nvFuser
    * ``fuser3`` - enables oneDNN Graph
    """
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()
    old_llga_state = torch._C._jit_llga_enabled()
    if name == "fuser0":  # legacy fuser
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    elif name == "fuser1":  # NNC
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    elif name == "fuser2":  # nvFuser
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._jit_set_llga_enabled(False)
    elif name == "fuser3":  # oneDNN Graph
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(True)
    elif name == "none":  # Turn PyTorch fuser off
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_llga_enabled(False)
    else:
        raise Exception(f"unrecognized fuser option (name: {name})")  # noqa: TRY002
    try:
        yield
    finally:
        if name in ["fuser1", "fuser3"]:  # NNC or oneDNN Graph
            torch._C._jit_set_profiling_executor(old_profiling_executor)  # type: ignore[possibly-undefined]
            torch._C._get_graph_executor_optimize(old_profiling_mode)  # type: ignore[possibly-undefined]
        # recover the previous values
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)
        torch._C._jit_set_llga_enabled(old_llga_state)


last_executed_optimized_graph = torch._C._last_executed_optimized_graph


def _get_differentiable_graph_node(node, diff_node) -> None:
    if node.kind() == "prim::DifferentiableGraph":
        diff_node.append(node)
    else:
        for block in node.blocks():
            for n in block.nodes():
                _get_differentiable_graph_node(n, diff_node)


def _graph_for(self, *args, **kwargs):
    return _script_method_graph_for(self, self, *args, **kwargs)


def _script_method_graph_for(self, parent, *args, **kwargs):
    try:
        dbs = parent.get_debug_state()
        eps = list(dbs.execution_plans.values())
        if len(eps) != 1:
            raise AssertionError(f"Expected exactly 1 execution plan, got {len(eps)}")
        graph = eps[0].graph.copy()

        # graph_executor_states for differentiable node
        fw_states = eps[0].code.differentiable_op_executor_states()
        diff_nodes: list[torch._C.Node] = []
        for n in graph.nodes():
            _get_differentiable_graph_node(n, diff_nodes)

        if len(fw_states) != len(diff_nodes):
            raise AssertionError(
                f"Expected fw_states ({len(fw_states)}) and diff_nodes "
                f"({len(diff_nodes)}) to have the same length"
            )
        # swap each differentiable graph with optimized graph in their execution plan
        for n, state in zip(diff_nodes, fw_states):
            fw_execution_plans = list(state.execution_plans.values())
            # we can only update the subgraph when there's a unique execution
            # plan. Avoid assert here so we would skip the ones that can't be
            # updated while try the best effort to update other nodes.
            if len(fw_execution_plans) == 1:
                n.g_("Subgraph", fw_execution_plans[0].graph)

        return graph
    except Exception:
        # fallback approach, we just ran the graph and return the recorded optimized
        # graph
        self(*args, **kwargs)
        return last_executed_optimized_graph()


def set_fusion_strategy(strategy: list[tuple[str, int]]):
    """Set the type and number of specializations that can occur during fusion.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    Usage: provide a list of pairs (type, depth) where type is one of "STATIC" or "DYNAMIC"
    and depth is an integer.

    Behavior - static vs dynamic:
        In STATIC fusion, fused ops are compiled to have fixed input shapes. The shape is determined
        based on some initial profiling runs.
        In DYNAMIC fusion, fused ops are compiled to have variable input shapes, so that multiple
        shapes are possible.

    In both cases, we also recompile on new striding behavior, device, or dtype.

    Behavior - fallback functions & depth:
        When an input doesn't match the format required by the specialized compiled op, it will run
        a fallback function. Fallback functions are recursively compiled and specialized based
        on the observed tensor shapes. Since compilation can be slow, the "depth" parameter is provided to
        limit the number of specializations that can be compiled, before giving up on recompiling and
        falling back to a completely un-fused, un-specialized implementation.

    The list of (type, depth) pairs controls the type of specializations and the number of
    specializations. For example: [("STATIC", 2), ("DYNAMIC", 2)] indicates that the first
    two specializations will use static fusions, the following two specializations will use
    dynamic fusion, and any inputs that satisfy none of the 4 options will run an
    unfused implementation.

    NB: in the future, as more and more fusion backends are added there may be more granular
    apis for specific fusers.
    """
    warnings.warn(
        "`torch.jit.set_fusion_strategy` is deprecated. Please use `torch.compile` instead.",
        DeprecationWarning,
    )
    return torch._C._jit_set_fusion_strategy(strategy)
<|endoftext|># mypy: allow-untyped-defs
# Functions for synthesizing magic methods for JIT-compiled dataclasses
import ast
import dataclasses
import inspect
import os
from collections.abc import Callable
from functools import partial

from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext


def _get_fake_filename(cls, method_name):
    return os.path.join(FAKE_FILENAME_PREFIX, cls.__name__, method_name)


def compose_fn(cls, name: str, body_lines: list[str], signature: str) -> ParsedDef:
    body = "\n".join(f"  {b}" for b in body_lines)
    decl = f"def {name}{signature}:\n{body}"

    # Parse the function declaration
    try:
        py_ast = ast.parse(decl)
    except SyntaxError as e:
        # This should only happen if there's some unforeseeable change
        # in the dataclasses module that makes our synthesized code fail
        raise RuntimeError(
            f"TorchScript failed to synthesize dataclass method '{name}' for class '{cls.__name__}'. "
            "Please file a bug report at <https://github.com/pytorch/pytorch/issues>"
        ) from e
    fake_filename = _get_fake_filename(cls, name)
    # Parse the function
    return ParsedDef(
        py_ast,
        ctx=SourceContext(
            source=decl, filename=fake_filename, file_lineno=0, leading_whitespace_len=0
        ),
        source=decl,
        filename=fake_filename,
        file_lineno=0,
    )


def synthesize__init__(cls) -> ParsedDef:
    # Supporting default factories in the way that people expect would sort of require us to
    # allow compiling lambda functions, which is not currently supported.
    if any(
        field.default_factory is not dataclasses.MISSING
        for field in dataclasses.fields(cls)
    ):
        raise NotImplementedError(
            "Default factory initializers are not supported in TorchScript dataclasses"
        )

    # Simply read off the generated __init__ signature from CPython's implementation. It'll be
    # almost correct except for InitVar annotations, which we need to handle specially.
    signature = inspect.signature(cls.__init__)

    # Handle InitVars if needed (only works on Python 3.8+, when a `type` attribute was added to InitVar);
    # see CPython commit here https://github.com/python/cpython/commit/01ee12ba35a333e8a6a25c4153c4a21838e9585c
    init_vars: list[str] = []
    params = []
    for name, param in signature.parameters.items():
        ann = param.annotation

        if isinstance(ann, dataclasses.InitVar):
            # The TorchScript interpreter can't handle InitVar annotations, so we unwrap the underlying type here
            init_vars.append(name)
            params.append(param.replace(annotation=ann.type))  # type: ignore[attr-defined]
        else:
            params.append(param)

    signature = signature.replace(parameters=params)

    body = [
        # Assign all attributes to self
        f"self.{field.name} = {field.name}"
        for field in dataclasses.fields(cls)
        if field.init and field.name not in init_vars
    ]
    # Call user's impl of __post_init__ if it exists
    if hasattr(cls, "__post_init__"):
        body.append("self.__post_init__(" + ", ".join(init_vars) + ")")

    return compose_fn(cls, "__init__", body or ["pass"], signature=str(signature))


# This is a placeholder at the moment since the TorchScript interpreter doesn't call __repr__
def synthesize__repr__(cls) -> ParsedDef:
    return compose_fn(
        cls,
        "__repr__",
        [
            f"return '{cls.__name__}("
            + ", ".join(
                [
                    f"{field.name}=self.{field.name}"
                    for field in dataclasses.fields(cls)
                    if field.repr
                ]
            )
            + ")'"
        ],
        signature="(self) -> str",
    )


def synthesize__hash__(cls) -> ParsedDef:
    return compose_fn(
        cls,
        "__hash__",
        [
            # This is just a placeholder to prevent compilation from failing; this won't even get called at
            # all right now because the TorchScript interpreter doesn't call custom __hash__ implementations
            "raise NotImplementedError('__hash__ is not supported for dataclasses in TorchScript')"
        ],
        signature="(self) -> int",
    )


# Implementation for __eq__ and __ne__
def synthesize_equality(cls, name: str, converse: str) -> ParsedDef:
    return synthesize_comparison(
        cls,
        name,
        allow_eq=True,
        raise_on_none=False,
        inner=[f"if val1 {converse} val2: return False"],
    )


def synthesize_inequality(cls, name: str, op: str, allow_eq: bool) -> ParsedDef:
    return synthesize_comparison(
        cls,
        name,
        allow_eq,
        raise_on_none=True,
        inner=[
            f"if val1 {op} val2: return True",
            f"elif val2 {op} val1: return False",
        ],
    )


def synthesize_comparison(
    cls, name: str, allow_eq: bool, raise_on_none: bool, inner: list[str]
) -> ParsedDef:
    body = []
    for field in dataclasses.fields(cls):
        if not field.compare:
            continue

        body.extend(
            [
                f"val1 = self.{field.name}",
                f"val2 = other.{field.name}",
            ]
        )
        body.extend(
            inner
            if not is_optional(field.type)
            else [
                # Type refinement for optional fields; we need this to avoid type errors from the interpreter
                "if val1 is not None and val2 is not None:",
                *["  " + line for line in inner],
                "elif (val1 is None) != (val2 is None):",
                f"  raise TypeError('Cannot compare {cls.__name__} with None')"
                if raise_on_none
                else "  return False",
            ]
        )

    body.append(f"return {allow_eq}")
    return compose_fn(
        cls, name, body, signature=f"(self, other: {cls.__name__}) -> bool"
    )


DATACLASS_MAGIC_METHODS: dict[str, Callable] = {
    "__init__": synthesize__init__,
    "__repr__": synthesize__repr__,
    "__hash__": synthesize__hash__,
    "__eq__": partial(synthesize_equality, name="__eq__", converse="!="),
    "__ne__": partial(synthesize_equality, name="__ne__", converse="=="),
    "__lt__": partial(synthesize_inequality, name="__lt__", op="<", allow_eq=False),
    "__le__": partial(synthesize_inequality, name="__le__", op="<", allow_eq=True),
    "__gt__": partial(synthesize_inequality, name="__gt__", op=">", allow_eq=False),
    "__ge__": partial(synthesize_inequality, name="__ge__", op=">", allow_eq=True),
}
<|endoftext|># mypy: allow-untyped-defs
"""Freezing.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import warnings

import torch
from torch.jit._script import RecursiveScriptModule, ScriptModule


def freeze(
    mod, preserved_attrs: list[str] | None = None, optimize_numerics: bool = True
):
    r"""Freeze ScriptModule, inline submodules, and attributes as constants.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned
    module's submodules, parameters, and attributes as constants in the TorchScript IR Graph.
    By default, `forward` will be preserved, as well as attributes & methods specified in
    `preserved_attrs`. Additionally, any attribute that is modified within a preserved
    method will be preserved.

    Freezing currently only accepts ScriptModules that are in eval mode.

    Freezing applies generic optimization that will speed up your model regardless of machine.
    To further optimize using server-specific settings, run `optimize_for_inference` after
    freezing.

    Args:
        mod (:class:`ScriptModule`): a module to be frozen
        preserved_attrs (Optional[List[str]]): a list of attributes to preserve in addition to the forward method.
            Attributes modified in preserved methods will also be preserved.
        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
            preserve numerics. Full details of optimization can be found at `torch.jit.run_frozen_optimizations`.

    Returns:
        Frozen :class:`ScriptModule`.

    Example (Freezing a simple module with a Parameter):

    .. testcode::
        import torch
        class MyModule(torch.nn.Module):
            def __init__(self, N, M):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(N, M))
                self.linear = torch.nn.Linear(N, M)

            def forward(self, input):
                output = self.weight.mm(input)
                output = self.linear(output)
                return output

        scripted_module = torch.jit.script(MyModule(2, 3).eval())
        frozen_module = torch.jit.freeze(scripted_module)
        # parameters have been removed and inlined into the Graph as constants
        assert len(list(frozen_module.named_parameters())) == 0
        # See the compiled graph as Python code
        print(frozen_module.code)

    Example (Freezing a module with preserved attributes)

    .. testcode::
        import torch
        class MyModule2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.modified_tensor = torch.tensor(10.)
                self.version = 1

            def forward(self, input):
                self.modified_tensor += 1
                return input + self.modified_tensor

        scripted_module = torch.jit.script(MyModule2().eval())
        frozen_module = torch.jit.freeze(scripted_module, preserved_attrs=["version"])
        # we've manually preserved `version`, so it still exists on the frozen module and can be modified
        assert frozen_module.version == 1
        frozen_module.version = 2
        # `modified_tensor` is detected as being mutated in the forward, so freezing preserves
        # it to retain model semantics
        assert frozen_module(torch.tensor(1)) == torch.tensor(12)
        # now that we've run it once, the next result will be incremented by one
        assert frozen_module(torch.tensor(1)) == torch.tensor(13)

    Note:
        Freezing submodule attributes is also supported:
        frozen_module = torch.jit.freeze(scripted_module, preserved_attrs=["submodule.version"])

    Note:
        If you're not sure why an attribute is not being inlined as a constant, you can run
        `dump_alias_db` on frozen_module.forward.graph to see if freezing has detected the
        attribute is being modified.

    Note:
        Because freezing makes weights constants and removes module hierarchy, `to` and other
        nn.Module methods to manipulate device or dtype no longer work. As a workaround,
        You can remap devices by specifying `map_location` in `torch.jit.load`, however
        device-specific logic may have been baked into the model.
    """
    warnings.warn(
        "`torch.jit.freeze` is deprecated. Please use `torch.compile` instead.",
        FutureWarning,
    )
    if not isinstance(mod, ScriptModule):
        raise RuntimeError(
            "Freezing expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
        )

    if mod.training:
        raise RuntimeError(
            "Freezing is currently only implemented for modules in eval mode. "
            "Please call .eval() on your module before freezing."
        )

    preserved_attrs = preserved_attrs if preserved_attrs is not None else []

    out = RecursiveScriptModule(torch._C._freeze_module(mod._c, preserved_attrs))
    RecursiveScriptModule._finalize_scriptmodule(out)

    preserved_methods = [x for x in preserved_attrs if mod._c._has_method(x)]
    run_frozen_optimizations(out, optimize_numerics, preserved_methods)

    return out


def run_frozen_optimizations(
    mod, optimize_numerics: bool = True, preserved_methods: list[str] | None = None
) -> None:
    r"""
    Run a series of optimizations looking for patterns that occur in frozen graphs.

    The current set of optimizations includes:
        - Dropout Removal
        - Pretranspose Linear Layers
        - Concat Linear Layers with same input Tensor
        - Conv -> Batchnorm folding
        - Conv -> Add/Sub folding
        - Conv -> Mul/Div folding

    Args:
        mod (:class:`ScriptModule`): a frozen module to be optimized

        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly
        preserve numerics. These optimizations preserve default rtol and atol of `torch.testing.assert_close`
        when applied on a single transformation, however in a module where many transformations are applied
        the rtol or atol may no longer fall within the default `assert_close` tolerance. Conv -> Batchnorm folding,
        Conv-Add/Sub, and Conv -> Mul/Div folding all may alter numerics.

    Returns:
        None

    Note:
        In rare occasions, this can result in slower execution.

    Example (Freezing a module with Conv->Batchnorm)
    .. code-block:: python
        import torch

        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True
        )
        bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        mod = torch.nn.Sequential(conv, bn)
        # set optimize to False here, by default freezing runs run_frozen_optimizations
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()), optimize=False)
        # inspect frozen mod
        assert "batch_norm" in str(frozen_mod.graph)
        torch.jit.run_frozen_optimizations(frozen_mod)
        assert "batch_norm" not in str(frozen_mod.graph)

    """
    if mod._c._has_method("forward"):
        torch._C._jit_pass_optimize_frozen_graph(mod.graph, optimize_numerics)

    if preserved_methods is None:
        preserved_methods = []

    for method in preserved_methods:
        torch._C._jit_pass_optimize_frozen_graph(
            mod.__getattr__(method).graph, optimize_numerics
        )


def optimize_for_inference(
    mod: ScriptModule, other_methods: list[str] | None = None
) -> ScriptModule:
    """
    Perform a set of optimization passes to optimize a model for the purposes of inference.

    .. deprecated:: 2.5
        TorchScript is deprecated, please use ``torch.compile`` instead.

    If the model is not already frozen, optimize_for_inference
    will invoke `torch.jit.freeze` automatically.

    In addition to generic optimizations that should speed up your model regardless
    of environment, prepare for inference will also bake in build specific settings
    such as the presence of CUDNN or MKLDNN, and may in the future make transformations
    which speed things up on one machine but slow things down on another. Accordingly,
    serialization is not implemented following invoking `optimize_for_inference` and
    is not guaranteed.

    This is still in prototype, and may have the potential to slow down your model.
    Primary use cases that have been targeted so far have been vision models on cpu
    and gpu to a lesser extent.

    Example (optimizing a module with Conv->Batchnorm)::

        import torch

        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True
        )
        bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        mod = torch.nn.Sequential(conv, bn)
        frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(mod.eval()))
        assert "batch_norm" not in str(frozen_mod.graph)
        # if built with MKLDNN, convolution will be run with MKLDNN weights
        assert "MKLDNN" in frozen_mod.graph
    """
    warnings.warn(
        "`torch.jit.optimize_for_inference` is deprecated. Please use `torch.compile` instead.",
        FutureWarning,
    )
    if not isinstance(mod, ScriptModule):
        raise RuntimeError(
            "optimize_for_inference expects a ScriptModule as input. "
            "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
        )

    if other_methods is None:
        other_methods = []

    if hasattr(mod, "training"):
        mod = freeze(mod.eval(), preserved_attrs=other_methods)

    torch._C._jit_pass_optimize_for_inference(mod._c, other_methods)

    return mod
<|endoftext|># mypy: allow-untyped-defs
import inspect
import textwrap

import torch.jit
from torch.jit._builtins import _find_builtin


# this file is for generating documentation using sphinx autodoc
# > help(torch.jit.supported_ops) will also give a nice list of the
# supported ops programmatically


def _hidden(name):
    return name.startswith("_") and not name.startswith("__")


def _emit_type(type):
    return str(type)


def _emit_arg(indent, i, arg):
    v = f"{arg.name} : {_emit_type(arg.type)}"
    default = arg.default_value
    if default is not None:
        v = f"{v}={str(default)}"
    if i > 0:
        v = f"\n{' ' * indent}{v}"
    return v


def _emit_args(indent, arguments):
    return ",".join(_emit_arg(indent, i, arg) for i, arg in enumerate(arguments))


def _emit_ret(ret):
    return _emit_type(ret.type)


def _emit_rets(returns):
    if len(returns) == 1:
        return _emit_ret(returns[0])
    return f"Tuple[{', '.join(_emit_ret(r) for r in returns)}]"


def _emit_schema(mod, name, schema, arg_start=0, padding=4):
    if mod is None:
        qualified_name = name
    else:
        qualified_name = f"{mod}.{name}"
    schema_str = (
        f"{qualified_name}"
        f"({_emit_args(len(qualified_name) + 1 + padding, schema.arguments[arg_start:])}) "
        f"-> {_emit_rets(schema.returns)}"
    )
    return schema_str


def _get_tensor_ops():
    def is_tensor_method(schema) -> bool:
        if len(schema.arguments) == 0:
            return False
        self = schema.arguments[0]
        if self.name != "self":
            return False
        if not self.type.isSubtypeOf(torch._C.TensorType.get()):
            return False
        return True

    methods = []
    # discover methods
    for elem in dir(torch.Tensor):
        if not _hidden(elem):
            schemas = torch._C._jit_get_schemas_for_operator("aten::" + elem)
            for schema in schemas:
                if is_tensor_method(schema):
                    methods.append(_emit_schema("Tensor", elem, schema, arg_start=1))

    return "Supported Tensor Methods", methods


def _get_nn_functional_ops():
    functions = []

    # Iterate over torch.nn.functional
    mod = torch.nn.functional
    name = mod.__name__
    for elem in dir(torch.nn.functional):
        attr = getattr(mod, elem)
        if not inspect.isfunction(attr) or _hidden(elem[0]):
            # Ignore non-functions and internal methods
            continue

        attr_module = inspect.getmodule(attr)
        if not attr_module:
            raise RuntimeError(f"Module for {attr} not found")

        if "torch.nn.functional" not in attr_module.__name__:
            # Ignore functions from outside torch.nn.functional
            continue

        try:
            # compile fn, get schema
            scripted = torch.jit.script(attr)
            scripted_schema = scripted.schema
            functions.append(_emit_schema(name, elem, scripted_schema))
        except:  # noqa: E722
            # Skip interpolate / boolean dispatched things
            pass

    # Iterate over modules that we know contain a lot of builtins
    for mod in torch.jit._builtins._modules_containing_builtins:
        name = mod.__name__
        for elem in dir(mod):
            builtin = _find_builtin(getattr(mod, elem))
            if builtin is not None:
                schemas = torch._C._jit_get_schemas_for_operator(builtin)
                for schema in schemas:
                    # remove _tan but not __and__
                    if not _hidden(elem):
                        functions.append(_emit_schema(name, elem, schema))
    return "Supported PyTorch Functions", functions


def _get_builtins_helper():
    builtins = []
    for fn, _builtin_name in torch.jit._builtins._builtin_ops:
        mod = inspect.getmodule(fn)

        if not hasattr(fn, "__name__"):
            # typing classes
            continue
        if not mod:
            continue
        if _hidden(fn.__name__) or _hidden(fn.__qualname__) or _hidden(mod.__name__):
            # skip internal-only methods
            continue

        if "torch._C" in mod.__name__:
            continue

        builtins.append((fn, _builtin_name))

    return builtins


def _is_math_fn(fn):
    mod = inspect.getmodule(fn)
    if not mod:
        raise RuntimeError(f"Module for {fn} not found")

    return mod.__name__ == "math"


def _get_torchscript_builtins():
    functions = []
    builtins = filter(lambda fn: not _is_math_fn(fn[0]), _get_builtins_helper())
    builtins_list = list(builtins)
    # Iterate over the specially added builtins
    for fn, _builtin_name in builtins_list:
        mod = inspect.getmodule(fn)
        if not mod:
            raise RuntimeError(f"Module for {fn} not found")
        builtin = _find_builtin(fn)
        if builtin is not None:
            schemas = torch._C._jit_get_schemas_for_operator(builtin)
            for schema in schemas:
                functions.append(_emit_schema(mod.__name__, fn.__name__, schema))

    return "TorchScript Builtin Functions", functions


def _get_math_builtins():
    functions = []
    builtins = filter(lambda fn: _is_math_fn(fn[0]), _get_builtins_helper())
    builtins_list = list(builtins)
    # Iterate over the specially added builtins
    for fn, _builtin_name in builtins_list:
        mod = inspect.getmodule(fn)
        if not mod:
            raise RuntimeError(f"Module for {fn} not found")
        builtin = _find_builtin(fn)
        if builtin is not None:
            schemas = torch._C._jit_get_schemas_for_operator(builtin)
            for schema in schemas:
                schema_str = _emit_schema(mod.__name__, fn.__name__, schema)
                if "Tensor" in schema_str:
                    # Skip Tensor ops that have the same name as math functions
                    # (they will show up in the tensor methods section)
                    continue
                functions.append(schema)

    return "``math`` Module", functions


def _get_global_builtins():
    # Taken from the 'globals' map in torch/csrc/jit/frontend/ir_emitter.cpp
    supported_builtins = [
        "print",
        "tuple",
        "float",
        "complex",
        "int",
        "bool",
        "str",
        "getattr",
        "hasattr",
        "isinstance",
        "len",
        "hex",
        "oct",
        "round",
        "hash",
        "min",
        "max",
        "abs",
        "all",
        "divmod",
        "list",
        "ord",
        "chr",
        "bin",
        "range",
        "zip",
        "enumerate",
        "sorted",
    ]

    op_renames = {
        "bool": "aten::Bool",
        "int": "aten::Int",
        "float": "aten::Float",
        "complex": "aten::Complex",
        "abs": "prim::abs",
        "max": "prim::max",
        "min": "prim::min",
        "range": "fake::does_not_exist",
    }

    schemaless_op_explanations = {
        "print": "Print any value",
        "tuple": "Lists cannot be converted to tuples with this method since their size is not statically known",
        "getattr": "Attribute name must be a literal string",
        "hasattr": "Attribute name must be a literal string",
        "isinstance": "Result is static",
        "zip": "Arguments must be iterable.",
        "enumerate": "Arguments must be iterable.",
        "range": "Can only be used as an iterator in a for loop",
    }

    magic_methods = [
        ("complex", "__complex__"),
        ("float", "__float__"),
        ("int", "__int__"),
        ("bool", "__bool__"),
        ("str", "__str__"),
        ("len", "__len__"),
        ("hex", "__hex__"),
        ("oct", "__oct__"),
    ]

    magic_methods_rows = []
    for fn, magic_method in magic_methods:
        # pyrefly: ignore [bad-argument-type]
        magic_methods_rows.append(f'"{fn}", "``{magic_method}``"')

    schematized_ops = []
    schemaless_ops = []

    for fn in supported_builtins:
        op_name = f"aten::{fn}"
        if fn in op_renames:
            op_name = op_renames[fn]
        schemas = torch._C._jit_get_schemas_for_operator(op_name)
        for s in schemas:
            schematized_ops.append(_emit_schema(None, fn, s, padding=0))
        if len(schemas) > 0:
            schematized_ops.append("")
        else:
            table_row = (
                f'":external+python:py:obj:`{fn}`", "{schemaless_op_explanations[fn]}"'
            )
            # pyrefly: ignore [bad-argument-type]
            schemaless_ops.append(table_row)

    schematized_ops_str = "\n".join(schematized_ops)
    schemaless_ops_str = "\n".join(schemaless_ops)
    magic_methods_rows_str = "\n".join(magic_methods_rows)
    schematized_ops_str = textwrap.indent(schematized_ops_str, "\t")
    schemaless_ops_str = textwrap.indent(schemaless_ops_str, "\t")
    magic_methods_rows_str = textwrap.indent(magic_methods_rows_str, "\t")
    section = f"""
The functions in the following table are supported but do not have a static schema

.. csv-table::
    :header: "Function", "Note"

{schemaless_ops_str}

The following functions will use the corresponding magic method on TorchScript classes

.. csv-table::
    :header: "Function", "Magic Method"

{magic_methods_rows_str}

These built-in functions use the schema

.. rst-class:: codeblock-height-limiter

::

{schematized_ops_str}
    """

    return "Python Built-in Functions", section


def _list_supported_ops():
    def emit_block(decls):
        return "\n.. rst-class:: codeblock-height-limiter\n\n::\n\n{}\n".format(
            "".join(f"    {d}\n\n" for d in decls)
        )

    body = ""
    op_gathering_fns = (
        _get_tensor_ops,
        _get_nn_functional_ops,
        _get_torchscript_builtins,
        _get_global_builtins,
        _get_math_builtins,
    )
    for fn in op_gathering_fns:
        header, items = fn()
        link_target = header.replace("`", "").replace("-", "").lower().replace(" ", "-")
        if isinstance(items, str):
            section = f"{header}\n{'~' * len(header)}\n{items}\n"
        else:
            section = f"{header}\n{'~' * len(header)}\n{emit_block(items)}"
        section = f".. _{link_target}:" + "\n\n" + section
        body += section

    return body


__doc__ = _list_supported_ops()
<|endoftext|>"""TorchScript.

This module contains functionality to support the JIT's scripting frontend, notably:
    - torch.jit.script

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import collections
import copy
import enum
import functools
import inspect
import pickle
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, TypeVar
from typing_extensions import deprecated, Self

import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _get_model_id, _qualified_name
from torch._utils_internal import log_torchscript_usage
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for
from torch.jit._monkeytype_config import (
    JitTypeTraceConfig,
    JitTypeTraceStore,
    monkeytype_trace,
)
from torch.jit._recursive import (
    _compile_and_register_class,
    infer_methods_to_compile,
    ScriptMethodStub,
    wrap_cpp_module,
)
from torch.jit._state import (
    _enabled,
    _set_jit_function_cache,
    _set_jit_overload_cache,
    _try_get_jit_cached_function,
    _try_get_jit_cached_overloads,
)
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module

from ._serialization import validate_map_location


_T = TypeVar("_T")


type_trace_db = JitTypeTraceStore()  # DB to hold all call traces from MonkeyType

torch._C.ScriptMethod.graph_for = _script_method_graph_for  # type: ignore[attr-defined]
torch._C.ScriptFunction.graph_for = _graph_for  # type: ignore[attr-defined]
ScriptFunction = torch._C.ScriptFunction
ScriptFunction.__doc__ = """
Functionally equivalent to a :class:`ScriptModule`, but represents a single
function and does not have any attributes or Parameters.
"""
ScriptFunction.__name__ = "ScriptFunction"
ScriptFunction.__qualname__ = "torch.jit.ScriptFunction"
set_module(ScriptFunction, "torch.jit")


# Throws an error if a jit function is pickled.
# Helps to avoid Python crashes for Python versions 3.9.5 + when protocol 0 or 1 is given as an argument.
def _reduce(cls):
    raise pickle.PickleError("ScriptFunction cannot be pickled")


ScriptFunction.__reduce__ = _reduce  # type: ignore[assignment]


if _enabled:
    Attribute = collections.namedtuple("Attribute", ["value", "type"])
else:

    def Attribute(value, type):  # type: ignore[no-redef]
        return value


Attribute.__doc__ = """
    This method is a pass-through function that returns `value`, mostly
    used to indicate to the TorchScript compiler that the left-hand side
    expression is a class instance attribute with type of `type`. Note that
    `torch.jit.Attribute` should only be used in `__init__` method of `jit.ScriptModule`
    subclasses.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume
      it is type `T` rather than `Optional[T]`

    In eager mode, it is simply a pass-through function that returns `value`
    without other implications.

    Example:

    .. testcode::

        import torch
        from typing import Dict

        class AttributeModule(torch.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.jit.Attribute(0.1, float)

                # we should be able to use self.foo as a float here
                assert 0.0 < self.foo

                self.names_ages = torch.jit.Attribute({}, Dict[str, int])
                self.names_ages["someone"] = 20
                assert isinstance(self.names_ages["someone"], int)

        m = AttributeModule()
        # m will contain two attributes
