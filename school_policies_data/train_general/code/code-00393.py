            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/variable_factories.h",
            "ops_headers": [
                f"#include <ATen/ops/{fn.root_name}.h>" for fn in factory_functions
            ],
            "function_definitions": list(mapMaybe(process_function, factory_functions)),
        },
    )


@with_native_function
def is_factory_function(f: NativeFunction) -> bool:
    if Variant.function not in f.variants:
        return False

    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    return has_tensor_options or name.endswith("_like")


@with_native_function
def process_function(f: NativeFunction) -> str | None:
    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    is_factory = has_tensor_options or name.endswith("_like")

    if Variant.function not in f.variants or not is_factory:
        return None

    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False)
    sigs = [cpp_sigs.signature]
    if cpp_sigs.symint_signature is not None:
        sigs.append(cpp_sigs.symint_signature)
    r = ""
    for sig in sigs:
        formals: list[str] = []
        exprs: list[str] = []
        requires_grad = "false"
        for arg in sig.arguments():
            qualified_type = fully_qualified_type(arg.type)
            if arg.default:
                formals.append(f"{qualified_type} {arg.name} = {arg.default}")
            else:
                formals.append(f"{qualified_type} {arg.name}")

            if isinstance(arg.argument, TensorOptionsArguments):
                # note: we remove the requires_grad setting from the TensorOptions because
                # it is ignored anyways (and we actually have an assertion that it isn't set
                # which would fail otherwise). We handle requires_grad explicitly here
                # instead of passing it through to the kernel.
                exprs.append(
                    f"at::TensorOptions({arg.name}).requires_grad(::std::nullopt)"
                )
                # Manually set the requires_grad bit on the result tensor.
                requires_grad = f"{arg.name}.requires_grad()"
            else:
                exprs.append(arg.name)

        r += f"""\
inline at::Tensor {sig.name()}({", ".join(formals)}) {{
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(at::{sig.name()}({", ".join(exprs)}), /*requires_grad=*/{requires_grad});
}}
"""
    return r
<|endoftext|>"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.autograd.gen_autograd \
       aten/src/ATen/native/native_functions.yaml \
       aten/src/ATen/native/tags.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/autograd/generated/
"""

# gen_autograd.py generates C++ autograd functions and Python bindings.
#
# It delegates to the following scripts:
#
#  gen_autograd_functions.py: generates subclasses of torch::autograd::Node
#  gen_variable_type.py: generates VariableType.h which contains all tensor methods
#  gen_python_functions.py: generates Python bindings to THPVariable
#

from __future__ import annotations

import argparse
import os

from torchgen.api import cpp
from torchgen.api.autograd import (
    match_differentiability_info,
    NativeFunctionWithDifferentiabilityInfo,
)
from torchgen.gen import parse_native_yaml
from torchgen.selective_build.selector import SelectiveBuilder

from . import gen_python_functions
from .gen_autograd_functions import (
    gen_autograd_functions_lib,
    gen_autograd_functions_python,
)
from .gen_inplace_or_view_type import gen_inplace_or_view_type
from .gen_trace_type import gen_trace_type
from .gen_variable_factories import gen_variable_factories
from .gen_variable_type import gen_variable_type
from .gen_view_funcs import gen_view_funcs
from .load_derivatives import load_derivatives


def gen_autograd(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
    operator_selector: SelectiveBuilder,
    disable_autograd: bool = False,
) -> None:
    # Parse and load derivatives.yaml
    differentiability_infos, used_dispatch_keys = load_derivatives(
        os.path.join(autograd_dir, "derivatives.yaml"), native_functions_path, tags_path
    )

    template_path = os.path.join(autograd_dir, "templates")

    native_funcs = parse_native_yaml(native_functions_path, tags_path).native_functions
    fns = sorted(
        filter(
            operator_selector.is_native_function_selected_for_training, native_funcs
        ),
        key=lambda f: cpp.name(f.func),
    )
    fns_with_diff_infos: list[NativeFunctionWithDifferentiabilityInfo] = (
        match_differentiability_info(fns, differentiability_infos)
    )

    # Generate VariableType.h/cpp
    if not disable_autograd:
        gen_variable_type(
            out,
            native_functions_path,
            tags_path,
            fns_with_diff_infos,
            template_path,
            used_dispatch_keys,
        )

        gen_inplace_or_view_type(
            out, native_functions_path, tags_path, fns_with_diff_infos, template_path
        )

        # operator filter not applied as tracing sources are excluded in selective build
        gen_trace_type(out, native_funcs, template_path)
    # Generate Functions.h/cpp
    gen_autograd_functions_lib(out, differentiability_infos, template_path)

    # Generate variable_factories.h
    gen_variable_factories(out, native_functions_path, tags_path, template_path)

    # Generate ViewFuncs.h/cpp
    gen_view_funcs(out, fns_with_diff_infos, template_path)


def gen_autograd_python(
    native_functions_path: str,
    tags_path: str,
    out: str,
    autograd_dir: str,
) -> None:
    differentiability_infos, _ = load_derivatives(
        os.path.join(autograd_dir, "derivatives.yaml"), native_functions_path, tags_path
    )

    template_path = os.path.join(autograd_dir, "templates")

    # Generate Functions.h/cpp
    gen_autograd_functions_python(out, differentiability_infos, template_path)

    # Generate Python bindings
    deprecated_path = os.path.join(autograd_dir, "deprecated.yaml")
    gen_python_functions.gen(
        out, native_functions_path, tags_path, deprecated_path, template_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate autograd C++ files script")
    parser.add_argument(
        "native_functions", metavar="NATIVE", help="path to native_functions.yaml"
    )
    parser.add_argument("tags", metavar="NATIVE", help="path to tags.yaml")
    parser.add_argument("out", metavar="OUT", help="path to output directory")
    parser.add_argument(
        "autograd", metavar="AUTOGRAD", help="path to autograd directory"
    )
    args = parser.parse_args()
    gen_autograd(
        args.native_functions,
        args.tags,
        args.out,
        args.autograd,
        SelectiveBuilder.get_nop_selector(),
    )


if __name__ == "__main__":
    main()
<|endoftext|>"""
For procedural tests needed for __torch_function__, we use this function
to export method names and signatures as needed by the tests in
test/test_overrides.py.

python -m tools.autograd.gen_annotated_fn_args \
       aten/src/ATen/native/native_functions.yaml \
       aten/src/ATen/native/tags.yaml \
       $OUTPUT_DIR \
       tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/testing/_internal/generated
"""

from __future__ import annotations

import argparse
import os
import textwrap
from collections import defaultdict
from typing import Any, TYPE_CHECKING

import torchgen.api.python as python
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.utils import FileManager

from .gen_python_functions import (
    is_py_fft_function,
    is_py_linalg_function,
    is_py_nn_function,
    is_py_special_function,
    is_py_torch_function,
    is_py_variable_method,
    should_generate_py_binding,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchgen.model import Argument, BaseOperatorName, NativeFunction


def gen_annotated(
    native_yaml_path: str, tags_yaml_path: str, out: str, autograd_dir: str
) -> None:
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    mappings = (
        (is_py_torch_function, "torch._C._VariableFunctions"),
        (is_py_nn_function, "torch._C._nn"),
        (is_py_linalg_function, "torch._C._linalg"),
        (is_py_special_function, "torch._C._special"),
        (is_py_fft_function, "torch._C._fft"),
        (is_py_variable_method, "torch.Tensor"),
    )
    annotated_args: list[str] = []
    for pred, namespace in mappings:
        groups: dict[BaseOperatorName, list[NativeFunction]] = defaultdict(list)
        for f in native_functions:
            if not should_generate_py_binding(f) or not pred(f):
                continue
            groups[f.func.name.name].append(f)
        for group in groups.values():
            for f in group:
                annotated_args.append(f"{namespace}.{gen_annotated_args(f)}")

    template_path = os.path.join(autograd_dir, "templates")
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template(
        "annotated_fn_args.py",
        "annotated_fn_args.py.in",
        lambda: {
            "annotated_args": textwrap.indent("\n".join(annotated_args), "    "),
        },
    )


@with_native_function
def gen_annotated_args(f: NativeFunction) -> str:
    def _get_kwargs_func_exclusion_list() -> list[str]:
        # functions that currently don't work with kwargs in test_overrides.py
        return [
            "diagonal",
            "round_",
            "round",
            "scatter_",
        ]

    def _add_out_arg(
        out_args: list[dict[str, Any]], args: Sequence[Argument], *, is_kwarg_only: bool
    ) -> None:
        for arg in args:
            if arg.default is not None:
                continue
            out_arg: dict[str, Any] = {}
            out_arg["is_kwarg_only"] = str(is_kwarg_only)
            out_arg["name"] = arg.name
            out_arg["simple_type"] = python.argument_type_str(
                arg.type, simple_type=True
            )
            size_t = python.argument_type_size(arg.type)
            if size_t:
                out_arg["size"] = size_t
            out_args.append(out_arg)

    out_args: list[dict[str, Any]] = []
    _add_out_arg(out_args, f.func.arguments.flat_positional, is_kwarg_only=False)
    if f"{f.func.name.name}" not in _get_kwargs_func_exclusion_list():
        _add_out_arg(out_args, f.func.arguments.flat_kwarg_only, is_kwarg_only=True)

    return f"{f.func.name.name}: {repr(out_args)},"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate annotated_fn_args script")
    parser.add_argument(
        "native_functions", metavar="NATIVE", help="path to native_functions.yaml"
    )
    parser.add_argument("tags", metavar="TAGS", help="path to tags.yaml")
    parser.add_argument("out", metavar="OUT", help="path to output directory")
    parser.add_argument(
        "autograd", metavar="AUTOGRAD", help="path to template directory"
    )
    args = parser.parse_args()
    gen_annotated(args.native_functions, args.tags, args.out, args.autograd)


if __name__ == "__main__":
    main()
<|endoftext|>from __future__ import annotations

import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import kernel_signature
from torchgen.context import with_native_function_and_index
from torchgen.model import BackendIndex, NativeFunction, NativeFunctionsGroup
from torchgen.utils import mapMaybe


@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> str | None:
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None
    else:
        prefix = "static" if backend_index.external else "TORCH_API"
        return f"{prefix} {sig.decl(name=metadata.kernel)};"


@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> list[str]:
    meta_name = meta.name(g)
    out_args = structured.impl_arguments(g)
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    prefix = "" if backend_index.external else "TORCH_API "
    return [
        f"""\
struct {prefix}structured_{metadata.kernel} : public at::meta::structured_{meta_name} {{
void impl({", ".join(a.decl() for a in out_args)});
}};
"""
    ]


# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function_and_index
def compute_native_function_declaration(
    g: NativeFunctionsGroup | NativeFunction, backend_index: BackendIndex
) -> list[str]:
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            if backend_index.external:
                # Structured hasn't been tested with external backends yet.
                raise AssertionError(
                    "Structured external backend functions are not implemented yet."
                )
            else:
                return gen_structured(g, backend_index)
        else:
            return list(
                mapMaybe(lambda f: gen_unstructured(f, backend_index), g.functions())
            )
    else:
        x = gen_unstructured(g, backend_index)
        return [] if x is None else [x]
<|endoftext|>from torchgen.dest.lazy_ir import (
    generate_non_native_lazy_ir_nodes as generate_non_native_lazy_ir_nodes,
    GenLazyIR as GenLazyIR,
    GenLazyNativeFuncDefinition as GenLazyNativeFuncDefinition,
    GenLazyShapeInferenceDefinition as GenLazyShapeInferenceDefinition,
)
from torchgen.dest.native_functions import (
    compute_native_function_declaration as compute_native_function_declaration,
)
from torchgen.dest.register_dispatch_key import (
    gen_registration_headers as gen_registration_headers,
    gen_registration_helpers as gen_registration_helpers,
    RegisterDispatchKey as RegisterDispatchKey,
)
from torchgen.dest.ufunc import (
    compute_ufunc_cpu as compute_ufunc_cpu,
    compute_ufunc_cpu_kernel as compute_ufunc_cpu_kernel,
    compute_ufunc_cuda as compute_ufunc_cuda,
)
<|endoftext|>from __future__ import annotations

import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any

import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
    getValueT,
    isValueType,
    LazyArgument,
    LazyIrProperties,
    LazyIrSchema,
    tensorListValueT,
)
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    deviceT,
    DispatcherSignature,
    kernel_signature,
    NativeSignature,
    OptionalCType,
    VectorCType,
)
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
)


def node_ctor_arg_rvalue_string(arg: LazyArgument) -> str:
    """
    Given a LazyArgument,
    generate a c++ string for materializing an rvalue of that arg for passing into
    a lazy Node constructor.
    """

    # TODO: Matching on CType seems wrong; should be matching on Type
    if isValueType(arg.lazy_type):
        if isinstance(arg.lazy_type, BaseCType):
            if arg.is_wrapped_scalar:
                return f"node_{arg.name}"
            elif arg.lazy_type.type is tensorListValueT:
                return f"lazy_{arg.name}_tensorlist"
            elif arg.is_symint_or_list:
                return f"GetSymIntValue({arg.name})"
            return f"lazy_{arg.name}->GetIrValue()"
        elif isinstance(arg.lazy_type, OptionalCType):
            if arg.is_symint_or_list:
                # TODO: I don't understand when you should put lazy_ in the name
                # or not
                return f"{arg.name} ? std::make_optional(GetSymIntValue(*{arg.name})) : ::std::nullopt"
            elif arg.is_wrapped_scalar:
                return f"node_{arg.name}"
            return (
                f"lazy_{arg.name} ? "
                f"std::make_optional(lazy_{arg.name}->GetIrValue()) : "
                "::std::nullopt"
            )
        else:
            raise AssertionError(
                f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})"
            )
    else:
        # NB: this is here because right now we aren't treating SymInt[] as a
        # value type; when we do this needs to move above
        # NB: we cannot test arg.lazy_type as we've already specified it is an
        # int64_t and so we cannot distinguish between SymInt and int64_t
        if isinstance(arg.orig_type, ListType) and arg.orig_type.elem == BaseType(
            BaseTy.SymInt
        ):
            if arg.symint:
                return f"GetSymIntArrayRefValue({arg.name})"
            else:
                return f"std::vector<int64_t>({arg.name}.begin(), {arg.name}.end())"
        elif isinstance(arg.lazy_type, VectorCType) and isinstance(
            arg.lazy_type.elem, BaseCType
        ):
            return f"std::vector<{arg.lazy_type.elem.type}>({arg.name}.begin(), {arg.name}.end())"
        elif (
            isinstance(arg.lazy_type, OptionalCType)
            and isinstance(arg.lazy_type.elem, VectorCType)
            and isinstance(arg.lazy_type.elem.elem, BaseCType)
        ):
            return f"torch::lazy::ToOptionalVector<{arg.lazy_type.elem.elem.type}>({arg.name})"
        else:
            return f"{arg.name}"


def node_ctor_inputs(schema: LazyIrSchema) -> str:
    """
    Produce a formatted string with the arguments as passed into the constructor of a node class.
    """
    node_ctor_values = [
        node_ctor_arg_rvalue_string(arg) for arg in schema.filtered_args()
    ]
    return ", ".join(node_ctor_values)


def gen_fallback_code(
    schema: LazyIrSchema,
    sig: DispatcherSignature | NativeSignature,
    overload_name: str,
) -> str:
    """
    Generate code that falls back to eager conditioned on a predicate
    """
    dispatcher_sig = DispatcherSignature.from_schema(schema.func)
    exprs = translate(sig.arguments(), dispatcher_sig.arguments())
    fallback_args = ",\n                ".join([a.expr for a in exprs])
    if len(overload_name):
        aten_op_str = f"ATEN_OP2({schema.aten_name}, {overload_name})"
    else:
        aten_op_str = f"ATEN_OP({schema.aten_name})"
    return f"""
        if (force_eager_fallback({aten_symbol(schema)})) {{
            return at::native::call_fallback_fn_symint<&ltc_eager_fallback, {aten_op_str}>::call(
                {fallback_args}
            );
        }}
"""


def aten_symbol(schema: LazyIrSchema) -> str:
    missing_interned_strings = {
        "sigmoid_backward",
    }
    if schema.aten_name in missing_interned_strings:
        return f'c10::Symbol::fromQualString("aten::{schema.aten_name}")'

    if not schema.aten_name.startswith("at::"):
        return f"at::aten::{schema.aten_name}"
    else:
        return schema.aten_name


# converts  all tensor-like arguments to meta tensors. Returns:
# (1) a string containing all of the logic that does the conversions.
# (2) a context, to be used by translate(), with all of the relevant bindings.
def convert_to_meta_tensors(sig: DispatcherSignature) -> tuple[str, list[Binding]]:
    context: list[Binding] = []
    unwrapped_tensor_args: list[str] = []
    for arg in sig.arguments():
        if isinstance(arg.argument, Argument) and arg.argument.type.is_tensor_like():
            unwrapped_name = f"{arg.name}_meta"
            unwrapped_tensor_args.append(
                f"auto {unwrapped_name} = to_meta({arg.name});"
            )
            context.append(arg.with_name(unwrapped_name))
        else:
            context.append(arg)
    unwrap_tensor_args_str = "\n        ".join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context


@dataclass(frozen=True)
class GenLazyIR(ABC):
    backend_index: BackendIndex
    backend_name: str
    node_base: str
    use_lazy_shape: bool

    @method_with_native_function
    def __call__(self, f: NativeFunctionsGroup | NativeFunction) -> list[str]:
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        metadata = self.backend_index.get_kernel(
            f.functional if isinstance(f, NativeFunctionsGroup) else f
        )
        schema = LazyIrSchema(
            func, symint=metadata is not None and metadata.supports_symint()
        )
        return self.gen(schema)

    # there is no lowering functionality generated unless this IR base class is subclassed and
    # implemented as a backend-specific node
    def lowering_function(self, schema: LazyIrSchema) -> str:
        return ""

    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        return ""

    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        return f"""bool CanBeReused({node_ctor_args}) const {{
    return false;
    }}"""

    def node_base_ctor_call(self, schema: LazyIrSchema) -> str:
        value_args = schema.filtered_args(values=True, scalars=False)
        # backends can customize the way the node base class constructor is called,
        # as long as all of its arguments can be generated from information available from the schema
        base_ctor_value_args_list = []
        for arg in value_args:
            if isinstance(arg.lazy_type, (BaseCType, VectorCType)):
                base_ctor_value_args_list.append(f"{arg.name}")
            elif isinstance(arg.lazy_type, OptionalCType):
                base_ctor_value_args_list.append(f"{arg.name}.value_or(kNullValue)")
            else:
                raise AssertionError(
                    f"Unsupported type ({arg.lazy_type}) - add support if necessary"
                )
        base_ctor_value_args = ", ".join(base_ctor_value_args_list)

        scalar_args = schema.filtered_args(values=False, scalars=True)

        # Shape construction.
        # Conditionally build shape depending on specified shape property
        if schema.properties.ShapePrecompute:
            shape_ctor_arg = "std::move(shapes),"
        elif schema.properties.ShapeCompute:
            shape_args = [a.name for a in value_args]
            shape_args.extend(a.name for a in scalar_args)
            shape_ctor_arg = f"compute_shape_{schema.name}({', '.join(shape_args)}),"
        elif schema.properties.ShapeCache:
            shape_args = [f"operand({i})" for i in range(len(value_args))]
            shape_args.extend(a.name for a in scalar_args)
            shape_ctor_arg = f"[&](){{ return compute_shape_{schema.name}({', '.join(shape_args)})[0]; }},"
        else:
            shape_ctor_arg = ""

        scalar_hashes = ", ".join(f"{a.name}" for a in scalar_args)

        return f"""{self.node_base}(
              {schema.node_name}::ClassOpKind(),
              OpList{{{base_ctor_value_args}}},
              {shape_ctor_arg}
              /* num_outputs */ {len(schema.returns)},
              torch::lazy::MHash({scalar_hashes}))"""

    def gen(self, schema: LazyIrSchema) -> list[str]:
        opkind = schema.opkind or aten_symbol(schema)

        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        all_args = schema.filtered_args()
        scalar_args = schema.filtered_args(values=False, scalars=True)

        ctor_args = [f"const {i.lazy_type.cpp_type()}& {i.name}" for i in all_args]
        reuse_ctor_args = ", ".join(ctor_args)
        if self.use_lazy_shape and schema.properties.ShapePrecompute:
            ctor_args.append("std::vector<torch::lazy::Shape>&& shapes")
        node_ctor_args = ", ".join(ctor_args)

        scalar_initializers = ",\n        ".join(
            [
                # This code is just special casing the mapping from string_view -> strings
                f"{a.name}({a.name}.has_value() ? ::std::make_optional(std::string(*{a.name})) : ::std::nullopt)"
                if a.lazy_type.cpp_type() == "::std::optional<c10::string_view>"
                else f"{a.name}({a.name})"
                for a in scalar_args
            ]
        )
        if len(scalar_initializers):
            scalar_initializers = f",\n        {scalar_initializers}"
        scalar_decls = "\n  ".join(
            [
                f"std::string {a.name};"
                if a.lazy_type.cpp_type() == "c10::string_view"
                else f"::std::optional<std::string> {a.name};"
                if a.lazy_type.cpp_type() == "::std::optional<c10::string_view>"
                else f"{a.lazy_type.cpp_type()} {a.name};"
                for a in scalar_args
            ]
        )
        optional_values = [
            arg.name
            for arg in schema.filtered_args(values=True, scalars=False)
            if isinstance(arg.lazy_type, OptionalCType)
        ]
        has_optional_decls = "\n  ".join(
            [f"bool has_{value}: 1;" for value in optional_values]
        )
        has_optional_defs = "\n    ".join(
            [f"has_{value} = !!{value};" for value in optional_values]
        )
        members_to_string = []
        for arg in scalar_args:
            if isinstance(arg.lazy_type, OptionalCType):
                value = f"{arg.name}.value()"
                if arg.is_generator:
                    value = '"torch.Generator()"'
                members_to_string.append(
                    f"""if ({arg.name}.has_value()) {{
      ss << ", {arg.name}=" << {value};
    }} else {{
      ss << ", {arg.name}=null";
    }}"""
                )
            else:
                members_to_string.append(f'ss << ", {arg.name}=" << {arg.name};')
        members_to_string_str = "\n    ".join(members_to_string)

        return [
            f"""\
class {schema.node_name} : public {self.node_base} {{
 public:
  static torch::lazy::OpKind ClassOpKind() {{
    return torch::lazy::OpKind({opkind});
  }}

  {schema.node_name}({node_ctor_args})
      : {self.node_base_ctor_call(schema)}{scalar_initializers}
  {{
    {has_optional_defs}
  }}

  std::string ToString() const override {{
    std::stringstream ss;
    ss << {self.node_base}::ToString();
    {members_to_string_str}
    return std::move(ss).str();
  }}

  {self.create_function(schema, reuse_ctor_args)}

  {self.can_be_reused_function(schema, reuse_ctor_args)}

  {self.lowering_function(schema)}

  {scalar_decls}
  {has_optional_decls}

}};

""",
        ]


@dataclass(frozen=True)
class GenTSLazyIR(GenLazyIR):
    def lowering_function(self, schema: LazyIrSchema) -> str:
        signature = """
  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      torch::lazy::TSLoweringContext* loctx) const override"""

        if schema.properties.LowerDeclOnly:
            return f"{signature};"
        elif schema.properties.Lower:
            return f"""{signature} {{
    {ts_lowering_body(schema)}
  }}
            """
        else:
            return ""

    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        signature = f"static NodePtr Create({node_ctor_args})"
        if schema.properties.CreateFnDeclOnly:
            return f"{signature};"
        elif not schema.properties.CreateFn:
            return ""
        return f"""{signature} {{
    return ReuseOrMakeNode<{schema.node_name}>(data);
  }}"""

    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        signature = f"bool CanBeReused({node_ctor_args}) const"
        if schema.properties.CanBeReusedDeclOnly:
            return f"{signature};"
        elif not schema.properties.CanBeReused:
            return ""
        value_comparison = []
        for arg in itertools.chain(schema.positional_values, schema.keyword_values):
            if isinstance(arg.lazy_type, OptionalCType):
                value_comparison.append(
                    f"nullable_operand(i++) == {arg.name}.value_or(kNullValue)"
                )
            else:
                value_comparison.append(f"operand(i++) == {arg.name}")
        for arg in itertools.chain(schema.positional_scalars, schema.keyword_scalars):
            if isinstance(arg.lazy_type, OptionalCType):
                value_comparison.append(
                    f"((!this->{arg.name}&&!{arg.name}) || (this->{arg.name}&&{arg.name} && *(this->{arg.name}) == *{arg.name}))"
                )
            else:
                value_comparison.append(f"this->{arg.name} == {arg.name}")
        value_comparison_str = " &&\n        ".join(value_comparison)

        return f"""{signature} {{
    size_t i = 0;
    return ({value_comparison_str});
  }}"""


@dataclass(frozen=True)
class GenLazyNativeFuncDefinition:
    class_method_name: str
    backend_index: BackendIndex
    tensor_class: str
    gen_forced_fallback_code: bool
    backend_namespace: str
    get_tensorlist: str
    get_tensor_or_wrap_number: str
    try_get_tensor: str
    metrics_counter: str
    create_tensor: str
    create_from_first_tensor: bool
    create_aten_from_ltc_tensor: str
    tuple_aten_from_ltc_tensors: str
    lazy_tensor_ptr: str
    get_device_fn: str

    def lazy_tensor_decls(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        value_args = schema.filtered_args(values=True, scalars=False)
        # Generates lazy_{name} variables for LazyTensors wrapping input tensors
        lazy_tensor_decls: list[str] = []
        for arg in value_args:
            if arg.is_wrapped_scalar:
                if isinstance(arg.lazy_type, OptionalCType):
                    lazy_tensor_decls.append(
                        f"""auto node_{arg.name} = {arg.name} ?
                std::make_optional(torch::lazy::LazyGraphExecutor::Get()->
                    GetIrValueForScalarFromCodegen(*{arg.name}, *common_device)):
                ::std::nullopt;"""
                    )
                else:
                    lazy_tensor_decls.append(
                        f"""auto node_{arg.name} = torch::lazy::LazyGraphExecutor::Get()->
                            GetIrValueForScalarFromCodegen({arg.name}, *common_device);"""
                    )
            elif arg.is_symint_or_list:
                continue  # values are extracted in isValueType
            elif isinstance(arg.lazy_type, BaseCType):
                if arg.lazy_type.type is tensorListValueT:
                    lazy_tensor_decls.append(
                        f"auto lazy_{arg.name}_tensorlist = "
                        f"{self.backend_namespace}::{self.get_tensorlist}({arg.name});"
                    )
                else:
                    lazy_tensor_decls.append(
                        f"{self.lazy_tensor_ptr} lazy_{arg.name} = "
                        f"{self.backend_namespace}::{self.get_tensor_or_wrap_number}({arg.name}, *common_device);"
                    )
            elif isinstance(arg.lazy_type, OptionalCType):
                if arg.lazy_type.elem != BaseCType(getValueT()):
                    raise AssertionError(
                        f"Expected OptionalCType elem to be {BaseCType(getValueT())}, "
                        f"got {arg.lazy_type.elem}"
                    )
                # TODO(alanwaketan): Maybe we want to apply GetLtcTensorOrCreateForWrappedNumber here, but hold it
                # until we encounter a real world example.
                lazy_tensor_decls.append(
                    f"{self.lazy_tensor_ptr} lazy_{arg.name} = "
                    f"{self.backend_namespace}::{self.try_get_tensor}({arg.name}.value_or(at::Tensor()));"
                )
            else:
                raise AssertionError(
                    f"TODO not sure if there are other valid types to handle here ({arg.lazy_type})"
                )
        return ("\n        ").join(lazy_tensor_decls)

    def force_eager_fallback(
        self,
        func: NativeFunction,
        schema: LazyIrSchema,
        metadata: BackendMetadata,
        sig: DispatcherSignature | NativeSignature,
    ) -> str:
        if self.gen_forced_fallback_code:
            return gen_fallback_code(
                schema, sig, overload_name=func.func.name.overload_name
            )
        return ""

    def metrics(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        return f"{self.metrics_counter};"

    def get_device(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        value_args = schema.filtered_args(values=True, scalars=False)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        value_types_names = [f"{a.name}" for a in value_args if not a.is_wrapped_scalar]
        optional_device = OptionalCType(BaseCType(deviceT))
        optional_devices = [
            a.name for a in scalar_args if a.lazy_type == optional_device
        ]
        if len(value_types_names) == 0 and len(optional_devices) == 0:
            raise AssertionError("Expected at least one Value or Device type")
        get_device_str = (
            f"{self.get_device_fn}({', '.join(value_types_names + optional_devices)})"
        )
        return f"""auto common_device = {get_device_str};
        TORCH_INTERNAL_ASSERT(common_device);
        """

    def shape_inference(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        metadata = self.backend_index.get_kernel(func)
        if metadata is None:
            raise AssertionError(f"No kernel metadata found for {func.func.name}")
        all_args = schema.filtered_args()
        returns_length = len(schema.returns)
        # call the meta kernel if it exists, to compute output shape/dtype for our IR
        # Note [Generated LTC Shape Functions]
        # LTC uses meta tensors from core to do shape inference when possible, and otherwise
        # we generate a shape function declaration that needs to be manually implemented.
        # How do we detect which ops are eligible to use meta tensors?
        # In general we should be able to use meta tensors not just on structured operators,
        # but also on composite operators that are implemented in terms of structured kernels.
        # We don't currently have a way of knowing at codegen time which ops are implemented that way.
        # This is the case for all view and view_copy operators however, so we're going to
        # use them specifically for all of the view_copy ops (instead of manually writing shape rules for all of them).
        is_view_copy_op = "view_copy" in func.tags
        is_structured = func.structured or func.structured_delegate is not None
        if is_structured or is_view_copy_op:
            meta_out = """
std::vector<torch::lazy::Shape> shapes{torch::lazy::Shape(out_meta.scalar_type(), out_meta.sizes().vec())};"""
            if returns_length > 1:

                def this_shape(i: int) -> str:
                    return f"torch::lazy::Shape(std::get<{i}>(out_meta).scalar_type(), std::get<{i}>(out_meta).sizes().vec())"

                shapes_str = ",".join([this_shape(i) for i in range(returns_length)])
                meta_out = "std::vector<torch::lazy::Shape> shapes{" + shapes_str + "};"

            # Convert tensor args to the meta device and call it.
            # (We can't pass in the input tensors directly, because they are "functional wrappers".
            # If any of the meta kernels call a tensor op and redispatch, we don't want to hit the functionalize kernels.)
            # Even at::meta:: functions might redispatch, e.g. if they call into view ops.
            dispatcher_sig = DispatcherSignature.from_schema(func.func)
            meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
            meta_call_args = [
                e.expr
                for e in translate(
                    meta_call_ctx, dispatcher_sig.arguments(), method=False
                )
            ]
            if is_view_copy_op:
                # view_copy ops always have a CompositeExplicitAutogradNonFunctional kernel
                if not func.has_composite_explicit_autograd_non_functional_kernel:
                    raise AssertionError(
                        f"view_copy op {func.func.name} must have "
                        "CompositeExplicitAutogradNonFunctional kernel"
                    )
                dispatch_ns = "compositeexplicitautogradnonfunctional"
            else:
                dispatch_ns = "meta"
            aten_name = schema.aten_name
            # TODO: this is trolling
            if func.func.has_symint() and metadata.supports_symint():
                aten_name += "_symint"
            shape_str = f"""\
        {meta_conversion_str}
        auto out_meta = at::{dispatch_ns}::{aten_name}({", ".join(meta_call_args)});
        {meta_out}"""
        else:
            shape_sig = ComputeShapeSignature(
                metadata.kernel, func, symint=metadata.supports_symint()
            )
            shape_str = f"""
            auto shapes = {shape_sig.shape_call};"""

        shape_str += f"""
            TORCH_INTERNAL_ASSERT(shapes.size() == {returns_length});"""

        # Calculating which dimensions are symbolic
        func_schema_str = "aten::" + str(func.func)
        shape_str += f"""
            if(torch::lazy::symbolicShapeEnabled()){{
                std::vector<torch::jit::IValue> inputs = {{ {", ".join(str(a.name) for a in all_args)} }};
                const char* schema_str = "{func_schema_str}";
                applySymbolicShapesOnLT(schema_str, inputs, shapes);
            }}
        """
        return shape_str

    def build_ir_node(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        node_ctor_input_str = node_ctor_inputs(schema)
        return f"""torch::lazy::NodePtr node = torch::lazy::ReuseNode<{schema.node_name}>({node_ctor_input_str});
        if (!node) {{
            {self.shape_inference(func, schema)}
            node = torch::lazy::MakeNode<{schema.node_name}>({node_ctor_input_str}, std::move(shapes));
            CacheNode(node);
        }}
        """

    def create_lazy_tensor(self, first_tensor_name: str | None = None) -> str:
        # xla uses an instance method for tensor creation, for the time being
        if self.create_from_first_tensor:
            # TODO(whc) remove this if XLA switches to using static method for creation
            if first_tensor_name is None:
                raise AssertionError("Requires first tensor to create lazy tensor")
            return f"{first_tensor_name}.{self.create_tensor}"
        return f"{self.backend_namespace}::{self.create_tensor}"

    def return_aten_tensor(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        returns_length = len(schema.returns)
        value_args = schema.filtered_args(values=True, scalars=False)
        value_types_names = [f"{a.name}" for a in value_args if not a.is_wrapped_scalar]
        first_tensor_name = value_types_names[0] if len(value_types_names) > 0 else None
        bridge_str = f"""auto result = {self.create_aten_from_ltc_tensor}(
                {self.create_lazy_tensor(first_tensor_name)}(std::move(node), *common_device));"""

        if returns_length > 1:
            if len(value_types_names) == 0:
                raise AssertionError(
                    "Code below assumes there is at least one tensor arg"
                )
            bridge_str = f"""std::vector<{self.lazy_tensor_ptr}> lazy_tensors;
        for (int i = 0; i < {returns_length}; i++) {{
            lazy_tensors.push_back({self.create_lazy_tensor(first_tensor_name)}({getValueT()}(node, i), *common_device));
        }}
        auto result = {self.tuple_aten_from_ltc_tensors}<{returns_length}>(lazy_tensors);"""

        if schema.name.name.inplace or func.func.is_out_fn():
            if returns_length != 1:
                raise AssertionError(
                    "We assumed there was no such case where an op is an in-place variant "
                    f"and has tuple outputs, but got tuple of len {returns_length}."
                )
            bridge_str = f"""lazy_{first_tensor_name}->SetInPlaceIrValue(node);
        auto& result = {first_tensor_name};"""

        bridge_str += """
        return result;"""
        return bridge_str

    @method_with_native_function
    def __call__(self, func: NativeFunction) -> list[str]:
        sig = kernel_signature(func, self.backend_index)
        metadata = self.backend_index.get_kernel(func)
        if metadata is None:
            raise AssertionError(f"No kernel metadata found for {func.func.name}")
        schema = LazyIrSchema(func.func, symint=metadata.supports_symint())
        return [
            f"""\
    {sig.decl(name=f"{self.class_method_name}::{metadata.kernel}")} {{
        {self.force_eager_fallback(func, schema, metadata, sig)}
        {self.metrics(func, schema)}
        {self.get_device(func, schema)}
        {self.lazy_tensor_decls(func, schema)}
        {self.build_ir_node(func, schema)}
        {self.return_aten_tensor(func, schema)}
    }}\n
    """
        ]


class ComputeShapeSignature:
    """
    Here we use the base name as the suffix of the signature to avoid generating for in-place variants.
    """

    def __init__(self, kernel_name: str, f: NativeFunction, *, symint: bool) -> None:
        self.__schema = LazyIrSchema(f.func, symint=symint)
        self.__dispatch_args = ", ".join(
            [a.decl() for a in dispatcher.arguments(f.func, symint=symint)]
        )
        self.__call_args = ", ".join(
            [f"{arg.name}" for arg in self.__schema.filtered_args(generator=True)]
        )
        self.__kernel_name = kernel_name

    def __decl_suffix(self) -> str:
        return f"{self.__kernel_name}({self.__dispatch_args})"

    def __call_suffix(self) -> str:
        return f"{self.__kernel_name}({self.__call_args})"

    @property
    def shape_decl(self) -> str:
        return f"TORCH_API std::vector<torch::lazy::Shape> compute_shape_{self.__decl_suffix()}"

    @property
    def shape_call(self) -> str:
        return f"torch::lazy::compute_shape_{self.__call_suffix()}"


@dataclass(frozen=True)
class GenLazyShapeInferenceDefinition:
    backend_index: BackendIndex
    tensor_class: str

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> list[str]:
        metadata = self.backend_index.get_kernel(f)
        if metadata is None:
            raise AssertionError(f"No kernel metadata found for {f.func.name}")

        # See Note [Generated LTC Shape Functions]
        is_view_copy_op = "view_copy" in f.tags
        is_structured = f.structured or f.structured_delegate is not None
        if is_structured or is_view_copy_op:
            return []
        else:
            shape_sig = ComputeShapeSignature(
                metadata.kernel, f, symint=metadata.supports_symint()
            )
            return ["\n".join([f"{shape_sig.shape_decl};"])]


def generate_non_native_lazy_ir_nodes(
    non_native: list[dict[str, Any]], gen_lazy_ir: GenLazyIR
) -> list[str]:
    """Generate the non-native lazy IR node classes"""
    nodes = []
    for op in non_native:
        # Set default properties for Non-Native IRs
        properties = LazyIrProperties("ShapeCache", "CanBeReused", "LowerDeclOnly")
        for p in op.get("properties", []):
            setattr(properties, p, True)

        # non-native is assumed to want symint bindings if you wrote symint
        schema = LazyIrSchema(FunctionSchema.parse(op["func"]), properties, symint=True)
        schema.opkind = op.get("opkind")
        nodes.append(gen_lazy_ir.gen(schema)[0])

    return nodes
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    CType,
    Expr,
    NamedCType,
    opmath_t,
    scalar_t,
    StructuredImplSignature,
    VectorizedCType,
)
from torchgen.context import with_native_function
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    DispatchKey,
    NativeFunctionsGroup,
    ScalarType,
    UfuncKey,
)
from torchgen.utils import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchgen.api.ufunc import UfunctorBindings


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                                  CUDA STUFF
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# NB: not bothering to generate dispatch stub forward declaration in header,
# we can just paste it wherever necessary

# TODO: use BackendIndex
# dispatch_key: DispatchKey  # only CPU/CUDA right now


# Represents functors for implementing CUDA ufuncs.
# Functors are templated by scalar_t because when USERS instantiate functors
# they are templated.  A functor looks something like this:
#
#   template <typename scalar_t>
#   struct CUDAFunctorOnSelf_add {
#     using opmath_t = at::opmath_type<scalar_t>;
#     opmath_t other_;
#     opmath_t alpha_;
#     CUDAFunctorOnSelf_add(opmath_t other, opmath_t alpha)
#         : other_(other), alpha_(alpha) {}
#     __device__ scalar_t operator()(scalar_t self) {
#       return ufunc::add(static_cast<opmath_t>(self), other_, alpha_);
#     }
#   };
#
@dataclass(frozen=True)
class UfunctorSignature:
    g: NativeFunctionsGroup
    scalar_tensor_idx: int | None
    name: str

    def arguments(self) -> UfunctorBindings:
        return ufunc.ufunctor_arguments(
            self.g, scalar_tensor_idx=self.scalar_tensor_idx, scalar_t=scalar_t
        )

    def fields(self) -> list[Binding]:
        # fields are renamed to have a trailing underscore, as is conventional
        return [b.rename(f"{b.name}_") for b in self.arguments().ctor]

    def returns_type(self) -> CType:
        # TODO: don't hardcode; return type will be inferred based on tags on
        # the native function
        return BaseCType(scalar_t)

    def decl_fields(self) -> str:
        return "\n".join(f"{f.type} {f.name};" for f in self.fields())

    def inline_defn_ctor(self) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments().ctor)
        # NB: hypothetically could do this with translate but the
        # transition here is very regular
        init_str = ", ".join(f"{a.name}_({a.name})" for a in self.arguments().ctor)
        return f"{self.name}({args_str}) : {init_str} {{}}"

    def decl_apply(self) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments().apply)
        return f"{self.returns_type().cpp_type()} operator()({args_str}) const"


@dataclass(frozen=True)
class UfuncSignature:
    g: NativeFunctionsGroup
    name: str
    compute_t: CType

    def arguments(self) -> list[Binding]:
        return ufunc.ufunc_arguments(self.g, compute_t=self.compute_t)

    def call(self, ctx: Sequence[Binding | Expr]) -> str:
        return f"{self.name}({', '.join(a.expr for a in translate(ctx, self.arguments()))})"


# steps:
#   1. take the functional signature
#   2. use api.ufunc to convert it to template signature.  this establishes
#      the type of the template function
#   3. use api.ufunc (II) to generate a split struct / operator() signature.
#      this establish context in which we call the template signature
#
# StructuredImplSignature context
#   ~> functor constructor sig
#
# Functor constructor context
#   ~> functor fields sig
#
# Functor apply context (functor fields + functor apply sig)
#   ~> template sig
#


def eligible_for_binary_scalar_specialization(g: NativeFunctionsGroup) -> bool:
    num_tensors = sum(
        1 for a in g.functional.func.arguments.flat_non_out if a.type.is_tensor_like()
    )
    return num_tensors == 2


def compute_ufunc_cuda_functors(
    g: NativeFunctionsGroup,
) -> tuple[dict[ScalarType, dict[UfuncKey, UfunctorSignature]], str]:
    # First, build the functors.
    ufunctor_sigs: dict[ScalarType, dict[UfuncKey, UfunctorSignature]] = {}
    ufunctors: list[str] = []
    loops = g.out.ufunc_inner_loop
    scalar_tensor_idx_lookup = {
        UfuncKey.CUDAFunctorOnSelf: 1,
        UfuncKey.CUDAFunctorOnOther: 0,
        UfuncKey.CUDAFunctor: None,
    }
    if eligible_for_binary_scalar_specialization(g):
        keys = [
            UfuncKey.CUDAFunctorOnSelf,
            UfuncKey.CUDAFunctorOnOther,
            UfuncKey.CUDAFunctor,
        ]
    else:
        keys = [UfuncKey.CUDAFunctor]
        for k in [UfuncKey.CUDAFunctorOnSelf, UfuncKey.CUDAFunctorOnOther]:
            if k in loops:
                raise AssertionError(f"cannot use {k} on non-binary function")
    for k in keys:
        # If the key was directly defined, skip functor codegen; we assume the
        # user already done it for us
        if k in loops:
            ufunctor_sig = UfunctorSignature(
                g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=loops[k].name
            )
            for dtype in loops[k].supported_dtypes:
                ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig
            continue

        # Note [ScalarOnly and Generic must match names for CUDA]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Otherwise, look in ANY of the generic entries.  For simplicity of
        # codegen, both ScalarOnly and Generic are defined, the ufunc name
        # must match  (if they didn't match, we'd have to generate distinct
        # functors per dtype, which is awful, so we're not going to do it unless
        # someone really forces us to)
        ufunc_name = None
        supported_dtypes: OrderedSet[ScalarType] = OrderedSet()
        for lk in [UfuncKey.ScalarOnly, UfuncKey.Generic]:
            if lk not in loops:
                continue
            if ufunc_name is None:
                ufunc_name = loops[lk].name
            else:
                # See Note [ScalarOnly and Generic must match names for CUDA]
                if ufunc_name != loops[lk].name:
                    raise AssertionError(
                        "ScalarOnly and Generic must have same ufunc name"
                    )
            supported_dtypes |= loops[lk].supported_dtypes
        if ufunc_name is None:
            raise AssertionError("ufunc_name must be non-None")

        name = f"{k}_{ufunc_name}"
        ufunctor_sig = UfunctorSignature(
            g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=name
        )
        for dtype in supported_dtypes:
            ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig

        ufunc_sig = UfuncSignature(
            g, name=f"ufunc::{ufunc_name}", compute_t=BaseCType(opmath_t)
        )
        apply_ctx = ufunctor_sig.fields() + ufunctor_sig.arguments().apply
        ufunctors.append(
            f"""
template <typename scalar_t>
struct {ufunctor_sig.name} {{
  using opmath_t = at::opmath_type<scalar_t>;
  {ufunctor_sig.decl_fields()}
  {ufunctor_sig.inline_defn_ctor()}
  __device__ {ufunctor_sig.decl_apply()} {{
    return {ufunc_sig.call(apply_ctx)};
  }}
}};
"""
        )

    return ufunctor_sigs, "\n".join(ufunctors)


@dataclass(frozen=True)
class BinaryScalarSpecializationConfig:
    scalar_idx: int
    ctor_tensor: str
    ufunc_key: UfuncKey


BinaryScalarSpecializationConfigs = [
    BinaryScalarSpecializationConfig(
        scalar_idx=0,
        ctor_tensor="self",
        ufunc_key=UfuncKey.CUDAFunctorOnOther,
    ),
    BinaryScalarSpecializationConfig(
        scalar_idx=1,
        ctor_tensor="other",
        ufunc_key=UfuncKey.CUDAFunctorOnSelf,
    ),
]


def compute_ufunc_cuda_dtype_body(
    g: NativeFunctionsGroup,
    dtype: ScalarType,
    inner_loops: dict[UfuncKey, UfunctorSignature],
    parent_ctx: Sequence[Binding],
) -> str:
    body = "using opmath_t = at::opmath_type<scalar_t>;"
    body += "if (false) {}\n"  # for ease of codegen
    for config in BinaryScalarSpecializationConfigs:
        if config.ufunc_key not in inner_loops:
            continue
        ufunctor_sig = inner_loops[config.ufunc_key]
        scalar_idx = config.scalar_idx + 1
        # Make a copy and at the same time widen the type (not permissible
        # without copy; we don't want to mutate the input argument anyway)
        ctx: list[Expr | Binding] = list(parent_ctx)
        ctx.append(
            Expr(
                expr=f"iter.scalar_value<opmath_t>({scalar_idx})",
                type=NamedCType(config.ctor_tensor, BaseCType(opmath_t)),
            )
        )
        ufunctor_ctor_exprs_str = ", ".join(
            a.expr for a in translate(ctx, ufunctor_sig.arguments().ctor)
        )

        # NB: ufunctor must be allocated before iter.remove_operand is called,
        # as it relies on iter
        body += f"""\
else if (iter.is_cpu_scalar({scalar_idx})) {{
  {ufunctor_sig.name}<scalar_t> ufunctor({ufunctor_ctor_exprs_str});
  iter.remove_operand({scalar_idx});
  gpu_kernel(iter, ufunctor);
}}"""

    ufunctor_sig = inner_loops[UfuncKey.CUDAFunctor]
    ufunctor_ctor_exprs_str = ", ".join(
        a.expr for a in translate(parent_ctx, ufunctor_sig.arguments().ctor)
    )
    body += f"""
else {{
  gpu_kernel(iter, {ufunctor_sig.name}<scalar_t>({ufunctor_ctor_exprs_str}));
}}
    """
    return body


@with_native_function
def compute_ufunc_cuda(g: NativeFunctionsGroup) -> str:
    # First, build the functors, indexing them by dtype
    ufunctor_sigs, ufunctors = compute_ufunc_cuda_functors(g)

    # Next, build the conditionals
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CUDA))
    dtype_cases = []
    for dtype, inner_ufunc_sigs in ufunctor_sigs.items():
        dtype_cases.append(
            f"""
AT_DISPATCH_CASE(at::ScalarType::{dtype},
  [&]() {{
    {compute_ufunc_cuda_dtype_body(g, dtype, inner_ufunc_sigs, sig.arguments())}
  }}
)
"""
        )

    dtype_cases_str = "\n".join(dtype_cases)

    stub_sig = StubSignature(g)

    return f"""
{ufunctors}

{stub_sig.type_defn()};
{stub_sig.dispatch_decl()}

{stub_sig.kernel_defn()} {{
  AT_DISPATCH_SWITCH(iter.common_dtype(), "{sig.name}",
    {dtype_cases_str}
  );
}}
REGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name})

{sig.defn()} {{
  {stub_sig.direct_call(sig.arguments())};
}}
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                                   CPU STUFF
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@dataclass(frozen=True)
class StubSignature:
    g: NativeFunctionsGroup

    @property
    def name(self) -> str:
        return f"{str(self.g.functional.func.name.name)}_stub"

    @property
    def kernel_name(self) -> str:
        return f"{str(self.g.functional.func.name.name)}_kernel"

    @property
    def type_name(self) -> str:
        return f"{str(self.g.functional.func.name.name)}_fn"

    def arguments(self) -> list[Binding]:
        return ufunc.stub_arguments(self.g)

    def type(self) -> str:
        cpp_args = self.arguments()
        return f"void(*)(TensorIteratorBase&, {', '.join(a.type for a in cpp_args)})"

    def dispatch_decl(self) -> str:
        return f"DECLARE_DISPATCH({self.type_name}, {self.name})"

    def dispatch_defn(self) -> str:
        return f"DEFINE_DISPATCH({self.name})"

    def kernel_defn(self) -> str:
        return f"void {self.kernel_name}(TensorIteratorBase& iter, {', '.join(a.defn() for a in self.arguments())})"

    def type_defn(self) -> str:
        return f"using {self.type_name} = {self.type()}"

    # must be called from context where this is TensorIteratorBase*
    def call(self, ctx: Sequence[Binding]) -> str:
        return f"{self.name}(device_type(), *this, {', '.join(a.expr for a in translate(ctx, self.arguments()))})"

    # used in CUDA to skip the unnecessary dynamic dispatch
    def direct_call(self, ctx: Sequence[Binding]) -> str:
        return f"{self.kernel_name}(*this, {', '.join(a.expr for a in translate(ctx, self.arguments()))})"


@with_native_function
def compute_ufunc_cpu(g: NativeFunctionsGroup) -> str:
    stub_sig = StubSignature(g)
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CPU))

    return f"""
{stub_sig.type_defn()};
{stub_sig.dispatch_decl()}
{stub_sig.dispatch_defn()};

{sig.defn()} {{
  {stub_sig.call(sig.arguments())};
}}
"""


def compute_ufunc_cpu_dtype_body(
    g: NativeFunctionsGroup,
    dtype: ScalarType,
    inner_loops: dict[UfuncKey, UfuncSignature],
    parent_ctx: Sequence[Binding],
) -> str:
    if UfuncKey.CPUScalar not in inner_loops:
        raise AssertionError(f"{dtype}, {inner_loops.keys()}")
    if not inner_loops.keys() <= {UfuncKey.CPUScalar, UfuncKey.CPUVector}:
        raise AssertionError(
            f"inner_loops keys must be subset of CPUScalar/CPUVector, got {inner_loops.keys()}"
        )
    scalar_loop = inner_loops[UfuncKey.CPUScalar]
    vec_loop = None
    if UfuncKey.CPUVector in inner_loops:
        vec_loop = inner_loops[UfuncKey.CPUVector]

    # NB: We DON'T use translate here, because translate is
    # incapable of CSE'ing the scalar accesses in case it is also
    # used by Vectorized; also, the unpacking here is very simple
    # and only affects Scalar; everything else is implicitly captured
    # by the lambda

    # Setup scalar in scope
    body = []
    ctx = []
    for b in parent_ctx:
        if isinstance(b.argument, Argument) and b.argument.type != BaseType(
            BaseTy.Scalar
        ):
            continue
        body.append(f"auto _s_{b.name} = {b.name}.to<scalar_t>();")
        ctx.append(Expr(f"_s_{b.name}", NamedCType(b.nctype.name, BaseCType(scalar_t))))
    if vec_loop is not None:
        for b in parent_ctx:
            if isinstance(b.argument, Argument) and b.argument.type != BaseType(
                BaseTy.Scalar
            ):
                continue
            body.append(
                f"auto _v_{b.name} = at::vec::Vectorized<scalar_t>(_s_{b.name});"
            )
            ctx.append(
                Expr(
                    f"_v_{b.name}",
                    NamedCType(b.nctype.name, VectorizedCType(BaseCType(scalar_t))),
                )
            )

    # Setup lambda signature
    # NB: simplified version of ufunctor_arguments
    scalar_bindings = []
    vec_bindings = []
    for a in g.functional.func.arguments.flat_non_out:
        if not a.type.is_tensor_like():
            continue
        if a.type != BaseType(BaseTy.Tensor):
            raise AssertionError(f"Expected Tensor type, got {a.type}")
        scalar_bindings.append(
            Binding(
                name=a.name,
                nctype=NamedCType(a.name, BaseCType(scalar_t)),
                argument=a,
            )
        )
        if vec_loop is not None:
            vec_bindings.append(
                Binding(
                    name=a.name,
                    nctype=NamedCType(a.name, VectorizedCType(BaseCType(scalar_t))),
                    argument=a,
                )
            )

    def with_ctx(b: Sequence[Binding]) -> list[Expr | Binding]:
        r: list[Expr | Binding] = []
        r.extend(ctx)
        r.extend(b)
        return r

    body_str = "\n".join(body)
    if vec_loop is not None:
        return f"""
{body_str}
cpu_kernel_vec(iter,
  [=]({", ".join(b.decl() for b in scalar_bindings)}) {{ return {scalar_loop.call(with_ctx(scalar_bindings))}; }},
  [=]({", ".join(b.decl() for b in vec_bindings)}) {{ return {vec_loop.call(with_ctx(vec_bindings))}; }}
);
"""
    else:
        return f"""
{body_str}
cpu_kernel(iter,
  [=]({", ".join(b.decl() for b in scalar_bindings)}) {{ return {scalar_loop.call(with_ctx(scalar_bindings))}; }}
);
"""


@with_native_function
def compute_ufunc_cpu_kernel(g: NativeFunctionsGroup) -> str:
    stub_sig = StubSignature(g)

    # Reindex the ufunc by dtypes; processing generic/scalaronly as well
    loops = g.out.ufunc_inner_loop
    ufunc_sigs: dict[ScalarType, dict[UfuncKey, UfuncSignature]] = {}
    for k in [UfuncKey.CPUScalar, UfuncKey.CPUVector]:
        lks = []
        # ORDER MATTERS: this specifies overriding precedence
        if k in loops:  # should happen rarely
            lks.append(k)
        if UfuncKey.ScalarOnly in loops and k is UfuncKey.CPUScalar:
            lks.append(UfuncKey.ScalarOnly)
        if UfuncKey.Generic in loops:
            lks.append(UfuncKey.Generic)
        # TODO: don't hardcode ufunc:: namespace here, should be centralized smh
        for lk in lks:
            for dtype in loops[lk].supported_dtypes:
                compute_t: CType
                if k is UfuncKey.CPUScalar:
                    compute_t = BaseCType(scalar_t)
                elif k is UfuncKey.CPUVector:
                    compute_t = VectorizedCType(BaseCType(scalar_t))
                else:
                    raise AssertionError
                inner_ufunc_sigs = ufunc_sigs.setdefault(dtype, {})
                if k not in inner_ufunc_sigs:
                    inner_ufunc_sigs[k] = UfuncSignature(
                        g, name=f"ufunc::{loops[lk].name}", compute_t=compute_t
                    )

    # Build the conditionals
    dtype_cases = []
    for dtype, inner_ufunc_sigs in ufunc_sigs.items():
        dtype_cases.append(
            f"""
AT_DISPATCH_CASE(at::ScalarType::{dtype},
  [&]() {{
    {compute_ufunc_cpu_dtype_body(g, dtype, inner_ufunc_sigs, stub_sig.arguments())}
  }}
)
"""
        )

    dtype_cases_str = "\n".join(dtype_cases)
    return f"""
namespace {{

{stub_sig.kernel_defn()} {{
  AT_DISPATCH_SWITCH(iter.common_dtype(), "{stub_sig.name}",
    {dtype_cases_str}
  );
}}

}} // anonymous namespace

{stub_sig.type_defn()};
{stub_sig.dispatch_decl()}
REGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name})
"""
<|endoftext|>from torchgen.api.lazy import LazyArgument, LazyIrSchema
from torchgen.api.types import OptionalCType


def ts_lowering_body(schema: LazyIrSchema) -> str:
    # for now, we just want one IR class decl and soon after also the method defs
    # and we use the functional version not out/inplace.
    emplace_arguments = []

    def get_value(arg: LazyArgument) -> str:
        if isinstance(arg.lazy_type, OptionalCType):
            return f"has_{arg.name} ? loctx->GetOutputOp(operand(i++)) : nullptr"
        return "loctx->GetOutputOp(operand(i++))"

    for arg in schema.positional_args:
        if arg.is_lazy_value:
            emplace_arguments.append(get_value(arg))
            continue
        emplace_arguments.append(f'"{arg.name}", {arg.name}')

    emplace_arguments_str = "\n    ".join(
        [f"arguments.emplace_back({a});" for a in emplace_arguments]
    )
    emplace_kwarg_values = [
        f'"{arg.name}", {get_value(arg)}' for arg in schema.keyword_values
    ]
    emplace_kwarg_scalars = [
        f'"{arg.name}", {arg.name}' for arg in schema.keyword_scalars
    ]
    emplace_kwarguments = "\n    ".join(
        [
            f"kwarguments.emplace_back({a});"
            for a in emplace_kwarg_values + emplace_kwarg_scalars
        ]
    )
    return f"""\
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve({len(emplace_arguments)});
    kwarguments.reserve({len(emplace_kwarg_values + emplace_kwarg_scalars)});
    size_t i = 0;
    {emplace_arguments_str}
    {emplace_kwarguments}
    torch::lazy::TSOpVector {schema.aten_name}_out = torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    TORCH_CHECK_EQ({schema.aten_name}_out.size(), {len(schema.returns)});

    return {schema.aten_name}_out;
"""
<|endoftext|>from __future__ import annotations

import itertools
import textwrap
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
from typing_extensions import assert_never

import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    ConstRefCType,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    Expr,
    kernel_signature,
    MutRefCType,
    NamedCType,
    NativeSignature,
    tensorT,
)
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
    Argument,
    BackendIndex,
    DeviceCheckType,
    DispatchKey,
    gets_generated_out_inplace_wrapper,
    is_cuda_dispatch_key,
    NativeFunction,
    NativeFunctionsGroup,
    SchemaKind,
    TensorOptionsArguments,
)
from torchgen.utils import mapMaybe, Target


if TYPE_CHECKING:
    from torchgen.selective_build.selector import SelectiveBuilder


def gen_registration_headers(
    backend_index: BackendIndex,
    per_operator_headers: bool,
    rocm: bool,
) -> list[str]:
    if per_operator_headers:
        headers = ["#include <ATen/ops/as_strided_native.h>"]
    else:
        headers = ["#include <ATen/NativeFunctions.h>"]

    if backend_index.dispatch_key in (DispatchKey.CPU, DispatchKey.Meta):
        headers.append("#include <ATen/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.CUDA:
        if rocm:
            headers.append("#include <ATen/hip/EmptyTensor.h>")
        else:
            headers.append("#include <ATen/cuda/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.MPS:
        headers.append("#include <ATen/mps/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.XPU:
        # XPU specific, this header resides in third_party/torch-xpu-ops
        headers.append("#include <ATen/xpu/EmptyTensor.h>")
    elif backend_index.dispatch_key == DispatchKey.MTIA:
        headers.append("#include <ATen/native/mtia/EmptyTensor.h>")
    elif per_operator_headers:
        headers += [
            "#include <ATen/ops/empty.h>",
            "#include <ATen/ops/empty_strided.h>",
            "#include <ATen/ops/_copy_from_and_resize.h>",
            "#include <ATen/ops/_copy_from.h>",
        ]
    else:
        headers.append("#include <ATen/Functions.h>")

    headers.append("#include <c10/macros/Macros.h>")
    return headers


def gen_empty_impl_names(
    backend_index: BackendIndex,
) -> tuple[str | None, str | None]:
    empty_impl = None
    empty_strided_impl = None

    if backend_index.dispatch_key in (
        DispatchKey.Meta,
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.MPS,
        DispatchKey.XPU,
        DispatchKey.MTIA,
    ):
        dispatch = str(backend_index.dispatch_key).lower()
        empty_impl = f"at::detail::empty_{dispatch}"
        empty_strided_impl = f"at::detail::empty_strided_{dispatch}"
    elif backend_index.dispatch_key in (
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.QuantizedCPU,
        DispatchKey.QuantizedCUDA,
    ):
        empty_impl = "at::empty"
        empty_strided_impl = "at::empty_strided"

    return empty_impl, empty_strided_impl


def gen_create_out_helper(backend_index: BackendIndex) -> list[str]:
    if backend_index.dispatch_key == DispatchKey.Meta:
        empty_options = "options.device(at::kMeta)"
    else:
        empty_options = "options"

    empty_impl, empty_strided_impl = gen_empty_impl_names(backend_index)
    if empty_impl is None:
        return []

    return [
        f"""
Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{
  if (strides.empty()) {{
      return {empty_impl}(sizes, {empty_options});
  }} else {{
      return {empty_strided_impl}(sizes, strides, {empty_options});
  }}
}}
"""
    ]


def gen_maybe_create_proxy_helper(backend_index: BackendIndex) -> list[str]:
    _, empty_strided_impl = gen_empty_impl_names(backend_index)
    return (
        []
        if empty_strided_impl is None
        else [
            f"""
std::optional<Tensor> maybe_create_proxy(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{
  if (out.strides() != strides) {{
    return {empty_strided_impl}(sizes, strides, options);
  }}
  return std::nullopt;
}}
"""
        ]
    )


def gen_resize_out_helper(backend_index: BackendIndex) -> list[str]:
    if backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
        # The function isn't used by this key (since only functional ops have a kernel for this key),
        # so we need to not include it to avoid a defined-but-not-used error.
        return []
    return [
        """
void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}
"""
    ]


def gen_check_inplace_helper(backend_index: BackendIndex) -> list[str]:
    return [
        """
void check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");
  TORCH_CHECK(options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");
  TORCH_CHECK(sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");
}
"""
    ]


def gen_registration_helpers(backend_index: BackendIndex) -> list[str]:
    return [
        'C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-function")',
        *gen_create_out_helper(backend_index),
        *gen_resize_out_helper(backend_index),
        *gen_check_inplace_helper(backend_index),
        *gen_maybe_create_proxy_helper(backend_index),
        "C10_DIAGNOSTIC_POP()",
    ]


# Generates Register{dispatch}.cpp (e.g., RegisterCPU.cpp).
#
#   - The primary function of this file is to register all of the
#     implementations for the given dispatch key to the dispatcher,
#     so they are available for use in PyTorch.  If dispatch is
#     None, we generate schema (def) registrations and catchall
#     registrations.
#   - The secondary function of this file is to generate a wrapper
#     around functions.  In CPUType these wrappers do nothing
#     (and should be removed), but in other cases they handle
#     DeviceGuard. A small extra benefit of wrappers is they
#     are not overloaded, so they can be used in the registration
#     API without having to disambiguate which overload you want
#     (as would be the case if you directly registered native::
#     functions).
#   - The tertiary function of this file is to generate *static*
#     cpp API bindings which can be used to bypass dispatcher
#     directly to kernels, but with user-friendly cpp-style API
@dataclass(frozen=True)
class RegisterDispatchKey:
    backend_index: BackendIndex

    target: Literal[
        Target.ANONYMOUS_DEFINITION,
        Target.NAMESPACED_DEFINITION,
        Target.NAMESPACED_DECLARATION,
        Target.REGISTRATION,
    ]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    # Whether or not we are actually code-genning for ROCm
    rocm: bool

    # Whether or not to generate symint registrations or not.  External users
    # of codegen who don't care about symints can set this to false to get
    # non-SymInt codegen
    symint: bool

    # The class that all unstructured native functions live under. This is used to improve
    # compiler error messages when a kernel writer adds a native function with the wrong signature.
    # This is only used in unstructured kernels, since structured kernels already live in a class.
    # Finally, this field is currently Optional because it is only used by external backends.
    # It would be nice if we can add the same logic to in-tree kernels too, but that requires updating
    # all of the existing kernel signatures scattered across aten/src/ATen/native.
    class_method_name: str | None

    # Only set to true in lightweight dispatch. If lightweight dispatch is enabled we are registering
    # operators into JIT op registry, thus we need to avoid generating code to register into the dispatcher.
    skip_dispatcher_op_registration: bool

    @staticmethod
    def gen_device_check(
        type: DeviceCheckType, args: list[Argument], method_name: str
    ) -> str:
        if type == DeviceCheckType.NoCheck:
            return "  // No device check\n"

        device_check = "std::optional<Device> common_device = std::nullopt;\n"
        device_check += "(void)common_device; // Suppress unused variable warning\n"
        for arg in args:
            # Only tensor like arguments are eligible
            if arg.type.is_tensor_like():
                device_check += f"""
  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");"""
        return device_check

    @method_with_native_function
    def __call__(self, f: NativeFunctionsGroup | NativeFunction) -> list[str]:
        if isinstance(f, NativeFunctionsGroup):
            g: NativeFunctionsGroup = f
            # Note: We call gen_structured() if the operator is marked structured, regardless of the backend.
            # gen_structured() has special logic to handle auto-generated kernels.
            if g.structured:
                return self.gen_structured(g)
            else:
                return list(
                    mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions())
                )
        elif isinstance(f, NativeFunction):
            r = self.gen_unstructured(f)
            return [] if r is None else [r]
        else:
            assert_never(f)

    def wrapper_kernel_sig(
        self, f: NativeFunction
    ) -> NativeSignature | DispatcherSignature:
        # The prefix is just to ensure uniqueness. The Dispatcher API doesn't guarantee unique kernel names.
        return DispatcherSignature.from_schema(
            f.func,
            prefix=f"wrapper_{self.backend_index.dispatch_key}_{f.func.name.overload_name}_",
            symint=self.symint,
        )

    def gen_out_inplace_wrapper(
        self, f: NativeFunction, g: NativeFunctionsGroup | None
    ) -> str | None:
        if g is None:
            return None
        k = f.func.kind()
        if k is SchemaKind.inplace:
            copy_op = "at::_copy_from"
        elif k is SchemaKind.out:
            copy_op = "at::_copy_from_and_resize"
        else:
            raise AssertionError("gen_out_inplace_wrapper called on a functional op")

        sig = self.wrapper_kernel_sig(f)
        name = sig.name()

        func_res = f"{name}_tmp"
        return_names = cpp.return_names(f)
        if len(return_names) > 1:
            updates = "\n  ".join(
                f"{copy_op}(std::get<{i}>({func_res}), {ret_name});"
                for i, ret_name in enumerate(return_names)
            )
            returns = f"{sig.returns_type().cpp_type()}({', '.join(return_names)})"
        elif len(return_names) == 1:
            ret_name = return_names[0]
            updates = f"{copy_op}({func_res}, {ret_name});"
            returns = ret_name
        else:
            if len(f.func.arguments.out) != 1:
                raise AssertionError(
                    f"Expected exactly 1 out argument, got {len(f.func.arguments.out)}"
                )
            returns = ""
            out_arg = f.func.arguments.out[0]
            if out_arg.type.is_list_like():
                updates = f"""\
    for (int64_t i = 0; i < {func_res}.size(); ++i) {{
        {copy_op}({func_res}[i], {out_arg.name}[i]);
    }}"""
            else:
                updates = f"{copy_op}({func_res}, {out_arg.name});"

        functional_sig = self.wrapper_kernel_sig(g.functional)

        return f"""\
{sig.defn(name=name)} {{
  auto {func_res} = {functional_sig.name()}({", ".join(e.expr for e in translate(sig.arguments(), functional_sig.arguments()))});
  {updates}
  return {returns};
}}
"""

    def gen_structured(self, g: NativeFunctionsGroup) -> list[str]:
        metadata = self.backend_index.get_kernel(g)
        if self.backend_index.dispatch_key == DispatchKey.Meta:
            if self.backend_index.has_kernel(g.out):
                raise AssertionError(
                    "Do not explicitly specify Meta dispatch key on structured "
                    "functions, they will be automatically generated for you"
                )
        elif (
            self.backend_index.dispatch_key
            == DispatchKey.CompositeExplicitAutogradNonFunctional
        ):
            if self.backend_index.has_kernel(g.out):
                raise AssertionError(
                    "Do not explicitly specify CompositeExplicitAutograd dispatch key on "
                    "structured functions, they will be automatically generated for you"
                )
        elif metadata is None or not metadata.structured:
            return list(mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions()))
        structured_gen = StructuredRegisterDispatchKey(
            self.backend_index,
            self.target,
            self.selector,
            self.rocm,
            self.symint,
            self.class_method_name,
            self.skip_dispatcher_op_registration,
            g,
        )
        return list(mapMaybe(structured_gen.gen_one, g.functions()))

    def gen_unstructured(
        self, f: NativeFunction, g: NativeFunctionsGroup | None = None
    ) -> str | None:
        with native_function_manager(f):
            inplace_meta = False
            gets_out_inplace_wrapper = False
            if not self.backend_index.has_kernel(f):
                if (
                    self.backend_index.dispatch_key == DispatchKey.Meta
                    and f.func.kind() is SchemaKind.inplace
                    and
                    # Defer to composites for meta implementation
                    not f.has_composite_kernel
                    and
                    # Inplace list operations are not supported
                    len(f.func.returns) == 1
                ):
                    inplace_meta = True
                elif (
                    not self.backend_index.use_out_as_primary
                    and g is not None
                    and gets_generated_out_inplace_wrapper(f, g, self.backend_index)
                ):
                    # We want to generate inplace/out wrappers, that don't have a kernel for the backend.
                    gets_out_inplace_wrapper = True
                else:
                    return None
            if f.manual_kernel_registration:
                return None

            if (
                self.target is Target.REGISTRATION
                and not self.selector.is_native_function_selected(f)
            ):
                return None

            sig = self.wrapper_kernel_sig(f)

            name = sig.name()
            returns_type = sig.returns_type().cpp_type()
            args = sig.arguments()
            args_str = ", ".join(a.defn() for a in args)

            # See Note [Direct dispatch bindings]
            cpp_sig_group = CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=False
            )

            # TODO: dedupe this with the structured codegen
            if self.target is Target.NAMESPACED_DECLARATION:
                result = ""
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    result += f"TORCH_API {cpp_sig.decl()};\n"
                return result
            elif self.target is Target.NAMESPACED_DEFINITION:

                def generate_defn(cpp_sig: CppSignature) -> str:
                    return f"""
{cpp_sig.defn()} {{
return {sig.name()}({", ".join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""

                result = ""
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    result += generate_defn(cpp_sig)
                return result

            elif self.target is Target.ANONYMOUS_DEFINITION:
                # short circuit for inplace_meta
                if inplace_meta:
                    if f.func.arguments.self_arg is None:
                        raise AssertionError(
                            "Expected self_arg to be non-None for inplace_meta"
                        )
                    self_arg_name = f.func.arguments.self_arg.argument.name
                    # TODO: handle in place on tensor list
                    return f"""
{returns_type} {name}({args_str}) {{
  TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),
    "Cannot inplace into non-meta tensor with meta tensor argument");
  return {self_arg_name};
}}
"""

                # short circuit for generated inplace/out wrappers
                if gets_out_inplace_wrapper:
                    return self.gen_out_inplace_wrapper(f, g)

                metadata = self.backend_index.get_kernel(f)
                if metadata is None:
                    return None
                if self.class_method_name is None:
                    impl_name = f"{metadata.cpp_namespace}::{metadata.kernel}"
                else:
                    impl_name = f"{metadata.cpp_namespace}::{self.class_method_name}::{metadata.kernel}"

                kernel_sig = kernel_signature(f, self.backend_index)

                args_exprs_str = ", ".join(
                    e.expr
                    for e in translate(
                        sig.arguments(), kernel_sig.arguments(), method=False
                    )
                )

                device_check = "  // No device check\n"
                # Backends that require device guards presumably also require device checks.
                if self.backend_index.device_guard:
                    device_check_args = itertools.chain(
                        f.func.arguments.out, f.func.arguments.flat_positional
                    )
                    device_check = RegisterDispatchKey.gen_device_check(
                        f.device_check, list(device_check_args), name
                    )

                device_guard = "// DeviceGuard omitted"  # default
                if f.device_guard and self.backend_index.device_guard:
                    has_tensor_options = any(
                        isinstance(a, TensorOptionsArguments)
                        for a in f.func.arguments.non_out
                    )
                    if has_tensor_options:
                        # kernel is creating a tensor
                        device_guard = """
  const DeviceGuard device_guard(device_or_default(device));"""

                        # CUDA requires special handling
                        if is_cuda_dispatch_key(self.backend_index.dispatch_key):
                            device_guard = f"globalContext().lazyInitDevice(c10::DeviceType::CUDA);\n{device_guard}"
                    else:
                        # kernel is operating on existing tensors

                        # There is precedence for which argument we use to do
                        # device guard.  This describes the precedence order.
                        self_arg = (
                            [f.func.arguments.self_arg.argument]
                            if f.func.arguments.self_arg is not None
                            else []
                        )
                        candidate_args = itertools.chain(
                            self_arg,
                            f.func.arguments.out,
                            f.func.arguments.flat_positional,
                        )

                        # Only tensor like arguments are eligible
                        device_of = next(
                            (
                                f"{a.name}"
                                for a in candidate_args
                                if a.type.is_tensor_like()
                            ),
                            None,
                        )
                        if device_of is not None:
                            device_guard = f"const OptionalDeviceGuard device_guard(device_of({device_of}));"

                return f"""\
namespace {{

{returns_type} {name}({args_str}) {{
  {device_check}

  {device_guard}
  return {impl_name}({args_exprs_str});
}}

}} // anonymous namespace
"""

            elif self.target is Target.REGISTRATION:
                if f.manual_kernel_registration or self.skip_dispatcher_op_registration:
                    return None
                else:
                    payload = f"TORCH_FN({name})"
                    return f'm.impl("{f.func.name}",\n{payload});\n'
            else:
                assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           STRUCTURED
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@dataclass(frozen=True)
class StructuredRegisterDispatchKey(RegisterDispatchKey):
    g: NativeFunctionsGroup

    def gen_class_set_output_functions(
        self, k: SchemaKind, parent_class: str, generate_super: bool
    ) -> str:
        if generate_super:
            set_output_super = f"{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options);"
        else:
            set_output_super = ""

        def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
            return f"""
void set_output_{name}(
    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
    TensorOptions options
) override {{
{textwrap.indent(self.gen_class_set_output_body(k, maybe_create_proxy), "    ")}
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
{textwrap.indent(set_output_super, "    ")}
}}
"""

        return f"""
{gen_set_output_function("strided", maybe_create_proxy=True)}
{gen_set_output_function("raw_strided", maybe_create_proxy=False)}
"""

    def gen_class_set_output_body(self, k: SchemaKind, maybe_create_proxy: bool) -> str:
        if self.backend_index.dispatch_key in [
            DispatchKey.CUDA,
            DispatchKey.MPS,
            DispatchKey.XPU,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        ]:
            maybe_set_guard = """
auto current_device = guard_.current_device();
if (C10_UNLIKELY(current_device.has_value())) {
  TORCH_INTERNAL_ASSERT(*current_device == options.device(),
    "structured kernels don't support multi-device outputs");
} else {
  guard_.reset_device(options.device());
}
"""
            maybe_set_guard_line = maybe_set_guard + "\n"
        else:
            maybe_set_guard_line = maybe_set_guard = ""

        if maybe_create_proxy:
            create_proxy = """
auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
if (C10_UNLIKELY(maybe_proxy.has_value())) {
    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();
}
"""
        else:
            create_proxy = ""

        if k is SchemaKind.functional:
            if self.backend_index.dispatch_key not in (
                DispatchKey.Meta,
                DispatchKey.CPU,
                DispatchKey.CUDA,
                DispatchKey.MPS,
                DispatchKey.XPU,
                DispatchKey.MTIA,
                DispatchKey.CompositeExplicitAutogradNonFunctional,
            ):
                raise AssertionError(
                    f"Unexpected dispatch key {self.backend_index.dispatch_key} "
                    "for functional schema"
                )
            return f"""{maybe_set_guard_line}
outputs_[output_idx] = create_out(sizes, strides, options);"""
        elif k is SchemaKind.inplace:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
check_inplace(out, sizes, options);
{create_proxy}"""
        elif k is SchemaKind.out:
            return f"""{maybe_set_guard_line}
const auto& out = outputs_[output_idx].get();
resize_out(out, sizes, strides, options);
{create_proxy}"""
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(
                f"{k} structured operators are currently not supported"
            )
        else:
            assert_never(k)

    # returns the definition of a ctor, as well as how to construct
    # this class to a variable named op
    def gen_class_ctor(self, k: SchemaKind, class_name: str, returns: int) -> str:
        if k is SchemaKind.functional:
            return ""
        elif k is SchemaKind.inplace:
            # TODO: Make sure out argument is guaranteed to be self
            return f"{class_name}(Tensor& self) : outputs_{{std::ref(self)}} {{}}"
        elif k is SchemaKind.out:
            out_args = ", ".join(f"Tensor& out{i}" for i in range(returns))
            out_refs = ", ".join(f"std::ref(out{i})" for i in range(returns))
            return f"{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}"
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(
                f"{k} structured operators are currently not supported"
            )
        else:
            assert_never(k)

    def gen_class(
        self,
        f: NativeFunction,
        k: SchemaKind,
        *,
        class_name: str,
        parent_class: str,
        generate_super: bool,
    ) -> str:
        if k is SchemaKind.functional:
            output_type = "Tensor"
            output_value = "outputs_[output_idx]"
            proxy_field = ""
        elif k is SchemaKind.inplace or k is SchemaKind.out:
            output_type = "std::reference_wrapper<Tensor>"
            output_value = "proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()"
            proxy_field = f"std::array<::std::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;"
        else:
            raise RuntimeError(f"Unsupported SchemaKind {k}")

        if self.backend_index.dispatch_key == DispatchKey.CUDA:
            guard_field = "c10::cuda::OptionalCUDAGuard guard_;"
        elif (
            self.backend_index.dispatch_key
            == DispatchKey.CompositeExplicitAutogradNonFunctional
        ):
            guard_field = "c10::OptionalDeviceGuard guard_;"
        elif self.backend_index.dispatch_key == DispatchKey.MPS:
            # TODO: Move to OptionalMPSGuard.
            guard_field = "c10::OptionalDeviceGuard guard_;"
        elif self.backend_index.dispatch_key == DispatchKey.XPU:
            guard_field = "c10::OptionalDeviceGuard guard_;"
        elif self.backend_index.dispatch_key == DispatchKey.MTIA:
            guard_field = "c10::OptionalDeviceGuard guard_;"
        else:
            guard_field = ""

        indent = " " * 4
        class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
        lines = (
            f"struct {class_name} final : public {parent_class} {{",
            f"{textwrap.indent(class_ctor_str, indent)}",
            f"{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}",
            "    const Tensor& maybe_get_output(int64_t output_idx) override {",
            f"      return {output_value};\n",  # type: ignore[possibly-undefined]  # TODO: audit
            "    }",
            # type: ignore[possibly-undefined]  # TODO: audit
            f"    std::array<{output_type}, {len(f.func.returns)}> outputs_;",
            f"{textwrap.indent(proxy_field, indent)}",  # type: ignore[possibly-undefined]  # TODO: audit
            f"{textwrap.indent(guard_field, indent)}",
            "};",
        )
        return "\n".join(line for line in lines if line)

    @method_with_native_function
    def gen_one(self, f: NativeFunction) -> str | None:
        if f.manual_kernel_registration:
            raise AssertionError(
                f"Function {f.func.name} has manual_kernel_registration=True"
            )

        if (
            self.target is Target.REGISTRATION
            and not self.selector.is_native_function_selected(f)
        ):
            return None

        # TODO: Now, there is something interesting going on here.  In the code below,
        # we generate CompositeExplicitAutogradNonFunctional implementations of functional and inplace
        # based on the out implementation.  But in fact, out is definable by
        # functional too (just not very efficiently), and this is honestly the
        # MORE likely situation for a backend implementer.  How do we pick?
        # Well, taking a page from Haskell type classes and default methods,
        # we could conceivably register a circular definition (out in terms
        # of functional, and functional in terms of out) and just require
        # someone to implement one or the other.  We'd have to do a little bit
        # of work to not register one of these "weak" definitions unless there
        # is a strong definition somewhere in the DAG!  So it's not implemented yet.
        if (
            self.backend_index.dispatch_key
            == DispatchKey.CompositeExplicitAutogradNonFunctional
            and f.func.kind() is SchemaKind.out
        ):
            # Never generate a default implementation for out, that's what you
            # have to define as a backend implementer
            return None

        # Note [Direct dispatch bindings]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Signature of the non-dispatched function we'll expose in a header
        # (e.g., at::cpu::add).  We don't generate methods (TODO: do this
        # when CPUTensor class is a thing); nor do we generate fallback
        # bindings for manual_cpp_binding functions.
        cpp_sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=False
        )

        # Signature of the wrapper function we'll register to the dispatcher
        kern = self.backend_index.get_kernel(f)
        sig = NativeSignature(
            f.func,
            prefix=f"wrapper_{self.backend_index.dispatch_key}_",
            symint=kern is not None and kern.supports_symint(),
        )

        if self.target is Target.NAMESPACED_DECLARATION:
            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += f"TORCH_API {cpp_sig.decl()};\n"
            return result

        elif self.target is Target.NAMESPACED_DEFINITION:

            def generate_defn(cpp_sig: CppSignature) -> str:
                return f"""
{cpp_sig.defn()} {{
return {sig.name()}({", ".join(e.expr for e in translate(cpp_sig.arguments(), sig.arguments()))});
}}
"""

            result = ""
            for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                result += generate_defn(cpp_sig)
            return result

        elif self.target is Target.ANONYMOUS_DEFINITION:
            k = f.func.kind()

            # Construct the body of the wrapper function with signature sig
            sig_body = []
            # We'll use context to keep track of any variables we've brought
            # into scope while generating code
            context: list[Binding | Expr] = list(sig.arguments())

            # Initialize the class corresponding to this structured
            # operator; feeding it the output argument(s) if it is known
            if self.backend_index.dispatch_key is DispatchKey.Meta:
                class_name = f"structured_{meta.name(self.g)}_meta_{k.name}"
                parent_class = f"at::meta::structured_{meta.name(self.g)}"
            elif (
                self.backend_index.dispatch_key
                is DispatchKey.CompositeExplicitAutogradNonFunctional
            ):
                # TODO: dedup this branch
                class_name = f"structured_{meta.name(self.g)}_default_backend_{k.name}"
                parent_class = f"at::meta::structured_{meta.name(self.g)}"
            else:
                metadata = self.backend_index.get_kernel(self.g)
                if metadata is None:
                    raise AssertionError(
                        f"No kernel metadata found for {self.g.functional.func.name}"
                    )
                class_name = f"structured_{metadata.kernel}_{k.name}"
                parent_class = f"{metadata.cpp_namespace}::structured_{metadata.kernel}"

            if self.backend_index.device_guard:
                device_check_args = itertools.chain(
                    f.func.arguments.out, f.func.arguments.flat_positional
                )
                sig_body.append(
                    RegisterDispatchKey.gen_device_check(
                        f.device_check, list(device_check_args), sig.name()
                    )
                )

            if k is SchemaKind.functional:
                sig_body.append(f"{class_name} op;")
            elif k is SchemaKind.inplace:
                sig_body.append(f"{class_name} op(self);")
            elif k is SchemaKind.out:
                out_args_str = ", ".join(a.name for a in f.func.arguments.out)
                sig_body.append(f"{class_name} op({out_args_str});")

            # Translate the input native arguments into structured
            # arguments for the meta call
            meta_exprs = ", ".join(
                e.expr
                for e in translate(
                    context, structured.meta_arguments(self.g), method=False
                )
            )

            if self.g.out.precomputed:
                # If this function group has precomputed elements, the meta function
                # returns a struct containing them which must be saved so that it
                # can be unpacked when generating code to call the impl.
                sig_body.append(f"auto precompute = op.meta({meta_exprs});")

                # Put all of the contents of the precompute struct into the context
                # so that translate will be able to return the correct args for the
                # call to the impl.
                precomputed_values = [
                    *self.g.out.precomputed.replace.values(),
                    self.g.out.precomputed.add,
                ]
                for precomputed_elems in precomputed_values:
                    context.extend(
                        Expr(
                            expr=f"precompute.{arg.name}",
                            type=structured.argument_type(arg, binds=arg.name),
                        )
                        for arg in precomputed_elems
                    )

                # Add a use of the precompute struct so FB internal compilers don't
                # complain that there is an unused variable.
                sig_body.append("(void)precompute;")
            else:
                sig_body.append(f"op.meta({meta_exprs});")

            # After running meta, op.outputs_ is guaranteed to be valid;
            # add it to the context
            out_args = structured.out_arguments(self.g)
            for i, out_arg in enumerate(out_args):
                if ConstRefCType(BaseCType(tensorT)) != out_arg.nctype.type:
                    raise AssertionError(
                        f"Expected out_arg type to be ConstRefCType(BaseCType(tensorT)), "
                        f"got {out_arg.nctype.type}"
                    )

                if k is SchemaKind.out:
                    expr = f"op.maybe_get_output({i})"
                else:
                    expr = f"op.outputs_[{i}]"

                context.append(
                    Expr(
                        expr=expr,
                        # TODO: Stop hardcoding that the output type is a Tensor.  Note
                        # that for the codegen here this is fine because outputs_ is
                        # hardcoded to be tensor already
                        type=NamedCType(
                            out_arg.nctype.name, MutRefCType(BaseCType(tensorT))
                        ),
                    )
                )

            # With the expanded context, do the impl call (if not a meta
            # function)
            if (
                self.backend_index.dispatch_key
                == DispatchKey.CompositeExplicitAutogradNonFunctional
            ):
                # TODO: https://github.com/pytorch/pytorch/issues/53023
                out_sig_group = CppSignatureGroup.from_native_function(
                    self.g.out, method=False, fallback_binding=f.manual_cpp_binding
                )
                out_sig = out_sig_group.most_faithful_signature()
                api_name = out_sig.name()
                out_exprs = ", ".join(
                    e.expr
                    for e in translate(context, out_sig.arguments(), method=False)
                )
                # TODO: I think this means structured won't work with method
                # only functions (but maybe you're saved by faithful? iunno.)
                # NB: Originally I wrote this as an at::redispatch call, but
                # I got in trouble because that meant I needed a DispatchKeySet
                # in the wrapper function, which meant I needed a DispatchKeySet
                # in the DispatchKeyFunctions declarations, but the defined API
                # there does NOT permit a dispatch key set.  I think you can
                # probably unwind this by calling some function to do the TLS
                # fetch and get the DispatchKeySet when you don't have it, but
                # I didn't do it for this version
                sig_body.append(f"at::{api_name}({out_exprs});")
            elif self.backend_index.dispatch_key != DispatchKey.Meta:
                impl_exprs = ", ".join(
                    e.expr
                    for e in translate(
                        context, structured.impl_arguments(self.g), method=False
                    )
                )
                sig_body.append(f"op.impl({impl_exprs});")

            # Go over each output, and check if there is a proxy created for it.
            # If so, copy it over to the original output.
            if k is SchemaKind.out or k is SchemaKind.inplace:
                for i in range(len(f.func.returns)):
                    sig_body.append(
                        f"if (op.proxy_outputs_[{i}].has_value()) op.outputs_[{i}].get().copy_(*op.proxy_outputs_[{i}]);"
                    )

            # Destructively return the final tensors
            # TODO: Do this in translate instead
            if k is SchemaKind.functional:
                if len(f.func.returns) == 1:
                    ret_expr = "std::move(op.outputs_[0])"  # small optimization
                else:
                    moved = ", ".join(
                        f"std::move(op.outputs_[{i}])"
                        for i in range(len(f.func.returns))
                    )
                    ret_expr = f"std::make_tuple({moved})"
            elif k is SchemaKind.inplace:
                ret_expr = "self"
            elif k is SchemaKind.out:
                if len(f.func.returns) == 1:
                    ret_expr = f.func.arguments.out[0].name
                else:
                    refs = ", ".join(a.name for a in f.func.arguments.out)
                    ret_expr = f"std::forward_as_tuple({refs})"
            sig_body.append(f"return {ret_expr};")  # type: ignore[possibly-undefined]  # TODO: audit

            sig_body_str = "\n".join(sig_body)

            # For an overview of what this template code looks like, see
            # https://github.com/pytorch/rfcs/pull/9
            return f"""\
{
                self.gen_class(
                    f,
                    k,
                    class_name=class_name,
                    parent_class=parent_class,
                    generate_super=self.g.out.structured_inherits is not None,
                )
            }

{sig.defn()} {{
{sig_body_str}
}}
"""

        elif self.target is Target.REGISTRATION:
            return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'
        else:
            assert_never(self.target)
            # Silence mypy's "Missing return statement" error
            return None
<|endoftext|>from __future__ import annotations

from typing import NoReturn, TYPE_CHECKING

from torchgen.api.types import (
    ArrayRefCType,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    deviceT,
    Expr,
    intArrayRefT,
    iOptTensorListRefT,
    layoutT,
    ListCType,
    longT,
    memoryFormatT,
    MutRefCType,
    NamedCType,
    opmath_t,
    OptionalCType,
    optionalIntArrayRefT,
    optionalScalarRefT,
    optionalSymIntArrayRefT,
    optionalTensorRefT,
    scalar_t,
    scalarT,
    scalarTypeT,
    SpecialArgName,
    symIntArrayRefT,
    SymIntT,
    tensorOptionsT,
    tensorT,
    VectorCType,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


# This file implements a small program synthesis engine that implements
# conversions between one API to another.
#
# The key data type in this file in NamedCType, short for Named C++ semantic type.  A NamedCType
# represents a C++ type, plus semantic information about what it represents.
# For example, consider the argument "bool pin_memory"; its normal C++ type is
# "bool", but its C++ semantic type also keeps track that this represents a
# "pin_memory"; you can't just use a random other boolean in a context where you
# need a "pin_memory"!
#
# The translator takes a list of needed NamedCTypes, and then figures out how
# to construct expressions with these NamedCTypes from the given bindings.  Many
# of these expressions are trivial (I need a Tensor other; there's a Tensor
# other scope); others are more nontrivial and may require packing/unpacking.
# Some examples of non-trivial action:
#
#   - Need the "dtype" binding?  Well, maybe "dtype" isn't available
#     in the context, instead, "options" is, and you need to extract
#     it from there.  (Gather)
#
#   - Need the "context" binding?  Well, maybe "context" isn't available
#     in the context, and you need to construct it from "dtype", "device",
#     etc.  (Scatter)
#
#   - Need the "memory_format" binding?  Well, actually, it's available
#     from both "memory_format" and "options", so you had better make sure
#     they are consistent.  (Join)

options_ctype = NamedCType("options", ConstRefCType(BaseCType(tensorOptionsT)))

out_tensor_ctype = NamedCType("out", ConstRefCType(BaseCType(tensorT)))

longVec_ctype = VectorCType(BaseCType(longT))
longSymVec_ctype = VectorCType(BaseCType(SymIntT))
optionalLongVec_ctype = OptionalCType(VectorCType(BaseCType(longT)))
optionalScalar_ctype = OptionalCType(BaseCType(scalarT))
optionalTensor_ctype = OptionalCType(BaseCType(tensorT))


class UnsatError(RuntimeError):
    pass


# Given a set of in-scope bindings and a set of target bindings, synthesize
# a list of expressions that uses only the in-scope bindings (bindings) that
# have all of the types of goals.  You may want to use this function if
# you're generating code for a function like:
#
#   void f({args}) {
#     g({exprs}); // g is a different API
#   }
#
# and you need to generate "exprs".
#
# Typically, a list of Bindings is convenient to get (you usually call something
# like arguments() to get them); but technically you only need less information:
# for 'bindings' an (un-ordered) list of Exprs is sufficient; similarly, for
# 'goals', an (ordered) list of NamedCType goals is sufficient.  If you are doing
# something more complicated, e.g., tracking the set of bindings in a context,
# you may find using these smaller types more convenient.
def translate(
    bindings: Sequence[Expr | Binding],
    goals: Sequence[NamedCType | Binding],
    *,
    method: bool = False,
    allow_expensive_conversions: bool = False,
) -> list[Expr]:
    binding_exprs: list[Expr] = []
    for b in bindings:
        if isinstance(b, Binding):
            binding_exprs.append(
                Expr(
                    expr=b.name,
                    type=b.nctype,
                )
            )
        else:
            binding_exprs.append(b)

    goal_ctypes: list[NamedCType] = []
    for g in goals:
        if isinstance(g, Binding):
            goal_ctypes.append(g.nctype)
        else:
            goal_ctypes.append(g)

    # Add all the bindings to the context
    ctx: dict[NamedCType, str] = {}
    for b in binding_exprs:
        ctx[b.type] = b.expr

        # While we're at it, do some simple forward inference, looking through
        # constructors.
        #
        # NB: When should you do forward inference versus backward inference?
        # The general idea:
        #
        #   - Backward inference WHEN the goal gets smaller
        #   - Forward inference WHEN the hypothesis gets smaller
        #
        # This helps ensure termination: backward inference starts with a goal
        # and tries to make it simpler and simpler until it's trivial; if the
        # goal can grow in size, we blow up to a really huge goal size.
        # Similarly, with forward inference we take hypotheses and decompose
        # them into simpler hypotheses; if hypotheses could expand in size,
        # we also have potential nontermination.  (In the code below, forward
        # inference is only ever carried out at a single step, but you could
        # imagine repeated application of forward inference being profitable.)
        #
        # A good starting point in the literature for exploring more about proof
        # search are these lecture notes
        # https://www.cs.cmu.edu/~fp/courses/oregon-m10/04-focusing.pdf
        #
        # TODO: My kingdom for a pattern matcher
        # https://www.python.org/dev/peps/pep-0634/
        #
        # TODO: This could get us in recomputation trouble if b.expr is nontrivial.
        # Fix this by implementing some sort of sharing so that if multiple
        # goals share the same expression, we only compute it once.  This seems
        # to matter in practice as compiler is often unwilling to CSE nontrivial
        # expressions like scalar.to<scalar_t>()
        t = b.type
        if (
            isinstance(t, ConstRefCType)
            and isinstance(t.elem, OptionalCType)
            and isinstance(t.elem.elem, BaseCType)
            and str(t.elem.elem.type) == "at::Tensor"
        ):
            ctx[NamedCType(t.elem.elem.name, ConstRefCType(BaseCType(tensorT)))] = (
                f"({b.expr}.has_value() ? *{b.expr} : at::Tensor())"
            )

        if t.type == ConstRefCType(OptionalCType(BaseCType(tensorT))):
            ctx[NamedCType(t.name, BaseCType(optionalTensorRefT))] = (
                f"(({b.expr}.has_value() && (*{b.expr}).defined()) ? at::OptionalTensorRef(*{b.expr}) : at::OptionalTensorRef())"
            )

        if t.type == ConstRefCType(BaseCType(scalarT)):
            ctx[NamedCType(t.name, BaseCType(opmath_t))] = f"({b.expr}).to<opmath_t>()"

        if t.type == ConstRefCType(OptionalCType(BaseCType(scalarT))):
            ctx[NamedCType(t.name, BaseCType(optionalScalarRefT))] = (
                f"({b.expr}.has_value() ? at::OptionalScalarRef(&({b.expr}.value())) : at::OptionalScalarRef())"
            )

        if t.type == BaseCType(scalar_t):
            ctx[NamedCType(t.name, BaseCType(opmath_t))] = (
                f"static_cast<opmath_t>({b.expr})"
            )

        # [Note: IOptTensorListRef]
        if t.type == ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT)))):
            ctx[NamedCType(t.name, BaseCType(iOptTensorListRefT))] = (
                f"at::IOptTensorListRef({b.expr})"
            )

    # Add implicit bindings if the generated code is inside a Tensor method
    if method:
        ctx[NamedCType("self", MutRefCType(BaseCType(tensorT)))] = (
            "const_cast<Tensor&>(*this)"
        )
        ctx[NamedCType("self", ConstRefCType(BaseCType(tensorT)))] = (
            "const_cast<Tensor&>(*this)"
        )
        # This is better!  Byte-for-byte compat
        # ctx[NamedCType("self", ConstRefCType(BaseCType(tensorT)))] = "*this"

    def unsat(goal: NamedCType) -> NoReturn:
        ctx_desc = "\n".join(
            f"  {t.cpp_type()} {t.name}; // {e}" for t, e in ctx.items()
        )
        raise UnsatError(
            f"""
Failed to synthesize the expression "{goal.cpp_type()} {goal.name}".
When I failed, the following bindings were available in the context:

{ctx_desc}

This probably means there is a missing rule in the rules of torchgen.api.translate.
Check this module for more information.
"""
        )

    # A shitty backtracking search implementation.  It's shitty because it
    # does backtracking via stack (bad idea!) and for the most part tries to
    # avoid backtracking.  In particular, if
    # direct=True, we won't try to do any fancy synthesis, just trivial
    # conversions (e.g., "T a" is OK for "const T& a").  So all of the
    # existing rules in this function simply try to solve immediately,
    # and bail if things don't work out.
    def solve(goal: NamedCType, *, direct: bool) -> str:
        def direct_solve(goal: NamedCType) -> str:
            return solve(goal, direct=True)

        if goal in ctx:
            # Trivial
            return ctx[goal]

        # const & is satisfied with mutable &
        if isinstance(goal.type, ConstRefCType):
            try:
                # WARNING: not strictly decreasing; be careful not
                # to add a direct conversion that goes satisfies
                # mutable& with const&
                return solve(
                    NamedCType(goal.name, MutRefCType(goal.type.elem)), direct=direct
                )
            except UnsatError:
                pass

        # mutable & is satisfied with value
        if isinstance(goal.type, MutRefCType):
            try:
                return solve(NamedCType(goal.name, goal.type.elem), direct=direct)
            except UnsatError:
                pass

        # TODO: These are referentially equal, shouldn't have to do this;
        # ensuring we don't use type synonym IntArrayRef in codegen would
        # help
        if goal.type == ArrayRefCType(BaseCType(longT)):
            return solve(NamedCType(goal.name, BaseCType(intArrayRefT)), direct=direct)

        if direct:
            unsat(goal)

        # For now, all of these rules are mutually exclusive.
        if goal == NamedCType("memory_format", OptionalCType(BaseCType(memoryFormatT))):
            memory_format = direct_solve(
                NamedCType(
                    SpecialArgName.possibly_redundant_memory_format,
                    OptionalCType(BaseCType(memoryFormatT)),
                )
            )
            # No need to join "memory_format" and "options" if the target API takes "options" directly.
            # Otherwise it will cause the redundant memory_format error.
            if options_ctype in goal_ctypes:
                return memory_format
            try:
                options = direct_solve(options_ctype)
                return f"c10::impl::check_tensor_options_and_extract_memory_format({options}, {memory_format})"
            except UnsatError:
                return memory_format
        elif goal == NamedCType("options", BaseCType(tensorOptionsT)):
            dtype = direct_solve(
                NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT)))
            )
            pin_memory = direct_solve(
                NamedCType("pin_memory", OptionalCType(BaseCType(boolT)))
            )
            device = direct_solve(
                NamedCType("device", OptionalCType(BaseCType(deviceT)))
            )
            layout = direct_solve(
                NamedCType("layout", OptionalCType(BaseCType(layoutT)))
            )
            return f"TensorOptions().dtype({dtype}).layout({layout}).device({device}).pinned_memory({pin_memory})"

        elif goal == NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT))):
            try:
                options = direct_solve(options_ctype)
                return f"c10::optTypeMetaToScalarType({options}.dtype_opt())"
            except UnsatError:
                out_tensor = direct_solve(out_tensor_ctype)
                return f"{out_tensor}.scalar_type()"

        elif goal == NamedCType("layout", OptionalCType(BaseCType(layoutT))):
            try:
                options = direct_solve(options_ctype)
                return f"{options}.layout_opt()"
            except UnsatError:
                out_tensor = direct_solve(out_tensor_ctype)
                return f"{out_tensor}.layout()"

        elif goal == NamedCType("device", OptionalCType(BaseCType(deviceT))):
            try:
                options = direct_solve(options_ctype)
                return f"{options}.device_opt()"
            except UnsatError:
                out_tensor = direct_solve(out_tensor_ctype)
                return f"{out_tensor}.device()"

        elif goal == NamedCType("pin_memory", OptionalCType(BaseCType(boolT))):
            try:
                options = direct_solve(options_ctype)
                return f"{options}.pinned_memory_opt()"
            except UnsatError:
                # If we're calling a factory op from its out= variant,
                # We don't actually care about the value of pin_memory.
                out_tensor = direct_solve(out_tensor_ctype)
                return "::std::nullopt"

        # We can always do translations from value types to reference types, like vector<int> -> IntArrayRef
        elif goal.type == BaseCType(intArrayRefT):
            try:
                return direct_solve(NamedCType(goal.name, longVec_ctype))
            except UnsatError:
                # We can also go SymIntArrayRef -> IntArrayRef
                symIntArrayRef_type = direct_solve(
                    NamedCType(goal.name, BaseCType(symIntArrayRefT))
                )
                return f"C10_AS_INTARRAYREF_SLOW({symIntArrayRef_type})"
        elif goal.type == BaseCType(symIntArrayRefT):
            try:
                r = direct_solve(NamedCType(goal.name, BaseCType(intArrayRefT)))
                return f"c10::fromIntArrayRefSlow({r})"
            except UnsatError:
                return direct_solve(NamedCType(goal.name, longSymVec_ctype))
        elif goal.type == BaseCType(SymIntT):
            return direct_solve(NamedCType(goal.name, BaseCType(longT)))
        elif goal.type == OptionalCType(BaseCType(SymIntT)):
            argname = direct_solve(
                NamedCType(goal.name, OptionalCType(BaseCType(longT)))
            )
            return f"{argname}.has_value() ? ::std::make_optional(c10::SymInt(*{argname})) : ::std::nullopt"
        elif goal.type == BaseCType(longT):
            symInt_type = direct_solve(NamedCType(goal.name, BaseCType(SymIntT)))
            return f"{symInt_type}.guard_int(__FILE__, __LINE__)"
        elif goal.type == OptionalCType(BaseCType(longT)):
            argname = direct_solve(
                NamedCType(goal.name, OptionalCType(BaseCType(SymIntT)))
            )
            return f"{argname}.has_value() ? ::std::make_optional({argname}->guard_int(__FILE__, __LINE__)) : ::std::nullopt"
        elif goal.type == BaseCType(optionalIntArrayRefT):
            try:
                return direct_solve(NamedCType(goal.name, optionalLongVec_ctype))
            except UnsatError:
                argname = direct_solve(
                    NamedCType(goal.name, BaseCType(optionalSymIntArrayRefT))
                )
                return f"{argname}.has_value() ? ::std::make_optional(C10_AS_INTARRAYREF_SLOW(*{argname})) : ::std::nullopt"
        elif goal.type == BaseCType(optionalSymIntArrayRefT):
            # TODO: You might also want to solve this from longSymVec_ctype or
            # an optional version of it
            argname = direct_solve(
                NamedCType(goal.name, BaseCType(optionalIntArrayRefT))
            )
            return f"{argname}.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*{argname})) : ::std::nullopt"
        elif goal.type == BaseCType(optionalScalarRefT):
            return direct_solve(NamedCType(goal.name, optionalScalar_ctype))
        elif goal.type == BaseCType(optionalTensorRefT):
            return direct_solve(NamedCType(goal.name, optionalTensor_ctype))

        # Note [translation from C++ reference to value types]
        # The below cases are all for when we have an argument with a reference type,
        # and a corresponding goal with a value type.
        # These are needed when we populate the inputs to a lambda capture and we need
        # to guarantee the lifetime of each captured argument.
        # We guard it with an explicit kwarg because converting to a value type is expensive
        # (O(n)) to convert from IntArrayRef to vector<int>),
        # so the caller of translate() should be explicit that they need it.
        if allow_expensive_conversions:
            if goal.type == VectorCType(BaseCType(longT)):
                intArrayRef_ctype = NamedCType(goal.name, BaseCType(intArrayRefT))
                argname = direct_solve(intArrayRef_ctype)
                return f"{argname}.vec()"
            if goal.type == VectorCType(BaseCType(SymIntT)):
                symIntArrayRef_ctype = NamedCType(goal.name, BaseCType(symIntArrayRefT))
                argname = direct_solve(symIntArrayRef_ctype)
                return f"{argname}.vec()"
            if goal.type == BaseCType(tensorT):
                const_ref_tensor_ctype = NamedCType(
                    goal.name, ConstRefCType(BaseCType(tensorT))
                )
                return direct_solve(const_ref_tensor_ctype)
            if goal.type == OptionalCType(BaseCType(tensorT)):
                const_ref_optional_tensor_ctype = NamedCType(
                    goal.name, ConstRefCType(OptionalCType(BaseCType(tensorT)))
                )
                return direct_solve(const_ref_optional_tensor_ctype)
            elif goal.type == OptionalCType(VectorCType(BaseCType(longT))):
                optionalIntArrayRef_ctype = NamedCType(
                    goal.name, BaseCType(optionalIntArrayRefT)
                )
                argname = direct_solve(optionalIntArrayRef_ctype)
                return f"{argname}.has_value() ? ::std::make_optional({argname}->vec()) : ::std::nullopt"
            elif goal.type == OptionalCType(BaseCType(scalarT)):
                optionalScalarRef_ctype = NamedCType(
                    goal.name, BaseCType(optionalScalarRefT)
                )
                argname = direct_solve(optionalScalarRef_ctype)
                return f"{argname}.has_value() ? ::std::make_optional({argname}) : ::std::nullopt"
            elif goal.type == OptionalCType(BaseCType(tensorT)):
                optionalTensorRef_ctype = NamedCType(
                    goal.name, BaseCType(optionalTensorRefT)
                )
                argname = direct_solve(optionalTensorRef_ctype)
                return f"{argname}.has_value() ? ::std::make_optional({argname}) : ::std::nullopt"
            # Technically, we also need to handle cases of C++ containers holding reference types.
            # But there currently aren't any ops that require lambda capture codegen
            # With arguments like ::std::vector<IntArrayRef>.
            # If that changes, we'll have to add the translation here.

        # We allow const casting on tensors, since const-correctness is a bit broken for at::Tensor.
        # We could probably generalize this to non-tensor types too.
        if goal.type == MutRefCType(BaseCType(tensorT)):
            const_ref_tensor_ctype = NamedCType(
                goal.name, ConstRefCType(BaseCType(tensorT))
            )
            argname = direct_solve(const_ref_tensor_ctype)
            return f"const_cast<Tensor&>({argname})"

        unsat(goal)

    return [Expr(solve(g, direct=False), g) for g in goal_ctypes]
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import assert_never

from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import (
    ArgName,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    deviceT,
    layoutT,
    ListCType,
    MutRefCType,
    NamedCType,
    OptionalCType,
    scalarT,
    scalarTypeT,
    tensorT,
)
from torchgen.model import (
    Argument,
    FunctionSchema,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


# This file describes the translation of JIT schema to the native functions API.
# This looks a lot like the C++ API (which makes historical sense, because the
# idea was you wrote native functions to implement functions in the C++ API),
# but over time we have evolved the C++ API without actually changing our
# native:: kernels.  The intention is to make native API and dispatcher API
# line up as closely as possible, since this results in the least overhead
# (no translation is needed from dispatcher API to native API).
#
# NB: this is symint aware, you will get the non-SymInt variant for some
# dispatch entries and SymInt for others.


def name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    # TODO: delete this!
    if func.is_out_fn():
        name += "_out"
    if func.name.overload_name:
        name += f"_{func.name.overload_name}"
    return name


def argumenttype_type(
    t: Type, *, mutable: bool, binds: ArgName, symint: bool
) -> NamedCType:
    if str(t) == "Tensor?":
        tensor_type: OptionalCType = OptionalCType(BaseCType(tensorT))
        if mutable and not local.use_const_ref_for_mutable_tensors():
            return NamedCType(binds, MutRefCType(tensor_type))
        else:
            return NamedCType(binds, ConstRefCType(tensor_type))
    elif str(t) == "Tensor?[]":
        return NamedCType(
            binds, ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT))))
        )
    elif str(t) == "Scalar":
        return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
    elif str(t) == "Scalar?":
        return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
    return cpp.argumenttype_type(t, mutable=mutable, binds=binds, symint=symint)


def returns_type(rs: Sequence[Return], *, symint: bool) -> CType:
    return cpp.returns_type(rs, symint=symint)


def argument_type(a: Argument, *, binds: ArgName, symint: bool) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds, symint=symint)


def argument(
    a: Argument | SelfArgument | TensorOptionsArguments,
    *,
    is_out: bool,
    symint: bool,
) -> list[Binding]:
    # Ideally, we NEVER default native functions.  However, there are a number
    # of functions that call native:: directly and rely on the defaulting
    # existing.  So for BC, we generate defaults for non-out variants (but not
    # for out variants, where it is impossible to generate an appropriate
    # default)
    should_default = not is_out
    if isinstance(a, Argument):
        default: str | None = None
        if should_default and a.default is not None:
            default = cpp.default_expr(a.default, a.type, symint=symint)
        return [
            Binding(
                nctype=argument_type(a, binds=a.name, symint=symint),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    elif isinstance(a, SelfArgument):
        # Erase SelfArgument from the distinction
        return argument(a.argument, is_out=is_out, symint=symint)
    elif isinstance(a, TensorOptionsArguments):
        default = None
        if should_default:
            default = "{}"
        # TODO: Not sure why the arguments assigned here are for
        # TensorOptionsArguments and not the constituent pieces.  It seems
        # to matter
        return [
            Binding(
                nctype=NamedCType("dtype", OptionalCType(BaseCType(scalarTypeT))),
                name="dtype",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("layout", OptionalCType(BaseCType(layoutT))),
                name="layout",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("device", OptionalCType(BaseCType(deviceT))),
                name="device",
                default=default,
                argument=a,
            ),
            Binding(
                nctype=NamedCType("pin_memory", OptionalCType(BaseCType(boolT))),
                name="pin_memory",
                default=default,
                argument=a,
            ),
        ]
    else:
        assert_never(a)


def arguments(func: FunctionSchema, *, symint: bool) -> list[Binding]:
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    args.extend(func.arguments.non_out)
    args.extend(func.arguments.out)
    return [
        r for arg in args for r in argument(arg, symint=symint, is_out=func.is_out_fn())
    ]
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import assert_never

from torchgen import local
from torchgen.api.types import (
    ArgName,
    ArrayCType,
    ArrayRefCType,
    BaseCType,
    BaseTypeToCppMapping,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    intArrayRefT,
    iTensorListRefT,
    ListCType,
    longT,
    MutRefCType,
    NamedCType,
    OptionalCType,
    optionalIntArrayRefT,
    optionalSymIntArrayRefT,
    scalarT,
    SpecialArgName,
    symIntArrayRefT,
    SymIntT,
    tensorListT,
    tensorOptionsT,
    tensorT,
    TupleCType,
    VectorCType,
    voidT,
)
from torchgen.model import (
    Argument,
    Arguments,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


# This file describes the translation of JIT schema to the public C++
# API, which is what people use when they call functions like at::add.
#
# Prominent characteristics of the C++ API:
#
#   - dtype, layout, device and pin_memory are collected into
#     a single C++ type TensorOptions  (the native functions API
#     also has this, but tensor options is really most relevant
#     for the C++ API; it makes calling kwarg factory functions
#     pleasant)
#
#   - defaulting lives here (in fact, the dispatcher is completely
#     oblivious of defaults!)
#
# BTW: policy on name collisions: we try not to have types with
# collisions, but functions are fair game to collide


def name(
    func: FunctionSchema,
    *,
    faithful_name_for_out_overloads: bool = False,
    symint_overload: bool = False,
) -> str:
    name = str(func.name.name)
    if symint_overload:
        name += "_symint"
    if func.is_out_fn():
        if faithful_name_for_out_overloads:
            name += "_outf"
        else:
            name += "_out"

    return name


# Translation of "value types" in JIT schema to C++ API type.  Value
# types look the same no matter if they are argument types or return
# types.  Returns None if the type in question is not a value type.
def valuetype_type(
    t: Type,
    *,
    binds: ArgName,
    mutable: bool = True,
    symint: bool = False,
) -> NamedCType | None:
    if isinstance(t, BaseType):
        if t.name in (BaseTy.Tensor, BaseTy.Scalar):
            return None
        elif str(t) == "SymInt":
            if symint:
                return NamedCType(binds, BaseCType(SymIntT))
            else:
                return NamedCType(binds, BaseCType(longT))
        # All other BaseType currently map directly to BaseCppTypes.
        return NamedCType(binds, BaseCType(BaseTypeToCppMapping[t.name]))
    elif isinstance(t, OptionalType):
        elem = valuetype_type(t.elem, binds=binds, mutable=mutable, symint=symint)
        if elem is None:
            return None
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if str(t.elem) == "bool":
            if t.size is None:
                raise AssertionError("bool ListType must have a size")
            return NamedCType(binds, ArrayCType(BaseCType(boolT), t.size))
        else:
            return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Translation of types occurring in JIT arguments to a C++ argument type.
# If remove_non_owning_ref_types is set, we'll guarantee that the output CType is not a non-owning reference type.
# For example, we'll return std::vector<int> instead of IntArrayRef, and at::Tensor instead of const at::Tensor&.
# See Note [translation from C++ reference to value types]
def argumenttype_type(
    t: Type,
    *,
    mutable: bool,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = False,
) -> NamedCType:
    # If it's a value type, do the value type translation
    r = valuetype_type(
        t,
        binds=binds,
        mutable=mutable,
        symint=symint,
    )
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(binds, MutRefCType(BaseCType(tensorT)))
            elif remove_non_owning_ref_types:
                return NamedCType(binds, BaseCType(tensorT))
            else:
                return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            if remove_non_owning_ref_types:
                return NamedCType(binds, BaseCType(scalarT))
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if str(t.elem) == "Tensor":
            if mutable and not local.use_const_ref_for_mutable_tensors():
                return NamedCType(
                    binds, MutRefCType(BaseCType(tensorT))
                )  # TODO: fix this discrepancy
            elif remove_non_owning_ref_types:
                return NamedCType(binds, OptionalCType(BaseCType(tensorT)))
            else:
                return NamedCType(
                    binds, ConstRefCType(OptionalCType(BaseCType(tensorT)))
                )
        elif str(t.elem) == "Scalar":
            if remove_non_owning_ref_types:
                return NamedCType(binds, OptionalCType(BaseCType(scalarT)))
            return NamedCType(binds, ConstRefCType(OptionalCType(BaseCType(scalarT))))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "int":
            return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "SymInt":
            if symint:
                return NamedCType(binds, BaseCType(optionalSymIntArrayRefT))
            else:
                return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds, symint=symint)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        # TODO: remove these special cases, ArrayRef fallthrough works fine
        if str(t.elem) == "int":
            if remove_non_owning_ref_types:
                return NamedCType(binds, VectorCType(BaseCType(longT)))
            else:
                return NamedCType(binds, BaseCType(intArrayRefT))
        if str(t.elem) == "SymInt":
            if remove_non_owning_ref_types:
                if symint:
                    return NamedCType(binds, VectorCType(BaseCType(SymIntT)))
                else:
                    return NamedCType(binds, VectorCType(BaseCType(longT)))
            else:
                if symint:
                    return NamedCType(binds, BaseCType(symIntArrayRefT))
                else:
                    return NamedCType(binds, BaseCType(intArrayRefT))
        if str(t.elem) == "Tensor":
            if local.use_ilistref_for_tensor_lists():
                return NamedCType(binds, ConstRefCType(BaseCType(iTensorListRefT)))
            else:
                return NamedCType(binds, BaseCType(tensorListT))
        elif str(t.elem) == "Scalar":
            return NamedCType(binds, ArrayRefCType(BaseCType(scalarT)))
        elif str(t.elem) == "Tensor?":
            return NamedCType(
                binds, ConstRefCType(ListCType(OptionalCType(BaseCType(tensorT))))
            )
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds, symint=symint)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Translate a JIT argument into its C++ type
def argument_type(a: Argument, *, binds: ArgName, symint: bool = False) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, symint=symint, binds=binds)


# Translation of a (non-multi) return type from JIT to C++
# N.B: returntype_type returns a CType, not a NamedCType.
# This is mostly because of the mismatch between return types and return names.
# e.g. a function with a return type of 'void' has 0 return names,
# and a function with a return type of 'std::tuple' has >1 return name.
def returntype_type(t: Type, *, mutable: bool, symint: bool = False) -> CType:
    # placeholder is ignored
    # NB: symint is ALWAYS respected for return types.  So symint argument
    # here is IGNORED
    r = valuetype_type(t, binds="__placeholder__", mutable=mutable, symint=True)
    if r is not None:
        return r.type

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                if local.use_const_ref_for_mutable_tensors():
                    return ConstRefCType(BaseCType(tensorT))
                else:
                    return MutRefCType(BaseCType(tensorT))
            else:
                # Note [Tensor Copy Returns]
                # Currently, we use "Argument.is_write" to determine
                # whether or not Tensor return types should be copies or references.
                # If that ever changes, take a look at other locations of this note!
                return BaseCType(tensorT)
        elif t.name == BaseTy.Scalar:
            return BaseCType(scalarT)
    elif isinstance(t, ListType):
        if mutable:
            raise AssertionError(
                "Native functions should never return a mutable tensor list. "
                "They should return void."
            )
        elem = returntype_type(t.elem, mutable=False)
        if t.size is not None:
            raise AssertionError(f"fixed size list returns not supported: {t}")
        return VectorCType(elem)
    elif isinstance(t, OptionalType):
        elem = returntype_type(t.elem, mutable=mutable)
        if str(t.elem) == "Tensor":
            return OptionalCType(elem)

    raise AssertionError(f"unrecognized return type {t}")


# Translation of a single return to its C++ type
def return_type(r: Return, *, symint: bool = False) -> CType:
    return returntype_type(r.type, mutable=r.is_write, symint=symint)


# Translation of a full (possibly multi) return from JIT to its C++ type
def returns_type(rs: Sequence[Return], *, symint: bool = False) -> CType:
    if len(rs) == 0:
        return BaseCType(voidT)
    elif len(rs) == 1:
        return return_type(rs[0], symint=symint)
    else:
        return TupleCType([return_type(r, symint=symint) for r in rs])


def return_names(f: NativeFunction, *, fallback_name: str = "result") -> Sequence[str]:
    returns: list[str] = []
    for i, r in enumerate(f.func.returns):
        # If we have an inplace function, the return argument is
        # implicitly named self.
        # TODO: Consider incorporating this into the data model
        if f.func.name.name.inplace:
            if i != 0:
                raise AssertionError("illegal inplace function with multiple returns")
            name = "self"
        # If we are out function, the name is the name of the
        # corresponding output function (r.name will get recorded
        # in field_name later.)
        elif f.func.is_out_fn():
            name = f.func.arguments.out[i].name
        # If the return argument is explicitly named...
        elif r.name:
            name_conflict = any(
                r.name == a.name for a in f.func.schema_order_arguments()
            )
            if name_conflict and not f.func.is_out_fn():
                name = f"{r.name}_return"
            else:
                name = r.name
        # If there is no explicit name and no fallback name was passed in, we just name the output result,
        # unless it's a multi-return, in which case it's result0,
        # result1, etc (zero-indexed)
        else:
            name = fallback_name if len(f.func.returns) == 1 else f"{fallback_name}{i}"
        returns.append(name)
    return returns


JIT_TO_CPP_DEFAULT = {
    "False": "false",
    "True": "true",
    "None": "::std::nullopt",  # UGH this one is type directed
    "Mean": "at::Reduction::Mean",
    "[]": "{}",
    "contiguous_format": "c10::MemoryFormat::Contiguous",
    "long": "at::kLong",
}


# Convert a JIT default into C++ expression representing the default
def default_expr(d: str, t: Type, *, symint: bool) -> str:
    if d == "None" and str(t) == "Tensor?":
        return "{}"
    if isinstance(t, BaseType) and t.name is BaseTy.str:
        # Schema allows single quotes but C++ needs double
        if len(d) >= 2 and d[0] == "'" and d[-1] == "'":
            s = ""
            i = 1
            while i + 1 < len(d):
                if d[i] != "\\":
                    if d[i] == '"':
                        s += '\\"'
                    else:
                        s += d[i]
                    i += 1
                else:
                    if d[i + 1] == "'":
                        s += "'"
                    else:
                        s += d[i : i + 2]
                    i += 2

            return f'"{s}"'

    if isinstance(t, OptionalType):
        if d == "None":
            return "::std::nullopt"

        return default_expr(d, t.elem, symint=symint)

    if isinstance(t, ListType):
        if d.startswith("[") and d.endswith("]"):
            return "{" + d[1:-1] + "}"
        elif symint and d.isdigit() and str(t.elem) == "SymInt":
            return f"c10::SymInt({d})"
        elif t.size is None:
            # NOTE: Sized lists can have scalar defaults
            raise ValueError(f"Expected a list default '[...]' but found: '{d}'")

    return JIT_TO_CPP_DEFAULT.get(d, d)


# Convert an argument into its C++ API form


def argument(
    a: Argument | TensorOptionsArguments | SelfArgument,
    *,
    cpp_no_default_args: set[str],
    method: bool,
    faithful: bool,
    symint: bool = False,
    has_tensor_options: bool,
) -> list[Binding]:
    def sub_argument(
        a: Argument | TensorOptionsArguments | SelfArgument,
    ) -> list[Binding]:
        return argument(
            a,
            cpp_no_default_args=cpp_no_default_args,
            method=method,
            faithful=faithful,
            symint=symint,
            has_tensor_options=has_tensor_options,
        )

    if isinstance(a, Argument):
        binds: ArgName
        if a.name == "memory_format" and has_tensor_options:
            binds = SpecialArgName.possibly_redundant_memory_format
        else:
            binds = a.name
        default: str | None = None
        if a.name not in cpp_no_default_args and a.default is not None:
            default = default_expr(a.default, a.type, symint=symint)
        return [
            Binding(
                nctype=argument_type(a, binds=binds, symint=symint),
                name=a.name,
                default=default,
                argument=a,
            )
        ]
    elif isinstance(a, TensorOptionsArguments):
        if faithful:
            return (
                sub_argument(a.dtype)
                + sub_argument(a.layout)
                + sub_argument(a.device)
                + sub_argument(a.pin_memory)
            )
        else:
            default = None
            # Enforced by NativeFunction.__post_init__
            if "options" in cpp_no_default_args:
                raise AssertionError("'options' should not be in cpp_no_default_args")
            if all(x.default == "None" for x in a.all()):
                default = "{}"
            elif a.dtype.default == "long":
                default = "at::kLong"  # TODO: this is wrong
            return [
                Binding(
                    nctype=NamedCType("options", BaseCType(tensorOptionsT)),
                    name="options",
                    default=default,
                    argument=a,
                )
            ]
    elif isinstance(a, SelfArgument):
        if method:
            # Caller is responsible for installing implicit this in context!
            return []
        else:
            return sub_argument(a.argument)
    else:
        assert_never(a)


def arguments(
    arguments: Arguments,
    *,
    faithful: bool,
    symint: bool = False,
    method: bool,
    cpp_no_default_args: set[str],
) -> list[Binding]:
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    if faithful:
        args.extend(arguments.non_out)
        args.extend(arguments.out)
    else:
        args.extend(arguments.out)
        args.extend(arguments.non_out)
    return [
        r.no_default() if faithful else r
        for a in args
        for r in argument(
            a,
            faithful=faithful,
            symint=symint,
            method=method,
            has_tensor_options=arguments.tensor_options is not None,
            cpp_no_default_args=cpp_no_default_args,
        )
    ]
<|endoftext|>from __future__ import annotations

from typing_extensions import assert_never

from torchgen.api import cpp
from torchgen.api.types import (
    ArgName,
    ArrayRefCType,
    BaseCType,
    Binding,
    ConstRefCType,
    intArrayRefT,
    iOptTensorListRefT,
    iTensorListRefT,
    NamedCType,
    OptionalCType,
    optionalIntArrayRefT,
    optionalScalarRefT,
    optionalTensorRefT,
    scalarT,
    tensorT,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    ListType,
    NativeFunctionsGroup,
    OptionalType,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)


# This file describes the translation of JIT schema to the structured functions API.
# This is similar to native API, but a number of historical problems with native
# API have been fixed.


# Translation of types occurring in JIT arguments to a C++ argument type.
# NB: For now, mutable doesn't do anything; but it could if we make
# some more nominal types
def argumenttype_type(t: Type, *, mutable: bool, binds: ArgName) -> NamedCType:
    # If it's a value type, do the value type translation
    # NB: structured kernels ALWAYS have symint off, since they involve actual
    # kernels that require real ints.  The one exception is the
    # CompositeExplicitAutograd and the meta function (which could
    # hypothetically be SymInt), but for simplicity we plan for these to just
    # be handled in Python
    r = cpp.valuetype_type(t, symint=False, binds=binds, mutable=mutable)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return NamedCType(binds, ConstRefCType(BaseCType(tensorT)))
        elif t.name == BaseTy.Scalar:
            return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
        else:
            raise AssertionError(f"base type should have been value type {t}")
    elif isinstance(t, OptionalType):
        if t.elem == BaseType(BaseTy.Tensor):
            return NamedCType(binds, BaseCType(optionalTensorRefT))
        elif t.elem == BaseType(BaseTy.Scalar):
            return NamedCType(binds, BaseCType(optionalScalarRefT))
        elif isinstance(t.elem, ListType) and str(t.elem.elem) == "int":
            return NamedCType(binds, BaseCType(optionalIntArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, OptionalCType(elem.type))
    elif isinstance(t, ListType):
        if t.elem == BaseType(BaseTy.Tensor):
            return NamedCType(binds, ConstRefCType(BaseCType(iTensorListRefT)))
        elif t.elem == OptionalType(BaseType(BaseTy.Tensor)):
            return NamedCType(binds, BaseCType(iOptTensorListRefT))
        # TODO: delete these special cases; see torchgen.api.cpp--these
        # must be changed in tandem, but there are problems; see
        # https://github.com/pytorch/pytorch/pull/51485
        elif str(t.elem) == "int":
            return NamedCType(binds, BaseCType(intArrayRefT))
        elem = argumenttype_type(t.elem, mutable=mutable, binds=binds)
        return NamedCType(binds, ArrayRefCType(elem.type))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


def argument_type(a: Argument, *, binds: ArgName) -> NamedCType:
    return argumenttype_type(a.type, mutable=a.is_write, binds=binds)


# returns_type intentionally omitted, because structured kernels never "return";
# instead, they always indirectly report their outputs (in the case of a meta
# function, by calling set_output; in the case of an impl function, by writing
# directly into the provided out argument).


# Structured kernels are never defaulted
def argument(a: Argument | SelfArgument | TensorOptionsArguments) -> list[Binding]:
    if isinstance(a, Argument):
        return [
            Binding(
                nctype=argument_type(a, binds=a.name),
                name=a.name,
                default=None,
                argument=a,
            )
        ]
    elif isinstance(a, SelfArgument):
        return argument(a.argument)
    elif isinstance(a, TensorOptionsArguments):
        raise AssertionError("structured kernels don't support TensorOptions yet")
    else:
        assert_never(a)


def impl_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []

    if g.out.precomputed:
        # A list of parameters for the impl function with
        # certain parameters replaced with precomputed counterparts
        # as specified in native_functions.yaml.
        non_out_args_replaced: list[
            Argument | TensorOptionsArguments | SelfArgument
        ] = []
        for a in g.out.func.arguments.non_out:
            if isinstance(a, Argument) and a.name in g.out.precomputed.replace:
                # If a is in precompute.replace, append the parameters
                # that should replace it onto non_out_args_replaced.
                non_out_args_replaced.extend(g.out.precomputed.replace[a.name])
            else:
                # If not, push a as it is.
                non_out_args_replaced.append(a)

        args.extend(non_out_args_replaced)
        # g.out.precomputed.add is the list of parameters that are added
        # without replacement after the non out args and just before the out args
        args.extend(g.out.precomputed.add)
    else:
        args.extend(g.out.func.arguments.non_out)

    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]


def meta_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    args.extend(g.functional.func.arguments.non_out)
    return [r for arg in args for r in argument(arg)]


def out_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    args: list[Argument | TensorOptionsArguments | SelfArgument] = []
    args.extend(g.out.func.arguments.out)
    return [r for arg in args for r in argument(arg)]
<|endoftext|>from __future__ import annotations

from torchgen.api import dispatcher
from torchgen.api.types import (
    BaseCppType,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    longT,
    NamedCType,
    tensorT,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsViewGroup,
)


# This file describes the translation of JIT schema to API's used
# when creating `ViewMeta` specializations that are used by the functionalization pass.
# These API's mostly follow the dispatcher API, with one difference:
# - While the forward function just directly calls into the at::_ops API
#   (following the dispatcher convention), the logic here for the reverse function
#   is responsible for generating both the call-site, and the declarations
#   (which are implemented manually in the at::functionalization::impl namespace).

# Define some specific lambda input arguments.
base_binding = Binding(
    name="base",
    nctype=NamedCType(name="base", type=ConstRefCType(BaseCType(tensorT))),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)

has_symbolic_inputs_binding = Binding(
    name="has_symbolic_inputs",
    nctype=NamedCType(name="has_symbolic_inputs", type=BaseCType(boolT)),
    argument=Argument(
        name="has_symbolic_inputs",
        type=BaseType(BaseTy.bool),
        default=None,
        annotation=None,
    ),
    default=None,
)
mutated_view_binding = Binding(
    name="mutated_view",
    nctype=NamedCType(name="mutated_view", type=ConstRefCType(BaseCType(tensorT))),
    argument=Argument(
        name="base", type=BaseType(BaseTy.Tensor), default=None, annotation=None
    ),
    default=None,
)
out_index_binding = Binding(
    name="out_index",
    nctype=NamedCType(name="out_index", type=BaseCType(longT)),
    argument=Argument(
        name="out_index", type=BaseType(BaseTy.int), default=None, annotation=None
    ),
    default=None,
)
reapply_views_binding = Binding(
    name="reapply_views",
    nctype=NamedCType(name="reapply_views", type=BaseCType(boolT)),
    argument=Argument(
        name="reapply_views", type=BaseType(BaseTy.bool), default=None, annotation=None
    ),
    default=None,
)

InverseReturnModeT = BaseCppType("at::functionalization", "InverseReturnMode")
inverse_return_mode_binding = Binding(
    name="inverse_return_mode",
    nctype=NamedCType(name="inverse_return_mode", type=BaseCType(InverseReturnModeT)),
    argument=Argument(
        name="inverse_return_mode",
        # NB: not actually a bool but it doesn't matter because this isn't used
        type=BaseType(BaseTy.bool),
        default=None,
        annotation=None,
    ),
    default=None,
)


# Name of the `ViewMeta` specialization class created.
def classname(func: FunctionSchema, with_namespace: bool = False) -> str:
    namespace = "at::functionalization::" if with_namespace else ""
    return f"{namespace}{func.name.unambiguous_name()}_ViewMeta"


# Name of the operation called inside the `forward`/`reverse` implementations.
def name(
    g: NativeFunctionsViewGroup,
    *,
    is_reverse: bool,
    include_namespace: bool,
    reapply_views: bool | None = None,
) -> str:
    if reapply_views is None:
        # reapply_views is only important for the fwd lambda,
        # since we always plumb the runtime "reapply_views" argument into the reverse function.
        if not is_reverse:
            raise AssertionError("reapply_views can only be None for reverse")
    if is_reverse:
        return reverse_name(g.view, include_namespace)
    # in the forward case, we just directly call into the at::_ops API (so we always need the namespace)
    if not include_namespace:
        raise AssertionError("include_namespace must be True for forward")
    if g.view_copy is None:
        raise AssertionError("view_copy must be non-None for forward")
    api_name = (
        g.view.func.name.unambiguous_name()
        if reapply_views
        else g.view_copy.func.name.unambiguous_name()
    )
    return f"at::_ops::{api_name}::call"


def reverse_name(f: NativeFunction, include_namespace: bool) -> str:
    # for the reverse: we plumb the "reapply_views" flag into that function and support
    # both copy and non-copy variants. (We could avoid doing that, but that would require
    # writing out twice as many view inverse functions).
    api_name = f.func.name.unambiguous_name()
    # in the reverse case, we codegen both the call-sites (which need the full namespace) and the declarations (which don't)
    if include_namespace:
        return f"at::functionalization::FunctionalInverses::{api_name}_inverse"
    else:
        return f"{api_name}_inverse"


def returns_type(func: FunctionSchema) -> CType:
    # Assertion: all view ops return tensor-like outputs
    if len(func.returns) < 1:
        raise AssertionError("Expected at least one return value")
    for ret in func.returns:
        if not ret.type.is_tensor_like():
            raise AssertionError(f"Expected tensor-like return type, got {ret.type}")
    # However, the return type of the lambda is always an individual tensor.
    # For multi-tensor outputs, each tensor needs to be tracked individually.
    return BaseCType(tensorT)


# Checks whether `func` might return more than one value.
def is_multi_output(func: FunctionSchema) -> bool:
    return len(func.returns) > 1 or (
        len(func.returns) == 1 and func.returns[0].type.is_list_like() is not None
    )


# `ViewMeta` specialization constructor parameters.
def base_ctor_arguments(func: FunctionSchema) -> list[Binding]:
    # All specializations are parematerized by `has_symbolic_inputs` flag.
    arguments = [has_symbolic_inputs_binding]

    # If `func` might return more than 1 value, we also parameterize this specialization
    # with the output index.
    if is_multi_output(func):
        arguments.append(out_index_binding)

    return arguments


# `ViewMeta` specialized class' constructor arguments.
#
# Values needed specifically by this specialization, that the base class does not need.
# Same as the class' attributes, but non-owning.
def extra_ctor_arguments(func: FunctionSchema) -> list[Binding]:
    return attributes(func, owning=False)


# `ViewMeta` specialized class' non-static member data.
#
# Essential data for calling the instance's `forward` and `reverse functions. You can
# think of them as values that should be captured from the functionalization kernel.
def attributes(func: FunctionSchema, owning: bool = True) -> list[Binding]:
    args = func.arguments.flat_all
    if args[0].type != BaseType(BaseTy.Tensor):
        raise AssertionError(f"Expected first arg to be Tensor, got {args[0].type}")
    return [
        reapply_views_binding,
        inverse_return_mode_binding,
        *[dispatcher.argument(a, remove_non_owning_ref_types=owning) for a in args[1:]],
    ]


def op_arguments(func: FunctionSchema, is_reverse: bool) -> list[Binding]:
    args = func.arguments.flat_all
    if args[0].type != BaseType(BaseTy.Tensor):
        raise AssertionError(f"Expected first arg to be Tensor, got {args[0].type}")
    non_self_args = args[1:]
    # The forward lambda calls the at::_ops API, while the reverse lambda calls the view inverse API.
    # Both of these follow the dispatcher API.
    non_self_bindings = [dispatcher.argument(a) for a in non_self_args]
    if not is_reverse:
        # the forward lambda swaps out the original tensor argument with the lambd arg "base"
        return [base_binding] + non_self_bindings
    else:
        # the reverse lambda does the same, but with an additional "mutated_view" arg
        # additionally, we have a calling convention: for view ops that return multiple tensor outputs
        # their corresponding view_inverse function takes in an additional index argument.
        if is_multi_output(func):
            return [
                base_binding,
                mutated_view_binding,
                inverse_return_mode_binding,
                out_index_binding,
            ] + non_self_bindings
        else:
            return [
                base_binding,
                mutated_view_binding,
                inverse_return_mode_binding,
            ] + non_self_bindings
<|endoftext|>from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsViewGroup,
    SchemaKind,
    Type,
)
from torchgen.utils import IDENT_REGEX


if TYPE_CHECKING:
    from collections.abc import Sequence


# Represents a saved attribute involved in backward calculation.
# Note that it can be a derived property of an input argument, e.g.:
# we could save `other.scalar_type()` instead of the entire `other` tensor.
@dataclass(frozen=True)
class SavedAttribute:
    # The NamedCType holds the updated name and cpp type of the attribute
    # for the name, Suffix is appended if it's derived property, e.g.: `other_scalar_type`
    nctype: NamedCType

    # The expression to read the derived property at save time, e.g.:
    # `other.scalar_type()`.
    expr: str


# Represents a backward formula that calculates derivatives for one
# or more tensors.
@dataclass(frozen=True)
class Derivative:
    # The formula string (legit C++ expression).
    # Note that expressions against input arguments have been replaced with the
    # corresponding saved attributes.
    # E.g.:
    #  raw formula: `mul_tensor_backward(grad, self, other.scalar_type())`
    #         here: `mul_tensor_backward(grad, self, other_scalar_type)`
    formula: str

    # The formula string before input argument replacement
    original_formula: str

    # Names of the arguments for which this formula calculates derivatives.
    var_names: tuple[str, ...]

    # Saved inputs that are referenced by the formula.
    saved_inputs: tuple[SavedAttribute, ...]

    # Saved outputs that are referenced by the formula.
    saved_outputs: tuple[SavedAttribute, ...]

    # Gradients that are referenced by name in the formula.
    named_gradients: set[str]


# Represents a forward formula that calculates forward derivatives
# for one tensor.
@dataclass(frozen=True)
class ForwardDerivative:
    # The formula string (legit C++ expression).
    # Note that special keywords such as "linear" or "element_wise" have been
    # replaced by the automatically generated formula.
    formula: str

    # Name of the output arguments for which this formula calculates forward
    # derivatives
    var_names: tuple[str, ...]

    # Type of the output arguments for which this formula calculates forward
    # derivatives
    var_types: tuple[Type, ...]

    # Inputs for which the forward derivatives are required for this formula
    required_inputs_fw_grad: tuple[str, ...] | None

    # Inputs for which the primal is required for this formula
    required_inputs_primal: tuple[str, ...] | None

    # Flag to specify if this formula requires the original value of self
    # This is only used by inplace operations
    required_original_self_value: bool

    # If this formula is specified in derivatives.yaml or if we are reusing the
    # out of place formula for inplace
    is_reusing_outplace_formula: bool


# Represents differentiability info for a NativeFunction.
@dataclass(frozen=True)
class DifferentiabilityInfo:
    # The base name read from derivatives.yaml.
    name: str

    # The matching native function.
    #
    # There can be multiple NativeFunction having the same base name:
    #  - different overloads with different types of input arguments;
    #  - in-place/out/functional variants of the same function;
    #
    # We first use the schema string (under the 'name' key) in derivatives.yaml
    # to find the NativeFunction having the same schema string.
    # Then we find the in-place/out/functional variants of the matching function.
    # Among these variants, we choose the one having the same name as the
    # derivatives.yaml entry. If there is no exact match, then we choose the
    # in-place variant.
    # TODO: maybe the logic to search for all variants is no longer necessary?
    func: NativeFunction

    # The name of the generated autograd function.
    # It's set only if we will calculate a derivative, i.e.
    # 'args_with_derivatives' is not empty.
    op: str | None

    # The derivatives formulae for this function.
    # Note that the length of this sequence is the number of differentiable inputs
    derivatives: Sequence[Derivative]

    # The forward derivatives formulae for this function.
    # Note that the length of this sequence is the number of differentiable outputs
    forward_derivatives: Sequence[ForwardDerivative]

    # The union of 'saved_inputs' of all 'derivatives'.
    all_saved_inputs: Sequence[SavedAttribute]

    # The union of 'saved_outputs' of all 'derivatives'.
    all_saved_outputs: Sequence[SavedAttribute]

    # All named gradients that are available for use, in the same
    # order as in the grads vector.
    available_named_gradients: Sequence[str]

    # The named gradients that are used in any of the derivatives.
    # Invariant: all(name in available_named_gradients for name in used_named_gradients)
    used_named_gradients: set[str]

    # The function's input arguments for which it calculates derivatives.
    # It's the union of 'var_names' of all 'derivatives', sorted by the
    # argument order in the function schema.
    args_with_derivatives: Sequence[Binding]

    # Names of arguments whose derivative formula is 'non_differentiable'.
    non_differentiable_arg_names: Sequence[str]

    # Raw data read from derivatives.yaml.
    output_differentiability: list[bool] | None

    # output_differentiability in derivatives.yaml can be a list of
    # conditions that express if the output is differentiable. In this case,
    # the number of conditions must match the number of outputs
    # (NB: we only support one condition right now).
    # output_differentiability gets populated with True for each condition,
    # while output_differentiability_conditions gets populated with the conditions
    output_differentiability_conditions: list[str] | None

    @property
    def has_derivatives(self) -> bool:
        return len(self.args_with_derivatives) > 0

    # Generates a new DifferentiabilityInfo using the exact same set of derivative information,
    # but with a new operator name.
    # This is used when generating "copy" variants of view ops,
    # which are able to use the exact same derivative formula as the original view op
    # See Note [Codegen'd {view}_copy Operators]
    def create_view_copy_from_view_derivative(
        self, g: NativeFunctionsViewGroup
    ) -> DifferentiabilityInfo | None:
        if g.view_copy is None:
            return None
        f = g.view_copy

        name_split_by_period = self.name.split(".", maxsplit=2)
        # Append a "_copy" to the base name of the operator (but keep the overload name the same)
        view_copy_name = f"{name_split_by_period[0]}_copy." + ".".join(
            name_split_by_period[1:]
        )
        view_copy_op_name = None if self.op is None else f"{self.op}_copy"

        return DifferentiabilityInfo(
            # Use the "_copy" version of name/func/op
            name=view_copy_name,
            func=f,
            op=view_copy_op_name,
            # But keep all derivative info the same
            derivatives=self.derivatives,
            forward_derivatives=self.forward_derivatives,
            all_saved_inputs=self.all_saved_inputs,
            all_saved_outputs=self.all_saved_outputs,
            available_named_gradients=self.available_named_gradients,
            used_named_gradients=self.used_named_gradients,
            args_with_derivatives=self.args_with_derivatives,
            non_differentiable_arg_names=self.non_differentiable_arg_names,
            output_differentiability=self.output_differentiability,
            output_differentiability_conditions=self.output_differentiability_conditions,
        )


def uses_ident(info: DifferentiabilityInfo | None, ident: str) -> bool:
    if info is None:
        return False
    for derivative in info.derivatives:
        formula = derivative.formula
        if re.search(IDENT_REGEX.format(ident), formula):
            return True
    return False


def uses_retain_variables(info: DifferentiabilityInfo | None) -> bool:
    return uses_ident(info, "retain_variables")


def uses_single_grad(info: DifferentiabilityInfo | None) -> bool:
    return uses_ident(info, "grad")


# Represents a differentiable `Argument`.
# How is it different from the `Argument` type?
# - It's processed Arguments which are differentiable and only used in the
#   context of the autograd codegen;
# - It can represent SelfArgument or regular Argument but not TensorOptionsArgument;
@dataclass(frozen=True)
class DifferentiableInput:
    name: str
    type: Type

    # TODO: only to keep it byte-for-byte compatible with the old codegen, should remove.
    cpp_type: str


# Represents a differentiable `Return`.
# How is it different from the `Return` type?
# - The name in `Return` is optional. Here it is always populated using the same
#   `cpp.return_names()` method.
#   TODO: some cpp naming logic (e.g. resolving name conflict) might be irrelevant?
# - It's processed Returns which are differentiable, in compliance with the
#   `output_differentiability` field defined in derivatives.yaml (if specified),
#   and are only used in the context of the autograd codegen;
@dataclass(frozen=True)
class DifferentiableOutput:
    name: str
    type: Type

    # TODO: only to keep it byte-for-byte compatible with the old codegen, should remove.
    cpp_type: str


@dataclass(frozen=True)
class NativeFunctionWithDifferentiabilityInfo:
    func: NativeFunction
    info: dict[str, DifferentiabilityInfo] | None
    fw_derivatives: dict[str, Sequence[ForwardDerivative]] | None


# TODO: Update comment below since it is out of date.
def dispatch_strategy(fn: NativeFunctionWithDifferentiabilityInfo) -> str:
    """How are we going to call the underlying implementation of a
    declaration?  There are two strategies:
        - use_derived: we want to call the implementation on CPUDoubleType
          (or a similar, derived Type instance).  Because these derived
          instances deal in Tensors, not Variables (it's a completely different
          object, so it doesn't dispatch back to VariableType), code on
          this dispatch path needs to wrap/unwrap tensors.  If the
          derived implementation takes and returns tensors, the
          implementation is usually differentiable (although we also use
          the derived dispatch path for non-differentiable functions
          that we still want to dispatch on the derived Type instance;
          e.g., size())
        - use_type: we want to call the implementation on Type, because
          it is implemented concretely, and the functions it invokes will
          get dispatched back to VariableType (which will ensure that they
          are differentiable.)
    """
    # fn is derived as long as any of its per-key differentiability infos
    # has_derivatives. dispatch_strategy() is used to guard generation of fns in VariableType
    # and ADInplaceOrViewType. We want to generate these functions as long as a
    # derivative is defined for ANY dispatch key.
    if fn.func.is_abstract or (
        fn.info is not None and any(info.has_derivatives for info in fn.info.values())
    ):
        # If the function is abstract (not implemented on at::Type), we must
        # call the implementation on the derived type with unpacked tensors.

        # If the function has a derivative specified and is concrete, we could
        # call either implementation. We prefer the calling the derived
        # type's implementation with unpacked tensors because it is more
        # performant in some cases: any internal calls to other ATen functions
        # won't have the history tracked.

        # If the function has a type dispatched argument (i.e. is a factory),
        # we prefer calling the derived type's implementation both because it is
        # more performant and to ensure factory functions return tensors with _version
        # of 0 (probably not strictly necessary, but nice to have to keeps versions simple
        # to understand.

        return "use_derived"
    else:
        # If the function is concrete (we don't have to override it) and we
        # didn't declare it in derivatives.yaml, we'll assume that it is
        # actually implemented out of differentiable functions. (This
        # assumption might not hold, but then you'll see gradcheck fail.)
        return "use_type"


def is_foreach_func(f: NativeFunction) -> bool:
    return f.func.name.name.base.startswith("_foreach_")


# note(crcrpar): Most foreach functions can reference an out-place `torch` function whose schema kind
# is functional for their backward derivatives (and forward derivatives in the future), i.e.,
# they would find such one in `functional_info_by_signature`. There however are some exceptions:
_foreach_with_inplace_ref = {"_foreach_zero_"}
_foreach_with_tensor_overload = {
    "_foreach_add.Tensor",
    "_foreach_mul.Tensor",
    "_foreach_div.Tensor",
}
# The following do not support the alpha kwarg, which the nonforeach versions support.
_skip_argument_len_check = {
    "_foreach_add.Scalar",
    "_foreach_add_.Scalar",
    "_foreach_add.ScalarList",
    "_foreach_add_.ScalarList",
    "_foreach_sub.Scalar",
    "_foreach_sub_.Scalar",
    "_foreach_sub.ScalarList",
    "_foreach_sub_.ScalarList",
}


# Checks if `function_schema` is a native, non-foreach function which `f`, a foreach function
# reference to generate derivatives.
def is_reference_for_foreach(
    f: NativeFunction,
    function_schema: FunctionSchema,
) -> bool:
    return (
        f.func.name.name.base.split("_foreach_")[-1] == function_schema.name.name.base
        and (
            not function_schema.name.name.inplace
            or str(f.func.name) in _foreach_with_inplace_ref
        )
        and (
            str(f.func.name) in _skip_argument_len_check
            or len(f.func.arguments.flat_non_out)
            == len(function_schema.arguments.flat_non_out)
        )
        and all(
            ref_arg.type in (arg.type, getattr(arg.type, "elem", None))
            for arg, ref_arg in zip(
                f.func.arguments.flat_non_out,
                function_schema.arguments.flat_non_out,
            )
        )
    )


# TODO(crcrpar): Avoid hard coding "Default" ideally.
def gen_foreach_derivativeinfo(
    foreach_function: NativeFunction,
    functional_info_by_signature: dict[
        FunctionSchema, dict[str, DifferentiabilityInfo]
    ],
    non_functional_info_by_signature: dict[
        FunctionSchema, dict[str, DifferentiabilityInfo]
    ],
    dispatch_key: str = "Default",
) -> tuple[DifferentiabilityInfo | None, bool]:
    """Generate DifferentiabilityInfo for out-place foreach function, return the existing one for in-place.

    The second return value indicates whether the info is generated in this function.
    """
    ref_diff_info: DifferentiabilityInfo | None = None

    for function_schema, diff_info in functional_info_by_signature.items():
        if not is_reference_for_foreach(foreach_function, function_schema):
            continue
        ref_diff_info = diff_info[dispatch_key]
        if ref_diff_info is not None:
            break
    # note(crcrpar): It seems like `zero`'s info isn't available in functional_info_by_signature
    # while the info of `zero_` is in non_functional_info_by_signature
    if (
        ref_diff_info is None
        and foreach_function.func.kind() == SchemaKind.inplace
        and str(foreach_function.func.name) in _foreach_with_inplace_ref
    ):
        for function_schema, diff_info in non_functional_info_by_signature.items():
            if not is_reference_for_foreach(foreach_function, function_schema):
                continue
            ref_diff_info = diff_info[dispatch_key]
            if ref_diff_info is not None:
                break
    if ref_diff_info is None:
        return None, False

    # non out-place uses the existing Derivative.
    if foreach_function.func.kind() == SchemaKind.inplace:
        return ref_diff_info, False

    map_refarg2foreacharg, map_name2arg = {}, {}
    for i, (arg, ref_arg) in enumerate(
        zip(
            foreach_function.func.arguments.flat_non_out,
            function_schema.arguments.flat_non_out,
        )
    ):
        map_refarg2foreacharg[ref_arg.name] = arg.name
        map_name2arg[arg.name] = arg

    all_saved_inputs, all_saved_outputs, all_var_names = [], [], []
    modified_derivative_formulas = []
    for i, derivative in enumerate(ref_diff_info.derivatives):
        modified_formula = derivative.formula.replace("grad", "grads[i]").replace(
            "result", "result[i]"
        )
        saved_inputs, saved_outputs = [], []
        # note(crcrpar): This context seems necessary to call `cpp.argument_type`
        with local.parametrize(
            use_const_ref_for_mutable_tensors=foreach_function.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=foreach_function.part_of_structured_group,
        ):
            for ref_input in derivative.saved_inputs:
                ref_input_jit_name = ref_input.expr.split(".")[0]
                mapped_name = map_refarg2foreacharg[ref_input_jit_name]
                if isinstance(map_name2arg[mapped_name].type, ListType):
                    mapped_expr = mapped_name + "[i]"
                else:
                    mapped_expr = mapped_name
                new_expr = ref_input.expr.replace(ref_input_jit_name, mapped_expr)
                modified_formula = modified_formula.replace(
                    cast(str, ref_input.nctype.name), new_expr
                )

                nctype = cpp.argument_type(map_name2arg[mapped_name], binds=mapped_name)
                canonical_nctype = NamedCType(
                    nctype.name, nctype.type.remove_const_ref()
                )
                saved_inputs.append(
                    SavedAttribute(nctype=canonical_nctype, expr=mapped_name)
                )
            for ref_output in derivative.saved_outputs:
                if ref_output.nctype.name == "result":
                    saved_outputs.append(
                        SavedAttribute(
                            nctype=NamedCType(
                                name="result", type=BaseCType(tensorListT)
                            ),
                            expr="result",
                        )
                    )
                else:
                    raise RuntimeError("")
        var_names = [map_refarg2foreacharg[var] for var in derivative.var_names]
        all_var_names.extend(var_names)
        all_saved_inputs.extend(saved_inputs)
        all_saved_outputs.extend(saved_outputs)
        modified_derivative = Derivative(
            formula=modified_formula,
            original_formula=derivative.formula,
            var_names=tuple(var_names),
            saved_inputs=tuple(saved_inputs),
            saved_outputs=tuple(saved_outputs),
            named_gradients=set(),
        )
        modified_derivative_formulas.append(modified_derivative)

    with local.parametrize(
        use_const_ref_for_mutable_tensors=foreach_function.use_const_ref_for_mutable_tensors,
        use_ilistref_for_tensor_lists=foreach_function.part_of_structured_group,
    ):
        args_with_derivatives = [
            Binding(
                name=arg.name,
                nctype=cpp.argument_type(arg, binds=arg.name),
                argument=arg,
                default=None,
            )
            for arg in foreach_function.func.arguments.flat_non_out
            if arg.name in all_var_names
        ]

    forward_derivatives: list[ForwardDerivative] = []
    fw_derivative: ForwardDerivative
    for fw_derivative in ref_diff_info.forward_derivatives:
        var_names: list[str] = list(fw_derivative.var_names)  # type: ignore[no-redef]
        var_types: list[Type] = list(fw_derivative.var_types)
        required_inputs_fw_grad: list[str] = []
        required_inputs_primal: list[str] = []
        if fw_derivative.required_inputs_fw_grad is not None:
            required_inputs_fw_grad = list(fw_derivative.required_inputs_fw_grad)
        if fw_derivative.required_inputs_primal:
            required_inputs_primal = list(fw_derivative.required_inputs_primal)
        modified_formula = fw_derivative.formula

        # Foreach's result is TensorList
        if "result" in modified_formula:
            modified_formula = fw_derivative.formula.replace("result", "result[i]")

        for foreach_arg, ref_arg in zip(
            foreach_function.func.arguments.flat_non_out,
            ref_diff_info.func.func.arguments.flat_non_out,
        ):
            # Modify reference forward formula
            if (
                isinstance(foreach_arg.type, ListType)
                and not foreach_arg.type.is_tensor_like()
            ):
                # Assuming ScalarList
                modified_formula = modified_formula.replace(
                    ref_arg.name, foreach_arg.name + "[i]"
                )
            elif foreach_arg.type.is_tensor_like():
                # Assuming TensorList / Tensor
                if not (
                    isinstance(foreach_arg.type, ListType)
                    or (
                        foreach_arg.type == BaseType(BaseTy.Tensor)
                        and str(foreach_function.func.name)
                        in _foreach_with_tensor_overload
                    )
                ):
                    raise AssertionError(
                        f"{foreach_function.func.name}, {foreach_arg.type}"
                    )
                for suffix in ("_p", "_t"):
                    curr_expr = ref_arg.name + suffix
                    if curr_expr in modified_formula:
                        new_expr = foreach_arg.name + suffix
                        modified_formula = modified_formula.replace(curr_expr, new_expr)
            else:
                # Assuming Scalar
                if foreach_arg.name != ref_arg.name:
                    modified_formula = modified_formula.replace(
                        ref_arg.name, foreach_arg.name
                    )

            # note(crcrpar): there should exist a cooler way...
            for i, name in enumerate(var_names):
                if name == ref_arg.name:
                    var_names[i] = foreach_arg.name
                    var_types[i] = foreach_arg.type
            for i, name in enumerate(required_inputs_fw_grad):
                if name == ref_arg.name:
                    required_inputs_fw_grad[i] = foreach_arg.name
            for i, name in enumerate(required_inputs_primal):
                if name == ref_arg.name:
                    required_inputs_primal[i] = foreach_arg.name
        forward_derivatives.append(
            ForwardDerivative(
                formula=modified_formula,
                var_names=tuple(var_names),
                var_types=tuple(var_types),
                required_inputs_fw_grad=tuple(required_inputs_fw_grad),
                required_inputs_primal=tuple(required_inputs_primal),
                required_original_self_value=fw_derivative.required_original_self_value,
                is_reusing_outplace_formula=fw_derivative.is_reusing_outplace_formula,
            )
        )

    return (
        DifferentiabilityInfo(
            name=foreach_function.func.name.name.base,
            func=foreach_function,
            op=f"Foreach{ref_diff_info.op}{foreach_function.func.name.overload_name}",
            derivatives=modified_derivative_formulas,
            forward_derivatives=forward_derivatives,
            all_saved_inputs=tuple(set(all_saved_inputs)),
            all_saved_outputs=tuple(set(all_saved_outputs)),
            available_named_gradients=(),
            used_named_gradients=set(),
            args_with_derivatives=args_with_derivatives,
            non_differentiable_arg_names=[],
            output_differentiability=None,
            output_differentiability_conditions=None,
        ),
        True,
    )


def match_differentiability_info(
    native_functions: list[NativeFunction],
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
) -> list[NativeFunctionWithDifferentiabilityInfo]:
    """Sets the "derivative" key on declarations to matching autograd function
    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """

    functional_info_by_signature = {
        schema.signature(strip_default=True): info_dict
        for schema, info_dict in differentiability_infos.items()
        if schema.kind() == SchemaKind.functional
    }
    non_functional_info_by_signature = {
        schema.signature(strip_default=True): info_dict
        for schema, info_dict in differentiability_infos.items()
        if schema.kind() != SchemaKind.functional
    }

    def find_info(
        f: NativeFunction,
    ) -> tuple[dict[str, DifferentiabilityInfo] | None, bool]:
        # Don't bother matching info to generated out= variants
        if "generated" in f.tags and f.func.kind() == SchemaKind.out:
            return None, False

        # (1) Check for an exact match
        if f.func in differentiability_infos:
            return differentiability_infos[f.func], True

        # (2) If no exact match, check if the out-of-place variant
        # of this operator has a match.
        # i.e mul() for mul_() or mul_out()
        # note(crcrpar): Check foreach or not because in-place foreach functions use backward defined for the existing
        # native functions instead of the out-place counterparts.
        f_sig = f.func.signature(strip_default=True)
        if f_sig in functional_info_by_signature and not is_foreach_func(f):
            return functional_info_by_signature[f_sig], False

        # (3) Some operators have a derivative explicitly defined for the mutable
        # variant, but get a code-generated out-of-place variant which does *not*
        # come with a derivative formula.
        # For the generated out-of-place variant, use the mutable variant's formula
        # if it exists.
        if "generated" in f.tags and f_sig in non_functional_info_by_signature:
            info_dict = non_functional_info_by_signature[f_sig]
            # See https://github.com/pytorch/pytorch/pull/76320/files#r874816389
            if any(
                any("self" in str(input.nctype.name) for input in info.all_saved_inputs)
                for info in info_dict.values()
            ):
                raise AssertionError(
                    f"Attempted to convert a derivative formula for a mutable operator "
                    f'to be used automatically by its functional variant ("{str(f.func)}"). '
                    "This is not currently supported (we'd need to fix up the formula in the codegen)."
                )
            return info_dict, False

        # (4) Generate derivative information of foreach functions if none is defined in `derivatives.yaml`
        if is_foreach_func(f):
            if f.func in differentiability_infos:
                raise AssertionError(
                    f"Foreach function {f.func.name} already has differentiability info"
                )
            diff_info, is_generated = gen_foreach_derivativeinfo(
                f,
                functional_info_by_signature,
                non_functional_info_by_signature,
            )
            if diff_info is None:
                return None, False
            # TODO(crcrpar): Avoid hard coding "Default" ideally.
            diff_info_dict = {"Default": diff_info}
            if is_generated:
                differentiability_infos[f.func] = diff_info_dict
                functional_info_by_signature[f.func] = diff_info_dict
            return diff_info_dict, is_generated

        return None, False

    result: list[NativeFunctionWithDifferentiabilityInfo] = []
    for f in native_functions:
        info_dict, is_exact_match = find_info(f)

        # Currently, the '.strides()' to 'strides_or_error' replacement does not support
        # 'self' derivatives of an inplace function, so we must check for this case.
        if f.func.kind() == SchemaKind.inplace and (info_dict is not None):
            for info in info_dict.values():
                for derivative in info.derivatives:
                    if "self" in derivative.var_names:
                        for saved_input in derivative.saved_inputs:
                            if "strides_or_error" in saved_input.expr:
                                raise AssertionError(
                                    "Calling '.strides()' in the 'self' derivative formula of an "
                                    f"in-place function is not supported: {f.func}"
                                )

        if not info_dict:
            result.append(
                NativeFunctionWithDifferentiabilityInfo(
                    func=f, info=None, fw_derivatives=None
                )
            )
            continue

        fw_derivative_dict: dict[str, Sequence[ForwardDerivative]] = {}
        for key, info in info_dict.items():
            if not info.forward_derivatives:
                fw_derivative_dict[key] = []
                continue

            forward_derivatives = info.forward_derivatives

            # For functions that have a single def for out-of-place and inplace (like abs())
            if f.func.kind() == SchemaKind.inplace:
                # For inplace functions there is a little bit of work to do:
                #  1) Validate the formula and make sure the input that is modified in not used:
                #    - If there is a formula for the inplace variant of the function (is_exact_match == True) then
                #      we make sure that the original value of the input that is being modified inplace (self_p) is
                #      not used in the formula. Note that the formula can use "original_self_p" here and that would
                #      trigger a clone of the original input.
                #    - If we are reusing the out of place formula (is_exact_match == False) then we replace every
                #      occurrence of self_p and self_t by original_self_p and original_self_t. These will be
                #      populated by cloned version of the original input (either the clone done by the backward AD
                #      logic if self is also used in a backward formula or a special clone that we add).
                #  2) At this point, there cannot be a self_p in the formula.
                #  3) Change "result" into "self_p" as by design, in the inplace function codegen, the result is
                #     simply called self (as it is modified inplace).
                #  4) Update the required primals data in case it used to contain "result" but should now contain
                #     "self"
                #  5) If it is not an exact match, the user formula is not modifying the existing forward grad
                #     inplace as it should. So add some code that makes sure that we do so if the forward grad
                #     already exists.

                if len(info.forward_derivatives) != 1:
                    raise AssertionError(
                        "Only single output inplace should exist, "
                        f"got {len(info.forward_derivatives)}"
                    )
                fw_info = info.forward_derivatives[0]
                formula = fw_info.formula

                def replace_self_with_original_self(formula: str, postfix: str) -> str:
                    def repl(m: re.Match[str]) -> str:
                        return f"{m.group(1)}original_self{postfix}{m.group(2)}"

                    return re.sub(IDENT_REGEX.format(f"self{postfix}"), repl, formula)

                if re.search(IDENT_REGEX.format("self_p"), formula):
                    if is_exact_match:
                        # For manually defined formulas, don't allow the original value to be used
                        raise RuntimeError(
                            f'The formula for "{f.func.name}" is using the original value of self '
                            "that is being modified inplace. This would lead to wrong forward gradients. "
                            'Please use "result" in the formula only.'
                        )
                    else:
                        # When the original formula is out of place, we save a clone of the primal
                        # value to be able to access this value if needed
                        # replace "self_p"/"self_t" from the formula by "original_self_p"/"original_self_t"
                        formula = replace_self_with_original_self(formula, "_p")
                        formula = replace_self_with_original_self(formula, "_t")

                # replace "result" from the formula by "self_p"
                def repl(m: re.Match[str]) -> str:
                    return f"{m.group(1)}self_p{m.group(2)}"

                formula = re.sub(IDENT_REGEX.format("result"), repl, formula)

                required_primals = fw_info.required_inputs_primal
                if re.search(IDENT_REGEX.format("self_p"), formula):
                    required_primals = (
                        required_primals + ("self",) if required_primals else ("self",)
                    )

                if not is_exact_match:
                    # NOTE [In-place forward AD formula Optimization]
                    #
                    # This optimization transforms the formula to directly do inplace, i.e.
                    # instead of self_t.copy_(self_t.op()) we do self_t.op_() when the following are met:
                    #
                    # 1) the formula satisfies the pattern: "self_t.op(*args)"
                    # 2) "op" in (1) needs to be the same as the op the derivative is for
                    #
                    # (2) may seem too strict, but currently the only ops that satisfy (1) also satisfy (2)
                    # If there is a need, we can relax (2) to allow any op that has an in-place variant
                    is_single_method_on_self_t = False
                    directly_do_inplace = False
                    op_name: str | None = None
                    between_parens: str | None = None
                    match = re.fullmatch(r"self_t.([\w]*)\((.*)\)", formula)
                    if match:
                        op_name, between_parens = match.group(1), match.group(2)

                        # We want to...
                        #   Match: self_t.op1(other_p.op2(arg))
                        #   Avoid: self_t.op1(args) + self_t.op2(args)
                        #   Avoid: self_t.op1(other_p.op2(arg)) + self_t.op2(args)
                        def check_parens_nest_level_gt_zero(s: str) -> bool:
                            level = 1
                            for ch in s:
                                if ch == ")":
                                    level -= 1
                                    if level == 0:
                                        return False
                                if ch == "(":
                                    level += 1
                            return True

                        is_single_method_on_self_t = check_parens_nest_level_gt_zero(
                            between_parens
                        )
                        directly_do_inplace = (
                            is_single_method_on_self_t and op_name == info.name
                        )

                    if directly_do_inplace:
                        if op_name is None:
                            raise AssertionError("op_name must be non-None for inplace")
                        if between_parens is None:
                            raise AssertionError(
                                "between_parens must be non-None for inplace"
                            )
                        formula = f"self_t_raw.defined() ? self_t_raw.{op_name}_({between_parens}) : {formula}"
                    else:
                        # Make sure that the forward grad is modified inplace when the original formula
                        # is out of place
                        formula = f"self_t_raw.defined() ? self_t_raw.copy_({formula}) : {formula}"

                required_original_self_value = bool(
                    re.search(IDENT_REGEX.format("original_self_p"), formula)
                ) or bool(re.search(IDENT_REGEX.format("original_self_t"), formula))

                forward_derivatives = [
                    ForwardDerivative(
                        formula=formula,
                        var_names=("self",),
