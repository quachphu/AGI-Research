            return "\n".join(map(str.rstrip, content.splitlines())).rstrip()

        def replace(match: re.Match[str]) -> str:
            indent = match.group(1)
            key = match.group(2)
            comma_before = ""
            comma_after = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ", "
                    key = key[1:]
                if key[-1] == ",":
                    comma_after = ", "
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ", ".join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)

        return self.substitution.sub(replace, self.pattern)


if __name__ == "__main__":
    c = CodeTemplate(
        """\
    int foo($args) {

        $bar
            $bar
        $a+$b
    }
    int commatest(int a${,stuff})
    int notest(int a${,empty,})
    """
    )
    print(
        c.substitute(
            args=["hi", 8],
            bar=["what", 7],
            a=3,
            b=4,
            stuff=["things...", "others"],
            empty=[],
        )
    )
<|endoftext|>from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

import torchgen.api.dispatcher as dispatcher
import torchgen.dest as dest
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import native_function_manager
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, context, FileManager, NamespaceHelper, Target
from torchgen.yaml_utils import YamlLoader


if TYPE_CHECKING:
    from collections.abc import Sequence


# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping)
ParsedExternalYaml = namedtuple(
    "ParsedExternalYaml",
    ["backend_key", "autograd_key", "class_name", "cpp_namespace", "backend_indices"],
)


def parse_backend_yaml(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: dict[DispatchKey, BackendIndex],
) -> ParsedExternalYaml:
    native_functions_map: dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(
            lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
            grouped_native_functions,
        )
    }

    with open(backend_yaml_path) as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    if not isinstance(yaml_values, dict):
        raise AssertionError(
            f"Expected yaml_values to be a dict, got {type(yaml_values)}"
        )

    valid_keys = [
        "backend",
        "class_name",
        "cpp_namespace",
        "extra_headers",
        "supported",
        "autograd",
        "full_codegen",
        "non_native",
        "ir_gen",
        "symint",
    ]

    backend = yaml_values.pop("backend", None)
    if backend is None:
        raise AssertionError('You must provide a value for "backend"')

    class_name = yaml_values.pop("class_name", None)

    cpp_namespace = yaml_values.pop("cpp_namespace", None)
    if cpp_namespace is None:
        raise AssertionError('You must provide a value for "cpp_namespace"')

    # Mostly just defaulting to false to stick with LazyTensor convention.
    use_out_as_primary = yaml_values.pop("use_out_as_primary", False)
    if not isinstance(use_out_as_primary, bool):
        raise AssertionError(
            f"You must provide either True or False for use_out_as_primary. Provided: {use_out_as_primary}"
        )

    use_device_guard = yaml_values.pop("device_guard", False)
    if not isinstance(use_device_guard, bool):
        raise AssertionError(
            f"You must provide either True or False for device_guard. Provided: {use_device_guard}"
        )

    supported = yaml_values.pop("supported", [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    if not isinstance(supported, list):
        raise AssertionError(
            f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'
        )

    symint = yaml_values.pop("symint", [])
    if symint is None:
        symint = []  # Allow an empty list of symint ops
    if not isinstance(symint, list):
        raise AssertionError(
            f'expected "symint" to be a list, but got: {symint} (of type {type(symint)})'
        )
    symint_set = set(symint)

    supported_autograd = yaml_values.pop("autograd", [])
    if not isinstance(supported_autograd, list):
        raise AssertionError(
            f'expected "autograd" to be a list, but got: {supported_autograd}'
        )

    # full_codegen is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    full_codegen = yaml_values.pop("full_codegen", [])
    supported.extend(full_codegen)

    # non_native is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    yaml_values.pop("non_native", {})

    # ir_gen is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    yaml_values.pop("ir_gen", {})

    if len(yaml_values.keys()) != 0:
        raise AssertionError(
            f"{backend_yaml_path} contains unexpected keys: {', '.join(yaml_values.keys())}. "
            f"Only the following keys are supported: {', '.join(valid_keys)}"
        )

    def create_backend_index(
        backend_ops: list[str],
        symint_ops: set[str],
        dispatch_key: DispatchKey,
        *,
        use_out_as_primary: bool,
        use_device_guard: bool,
    ) -> BackendIndex:
        metadata: dict[OperatorName, BackendMetadata] = {}
        for op in backend_ops:
            op_name = OperatorName.parse(op)
            if op_name not in native_functions_map:
                raise AssertionError(f"Found an invalid operator name: {op_name}")
            # See Note [External Backends Follow Dispatcher API]
            kernel_name = dispatcher.name(native_functions_map[op_name].func)
            if op in symint_ops:
                kernel_name += "_symint"
            # TODO: allow structured external backends later.
            m = BackendMetadata(
                kernel=kernel_name, structured=False, cpp_namespace=cpp_namespace
            )
            metadata[op_name] = m
        return BackendIndex(
            dispatch_key=dispatch_key,
            use_out_as_primary=use_out_as_primary,
            external=True,
            device_guard=use_device_guard,
            index=metadata,
        )

    backend_key: DispatchKey | None = None
    if len(supported) > 0:
        with context(
            lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'
        ):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(
            supported,
            symint_set,
            backend_key,
            use_out_as_primary=use_out_as_primary,
            use_device_guard=use_device_guard,
        )
        if backend_key in backend_indices:
            raise AssertionError(f"Duplicate backend key: {backend_key}")
        backend_indices[backend_key] = backend_idx

    autograd_key: DispatchKey | None = None
    if len(supported_autograd) > 0:
        with context(
            lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'
        ):
            autograd_key = DispatchKey.parse(f"Autograd{backend}")

        autograd_idx = create_backend_index(
            supported_autograd,
            symint_set,
            autograd_key,
            use_out_as_primary=use_out_as_primary,
            use_device_guard=use_device_guard,
        )
        if autograd_key in backend_indices:
            raise AssertionError(f"Duplicate autograd key: {autograd_key}")
        backend_indices[autograd_key] = autograd_idx

    for g in grouped_native_functions:
        if isinstance(g, NativeFunction):
            forward_kernels = (
                []
                if backend_key is None
                else [
                    m
                    for m in [backend_indices[backend_key].get_kernel(g)]
                    if m is not None
                ]
            )
            backward_kernels = (
                []
                if autograd_key is None
                else [
                    m
                    for m in [backend_indices[autograd_key].get_kernel(g)]
                    if m is not None
                ]
            )
        else:
            forward_kernels = (
                []
                if backend_key is None
                else [
                    m
                    for m in [
                        backend_indices[backend_key].get_kernel(f)
                        for f in g.functions()
                    ]
                    if m is not None
                ]
            )
            backward_kernels = (
                []
                if autograd_key is None
                else [
                    m
                    for m in [
                        backend_indices[autograd_key].get_kernel(f)
                        for f in g.functions()
                    ]
                    if m is not None
                ]
            )

        forward_kernels = [f for f in forward_kernels if f is not None]
        backward_kernels = [f for f in backward_kernels if f is not None]
        if not (len(forward_kernels) == 0 or len(backward_kernels) == 0):
            raise AssertionError(
                f"Currently, all variants of an op must either be registered to a backend key, "
                f"or to a backend's autograd key. They cannot be mix and matched. "
                f"If this is something you need, feel free to create an issue! "
                f'{forward_kernels[0].kernel} is listed under "supported", '
                f'but {backward_kernels[0].kernel} is listed under "autograd".'
            )

    return ParsedExternalYaml(
        backend_key, autograd_key, class_name, cpp_namespace, backend_indices
    )


def error_on_missing_kernels(
    native_functions: Sequence[NativeFunction],
    backend_indices: dict[DispatchKey, BackendIndex],
    backend_key: DispatchKey,
    autograd_key: DispatchKey | None,
    class_name: str,
    kernel_defn_file_path: str,
    full_codegen: list[OperatorName] | None = None,
) -> None:
    try:
        with open(kernel_defn_file_path) as f:
            backend_defns = f.read()
    except OSError as e:
        raise AssertionError(
            f"Unable to read from the specified impl_path file: {kernel_defn_file_path}"
        ) from e

    if full_codegen is None:
        full_codegen = []

    indices = [backend_indices[backend_key].index] + (
        [] if autograd_key is None else [backend_indices[autograd_key].index]
    )
    # Quick mapping from each OperatorName used by the external backend
    # to its backend kernel name
    expected_backend_op_names: dict[OperatorName, str] = dict(
        list(
            concatMap(
                lambda index: [
                    (op_name, metadata.kernel) for op_name, metadata in index.items()
                ],
                indices,
            )
        )
    )
    expected_backend_native_funcs: list[NativeFunction] = [
        f
        for f in native_functions
        if f.func.name in expected_backend_op_names and f.func.name not in full_codegen
    ]
    expected_backend_kernel_name_counts: dict[str, list[NativeFunction]] = defaultdict(
        list
    )
    for native_f in expected_backend_native_funcs:
        expected_backend_kernel_name_counts[
            expected_backend_op_names[native_f.func.name]
        ].append(native_f)

    # This just looks for lines containing "foo(", and assumes that the kernel foo has been implemented.
    # It might cause false negatives (we won't catch all cases), but that's ok - if we catch a missing kernel
    # here, then we get a nicer error message. If we miss it, you get a linker error.
    kernel_defn_regex = rf"(.*){class_name}::\s*([\w\d]*)\("
    actual_backend_kernel_name_counts = Counter(
        # A bit unwieldy (this could probably be moved into regex),
        # but we don't want to include kernel names that come from function calls,
        # like "return torch_xla::XLANativeFunctions::empty_strided_symint(...)".
        # Easy check is to ignore any lines with colons before the class name.
        [
            y
            for (x, y) in re.findall(kernel_defn_regex, backend_defns)
            if not x.endswith(":")
        ]
    )

    missing_kernels_err_msg = ""
    for expected_name, funcs in expected_backend_kernel_name_counts.items():
        expected_overload_count = len(funcs)
        actual_overload_count = actual_backend_kernel_name_counts[expected_name]
        if expected_overload_count != actual_overload_count:

            def create_decl(f: NativeFunction) -> str:
                with native_function_manager(f):
                    return DispatcherSignature.from_schema(f.func).decl()

            expected_schemas_str = "\n".join([create_decl(f) for f in funcs])
            missing_kernels_err_msg += f"""
{class_name} is missing a kernel definition for {expected_name}. We found {actual_overload_count} kernel(s) with that name,
but expected {expected_overload_count} kernel(s). The expected function schemas for the missing operator are:
{expected_schemas_str}

"""
    if missing_kernels_err_msg != "":
        raise AssertionError(missing_kernels_err_msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backend stub files")
    parser.add_argument(
        "-s",
        "--source-yaml",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    parser.add_argument("-o", "--output-dir", "--output_dir", help="output directory")
    parser.add_argument(
        "--dry-run", "--dry_run", type=bool, default=False, help="output directory"
    )
    parser.add_argument(
        "--impl-path",
        "--impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions",
    )
    options = parser.parse_args()

    run(options.source_yaml, options.output_dir, options.dry_run, options.impl_path)


def gen_dispatchkey_nativefunc_headers(
    fm: FileManager,
    class_name: str,
    cpp_namespace: str,
    backend_indices: dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_dispatch_key: DispatchKey,
    autograd_dispatch_key: DispatchKey | None,
    backend_name: str = "",
) -> None:
    if class_name is None:
        raise AssertionError("class_name must not be None")
    generated_comment = (
        "Autogenerated file by gen_backend_stubs.py. Do not edit directly!"
    )

    # Convert to a set first to remove duplicate kernel names.
    # Backends are allowed to repeat kernel names; only generate the declaration once!
    # Sort for deterministic output.
    backend_declarations = sorted(
        set(
            concatMap(
                lambda f: dest.compute_native_function_declaration(
                    f, backend_indices[backend_dispatch_key]
                ),
                grouped_native_functions,
            )
        )
    )
    autograd_declarations = sorted(
        set(
            concatMap(
                lambda f: []
                if autograd_dispatch_key is None
                else dest.compute_native_function_declaration(
                    f, backend_indices[autograd_dispatch_key]
                ),
                grouped_native_functions,
            )
        )
    )

    ns_helper = NamespaceHelper(cpp_namespace)
    fm.write_with_template(
        f"{backend_dispatch_key}NativeFunctions.h",
        "DispatchKeyNativeFunctions.h",
        lambda: {
            "generated_comment": generated_comment,
            "namespace_prologue": ns_helper.prologue,
            "class_name": class_name,
            "namespace_epilogue": ns_helper.epilogue,
            "dispatch_declarations": backend_declarations + autograd_declarations,
            "BackendName": backend_name,
            "DispatchKey": backend_dispatch_key,
        },
    )


def gen_dispatcher_registrations(
    fm: FileManager,
    output_dir: str,
    class_name: str,
    backend_indices: dict[DispatchKey, BackendIndex],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_dispatch_key: DispatchKey,
    dispatch_key: DispatchKey,
    selector: SelectiveBuilder,
    # build_in_tree is true for lazy TS backend and affects include paths, not used for external backends
    build_in_tree: bool = False,
    per_operator_headers: bool = False,
    backend_name: str = "",
    eager_registration: bool = True,
) -> None:
    headers = [
        f"{output_dir}/{backend_dispatch_key}NativeFunctions.h",
    ]
    if build_in_tree:
        external_backend_headers_str = "\n".join(f"#include <{h}>" for h in headers)
    else:
        external_backend_headers_str = "\n".join(f'#include "{h}"' for h in headers)

    if class_name is None:
        raise AssertionError("class_name must not be None")
    backend_index = backend_indices[dispatch_key]

    dispatch_registrations_body = list(
        concatMap(
            dest.RegisterDispatchKey(
                backend_index,
                Target.REGISTRATION,
                selector,
                rocm=False,
                symint=True,
                class_method_name=f"{class_name}",
                skip_dispatcher_op_registration=False,
            ),
            grouped_native_functions,
        )
    )
    newline = "\n"
    ns_helper = NamespaceHelper(namespace_str="at")
    deferred_dispatch_registrations = ""
    static_init_dispatch_registrations = ""
    if eager_registration:
        static_template = CodeTemplate(
            """\
TORCH_LIBRARY_IMPL(aten, $dispatch_key, m) {
    $dispatch_registrations_body
}"""
        )
        static_init_dispatch_registrations = static_template.substitute(
            dispatch_key=dispatch_key,
            dispatch_registrations_body=dispatch_registrations_body,
        )
    else:
        deferred_template = CodeTemplate(
            """\
TORCH_API void Register${backend_name}${dispatch_key}NativeFunctions();
TORCH_API void Register${backend_name}${dispatch_key}NativeFunctions() {
    static auto m = MAKE_TORCH_LIBRARY_IMPL(aten, $dispatch_key);
    $dispatch_registrations_body
}"""
        )
        deferred_dispatch_registrations = deferred_template.substitute(
            backend_name=backend_name,
            dispatch_key=dispatch_key,
            dispatch_registrations_body=dispatch_registrations_body,
        )

    fm.write_with_template(
        f"Register{dispatch_key}.cpp",
        "RegisterDispatchKey.cpp",
        lambda: {
            "extra_cuda_headers": "",
            "external_backend_headers": external_backend_headers_str,
            "ops_headers": "#include <ATen/Functions.h>"
            if not per_operator_headers
            else "",
            "DispatchKey": dispatch_key,
            "dispatch_namespace": dispatch_key.lower(),
            "dispatch_headers": dest.gen_registration_headers(
                backend_index, per_operator_headers=per_operator_headers, rocm=False
            ),
            "dispatch_helpers": dest.gen_registration_helpers(backend_index),
            "dispatch_definitions": fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",
                lambda: {
                    "ns_prologue": ns_helper.prologue,
                    "ns_epilogue": ns_helper.epilogue,
                    "static_init_dispatch_registrations": static_init_dispatch_registrations,
                    "deferred_dispatch_registrations": deferred_dispatch_registrations,
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_definitions": "",
                    "dispatch_anonymous_definitions": list(
                        concatMap(
                            dest.RegisterDispatchKey(
                                backend_index,
                                Target.ANONYMOUS_DEFINITION,
                                selector,
                                rocm=False,
                                symint=True,
                                class_method_name=f"{class_name}",
                                skip_dispatcher_op_registration=False,
                            ),
                            grouped_native_functions,
                        )
                    ),
                },
            ).split(newline),
        },
    )


def run(
    source_yaml: str, output_dir: str, dry_run: bool, impl_path: str | None = None
) -> None:
    # Assumes that this file lives at torchgen/gen_backend_stubs.py
    root = Path(__file__).absolute().parent.parent
    common_dir = os.path.join(root, "aten/src")  # Assumes root is pytorch_root
    if not os.path.exists(common_dir):  # This file is out-of-tree.
        common_dir = os.path.join(root, "torchgen/packaged")

    template_dir = os.path.join(common_dir, "ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(common_dir, "ATen/native/native_functions.yaml")
    tags_yaml_path = os.path.join(common_dir, "ATen/native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(
        source_yaml, grouped_native_functions, backend_indices
    )
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    class_name = parsed_backend_yaml.class_name
    backend_indices = parsed_backend_yaml.backend_indices

    selector = SelectiveBuilder.get_nop_selector()

    if backend_key is None:
        # This could be useful if a backend wants to quickly set up a noop yaml file but doesn't have any kernels ready yet.
        return

    if class_name is None:
        # class_name is an optional argument to backend yaml file.
        # if specified it allows an external backend to override
        # the name of the class that all generated kernel definitions live under.
        # if not specified, its value is given as native_function_class_name.
        class_name = backend_indices[backend_key].native_function_class_name()
    if class_name is None:
        raise AssertionError("class_name must not be None")

    if impl_path is not None:
        error_on_missing_kernels(
            native_functions,
            backend_indices,
            backend_key,
            autograd_key,
            class_name,
            impl_path,
        )

    gen_dispatchkey_nativefunc_headers(
        fm,
        class_name,
        cpp_namespace,
        backend_indices,
        grouped_native_functions,
        backend_key,
        autograd_key,
    )

    for dispatch_key in (
        [backend_key] if autograd_key is None else [backend_key, autograd_key]
    ):
        gen_dispatcher_registrations(
            fm,
            output_dir,
            class_name,
            backend_indices,
            grouped_native_functions,
            backend_key,
            dispatch_key,
            selector,
        )


if __name__ == "__main__":
    main()
<|endoftext|>from __future__ import annotations

import argparse
import os
from collections import namedtuple
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

import torchgen.dest as dest
from torchgen.api.lazy import setValueT
from torchgen.api.types import BaseCppType
from torchgen.dest.lazy_ir import GenLazyIR, GenLazyNativeFuncDefinition, GenTSLazyIR
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.gen_backend_stubs import (
    error_on_missing_kernels,
    gen_dispatcher_registrations,
    gen_dispatchkey_nativefunc_headers,
    parse_backend_yaml,
)
from torchgen.model import NativeFunction, NativeFunctionsGroup, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import FileManager, NamespaceHelper
from torchgen.yaml_utils import YamlLoader


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        Lazy Tensor Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Overview
# ~~~~~~~~
#
# This codegen script builds on existing data models and helpers used
# by all ATen backends, and adds new functionality specific to lazy
# tensor backends.
#
# Inputs:
# - <backend>_native_functions.yaml: controls which operators are
#   supported by the backend.
#
# Outputs:
# (for all backends)
# <DispatchKey>Ir.h defines Lazy IR classes to be constructed during tracing
# - opt-in: also generate 'lowering' methods for the TorchScript backend only
# <DispatchKey>NativeFunctions.cpp defines implementations of native functions which perform lazy tracing
# - opt-in: 'full_codegen' section of backend yaml; 'supported' section omits these implementations
# <DispatchKey>NativeFunctions.h declares implementations of native functions for both 'supported' and 'full_codegen'
# ops
#
# Register<DispatchKey>.cpp registers all op implementations with the dispatcher
# RegisterAutograd<DispatchKey>.cpp registers all autograd implementations with the dispatcher
#
# Validation Helpers:
# - Shape Inference: errs if any ops in backend yaml require shape inference not provided by meta kernels or
#   implementations in torch/csrc/lazy/core/shape_inference.*
# - native function impls: errs if any 'supported' ops do not have an implementation defined in the backend
#   (non-codegen) implementation file
#
#
# About the Data Model
# ~~~~~~~~~~~~~~~~~~~~
#
# Modeled after ATen codegen, the first step is to parse yaml and build a data model for the operators
# we care about.  In this case, the <backend>_native_functions yaml defines a subset of the core operators
# (defined in more detail in the main native_functions.yaml), which will be supported by your backend.
# Backends can list ops in two categories:
#  - `supported` ops require hand-implementations but still get codegenned declarations and registrations
#  - `full_codegen` ops get implementations (and IR classes) generated too
#
# Each native function is modeled as an object with a schema, and each schema has objects representing their
# arguments.  Much of the codegen is manipulation of the arguments and their types.  For example, lazy tensor
# backends need to transform 'at::Tensor' arguments into 'lazy::Value' objects, as well as replacing reference
# types (stringref) with actual string objects, and this is done by manipulating the data model objects.
# - see api/lazy.py for the lazy data model
#
# Once the data model is set up, the rest of this script processes a number of templates for output CPP file
# and fills in the template values using helpers in `dest/lazy_ir.py` and `dest/lazy_ts_lowering.py`.  These
# helpers mostly iterate over functions and their arguments, outputting different c++ snippets.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping, full_codegen)
ParsedExternalYaml = namedtuple(
    "ParsedExternalYaml",
    ["backend_key", "autograd_key", "cpp_namespace", "backend_indices", "full_codegen"],
)


def parse_native_functions_keys(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
) -> tuple[list[OperatorName], list[Any], list[OperatorName]]:
    with open(backend_yaml_path) as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    if not isinstance(yaml_values, dict):
        raise AssertionError(f"Expected dict from YAML, got {type(yaml_values)}")

    full_codegen = yaml_values.pop("full_codegen", [])
    non_native = yaml_values.pop("non_native", [])
    ir_gen = yaml_values.pop("ir_gen", [])
    if not isinstance(full_codegen, list):
        raise AssertionError(
            f"Expected full_codegen to be list, got {type(full_codegen)}"
        )
    if not isinstance(non_native, list):
        raise AssertionError(f"Expected non_native to be list, got {type(non_native)}")
    if not isinstance(ir_gen, list):
        raise AssertionError(f"Expected ir_gen to be list, got {type(ir_gen)}")
    full_codegen_opnames = [OperatorName.parse(name) for name in full_codegen]
    ir_gen_opnames = [OperatorName.parse(name) for name in ir_gen]
    return full_codegen_opnames, non_native, ir_gen_opnames


def validate_shape_inference_header(
    shape_inference_hdr: str, expected_shape_infr_decls: list[str]
) -> None:
    try:
        with open(shape_inference_hdr) as f:
            shape_infr_decls = f.read()
            shape_infr_decl_lines = set(shape_infr_decls.split("\n"))
    except OSError as e:
        raise AssertionError(
            f"Unable to read from the specified shape_inference_hdr file: {shape_inference_hdr}"
        ) from e

    # TODO(whc) add a check for shape inference functions that have meta kernels implement and should be retired.

    missing_decls = [
        decl for decl in expected_shape_infr_decls if decl not in shape_infr_decl_lines
    ]
    if missing_decls:
        raise Exception(  # noqa: TRY002
            f"""Missing shape inference function.\n
Please add declare this function in {shape_inference_hdr}:\n
and implement it in the corresponding shape_inference.cpp file.\n
{os.linesep.join(missing_decls)}"""
        )


# Some helper functions for the codegen.
def get_ltc_helper_fns() -> str:
    return """\
at::Tensor to_meta(const at::Tensor& tensor) {
  // undefined tensors can't be converted to the meta device, since they don't have sizes/strides
  if (!tensor.defined()) return tensor;
  auto out = at::native::empty_strided_meta_symint(tensor.sym_sizes(), tensor.sym_strides(), \
/*dtype=*/tensor.scalar_type(), /*layout=*/tensor.layout(), \
/*device=*/c10::Device(c10::kMeta), /*pin_memory=*/std::nullopt);
  // needs to handle wrapped numbers, so dtype promotion works properly.
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }
  return out;
}
std::optional<at::Tensor> to_meta(const std::optional<at::Tensor>& tensor) {
  if (tensor.has_value()) {
    return to_meta(*tensor);
  }
  return std::nullopt;
}

std::vector<at::Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<at::Tensor> outs;
  outs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    outs.push_back(to_meta(tensor));
  }
  return outs;
}
"""


class default_args:
    node_base: str = "Node"
    node_base_hdr: str | None = None
    shape_inference_hdr: str = "torch/csrc/lazy/core/shape_inference.h"
    tensor_class: str = "torch::lazy::LazyTensor"
    tensor_class_hdr: str = "torch/csrc/lazy/core/tensor.h"
    lazy_ir_generator: type[GenLazyIR] = GenLazyIR
    native_func_definition_generator: type[GenLazyNativeFuncDefinition] = (
        GenLazyNativeFuncDefinition
    )
    backend_name: str = "TorchScript"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lazy Tensor backend files")
    parser.add_argument(
        "-s",
        "--source-yaml",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    parser.add_argument("-o", "--output-dir", "--output_dir", help="output directory")
    parser.add_argument(
        "--dry-run", "--dry_run", type=bool, default=False, help="output directory"
    )
    parser.add_argument(
        "--impl-path",
        "--impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions",
    )
    parser.add_argument(
        "--gen-ts-lowerings",
        "--gen_ts_lowerings",
        action="store_true",
        help="Generate TorchScript lowerings in addition to Lazy IR and NativeFunctions",
    )
    parser.add_argument(
        "--node-base",
        "--node_base",
        type=str,
        default=default_args.node_base,
        help="Name of backend specific custom Lazy IR Node base class",
    )
    parser.add_argument(
        "--node-base-hdr",
        "--node_base_hdr",
        type=str,
        default=default_args.node_base_hdr,
        help="Path to header file defining custom Lazy IR Node base class",
    )
    parser.add_argument(
        "--shape-inference-hdr",
        "--shape_inference_hdr",
        type=str,
        default=default_args.shape_inference_hdr,
        help="Path to header file defining custom Lazy shape inference functions",
    )
    parser.add_argument(
        "--tensor-class",
        "--tensor_class",
        type=str,
        default=default_args.tensor_class,
        help="Name of backend specific custom Lazy Tensor class",
    )
    parser.add_argument(
        "--tensor-class-hdr",
        "--tensor_class_hdr",
        type=str,
        default=default_args.tensor_class_hdr,
        help="Path to header file defining custom Lazy Tensor class",
    )
    parser.add_argument(
        "--backend-name",
        "--backend_name",
        type=str,
        default=default_args.backend_name,
        help="Name of the backend to generate",
    )
    options = parser.parse_args()

    # Assumes that this file lives at PYTORCH_ROOT/torchgen/gen_backend_stubs.py
    torch_root = Path(__file__).absolute().parents[2]
    aten_path = str(torch_root / "aten" / "src" / "ATen")
    lazy_ir_generator: type[GenLazyIR] = default_args.lazy_ir_generator
    if options.gen_ts_lowerings:
        lazy_ir_generator = GenTSLazyIR
    native_func_definition_generator: type[GenLazyNativeFuncDefinition] = (
        default_args.native_func_definition_generator
    )

    run_gen_lazy_tensor(
        aten_path,
        options.source_yaml,
        options.output_dir,
        options.dry_run,
        options.impl_path,
        options.node_base,
        options.node_base_hdr,
        options.tensor_class,
        options.tensor_class_hdr,
        options.shape_inference_hdr,
        lazy_ir_generator,
        native_func_definition_generator,
        options.backend_name,
    )


def run_gen_lazy_tensor(
    aten_path: str,
    source_yaml: str,
    output_dir: str,
    dry_run: bool,
    impl_path: str | None,
    node_base: str = default_args.node_base,
    node_base_hdr: str | None = default_args.node_base_hdr,
    tensor_class: str = default_args.tensor_class,
    tensor_class_hdr: str = default_args.tensor_class_hdr,
    shape_inference_hdr: str = default_args.shape_inference_hdr,
    lazy_ir_generator: type[GenLazyIR] = default_args.lazy_ir_generator,
    native_func_definition_generator: type[
        GenLazyNativeFuncDefinition
    ] = default_args.native_func_definition_generator,
    # build_in_tree is true for TS backend and affects include paths
    build_in_tree: bool = False,
    # per_operator_headers changes whether ATen/Functions.h or individual operator headers are used
    # it must match how ATen was built
    per_operator_headers: bool = False,
    backend_name: str = default_args.backend_name,
    gen_forced_fallback_code: bool = False,
    use_lazy_shape: bool = True,
    # the following arguments are temporary customization points for xla backend migration.
    # do not rely on them otherwise, they should be removed once migration is complete
    backend_namespace: str = "torch::lazy",
    get_tensorlist: str = "GetTensorList",
    get_tensor_or_wrap_number: str = "GetLtcTensorOrCreateForWrappedNumber",
    try_get_tensor: str = "TryGetLtcTensor",
    metrics_counter: str = 'TORCH_LAZY_FN_COUNTER("lazy::")',
    create_tensor: str = "LazyTensor::Create",
    create_from_first_tensor: bool = False,
    create_aten_from_ltc_tensor: str = "torch::lazy::CreateAtenFromLtcTensor",
    tuple_aten_from_ltc_tensors: str = "torch::lazy::TupleAtenFromLtcTensors",
    lazy_value_class: str = "torch::lazy::Value",
    lazy_tensor_ptr: str = "LazyTensorPtr",
    get_device_fn: str = "torch::lazy::GetBackendDevice",
) -> None:
    lv_tokens = lazy_value_class.split("::")
    lv_class = lv_tokens[-1]
    lv_ns = "::".join(lv_tokens[:-1])
    setValueT(BaseCppType(lv_ns, lv_class))
    template_dir = os.path.join(aten_path, "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(aten_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(aten_path, "native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    grouped_native_functions = get_grouped_native_functions(native_functions)

    def sort_native_function(f: NativeFunctionsGroup | NativeFunction) -> str:
        """
        We sort the native function because of the note in concat_map_codegen.
        TODO(alanwaketan): Remove this sorting hack once all ops are grouped properly.
        """
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        return str(func.name.name)

    grouped_native_functions = sorted(
        grouped_native_functions, key=sort_native_function
    )

    parsed_backend_yaml = parse_backend_yaml(
        source_yaml, grouped_native_functions, backend_indices
    )
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices
    # the following 3 keys are all processed differently
    # for full_codegen, we generate IR, kernels, etc
    # for ir_gen, we generate only IR
    # non_native is used to register kernels not declared in
    # native_functions.yaml
    full_codegen, non_native, ir_gen = parse_native_functions_keys(
        source_yaml, grouped_native_functions
    )

    def concat_map_codegen(
        func: Callable[[NativeFunction], Sequence[str]],
        xs: Iterable[NativeFunctionsGroup | NativeFunction],
        ops_list: list[OperatorName] = full_codegen,
    ) -> Iterator[str]:
        """
        We code-gen for the functional variant, which is all we need for IR classes/lowerings/shape inferences, but we
        only code-gen additional entries for the inplace variant for the native functions.
        """

        for x in xs:
            fs = list(x.functions()) if isinstance(x, NativeFunctionsGroup) else [x]
            for f in fs:
                if f.func.name in ops_list:
                    yield from func(f)

    selector = SelectiveBuilder.get_nop_selector()

    if backend_key is None:
        raise AssertionError("backend_key must be non-None")
    class_name = backend_indices[backend_key].native_function_class_name()

    if impl_path is not None:
        error_on_missing_kernels(
            native_functions,
            backend_indices,
            backend_key,
            autograd_key,
            class_name,
            impl_path,
            full_codegen,
        )

    """ Validate Shape Inference Definitions

    Generated lazy native functions all perform shape inference, by first using a meta:: kernel
    if available for that op, and otherwise using a 'compute_shape_{op}' function instead.  The generator
    knows the call signature for compute_shape_{op} because it matches the nativefunction (and meta::) signature,
    so it just has to check whether the op is structured and generate a call for one or the other.  It's up to the dev
    to supply the missing compute_shape_{op} function, but the codegen at least warns you about this and provides
    the expected signature which can be copy-pasted into shape_inference.h.

    compute_shape_{op} functions are handwritten and should be replaced over time as ops get ported
    to structured kernels.

    See torch/csrc/lazy/core/shape_inference.cpp #READ THIS! for more information.
    """
    if shape_inference_hdr is not None:
        expected_shape_infr_decls = list(
            concat_map_codegen(
                dest.GenLazyShapeInferenceDefinition(
                    backend_indices[backend_key], tensor_class
                ),
                grouped_native_functions,
            )
        )

        validate_shape_inference_header(shape_inference_hdr, expected_shape_infr_decls)
    if class_name is None:
        raise AssertionError("class_name must be non-None")

    # Generate nativefunction declarations
    # Note, eager registrations is set to False for the lazy TS backend as another LTC backend
    # may want to register their own lazy kernels instead of registering the TS ones.
    # The registration will lazily happen when init_ts_backend is called.
    gen_dispatchkey_nativefunc_headers(
        fm,
        class_name,
        cpp_namespace,
        backend_indices,
        grouped_native_functions,
        backend_key,
        autograd_key,
        backend_name,
    )

    # Generate Dispatcher registrations which hook up the nativefunctions
    for dispatch_key in (
        [backend_key] if autograd_key is None else [backend_key, autograd_key]
    ):
        gen_dispatcher_registrations(
            fm,
            output_dir,
            class_name,
            backend_indices,
            grouped_native_functions,
            backend_key,
            dispatch_key,
            selector,
            build_in_tree=build_in_tree,
            per_operator_headers=per_operator_headers,
            backend_name=backend_name,
            eager_registration=False,
        )

    # Generate native function impls that build IR nodes
    ns_helper = NamespaceHelper(cpp_namespace)
    fm.write_with_template(
        f"{backend_key}NativeFunctions.cpp",
        "DispatchKeyNativeFunctions.cpp",
        lambda: {
            "includes": [
                f"#include <{path}>"
                for path in [
                    tensor_class_hdr,
                    shape_inference_hdr,
                    "ATen/Functions.h",
                    "ATen/native/TensorConversions.h",
                    "ATen/NativeFunctions.h",
                    "ATen/CompositeExplicitAutogradNonFunctionalFunctions.h",
                    "ATen/MetaFunctions.h",
                    "ATen/Operators.h",
                    "ATen/native/CPUFallback.h",
                    "torch/csrc/lazy/core/ir_builder.h",
                    "torch/csrc/lazy/core/lazy_graph_executor.h",
                    "torch/csrc/lazy/core/metrics.h",
                    "torch/csrc/lazy/core/shape.h",
                    f"{output_dir}/{backend_key}NativeFunctions.h",
                    f"{output_dir}/LazyIr.h",
                ]
                + (
                    ["torch/csrc/lazy/ts_backend/ts_eager_fallback.h"]
                    if gen_forced_fallback_code
                    else []
                )
            ],
            "helper_fns": get_ltc_helper_fns(),
            "native_functions_include": "",
            "namespace_prologue": ns_helper.prologue,
            "namespace_epilogue": ns_helper.epilogue,
            "native_function_definitions": list(
                concat_map_codegen(
                    native_func_definition_generator(
                        f"{backend_key}NativeFunctions",
                        backend_indices[backend_key],
                        tensor_class,
                        gen_forced_fallback_code,
                        backend_namespace,
                        get_tensorlist,
                        get_tensor_or_wrap_number,
                        try_get_tensor,
                        metrics_counter,
                        create_tensor,
                        create_from_first_tensor,
                        create_aten_from_ltc_tensor,
                        tuple_aten_from_ltc_tensors,
                        lazy_tensor_ptr,
                        get_device_fn,
                    ),
                    grouped_native_functions,
                )
            ),
        },
    )
    # Generate IR node classes
    lazy_ir_obj = lazy_ir_generator(
        backend_indices[backend_key], backend_name, node_base, use_lazy_shape
    )

    fm.write_with_template(
        "LazyIr.h",
        "LazyIr.h",
        lambda: {
            "lazy_ir_sysinc": [
                f"#include <{path}>"
                for path in [
                    "ATen/core/Formatting.h",
                    "c10/core/ScalarType.h",
                    "torch/csrc/lazy/core/hash.h",
                    "torch/csrc/lazy/core/ir.h",
                    "torch/csrc/lazy/core/shape.h",
                    "optional",
                    "vector",
                ]
            ],
            "lazy_ir_inc": [f'#include "{node_base_hdr}"']
            if node_base_hdr is not None
            else [],
            "ir_declarations": list(
                concat_map_codegen(
                    lazy_ir_obj, grouped_native_functions, full_codegen + ir_gen
                )
            ),
            "namespace_prologue": ns_helper.prologue,
            "namespace_epilogue": ns_helper.epilogue,
        },
    )

    # Generate Non Native IR Node classes
    fm.write_with_template(
        "LazyNonNativeIr.h",
        "LazyNonNativeIr.h",
        lambda: {
            "lazy_non_native_ir_inc": [
                f"#include <{path}>"
                for path in [
                    "torch/csrc/lazy/core/ir.h",
                    "torch/csrc/lazy/core/ir_builder.h",
                    "torch/csrc/lazy/core/internal_ops/ltc_ops.h",
                    "torch/csrc/lazy/core/shape_inference.h",
                ]
                + ([node_base_hdr] if node_base_hdr else [])
                if path
            ],
            "non_native_ir_nodes": dest.generate_non_native_lazy_ir_nodes(
                non_native, lazy_ir_obj
            ),
            "namespace_prologue": ns_helper.prologue,
            "namespace_epilogue": ns_helper.epilogue,
        },
    )


if __name__ == "__main__":
    main()
<|endoftext|>from __future__ import annotations

import string
from collections import defaultdict
from typing import TYPE_CHECKING

import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
    Annotation,
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseOperatorName,
    BaseTy,
    BaseType,
    DEFAULT_KERNEL_NAMESPACE,
    DeviceCheckType,
    DispatchKey,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    Return,
    SchemaKind,
    Variant,
)
from torchgen.utils import concatMap


if TYPE_CHECKING:
    from collections.abc import Sequence


# See Note: [Out ops with functional variants that don't get grouped properly]
OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY = [
    # This has a functional variant, but it's currently marked private.
    # This function should be marked private as well (*_backward ops aren't exposed to python anyway).
    "adaptive_avg_pool3d_backward.grad_input",
    # There's a functional variant, _slow_conv2d_backward.output_mask, that isn't grouped properly.
    # Maybe we can kill this operator in favor of convolution_backward?
    "_slow_conv2d_backward.grad_input",
]


# See Note: [Mutable ops that cannot get an out variant]
MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT = [
    # should be out=?
    "_cummax_helper",
    # should be out=?
    "_cummin_helper",
]

# All of these operators don't have any tensor like returns
FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT = [
    "_assert_async",  # no return
    "_assert_async.msg",  # no return
    "_assert_tensor_metadata",  # no return
    "_cslt_sparse_mm_search",  # returns an int
    "_assert_scalar",  # no return
    "_dimI",  # returns an int
    "_dimV",  # returns an int
    "_has_same_storage_numel",  # returns a boolean
    "_linalg_check_errors",  # no return
    "_local_scalar_dense",  # returns a Scalar
    "_nested_tensor_from_mask_left_aligned",  # returns a boolean
    "_nnz",  # returns an int
    "_use_cudnn_ctc_loss",  # returns a boolean
    "_use_cudnn_ctc_loss.Tensor",  # returns a boolean
    "_use_miopen_ctc_loss",  # returns a boolean
    "_use_miopen_ctc_loss.Tensor",  # returns a boolean
    "_validate_compressed_sparse_indices",  # no return
    "allclose",  # returns a boolean
    "dense_dim",  # returns an int
    "equal",  # returns a boolean
    "is_coalesced",  # returns an boolean
    "is_pinned",  # returns a boolean
    "is_same_size",  # returns a boolean
    "is_set_to",  # returns a boolean
    "q_per_channel_axis",  # returns an int
    "q_scale",  # returns a float
    "q_zero_point",  # returns an int
    "qscheme",  # returns a QScheme
    "record_stream",  # no return
    "sparse_dim",  # returns an int
    "sym_constrain_range",  # no return
    "sym_constrain_range_for_size",  # no return
    "_nested_tensor_storage_offsets",  # returns a vector of ints
    "_chunk_grad_outputs_efficient_attention",  # returns a bool
    "_fused_sdp_choice",  # returns an int
    "_print",  # no return
    "_sink_tokens",  # no return
    "_nested_get_ragged_idx",  # returns an int
]

INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY = [
    # polygamma and polygamma.out both exist, but have a
    # pre-self arg (while polygamma_ does not)
    # We should either fix this schema so it can be grouped properly,
    # or allow the codegen to generate new functional/out= NativeFunctions for this op
    # (which would require changing its overload name to prevent overload ambiguity).
    "polygamma_"
]


# Groups "similar" NativeFunctions together
# example add.Tensor, add_.Tensor, add.out
# "similar" NativeFunctions are all expected to have an identical `signature()`,
# But have differing SchemaKinds.
def pre_group_native_functions(
    native_functions: Sequence[NativeFunction],
) -> dict[FunctionSchema, dict[SchemaKind, NativeFunction]]:
    pre_grouped_native_functions: dict[
        FunctionSchema, dict[SchemaKind, NativeFunction]
    ] = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        if f.func.kind() in d:
            raise AssertionError(f"Duplicate schema kind {f.func.kind()} for {f.func}")
        d[f.func.kind()] = f
    return pre_grouped_native_functions


# Returns the out variant overload name given a base function overload name
def get_expected_out_variant_overload_name(overload_name: str | None) -> str:
    return "out" if not overload_name else f"{overload_name}_out"


# Helper function: given an inplace FunctionSchema, generate its corresponding out= variant
# Example before:
#   _add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
# Example after:
#   _add_relu.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out)
def self_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # Generating an out= schema from an inplace schema.
    if func.kind() != SchemaKind.inplace:
        raise AssertionError(f"Expected inplace schema, got {func.kind()}")
    if func.arguments.self_arg is None:
        raise AssertionError("Expected self_arg to be non-None")
    # The new out= schema has:
    # - a new out argument with the same type as "func" (but with a mutable annotation)
    # - The returns (if any) now alias the out= argument instead of "func"
    # - an "out" overload name
    return FunctionSchema(
        name=func.name.remove_inplace().with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.remove_self_annotation().with_out_args(
            [
                Argument(
                    name="out",
                    type=func.arguments.self_arg.argument.type,
                    default=None,
                    annotation=func.arguments.self_arg.argument.annotation,
                )
            ]
        ),
        returns=func.returns,
    )


# Helper function: given a functional FunctionSchema, generate its corresponding out= variant
# Example before:
#   _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#       bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
# Example after:
#   _to_copy._out(Tensor self, *, bool non_blocking=False, MemoryFormat? memory_format=None,
#       Tensor(a!) out) -> Tensor(a!)
def functional_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # Generating an out= schema from a functional schema.
    if func.kind() != SchemaKind.functional:
        raise AssertionError(f"Expected functional schema, got {func.kind()}")

    new_returns, new_out_args = generate_out_args_from_schema(func)
    # The new out= schema has:
    # - one or more new out argument(s) with the same type as returns (but with a mutable annotation)
    # - The returns now alias the out= arguments
    # - an "_out" overload name
    return FunctionSchema(
        name=func.name.with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.signature().with_out_args(
            new_out_args,
        ),
        returns=tuple(new_returns),
    )


# Helper function: given a function schema, generate corresponding out arguments, also the updated return annotations.
def generate_out_args_from_schema(
    func: FunctionSchema,
) -> tuple[list[Return], list[Argument]]:
    # More of a sanity check - our existing restrictions on schemas should enforce that
    # mutable schema kinds never return their mutable arguments.
    if any(r.annotation is not None and r.annotation.is_write for r in func.returns):
        raise AssertionError("Mutable schema kinds should not return mutable arguments")

    tensorlike_rets = [r for r in func.returns if r.type.is_tensor_like()]
    if len(tensorlike_rets) == 0:
        raise AssertionError("Expected at least one tensor-like return")

    used_annotations = concatMap(
        lambda a: [] if a.annotation is None else a.annotation.alias_set,
        func.arguments.flat_all,
    )
    valid_annotations = [x for x in string.ascii_lowercase if x not in used_annotations]

    all_rets_are_tensors = all(r.type == BaseType(BaseTy.Tensor) for r in func.returns)

    new_out_args: list[Argument] = []
    # The end result of new_returns is that:
    # - If every return is a plain tensor, then the new returns == the old returns, but with the out= alias annotations added.
    # - Otherwise, none of the out arguments show up in the returns (and we're only left with non-tensor-like returns, if any).
    new_returns: list[Return] = []
    for i, r in enumerate(func.returns):
        if r.type.is_tensor_like():
            new_out = Argument(
                name="out" if len(func.returns) == 1 else f"out{i}",
                type=r.type,
                default=None,
                annotation=Annotation.parse(f"{valid_annotations[i]}!"),
            )
            new_out_args.append(new_out)
            if all_rets_are_tensors:
                # The convention for out= schemas is that they only return their out arguments
                # if the return is a plain Tensor (or if it's a tuple of plain Tensors)
                new_ret = Return(
                    name=None, type=new_out.type, annotation=new_out.annotation
                )
                new_returns.append(new_ret)
        else:
            new_returns.append(r)
    return new_returns, new_out_args


# Helper function: given a mutable FunctionSchema, generate its corresponding out= variant
# Example before:
#   _fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)
# Example after:
#   _fused_moving_avg_obs_fq_helper._out(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False, *, Tensor(e!) out0, Tensor(f!) out1) -> (Tensor(e!), Tensor(f!))
def mutable_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    # Generating an out= schema from a mutable schema.
    if func.kind() != SchemaKind.mutable:
        raise AssertionError(f"Expected mutable schema, got {func.kind()}")
    # The new out= schema has:
    # - Any non-aliased tensor-like returns are converted to mutable, aliased out= arguments
    #   (if the argument is a tensor then we also return it for method chaining,
    #   otherwise we return nothing)
    # - an "out" overload name
    #
    # Note that:
    # (1) This also means that we can *only* generate an out= variant from a mutable schema
    #     if the mutable schema has at least one tensor-like non-aliasing return.
    # (2) The generated out= variant still has mutable positional arguments,
    #     but if necessary we could probably add another out= variant that also
    #     functionalizes the mutable arguments (a functional_out variant)

    new_returns, new_out_args = generate_out_args_from_schema(func)

    return FunctionSchema(
        name=func.name.remove_inplace().with_overload(
            get_expected_out_variant_overload_name(func.name.overload_name)
        ),
        arguments=func.arguments.with_out_args(new_out_args),
        returns=tuple(new_returns),
    )


# This function, given function of one SchemaKind, as well as a target SchemaKind,
# generates a new NativeFunction with the same properties, but using the target SchemaKind.
# We only actually generate functions for either functional or out= SchemaKinds.
# This function returns a tuple, with:
# - The generated NativeFunction
# - a dictionary of `BackendIndex` objects, describing which dispatch keys
#   we will generate kernels for, for the new NativeFunction.
#   Details are in the function, but we only generate composite kernels (in some cases) today.
def generate_function(
    f: NativeFunction, k: SchemaKind
) -> tuple[NativeFunction, dict[DispatchKey, dict[OperatorName, BackendMetadata]]]:
    from torchgen.api import cpp

    if k == SchemaKind.functional:
        if f.func.kind() == SchemaKind.functional:
            raise AssertionError("Cannot generate functional from functional schema")
        # The new "functional" NativeFunction has:
        # - any mutable arguments have been converted into (immutable) returns.
        #   (if a mutable argument was not also a return, it gets converted to one)
        # - "_functional" appended to the base name, ONLY IF this op has a mutable variant.
        #   See Note [Overload Ambiguity With Functional Variants]
        # The default grouping logic in signature() actually already does this,
        # so we can piggy-back off it (but we still want return names)
        func = f.func.signature(keep_return_names=True).with_name(
            OperatorName(
                name=BaseOperatorName(
                    base=f.func.name.name.base,
                    inplace=False,
                    dunder_method=f.func.name.name.dunder_method,
                    # See Note [Overload Ambiguity With Functional Variants]
                    functional_overload=f.func.kind() == SchemaKind.mutable,
                ),
                overload_name=f.func.name.overload_name,
            )
        )
    elif k == SchemaKind.out:
        # We generate out= ops mostly just so that we can pair up NativeFunctions into groups easily,
        # but at least today, there is no good reason to actually use them.
        # we'll generate a dispatcher entry for them, but won't actually register any kernels for them.
        if f.func.kind() == SchemaKind.inplace:
            func = self_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.mutable:
            func = mutable_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.functional:
            func = functional_to_out_signature(f.func)
        else:
            raise AssertionError(
                "We only bother generating out= functions from either inplace or mutable or functional variants"
            )
    else:
        raise AssertionError(
            "We currently only generate either functional or out= NativeFunctions"
        )

    # Generated kernel naming convention for out: <op_name>_<overload_name>. The reason for this is to
    # disambiguate operator with the same name but different overload name, e.g., `randn.names_out` and
    # `randn.generator_with_names_out`.
    kernel_name = (
        func.name.unambiguous_name()
        if func.kind() == SchemaKind.out
        else cpp.name(func)
    )
    if f.func.has_symint():
        kernel_name += "_symint"
    backend_metadata = {
        DispatchKey.CompositeExplicitAutograd: {
            func.name: BackendMetadata(
                kernel=kernel_name,
                structured=False,
                cpp_namespace=DEFAULT_KERNEL_NAMESPACE,
            )
        }
    }
    tags = {"generated"} | set(
        f.tags & {"nondeterministic_seeded", "view_copy", "pt2_compliant_tag"}
    )
    if func.is_out_fn():
        tags.add("out")
    if func.name.name.inplace:
        tags.add("inplace")

    return (
        NativeFunction(
            func=func,
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            # These generated fn's aren't meant to be user friendly- don't generate methods.
            variants={Variant.function},
            structured=False,
            structured_delegate=None,
            structured_inherits=None,
            precomputed=None,
            autogen=[],
            ufunc_inner_loop={},
            manual_kernel_registration=False,
            manual_cpp_binding=False,
            python_module=None,
            category_override=None,
            device_guard=False,
            device_check=DeviceCheckType.NoCheck,
            loc=f.loc,
            cpp_no_default_args=set(),
            is_abstract=f.is_abstract,
            has_composite_implicit_autograd_kernel=False,
            has_composite_implicit_autograd_nested_tensor_kernel=False,
            has_composite_explicit_autograd_kernel=True,
            has_composite_explicit_autograd_non_functional_kernel=False,
            # Every generated NativeFunction gets a "generated" tag, so it's easy to tell
            # which NativeFunction objects did not come directly from native_functions.yaml.
            tags=tags,
            namespace=f.namespace,
        ),
        backend_metadata,
    )


# This function is responsible for adding generated NativeFunctions which don't appear
# explicitly in the codegen.
# You can inspect the full list of NativeFunctions yourself with the torchgen package, by running
# torchgen.parse_native_yaml("aten/src/ATen/native/native_functions.yaml", "aten/src/ATen/native/tags.yaml")
# (Maybe we should make a friendly API for this)
#
# Note: this function *mutates* its two inputs,
# adding the new NativeFunctions / BackendMetadata to them
def add_generated_native_functions(
    rs: list[NativeFunction],
    indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]],
) -> None:
    # The main code for generating new NativeFunctions
    # First we group of NativeFunctions by schema kind,
    # then we detect which ones are missing and generate them.
    pre_grouped_native_functions = pre_group_native_functions(rs)
    for d in pre_grouped_native_functions.values():
        has_functional = SchemaKind.functional in d
        has_inplace = SchemaKind.inplace in d
        has_mutable = SchemaKind.mutable in d
        has_out = SchemaKind.out in d
        is_core = any("core" in variant.tags for variant in d.values())

        # We automatically generate a few native functions that don't exist in the yaml, for a few reasons:
        # (1) If an operator has an inplace/out= variant but no functional variant, we can generate
        #     a simple functional variant that the functionalization pass can consume.
        # (2) If an operator has an inplace or functional but no out= variant, we generate an out=
        #     variant, mostly so we can easily pair up functions into NativeFunctionsGroup,
        #     while maintaining the constraint that the out= variant is "required".
        if has_mutable or has_inplace or has_out or has_functional:
            # Don't bother generating functions trio's for native functions that bypass the dispatcher.
            are_manual = all(f.manual_cpp_binding for f in d.values())
            # Don't bother generating functional + out= variants for view operators
            # set_ is technically an inplace_view, but for now it is treated
            # as a normal inplace op in the codegen
            has_view_ops = any(
                f.is_view_op and str(f.func.name.name) != "set_" for f in d.values()
            )
            # Don't generate the other variants for non-core CompositeImplicitAutograd operators.
            # We could probably do this, but the main benefit of generating the function triplets
            # is for transforms that need them, and transforms don't need to act directly
            # on CompositeImplicitAutograd operators (since we let them decompose).
            are_composite_implicit = all(
                f.has_composite_implicit_autograd_kernel for f in d.values()
            )
            if are_manual or has_view_ops or are_composite_implicit and not is_core:
                continue
            if has_out and len(d.values()) == 1:
                # Note: [Out ops with functional variants that don't get grouped properly]
                # In theory we could validly have an out= operator in native_functions.yaml
                # that has no other variants.
                # But today, all of the operators where that's the case actually do have
                # functional variants, that we are just unable to pair up properly.
                # I think banning this all together is probably safer
                # (you can always add a functional variant yourself if you want to add a new out= operator).
                #
                # We should probably fix the existing cases; this check is to prevent us from adding more over time.
                if (
                    str(d[SchemaKind.out].func.name)
                    not in OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY
                ):
                    raise AssertionError(
                        f"Found an out= operator that we could not find any other variants of: {str(d[SchemaKind.out].func)}"
                    )
                continue

            # Some inplace ops that have problematic schemas (that we should fix), which prevent us
            # from generating out= and functional variants
            if (
                has_inplace
                and str(d[SchemaKind.inplace].func.name)
                in INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY
            ):
                continue

            base_fn = (
                d[SchemaKind.mutable]
                if has_mutable
                else d[SchemaKind.inplace]
                if has_inplace
                else d[SchemaKind.out]
                if has_out
                else d[SchemaKind.functional]
            )

            # Note: [Mutable ops that cannot get an out variant]
            # We can only generate an out= variant if either:
            # - the original function has tensor-like returns (since we can convert them to out kwargs)
            # - or it's inplace (since we can convert `self` to an out kwarg)
            # There are only two functions that don't fit this criteria today though,
            # and they both look like they should be fixed to be out= variants,
            # so if feels safer to ban this schema all-together
            base_fn_valid = base_fn.func.kind() == SchemaKind.inplace or any(
                r.type.is_tensor_like() for r in base_fn.func.returns
            )
            # Note: [Loosen the assertion that all functional should have out variant]
            # By design all functional operators should have our variants. The needs_out check
            # is loosening this requirement, changing it to only generate out variant if there's
            # an `autogen` block in the native function, in the long run it should be removed.
            # FIXME: Remove this after figuring out CI job failures related to min, max, mean
            needs_out = any("out" in str(op_name) for op_name in base_fn.autogen)
            gets_out_variant = not has_out and base_fn_valid and needs_out
            if not has_out and not base_fn_valid:
                if (
                    str(base_fn.func.name)
                    not in MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
                    and str(base_fn.func.name)
                    not in FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
                ):
                    raise AssertionError(
                        f"""Found an operator that we could not generate an out= variant for: {str(base_fn.func)}.
This type of operators don't have tensor-like return, making it difficult to generate a proper out= variant. If
out= variant is not needed, please add the function name into FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT list."""
                    )

            # Generate an out= variant
            if gets_out_variant:
                fn, metadata = generate_function(base_fn, SchemaKind.out)
                d[SchemaKind.out] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)

            # Generate a functional variant, but only do it if the operator got an out= variant
            # (Functional variants are only useful if we can group up the variants,
            # which we can only do if they have an out= variant)
            if not has_functional and (has_out or gets_out_variant):
                fn, metadata = generate_function(base_fn, SchemaKind.functional)
                d[SchemaKind.functional] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)


def return_str(rets: tuple[Return, ...], names: list[str]) -> str:
    if len(rets) != len(names):
        raise AssertionError(
            f"Returns and names length mismatch: {len(rets)} vs {len(names)}"
        )
    if len(rets) == 0:
        return ""
    elif len(rets) == 1:
        return f"return {names[0]};"
    else:
        return f"return {dispatcher.returns_type(rets).cpp_type()}({', '.join(names)});"


# Given a function, and the name of a variable corresponding to the output of that function,
# gather up all of the individual returns that are not aliased
def gather_nonaliased_inner_rets(func: FunctionSchema, out_var: str) -> list[str]:
    aliased_rets = func.aliased_return_names()
    non_aliased_names = []
    is_out_var_a_tuple = len(func.returns) > 1
    for i, r in enumerate(aliased_rets):
        if r is None:
            non_aliased_names.append(
                f"std::get<{i}>({out_var})" if is_out_var_a_tuple else out_var
            )
    return non_aliased_names


# Generates functional kernels in terms of their inplace.mutable counterparts.
# We only do this for "generated" NativeFunctions
@with_native_function
def gen_composite_functional_kernel(g: NativeFunctionsGroup) -> str | None:
    # We should only be generating these for code-generated NativeFunctions
    if "generated" not in g.functional.tags:
        return None
    # And we always write the kernel for a generated op in terms of a non-generated op.
    if g.inplace is not None and "generated" not in g.inplace.tags:
        target_f = g.inplace
    elif g.mutable is not None and "generated" not in g.mutable.tags:
        target_f = g.mutable
    else:
        # We should be guaranteed to have a valid inplace/mutable variant to call into.
        # See Note: [Mutable Ops Not Using Functionalization]
        raise AssertionError(str(g.functional.func))

    sig = DispatcherSignature(g.functional.func)
    target_sig = DispatcherSignature(target_f.func)

    context: list[Binding | Expr] = []
    clone_mutable_inputs = []
    cloned_return_names = []
    # We can't just directly pass all of the arguments from the functional op into the mutating op.
    # We need to check for which inputs to the mutating operator are mutable,
    # and clone those inputs first.
    for a_curr, a_tgt in zip(
        dispatcher.jit_arguments(g.functional.func),
        dispatcher.jit_arguments(target_f.func),
    ):
        if a_tgt.annotation is not None and a_tgt.annotation.is_write:
            clone_mutable_inputs.append(
                f"auto {a_curr.name}_clone = clone_arg({a_curr.name});"
            )
            context.append(
                Expr(
                    expr=f"{a_curr.name}_clone",
                    type=dispatcher.argument_type(a_curr, binds=a_curr.name),
                )
            )
            # Invariant: mutable arguments on the inner mutable op are always returns on the functional op.
            cloned_return_names.append(f"{a_curr.name}_clone")
        else:
            context.append(dispatcher.argument(a_curr))
    exprs = ", ".join([e.expr for e in translate(context, target_sig.arguments())])

    out_name = "output"
    maybe_assign = f"auto {out_name} = " if len(target_f.func.returns) > 0 else ""
    inner_return_names = gather_nonaliased_inner_rets(target_f.func, out_name)
    ret_str = return_str(
        g.functional.func.returns, inner_return_names + cloned_return_names
    )

    clone_mutable_inputs_str = "\n".join(clone_mutable_inputs)
    return f"""
{sig.defn(name=sig.name() + ("_symint" if g.out.func.has_symint() else ""))} {{
  {clone_mutable_inputs_str}
  {maybe_assign}at::_ops::{target_f.func.name.unambiguous_name()}::call({exprs});
  {ret_str}
}}
"""


# Generates out= kernels in terms of their functional counterparts.
# We only do this for "generated" NativeFunctions
@with_native_function
def gen_composite_out_kernel(g: NativeFunctionsGroup) -> str | None:
    # We should only be generating these for code-generated NativeFunctions
    if "generated" not in g.out.tags:
        return None
    # And we always write the kernel for the out= op in terms of the functional.
    # Note that the functional op might have also been generated, but we don't have to
    # worry about cycles, because the generated functional kernels are always implemented
    # in terms of non-generated kernels (see gen_composite_functional_kernel).

    sig = DispatcherSignature(g.out.func)
    target_sig = DispatcherSignature(g.functional.func)

    exprs = ", ".join(
        [e.expr for e in translate(sig.arguments(), target_sig.arguments())]
    )

    copy_outs = []
    out_name = "tmp_output"
    for i, out_arg in enumerate(g.out.func.arguments.out):
        functional_return_name = (
            out_name
            if len(g.functional.func.returns) == 1
            else f"std::get<{i}>({out_name})"
        )
        copy_outs.append(
            f"""\
  resize_out_helper({out_arg.name}, {functional_return_name});
  copy_arg({out_arg.name}, {functional_return_name});"""
        )

    rets = []
    # For each return arg in the calling (out=) operator,
    # If it corresponds to an aliased input, return the input.
    # Otherwise, return the corresponding output from calling the functional operator.
    for i, ret_name in enumerate(g.out.func.aliased_return_names()):
        if ret_name is not None:
            rets.append(ret_name)
        else:
            functional_return_name = (
                out_name
                if len(g.functional.func.returns) == 1
                else f"std::get<{i}>({out_name})"
            )
            rets.append(functional_return_name)

    copy_outs_str = "\n".join(copy_outs)

    # Kernel name needs to follow the naming convention defined in `generate_function()`
    return f"""
{sig.defn(name=g.out.func.name.unambiguous_name() + ("_symint" if g.out.func.has_symint() else ""))} {{
  auto {out_name} = at::_ops::{g.functional.func.name.unambiguous_name()}::call({exprs});
  {copy_outs_str}
  {return_str(g.out.func.returns, rets)}
}}
"""
<|endoftext|>from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from dataclasses import is_dataclass
from enum import auto, Enum
from pathlib import Path
from pprint import pformat
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never, Self

from torchgen.code_template import CodeTemplate


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Iterable, Iterator, Sequence


TORCHGEN_ROOT = Path(__file__).absolute().parent
REPO_ROOT = TORCHGEN_ROOT.parent


# Many of these functions share logic for defining both the definition
# and declaration (for example, the function signature is the same), so
# we organize them into one function that takes a Target to say which
# code we want.
#
# This is an OPEN enum (we may add more cases to it in the future), so be sure
# to explicitly specify with Literal[Target.XXX] or Literal[Target.XXX, Target.YYY]
# what targets are valid for your use.
class Target(Enum):
    # top level namespace (not including at)
    DEFINITION = auto()
    DECLARATION = auto()
    # TORCH_LIBRARY(...) { ... }
    REGISTRATION = auto()
    # namespace { ... }
    ANONYMOUS_DEFINITION = auto()
    # namespace cpu { ... }
    NAMESPACED_DEFINITION = auto()
    NAMESPACED_DECLARATION = auto()


# Matches "foo" in "foo, bar" but not "foobar". Used to search for the
# occurrence of a parameter in the derivative formula
IDENT_REGEX = r"(^|\W){}($|\W)"


# TODO: Use a real parser here; this will get bamboozled
def split_name_params(schema: str) -> tuple[str, list[str]]:
    m = re.match(r"(\w+)(\.\w+)?\((.*)\)", schema)
    if m is None:
        raise RuntimeError(f"Unsupported function schema: {schema}")
    name, _, params = m.groups()
    return name, params.split(", ")


T = TypeVar("T")
S = TypeVar("S")

# These two functions purposely return generators in analogy to map()
# so that you don't mix up when you need to list() them


# Map over function that may return None; omit Nones from output sequence
def mapMaybe(func: Callable[[T], S | None], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        r = func(x)
        if r is not None:
            yield r


# Map over function that returns sequences and cat them all together
def concatMap(func: Callable[[T], Sequence[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        yield from func(x)


# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg_fn: Callable[[], str]) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        # TODO: this does the wrong thing with KeyError
        msg = msg_fn()
        msg = textwrap.indent(msg, "  ")
        msg = f"{e.args[0]}\n{msg}" if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise


@functools.cache
def _read_template(template_fn: str) -> CodeTemplate:
    return CodeTemplate.from_file(template_fn)


# String hash that's stable across different executions, unlike builtin hash
def string_stable_hash(s: str) -> int:
    sha1 = hashlib.sha1(s.encode("latin1"), usedforsecurity=False).digest()
    return int.from_bytes(sha1, byteorder="little")


# A small abstraction for writing out generated files and keeping track
# of what files have been written (so you can write out a list of output
# files)
class FileManager:
    def __init__(
        self,
        install_dir: str | Path,
        template_dir: str | Path,
        dry_run: bool,
    ) -> None:
        self.install_dir = Path(install_dir)
        self.template_dir = Path(template_dir)
        self.files: set[Path] = set()
        self.dry_run = dry_run

    @property
    def filenames(self) -> frozenset[str]:
        return frozenset({file.as_posix() for file in self.files})

    def _write_if_changed(self, filename: str | Path, contents: str) -> None:
        file = Path(filename)
        old_contents: str | None = None
        try:
            old_contents = file.read_text(encoding="utf-8")
        except OSError:
            pass
        if contents != old_contents:
            # Create output directory if it doesn't exist
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(contents, encoding="utf-8")

    # Read from template file and replace pattern with callable (type could be dict or str).
    def substitute_with_template(
        self,
        template_fn: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> str:
        if Path(template_fn).is_absolute():
            raise AssertionError(f"template_fn must be relative: {template_fn}")
        template_path = self.template_dir / template_fn
        env = env_callable()
        if isinstance(env, dict):
            if "generated_comment" not in env:
                generator_default = TORCHGEN_ROOT / "gen.py"
                try:
                    generator = Path(
                        sys.modules["__main__"].__file__ or generator_default
                    ).absolute()
                except (KeyError, AttributeError):
                    generator = generator_default.absolute()

                try:
                    generator_path = generator.relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    generator_path = generator.name

                env = {
                    **env,  # copy the original dict instead of mutating it
                    "generated_comment": (
                        "@" + f"generated by {generator_path} from {template_fn}"
                    ),
                }
            template = _read_template(template_path)
            substitute_out = template.substitute(env)
            # Ensure an extra blank line between the class/function definition
            # and the docstring of the previous class/function definition.
            # NB: It is generally not recommended to have docstrings in pyi stub
            #     files. But if there are any, we need to ensure that the file
            #     is properly formatted.
            return re.sub(
                r'''
                (""")\n+             # match triple quotes
                (
                    (\s*@.+\n)*     # match decorators if any
                    \s*(class|def)  # match class/function definition
                )
                ''',
                r"\g<1>\n\n\g<2>",
                substitute_out,
                flags=re.VERBOSE,
            )
        if isinstance(env, str):
            return env
        assert_never(env)

    def write_with_template(
        self,
        filename: str | Path,
        template_fn: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> None:
        filename = Path(filename)
        if filename.is_absolute():
            raise AssertionError(f"filename must be relative: {filename}")
        file = self.install_dir / filename
        if file in self.files:
            raise AssertionError(f"duplicate file write {file}")
        self.files.add(file)
        if not self.dry_run:
            substitute_out = self.substitute_with_template(
                template_fn=template_fn,
                env_callable=env_callable,
            )
            self._write_if_changed(filename=file, contents=substitute_out)

    def write(
        self,
        filename: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> None:
        self.write_with_template(filename, filename, env_callable)

    def write_sharded(
        self,
        filename: str | Path,
        items: Iterable[T],
        *,
        key_fn: Callable[[T], str],
        env_callable: Callable[[T], dict[str, list[str]]],
        num_shards: int,
        base_env: dict[str, Any] | None = None,
        sharded_keys: set[str],
    ) -> None:
        self.write_sharded_with_template(
            filename,
            filename,
            items,
            key_fn=key_fn,
            env_callable=env_callable,
            num_shards=num_shards,
            base_env=base_env,
            sharded_keys=sharded_keys,
        )

    def write_sharded_with_template(
        self,
        filename: str | Path,
        template_fn: str | Path,
        items: Iterable[T],
        *,
        key_fn: Callable[[T], str],
        env_callable: Callable[[T], dict[str, list[str]]],
        num_shards: int,
        base_env: dict[str, Any] | None = None,
        sharded_keys: set[str],
    ) -> None:
        file = Path(filename)
        if file.is_absolute():
            raise AssertionError(f"filename must be relative: {filename}")
        everything: dict[str, Any] = {"shard_id": "Everything"}
        shards: list[dict[str, Any]] = [
            {"shard_id": f"_{i}"} for i in range(num_shards)
        ]
        all_shards = [everything] + shards

        if base_env is not None:
            for shard in all_shards:
                shard.update(base_env)

        for key in sharded_keys:
            for shard in all_shards:
                if key in shard:
                    if not isinstance(shard[key], list):
                        raise AssertionError("sharded keys in base_env must be a list")
                    shard[key] = shard[key].copy()
                else:
                    shard[key] = []

        def merge_env(into: dict[str, list[str]], from_: dict[str, list[str]]) -> None:
            for k, v in from_.items():
                if k not in sharded_keys:
                    raise AssertionError(f"undeclared sharded key {k}")
                into[k] += v

        if self.dry_run:
            # Dry runs don't write any templates, so incomplete environments are fine
            items = ()

        for item in items:
            key = key_fn(item)
            sid = string_stable_hash(key) % num_shards
            env = env_callable(item)

            merge_env(shards[sid], env)
            merge_env(everything, env)

        for shard in all_shards:
            shard_id = shard["shard_id"]
            self.write_with_template(
                file.with_stem(f"{file.stem}{shard_id}"),
                template_fn,
                lambda: shard,
            )

        # filenames is used to track compiled files, but FooEverything.cpp isn't meant to be compiled
        self.files.discard(self.install_dir / file.with_stem(f"{file.stem}Everything"))

    def write_outputs(self, variable_name: str, filename: str | Path) -> None:
        """Write a file containing the list of all outputs which are generated by this script."""
        content = "\n".join(
            (
                "set(",
                variable_name,
                # Use POSIX paths to avoid invalid escape sequences on Windows
                *(f'    "{file.as_posix()}"' for file in sorted(self.files)),
                ")",
            )
        )
        self._write_if_changed(filename, content)

    def template_dir_for_comments(self) -> str:
        """
        This needs to be deterministic. The template dir is an absolute path
        that varies across builds. So, just use the path relative to this file,
        which will point to the codegen source but will be stable.
        """
        return os.path.relpath(self.template_dir, os.path.dirname(__file__))


# Helper function to generate file manager
def make_file_manager(
    options: Namespace,
    install_dir: str | Path | None = None,
) -> FileManager:
    template_dir = os.path.join(options.source_path, "templates")
    install_dir = install_dir if install_dir else options.install_dir
    return FileManager(
        install_dir=install_dir,
        template_dir=template_dir,
        dry_run=options.dry_run,
    )


# Helper function to create a pretty representation for dataclasses
def dataclass_repr(
    obj: Any,
    indent: int = 0,
    width: int = 80,
) -> str:
    return pformat(obj, indent, width)


def _format_dict(
    attr: dict[Any, Any],
    indent: int,
    width: int,
    curr_indent: int,
) -> str:
    curr_indent += indent + 3
    dict_repr = []
    for k, v in attr.items():
        k_repr = repr(k)
        v_str = (
            pformat(v, indent, width, curr_indent + len(k_repr))
            if is_dataclass(v)
            else repr(v)
        )
        dict_repr.append(f"{k_repr}: {v_str}")

    return _format(dict_repr, indent, width, curr_indent, "{", "}")


def _format_list(
    attr: list[Any] | set[Any] | tuple[Any, ...],
    indent: int,
    width: int,
    curr_indent: int,
) -> str:
    curr_indent += indent + 1
    list_repr = [
        pformat(l, indent, width, curr_indent) if is_dataclass(l) else repr(l)
        for l in attr
    ]
    start, end = ("[", "]") if isinstance(attr, list) else ("(", ")")
    return _format(list_repr, indent, width, curr_indent, start, end)


def _format(
    fields_str: list[str],
    indent: int,
    width: int,
    curr_indent: int,
    start: str,
    end: str,
) -> str:
    delimiter, curr_indent_str = "", ""
    # if it exceed the max width then we place one element per line
    if len(repr(fields_str)) >= width:
        delimiter = "\n"
        curr_indent_str = " " * curr_indent

    indent_str = " " * indent
    body = f", {delimiter}{curr_indent_str}".join(fields_str)
    return f"{start}{indent_str}{body}{end}"


class NamespaceHelper:
    """A helper for constructing the namespace open and close strings for a nested set of namespaces.

    e.g. for namespace_str torch::lazy,

    prologue:
    namespace torch {
    namespace lazy {

    epilogue:
    } // namespace lazy
    } // namespace torch
    """

    def __init__(
        self,
        namespace_str: str,
        entity_name: str = "",
        max_level: int = 2,
    ) -> None:
        # cpp_namespace can be a colon joined string such as torch::lazy
        cpp_namespaces = namespace_str.split("::")
        if len(cpp_namespaces) > max_level:
            raise AssertionError(
                f"Codegen doesn't support more than {max_level} level(s) of "
                f"custom namespace. Got {namespace_str}."
            )
        self.cpp_namespace_ = namespace_str
        self.prologue_ = "\n".join([f"namespace {n} {{" for n in cpp_namespaces])
        self.epilogue_ = "\n".join(
            [f"}} // namespace {n}" for n in reversed(cpp_namespaces)]
        )
        self.namespaces_ = cpp_namespaces
        self.entity_name_ = entity_name

    @staticmethod
    def from_namespaced_entity(
        namespaced_entity: str,
        max_level: int = 2,
    ) -> NamespaceHelper:
        """
        Generate helper from nested namespaces as long as class/function name. E.g.: "torch::lazy::add"
        """
        names = namespaced_entity.split("::")
        entity_name = names[-1]
        namespace_str = "::".join(names[:-1])
        return NamespaceHelper(
            namespace_str=namespace_str, entity_name=entity_name, max_level=max_level
        )

    @property
    def prologue(self) -> str:
        return self.prologue_

    @property
    def epilogue(self) -> str:
        return self.epilogue_

    @property
    def entity_name(self) -> str:
        return self.entity_name_

    # Only allow certain level of namespaces
    def get_cpp_namespace(self, default: str = "") -> str:
        """
        Return the namespace string from joining all the namespaces by "::" (hence no leading "::").
        Return default if namespace string is empty.
        """
        return self.cpp_namespace_ if self.cpp_namespace_ else default


class OrderedSet(Generic[T]):
    storage: dict[T, None]

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        if iterable is None:
            self.storage = {}
        else:
            self.storage = dict.fromkeys(iterable)

    def __contains__(self, item: T) -> bool:
        return item in self.storage

    def __iter__(self) -> Iterator[T]:
        return iter(self.storage.keys())

    def update(self, items: OrderedSet[T]) -> None:
        self.storage.update(items.storage)

    def add(self, item: T) -> None:
        self.storage[item] = None

    def copy(self) -> OrderedSet[T]:
        ret: OrderedSet[T] = OrderedSet()
        ret.storage = self.storage.copy()
        return ret

    @staticmethod
    def union(*args: OrderedSet[T]) -> OrderedSet[T]:
        ret = args[0].copy()
        for s in args[1:]:
            ret.update(s)
        return ret

    def __or__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return OrderedSet.union(self, other)

    def __ior__(self, other: OrderedSet[T]) -> Self:
        self.update(other)
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return self.storage == other.storage
        else:
            return set(self.storage.keys()) == other
<|endoftext|>from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    SchemaKind,
    Type,
)
from torchgen.utils import mapMaybe


if TYPE_CHECKING:
    from collections.abc import Sequence


def is_tensor(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.Tensor


def is_optional_tensor(typ: Type) -> bool:
    return isinstance(typ, OptionalType) and is_tensor(typ.elem)


def is_tensor_list(typ: Type) -> bool:
    return isinstance(typ, ListType) and is_tensor(typ.elem)


def unwrap_tensor(name: str, cur_level_var: str) -> list[str]:
    result = f"""\
    auto [{name}_value, {name}_bdim] = unwrapTensorAtLevel({name}, {cur_level_var});"""
    return textwrap.dedent(result).split("\n")


def unwrap_optional_tensor(name: str, cur_level_var: str) -> list[str]:
    result = f"""\
    std::optional<Tensor> {name}_value;
    std::optional<int64_t> {name}_bdim;
    if ({name}) {{
        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), {cur_level_var});
    }}"""
    return textwrap.dedent(result).split("\n")


def gen_unwraps(
    flat_arguments: Sequence[Argument], cur_level_var: str
) -> tuple[str, list[str]]:
    arg_names = [a.name for a in flat_arguments]
    arg_types = [a.type for a in flat_arguments]

    tensors = [name for typ, name in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [
        name for typ, name in zip(arg_types, arg_names) if is_optional_tensor(typ)
    ]

    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor, cur_level_var)

    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor, cur_level_var)
    unwrap_code = "\n".join(unwraps)

    unwrapped_arg_list = []
    for arg in arg_names:
        if arg in tensors or arg in optional_tensors:
            unwrapped_arg_list += [f"{arg}_value", f"{arg}_bdim"]
        else:
            unwrapped_arg_list.append(arg)
    return unwrap_code, unwrapped_arg_list


def gen_case_where_all_bdims_are_none(
    outer_sig: DispatcherSignature, schema: FunctionSchema, cur_level_var: str
) -> str:
    conditions = []
    flat_args = schema.arguments.flat_all
    for arg in flat_args:
        if not arg.type.is_tensor_like():
            continue
        conditions.append(f"!isBatchedAtLevel({arg.name}, {cur_level_var})")

    sig = DispatcherSignature.from_schema(schema)
    translated_args = ", ".join(
        e.expr for e in translate(outer_sig.arguments(), sig.arguments())
    )
    return f"""\
if ({" && ".join(conditions)}) {{
  return at::_ops::{sig.func.name.unambiguous_name()}::call({translated_args});
}}"""


def gen_returns(
    returns: tuple[Return, ...], cur_level_var: str, results_var: str
) -> str:
    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret.type):
            wrapped_returns.append(
                f"makeBatched(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})"
            )
            idx += 2
        elif is_tensor_list(ret.type):
            wrapped_returns.append(
                f"makeBatchedVector(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})"
            )
            idx += 2
        else:
            wrapped_returns.append(f"std::get<{idx}>({results_var})")
            idx += 1
    if len(wrapped_returns) == 1:
        result = f"return {wrapped_returns[0]};"
    else:
        result = f"return std::make_tuple({', '.join(wrapped_returns)});"
    return result


def accepts_at_least_one_tensor_input(schema: FunctionSchema) -> bool:
    return any(a.type.is_tensor_like() for a in schema.arguments.flat_all)


def is_mutated_arg(argument: Argument) -> bool:
    return argument.annotation is not None and argument.annotation.is_write


def gen_vmap_inplace_plumbing(native_function: NativeFunction) -> str | None:
    # Assumptions:
    # - only one argument is being modified in-place
    # - the argument that is being modified in-place is the first argument
    # - all returns are either Tensor, tuple of Tensor, or TensorList
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    # Check assumptions. If these are invalid we return None
    # and punt the work to handle them to the future.
    if schema.kind() != SchemaKind.inplace:
        raise AssertionError(f"Expected inplace schema, got {schema.kind()}")
    if not is_mutated_arg(schema.arguments.flat_all[0]):
        return None
    if len([arg for arg in schema.arguments.flat_all if is_mutated_arg(arg)]) != 1:
        return None

    # Only support cases where all returns are Tensors or vector<Tensor>
    if len(returns) == 0:
        return None
    if not all(is_tensor(ret.type) or is_tensor_list(ret.type) for ret in returns):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None

    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)

    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + "_generated_plumbing")} {{
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  batch_rule({", ".join(unwrapped_arg_list)});
  return {schema.arguments.flat_all[0].name};
}}"""


def gen_vmap_plumbing_no_returns(native_function: NativeFunction) -> str:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)

    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + "_generated_plumbing")} {{
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  batch_rule({", ".join(unwrapped_arg_list)});
}}"""


def gen_vmap_plumbing(native_function: NativeFunction) -> str | None:
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns

    # Only support cases where all returns are Tensors or vector<Tensor>
    if not accepts_at_least_one_tensor_input(schema):
        return None
    if len(returns) == 0:
        return gen_vmap_plumbing_no_returns(native_function)
    return_symint_overrides = [
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_flash_attention_quantized",
    ]
    if (
        not all(ret.type.is_tensor_like() for ret in returns)
        and schema.name.unambiguous_name() not in return_symint_overrides
    ):
        return None
    # in-place views need special handling
    if "inplace_view" in native_function.tags:
        return None

    if schema.kind() == SchemaKind.inplace:
        return gen_vmap_inplace_plumbing(native_function)

    # Don't support these (mutable, out, scratch)
    if schema.kind() != SchemaKind.functional:
        return None

    results_var = "results"
    cur_level_var = "cur_level"

    unwraps, unwrapped_arg_list = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)

    wrapped_returns = gen_returns(returns, cur_level_var, results_var)
    return f"""\
template <typename batch_rule_t, batch_rule_t batch_rule>
{sig.decl(name=schema.name.unambiguous_name() + "_generated_plumbing")} {{
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
  auto maybe_layer = maybeCurrentDynamicLayer();
  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");
  int64_t {cur_level_var} = maybe_layer->layerId();
{textwrap.indent(bdims_all_none_case, "  ")}
{textwrap.indent(unwraps, "  ")}
  auto {results_var} = batch_rule({", ".join(unwrapped_arg_list)});
  {wrapped_returns}
}}"""


@dataclass(frozen=True)
class ComputeBatchRulePlumbing:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        result = gen_vmap_plumbing(f)
        return result


def gen_all_vmap_plumbing(native_functions: Sequence[NativeFunction]) -> str:
    body = "\n".join(list(mapMaybe(ComputeBatchRulePlumbing(), native_functions)))
    return f"""
#pragma once
#include <ATen/Operators.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at {{ namespace functorch {{

{body}

}}}} // namespace at::functorch
"""
<|endoftext|>from __future__ import annotations

import difflib
import os
import re
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.aoti.fallback_ops import aten_shimified_ops, inductor_fallback_ops
from torchgen.api.types import DispatcherSignature
from torchgen.api.types.signatures import CppSignature, CppSignatureGroup
from torchgen.context import method_with_native_function
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    DispatchKey,
    FunctionSchema,
    is_cuda_dispatch_key,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    OptionalType,
    Type,
    Variant,
)
from torchgen.utils import FileManager, mapMaybe


if TYPE_CHECKING:
    from collections.abc import Sequence


base_type_to_c_type = {
    BaseTy.Tensor: "AtenTensorHandle",
    BaseTy.bool: "int32_t",  # Use int to pass bool
    BaseTy.int: "int64_t",
    BaseTy.SymInt: "int64_t",  # Inductor-generated code won't see a SymInt
    BaseTy.Scalar: "double",  # Use double to pass both integer and floating point
    BaseTy.float: "double",  # TODO: how about other floating point types?
    BaseTy.str: "const char*",
    BaseTy.DeviceIndex: "int32_t",
    BaseTy.Layout: "int32_t",  # Represent enum as int
    BaseTy.MemoryFormat: "int32_t",  # Represent enum as int
    BaseTy.ScalarType: "int32_t",  # Represent enum as int
    BaseTy.Generator: "AtenGeneratorHandle",
}

base_type_to_aten_type = {
    BaseTy.Tensor: "at::Tensor",
    BaseTy.bool: "bool",
    BaseTy.int: "int64_t",
    BaseTy.SymInt: "c10::SymInt",
    BaseTy.Scalar: "c10::Scalar",
    BaseTy.float: "double",
    BaseTy.str: "::std::string_view",
    BaseTy.DeviceIndex: "c10::DeviceIndex",
    BaseTy.Layout: "c10::Layout",
    BaseTy.MemoryFormat: "c10::MemoryFormat",
    BaseTy.ScalarType: "c10::ScalarType",
    BaseTy.Generator: "at::Generator",
}

base_type_to_callsite_expr = {
    BaseTy.Tensor: "resolve_tensor_dispatch_flags",
    BaseTy.bool: "",
    BaseTy.int: "",
    BaseTy.SymInt: "",
    BaseTy.Scalar: "",
    BaseTy.float: "",
    BaseTy.str: "",
    BaseTy.DeviceIndex: "static_cast<c10::DeviceIndex>",
    BaseTy.Layout: "static_cast<c10::Layout>",
    BaseTy.MemoryFormat: "static_cast<c10::MemoryFormat>",
    BaseTy.ScalarType: "static_cast<c10::ScalarType>",
    BaseTy.Generator: "*generator_handle_to_generator_pointer",
}


# convert args to C types, names in declarations, and expressions in function bodies
def convert_arg_type_and_name(
    typ: Type,
    name: str,
    is_write: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    if isinstance(typ, BaseType):
        if typ.name in base_type_to_c_type:
            if typ.name == BaseTy.Tensor and is_write:
                # For output tensors, our normal call to resolve_tensor_dispatch_flags
                # results in an rvalue tensor, which can't be passed to at::Tensor&.
                # Override this case specifically.
                callsite_expr = [f"*tensor_handle_to_tensor_pointer({name})"]
            else:
                callsite_expr = [
                    f"{base_type_to_callsite_expr[typ.name]}({name})"
                    if base_type_to_callsite_expr[typ.name]
                    else name
                ]

            return (
                [base_type_to_c_type[typ.name]],
                [name],
                [base_type_to_aten_type[typ.name]],
                callsite_expr,
            )
        elif typ.name == BaseTy.Device:
            return (
                ["int32_t", "int32_t"],
                [name, name + "_index_"],
                ["c10::Device"],
                [
                    f"c10::Device(static_cast<c10::DeviceType>({name}), static_cast<c10::DeviceIndex>({name}_index_))"
                ],
            )
        else:
            raise NotImplementedError(f"TODO: add support for arg type {repr(typ)}")
    elif isinstance(typ, OptionalType):
        c_types, names, aten_types, callsite_exprs = convert_arg_type_and_name(
            typ.elem, name
        )
        j = 0  # index for names
        new_aten_types = []
        new_callsite_exprs = []
        for aten_type in aten_types:
            # Use pointer to denote optional type
            c_types[j] = c_types[j] + "*"
            if aten_type.startswith("c10::ArrayRef<"):
                # ArrayRef is passed as pointer + size, but no need to add "*" to the size argument
                new_aten_types.append(f"::std::optional<{aten_type}>")
                base_type = aten_type[len("c10::ArrayRef<") : -1]
                new_callsite_exprs.append(
                    f"pointer_to_optional_list<{base_type}>({names[j]}, {names[j + 1]})"
                )
                j += 2
            elif aten_type == "c10::Device":
                # Device is passed as device_type + device_index
                new_aten_types.append("::std::optional<c10::Device>")
                new_callsite_exprs.append(
                    f"pointer_to_optional_device({names[j]}, {names[j + 1]})"
                )
                j += 2
            elif aten_type == "at::Tensor":
                new_aten_types.append(f"::std::optional<{aten_type}>")
                new_callsite_exprs.append(f"resolve_tensor_dispatch_flags({names[j]})")
                j += 1
            else:
                new_aten_types.append(f"::std::optional<{aten_type}>")
                new_callsite_exprs.append(
                    f"pointer_to_optional<{aten_type}>({names[j]})"
                )
                j += 1

        return (
            c_types,
            names,
            new_aten_types,
            new_callsite_exprs,
        )
    elif isinstance(typ, ListType):
        # Need to explicitly pass the list as pointer + length
        c_types, names, aten_types, _ = convert_arg_type_and_name(typ.elem, name)
        if len(c_types) != 1:
            raise AssertionError(f"ListType with unsupported element type {repr(typ)}")

        # The list content should never be modified
        c_types[0] = f"const {c_types[0]}*"
        c_types.append("int64_t")
        name = names[0]
        names.append(name + "_len_")

        atype = aten_types[0]
        callsite_exprs = []
        if atype == "bool":
            # no converter from std::vector<bool> to c10::ArrayRef<bool>
            # construct std::array<bool, N> instead
            if typ.size is None:
                raise AssertionError("bool ListType must have a size")
            callsite_exprs.append(f"pointer_to_list<{typ.size}>({name})")
        elif atype == "at::Tensor" and not is_write:
            callsite_exprs.append(
                f"resolve_tensor_list_dispatch_flags({name}, {name}_len_)"
            )
        elif atype == "::std::optional<at::Tensor>":
            # convert from std::vector<::std::optional<at::Tensor>> to c10::List<::std::optional<at::Tensor>>
            callsite_exprs.append(
                f"c10::List<{atype}>(c10::ArrayRef<{atype}>(resolve_tensor_list_dispatch_flags({name}, {name}_len_)))"
            )
        else:
            callsite_exprs.append(f"pointer_to_list<{atype}>({name}, {name}_len_)")

        aten_types = [f"c10::ArrayRef<{t}>" for t in aten_types]
        return (
            c_types,
            names,
            aten_types,
            callsite_exprs,
        )
    raise NotImplementedError(f"Argument type {repr(typ)} not supported!")


def zip_type_and_name(types: list[str], names: list[str]) -> list[str]:
    return [typ + " " + name for typ, name in zip(types, names)]


# Generate argument declarations and callsite expressions
def gen_arguments(
    flat_arguments: Sequence[Argument], skipped_args: set[str]
) -> tuple[list[str], list[str]]:
    types: list[str] = []
    new_names: list[str] = []
    callsite_exprs: list[str] = []
    for arg in flat_arguments:
        if arg.name in skipped_args:
            # Pass the arg's schema default when available (e.g. "false" for
            # a bool arg with default=False), so non-optional args with defaults
            # can be versioned too. Fall back to std::nullopt for optional args
            # with no default (matches historical behavior).
            if arg.default is not None:
                from torchgen.api.cpp import default_expr

                callsite_exprs.append(default_expr(arg.default, arg.type, symint=False))
            else:
                callsite_exprs.append("std::nullopt")
            continue
        new_types, names, _, new_callsite_exprs = convert_arg_type_and_name(
            arg.type, arg.name, arg.is_write
        )
        types.extend(new_types)
        new_names.extend(names)
        callsite_exprs.extend(new_callsite_exprs)
    return zip_type_and_name(types, new_names), callsite_exprs


# Return values are passed out as pointer arguments because all the C shim functions
# are expected to return AOTITorchError.
# Generate returns as declarations and callsite expressions
def gen_returns(schema: FunctionSchema) -> tuple[list[str], list[str]]:
    types = []
    names = []
    for idx, ret in enumerate(schema.returns):
        names.append(f"ret{idx}")
        if isinstance(ret.type, BaseType) and ret.type.name in base_type_to_c_type:
            types.append(base_type_to_c_type[ret.type.name] + "*")
        else:
            raise NotImplementedError(
                f"TODO: add support for return type {repr(ret.type)}"
            )

    def convert_return(typ: BaseType, val: str) -> str:
        if typ.name == BaseTy.Tensor:
            return f"new_tensor_handle(std::move({val}))"
        elif typ.name == BaseTy.SymInt:
            return f"{val}.expect_int()"
        elif typ.name == BaseTy.Scalar:
            return f"{val}.toDouble()"
        else:
            return val

    ret_pointer_can_be_null = False
    unambiguous_name = schema.name.unambiguous_name()
    for name in (
        "_functional_sym_constrain_range",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_efficient_attention_backward",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_fused_attention_overrideable",
        "_thhn_fused_lstm_cell_backward_impl",
        "convolution_backward",
        "grid_sampler_2d_backward",
        "grid_sampler_3d_backward",
        "linear_backward",
    ):
        if name in unambiguous_name:
            ret_pointer_can_be_null = True
            break

    callsite_exprs: list[str] = []
    for idx, ret in enumerate(schema.returns):
        tmp = "tmp_result" if len(names) == 1 else f"std::get<{idx}>(tmp_result)"
        if not isinstance(ret.type, BaseType):
            raise AssertionError(f"Expected BaseType for return, got {type(ret.type)}")
        rval = convert_return(ret.type, tmp)
        if ret_pointer_can_be_null:
            callsite_exprs.append(f"if ({names[idx]}) {{ *{names[idx]} = {rval}; }}")
        else:
            callsite_exprs.append(f"*{names[idx]} = {rval};")

    return zip_type_and_name(types, names), callsite_exprs


# gen.py generates header first and then src, so caching the result here to avoid duplicate work
declaration_definition_cache: dict[tuple[str, str, str], tuple[str, str]] = {}


_TORCH_VERSION_PATTERN = re.compile(r"^TORCH_VERSION_\d+_\d+_\d+$")


def _get_earliest_torch_version_for_op_variant(
    op_metadata: dict[str, str | dict[str, list[str] | str]],
    op_version: int,
) -> str | None:
    """
    Return the TORCH_VERSION_X_Y_Z macro string at which the given op variant became
    available, or None if the variant is ungated.

    op_metadata contains the entry for an op in a dictionary like aten_shimified_ops
    in torchgen/aoti/fallback_ops.py

    op_version is the integer extracted from a "vN" key in `op_metadata`. The base op
    has op_version == 1 and reads from a top-level "since" key; later variants read
    "since" from their own dict-form entry {"new_args": [...], "since": "..."}.
    """
    if op_version == 1:
        since = op_metadata.get("since")
    else:
        variant = op_metadata.get(f"v{op_version}")
        since = variant.get("since") if isinstance(variant, dict) else None
    if since is None:
        return None
    if not isinstance(since, str) or not _TORCH_VERSION_PATTERN.match(since):
        raise AssertionError(
            f"`since` value {since!r} is not of the form TORCH_VERSION_X_Y_Z"
        )
    return since


def gen_declaration_and_definition(
    schema: FunctionSchema,
    device: str,
    backend_call: str,
    op_metadata: dict[str, str | dict[str, list[str] | str]],
) -> tuple[str, str]:
    base_name = schema.name.unambiguous_name()

    global declaration_definition_cache
    if (base_name, device, backend_call) in declaration_definition_cache:
        return declaration_definition_cache[(base_name, device, backend_call)]

    # Check the validity of op_metadata. Each "vN" entry is a dict
    # {"new_args": [...], "since": "TORCH_VERSION_X_Y_Z"} where "since" is optional;
    # op_metadata may also carry a top-level "since" key that would version-gate v1.
    indexed_op_metadata: dict[int, list[str]] = {1: []}
    for ver_str, payload in sorted(op_metadata.items()):
        # since will get processed later per op
        if ver_str == "since":
            continue
        if not ver_str.startswith("v"):
            raise AssertionError(
                f"Version number for {base_name} is {ver_str}, not starting with 'v'"
            )
        try:
            ver_id = int(ver_str[1:])
        except ValueError as e:
            raise AssertionError(
                f"Version number for {base_name} is {ver_str}, not a valid integer after 'v'"
            ) from e
        if ver_id in indexed_op_metadata:
            raise AssertionError(f"{ver_str} for {base_name} has already been defined")
        if not isinstance(payload, dict):
            raise AssertionError(
                f"Variant {ver_str} for {base_name} must be a dict of the form "
                "{'new_args': [...], 'since': 'TORCH_VERSION_X_Y_Z'}"
            )
        new_args = payload.get("new_args", [])
        if not isinstance(new_args, list):
            raise AssertionError(
                f"Variant {ver_str} for {base_name} has non-list 'new_args' field when a "
                "list of arg names differing from prev versions is expected."
            )
        indexed_op_metadata[ver_id] = new_args

    declarations: list[str] = []
    definitions: list[str] = []
    skipped_args: set[str] = set()

    for ver_id, new_args in sorted(indexed_op_metadata.items(), reverse=True):
        # Iterate in the reverse order, so the latest version of an op will get generated first
        # with all the arguments included, while a set of to-be-trimmed args is carried down
        # to generate earlier version of the op.
        func_name = base_name if ver_id == 1 else f"{base_name}_v{ver_id}"
        if schema.is_out_fn():
            # out_variant has out arguments in the front, and it's ok to ignore return values
            # because C shim functions only return AOTITorchError
            args, callsite_exprs = gen_arguments(
                [*schema.arguments.out, *schema.arguments.flat_non_out], skipped_args
            )
            ret_assignments: list[str] = []
        else:
            args, callsite_exprs = gen_arguments(
                schema.arguments.flat_all, skipped_args
            )
            # ignore return values for inplace ops
            ret_declarations, ret_assignments = (
                ([], []) if schema.name.name.inplace else gen_returns(schema)
            )
            args.extend(ret_declarations)

        declaration = textwrap.dedent(
            f"AOTITorchError aoti_torch_{device}_{func_name}({', '.join(args)})"
        )

        tmp_result = "auto tmp_result = " if ret_assignments else ""
        indent = "\t\t"
        ret_assignments_str = (
            "\n".join(indent + r for r in ret_assignments) if ret_assignments else ""
        )
        definition = (
            textwrap.dedent(f"""
        {declaration} {{
            AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({{
                {tmp_result}{backend_call}(
                    {", ".join(callsite_exprs)}
                );
        """)
            + ret_assignments_str
            + textwrap.dedent("""
            });
        }
        """)
        )
        skipped_args.update(new_args)
        decl = f"AOTI_TORCH_EXPORT {declaration};"
        since = _get_earliest_torch_version_for_op_variant(op_metadata, ver_id)
        if since is not None:
            decl = (
                f"#if TORCH_FEATURE_VERSION >= {since}\n"
                f"{decl}\n"
                f"#endif // TORCH_FEATURE_VERSION >= {since}"
            )
        declarations.append(decl)
        definitions.append(definition)

    declaration_definition_cache[(base_name, device, backend_call)] = (
        "\n".join(declarations),
        "\n".join(definitions),
    )
    return declaration_definition_cache[(base_name, device, backend_call)]


def gen_static_dispatch_backend_call_signature(f: NativeFunction) -> CppSignature:
    sig = DispatcherSignature.from_schema(f.func)
    cpp_sigs = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    if cpp_sig is None:
        raise AssertionError(f"No cpp signature found for {f.func.name}")
    return cpp_sig


def gen_static_dispatch_backend_call(
    f: NativeFunction,
    backend_index: BackendIndex | None = None,
) -> str:
    sig = DispatcherSignature.from_schema(f.func)
    cpp_sig = gen_static_dispatch_backend_call_signature(f)

    if backend_index is None:
        # Check if this is a symint function and if the function only has method variants
        if sig.symint and f.func.has_symint():
            has_function_variant = Variant.function in f.variants

            if not has_function_variant:
                # Functions with both function and method variants can use the at::{*}_symint version
                # (e.g., narrow -> at::narrow_symint), BUT
                # Method-only functions with symint parameters should use at::symint:: namespace
                # Remove the _symint suffix since at::symint:: namespace uses the base name
                # (e.g., new_empty -> at::symint::new_empty<c10::SymInt>)
                base_name = cpp_sig.name()
                base_name = base_name.removesuffix("_symint")  # Remove "_symint" suffix
                return f"at::symint::{base_name}<c10::SymInt>"

        return f"at::{cpp_sig.name()}"
    else:
        return f"at::{backend_index.dispatch_key.lower()}::{cpp_sig.name()}"


def get_backend_index_for_aoti(
    func: NativeFunction,
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: DispatchKey | None,
    backend_indices: dict[DispatchKey, BackendIndex],
    extend_aoti_c_shim: bool,
) -> BackendIndex | None:
    backend_index = None

    if dispatch_key is None:
        return backend_index

    if backend_indices[dispatch_key].has_kernel(func) or (
        func.structured_delegate is not None
        and func.structured_delegate in func_group_mapping
        and backend_indices[dispatch_key].has_kernel(
            func_group_mapping[func.structured_delegate]
        )
    ):
        backend_index = backend_indices[dispatch_key]
    else:
        # for the extend out-of-tree kernels, we don't need to
        # duplicatly create C shim wrappers for other dispatch keys
        if extend_aoti_c_shim:
            return backend_index

        elif backend_indices[DispatchKey.CompositeExplicitAutograd].has_kernel(func):
            # We need to create C shim wrappers for CompositeExplicitAutograd kernels
            backend_index = backend_indices[DispatchKey.CompositeExplicitAutograd]
        elif backend_indices[
            DispatchKey.CompositeExplicitAutogradNonFunctional
        ].has_kernel(func):
            # We need to create C shim wrappers for CompositeExplicitAutogradNonFunctional kernels
            backend_index = backend_indices[
                DispatchKey.CompositeExplicitAutogradNonFunctional
            ]
        elif backend_indices[DispatchKey.CompositeImplicitAutograd].has_kernel(func):
            backend_index = backend_indices[DispatchKey.CompositeImplicitAutograd]

    return backend_index


def get_header_for_aoti(
    func: NativeFunction,
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: DispatchKey | None,
    backend_indices: dict[DispatchKey, BackendIndex],
    extend_aoti_c_shim: bool,
) -> str | None:
    backend_index = get_backend_index_for_aoti(
        func, func_group_mapping, dispatch_key, backend_indices, extend_aoti_c_shim
    )
    if backend_index is None:
        if dispatch_key is None:
            return f"#include <ATen/ops/{func.root_name}.h>"
        return None

    return f"#include <ATen/ops/{func.root_name}_{backend_index.dispatch_key.lower()}_dispatch.h>"


def get_fallback_op_name(func: NativeFunction) -> str:
    return (
        f"{func.namespace}.{func.func.name.name}.{func.func.name.overload_name}"
        if func.func.name.overload_name
        else f"{func.namespace}.{func.func.name.name}.default"
    )


def gen_c_shim(
    func: NativeFunction,
    version_info: dict[str, str | dict[str, list[str] | str]],
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: DispatchKey | None,
    backend_indices: dict[DispatchKey, BackendIndex],
    header: bool,
    extend_aoti_c_shim: bool,
) -> str | None:
    backend_index = get_backend_index_for_aoti(
        func, func_group_mapping, dispatch_key, backend_indices, extend_aoti_c_shim
    )
    if backend_index is None and dispatch_key is not None:
        return None

    schema = func.func
    device = "aten" if dispatch_key is None else dispatch_key.lower()
    backend_call = gen_static_dispatch_backend_call(
        func,
        backend_index,
    )

    try:
        if header:
            declaration, _ = gen_declaration_and_definition(
                schema, device, backend_call, version_info
            )
            return declaration
        else:
            _, definition = gen_declaration_and_definition(
                schema, device, backend_call, version_info
            )
            return definition

    except NotImplementedError:
        return None


@dataclass(frozen=True)
class ShimGenerator:
    inductor_fallback_ops: dict[str, dict[str, str | dict[str, list[str] | str]]]
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup]
    dispatch_key: DispatchKey | None
    backend_indices: dict[DispatchKey, BackendIndex]
    header: bool  # True to generate .h and False to generate .cpp
    extend_aoti_c_shim: bool

    @method_with_native_function
    def __call__(
        self,
        func: NativeFunction,
    ) -> str | None:
        version_info = self.inductor_fallback_ops[get_fallback_op_name(func)]
        result = gen_c_shim(
            func,
            version_info,
            self.func_group_mapping,
            self.dispatch_key,
            self.backend_indices,
            self.header,
            self.extend_aoti_c_shim,
        )
        return result


def gen_aoti_c_shim(
    native_functions: Sequence[NativeFunction],
    inductor_fallback_ops: dict[str, dict[str, str | dict[str, list[str] | str]]],
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: DispatchKey | None,
    backend_indices: dict[DispatchKey, BackendIndex],
    header: bool,
    extend_aoti_c_shim: bool,
    includes: str = "",
) -> str:
    body = "\n".join(
        list(
            mapMaybe(
                ShimGenerator(
                    inductor_fallback_ops,
                    func_group_mapping,
                    dispatch_key,
                    backend_indices,
                    header,
                    extend_aoti_c_shim,
                ),
                native_functions,
            )
        )
    )
    device = "aten" if dispatch_key is None else dispatch_key.lower()
    include_device_functions = (
        "#include <ATen/Functions.h>"
        if dispatch_key is None
        else f"#include <ATen/{str(dispatch_key)}Functions.h>"
    )
    aten_warning = (
        (
            "\n\n// This file corresponds to the aten_shimified_ops list in torchgen/aoti/fallback_ops.py\n"
        )
        if dispatch_key is None
        else ""
    )
    warning = """

// WARNING: THIS FILE IS AUTOGENERATED BY torchgen. DO NOT MODIFY BY HAND.
// See https://github.com/pytorch/pytorch/blob/7e86a7c0155295539996e0cf422883571126073e/torchgen/gen.py#L2424-L2436 for details"""

    if header:
        return (
            warning
            + aten_warning
            + textwrap.dedent("""

            #pragma once

            #include <torch/csrc/inductor/aoti_torch/c/shim.h>

            #ifdef __cplusplus
            extern "C" {
            #endif

            """)
            + body
            + textwrap.dedent("""

            #ifdef __cplusplus
            } // extern "C"
            #endif
            """)
        )
    else:
        return (
            warning
            + aten_warning
            + textwrap.dedent(f"""

            #include <torch/csrc/inductor/aoti_torch/generated/{"extend/" if extend_aoti_c_shim else ""}c_shim_{device}.h>
            #include <torch/csrc/inductor/aoti_torch/utils.h>

            #ifndef AT_PER_OPERATOR_HEADERS
            {include_device_functions}
            #include <ATen/CompositeExplicitAutogradFunctions.h>
            #include <ATen/CompositeExplicitAutogradNonFunctionalFunctions.h>
            #include <ATen/CompositeImplicitAutogradFunctions.h>
            #else
            """)
            + includes
            + textwrap.dedent("""
            #endif // AT_PER_OPERATOR_HEADERS

            using namespace torch::aot_inductor;

            """)
            + body
        )


def gen_aoti_c_shim_files(
    aoti_fm: FileManager,
    aoti_backends: set[DispatchKey | None],
    native_functions: Sequence[NativeFunction],
    backend_indices: dict[DispatchKey, BackendIndex],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    extra_cuda_headers: str,
    extend_aoti_c_shim: bool,
    update_aoti_c_shim: bool,
) -> None:
    structured_func_group_dict = {}
    for func_group in structured_native_functions:
        for func in func_group.functions():
            if func.structured_delegate is not None:
                structured_func_group_dict[func.structured_delegate] = func_group
                break

    for dispatch_key in aoti_backends:
        # Use aten_shimified_ops for the aten backend, inductor_fallback_ops for others
        fallback_ops_dict = (
            aten_shimified_ops if dispatch_key is None else inductor_fallback_ops
        )
        fallbacks = {}
        for func in native_functions:
            op_name = get_fallback_op_name(func)
            if op_name in fallback_ops_dict:
                fallbacks[op_name] = func
        fallback_native_functions = tuple(
            value for _, value in sorted(fallbacks.items())
        )

        # Use "aten" as the device name when dispatch_key is Generic
        device_name = "aten" if dispatch_key is None else dispatch_key.lower()

        # header files were checked in for ABI-compatibility checking
        header_file_name = f"c_shim_{device_name}.h"
        new_header = gen_aoti_c_shim(
            fallback_native_functions,
            fallback_ops_dict,
            structured_func_group_dict,
            dispatch_key,
            backend_indices,
            header=True,
            extend_aoti_c_shim=extend_aoti_c_shim,
            includes="",
        )
        if update_aoti_c_shim:
            aoti_fm.write(
                header_file_name,
                lambda: new_header,
            )
        else:
            try:
                with open(
                    os.path.join(aoti_fm.install_dir, header_file_name)
                ) as old_file:
                    old_header = old_file.read()

                    if old_header != new_header:
                        diff = "\n".join(
                            difflib.unified_diff(
                                old_header.splitlines(),
                                new_header.splitlines(),
                                fromfile="expected",
                                tofile="actual",
                                lineterm="",
                            )
                        )

                        raise RuntimeError(f"""
The generated AOTInductor C shim header files have unexpectedly changed. This
indicates an AOTInductor fallback operator ABI backward compatibility breakage!!!
Only in a limited number of situations, this is allowed:

1. You added a fallback op to the inductor_fallback_ops list in torchgen/aoti/fallback_ops.py.
If that's the case, run `python torchgen/gen.py --update-aoti-c-shim` to add a new entry to
existing C shim header files.

2. You added a new default argument to an existing fallback op. This is clearly a BC breaking
change in the AOTInductor land. You need to annotate the new default argument in
torchgen/aoti/fallback_ops.py, and then run `python torchgen/gen.py --update-aoti-c-shim` to
update the C shim header files by creating different versions of the fallback op. See
https://github.com/pytorch/pytorch/pull/154848 as an example.

{diff}
                    """)
            except FileNotFoundError:
                print(
                    f"{os.path.join(aoti_fm.install_dir, header_file_name)} not found"
                )

        # cpp files are always generated on-the-fly
        def headers_for_aoti() -> str:
            headers = []
            for func in fallback_native_functions:
                header = get_header_for_aoti(
                    func,
                    structured_func_group_dict,
                    dispatch_key,
                    backend_indices,
                    extend_aoti_c_shim=extend_aoti_c_shim,
                )
                if header is not None:
                    headers.append(header)
            return "\n".join(sorted(set(headers)))

        extra_headers = (
            extra_cuda_headers
            if dispatch_key is not None and is_cuda_dispatch_key(dispatch_key)
            else ""
        )

        aoti_fm.write(
            f"c_shim_{device_name}.cpp",
            lambda: gen_aoti_c_shim(
                fallback_native_functions,
                fallback_ops_dict,
                structured_func_group_dict,
                dispatch_key,
                backend_indices,
                header=False,
                extend_aoti_c_shim=extend_aoti_c_shim,
                includes=headers_for_aoti() + "\n" + extra_headers,
            ),
        )
<|endoftext|># Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[assignment, misc]

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper  # type: ignore[assignment, misc]
YamlDumper = Dumper


# A custom loader for YAML that errors on duplicate keys.
# This doesn't happen by default: see https://github.com/yaml/pyyaml/issues/165
class YamlLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            if key in mapping:
                raise AssertionError(
                    f"Found a duplicate key in the yaml. key={key}, line={node.start_mark.line}"
                )
            mapping.append(key)
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        return mapping
<|endoftext|>from __future__ import annotations

import argparse
import functools
import json
import keyword
import os
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never

import yaml

import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
    Binding,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    NamedCType,
    NativeSignature,
    SpecialArgName,
)
from torchgen.context import (
    method_with_native_function,
    native_function_manager,
    with_native_function,
    with_native_function_and_indices,
)
from torchgen.gen_aoti_c_shim import (
    gen_aoti_c_shim_files,
    gen_static_dispatch_backend_call_signature,
)
from torchgen.gen_functionalization_type import (
    gen_functionalization_definition,
    gen_functionalization_registration,
    gen_functionalization_view_inverse_declaration,
    gen_functionalization_view_meta_classes_decl,
    gen_functionalization_view_meta_classes_impl,
    GenCompositeViewCopyKernel,
)
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseOperatorName,
    DEFAULT_KERNEL_NAMESPACE,
    dispatch_device_map,
    DispatchKey,
    FRAGMENT_NAMESPACES,
    FunctionSchema,
    is_cuda_dispatch_key,
    is_generic_dispatch_key,
    is_ufunc_dispatch_key,
    is_xpu_dispatch_key,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    OperatorName,
    OptionalType,
    SchemaKind,
    SelfArgument,
    STRUCTURED_DISPATCH_KEYS,
    TensorOptionsArguments,
    Type,
    Variant,
    ViewSchemaKind,
)
from torchgen.native_function_generation import (
    add_generated_native_functions,
    gen_composite_functional_kernel,
    gen_composite_out_kernel,
    pre_group_native_functions,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
    concatMap,
    context,
    FileManager,
    make_file_manager,
    mapMaybe,
    NamespaceHelper,
    Target,
)
from torchgen.yaml_utils import YamlDumper, YamlLoader


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


T = TypeVar("T")

# Welcome to the ATen code generator v2!  The ATen code generator is
# responsible for parsing native_functions.yaml and then generating
# various generated files (e.g., TypeDefault.cpp) based on the operators
# defined in this file.  This means that the code generator knows how to
# parse function schema, and then translate this into various C++ types
# and boilerplate code.
#
# Some things to know about this file when you modify it:
#
# - This file has STRICT mypy typechecking.  Typecheck it with
#   `mypy --config mypy-strict.ini` in the root source directory
#
# - Most of the heavy lifting lives in external modules:
#   - 'model' has the data model for native_functions.yaml.  The classes
#     in those file represent what you see when you look at
#     a native_functions.yaml
#   - 'api' has conversions for how to translate JIT schema into
#     the various C++ APIs that the codegen interacts with.  There
#     are in fact THREE different C++ APIs: the public C++ API,
#     the dispatcher API, and the legacy dispatcher API.  See each
#     of these respective files for more information

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         HELPER FUNCTIONS
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# A custom loader for YAML to let us also keep track of line numbers
# of each entry in the YAML file
class LineLoader(YamlLoader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        # Add 1 so line numbering starts at 1
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple("ParsedYaml", ["native_functions", "backend_indices"])


_GLOBAL_PARSE_NATIVE_YAML_CACHE: dict[str, ParsedYaml] = {}
_GLOBAL_PARSE_TAGS_YAML_CACHE: dict[str, set[str]] = {}


def file_manager_from_dispatch_key(
    dispatch_key: DispatchKey,
    device_fms: dict[str, FileManager],
    default_fm: FileManager,
) -> FileManager:
    fm = device_fms.get(
        next(
            (
                device
                for check, device in dispatch_device_map.items()
                if check(dispatch_key)
            ),
            "",
        ),
        default_fm,
    )
    return fm


def parse_native_yaml_struct(
    es: object,
    valid_tags: set[str],
    ignore_keys: set[DispatchKey] | None = None,
    path: str = "<stdin>",
    skip_native_fns_gen: bool = False,
) -> ParsedYaml:
    if not isinstance(es, list):
        raise AssertionError(f"Expected 'es' to be a list, but got {type(es)}")
    rs: list[NativeFunction] = []
    bs: dict[DispatchKey, dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    for e in es:
        if not isinstance(e, dict):
            raise AssertionError(f"Expected to be dict: {e}")
        if not isinstance(e.get("__line__"), int):
            raise AssertionError(f"Expected '__line__' to be int: {e}")
        loc = Location(path, e["__line__"])
        funcs = e.get("func")
        if funcs is None:
            raise AssertionError(f"Missed 'func' in {e}")
        with context(lambda: f"in {loc}:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
            rs.append(func)
            BackendIndex.grow_index(bs, m)
    error_check_native_functions(rs)
    # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
    indices: dict[DispatchKey, BackendIndex] = defaultdict(
        lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined,
            use_out_as_primary=True,
            external=False,
            device_guard=False,
            # I'm actually not sure about this; undefined could be hit on
            # empty TensorList, hypothetically that could have sizes in it
            index={},
        )
    )
    if not skip_native_fns_gen:
        add_generated_native_functions(rs, bs)
    for k, v in bs.items():
        # All structured in-tree operators are implemented in terms of their out operator.
        indices[k] = BackendIndex(
            dispatch_key=k,
            use_out_as_primary=True,
            external=False,
            # Only cuda-like devices in tree require device guards
            device_guard=is_cuda_dispatch_key(k) or is_xpu_dispatch_key(k),
            index=v,
        )
    return ParsedYaml(rs, indices)


def parse_tags_yaml_struct(es: object, path: str = "<stdin>") -> set[str]:
    if not isinstance(es, list):
        raise AssertionError(f"Expected 'es' to be a list, but got {type(es)}")
    rs: set[str] = set()
    for e in es:
        if not isinstance(e.get("__line__"), int):
            raise AssertionError(f"Expected '__line__' to be int: {e}")
        loc = Location(path, e["__line__"])
        tags = e.get("tag")
        with context(lambda: f"in {loc}:\n  {tags}"):
            e_i = e.copy()
            name = e_i.pop("tag")
            desc = e_i.pop("desc", "")
            # ensure that each tag has a non-empty description
            if desc == "":
                raise AssertionError(f"Tag '{name}' must have a non-empty description")
            rs.add(name)
    return rs


@functools.cache
def parse_tags_yaml(path: str) -> set[str]:
    global _GLOBAL_PARSE_TAGS_YAML_CACHE
    if path not in _GLOBAL_PARSE_TAGS_YAML_CACHE:
        with open(path) as f:
            es = yaml.load(f, Loader=LineLoader)
            _GLOBAL_PARSE_TAGS_YAML_CACHE[path] = parse_tags_yaml_struct(es, path=path)

    return _GLOBAL_PARSE_TAGS_YAML_CACHE[path]


def parse_native_yaml(
    path: str,
    tags_yaml_path: str,
    ignore_keys: set[DispatchKey] | None = None,
    *,
    skip_native_fns_gen: bool = False,
    loaded_yaml: object | None = None,
) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tags_yaml_path)

        # if a loaded yaml is provided, use that instead of reading from path
        if loaded_yaml is None:
            with open(path) as f:
                es = yaml.load(f, Loader=LineLoader)
        else:
            es = loaded_yaml

        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(
            es,
            valid_tags,
            ignore_keys,
            path=path,
            skip_native_fns_gen=skip_native_fns_gen,
        )

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]


# Some assertions are already performed during parsing, but those are only within a single NativeFunction.
# Assertions here are meant to be performed across NativeFunctions.
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    func_map: dict[OperatorName, NativeFunction] = {}
    base_func_map: dict[BaseOperatorName, list[NativeFunction]] = defaultdict(list)
    for f in funcs:
        func_map[f.func.name] = f
        base_func_map[f.func.name.name].append(f)
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map.get(f.structured_delegate)
            if delegate_func is None:
                raise AssertionError(
                    f"{f.func.name} is marked as a structured_delegate pointing to "
                    f"{f.structured_delegate}, but {f.structured_delegate} is missing."
                )
            if not delegate_func.structured:
                raise AssertionError(
                    f"{f.func.name} is marked as a structured_delegate pointing to "
                    f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. "
                    f"Consider adding 'structured=True' to the delegated operator"
                )

        # Check for reserved Python keywords
        PYTHON_RESERVED_KEYWORDS = set(keyword.kwlist)
        # List of pre-existing operators that are known to have reserved keywords
        # Exclusion list is used to suppress the assertion for these operators
        EXCLUSION_LIST = {
            ("_has_compatible_shallow_copy_type", "from"),
            ("random_.from", "from"),
            ("uniform_", "from"),
        }

        for arg in f.func.arguments.flat_all:
            if arg.name in PYTHON_RESERVED_KEYWORDS:
                if (str(f.func.name), arg.name) not in EXCLUSION_LIST:
                    raise AssertionError(
                        f"Argument name '{arg.name}' in function '{f.func.name}' is a reserved Python keyword."
                    )
        # See Note [resize_ in Functionalization]
        # resize_() is technically an inplace view op (and therefore needs the tag),
        # but it would be overkill to add a true "view" variant of resize.
        # Instead, resize_() gets special treatment in functionalization,
        # and we have a resize() op that is non-aliasing + functional.
        if (
            "inplace_view" in f.tags
            and str(f.func.name) != "resize_"
            and str(f.func.name) != "resize_as_"
            and str(f.func.name.name) != "set_"
        ):
            base_name = f.func.name.name
            if not base_name.inplace:
                raise AssertionError(
                    f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming "
                    "convention for inplace ops - the codegen expects the base name to have a trailing underscore."
                )
            out_of_place_base_name = BaseOperatorName(
                base_name.base, False, base_name.dunder_method
            )
            if len(base_func_map[out_of_place_base_name]) == 0:
                raise AssertionError(
                    f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding "
                    f"out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one."
                )


def cpp_string(s: str) -> str:
    """Convert a python string into a c++ string literal"""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\a", "\\a")
    s = s.replace("\b", "\\b")
    s = s.replace("\f", "\\f")
    s = s.replace("\n", "\\n")
    s = s.replace("\v", "\\v")
    s = s.replace("\t", "\\t")
    return f'"{s}"'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Most functions in this section are curried: they consist of a function
# that takes some parameters (e.g., what is to be generated) which itself
# returns a function that actually maps NativeFunction to the code
# to be generated.  This pattern makes it convenient to use map, concatMap
# and similar functional combinators.


def static_dispatch_keys(backends: list[BackendIndex]) -> list[DispatchKey]:
    if len(backends) == 0:
        return []
    else:
        return [backend.dispatch_key for backend in backends] + [
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.CompositeImplicitAutogradNestedTensor,
            DispatchKey.CompositeExplicitAutograd,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        ]


def get_static_dispatch_backend(
    f: NativeFunction, backend_index: BackendIndex
) -> DispatchKey | None:
    if f.structured_delegate is not None or backend_index.has_kernel(f):
        # TODO: for ops with structured_delegate it should check the dispatch table of
        # the out variant instead. For now, these structured ops all have CPU/CUDA kernels
        # so we always dispatch to the `backend`, but this could be wrong when we
        # migrate math/default_backend ops to use structured delegate.
        return backend_index.dispatch_key
    elif f.has_composite_explicit_autograd_kernel:
        return DispatchKey.CompositeExplicitAutograd
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return DispatchKey.CompositeExplicitAutogradNonFunctional
    elif f.has_composite_implicit_autograd_kernel:
        return DispatchKey.CompositeImplicitAutograd
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return DispatchKey.CompositeImplicitAutogradNestedTensor
    return None


def static_dispatch_ops_header(
    f: NativeFunction, backend_index: list[BackendIndex]
) -> str | None:
    if backend_index is None or f.manual_kernel_registration:
        return None

    output = []
    for index in backend_index:
        dispatch_key = get_static_dispatch_backend(f, index)
        if dispatch_key is not None:
            output.append(
                f"#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>"
            )
    return "\n".join(output)


def static_dispatch_extra_headers(backends: list[BackendIndex]) -> list[str]:
    return [
        f"#include <ATen/{dispatch_key}Functions.h>"
        for dispatch_key in static_dispatch_keys(backends)
    ]


# Translates arguments of `sig` to CppSignature bindings.
# Note that we have a special case for `memory_format` argument and this case is not covered by
# tools.codegen.api.translate() yet as its application is limited to static dispatch.
def translate_args(
    sig: CppSignature | DispatcherSignature,
    cpp_sig: CppSignature,
) -> str:
    # Adds SpecialArgName.possibly_redundant_memory_format NamedCType for memory_format bindings
    def add_spl_memory_format_binding(input_bindings: list[Binding]) -> list[Binding]:
        output_bindings: list[Binding] = []
        for binding in input_bindings:
            if binding.name == "memory_format":
                spl_mem_format_binding = Binding(
                    nctype=NamedCType(
                        SpecialArgName.possibly_redundant_memory_format,
                        binding.nctype.type,
                    ),
                    name=binding.name,
                    default=binding.default,
                    argument=binding.argument,
                )
                output_bindings.append(spl_mem_format_binding)
            else:
                output_bindings.append(binding)
        return output_bindings

    src_bindings = list(sig.arguments())
    goal_bindings = list(cpp_sig.arguments())
    # When last argument of CPP signature has SpecialArgName.possibly_redundant_memory_format NCType,
    # get memory_format bindings of dispatcher signature to have the same NCType as well
    for arg in goal_bindings:
        if arg.nctype.name == SpecialArgName.possibly_redundant_memory_format:
            src_bindings = add_spl_memory_format_binding(src_bindings)
            break
    exprs = translate(src_bindings, goal_bindings)
    return ", ".join(a.expr for a in exprs)


def generate_static_dispatch_backend_call(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_index: BackendIndex,
) -> str:
    cpp_sig = gen_static_dispatch_backend_call_signature(f)
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    backend_metadata = backend_index.get_kernel(f)
    kernel_ns = (
        backend_metadata.cpp_namespace
        if backend_metadata and backend_metadata.cpp_namespace
        else DEFAULT_KERNEL_NAMESPACE
    )
    ns = kernel_ns.replace("::native", "")
    return f"return {ns}::{backend_index.dispatch_key.lower()}::{name}({exprs});"


def generate_static_dispatch_fallback_call(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_indices: list[BackendIndex],
) -> str:
    cpp_sigs = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    if cpp_sig is None:
        raise AssertionError("Expected cpp_sig to be non-None")
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    ns = DEFAULT_KERNEL_NAMESPACE.replace("::native", "")
    if f.has_composite_explicit_autograd_kernel:
        return f"return {ns}::{DispatchKey.CompositeExplicitAutograd.lower()}::{name}({exprs});"
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return f"return {ns}::{DispatchKey.CompositeExplicitAutogradNonFunctional.lower()}::{name}({exprs});"
    elif f.has_composite_implicit_autograd_kernel:
        return f"return {ns}::{DispatchKey.CompositeImplicitAutograd.lower()}::{name}({exprs});"
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return f"return {ns}::{DispatchKey.CompositeImplicitAutogradNestedTensor.lower()}::{name}({exprs});"
    else:
        return f"""TORCH_CHECK(false, "Static dispatch does not support {name} for\
{", ".join([str(index.dispatch_key) for index in backend_indices])} ");"""


def static_dispatch(
    sig: CppSignature | DispatcherSignature,
    f: NativeFunction,
    backend_indices: list[BackendIndex],
) -> str:
    """
    For a given `NativeFunction`, find out the corresponding backend and dispatch to it. If more than one
    backends exist, fallback to static dispatch by determining dispatch key from inputs.
    Arguments:
        sig: A CppSignature or DispatcherSignature for this native function we want to use.
        f: NativeFunction to generate static dispatch.
        backend_indices: All available backends.
    Return:
        C++ code to call backend-specific functions, e.g., "return at::cpu::add(self, other, scale);"
    """
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ""

    keys = [
        b
        for b in backend_indices
        if b.has_kernel(f)
        or (
            f.structured_delegate is not None
            and b.dispatch_key in STRUCTURED_DISPATCH_KEYS
        )
    ]
    if len(keys) == 1:
        return generate_static_dispatch_backend_call(sig, f, keys[0])
    elif len(keys) == 0:
        return generate_static_dispatch_fallback_call(sig, f, backend_indices)

    native_tensor_args = [
        a.name
        for a in sig.arguments()
        if isinstance(a.argument, SelfArgument)
        or isinstance(a.argument, Argument)
        and a.argument.type.is_tensor_like()
    ]
    tensor_args = ", ".join(native_tensor_args)
    tensor_opts = f.func.arguments.tensor_options

    stmts = []
    subexprs: list[str] = []
    if tensor_opts is not None:
        subexprs.append(
            "DispatchKeySet(c10::computeDispatchKey(dtype, layout, device))"
        )
    if tensor_args != "":
        subexprs.append(f"c10::detail::multi_dispatch_key_set({tensor_args})")
    stmts.append(f"""DispatchKeySet _dk_set = {" | ".join(subexprs)};""")
    stmts.append("DispatchKey _dk = c10::highestPriorityBackendTypeId(_dk_set);")

    dispatch_code = []
    for index in keys:
        dispatch_code.append(f"""case DispatchKey::{index.dispatch_key}:""")
        dispatch_code.append(
            f"""\t{generate_static_dispatch_backend_call(sig, f, index)};"""
        )

    fallback = generate_static_dispatch_fallback_call(sig, f, backend_indices)
    connector = "\n\t\t"

    return f"""
    {connector.join(stmts)}
    switch (_dk) {{
        {connector.join(dispatch_code)}
        default:
            {fallback}
    }}
    """


# Generates RegisterSchema.cpp.  Depending on the selector, either
# all schemas are registered, or only some are (in the case of
# selective build)
@dataclass(frozen=True)
class RegisterSchema:
    selector: SelectiveBuilder
    known_tags: dict[str, int] = field(default_factory=dict)

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if not self.selector.is_native_function_selected(f):
            return None
        tags = "{" + ", ".join(f"at::Tag::{tag}" for tag in sorted(f.tags)) + "}"
        if tags == "{}":
            return f"m.def({cpp_string(str(f.func))}, {{}});\n"
        maybe_tags = ""
        if tags not in self.known_tags:
            idx = len(self.known_tags)
            self.known_tags[tags] = idx
            maybe_tags = f"const std::vector<at::Tag> tags_{idx} = {tags};\n"
        return f"{maybe_tags}m.def({cpp_string(str(f.func))}, tags_{self.known_tags[tags]});\n"


# Generates Operators.h and Operators.cpp.
# These provide macros that, given an operator and overload name, allow users
# to access an "un-overloaded" function version of the operator. This
# is useful for extension writers who want to (1) want to decltype the operator
# and (2) don't want to worry about method-only operators.
@dataclass(frozen=True)
class ComputeOperators:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: list[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig = DispatcherSignature.from_schema(f.func)
        name = f.func.name.unambiguous_name()

        if self.target is Target.DECLARATION:
            # Note [The ATen Operators API]
            # The ATen Operators API lives in the at::_ops namespace, and contains compile-time
            # metadata about each operator + entry points into the Dispatcher.
            # The C++ function, method, and redispatch API's are all implemented as wrappers
            # into various bits of the structs defined here.
            #
            # Important characteristics about the Operators API:
            # (1) It follows the Dispatcher API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Overload names are disambiguated.
            #     This is helpful for pytorch extenders who would like to decltype() an aten operator,
            #     that has overloads, e.g. decltype(at::_ops::mul_Tensor::call)
            # (3) No argument defaulting is allowed.
            #     This is more of an implementation detail to avoid #include cycles,
            #     since TensorBody.h (which defines the Tensor class) needs to include this file.
            # (4) manual_cpp_bindings and faithful names are not included in the API.
            #     This applies to stuff like __dispatch__is_complex(), and add_outf().
            #     These aren't "real aten ops", they're just additional functions provided by the C++ API.
            #     They're implemented as wrappers in Functions.h that call into the actual operators
            #     defined here, i.e. at::_ops::is_complex::call() and at::_ops::add_out::call().
            #     This means that ATEN_OP(is_complex) will not fastpath, and will go through the dispatcher.
            return f"""
struct TORCH_API {name} {{
  using schema = {sig.type()};
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::{f.func.name.name}";
  static constexpr const char* overload_name = "{f.func.name.overload_name}";
  static constexpr const char* schema_str = {cpp_string(str(f.func))};
  static {sig.defn(name="call", is_redispatching_fn=False)};
  static {sig.defn(name="redispatch", is_redispatching_fn=True)};
}};"""

        elif self.target is Target.DEFINITION:
            defns = f"""
// aten::{f.func}
static C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow({name}::name, {name}::overload_name)
      .typed<{name}::schema>();
}}
"""
            for is_redispatching_fn in [False, True]:
                if is_redispatching_fn:
                    dispatcher_exprs_str = ", ".join(
                        ["dispatchKeySet"] + [a.name for a in sig.arguments()]
                    )
                    method_base = "redispatch"
                else:
                    dispatcher_exprs_str = ", ".join([a.name for a in sig.arguments()])
                    method_base = "call"

                dispatcher_call = method_base
                method_name = f"{name}::{method_base}"

                fn_body = f"""
    static auto op = create_{name}_typed_handle();
    return op.{dispatcher_call}({dispatcher_exprs_str});"""

                if (
                    not is_redispatching_fn
                    and len(self.static_dispatch_backend_indices) > 0
                ):
                    # call() should go through static dispatch
                    fn_body = static_dispatch(
                        sig, f, backend_indices=self.static_dispatch_backend_indices
                    )
                defns += f"""
// aten::{f.func}
{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{
    {fn_body}
}}
"""
            return defns
        else:
            assert_never(self.target)


# Generates Functions.h, which provides the functional public C++ API,
# and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeFunction:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )
        has_symint = f.func.has_symint()

        result = ""
        for sig in sig_group.signatures():
            # See Note [The ATen Operators API]
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ", ".join([e.expr for e in exprs])

            if sig.symint:
                intlike_t = "c10::SymInt"
            else:
                intlike_t = "int64_t"

            if Variant.function in f.variants:
                result += f"""
// aten::{f.func}
inline {sig.decl()} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}"""

            # The template function can be used from template situations
            # where you want to switch between the symint or not version
            # depending on a template argument
            #
            # NB: we ALWAYS generate this even for methods.  But we put it in
            # this header so it can take advantage of per-op headers
            if has_symint:
                result += f"""
namespace symint {{
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, {intlike_t}>>>
  {sig.decl(suppress_symint_suffix=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
  }}
}}
"""
        return result


# Generates TensorBody.h. This file provides the object-oriented (method-based)
# public C++ API, and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: list[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if Variant.method not in f.variants:
            return None

        if f.func.is_out_fn():
            raise AssertionError(f"Method variant cannot be an out function: {f.func}")
        if f.func.arguments.self_arg is None:
            raise AssertionError(f"Method variant must have self_arg: {f.func}")

        sig_group = CppSignatureGroup.from_native_function(
            f, method=True, fallback_binding=f.manual_cpp_binding
        )

        if self.target is Target.DECLARATION:
            result = ""
            for sig in sig_group.signatures():
                result += f"{sig.decl()} const;\n"
            return result

        if self.target is not Target.DEFINITION:
            assert_never(self.target)

        result = ""

        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ", ".join([e.expr for e in exprs])

            result += f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""

        return result


# Generates RedispatchFunctions.h.
# This is similar to the C++ API defined in Functions.h, but provides access
# to the dispatcher's redispatch API.
@dataclass(frozen=True)
class ComputeRedispatchFunction:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        # We unconditionally generate function variants of the redispatch API.
        # This is mainly because we can namespace functions separately, but not methods,
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )

        result = ""
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ", ".join(["dispatchKeySet"] + [a.expr for a in exprs])

            result += f"""
// aten::{f.func}
inline {sig.decl(is_redispatching_fn=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::redispatch({exprs_str});
}}
"""

        return result


# Generates ATenOpList.cpp, a runtime accessible list of all aten
# operators.
# TODO: This was historically used to help some JIT interop code
# figure out whether or not to treat aten namespace'd operators
# one way or another, we should reevaluate if this is actually needed.
@with_native_function
def compute_aten_op(f: NativeFunction) -> str:
    return f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},'


# Generates MetaFunctions.h
def compute_meta_function_declaration(g: NativeFunctionsGroup) -> str | None:
    if not g.structured:
        return None
    with native_function_manager(g.out):
        name = meta.name(g)
        args = structured.meta_arguments(g)
        args_str = ", ".join(a.decl() for a in args)
        parent_class = g.out.structured_inherits
        if parent_class is None:
            parent_class = "at::impl::MetaBase"
        meta_return = "void"
        precomputed = g.out.precomputed if g.structured else None

        if precomputed:
            # Generate the template declaration with one bool parameter for each
            # precomputed element. Each parameter is true if the corresponding (in
            # terms of position) precomputed element has been set.
            precomputed_values = [*precomputed.replace.values(), precomputed.add]
            precomputed_elements = [
                elem for replace_list in precomputed_values for elem in replace_list
            ]
            precomputed_template_parameters = [
                elem.name.upper() for elem in precomputed_elements
            ]
            precomputed_template_params_str = ", ".join(
                f"bool {param} = false" for param in precomputed_template_parameters
            )
            precompute_template_decl = f"template <{precomputed_template_params_str}>"

            # Generate a string containing declarations of all precomputed elements.
            precomputed_elements_with_cpp_types = [
                structured.argument_type(elem, binds=elem.name)
                for elem in precomputed_elements
            ]

            precomputed_elements_decl = ";\n".join(
                f"{elem.cpp_type(strip_ref=True)} {elem.name}"
                for elem in precomputed_elements_with_cpp_types
            )

            # Generate "setter" methods for each precomputed element. Each method will return
            # a new instance of precompute_out with the template parameter that corresponds to
            # the member set by the method to true (to indicate that it has been set).
            setter_methods = []
            for i, elem in enumerate(precomputed_elements):
                # Generate the signature. The return type will be the same
                # as the type of `this` but with the template parameter
                # corresponding to the element set by this method set to true.
                # The assert generated below will ensure that this template
                # parameter is false on the type of `this`.
                return_ty_templates = ", ".join(
                    precomputed_template_parameters[:i]
                    + ["true"]
                    + precomputed_template_parameters[i + 1 :]
                )
                return_ty = f"precompute_out<{return_ty_templates}>"
                elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(
                    strip_ref=True
                )
                signature = f"{return_ty} set_{elem.name}({elem_cpp_ty} value)"

                # Generate an assert which checks that the
                # template parameter corresponding to the precomputed
                # element that is set by this method is false on the
                # class corresponding to the object that `this` points to.
                # This ensures that each element can be set only once.
                assert_msg = f'"{elem.name} already set"'
                assert_stmt = f"static_assert({precomputed_template_parameters[i]} == false, {assert_msg});"

                # Generate the new object construction block. All state
                # except the element that this method sets is copied from the
                # object that `this` points to. The value for the element that
                # the method sets is taken from a method parameter.
                construction_stmts = []
                construction_stmts.append(f"{return_ty} ret;")

                for j, elem in enumerate(precomputed_elements):
                    if i == j:
                        construction_stmts.append(f"ret.{elem.name} = value;")
                    else:
                        construction_stmts.append(
                            f"ret.{elem.name} = this->{elem.name};"
                        )

                construction_stmts.append("return ret;")
                construction_block = "\n".join(construction_stmts)

                setter_methods.append(
                    f"""
                    {signature} {{
                        {assert_stmt}
                        {construction_block}
                    }}
                """
                )
            setter_methods_decl = "\n".join(setter_methods)

            # Meta should return an instance of the struct containing the precomputed elements.
            meta_return_template_params = ", ".join(
                ["true"] * len(precomputed_template_parameters)
            )
            # This typedef (actually a using statement) is needed so that TORCH_META_FUNC can reuse the return
            # type (which has a variable number of template parameters).
            meta_return_typedef = f"using meta_return_ty = precompute_out <{meta_return_template_params}>;"
            meta_return = "meta_return_ty"
            precomputed_decl = f"""
                {precompute_template_decl}
                struct TORCH_API precompute_out {{
                    {setter_methods_decl}
                    {precomputed_elements_decl};
            }};"""
        else:
            meta_return_typedef = ""
            precomputed_decl = ""

        return f"""\
struct TORCH_API structured_{name} : public {parent_class} {{
    {precomputed_decl}
    {meta_return_typedef}
    {meta_return} meta({args_str});
}};
"""


def needs_backend_select(f: NativeFunction, selector: SelectiveBuilder) -> bool:
    name = str(f.func.name.name)
    if name.endswith("_like") or name.startswith("new_"):
        return False
    if f.func.arguments.tensor_options is None:
        return False
    return selector.is_native_function_selected(f)


# Generates RegisterBackendSelect.cpp, a series of kernels which provide
# specialized computation of dispatch key for operator signatures which cannot
# be easily done automatically using templating.
@dataclass(frozen=True)
class ComputeBackendSelect:
    target: Literal[Target.DEFINITION, Target.REGISTRATION]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if not needs_backend_select(f, self.selector):
            return None

        name = native.name(f.func)
        # BackendSelect can go to Meta, so it must preserve symints
        native_sig = NativeSignature(f.func, symint=True)

        native_tensor_args = [
            a
            for a in native_sig.arguments()
            if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
        ]

        dispatcher_sig = DispatcherSignature.from_schema(f.func)

        sig: NativeSignature | DispatcherSignature
        sig = dispatcher_sig
        dispatcher_exprs = dispatcher_sig.exprs()
        dispatch_key = "c10::computeDispatchKey(dtype, layout, device)"

        if self.target is Target.DEFINITION:
            # I don't think there's actually a good reason to generate
            # these two cases differently
            # The first case could probably be improved though- it calls computeDispatchKeySet(),
            # which looks at TLS dispatch keys- there should not be any by the time we reach backend select.
            if native_tensor_args:
                if not f.func.arguments.has_tensor_arg():
                    raise AssertionError(
                        f"Expected function to have tensor args: {f.func}"
                    )
                tensor_args = ", ".join(a.name for a in native_tensor_args)
                compute_dk = f"""\
DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});
DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
DispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);"""
            else:
                if f.func.arguments.has_tensor_arg():
                    raise AssertionError(
                        f"Expected function to not have tensor args: {f.func}"
                    )
                compute_dk = (
                    f"DispatchKeySet _dk = c10::DispatchKeySet({dispatch_key});"
                )
            return f"""\
// aten::{f.func}
C10_ALWAYS_INLINE
{sig.defn(name)} {{
  {compute_dk}
  return at::_ops::{f.func.name.unambiguous_name()}::redispatch(
      _dk, {", ".join(a.expr for a in dispatcher_exprs)});
}}
"""
        elif self.target is Target.REGISTRATION:
            return f"""m.impl("aten::{f.func.name}", TORCH_FN({name}));"""
        else:
            assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       YAML CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def format_yaml(data: object) -> str:
    # Ignore alias in Dumper
    YamlDumper.ignore_aliases = lambda self, data: True  # type: ignore[assignment]

    # Support serializing OrderedDict
    def dict_representer(dumper: Any, data: Any) -> Any:
        return dumper.represent_dict(data.items())

    YamlDumper.add_representer(OrderedDict, dict_representer)  # type: ignore[no-untyped-call]
    # Some yaml parsers (e.g. Haskell's) don't understand line breaks.
    # width=1e9 turns off optional line breaks and improves
    # the portability of the outputted yaml.
    return yaml.dump(data, default_flow_style=False, Dumper=YamlDumper, width=1e9)  # type: ignore[no-any-return, call-overload]


# For some reason, some defaults we write to YAML are written as native
# YAML objects, rather than doing them uniformly as strings.  This
# function detects those cases and converts them into native Python
# objects.
def pythonify_default(s: str) -> object:
    if s == "true":
        return True
    elif s == "false":
        return False

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


# What is a dynamic type?  Over time, the semantic meaning of
# dynamic type has degraded to meaninglessness (in the old days,
# it captured dtype-ness of types, but that has gone away with
# the removal of TH).  These days, it's mostly the same thing as
# the C++ API argument type, except that Tensor and Tensor?
# arguments simply present as Tensor.
#
# TODO: Get rid of dynamic_type, after getting tools/autograd
# to use the new codegen framework
def dynamic_type(t: Type) -> str:
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
    # Note we don't use t.is_tensor_like() here because it would
    # also include Tensor[]
    if str(t) == "Tensor":
        return "at::Tensor"
    # This is a legacy concept, so never report SymInt
    return cpp.argumenttype_type(
        t, mutable=False, binds="__placeholder__", symint=False
    ).cpp_type()


def compute_method_of_yaml(variants: set[Variant]) -> list[str]:
    # This is written out explicitly to ensure that Tensor and
    # namespace are put into the list in the right order
    method_of = ["Type"]
    if Variant.method in variants:
        method_of.append("Tensor")
    if Variant.function in variants:
        method_of.append("namespace")
    return method_of


def compute_returns_yaml(
    f: NativeFunction,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    # Note [name and field_name]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # To understand name_to_field_name, we must first talk about this
    # schema:
    #
    #   lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
    #
    # There is something very odd about this schema: it is an out
    # variant of the function (that is to say, it will convert into
    # at::lstsq_out() in the C++ API), but the names of the output
    # return arguments don't match the keyword argument names of
    # the inputs.  It TURNS OUT that in this situation, the historical
    # Declarations.yaml we want to output is this (abbreviated to
    # only show relevant fields):
    #
    #   arguments:
    #     ...
    #   - field_name: solution
    #     name: X
    #   - field_name: QR
    #     name: qr
    #     ...
    #
    #   returns:
    #   - field_name: solution
    #     name: X
    #   - field_name: QR
    #     name: qr
    #
    # The name of the return fields is stored in 'field_name', and the
    # name of the arguments is stored in 'name'.  So when we process
    # arguments, we need a way to get at the corresponding return.  At
    # the moment, this is most conveniently done by constructing a
    # mapping from name (the argument concept) to field_name (the
    # return concept) while processing return arguments, since we don't
    # directly maintain this correspondence in the modeling of function
    # schema itself.
    #
    # See also https://github.com/pytorch/pytorch/issues/43114
    name_to_field_name: dict[str, str] = {}

    # Compute the returns field of the YAML entry
    names = cpp.return_names(f)
    returns = []
    for i, (r, name) in enumerate(zip(f.func.returns, names)):
        ret = {
            "dynamic_type": dynamic_type(r.type),
            "name": name,
            # legacy, report ints
            "type": cpp.return_type(r, symint=False).cpp_type(),
        }

        if r.name:
            # See Note [name and field_name]
            ret["field_name"] = r.name
            if f.func.is_out_fn():
                name_to_field_name[f.func.arguments.out[i].name] = r.name

        returns.append(ret)

    return returns, name_to_field_name


# arguments in yaml roughly corresponds to the public C++ API
def compute_cpp_argument_yaml(
    cpp_a: Binding,
    *,
    schema_order: bool,
    kwarg_only_set: set[str],
    out_arg_set: set[str],
    name_to_field_name: dict[str, str],
) -> object:
    if isinstance(cpp_a.argument, TensorOptionsArguments):
        arg: dict[str, object] = {
            "annotation": None,
            "dynamic_type": "at::TensorOptions",
            "is_nullable": False,
            "name": cpp_a.name,
            "type": cpp_a.type,
            "kwarg_only": True,
        }
        if cpp_a.default is not None:
            arg["default"] = cpp_a.default
        return arg
    elif isinstance(cpp_a.argument, SelfArgument):
        raise AssertionError
    elif isinstance(cpp_a.argument, Argument):
        return compute_argument_yaml(
            cpp_a.argument,
            schema_order=schema_order,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )


def compute_argument_yaml(
    a: Argument,
    *,
    schema_order: bool,
    kwarg_only_set: set[str],
    out_arg_set: set[str],
    name_to_field_name: dict[str, str],
) -> object:
    arg: dict[str, object] = {
        "annotation": str(a.annotation) if a.annotation else None,
        "dynamic_type": dynamic_type(a.type),
        "is_nullable": a.type.is_nullable(),
        "name": a.name,
        # legacy, report ints
        "type": cpp.argument_type(a, binds="__placeholder__", symint=False).cpp_type(),
    }
    if a.default is not None:
        arg["default"] = pythonify_default(
            cpp.default_expr(a.default, a.type, symint=False)
        )
    if a.name in kwarg_only_set:
        arg["kwarg_only"] = True
    if a.name in out_arg_set:
        arg["output"] = True
        arg["allocate"] = True
        # See Note [name and field_name]
        if a.name in name_to_field_name:
            arg["field_name"] = name_to_field_name[a.name]
    # Historically, booleans don't get their size recorded, because it
    # is already built into the cpp type (e.g., std::array<bool, 4>)
    l = a.type.is_list_like()
    if l is not None and l.size is not None and str(l.elem) != "bool":
        arg["size"] = l.size
    return arg


@with_native_function
def compute_declaration_yaml(f: NativeFunction) -> object:
    returns, name_to_field_name = compute_returns_yaml(f)

    # These sets are used to conveniently test if an argument is a
    # kwarg-only or out argument
    kwarg_only_set = {a.name for a in f.func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in f.func.arguments.out}

    sig_group = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    cpp_args = sig_group.signature.arguments()
    arguments = [
        compute_cpp_argument_yaml(
            cpp_a,
            schema_order=False,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for cpp_a in cpp_args
    ]

    schema_order_jit_arguments = list(f.func.schema_order_arguments())

    schema_order_arguments = [
        compute_argument_yaml(
            a,
            schema_order=True,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for a in schema_order_jit_arguments
    ]

    cpp_schema_order_types = [
        # NB: method here doesn't matter
        r.type
        for a in schema_order_jit_arguments
        for r in cpp.argument(
            a,
            method=False,
            cpp_no_default_args=set(),
            faithful=False,
            symint=False,
            has_tensor_options=False,
        )
    ]

    # legacy, report ints
    cpp_returns = cpp.returns_type(f.func.returns, symint=False).cpp_type()
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"

    is_factory_method = (
        any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args)
        and Variant.method not in f.variants
    )

    return OrderedDict(
        [
            ("name", cpp.name(f.func)),
            ("operator_name", str(f.func.name.name)),
            ("overload_name", str(f.func.name.overload_name)),
            ("manual_kernel_registration", f.manual_kernel_registration),
            (
                "category_override",
                f.category_override if f.category_override is not None else "",
            ),
            ("schema_string", f"aten::{f.func}"),
            ("arguments", arguments),
            ("schema_order_cpp_signature", schema_order_cpp_signature),
            ("schema_order_arguments", schema_order_arguments),
            ("method_of", compute_method_of_yaml(f.variants)),
            ("mode", "native"),
            ("python_module", "" if f.python_module is None else f.python_module),
            ("returns", returns),
            ("inplace", f.func.name.name.inplace),
            ("is_factory_method", is_factory_method),
            ("abstract", f.is_abstract),
            ("device_guard", f.device_guard),
            ("with_gil", False),
            ("deprecated", False),
            ("has_math_kernel", f.has_composite_implicit_autograd_kernel),
        ]
    )


# See Note [Auto generated composite kernels]
def has_autogenerated_composite_kernel(f: NativeFunction) -> bool:
    return (f.structured or f.structured_delegate is not None) and (
        f.func.kind() == SchemaKind.functional or f.func.kind() == SchemaKind.inplace
    )


@with_native_function_and_indices
def compute_registration_declarations(
    f: NativeFunction, backend_indices: dict[DispatchKey, BackendIndex]
) -> str:
    name = dispatcher.name(f.func)
    returns_type = dispatcher.returns_type(f.func.returns).cpp_type()
    args = dispatcher.arguments(f.func)
    args_str = ", ".join(a.no_default().decl() for a in args)
    comment_data: dict[str, str] = {
        "schema": f"aten::{f.func}",
        # TODO: What exactly is the semantics of the 'dispatch' field?
        "dispatch": str(
            {k for k, v in backend_indices.items() if v.has_kernel(f)}
            != {DispatchKey.CompositeImplicitAutograd}
            and {k for k, v in backend_indices.items() if v.has_kernel(f)}
            != {
                DispatchKey.CompositeImplicitAutograd,
                DispatchKey.CompositeImplicitAutogradNestedTensor,
            }
        ),
        "default": str(f.has_composite_kernel or has_autogenerated_composite_kernel(f)),
    }
    return f"""{returns_type} {name}({args_str}); // {json.dumps(comment_data)}
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def get_custom_build_selector(
    provided_op_registration_allowlist: list[str] | None,
    op_selection_yaml_path: str | None,
) -> SelectiveBuilder:
    if (
        provided_op_registration_allowlist is not None
        and op_selection_yaml_path is not None
    ):
        raise AssertionError(
            "Both provided_op_registration_allowlist and op_selection_yaml_path "
            "can NOT be provided at the same time."
        )

    op_registration_allowlist: set[str] | None = None
    if provided_op_registration_allowlist is not None:
        op_registration_allowlist = set(provided_op_registration_allowlist)

    if op_registration_allowlist is not None:
        selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
            op_registration_allowlist,
            True,
            False,
        )
    elif op_selection_yaml_path is not None:
        selector = SelectiveBuilder.from_yaml_path(op_selection_yaml_path)
    else:
        selector = SelectiveBuilder.get_nop_selector()

    return selector


def get_grouped_by_view_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Sequence[NativeFunction | NativeFunctionsViewGroup]:
    def maybe_create_view_group(
        d: dict[ViewSchemaKind | SchemaKind, NativeFunction],
    ) -> list[NativeFunction | NativeFunctionsViewGroup]:
        funcs: list[NativeFunction | NativeFunctionsViewGroup] = []
        if ViewSchemaKind.aliasing in d:
            view = d.pop(ViewSchemaKind.aliasing)
            view_inplace = d.pop(ViewSchemaKind.aliasing_inplace, None)
            view_copy = d.pop(SchemaKind.functional, None)

            funcs.append(
                NativeFunctionsViewGroup(
                    view=view,
                    view_copy=view_copy,
                    view_inplace=view_inplace,
                )
            )
        # Take the remaining functions that weren't part of the view group
        # and emit them separately
        funcs.extend(d.values())
        return funcs

    grouped_by_views: dict[
        FunctionSchema, dict[SchemaKind | ViewSchemaKind, NativeFunction]
    ] = defaultdict(dict)
    for f in native_functions:
        schema = f.func.view_signature()
        view_kind: ViewSchemaKind = f.view_schema_kind
        # We need to group up ops relevant to the same "view", consisting of:
        # view op (ViewSchemaKind.aliasing)
        # view_inplace op (ViewSchemaKind.aliasing_inplace)
        # view_copy op (SchemaKind.functional)
        if view_kind == ViewSchemaKind.non_aliasing:
            kind = f.func.kind()
            if kind in grouped_by_views[schema]:
                raise AssertionError(
                    f"Duplicate schema kind {kind} in {grouped_by_views[schema].keys()}"
                )
            grouped_by_views[schema][kind] = f
        else:
            if view_kind in grouped_by_views[schema]:
                raise AssertionError(
                    f"{view_kind} already in {grouped_by_views[schema].keys()}"
                )
            grouped_by_views[schema][view_kind] = f

    return list(concatMap(maybe_create_view_group, grouped_by_views.values()))


def get_grouped_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Sequence[NativeFunction | NativeFunctionsGroup]:
    def flatten_pre_group(
        d: dict[SchemaKind, NativeFunction],
    ) -> Sequence[NativeFunction | NativeFunctionsGroup]:
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            # Invariant: any NativeFunctions that are code-generated
            # should have been grouped into NativeFunctionsGroup objects
            if any("generated" in f.tags for f in d.values()):
                raise AssertionError(
                    "Generated NativeFunctions should have been grouped into "
                    f"NativeFunctionsGroup objects: {list(d.values())}"
                )
            return list(d.values())
        else:
            return [r]

    # TODO: how come ValuesView isn't a Sequence lol
    pre_grouped_native_functions = pre_group_native_functions(native_functions)
    return list(
        concatMap(flatten_pre_group, list(pre_grouped_native_functions.values()))
    )


def get_ns_grouped_kernels(
    *,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: dict[DispatchKey, BackendIndex],
    native_function_decl_gen: Callable[
        [NativeFunctionsGroup | NativeFunction, BackendIndex], list[str]
    ] = dest.compute_native_function_declaration,
) -> dict[str, list[str]]:
    ns_grouped_kernels: dict[str, list[str]] = defaultdict(list)
    for f in grouped_native_functions:
        native_function_namespaces = set()
        dispatch_keys = set()
        for dispatch_key, backend_idx in backend_indices.items():
            backend_metadata = backend_idx.get_kernel(f)
            if backend_metadata:
                namespace = backend_metadata.cpp_namespace
                dispatch_keys.add(dispatch_key)
                native_function_namespaces.add(namespace)
            else:
                namespace = DEFAULT_KERNEL_NAMESPACE
            if len(native_function_namespaces) > 1:
                raise AssertionError(
                    f"Codegen only supports one namespace per operator, "
                    f"got {native_function_namespaces} from {dispatch_keys}"
                )
            ns_grouped_kernels[namespace].extend(
                native_function_decl_gen(f, backend_idx)
            )
    return ns_grouped_kernels


def get_native_function_declarations_from_ns_grouped_kernels(
    *,
    ns_grouped_kernels: dict[str, list[str]],
) -> list[str]:
    declarations: list[str] = []
    newline = "\n"
    for namespace, kernels in ns_grouped_kernels.items():
        ns_helper = NamespaceHelper(
            namespace_str=namespace,
            entity_name="",
            max_level=4,
        )
        # Convert to a set first to remove duplicate kernel names. Backends are
        # allowed to repeat kernel names; only generate the declaration once!
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(
            f"""
{ns_helper.prologue}
{newline.join(ordered_kernels)}
{ns_helper.epilogue}
        """.split(newline)
        )
    return declarations


# Return native function declarations grouped by their namespaces.
def get_native_function_declarations(
    *,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: dict[DispatchKey, BackendIndex],
    native_function_decl_gen: Callable[
        [NativeFunctionsGroup | NativeFunction, BackendIndex], list[str]
    ] = dest.compute_native_function_declaration,
) -> list[str]:
    """
    Generate kernel declarations, in `NativeFunction(s).h`.
    :param grouped_native_functions: a sequence of `NativeFunction` or `NativeFunctionGroup`.
    :param backend_indices: kernel collections grouped by dispatch key.
    :param native_function_decl_gen: callable to generate kernel declaration for each `NativeFunction`.
    :return: a list of string, from the string with all declarations, grouped by namespaces, split by newline.
    """

    ns_grouped_kernels = get_ns_grouped_kernels(
        grouped_native_functions=grouped_native_functions,
        backend_indices=backend_indices,
        native_function_decl_gen=native_function_decl_gen,
    )
    return get_native_function_declarations_from_ns_grouped_kernels(
        ns_grouped_kernels=ns_grouped_kernels
    )


def get_kernel_namespace(
    *, f: NativeFunction | NativeFunctionsGroup, backend_idx: BackendIndex
) -> str:
    backend_metadata = backend_idx.get_kernel(f)
    if backend_metadata and "::native" not in backend_metadata.cpp_namespace:
        func_name = (
            f.func.name if isinstance(f, NativeFunction) else f.functional.func.name
        )
        raise AssertionError(
            f"The kernel for function {func_name} "
            f"with dispatch key {backend_idx.dispatch_key} "
            f"has a namespace {backend_metadata.cpp_namespace} and it's not ending with '::native'."
        )
    return (
        backend_metadata.cpp_namespace if backend_metadata else DEFAULT_KERNEL_NAMESPACE
    )


# Return native function definitions grouped by dispatch key and custom namespace.
# Used in RegisterDispatchKey.cpp and etc.
def get_native_function_definitions(
    *,
    fm: FileManager,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    rocm: bool,
    symint: bool,
    skip_dispatcher_op_registration: bool,
    gen_dispatch_helpers: bool,
) -> list[str]:
    definitions: list[str] = []
