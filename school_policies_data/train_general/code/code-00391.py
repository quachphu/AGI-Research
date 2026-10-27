    ns_definitions: dict[str, list[str]] = defaultdict(list)
    anonymous_definitions: dict[str, list[str]] = defaultdict(list)
    registrations: dict[str, dict[str, list[str]]] = defaultdict(dict)
    newline = "\n"
    ns_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.NAMESPACED_DEFINITION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    anonymous_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.ANONYMOUS_DEFINITION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    reg_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.REGISTRATION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    for f in grouped_native_functions:
        kernel_namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace(
            "::native", ""
        )

        ns_definitions[kernel_namespace].extend(
            ns_gen(f),
        )
        anonymous_definitions[kernel_namespace].extend(
            anonymous_gen(f),
        )
        namespace = (
            f.namespace if isinstance(f, NativeFunction) else f.functional.namespace
        )
        if namespace not in registrations[kernel_namespace]:
            registrations[kernel_namespace] = defaultdict(list)
        registrations[kernel_namespace][namespace].extend(
            reg_gen(f),
        )

    for kernel_namespace in ns_definitions:
        if len(ns_definitions[kernel_namespace]) == 0:
            continue
        ns_helper = NamespaceHelper(namespace_str=kernel_namespace)
        registration_body = ""
        for namespace in registrations[kernel_namespace]:
            if not registrations[kernel_namespace][namespace]:
                continue
            registration_body += f"""
TORCH_LIBRARY_IMPL({namespace}, {dispatch_key}, m) {{
    {newline.join(registrations[kernel_namespace][namespace])}
}}"""
        definitions.extend(
            fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",
                lambda: {
                    "ns_prologue": ns_helper.prologue,
                    "ns_epilogue": ns_helper.epilogue,
                    "dispatch_anonymous_definitions": anonymous_definitions[
                        kernel_namespace
                    ],
                    "static_init_dispatch_registrations": ""
                    if skip_dispatcher_op_registration
                    else registration_body,
                    "deferred_dispatch_registrations": "",
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_definitions": ns_definitions[kernel_namespace],
                },
            ).split(newline)
        )

    return definitions


# Return native function declarations grouped by dispatch key and custom namespace.
# Used in CPUFunctions_inl.h and etc.
def get_namespaced_declaration(
    *,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    rocm: bool,
    symint: bool,
) -> list[str]:
    declarations: list[str] = []
    ns_grouped_kernels: dict[str, list[str]] = defaultdict(list)
    newline = "\n"
    func = dest.RegisterDispatchKey(
        backend_idx,
        Target.NAMESPACED_DECLARATION,
        selector,
        rocm=rocm,
        class_method_name=None,
        skip_dispatcher_op_registration=False,
        symint=symint,
    )
    for f in grouped_native_functions:
        namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace(
            "native", dispatch_key.lower()
        )

        ns_grouped_kernels[namespace].extend(
            func(f),
        )

    for namespace, kernels in ns_grouped_kernels.items():
        if len(kernels) == 0:
            continue
        ns_helper = NamespaceHelper(
            namespace_str=namespace, entity_name="", max_level=3
        )
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(
            f"""
{ns_helper.prologue}
{newline.join(ordered_kernels)}
{ns_helper.epilogue}
        """.split(newline)
        )
    return declarations


# Return native function schema registration code for aten and other namespaces.
def get_native_function_schema_registrations(
    *,
    native_functions: Sequence[NativeFunction],
    schema_selector: SelectiveBuilder,
) -> tuple[list[str], str]:
    ns_native_functions: dict[str, list[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_native_functions[native_function.namespace].append(native_function)
    schema_registrations = ""
    aten_schema_registrations = []
    custom_namespace = None
    for namespace, funcs in ns_native_functions.items():
        schema_registrations_body = list(
            mapMaybe(RegisterSchema(schema_selector), funcs)
        )
        # NB: we have to separate aten namespace registration from other namespaces,
        # because in the template we hardcoded an operator for ATen already.
        if namespace == "aten":
            aten_schema_registrations = schema_registrations_body
        else:
            custom_namespace = namespace
            tab = "\t"
            # if the namespace is predefined, we should use define a library fragment
            # instead of a new library
            torch_library_macro = (
                "TORCH_LIBRARY_FRAGMENT"
                if namespace in FRAGMENT_NAMESPACES
                else "TORCH_LIBRARY"
            )
            schema_registrations += f"""
{torch_library_macro}({custom_namespace}, m) {{
  {tab.join(schema_registrations_body)}
}};"""
    return (aten_schema_registrations, schema_registrations)


def gen_aggregated_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    static_dispatch_idx: list[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: dict[DispatchKey, BackendIndex],
    cpu_fm: FileManager,
    device_fms: dict[str, FileManager],
    functions_keys: set[DispatchKey],
    dispatch_keys: Sequence[DispatchKey],
    rocm: bool,
) -> None:
    # Buck doesn't support dynamic output files, so we aggregate all operator
    # headers into a single file
    cpu_fm.write(
        "NativeMetaFunctions.h",
        lambda: {
            "NativeMetaFunctions_includes": [],
            "NativeMetaFunctions_declarations": list(
                mapMaybe(compute_meta_function_declaration, structured_native_functions)
            ),
        },
    )
    method_native_functions = [
        fn for fn in native_functions if Variant.method in fn.variants
    ]
    non_method_native_functions = [
        fn for fn in native_functions if fn not in method_native_functions
    ]
    cpu_fm.write(
        "MethodOperators.h",
        lambda: {
            "MethodOperators_includes": [],
            "MethodOperators_declarations": list(
                mapMaybe(
                    ComputeOperators(
                        Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    method_native_functions,
                )
            ),
        },
    )
    cpu_fm.write(
        "Operators.h",
        lambda: {
            "Operators_includes": ["#include <ATen/MethodOperators.h>"],
            "Operators_declarations": list(
                mapMaybe(
                    ComputeOperators(
                        Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    non_method_native_functions,
                )
            ),
        },
    )
    cpu_fm.write(
        "Functions.h",
        lambda: {
            "static_dispatch_extra_headers": static_dispatch_extra_headers(
                static_dispatch_idx
            ),
            "Functions_includes": ["#include <ATen/Operators.h>"],
            "Functions_declarations": list(
                mapMaybe(
                    ComputeFunction(),
                    native_functions,
                )
            ),
        },
    )
    declarations = get_native_function_declarations(
        grouped_native_functions=grouped_native_functions,
        backend_indices=backend_indices,
    )
    cpu_fm.write(
        "NativeFunctions.h",
        lambda: {
            "NativeFunctions_includes": ["#include <ATen/NativeMetaFunctions.h>"],
            "NativeFunctions_declarations": declarations,
        },
    )

    for dispatch_key in dispatch_keys:
        fm = file_manager_from_dispatch_key(dispatch_key, device_fms, cpu_fm)
        if dispatch_key in functions_keys:
            inl_headers = f"#include <ATen/{dispatch_key}Functions_inl.h>"

            fm.write_with_template(
                f"{dispatch_key}Functions.h",
                "DispatchKeyFunctions.h",
                lambda: {
                    "dispatch_key": str(dispatch_key),
                    "inline_headers": inl_headers,
                },
            )
            fm.write_with_template(
                f"{dispatch_key}Functions_inl.h",
                "DispatchKeyFunctions_inl.h",
                lambda: {
                    "DispatchKeyFunctions_inl_includes": [],
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_declarations": get_namespaced_declaration(
                        grouped_native_functions=grouped_native_functions,
                        dispatch_key=dispatch_key,
                        backend_idx=backend_indices[dispatch_key],
                        selector=selector,
                        rocm=rocm,
                        symint=True,
                    ),
                },
            )

        del fm


def gen_per_operator_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    static_dispatch_idx: list[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: dict[DispatchKey, BackendIndex],
    cpu_fm: FileManager,
    device_fms: dict[str, FileManager],
    ops_fm: FileManager,
    functions_keys: set[DispatchKey],
    dispatch_keys: Sequence[DispatchKey],
    rocm: bool,
) -> None:
    # For CMake builds, split operator declarations into separate headers in
    # the ATen/ops folder to split up header dependencies
    functions_by_root_name: dict[str, list[NativeFunction]] = defaultdict(list)
    for fn in native_functions:
        functions_by_root_name[fn.root_name].append(fn)

    grouped_functions_by_root_name: dict[
        str, list[NativeFunction | NativeFunctionsGroup]
    ] = defaultdict(list)
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)

    for name, functions in functions_by_root_name.items():
        ops_fm.write_with_template(
            f"{name}_ops.h",
            "Operator.h",
            lambda: {
                "declarations": list(
                    mapMaybe(
                        ComputeOperators(
                            Target.DECLARATION,
                            static_dispatch_backend_indices=static_dispatch_idx,
                        ),
                        functions,
                    )
                ),
            },
        )

        ops_fm.write_with_template(
            f"{name}.h",
            "Function.h",
            lambda: {
                "static_dispatch_ops_headers": list(
                    mapMaybe(
                        lambda fn: static_dispatch_ops_header(
                            fn, backend_index=static_dispatch_idx
                        ),
                        functions,
                    )
                ),
                "operator_includes": f"#include <ATen/ops/{name}_ops.h>",
                "function_definitions": list(
                    mapMaybe(
                        ComputeFunction(),
                        functions,
                    )
                ),
            },
        )

        grouped_functions = grouped_functions_by_root_name.get(name, [])
        structured_functions = [
            fn
            for fn in grouped_functions
            if isinstance(fn, NativeFunctionsGroup) and fn.structured
        ]
        is_structured = len(structured_functions) > 0

        if is_structured:
            ops_fm.write_with_template(
                f"{name}_meta.h",
                "NativeMetaFunction.h",
                lambda: {
                    "meta_function_declarations": list(
                        mapMaybe(
                            compute_meta_function_declaration, structured_functions
                        )
                    ),
                },
            )
        declarations = get_native_function_declarations(
            grouped_native_functions=grouped_functions,
            backend_indices=backend_indices,
            native_function_decl_gen=dest.compute_native_function_declaration,
        )
        ops_fm.write_with_template(
            f"{name}_native.h",
            "NativeFunction.h",
            lambda: {
                "extra_includes": (
                    f"#include <ATen/ops/{name}_meta.h>" if is_structured else []
                ),
                "native_function_declarations": declarations,
            },
        )

    for category, suffix in [
        ("Functions", ""),
        ("Operators", "_ops"),
        ("NativeMetaFunctions", "_meta"),
        ("NativeFunctions", "_native"),
    ]:
        cpu_fm.write(
            f"{category}.h",
            lambda: {
                f"{category}_includes": [
                    f"#include <ATen/ops/{name}{suffix}.h>"
                    for name in sorted(functions_by_root_name.keys())
                ],
                f"{category}_declarations": [],
            },
        )

    for dispatch_key in dispatch_keys:
        if dispatch_key not in functions_keys:
            continue

        dispatch_namespace = dispatch_key.lower()
        dispatch_names = []

        for name, functions in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(
                concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DECLARATION,
                        selector,
                        rocm=rocm,
                        symint=True,
                        class_method_name=None,
                        skip_dispatcher_op_registration=False,
                    ),
                    grouped_functions,
                )
            )

            if len(declarations) == 0:
                continue

            dispatch_names.append(name)
            ops_fm.write_with_template(
                f"{name}_{dispatch_namespace}_dispatch.h",
                "DispatchKeyFunction.h",
                lambda: {
                    "dispatch_namespace": dispatch_namespace,
                    "dispatch_namespaced_declarations": declarations,
                },
            )

        fm = file_manager_from_dispatch_key(dispatch_key, device_fms, cpu_fm)
        inl_headers = f"#include <ATen/{dispatch_key}Functions_inl.h>"

        fm.write_with_template(
            f"{dispatch_key}Functions.h",
            "DispatchKeyFunctions.h",
            lambda: {
                "dispatch_key": str(dispatch_key),
                "inline_headers": inl_headers,
            },
        )
        fm.write_with_template(
            f"{dispatch_key}Functions_inl.h",
            "DispatchKeyFunctions_inl.h",
            lambda: {
                "dispatch_namespace": dispatch_namespace,
                "DispatchKeyFunctions_inl_includes": [
                    f"#include <ATen/ops/{name}_{dispatch_namespace}_dispatch.h>"
                    for name in sorted(dispatch_names)
                ],
                "dispatch_namespaced_declarations": [],
            },
        )
        del fm

    cpu_fm.write(
        "MethodOperators.h",
        lambda: {
            "MethodOperators_includes": sorted(
                f"#include <ATen/ops/{name}_ops.h>"
                for name, functions in functions_by_root_name.items()
                if any(Variant.method in fn.variants for fn in functions)
            ),
            "MethodOperators_declarations": [],
        },
    )


def gen_headers(
    *,
    native_functions: Sequence[NativeFunction],
    valid_tags: set[str],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    static_dispatch_idx: list[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: dict[DispatchKey, BackendIndex],
    headeronly_fm: FileManager,
    core_fm: FileManager,
    cpu_fm: FileManager,
    device_fms: dict[str, FileManager],
    ops_fm: FileManager,
    dispatch_keys: Sequence[DispatchKey],
    functions_keys: set[DispatchKey],
    rocm: bool,
    per_operator_headers: bool,
) -> None:
    if per_operator_headers:
        gen_per_operator_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            device_fms=device_fms,
            ops_fm=ops_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
        )
    else:
        gen_aggregated_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            device_fms=device_fms,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
        )

    core_fm.write(
        "TensorBody.h",
        lambda: {
            "tensor_method_declarations": list(
                mapMaybe(
                    ComputeTensorMethod(
                        target=Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    native_functions,
                )
            ),
            "tensor_method_definitions": list(
                mapMaybe(
                    ComputeTensorMethod(
                        target=Target.DEFINITION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    native_functions,
                )
            ),
        },
    )

    cpu_fm.write(
        "RedispatchFunctions.h",
        lambda: {
            "function_redispatch_definitions": list(
                mapMaybe(ComputeRedispatchFunction(), native_functions)
            ),
        },
    )

    cpu_fm.write(
        "RegistrationDeclarations.h",
        lambda: {
            "registration_declarations": [
                compute_registration_declarations(f, backend_indices)
                for f in native_functions
            ],
        },
    )

    cpu_fm.write(
        "VmapGeneratedPlumbing.h", lambda: gen_all_vmap_plumbing(native_functions)
    )

    def gen_aten_interned_strings() -> dict[str, str]:
        attrs: set[str] = set()  # All function argument names
        names = set()  # All ATen function names
        for func in native_functions:
            names.add(str(func.func.name.name))
            # Some operators don't have a functional variant but we still create a
            # symbol without the underscore
            names.add(func.func.name.name.base)

            attrs.update(arg.name for arg in func.func.schema_order_arguments())

        # These are keywords in C++, so aren't valid symbol names
        # https://en.cppreference.com/w/cpp/language/operator_alternative
        names -= {
            "and",
            "and_eq",
            "bitand",
            "bitor",
            "compl",
            "not",
            "not_eq",
            "or",
            "or_eq",
            "xor",
            "xor_eq",
        }

        return {
            "aten_symbols": " \\\n".join(
                [f"_(aten, {name})" for name in sorted(names)]
            ),
            "attr_symbols": " \\\n".join(
                [f"_(attr, {name})" for name in sorted(attrs)]
            ),
        }

    core_fm.write("aten_interned_strings.h", gen_aten_interned_strings)

    def gen_tags_enum() -> dict[str, str]:
        return {"enum_of_valid_tags": (",\n".join(sorted(valid_tags)))}

    headeronly_fm.write("enum_tag.h", gen_tags_enum)


def gen_source_files(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    view_groups: Sequence[NativeFunctionsViewGroup],
    selector: SelectiveBuilder,
    static_dispatch_idx: list[BackendIndex],
    backend_indices: dict[DispatchKey, BackendIndex],
    aoti_fm: FileManager,
    core_fm: FileManager,
    cpu_vec_fm: FileManager,
    cpu_fm: FileManager,
    device_fms: dict[str, FileManager],
    dispatch_keys: Sequence[DispatchKey],
    functions_keys: set[DispatchKey],
    rocm: bool,
    force_schema_registration: bool,
    per_operator_headers: bool,
    skip_dispatcher_op_registration: bool,
    update_aoti_c_shim: bool,
    aoti_backends: set[DispatchKey | None],
    extend_aoti_c_shim: bool,
) -> None:
    extra_cuda_headers = """\
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>"""
    if rocm:
        extra_cuda_headers = """\
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>"""

    for dispatch_key in dispatch_keys:
        fm = file_manager_from_dispatch_key(dispatch_key, device_fms, cpu_fm)
        if per_operator_headers:

            def operator_headers() -> list[str]:
                headers = []
                for g in grouped_native_functions:
                    is_registered = False
                    if backend_index.has_kernel(g):
                        is_registered = True
                    # The above has_kernel test on a group will only test for
                    # the existence of out dispatch, because that's how
                    # structured kernels work. But sometimes functions can be
                    # grouped but not be structured, and then you need to check
                    # each individual piece, as they may have manual dispatch
                    # entries.
                    elif isinstance(g, NativeFunctionsGroup) and any(
                        backend_index.has_kernel(fn) for fn in g.functions()
                    ):
                        is_registered = True
                    # TODO: this condition is a bit questionable
                    # (It has to do with the fact that structured kernels get generated kernels
                    # to the Meta + CompositeExplicitAutogradNonFunctional keys).
                    elif g.structured and dispatch_key in (
                        DispatchKey.Meta,
                        DispatchKey.CompositeExplicitAutogradNonFunctional,
                    ):
                        is_registered = True
                    if not is_registered:
                        continue

                    headers.append(f"#include <ATen/ops/{g.root_name}_native.h>")
                    if (
                        dispatch_key
                        == DispatchKey.CompositeExplicitAutogradNonFunctional
                    ):
                        headers.append(f"#include <ATen/ops/{g.root_name}.h>")
                    if dispatch_key in functions_keys:
                        headers.append(
                            f"#include <ATen/ops/{g.root_name}_{dispatch_namespace}_dispatch.h>"
                        )

                return sorted(set(headers))

        else:

            def operator_headers() -> list[str]:
                headers = ["#include <ATen/NativeFunctions.h>"]
                if dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
                    headers.append("#include <ATen/Functions.h>")
                if dispatch_key in functions_keys:
                    headers.append(f"#include <ATen/{dispatch_key!s}Functions.h>")
                return headers

        backend_index = backend_indices[dispatch_key]
        ns_grouped_native_functions = defaultdict(list)
        for grouped_native_function in grouped_native_functions:
            namespace = (
                grouped_native_function.namespace
                if isinstance(grouped_native_function, NativeFunction)
                else grouped_native_function.functional.namespace
            )
            ns_grouped_native_functions[namespace].append(grouped_native_function)

        dispatch_namespace = str(dispatch_key).lower()

        # CompositeImplicitAutogradNestdTensor does not currently user the helpers generated
        # compilation will fail when `-Werror=unused-function` flag is set
        gen_dispatch_helpers: bool = (
            dispatch_key != DispatchKey.CompositeImplicitAutogradNestedTensor
        )

        register_dispatch_key_base_env = {
            "extra_cuda_headers": extra_cuda_headers
            if is_cuda_dispatch_key(dispatch_key)
            else "",
            "external_backend_headers": "",
            "dispatch_headers": dest.gen_registration_headers(
                backend_index, per_operator_headers, rocm
            ),
            # ops_headers *could* be sharded, but doesn't seem necessary?
            "ops_headers": operator_headers(),
            "dispatch_helpers": (
                dest.gen_registration_helpers(backend_index)
                if gen_dispatch_helpers
                else []
            ),
        }

        def register_dispatch_key_env_callable(
            gnf: NativeFunction | NativeFunctionsGroup,
        ) -> dict[str, list[str]]:
            return {
                "dispatch_definitions": get_native_function_definitions(
                    fm=fm,  # noqa: F821
                    grouped_native_functions=[gnf],
                    dispatch_key=dispatch_key,
                    backend_idx=backend_index,
                    selector=selector,
                    rocm=rocm,
                    symint=True,
                    skip_dispatcher_op_registration=skip_dispatcher_op_registration,
                    gen_dispatch_helpers=gen_dispatch_helpers,
                )
            }

        fm.write_sharded_with_template(
            f"Register{dispatch_key}.cpp",
            "RegisterDispatchKey.cpp",
            grouped_native_functions,
            key_fn=lambda x: x.root_name,
            env_callable=register_dispatch_key_env_callable,
            num_shards=4 if dispatch_key == DispatchKey.CPU else 1,
            base_env=register_dispatch_key_base_env,
            sharded_keys={"dispatch_definitions"},
        )

        for g in structured_native_functions:
            if not g.out.ufunc_inner_loop or not is_ufunc_dispatch_key(dispatch_key):
                continue
            name = g.functional.func.name.name
            if dispatch_key is DispatchKey.CPU:
                if fm is not cpu_fm:
                    raise AssertionError("Expected fm to be cpu_fm for DispatchKey.CPU")
                fm.write_with_template(
                    f"UfuncCPU_{name}.cpp",
                    "UfuncCPU.cpp",
                    lambda: {
                        "meta_declaration": compute_meta_function_declaration(g),
                        "native_declaration": dest.compute_native_function_declaration(
                            g, backend_indices[dispatch_key]
                        ),
                        "native_definitions": dest.compute_ufunc_cpu(g),
                    },
                )
                cpu_vec_fm.write_with_template(
                    f"UfuncCPUKernel_{name}.cpp",
                    "UfuncCPUKernel.cpp",
                    lambda: {
                        "name": name,
                        "native_definitions": dest.compute_ufunc_cpu_kernel(g),
                    },
                )
            elif dispatch_key is DispatchKey.CUDA:
                cuda_headers = "#include <ATen/native/cuda/Loops.cuh>"
                if rocm:
                    cuda_headers = "#include <ATen/native/hip/Loops.cuh>"
                fm.write_with_template(
                    f"UfuncCUDA_{name}.cu",
                    "UfuncCUDA.cu",
                    lambda: {
                        "name": name,
                        "cuda_headers": cuda_headers,
                        "meta_declaration": compute_meta_function_declaration(g),
                        "native_declaration": dest.compute_native_function_declaration(
                            g, backend_indices[dispatch_key]
                        ),
                        "native_definitions": dest.compute_ufunc_cuda(g),
                    },
                )
            else:
                raise AssertionError(f"unrecognized {dispatch_key} for ufunc")

        del fm

    gen_aoti_c_shim_files(
        aoti_fm=aoti_fm,
        aoti_backends=aoti_backends,
        native_functions=native_functions,
        backend_indices=backend_indices,
        structured_native_functions=structured_native_functions,
        extra_cuda_headers=extra_cuda_headers,
        update_aoti_c_shim=update_aoti_c_shim,
        extend_aoti_c_shim=extend_aoti_c_shim,
    )

    # BackendSelect is generated specially
    def gen_backend_select() -> dict[str, list[str]]:
        relevant_fns = [
            fn for fn in native_functions if needs_backend_select(fn, selector)
        ]
        return {
            "ops_headers": [
                f"#include <ATen/ops/{fn.root_name}_ops.h>" for fn in relevant_fns
            ],
            "backend_select_method_definitions": list(
                mapMaybe(
                    ComputeBackendSelect(Target.DEFINITION, selector), relevant_fns
                )
            ),
            "backend_select_function_registrations": list(
                mapMaybe(
                    ComputeBackendSelect(Target.REGISTRATION, selector), relevant_fns
                )
            ),
        }

    cpu_fm.write("RegisterBackendSelect.cpp", gen_backend_select)

    schema_selector = selector
    if force_schema_registration:
        schema_selector = SelectiveBuilder.get_nop_selector()

    (
        aten_schema_registrations,
        schema_registrations,
    ) = get_native_function_schema_registrations(
        native_functions=native_functions, schema_selector=schema_selector
    )
    cpu_fm.write(
        "RegisterSchema.cpp",
        lambda: {
            "aten_schema_registrations": []
            if skip_dispatcher_op_registration
            else aten_schema_registrations,
            "schema_registrations": []
            if skip_dispatcher_op_registration
            else schema_registrations,
        },
    )

    def key_func(
        fn: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
    ) -> str:
        return fn.root_name

    cpu_fm.write_sharded(
        "Operators.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "operator_headers": [f"#include <ATen/ops/{fn.root_name}.h>"],
            "definitions": [
                ComputeOperators(
                    Target.DEFINITION,
                    static_dispatch_backend_indices=static_dispatch_idx,
                )(fn)
            ],
        },
        base_env={
            "static_dispatch_extra_headers": static_dispatch_extra_headers(
                static_dispatch_idx
            ),
        },
        num_shards=5,
        sharded_keys={
            "operator_headers",
            "definitions",
            "static_dispatch_extra_headers",
        },
    )

    cpu_fm.write("Functions.cpp", dict)

    core_fm.write("TensorMethods.cpp", dict)

    core_fm.write(
        "ATenOpList.cpp",
        lambda: {
            "aten_ops": list(mapMaybe(compute_aten_op, native_functions)),
        },
    )

    def gen_op_headers(
        g: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
    ) -> list[str]:
        if isinstance(g, NativeFunctionsViewGroup):
            # view ops always get a functionalization kernel
            headers = [
                f"#include <ATen/ops/{g.view.root_name}_native.h>",
                f"#include <ATen/ops/{g.view.root_name}_ops.h>",
            ]
            if g.view_copy is not None:
                headers += [
                    f"#include <ATen/ops/{g.view_copy.root_name}_native.h>",
                    f"#include <ATen/ops/{g.view_copy.root_name}_ops.h>",
                ]
            return headers
        elif isinstance(g, NativeFunctionsGroup):
            headers = [
                f"#include <ATen/ops/{g.functional.root_name}_native.h>",
                f"#include <ATen/ops/{g.functional.root_name}_ops.h>",
                f"#include <ATen/ops/{g.out.root_name}_native.h>",
                f"#include <ATen/ops/{g.out.root_name}_ops.h>",
            ]
            if g.inplace is not None:
                headers += [
                    f"#include <ATen/ops/{g.inplace.root_name}_native.h>",
                    f"#include <ATen/ops/{g.inplace.root_name}_ops.h>",
                ]
            if g.mutable is not None:
                headers += [
                    f"#include <ATen/ops/{g.mutable.root_name}_native.h>",
                    f"#include <ATen/ops/{g.mutable.root_name}_ops.h>",
                ]
            return headers
        else:
            return [
                f"#include <ATen/ops/{g.root_name}_native.h>",
                f"#include <ATen/ops/{g.root_name}_ops.h>",
            ]

    def functionalization_env_callable(
        g: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
    ) -> dict[str, list[str]]:
        return {
            "ops_headers": gen_op_headers(g),
            "func_definitions": gen_functionalization_definition(
                selector,
                g,
            ),
            "func_registrations": gen_functionalization_registration(
                selector,
                g,
                backend_indices[DispatchKey.CompositeImplicitAutograd],
            ),
        }

    all_groups: list[
        NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup
    ] = list(structured_native_functions) + list(
        view_groups  # type: ignore[assignment, arg-type, operator]
    )
    # Note: all operators that functionalization needs to handle (mutable and aliasing ops) should be grouped properly.
    # The only reason we really need to deal with direct NativeFunctions here (instead of the groups) is because:
    # (1) We can provide better error checking (error out if someone introduces a mutable op that doesn't obey the grouping logic)
    # (2) functionalization needs to manually register CompositeImplicitAutograd kernels, which might not be grouped.
    #     Although this could go away long-term if we add a dedicated dispatch key for decompositions.
    structured_map: dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(lambda g: list(g.functions()), structured_native_functions)
    }
    view_map: dict[OperatorName, NativeFunction] = {
        f.func.name: f for f in concatMap(lambda g: list(g.functions()), view_groups)
    }
    all_groups.extend(
        f
        for f in native_functions
        if f.func.name not in structured_map and f.func.name not in view_map
    )

    cpu_fm.write_sharded(
        "RegisterFunctionalization.cpp",
        all_groups,
        key_fn=key_func,
        env_callable=functionalization_env_callable,
        num_shards=4,
        sharded_keys={
            "ops_headers",
            "func_definitions",
            "func_registrations",
            "func_add_back_views_definitions",
            "func_add_back_views_registrations",
        },
    )

    cpu_fm.write(
        "FunctionalInverses.h",
        lambda: {
            "view_inverse_declarations": list(
                mapMaybe(
                    lambda g: gen_functionalization_view_inverse_declaration(
                        selector, g
                    ),
                    view_groups,
                )
            )
        },
    )

    cpu_fm.write(
        "ViewMetaClasses.h",
        lambda: {
            "view_meta_declarations": list(
                concatMap(
                    lambda g: gen_functionalization_view_meta_classes_decl(selector, g),
                    view_groups,
                )
            )
        },
    )

    cpu_fm.write(
        "ViewMetaClasses.cpp",
        lambda: {
            "view_meta_implementations": list(
                concatMap(
                    lambda g: gen_functionalization_view_meta_classes_impl(selector, g),
                    view_groups,
                )
            ),
            "op_headers": list(concatMap(gen_op_headers, view_groups)),
        },
    )

    # Note [view_copy NativeFunctions]
    # Every view operator in native_functions.yaml that is not CompositeImplicitAutograd
    # needs to have a corresponding non-aliasing {view}_copy variant.
    # Backends that use functionalization and don't know how to handle aliasing ops
    # are expected to implement kernels for these {view}_copy kernels instead.
    # The code for {view}_copy operators in core is pretty boilerplate-heavy however,
    # so we codegen the following:
    # (1) A CompositeExplicitAutogradNonFunctional kernel for every {view}_copy operator.
    #     These are never explicitly invoked by the functionalization pass,
    #     but they could theoretically be called from user code (I added these kernels for completeness,
    #     since the ops are part of the public API).
    # (2) A derivative formula for every {view}_copy operator
    #     {view}_copy operators can reuse the same derivative formulas as their {view} op counterparts,
    #     so rather than stamping all of the entries out in derivatives.yaml,
    #     we codegen them in.
    #     This is similar to how autograd codegen doesn't require inplace ops to have a derivatives.yaml entry.
    cpu_fm.write(
        "CompositeViewCopyKernels.cpp",
        lambda: {
            "ops_headers": [
                "\n".join(
                    f"#include <ATen/ops/{f.root_name}_ops.h>\n"
                    # NB: this include is important as it ensures we
                    # set the visibility on generated view_copy kernels
                    # correctly
                    f"#include <ATen/ops/{f.root_name}_native.h>"
                    for f in (
                        [g.view] if g.view_copy is None else [g.view, g.view_copy]
                    )
                )
                for g in view_groups
            ]
            + [
                "\n".join(
                    f"#include <ATen/ops/{f.root_name}_ops.h>\n"
                    # NB: this include is also important for correct visibility
                    f"#include <ATen/ops/{f.root_name}_native.h>"
                    for f in [g.inplace, g.mutable, g.functional]
                    if f is not None and "generated" not in f.tags
                )
                for g in structured_native_functions
            ],
            "CompositeViewCopyKernel_Definitions": list(
                mapMaybe(
                    GenCompositeViewCopyKernel(
                        backend_indices[
                            DispatchKey.CompositeExplicitAutogradNonFunctional
                        ]
                    ),
                    view_groups,
                )
            ),
            "GeneratedCompositeFunctional_Definitions": list(
                mapMaybe(
                    gen_composite_functional_kernel,
                    structured_native_functions,
                )
            ),
            "GeneratedCompositeOut_Definitions": list(
                mapMaybe(
                    gen_composite_out_kernel,
                    structured_native_functions,
                )
            ),
        },
    )


def gen_declarations_yaml(
    cpu_fm: FileManager, native_functions: Sequence[NativeFunction]
) -> None:
    cpu_fm.write(
        "Declarations.yaml",
        lambda: format_yaml([compute_declaration_yaml(f) for f in native_functions]),
    )


def get_torchgen_root() -> Path:
    """
    If you're depending on torchgen out-of-tree, you can use the root to figure
    out the path to native_functions.yaml
    """
    return Path(__file__).parent.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATen source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="aten/src/ATen",
    )
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    parser.add_argument(
        "--per-operator-headers",
        action="store_true",
        help="generate separate headers per operator in ATen/ops",
    )
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/aten/src/ATen",
    )
    parser.add_argument(
        "--aoti-install-dir",
        "--aoti_install_dir",
        help="output directory for AOTInductor shim",
        default="torch/csrc/inductor/aoti_torch/generated",
    )
    parser.add_argument(
        "--headeronly-install-dir",
        "--headeronly_install_dir",
        help="output directory for header-only generated files (e.g. enum_tag.h). "
        "Defaults to `<install-dir>/core` when --install-dir is set, otherwise "
        "`build/torch/headeronly/core`.",
        default=None,
    )
    parser.add_argument(
        "--rocm",
        action="store_true",
        help="reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly",
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Generate MPS registration code when set",
    )
    parser.add_argument(
        "--xpu",
        action="store_true",
        help="Generate XPU registration code when set",
    )
    parser.add_argument(
        "--mtia",
        action="store_true",
        help="Generate MTIA registration code when set",
    )

    # TODO: --op-registration-whitelist will be removed when all call-sites
    # for gen.py are moved over to using the operator YAML file for mobile
    # custom build.
    parser.add_argument(
        "--op-registration-whitelist",
        "--op_registration_whitelist",
        nargs="*",
        help="filter op registrations by the whitelist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    parser.add_argument(
        "--op-selection-yaml-path",
        "--op_selection_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "that contains the information about the set of selected operators "
        "and their categories (training, ...). Each operator is either a "
        "full operator name with overload or just a bare operator name. "
        "The operator names also contain the namespace prefix (e.g. aten::)",
    )
    parser.add_argument(
        "--backend-whitelist",
        "--backend_whitelist",
        nargs="*",
        help="filter dispatch backend by the whitelist (if set), "
        "e.g.: CPU CUDA QuantizedCPU ...",
    )
    parser.add_argument(
        "--static-dispatch-backend",
        "--static_dispatch_backend",
        nargs="*",
        help="generate static dispatch code for the specific backend (if set)",
    )
    parser.add_argument(
        "--skip-dispatcher-op-registration",
        "--skip_dispatcher_op_registration",
        action="store_true",
        help="Avoid registering operators into the dispatcher.",
    )
    parser.add_argument(
        "--force-schema-registration",
        "--force_schema_registration",
        action="store_true",
        help="force it to generate schema-only registrations for all ops, including"
        "those that are not listed on --op-registration-whitelist",
    )
    parser.add_argument(
        "--generate",
        type=str,
        nargs="*",
        choices=["headers", "sources", "declarations_yaml"],
        default=["headers", "sources", "declarations_yaml"],
        help="Generate only a subset of files",
    )
    parser.add_argument(
        "--update-aoti-c-shim",
        action="store_true",
        help="Update AOTInductor C shim after adding an entry to inductor_fallback_ops in torchgen/aoti/fallback_ops.py. "
        "WARNING: Do not use this unless you are sure what you are doing!!!",
    )
    parser.add_argument(
        "--extend-aoti-c-shim",
        action="store_true",
        help="This Flag indicates the generation of c shims for out-of-tree ATen ops,"
        "which is an extension to the In-tree ATen op c shims. This flag needs to be combined with"
        "---source-path=<out-of-tree native_functions.yaml>"
        "--aoti-install-dir=<in-tree aoti_install_dir>/extend"
        "   default is torch/csrc/inductor/aoti_torch/generated/extend"
        "WARNING: Do not use this unless you are sure what you are doing!!!",
    )

    options = parser.parse_args()

    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")

    from torchgen.model import dispatch_keys

    # Only a limited set of dispatch keys get CPUFunctions.h headers generated
    # for them; this is the set
    functions_keys = {
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeImplicitAutogradNestedTensor,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.Meta,
        DispatchKey.MTIA,
    }

    aoti_backends = {
        DispatchKey.CPU,
        DispatchKey.CUDA,
        # None will generate the aten shim based on aten_shimified_ops
        # which does not bypass the dispatcher
        None,
    }

    # TODO: stop generating CUDA kernels for non-CUDA builds
    ignore_keys = set()

    MPS_KEYS = {DispatchKey.MPS, DispatchKey.SparseMPS, DispatchKey.SparseCsrMPS}
    if options.mps or options.update_aoti_c_shim:
        functions_keys.update(MPS_KEYS)
        aoti_backends.add(DispatchKey.MPS)
    else:
        ignore_keys.update(MPS_KEYS)
        dispatch_keys[:] = [k for k in dispatch_keys if k not in MPS_KEYS]

    XPU_KEYS = {
        DispatchKey.XPU,
        DispatchKey.SparseXPU,
        DispatchKey.SparseCsrXPU,
        DispatchKey.NestedTensorXPU,
    }
    if options.xpu or options.update_aoti_c_shim:
        functions_keys.add(DispatchKey.XPU)
        aoti_backends.add(DispatchKey.XPU)
    else:
        ignore_keys.update(XPU_KEYS)
        dispatch_keys[:] = [k for k in dispatch_keys if k not in XPU_KEYS]

    if not options.mtia:
        ignore_keys.add(DispatchKey.MTIA)

        if DispatchKey.MTIA in dispatch_keys:
            del dispatch_keys[dispatch_keys.index(DispatchKey.MTIA)]

    if options.backend_whitelist:
        dispatch_keys = [
            k
            for k in dispatch_keys
            if is_generic_dispatch_key(k) or str(k) in options.backend_whitelist
        ]

    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path, ignore_keys)
    valid_tags = _GLOBAL_PARSE_TAGS_YAML_CACHE[tags_yaml_path]
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    grouped_native_functions = get_grouped_native_functions(native_functions)

    structured_native_functions = [
        g for g in grouped_native_functions if isinstance(g, NativeFunctionsGroup)
    ]
    native_functions_with_view_groups = get_grouped_by_view_native_functions(
        native_functions
    )
    view_groups = [
        g
        for g in native_functions_with_view_groups
        if isinstance(g, NativeFunctionsViewGroup)
    ]

    # NB: It is mandatory to NOT use os.path.join here, as the install directory
    # will eventually be ingested by cmake, which does not respect Windows style
    # path slashes.  If you switch this to use os.path.join, you'll get an error
    # like:
    #
    #   Syntax error in cmake code when parsing string
    #
    #     C:/Jenkins/workspace/pytorch-builds/pytorch-win-ws2016-cuda9-cudnn7-py3-build/build/aten/src/ATen\core/TensorMethods.h
    #
    #   Invalid character escape '\c'.
    core_install_dir = f"{options.install_dir}/core"
    Path(core_install_dir).mkdir(parents=True, exist_ok=True)
    ops_install_dir = f"{options.install_dir}/ops"
    Path(ops_install_dir).mkdir(parents=True, exist_ok=True)

    aoti_install_dir = f"{options.aoti_install_dir}"
    Path(aoti_install_dir).mkdir(parents=True, exist_ok=True)

    if options.headeronly_install_dir is not None:
        headeronly_install_dir = options.headeronly_install_dir
    elif options.install_dir is not None:
        headeronly_install_dir = f"{options.install_dir}/core"
    else:
        headeronly_install_dir = "build/torch/headeronly/core"
    Path(headeronly_install_dir).mkdir(parents=True, exist_ok=True)

    core_fm = make_file_manager(options=options, install_dir=core_install_dir)
    cpu_fm = make_file_manager(options=options)
    cpu_vec_fm = make_file_manager(options=options)
    cuda_fm = make_file_manager(options=options)
    ops_fm = make_file_manager(options=options, install_dir=ops_install_dir)
    aoti_fm = make_file_manager(options=options, install_dir=aoti_install_dir)
    headeronly_fm = make_file_manager(
        options=options, install_dir=headeronly_install_dir
    )
    device_fms = {"cuda": cuda_fm}
    if options.xpu:
        device_fms["xpu"] = make_file_manager(options=options)

    static_dispatch_idx: list[BackendIndex] = []
    if options.static_dispatch_backend:
        static_dispatch_idx = [
            backend_indices[DispatchKey.parse(key)]
            for key in options.static_dispatch_backend
        ]
        for key in options.static_dispatch_backend:
            dp_key = DispatchKey.parse(key)
            if dp_key not in functions_keys:
                functions_keys.add(dp_key)

    if "sources" in options.generate:
        gen_source_files(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            view_groups=view_groups,
            selector=selector,
            static_dispatch_idx=static_dispatch_idx,
            backend_indices=backend_indices,
            aoti_fm=aoti_fm,
            core_fm=core_fm,
            cpu_vec_fm=cpu_vec_fm,
            cpu_fm=cpu_fm,
            device_fms=device_fms,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=options.rocm,
            force_schema_registration=options.force_schema_registration,
            per_operator_headers=options.per_operator_headers,
            skip_dispatcher_op_registration=options.skip_dispatcher_op_registration,
            update_aoti_c_shim=options.update_aoti_c_shim,
            aoti_backends=aoti_backends,
            extend_aoti_c_shim=options.extend_aoti_c_shim,
        )

    if "headers" in options.generate:
        gen_headers(
            native_functions=native_functions,
            valid_tags=valid_tags,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            headeronly_fm=headeronly_fm,
            core_fm=core_fm,
            cpu_fm=cpu_fm,
            device_fms=device_fms,
            ops_fm=ops_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=options.rocm,
            per_operator_headers=options.per_operator_headers,
        )

    if "declarations_yaml" in options.generate:
        gen_declarations_yaml(native_functions=native_functions, cpu_fm=cpu_fm)

    if options.output_dependencies:
        depfile_path = Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        for fm, prefix in [
            (cpu_fm, ""),
            (cpu_vec_fm, "cpu_vec_"),
            (core_fm, "core_"),
            (ops_fm, "ops_"),
        ] + [(device_fm, f"{device}_") for device, device_fm in device_fms.items()]:
            varname = prefix + depfile_stem
            path = depfile_path.parent / (prefix + depfile_name)
            fm.write_outputs(varname, str(path))


if __name__ == "__main__":
    main()
<|endoftext|>from __future__ import annotations

from torchgen.model import NativeFunctionsGroup, NativeFunctionsViewGroup


def func_name_base_str(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> str:
    if isinstance(g, NativeFunctionsGroup):
        return str(g.functional.func.name.name.base)
    else:
        return str(g.view.root_name)


is_hand_written_ops_ = frozenset(
    (
        "abs",
        "add",
        "addmm",
        "all",
        "any",
        "argmin",
        "bmm",
        "clamp",
        "clamp_min",
        "cumsum",
        "div",
        "fmod",
        "index_select",
        "leaky_relu",
        "linear",
        "log",
        "matmul",
        "mul",
        "narrow_copy",
        "nonzero",
        "pow",
        "remainder",
        "sigmoid",
        "sign",
        "sub",
        "tanh",
        "detach",
        "expand_as",
        "flatten",
        "narrow",
        "reshape_as",
        "select",
        "slice",
        "softmax",
        "split",
        "squeeze",
        "transpose",
        "view",
        "where",
    )
)


def is_hand_written(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> bool:
    name_base = func_name_base_str(g)
    return name_base in is_hand_written_ops_


def override_test_values(arg_map: dict[str, str], op_name: str, index: int) -> None:
    if index not in (0, 1):
        raise AssertionError(f"index must be 0 or 1, got {index}")
    if op_name == "addr":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["vec1"] = "at::rand({6})"
            arg_map["vec2"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["vec1"] = "at::rand({22})"
            arg_map["vec2"] = "at::rand({22})"
        return
    if op_name == "mv":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["vec"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["vec"] = "at::rand({22})"
        return
    if op_name == "addbmm":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
        return
    if op_name == "cross":
        if index == 0:
            arg_map["self"] = "at::rand({3, 3, 3})"
            arg_map["other"] = "at::rand({3, 3, 3})"
        else:
            arg_map["self"] = "at::rand({22, 3, 22})"
            arg_map["other"] = "at::rand({22, 3, 22})"
        return
    if op_name == "take":
        if index == 0:
            arg_map["index"] = "at::randint(0, 216, {20}, torch::kInt64)"
        else:
            arg_map["index"] = "at::randint(0, 1000, {100}, torch::kInt64)"
        return
    if op_name == "take_along_dim":
        if index == 0:
            arg_map["indices"] = "at::argsort(self0, 1, true)"
        else:
            arg_map["indices"] = "at::argsort(self1, 1, true)"
        return
    if op_name == "masked_select":
        if index == 0:
            arg_map["mask"] = "at::randn({6, 6, 6}) > 0.5"
        else:
            arg_map["mask"] = "at::rand({22, 22, 22}) > 0.5"
        return
    if op_name == "orgqr":
        if index == 0:
            arg_map["input2"] = "at::rand({6, 6})"
        else:
            arg_map["input2"] = "at::rand({22, 22})"
        return
    if op_name == "ormqr":
        if index == 0:
            arg_map["input2"] = "at::rand({6, 6})"
        else:
            arg_map["input2"] = "at::rand({22, 22})"
        return
    if op_name == "quantile":
        if index == 0:
            arg_map["q"] = "at::rand({6})"
            arg_map["interpolation"] = '"linear"'
        else:
            arg_map["q"] = "at::rand({22})"
            arg_map["interpolation"] = '"linear"'
        return
    if op_name == "nanquantile":
        if index == 0:
            arg_map["q"] = "at::rand({6})"
            arg_map["interpolation"] = '"linear"'
        else:
            arg_map["q"] = "at::rand({22})"
            arg_map["interpolation"] = '"linear"'
        return
    if op_name == "multi_margin_loss":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["target"] = "at::randint(6, {6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["target"] = "at::randint(22, {22}, torch::kInt64)"
            arg_map["weight"] = "at::rand({22})"
        return
    if op_name == "multilabel_margin_loss":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["target"] = "at::randint(6, {6, 6}, torch::kInt64)"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["target"] = "at::randint(22, {22, 22}, torch::kInt64)"
        return
    if op_name == "nll_loss":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["target"] = "at::randint(6, {6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["target"] = "at::randint(22, {22}, torch::kInt64)"
            arg_map["weight"] = "at::rand({22})"
        return
    if op_name == "nll_loss2d":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6, 6, 6})"
            arg_map["target"] = "at::randint(6, {6, 6, 6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22, 22, 22})"
            arg_map["target"] = "at::randint(22, {22, 22, 22}, torch::kInt64)"
            arg_map["weight"] = "at::rand({22})"
        return
    if op_name in (
        "fft_fft",
        "fft_ifft",
        "fft_rfft",
        "fft_irfft",
        "fft_hfft",
        "fft_ihfft",
    ):
        arg_map["norm"] = '"forward"'
        return
    if op_name == "linalg_tensorinv":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6, 6, 6})"
            arg_map["ind"] = "2"
        else:
            arg_map["self"] = "at::rand({22, 22, 22, 22})"
            arg_map["ind"] = "2"
        return
    if op_name == "addmv":
        if index == 0:
            arg_map["self"] = "at::rand({2})"
            arg_map["mat"] = "at::rand({2, 2})"
            arg_map["vec"] = "at::rand({2})"
        else:
            arg_map["self"] = "at::rand({35})"
            arg_map["mat"] = "at::rand({35, 35})"
            arg_map["vec"] = "at::rand({35})"
        return
    if op_name == "acosh":
        if index == 0:
            arg_map["self"] = "at::rand({2, 2, 2}) + at::ones({2, 2, 2})"
        else:
            arg_map["self"] = "at::rand({5, 5, 5}) + at::ones({5, 5, 5})"
        return
    if op_name == "adaptive_max_pool2d_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({2, 2, 2}, at::kFloat)"
            arg_map["self"] = "at::rand({2, 2, 2}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {2, 2, 2}, at::kLong)"
        else:
            arg_map["grad_output"] = "at::rand({3, 3, 3}, at::kFloat)"
            arg_map["self"] = "at::rand({3, 3, 3}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {3, 3, 3}, at::kLong)"
        return
    if op_name == "adaptive_max_pool3d_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({2, 2, 2, 2}, at::kFloat)"
            arg_map["self"] = "at::rand({2, 2, 2, 2}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {2, 2, 2, 2}, at::kLong)"
        else:
            arg_map["grad_output"] = "at::rand({3, 3, 3, 3}, at::kFloat)"
            arg_map["self"] = "at::rand({3, 3, 3, 3}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {3, 3, 3, 3}, at::kLong)"
        return
    if op_name == "bitwise_left_shift":
        if index == 0:
            arg_map["self"] = "at::randint(1, 1 << 4, {6, 6, 6}, at::kInt)"
            arg_map["other"] = "at::randint(1, 26, {6, 6, 6}, at::kInt)"
        else:
            arg_map["self"] = "at::randint(1, 1 << 4, {22, 22, 22}, at::kInt)"
            arg_map["other"] = "at::randint(1, 26, {22, 22, 22}, at::kInt)"
        return
    if op_name == "bitwise_right_shift":
        if index == 0:
            arg_map["self"] = "at::randint(1 << 21, 1 << 30, {6, 6, 6}, at::kInt)"
            arg_map["other"] = "at::randint(1, 22, {6, 6, 6}, at::kInt)"
        else:
            arg_map["self"] = "at::randint(1 << 21, 1 << 30, {22, 22, 22}, at::kInt)"
            arg_map["other"] = "at::randint(1, 22, {22, 22, 22}, at::kInt)"
        return
    if op_name == "gather":
        if index == 0:
            arg_map["self"] = "at::randint(1, 100, {2,2,2}, at::kInt)"
            arg_map["dim"] = "1"
            arg_map["index"] = "at::randint(0, 1, {2,2,2}, torch::kInt64)"
            arg_map["sparse_grad"] = "false"
        else:
            arg_map["self"] = "at::randint(1, 100, {5,5,5}, at::kInt)"
            arg_map["dim"] = "1"
            arg_map["index"] = "at::randint(0, 4, {5,5,5}, torch::kInt64)"
            arg_map["sparse_grad"] = "false"
        return
    if op_name == "gelu":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6, 6})"
            arg_map["approximate"] = '"tanh"'
        else:
            arg_map["self"] = "at::rand({22, 22, 22})"
            arg_map["approximate"] = '"tanh"'
        return
    if op_name == "gelu_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({6, 6, 6})"
            arg_map["self"] = "at::rand({6, 6, 6})"
            arg_map["approximate"] = '"tanh"'
        else:
            arg_map["grad_output"] = "at::rand({22, 22, 22})"
            arg_map["self"] = "at::rand({22, 22, 22})"
            arg_map["approximate"] = '"tanh"'
        return
    if op_name == "index_add":
        if index == 0:
            arg_map["self"] = "at::rand({2})"
            arg_map["dim"] = "0"
            arg_map["index"] = "at::randint(0, 1, {2}, at::kInt)"
            arg_map["source"] = "at::rand({2})"
            arg_map["alpha"] = "2"
        else:
            arg_map["self"] = "at::rand({16})"
            arg_map["dim"] = "0"
            arg_map["index"] = "at::randint(0, 10, {16}, at::kInt)"
            arg_map["source"] = "at::rand({16})"
            arg_map["alpha"] = "2"
        return
    if op_name == "index_copy":
        if index == 0:
            arg_map["self"] = "at::rand({2})"
            arg_map["dim"] = "0"
            arg_map["index"] = "at::randint(0, 1, {2}, at::kLong)"
            arg_map["source"] = "at::rand({2})"
        else:
            arg_map["self"] = "at::rand({32})"
            arg_map["dim"] = "0"
            arg_map["index"] = "at::randint(0, 10, {32}, at::kLong)"
            arg_map["source"] = "at::rand({32})"
        return
    if op_name == "linalg_cross":
        if index == 0:
            arg_map["self"] = "at::rand({6, 3, 6})"
            arg_map["other"] = "at::rand({6, 3, 6})"
            arg_map["dim"] = "1"
        else:
            arg_map["self"] = "at::rand({22, 3, 22})"
            arg_map["other"] = "at::rand({22, 3, 22})"
            arg_map["dim"] = "1"
        return
    if op_name == "nll_loss_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({})"
            arg_map["self"] = "at::rand({6})"
            arg_map["target"] = "at::randint(0, 5, {6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
            arg_map["reduction"] = "1"
            arg_map["ignore_index"] = "1"
            arg_map["total_weight"] = "at::rand({})"
        else:
            arg_map["grad_output"] = "at::rand({})"
            arg_map["self"] = "at::rand({36})"
            arg_map["target"] = "at::randint(0, 11, {36}, torch::kInt64)"
            arg_map["weight"] = "at::rand({36})"
            arg_map["reduction"] = "1"
            arg_map["ignore_index"] = "1"
            arg_map["total_weight"] = "at::rand({})"
        return
    if op_name in ["scatter", "scatter_add", "_scatter_reduce"]:
        if index == 0:
            arg_map["self"] = "at::randint(1, 100, {2,2,2}, torch::kInt64)"
            arg_map["index"] = "at::randint(0, 1, {2,2,2}, torch::kInt64)"
            arg_map["src"] = "at::randint(1, 100, {2,2,2}, torch::kInt64)"
        else:
            arg_map["self"] = "at::randint(1, 100, {5,5,5}, torch::kInt64)"
            arg_map["index"] = "at::randint(0, 1, {5,5,5}, torch::kInt64)"
            arg_map["src"] = "at::randint(1, 100, {5,5,5}, torch::kInt64)"
        if "reduce" in arg_map:
            arg_map["reduce"] = '"sum"' if op_name == "_scatter_reduce" else '"add"'
        return
    if op_name == "scatter_reduce":
        arg_map["reduce"] = '"mean"'
        if index == 0:
            arg_map["index"] = "at::randint(6, {6, 6, 6}, torch::kInt64)"
        else:
            arg_map["index"] = "at::randint(22, {22, 22, 22}, torch::kInt64)"
        return
    if op_name == "special_zeta":
        if index == 0:
            arg_map["self"] = "at::rand({2,2,2}, at::kDouble) + at::ones({2,2,2})"
            arg_map["other"] = "at::rand({2,2,2}, at::kDouble) + at::ones({2,2,2})"
        else:
            arg_map["self"] = "at::rand({5,5,5}, at::kDouble) + at::ones({5,5,5})"
            arg_map["other"] = "at::rand({5,5,5}, at::kDouble) + at::ones({5,5,5})"
        return
    if op_name == "_convert_indices_from_csr_to_coo":
        if index == 0:
            arg_map["crow_indices"] = "torch::tensor({1}, torch::kInt32)"
            arg_map["col_indices"] = "torch::tensor({0, 1, 0}, torch::kInt32)"
            arg_map["out_int32"] = "false"
        else:
            arg_map["crow_indices"] = "torch::tensor({0}, torch::kInt32)"
            arg_map["col_indices"] = (
                "torch::tensor({0, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2}, torch::kInt32)"
            )
            arg_map["out_int32"] = "false"
        return
    if op_name == "_convert_indices_from_coo_to_csr":
        if index == 0:
            arg_map["self"] = "at::randint(0, 3, {2}, at::kInt)"
            arg_map["size"] = "10"
            arg_map["out_int32"] = "false"
        else:
            arg_map["self"] = "at::randint(0, 3, {12}, at::kInt)"
            arg_map["size"] = "24"
            arg_map["out_int32"] = "false"
        return
    if op_name in ("diagonal", "linalg_diagonal"):
        arg_map["offset"] = "0"
        arg_map["dim1"] = "2"
        arg_map["dim2"] = "1"
        return
<|endoftext|>from __future__ import annotations

import argparse
import itertools
import os
from typing import TYPE_CHECKING, TypeVar

from libfb.py.log import set_simple_logging  # type: ignore[import]

from torchgen import gen
from torchgen.context import native_function_manager
from torchgen.model import DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup
from torchgen.static_runtime import config, generator


if TYPE_CHECKING:
    from collections.abc import Sequence


# Given a list of `grouped_native_functions` sorted by their op names, return a list of
# lists each of which groups ops that share the base name. For example, `mean` and
# `mean.dim` are grouped together by this function.

NativeGroupT = TypeVar(
    "NativeGroupT",
    bound=NativeFunctionsGroup | NativeFunctionsViewGroup,
)


def group_functions_by_op_name(
    grouped_native_functions: Sequence[NativeGroupT],
) -> Sequence[Sequence[NativeGroupT]]:
    if not grouped_native_functions:
        return []
    groups = []

    def is_supported(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> bool:
        with native_function_manager(g):
            return generator.is_supported(g)

    eligible_ops = (g for g in grouped_native_functions if is_supported(g))
    groups = [
        list(group)
        for k, group in (
            itertools.groupby(
                eligible_ops,
                key=config.func_name_base_str,
            )
        )
    ]

    return groups


def clang_format(cpp_file_path: str) -> None:
    import subprocess

    subprocess.check_call(["clang-format", "-i", cpp_file_path])


def write_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    code = "\n".join(cpp_ops)
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch {{
namespace jit {{

{code}

}} // namespace jit
}} // namespace torch
"""
    with open(file_path, "w") as f:
        f.write(generated)
    clang_format(file_path)


def write_test_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    code = "\n".join(cpp_ops)
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>

#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

{code}

"""
    with open(file_path, "w") as f:
        f.write(generated)
    clang_format(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATen source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="caffe2/aten/src/ATen",
    )
    parser.add_argument(
        "-p",
        "--generated-ops-cpp-path",
        help="path to directory to generate op dispatcher .cpp file",
        default="caffe2/torch/csrc/jit/runtime/static/generated_ops.cpp",
    )
    parser.add_argument(
        "-t",
        "--generated-ops-test-cpp-path",
        help="path to directory to generate op dispatcher .cpp file",
        default="caffe2/benchmarks/static_runtime/test_generated_ops.cc",
    )
    options = parser.parse_args()
    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")
    parsed_yaml = gen.parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    op_generator = generator.GenOpDispatcher()
    test_case_generator = generator.GenOpTestCase()

    native_functions_groups = [
        g
        for g in gen.get_grouped_native_functions(native_functions)
        if isinstance(g, NativeFunctionsGroup)
    ]

    supported_functions_groups = group_functions_by_op_name(native_functions_groups)

    out_variant_op_result = [
        op_generator.out_variant(groups, backend_indices[DispatchKey.CPU])
        for groups in supported_functions_groups
    ]
    out_variant_test_result = [
        test_case_generator.out_variant(groups) for groups in supported_functions_groups
    ]

    native_functions_view_groups = [
        g
        for g in gen.get_grouped_by_view_native_functions(native_functions)
        if isinstance(g, NativeFunctionsViewGroup)
    ]

    supported_functions_view_groups = group_functions_by_op_name(
        native_functions_view_groups
    )

    view_op_result = [
        op_generator.view(groups, backend_indices[DispatchKey.CPU])
        for groups in supported_functions_view_groups
    ]
    view_test_result = [
        test_case_generator.view(groups) for groups in supported_functions_view_groups
    ]

    op_result = out_variant_op_result + ["\n\n"] + view_op_result
    test_result = out_variant_test_result + ["\n\n"] + view_test_result

    write_cpp(op_result, options.generated_ops_cpp_path)
    write_test_cpp(test_result, options.generated_ops_test_cpp_path)

    print(
        f"\ntotal grouped native ops: {len(gen.get_grouped_native_functions(native_functions)):d}"
    )

    print(f"grouped native ops with out variant: {len(native_functions_groups):d}")
    supported_functions_num = sum(len(groups) for groups in supported_functions_groups)
    print(f"generated functions groups with out variant: {supported_functions_num:d}")

    print(f"\nview grouped native ops: {len(native_functions_view_groups):d}")
    supported_view_functions_num = sum(
        len(groups) for groups in supported_functions_view_groups
    )
    print(f"generated functions view groups: {supported_view_functions_num:d}")

    print(
        f"\noverall generated : {supported_functions_num + supported_view_functions_num:d}"
    )


if __name__ == "__main__":
    set_simple_logging(escape_newlines=False)
    main()
<|endoftext|>from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING

import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    FunctionSchema,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    OptionalType,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.static_runtime import config


if TYPE_CHECKING:
    from collections.abc import Sequence


logger: logging.Logger = logging.getLogger()


def has_alias(
    arguments: Sequence[Argument | SelfArgument | TensorOptionsArguments],
) -> bool:
    for arg in arguments:
        annotation = getattr(arg, "annotation", None)
        if not annotation:
            continue
        alias_set = getattr(annotation, "alias_set", ())
        if alias_set:
            return True
    return False


BLOCKED_OPS = frozenset(
    (
        # non cpu ops
        "sparse_sampled_addmm",
        "hspmm",
        "linalg_svdvals",
        # sparse ops
        "sspaddmm",
        "coalesce",
        "_indices",
        "indices",
        "_values",
        "values",
        "crow_indices",
        "col_indices",
        # deprecated ops
        "floor_divide",
        "ger",
        # buggy ops
        "conj_physical",  # P495807361
        "binary_cross_entropy",  # P496394764
        "arccosh",
        # uncommon ops
        "cholesky",
        "lu_solve",
        "linalg_cholesky",
        "linalg_householder_product",
        "linalg_ldl_solve",
        "_compute_linear_combination",
        # training related ops
        "_make_dual",
        # cannot call directly
        "_fw_primal",
        # no documentation
        "_index_reduce",
        # TODO: these ones got added recently and need manual inspection
        "_new_zeros_with_same_feature_meta",
        "_conj_physical",
        "binary_cross_entropy_with_logits",
        "bincount",
        "conv_tbc",
        "copy",
        "_copy_from",
        "_copy_from_and_resize",
        "count_nonzero",
        "cudnn_affine_grid_generator",
        "cudnn_affine_grid_generator_backward",
        "cudnn_grid_sampler",
        "diag_embed",
        "embedding",
        "embedding_dense_backward",
        "_embedding_bag_dense_backward",
        "_embedding_bag_per_sample_weights_backward",
        "grid_sampler_2d",
        "_grid_sampler_2d_cpu_fallback",
        "grid_sampler_3d",
        "isnan",
        "mkldnn_linear",
        "median",
        "nanmedian",
        "_sparse_sparse_matmul",
        "batch_norm_backward_elemt",
        "_euclidean_dist",
        "pixel_shuffle",
        "pixel_unshuffle",
        "channel_shuffle",
        "_reshape_nested_backward",
        "relu",
        "prelu",
        "celu",
        "slice_scatter",
        "select_scatter",
        "diagonal_scatter",
        "sum",
        "_mkldnn_transpose",
        "_nested_tensor_from_mask",
        "_nested_from_padded",
        "_nested_tensor_size",
        "_nested_from_padded_and_nested_example",
        "_standard_gamma_grad",
        "_dirichlet_grad",
        "native_norm",
        "_sparse_softmax",
        "_sparse_softmax_backward_data",
        "_sparse_log_softmax",
        "_sparse_log_softmax_backward_data",
        "zero",
        "_sparse_addmm",
        "sparse_mask",
        "_sparse_mask_projection",
        "_to_dense",
        "_coalesce",
        "_coalesced",
        "copy_sparse_to_sparse",
        "to_sparse",
        "to_sparse_csr",
        "to_sparse_csc",
        "to_mkldnn",
        "quantize_per_tensor_dynamic",
        "quantize_per_channel",
        "q_per_channel_scales",
        "q_per_channel_zero_points",
        "int_repr",
        "_make_per_channel_quantized_tensor",
        "set",
        "lift",
        "lift_fresh",
        "lift_fresh_copy",
        "masked_scatter",
        "_masked_softmax",
        "_masked_softmax_backward",
        "put",
        "index_reduce",
        "trace",
        "_cholesky_solve_helper",
        "dist",
        "max",
        "_torch_cuda_cu_linker_symbol_op",
        "glu_jvp",
        "glu_backward_jvp",
        "hardswish_backward",
        "rrelu_with_noise_backward",
        "mkldnn_adaptive_avg_pool2d_backward",
        "_adaptive_avg_pool2d_backward",
        "_adaptive_avg_pool3d_backward",
        "isinf",
        "linalg_lu_solve",
        "linalg_vecdot",
        "linalg_matrix_exp",
        "linalg_eigvalsh",
        "_test_warn_in_autograd",
        "_test_autograd_multiple_dispatch_view",
        "_test_autograd_multiple_dispatch_view_copy",
        "_segment_reduce",
        "_segment_reduce_backward",
        "_fw_primal_copy",
        "_make_dual_copy",
        "view_as_real_copy",
        "view_as_complex_copy",
        "_conj_copy",
        "_neg_view_copy",
        "diagonal_copy",
        "detach_copy",
        "squeeze_copy",
        "t_copy",
        "unsqueeze_copy",
        "_indices_copy",
        "_values_copy",
        "indices_copy",
        "values_copy",
        "crow_indices_copy",
        "col_indices_copy",
        "ccol_indices",
        "ccol_indices_copy",
        "row_indices",
        "row_indices_copy",
        "unfold_copy",
        "alias_copy",
        "_triton_multi_head_attention",
        "special_airy_ai",
        "special_bessel_j0",
        "special_bessel_j1",
        "special_bessel_y0",
        "special_bessel_y1",
        "special_chebyshev_polynomial_t",
        "special_chebyshev_polynomial_u",
        "special_chebyshev_polynomial_v",
        "special_chebyshev_polynomial_w",
        "special_hermite_polynomial_h",
        "special_hermite_polynomial_he",
        "special_laguerre_polynomial_l",
        "special_legendre_polynomial_p",
        "special_modified_bessel_i0",
        "special_modified_bessel_i1",
        "special_modified_bessel_k0",
        "special_modified_bessel_k1",
        "special_scaled_modified_bessel_k0",
        "special_scaled_modified_bessel_k1",
        "special_shifted_chebyshev_polynomial_t",
        "special_shifted_chebyshev_polynomial_u",
        "special_shifted_chebyshev_polynomial_v",
        "special_shifted_chebyshev_polynomial_w",
        "special_spherical_bessel_j0",
        "_foobar",
        "_nested_tensor_strides",
        "_nested_tensor_storage_offsets",
        "_nested_get_values",  # no CPU backend
        "_nested_get_values_copy",  # no CPU backend
        "_nested_view_from_jagged",  # testing needs to be patched
        "_nested_view_from_jagged_copy",  # testing needs to be patched
        "_nested_view_from_buffer",  # testing needs to be patched
        "_nested_view_from_buffer_copy",  # testing needs to be patched
        "_int_mm",  # testing needs to be patched
        "_to_sparse_csc",  # testing needs to be patched
        "_to_sparse_csr",  # testing needs to be patched
        "segment_reduce",  # testing needs to be patched
    )
)


def is_supported(g: NativeFunctionsGroup | NativeFunctionsViewGroup) -> bool:
    base_op_name = ""
    func = None
    if isinstance(g, NativeFunctionsViewGroup):
        base_op_name = g.view.root_name
        func = g.view.func
    else:
        base_op_name = g.out.func.name.name.base
        func = g.out.func
    if config.is_hand_written(g):
        logger.info("HAND WRITTEN: %s", base_op_name)
        return False
    if base_op_name in BLOCKED_OPS:
        logger.info("BLOCKED: %s", base_op_name)
        return False
    for arg in func.schema_order_arguments():
        maybe_method = ivalue_type_conversion_method(arg.type)
        if not maybe_method:
            # Type converting is unsupported yet.
            logger.info("NOT SUPPORTED TYPE CONVERTING: %s", func)
            return False

    if isinstance(g, NativeFunctionsViewGroup):
        # TODO: stop doing type tests by converting to C++ and then testing
        # the string, just test the dang thing directly
        if "at::Tensor" != cpp.returns_type(func.returns, symint=False).cpp_type():
            # Returns a non-Tensor value.
            logger.info("NON-TENSOR RET TYPE: %s", func)
            return False
        return True

    # For out variant ops, we need to check the arguments of its functional func.
    for arg in g.functional.func.schema_order_arguments():
        maybe_method = ivalue_type_conversion_method(arg.type)
        if not maybe_method:
            # Type converting is unsupported yet.
            logger.info("NOT SUPPORTED TYPE CONVERTING: %s", g.functional.func)
            return False

    if not g.structured:
        # In case of unstructured op, we check if it has out variant implementation.
        # The out variant implementation satisfies the minimum requirement that it has the output tensor as the last
        # parameter.
        if (
            not hasattr(g, "out")
            or not str(func).endswith("Tensor(a!) out) -> Tensor(a!)")
            or not str(func.name).endswith(".out")
        ):
            return False
    # TODO: stop type testing by converting to C++
    if "at::Tensor &" != cpp.returns_type(func.returns, symint=False).cpp_type():
        logger.info("NON_TENSOR RET TYPE: %s", func)
        return False
    if has_alias(func.arguments.non_out):
        # This op may create an alias of inputs.
        logger.info("INPUTS ALIAS: %s", base_op_name)
        return False
    return True


def ivalue_type_conversion_method(
    arg_type: BaseType | OptionalType | Type,
) -> tuple[bool, str] | None:
    """
    Return the method call expression of `c10::ivalue' to convert its contained value to
    the expected value of `arg_type` type. For example, for `arg_type` == BaseTy.Tensor,
    this function returns ".toTensor()", so that it can be appended to the ivalue's
    variable name to get the value of the expected type.
    """
    type_conversion_methods = {
        BaseTy.Tensor: ((True, "toTensor()"), (False, "toOptional<at::Tensor>()")),
        BaseTy.int: ((False, "toInt()"), (False, "toOptional<int64_t>()")),
        BaseTy.bool: ((False, "toBool()"), (False, "toOptional<bool>()")),
        BaseTy.Scalar: ((False, "toScalar()"), (False, "toOptional<at::Scalar>()")),
        BaseTy.ScalarType: (
            (False, "toScalarType()"),
            (False, "toOptional<at::ScalarType>()"),
        ),
        BaseTy.str: (
            (False, "toStringView()"),
            (False, "toOptional<c10::string_view>()"),
            (False, "toOptional<::std::string_view>()"),
        ),
    }

    base_ty_object = None
    if isinstance(arg_type, BaseType):
        base_ty_object = arg_type.name
    elif isinstance(arg_type, OptionalType):
        if not isinstance(arg_type.elem, BaseType):
            # ListType is currently unsupported.
            return None
        base_ty_object = arg_type.elem.name
    else:
        return None

    if base_ty_object not in type_conversion_methods:
        return None
    methods = type_conversion_methods[base_ty_object]
    if isinstance(arg_type, BaseType):
        return methods[0]
    return methods[1]


should_use_int_tensor_ops_ = frozenset(
    (
        "bitwise_not",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "gcd",
        "lcm",
        "scatter",
        "gather",
        "_convert_indices_from_coo_to_csr",
        "_convert_indices_from_csr_to_coo",
    )
)
should_use_complex_tensor_ops_ = frozenset(("view_as_real", "imag", "_conj"))


def should_use_int_tensor(op_name: str) -> bool:
    return op_name in should_use_int_tensor_ops_


def should_use_complex_tensor(op_name: str) -> bool:
    return op_name in should_use_complex_tensor_ops_


test_tensor_dim_ops_1_ = frozenset(
    (
        "addmv",
        "index_add",
        "_convert_indices_from_coo_to_csr",
        "_convert_indices_from_csr_to_coo",
        "nll_loss_backward",
        "dot",
        "vdot",
        "outer",
        "ger",
    )
)
test_tensor_dim_ops_2_ = frozenset(
    ("addmm", "mm", "nuclear_norm", "diag", "_addmm_activation", "matrix_H", "t")
)


def test_tensor_dim(op_name: str) -> int:
    if op_name in test_tensor_dim_ops_1_:
        return 1
    if op_name in test_tensor_dim_ops_2_:
        return 2
    return 3


test_tensor_shapes_string = '{"view_as_complex": "{2, 2}"}'
test_tensor_shape_json: dict[str, str] = json.loads(test_tensor_shapes_string)


def test_tensor_shape(op_name: str) -> str:
    if op_name in test_tensor_shape_json:
        return test_tensor_shape_json[op_name]
    else:
        return ""


def test_value_expression(
    arg_type: BaseType | OptionalType | Type, index: int, op_name: str
) -> str:
    tensor_size_ex = test_tensor_shape(op_name)
    if tensor_size_ex == "":
        num_tensors = 16 if index == 0 else 64
        num_dim = test_tensor_dim(op_name)
        size_per_dim = math.ceil(num_tensors / float(num_dim))
        size_per_dim += size_per_dim % 2
        tensor_size_ex = "{{{}}}".format(",".join([f"{size_per_dim}"] * num_dim))
    if should_use_int_tensor(op_name):
        tensor_expression = f"at::randint(1, 100, {tensor_size_ex}, at::kInt)"
    elif should_use_complex_tensor(op_name):
        tensor_expression = f"at::randn({tensor_size_ex}, at::kComplexFloat)"
    else:
        tensor_expression = f"at::rand({tensor_size_ex})"

    value_expressions = {
        BaseTy.Tensor: tensor_expression,
        BaseTy.int: "1",
        BaseTy.bool: "false",
        BaseTy.Scalar: "2",
        BaseTy.ScalarType: "at::ScalarType::Float",
        BaseTy.str: '"floor"',
    }

    base_ty_object = None
    if isinstance(arg_type, BaseType):
        base_ty_object = arg_type.name
    else:
        if not (
            isinstance(arg_type, OptionalType) and isinstance(arg_type.elem, BaseType)
        ):
            raise AssertionError(
                f"Expected OptionalType with BaseType elem, got {type(arg_type)}"
            )
        base_ty_object = arg_type.elem.name
    if base_ty_object not in value_expressions:
        raise AssertionError(f"Unexpected type: {base_ty_object}")
    value_expression = value_expressions[base_ty_object]
    return value_expression


def generate_test_value_definitions(schema: FunctionSchema, index: int) -> str:
    if schema.is_out_fn():
        raise AssertionError(f"Expected non-out function, got {schema}")
    schema_name = schema.name.name.base
    arg_map = {}
    for arg in schema.schema_order_arguments():
        test_value_exp = test_value_expression(arg.type, index, schema_name)
        arg_map[arg.name] = test_value_exp
    config.override_test_values(arg_map, schema_name, index)
    arg_populations = []
    for arg_name, arg_value in arg_map.items():
        arg_populations.append(f"auto {arg_name}{index} = {arg_value}")
    return ";\n    ".join(arg_populations) + ";"


def generate_test_value_names(schema: FunctionSchema, index: int) -> str:
    if schema.is_out_fn():
        raise AssertionError(f"Expected non-out function, got {schema}")
    return ",".join(f"{arg.name}{index}" for arg in schema.schema_order_arguments())


generate_test_ir_arguments_base_ty_to_type_str_ = {
    BaseTy.Tensor: "Tensor",
    BaseTy.int: "int",
    BaseTy.float: "float",
    BaseTy.str: "str",
    BaseTy.Scalar: "int",
    BaseTy.ScalarType: "int",
    BaseTy.bool: "bool",
}


def generate_test_ir_arguments(
    schema: FunctionSchema,
) -> list[tuple[str, str | None]]:
    def ir_argument(arg: Argument) -> tuple[str, str | None]:
        t = arg.type
        add_optional = False
        if isinstance(t, OptionalType):
            t = t.elem
            add_optional = True
        if not isinstance(t, BaseType):
            raise AssertionError(f"Expected BaseType, got {type(t)}")
        type_str = None
        if t.name in generate_test_ir_arguments_base_ty_to_type_str_:
            type_str = generate_test_ir_arguments_base_ty_to_type_str_[t.name]
        if type_str and add_optional:
            type_str = f"{type_str}?"
        return ("%" + arg.name, type_str)

    return [ir_argument(arg) for arg in schema.schema_order_arguments()]


def generate_arg_extraction(schema: FunctionSchema) -> str:
    arg_populations = []
    for i, arg in enumerate(schema.schema_order_arguments()):
        maybe_method = ivalue_type_conversion_method(arg.type)
        if not maybe_method:
            raise AssertionError(
                f"No type conversion method for {arg.name}: {arg.type}"
            )
        is_reference, type_conversion_method = maybe_method
        reference = "&" if is_reference else ""
        arg_populations.append(
            f"const auto{reference} {arg.name} = p_node->Input({i}).{type_conversion_method}"
        )
    return ";\n    ".join(arg_populations) + ";"


def get_kernel_name(g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
    kernel = backend_index.get_kernel(g.functional)
    if g.structured or kernel is None:
        return cpp.name(g.functional.func)
    return kernel.kernel


def get_out_kernel_name(g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
    kernel = backend_index.get_kernel(g.out)
    if g.structured or kernel is None:
        return cpp.name(g.out.func)
    return kernel.kernel


def generate_non_out_variant_call(
    g: NativeFunctionsGroup, backend_index: BackendIndex
) -> str:
    schema = g.functional.func
    if schema.is_out_fn():
        raise AssertionError(f"Expected non-out function, got {schema}")
    kernel_name = get_kernel_name(g, backend_index)
    arg_names = (arg.name for arg in schema.schema_order_arguments())
    namespace_name = "cpu" if g.structured else "native"
    return f"at::{namespace_name}::{kernel_name}({','.join(arg_names)})"


def generate_call_to_view_ops(
    g: NativeFunctionsViewGroup, backend_index: BackendIndex
) -> str:
    schema = g.view.func
    kernel_name = cpp.name(schema)
    kernel = backend_index.get_kernel(g.view)
    if kernel:
        kernel_name = kernel.kernel
    arg_names = (arg.name for arg in schema.schema_order_arguments())
    namespace_name = "native"
    return f"at::{namespace_name}::{kernel_name}({','.join(arg_names)})"


def generate_out_variant_call(
    g: NativeFunctionsGroup, backend_index: BackendIndex
) -> str:
    schema = g.out.func
    if not schema.is_out_fn():
        raise AssertionError(f"Expected out function, got {schema}")
    arg_names = []
    kernel_name = get_out_kernel_name(g, backend_index)
    if g.structured:
        # structured op starts with the output tensor argument.
        arg_names = [out_arg.name for out_arg in schema.arguments.out]
    else:
        arg_names = []
    for arg in schema.arguments.non_out:
        if isinstance(arg, SelfArgument):
            arg_names.append(arg.argument.name)
        else:
            if not isinstance(arg, Argument):
                raise AssertionError(f"Expected Argument, got {type(arg)}")
            arg_names.append(arg.name)
    if not g.structured:
        if len(schema.arguments.out) != 1:
            raise AssertionError(
                f"Expected 1 out argument, got {len(schema.arguments.out)}"
            )
        arg_names.append(schema.arguments.out[0].name)
    cpp_arg_names = ",".join(arg_names)
    namespace_name = "cpu" if g.structured else "native"
    return f"at::{namespace_name}::{kernel_name}({cpp_arg_names})"


no_memory_resize_ops = frozenset(
    (
        "isin.Scalar_Tensor",
        "index_add",
        "dot",
        "vdot",
        "nuclear_norm",
        "histc",
        "l1_loss",
        "multi_margin_loss",
        "multilabel_margin_loss",
        "nll_loss",
        "nll_loss2d",
        "prod",
    )
)


def should_check_resize(schema: FunctionSchema) -> bool:
    schema_str = str(schema)
    type_variant_op_name = schema_str[: schema_str.find("(")]
    return type_variant_op_name not in no_memory_resize_ops


def op_name_from_group(g: NativeFunctionsGroup) -> str:
    return g.functional.func.name.name.base


class GenOpDispatcher:
    def out_variant(
        self, groups: Sequence[NativeFunctionsGroup], backend_index: BackendIndex
    ) -> str:
        if not groups:
            return ""
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                if not is_supported(g):
                    raise AssertionError(f"Unsupported function group: {g}")
                if not isinstance(g, NativeFunctionsGroup):
                    raise AssertionError(
                        f"Expected NativeFunctionsGroup, got {type(g)}"
                    )
                generated_type_variant = self.out_variant_op_generator(g, backend_index)
                generated_type_variants.append(generated_type_variant)
        op_name = op_name_from_group(groups[0])
        body = "\n".join(generated_type_variants)
        generated = f"""
REGISTER_OPERATOR_FUNCTOR(
    aten::{op_name},
    aten_{op_name},
    [](Node* n) -> SROperator {{
      {body}
      LogAndDumpSchema(n);
      return nullptr;
    }})
"""
        return generated

    def view(
        self, groups: Sequence[NativeFunctionsViewGroup], backend_index: BackendIndex
    ) -> str:
        if not groups:
            return ""
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                if not is_supported(g):
                    raise AssertionError(f"Unsupported view group: {g}")
                if not isinstance(g, NativeFunctionsViewGroup):
                    raise AssertionError(
                        f"Expected NativeFunctionsViewGroup, got {type(g)}"
                    )
                generated_type_variant = self.view_op_generator(g, backend_index)
                generated_type_variants.append(generated_type_variant)
        op_name = config.func_name_base_str(groups[0])
        body = "\n".join(generated_type_variants)
        generated = f"""
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::{op_name},
    aten_{op_name},
    [](Node* n) -> SROperator {{
      {body}
      LogAndDumpSchema(n);
      return nullptr;
    }});
"""
        return generated

    def out_variant_op_generator(
        self, g: NativeFunctionsGroup, backend_index: BackendIndex
    ) -> str:
        functional = g.functional
        schema = str(functional.func)
        populated_argument = generate_arg_extraction(g.functional.func)
        functional_variant_call = generate_non_out_variant_call(g, backend_index)
        if len(g.out.func.arguments.out) != 1:
            raise AssertionError(
                f"Expected 1 out argument, got {len(g.out.func.arguments.out)}"
            )
        out_variable_name = str(g.out.func.arguments.out[0].name)
        out_variant_call = generate_out_variant_call(g, backend_index)
        generated = f"""
      if (n->matches(torch::schema("aten::{schema}"))) {{
        return [](ProcessedNode* p_node) {{
          {populated_argument}
          if (p_node->Output(0).isNone()) {{
            p_node->Output(0) = {functional_variant_call};
            return;
          }}
          auto& {out_variable_name} = p_node->Output(0).toTensor();
          fastResizeToZero({out_variable_name});
          {out_variant_call};
        }};
      }}"""
        return generated

    def view_op_generator(
        self, g: NativeFunctionsViewGroup, backend_index: BackendIndex
    ) -> str:
        schema = str(g.view.func)
        populated_argument = generate_arg_extraction(g.view.func)
        functional_variant_call = generate_call_to_view_ops(g, backend_index)
        generated = f"""
      if (n->matches(torch::schema("aten::{schema}"))) {{
        return [](ProcessedNode* p_node) {{
          {populated_argument}
            p_node->Output(0) = {functional_variant_call};
        }};
      }}"""
        return generated


class GenOpTestCase:
    def out_variant(self, groups: Sequence[NativeFunctionsGroup]) -> str:
        if not groups:
            return ""
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                if not is_supported(g):
                    raise AssertionError(f"Unsupported function group: {g}")
                if not isinstance(g, NativeFunctionsGroup):
                    raise AssertionError(
                        f"Expected NativeFunctionsGroup, got {type(g)}"
                    )
                generated_type_variant = self.out_variant_op_test_case_generator(g)
                generated_type_variants.append(generated_type_variant)
        return "\n".join(generated_type_variants)

    def view(self, groups: Sequence[NativeFunctionsViewGroup]) -> str:
        if not groups:
            return ""
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                if not is_supported(g):
                    raise AssertionError(f"Unsupported view group: {g}")
                if not isinstance(g, NativeFunctionsViewGroup):
                    raise AssertionError(
                        f"Expected NativeFunctionsViewGroup, got {type(g)}"
                    )
                generated_type_variant = self.view_op_test_case_generator(g)
                generated_type_variants.append(generated_type_variant)
        return "\n".join(generated_type_variants)

    def out_variant_op_test_case_generator(self, g: NativeFunctionsGroup) -> str:
        schema = g.functional.func
        schema_str = str(schema)
        if schema_str.find("(") <= 0:
            raise AssertionError(f"Invalid schema string: {schema_str}")
        type_variant_op_name = schema_str[: schema_str.find("(")].replace(".", "_")
        op_name = op_name_from_group(g)
        if not type_variant_op_name.startswith(op_name):
            raise AssertionError(
                f"Type variant op name {type_variant_op_name} doesn't start with {op_name}"
            )

        arg_types = generate_test_ir_arguments(schema)
        arg_declarations = ", ".join(
            (
                arg_name if arg_type is None else f"{arg_name}: {arg_type}"
                for arg_name, arg_type in arg_types
            )
        )
        arg_names = ", ".join((arg_name for arg_name, _ in arg_types))
        if not (
            len(schema.returns) == 1
            and isinstance(schema.returns[0].type, BaseType)
            and schema.returns[0].type.name is BaseTy.Tensor
        ):
            raise AssertionError(f"Expected single Tensor return, got {schema.returns}")
        test_value_definitions = generate_test_value_definitions(schema, 0)
        test_value_names = generate_test_value_names(schema, 0)
        test_value_definitions2 = generate_test_value_definitions(schema, 1)
        test_value_names2 = generate_test_value_names(schema, 1)
        check_resize = "true" if should_check_resize(schema) else "false"
        generated = f"""
TEST(StaticRuntime, autogen_{type_variant_op_name}) {{
  const std::string script = R"IR(
    graph({arg_declarations}):
        %bias: None = prim::Constant()
        %ret = aten::{op_name}({arg_names})
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  {test_value_definitions}
  std::vector<IValue> args{{{test_value_names}}};
  testStaticRuntime(script, args, {{}}, /*use_allclose=*/false, /*use_equalnan=*/false, /*check_resize=*/{check_resize});

  {test_value_definitions2}
  std::vector<IValue> args2{{{test_value_names2}}};
  testStaticRuntime(script, args, args2, /*use_allclose=*/false, /*use_equalnan=*/false, /*check_resize=*/{check_resize});

}}
"""
        return generated

    def view_op_test_case_generator(self, g: NativeFunctionsViewGroup) -> str:
        schema = g.view.func
        schema_str = str(schema)
        if schema_str.find("(") <= 0:
            raise AssertionError(f"Invalid schema string: {schema_str}")
        type_variant_op_name = schema_str[: schema_str.find("(")].replace(".", "_")
        op_name = g.view.root_name
        if not type_variant_op_name.startswith(op_name):
            raise AssertionError(
                f"Type variant op name {type_variant_op_name} doesn't start with {op_name}"
            )

        arg_types = generate_test_ir_arguments(schema)
        arg_declarations = ", ".join(
            (
                arg_name if arg_type is None else f"{arg_name}: {arg_type}"
                for arg_name, arg_type in arg_types
            )
        )
        arg_names = ", ".join((arg_name for arg_name, _ in arg_types))
        if not (
            len(schema.returns) == 1
            and isinstance(schema.returns[0].type, BaseType)
            and schema.returns[0].type.name is BaseTy.Tensor
        ):
            raise AssertionError(f"Expected single Tensor return, got {schema.returns}")
        test_value_definitions = generate_test_value_definitions(schema, 0)
        test_value_names = generate_test_value_names(schema, 0)
        generated = f"""
TEST(StaticRuntime, autogen_{type_variant_op_name}) {{
  const std::string script = R"IR(
    graph({arg_declarations}):
        %bias: None = prim::Constant()
        %ret = aten::{op_name}({arg_names})
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  {test_value_definitions}
  std::vector<IValue> args{{{test_value_names}}};
  testStaticRuntime(script, args);
}}
"""

        return generated
<|endoftext|>from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from torchgen.selective_build.operator import (
    merge_debug_info,
    merge_operator_dicts,
    SelectiveBuildOperator,
    strip_operator_overload_name,
)


if TYPE_CHECKING:
    from torchgen.model import NativeFunction


# A SelectiveBuilder holds information extracted from the selective build
# YAML specification.
#
# It includes information about the build's selectivity, the debug_info
# associated with this selective build (opaque string), and the set of
# operators that should be included in the build.
#
@dataclass(frozen=True)
class SelectiveBuilder:
    # If true, then the build is not selective, and includes all
    # operators.
    include_all_operators: bool

    # Debug Information at the selective/custom build level.
    _debug_info: tuple[str, ...] | None

    # A dictionary of operator -> operator metadata.
    operators: dict[str, SelectiveBuildOperator]

    # A dictionary of selected kernel tags and dtypes. Typically a
    # PyTorch Operator Kernel (function) may have many code paths
    # that are specialized for many many Tensor dtypes, so it's not
    # one per kernel function, but there could be many per kernel
    # function. The tag isn't a kernel function name, but some fragment
    # of the kernel function implementation itself.
    kernel_metadata: dict[str, list[str]]

    # ExecuTorch only. A dictionary of kernel tag -> list of (list of input
    # dtypes for tensor-like input args).
    # This is from selective.yaml
    et_kernel_metadata: dict[str, list[str]]

    # A set of all the custom torch bind classes used by the selected models
    # Stored as a set internally to remove duplicates proactively, but written
    # as a list to yamls
    custom_classes: set[str]

    # A set of all the build features used by the selected models
    # Stored as a set internally to remove duplicates proactively, but written
    # as a list to yamls
    build_features: set[str]

    # If true, then fragments for all dtypes for all kernel functions
    # are included as well as all custom classes. This is typically set when any one of the
    # operator lists is generated from a mechanism other than
    # tracing based selective build.
    include_all_non_op_selectives: bool

    @staticmethod
    def get_nop_selector() -> SelectiveBuilder:
        return SelectiveBuilder.from_yaml_dict({"include_all_operators": True})

    @staticmethod
    def from_yaml_dict(data: dict[str, object]) -> SelectiveBuilder:
        valid_top_level_keys = {
            "include_all_non_op_selectives",
            "include_all_operators",
            "debug_info",
            "operators",
            "kernel_metadata",
            "et_kernel_metadata",
            "custom_classes",
            "build_features",
        }
        top_level_keys = set(data.keys())
        if len(top_level_keys - valid_top_level_keys) > 0:
            raise Exception(  # noqa: TRY002
                "Got unexpected top level keys: {}".format(
                    ",".join(top_level_keys - valid_top_level_keys),
                )
            )
        include_all_operators = data.get("include_all_operators", False)
        if not isinstance(include_all_operators, bool):
            raise AssertionError(
                f"Expected 'include_all_operators' to be bool, got {type(include_all_operators)}"
            )

        debug_info = None
        if "debug_info" in data:
            di_list = data["debug_info"]
            if not isinstance(di_list, list):
                raise AssertionError(
                    f"Expected 'debug_info' to be list, got {type(di_list)}"
                )

            debug_info = tuple(str(x) for x in di_list)

        operators = {}
        operators_dict = data.get("operators", {})
        if not isinstance(operators_dict, dict):
            raise AssertionError(
                f"Expected 'operators' to be dict, got {type(operators_dict)}"
            )

        for k, v in operators_dict.items():
            operators[k] = SelectiveBuildOperator.from_yaml_dict(k, v)

        kernel_metadata = {}
        kernel_metadata_dict = data.get("kernel_metadata", {})
        if not isinstance(kernel_metadata_dict, dict):
            raise AssertionError(
                f"Expected 'kernel_metadata' to be dict, got {type(kernel_metadata_dict)}"
            )

        for k, v in kernel_metadata_dict.items():
            kernel_metadata[str(k)] = [str(dtype) for dtype in v]

        et_kernel_metadata = data.get("et_kernel_metadata", {})
        if not isinstance(et_kernel_metadata, dict):
            raise AssertionError(
                f"Expected 'et_kernel_metadata' to be dict, got {type(et_kernel_metadata)}"
            )

        custom_classes = data.get("custom_classes", [])
        if not isinstance(custom_classes, Iterable):
            raise AssertionError(
                f"Expected 'custom_classes' to be Iterable, got {type(custom_classes)}"
            )
        custom_classes = set(custom_classes)

        build_features = data.get("build_features", [])
        if not isinstance(build_features, Iterable):
            raise AssertionError(
                f"Expected 'build_features' to be Iterable, got {type(build_features)}"
            )
        build_features = set(build_features)

        include_all_non_op_selectives = data.get("include_all_non_op_selectives", False)
        if not isinstance(include_all_non_op_selectives, bool):
            raise AssertionError(
                f"Expected 'include_all_non_op_selectives' to be bool, "
                f"got {type(include_all_non_op_selectives)}"
            )

        return SelectiveBuilder(
            include_all_operators,
            debug_info,
            operators,
            kernel_metadata,
            et_kernel_metadata,
            custom_classes,  # type: ignore[arg-type]
            build_features,  # type: ignore[arg-type]
            include_all_non_op_selectives,
        )

    @staticmethod
    def from_yaml_str(config_contents: str) -> SelectiveBuilder:
        contents = yaml.safe_load(config_contents)
        return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_yaml_path(config_path: str) -> SelectiveBuilder:
        with open(config_path) as f:
            contents = yaml.safe_load(f)
            return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_legacy_op_registration_allow_list(
        allow_list: set[str], is_root_operator: bool, is_used_for_training: bool
    ) -> SelectiveBuilder:
        operators = {}
        for op in allow_list:
            operators[op] = {
                "name": op,
                "is_root_operator": is_root_operator,
                "is_used_for_training": is_used_for_training,
                "include_all_overloads": True,
            }
        return SelectiveBuilder.from_yaml_dict(
            {
                "operators": operators,
                "include_all_non_op_selectives": True,
            }
        )

    def is_operator_selected(self, name: str) -> bool:
        if self.include_all_operators:
            return True

        if name in self.operators:
            return True
        name = strip_operator_overload_name(name)
        return name in self.operators and self.operators[name].include_all_overloads

    def is_native_function_selected(self, func: NativeFunction) -> bool:
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected(op_name)

    def is_operator_selected_for_training(self, name: str) -> bool:
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True

        not_training_op = SelectiveBuildOperator(
            name="",
            is_root_operator=False,
            is_used_for_training=False,
            include_all_overloads=False,
            _debug_info=None,
        )
        op = not_training_op
        if name in self.operators:
            op = self.operators[name]

        name = strip_operator_overload_name(name)
        base_op = not_training_op
        if name in self.operators:
            base_op = self.operators[name]

        return op.is_used_for_training or (
            base_op.include_all_overloads and base_op.is_used_for_training
        )

    def is_native_function_selected_for_training(self, func: NativeFunction) -> bool:
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected_for_training(op_name)

    def is_root_operator(self, name: str) -> bool:
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True

        if name in self.operators:
            op: SelectiveBuildOperator = self.operators[name]
            return op.is_root_operator
        name = strip_operator_overload_name(name)
        if name not in self.operators:
            return False
        base_op: SelectiveBuildOperator = self.operators[name]
        return base_op.include_all_overloads and base_op.is_root_operator

    def is_kernel_dtype_selected(self, kernel_tag: str, dtype: str) -> bool:
        if self.include_all_operators or self.include_all_non_op_selectives:
            return True

        return (
            kernel_tag in self.kernel_metadata
            and dtype in self.kernel_metadata[kernel_tag]
        )

    def et_get_selected_kernels(self, op_name: str, kernel_key: list[str]) -> list[str]:
        """
        Return a list of kernel keys that cover the used ops
        """
        # If no kernel metadata, either it's implied by include_all_operators=True or the op is not used.
        if op_name not in self.et_kernel_metadata:
            return kernel_key if self.include_all_operators else []
        # Otherwise, only return the specific kernel keys.

        result_set = set()

        for model_kernel_keys in self.et_kernel_metadata[op_name]:
            key_found = False
            for key in kernel_key:
                # Don't compare the version for now
                if (
                    key != "default"
                    and key.split("/")[1] == model_kernel_keys.split("/")[1]
                ):
                    result_set.add(key)
                    key_found = True
                    break
            if not key_found:
                if "default" not in kernel_key:
                    raise Exception("Missing kernel for the model")  # noqa: TRY002
                else:
                    result_set.add("default")

        return list(result_set)

    def to_dict(self) -> dict[str, object]:
        ret: dict[str, object] = {
            "include_all_non_op_selectives": self.include_all_non_op_selectives,
            "include_all_operators": self.include_all_operators,
        }
        operators = {}
        for op_name, op in self.operators.items():
            operators[op_name] = op.to_dict()
        ret["operators"] = operators

        if self._debug_info is not None:
            ret["debug_info"] = sorted(self._debug_info)

        ret["kernel_metadata"] = {
            k: sorted(v) for (k, v) in self.kernel_metadata.items()
        }

        ret["et_kernel_metadata"] = self.et_kernel_metadata

        ret["custom_classes"] = sorted(self.custom_classes)

        ret["build_features"] = sorted(self.build_features)

        return ret


def merge_kernel_metadata(
    lhs: dict[str, list[str]],
    rhs: dict[str, list[str]],
) -> dict[str, list[str]]:
    kernel_metadata: dict[str, list[str]] = {}
    for tag_name, dtypes in list(lhs.items()) + list(rhs.items()):
        dtypes_copy = set(dtypes)
        if tag_name in kernel_metadata:
            dtypes_copy |= set(kernel_metadata[tag_name])

        kernel_metadata[tag_name] = list(dtypes_copy)

    return kernel_metadata


def merge_et_kernel_metadata(
    lhs: dict[str, list[str]],
    rhs: dict[str, list[str]],
) -> dict[str, list[str]]:
    merge_et_kernel_metadata: dict[str, set[str]] = defaultdict(set)
    for op in list(lhs.keys()) + list(rhs.keys()):
        merge_et_kernel_metadata[op].update(lhs.get(op, []))
        merge_et_kernel_metadata[op].update(rhs.get(op, []))

    return {op: sorted(val) for op, val in merge_et_kernel_metadata.items()}


def combine_selective_builders(
    lhs: SelectiveBuilder, rhs: SelectiveBuilder
) -> SelectiveBuilder:
    include_all_operators = lhs.include_all_operators or rhs.include_all_operators
    debug_info = merge_debug_info(lhs._debug_info, rhs._debug_info)
    operators = merge_operator_dicts(lhs.operators, rhs.operators)
    kernel_metadata = merge_kernel_metadata(lhs.kernel_metadata, rhs.kernel_metadata)
    et_kernel_metadata = merge_et_kernel_metadata(
        lhs.et_kernel_metadata, rhs.et_kernel_metadata
    )
    include_all_non_op_selectives = (
        lhs.include_all_non_op_selectives or rhs.include_all_non_op_selectives
    )
    custom_classes = lhs.custom_classes.union(rhs.custom_classes)
    build_features = lhs.build_features.union(rhs.build_features)
    return SelectiveBuilder(
        include_all_operators,
        debug_info,
        operators,
        kernel_metadata,
        et_kernel_metadata,
        custom_classes,
        build_features,
        include_all_non_op_selectives,
    )


def op_name_from_native_function(f: NativeFunction) -> str:
    # This was originally read from the 'operator_name_with_overload' field in the
    # declaration dict, which was the part before the first '(' in 'schema_string'.
    return f"{f.namespace}::{f.func.name}"
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass


# This class holds information about a single operator used to determine
# the outcome of a selective/custom PyTorch build that doesn't include
# registration code for all the supported operators. This is done to
# reduce the size of the generated binary so that it can be deployed in
# situations where binary size comes at a premium.
#
@dataclass(frozen=True)
class SelectiveBuildOperator:
    # The name of the operator. This includes the aten::, etc... prefix
    # The operator name may or may not have the overload name. If this
    # operator name does not specify an overload name, the way to determine
    # if this entry refers to the family of operators with this base name
    # or just the operator with this name is to look at the value of the
    # 'include_all_overloads' flag in this class.
    name: str

    # True if this is a root operator (i.e. called directly from a
    # TorchScript model, etc...). An operator is considered to be a
    # root operator if it is called directly from any one of the models
    # that this instance of the pytorch library was built for. Hence, it
    # may not be a root operator in all of the models that are used in
    # this instance of the pytorch library.
    is_root_operator: bool

    # Is this operator used for on-device training? If True, then we need to
    # use the information to generate code in VariableType_N.cpp for registration
    # of training related operators. Again, this is True if this operator
    # is used for training in one or more models used by this instance of the
    # pytorch library.
    is_used_for_training: bool

    # If True, it indicates that this operator instance (object) refers to an
    # operator without the overload name and should apply to all overloads
    # which have this operator name as the base name. This flag is applicable
    # only for objects that have operator names without a DOT (period) character
    # in them.
    #
    # Note: This flag is a temporary workaround to grandfather in the current
    # static selective (custom) build mechanism, which largely ignores overload
    # names when determining whether to select operators for registration
    # purposes.
    include_all_overloads: bool

    # Debug Information at the operator level
    _debug_info: tuple[str, ...] | None

    @staticmethod
    def from_yaml_dict(
        op_name: str, op_info: dict[str, object]
    ) -> SelectiveBuildOperator:
        allowed_keys = {
            "name",
            "is_root_operator",
            "is_used_for_training",
            "include_all_overloads",
            "debug_info",
        }

        if len(set(op_info.keys()) - allowed_keys) > 0:
            raise Exception(  # noqa: TRY002
                "Got unexpected top level keys: {}".format(
                    ",".join(set(op_info.keys()) - allowed_keys),
                )
            )

        if "name" in op_info:
            if op_name != op_info["name"]:
                raise AssertionError(
                    f"op_name mismatch: {op_name} != {op_info['name']}"
                )

        is_root_operator = op_info.get("is_root_operator", True)
        if not isinstance(is_root_operator, bool):
            raise AssertionError(
                f"Expected 'is_root_operator' to be bool, got {type(is_root_operator)}"
            )

        is_used_for_training = op_info.get("is_used_for_training", True)
        if not isinstance(is_used_for_training, bool):
            raise AssertionError(
                f"Expected 'is_used_for_training' to be bool, got {type(is_used_for_training)}"
            )

        include_all_overloads = op_info.get("include_all_overloads", True)
        if not isinstance(include_all_overloads, bool):
            raise AssertionError(
                f"Expected 'include_all_overloads' to be bool, got {type(include_all_overloads)}"
            )

        debug_info: tuple[str, ...] | None = None
        if "debug_info" in op_info:
            di_list = op_info["debug_info"]
            if not isinstance(di_list, list):
                raise AssertionError(
                    f"Expected 'debug_info' to be list, got {type(di_list)}"
                )
            debug_info = tuple(str(x) for x in di_list)

        return SelectiveBuildOperator(
            name=op_name,
            is_root_operator=is_root_operator,
            is_used_for_training=is_used_for_training,
            include_all_overloads=include_all_overloads,
            _debug_info=debug_info,
        )

    @staticmethod
    def from_legacy_operator_name_without_overload(
        name: str,
    ) -> SelectiveBuildOperator:
        return SelectiveBuildOperator(
            name=name,
            is_root_operator=True,
            is_used_for_training=True,
            include_all_overloads=True,
            _debug_info=None,
        )

    def to_dict(self) -> dict[str, object]:
        ret: dict[str, object] = {
            "is_root_operator": self.is_root_operator,
            "is_used_for_training": self.is_used_for_training,
            "include_all_overloads": self.include_all_overloads,
        }
        if self._debug_info is not None:
            ret["debug_info"] = self._debug_info

        return ret


def merge_debug_info(
    lhs: tuple[str, ...] | None,
    rhs: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    # Ensure that when merging, each entry shows up just once.
    if lhs is None and rhs is None:
        return None

    return tuple(set((lhs or ()) + (rhs or ())))


def combine_operators(
    lhs: SelectiveBuildOperator, rhs: SelectiveBuildOperator
) -> SelectiveBuildOperator:
    if str(lhs.name) != str(rhs.name):
        raise Exception(  # noqa: TRY002
            f"Expected both arguments to have the same name, but got '{str(lhs.name)}' and '{str(rhs.name)}' instead"
        )

    return SelectiveBuildOperator(
        name=lhs.name,
        # Consider this operator to be a root operator if it is a
        # root operator in any of the models used in this instance of
        # the pytorch library.
        is_root_operator=lhs.is_root_operator or rhs.is_root_operator,
        # Consider this operator to be a training operator if it is
        # an operator used for training in any of the models used
        # in this instance of the pytorch library.
        is_used_for_training=lhs.is_used_for_training or rhs.is_used_for_training,
        include_all_overloads=lhs.include_all_overloads or rhs.include_all_overloads,
        _debug_info=merge_debug_info(lhs._debug_info, rhs._debug_info),
    )


def merge_operator_dicts(
    lhs: dict[str, SelectiveBuildOperator],
    rhs: dict[str, SelectiveBuildOperator],
) -> dict[str, SelectiveBuildOperator]:
    operators: dict[str, SelectiveBuildOperator] = {}
    for op_name, op in list(lhs.items()) + list(rhs.items()):
        new_op = op
        if op_name in operators:
            new_op = combine_operators(operators[op_name], op)

        operators[op_name] = new_op

    return operators


def strip_operator_overload_name(op_name: str) -> str:
    return op_name.split(".", maxsplit=1)[0]
<|endoftext|>from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager


if TYPE_CHECKING:
    from collections.abc import Sequence


# Note [Manual Backend kernels]
# For these ops, we want to manually register to dispatch key Backend and
# skip codegen-ed registration to all keys before Backend.
# For codegen this means:
#   - op set below must match ops with manual_kernel_registration=True in native_functions.yaml
#     where we skip codegen backend kernels
#   - all ops below are part of MANUAL_AUTOGRAD to skip codegen Autograd kernel registration
#   - all ops below are part of MANUAL_TRACER to skip codegen Tracer kernel registration
# Note: we still register to dispatch key Profiler for these ops, keeping it untouched for now.
# You can find the manual registration in torch/csrc/autograd/VariableTypeManual.cpp
MANUAL_BACKEND = {
    "options",
    "data",
    "set_data",
    "is_leaf",
    "output_nr",
    "_version",
    "retain_grad",
    "_backward",
    "requires_grad_",
}

# For these ops we want to skip the codegen-ed registration to both Autograd and Tracer keys.
# You can find the manual registration in torch/csrc/autograd/VariableTypeManual.cpp
MANUAL_AUTOGRAD_AND_TRACER = {
    "resize_",
    "resize_as_",
    "detach",
    "detach_",
    "copy_",
    "_fw_primal",
    "_make_dual",
}

# Currently MANUAL_AUTOGRAD and MANUAL_TRACER share the same set of ops:
#   union(MANUAL_BACKEND, MANUAL_AUTOGRAD_AND_TRACER)
# You can find the manual registration in torch/csrc/autograd/VariableTypeManual.cpp
MANUAL_AUTOGRAD = MANUAL_TRACER = MANUAL_BACKEND | MANUAL_AUTOGRAD_AND_TRACER

# These functions we don't want to record for tracing, because we always want
# to trace their constituent parts.  This is a temporary hack in lieue
# of proper scopes, where subsequent compilation passes can ask for the unfolding
# on demand.  Only concrete ATen methods can be disabled this way; it will have
# NO EFFECT otherwise.
DONT_RECORD_TRACE = {
    "convolution",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "lstm_cell",
    "gru_cell",
    "rnn_tanh_cell",
    "rnn_relu_cell",
    # FIXME: figure out a better way when we support sparse tensors in jit
    "_coalesced",
}


def should_trace(f: NativeFunction) -> bool:
    # Operations involving Storage or Type are not traceable at the moment
    if any(
        str(arg.type) in {"Storage", "Type"} for arg in f.func.schema_order_arguments()
    ):
        return False
    # We can't trace functions which don't have any Tensor or TensorList returns
    if not any(r.type.is_tensor_like() for r in f.func.returns):
        return False
    return f.func.name.name.base not in DONT_RECORD_TRACE


SELECT = CodeTemplate(
    """\

if (${cond}) {
  ${true}
} else {
  ${false}
}
"""
)

OP_NAME = CodeTemplate(
    """\
op_name = c10::Symbol::fromQualString("aten::${trace_name}");
"""
)

# These functions have their names recorded under trace renamed,
RENAME_TRACE = {
    "zero": "zeros_like",  # replacing aten::zero_ with aten::zeros_like
    "fill": "full_like",  # replacing aten::fill_ with aten::full_like
}


def format_trace_op_name(f: NativeFunction) -> str:
    # TODO: byte-for-byte compatible with old codegen behavior - should clean up
    if (
        f.func.kind() in (SchemaKind.functional, SchemaKind.out)
        or f.func.name.name.dunder_method
    ):
        # special case for *_out functions: the in-place and out-of-place ops
        # are overloaded with the same name in the JIT
        trace_name = str(f.func.name.name)
        trace_name = RENAME_TRACE.get(trace_name, trace_name)
        return OP_NAME.substitute(trace_name=trace_name)

    # otherwise, this is an in-place op and we need to emit both in- and
    # out-of-place versions
    outplace_trace_name = f.func.name.name.base
    inplace_trace_name = cpp.name(f.func)
    outplace_trace_name = RENAME_TRACE.get(outplace_trace_name, outplace_trace_name)
    inplace_trace_name = RENAME_TRACE.get(inplace_trace_name, inplace_trace_name)

    return SELECT.substitute(
        cond="tracer_state->force_outplace",
        true=OP_NAME.substitute(trace_name=outplace_trace_name),
        false=OP_NAME.substitute(trace_name=inplace_trace_name),
    )


ADD_TRACE_INPUT = CodeTemplate("""jit::tracer::addInputs(node, "${name}", ${input});""")


def format_trace_inputs(f: NativeFunction) -> str:
    def dispatch_trace_input(arg: Argument | TensorOptionsArguments) -> Sequence[str]:
        if isinstance(arg, TensorOptionsArguments):
            name = "options"
            return [
                ADD_TRACE_INPUT.substitute(
                    name=name, input="c10::optTypeMetaToScalarType(options.dtype_opt())"
                ),
                ADD_TRACE_INPUT.substitute(name=name, input="options.layout()"),
                ADD_TRACE_INPUT.substitute(name=name, input="options.device()"),
                ADD_TRACE_INPUT.substitute(name=name, input="options.pinned_memory()"),
            ]
        else:
            name = arg.name
            if str(arg.type) == "Tensor?[]":
                return [f'jit::tracer::addInputs(node, "{name}", {name});']
            else:
                return [ADD_TRACE_INPUT.substitute(name=name, input=name)]

    args: list[Argument | TensorOptionsArguments] = list(
        f.func.schema_order_arguments()
    )

    if f.func.is_out_fn():
        # *_out functions take the result as a separate argument, but we don't want to
        # trace that argument directly. Instead, we trace its TensorOptions.
        # So first, we need to remove the out argument from the list of arguments to trace.
        num_out_args = len(f.func.arguments.out)
        args = args[:-num_out_args]

    trace_inputs = itertools.chain.from_iterable(
        dispatch_trace_input(arg) for arg in args
    )

    if f.func.is_out_fn():
        # for *_out functions, handle the result argument differently for inplace/outplace.
        # For inplace: just add the input to the end to confirm with the JIT schema
        inplace = [
            ADD_TRACE_INPUT.substitute(
                name=f.func.arguments.out[i].name, input=f.func.arguments.out[i].name
            )
            # pyrefly: ignore [unbound-name]
            for i in range(num_out_args)
        ]

        # for outplace: do nothing, except if the function is a factory.
        # Factories are a bit special because their out-of-place overloads
        # take an extra TensorOptions argument, which is missing in the _out function
        has_tensor_return = any(r.type.is_tensor_like() for r in f.func.returns)
        has_tensor_input_arg = any(
            a.type.is_tensor_like() for a in f.func.arguments.flat_non_out
        )
        is_factory_method = f.category_override == "factory" or (
            has_tensor_return and not has_tensor_input_arg
        )

        # HACK: preserve old codegen behavior - the old codegen set the `is_factory_method`
        # flag for the whole family of ops with the same basename if any of them is a
        # factory method. For most cases the whole family of ops are indeed all factory
        # method - 'normal' is the only exception. So we handle it specially here to avoid
        # cloning the old logic.
        if f.func.name.name.base == "normal":
            is_factory_method = True

        if is_factory_method:
            outplace = [
                ADD_TRACE_INPUT.substitute(
                    name="out",
                    input="c10::optTypeMetaToScalarType(out.options().dtype_opt())",
                ),
                ADD_TRACE_INPUT.substitute(name="out", input="out.options().layout()"),
                ADD_TRACE_INPUT.substitute(name="out", input="out.options().device()"),
                ADD_TRACE_INPUT.substitute(
                    name="out", input="out.options().pinned_memory()"
                ),
            ]
        else:
            outplace = []

        trace_inputs = itertools.chain(
            trace_inputs,
            [
                SELECT.substitute(
                    cond="tracer_state->force_outplace",
                    true="\n".join(outplace),
                    false="\n".join(inplace),
                )
            ],
        )

    return "\n".join(trace_inputs)


# `torch.jit.trace` have undocumented keyword argument `_force_outplace`,
# which force jit to replace functions with outplace variants (for
# example `aten::add_` becomes `aten::add`).
#
# This replacement implemented in-place with minimum modifications of
# arguments stack (as it assumes that outplace call has the same arguments
# as inplace version).
#
# However there are no such substitutions available for `aten::fill_`
# and `aten::zero_` operators, as we never implemented `aten::fill`
# and `aten::zero`. So jit tracing hack replacing `aten::zero_` with
# `aten::zeros_like` and replacing `aten::fill_` with `aten::full_like`.
#
# But as they potentially can have different arguments, we also have
# to hack into the stack and add missing ones.
#
# A possible alternative would be:
#
#  - Add `aten::fill` and `aten::zero`
#
#  - Or keep `aten::zeros_like` arguments aligned with `aten::zero_`
# arguments (inside of the `native_functions.yaml`)
RENAME_TRACE_ADD_ARGS = {
    "fill": """\
    jit::tracer::addInputs(node, "options", ::std::optional<ScalarType>());
    jit::tracer::addInputs(node, "options", layout_or_default(::std::nullopt));
    jit::tracer::addInputs(node, "options", device_or_default(::std::nullopt));
    jit::tracer::addInputs(node, "options", pinned_memory_or_default(::std::nullopt));
    ::std::optional<MemoryFormat> memory_format = c10::MemoryFormat::Preserve;
    jit::tracer::addInputs(node, "memory_format", memory_format);
""",
    "zero": """\
    jit::tracer::addInputs(node, "options", ::std::optional<ScalarType>());
    jit::tracer::addInputs(node, "options", layout_or_default(::std::nullopt));
    jit::tracer::addInputs(node, "options", device_or_default(::std::nullopt));
    jit::tracer::addInputs(node, "options", pinned_memory_or_default(::std::nullopt));
    ::std::optional<MemoryFormat> memory_format = c10::MemoryFormat::Preserve;
    jit::tracer::addInputs(node, "memory_format", memory_format);
""",
}

INPLACE_GUARD = CodeTemplate(
    """\
jit::tracer::ensureUniqueIfOutOfPlaced("${name}", ${mutable_input});
"""
)

PRE_RECORD_TRACE = CodeTemplate(
    """\
torch::jit::Node* node = nullptr;
std::shared_ptr<jit::tracer::TracingState> tracer_state;
if (jit::tracer::isTracing()) {
  tracer_state = jit::tracer::getTracingState();
  at::Symbol op_name;
  ${set_op_name}
  node = tracer_state->createNode(op_name, /*num_outputs=*/0);
  jit::tracer::recordSourceLocation(node);
  ${add_trace_inputs}
  tracer_state->insertNode(node);
  ${inplace_guard}
  jit::tracer::setTracingState(nullptr);
}
"""
)


def format_prerecord_trace(f: NativeFunction) -> str:
    if not should_trace(f):
        return ""

    # TODO: clean up old codegen behavior
    is_inplace = (
        f.func.kind() in (SchemaKind.inplace, SchemaKind.out)
        and not f.func.name.name.dunder_method
    )
    add_args = (
        RENAME_TRACE_ADD_ARGS.get(f.func.name.name.base, "") if is_inplace else ""
    )
    additional_inputs = (
        SELECT.substitute(
            cond="tracer_state->force_outplace",
            true=add_args,
            false="",
        )
        if add_args
        else ""
    )

    return PRE_RECORD_TRACE.substitute(
        set_op_name=format_trace_op_name(f),
        add_trace_inputs=format_trace_inputs(f) + additional_inputs,
        inplace_guard=INPLACE_GUARD.substitute(
            name=cpp.name(f.func),
            mutable_input=f.func.arguments.out[0].name
            if f.func.arguments.out
            else "self",
        )
        if is_inplace
        else "",
    )


POST_RECORD_TRACE = CodeTemplate(
    """\
if (tracer_state) {
  jit::tracer::setTracingState(std::move(tracer_state));
  ${add_trace_outputs}
}
"""
)


def format_postrecord_trace(f: NativeFunction) -> str:
    if not should_trace(f):
        return ""

    # For outplacing ops, *_out overloads require special handling to move the
    # output *argument* to a return value
    if f.func.is_out_fn():
        output_names_outplace = [arg.name for arg in f.func.arguments.out]
        output_names_inplace = cpp.return_names(f)

        # Code size optimization: the common case is that the return value is
        # the same for both variants
        if output_names_outplace == output_names_inplace:
            outputs = [
                f"jit::tracer::addOutput(node, {n});" for n in output_names_outplace
            ]
            return POST_RECORD_TRACE.substitute(add_trace_outputs=outputs)

        selection = SELECT.substitute(
            cond="force_outplace",
            true="\n".join(
                f"jit::tracer::addOutput(node, {n});" for n in output_names_outplace
            ),
            false="\n".join(
                f"jit::tracer::addOutput(node, {n});" for n in output_names_inplace
            ),
        )
        return POST_RECORD_TRACE.substitute(add_trace_outputs=selection)
    else:
        output_names = cpp.return_names(f)
        outputs = [f"jit::tracer::addOutput(node, {n});" for n in output_names]
        return POST_RECORD_TRACE.substitute(add_trace_outputs=outputs)


def tie_return_values(f: NativeFunction) -> str:
    if len(f.func.returns) == 1:
        return f"auto {f.func.returns[0].name or 'result'}"
    names = cpp.return_names(f)
    return f"auto [{', '.join(names)}]"


def get_return_value(f: NativeFunction) -> str:
    names = cpp.return_names(f)
    if len(f.func.returns) == 1:
        return names[0]
    if f.func.kind() == SchemaKind.out:
        return f"std::forward_as_tuple({', '.join(names)})"
    else:
        moved = ", ".join(f"std::move({name})" for name in names)
        return f"std::make_tuple({moved})"


TRACE_DISPATCH = CodeTemplate(
    """\
${assign_return_values}at::_ops::${unambiguous_name}::redispatch(${unpacked_args});"""
)


def emit_trace_body(f: NativeFunction) -> list[str]:
    trace_body: list[str] = []

    trace_body.append(format_prerecord_trace(f))

    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()

    # code-generated tracing kernels plumb and recompute dispatch keys directly through the kernel for performance.
    # See Note [Plumbing Keys Through The Dispatcher] for details.
    dispatch_key_set = "ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer)"
    redispatch_args = ", ".join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])

    assign_return_values = (
        f"{tie_return_values(f)} = "
        if f.func.kind() in [SchemaKind.functional, SchemaKind.mutable]
        and f.func.returns
        else ""
    )

    # Note that this calls the slow, dispatching variants of manual_cpp_binding ops.
    # We could probably work harder to ensure that the fast variants are
    # called instead, but the perf benefit would be minimal.
    trace_body.append(
        TRACE_DISPATCH.substitute(
            assign_return_values=assign_return_values,
            unambiguous_name=f.func.name.unambiguous_name(),
            unpacked_args=redispatch_args,
        )
    )

    trace_body.append(format_postrecord_trace(f))
    if f.func.returns:
        trace_body.append(f"return {get_return_value(f)};")
    return trace_body


METHOD_DEFINITION = CodeTemplate(
    """\
${return_type} ${type_wrapper_name}(${formals}) {
  ${type_definition_body}
}
"""
)


def type_wrapper_name(f: NativeFunction, key: str = "Default") -> str:
    if f.func.name.overload_name:
        name = f"{cpp.name(f.func)}_{f.func.name.overload_name}"
    else:
        name = cpp.name(f.func)

    # The key argument is only used in gen_variable_type where we need fns per autograd dispatch key.
    # In gen_trace_type and gen_inplace_view_type where only one fn per native_fn must be generated,
    # the key argument should not be passed.
    # We do not append key if it is Default so that generated functions from
    # before per-dispatch-key derivatives were added retain the same names.
    if key != "Default":
        name = name + f"_{key}"
    return name


@with_native_function
def method_definition(f: NativeFunction) -> str:
    if cpp.name(f.func) in MANUAL_TRACER:
        raise AssertionError(f"Function {cpp.name(f.func)} is in MANUAL_TRACER")

    formals = ", ".join(
        # code-generated tracing kernels plumb and recompute dispatch keys directly through the kernel for performance.
        # See Note [Plumbing Keys Through The Dispatcher] for details.
        ["c10::DispatchKeySet ks"]
        + [
            f"{cpp.argument_type(a, binds='__placeholder__', symint=True).cpp_type()} {a.name}"
            for a in f.func.schema_order_arguments()
        ]
    )

    return METHOD_DEFINITION.substitute(
        return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(),
        type_wrapper_name=type_wrapper_name(f),
        formals=formals,
        type_definition_body=emit_trace_body(f),
    )


WRAPPER_REGISTRATION = CodeTemplate(
    """\
m.impl("${name}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
"""
)


@with_native_function
def method_registration(f: NativeFunction) -> str:
    if cpp.name(f.func) in MANUAL_TRACER:
        raise AssertionError(f"Function {cpp.name(f.func)} is in MANUAL_TRACER")

    return WRAPPER_REGISTRATION.substitute(
        name=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type="TraceType",
    )


def gen_trace_type_func(fn: NativeFunction) -> dict[str, list[str]]:
    return {
        "ops_headers": [f"#include <ATen/ops/{fn.root_name}_ops.h>"],
        "trace_method_definitions": [method_definition(fn)],
        "trace_wrapper_registrations": [method_registration(fn)],
    }


def gen_trace_type(
    out: str, native_functions: list[NativeFunction], template_path: str
) -> None:
    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_sharded(
        "TraceType.cpp",
        [fn for fn in native_functions if cpp.name(fn.func) not in MANUAL_TRACER],
        key_fn=lambda fn: fn.root_name,
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/TraceType.cpp",
        },
        env_callable=gen_trace_type_func,
        num_shards=10,
        sharded_keys={
            "ops_headers",
            "trace_method_definitions",
            "trace_wrapper_registrations",
        },
    )
<|endoftext|>import functools
from collections.abc import Callable

from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo as NFWDI
from torchgen.context import native_function_manager
from torchgen.utils import T


# Like tools.api.context.with_native_function, but for
# NativeFunctionWithDifferentiabilityInfo.
def with_native_function_with_differentiability_info(
    func: Callable[[NFWDI], T],
) -> Callable[[NFWDI], T]:
    @functools.wraps(func)
    def wrapper(f: NFWDI) -> T:
        with native_function_manager(f.func):
            return func(f)

    return wrapper


# Like the above but with an additional dispatch key string argument
def with_native_function_with_differentiability_info_and_key(
    func: Callable[[NFWDI, str], T],
) -> Callable[[NFWDI, str], T]:
    @functools.wraps(func)
    def wrapper(f: NFWDI, key: str) -> T:
        with native_function_manager(f.func):
            return func(f, key)

    return wrapper
<|endoftext|># Parses derivatives.yaml into autograd functions
#
# Each autograd function is represented by `DifferentiabilityInfo` containing
# a list of `Derivative`. See `torchgen.api.autograd` for the data models.

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, TYPE_CHECKING

import yaml

from torchgen.api import cpp
from torchgen.api.autograd import (
    Derivative,
    DifferentiabilityInfo,
    ForwardDerivative,
    SavedAttribute,
)
from torchgen.api.types import (
    BaseCType,
    Binding,
    boolT,
    CppSignatureGroup,
    layoutT,
    longT,
    NamedCType,
    OptionalCType,
    scalarTypeT,
    SpecialArgName,
    stringT,
    symIntArrayRefT,
    SymIntT,
    tensorGeometryT,
    tensorOptionsT,
    typeAndSizeT,
    VectorCType,
)
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
    AUTOGRAD_KEYS,
    FunctionSchema,
    NativeFunction,
    NativeFunctionsViewGroup,
    OperatorName,
    SchemaKind,
    Type,
    Variant,
)
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader


if TYPE_CHECKING:
    from collections.abc import Sequence


DerivativeRet = tuple[dict[FunctionSchema, dict[str, DifferentiabilityInfo]], set[str]]

_GLOBAL_LOAD_DERIVATIVE_CACHE: dict[tuple[str, str], DerivativeRet] = {}

_VALID_AUTOGRAD_KEYS = set(AUTOGRAD_KEYS)


# This function directly adds per-dispatchkey derivative entries for {view}_copy variants of each view op.
# Since every {view} and {view}_copy op shares the same derivative formula,
# we generate them here instead of duplicating them in the yaml.
# See Note [Codegen'd {view}_copy Operators]
def add_view_copy_derivatives(
    infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
    view_groups: list[NativeFunctionsViewGroup],
) -> None:
    # Get the map from each view op's name to its corresponding view group
    view_name_to_group: dict[OperatorName, NativeFunctionsViewGroup] = {
        g.view.func.name: g for g in view_groups
    }

    view_infos = {}

    for info_dispatch_dict in infos.values():
        # maybe_view_group only needs to be calculated once per info_dispatch_dict
        maybe_view_group = None
        view_copy_differentiability_infos = {}
        for dispatch_key, info in info_dispatch_dict.items():
            maybe_view_group = view_name_to_group.get(info.func.func.name, None)
            if maybe_view_group is not None and maybe_view_group.view_copy is not None:
                view_copy_info = info.create_view_copy_from_view_derivative(
                    maybe_view_group
                )
                if view_copy_info is not None:
                    fn_schema = view_copy_info.func.func
                    view_copy_differentiability_infos[dispatch_key] = view_copy_info
            else:
                break
        # prefer manually-defined derivatives if any
        # pyrefly: ignore [unbound-name]
        if len(view_copy_differentiability_infos) > 0 and fn_schema not in infos:
            if fn_schema is None:
                raise AssertionError("Expected fn_schema to be non-None")
            view_infos[fn_schema] = view_copy_differentiability_infos

    infos.update(view_infos)


def load_derivatives(
    derivatives_yaml_path: str, native_yaml_path: str, tags_yaml_path: str
) -> DerivativeRet:
    # Do some caching as this is a deterministic function
    global _GLOBAL_LOAD_DERIVATIVE_CACHE
    key = (derivatives_yaml_path, native_yaml_path)
    if key not in _GLOBAL_LOAD_DERIVATIVE_CACHE:
        with open(derivatives_yaml_path) as f:
            definitions = yaml.load(f, Loader=YamlLoader)

        funcs = parse_native_yaml(native_yaml_path, tags_yaml_path).native_functions
        # From the parsed native functions, separate out the (generated) view_copy functions,
        # so we can generate derivatives for them separately.
        native_functions_with_view_groups = get_grouped_by_view_native_functions(funcs)
        native_functions = concatMap(
            lambda g: [g]
            if isinstance(g, NativeFunction)
            else list(g.functions(include_copy=True)),
            native_functions_with_view_groups,
        )
        view_groups = [
            g
            for g in native_functions_with_view_groups
            if isinstance(g, NativeFunctionsViewGroup)
        ]

        # What's the difference between function schema v.s. signature?
        # function schema is the complete declaration including mutability annotation / default value and etc.
        # signature is the canonical schema for a group of functions (in-place/out/functional variants)
        # that are semantically related.
        functions_by_signature: dict[FunctionSchema, list[NativeFunction]] = (
            defaultdict(list)
        )
        functions_by_schema: dict[str, NativeFunction] = {}
        for function in native_functions:
            functions_by_signature[function.func.signature()].append(function)
            if str(function.func) in functions_by_schema:
                raise AssertionError(f"Duplicate function schema: {str(function.func)}")
            functions_by_schema[str(function.func)] = function

        # Keep track of how many of which ops we've seen so we can
        # disambiguate them with a numeric suffix.
        op_counter = Counter[str]()

        # infos is a dict that maps FunctionSchema -> a dict of per dispatch key DifferentiabilityInfos
        # this is useful because in tools/autograd/gen_autograd.py:match_differentiability_info
        # we ultimately need to categorize the DifferentiabilityInfos by FunctionSchema
        infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]] = {}
        used_dispatch_keys: set[str] = set()
        for defn_dict in definitions:
            # Ensure that the old derivatives.yaml schema with no dispatch key can be loaded.
            if "dispatch" not in defn_dict:
                specification = defn_dict.pop("name")
                output_differentiability = defn_dict.pop(
                    "output_differentiability", None
                )
                defn_dict = {"name": specification, "dispatch": {"Default": defn_dict}}
                if output_differentiability:
                    defn_dict["output_differentiability"] = output_differentiability
            name, per_dispatch_diffinfos = create_differentiability_info(
                defn_dict,
                functions_by_signature,
                functions_by_schema,
                op_counter,
                used_dispatch_keys,
            )
            infos[name] = per_dispatch_diffinfos

        add_view_copy_derivatives(infos, view_groups)

        # cache both loaded infos as well as a set of all the dispatch_keys/aliases
        # that appear in derivatives.yaml. used_dispatch_keys is useful for generating
        # VariableType.cpp where we need a TORCH_LIBRARY_IMPL for every autograd dispatch key used
        _GLOBAL_LOAD_DERIVATIVE_CACHE[key] = infos, used_dispatch_keys

    return _GLOBAL_LOAD_DERIVATIVE_CACHE[key]


# TODO: Why is this going through CppSignatureGroup, that doesn't make sense...
@with_native_function
def cpp_arguments(f: NativeFunction) -> Sequence[Binding]:
    sigs = CppSignatureGroup.from_native_function(f, method=False)
    if sigs.symint_signature is not None:
        return sigs.symint_signature.arguments()
    else:
        return sigs.signature.arguments()


def create_derivative(
    f: NativeFunction,
    formula: str,
    var_names: tuple[str, ...],
    available_named_gradients: Sequence[str],
) -> Derivative:
    original_formula = formula
    arguments: list[NamedCType] = [
        a.nctype.remove_const_ref() for a in cpp_arguments(f)
    ]

    return_names = tuple(n if n != "self" else "result" for n in cpp.return_names(f))
    return_types = tuple(
        cpp.return_type(r, symint=True).remove_const_ref() for r in f.func.returns
    )

    named_returns = [
        NamedCType(name, type) for name, type in zip(return_names, return_types)
    ]

    formula, saved_inputs = saved_variables(formula, arguments, var_names)
    formula, saved_outputs = saved_variables(formula, named_returns, var_names)

    used_named_gradients = {
        name
        for name in available_named_gradients
        if re.search(IDENT_REGEX.format(name), formula)
    }

    # Check that the referenced derivatives in the formula are in bounds
    for i in used_gradient_indices(formula):
        if i >= len(f.func.returns):
            raise RuntimeError(
                f"Out of bounds grads access: derivative formula for {cpp.name(f.func)} "
                f"used grads[{i}], but the forward only returns {len(f.func.returns)} outputs."
            )

    return Derivative(
        formula=formula,
        original_formula=original_formula,
        var_names=var_names,
        saved_inputs=saved_inputs,
        saved_outputs=saved_outputs,
        named_gradients=used_named_gradients,
    )


def create_forward_derivative(
    f: NativeFunction, formula: str, names: tuple[str, ...]
) -> ForwardDerivative:
    var_names = names
    var_types: tuple[Type, ...] | None = None
    for r in f.func.returns:
        if r.name in var_names:
            if var_types is None:
                var_types = ()
            var_types = var_types + (r.type,)

    # Handle default return names
    if var_types is None:
        if var_names == ("result",):
            if len(f.func.returns) != 1:
                raise AssertionError(
                    f"Expected 1 return for 'result', got {len(f.func.returns)}"
                )
            var_types = (f.func.returns[0].type,)
        else:
            for var_name in var_names:
                res = re.findall(r"^result(\d+)$", var_name)
                if len(res) == 1:
                    if var_types is None:
                        var_types = ()
                    arg_idx = int(res[0])
                    var_types = var_types + (f.func.returns[arg_idx].type,)

    if var_types is None:
        raise AssertionError("No matching output for forward derivative definition")
    return ForwardDerivative(
        formula=formula,
        var_names=var_names,
        var_types=var_types,
        required_inputs_fw_grad=None,
        required_inputs_primal=None,
        required_original_self_value=False,
        is_reusing_outplace_formula=False,
    )


def postprocess_forward_derivatives(
    f: NativeFunction,
    defn_name: str,
    all_arg_names: list[str],
    derivatives: list[Derivative],
    forward_derivatives: list[ForwardDerivative],
    args_with_derivatives: Sequence[Binding],
) -> list[ForwardDerivative]:
    def find_required_inputs(formula: str, postfix: str) -> tuple[str, ...]:
        is_foreach = f.func.name.name.base.startswith("_foreach_")
        required_inputs = set()
        for arg in args_with_derivatives:
            if (
                arg.type in ("at::TensorList", "const at::ITensorListRef &")
                and not is_foreach
            ):
                # The functions taking TensorList handle everything internally
                continue
            arg_name = arg.name

            found = re.search(IDENT_REGEX.format(arg_name), formula)
            if found:
                raise RuntimeError(
                    f"The forward formula for {defn_name} is using the base name of the {arg_name} "
                    f"argument which is ambiguous. You should use {arg_name}_p to access the primal "
                    f"value and {arg_name}_t to access the tangent."
                )

            found = re.search(IDENT_REGEX.format(arg_name + postfix), formula)
            if found:
                required_inputs.add(arg_name)

        return tuple(required_inputs)

    updated_derivatives: list[ForwardDerivative] = []

    for defn in forward_derivatives:
        formula = defn.formula
        required_inputs_tangent = find_required_inputs(formula, "_t")
        if formula == "auto_element_wise":
            if f.func.kind() == SchemaKind.inplace:
                raise AssertionError(
                    f"Cannot use auto_element_wise with {f.func.name} because it is an in-place variant"
                )
            if (
                (not len(args_with_derivatives) == 1)
                or len(forward_derivatives) > 1
                or len(forward_derivatives[0].var_names) > 1
            ):
                raise RuntimeError(
                    f"Derivative definition of {defn_name} in derivatives.yaml defines the "
                    "forward definition of gradient as element_wise but this only "
                    "works for functions with a single differentiable input and a "
                    "single differentiable output."
                )
            if not len(derivatives) == 1:
                raise RuntimeError(
                    f"Derivative definition of {defn_name} in derivatives.yaml defines the "
                    "forward definition of gradient as element_wise but it does not "
                    "defines the gradient formula for its argument which is required."
                )
            # This transformation is based on the observation that for element-wise functions, the Jacobian
            # matrix is diagonal and thus doing J * v is the same as (v^T J)^T (in practice, we ignore the transpositions)
            # For the complex case, we use hermitian transpose and get (v.conj() J).conj()
            # So here we are going to reuse the backward formula and replace two things:
            # 1) all occurrences of "grad" with "foo_t.conj()", where foo is the name of the unique differentiable input.
            # 2) all usage of an original input "foo" with its primal value "foo_p".
            # 3) conjugate the final result
            # For example, for abs, the backward formula is:
            #   grad * self.sgn()
            # And this function generates a forward formula that is:
            #   (self_t.conj() * self_p.sgn()).conj()

            backward_formula = derivatives[0].original_formula
            input_name = args_with_derivatives[0].name

            # Do replacement 1) of the grad
            def repl(m: Any) -> str:
                return f"{m.group(1)}{input_name}_t.conj(){m.group(2)}"

            fw_formula = re.sub(IDENT_REGEX.format("grad"), repl, backward_formula)

            # Do replacement 2) of the input variables
            for arg in args_with_derivatives:
                arg_name = arg.name

                def repl(m: Any) -> str:
                    return f"{m.group(1)}{arg_name}_p{m.group(2)}"

                fw_formula = re.sub(IDENT_REGEX.format(arg_name), repl, fw_formula)

            # Do the final conjugate 3)
            fw_formula = f"({fw_formula}).conj()"

            # Since there is a single differentiable inputs and we necessarily need its tangent we can
            # simply require all differentiable input's tangent.
            required_inputs_tangent = tuple(all_arg_names)
            formula = fw_formula
        elif formula == "auto_linear":
            if (
                len(forward_derivatives) > 1
                or len(forward_derivatives[0].var_names) > 1
            ):
                raise RuntimeError(
                    f"Derivative definition of {defn_name} in derivatives.yaml defines the "
                    "forward definition of gradient as linear but this only works "
                    "for functions with a single differentiable output."
                )
            # This transformation is based on the observation that linear functions can be written as:
            #   y = f(x) = A * x
            # For some matrix A and the Jacobian of the function f is also A.
            # So doing J * v = A * v = f(v).
            # Hence to do the jvp, we simply need to evaluate the function at the point v instead of x.
            # We do this by calling the forward again by replacing any occurrence of the differentiable
            # input "foo" by it's tangent "foo_t".
            # Note that multiple inputs are not a problem as long as the function is truly linear wrt to
            # the vector where all the differentiable inputs are stacked.

            diff_arg_names = [arg.name for arg in args_with_derivatives]
            if len(diff_arg_names) == 0:
                raise AssertionError("Expected at least one differentiable argument")

            # Do replacement of input variables
            new_args = []
            for arg_name in all_arg_names:
                if arg_name in diff_arg_names:
                    arg_name = arg_name + "_t"
                new_args.append(arg_name)

            # TODO we are trolling
            if f.func.has_symint():
                defn_name += "_symint"

            # Call into the forward again. We need two cases here to handle both Tensor methods and at:: functions.
            if Variant.function in f.variants:
                fw_formula = f"at::{defn_name}({', '.join(new_args)})"
            else:
                if Variant.method not in f.variants:
                    raise AssertionError(
                        f"Expected Variant.method in variants for {f.func.name}"
                    )
                fw_formula = f"{new_args[0]}.{defn_name}({', '.join(new_args[1:])})"

            # All of the input tangents are always used so all of them are required here.
            required_inputs_tangent = tuple(diff_arg_names)
            formula = fw_formula

        # At this point, the formula is final and is not modified anymore.

        # During forward formula, we use the primal instead of the input Tensors.
        # This call inspects the formula to find for which input's primal are used.
        required_inputs_primal = find_required_inputs(formula, "_p")

        updated_derivatives.append(
            ForwardDerivative(
                formula=formula,
                var_names=defn.var_names,
                var_types=defn.var_types,
                required_inputs_fw_grad=required_inputs_tangent,
                required_inputs_primal=required_inputs_primal,
                required_original_self_value=False,
                is_reusing_outplace_formula=False,
            )
        )

    return updated_derivatives


def is_forward_derivative_definition(
    all_arg_names: list[str], names: tuple[str, ...]
) -> bool:
    for name in names:
        return name not in all_arg_names
    raise RuntimeError("Expected `names` to be non-empty")


def create_differentiability_info(
    defn_dict: dict[Any, Any],
    functions_by_signature: dict[FunctionSchema, list[NativeFunction]],
    functions_by_schema: dict[str, NativeFunction],
    op_counter: Counter[str],
    used_dispatch_keys: set[str],
) -> tuple[FunctionSchema, dict[str, DifferentiabilityInfo]]:
    """Processes a single entry `defn` in derivatives.yaml"""

    def canonical_function(
        functions: Sequence[NativeFunction], name: str
    ) -> NativeFunction:
        for f in functions:
            if (
                not f.func.is_functional_fn()
                and not f.func.is_out_fn()
                and name == str(f.func.name.name)
            ):
                return f
        # some functions only have in-place variants
        if name + "_" != cpp.name(functions[0].func):
            raise AssertionError(
                f"Expected inplace function name '{name}_', got '{cpp.name(functions[0].func)}'"
            )
        return functions[0]

    def split_names(raw_names: str) -> tuple[str, ...]:
        """Given "foo, bar", return ["foo", "bar"]."""
        return tuple(x.strip() for x in raw_names.split(","))

    def check_grad_usage(defn_name: str, derivatives: Sequence[Derivative]) -> None:
        """
        Check for some subtle mistakes one might make when writing derivatives.
        These mistakes will compile, but will be latent until a function is
        used with double backwards.
        """

        uses_grad = False  # true if any derivative uses "grad"
        num_grads_uses = 0  # count of uses of "grads" or "grads[INDEX]"
        uses_named_grads = False  # true if any derivative uses "grad_{name}"
        used_grads_indices: list[int] = []  # which indices of grads are used
        for d in derivatives:
            formula = d.formula
            uses_grad = uses_grad or bool(
                re.findall(IDENT_REGEX.format("grad"), formula)
            )
            num_grads_uses += len(re.findall(IDENT_REGEX.format("grads"), formula))
            uses_named_grads = uses_named_grads or bool(d.named_gradients)
            used_grads_indices.extend(used_gradient_indices(formula))
        # This is a basic sanity check: the number of places we see
        # "grads" should be no fewer than the number of indices we see
        # inside "grads". They may not be equal because we may use
        # "grads" without an index.
        if num_grads_uses < len(used_grads_indices):
            raise AssertionError(
                f"num_grads_uses ({num_grads_uses}) < len(used_grads_indices) ({len(used_grads_indices)})"
            )
        # Thus if the number is equal, every use of grads is also
        # indexed.
        only_used_grads_indices = num_grads_uses == len(used_grads_indices)

        if uses_grad and num_grads_uses > 0:
            raise RuntimeError(
                f"Derivative definition of {defn_name} in derivatives.yaml illegally "
                "mixes use of 'grad' and 'grads'. Consider replacing "
                "occurrences of 'grad' with 'grads[0]'"
            )

        if only_used_grads_indices and set(used_grads_indices) == {0}:
            raise RuntimeError(
                f"Derivative definition of {defn_name} in derivatives.yaml solely "
                "refers to 'grads[0]'.  If the first output is indeed the "
                "only differentiable output, replace 'grads[0]' with 'grad'; "
                "otherwise, there is a likely error in your derivatives "
                "declaration."
            )

        if uses_named_grads and (uses_grad or num_grads_uses > 0):
            raise RuntimeError(
                f"Derivative definition of {defn_name} in derivatives.yaml illegally "
                'mixes use of "grad_RETURN_NAME" and "grad" or "grads[x]". Use '
                "only one method for identifying gradients."
            )

    @with_native_function
    def set_up_derivatives(
        f: NativeFunction,
    ) -> tuple[
        Sequence[Derivative],
        Sequence[ForwardDerivative],
        Sequence[Binding],
        Sequence[str],
        Sequence[str],
    ]:
        # Set up the derivative information
        derivatives: list[Derivative] = []
        forward_derivatives: list[ForwardDerivative] = []
        non_differentiable_arg_names: list[str] = []
        args_with_derivatives_set: set[str] = set()

        all_arg_names = [a.name for a in cpp_arguments(f)]
        all_ret_names = [
            r.name for r in f.func.returns
        ]  # only used for the assert below
        # output_differentiability is captured from the enclosed
        # scope. Don't modify it.
        #
        # If it is not present, then no output is explicitly
        # undifferentiable.
        #
        # It may be present and shorter than the length of return
        # values. If that's the case, any return value that does not
        # have a corresponding entry is considered not differentiable.
        differentiability = output_differentiability or [True] * len(f.func.returns)
        # A return is available as a named gradient ...
        available_named_gradients = [
            f"grad_{ret.name}"
            for ret, differentiable in zip(f.func.returns, differentiability)
            # if it has not been explicitly made undifferentiable
            if differentiable
            # and if it has a name
            and ret.name is not None
            # and if its type is differentiable
            and ret.type.is_tensor_like()
        ]

        for raw_names in sorted(defn.keys()):
            formula = defn[raw_names]
            names = split_names(raw_names)

            for name in names:
                if name in all_arg_names and name in all_ret_names:
                    raise AssertionError(
                        f"While processing the derivative formula for '{f.func.name}' wrt '{name}', "
                        f"expected '{name}' to not be both an input arg and named return."
                    )

            if is_forward_derivative_definition(all_arg_names, names):
                forward_derivatives.append(create_forward_derivative(f, formula, names))
            else:
                if formula.lower().strip() == "non_differentiable":
                    non_differentiable_arg_names += names
                else:
                    derivative = create_derivative(
                        f, formula, names, available_named_gradients
                    )
                    derivatives.append(derivative)
                    args_with_derivatives_set |= set(names)

        overlap = args_with_derivatives_set.intersection(non_differentiable_arg_names)
        if overlap:
            raise RuntimeError(
                f"derivatives definition for {defn} have overlapped non_differentiable "
                f"and differentiable variables: {overlap}"
            )

        # Next, let us determine the list of inputs in order.
        # TODO: do we need eagerly calculate and save it here? Can it be derived
        # from NativeFunction and `derivatives` on callsites instead?
        args_with_derivatives = [
            a for a in cpp_arguments(f) if a.name in args_with_derivatives_set
        ]

        # Postprocess forward derivatives definitions now that we know the differentiable arguments
        forward_derivatives = postprocess_forward_derivatives(
            f,
            defn_name,
            all_arg_names,
            derivatives,
            forward_derivatives,
            args_with_derivatives,
        )

        # Test to see if the use of 'grads' makes sense.
        check_grad_usage(defn_name, derivatives)

        return (
            derivatives,
            forward_derivatives,
            args_with_derivatives,
            non_differentiable_arg_names,
            available_named_gradients,
        )

    # NB: Removes 'name' from defn dictionary
    specification = defn_dict.pop("name")
    defn_name, _ = split_name_params(specification)
    # NB: Removes 'output_differentiability' from defn dictionary
    #     `None` means all differentiable.
    output_differentiability = defn_dict.pop("output_differentiability", None)
    output_differentiability_conditions = None
    if output_differentiability and any(
        isinstance(diff, str) for diff in output_differentiability
    ):
        if len(output_differentiability) != 1:
            raise RuntimeError(
                f"Not supported: for {specification},"
                f"output_differentiability must either be "
                f"list[bool] or a list[str] where each str is a "
                f"condition. In the case where it is a condition, "
                f"we only support single-output functions. "
                f"Please file us an issue. "
            )
        output_differentiability_conditions = output_differentiability
        output_differentiability = [True]

    schema_function = functions_by_schema.get(specification)
    if not schema_function:
        avail = "\n".join(
            k for k, v in functions_by_schema.items() if cpp.name(v.func) == defn_name
        )
        raise RuntimeError(
            f"could not find ATen function for schema: {specification} "
            f".  Available signatures:\n{avail}"
        )

    # now map this to the legacy schema; this isn't technically necessary, but we'd need some logic here
    # to map in-place schemas to the out-of-place variants.
    # TODO: maybe the logic to handle the legacy schema is no longer necessary?
    signature = schema_function.func.signature()
    functions = functions_by_signature[signature]
    if len(functions) == 0:
        avail = "\n".join(
            str(k)
            for k, v in functions_by_signature.items()
            if cpp.name(k) == defn_name
        )
        raise RuntimeError(
            f"could not find ATen function for legacy signature: {signature} "
            f"corresponding to schema {specification}.  Please report a bug to PyTorch. "
            f"Available signatures:\n{avail}"
        )

    canonical = canonical_function(functions, defn_name)
    if "grad_input_mask" in (a.name for a in cpp_arguments(canonical)):
        raise RuntimeError(
            f"Schema for {defn_name} has an argument named grad_input_mask, "
            "but this name would be shadowed by our codegen. "
            "Please use a different name in native_functions.yaml."
        )

    if "result" in (a.name for a in cpp_arguments(canonical)):
        raise RuntimeError(
            f"Schema for {defn_name} has an argument named result, "
            "but this is only allowed for outputs."
            "Please use a different name in native_functions.yaml."
        )

    diffinfo_dict = {}
    for key, defn in defn_dict["dispatch"].items():
        if key != "Default" and key not in _VALID_AUTOGRAD_KEYS:
            raise RuntimeError(
                f"Invalid dispatch key {key} in derivatives.yaml for {specification},"
                f" expected key to be one of {_VALID_AUTOGRAD_KEYS}"
            )
        if key not in used_dispatch_keys:
            used_dispatch_keys.add(key)

        (
            derivatives,
            forward_derivatives,
            args_with_derivatives,
            non_differentiable_arg_names,
            available_named_gradients,
        ) = set_up_derivatives(canonical)

        used_named_gradients: set[str] = set()
        for d in derivatives:
            used_named_gradients |= d.named_gradients

        # only assign an op name if we are actually going to calculate a derivative
        op = None
        if args_with_derivatives:
            op_prefix = _create_op_prefix(defn_name)
            if key != "Default":
                op_prefix = op_prefix + key
            op = f"{op_prefix}{op_counter[op_prefix]}"
            op_counter[op_prefix] += 1

        diffinfo_dict[key] = DifferentiabilityInfo(
            name=defn_name,
            func=canonical,
            op=op,
            derivatives=derivatives,
            forward_derivatives=forward_derivatives,
            all_saved_inputs=dedup_vars(
                [v for d in derivatives for v in d.saved_inputs]
            ),
            all_saved_outputs=dedup_vars(
                [v for d in derivatives for v in d.saved_outputs]
            ),
            available_named_gradients=available_named_gradients,
            used_named_gradients=used_named_gradients,
            args_with_derivatives=args_with_derivatives,
            non_differentiable_arg_names=non_differentiable_arg_names,
            output_differentiability=output_differentiability,
            output_differentiability_conditions=output_differentiability_conditions,
        )

    return canonical.func, diffinfo_dict


GRAD_INDEX_REGEX = r"(?:^|\W)grads\[(\d+)\]"


def used_gradient_indices(formula: str) -> list[int]:
    """Determine a list of gradient indices (the i in grads[i]) that
    are used by the formula.

    >>> used_gradient_indices("foo(grads[0], grads[1])")
    [0, 1]
    """
    return [int(i) for i in re.findall(GRAD_INDEX_REGEX, formula)]


def saved_variables(
    formula: str,
    nctypes: list[NamedCType],
    var_names: tuple[str, ...],
) -> tuple[str, tuple[SavedAttribute, ...]]:
    def stride_expr(name: str) -> str:
        if var_names != (name,):
            raise AssertionError(
                'Replacement for ".strides()" is currently only supported for single derivatives of the same tensor '
                'that ".strides()" is being called on.'
            )
        return f'strides_or_error({name}, "{name}")'

    REPLACEMENTS: list[tuple[str, dict[str, Any]]] = [
        # replace self.sym_sizes() with self_sym_sizes
        (
            r"{}.sym_sizes\(\)",
            {
                "suffix": "_sym_sizes",
                "nctype": lambda name: NamedCType(name, BaseCType(symIntArrayRefT)),
            },
        ),
        # replace self->sym_sizes() with self_sym_sizes_opt
        (
            r"{}->sym_sizes\(\)",
            {
                "suffix": "_sym_sizes_opt",
                "nctype": lambda name: NamedCType(
                    name, OptionalCType(BaseCType(symIntArrayRefT))
                ),
                "expr": lambda name: f"{name}.has_value() ? std::optional<c10::SymIntArrayRef>({name}->sym_sizes()) : std::nullopt",
            },
        ),
        # replace self.sym_blocksize() with self_sym_blocksize_opt
        (
            r"{}.sym_blocksize\(\)",
            {
                "suffix": "_self_sym_blocksize_opt",
                "nctype": lambda name: NamedCType(
                    name, OptionalCType(BaseCType(symIntArrayRefT))
                ),
                "expr": lambda name: f"at::sparse_csr::getSymIntBlockSize({name})",
            },
        ),
        # replace self.options() with self_options
        (
            r"{}.options\(\)",
            {
                "suffix": "_options",
                "nctype": lambda name: NamedCType(name, BaseCType(tensorOptionsT)),
            },
        ),
        # replace zeros_like(self) with self_info
        (
            r"zeros_like\({}\)",
            {
                "suffix": "_info",
                "nctype": lambda name: NamedCType(name, BaseCType(typeAndSizeT)),
                "expr": lambda name: name,  # at save-time
                "res": lambda name: name + "_info.zeros()",  # at eval-time
            },
        ),
        # replace self.sym_size(2) with self_sym_size_2
        (
            r"{}.sym_size\((-?\w+)\)",
            {
                "suffix": lambda m: f"_sym_argsize_{m.groups()[0].replace('-', 'minus_')}",
                "nctype": lambda name: NamedCType(name, BaseCType(SymIntT)),
            },
        ),
        # replace self.numel() with self_numel
        (
            r"{}.numel\(\)",
            {
                "suffix": "_numel",
                "nctype": lambda name: NamedCType(name, BaseCType(longT)),
            },
        ),
        # replace self.sym_numel() with self_sym_numel
        (
            r"{}.sym_numel\(\)",
            {
                "suffix": "_sym_numel",
                "nctype": lambda name: NamedCType(name, BaseCType(SymIntT)),
            },
        ),
        # replace to_args_sizes(self) with self_args_sizes
        (
            r"to_args_sizes\({}\)",
            {
                "suffix": "_args_sizes",
                "nctype": lambda name: NamedCType(
                    name, VectorCType(VectorCType(BaseCType(longT)))
                ),
            },
        ),
        # replace to_args_sizes_symint(self) with self_args_sizes
        (
            r"to_args_sizes_symint\({}\)",
            {
                "suffix": "_args_sizes_symint",
                "nctype": lambda name: NamedCType(
                    name, VectorCType(VectorCType(BaseCType(SymIntT)))
                ),
            },
        ),
        # replace to_args_scalartypes(self) with self_args_scalartypes
        (
            r"to_args_scalartypes\({}\)",
            {
                "suffix": "_args_scalartypes",
                "nctype": lambda name: NamedCType(
                    name, VectorCType(BaseCType(scalarTypeT))
                ),
            },
        ),
        # replace TensorGeometry(self) with self_geometry
        (
            r"TensorGeometry\({}\)",
            {
                "suffix": "_geometry",
                "nctype": lambda name: NamedCType(name, BaseCType(tensorGeometryT)),
            },
        ),
        (
            r"{}.scalar_type\(\)",
            {
                "suffix": "_scalar_type",
                "nctype": lambda name: NamedCType(name, BaseCType(scalarTypeT)),
            },
        ),
        # replace self.dim() with self_dim
        (
            r"{}.dim\(\)",
            {
                "suffix": "_dim",
                "nctype": lambda name: NamedCType(name, BaseCType(longT)),
            },
        ),
        # replace self.sym_strides() with self_sym_strides
        (
            r"{}.sym_strides\(\)",
            {
                "suffix": "_sym_strides",
                "nctype": lambda name: NamedCType(name, BaseCType(symIntArrayRefT)),
                "expr": stride_expr,
            },
        ),
        # replace self.layout() with self_layout
        (
            r"{}.layout\(\)",
            {
                "suffix": "_layout",
                "nctype": lambda name: NamedCType(name, BaseCType(layoutT)),
            },
        ),
        # replace self.is_conj() with self_conjugate
        (
            r"{}.is_conj\(\)",
            {
                "suffix": "_conjugate",
                "nctype": lambda name: NamedCType(name, BaseCType(boolT)),
            },
        ),
    ]

    # find which arguments need to be saved
    saved: list[SavedAttribute] = []

    if ".sizes()" in formula or "->sizes()" in formula:
        raise RuntimeError(
            ".sizes() is not supported in derivative formulas. Instead, please use the SymInt version,"
            + f".sym_sizes(), which returned a c10::SymIntArrayRef. formula={formula}"
        )
    if re.search(r"\.size\([-]?\d+\)", formula) or re.search(
        r"->size\([-]?\d+\)", formula
    ):
        raise RuntimeError(
            ".size(int) is not supported in derivative formulas. Instead, please use the SymInt version,"
            + f".sym_size(int), which returned a c10::SymIntArrayRef. formula={formula}"
        )
    if ".strides()" in formula or "->strides()" in formula:
        raise RuntimeError(
            ".strides() is not supported in derivative formulas. Instead, please use the SymInt version,"
            + f".sym_strides(), which returned a c10::SymIntArrayRef. formula={formula}"
        )
    for nctype in nctypes:
        name = (
            nctype.name.name if isinstance(nctype.name, SpecialArgName) else nctype.name
        )
        # First search the formula for expressions which can be evaluated
        # when the autograd Function is created to avoid saving variables
        for regex, info in REPLACEMENTS:

            def repl(m: re.Match[str]) -> str:
                suffix: str = (
                    # pyrefly: ignore [bad-assignment]
                    info["suffix"](m) if callable(info["suffix"]) else info["suffix"]
                )
                expr: str = info["expr"](name) if "expr" in info else m.group(0)
                saved.append(
                    SavedAttribute(
                        nctype=info["nctype"](name + suffix),
                        expr=expr,
                    )
                )
                if "res" in info:
                    replacement: str = info["res"](name)
                    return replacement
                return name + suffix

            formula = re.sub(regex.format(name), repl, formula)

        # std::optional<std::string> types stored in Backward nodes must be
        # converted to std::optional<std::string_view> before being passed into
        # the backward function
        if nctype.type == OptionalCType(BaseCType(stringT)):
            formula = re.sub(
                rf"\b{name}\b",
                f"{name}.has_value() ? std::optional<std::string_view>({name}.value()) : std::nullopt",
                formula,
            )

        # Find any variables which remain in the formula and save them
        if re.search(IDENT_REGEX.format(name), formula):
            saved.append(
                SavedAttribute(
                    nctype=nctype,
                    expr=name,
                )
            )

    return formula, tuple(saved)


def _create_op_prefix(name: str) -> str:
    r"""Takes a native function name converts to an op prefix name.

    Note that the "name" parameter must be the native function name
    without the optional variant suffix, so "add" instead of
    "add.out".

    OP names correspond to classes, hence the change to title case.

    Example::

        >>> _create_op_prefix("add")
        'AddBackward'
    """
    camel_case = "".join([p.title() for p in name.split("_")])
    return (camel_case + "Backward").replace("ForwardBackward", "Backward")


def dedup_vars(vars: Sequence[SavedAttribute]) -> Sequence[SavedAttribute]:
    seen: set[str] = set()
    saved: list[SavedAttribute] = []
    for var in vars:
        name = (
            var.nctype.name.name
            if isinstance(var.nctype.name, SpecialArgName)
            else var.nctype.name
        )
        if name in seen:
            continue
        seen.add(name)
        saved.append(var)
    return saved
<|endoftext|># Generates C++ autograd functions for the derivatives of ATen operations
#
# This writes two files:
#  Functions.h/cpp: subclasses of autograd::Node
#  python_functions.h/cpp: Python bindings for the above classes
#

from __future__ import annotations

from typing import TYPE_CHECKING

from torchgen.api.autograd import (
    Derivative,
    DifferentiabilityInfo,
    SavedAttribute,
    uses_retain_variables,
    uses_single_grad,
)
from torchgen.api.types import (
    ArrayRefCType,
    BaseCppType,
    BaseCType,
    Binding,
    boolT,
    doubleT,
    intArrayRefT,
    iTensorListRefT,
    ListCType,
    longT,
    MutRefCType,
    OptionalCType,
    optionalIntArrayRefT,
    optionalSymIntArrayRefT,
    scalarT,
    stringT,
    symIntArrayRefT,
    SymIntT,
    TENSOR_LIST_LIKE_CTYPES,
    tensorListT,
    tensorT,
    VectorCType,
)
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, FunctionSchema
from torchgen.utils import FileManager

from .gen_inplace_or_view_type import VIEW_FUNCTIONS


if TYPE_CHECKING:
    from collections.abc import Sequence


FUNCTION_DECLARATION = CodeTemplate(
    """\
#ifdef _WIN32
struct ${op} : public ${superclass} {
  TORCH_API ${op}() = default;
#else
struct TORCH_API ${op} : public ${superclass} {
#endif
  using ${superclass}::${superclass};
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "${op}"; }
  void release_variables() override {
    ${thread_lock}
    ${release_variables}
  }
  ${will_release_variables}
  void compiled_args(CompiledNodeArgs& args) const override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ${saved_variables}
  ${saved_list_sizes}
};
"""
)

WILL_RELEASE_VARIABLES = CodeTemplate(
    """\
bool retain_variables = true;
void will_release_variables() override {
  retain_variables = false;
}
"""
)

# We generate e.g. MulBackward0::apply and have that call into
# MulBackward0_apply_functional. The apply_functional is a pure function,
# that is, it does not rely on global state. MulBackward0::apply
# is responsible for querying the autograd engine for which outputs should
# be computed (needs_input_grad), applying locks,
# and unpacking saved variables to pass to MulBackward0_apply_functional.
#
# needs_input_grad is a mapping from input index to if that input needs
# gradients computed. For operators that take in List[Tensor], the List[Tensor]
# is one element in the needs_input_grad that specifies if *any* of the
# List[Tensor] needs input grad. In theory this could be optimized.
FUNCTION_DEFINITION = CodeTemplate(
    """\
static variable_list ${op}_apply_functional(
  variable_list&& grads,
  std::array<bool,${num_inputs}> needs_input_grad${,apply_functional_args_signature})
{
  IndexRangeGenerator gen;
  ${compute_index_ranges}
  variable_list grad_inputs(gen.size());
  ${body}
  return grad_inputs;
}
inline variable_list ${op}_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, ${num_inputs}>>();
  ${unpack_ivalues}
  return ${op}_apply_functional(variable_list(grads), needs_input_grad${,apply_functional_args});
#endif
}

variable_list ${op}::apply(variable_list&& grads) {
  ${thread_lock}
  ${asserts}
  ${unpacks}
  ${compute_needs_input_grad}
  return ${op}_apply_functional(std::move(grads), needs_input_grad${,apply_functional_args});
}

void ${op}::compiled_args(CompiledNodeArgs& args) const {
    ${compiled_args}
}
variable_list ${op}::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  ${apply_with_saved_before}

  static bool called = false;
  if (!called) {
    called = true;
    ${compute_schema}
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), ${op}_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;
  ${asserts}
  ${unpacks}
  ${compute_needs_input_grad}
  packed_args.pack(needs_input_grad);
  ${get_packed_args}

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  ${apply_with_saved_after}
  return output_result;
#endif
}

"""
)

GRAD_INPUT_MASK = CodeTemplate(
    """\
  auto grad_input_mask = std::array<bool, ${n}>{
    ${masks}
  };
"""
)

COMPUTE_NEEDS_INPUT_GRAD = CodeTemplate(
    """\
IndexRangeGenerator gen;
${compute_index_ranges}
auto needs_input_grad = std::array<bool, ${n}>{
  ${masks}
};\
"""
)


DERIVATIVE_SINGLE = CodeTemplate(
    """\
if (needs_input_grad[/*${name}*/${idx}]) {
  auto grad_result = ${derivative};
  copy_range(grad_inputs, ${name}_ix, grad_result);
}
"""
)

# note(crcrpar): `self` argument and other optional positional argument
# of foreach functions are basically a list of n `Tensor`s thus iterating over
# `grads` in order to utilize and apply the existing derivative definitions
# to each `Tensor`(s) of `self`, and the others.
DERIVATIVE_SINGLE_FOREACH = CodeTemplate(
    """\
if (needs_input_grad[/*${name}*/${idx}]) {  // ${name}
  std::vector<Tensor> grad_result;
  grad_result.reserve(grads.size());
  for (const auto & i : c10::irange(grads.size())) {
    if (grads[i].defined()) {
      grad_result.emplace_back(${derivative});
    } else {
      grad_result.emplace_back(Tensor());
    }
  }
  copy_range(grad_inputs, ${name}_ix, grad_result);
}
"""
)

DERIVATIVE_MULTI_COPY_RANGE = CodeTemplate(
    """\
  if (needs_input_grad[/*${name}*/${idx}]) {
    copy_range(grad_inputs, ${name}_ix, std::get<${i}>(grad_result));
  }
"""
)

DERIVATIVE_MULTI = CodeTemplate(
    """\
if (${needs_input_grad}) {
  ${grad_input_mask}
  auto grad_result = ${derivative};
  ${copy_ranges}
}
"""
)

# Generates python bindings
#
# This generates the definitions for:
#   (1) The PyTypeObject for each backward grad_fn subclassing Node
#   (2) The entry for PyTypeObject's tp_getset slot (an array of PyGetSetDef structs)
#       We generate one PyGetSetDef struct for each of grad_fn's saved inputs and outputs
#       Each PyGetSetDef has a function ptr to a getter, also defined here (3).
#   (3) Getters for each of grad_fn's saved inputs and outputs.
#
PY_FUNCTION_DEFINITION = CodeTemplate(
    """\
static PyTypeObject ${op}Class;
addClass<${op}>(module, ${op}Class, "${op}", ${op}_properties);
"""
)

PY_FUNCTION_PROPS_AND_GETTERS = CodeTemplate(
    """\
${all_getter_definitions}

static struct PyGetSetDef ${op}_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  ${all_getsetdef_structs}
  {nullptr} /* sentinel */
};

"""
)

PY_GETSETDEF_STRUCT = CodeTemplate(
    """\
{(char*)"_saved_${name}", (getter)THP${op}_${name}_getter, nullptr, nullptr, nullptr}"""
)

PY_RAW_GETSETDEF_STRUCT = CodeTemplate(
    """\
{(char*)"_raw_saved_${name}", (getter)THP${op}_${name}_raw_getter, nullptr, nullptr, nullptr}"""
)

# Getter templates
GETTER_DEFINITION = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<${op}*>(self->cdata.get())->${name};
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_SAVEDVAR = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<${op}*>(self->cdata.get())->${name}_;
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_RAW_SAVEDVAR = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<${op}*>(self->cdata.get())->${name}_;
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_VEC_SAVEDVAR = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<${op}*>(self->cdata.get());
  const auto& prop = node->${name}_;
  if (node->${name}_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_RAW_VEC_SAVEDVAR = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<${op}*>(self->cdata.get());
  const auto& prop = node->${name}_;
  if (node->${name}_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_OPT = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<${op}*>(self->cdata.get())->${name};
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_OPT_ARRAYREF = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<${op}*>(self->cdata.get())->${name};
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

# Getter body
GETTER_BODY_SAVEDVAR = """\
return THPVariable_Wrap(prop.unpack(self->cdata));
"""

GETTER_BODY_RAW_SAVEDVAR = """\
pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
return obj.release().ptr();
"""

GETTER_BODY_VEC_SAVEDVAR = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i: c10::irange(prop.size())) {
  PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
}
return tup;
"""

GETTER_BODY_RAW_VEC_SAVEDVAR = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i : c10::irange(prop.size())) {
  pybind11::object obj = pybind11::cast(prop[i], pybind11::return_value_policy::reference);
  PyTuple_SetItem(tup, (Py_ssize_t) i, obj.release().ptr());
}
return tup;
"""

GETTER_BODY_ARRAYREF_LONG = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i : c10::irange(prop.size())) {
  PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
}
return tup;
"""

GETTER_BODY_ARRAYREF_SYMINT = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i : c10::irange(prop.size())) {
    auto si = prop[i];
    if (auto m = si.maybe_as_int()) {
      PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
    } else {
      auto py_symint = py::cast(si).release().ptr();
      PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
    }
}
return tup;
"""

GETTER_BODY_ARRAYREF_DOUBLE = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i : c10::irange(prop.size())) {
  PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
}
return tup;
"""

GETTER_BODY_INT64_T = """\
