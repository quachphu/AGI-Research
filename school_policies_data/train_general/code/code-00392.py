return PyLong_FromUnsignedLong((int64_t) prop);
"""

GETTER_BODY_SYMINT = """\
if (auto m = prop.maybe_as_int()) {
  return PyLong_FromUnsignedLong(*m);
} else {
  return py::cast(prop).release().ptr();
}
"""

GETTER_BODY_DOUBLE = """\
return PyFloat_FromDouble((double) prop);
"""

GETTER_BODY_BOOL = """\
if (prop) {
  Py_RETURN_TRUE;
} else {
  Py_RETURN_FALSE;
}
"""

GETTER_BODY_STRING = """\
return PyUnicode_FromStringAndSize(prop.data(), prop.size());
"""

GETTER_BODY_SCALAR = """\
if (prop.isComplex()) {
  auto cprop = prop.to<c10::complex<double>>();
  return PyComplex_FromDoubles(cprop.real(), cprop.imag());
} else if (prop.isFloatingPoint()) {
  return PyFloat_FromDouble(prop.to<double>());
} else if (prop.isIntegral(/*includeBool=*/false)) {
  return PyLong_FromLong(prop.to<int64_t>());
} else if (prop.isBoolean()) {
  if (prop.to<bool>()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
} else {
  PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
  return nullptr;
}
"""


GETTER_BODY_VEC_SCALAR = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i: c10::irange(prop.size())) {
  if (prop[i].isComplex()) {
    auto cprop = prop[i].to<c10::complex<double>>();
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyComplex_FromDoubles(cprop.real(), cprop.imag()));
  } else if (prop[i].isFloatingPoint()) {
    auto double_prop = prop[i].to<double>();
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble(double_prop));
  } else if (prop[i].isIntegral(/*includeBool=*/false)) {
    auto long_prop = prop[i].to<int64_t>();
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromLong(long_prop));
  } else if (prop[i].isBoolean()) {
    if (prop[i].to<bool>()) {
      PyTuple_SetItem(tup, (Py_ssize_t) i, Py_True);
    } else {
      PyTuple_SetItem(tup, (Py_ssize_t) i, Py_False);
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
}
return tup;
"""


MISC_GETTER_DEFS = {
    OptionalCType(BaseCType(longT)): (GETTER_DEFINITION_OPT, GETTER_BODY_INT64_T),
    OptionalCType(BaseCType(SymIntT)): (GETTER_DEFINITION_OPT, GETTER_BODY_SYMINT),
    BaseCType(doubleT): (GETTER_DEFINITION, GETTER_BODY_DOUBLE),
    OptionalCType(BaseCType(doubleT)): (GETTER_DEFINITION_OPT, GETTER_BODY_DOUBLE),
    BaseCType(boolT): (GETTER_DEFINITION, GETTER_BODY_BOOL),
    BaseCType(scalarT): (GETTER_DEFINITION, GETTER_BODY_SCALAR),
    OptionalCType(BaseCType(scalarT)): (GETTER_DEFINITION_OPT, GETTER_BODY_SCALAR),
}

# These functions have backwards which cannot be traced, and so must have
# their backward functions traced opaquely.
# VIEW_FUNCTIONS are not traceable because they use as_strided, which
# has an untraceable backwards, see
# https://github.com/pytorch/pytorch/issues/4250
# TODO: This is probably not exhaustive, but it's a start
UNTRACEABLE_FUNCTIONS = VIEW_FUNCTIONS


def get_infos_with_derivatives_list(
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
) -> list[DifferentiabilityInfo]:
    diff_info_list = [
        info
        for diffinfo_dict in differentiability_infos.values()
        for info in diffinfo_dict.values()
    ]

    return list(filter(lambda info: info.args_with_derivatives, diff_info_list))


def gen_autograd_functions_lib(
    out: str,
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
    template_path: str,
) -> None:
    """Functions.h and Functions.cpp body

    These contain the auto-generated subclasses of torch::autograd::Node
    for each every differentiable torch function.
    """

    # get a 1D list of diffinfos, we do not need them to be per FunctionSchema/DispatchKey here
    # infos with the diff dispatchkeys but the same name will still be in the same shard.
    infos = get_infos_with_derivatives_list(differentiability_infos)
    declarations = [process_function(f, FUNCTION_DECLARATION) for f in infos]
    definitions = [process_function(f, FUNCTION_DEFINITION) for f in infos]

    file_basename = "Functions"
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for suffix in [".h", ".cpp"]:
        fname = file_basename + suffix
        fm.write_with_template(
            fname,
            fname,
            lambda: {
                "generated_comment": "@"
                + f"generated from {fm.template_dir_for_comments()}/{fname}",
                "autograd_function_declarations": declarations,
                "autograd_function_definitions": definitions,
            },
        )


def gen_autograd_functions_python(
    out: str,
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
    template_path: str,
) -> None:
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    num_shards = 5
    fm.write(
        "python_functions.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/python_functions.h",
            "shard_forward_declare": [
                f"void initialize_autogenerated_functions_{i}(PyObject* module);"
                for i in range(num_shards)
            ],
            "shard_call": [
                f"initialize_autogenerated_functions_{i}(module);"
                for i in range(num_shards)
            ],
        },
    )

    # get a 1D list of diffinfos, we do not need them to be per FunctionSchema/DispatchKey here
    # infos with the diff dispatchkeys but the same name will still be in the same shard.
    infos = get_infos_with_derivatives_list(differentiability_infos)
    fm.write_sharded(
        "python_functions.cpp",
        infos,
        key_fn=lambda info: info.name,
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/python_functions.cpp",
        },
        env_callable=lambda info: {
            "py_function_initializers": [
                process_function(info, PY_FUNCTION_DEFINITION)
            ],
            "py_function_props_and_getters": [
                process_function(info, PY_FUNCTION_PROPS_AND_GETTERS)
            ],
        },
        num_shards=num_shards,
        sharded_keys={"py_function_initializers", "py_function_props_and_getters"},
    )


def process_function(info: DifferentiabilityInfo, template: CodeTemplate) -> str:
    saved_variables: list[str] = []
    release_variables: list[str] = []
    saved_list_sizes: list[str] = []
    unpack: list[str] = []
    asserts: list[str] = []
    compute_index_ranges: list[str] = []
    getter_definitions: list[str] = []
    py_getsetdef_structs: list[str] = []
    compiled_args: list[str] = []
    apply_with_saved_before: list[str] = []
    apply_with_saved_after: list[str] = []
    apply_functional_args: list[str] = []
    apply_functional_args_ref_types: list[str] = []
    # Maps the name of an input (to the original forward operator;
    # examples are "self", "other") to the order in which they appear in the
    # operator.
    # For example; if the operator is foo(Tensor self, int64_t k, Tensor other),
    # the mapping is: {"self": 0, "other": 1}.
    # We use this mapping to populate needs_input_grad in some order and then grab
    # values from it.
    input_name_to_idx: dict[str, int] = {}

    for idx, arg in enumerate(info.args_with_derivatives):
        if arg.type in TENSOR_LIST_LIKE_CTYPES:
            size = f"{arg.name}_size_"
            saved_list_sizes.append(f"size_t {arg.name}_size_;")
            apply_functional_args.append(f"{arg.name}_size_")
            apply_functional_args_ref_types.append("size_t")
        else:
            size = "1"
        compute_index_ranges.append(f"auto {arg.name}_ix = gen.range({size});")
        input_name_to_idx[arg.name] = idx

    def save_var(var: SavedAttribute, is_output: bool) -> None:
        name = var.nctype.name
        type = var.nctype.type
        should_append_getsetdef = True
        should_append_raw_getsetdef = False
        visit_name = name
        uses_cpp_saved_variable_cls = False
        unpacked_ref_type = None

        if (
            type == BaseCType(tensorT)
            or type == OptionalCType(BaseCType(tensorT))
            or type == MutRefCType(OptionalCType(BaseCType(tensorT)))
            or (type == BaseCType(scalarT) and is_output)
        ):
            uses_cpp_saved_variable_cls = True
            saved_variables.append(f"SavedVariable {name}_;")
            release_variables.append(f"{name}_.reset_data();")
            ptr = "getptr()" if is_output else ""
            unpack.append(f"auto {name} = {name}_.unpack({ptr});")
            getter_definitions.append(
                GETTER_DEFINITION_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_SAVEDVAR
                )
            )
            getter_definitions.append(
                GETTER_DEFINITION_RAW_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_RAW_SAVEDVAR
                )
            )
            should_append_raw_getsetdef = True
            visit_name = f"{name}_"
            unpacked_ref_type = "Tensor&"
        elif (
            type == BaseCType(tensorListT)
            or type == BaseCType(iTensorListRefT)
            or type == VectorCType(BaseCType(tensorT))
        ):
            # note(crcrpar): [nuanced return type of out-of-place foreach functions]
            # When an out-of-place foreach function whose return signature is `Tensor[]`
            # spells out its backward definitions in `derivatives.yaml`, and some of them depend on
            # `result`, `result`'s type is interpreted and treated as `std::vector<Tensor>`.
            # An out-of-place foreach whose backwards rely on their output doesn't suffer from this
            # difference if the definitions are codegen'ed.
            # This special case is needed for `_foreach_pow.List` and `_foreach_pow.ScalarAndTensor`
            # as of https://github.com/pytorch/pytorch/pull/105504.
            if type == VectorCType(BaseCType(tensorT)):
                if not (
                    info.func.func.name.name.base.startswith("_foreach") and is_output
                ):
                    raise AssertionError(
                        "VectorCType(BaseCType(tensorT)) requires foreach function and is_output"
                    )
            uses_cpp_saved_variable_cls = True
            saved_variables.append(f"std::vector<SavedVariable> {name}_;")
            saved_variables.append(f"bool {name}_released_ = false;")
            # Just clear() is sufficient, we don't need to loop and clear each variable.
            # Because the SavedVariable owns a tensor and a grad_fn, removing the SavedVariable makes them go away as well.
            release_variables.append(f"{name}_.clear();")
            release_variables.append(f"{name}_released_ = true;")
            ptr = "getptr()" if is_output else "nullptr"
            unpack.append(f"auto {name} = unpack_list({name}_, {ptr});")
            asserts.append(f"TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);")
            getter_definitions.append(
                GETTER_DEFINITION_VEC_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_VEC_SAVEDVAR
                )
            )
            getter_definitions.append(
                GETTER_DEFINITION_RAW_VEC_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_RAW_VEC_SAVEDVAR
                )
            )
            should_append_raw_getsetdef = True
            visit_name = f"{name}_"
            unpacked_ref_type = "std::vector<Tensor>&"
        elif type == ListCType(OptionalCType(BaseCType(tensorT))):
            uses_cpp_saved_variable_cls = True
            saved_variables.append(f"std::vector<SavedVariable> {name}_;")
            saved_variables.append(f"bool {name}_released_ = false;")
            # Just clear() is sufficient, we don't need to loop and clear each variable.
            # Because the SavedVariable owns a tensor and a grad_fn, removing the SavedVariable makes them go away as well.
            release_variables.append(f"{name}_.clear();")
            release_variables.append(f"{name}_released_ = true;")
            unpack.append(f"auto {name} = unpack_opt_list({name}_);")
            asserts.append(f"TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);")
            getter_definitions.append(
                GETTER_DEFINITION_VEC_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_VEC_SAVEDVAR
                )
            )
            getter_definitions.append(
                GETTER_DEFINITION_RAW_VEC_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_RAW_VEC_SAVEDVAR
                )
            )
            should_append_raw_getsetdef = True
            visit_name = f"{name}_"
            unpacked_ref_type = "torch::List<std::optional<Tensor>>&"
        elif type == BaseCType(intArrayRefT):
            saved_variables.append(f"std::vector<int64_t> {name};")
            getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG
                )
            )
        elif type == BaseCType(symIntArrayRefT):
            saved_variables.append(f"std::vector<c10::SymInt> {name};")
            getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT
                )
            )
        elif type == BaseCType(optionalIntArrayRefT):
            saved_variables.append(f"c10::OptionalArray<int64_t> {name};")
            getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG
                )
            )
        elif type == BaseCType(optionalSymIntArrayRefT):
            saved_variables.append(f"c10::OptionalArray<c10::SymInt> {name};")
            getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT
                )
            )
        elif type == OptionalCType(BaseCType(intArrayRefT)):
            saved_variables.append(f"c10::OptionalArray<int64_t> {name};")
            getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG
                )
            )
        elif type == OptionalCType(BaseCType(symIntArrayRefT)):
            saved_variables.append(f"c10::OptionalArray<c10::SymInt> {name};")
            getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT
                )
            )
        elif type == OptionalCType(ArrayRefCType(BaseCType(doubleT))):
            saved_variables.append(f"c10::OptionalArray<double> {name};")
            getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_DOUBLE
                )
            )
        elif type == BaseCType(longT):
            saved_variables.append(f"{type.cpp_type()} {name} = 0;")
            getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_INT64_T
                )
            )
        elif type == BaseCType(SymIntT):
            saved_variables.append(f"c10::SymInt {name};")
            getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_SYMINT
                )
            )
        elif type == BaseCType(stringT):
            saved_variables.append(f"std::string {name};")
            getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_STRING
                )
            )
        elif type == OptionalCType(BaseCType(stringT)):
            saved_variables.append(f"std::optional<std::string> {name};")
            getter_definitions.append(
                GETTER_DEFINITION_OPT.substitute(
                    op=info.op, name=name, body=GETTER_BODY_STRING
                )
            )
        elif type == ArrayRefCType(
            elem=BaseCType(type=BaseCppType(ns="at", name="Scalar"))
        ):
            saved_variables.append(f"std::vector<at::Scalar> {name};")
            unpacked_ref_type = "std::vector<at::Scalar>&"
            saved_variables.append(f"bool {name}_released_ = false;")
            # Just clear() is sufficient, we don't need to loop and clear each variable.
            # Because the SavedVariable owns a tensor and a grad_fn, removing the SavedVariable makes them go away as well.
            release_variables.append(f"{name}.clear();")
            # release_variables.append(f"{name}_released_ = true;")
            # unpack.append(f"auto {name} = unpack_list({name}_);")
            # asserts.append(f"TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);")
            getter_definitions.append(
                CodeTemplate(
                    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<${op}*>(self->cdata.get());
  const auto& prop = node->${name};
  if (node->${name}_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  ${body}
  END_HANDLE_TH_ERRORS
}
                            """
                ).substitute(
                    op=info.op,
                    name=name,
                    body=GETTER_BODY_VEC_SCALAR,
                )
            )
        else:
            # Check for indicators that you're putting a non-owning reference
            # into the saved variable field.  If this is spuriously firing,
            # edit this field.  Otherwise, you probably need to add a case
            # above.
            if not (
                "ref" not in type.cpp_type().lower()
                and "view" not in type.cpp_type().lower()
                and "*" not in type.cpp_type()
                and "&" not in type.cpp_type()
            ):
                raise AssertionError(
                    f"{type.cpp_type()} looks like it contains a non-owning reference"
                )
            saved_variables.append(f"{type.cpp_type()} {name};")

            if type in MISC_GETTER_DEFS:
                getter_def, body = MISC_GETTER_DEFS[type]
                getter_definitions.append(
                    getter_def.substitute(op=info.op, name=name, body=body)
                )
            else:
                # Types we don't expose python bindings to yet:
                #   TypeAndSize, at::ScalarType, TensorOptions, TensorGeometry,
                #   std::vector<std::vector<int64_t>>, std::vector<at::ScalarType>
                should_append_getsetdef = False

        if should_append_getsetdef:
            py_getsetdef_structs.append(
                PY_GETSETDEF_STRUCT.substitute(op=info.op, name=name)
            )
        if should_append_raw_getsetdef:
            py_getsetdef_structs.append(
                PY_RAW_GETSETDEF_STRUCT.substitute(op=info.op, name=name)
            )

        if uses_cpp_saved_variable_cls:
            compiled_args.append(
                f"args.collect({visit_name}, {'true' if is_output else 'false'});"
            )
        else:
            compiled_args.append(f"args.collect({visit_name});")
        apply_with_saved_before.append(f"saved.before({visit_name});")
        apply_with_saved_after.append(f"saved.after({visit_name});")

        if unpacked_ref_type is None:
            unpacked_ref_type = f"{saved_variables[-1].split(' ')[0]}&"
        apply_functional_args.append(str(name))
        apply_functional_args_ref_types.append(unpacked_ref_type)

    for var in sorted(info.all_saved_inputs, key=lambda sa: str(sa.nctype.name)):
        save_var(var, is_output=False)
    for var in sorted(info.all_saved_outputs, key=lambda sa: str(sa.nctype.name)):
        save_var(var, is_output=True)

    # lock the mutex when we release variables and in Node::apply to protect thread safety
    # see Note [Thread Safety on Autograd Node]
    if len(release_variables) > 0:
        thread_lock = "std::lock_guard<std::mutex> lock(mutex_);"
    else:
        thread_lock = ""

    if uses_retain_variables(info):
        apply_functional_args.append("retain_variables")
        apply_functional_args_ref_types.append("bool")
        will_release_variables = WILL_RELEASE_VARIABLES.substitute()
    else:
        will_release_variables = ""

    body: list[str] = []

    if uses_single_grad(info):
        body.append("const auto& grad = grads[0];")
    else:
        # Generate aliases for gradients named for returned values.
        body.extend(
            f"const auto& {name} = grads[{info.available_named_gradients.index(name)}];"
            for name in sorted(info.used_named_gradients)
        )

    def emit_derivative(
        derivative: Derivative,
        args_with_derivatives: Sequence[Binding],
    ) -> tuple[bool, str]:
        formula = derivative.formula
        var_names = derivative.var_names

        if len(var_names) == 1:
            checks_any_grad_defined = False
            if "not_implemented" not in formula:
                matching_args = [
                    arg for arg in args_with_derivatives if arg.name == var_names[0]
                ]
                if len(matching_args) == 1:
                    # We can add undefined grad support if the input variable is a Tensor
                    arg = matching_args[0]
                    if isinstance(arg.argument, Argument) and str(
                        arg.argument.type
                    ) in ("Tensor", "Tensor?"):
                        formula = "any_grad_defined ? (" + formula + ") : Tensor()"
                        checks_any_grad_defined = True
            if info.name.startswith("_foreach_"):
                derivative_template = DERIVATIVE_SINGLE_FOREACH
            else:
                derivative_template = DERIVATIVE_SINGLE
            return (
                checks_any_grad_defined,
                derivative_template.substitute(
                    name=var_names[0],
                    derivative=formula,
                    idx=input_name_to_idx[var_names[0]],
                ),
            )

        else:
            if "grad_input_mask" in formula:
                masks = [
                    f"needs_input_grad[{input_name_to_idx[name]}],"
                    for name in var_names
                ]
                grad_input_mask = GRAD_INPUT_MASK.substitute(
                    n=len(var_names), masks=masks
                )
            else:
                grad_input_mask = ""
            needs_input_grad = [
                f"needs_input_grad[{input_name_to_idx[name]}]" for name in var_names
            ]
            needs_input_grad = " || ".join(needs_input_grad)
            copy_ranges: list[str] = []
            for i, n in enumerate(var_names):
                copy_ranges.append(
                    DERIVATIVE_MULTI_COPY_RANGE.substitute(
                        name=n, i=i, idx=input_name_to_idx[n]
                    )
                )
            return False, DERIVATIVE_MULTI.substitute(
                needs_input_grad=needs_input_grad,
                copy_ranges=copy_ranges,
                derivative=formula,
                grad_input_mask=grad_input_mask,
            )

    masks = []

    need_any_grad_defined_var = False
    for derivative in info.derivatives:
        checks_any_grad_defined, derivative_text = emit_derivative(
            derivative, info.args_with_derivatives
        )
        body.append(derivative_text)
        need_any_grad_defined_var |= checks_any_grad_defined

    for name in input_name_to_idx:
        masks.append(f"task_should_compute_output({{ {name}_ix }}),")

    # Since single-output derivative formulas need to check if grads are
    # defined, only perform the check once, before all the formulas
    if need_any_grad_defined_var:
        body.insert(
            -len(info.derivatives),
            "bool any_grad_defined = any_variable_defined(grads);",
        )

    if info.name in UNTRACEABLE_FUNCTIONS:
        superclass = "Node"
    else:
        superclass = "TraceableFunction"

    all_getsetdef_structs = (
        ",\n".join(py_getsetdef_structs) + "," if len(py_getsetdef_structs) != 0 else ""
    )
    all_getter_definitions = "\n".join(getter_definitions)

    compute_needs_input_grad = COMPUTE_NEEDS_INPUT_GRAD.substitute(
        n=len(masks), compute_index_ranges=compute_index_ranges, masks=masks
    )
    apply_functional_args_signature = [
        f"{T} {x}"
        for T, x in zip(apply_functional_args_ref_types, apply_functional_args)
    ]
    get_packed_args = "\n".join(
        f"packed_args.pack({name});" for name in apply_functional_args
    )
    unpack_ivalues = []
    for typ, name in zip(apply_functional_args_ref_types, apply_functional_args):
        typ = typ.removesuffix("&")
        unpack_ivalues.append(f"auto {name} = packed_args.unpack<{typ}>();")

    schema_args = [f"std::array<bool, {len(input_name_to_idx)}>"]
    for typ in apply_functional_args_ref_types:
        typ = typ.removesuffix("&")
        typ = typ.removeprefix("const")
        schema_args.append(typ.strip())
    compute_schema = ["std::vector<at::TypePtr> schema = {"]
    for schema_arg in schema_args:
        compute_schema.append(
            f"  torch::dynamo::autograd::IValuePacker<{schema_arg}>::packed_type(),"
        )
    compute_schema.append("};")

    return template.substitute(
        unpacks="\n".join(unpack),
        op=info.op,
        compute_schema="\n".join(compute_schema),
        apply_functional_args=apply_functional_args,
        apply_functional_args_signature=apply_functional_args_signature,
        compute_needs_input_grad=compute_needs_input_grad,
        num_inputs=len(input_name_to_idx),
        unpack_ivalues="\n".join(unpack_ivalues),
        compute_index_ranges=compute_index_ranges,
        saved_variables=saved_variables,
        release_variables=release_variables,
        saved_list_sizes=saved_list_sizes,
        asserts=asserts,
        thread_lock=thread_lock,
        will_release_variables=will_release_variables,
        body=body,
        superclass=superclass,
        all_getter_definitions=all_getter_definitions,
        all_getsetdef_structs=all_getsetdef_structs,
        compiled_args=compiled_args,
        apply_with_saved_before=apply_with_saved_before,
        apply_with_saved_after=apply_with_saved_after,
        get_packed_args=get_packed_args,
    )
<|endoftext|># Generates VariableType.h/cpp
#
# **If any changes are being made to the VariableType codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
#
# VariableType is a subclass of at::Type that provides the binding code
# necessary to provide a differentiable version of ATen operators. There are a
# number of different things we could mean:
#
#   - Given a non-differentiable forward implementation, we might
#     directly associate it with a backward implementation to make
#     it differentiable.  This is the common case.
#
#   - Some functions don't need a backwards implementation, because
#     backpropagation will never propagate beyond them.  There are a
#     number of different reasons why this may be the case:
#
#       - The function has no differentiable inputs
#       - The function's output is not differentiable
#       - The function has no data dependency on its input
#
#   - Some function don't need a backwards implementation because they
#     are implemented as a composition of other (differentiable) ATen
#     functions.  These are dispatched directly to the Type superclass,
#     which will in turn dispatch back to VariableType for its
#     differentiable subcomponents.
#

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from torchgen.api import cpp
from torchgen.api.autograd import (
    DifferentiableInput,
    dispatch_strategy,
    ForwardDerivative,
    gen_differentiable_outputs,
    is_differentiable,
    NativeFunctionWithDifferentiabilityInfo,
    SavedAttribute,
)
from torchgen.api.types import (
    ArrayRefCType,
    BaseCppType,
    BaseCType,
    Binding,
    intArrayRefT,
    iTensorListRefT,
    ListCType,
    MutRefCType,
    OptionalCType,
    scalarT,
    SpecialArgName,
    stringT,
    symIntArrayRefT,
    TENSOR_LIST_LIKE_CTYPES,
    tensorListT,
    tensorT,
    TupleCType,
    VectorCType,
)
from torchgen.code_template import CodeTemplate
from torchgen.context import (
    native_function_manager,
    with_native_function,
    with_native_function_and,
)
from torchgen.model import (
    Argument,
    BaseType,
    ListType,
    NativeFunction,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
)
from torchgen.utils import FileManager, mapMaybe

from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
    ALL_VIEW_FUNCTIONS,
    ASSIGN_RETURN_VALUE,
    AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION,
    gen_formals,
    get_base_name,
    get_view_info,
    is_tensor_list_type,
    is_tensor_type,
    METHOD_DEFINITION,
    modifies_arguments,
    TMP_VAR,
    unpack_args,
    unpacked_name,
    use_derived,
    WRAPPER_REGISTRATION,
)
from .gen_trace_type import (
    get_return_value,
    MANUAL_AUTOGRAD_AND_TRACER,
    MANUAL_BACKEND,
    tie_return_values,
    type_wrapper_name,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# We don't set or modify grad_fn on these methods. Generally, they return
# tensors that have requires_grad=False. In-place functions listed here will
# not examine or modify requires_grad or grad_fn.
# NB: this does NOT include overload name
DONT_REQUIRE_DERIVATIVE = {
    # These only depend on the input Tensor's shape and device, not the data
    "empty_like",
    "ones_like",
    "full_like",
    "zeros_like",
    "rand_like",
    "randn_like",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_zeros",
    "new_ones",
    # These are only implemented on integral types
    "__and__",
    "__iand__",
    "__ilshift__",
    "__ior__",
    "__irshift__",
    "__ixor__",
    "__lshift__",
    "__or__",
    "__rshift__",
    "__xor__",
    # These work on integral data types, and hence don't require derivative
    "_sobol_engine_draw",
    "_sobol_engine_ff",
    "_sobol_engine_scramble_",
    "_sobol_engine_initialize_state_",
    # This is an unsafe method that is meant to be out of reach of autograd.
    "_coalesced_",
    # Quantize functions should not record gradients
    "quantize_per_tensor",
    "quantize_per_channel",
    # Functions that return integers should not have output that require gradients
    "argmax",
    "argmin",
    "argsort",
    "searchsorted",
    "bucketize",
    # Functions that return booleans are not differentiable
    "isnan",
    "isposinf",
    "isneginf",
    "isinf",
    "signbit",
    "isin",
    "allclose",
    # Functions return none are not differentiable
    "record_stream",
    # These functions are not differentiable
    "logical_and",
    "logical_xor",
    "logical_not",
    "logical_or",
    # This function returns nested_tensor shape as a tensor that is non-differentiable
    "_nested_tensor_size",
    "_nested_tensor_strides",
    "_nested_tensor_storage_offsets",
}

# The C -> R functions at the time of adding this are still being audited and tested
# but will not error out.
# C -> C, R -> C functions for which backward is correctly implemented and tested
GRADIENT_IMPLEMENTED_FOR_COMPLEX = {
    "fill",
    "t",
    "t_copy",
    "view",
    "reshape",
    "reshape_as",
    "view_as",
    "view_copy",
    "roll",
    "clone",
    "block_diag",
    "diag_embed",
    "repeat",
    "expand",
    "expand_copy",
    "flip",
    "fliplr",
    "flipud",
    "rot90",
    "nanmean",
    "nansum",
    "transpose",
    "transpose_copy",
    "permute",
    "permute_copy",
    "squeeze",
    "squeeze_copy",
    "unsqueeze",
    "unsqueeze_copy",
    "resize",
    "resize_as",
    "tril",
    "triu",
    "chunk",
    "zero_",
    "eq_",
    "ne_",
    "add",
    "__radd__",
    "sum",
    "_conj",
    "sin",
    "cos",
    "mul",
    "sinc",
    "sinh",
    "cosh",
    "__rmul__",
    "sgn",
    "asin",
    "acos",
    "sub",
    "div",
    "cat",
    "view_as_complex",
    "index_put",
    "neg",
    "complex",
    "select",
    "where",
    "as_strided",
    "as_strided_copy",
    "as_strided_scatter",
    "slice",
    "constant_pad_nd",
    "unbind",
    "unbind_copy",
    "split",
    "split_with_sizes",
    "unsafe_split",
    "split_with_sizes_backward",
    "dot",
    "vdot",
    "triangular_solve",
    "mm",
    "_unsafe_view",
    "mv",
    "outer",
    "bmm",
    "diagonal",
    "alias",
    "atan",
    "ldexp",
    "linear",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logsumexp",
    "logcumsumexp",
    "reciprocal",
    "tan",
    "pow",
    "rsqrt",
    "tanh",
    "tanh_backward",
    "asinh",
    "acosh",
    "atanh",
    "take",
    "fill_",
    "exp",
    "exp2",
    "expm1",
    "nonzero",
    "mean",
    "std_mean",
    "var_mean",
    "inverse",
    "solve",
    "linalg_cholesky",
    "addcmul",
    "addcdiv",
    "matrix_exp",
    "linalg_matrix_exp",
    "linalg_matrix_sqrth",
    "linalg_polar",
    "_linalg_eigh",
    "cholesky_solve",
    "linalg_qr",
    "_linalg_svd",
    "_fft_c2c",
    "_fft_r2c",
    "linalg_solve",
    "sqrt",
    "stack",
    "gather",
    "index_select",
    "index_add_",
    "linalg_inv",
    "linalg_inv_ex",
    "baddbmm",
    "addbmm",
    "addmm",
    "addmv",
    "addr",
    "linalg_householder_product",
    "ormqr",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "linalg_cholesky_ex",
    "linalg_eig",
    "diagonal_copy",
    "diagonal_scatter",
    "alias_copy",
    "select_backward",
    "diagonal_backward",
    "slice_backward",
    "reflection_pad1d_backward",
    "reflection_pad2d_backward",
    "reflection_pad3d_backward",
    "_sparse_sparse_matmul",
    "replication_pad1d",
    "replication_pad2d",
    "replication_pad3d",
    "put",
    "put_",
    "_to_copy",
    "replication_pad1d_backward",
    "replication_pad2d_backward",
    "replication_pad3d_backward",
    "diag",
    "masked_scatter",
    "masked_select",
    "index_add",
    "index_fill",
    "trace",
    "polar",
    "cumsum",
    "rsub",
    "eig",
    "lerp",
    "linalg_vector_norm",
    "cumprod",
    "prod",
    "index_copy",
    "lu",
    "unfold",
    "unfold_backward",
    "index",
    "masked_fill",
    "masked_scatter_backward",
    "linalg_cross",
    "lu_unpack",
    "renorm",
    "_conj_physical",
    "linalg_lu_factor_ex",
    "scatter",
    "scatter_add",
    "sigmoid",
    "sigmoid_backward",
    "sparse_mask",
    "trapezoid",
    "cumulative_trapezoid",
    "conj_physical_",
    "_neg_view",
    "_reshape_alias",
    "_reshape_copy",
    "narrow_copy",
    "_linalg_det",
    "lu_solve",
    "linalg_solve_triangular",
    "linalg_pinv",
    "linalg_lstsq",
    "unfold_copy",
    "col2im",
    "im2col",
    "cholesky_inverse",
    "to_sparse",
    "sparse_sampled_addmm",
    "linalg_lu",
    "pixel_shuffle",
    "pixel_unshuffle",
    "channel_shuffle",
    "linalg_lu_solve",
    "_linalg_slogdet",
    "_linalg_solve_ex",
    "_unsafe_index",
    "_unsafe_index_put",
    "_unsafe_masked_index",
    "_unsafe_masked_index_put_accumulate",
}

GRADIENT_IMPLEMENTED_FOR_SPARSE_COMPLEX = {
    "_to_dense",
    "_coalesce",
    "coalesce",
    "values",
    "_sparse_coo_tensor_with_dims_and_tensors",
    "_sparse_addmm",
}

GRADIENT_IMPLEMENTED_FOR_COMPLEX.update(GRADIENT_IMPLEMENTED_FOR_SPARSE_COMPLEX)

# Some operators invalidate the grad_accumulator. Let's reset it.
RESET_GRAD_ACCUMULATOR = {"set_", "resize_"}

# NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
#
# We check the following properties:
#   1) A function should never change the input tensors' underlying c10::TensorImpl
#      pointers or c10::Storage pointers, even if it modifies its input tensors (via
#      inplace or out-variants)
# If the function does not modify its arguments, we also check the following properties
# pertaining to its output:
#   2) Its TensorImpl has use_count of 1 (or 2 if it has a PyObject)
#   3) If the function is a view function, it has the same StorageImpl as that of
#      the input it is aliased with. Otherwise, its StorageImpl has use_count of 1
#
# The following code templates implement the checks for this invariant:
SAVE_TENSOR_STORAGE = CodeTemplate(
    """\
auto ${tensor_name}_storage_saved =
  ${tensor_name}.has_storage() ? ::std::optional<Storage>(${tensor_name}.storage()) : ::std::nullopt;
"""
)


# If tensor_name == out_tensor_name, used to enforce (1), otherwise used for (2)
ENFORCE_SAME_TENSOR_STORAGE = CodeTemplate(
    """\
if (${tensor_name}_storage_saved.has_value() &&
    !at::impl::dispatch_mode_enabled() &&
    !at::impl::tensor_has_dispatch(${tensor_name}) &&
    !at::impl::tensor_has_dispatch(${out_tensor_name}))
  TORCH_INTERNAL_ASSERT(${tensor_name}_storage_saved.value().is_alias_of(${out_tensor_name}.storage()));
"""
)

SAVE_TENSORLIST_STORAGE = CodeTemplate(
    """\
std::vector<::std::optional<Storage>> ${tensorlist_name}_storage_saved(${tensorlist_name}.size());
for (const Tensor& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_storage() ? ::std::optional<Storage>(tensor.storage()) : ::std::nullopt);
"""
)

ENFORCE_SAME_TENSORLIST_STORAGE = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_storage_saved[i].has_value() && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
    TORCH_INTERNAL_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(${tensorlist_name}[i].storage()));
}
"""
)

SAVE_OPTIONALTENSORLIST_STORAGE = CodeTemplate(
    """\
std::vector<::std::optional<Storage>> ${tensorlist_name}_storage_saved(${tensorlist_name}.size());
for (const ::std::optional<Tensor>& tensor : ${tensorlist_name})
  ${tensorlist_name}_storage_saved.push_back(
    tensor.has_value() && tensor->has_storage() ? ::std::optional<Storage>(tensor->storage()) : ::std::nullopt);
"""
)

ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_storage_saved[i].has_value() && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
    TORCH_INTERNAL_ASSERT(${tensorlist_name}_storage_saved[i].value().is_alias_of(
        static_cast<::std::optional<Tensor>>(${tensorlist_name}[i])->storage()));
}
"""
)

SAVE_TENSOR_IMPL = CodeTemplate(
    """\
c10::intrusive_ptr<TensorImpl> ${tensor_name}_impl_saved;
if (${tensor_name}.defined()) ${tensor_name}_impl_saved = ${tensor_name}.getIntrusivePtr();
"""
)

ENFORCE_SAME_TENSOR_IMPL = CodeTemplate(
    """\
if (${tensor_name}_impl_saved && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name}))
  TORCH_INTERNAL_ASSERT(${tensor_name}_impl_saved == ${tensor_name}.getIntrusivePtr());
"""
)

ENFORCE_TENSOR_IMPL_USE_COUNT = CodeTemplate(
    """\
if (!at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name}))
  TORCH_INTERNAL_ASSERT(${tensor_name}.use_count() == expected_fresh_use_count(${tensor_name}), "function: ${fn_name}");
"""
)

ENFORCE_TENSOR_STORAGE_USE_COUNT_EQUALS_ONE = CodeTemplate(
    """\
if (${tensor_name}.has_storage() && !at::impl::dispatch_mode_enabled() && !at::impl::tensor_has_dispatch(${tensor_name})) {
  TORCH_INTERNAL_ASSERT(${tensor_name}.storage().use_count() == 1, "function: ${fn_name}");
}
"""
)

SAVE_TENSORLIST_IMPL = CodeTemplate(
    """\
std::vector<c10::intrusive_ptr<TensorImpl>> ${tensorlist_name}_impl_saved(${tensorlist_name}.size());
for (size_t i=0; i<${tensorlist_name}.size(); i++)
  if (${tensorlist_name}[i].defined()) ${tensorlist_name}_impl_saved[i] = ${tensorlist_name}[i].getIntrusivePtr();
"""
)

ENFORCE_SAME_TENSORLIST_IMPL = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_impl_saved[i] && !at::impl::tensorlist_has_dispatch(${tensorlist_name}))
    TORCH_INTERNAL_ASSERT(${tensorlist_name}_impl_saved[i] == ${tensorlist_name}[i].getIntrusivePtr());
}
"""
)

SAVE_OPTIONALTENSORLIST_IMPL = CodeTemplate(
    """\
std::vector<c10::intrusive_ptr<TensorImpl>> ${tensorlist_name}_impl_saved(${tensorlist_name}.size());
for (size_t i=0; i<${tensorlist_name}.size(); i++) {
  ::std::optional<Tensor> t = ${tensorlist_name}[i];
  if (t.has_value() && t->defined()) ${tensorlist_name}_impl_saved[i] = t->getIntrusivePtr();
}
"""
)

ENFORCE_SAME_OPTIONALTENSORLIST_IMPL = CodeTemplate(
    """\
for (size_t i=0; i<${tensorlist_name}.size() && !at::impl::dispatch_mode_enabled(); i++) {
  if (${tensorlist_name}_impl_saved[i])
    TORCH_INTERNAL_ASSERT(
      ${tensorlist_name}_impl_saved[i] == static_cast<::std::optional<Tensor>>(${tensorlist_name}[i])->getIntrusivePtr());
}
"""
)

# The following list contains functions that we don't enforce the invariant on.
DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE = {
    # These functions are expected to change impl or storage of input tensors
    "set_",
    "_cudnn_rnn_flatten_weight",
    "_unsafe_masked_index",
    "_unsafe_masked_index_put_accumulate",
}
DONT_ENFORCE_TENSOR_IMPL_USE_COUNT = {
    # These non-inplace, non-out functions return tensors with use_count > 1
    # Therefore, they MAY (but not necessarily) return one of its inputs as-is
    # See https://github.com/pytorch/pytorch/issues/60426 for more information
    "_embedding_bag",
    "_embedding_bag_forward_only",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "lu_unpack",
    "_cudnn_rnn_backward",
    # The below failed StorageImpl use_count check but we skip tensor_impl check
    # just in case
    "_cudnn_rnn",
    "dequantize_self",
    # lift() should never actually be called with a requires_grad=True tensor,
    "lift",
    "lift_fresh",
    "lift_fresh_copy",
    # Nested Tensors related functions
    # _nested_tensor_size() should never actually be called with requires_grad=True tensor
    "_nested_tensor_size",
    "_nested_tensor_strides",
    "_nested_tensor_storage_offsets",
}

DONT_ENFORCE_STORAGE_IMPL_USE_COUNT = {
    # These non-view functions return tensors with storage use_count != 1
    "_slow_conv2d_forward",
    "slow_conv3d_forward",
    "channel_shuffle",
    # If an input is returned as-is in output, we cannot guarantee its storage_impl
    # use count to be 1 either.
    *DONT_ENFORCE_TENSOR_IMPL_USE_COUNT,
}
# END CHECKS FOR [ TensorImpl and Storage Pointer Sanity Checks ]

DECLARE_GRAD_FN = CodeTemplate(
    """\
c10::intrusive_ptr<${op}> grad_fn;
"""
)

DECLARE_VECTOR_OF_GRAD_FN = CodeTemplate(
    """\
std::vector<c10::intrusive_ptr<${op}>> grad_fns;
"""
)

SETUP_ANY_REQUIRES_GRAD = CodeTemplate(
    """\
[[maybe_unused]] auto _any_requires_grad = compute_requires_grad( ${args_with_derivatives} );
${extra_differentiability_conditions}
"""
)

SETUP_DERIVATIVE = CodeTemplate(
    """\
if (_any_requires_grad) {
  ${setup}
}
"""
)

SETUP_NONE_REQUIRES_GRAD = CodeTemplate(
    """\
if (compute_requires_grad( ${args_to_check} )) {
  throw_error_out_requires_grad("${base_name}");
}
"""
)

ASSIGN_GRAD_FN = CodeTemplate(
    """\
grad_fn = c10::make_intrusive<${op}>(${op_ctor});
grad_fn->set_next_edges(collect_next_edges( ${args_with_derivatives} ));
"""
)

# note(crcrpar): `compute_requires_grad` in the template below is supplied with arguments indexed with `i`
# while the `SETUP_ANY_REQUIRES_GRAD` above takes whole tensors and scalars.
ASSIGN_VECTOR_OF_GRAD_FN = CodeTemplate(
    """\
for (const auto& i : c10::irange( ${irange} )) {
  const auto ith_requires_grad = compute_requires_grad(${args_with_derivatives});
  check_inplace(self[i], ith_requires_grad);
  grad_fns.push_back([&]() -> c10::intrusive_ptr<${op}> {
      if (!ith_requires_grad) {
          return nullptr;
      } else {
          auto grad_fn = c10::make_intrusive<${op}>(${op_ctor});
          grad_fn->set_next_edges(collect_next_edges( ${args_with_derivatives} ));
          return grad_fn;
      }
  }());
}
"""
)

CALL_REDISPATCH = CodeTemplate(
    """\
at::redispatch::${api_name}(${unpacked_args})"""
)
# If the non-variable operation has return values, we use the `tmp` variable to hold the
# values temporarily and pass the values to the return variables outside of the
# `at::AutoDispatchBelowAutograd` guard block.
DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES_JVP_DECOMP = CodeTemplate(
    """\
auto ${tmp_var} = ([&]() {
  if (${any_has_forward_grad}) {
    static c10::OperatorName full_name("aten::${op_name}", "${op_overload}");
    static ::std::optional<c10::OperatorHandle> opt_op = c10::Dispatcher::singleton().findSchema(full_name);
    return impl::run_jit_decomposition_with_args_for_jvp<${return_types}>("${op_name}", *opt_op, ks, ${arg_names});
  } else {
    ${guard}
    return ${base_type_call};
  }
})();
"""
)

DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES = CodeTemplate(
    """\
auto ${tmp_var} = ([&]() {
  ${guard}
  return ${base_type_call};
})();
"""
)

DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES = CodeTemplate(
    """\
{
  ${guard}
  ${base_type_call};
}
"""
)

SET_HISTORY = CodeTemplate(
    """\
if (grad_fn) {
    ${fn}_history(${differentiable_outputs}, grad_fn);
}
"""
)

# rebase_history returns the node actually attached as the new history (the
# CopySlices node when rebasing through a view). It is captured so that node
# creation hooks fire on the composed node, not the inner one.
REBASE_HISTORY = CodeTemplate(
    """\
c10::intrusive_ptr<Node> _rebased_fn;
if (grad_fn) {
    _rebased_fn = rebase_history(${differentiable_outputs}, grad_fn);
}
"""
)

# Fired at the very end of the function, after saved variables (including
# saved outputs) have been stored on grad_fn.
FIRE_NODE_CREATION_HOOKS = CodeTemplate(
    """\
if (${node}) {
    fire_node_creation_hooks(${node});
}
"""
)

LOOP_OVER_VECTOR_OF_GRAD_FNS = CodeTemplate(
    """\
if (!grad_fns.empty()) {
    ${preamble}
    for (const auto& i : c10::irange(grad_fns.size())) {
        auto grad_fn = grad_fns[i];
        if (grad_fn != nullptr) {
            ${statements}
        }
    }
}
"""
)

CONDITIONAL = CodeTemplate(
    """\
if (${cond}) {
  ${statements}
}
"""
)

RUN_ONLY_IN_DEBUG_MODE = CodeTemplate(
    """\
#ifndef NDEBUG
${statements}
#endif
"""
)

FW_DERIVATIVE_CHECK_TEMPLATE = CodeTemplate(
    """\
isFwGradDefined(${req_inp})\
"""
)
FW_DERIVATIVE_SIZE_CHECK_TEMPLATE = CodeTemplate(
    """\
TORCH_CHECK(
    self.size() == ${inp_name}.size(),
      "Tensor lists must have the same number of tensors, got ",
    self.size(),
      " and ",
    ${inp_name}.size());
"""
)

FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE = CodeTemplate(
    """\
isFwGradDefinedTensorList(${req_inp})\
"""
)

FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE = CodeTemplate(
    """\
auto ${inp_name}_t_raw = toNonOptFwGrad(${inp});
auto ${inp_name}_tensor = toNonOptTensor(${inp});
auto ${inp_name}_t = (${inp_name}_t_raw.defined() || !${inp_name}_tensor.defined())
  ? ${inp_name}_t_raw : at::${zeros_fn}(${inp_name}_tensor.sym_sizes(), ${inp_name}_tensor.options());
"""
)

FW_DERIVATIVE_UPDATE_WRAPPED_NUM_TEMPLATE = CodeTemplate(
    """\
update_wrapped_number(${inp_name}_tensor, ${inp_name}_t);
"""
)

FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE = CodeTemplate(
    """\
auto ${inp_name}_p = toNonOptPrimal(${inp});
"""
)

FW_DERIVATIVE_SETTER_TENSOR = CodeTemplate(
    """\
if (${out_arg}_new_fw_grad_opt.has_value() && ${out_arg}_new_fw_grad_opt.value().defined() && ${out_arg}.defined()) {
  // The hardcoded 0 here will need to be updated once we support multiple levels.
  ${out_arg}._set_fw_grad(${out_arg}_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ ${is_inplace});
}
"""
)

FW_DERIVATIVE_SETTER_TENSOR_FOREACH = CodeTemplate(
    """\
for (const auto& i : c10::irange(${out_arg}_new_fw_grad_opts.size())) {
  auto& ${out_arg}_new_fw_grad_opt = ${out_arg}_new_fw_grad_opts[i];
  if (${out_arg}_new_fw_grad_opt.has_value() && ${out_arg}_new_fw_grad_opt.value().defined() && ${out_arg}[i].defined()) {
    // The hardcoded 0 here will need to be updated once we support multiple levels.
    ${out_arg}[i]._set_fw_grad(${out_arg}_new_fw_grad_opt.value(), /* level */ 0, /* is_inplace_op */ ${is_inplace});
  }
}
"""
)

FW_DERIVATIVE_SETTER_MULTI_OUTPUT = CodeTemplate(
    """\
if (${all_res}_new_fw_grad_opt.has_value() && std::get<${idx}>(${all_res}_new_fw_grad_opt.value()).defined()
    && ${out_arg}.defined()) {
  ${out_arg}._set_fw_grad(std::get<${idx}>(${all_res}_new_fw_grad_opt.value()), /* level */ 0, /* is_inplace_op */ false);
}
"""
)

FW_DERIVATIVE_SETTER_TENSOR_LIST = CodeTemplate(
    """\
if (${out_arg}_new_fw_grad_opt.has_value()) {
  auto ${out_arg}_new_fw_grad = ${out_arg}_new_fw_grad_opt.value();
  TORCH_INTERNAL_ASSERT(${out_arg}.size() == ${out_arg}_new_fw_grad.size());
  for (const auto i : c10::irange(${out_arg}.size())) {
    if (${out_arg}_new_fw_grad[i].defined() && ${out_arg}[i].defined()) {
      // The hardcoded 0 here will need to be updated once we support multiple levels.
      ${out_arg}[i]._set_fw_grad(${out_arg}_new_fw_grad[i], /* level */ 0, /* is_inplace_op */ ${is_inplace});
    }
  }
}
"""
)

FW_DERIVATIVE_TEMPLATE = CodeTemplate(
    """\
${fw_grad_opt_definition}
if (${requires_fw_grad}) {
    ${unpacked_arguments}
    ${out_arg}_new_fw_grad_opt = ${formula};
}
"""
)

FW_DERIVATIVE_FOREACH_TEMPLATE = CodeTemplate(
    """\
${fw_grad_opt_definition}
for (const auto& i : c10::irange(${vector_of_optional_tensor}.size())) {
  if (${any_has_forward_grad_for_current_index}) {
      ${unpacked_arguments}
      ${vector_of_optional_tensor}[i] = ${formula};
  }
}
"""
)

FW_DERIVATIVE_FORBID_TEMPLATE = CodeTemplate(
    """\
TORCH_CHECK_NOT_IMPLEMENTED(!(${cond}), "Trying to use forward AD with ${name} that does not support it ${msg}");
"""
)

FW_DERIVATIVE_FORBID_LIST_TEMPLATE = CodeTemplate(
    """\
for (const auto& _t: ${arg}) {
    TORCH_CHECK_NOT_IMPLEMENTED(!(${cond}), "Trying to use forward AD with ${name} that does not support it ${msg}");
}
"""
)


def gen_variable_type(
    out: str,
    native_yaml_path: str,
    tags_yaml_path: str,
    fns_with_diff_infos: list[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
    used_keys: set[str],
) -> None:
    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write(
        "VariableType.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/VariableType.h"
        },
    )

    # helper that generates a TORCH_LIBRARY_IMPL macro for each
    # dispatch key that appears in derivatives.yaml
    def wrapper_registrations(used_keys: set[str]) -> str:
        library_impl_macro_list: list[str] = []
        for key in sorted(used_keys):
            dispatch_key = key
            if key == "Default":
                dispatch_key = "Autograd"
            library_impl_macro = (
                f"TORCH_LIBRARY_IMPL(aten, {dispatch_key}, m) "
                + "{\n"
                + "${"
                + f"wrapper_registrations_{key}"
                + "}\n}"
            )
            library_impl_macro_list += [library_impl_macro]
        return "\n\n".join(library_impl_macro_list)

    # Generate a new template from VariableType.cpp which replaces ${wrapper_registrations}
    # with per key TORCH_LIBRARY_IMPL macros for each key that appears in derivatives.yaml
    fm1 = FileManager(
        install_dir=out + "/templates", template_dir=template_path, dry_run=False
    )
    fm1.write(
        "VariableType.cpp",
        lambda: {
            "type_derived_method_definitions": "\n\n".join(
                [
                    "${" + f"type_derived_method_definitions_{key}" + "}"
                    for key in sorted(used_keys)
                ]
            ),
            "wrapper_registrations": wrapper_registrations(used_keys),
        },
    )

    # Generate final VariableType_*.cpp files from the generated template
    fm2 = FileManager(install_dir=out, template_dir=out + "/templates", dry_run=False)

    sharded_keys = set(
        [f"type_derived_method_definitions_{key}" for key in sorted(used_keys)]
        + [f"wrapper_registrations_{key}" for key in sorted(used_keys)]
    )
    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.
    fm2.write_sharded(
        "VariableType.cpp",
        [fn for fn in fns_with_diff_infos if use_derived(fn)],
        key_fn=lambda fn: cpp.name(fn.func.func),
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/VariableType.cpp",
        },
        env_callable=gen_variable_type_func,
        num_shards=10,
        sharded_keys=sharded_keys,
    )


@with_native_function_and
def gen_wrapper_registration(f: NativeFunction, key: str = "Default") -> str:
    return WRAPPER_REGISTRATION.substitute(
        unqual_operator_name_with_overload=f.func.name,
        type_wrapper_name=type_wrapper_name(f, key),
        class_type="VariableType",
    )


def gen_variable_type_func(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> dict[str, list[str]]:
    f = fn.func
    result = {}
    with native_function_manager(f):
        name = cpp.name(f.func)
        formals = gen_formals(f)

        if (
            fn.info is None
            and str(f.func.name.name) not in RESET_GRAD_ACCUMULATOR
            and get_base_name(f) not in DONT_REQUIRE_DERIVATIVE
            and len(gen_differentiable_outputs(fn)) > 0
            and cpp.name(f.func) not in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE
            and type_wrapper_name(f) not in DONT_ENFORCE_STORAGE_IMPL_USE_COUNT
            and type_wrapper_name(f) not in DONT_ENFORCE_TENSOR_IMPL_USE_COUNT
        ):
            # NOTE: [ Registering AutogradNotImplemented boxed kernel ]
            #
            # When there is no derivatives.yaml entry, we register a generic boxed
            # NotImplemented kernel to set grad_fn to be NotImplemented, so that forward
            # proceeds as usual but an error is properly produced on backward.
            # TODO: it would be nice to not have these special cases
            #
            # There are several cases where still let codegen handle it:
            # 1) ops that need to reset grad accumulator (we let codegen handle this case
            #     because) the list is (currently) only accessible in Python.
            # 2) User explicitly specifies DONT_REQUIRE_DERIVATIVE. This basically makes
            #    autograd a fallthrough with NDEBUG checks. This can be useful for when all
            #    outputs are integral.
            # 3) When there are no differentiable outputs. This is similar to (2).
            # 4) There are certain ops where we skip certain NDEBUG checks. this is similar
            #    to (1).
            type_definition = ""
            wrapper_registration = AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION.substitute(
                unqual_operator_name_with_overload=f.func.name
            )
            result["type_derived_method_definitions_Default"] = [type_definition]
            result["wrapper_registrations_Default"] = [wrapper_registration]
        else:
            if not fn.info:
                key = "Default"
                type_definition = METHOD_DEFINITION.substitute(
                    return_type=cpp.returns_type(
                        f.func.returns, symint=True
                    ).cpp_type(),
                    type_wrapper_name=type_wrapper_name(f, key),
                    type_definition_body=emit_body(fn, key),
                    formals=formals,
                )
                wrapper_registration = gen_wrapper_registration(f, key)
                result[f"type_derived_method_definitions_{key}"] = [type_definition]
                result[f"wrapper_registrations_{key}"] = [wrapper_registration]
            else:
                for key in fn.info:
                    type_definition = METHOD_DEFINITION.substitute(
                        return_type=cpp.returns_type(
                            f.func.returns, symint=True
                        ).cpp_type(),
                        type_wrapper_name=type_wrapper_name(f, key),
                        type_definition_body=emit_body(fn, key),
                        formals=formals,
                    )
                    wrapper_registration = gen_wrapper_registration(f, key)
                    result[f"type_derived_method_definitions_{key}"] = [type_definition]
                    result[f"wrapper_registrations_{key}"] = [wrapper_registration]
    # See Note [Manual Backend kernels]
    if (name in MANUAL_BACKEND) != f.manual_kernel_registration:
        raise AssertionError(
            f"(name in MANUAL_BACKEND) != f.manual_kernel_registration: {name in MANUAL_BACKEND} != {f.manual_kernel_registration}"
        )
    # If you want to register a kernel to Autograd, you must make the op abstract.
    # In other words, this op must have dispatch section in native_functions.yaml.
    if name in MANUAL_AUTOGRAD_AND_TRACER or (
        fn.info and any(info.has_derivatives for info in fn.info.values())
    ):
        if not f.is_abstract:
            raise AssertionError(
                f"There's a formula for {name}(or its functional variant) in derivatives.yaml. "
                f"It's required to add a dispatch section for it with explicit supported backends e.g CPU/CUDA "
                f"or CompositeExplicitAutograd in native_functions.yaml. Please see "
                f"https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native#choosing-the-right-dispatch-keyword "
                f"for instructions to choose the right dispatch keyword."
            )

    return result


_foreach_ops_without_differentiability_info = {
    # No reference backward available due to the lack of `{maximum, minimum}(tensor, scalar)`.
    ("_foreach_maximum", "Scalar"),
    ("_foreach_maximum", "ScalarList"),
    ("_foreach_minimum", "Scalar"),
    ("_foreach_minimum", "ScalarList"),
    # No reference backward available as addcdiv/addcmul don't support Tensor as scaling factor.
    ("_foreach_addcdiv", "Tensor"),
    ("_foreach_addcmul", "Tensor"),
    ("_foreach_copy", ""),
}

_foreach_ops_with_different_arity = {
    # These ops lack `alpha` of scaling factor to applied to the right hand side argument.
    ("_foreach_add", "Scalar"),
    ("_foreach_add", "ScalarList"),
    ("_foreach_sub", "Scalar"),
    ("_foreach_sub", "ScalarList"),
}


@with_native_function_with_differentiability_info_and_key
def emit_body(
    fn: NativeFunctionWithDifferentiabilityInfo, key: str = "Default"
) -> list[str]:
    if dispatch_strategy(fn) != "use_derived":
        raise AssertionError(
            f"dispatch_strategy(fn) is {dispatch_strategy(fn)}, expected 'use_derived'"
        )
    f = fn.func
    info = fn.info[key] if fn.info else None
    fw_derivatives = fn.fw_derivatives.get(key, []) if fn.fw_derivatives else []

    name = cpp.name(f.func)
    inplace = f.func.kind() == SchemaKind.inplace
    is_out_fn = f.func.kind() == SchemaKind.out
    returns_void = len(f.func.returns) == 0
    base_name = get_base_name(f)
    view_info = get_view_info(f)

    is_foreach = name.startswith("_foreach")
    is_inplace_foreach = is_foreach and inplace
    if is_inplace_foreach:
        inplace_foreacharg2refarg: dict[Argument, Argument] = {}
        refargname2inplace_foreacharg: dict[str, Argument] = {}
        base_name_and_overload_name = (f.func.name.name.base, f.func.name.overload_name)
        if info is None:
            if (
                base_name_and_overload_name
                not in _foreach_ops_without_differentiability_info
            ):
                raise AssertionError(
                    f"{'.'.join(base_name_and_overload_name)} should have a differentiability info"
                )
        else:
            if not (
                len(f.func.arguments.flat_non_out)
                == len(info.func.func.arguments.flat_non_out)
            ) and (
                base_name_and_overload_name not in _foreach_ops_with_different_arity
            ):
                raise AssertionError(
                    f"{'.'.join(base_name_and_overload_name)} has {len(f.func.arguments.flat_non_out)} args "
                    f"but the reference has {len(info.func.func.arguments.flat_non_out)}"
                )
            for foreach_arg, ref_arg in zip(
                f.func.arguments.flat_non_out, info.func.func.arguments.flat_non_out
            ):
                foreach_arg_type = foreach_arg.type
                if isinstance(foreach_arg_type, ListType):
                    foreach_arg_type = foreach_arg_type.elem
                if foreach_arg_type != ref_arg.type:
                    raise AssertionError(
                        f"foreach_arg_type ({foreach_arg_type}) != ref_arg.type ({ref_arg.type})"
                    )
                inplace_foreacharg2refarg[foreach_arg] = ref_arg
                refargname2inplace_foreacharg[ref_arg.name] = foreach_arg

    def gen_differentiable_input(
        arg: Argument | SelfArgument | TensorOptionsArguments,
    ) -> DifferentiableInput | None:
        if isinstance(arg, TensorOptionsArguments):
            return None
        a: Argument = arg.argument if isinstance(arg, SelfArgument) else arg

        # TODO: `cpp_type` is only to keep it byte-for-byte compatible with the old codegen, should remove.
        # NB: This is not a clone of cpp.argument() - TensorOptionsArguments / faithful / binds are
        # not handled properly as they are irrelevant for this codegen.
        cpp_type = cpp.argument_type(a, binds=a.name, symint=True).cpp_type()

        if not is_differentiable(a.name, a.type, info):
            return None
        return DifferentiableInput(
            name=a.name,
            type=a.type,
            cpp_type=cpp_type,
        )

    @with_native_function
    def gen_differentiable_inputs(f: NativeFunction) -> list[DifferentiableInput]:
        arguments = list(f.func.arguments.non_out)
        if is_inplace_foreach and info is not None:
            for i, arg in enumerate(f.func.arguments.flat_non_out):
                if arg in inplace_foreacharg2refarg:
                    # note(crcrpar): From what I understand, what matters is only the name.
                    # Thus originally I only replace argument only when the names are different.
                    # TODO(crcrpar): Make it simpler.
                    mapped_arg = inplace_foreacharg2refarg[arg]
                    arguments[i] = Argument(
                        mapped_arg.name,
                        mapped_arg.type,
                        mapped_arg.default,
                        mapped_arg.annotation,
                    )
        return list(mapMaybe(gen_differentiable_input, arguments))

    def find_args_with_derivatives(
        differentiable_inputs: list[DifferentiableInput],
    ) -> list[DifferentiableInput]:
        """Find arguments that have derivative definitions"""
        if info is None or not info.has_derivatives:
            return differentiable_inputs
        names = {name for d in info.derivatives for name in d.var_names}
        differentiable = [arg for arg in differentiable_inputs if arg.name in names]
        if len(differentiable) != len(names):
            missing = names - {arg.name for arg in differentiable}
            raise RuntimeError(
                f"Missing arguments for derivatives: {missing} in {info.name}"
            )
        return differentiable

    differentiable_inputs = gen_differentiable_inputs(f)
    args_with_derivatives = find_args_with_derivatives(differentiable_inputs)
    differentiable_outputs = gen_differentiable_outputs(fn, key)

    undifferentiable = (base_name in DONT_REQUIRE_DERIVATIVE) or (
        name in DONT_REQUIRE_DERIVATIVE
    )

    requires_derivative = (
        (not undifferentiable)
        and (len(differentiable_inputs) > 0)
        and (
            (len(differentiable_outputs) > 0)
            # note(crcrpar): In-place foreach functions are a void function.
            or is_inplace_foreach
        )
    )

    if (
        info is not None
        and info.has_derivatives
        and not requires_derivative
        # out= ops are allowed to have zero returns which cause requires_derivative to be False
        # we shouldn't error out though (out= ops for autograd just redispatch)
        and len(f.func.returns) > 0
    ):
        raise RuntimeError(
            f"ERROR: derivative ignored for {name} -- specified an autograd function without derivative"
        )

    # note(crcrpar): In-place foreach functions do not support forward AD
    if requires_derivative and len(fw_derivatives) > 0 and not is_inplace_foreach:
        num_fw_derivative_var_names = sum(
            len(derivative.var_names) for derivative in fw_derivatives
        )
        if num_fw_derivative_var_names != len(differentiable_outputs):
            raise AssertionError(
                f"Expected the number of forward derivatives implemented ({num_fw_derivative_var_names}) to match the "
                f"number of differentiable outputs ({len(differentiable_outputs)}). NB: This only applies when at least "
                "one forward derivative is implemented. Not implementing any forward "
                "derivatives is also okay, and we would require inputs to the op to "
                "not have associated tangents in that case."
            )

    try_jit_decomposition = (
        requires_derivative
        and len(fw_derivatives) == 0
        and (not modifies_arguments(f))
        and (not returns_void)
    )

    def emit_save_inputs() -> list[str]:
        setup: list[str] = []
        if info is None or not info.has_derivatives:
            return setup

        has_tensorlist_arg = any(
            is_tensor_list_type(arg.type) for arg in args_with_derivatives
        )

        # We don't want to save tensors if we know that they will never be used
        # when computing the derivative, so we add guards to those statements
        def guard_for(arg: SavedAttribute) -> str | None:
            if info is None:
                raise AssertionError("info is None in guard_for")

            # It's hard to determine the edge offset if we have TensorLists
            # NOTE(crcrpar): in-place foreach functions' arguments include tensorlist
            # but their derivatives don't use it, so let them bypass this check.
            if has_tensorlist_arg and (not is_inplace_foreach):
                return None

            # Empirical evaluation of the cases where we insert those guards in
            # backward show that they are somewhat useless. E.g. there's no need
            # to guard on some values captured from forward, because they had to
            # require_grad if the backward function even gets executed. I don't
            # have any good ideas for detecting those cases, so I simply disabled the
            # checks.
            if "backward" in info.name:
                return None

            # If there's a single derivative we could compute, we already have
            # a requires_grad check that is sufficient
            if len(args_with_derivatives) <= 1:
                return None

            # We really only care about trimming down the amount of tensors we save
            if arg.nctype.type != BaseCType(tensorT):
                return None

            # We want to emit simple guards, so we only allow that if checking one
            # input is enough to determine whether we need that value
            used_in = [d for d in info.derivatives if arg in d.saved_inputs]
            if len(used_in) == 0:
                raise AssertionError(f"used_in is empty for arg {arg.nctype.name}")
            if len(used_in) != 1:
                return None
            derivative = used_in[0]

            # Case with multioutput formulas
            # TODO: process all derivative formulas!!!
            if len(derivative.var_names) != 1:
                wrap_opt_if_start = derivative.formula.find(
                    f"wrap_opt_if({arg.nctype.name}"
                )
                if wrap_opt_if_start == -1:
                    return None

                wrap_opt_if_match = re.match(
                    rf"wrap_opt_if\({arg.nctype.name},(.*?)\)",
                    derivative.formula[wrap_opt_if_start:],
                )
                if wrap_opt_if_match is None:
                    raise AssertionError(
                        f"wrap_opt_if_match is None for {arg.nctype.name} in {derivative.formula}"
                    )

                # Condition is between 'wrap_opt_if(var_name,' and ')'.
                condition_slice = slice(len(rf"wrap_opt_if\({arg.nctype.name},"), -1)
                wrap_opt_if_condition = wrap_opt_if_match.group(0)[
                    condition_slice
                ].strip()
                # replace 'grad_input_mask[num]' with 'grad_fn->should_compute_output(num)'
                wrap_opt_if_condition = re.sub(
                    r"grad_input_mask\[(\d+)\]",
                    r"grad_fn->should_compute_output(\1)",
                    wrap_opt_if_condition,
                )
                return f"{wrap_opt_if_condition}"

            # Figure out the offset of the edge that uses this variable
            derivative_var_name = derivative.var_names[0]
            for edge_off, a in enumerate(args_with_derivatives):
                if a.name == derivative_var_name:
                    break
            else:
                raise AssertionError
            return f"grad_fn->should_compute_output({edge_off})"

        if is_inplace_foreach:
            save_input_stmts = save_variables(info.all_saved_inputs, False, guard_for)
            if save_input_stmts:
                setup.append(
                    LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(
                        preamble="", statements=save_input_stmts
                    )
                )
        else:
            setup.extend(save_variables(info.all_saved_inputs, False, guard_for))
            for arg in args_with_derivatives:
                if is_tensor_list_type(arg.type):
                    setup.append(f"grad_fn->{arg.name}_size_ = {arg.name}.size();")
        return setup

    def setup_derivative(differentiable_inputs: list[DifferentiableInput]) -> list[str]:
        body: list[str] = []
        if is_out_fn:
            # For out functions, ensure that no input or output requires grad
            body.append(DECLARE_GRAD_FN.substitute(op="Node"))
            body.append(
                SETUP_NONE_REQUIRES_GRAD.substitute(
                    base_name=base_name,
                    args_to_check=[arg.name for arg in differentiable_inputs],
                )
            )
            body.append(
                SETUP_NONE_REQUIRES_GRAD.substitute(
                    base_name=base_name,
                    args_to_check=[arg.name for arg in differentiable_outputs],
                )
            )
            return body

        op = info.op if info is not None and info.has_derivatives else "NotImplemented"
        setup = []
        if not is_inplace_foreach:
            setup.extend(
                ASSIGN_GRAD_FN.substitute(
                    op=op,
                    op_ctor=""
                    if info is not None and info.has_derivatives
                    else f'"{cpp.name(f.func)}"',
                    args_with_derivatives=[arg.name for arg in args_with_derivatives],
                ).split("\n")
            )
        else:
            # note(crcrpar): Assuming in-place foreach function's self_arg is always TensorList.
            list_like_arg = "self"
            args = [arg.name for arg in args_with_derivatives]
            for i, arg in enumerate(args):
                if is_inplace_foreach and info is not None:
                    if arg in refargname2inplace_foreacharg:
                        foreach_arg = refargname2inplace_foreacharg[arg]
                        args[i] = foreach_arg.name + (
                            "[i]" if isinstance(foreach_arg.type, ListType) else ""
                        )
                else:
                    if arg == list_like_arg:
                        args[i] = arg + "[i]"
            setup.extend(
                ASSIGN_VECTOR_OF_GRAD_FN.substitute(
                    op=op,
                    op_ctor=""
                    if info is not None and info.has_derivatives
                    else f'"{cpp.name(f.func)}"',
                    args_with_derivatives=args,
                    irange=f"{list_like_arg}.size()",
                ).split("\n")
            )
        setup.extend(emit_save_inputs())

        body.extend(
            emit_check_no_requires_grad(differentiable_inputs, args_with_derivatives)
        )
        declare_grad_fn_template = (
            DECLARE_GRAD_FN if not is_inplace_foreach else DECLARE_VECTOR_OF_GRAD_FN
        )
        body.append(declare_grad_fn_template.substitute(op=op))
        body.append(SETUP_DERIVATIVE.substitute(setup=setup))
        return body

    def emit_check_if_in_complex_autograd_allowlist() -> list[str]:
        body: list[str] = []
        if base_name in GRADIENT_IMPLEMENTED_FOR_COMPLEX:
            return body
        for arg in differentiable_outputs:
            name = arg.name
            # TODO: should be `arg.type.is_tensor_like()`?
            if arg.cpp_type == "at::Tensor" or arg.cpp_type in TENSOR_LIST_LIKE_CTYPES:
                body.append(f'throw_error_for_complex_autograd({name}, "{base_name}");')
        return body

    def emit_check_no_requires_grad(
        tensor_args: list[DifferentiableInput],
        args_with_derivatives: list[DifferentiableInput],
    ) -> list[str]:
        """Checks that arguments without derivatives don't require grad"""
        body: list[str] = []
        for arg in tensor_args:
            if arg in args_with_derivatives:
                continue
            arg_name = arg.name
            if info and arg_name in info.non_differentiable_arg_names:
                continue
            if arg_name == "output":
                # Double-backwards definitions sometimes take in 'input' and
                # 'output', but only define the derivative for input.
                continue
            body.append(f'check_no_requires_grad({arg_name}, "{arg_name}", "{name}");')
        return body

    def emit_original_self_definition() -> list[str]:
        body: list[str] = []
        if inplace:
            if is_inplace_foreach:
                body.append(
                    "std::vector<::std::optional<at::Tensor>> original_selfs(self.size());"
                )
            else:
                body.append("::std::optional<at::Tensor> original_self;")

            all_forward_grad_cond = []
            for derivative in fw_derivatives:
                if derivative.required_original_self_value:
                    all_forward_grad_cond.append(
                        get_any_has_forward_grad_name(derivative.var_names)
                    )

            if all_forward_grad_cond:
                if not is_inplace_foreach:
                    body.append(f"if ({' || '.join(all_forward_grad_cond)}) {{")
                    body.append("  original_self = self.clone();")
                    body.append("}")
                else:
                    current_all_forward_grad_cond = [
                        f"{cond}[i]" for cond in all_forward_grad_cond
                    ]
                    body.append("for (const auto& i : c10::irange(self.size())) {")
                    body.append(
                        f"  if ({' || '.join(current_all_forward_grad_cond)}) {{"
                    )
                    body.append("    original_selfs[i] = self[i].clone();")
                    body.append("  }")
                    body.append("}")

        return body

    def save_variables(
        saved_variables: Sequence[SavedAttribute],
        is_output: bool,
        guard_for: Callable[[SavedAttribute], str | None] = lambda name: None,
    ) -> Sequence[str]:
        # assign the saved variables to the generated grad_fn
        stmts: list[str] = []
        for arg in sorted(saved_variables, key=lambda sa: str(sa.nctype.name)):
            name = (
                arg.nctype.name.name
                if isinstance(arg.nctype.name, SpecialArgName)
                else arg.nctype.name
            )
            foreacharg: Argument | None = None
            is_foreacharg_list_type: bool = False
            type = arg.nctype.type
            expr = arg.expr
            stmts_prepend = None
            if is_inplace_foreach and info is not None:
                # todo(crcrpar): See if we can add some check e.g. `assert foreacharg is not None`.
                # for now the example assert would fail.
                name_to_query = name.split("_scalar_type")[0]
                if name_to_query in refargname2inplace_foreacharg:
                    foreacharg = refargname2inplace_foreacharg[name_to_query]
                    is_foreacharg_list_type = isinstance(foreacharg.type, ListType)
                if foreacharg is not None:
                    name_in_expr = (
                        f"{foreacharg.name}{'[i]' if is_foreacharg_list_type else ''}"
                    )
                    src_name = name
                    if "_scalar_type" in src_name:
                        split_src_name = src_name.split("_scalar_type")
                        if len(split_src_name) != 2:
                            raise AssertionError(
                                f"expected 2 parts after split, got {len(split_src_name)}: {split_src_name}"
                            )
                        src_name = split_src_name[0]
                    expr = expr.replace(src_name, name_in_expr)
            if (
                type == BaseCType(tensorT)
                or type == OptionalCType(BaseCType(tensorT))
                or type == MutRefCType(OptionalCType(BaseCType(tensorT)))
                or (is_output and type == BaseCType(scalarT))
            ):
                # note(crcrpar): Here `expr` is generated from scratch, `arg.expr` is ignored.
                var = name
                name += "_"
                if var == "self" and inplace:
                    original_self_var = (
                        "original_self"
                        if not is_inplace_foreach
                        else "original_selfs[i]"
                    )
                    self_var = var if not is_inplace_foreach else var + "[i]"
                    stmts_prepend = f"if (!{original_self_var}.has_value()) {original_self_var} = {self_var}.clone()"
                    var = f"{original_self_var}.value()"
                    if is_output:
                        raise AssertionError(
                            "is_output should be False when var == 'self' and inplace"
                        )
                if inplace and is_output:
                    if name != "result_":
                        raise AssertionError(
                            f"expected name to be 'result_' for inplace output, got {name}"
                        )
                    var = (
                        "self[i]"
                        if is_inplace_foreach or is_foreacharg_list_type
                        else "self"
                    )
                    is_inplace_view = f"{var}.is_view()"
                    expr = f"SavedVariable({var}, {str(is_output).lower()}, {is_inplace_view})"
                else:
                    expr = f"SavedVariable({var}, {str(is_output).lower()})"
                    if foreacharg is not None and "original_selfs" not in expr:
                        # pyrefly: ignore [unbound-name]
                        expr = expr.replace(src_name, name_in_expr)
            elif (
                type == BaseCType(tensorListT)
                or type == ListCType(OptionalCType(BaseCType(tensorT)))
                or type == BaseCType(iTensorListRefT)
                or type == VectorCType(BaseCType(tensorT))
            ):
                # See Note [nuanced return type of out-of-place foreach functions]
                if type == VectorCType(BaseCType(tensorT)):
                    if not (is_foreach and is_output):
                        raise AssertionError(
                            f"VectorCType(BaseCType(tensorT)) requires is_foreach and is_output, "
                            f"got is_foreach={is_foreach}, is_output={is_output}"
                        )
                expr = f"make_saved_variable_list({name}, {str(is_foreach and is_output).lower()})"
                name += "_"
            elif type == BaseCType(intArrayRefT):
                expr = expr + ".vec()"
            elif type == BaseCType(symIntArrayRefT):
                expr = expr + ".vec()"
            elif type == BaseCType(stringT):
                expr = f"std::string({expr})"
            elif type == OptionalCType(BaseCType(stringT)):
                expr = f"{expr}.has_value() ? ::std::optional<std::string>(std::string({expr}.value())) : ::std::nullopt"
            elif type == ArrayRefCType(
                elem=BaseCType(type=BaseCppType(ns="at", name="Scalar"))
            ):
                expr = expr + ".vec()"

            guard = guard_for(arg)
            if guard is None:
                if stmts_prepend:
                    stmts.append(f"{stmts_prepend};")
                stmts.append(f"grad_fn->{name} = {expr};")
            else:
                stmts.append(f"if ({guard}) {{")
                if stmts_prepend:
                    stmts.append(f"  {stmts_prepend};")
                stmts.append(f"  grad_fn->{name} = {expr};")
                stmts.append("}")
        return stmts

    # Generates a Dispatcher::redispatch() call into the dispatcher. We do this mainly for performance reasons:
    #  - Pre-compute the full DispatchKeySet. This saves the dispatcher from having to read from TLS.
    #  - redispatch() avoids a redundant call to RecordFunction, which was already called right before
    #    we entered this autograd kernel.
    def emit_dispatch_call(
        f: NativeFunction, input_base: str, unpacked_args: Sequence[str]
    ) -> str:
        """Dispatch call via function in a namespace or method on Tensor."""
        # code-generated autograd kernels plumb and recompute dispatch keys directly through the kernel for performance.
        # Ops also always have a function variant of the redispatch API.
        # See Note [Plumbing Keys Through The Dispatcher] for details.
        dispatch_key_set = "ks & c10::after_autograd_keyset"
        call = CALL_REDISPATCH.substitute(
            api_name=cpp.name(
                f.func,
                faithful_name_for_out_overloads=True,
                symint_overload=f.func.has_symint(),
            ),
            unpacked_args=[dispatch_key_set] + list(unpacked_args),
        )
        return call

    def wrap_output(
        f: NativeFunction, unpacked_bindings: list[Binding], var: str
    ) -> str:
        call = ""
        rhs_value: str | None = None
        if not any(r.type.is_tensor_like() for r in f.func.returns):
            rhs_value = var
        else:
            rhs_value = f"std::move({var})"
        if rhs_value is None:
            raise AssertionError("rhs_value is None")
        call += ASSIGN_RETURN_VALUE.substitute(
            return_values=tie_return_values(f), rhs_value=rhs_value
        )
        return call

    def check_tensorimpl_and_storage(
        call: str, unpacked_bindings: list[Binding]
    ) -> str:
        # See NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
        stmts_before_call: list[str] = []
        stmts_after_call: list[str] = []

        if cpp.name(f.func) in DONT_ENFORCE_SAME_TENSOR_IMPL_OR_STORAGE:
            return call

        # Check properties of inputs (enforce (1))
        for unpacked_binding in unpacked_bindings:
            arg = unpacked_binding.name
            noref_cpp_type = unpacked_binding.nctype.type.remove_const_ref()
            if noref_cpp_type == BaseCType(tensorListT) or noref_cpp_type == BaseCType(
                iTensorListRefT
            ):
                stmts_before_call += [
                    SAVE_TENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                    SAVE_TENSORLIST_IMPL.substitute(tensorlist_name=arg),
                ]
                stmts_after_call += [
                    ENFORCE_SAME_TENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                    ENFORCE_SAME_TENSORLIST_IMPL.substitute(tensorlist_name=arg),
                ]
            elif noref_cpp_type == ListCType(OptionalCType(BaseCType(tensorT))):
                stmts_before_call += [
                    SAVE_OPTIONALTENSORLIST_STORAGE.substitute(tensorlist_name=arg),
                    SAVE_OPTIONALTENSORLIST_IMPL.substitute(tensorlist_name=arg),
                ]
                stmts_after_call += [
                    ENFORCE_SAME_OPTIONALTENSORLIST_STORAGE.substitute(
                        tensorlist_name=arg
                    ),
                    ENFORCE_SAME_OPTIONALTENSORLIST_IMPL.substitute(
                        tensorlist_name=arg
                    ),
                ]
            elif noref_cpp_type == BaseCType(tensorT):
                stmts_before_call += [
                    SAVE_TENSOR_STORAGE.substitute(tensor_name=arg),
                    SAVE_TENSOR_IMPL.substitute(tensor_name=arg),
                ]
                stmts_after_call += [
                    ENFORCE_SAME_TENSOR_STORAGE.substitute(
                        tensor_name=arg, out_tensor_name=arg
                    ),
                    ENFORCE_SAME_TENSOR_IMPL.substitute(tensor_name=arg),
                ]

        if not (
            (stmts_before_call and stmts_after_call)
            or (not stmts_before_call and not stmts_after_call)
        ):
            raise AssertionError(
                "stmts_before_call and stmts_after_call must be both empty or both non-empty"
            )

        # Check properties of outputs (enforce (2), (3))
        if f.func.kind() not in (SchemaKind.inplace, SchemaKind.out):
            base_name = f.func.name.name.base  # TODO: should be str(f.func.name.name)?
            aliased_arg_name = ALL_VIEW_FUNCTIONS.get(base_name, None)
            if aliased_arg_name is not None:
                aliased_arg_name = unpacked_name(aliased_arg_name)
            for i, (ret, ret_name) in enumerate(
                zip(f.func.returns, cpp.return_names(f))
            ):
                noref_cpp_type = cpp.return_type(ret, symint=True).remove_const_ref()
                if noref_cpp_type == BaseCType(tensorT):
                    if aliased_arg_name is not None:
                        if i != 0:
                            raise AssertionError(
                                f"Expect non-CompositeImplicitAutograd view function {base_name} "
                                f"to return single output, got index {i}"
                            )
                        stmts_after_call += [
                            ENFORCE_SAME_TENSOR_STORAGE.substitute(
                                tensor_name=aliased_arg_name, out_tensor_name=ret_name
                            )
                        ]
                    else:
                        if (
                            type_wrapper_name(f)
                            not in DONT_ENFORCE_STORAGE_IMPL_USE_COUNT
                        ):
                            stmts_after_call += [
                                ENFORCE_TENSOR_STORAGE_USE_COUNT_EQUALS_ONE.substitute(
                                    tensor_name=ret_name, fn_name=type_wrapper_name(f)
                                )
                            ]

                    if type_wrapper_name(f) not in DONT_ENFORCE_TENSOR_IMPL_USE_COUNT:
                        stmts_after_call += [
                            ENFORCE_TENSOR_IMPL_USE_COUNT.substitute(
                                tensor_name=ret_name, fn_name=type_wrapper_name(f)
                            )
                        ]

                # Currently we don't have any functions that return the following types, but
                # we should update the checks once we do
                elif noref_cpp_type == ListCType(OptionalCType(BaseCType(tensorT))):
                    raise AssertionError(
                        f"Please add use_count checks for {noref_cpp_type}"
                    )
                elif noref_cpp_type == BaseCType(tensorListT):
                    raise AssertionError(
                        f"Please add use_count checks for {noref_cpp_type}"
                    )

        if stmts_before_call and stmts_after_call:
            call = (
                RUN_ONLY_IN_DEBUG_MODE.substitute(statements=stmts_before_call)
                + call
                + RUN_ONLY_IN_DEBUG_MODE.substitute(statements=stmts_after_call)
            )
        return call

    def emit_call(
        f: NativeFunction, unpacked_bindings: list[Binding], try_jit_decomposition: bool
    ) -> str:
        # We only care about adding `at::AutoDispatchBelowAutograd` guard for non-variable dispatch
        # (which corresponds to 'use_derived' strategy). The purpose of this guard is to make sure
        # the baseType operations still dispatch to non-Variable type, even if the arguments passed
        # in are now Variables.
        # See NOTE [ Treating Variables as non-Variables in type dispatch ] for details.
        unpacked_args = [b.name for b in unpacked_bindings]
        base_type_call = emit_dispatch_call(f, "self_", unpacked_args)

        if get_view_info(f) is not None or modifies_arguments(f):
            guard = "at::AutoDispatchBelowAutograd guard;"
        else:
            guard = "at::AutoDispatchBelowADInplaceOrView guard;"

        any_has_forward_grad = (
            get_any_has_fw_grad_cond(derivative=None)
            if requires_derivative
            else "false"
        )
        return_types = ", ".join(
            [cpp.return_type(a, symint=True).cpp_type() for a in f.func.returns]
        )
        if len(f.func.returns) > 1:
            return_types = f"std::tuple<{return_types}>"

        arg_names = [
            a.name
            for a in cpp.arguments(
                f.func.arguments,
                faithful=True,
                symint=True,
                method=False,
                cpp_no_default_args=set(),
            )
        ]

        if not modifies_arguments(f) and not returns_void:
            if try_jit_decomposition:
                call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES_JVP_DECOMP.substitute(
                    base_type_call=base_type_call,
                    tmp_var=TMP_VAR,
                    guard=guard,
                    any_has_forward_grad=any_has_forward_grad,
                    op_name=cpp.name(f.func),
                    op_overload=f.func.name.overload_name,
                    return_types=return_types,
                    arg_names=arg_names,
                )
            else:
                call = DISPATCH_TO_NON_VAR_TYPE_WITH_TMP_RETURN_VALUES.substitute(
                    base_type_call=base_type_call,
                    tmp_var=TMP_VAR,
                    guard=guard,
                )

            call += wrap_output(f, unpacked_bindings, TMP_VAR)
        else:
            if try_jit_decomposition:
                raise AssertionError(
                    "try_jit_decomposition should be False for functions with no return values or that modify arguments"
                )
            call = DISPATCH_TO_NON_VAR_TYPE_WITHOUT_RETURN_VALUES.substitute(
                base_type_call=base_type_call, guard=guard
            )
        call = check_tensorimpl_and_storage(call, unpacked_bindings)
        return call

    def emit_history() -> str:
        fn = "rebase" if modifies_arguments(f) and view_info is None else "set"
        output_names = [r.name for r in differentiable_outputs]
        # TODO: flatten allocates a std::vector, which could be expensive
        outs = CodeTemplate("flatten_tensor_args( ${outs} )").substitute(
            outs=output_names if not is_inplace_foreach else "self"
        )
        if not is_inplace_foreach:
            if fn == "rebase":
                return REBASE_HISTORY.substitute(differentiable_outputs=outs)
            return SET_HISTORY.substitute(fn=fn, differentiable_outputs=outs)
        else:
            decl = ""
            capture = ""
            if fn == "rebase":
                decl = "c10::SmallVector<c10::intrusive_ptr<Node>, 8> _rebased_fns(grad_fns.size());\n"
                capture = "_rebased_fns[i] = "
            return decl + LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(
                preamble=(
                    f"auto differentiable_outputs = {outs};\n"
                    f"TORCH_INTERNAL_ASSERT(differentiable_outputs.size() == grad_fns.size());"
                ),
                statements=f"{capture}{fn}_history(differentiable_outputs[i], grad_fns[i]);",
            )

    def emit_fire_creation_hooks() -> str:
        rebase = modifies_arguments(f) and view_info is None
        if not is_inplace_foreach:
            node = "_rebased_fn" if rebase else "grad_fn"
            return FIRE_NODE_CREATION_HOOKS.substitute(node=node)
        else:
            node = "_rebased_fns[i]" if rebase else "grad_fns[i]"
            return (
                "for (const auto& i : c10::irange(grad_fns.size())) {\n"
                f"    if ({node}) {{\n"
                f"        fire_node_creation_hooks({node});\n"
                "    }\n"
                "}\n"
            )

    def emit_save_outputs() -> str:
        if is_out_fn:
            # out functions don't currently support differentiation
            return ""
        if info is not None and info.has_derivatives:
            stmts = save_variables(info.all_saved_outputs, True)
            if len(stmts) == 0:
                return ""
            if not is_inplace_foreach:
                return CONDITIONAL.substitute(cond="grad_fn", statements=stmts)
            else:
                return LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(
                    preamble="", statements=stmts
                )
        return ""

    def emit_any_requires_grad() -> list[str]:
        extra_condition = ""
        if info and info.output_differentiability_conditions:
            if len(info.output_differentiability_conditions) != 1:
                raise AssertionError(
                    f"expected 1 output_differentiability_condition, got {len(info.output_differentiability_conditions)}"
                )
            extra_condition = f"_any_requires_grad &= ({info.output_differentiability_conditions[0]});"
        names_of_args_with_derivatives = [arg.name for arg in args_with_derivatives]
        if is_inplace_foreach and info is not None:
            for i, arg in enumerate(names_of_args_with_derivatives):
                for f_arg, r_arg in inplace_foreacharg2refarg.items():
                    if arg == r_arg.name:
                        names_of_args_with_derivatives[i] = f_arg.name
        return [
            SETUP_ANY_REQUIRES_GRAD.substitute(
                args_with_derivatives=names_of_args_with_derivatives,
                extra_differentiability_conditions=extra_condition,
            )
        ]

    def get_any_has_forward_grad_name(var_names: tuple[str, ...]) -> str:
        if len(var_names) == 1:
            return f"_any_has_forward_grad_{var_names[0]}"
        else:
            return f"_any_has_forward_grad_{'_'.join(var_names)}"

    def emit_any_has_forward_grad() -> list[str]:
        content: list[str] = []
        if not is_foreach:
            for derivative in fw_derivatives:
                requires_fw_grad = get_any_has_fw_grad_cond(derivative=derivative)
                if info and info.output_differentiability_conditions:
                    if len(info.output_differentiability_conditions) != 1:
                        raise AssertionError(
                            f"expected 1 output_differentiability_condition, got {len(info.output_differentiability_conditions)}"
                        )
                    requires_fw_grad = f"({info.output_differentiability_conditions[0]}) && {requires_fw_grad}"
                content.append(
                    f"[[maybe_unused]] auto {get_any_has_forward_grad_name(derivative.var_names)} = {requires_fw_grad};"
                )
        else:
            for derivative in fw_derivatives:
                bool_vector_name = get_any_has_forward_grad_name(derivative.var_names)
                cur_derivative_conditions = []
                for inp in differentiable_inputs:
                    if derivative.required_inputs_fw_grad is None:
                        continue
                    if inp.name not in derivative.required_inputs_fw_grad:
                        continue
                    inp_name = (
                        inp.name
                        if not inplace
                        else refargname2inplace_foreacharg[inp.name].name
                    )
                    inp_type = (
                        inp.type
                        if not inplace
                        else refargname2inplace_foreacharg[inp.name].type
                    )
                    is_list_type = is_tensor_list_type(inp_type)
                    if is_list_type:
                        if inp_name != "self":
                            content.append(
                                FW_DERIVATIVE_SIZE_CHECK_TEMPLATE.substitute(
                                    inp_name=inp_name
                                )
                            )
                        cur_derivative_conditions.append(
                            FW_DERIVATIVE_CHECK_TEMPLATE.substitute(
                                req_inp=inp_name + "[i]"
                            )
                        )
                    else:
                        cur_derivative_conditions.append(
                            FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp_name)
                        )

                content.append(f"std::vector<bool> {bool_vector_name}(self.size());")
                content.append("for (const auto& i : c10::irange(self.size())) {")
                content.append(
                    f"  {bool_vector_name}[i] = {' || '.join(cur_derivative_conditions)};"
                )
                content.append("}")
        return content

    def emit_check_inplace() -> list[str]:
        if not inplace:
            return []
        return [
            f"check_inplace({arg.name}, _any_requires_grad);"
            for arg in differentiable_outputs
        ]

    def emit_fw_derivatives() -> list[str]:
        content: list[str] = []
        fw_grad_setters: list[str] = []
        for derivative in fw_derivatives:
            res = derivative.var_names
            if f.func.name.name.inplace:
                if len(res) != 1:
                    raise AssertionError(
                        f"Expected number of outputs to be 1 if function is inplace, got {len(res)}"
                    )
                # TODO update this when inplace namings are unified
                res = ("self",)

            if derivative.required_inputs_fw_grad is None:
                raise AssertionError("derivative.required_inputs_fw_grad is None")

            unpacked_arguments = ""
            for inp in differentiable_inputs:
                inp_name = inp.name
                is_input_tensorlist = is_foreach and is_tensor_list_type(
                    inp.type
                    if not inplace
                    else refargname2inplace_foreacharg[inp.name].type
                )
                input_suffix = "[i]" if is_input_tensorlist else ""
                if is_inplace_foreach:
                    if inp.name in refargname2inplace_foreacharg:
                        inp_name = refargname2inplace_foreacharg[inp.name].name
                zeros_fn = (
                    "zeros_symint"
                    if inplace and inp.name == "self"
                    else "_efficientzerotensor_symint"
                )
                if inp.name in derivative.required_inputs_fw_grad:
                    unpacked_arguments += (
                        FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE.substitute(
                            inp_name=inp.name,
                            inp=inp_name + input_suffix,
                            zeros_fn=zeros_fn,
                        )
                    )
                    if zeros_fn == "_efficientzerotensor_symint":
                        unpacked_arguments += (
                            FW_DERIVATIVE_UPDATE_WRAPPED_NUM_TEMPLATE.substitute(
                                inp_name=inp.name
                            )
                        )

                if inp.name in (derivative.required_inputs_primal or []):
                    unpacked_arguments += (
                        FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE.substitute(
                            inp_name=inp.name,
                            inp=inp_name + input_suffix,
                        )
                    )
            if derivative.required_original_self_value:
                input_suffix = "s[i]" if is_inplace_foreach else ""
                unpacked_arguments += FW_DERIVATIVE_DEFINED_GRAD_TEMPLATE.substitute(
                    inp_name="original_self",
                    inp="original_self" + input_suffix,
                    # pyrefly: ignore [unbound-name]
                    zeros_fn=zeros_fn,
                )
                unpacked_arguments += FW_DERIVATIVE_DEFINED_PRIMAL_TEMPLATE.substitute(
                    inp_name="original_self",
                    inp="original_self" + input_suffix,
                )
            elif inplace and derivative.is_reusing_outplace_formula:
                # The gradient wasn't already cloned, do it if grad mode is enabled
                unpacked_arguments += (
                    "self_t = GradMode::is_enabled() ? self_t.clone() : self_t;"
                )

            if inplace:
                is_inplace_str = "true"
            else:
                is_inplace_str = "false"

            requires_fw_grad = get_any_has_forward_grad_name(derivative.var_names)

            if all(
                (isinstance(var_type, BaseType) and var_type.is_tensor_like())
                for var_type in derivative.var_types
            ):
                # Is there a way to get from BaseType to BaseCType
                if len(derivative.var_types) == 1:
                    opt_res_grad_type = OptionalCType(BaseCType(tensorT)).cpp_type()
                    if not is_foreach:
                        fw_grad_setters.append(
                            FW_DERIVATIVE_SETTER_TENSOR.substitute(
                                out_arg=res[0], is_inplace=is_inplace_str
                            )
                        )
                    else:
                        expected_res = "result" if not inplace else "self"
                        if res[0] != expected_res:
                            raise AssertionError(
                                f"res[0] is {res[0]}, expected {expected_res}"
                            )
                        fw_grad_setters.append(
                            FW_DERIVATIVE_SETTER_TENSOR_FOREACH.substitute(
                                out_arg=res[0], is_inplace=is_inplace_str
                            )
                        )
                    requires_fw_grad += f" && ({derivative.var_names[0]}.defined())"
                else:
                    tuple_type = TupleCType(
                        [BaseCType(tensorT)] * len(derivative.var_types)
                    )
                    opt_res_grad_type = OptionalCType(tuple_type).cpp_type()
                    for idx, single_res in enumerate(res):
                        fw_grad_setters.append(
                            FW_DERIVATIVE_SETTER_MULTI_OUTPUT.substitute(
                                idx=idx, all_res="_".join(res), out_arg=single_res
                            )
                        )
            elif (
                isinstance(derivative.var_types[0], ListType)
                and derivative.var_types[0].is_tensor_like()
            ):
                if len(derivative.var_types) != 1:
                    raise AssertionError(
                        f"Expected number of outputs to be 1 if function returns ListType, got {len(derivative.var_types)}"
                    )
                if not is_foreach:
                    opt_res_grad_type = OptionalCType(
                        VectorCType(BaseCType(tensorT))
                    ).cpp_type()
                    fw_grad_setters.append(
                        FW_DERIVATIVE_SETTER_TENSOR_LIST.substitute(
                            out_arg=res[0], is_inplace=is_inplace_str
                        )
                    )
                else:
                    # TODO(crcrpar): Should this (= the foreach specific logic) be refactored somehow?
                    # Only out-place foreach functions that have entries in `tools/autograd/derivatives.yaml`
                    # can reach here.
                    opt_res_grad_type = OptionalCType(BaseCType(tensorT)).cpp_type()
                    fw_grad_setters.append(
                        FW_DERIVATIVE_SETTER_TENSOR_FOREACH.substitute(
                            out_arg=res[0], is_inplace=is_inplace_str
                        )
                    )
            else:
                raise RuntimeError("Unsupported output type for forward derivative")

            if not is_foreach:
                fw_grad_opt_definition = f"{opt_res_grad_type} {'_'.join(res)}_new_fw_grad_opt = ::std::nullopt;"
                # View ops create fw_grad that already is a view of the base's fw_grad so just use that
                content.append(
                    FW_DERIVATIVE_TEMPLATE.substitute(
                        fw_grad_opt_definition=fw_grad_opt_definition,
                        requires_fw_grad=requires_fw_grad,
                        formula=derivative.formula,
                        out_arg="_".join(res),
                        unpacked_arguments=unpacked_arguments,
                    )
                )
            else:
                # note(crcrpar): Assuming `self` is TensorList.
                fw_grad_opt_definition = (
                    f"std::vector<{opt_res_grad_type}> {'_'.join(res)}_new_fw_grad_opts"
                    "(self.size(), ::std::nullopt);"
                )
                foreach_forward_grad_formula = derivative.formula
                _foreach_arg: Argument | DifferentiableInput
                if inplace:
                    for _foreach_arg, _ref_arg in inplace_foreacharg2refarg.items():
                        # note(crcrpar): Massage only Scalar and ArrayRef<Scalar> here.
                        if not (
                            is_tensor_type(_foreach_arg.type)
                            or is_tensor_list_type(_foreach_arg.type)
                        ):
                            pattern = _foreach_arg.name
                            if isinstance(_foreach_arg.type, ListType):
                                pattern += "[i]"
                            foreach_forward_grad_formula = (
                                foreach_forward_grad_formula.replace(
                                    _ref_arg.name, pattern
                                )
                            )
                else:
                    if (
                        "result" in foreach_forward_grad_formula
                        and "result[i]" not in foreach_forward_grad_formula
                    ):
                        foreach_forward_grad_formula = (
                            foreach_forward_grad_formula.replace("result", "result[i]")
                        )

                content.append(
                    FW_DERIVATIVE_FOREACH_TEMPLATE.substitute(
                        fw_grad_opt_definition=fw_grad_opt_definition,
                        vector_of_optional_tensor=f"{'_'.join(res)}_new_fw_grad_opts",
                        any_has_forward_grad_for_current_index=" || ".join(
                            get_any_has_forward_grad_name(derivative.var_names) + "[i]"
                            for derivative in fw_derivatives
                        ),
                        formula=foreach_forward_grad_formula,
                        unpacked_arguments=unpacked_arguments,
                    )
                )

        # Set all the grads at the end to avoid: https://github.com/pytorch/pytorch/issues/67367
        content.append("\n".join(fw_grad_setters))
        return content

    def get_any_has_fw_grad_cond(derivative: ForwardDerivative | None) -> str:
        #
        # Produces a condition string (e.g, "isFwGradDefined(grad_output) || isFwGradDefined(output)")
        #
        if derivative is None:
            # (1) If a derivative is NOT provided, cond will check fw_grad of ALL differentiable inputs
            # - Used in the out_fn case when we want to forbid fw derivatives
            # - Used in the case where the fw_derivative is not defined, but we want
            #   To check if there is a decomposition registered for jvp
            to_check: list[str] = []
            for inp in list(
                mapMaybe(
                    gen_differentiable_input,
                    f.func.arguments.non_out + list(f.func.arguments.out),  # type: ignore[operator]
                )
            ):
                if is_tensor_type(inp.type):
                    to_check.append(
                        FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp.name)
                    )
                elif is_tensor_list_type(inp.type):
                    to_check.append(
                        FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE.substitute(
                            req_inp=inp.name
                        )
                    )
                else:
                    raise RuntimeError(
                        f'Unsupported input type for "{name}" when forbidding forward AD usage.'
                    )
            return f"({' || '.join(to_check)})"
        else:
            # (2) If derivative is provided, use that information to determine which inputs
            #     to check fw_grad for
            if derivative.required_inputs_fw_grad is None:
                raise AssertionError("derivative.required_inputs_fw_grad is None")

            if len(derivative.required_inputs_fw_grad) == 0:
                # Handle functions like stack
                # For these, we don't unpack anything and always call the user function
                if not (
                    len(differentiable_inputs) == 1
                    and is_tensor_list_type(differentiable_inputs[0].type)
                ):
                    raise RuntimeError(
                        f'No differentiable input to "{name}" is a differentiable Tensor (as the provided '
                        "forward AD formula does not use any input tangent) even though a forward gradient "
                        "formula has been defined for it. This case should only happen for function that "
                        "take a single TensorList as input. All other cases are not supported right now."
                    )
                any_has_fw_grad = "true"
            else:
                any_has_fw_grad = " || ".join(
                    [
                        (
                            FW_DERIVATIVE_TENSORLIST_CHECK_TEMPLATE
                            if is_tensor_list_type(inp.type)
                            else FW_DERIVATIVE_CHECK_TEMPLATE
                        ).substitute(req_inp=inp.name)
                        for inp in differentiable_inputs
                        if inp.name in derivative.required_inputs_fw_grad
                    ]
                )
                any_has_fw_grad = f"({any_has_fw_grad})"

            return any_has_fw_grad

    def emit_forbid_fw_derivatives(is_out_fn: bool = False) -> str:
        if is_out_fn:
            msg = "because it is an out= function"
        else:
            msg = (
                "because it has not been implemented yet.\\nPlease file an issue "
                "to PyTorch at https://github.com/pytorch/pytorch/issues/new?template=feature-request.yml "
                "so that we can prioritize its implementation."
            )
        cond = get_any_has_fw_grad_cond(derivative=None)
        return (
            FW_DERIVATIVE_FORBID_TEMPLATE.substitute(cond=cond, name=name, msg=msg)
            if cond != ""
            else ""
        )

    body: list[str] = []
    unpack_args_stats, unpacked_bindings = unpack_args(f)

    body.extend(unpack_args_stats)
    if requires_derivative:
        body.extend(emit_any_requires_grad())
        body.extend(emit_any_has_forward_grad())
        body.extend(emit_check_inplace())
        body.extend(emit_original_self_definition())
        body.extend(setup_derivative(differentiable_inputs))

    body.append(emit_call(f, unpacked_bindings, try_jit_decomposition))
    if requires_derivative:
        # set_flags has to appear after version_counter, because rebase_history
        # requires that the counter is incremented before it is called
        body.append(emit_history())
        body.extend(emit_check_if_in_complex_autograd_allowlist())

    if is_out_fn:
        body.append(emit_forbid_fw_derivatives(is_out_fn=True))
    else:
        if requires_derivative and not try_jit_decomposition:
            if len(fw_derivatives) > 0:
                body.extend(emit_fw_derivatives())
            else:
                body.append(emit_forbid_fw_derivatives())

    if requires_derivative:
        # Save only after the forward AD has been set up
        body.append(emit_save_outputs())
        # Fire node creation hooks only once the node is fully populated,
        # including saved outputs above.
        body.append(emit_fire_creation_hooks())

    if str(f.func.name.name) in RESET_GRAD_ACCUMULATOR:
        # `inplace` implies that there is exactly one output named `self`,
        # so we can keep the generated code easy. If you need to
        # `reset_grad_accumulator` in an operator that's not `inplace`, you can
        # remove this check but the code generation will get more elaborate
        if not inplace:
            raise AssertionError(
                f"expected inplace=True for {f.func.name.name} which is in RESET_GRAD_ACCUMULATOR"
            )
        body.append("reset_grad_accumulator(self);")
    if not returns_void:
        body.append(f"return {get_return_value(f)};")
    return body
<|endoftext|># Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn. torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._sparse
# or torch._C._special objects.
#

# Code tries to stick to the following rules:
#
# - templates should be colocated with the functions that use them.
#   no templates are currently shared between functions, but if that
#   happens, maybe put the template with the first one
#
# - don't use environment dictionaries when calling template.substitute().
#   pass named arguments directly for everything, otherwise it's much too
#   hard to track what's actually being used and by who
#
# - colocate any new hacks/adjustments with existing ones of the same kind.
#   ideally in a data structure rather than code if possible. See e.g.
#   SCHEMA_DEFAULT_CONVERSION_HACKS, etc.
#
# - similarly, conversions from one format to another should ideally happen
#   all at once in a single place.
#
# - no nontrivial nested functions. couple-liners are ok but please no more.
#   especially avoid functions that read/write outer variables defined far away.
#
# - raise RuntimeError instead of asserting, and put as much
#   information as is available into the message. I.e. no need to
#   plumb in new params whose only purpose is to fill out an error
#   message, but use what's there
#

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import yaml

from torchgen.api import cpp
from torchgen.api.python import (
    arg_parser_output_exprs,
    cpp_dispatch_exprs,
    cpp_dispatch_target,
    dispatch_lambda_args,
    dispatch_lambda_exprs,
    dispatch_lambda_return_str,
    has_tensor_options,
    PythonSignature,
    PythonSignatureDeprecated,
    PythonSignatureGroup,
    PythonSignatureNativeFunctionPair,
    signature,
    signature_from_schema,
    structseq_fieldnames,
)
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
    Argument,
    BaseOperatorName,
    FunctionSchema,
    NativeFunction,
    SchemaKind,
    Type,
    Variant,
)
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader

from .gen_inplace_or_view_type import is_tensor_list_type
from .gen_trace_type import should_trace


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


#
# declarations blocklist
# We skip codegen for these functions, for various reasons.
# Future PRs will categorize this list and eliminate or hoist
# them out of eager-only codegen.
# See https://github.com/pytorch/pytorch/issues/30788
#

# These functions require manual Python bindings or are not exposed to Python
_SKIP_PYTHON_BINDINGS = [
    "alias",
    "contiguous",
    "dim",
    "get_device",
    "is_contiguous",
    "is_cuda",
    "is_sparse",
    "is_sparse_csr",
    "numel",
    "size",
    "storage_offset",
    "stride",
    "sym_is_contiguous",
    "sym_size",
    "sym_stride",
    "sym_storage_offset",
    "sym_numel",
    ".*_backward",
    ".*_backward_(out|input|weight|bias)",
    ".*_forward",
    ".*_forward_out",
    ".*_jvp",
    "_unsafe_view",
    "tensor",
    "_?sparse_(coo|compressed|csr|csc|bsr|bsc)_tensor.*",
    "_range.*",
    "_sparse_add_out",
    "_sparse_div.*",
    "_sparse_mul.*",
    "_sparse_sub.*",
    "_sparse_dense_add_out",
    "index",
    "index_out",
    "unique_dim_consecutive",
    "_cumsum.*",
    "_cumprod.*",
    "_sum.*",
    "_prod.*",
    "_th_.*",
    "_thnn_.*",
    "range.*",
    "_solve.*",
    "_inverse.*",
    "_cholesky.*",
    "_triangular_solve.*",
    "_qr.*",
    "_svd.*",
    "slice",
    "item",
    "_local_scalar_dense",
    "to",
    "_to_copy",
    "_to_copy_out",
    "_reshape_copy",
    "_reshape_copy_out",
    "copy_sparse_to_sparse_",
    "copy_",
    "_foreach_copy",
    "numpy_T",
    "matrix_H",
    "mT",
    "mH",  # these need to be an attributes in Python, not functions
    "nonzero(_(out|numpy))?",
    "set_data",
    ".*_overrideable",  # overridable functions for backend extension
    "data",
    "is_leaf",
    "output_nr",
    "_version",
    "requires_grad_",
    "retains_grad",
    "set_",
    "_fw_primal",
    "fake_quantize_per_tensor_affine_cachemask",
    "fake_quantize_per_channel_affine_cachemask",
    "_new_zeros_with_same_feature_meta",
    "_has_same_storage_numel",  # used for forward AD internals
    "_reshape_alias",
    "replace_",  # only used by the functionalization pass, doesn't need to be exposed to python
    "copy",  # only used by the functionalization pass
    "fill.Tensor",  # only used by the functionalization pass
    "fill.Scalar",  # only used by the functionalization pass
    "lift.*",
    "normal_functional",  # only used by the functionalization pass
    "nbytes",
    "itemsize",
    "_batch_norm_with_update",
    "_batch_norm_with_update_out",
    "_batch_norm_no_update",
]

SKIP_PYTHON_BINDINGS = [
    re.compile(rf"^{pattern}$") for pattern in _SKIP_PYTHON_BINDINGS
]

# These function signatures are not exposed to Python. Note that this signature
# list does not support regex.
SKIP_PYTHON_BINDINGS_SIGNATURES = [
    "add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
    "sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
    "mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
    "div.Scalar(Tensor self, Scalar other) -> Tensor",
    "div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
]


@with_native_function
def should_generate_py_binding(f: NativeFunction) -> bool:
    # NativeFunctions that are entirely code-generated should not get python bindings
    # because these codegen implementations are often inefficient. A handful of
    # view_copy style ops were exposed accidentally when they were handwritten and now
    # that we are moving them to codegen for bc reasons we need to keep them exposed in
    # python.
    if "generated" in f.tags and "view_copy" not in f.tags:
        return False

    name = cpp.name(f.func)
    for skip_regex in SKIP_PYTHON_BINDINGS:
        if skip_regex.match(name):
            return False

    signature = str(f.func)
    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == signature:
            return False
    return True


def get_pycname(name: BaseOperatorName) -> str:
    return f"THPVariable_{name}"


def is_noarg(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> bool:
    return len(overloads) == 1 and overloads[0].signature.arguments_count() == 0


def is_py_variable_method(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.method in f.variants


def is_py_torch_function(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.function in f.variants


def is_py_nn_function(f: NativeFunction) -> bool:
    return f.python_module == "nn"


def is_py_fft_function(f: NativeFunction) -> bool:
    return f.python_module == "fft"


def is_py_linalg_function(f: NativeFunction) -> bool:
    return f.python_module == "linalg"


def is_py_nested_function(f: NativeFunction) -> bool:
    return f.python_module == "nested"


def is_py_sparse_function(f: NativeFunction) -> bool:
    return f.python_module == "sparse"


def is_py_special_function(f: NativeFunction) -> bool:
    return f.python_module == "special"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                            Main Function
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def gen(
    out: str,
    native_yaml_path: str,
    tags_yaml_path: str,
    deprecated_yaml_path: str,
    template_path: str,
    *,
    symint: bool = True,
) -> None:
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    native_functions = list(filter(should_generate_py_binding, native_functions))

    methods = load_signatures(native_functions, deprecated_yaml_path, method=True)
    create_python_bindings(
        fm,
        methods,
        is_py_variable_method,
        None,
        "python_variable_methods.cpp",
        method=True,
        symint=symint,
    )

    # NOTE: num_shards here must be synced with gatherTorchFunctions in
    #       torch/csrc/autograd/python_torch_functions_manual.cpp
    functions = load_signatures(native_functions, deprecated_yaml_path, method=False)
    create_python_bindings_sharded(
        fm,
        functions,
        is_py_torch_function,
        "torch",
        "python_torch_functions.cpp",
        method=False,
        num_shards=3,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_nn_function,
        "torch.nn",
        "python_nn_functions.cpp",
        method=False,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_fft_function,
        "torch.fft",
        "python_fft_functions.cpp",
        method=False,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_linalg_function,
        "torch.linalg",
        "python_linalg_functions.cpp",
        method=False,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_nested_function,
        "torch.nested",
        "python_nested_functions.cpp",
        method=False,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_sparse_function,
        "torch.sparse",
        "python_sparse_functions.cpp",
        method=False,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_special_function,
        "torch.special",
        "python_special_functions.cpp",
        method=False,
        symint=symint,
    )

    # Currently, we only use `functions` to generate `return_types` bindings.
    # All methods which return structseq have function variant at this point.
    # If any method only operator with structseq is added in the future,
    # we will have to address that.
    create_python_return_type_bindings(
        fm, functions, lambda fn: True, "python_return_types.cpp"
    )
    create_python_return_type_bindings_header(
        fm, functions, lambda fn: True, "python_return_types.h"
    )

    valid_tags = parse_tags_yaml(tags_yaml_path)

    def gen_tags_enum() -> dict[str, str]:
        return {
            "enum_of_valid_tags": (
                "".join(
                    [f'\n.value("{tag}", at::Tag::{tag})' for tag in sorted(valid_tags)]
                )
            )
        }

    fm.write("python_enum_tag.cpp", gen_tags_enum)


def group_filter_overloads(
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
) -> dict[BaseOperatorName, list[PythonSignatureNativeFunctionPair]]:
    grouped: dict[BaseOperatorName, list[PythonSignatureNativeFunctionPair]] = (
        defaultdict(list)
    )
    for pair in pairs:
        if pred(pair.function):
            grouped[pair.function.func.name.name].append(pair)
    return grouped


def create_python_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: str | None,
    filename: str,
    *,
    method: bool,
    symint: bool = True,
) -> None:
    """Generates Python bindings to ATen functions"""
    py_methods: list[str] = []
    ops_headers: list[str] = []
    py_method_defs: list[str] = []
    py_forwards: list[str] = []

    grouped = group_filter_overloads(pairs, pred)

    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        py_methods.append(
            method_impl(name, module, overloads, method=method, symint=symint)
        )
        py_method_defs.append(method_def(name, module, overloads, method=method))
        py_forwards.extend(forward_decls(name, overloads, method=method))
        ops_headers.append(f"#include <ATen/ops/{name.base}.h>")

    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "ops_headers": ops_headers,
            "py_forwards": py_forwards,
            "py_methods": py_methods,
            "py_method_defs": py_method_defs,
        },
    )


def create_python_return_type_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    filename: str,
) -> None:
    """
    Generate function to initialize and return named tuple for native functions
    which returns named tuple and registration invocations in `python_return_types.cpp`.
    """
    py_return_types_definition: list[str] = []
    py_return_types_registrations: list[str] = []

    grouped = group_filter_overloads(pairs, pred)

    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        definitions, registrations = generate_return_type_definition_and_registrations(
            overloads
        )
        py_return_types_definition.append(
            "" if not definitions else "\n".join(definitions)
        )
        py_return_types_registrations.append(
            "" if not registrations else "\n".join(registrations)
        )

    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "py_return_types": py_return_types_definition,
            "py_return_types_registrations": py_return_types_registrations,
        },
    )


def create_python_return_type_bindings_header(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    filename: str,
) -> None:
    """
    Generate function to initialize and return named tuple for native functions
    which returns named tuple and relevant entry for the map in `python_return_types.cpp`.
    """
    py_return_types_declarations: list[str] = []

    grouped = group_filter_overloads(pairs, pred)

    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        declarations = generate_return_type_declarations(overloads)
        py_return_types_declarations.append(
            "" if not declarations else "\n".join(declarations)
        )

    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "py_return_types_declarations": py_return_types_declarations,
        },
    )


def create_python_bindings_sharded(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: str | None,
    filename: str,
    *,
    method: bool,
    num_shards: int,
    symint: bool = True,
) -> None:
    """Generates Python bindings to ATen functions"""
    grouped = group_filter_overloads(pairs, pred)

    def key_func(
        kv: tuple[BaseOperatorName, list[PythonSignatureNativeFunctionPair]],
    ) -> str:
        return kv[0].base

    def env_func(
        kv: tuple[BaseOperatorName, list[PythonSignatureNativeFunctionPair]],
    ) -> dict[str, list[str]]:
        name, fn_pairs = kv
        return {
            "ops_headers": [f"#include <ATen/ops/{name.base}.h>"],
            "py_forwards": list(forward_decls(name, fn_pairs, method=method)),
            "py_methods": [
                method_impl(name, module, fn_pairs, method=method, symint=symint)
            ],
            "py_method_defs": [method_def(name, module, fn_pairs, method=method)],
        }

    fm.write_sharded(
        filename,
        grouped.items(),
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
        },
        key_fn=key_func,
        env_callable=env_func,
        num_shards=num_shards,
        sharded_keys={"ops_headers", "py_forwards", "py_methods", "py_method_defs"},
    )


def load_signatures(
    native_functions: list[NativeFunction],
    deprecated_yaml_path: str,
    *,
    method: bool,
    skip_deprecated: bool = False,
    pyi: bool = False,
) -> Sequence[PythonSignatureNativeFunctionPair]:
    @with_native_function
    def gen_signature_pairs(f: NativeFunction) -> PythonSignatureNativeFunctionPair:
        return PythonSignatureNativeFunctionPair(
            signature=signature(f, method=method, pyi=pyi),
            function=f,
        )

    pairs = list(map(gen_signature_pairs, native_functions))
    deprecated = load_deprecated_signatures(
        pairs, deprecated_yaml_path, method=method, pyi=pyi
    )
    return pairs if skip_deprecated else pairs + deprecated


def load_deprecated_signatures(
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    deprecated_yaml_path: str,
    *,
    method: bool,
    pyi: bool,
) -> list[PythonSignatureNativeFunctionPair]:
    # The deprecated.yaml doesn't have complete type information, we need
    # find and leverage the original ATen signature (to which it delegates
    # the call) to generate the full python signature.
    # We join the deprecated and the original signatures using type-only form.

    # group the original ATen signatures by name
    grouped: dict[str, list[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.signature.name].append(pair)

    # find matching original signatures for each deprecated signature
    results: list[PythonSignatureNativeFunctionPair] = []

    with open(deprecated_yaml_path) as f:
        deprecated_defs = yaml.load(f, Loader=YamlLoader)

    for deprecated in deprecated_defs:
        schema = FunctionSchema.parse(deprecated["name"])
        aten_name, call_args = split_name_params(deprecated["aten"])
        is_out = aten_name.endswith("_out")
        if is_out:
            aten_name = aten_name.replace("_out", "")

        # HACK: these are fixed constants used to pass the aten function.
        # The type must be known ahead of time
        known_constants = {
            "1": Type.parse("Scalar"),
        }
        schema_args_by_name = {a.name: a for a in schema.arguments.flat_all}
        for name in call_args:
            if name not in schema_args_by_name and name not in known_constants:
                raise AssertionError(
                    f"deprecation definition: Unrecognized value {name}"
                )

        # Map deprecated signature arguments to their aten signature and test
        # if the types and alias annotation match.
        def is_schema_compatible(
            aten_schema: FunctionSchema,
        ) -> bool:
            arguments: Iterable[Argument]
            if is_out:
                arguments = itertools.chain(
                    aten_schema.arguments.out, aten_schema.arguments.flat_non_out
                )
            else:
                arguments = aten_schema.arguments.flat_all

            for i, arg in enumerate(arguments):
                if i < len(call_args):
                    arg_name = call_args[i]
                    if arg_name in known_constants:
                        schema_type = known_constants[arg_name]
                        schema_annotation = None
                    else:
                        schema_arg = schema_args_by_name[arg_name]
                        schema_type = schema_arg.type
                        schema_annotation = schema_arg.annotation

                    if schema_type != arg.type or schema_annotation != arg.annotation:
                        return False
                else:
                    if arg.default is None:
                        return False

            return len(schema.returns) == len(aten_schema.returns) and all(
                a == b for a, b in zip(schema.returns, aten_schema.returns)
            )

        any_schema_found = False
        for pair in grouped[aten_name]:
            if not is_schema_compatible(pair.function.func):
                continue
            any_schema_found = True

            python_sig = signature_from_schema(
                schema,
                category_override=pair.function.category_override,
                method=method,
                pyi=pyi,
            )

            results.append(
                PythonSignatureNativeFunctionPair(
                    signature=PythonSignatureDeprecated(
                        name=python_sig.name,
                        input_args=python_sig.input_args,
                        input_kwargs=python_sig.input_kwargs,
                        output_args=python_sig.output_args,
                        tensor_options_args=python_sig.tensor_options_args,
                        method=python_sig.method,
                        deprecated_schema=schema,
                        deprecated_args_exprs=tuple(call_args),
                        returns=python_sig.returns,
                    ),
                    function=pair.function,
                )
            )
        if not any_schema_found:
            raise AssertionError(
                f"No native function with name {aten_name} matched signature:\n  {str(schema)}"
            )

    return results


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         Named Tuple Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@with_native_function
def gen_structseq_typename_key(f: NativeFunction) -> str:
    name = cpp.name(f.func)
    fieldnames = structseq_fieldnames(f.func.returns)
    return "_".join([name] + fieldnames)


def emit_structseq_call(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> tuple[list[str], dict[str, str]]:
    """
    Generate block of named tuple type def inits, and add typeref snippets
    to declarations that use them
    """
    typenames: dict[
        str, str
    ] = {}  # map from unique name + field name lists to typedef name
    typedefs: list[str] = []  # typedef declarations and init code

    for overload in overloads:
        fieldnames = structseq_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue

        name = cpp.name(overload.function.func)  # use @with_native_function?
        tn_key = gen_structseq_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f"NamedTuple{'' if not typedefs else len(typedefs)}"
            typenames[tn_key] = typename
            typedefs.append(
                f"""\
static PyTypeObject* {typename} = generated::get_{name}_structseq();"""
            )

    return typedefs, typenames


def generate_return_type_definition_and_registrations(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> tuple[list[str], list[str]]:
    """
    Generate block of function in `python_return_types.cpp` to initialize
    and return named tuple for a native function which returns named tuple
    and registration invocations in same file.
    """
    typenames: dict[
        str, str
    ] = {}  # map from unique name + field name lists to typedef name
    definitions: list[str] = []  # function definition to register the typedef
    registrations: list[str] = []  # register call for the typedef

    for overload in overloads:
        fieldnames = structseq_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue

        fields = ", ".join(f'{{"{fn}", ""}}' for fn in fieldnames)

        name = cpp.name(overload.function.func)  # use @with_native_function?
        tn_key = gen_structseq_typename_key(overload.function)
        typename = typenames.get(tn_key)

        if typename is None:
            typename = f"{name}NamedTuple{'' if not definitions else len(definitions)}"
            typenames[tn_key] = typename
            definitions.append(
                f"""\
PyTypeObject* get_{name}_structseq() {{
    static PyStructSequence_Field NamedTuple_fields[] = {{ {fields},  {{nullptr}} }};
    static PyTypeObject {typename};
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = {{ "torch.return_types.{name}", nullptr, NamedTuple_fields, {len(fieldnames)} }};
    if (!is_initialized) {{
        PyStructSequence_InitType(&{typename}, &desc);
        {typename}.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }}
    return &{typename};
}}
"""
            )
            registrations.append(
                f'addReturnType(return_types_module, "{name}", generated::get_{name}_structseq());'
            )

    return definitions, registrations


def generate_return_type_declarations(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> list[str]:
    """
    Generate block of function declarations in `python_return_types.h` to initialize
    and return named tuple for a native function.
    """
    typenames: dict[
        str, str
    ] = {}  # map from unique name + field name lists to typedef name
    declarations: list[str] = []  # function declaration to register the typedef

    for overload in overloads:
        fieldnames = structseq_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue

        name = cpp.name(overload.function.func)  # use @with_native_function?
        tn_key = gen_structseq_typename_key(overload.function)
        typename = typenames.get(tn_key)

        if typename is None:
            typename = (
                f"{name}NamedTuple{'' if not declarations else len(declarations)}"
            )
            typenames[tn_key] = typename
            declarations.append(f"PyTypeObject* get_{name}_structseq();")

    return declarations


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         Method Impl Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# python binding for all overloads of a particular function/method
PY_VARIABLE_METHOD_VARARGS = CodeTemplate(
    r"""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  ${method_header}
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});

  ParsedArgs<${max_args}> parsed_args;
  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);
  ${check_has_torch_function}
  switch (_r.idx) {
    ${dispatch}
  }
  ${method_footer}
}

"""
)

# handler for a single parsed signature - may be a single overload or
# a pair of overloads that whose signatures only differ in output params
# (plugged into PY_VARIABLE_METHOD_VARARGS as an item in ${dispatch})
PY_VARIABLE_CASE = CodeTemplate(
    """\
case ${overload_index}: {
  ${body}
}
"""
)

# python binding for single-overload function/method
PY_VARIABLE_METHOD_VARARGS_SINGLETON = CodeTemplate(
    """\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  ${method_header}
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});

  ParsedArgs<${max_args}> parsed_args;
  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);
  ${check_has_torch_function}
  ${dispatch}
  ${method_footer}
}

"""
)

# python binding for a method with no args, shortcuts parsing
PY_VARIABLE_METHOD_NOARGS = CodeTemplate(
    """\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args)
{
  ${method_header}
  ${check_has_torch_function}
  ${dispatch}
  ${method_footer}
}

"""
)


def method_impl(
    name: BaseOperatorName,
    module: str | None,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
    symint: bool = True,
) -> str:
    """
    Generate a python binding for all overloads of an op.
    """
    pycname = get_pycname(name)
    noarg = is_noarg(overloads)
    structseq_inits, structseq_typenames = emit_structseq_call(overloads)

    method_header = ["HANDLE_TH_ERRORS"]
    method_header += structseq_inits
    method_header += (
        ["const Tensor& self = THPVariable_Unpack(self_);"] if method else []
    )

    method_footer = ([] if noarg else ["Py_RETURN_NONE;"]) + ["END_HANDLE_TH_ERRORS"]

    traceable = "true" if all(should_trace(o.function) for o in overloads) else "false"

    grouped_overloads: Sequence[PythonSignatureGroup] = group_overloads(
        overloads, symint=symint
    )
    is_singleton = len(grouped_overloads) == 1
    signatures: list[str] = []
    dispatch: list[str] = []
    for overload_index, overload in enumerate(grouped_overloads):
        signature = overload.signature.signature_str(symint=symint)
        signatures.append(f"{cpp_string(str(signature))},")
        dispatch_body = emit_dispatch_case(overload, structseq_typenames, symint=symint)
        dispatch.append(
            PY_VARIABLE_CASE.substitute(
                overload_index=overload_index, body=dispatch_body
            )
            if not is_singleton
            else dispatch_body
        )

    if noarg:
        template = PY_VARIABLE_METHOD_NOARGS
    elif is_singleton:
        template = PY_VARIABLE_METHOD_VARARGS_SINGLETON
    else:
        template = PY_VARIABLE_METHOD_VARARGS

    return template.substitute(
        name=name,
        pycname=pycname,
        method_header=method_header,
        max_args=max(o.signature.arguments_count() for o in overloads),
        signatures=signatures,
        traceable=traceable,
        check_has_torch_function=gen_has_torch_function_check(
            name=name,
            module=module,
            noarg=noarg,
            method=method,
        ),
        dispatch=dispatch,
        method_footer=method_footer,
        self_="self_" if method else "nullptr",
    )


def gen_has_torch_function_check(
    name: BaseOperatorName, module: str | None, *, noarg: bool, method: bool
) -> str:
    if noarg:
        if method:
            return f"""\
if (has_torch_function(self_)) {{
  return handle_torch_function(self_, "{name}");
}}
"""
        else:
            return ""

    self_ = "self_" if method else "nullptr"
    namespace = (
        {
            "torch": "THPVariableFunctionsModule",
            "torch.nn": "THPNNVariableFunctionsModule",
            "torch.fft": "THPFFTVariableFunctionsModule",
            "torch.linalg": "THPLinalgVariableFunctionsModule",
            "torch.nested": "THPNestedVariableFunctionsModule",
            "torch.sparse": "THPSparseVariableFunctionsModule",
            "torch.special": "THPSpecialVariableFunctionsModule",
        }[module]
        if module
        else "THPVariableClass"
    )

    return f"""\
if(_r.has_torch_function()) {{
  return handle_torch_function(_r, {self_}, args, kwargs, {namespace}, "{module or "torch.Tensor"}");
}}
"""


# handler for output/no-output overload pair
PY_VARIABLE_OUT = CodeTemplate(
    """\
if (_r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  ${call_dispatch_out}
}
"""
)


def emit_dispatch_case(
    overload: PythonSignatureGroup,
    structseq_typenames: dict[str, str],
    *,
    symint: bool = True,
) -> str:
    """
    Emit dispatch code for a single parsed signature. This corresponds to either
    a single native function, or a pair that differ only in output params. In the
    latter case, a single python signature is used for both and dispatching
    switches on the presence/absence of passed output args.
    """
    if overload.outplace is not None:
        # dispatch output and no-output variants, branch on _r.isNone(<out_idx>)
        return PY_VARIABLE_OUT.substitute(
            out_idx=overload.signature.output_idx(),
            call_dispatch=emit_single_dispatch(
                overload.signature, overload.base, structseq_typenames, symint=symint
            ),
            call_dispatch_out=emit_single_dispatch(
                overload.signature,
                overload.outplace,
                structseq_typenames,
                symint=symint,
            ),
        )
    else:
        # no-output version only
        return emit_single_dispatch(
            overload.signature, overload.base, structseq_typenames, symint=symint
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                    Forward Declarations Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def forward_decls(
    name: BaseOperatorName,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
) -> tuple[str, ...]:
    if method:
        return ()

    pycname = get_pycname(name)
    if is_noarg(overloads):
        return (
            f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args);
""",
        )
    else:
        return (
            f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args, PyObject* kwargs);
""",
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#              Method Def (Binding Table Entry) Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def method_def(
    name: BaseOperatorName,
    module: str | None,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
) -> str:
    """
    Generate method def entry.
    """
    pycname = get_pycname(name)

    if name.dunder_method:
        # PyMethodDef entry for binary op, throws not implemented error
        pycname = f"TypeError_to_NotImplemented_<{pycname}>"

    if is_noarg(overloads):
        flags = "METH_NOARGS" if method else "METH_VARARGS | METH_KEYWORDS"
    else:
        pycname = f"castPyCFunctionWithKeywords({pycname})"
        flags = "METH_VARARGS | METH_KEYWORDS"

    if module == "torch":
        flags += " | METH_STATIC"

    return f'{{"{name}", {pycname}, {flags}, nullptr}},'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                   Overload Sorting and Grouping
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def group_overloads(
    overloads: Sequence[PythonSignatureNativeFunctionPair], *, symint: bool = True
) -> Sequence[PythonSignatureGroup]:
    bases: dict[str, PythonSignatureNativeFunctionPair] = {}
    outplaces: dict[str, PythonSignatureNativeFunctionPair] = {}

    # first group by signature ignoring out arguments
    for overload in overloads:
        sig = overload.signature.signature_str(skip_outputs=True, symint=symint)
        if overload.function.func.is_out_fn():
            if sig in outplaces:
                raise RuntimeError(
                    f"Found duplicated function definition:\n- {overload.function.func}.\n"
                    f"Existing definition:\n- {outplaces[sig].function.func}."
                )
            outplaces[sig] = overload
        else:
            if sig in bases:
                raise RuntimeError(
                    f"Found duplicated function definition:\n- {overload.function.func}.\n"
                    f"Existing definition:\n- {bases[sig].function.func}."
                )
            bases[sig] = overload

    for sig, out in outplaces.items():
        if sig not in bases:
            candidates: list[str] = []
            for overload in overloads:
                if (
                    str(overload.function.func.name.name)
                    == str(out.function.func.name.name)
                    and not overload.function.func.is_out_fn()
                    and not overload.signature.deprecated
                ):
                    candidates.append(
                        overload.signature.signature_str(
                            skip_outputs=True, symint=symint
                        )
                    )
            out_sig = out.signature.signature_str(symint=symint)
            raise RuntimeError(
                f"While identifying overloads, we found an out schema {out_sig} without a corresponding non-out variant. "
                f"We expected the non-out variant to have schema: \n- {sig}\nPlease check that you spelled the schema "
                "correctly in native_functions.yaml. We discovered the following candidate(s): \n"
                + "\n".join(f"- {candidate}" for candidate in candidates)
            )

    grouped = [
        PythonSignatureGroup.from_pairs(
            functional=base,
            out=outplaces.get(sig),
        )
        for sig, base in bases.items()
    ]
    return sort_overloads(grouped, symint=symint)


# This function declares a partial order on declarations, and sorts them according
# to its linear extension. This is necessary, because there's some ambiguity in the
# choice of overload, and we want a different order.
#
# See Note[Order of overloads matters]
#
# A few examples of ambiguous python signature pairs.
#
#   All parameters have the same type, except one taking Tensor the other taking
#   Scalar. A numeric PyObject can be casted into Tensor, and a zero-dim Tensor
#   object can be accepted as Scalar type parameter (see python_arg_parser.cpp).
#   Therefore, same input arguments might be accepted by either python signature.
#   We want to always parse the one taking Tensor first.
#
#     bitwise_and(Tensor input, Tensor other, *, Tensor out=None)
#     bitwise_and(Tensor input, Scalar other, *, Tensor out=None)
#
#   If they have different number of parameters then they are not ambiguous - but
#   the difference on output param can be ignored as it's optional.
#
#     multiply(Tensor input, Tensor other, *, Tensor out=None)
#     multiply(Tensor input, Scalar other)
#
#   Both positional args and keyword-only args are considered together.
#
#     subtract(Tensor other, *, Scalar alpha=1)
#     subtract(Scalar other, Scalar alpha=1)
#
# A few ambiguous cases which it does NOT handle yet.
#
#   If there is any difference in other parameters besides the Tensor/Scalar
#   difference, then they are not considered ambiguous by this method anymore.
#   However, the difference could be too trivial to disambiguate.
#
#     foo(Tensor input, Scalar other, Scalar bar)
#     foo(Tensor input, Tensor other, double bar)
#
#   If they are taking different number of parameters then they are not considered
#   ambiguous anymore, even if the difference is only on optional kwargs.
#
#     foo(Scalar other, Scalar alpha=1)
#     foo(Tensor other, *, Scalar alpha=1, Scalar beta=1)
#


def sort_overloads(
    grouped_overloads: Sequence[PythonSignatureGroup], *, symint: bool = True
) -> Sequence[PythonSignatureGroup]:
    # NB: Smaller here means lower priority

    def is_arg_smaller(t1: Type, t2: Type) -> bool:
        return (
            str(t1) == "Scalar"
            and str(t2) == "Tensor"
            or str(t1) == "Scalar?"
            and str(t2) == "Tensor?"
            or
            # In the discussion https://github.com/pytorch/pytorch/issues/54555 it has been
            # discussed why it is important to prioritize int/int? over int[]
            str(t1) == "int[]"
            and (str(t2) == "int" or str(t2) == "int?")
            or
            # TensorList currently throws an error during argument parsing, that's why it needs to be
            # last in signature ordering. See discussion: https://github.com/pytorch/pytorch/issues/58087
            str(t1) == "Tensor[]"
            and str(t2).find("[]") != -1
            or
            # Prioritize IntArrayRef overload over SymIntArrayRef
            str(t1) == "SymInt[]"
            and str(t2) == "int[]"
            or
            # Make sure both in, SymInt are sorted consistently w.r.t. Tensor since Tensor can be implicitly
            # converted to either int or SymInt.  Prioritize the Tensor overload since it otherwise gets shadowed.
            (str(t1) == "SymInt" or str(t1) == "int")
            and str(t2) == "Tensor"
        )

    def is_smaller(s1: PythonSignature, s2: PythonSignature) -> bool:
        """Returns True if s1 < s2 in the partial order."""
        args1, args2 = s1.arguments(skip_outputs=True), s2.arguments(skip_outputs=True)
        if len(args1) != len(args2):
            return False
        # TODO: should use some canonical form instead of 'str(arg.type)' - see comments
        # above. The old codegen used the deprecated 'dynamic_type(arg.type)', which
        # ignores the optional annotation, i.e. 'Scalar' and 'Scalar?'.
        equal = all(arg1.type == arg2.type for arg1, arg2 in zip(args1, args2))
        smaller_or_equal = all(
            str(arg1.type) == str(arg2.type) or is_arg_smaller(arg1.type, arg2.type)
            for arg1, arg2 in zip(args1, args2)
        )
        return smaller_or_equal and not equal

    # First sort by signature
    grouped_overloads = sorted(
        grouped_overloads, key=lambda x: x.signature.signature_str(symint=symint)
    )

    # Construct the relation graph
    larger_than: dict[int, set[int]] = defaultdict(set)
    for i1, overload1 in enumerate(grouped_overloads):
        for i2, overload2 in enumerate(grouped_overloads):
            if is_smaller(overload1.signature, overload2.signature):
                larger_than[i1].add(i2)

    if not larger_than:
        return list(grouped_overloads)

    # Use a topological sort to sort overloads according to the partial order.
    N = len(grouped_overloads)
    sorted_ids: list[int] = list(filter(lambda x: x not in larger_than, range(N)))

    for idx in range(N):
        # The size of sorted_ids will grow to N eventually.
        i = sorted_ids[idx]
        for j in sorted(larger_than.keys()):
            larger = larger_than[j]
            larger.discard(i)
            if not larger:
                del larger_than[j]
                sorted_ids.append(j)

    return [grouped_overloads[x] for x in sorted_ids]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       Codegen API Integration
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def emit_single_dispatch(
    ps: PythonSignature,
    f: NativeFunction,
    structseq_typenames: dict[str, str],
    *,
    symint: bool = True,
) -> str:
    """
    Emit dispatch code for a single native function.
    """

    @with_native_function
    def go(f: NativeFunction) -> str:
        # header comments
        if isinstance(ps, PythonSignatureDeprecated):
            schema_comment = f"// [deprecated] aten::{ps.deprecated_schema}"
        else:
            schema_comment = f"// aten::{f.func}"

        # dispatch lambda signature
        name = cpp.name(f.func)
        lambda_formals = ", ".join(
            f"{a.type_str} {a.name}" for a in dispatch_lambda_args(ps, f, symint=symint)
        )
        lambda_return = dispatch_lambda_return_str(f)

        # dispatch lambda body
        dispatch_callee = cpp_dispatch_target(f)
        dispatch_args = ", ".join(cpp_dispatch_exprs(f, python_signature=ps))

        # from arg parser outputs to dispatch lambda arguments
        parser_outputs = arg_parser_output_exprs(ps, f, symint=symint)
        lambda_arg_exprs = dispatch_lambda_exprs(ps, f, symint=symint)
        inits = "\n".join(lambda_arg_exprs.inits)
        lambda_args = ", ".join(lambda_arg_exprs.exprs)

        # scatter fields
        # TODO: Checking `ps.method and ('requires_grad' in parser_outputs)` is a hacky
        #       solution for enabling the 'requires_grad' argument for tensor methods
        #       new_full, new_empty, and new_zeros. A much better but more difficult to
        #       implement solution involves refactoring according to Ed's description here:
        #       https://github.com/pytorch/pytorch/issues/36455#issuecomment-614767589
        need_set_requires_grad = ps.tensor_options_args and (
            not has_tensor_options(f)
            or (ps.method and ("requires_grad" in parser_outputs))
        )
        set_requires_grad = (
            f".set_requires_grad({parser_outputs['requires_grad'].expr})"
            if need_set_requires_grad
            else ""
        )

        if lambda_return == "void":
            # Make in-place foreach return `self` at python-binding level.
            # ref: https://github.com/pytorch/pytorch/pull/118622#pullrequestreview-1904804954
            self_arg = f.func.arguments.self_arg
            return_stmt: str
            if (
                str(f.func.name).startswith("_foreach_")
                and f.func.kind() == SchemaKind.inplace
            ):
                # note(crcrpar): `_foreach_pow.ScalarAndTensor` does NOT have its in-place
                # variant and it unlikely to have it in the future. Thus it's safe to have the following check.
                if self_arg is None or not is_tensor_list_type(self_arg.argument.type):
                    raise AssertionError(
                        "Expected self_arg to be a tensor list type for inplace foreach"
                    )
                return_stmt = """PyObject* self_tensorlist = _r.args[0];
Py_INCREF(self_tensorlist);
return self_tensorlist;
"""
            else:
                return_stmt = "Py_RETURN_NONE;"
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  pybind11::gil_scoped_release no_gil;
  {dispatch_callee}({dispatch_args});
}};
dispatch_{name}({lambda_args}){set_requires_grad};
{return_stmt}
"""
        else:
            typename = structseq_typenames.get(gen_structseq_typename_key(f))
            structseq_typeref = f"{typename}, " if typename is not None else ""
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  pybind11::gil_scoped_release no_gil;
  return {dispatch_callee}({dispatch_args});
}};
return wrap({structseq_typeref}dispatch_{name}({lambda_args}){set_requires_grad});
"""

    return go(f)
<|endoftext|># Generates ADInplaceOrViewType.h/cpp
#
# NOTE: If any changes are being made to the ADInplaceOrView codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
# The fallback is expected to mimic this codegen, so we should keep the two in sync.

from __future__ import annotations

from torchgen.api import cpp
from torchgen.api.autograd import (
    dispatch_strategy,
    gen_differentiable_outputs,
    NativeFunctionWithDifferentiabilityInfo,
)
from torchgen.api.types import (
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    CType,
    DispatcherSignature,
    intArrayRefT,
    longT,
    OptionalCType,
    symIntArrayRefT,
    SymIntT,
    tensorT,
)
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import (
    NativeFunction,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import FileManager

from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
    get_return_value,
    MANUAL_AUTOGRAD,
    tie_return_values,
    type_wrapper_name,
)


# See NOTE [ Autograd View Variables ] in variable.h for details.
# If you update list VIEW_FUNCTIONS or RETURNS_VIEWS_OF_INPUT,
# you **MUST** also update the public list of view ops accordingly in
# docs/source/tensor_view.rst. Note not all ATen functions are exposed to public,
# e.g alias & sparse_coo_tensor_with_dims_and_tensors.
#
# A map: function name => name of the argument that all outputs are view of

VIEW_FUNCTIONS_WITH_METADATA_CHANGE = [
    "view_as_complex",
    "view_as_real",
    "_conj",
    "_neg_view",
    "_nested_get_values",
    "_nested_view_from_buffer",
    "_nested_view_from_jagged",
]

VIEW_FUNCTIONS = {
    "numpy_T": "self",
    "alias": "self",
    "as_strided": "self",
    "diagonal": "self",
    "expand": "self",
    "permute": "self",
    "select": "self",
    "slice": "self",
    "slice_inverse": "self",
    "split": "self",
    "split_with_sizes": "self",
    "squeeze": "self",
    "t": "self",
    "transpose": "self",
    "unfold": "self",
    "unsqueeze": "self",
    "flatten": "self",
    "view": "self",
    "unbind": "self",
    "_indices": "self",
    "_values": "self",
    "indices": "self",
    "values": "self",
    "crow_indices": "self",
    "col_indices": "self",
    "ccol_indices": "self",
    "row_indices": "self",
    # sparse_coo ctor output should really be views of both indices and values,
    # but we only supports making as view of a single variable, and indices is
    # discrete anyways.
    # FIXME: clone indices on construction.
    "sparse_coo_tensor_with_dims_and_tensors": "values",
    "_reshape_alias": "self",
    "_test_autograd_multiple_dispatch_view": "self",
}

for key in VIEW_FUNCTIONS_WITH_METADATA_CHANGE:
    VIEW_FUNCTIONS[key] = "self"

# note: some VIEW_FUNCTIONS are just compositions of the view functions above
# this list contains both the root view functions and any that are purely composed
# of viewing functions, and is used by the JIT to determine when an operator
# may return a view of its inputs; however they may sometimes return a copy.
# (e.g. `contiguous`)
RETURNS_VIEWS_OF_INPUT = set(VIEW_FUNCTIONS.keys()).union(
    {
        "chunk",
        "detach",
        "contiguous",
        "reshape",
        "reshape_as",
        "expand_as",
        "view_as",
        "real",
        "imag",
        "narrow",
        "movedim",
        "tensor_split",
        "swapdims",
        "swapaxes",
        "mT",
        "mH",
        "adjoint",
        "matrix_H",
    }
)

# These are the functions we consider views for the purposes of validating
# StorageImpl and TensorImpl in gen_variable_type.
# `_unsafe_view` is not included in VIEW_FUNCTIONS above because it is not a
# view for the purposes of ADInplaceOrView kernel, we do not want to call as_view
# See NOTE [Unsafe View] for more info.
ALL_VIEW_FUNCTIONS = {
    **VIEW_FUNCTIONS,
    "_unsafe_view": "self",
}

ARRAYREF_TO_VEC = CodeTemplate(
    """\
auto ${vec} = ${arg}.vec();
"""
)

OPTIONAL_TO_VAL = CodeTemplate(
    """\
auto ${val} = ${arg}.value_or(${default});
"""
)

CALL_DISPATCH = CodeTemplate(
    """\
at::_ops::${unambiguous_name}::call(${unpacked_args})"""
)

REVERSE_VIEW_DISPATCH = CodeTemplate(
    """\
${reverse_name}(${unpacked_args})"""
)

MULTI_OUTPUT_VIEW_ITERATION = CodeTemplate(
    """\
for (auto ${view_idx} : c10::irange(${var}.size())) {
  ${body}
}
"""
)

SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE = CodeTemplate(
    """\
std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
if (${is_view_with_metadata_change} ||
    !self.unsafeGetTensorImpl()->support_as_strided() ||
    self.unsafeGetTensorImpl()->is_python_dispatch() ||
    c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
  ${replay_view_func}
  ${reverse_replay_view_func}
}
"""
)

REPLAY_VIEW_FUNC = CodeTemplate(
    """\
func = std::make_unique<${view_func_name}>(${view_func_args});
"""
)

REVERSE_REPLAY_VIEW_LAMBDA_FUNC = CodeTemplate(
    """\
rev_func = [=](const at::Tensor& ${input_view}) {
  return ${reverse_replay_view_call};
};
"""
)

METHOD_DEFINITION = CodeTemplate(
    """\
${return_type} ${type_wrapper_name}(${formals}) {
  ${type_definition_body}
}
"""
)

WRAPPER_REGISTRATION = CodeTemplate(
    """\
m.impl("${unqual_operator_name_with_overload}",
       TORCH_FN(${class_type}::${type_wrapper_name})
);
"""
)

AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION = CodeTemplate(
    """\
m.impl("${unqual_operator_name_with_overload}", torch::autograd::autogradNotImplementedFallback());
"""
)

INPLACE_REDISPATCH = CodeTemplate(
    """\
{
  at::AutoDispatchBelowADInplaceOrView guard;
  at::_ops::${unambiguous_name}::redispatch(${unpacked_args});
}
"""
)

ASSIGN_RETURN_VALUE = CodeTemplate(
    """\
${return_values} = ${rhs_value};
"""
)

VIEW_REDISPATCH = CodeTemplate(
    """\
${assign_return_values} ([&]() {
  at::AutoDispatchBelowADInplaceOrView guard;
  return at::_ops::${unambiguous_name}::redispatch(${unpacked_args});
})();
"""
)

TMP_VAR = "_tmp"


# FIXME: Ideally these functions should be methods on Type class, but we have a
#        comment in codegen/model.py there saying these concepts are not well defined.
#        Thus we put a version that commonly used by autograd codegen here.
def is_tensor_type(t: Type) -> bool:
    # TODO: Should handle optional here?
    return t.is_tensor_like() and t.is_list_like() is None


def is_tensor_list_type(t: Type) -> bool:
    # TODO: Should handle optional here?
    return t.is_tensor_like() and t.is_list_like() is not None


UNPACK_TENSOR = CodeTemplate(
    """\
auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});"""
)


def unpacked_name(arg_name: str) -> str:
    return arg_name + "_"


# e.g. select.int -> select_copy_int_inverse()
def inverse_view_name(f: NativeFunction) -> str:
    copy_variant = f"{f.root_name}_copy"
    overload = f"{f.func.name.overload_name}"
    if overload != "":
        overload = "_" + overload
    return f"{copy_variant}{overload}_inverse"


def extract_bindings(f: NativeFunction) -> list[Binding]:
    return [
        r
        for a in f.func.schema_order_arguments()
        for r in cpp.argument(
            a,
            method=False,
            symint=True,
            cpp_no_default_args=set(),
            faithful=False,
            has_tensor_options=False,
        )
    ]


@with_native_function
def unpack_args(f: NativeFunction) -> tuple[list[str], list[Binding]]:
    body: list[str] = []
    unpacked_bindings: list[Binding] = []

    for i, binding in enumerate(extract_bindings(f)):
        if isinstance(binding.argument, SelfArgument):
            raise AssertionError("Binding argument should not be SelfArgument")
        if isinstance(binding.argument, TensorOptionsArguments):
            raise RuntimeError("VariableKernel shouldn't take TensorOptions")

        is_nullable = binding.argument.type.is_nullable()
        if not binding.argument.type.is_tensor_like() or is_nullable:
            unpacked_bindings.append(binding)
            continue

        is_tensor_list = is_tensor_list_type(binding.argument.type)
        ref = (not is_nullable) and not is_tensor_list
        suffix = "_opt" if is_nullable and not is_tensor_list else ""
        body.append(
            UNPACK_TENSOR.substitute(
                arg_name=binding.name,
                arg_pos=i,
                suffix=suffix,
                ref="&" if ref else "",
            )
        )
        unpacked_bindings.append(
            Binding(
                name=unpacked_name(binding.name),
                nctype=binding.nctype,
                argument=binding.argument,
                default=binding.default,
            )
        )

    return body, unpacked_bindings


def get_base_name(f: NativeFunction) -> str:
    return f.func.name.name.base  # TODO: should be str(f.func.name.name)?


def get_view_info(f: NativeFunction) -> str | None:
    base_name = get_base_name(f)
    view_info = VIEW_FUNCTIONS.get(base_name)
    if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
        view_info = "self"
    return view_info


def emit_view_func(
    f: NativeFunction, bindings: list[Binding], view_idx: str | None = None
) -> str:
    """Generate an additional lambda function to recover views in backward when as_strided is not supported.
    See Note [View + Inplace update for base tensor] and [View + Inplace update for view tensor] for more details.
    """
    # TODO: Clean this logic up if we get rid of reverse view funcs or reify them.
    input_base = "input_base"
    replay_view_func = ""
    updated_args: list[str] = []
    known_view_arg_simple_types: list[CType] = [
        BaseCType(longT),
        OptionalCType(BaseCType(longT)),
        BaseCType(SymIntT),
        OptionalCType(BaseCType(SymIntT)),
        BaseCType(boolT),
        BaseCType(intArrayRefT),
        BaseCType(symIntArrayRefT),
        ConstRefCType(BaseCType(tensorT)),
        ConstRefCType(OptionalCType(BaseCType(tensorT))),
    ]
    for binding in bindings:
        arg, arg_type = binding.name, binding.nctype.type
        if arg == "self":
            updated_args.append(input_base)
            continue
        if arg_type not in known_view_arg_simple_types:
            known_types_str = ", ".join([str(t) for t in known_view_arg_simple_types])
            raise TypeError(
                f"You are adding an {arg_type} {arg} argument to op {cpp.name(f.func)} in addition to known types: "
                f"{known_types_str}. Please update the list or materialize it so that it can be closed "
                "over by value, also add a test in pytorch/xla/test/test_operations.py where this code "
                "is exercised."
            )
        if arg_type == BaseCType(intArrayRefT) or arg_type == BaseCType(
            symIntArrayRefT
        ):
            # It's not safe to close over IntArrayRef by value, since this is a
            # reference type, so materialize a vector to close over by value
            arg_vec = arg + "_vec"
            replay_view_func += ARRAYREF_TO_VEC.substitute(arg=arg, vec=arg_vec)
            updated_args.append(arg_vec)
        elif arg_type == OptionalCType(BaseCType(longT)):
            # Materialize int64_t? to int64_t
            arg_value = arg + "_val"
            replay_view_func += OPTIONAL_TO_VAL.substitute(
                arg=arg, val=arg_value, default="0"
            )
            updated_args.append(arg_value)
        elif arg_type == ConstRefCType(BaseCType(tensorT)) or arg_type == ConstRefCType(
            OptionalCType(BaseCType(tensorT))
        ):
            # NB: Closing over a tensor. If a user modifies this tensor, this will be silently
            # incorrect. The proper thing to do is to store the version counter and copy on write.
            updated_args.append(arg)
        else:
            updated_args.append(arg)

    from .gen_view_funcs import view_func_name

    view_func_args = [b.name for b in bindings if b.name != "self"]
    if view_idx is not None:
        view_func_args.append(f"{view_idx}")
    replay_view_func += REPLAY_VIEW_FUNC.substitute(
        view_func_name=view_func_name(f, include_namespace=True),
        view_func_args=view_func_args,
    )

    input_view = "input_view"
    reverse_unpacked_args = [
        "self",
        f"{input_view}",
        # inverse_return_mode=
        "at::functionalization::InverseReturnMode::AlwaysView",
        *(() if view_idx is None else (f"{view_idx}",)),
        # skip input_base arg
        *updated_args[1:],
    ]

    from torchgen.api.functionalization import reverse_name

    reverse_replay_view_call = REVERSE_VIEW_DISPATCH.substitute(
        reverse_name=reverse_name(f, include_namespace=True),
        unpacked_args=reverse_unpacked_args,
    )
    reverse_replay_view_func = REVERSE_REPLAY_VIEW_LAMBDA_FUNC.substitute(
        input_view=input_view, reverse_replay_view_call=reverse_replay_view_call
    )

    is_view_with_metadata_change = (
        "true" if cpp.name(f.func) in VIEW_FUNCTIONS_WITH_METADATA_CHANGE else "false"
    )

    return SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE.substitute(
        is_view_with_metadata_change=is_view_with_metadata_change,
        replay_view_func=replay_view_func,
        reverse_replay_view_func=reverse_replay_view_func,
    )


def emit_view_body(
    fn: NativeFunctionWithDifferentiabilityInfo, var: str
) -> tuple[str, str]:
    # See NOTE [ Autograd View Variables ] in variable.h for details.
    f = fn.func
    base_name = get_base_name(f)
    view_info = get_view_info(f)
    call = ""
    differentiable_outputs = gen_differentiable_outputs(fn)
    differentiable_output_vars = {r.name for r in differentiable_outputs}
    if not isinstance(view_info, str):
        raise TypeError(
            f"The view info should be a string for {base_name}, but it is: {view_info}"
        )
    if len(differentiable_output_vars) == 0:
        # no output is differentiable (.indices() for SparseTensors for example)
        rhs_value = (
            f"as_view({view_info}, {var}, "
            f"/* is_bw_differentiable */ false, /* is_fw_differentiable */ false)"
        )
    elif len(differentiable_output_vars) == 1:
        # Single differentiable output (Tensor or Tensor[])
        return_info = differentiable_outputs[0]
        # We only support simple Tensor or a TensorList for functions that return views
        if not is_tensor_type(return_info.type) and not is_tensor_list_type(
            return_info.type
        ):
            raise RuntimeError(
                f"{base_name} that return differentiable views can only return Tensor or Tensor[]"
            )

        # See Note [ View + Inplace detection]
        def get_creation_meta_in_mode(original: str) -> str:
            creation_meta_with_grad_mode = f"(at::GradMode::is_enabled() ? {original} : CreationMeta::NO_GRAD_MODE)"
            return f"InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : {creation_meta_with_grad_mode}"

        # Only allow rebasing of the history if we return a single Tensor
        # If we are in a no grad block, raise a warning
        # See NOTE [ View + Inplace detection ] for more details about this logic
        if is_tensor_list_type(return_info.type):
            creation_meta = get_creation_meta_in_mode("CreationMeta::MULTI_OUTPUT_NODE")
            view_idx = "view_idx"
            view_func = emit_view_func(
                f, extract_bindings(f), view_idx=view_idx
            ).strip()
            as_view_call = (
                f"as_view(/* base */ {view_info}, /* output */ {var}[{view_idx}], "
                "/* is_bw_differentiable */ true, /* is_fw_differentiable */ true, "
                "/* view_func */ std::move(func), /* rev_view_func */ rev_func, "
                f"/* creation_meta */ {creation_meta});"
            )
            call += MULTI_OUTPUT_VIEW_ITERATION.substitute(
                var=var, view_idx=view_idx, body=f"{view_func}\n{as_view_call}"
            )
            rhs_value = f"std::move({var})"
        else:
            call += emit_view_func(f, extract_bindings(f), view_idx=None)
            creation_meta = get_creation_meta_in_mode("CreationMeta::DEFAULT")
            rhs_value = (
                f"as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, "
                "/* is_fw_differentiable */ true, "
                f"/* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ {creation_meta})"
            )
    else:
        # This could be supported but we don't need it at the moment, so keeping things simple.
        raise RuntimeError(
            "Function that return multiple differentiable output "
            "when at least one of them is view is not supported."
        )
    return call, rhs_value


def modifies_arguments(f: NativeFunction) -> bool:
    return f.func.kind() in [SchemaKind.inplace, SchemaKind.out]


@with_native_function_with_differentiability_info
def emit_inplace_or_view_body(fn: NativeFunctionWithDifferentiabilityInfo) -> list[str]:
    f = fn.func
    inplace_view_body: list[str] = []

    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()

    # code-generated ADInplaceOrView kernels plumb and recompute dispatch keys directly through the kernel for performance.
    # See Note [Plumbing Keys Through The Dispatcher] for details.
    dispatch_key_set = "ks & c10::after_ADInplaceOrView_keyset"
    redispatch_args = ", ".join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])

    # Note that this calls the slow, dispatching variants of manual_cpp_binding ops.
    # We could probably work harder to ensure that the fast variants are called instead, but the perf benefit would be minimal.
    if modifies_arguments(f):  # inplace op
        inplace_view_body.append(
            INPLACE_REDISPATCH.substitute(
                unambiguous_name=f.func.name.unambiguous_name(),
                unpacked_args=redispatch_args,
            )
        )
        for r in cpp.return_names(f):
            inplace_view_body.append(f"increment_version({r});")
    else:
        if get_view_info(f) is None:
            raise AssertionError("Expected view info to be non-None")
        inplace_view_body.append(
            VIEW_REDISPATCH.substitute(
                assign_return_values="auto " + TMP_VAR + " = ",
                unambiguous_name=f.func.name.unambiguous_name(),
                unpacked_args=redispatch_args,
            )
        )
        call, rhs_value = emit_view_body(fn, TMP_VAR)
        inplace_view_body.append(call)
        if rhs_value is None:
            raise AssertionError("Expected rhs_value to be non-None")
        inplace_view_body.append(
            ASSIGN_RETURN_VALUE.substitute(
                return_values=tie_return_values(f), rhs_value=rhs_value
            )
        )
    if f.func.returns:
        inplace_view_body.append(f"return {get_return_value(f)};")
    return inplace_view_body


@with_native_function
def gen_formals(f: NativeFunction) -> str:
    return ", ".join(
        # code-generated autograd kernels plumb and recompute dispatch keys directly through the kernel for performance.
        # See Note [Plumbing Keys Through The Dispatcher] for details.
        ["c10::DispatchKeySet ks"]
        + [
            f"{cpp.argument_type(a, binds='__placeholder__', symint=True).cpp_type()} {a.name}"
            for a in f.func.schema_order_arguments()
        ]
    )


@with_native_function_with_differentiability_info
def inplace_or_view_method_definition(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> str | None:
    f = fn.func
    if get_view_info(f) is None and (
        # For functions that modify their inputs but don't return them,
        # we can't give them autograd support.
        # See https://github.com/pytorch/pytorch/issues/53796
        not modifies_arguments(f) or len(f.func.returns) == 0
    ):
        return None
    return METHOD_DEFINITION.substitute(
        return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(),
        type_wrapper_name=type_wrapper_name(f),
        formals=gen_formals(f),
        type_definition_body=emit_inplace_or_view_body(fn),
    )


@with_native_function_with_differentiability_info
def inplace_or_view_method_registration(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> str | None:
    f = fn.func
    if get_view_info(f) is None and (
        not modifies_arguments(f) or len(f.func.returns) == 0
    ):
        return None
    return WRAPPER_REGISTRATION.substitute(
        unqual_operator_name_with_overload=f.func.name,
        type_wrapper_name=type_wrapper_name(f),
        class_type="ADInplaceOrView",
    )


def use_derived(fn: NativeFunctionWithDifferentiabilityInfo) -> bool:
    f = fn.func
    name = cpp.name(f.func)
    return name not in MANUAL_AUTOGRAD and dispatch_strategy(fn) == "use_derived"


def gen_inplace_or_view_type_env(
    fn: NativeFunctionWithDifferentiabilityInfo,
) -> dict[str, list[str]]:
    definition = inplace_or_view_method_definition(fn)
    registration = inplace_or_view_method_registration(fn)

    return {
        "ops_headers": (
            [f"#include <ATen/ops/{fn.func.root_name}_ops.h>"]
            if definition is not None
            else []
        ),
        "inplace_or_view_method_definitions": [definition]
        if definition is not None
        else [],
        "inplace_or_view_wrapper_registrations": [registration]
        if registration is not None
        else [],
    }


def gen_inplace_or_view_type(
    out: str,
    native_yaml_path: str,
    tags_yaml_path: str,
    fns_with_infos: list[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    # NOTE: see Note [Sharded File] at the top of the VariableType.cpp
    # template regarding sharding of the generated files.

    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_sharded(
        "ADInplaceOrViewType.cpp",
        [fn for fn in fns_with_infos if use_derived(fn)],
        key_fn=lambda fn: fn.func.root_name,
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/ADInplaceOrViewType.cpp",
        },
        env_callable=gen_inplace_or_view_type_env,
        num_shards=2,
        sharded_keys={
            "ops_headers",
            "inplace_or_view_method_definitions",
            "inplace_or_view_wrapper_registrations",
        },
    )
<|endoftext|># Generates ViewFuncs.h/cpp
#
# NOTE: If any changes are being made to the ViewFunc codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
# The fallback is expected to mimic this codegen, so we should keep the two in sync.

from __future__ import annotations

from typing import TYPE_CHECKING

import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    NamedCType,
    SymIntT,
    tensorT,
    VectorCType,
)
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, NativeFunction, OptionalType
from torchgen.utils import FileManager

from .gen_inplace_or_view_type import (
    CALL_DISPATCH,
    extract_bindings,
    get_view_info,
    modifies_arguments,
    use_derived,
)


if TYPE_CHECKING:
    from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo


FUNCTION_DECLARATION = CodeTemplate(
    """\
#define ${uppercase_op}_AVAILABLE
struct ${op} : public ${superclass} {
  ${op}(${constructor_args}) ${initializer_list}
  {}
  virtual ~${op}() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ${state}
};

"""
)

FUNCTION_DEFINITION = CodeTemplate(
    """\
std::vector<c10::SymInt> ${op}::get_symints() const {
  ${get_symints}
}

size_t ${op}::num_symints() const {
  return static_cast<size_t>(${num_symints});
}

void ${op}::set_symints(std::vector<c10::SymInt> ${symints_vec}) {
  TORCH_INTERNAL_ASSERT(${symints_vec}.size() == num_symints());
  ${set_symints}
}

std::vector<at::Tensor> ${op}::get_tensors() const {
  ${get_tensors}
}

size_t ${op}::num_tensors() const {
  return static_cast<size_t>(${num_tensors});
}

void ${op}::set_tensors(std::vector<at::Tensor> ${tensors_vec}) {
  TORCH_INTERNAL_ASSERT(${tensors_vec}.size() == num_tensors());
  ${set_tensors}
}

at::Tensor ${op}::operator()(const at::Tensor& ${call_input_name}) const {
  return ${op_call};
}

std::unique_ptr<ViewFunc> ${op}::clone_and_set(
    std::optional<std::vector<c10::SymInt>> ${symints_vec},
    std::optional<std::vector<at::Tensor>> ${tensors_vec}) const {
  auto output = std::make_unique<${op}>(${clone_args});
  if (${symints_vec}.has_value()) {
    output->set_symints(std::move(*(${symints_vec})));
  }
  if (${tensors_vec}.has_value()) {
    output->set_tensors(std::move(*(${tensors_vec})));
  }
  return output;
}

"""
)


# e.g. as_strided -> AsStridedViewFunc for camel case or
# as_strided_view_func otherwise
def view_func_name(
    f: NativeFunction, include_namespace: bool = False, camel_case: bool = True
) -> str:
    name = f.func.name.unambiguous_name()
    view_func_name = f"{name.replace('.', '_')}_view_func"
    if camel_case:
        is_private = view_func_name.startswith("_")
        view_func_name = "".join(
            [p.title() for p in view_func_name.replace(".", "_").split("_")]
        )
        if is_private:
            # put the leading underscore back in
            view_func_name = f"_{view_func_name}"
    namespace = "torch::autograd::generated::" if include_namespace else ""
    return f"{namespace}{view_func_name}"


def is_symint_or_tensor(arg: Argument) -> bool:
    return arg.type.is_tensor_like() or arg.type.is_symint_like()


def remove_const_ref(binding: Binding) -> Binding:
    return Binding(
        name=binding.name,
        nctype=binding.nctype.remove_const_ref(),
        argument=binding.argument,
        default=binding.default,
    )


def returns_multi_tensor(fn: NativeFunction) -> bool:
    returns = fn.func.returns
    if len(returns) != 1:
        raise AssertionError(f"Expected 1 return, got {len(returns)}")
    returns_list_like = returns[0].type.is_list_like() is not None
    returns_tensor_like = returns[0].type.is_tensor_like()
    return returns_list_like and returns_tensor_like


# Generates strings with logic for getting / setting state of a particular type.
#
# Args:
#   bindings (list): List of state bindings of interest (may be empty)
#   state_vec_type (NamedCType): Type of vector to either return or copy from
#
# Returns:
#   tuple: (list of getter logic strings, list of setter logic strings, string
#     with num items expression)
def generate_state_getter_setter(
    bindings: list[Binding],
    state_vec_type: NamedCType,
) -> tuple[list[str], list[str], str]:
    getter_logic = []
    setter_logic = []

    state_vec = state_vec_type.name
    getter_logic.append(f"{state_vec_type.cpp_type()} {state_vec};")
    if len(bindings) > 0:
        setter_logic.append("auto i = 0;")

    num_exprs = []
    for i, b in enumerate(bindings):
        if not isinstance(b.argument, Argument):
            raise AssertionError(f"Expected Argument, got {type(b.argument)}")
        if b.argument.type.is_list_like():
            # Handle list-likes.
            num_expr = f"{b.name}.size()"
            num_exprs.append(num_expr)
            getter = f"{state_vec}.insert({state_vec}.end(), {b.name}.begin(), {b.name}.end());"
            setter = f"std::copy({state_vec}.begin() + i, {state_vec}.begin() + i + {b.name}.size(), {b.name}.begin());"
        elif isinstance(b.argument.type, OptionalType):
            # Handle optionals.
            num_expr = f"({b.name}.has_value() ? 1 : 0)"
            num_exprs.append(num_expr)
            conditional = f"if({b.name}.has_value())"
            getter = (
                f"{conditional} {state_vec}.insert({state_vec}.end(), *({b.name}));"
            )
            setter = f"{conditional} {b.name} = {state_vec}[i];"
        else:
            num_expr = "1"
            num_exprs.append(num_expr)
            getter = f"{state_vec}.push_back({b.name});"
            setter = f"{b.name} = {state_vec}[i];"

        getter_logic.append(getter)
        setter_logic.append(setter)
        if i < len(bindings) - 1:
            setter_logic.append(f"i += {num_expr};")

    # Reserve / assert based on the total number of items expression.
    num_items = "0" if len(num_exprs) == 0 else " + ".join(num_exprs)
    if len(bindings) > 0:
        getter_logic.insert(1, f"{state_vec}.reserve({num_items});")

    getter_logic.append(f"return {state_vec};")

    return getter_logic, setter_logic, num_items


def process_function(fn: NativeFunction, template: CodeTemplate) -> str:
    bindings = extract_bindings(fn)
    non_self_bindings = [b for b in bindings if b.name != "self"]

    non_self_args = fn.func.arguments.flat_all[1:]
    non_self_value_bindings = [
        dispatcher.argument(a, remove_non_owning_ref_types=True) for a in non_self_args
    ]

    # Generate constructor / clone args for the generated struct.
    constructor_args = [b.defn() for b in non_self_bindings]
    clone_args = [b.name for b in non_self_bindings]

    # Generate state variable declarations for the generated struct.
    state_variables = [
        f"{remove_const_ref(b).defn()};" for b in non_self_value_bindings
    ]

    # Generate initializer list expressions for the generated struct.
    # allow_expensive_conversions=True because we need to store e.g. SymIntArrayRefs as
    # vector<SymInt>s.
    init_exprs = translate(
        non_self_bindings, non_self_value_bindings, allow_expensive_conversions=True
    )
    initializers = []
    for b, init_expr in zip(non_self_bindings, init_exprs):
        name = b.nctype.name
        if not isinstance(name, str):
            raise AssertionError(f"Expected name to be str, got {type(name)}")
        initializers.append(f"{name}({init_expr.expr})")

    # Generate call to underlying view op
    call_input_name = "input_base"
    op_call_args = [call_input_name, *(b.name for b in non_self_bindings)]
    op_call = CALL_DISPATCH.substitute(
        unambiguous_name=fn.func.name.unambiguous_name(),
        unpacked_args=op_call_args,
    )

    # Multi-output views additionally require a view_idx for disambiguation.
    if returns_multi_tensor(fn):
        view_idx_name = "view_idx"
        view_idx_typename = "int64_t"
        view_idx_decl = f"{view_idx_typename} {view_idx_name}"
        constructor_args.append(view_idx_decl)
        clone_args.append(view_idx_name)
        state_variables.append(f"{view_idx_decl};")
        initializers.append(f"{view_idx_name}({view_idx_name})")
        op_call += f"[{view_idx_name}]"

    # Generate initializer list for the generated struct.
    initializer_list = f": {', '.join(initializers)}" if len(initializers) > 0 else ""

    # Generate getter / setter logic for any symints.
    symint_bindings = [
        b
        for b in non_self_bindings
        if isinstance(b.argument, Argument) and b.argument.type.is_symint_like()
    ]
    symints_vec_type = NamedCType("symints", VectorCType(BaseCType(SymIntT)))
    get_symints, set_symints, num_symints = generate_state_getter_setter(
        symint_bindings, symints_vec_type
    )

    # Generate getter / setter logic for any tensors.
    tensor_bindings = [
        b
        for b in non_self_bindings
        if isinstance(b.argument, Argument) and b.argument.type.is_tensor_like()
    ]
    tensors_vec_type = NamedCType("tensors", VectorCType(BaseCType(tensorT)))
    get_tensors, set_tensors, num_tensors = generate_state_getter_setter(
        tensor_bindings, tensors_vec_type
    )

    return template.substitute(
        op=view_func_name(fn),
        uppercase_op=view_func_name(fn, camel_case=False).upper(),
        superclass="torch::autograd::ViewFunc",
        initializer_list=initializer_list,
        state=state_variables,
        constructor_args=constructor_args,
        clone_args=clone_args,
        symints_vec=symints_vec_type.name,
        get_symints=get_symints,
        set_symints=set_symints,
        num_symints=num_symints,
        tensors_vec=tensors_vec_type.name,
        get_tensors=get_tensors,
        set_tensors=set_tensors,
        num_tensors=num_tensors,
        call_input_name=call_input_name,
        op_call=op_call,
    )


def gen_view_funcs(
    out: str,
    fns_with_infos: list[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    # don't need the info parts, just the function
    fns = [fn.func for fn in fns_with_infos if use_derived(fn)]
    # only want out-of-place views
    view_fns = [
        fn for fn in fns if get_view_info(fn) is not None and not modifies_arguments(fn)
    ]

    declarations = [process_function(fn, FUNCTION_DECLARATION) for fn in view_fns]
    definitions = [process_function(fn, FUNCTION_DEFINITION) for fn in view_fns]
    ops_headers = [f"#include <ATen/ops/{fn.root_name}_ops.h>" for fn in view_fns]

    file_basename = "ViewFuncs"
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for suffix in [".h", ".cpp"]:
        fname = file_basename + suffix
        fm.write_with_template(
            fname,
            fname,
            lambda: {
                "generated_comment": "@"
                + f"generated from {fm.template_dir_for_comments()}/{fname}",
                "view_func_declarations": declarations,
                "view_func_definitions": definitions,
                "ops_headers": ops_headers,
            },
        )
<|endoftext|># Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

from __future__ import annotations

import re

import torchgen.api.python as python
from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, TensorOptionsArguments, Variant
from torchgen.utils import FileManager, mapMaybe


OPTIONAL_TYPE_PATTERN = re.compile(r"std::optional<(.+)>")
TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")


# Add 'at::' to types defined in ATen namespace, e.g. Tensor, TensorList, IntArrayRef and etc.
# TODO: maybe update the cpp argument API to take optional namespace argument?
def fully_qualified_type(argument_type: str) -> str:
    def maybe_optional_type(type: str, is_opt: bool) -> str:
        return f"std::optional<{type}>" if is_opt else type

    opt_match = OPTIONAL_TYPE_PATTERN.match(argument_type)
    is_opt = opt_match is not None
    if opt_match:
        argument_type = argument_type[opt_match.start(1) : opt_match.end(1)]
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return maybe_optional_type(argument_type, is_opt)
    index = match.start(1)
    qualified_type = f"{argument_type[:index]}at::{argument_type[index:]}"
    return maybe_optional_type(qualified_type, is_opt)


def gen_variable_factories(
    out: str, native_yaml_path: str, tags_yaml_path: str, template_path: str
) -> None:
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    factory_functions = [fn for fn in native_functions if is_factory_function(fn)]
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template(
        "variable_factories.h",
        "variable_factories.h",
        lambda: {
