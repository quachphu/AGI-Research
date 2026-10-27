                        var_types=fw_info.var_types,
                        required_inputs_fw_grad=fw_info.required_inputs_fw_grad,
                        required_inputs_primal=required_primals,
                        required_original_self_value=required_original_self_value,
                        is_reusing_outplace_formula=not is_exact_match,
                    ),
                ]

            fw_derivative_dict[key] = forward_derivatives

        result.append(
            NativeFunctionWithDifferentiabilityInfo(
                func=f, info=info_dict, fw_derivatives=fw_derivative_dict
            )
        )

    return result


def is_differentiable(
    name: str, type: Type, info: DifferentiabilityInfo | None
) -> bool:
    return type.is_tensor_like() and (
        info is None or name not in info.non_differentiable_arg_names
    )


def gen_differentiable_outputs(
    fn: NativeFunctionWithDifferentiabilityInfo, key: str = "Default"
) -> list[DifferentiableOutput]:
    f = fn.func
    info = fn.info[key] if fn.info else None
    outputs: list[DifferentiableOutput] = [
        DifferentiableOutput(
            name=name,
            type=ret.type,
            cpp_type=cpp.return_type(ret, symint=True).cpp_type(),
        )
        for name, ret in zip(cpp.return_names(f), f.func.returns)
    ]
    output_differentiability = info.output_differentiability if info else None
    if output_differentiability is not None:
        if len(output_differentiability) != len(outputs):
            raise RuntimeError(
                f"The length of output_differentiability ({len(output_differentiability)}), "
                f"does not match the number of outputs ({len(outputs)})."
            )
        differentiable_outputs: list[DifferentiableOutput] = []
        if False in output_differentiability and f.func.kind() == SchemaKind.inplace:
            raise RuntimeError(
                "output_differentiability=False for inplace operation (version_counter won't get updated)"
            )
        for differentiable, output in zip(output_differentiability, outputs):
            if differentiable:
                differentiable_outputs.append(output)
        return differentiable_outputs
    candidate_differentiable_outputs = list(
        filter(lambda r: is_differentiable(r.name, r.type, info), outputs)
    )
    if uses_single_grad(info):
        return candidate_differentiable_outputs[:1]
    else:
        return candidate_differentiable_outputs
<|endoftext|>from __future__ import annotations

from typing import Any

from torchgen.api.types import (
    BaseCppType,
    BaseCType,
    boolT,
    CType,
    deviceT,
    doubleT,
    generatorT,
    layoutT,
    ListCType,
    longT,
    memoryFormatT,
    NamedCType,
    OptionalCType,
    scalarT,
    scalarTypeT,
    stringT,
    SymIntT,
    VectorCType,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    OperatorName,
    OptionalType,
    Return,
    TensorOptionsArguments,
    Type,
)


_valueT: BaseCppType | None = None


# A ValueT is an IR type which represents the computation of a Tensor.  In other
# words, a PyTorch user will do operations on lazy tensors, and each output lazy
# tensor internally tracks a ValueT representing the IR node that would have
# actually produced the value of this tensor for real.
#
# This is configurable because different lazy tensor backends (LTC vs XLA) will
# have different IR representations.  (Though, arguably, after unification they
# shouldn't!)
def getValueT() -> BaseCppType:
    global _valueT
    if not _valueT:
        raise NotImplementedError(
            "The value type needs to be set with setValueT() in run_gen_lazy_tensor()"
        )

    return _valueT


def setValueT(val: BaseCppType) -> None:
    global _valueT
    _valueT = val


# this is a bad hack. I need to refactor the data model to represent each arg in the schema as an object,
# making it easier to represent special properties of an arg.
tensorListValueT = BaseCppType("torch::lazy", "Value")


def process_ir_type(
    typ: Type, properties: LazyIrProperties, *, symint: bool
) -> BaseCType | VectorCType | OptionalCType | ListCType:
    """
    This function takes a type from NativeFunctions and converts it for use with
    lazy tensor codegen.

    Type conversion for lazy currently consists of
     (1) changing at::Tensors into lazy::Values
     (2) wrapping everything in a BaseCType
     (3) making cpp-reference types into cpp-value types (e.g. vector instead of IntArrayRef)

    (1) converts at::Tensors to lazy::Values (which wrap lazy::Nodes, with which Lazy IR represents tensors.)
    There is special handling for Optional[Tensor] or list[Tensor], etc- hence 'tensor-like'

    This is incomplete- there are assertions in places that it's expected to need to add
    more types as the codegen is used with more operators.
    """
    if isinstance(typ, BaseType):
        if typ.name == BaseTy.Tensor:
            return BaseCType(getValueT())
        elif typ.name == BaseTy.Scalar:
            if properties.TreatScalarsAsConstants:
                return BaseCType(scalarT)
            # at::scalar has special handling,
            # and is wrapped in an lazy::Value just like at::tensor
            return BaseCType(getValueT())
        elif typ.name == BaseTy.ScalarType:
            return BaseCType(scalarTypeT)
        elif typ.name == BaseTy.int:
            return BaseCType(longT)
        elif typ.name == BaseTy.SymInt:
            if symint:
                return BaseCType(getValueT())
            else:
                return BaseCType(longT)
        elif typ.name == BaseTy.bool:
            return BaseCType(boolT)
        elif typ.name == BaseTy.float:
            return BaseCType(doubleT)
        elif typ.name == BaseTy.str:
            return BaseCType(stringT)
        elif typ.name == BaseTy.Device:
            return BaseCType(deviceT)
        elif typ.name == BaseTy.Generator:
            return BaseCType(generatorT)
        elif typ.name == BaseTy.Layout:
            return BaseCType(layoutT)
        elif typ.name == BaseTy.MemoryFormat:
            return BaseCType(memoryFormatT)
        else:
            raise AssertionError(f"TODO add support for type {repr(typ)}")
    elif isinstance(typ, OptionalType):
        return OptionalCType(process_ir_type(typ.elem, properties, symint=symint))
    elif isinstance(typ, ListType):
        if str(typ.elem) == "Tensor?":
            # TODO(whc) is this actually correct? or should it use a Vector like above
            return ListCType(OptionalCType(BaseCType(getValueT())))
        elif str(typ.elem) == "Tensor":
            # this is a TensorList which comes in from GetTensorList as a Value
            return BaseCType(tensorListValueT)
        elif typ.elem == BaseType(BaseTy.SymInt):
            # TODO: return a value type.  The problem here is analogous to
            # the problem with tensorListValueT: if you have SymInt[] you
            # cannot conveniently save the list of Value directly, as nodes
            # expect to save values as a vector for ALL arguments.  So you
            # need a separate IR node that represents all of the size nodes
            # assembled into a list.  I'm not an LTC dev so I don't want to
            # figure it out right now.  Y'all figure it out...
            return VectorCType(BaseCType(longT))

        else:
            return VectorCType(process_ir_type(typ.elem, properties, symint=symint))
    else:
        raise AssertionError(f"unrecognized type {repr(typ)}")


# TODO: Determining this based off of CType is bad; this should be computed
# from Type directly; then the same logic as process_ir_type can be used
#
# Invariant: passed typ should be an *owning* CType (e.g., we will report
# that ArrayRef<Value> is NOT a value type)
def isValueType(typ: CType, properties: LazyIrProperties | None = None) -> bool:
    """
    Given a type, determine if it is a Value-like type.  This is equivalent to
    being Tensor-like, but assumes the type has already been transformed.
    """
    if isinstance(typ, BaseCType):
        # I am regretting my naming conventions, but now we are wrapping at::scalar in
        # lazy value, while preserving other 'scalar' types as scalars in the IR
        treat_scalars_as_constants = properties and properties.TreatScalarsAsConstants
        return (
            typ.type == getValueT()
            or (typ.type == scalarT and not treat_scalars_as_constants)
            or typ.type == SymIntT
        )
    elif typ == VectorCType(BaseCType(SymIntT)):
        # TODO: report True for this
        return False
    elif isinstance(typ, (OptionalCType, ListCType, VectorCType)):
        return isValueType(typ.elem, properties)
    return False


def isSymIntType(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.SymInt


def isWrappedScalarType(typ: Type) -> bool:
    """
    Given a type, determine if it is a c10::scalar which we will wrap in a lazy Value.
    Since we literally change the type from scalarT to valueT, information is lost.
    This function helps build a list of wrapped scalars to save that information
    """
    if isinstance(typ, BaseType):
        # I am regretting my naming conventions, but now we are wrapping at::scalar in
        # lazy value, while preserving other 'scalar' types as scalars in the IR
        return typ.name == BaseTy.Scalar
    elif isinstance(typ, (OptionalType, ListType)):
        return isWrappedScalarType(typ.elem)
    return False


# TODO: dedupe with Type.is_generator_like
def isGeneratorType(typ: Type) -> bool:
    if isinstance(typ, BaseType):
        return typ.name == BaseTy.Generator
    elif isinstance(typ, (OptionalType)):
        return isGeneratorType(typ.elem)
    return False


# This class caches a few derived properties computed from an Argument
# and LazyIrProperties
class LazyArgument:
    name: str
    orig_type: Type
    lazy_type_: CType | None
    is_wrapped_scalar: bool
    is_generator: bool
    # TODO: this is lies, it is false for symint list
    is_symint_or_list: bool

    # Whether or not we are treating this as symint or not
    symint: bool

    # true if this argument is or contains a lazy IR value
    is_lazy_value: bool

    def __init__(
        self, arg: Argument, properties: LazyIrProperties, *, symint: bool
    ) -> None:
        self.name = arg.name
        self.orig_type = arg.type
        self.symint = symint
        self.is_optional = isinstance(arg.type, OptionalType)
        self.is_generator = isGeneratorType(arg.type)
        self.lazy_type_ = process_ir_type(arg.type, properties, symint=symint)
        self.is_wrapped_scalar = isWrappedScalarType(arg.type)
        self.is_symint_or_list = symint and (
            isSymIntType(arg.type)
            or (isinstance(arg.type, OptionalType) and isSymIntType(arg.type.elem))
            # TODO: lists of symints are not currently treated as value types
            # or (isinstance(arg.type, ListType) and isSymIntType(arg.type.elem))
        )

        self.is_lazy_value = isValueType(self.lazy_type, properties)

    @property
    def lazy_type(self) -> CType:
        if self.lazy_type_ is None:
            raise AssertionError(
                f"Attempted to access lazy_type for invalid argument {self.name}"
            )
        return self.lazy_type_


class LazyIrProperties:
    """Collection of properties for an IR node

    The property groups are listed below. Each group is mutually
    exclusive, meaning that only one property from each group can be True
    at any one time. The properties can be accessed as if they were normal
    attributes. The mutual exclusivity is automatically handled.
    """

    Properties: tuple[tuple[str, ...], ...] = (
        (
            "ShapePrecompute",  # Assume shape has been precomputed
            "ShapeCompute",  # Need to compute the shape on construction
            "ShapeCache",  # Utilize the shape cache to defer computation
        ),
        (
            "Lower",  # Codegen full lower function
            "LowerDeclOnly",  # Codegen only lower function declaration
        ),
        (
            "CanBeReused",  # Codegen full reuse function
            "CanBeReusedDeclOnly",  # Codegen only reuse function declaration
        ),
        (
            "CreateFn",  # Codegen full create function
            "CreateFnDeclOnly",  # Codegen only create function declaration
        ),
        (
            "TreatScalarsAsConstants",  # Treat Scalars as constants instead of handling like values
        ),
    )

    def __init__(self, *default_properties: str) -> None:
        properties: dict[tuple[str, ...], str | None] = dict.fromkeys(
            LazyIrProperties.Properties
        )
        self.__dict__["properties"] = properties
        for p in default_properties:
            setattr(self, p, True)

    def __getattr__(self, key: str) -> Any:
        properties = self.__dict__["properties"]
        for values in LazyIrProperties.Properties:
            if key in values:
                return properties[values] == key

        return self.__getattribute__(key)

    def __setattr__(self, key: str, value: Any) -> Any:
        properties = self.__dict__["properties"]
        for values in LazyIrProperties.Properties:
            if key in values:
                properties[values] = key if value else None
                return value

        raise KeyError(f"Invalid property: {key}")


# Inspired by a FunctionSchema object, a LazyIrSchema holds the schema of a Lazy IR node.
# Unlike a FunctionSchema, it has no round-trippable string form (relating to the YAML),
# but carries type information from a native FunctionSchema modified for use with IR nodes,
# and preserving original argument names.
#
# TODO: This is not idiomatic with how other torchgen APIs transform on schema.
class LazyIrSchema:
    # The name of the operator this function schema describes.
    name: OperatorName

    positional_args: tuple[LazyArgument, ...]
    keyword_args: tuple[LazyArgument, ...]

    # TODO: Need to handle collisions with argument names at some point
    returns: tuple[Return, ...]

    # if this schema has a Generator arg, list its orig ctype/name but don't
    # build a LazyArgument since lazy IR doesn't support it
    generator_arg: NamedCType | None = None

    # original function schema
    func: FunctionSchema

    # Whether or not we are code-genning for SymInt or not
    symint: bool

    properties: LazyIrProperties = LazyIrProperties(
        # default properties
        "ShapePrecompute",
        "Lower",
        "CanBeReused",
    )
    opkind: str | None = None

    def __init__(
        self,
        func: FunctionSchema,
        properties: LazyIrProperties | None = None,
        *,
        symint: bool,
    ) -> None:
        if properties:
            self.properties = properties

        self.func = func
        self.symint = symint
        positional_args: list[LazyArgument] = []
        for arg_field in ["pre_self_positional", "self_arg", "post_self_positional"]:
            if arg_field == "self_arg" and func.arguments.self_arg is not None:
                arg = func.arguments.self_arg.argument
                positional_args.append(
                    LazyArgument(arg, self.properties, symint=symint)
                )
            elif getattr(func.arguments, arg_field) is not None:
                positional_args.extend(
                    LazyArgument(arg, self.properties, symint=symint)
                    for arg in getattr(func.arguments, arg_field)
                )
        self.positional_args = tuple(positional_args)

        keyword_args: list[LazyArgument] = []
        for arg_field in [
            "pre_tensor_options_kwarg_only",
            "tensor_options",
            "post_tensor_options_kwarg_only",
            "out",
        ]:
            curr_args = getattr(func.arguments, arg_field)
            if curr_args is not None:
                if isinstance(curr_args, TensorOptionsArguments):
                    curr_args = curr_args.all()
                for arg in curr_args:
                    if isGeneratorType(arg.type):
                        if self.generator_arg is not None:
                            raise AssertionError(
                                "We expect there is only one generator arg"
                            )
                        self.generator_arg = NamedCType(
                            arg.name,
                            arg.type,  # type:ignore[arg-type]
                        )
                keyword_args.extend(
                    LazyArgument(arg, self.properties, symint=symint)
                    for arg in curr_args
                )
        self.keyword_args = tuple(keyword_args)
        self.name = func.name
        self.returns = func.returns

    @property
    def node_name(self) -> str:
        """
        Return camel-case version of op in node.

        Note: This function also appends any `overload_name` in the operation.
        For example, if the op is `bitwise_and.Tensor`, the returned name
        will be `BitwiseAndTensor`.
        """
        op_name = f"{self.name.name}_{self.name.overload_name}".lower()
        return "".join(word.capitalize() or "" for word in op_name.split("_"))

    @property
    def aten_name(self) -> str:
        return str(self.name.name)

    @property
    def base_name(self) -> str:
        return f"{self.name.name.base}"

    def filtered_args(
        self,
        positional: bool = True,
        keyword: bool = True,
        values: bool = True,
        scalars: bool = True,
        generator: bool = True,
    ) -> list[LazyArgument]:
        # This function maintains the sorted order of arguments but provides different filtered views.
        # Some parts of the code care about kwargs vs args (TS lowerings),
        # other parts care about whether they need to wrap the arg in a lazy value or leave it alone.
        # Generators are special cased, as they are needed for fallback/shape-inference but not supported
        # in TS lowerings and therefore also omitted from lazy IR.
        args: list[LazyArgument] = []
        if positional:
            args.extend(self.positional_args)
        if keyword:
            args.extend(self.keyword_args)

        if values and scalars and generator:
            return args
        elif values and scalars:
            return [a for a in args if not a.is_generator]
        elif values:
            return [a for a in args if a.is_lazy_value]
        elif scalars:
            return [
                a
                for a in args
                if not a.is_lazy_value and (generator or not a.is_generator)
            ]

        return []

    @property
    def positional_values(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=True, keyword=False, values=True, scalars=False
        )

    @property
    def positional_scalars(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=True, keyword=False, values=False, scalars=True
        )

    @property
    def keyword_values(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=False, keyword=True, values=True, scalars=False
        )

    @property
    def keyword_scalars(self) -> list[LazyArgument]:
        return self.filtered_args(
            positional=False, keyword=True, values=False, scalars=True
        )
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    OptionalType,
    Return,
    Type,
    Variant,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           Data Models
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# [Notes] python binding codegen
#
# The Python binding codegen produces code that takes the input list of
# PyObjects, finds the matching ATen C++ function using PythonArgParser,
# converts the PyObjects into C++ types and calls the ATen C++ function:
#
# +--------+  parsing   +------------------------+  binding   +-----------------------+
# | PyObjs | ---------> | PythonArgParser Output | ---------> | Cpp Function Dispatch |
# +--------+            +------------------------+            +-----------------------+
#
# The following examples demonstrate the data models the Python binding
# codegen needs to deal with and the tasks it needs to accomplish. It
# helps understand the purpose of the new data types we introduced below.
#
#  - Function Schema (source of truth)
#
#      aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None,
#                  Device? device=None, bool? pin_memory=None) -> Tensor
#
#  - Python Signature
#
#    It's used to generate input schema string for PythonArgParser.
#    Note: TensorOptions fields are reordered and the additional
#    'requires_grad' field is added:
#
#      zeros(IntArrayRef size, *, ScalarType dtype=None,
#            Layout layout=torch.strided, Device device=None,
#            bool pin_memory=False, bool requires_grad=False)
#
#  - C++ Signature
#
#    It's used to generate C++ lambda formals & dispatch call.
#    Note: the scattered TensorOptions fields are packed into 'options'.
#
#      auto dispatch_zeros =
#          [](IntArrayRef size, const TensorOptions & options) -> Tensor {
#          pybind11::gil_scoped_release no_gil;
#          return torch::zeros(size, options);
#      };
#
#  - Binding between Python Arguments and C++ Arguments
#
#    Given a set of Python Arguments in scope, we need produce the
#    binding expressions that translate the Python API into C++ API:
#
#            Python Args               Cpp Args       Binding Exprs
#     -----------------------------------------------------------------
#         0: size                      size           '_r.intlist(0)'
#         1: dtype         -----+
#         2: layout            +--> options        'options' [special packing]
#         3: device           /
#         4: pin_memory      /
#         5: requires_grad -+
#
#    So the full dispatch expression would look like:
#
#      dispatch_zeros(_r.intlist(0), options)
#
#    Where does 'options' come from? It involves special local init
#    for TensorOptions. Note that Python side has the additional
#    'requires_grad' field:
#
#      const auto options = TensorOptions()
#          .dtype(_r.scalartype(1))
#          .device(_r.device(3))
#          .layout(_r.layoutOptional(2))
#          .requires_grad(_r.toBool(5))
#          .pinned_memory(_r.toBool(4));
#
#    In some other cases one Python Argument can map to multiple C++
#    Arguments. For example:
#
#     aten::max.dim(Tensor self, int dim, bool keepdim=False)
#       -> (Tensor values, Tensor indices)
#
#            Python Args               Cpp Args          Binding Exprs
#     ---------------------------------------------------------------------
#                               +----> max               'out[0]'
#                              /-----> max_values        'out[1]
#         0: input            /        self              '_r.tensor(0)'
#         1: dim             /         dim               '_r.toInt64(1)'
#         2: keepdim        /          keepdim           '_r.toBool(2)'
#         3: out      -----+           [local init] out  '_r.tensorlist_n<2>(3)'
#
#    As demonstrated above, the binding can involve reordering,
#    packing, unpacking and special local inits.
#
#
#  Let's look at a concrete example:
#
#      static PythonArgParser parser({
#        "abs(Tensor input, *, Tensor out=None)",
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ^
#         +--- Python Schema, represented by PythonSignature and PythonArgument
#
#      }, /*traceable=*/true);
#
#      ParsedArgs<2> parsed_args;
#      auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
#
#      ...
#
#      if (_r.isNone(1)) {
#          ~~~~~~~~~~~~  <--- Scattered PythonArgParser output (arg name = 'out')
#                             represented by PythonArgParserOutputExpr
#
#        // aten::abs(Tensor self) -> Tensor
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ^
#         +--- NativeFunction schema, base version
#
#        auto dispatch_abs = [](const Tensor & self) -> Tensor {
#                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                             ^
#                             +--- dispatch_lambda_args / dispatch_lambda_return_str
#                                  generated from NativeFunction / CppSignature
#                                  (deprecated PythonSignature is special)
#                                  arguments are represented by DispatchLambdaArgument
#
#          pybind11::gil_scoped_release no_gil;
#          return self.abs();
#                 ~~~~~~~~~~~  <--- cpp_dispatch_target / cpp_dispatch_exprs
#                                   generated from NativeFunction / CppSignature
#        };
#        return wrap(dispatch_abs(_r.tensor(0)));
#                                 ~~~~~~~~~~~~~
#                                  ^
#                                  +--- dispatch_lambda_exprs
#                                       binding PythonArgParserOutputExpr (python args)
#                                       and DispatchLambdaArgument (c++ args)
#
#      } else {
#        // aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
#        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         ^
#         +--- NativeFunction schema, out-variant
#
#        auto dispatch_abs_out = [](Tensor out, const Tensor & self) -> Tensor {
#          pybind11::gil_scoped_release no_gil;
#          return at::abs_out(out, self);
#        };
#        return wrap(dispatch_abs_out(_r.tensor(1), _r.tensor(0)));
#      }
#
#
# [Notes] python interface codegen
# The python dataclasses below are used used to generate both python binding code
# and pyi type hint signatures.
# In theory these two should look very similar, but there are number of differences
# in how pyi signatures vs. python_arg_parser signatures are generated.
# These differences have been encapsulated in signature_str() vs. signature_str_pyi()
# to display the full signatures, and argument_str() vs argument_str_pyi() to display arguments.
# For examples, only pyi signatures include return types.


def format_function_signature(
    name: str, arguments: Iterable[str] = (), return_type: str | None = None
) -> str:
    if not isinstance(arguments, (list, tuple)):
        arguments = tuple(arguments)
    return_type = f" -> {return_type}" if return_type is not None else ""

    sig = f"def {name}({', '.join(arguments)}){return_type}: ..."
    if len(sig) <= 80 or len(arguments) == 0 or tuple(arguments) == ("self",):
        return sig

    lines = [
        f"def {name}(",
        *(f"    {arg}," for arg in arguments),
        f"){return_type}: ...",
    ]
    sig = "\n".join(lines)
    if all(len(line) <= 80 for line in lines):
        return sig
    # ruff format bug for compound statements: https://github.com/astral-sh/ruff/issues/18658
    # use `skip` instead of `on` + `off`
    return sig.removesuffix(" ...") + "  # fmt: skip\n    ..."


@dataclass(frozen=True)
class PythonReturns:
    returns: tuple[Return, ...]


@dataclass(frozen=True)
class PythonArgument:
    name: str
    type: Type
    default: str | None

    # Used to generate the default init expr for some PythonArgParser outputs, e.g.:
    #
    #   _r.layoutWithDefault(3, layout_from_backend(self.options().backend())))
    #                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                            ^
    #                            +--- default_init str
    default_init: str | None

    # Compute argument formal for python argument parsing.
    # Needs to be consistent with torch/csrc/utils/python_arg_parser.h.
    def argument_str(self, *, method: bool = False, symint: bool = True) -> str:
        type_str = (
            argument_type_str(self.type, symint=symint)
            .replace("const ", "")
            .replace(" &", "")
        )

        name = self.name
        # s/self/input/ outside method bindings
        # [old codegen] TODO: remove this? doesn't rename in codegen, it's just
        # for the parse string
        if name == "self" and type_str in ["Tensor", "Number"] and not method:
            name = "input"

        # add default
        if self.default is not None:
            default = {
                "nullptr": "None",
                "::std::nullopt": "None",
                "std::nullopt": "None",
                "{}": "None",
            }.get(self.default, self.default)
            return f"{type_str} {name}={default}"
        else:
            return f"{type_str} {name}"

    def argument_str_pyi(
        self, *, method: bool = False, deprecated: bool = False
    ) -> str:
        type_str = argument_type_str_pyi(self.type)

        name = self.name
        # s/self/input/ outside method bindings
        # [old codegen] TODO: remove this? doesn't rename in codegen, it's just
        # for the parse string
        if name == "self" and type_str == "Tensor" and not method and not deprecated:
            name = "input"

        if name == "from":  # from is a Python keyword...
            name += "_"

        # pyi merges the _out and functional variants into the same signature, with an optional out arg
        if name == "out" and not deprecated:
            type_str = _append_optional_pyi(type_str)

        # pyi deprecated signatures don't get defaults for their out arg
        treat_as_no_default = (
            deprecated
            and isinstance(self, PythonOutArgument)
            and self.default == "None"
        )

        # add default
        if self.default is not None and not treat_as_no_default:
            if (
                isinstance(self.type, ListType)
                and self.type.elem == BaseType(BaseTy.int)
                and self.default.startswith("{")
                and self.default.endswith("}")
            ):
                default = (
                    "(" + ", ".join(map(str.strip, self.default[1:-1].split(","))) + ")"
                )
            else:
                default = {
                    "nullptr": "None",
                    "::std::nullopt": "None",
                    "std::nullopt": "None",
                    "{}": "None",
                    "c10::MemoryFormat::Contiguous": "contiguous_format",
                    "QScheme::PER_TENSOR_AFFINE": "per_tensor_affine",
                }.get(self.default, self.default)
            return f"{name}: {type_str} = {default}"
        else:
            return f"{name}: {type_str}"


@dataclass(frozen=True)
class PythonOutArgument(PythonArgument):
    # In Python signature multiple output fields are packed into one 'out' argument.
    # When binding to C++, it's first binded to a local 'out' variable:
    #   'auto out = _r.tensorlist_n<2>(2);',
    # then binded to scattered C++ output arguments as 'out[0]', 'out[1]', and etc.
    # TODO: maybe don't need keep scattered out fields for python signature?
    outputs: tuple[PythonArgument, ...]

    @staticmethod
    def from_outputs(outputs: tuple[PythonArgument, ...]) -> PythonOutArgument | None:
        if not outputs:
            return None

        size = len(outputs)
        if size == 1:
            return PythonOutArgument(
                name=outputs[0].name,
                type=outputs[0].type,
                default="None",
                default_init=None,
                outputs=outputs,
            )
        elif size > 1:
            if any(not a.type.is_tensor_like() for a in outputs):
                raise RuntimeError(f"Unsupported output type: {outputs}")
            return PythonOutArgument(
                name="out",
                # TODO: shouldn't this be OptionalType[ListType[...]], since it defaults to None?
                type=ListType(BaseType(BaseTy.Tensor), size),
                default="None",
                default_init=None,
                outputs=outputs,
            )
        raise AssertionError(r"Unexpected PythonOutArgument size")


@dataclass(frozen=True)
class PythonSignature:
    # Base operator name, without inplace/outplace suffix.
    name: str

    # Positional arguments.
    # TODO: create a dedicated SelfArgument type for 'self'?
    input_args: tuple[PythonArgument, ...]

    # Keyword arguments excluding the 'out' argument and scattered kwargs belonging
    # to TensorOptions (dtype, layout, device, pin_memory, requires_grad, etc).
    input_kwargs: tuple[PythonArgument, ...]

    output_args: PythonOutArgument | None

    # Return types, which are only used by pyi
    returns: PythonReturns

    # These are scattered kwargs arguments belonging to TensorOptions.
    # When binding to C++, they are packed into a TensorOptions object 'options'.
    # It's possible that the C++ signature doesn't take TensorOptions object (e.g.
    # for out variant), in which case they will be used as scattered fields without
    # being packed into 'options'.
    # TODO: maybe create a PythonTensorOptionsArgument?
    tensor_options_args: tuple[PythonArgument, ...]

    # method or function signature?
    method: bool

    @property
    def deprecated(self) -> bool:
        return False

    def arguments(
        self, *, skip_outputs: bool = False, skip_tensor_options: bool = False
    ) -> tuple[PythonArgument | PythonOutArgument, ...]:
        result: list[PythonArgument | PythonOutArgument] = []
        result.extend(self.input_args)
        result.extend(self.input_kwargs)
        if self.output_args is not None and not skip_outputs:
            result.append(self.output_args)
        if not skip_tensor_options:
            result.extend(self.tensor_options_args)
        return tuple(result)

    def arguments_count(self) -> int:
        return len(self.arguments())

    def output_idx(self) -> int:
        return len(self.input_args) + len(self.input_kwargs)

    # [old codegen] Compute the Python function signature for argument parsing,
    # as specified in torch/csrc/utils/python_arg_parser.h.  WARNING:
    # this is NOT the same type signature as specified by PEP 484
    # as understood by mypy; our format was independently developed
    # and has some quirks to make it more suitable specifically
    # for error parsing.
    #
    # For a translation to mypy-valid type signatures, see
    # signature_str_pyi().
    def signature_str(self, *, skip_outputs: bool = False, symint: bool = True) -> str:
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: list[str] = [
            a.argument_str(method=self.method, symint=symint) for a in args
        ]
        positional_argc = len(self.input_args)
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, "*")

        return f"{self.name}({', '.join(schema_formals)})"

    def signature_str_pyi(self, *, skip_outputs: bool = False) -> str:
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: list[str] = [
            a.argument_str_pyi(method=self.method) for a in args
        ]
        positional_argc = len(self.input_args)
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, "*")

        # only pyi signatures include returns
        returns_str = returns_str_pyi(self)
        # pyi also includes self (with no typing/defaults) for methods
        if self.method:
            schema_formals.insert(0, "self")
        return format_function_signature(self.name, schema_formals, returns_str)

    def signature_str_pyi_vararg(self, *, skip_outputs: bool = False) -> str | None:
        # only pyi uses vararg signatures
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: list[str] = [
            a.argument_str_pyi(method=self.method) for a in args
        ]
        # vararg only applies to pyi signatures. vararg variants are not generated for all signatures
        num_args = self.arguments_count()
        if num_args == 0:
            return None

        num_positionalargs = len(self.input_args)

        vararg_type = args[0].type
        if not (
            isinstance(vararg_type, ListType)
            and str(vararg_type.elem) in ["int", "SymInt"]
            and num_positionalargs == 1
        ):
            return None

        # Below are the major changes in vararg vs. regular pyi signatures
        # vararg signatures also omit the asterix
        schema_formals[0] = (
            "*" + args[0].name + ": " + argument_type_str_pyi(vararg_type.elem)
        )

        returns_str = returns_str_pyi(self)
        # pyi also includes self (with no typing/defaults) for methods
        if self.method:
            schema_formals.insert(0, "self")
        return format_function_signature(self.name, schema_formals, returns_str)


# The deprecated python signature involves some special logic, so create a
# dedicated data model to store these extra properties.
@dataclass(frozen=True)
class PythonSignatureDeprecated(PythonSignature):
    # Schema for the deprecated function
    deprecated_schema: FunctionSchema

    # The deprecated signature might miss some arguments that the corresponding
    # C++ signature expects. We need store the constant default values to pass in.
    # For example:
    #   [deprecate signature]: addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2)
    #   [func schema]: aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    #   [func call]: self.addmm(mat1, mat2, beta, 1)
    # We store ['self', 'mat1', 'mat2', 'beta', '1'] in this case.
    deprecated_args_exprs: tuple[str, ...]

    @property
    def deprecated(self) -> bool:
        return True

    def signature_str(self, *, skip_outputs: bool = False, symint: bool = True) -> str:
        return (
            PythonSignature.signature_str(
                self, skip_outputs=skip_outputs, symint=symint
            )
            + "|deprecated"
        )

    def signature_str_pyi(self, *, skip_outputs: bool = False) -> str:
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: list[str] = [
            a.argument_str_pyi(method=self.method, deprecated=True) for a in args
        ]
        positional_argc = len(self.input_args)
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, "*")

        returns_str = returns_str_pyi(self)
        return format_function_signature(self.name, schema_formals, returns_str)

    def signature_str_pyi_vararg(self, *, skip_outputs: bool = False) -> str | None:
        # the codegen doesn't include vararg variants for deprecated signatures
        return None


# This struct is used to hold the PythonSignature and its corresponding
# NativeFunction BEFORE grouping base and out-variant functions.
# Why not store NativeFunction in PythonSignature or construct PythonSignature
# from NativeFunction? Because they are not 1-1 mapped.
# One native function could have both deprecated and non-deprecated python
# signatures - NativeFunction doesn't contain information to construct the
# deprecated python signature.
# One python signature is used to handle both the base and the out-variant
# function - see 'PythonSignatureGroup'.
@dataclass(frozen=True)
class PythonSignatureNativeFunctionPair:
    signature: PythonSignature
    function: NativeFunction


# We merge pairs of functions with signatures that are equivalent mod
# output arguments, and use a single entry in the python_arg_parser sig
# list for both (output arguments become optional).
@dataclass(frozen=True)
class PythonSignatureGroup:
    # The signature used for Python argument parsing. The outplace signature
    # is preferred if exists, because it can be used to parse inputs for both
    # the out-place variant and the base version (with output omitted).
    signature: PythonSignature

    # The regular ATen declaration (e.g. conv2d)
    base: NativeFunction

    # The out variant (e.g. conv2d_out)
    outplace: NativeFunction | None

    @classmethod
    def from_pairs(
        cls,
        functional: PythonSignatureNativeFunctionPair,
        out: PythonSignatureNativeFunctionPair | None,
    ) -> PythonSignatureGroup:
        if out is None:
            return PythonSignatureGroup(
                signature=functional.signature,
                base=functional.function,
                outplace=None,
            )

        # prefer the signature with optional out=... arguments because it's the
        # superset that can be used to parse input for both base and outplace.
        signature_kwargs = out.signature.__dict__.copy()

        # Out overloads in C++ don't have TensorOptions arguments,
        # so take these from the functional variant
        signature_kwargs["tensor_options_args"] = (
            functional.signature.tensor_options_args
        )

        return PythonSignatureGroup(
            signature=type(out.signature)(**signature_kwargs),
            base=functional.function,
            outplace=out.function,
        )


# C++ function dispatch is wrapped in a lambda function. The lambda function
# has almost the same signature as the C++ function, only with some small
# variants - see details below.
# This data model is used to represent arguments of the lambda function
# signature.
@dataclass(frozen=True)
class DispatchLambdaArgument:
    name: str
    type_str: str
    is_out_arg: bool


# To pass PyObjects arguments to C++ function (via the lambda wrapper),
# we need first convert PyObjects into simple C++ objects. This work
# is done by PythonArgParser.
# This data model is used to represent the output of PythonArgParser.
# It has 1-1 mapping with PythonArgument in PythonSignature.
@dataclass(frozen=True)
class PythonArgParserOutputExpr:
    # argument name
    name: str

    # RHS expression to reference PythonArgParser output.
    expr: str

    # In some special cases we need create different expr, e.g.:
    # '_r.isNone(1)' instead of '_r.tensor(1)'.
    index: int

    # The python argument it maps to.
    argument: PythonArgument

    @property
    def is_none_expr(self) -> str:
        return f"_r.isNone({self.index})"


# To pass PythonArgParser output to the lambda wrapper, we need bind
# PythonArgParserOutputExpr to DispatchLambdaArgument.
# They are not always 1-1 mapped, e.g. scattered TensorOptions fields
# need be packed into a TensorOptions object, which is the argument
# that the lambda function wrapper takes.
@dataclass(frozen=True)
class DispatchLambdaArgumentExprs:
    # The exprs that provide the binding for lambda arguments, e.g.:
    #
    #   'self' -> '_r.tensor(0)'
    #   'min' -> 'out[0]' / 'min_indices' -> 'out[1]'
    #   'options' -> 'options'
    #
    # It has 1-1 mapping with DispatchLambdaArgument.
    exprs: Sequence[str]

    # Special local inits, which might introduce new variables that
    # the 'exprs' above reference, e.g.:
    #
    #   'auto out = _r.tensorlist_n<2>(2);'
    #
    inits: Sequence[str]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                          Helper Functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _cpp_signature(f: NativeFunction, *, method: bool = False) -> CppSignature:
    return CppSignatureGroup.from_native_function(f, method=method).signature


def has_tensor_options(f: NativeFunction) -> bool:
    return f.func.arguments.tensor_options is not None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                          Python Signature
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# 'simple_type' was introduced by the old codegen, which is slightly
# different from the python schema type, e.g.: doesn't have '?' suffix
# for optional Tensor/TensorList; doesn't have '[size]' suffix for list type.
def argument_type_str(
    t: Type, *, simple_type: bool = False, symint: bool = True
) -> str:
    if isinstance(t, BaseType):
        if t.name == BaseTy.int:
            return "int64_t"
        elif t.name == BaseTy.float:
            return "double"
        elif t.name == BaseTy.str:
            return "c10::string_view"
        elif t.name in [
            BaseTy.Tensor,
            BaseTy.bool,
            BaseTy.QScheme,
            BaseTy.Scalar,
            BaseTy.ScalarType,
            BaseTy.Generator,
            BaseTy.Storage,
            BaseTy.Layout,
            BaseTy.Device,
            BaseTy.DeviceIndex,
            BaseTy.MemoryFormat,
            BaseTy.Stream,
            BaseTy.SymInt,
        ]:
            # These python schema type names line up with their function schema names
            return t.name.name

    elif isinstance(t, OptionalType):
        elem = argument_type_str(t.elem, simple_type=simple_type, symint=symint)
        return f"{elem}?"
    elif isinstance(t, ListType):
        size = t.size if not simple_type else None
        if str(t.elem) == "bool":
            if t.size is None:
                raise AssertionError("bool ListType must have a size")
            return f"::std::array<bool,{t.size}>"
        elif str(t.elem) == "int":
            return f"IntArrayRef[{size}]" if size is not None else "IntArrayRef"
        elif str(t.elem) == "SymInt":
            if symint:
                return (
                    f"SymIntArrayRef[{size}]" if size is not None else "SymIntArrayRef"
                )
            else:
                return f"IntArrayRef[{size}]" if size is not None else "IntArrayRef"
        elif str(t.elem) == "Tensor":
            return f"TensorList[{size}]" if size is not None else "TensorList"
        elif str(t.elem) == "Scalar":
            return f"ScalarList[{size}]" if size is not None else "ScalarList"
        elif str(t.elem) == "Tensor?":
            if simple_type:
                return "c10::List<::std::optional<Tensor>>"
            else:
                return "const c10::List<::std::optional<Tensor>> &"
        elem = argument_type_str(t.elem, simple_type=simple_type, symint=symint)
        return f"ArrayRef<{elem}>"

    raise RuntimeError(f"unrecognized type {repr(t)}")


def argument_type_size(t: Type) -> int | None:
    l = t.is_list_like()
    if l is not None and str(l.elem) != "bool":
        return l.size
    else:
        return None


def argument(a: Argument) -> PythonArgument:
    return PythonArgument(
        name=a.name,
        type=a.type,
        # TODO: directly translate a.default to python default
        default=(
            str(pythonify_default(cpp.default_expr(a.default, a.type, symint=False)))
            if a.default is not None
            else None
        ),
        default_init=None,
    )


# Generates a PythonSignature that can be used for either .pyi or PythonArgParser codegen
def signature(
    f: NativeFunction, *, method: bool = False, pyi: bool = False
) -> PythonSignature:
    return signature_from_schema(
        f.func, category_override=f.category_override, method=method, pyi=pyi
    )


def signature_from_schema(
    func: FunctionSchema,
    *,
    category_override: str | None,
    method: bool = False,
    pyi: bool = False,
) -> PythonSignature:
    args: list[Argument] = []
    args.extend(func.arguments.pre_self_positional)
    # Skip SelfArgument if this is method.
    if not method and func.arguments.self_arg is not None:
        args.append(func.arguments.self_arg.argument)
    args.extend(func.arguments.post_self_positional)
    args.extend(func.arguments.pre_tensor_options_kwarg_only)
    # Skip TensorOptionsArguments. Python side TensorOptions
    # arguments are created based on different rules - see below.
    args.extend(func.arguments.post_tensor_options_kwarg_only)
    args.extend(func.arguments.out)

    input_arg_set = {a.name for a in func.arguments.flat_positional}
    kwarg_only_set = {a.name for a in func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in func.arguments.out}

    input_args = tuple(map(argument, filter(lambda a: a.name in input_arg_set, args)))
    input_kwargs = tuple(
        map(argument, filter(lambda a: a.name in kwarg_only_set, args))
    )
    outputs = tuple(map(argument, filter(lambda a: a.name in out_arg_set, args)))

    # Reintroduce the scattered fields of TensorOptions for Python.
    # Compared to the cpp counterpart, the python arguments have new property
    # (default_init) and a new argument 'requires_grad', which require some
    # special handlings.
    # [old codegen] TODO: because these aren't guaranteed to be 100% faithful
    # to the original versions in the yaml, this recreation is a potential
    # source of drift between eager and JIT. Pull this logic out to a shared place.

    has_tensor_input_arg = any(
        a.type.is_tensor_like() for a in func.arguments.flat_non_out
    )
    if any(a.name == "requires_grad" for a in func.schema_order_arguments()):
        raise ValueError(
            "argument named requires_grad is reserved, should not explicitly add it in the schema"
        )

    # [old codegen] this probably won't work if one of the returns is not a tensor,
    # but it will produce a compile-time error that is obvious.
    has_tensor_return = any(r.type.is_tensor_like() for r in func.returns)

    name: str = cpp.name(func)
    is_factory_function = category_override == "factory" or (
        has_tensor_return and not has_tensor_input_arg
    )
    is_like_or_new_function = (
        category_override in ("new", "like")
        or name.startswith("new_")
        or name.endswith("_like")
    )
    is_dummy_function = category_override == "dummy"

    tensor_options_args: list[PythonArgument] = []
    if (is_factory_function or is_like_or_new_function) and not is_dummy_function:

        def topt_default_init(name: str) -> str | None:
            topt_args = func.arguments.tensor_options
            if topt_args is None:
                return None
            a = getattr(topt_args, name)
            if a.default is None or a.default == "None":
                return None
            return cpp.default_expr(a.default, a.type, symint=False)

        tensor_options_args.append(
            PythonArgument(
                name="dtype",
                type=OptionalType(BaseType(BaseTy.ScalarType)),
                default="None",
                default_init=(
                    None if is_like_or_new_function else topt_default_init("dtype")
                ),
            )
        )
        tensor_options_args.append(
            PythonArgument(
                name="layout",
                type=OptionalType(BaseType(BaseTy.Layout)),
                default="None",
                default_init=(
                    None if is_like_or_new_function else topt_default_init("layout")
                ),
            )
        )
        tensor_options_args.append(
            PythonArgument(
                name="device",
                type=OptionalType(BaseType(BaseTy.Device)),
                default="None",
                default_init=(
                    None
                    if is_like_or_new_function
                    else (
                        topt_default_init("device")
                        or "torch::tensors::get_default_device()"
                    )
                ),
            )
        )
        tensor_options_args.append(
            PythonArgument(
                name="pin_memory",
                type=OptionalType(BaseType(BaseTy.bool)),
                default="False",
                default_init=None,
            )
        )
        tensor_options_args.append(
            PythonArgument(
                name="requires_grad",
                type=OptionalType(BaseType(BaseTy.bool)),
                default="False",
                default_init=None,
            )
        )

    returns = PythonReturns(returns=func.returns)

    return PythonSignature(
        name=str(func.name.name),
        input_args=input_args,
        input_kwargs=input_kwargs,
        output_args=PythonOutArgument.from_outputs(outputs),
        tensor_options_args=tuple(tensor_options_args),
        returns=returns,
        method=method,
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                          Python Interface
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def structseq_fieldnames(returns: tuple[Return, ...]) -> list[str]:
    if len(returns) <= 1 or all(r.name is None for r in returns):
        return []
    else:
        if any(r.name is None for r in returns):
            # When building on Windows, `PyStructSequence_UnnamedField` could not be
            # resolved by the linker for some reason, which cause error in building:
            #
            # python_nn_functions.cpp.obj : error LNK2001: unresolved external symbol
            # PyStructSequence_UnnamedField
            #
            # Thus, at this point in time, we do not support unnamed
            # fields in structseq; you must either name all fields,
            # or none of them.
            raise ValueError("Unnamed field is not supported by codegen")

        return [str(r.name) for r in returns]


def _append_optional_pyi(type_str: str) -> str:
    # Append `| None`, avoiding a doubled `| None` if the type is already optional.
    return f"{type_str} | None".replace(" | None | None", " | None")


def argument_type_str_pyi(t: Type) -> str:
    add_optional = False
    if isinstance(t, OptionalType):
        t = t.elem
        add_optional = True

    ret = ""
    if isinstance(t, BaseType):
        if t.name in [BaseTy.int, BaseTy.DeviceIndex]:
            ret = "_int"
        if t.name == BaseTy.SymInt:
            ret = "_int | SymInt"
        elif t.name == BaseTy.float:
            ret = "_float"
        elif t.name == BaseTy.str:
            ret = "str"
        elif t.name == BaseTy.Scalar:
            ret = "Number | _complex | PySymType"
        elif t.name == BaseTy.ScalarType:
            ret = "_dtype"
        elif t.name == BaseTy.bool:
            ret = "_bool"
        elif t.name == BaseTy.QScheme:
            ret = "_qscheme"
        elif t.name == BaseTy.Layout:
            ret = "_layout"
        elif t.name == BaseTy.Device:
            ret = "DeviceLikeType | None"
        elif t.name == BaseTy.MemoryFormat:
            ret = "memory_format"
        elif t.name == BaseTy.Storage:
            ret = "Storage | UntypedStorage"
        elif t.name in [BaseTy.Tensor, BaseTy.Generator, BaseTy.Stream]:
            # These python schema type names line up with their function schema names
            ret = t.name.name

    elif isinstance(t, ListType):
        if str(t.elem) == "int":
            ret = "_int | _size" if t.size is not None else "_size"
        elif t.is_tensor_like():
            # Tensor?[] translates to tuple[Tensor | None, ...] | list[Tensor | None] | None
            # Tensor[] translates to tuple[Tensor, ...] | list[Tensor]
            if isinstance(t.elem, OptionalType):
                add_optional = True
                elem_str = "Tensor | None"
            else:
                elem_str = "Tensor"
            ret = (
                f"Tensor | tuple[{elem_str}, ...] | list[{elem_str}]"
                if t.size is not None
                else f"tuple[{elem_str}, ...] | list[{elem_str}]"
            )
        elif str(t.elem) == "float":
            ret = "Sequence[_float]"
        elif str(t.elem) == "SymInt" and t.size is not None:
            elem = argument_type_str_pyi(t.elem)
            ret = f"{elem} | Sequence[{elem}]"
        else:
            elem = argument_type_str_pyi(t.elem)
            ret = f"Sequence[{elem}]"

    else:
        raise RuntimeError(f"unrecognized type {repr(t)}")

    if add_optional:
        ret = _append_optional_pyi(ret)

    return ret


def return_type_str_pyi(t: Type) -> str:
    # Where arguments are open to accepting Union, return types should return
    # concrete types

    if isinstance(t, OptionalType):
        inner = return_type_str_pyi(t.elem)
        return _append_optional_pyi(inner)

    if isinstance(t, BaseType):
        if t.name == BaseTy.Device:
            return "_device"
        else:
            return argument_type_str_pyi(t)

    if isinstance(t, ListType):
        inner = return_type_str_pyi(t.elem)
        return f"tuple[{inner}, ...]"

    return argument_type_str_pyi(t)


def returns_structseq_pyi(signature: PythonSignature) -> tuple[str, str] | None:
    python_returns = [return_type_str_pyi(r.type) for r in signature.returns.returns]
    structseq_name = signature.name
    field_names = structseq_fieldnames(signature.returns.returns)
    if field_names:
        # These types are structseq objects which act like named NamedTuples, but
        # the constructor acts like the constructor of tuple. Using typing.NamedTuple
        # does not allow us to override __init__.
        seq_type = f"tuple[{', '.join(python_returns)}]"
        structseq_def_lines = [
            f"class {structseq_name}({seq_type}):  # fmt: skip",
        ]
        for name, ret_type in zip(field_names, python_returns):
            structseq_def_lines.extend(
                [
                    "    @property",
                    f"    def {name}(self) -> {ret_type}: ...",
                ]
            )
        structseq_def_lines.extend(
            [
                "    def __new__(",
                "        cls,",
                f"        sequence: {seq_type},",
                "    ) -> Self:  # fmt: skip",
                "        ...",
                f"    n_fields: Final[_int] = {len(field_names)}",
                f"    n_sequence_fields: Final[_int] = {len(field_names)}",
                "    n_unnamed_fields: Final[_int] = 0",
                "    def __init_subclass__(cls) -> NoReturn: ...  # prohibit subclassing",
                "",  # add an extra newline
            ]
        )
        structseq_def = "\n".join(structseq_def_lines)
        # Example:
        # structseq_def = (
        #     "class max(tuple[Tensor, Tensor]):  # fmt: skip\n"
        #     "    @property\n"
        #     "    def values(self) -> Tensor: ...\n"
        #     "    @property\n"
        #     "    def indices(self) -> Tensor: ...\n"
        #     "    def __new__(\n"
        #     "        cls,\n"
        #     "        sequence: tuple[Tensor, Tensor],\n"
        #     "    ) -> Self:  # fmt: skip\n"
        #     "        ...\n"
        #     "    n_fields: Final[_int] = 2",
        #     "    n_sequence_fields: Final[_int] = 2",
        #     "    n_unnamed_fields: Final[_int] = 0",
        #     "    def __init_subclass__(cls) -> NoReturn: ...  # prohibit subclassing",
        # )
        return structseq_name, structseq_def
    return None


def returns_str_pyi(signature: PythonSignature) -> str:
    field_names = structseq_fieldnames(signature.returns.returns)
    if field_names:
        return f"torch.return_types.{signature.name}"

    python_returns = [return_type_str_pyi(r.type) for r in signature.returns.returns]
    if len(python_returns) > 1:
        return "tuple[" + ", ".join(python_returns) + "]"
    if len(python_returns) == 1:
        return python_returns[0]
    return "None"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ Function Dispatch
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# This section provides APIs to generate the code that does C++ function
# dispatch. The C++ function call is wrapped by a lambda function.
# For example:
#
#    // aten::selu_(Tensor(a!) self) -> Tensor(a!)
#    auto dispatch_selu_ = [](Tensor self) -> Tensor {
#      pybind11::gil_scoped_release no_gil;
#      return at::selu_(self);
#    };
#
# The lambda function's signature follows the C++ signature in common
# cases, e.g.:
#
#   // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
#   [](const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor
#
# For out variant the 'out' argument's type is changed from 'Tensor &'
# to 'Tensor'. It's because when calling the lambda it passes in the
# PythonArgParser output '_r.tensor(3)', which is stack allocated object
# and needs to pass by value. Also see comments in 'dispatch_lambda_return_str()'.
#
#   // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
#   [](Tensor out, const Tensor & self, const Tensor & other, Scalar alpha) -> Tensor
#
# For multi-output case it can keep using reference type because the
# PythonArgParser output has been unpacked to local variables, e.g.:
#
#   // aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *,
#   //     Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
#   [](Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) -> std::tuple<Tensor,Tensor>
#
# For deprecated python signature, it should follow deprecated python arg order.
# TODO: This is to keep same byte-for-byte result as the old codegen - maybe unnecessary?


def dispatch_lambda_args(
    ps: PythonSignature, f: NativeFunction, symint: bool = True
) -> tuple[DispatchLambdaArgument, ...]:
    if isinstance(ps, PythonSignatureDeprecated):
        schema = ps.deprecated_schema
    else:
        schema = f.func

    # Start with cpp arguments - dispatch lambda signature always include 'self'
    cpp_args = cpp.arguments(
        arguments=schema.arguments,
        faithful=False,
        symint=symint,
        method=False,
        cpp_no_default_args=f.cpp_no_default_args,
    )
    out_args: set[str] = {a.name for a in schema.arguments.out}

    # Convert from cpp argument to lambda argument
    def dispatch_lambda_arg(cpp_arg: Binding) -> DispatchLambdaArgument:
        type_str = cpp_arg.type
        is_out_arg = cpp_arg.name in out_args
        if ps.method and cpp_arg.name == "self":
            # For method's 'self', we can use 'const Tensor &' and simply ignore mutability!
            type_str = "const at::Tensor &"
        else:
            # For other cases we need prevent dangling refs to temps (unless it's
            # unpacked scattered output)
            # The reason is explained in the comments above and in 'dispatch_lambda_return_str()'.
            # TODO: avoid this special handling?
            ensure_temp_safe = len(out_args) <= 1 or not is_out_arg
            if ensure_temp_safe:
                type_str = {
                    "at::Tensor &": "at::Tensor",
                }.get(type_str, type_str)
        return DispatchLambdaArgument(
            name=cpp_arg.name,
            type_str=type_str,
            is_out_arg=is_out_arg,
        )

    return tuple(map(dispatch_lambda_arg, cpp_args))


# [old codegen] XXX: if you got here because of an assertion failure, it doesn't mean
# it's enough to just extend the list here. Before you do this, make sure
# to add an appropriate wrap() overload in torch/csrc/autograd/utils/wrap_outputs.h.
SUPPORTED_RETURN_TYPES = {
    "at::Tensor",
    "::std::tuple<at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,int64_t>",
    "::std::tuple<at::Tensor,at::Tensor,double,int64_t>",
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t>",
    "::std::tuple<at::Tensor,at::Tensor,double,at::Tensor,int64_t>",
    "::std::tuple<double,int64_t>",
    "::std::tuple<at::Tensor,::std::vector<at::Tensor>>",
    "::std::vector<at::Tensor>",
    # Needed for flash attention forw/backward
    "::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::SymInt,c10::SymInt,at::Tensor,at::Tensor,at::Tensor>",
    "at::Scalar",
    "bool",
    "int64_t",
    "void*",
    "void",
    "at::QScheme",
    "double",
    "at::IntArrayRef",
    "at::ScalarType",
    "at::Stream",
}


def dispatch_lambda_return_str(f: NativeFunction) -> str:
    # [old codegen] Remove type annotation (e.g. 'Tensor' rather than 'Tensor &')
    # because the dispatch lambdas take mutable arguments *by value*, not
    # by reference. If you then return a reference to such an argument, you
    # will now have a pointer to a dangling stack entry. Not good.
    #
    # You want:
    #
    #   auto dispatch_selu_ = [](Tensor self) -> Tensor { ...; return at::selu_(self); };
    #                                            ^^^^^^
    #
    # *not*
    #
    #   auto dispatch_selu_ = [](Tensor self) -> Tensor& { ...; return at::selu_(self); };
    #                                            ^^^^^^^
    #
    # (NB: We can't make dispatch_selu_ take Tensor&, because the enclosing
    # codegen looks like dispatch_selu_(_r.tensor(0)), and you can't take a
    # mutable reference to temporary.  Maybe we could assign it to a
    # variable itself.)
    returns_without_annotation = tuple(
        Return(r.name, r.type, None) for r in f.func.returns
    )
    return_str = cpp.returns_type(returns_without_annotation, symint=True).cpp_type()
    if return_str not in SUPPORTED_RETURN_TYPES:
        raise RuntimeError(f"{f.func.name} returns unsupported type {return_str}")
    return return_str


def cpp_dispatch_target(f: NativeFunction) -> str:
    symint = f.func.has_symint()
    name = cpp.name(f.func, symint_overload=symint)
    if Variant.method in f.variants:
        return f"self.{name}"
    if Variant.function in f.variants:
        if has_tensor_options(f) or f.func.name.name.base.endswith("_like"):
            namespace = "torch"
        else:
            namespace = "at"
        return f"{namespace}::{name}"
    raise RuntimeError(f"could not dispatch, neither function nor method: {f.func}")


def cpp_dispatch_exprs(
    f: NativeFunction,
    *,
    python_signature: PythonSignature | None = None,
) -> tuple[str, ...]:
    cpp_args: Sequence[Binding] = _cpp_signature(f, method=False).arguments()

    exprs: tuple[str, ...] = ()
    if not isinstance(python_signature, PythonSignatureDeprecated):
        # By default the exprs are consistent with the C++ signature.
        exprs = tuple(a.name for a in cpp_args)
    else:
        # For deprecated python signature we may need fill in some constants.
        exprs = tuple(
            filter(
                lambda n: n != "out" or f.func.is_out_fn(),
                python_signature.deprecated_args_exprs,
            )
        )

    if Variant.method in f.variants:
        exprs = tuple(filter("self".__ne__, exprs))

    return exprs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                     Python / C++ Args Binding
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# We explicitly enumerate the PythonArgParser unpacking methods for all
# supported types. This might be more verbose than necessary, partially
# because of the irregularity of unpacking method naming, partially
# because we want to mimic the old codegen behavior - to reject
# unexpected and/or unsupported cases which the old codegen rejects.
# For certain cases it is intentionally more restrictive than necessary,
# e.g.: it doesn't accepts doublelist with definite size.
def arg_parser_unpack_method(
    t: Type, default: str | None, default_init: str | None, *, symint: bool = True
) -> str:
    has_default_init = default_init is not None
    if has_default_init and str(t) not in (
        "ScalarType?",
        "ScalarType",
        "Device",
        "Device?",
        "Layout",
        "Layout?",
        "bool",
        "bool?",
    ):
        raise RuntimeError(f"type '{t}' does not supported unpacking with default")

    if isinstance(t, BaseType):
        if t.name in [
            BaseTy.Tensor,
            BaseTy.Stream,
            BaseTy.Storage,
            BaseTy.Scalar,
        ]:
            # These unpack methods line up with their schema names
            return t.name.name.lower()
        elif t.name == BaseTy.ScalarType:
            return "scalartypeWithDefault" if has_default_init else "scalartype"
        elif t.name == BaseTy.Device:
            return "deviceWithDefault" if has_default_init else "device"
        elif t.name == BaseTy.DeviceIndex:
            return "toInt64"
        elif t.name == BaseTy.int:
            return "toInt64"
        elif t.name == BaseTy.SymInt:
            return "toSymInt" if symint else "toInt64"
        elif t.name == BaseTy.bool:
            return "toBoolWithDefault" if has_default_init else "toBool"
        elif t.name == BaseTy.float:
            return "toDouble"
        elif t.name == BaseTy.str:
            return "stringView"
        elif t.name == BaseTy.Layout:
            return "layoutWithDefault" if has_default_init else "layout"
        elif t.name == BaseTy.MemoryFormat:
            return "memoryformat"

    elif isinstance(t, OptionalType):
        if str(t.elem) == "Tensor":
            return "optionalTensor"
        elif str(t.elem) == "Generator":
            return "generator"
        elif not has_default_init and default in (
            None,
            "None",
            "::std::nullopt",
            "std::nullopt",
        ):
            # If default is None: append 'Optional' to elem's unpacking method
            return (
                arg_parser_unpack_method(t.elem, None, None, symint=symint) + "Optional"
            )
        else:
            # Otherwise, load as underlying type with default
            return arg_parser_unpack_method(
                t.elem, default, default_init, symint=symint
            )

    elif isinstance(t, ListType):
        if str(t.elem) == "Tensor":
            # accept and use definite size
            return f"tensorlist_n<{t.size}>" if t.size is not None else "tensorlist"
        elif str(t.elem) == "Tensor?":
            return "list_of_optional_tensors"
        elif str(t.elem) == "int":
            # accept definite size
            return "intlist"
        elif str(t.elem) == "float":
            return "doublelist"
        elif str(t.elem) == "SymInt":
            # accept definite size
            return "symintlist" if symint else "intlist"
        elif str(t.elem) == "Scalar":
            return "scalarlist"
    raise RuntimeError(f"type '{t}' is not supported by PythonArgParser")


# Return RHS expression for python argument using PythonArgParser output.
# e.g. for arg name 'foo', arg type 'bool', arg_index = 2, returns '_r.toBool(2)'
def arg_parser_output_expr(
    arg_index: int, a: PythonArgument, *, symint: bool = True
) -> PythonArgParserOutputExpr:
    has_default = a.default_init is not None
    unpack_method = arg_parser_unpack_method(
        t=a.type, default=a.default, default_init=a.default_init, symint=symint
    )
    default = f", {a.default_init}" if has_default else ""
    expr = f"_r.{unpack_method}({arg_index}{default})"

    return PythonArgParserOutputExpr(
        name=a.name,
        expr=expr,
        index=arg_index,
        argument=a,
    )


# Returns a map with key = arg_name and value = PythonArgParserOutputExpr.
def arg_parser_output_exprs(
    ps: PythonSignature, f: NativeFunction, *, symint: bool = True
) -> dict[str, PythonArgParserOutputExpr]:
    return {
        e.name: e
        for i, a in enumerate(ps.arguments())
        for e in (arg_parser_output_expr(i, a, symint=symint),)
    }


# argument name to type for scattered tensor options fields
TENSOR_OPTIONS_FIELDS = {
    "dtype": "ScalarType?",
    "device": "Device?",
    "layout": "Layout?",
    "pin_memory": "bool?",
    "requires_grad": "bool?",
}


# bind arg parser outputs (python args) with dispatch lambda arguments (c++ args).
def dispatch_lambda_exprs(
    ps: PythonSignature, f: NativeFunction, *, symint: bool = True
) -> DispatchLambdaArgumentExprs:
    # This method is to bind 'arg_parser_outputs' and 'lambda_args' by producing
    # 'inits' and 'lambda_args_exprs' for each lambda argument using arg parser
    # outputs.
    arg_parser_outputs = arg_parser_output_exprs(ps, f, symint=symint)
    lambda_args = dispatch_lambda_args(ps, f, symint=symint)
    inits: list[str] = []
    lambda_args_exprs: dict[str, str] = {}

    has_toptions = has_tensor_options(f)

    # 1. special inits/unpacking to provide binding exprs for lambda arguments.
    for a in ps.arguments(skip_tensor_options=True):
        name = a.name
        arg_parser_expr = arg_parser_outputs[a.name].expr

        if has_toptions and name == "self":
            # TODO: why this needs to be special case?
            inits.extend(
                [
                    f"auto self = {arg_parser_expr};",
                ]
            )
            lambda_args_exprs[name] = name
        elif (
            isinstance(a, PythonOutArgument)
            and len(a.outputs) > 1
            and f.func.is_out_fn()
        ):
            inits.extend(
                [
                    f"auto out = {arg_parser_expr};",
                ]
            )
            for i, out_arg in enumerate(a.outputs):
                lambda_args_exprs[out_arg.name] = f"out[{i}]"
        else:
            # default case - directly using PythonArgParser output expr
            lambda_args_exprs[name] = arg_parser_expr

    # method's self is passed directly to python binding, rather than parsed
    if ps.method:
        lambda_args_exprs["self"] = "self"

    # 2. special packing/checking for TensorOptions.
    tensor_options_args_names = [a.name for a in ps.tensor_options_args]
    if has_toptions:
        if f.func.is_out_fn():
            raise RuntimeError(f"{f.func}: tensor options with output arg")
        for a in ps.tensor_options_args:
            if a.name not in TENSOR_OPTIONS_FIELDS:
                raise RuntimeError(
                    f"{f.func}: unrecognized tensor options field '{a.name}' in python binding arguments"
                )
            if str(a.type) != TENSOR_OPTIONS_FIELDS.get(a.name):
                raise RuntimeError(
                    f"{f.func}: unrecognized type '{str(a.type)}' for tensor options field '{a.name}'"
                )
        if not all(a in tensor_options_args_names for a in TENSOR_OPTIONS_FIELDS):
            raise RuntimeError(
                f"{f.func}: incomplete tensor options args: {tensor_options_args_names}"
            )

        maybe_initialize_device = (
            """\
if (!at::impl::tensor_has_dispatch(self)) {
  torch::utils::maybe_initialize_device(options);
}
"""
            if ps.method
            else "torch::utils::maybe_initialize_device(options);\n"
        )

        inits.append(
            f"""\
const auto options = TensorOptions()
    .dtype({arg_parser_outputs["dtype"].expr})
    .device({arg_parser_outputs["device"].expr})
    .layout({arg_parser_outputs["layout"].expr})
    .requires_grad({arg_parser_outputs["requires_grad"].expr})
    .pinned_memory({arg_parser_outputs["pin_memory"].expr});
{maybe_initialize_device}\
"""
        )
        lambda_args_exprs["options"] = "options"

    # 3. special case - access scattered TensorOptions fields without packing
    # TODO: maybe move to the generator side as it's not related to binding.
    if not has_toptions and tensor_options_args_names:
        if "dtype" in tensor_options_args_names:
            # we're an output-arg variant, check these args against output tensor
            if not f.func.is_out_fn():
                raise RuntimeError(
                    f"{f.func}: dtype in tensor_options_args without output arg, {ps} {ps.arguments}"
                )
            if not all(a in tensor_options_args_names for a in ("layout", "device")):
                raise RuntimeError(
                    f"{f.func}: incomplete tensor options for output check"
                )

            inits.append(
                f"""\
check_out_type_matches({arg_parser_outputs["out"].expr}, {arg_parser_outputs["dtype"].expr},
                       {arg_parser_outputs["dtype"].is_none_expr}, {arg_parser_outputs["layout"].expr},
                       {arg_parser_outputs["device"].expr}, {arg_parser_outputs["device"].is_none_expr});
"""
            )
        # we'll set requires_grad on outgoing tensor
        if "requires_grad" not in tensor_options_args_names:
            raise RuntimeError(
                f'{f.func}: expected "requires_grad" in tensor_options_args absent, but found [{tensor_options_args_names}]'
            )

    return DispatchLambdaArgumentExprs(
        exprs=tuple(lambda_args_exprs[a.name] for a in lambda_args),
        inits=inits,
    )
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass

import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
    ArgName,
    BaseCppType,
    BaseCType,
    Binding,
    ConstRefCType,
    CType,
    NamedCType,
    scalarT,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    DispatchKey,
    FunctionSchema,
    NativeFunctionsGroup,
    Type,
)


def schema_kernel_name(func: FunctionSchema, dispatch_key: DispatchKey) -> str:
    if not func.is_out_fn():
        raise AssertionError("ufunc.kernel_name should only be invoked on out schemas")
    return f"ufunc_{func.name.name}_{dispatch_key}"


def kernel_name(g: NativeFunctionsGroup, dispatch_key: DispatchKey) -> str:
    return schema_kernel_name(g.out.func, dispatch_key)


# Tensors are omitted (as they are stored in TensorIterator), everything else is
# passed along  (technically, we can pass tensors along too, it just wastes
# argument registers)
#
# NB: used for CPU only
def dispatchstub_type(t: Type, *, binds: ArgName) -> NamedCType | None:
    # Dispatch stubs are always plain ints
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    if r is not None:
        return r

    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
    elif t == BaseType(BaseTy.Tensor):
        return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


def opmath_type(scalar_t: BaseCppType) -> BaseCppType:
    if scalar_t == api_types.scalar_t:
        return api_types.opmath_t
    raise NotImplementedError


# NB: Tensors in constructor are stored in opmath_t, not scalar_t
# because Tensor in constructor = its a scalar tensor partially applied =
# it can be higher precision and we want to compute in that higher precision
#
# NB: CUDA only
def ufunctor_ctor_type(t: Type, *, binds: ArgName, scalar_t: BaseCppType) -> NamedCType:
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    if r is not None:
        return r

    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Only Tensors ever get passed directly to operator()
#
# NB: CUDA only
# (Actually, this works for CPU too)
def ufunctor_apply_type(
    t: Type, *, binds: ArgName, scalar_t: BaseCppType
) -> NamedCType:
    if t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(scalar_t))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# The actual ufunc template function the user writes.  Everything here
# is done in the computation type.  compute_t is opmath_t in CUDA and scalar_t
# in CPU
def ufunc_type(t: Type, *, binds: ArgName, compute_t: CType) -> NamedCType:
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    if r is not None:
        return r

    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, compute_t)
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, compute_t)
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


def ufunctor_ctor_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    return Binding(
        nctype=ufunctor_ctor_type(a.type, binds=a.name, scalar_t=scalar_t),
        name=a.name,
        default=None,
        argument=a,
    )


def ufunctor_apply_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    return Binding(
        nctype=ufunctor_apply_type(a.type, binds=a.name, scalar_t=scalar_t),
        name=a.name,
        default=None,
        argument=a,
    )


def ufunc_argument(a: Argument, compute_t: CType) -> Binding:
    return Binding(
        nctype=ufunc_type(a.type, binds=a.name, compute_t=compute_t),
        name=a.name,
        default=None,
        argument=a,
    )


@dataclass(frozen=True)
class UfunctorBindings:
    ctor: list[Binding]
    apply: list[Binding]


# ufunctors are a CUDA-only concept representing functors that take some of
# their arguments on a host-side constructor, and the rest in the device-side
# apply.  E.g.,
#
# template <typename scalar_t>
# struct CUDAFunctorOnSelf_add {
#   using opmath_t = at::opmath_type<scalar_t>;
#   opmath_t other_;
#   opmath_t alpha_;
#   CUDAFunctorOnSelf_add(opmath_t other, opmath_t alpha) : other_(other), alpha_(alpha) {}
#   __device__ scalar_t operator()(scalar_t self) {
#     return ufunc::add(static_cast<opmath_t>(self), other_, alpha_);
#   }
# };
#
# The ctor refers to the constructor CUDAFunctorOnSelf_add, while apply refers
# to the operator() definition
def ufunctor_arguments(
    g: NativeFunctionsGroup, *, scalar_tensor_idx: int | None, scalar_t: BaseCppType
) -> UfunctorBindings:
    ctor = []
    apply = []
    for a in g.functional.func.arguments.flat_non_out:
        if a.type.is_tensor_like():
            if scalar_tensor_idx == 0:
                # put it in the ctor anyway
                ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
                scalar_tensor_idx = None
            else:
                if scalar_tensor_idx is not None:
                    scalar_tensor_idx -= 1
                apply.append(ufunctor_apply_argument(a, scalar_t=scalar_t))
        else:
            ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
    if scalar_tensor_idx is not None:
        raise AssertionError("scalar_tensor_idx should be None at end of processing")
    return UfunctorBindings(ctor=ctor, apply=apply)


# ufuncs are the inner loop template functions that you wrote in ufunc/add.h
# which do the actual computation in question.  E.g.,
#
# template <typename T>
# C10_HOST_DEVICE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
#   return self + alpha * other;
# }
#
# In this file, we refer to T as compute_t which is bound by caller
def ufunc_arguments(g: NativeFunctionsGroup, *, compute_t: CType) -> list[Binding]:
    return [
        ufunc_argument(a, compute_t=compute_t)
        for a in g.functional.func.arguments.flat_non_out
    ]


# Stubs are the DispatchStub trampolines that CPU kernels use to get to their
# vectorized versions.  E.g.,
#
# using structured_binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
# DECLARE_DISPATCH(structured_binary_fn_alpha, add_stub);
def stub_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    # stubs drop all tensor arguments (they are implicit in the TensorIterator
    # argument and keep everything else)
    return [
        r
        for a in g.out.func.arguments.flat_non_out
        if not a.type.is_tensor_like()
        for r in structured.argument(a)
    ]
<|endoftext|>from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from typing_extensions import assert_never

from torchgen.api import cpp
from torchgen.api.types import ArgName, Binding, CType, NamedCType
from torchgen.model import (
    Argument,
    FunctionSchema,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import concatMap


if TYPE_CHECKING:
    from collections.abc import Sequence


# This file describes the translation of JIT schema to the dispatcher
# API, the *unboxed* calling convention by which invocations through
# the dispatcher are made.  Historically, the dispatcher API matched
# the C++ API, but with the establishment of the boxed API, we've
# made changes to the dispatcher API to so that the unboxed API
# better aligns with the boxed API.  The dispatcher API hooks heavily
# into our template based boxing/unboxing machinery, so changes
# to this convention will usually need template updates too.
#
# Prominent characteristics of the dispatcher API:
#
#   - dtype, layout, device and pin_memory are represented as separate
#     arguments.
#


def name(func: FunctionSchema) -> str:
    return cpp.name(func)


def argumenttype_type(
    t: Type,
    *,
    mutable: bool,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = True,
) -> NamedCType:
    # This is a faux amis.  If it makes sense in the future to add
    # more special cases here, or invert things so cpp.argument_type
    # calls this, or just completely inline the function, please do
    # it.
    return cpp.argumenttype_type(
        t,
        mutable=mutable,
        binds=binds,
        symint=symint,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
    )


def argument_type(
    a: Argument,
    *,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = True,
) -> NamedCType:
    return argumenttype_type(
        a.type,
        mutable=a.is_write,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
        symint=symint,
    )


def returns_type(rs: Sequence[Return], *, symint: bool = True) -> CType:
    # At present, there is no difference. But there could be!
    return cpp.returns_type(rs, symint=symint)


def jit_arguments(func: FunctionSchema) -> list[Argument]:
    def to_argument(
        a: Argument | TensorOptionsArguments | SelfArgument,
    ) -> list[Argument]:
        if isinstance(a, Argument):
            return [a]
        elif isinstance(a, SelfArgument):
            return [a.argument]
        elif isinstance(a, TensorOptionsArguments):
            return [a.dtype, a.layout, a.device, a.pin_memory]
        else:
            assert_never(a)

    return list(
        concatMap(
            to_argument,
            itertools.chain(
                func.arguments.positional, func.arguments.kwarg_only, func.arguments.out
            ),
        )
    )


def argument(
    a: Argument, *, remove_non_owning_ref_types: bool = False, symint: bool = True
) -> Binding:
    return Binding(
        nctype=argument_type(
            a,
            binds=a.name,
            remove_non_owning_ref_types=remove_non_owning_ref_types,
            symint=symint,
        ),
        name=a.name,
        argument=a,
    )


def arguments(func: FunctionSchema, *, symint: bool = True) -> list[Binding]:
    return [argument(a, symint=symint) for a in jit_arguments(func)]
<|endoftext|>from torchgen.model import NativeFunctionsGroup


# Follows dispatcher calling convention, but:
#   - Mutable arguments not allowed.  Meta functions are always
#     written in functional form.  Look at FunctionSchema.signature()
#   - No tensor returns; instead we return a TensorMeta describing
#     the tensor in question


def name(g: NativeFunctionsGroup) -> str:
    # use the overload name from the functional version
    return str(g.functional.func.name).replace(".", "_")
<|endoftext|>from __future__ import annotations

from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignatureGroup, CType
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    OptionalType,
    Type,
)


# This file generates the code for unboxing wrappers, i.e., the glue logic to unbox a boxed operator and convert the
# ivalues from stack to correct arguments to the unboxed kernel, based on corresponding JIT schema. This codegen is
# an alternative way to generate unboxing wrappers similar to the existing C++ metaprogramming approach but gets the
# job done statically. These generated unboxing wrappers will be useful under the scenario where we need to register
# a fixed set of operators known at compile time and thus can save some time in runtime initialization phase.
#
# Here's an example on how the codegen works:
#
# - Function Schema (source of truth)
#
#      aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None,
#                  Device? device=None, bool? pin_memory=None) -> Tensor
#
# - Argument Conversion
#       Generates C++ code to convert an ivalue (from stack) to its underlying C++ type.
#    - int[] size
#        ```cpp
#           const c10::List<c10::IValue> size_list_in = (std::move(peek(stack, 0, 5))).toList();
#
#           std::vector<int64_t> size_vec;
#           for (c10::IValue size_elem: size_list_in) {
#               int64_t size_base = size_elem.to<int64_t>();
#               size_vec.push_back(size_base);
#           }
#           at::ArrayRef<int64_t> size_list_out(size_vec);
#                                 ~~~~~~~~~~~~~ <-- The converted argument from ivalues in the stack.
#                                                   Will be passed to unboxed kernel.
#       ```
#    - ScalarType? dtype
#       ```cpp
#           ::std::optional<c10::IValue> dtype_opt = (std::move(peek(stack, 1, 5))).toOptional<c10::IValue>();
#           ::std::optional<at::ScalarType> dtype_opt_out;
#           if (dtype_opt.has_value()) {
#                         ~~~~~~~~~~~ <-- Unwrapping optional shell
#               const c10::IValue dtype_opt_in = dtype_opt.value();
#               at::ScalarType dtype_base = dtype_opt_in.to<at::ScalarType>();
#                                                        ~~~~~~~~~~~~~~~~~~~~ <-- For base types, convert ivalue to it
#                                                                                 directly using ".to<T>()" API.
#               dtype_opt_out = ::std::optional<at::ScalarType>(dtype_base);
#           } else {
#               dtype_opt_out = ::std::optional<at::ScalarType>();
#           }
#       ```
#
# - Unboxed Kernel Call
#   ```cpp
#       auto result_ = torch::zeros(
#           size_list_out,
#           options
#       );
#   ```
#
# - Push Result Back to Stack
#   ```cpp
#       drop(stack, 5);
#       pack(stack, std::move(result_));
#   ```
connector = "\n\t"


# Return unboxing function name for a NativeFunction
def name(f: NativeFunction) -> str:
    return f.func.name.unambiguous_name()


# Convert all the arguments in a NativeFunction to C++ code
def convert_arguments(f: NativeFunction) -> tuple[list[Binding], list[str]]:
    # we need the 'self' argument so method needs to be False
    args = (
        CppSignatureGroup.from_native_function(f, method=False)
        .most_faithful_signature()
        .arguments()
    )
    code_list = [
        f"c10::IValue {args[i].name} = std::move(peek(stack, {i}, {len(args)}));"
        for i in range(len(args))
    ] + [""]
    binding_list = []
    for arg in args:
        # expecting only Argument
        if not isinstance(arg.argument, Argument):
            raise Exception(  # noqa: TRY002
                f"Unexpected argument type, expecting `Argument` but got {arg}"
            )
        argument: Argument = arg.argument
        unboxed_name, _, code, decl = argumenttype_ivalue_convert(
            argument.type,
            argument.name,
            mutable=argument.is_write,
        )
        code_list.extend(decl)
        code_list.extend(code)
        binding_list.append(arg.with_name(unboxed_name))
    return binding_list, code_list


# Takes in the type, name and mutability corresponding to an argument, and generates a tuple of:
# (1) the C++ code necessary to unbox the argument
# (2) A Binding corresponding to the newly created unboxed variable, including variable name and its CType
def argumenttype_ivalue_convert(
    t: Type, arg_name: str, *, mutable: bool = False
) -> tuple[str, CType, list[str], list[str]]:
    # Unboxing is for mobile, which doesn't care about SymInts
    ctype = cpp.argumenttype_type(
        t=t, mutable=mutable, binds=arg_name, symint=False
    ).type

    if isinstance(t, BaseType):
        out_name = f"{arg_name}_base"
        code, decl = _gen_code_base_type(
            arg_name=arg_name, out_name=out_name, ctype=ctype
        )
    elif isinstance(t, OptionalType):
        out_name = f"{arg_name}_opt_out"
        code, decl = _gen_code_optional_type(
            arg_name=arg_name,
            out_name=out_name,
            t=t,
            ctype=ctype,
        )
    elif isinstance(t, ListType):
        out_name = f"{arg_name}_list_out"
        code, decl = _gen_code_list_type(
            arg_name=arg_name,
            out_name=out_name,
            t=t,
            ctype=ctype,
        )
    else:
        raise Exception(f"Cannot handle type {t}. arg_name: {arg_name}")  # noqa: TRY002
    return out_name, ctype, code, decl


def _gen_code_base_type(
    arg_name: str, out_name: str, ctype: CType
) -> tuple[list[str], list[str]]:
    return [
        f"{ctype.cpp_type(strip_ref=True)} {out_name} = {arg_name}.to<{ctype.cpp_type(strip_ref=True)}>();"
    ], []


def _gen_code_optional_type(
    arg_name: str, out_name: str, t: OptionalType, ctype: CType
) -> tuple[list[str], list[str]]:
    in_name = f"{arg_name}_opt_in"
    res_name, _, res_code, decl = argumenttype_ivalue_convert(t.elem, in_name)
    return (
        f"""
auto {arg_name}_opt = {arg_name}.toOptional<c10::IValue>();
{ctype.cpp_type(strip_ref=True)} {out_name};
if ({arg_name}_opt.has_value()) {{
    const c10::IValue {in_name} = {arg_name}_opt.value();
    {connector.join(res_code)}
    {out_name} = {ctype.cpp_type(strip_ref=True)}({res_name});
}} else {{
    {out_name} = {ctype.cpp_type(strip_ref=True)}();
}}
        """.split("\n"),
        decl,
    )


def _gen_code_list_type(
    arg_name: str, out_name: str, t: ListType, ctype: CType
) -> tuple[list[str], list[str]]:
    in_name = f"{arg_name}_list_in"
    elem_name = f"{arg_name}_elem"
    code = [f"const c10::List<c10::IValue> {in_name} = {arg_name}.toList();"]
    res_name, res_ctype, res_code, decl = argumenttype_ivalue_convert(t.elem, elem_name)
    # handle list type with size, e.g., bool[4]
    if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool and t.size:
        code.extend(
            f"""
{ctype.cpp_type(strip_ref=True)} {out_name} = as_array<{res_ctype.cpp_type(strip_ref=True)}, {t.size}>({in_name});
            """.split("\n")
        )
    # we have to use c10::List for optional element. e.g., Tensor?[] -> c10::List<::std::optional<at::Tensor>>
    elif isinstance(t.elem, OptionalType):
        code.extend(
            f"""
{ctype.cpp_type(strip_ref=True)} {out_name};
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res_code)}
    {out_name}.push_back({res_name});
}}
            """.split("\n")
        )
    else:
        # use ArrayRef as default.
        vec_name = arg_name + "_vec"
        # need to bring vector instantiation out of scope so that ArrayRef has valid data
        decl.append(f"std::vector<{res_ctype.cpp_type(strip_ref=True)}> {vec_name};")
        code.extend(
            f"""
for (c10::IValue {elem_name}: {in_name}) {{
    {connector.join(res_code)}
    {vec_name}.push_back({res_name});
}}
{ctype.cpp_type(strip_ref=True)} {out_name}({vec_name});
            """.split("\n")
        )
    return code, decl
<|endoftext|>"""
Where should I add a new type? `types_base.py` vs `types.py`

This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.

`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.

The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn't
contain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reused
if we want to generate code for another C++ library.

Add new types to `types.py` if these types are ATen/c10 related.
Add new types to `types_base.py` if they are basic and not attached to ATen/c10.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchgen.api.types.types_base import (
    BaseCppType,
    BaseCType,
    boolT,
    byteT,
    charT,
    CType,
    doubleT,
    floatT,
    int32T,
    longT,
    shortT,
)
from torchgen.model import BaseTy, ScalarType


TENSOR_LIST_LIKE_CTYPES = [
    "at::TensorList",
    "const c10::List<::std::optional<at::Tensor>> &",
    "const at::ITensorListRef &",
]


halfT = BaseCppType("at", "Half")
complexHalfT = BaseCppType(
    "c10", "complex<c10::Half>"
)  # stuffing template param here is an abuse
complexFloatT = BaseCppType("c10", "complex<float>")
complexDoubleT = BaseCppType("c10", "complex<double>")
bfloat16T = BaseCppType("at", "BFloat16")
float8_e5m2T = BaseCppType("at", "Float8_e5m2")
float8_e5m2fnuzT = BaseCppType("at", "Float8_e5m2fnuz")
float8_e4m3fnT = BaseCppType("at", "Float8_e4m3fn")
float8_e4m3fnuzT = BaseCppType("at", "Float8_e4m3fnuz")
float8_e8m0fnuT = BaseCppType("at", "Float8_e8m0fnu")
stringT = BaseCppType("c10", "string_view")
generatorT = BaseCppType("at", "Generator")
scalarTypeT = BaseCppType("at", "ScalarType")
tensorT = BaseCppType("at", "Tensor")
optionalTensorRefT = BaseCppType("at", "OptionalTensorRef")
tensorListT = BaseCppType("at", "TensorList")
iTensorListRefT = BaseCppType("at", "ITensorListRef")
iOptTensorListRefT = BaseCppType("at", "IOptTensorListRef")
dimVectorT = BaseCppType("at", "DimVector")
layoutT = BaseCppType("at", "Layout")
deviceT = BaseCppType("at", "Device")
deviceIndexT = BaseCppType("at", "DeviceIndex")
scalarT = BaseCppType("at", "Scalar")
optionalScalarRefT = BaseCppType("at", "OptionalScalarRef")
memoryFormatT = BaseCppType("at", "MemoryFormat")
qschemeT = BaseCppType("at", "QScheme")
storageT = BaseCppType("at", "Storage")
streamT = BaseCppType("at", "Stream")
intArrayRefT = BaseCppType("at", "IntArrayRef")
optionalIntArrayRefT = BaseCppType("at", "OptionalIntArrayRef")
optionalSymIntArrayRefT = BaseCppType("at", "OptionalSymIntArrayRef")
tensorOptionsT = BaseCppType("at", "TensorOptions")
typeAndSizeT = BaseCppType("torch::autograd::generated", "TypeAndSize")
tensorGeometryT = BaseCppType("at", "TensorGeometry")
SymIntT = BaseCppType("c10", "SymInt")
SymBoolT = BaseCppType("c10", "SymBool")
symIntArrayRefT = BaseCppType("c10", "SymIntArrayRef")

# Types representing template parameters.  Technically, we probably shouldn't
# represent them this way in codegen, but it was pretty convenient.
scalar_t = BaseCppType("", "scalar_t")
opmath_t = BaseCppType("", "opmath_t")

ScalarTypeToCppMapping: dict[ScalarType, BaseCppType] = {
    ScalarType.Byte: byteT,
    ScalarType.Char: charT,
    ScalarType.Short: shortT,
    ScalarType.Int: int32T,
    ScalarType.Long: longT,
    ScalarType.Half: halfT,
    ScalarType.Float: floatT,
    ScalarType.Double: doubleT,
    ScalarType.ComplexHalf: complexHalfT,
    ScalarType.ComplexFloat: complexFloatT,
    ScalarType.ComplexDouble: complexDoubleT,
    ScalarType.Bool: boolT,
    ScalarType.Float8_e5m2: float8_e5m2T,
    ScalarType.Float8_e5m2fnuz: float8_e5m2fnuzT,
    ScalarType.Float8_e4m3fn: float8_e4m3fnT,
    ScalarType.Float8_e4m3fnuz: float8_e4m3fnuzT,
    ScalarType.Float8_e8m0fnu: float8_e8m0fnuT,
}

BaseTypeToCppMapping: dict[BaseTy, BaseCppType] = {
    BaseTy.int: longT,
    BaseTy.float: doubleT,
    BaseTy.bool: boolT,
    BaseTy.str: stringT,
    BaseTy.Generator: generatorT,
    BaseTy.ScalarType: scalarTypeT,
    BaseTy.Tensor: tensorT,
    BaseTy.DimVector: dimVectorT,
    BaseTy.Layout: layoutT,
    BaseTy.Device: deviceT,
    BaseTy.DeviceIndex: deviceIndexT,
    BaseTy.Scalar: scalarT,
    BaseTy.MemoryFormat: memoryFormatT,
    BaseTy.QScheme: qschemeT,
    BaseTy.Storage: storageT,
    BaseTy.Stream: streamT,
    BaseTy.SymInt: SymIntT,
    BaseTy.SymBool: SymBoolT,
}

# CTypes encode C++ type structure as needed for translation.


@dataclass(frozen=True)
class OptionalCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::optional<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return OptionalCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ListCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"c10::List<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return ListCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayRefCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"at::ArrayRef<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return ArrayRefCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class VectorizedCType(CType):
    # This template is explicitly specialized, so the only valid
    # elems are those we have specializations for (e.g., float, double, ...)
    # scalar_t is also a common argument here (when we are codegen in
    # a templated context)
    elem: BaseCType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return f"at::vec::Vectorized<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return self
<|endoftext|>"""
Where should I add a new type? `types_base.py` vs `types.py`

This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.

`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.

The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn't
contain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reused
if we want to generate code for another C++ library.

Add new types to `types.py` if these types are ATen/c10 related.
Add new types to `types_base.py` if they are basic and not attached to ATen/c10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torchgen.model import Argument, SelfArgument, TensorOptionsArguments


# An ArgName is just the str name of the argument in schema;
# but in some special circumstances, we may add a little extra
# context.  The Enum SpecialArgName covers all of these cases;
# grep for their construction sites to see when they can occur.


class SpecialArgName(Enum):
    possibly_redundant_memory_format = auto()


ArgName = str | SpecialArgName


# This class shouldn't be created directly; instead, use/create one of the singletons below.
@dataclass(frozen=True)
class BaseCppType:
    ns: str | None
    name: str

    def __str__(self) -> str:
        if self.ns is None or self.ns == "":
            return self.name
        return f"{self.ns}::{self.name}"


# The set of all non-templated, valid, fully-qualified names of C++ types that are used in the codegen.
# Templated types get their own dataclass, mainly to make namespace parsing easier.
byteT = BaseCppType("", "uint8_t")
charT = BaseCppType("", "int8_t")
shortT = BaseCppType("", "int16_t")
# It would be more symmetric for this to be called intT, but it easy to mix
# this up with JIT int (which is int64_t in C++), so we intentionally don't
# define intT to make it obvious when you've stuffed it up
int32T = BaseCppType("", "int32_t")
longT = BaseCppType("", "int64_t")
doubleT = BaseCppType("", "double")
floatT = BaseCppType("", "float")
boolT = BaseCppType("", "bool")
voidT = BaseCppType("", "void")


class CType(ABC):
    @abstractmethod
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        raise NotImplementedError

    @abstractmethod
    def remove_const_ref(self) -> CType:
        return self


@dataclass(frozen=True)
class BaseCType(CType):
    type: BaseCppType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return str(self.type)

    def remove_const_ref(self) -> CType:
        return self


@dataclass(frozen=True)
class ConstRefCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f"const {self.elem.cpp_type()} &"

    def remove_const_ref(self) -> CType:
        return self.elem.remove_const_ref()


@dataclass(frozen=True)
class VectorCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::vector<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return VectorCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayCType(CType):
    elem: CType
    size: int

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::array<{self.elem.cpp_type()},{self.size}>"

    def remove_const_ref(self) -> CType:
        return ArrayCType(self.elem.remove_const_ref(), self.size)


@dataclass(frozen=True)
class TupleCType(CType):
    elems: list[CType]

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::tuple<{','.join([e.cpp_type() for e in self.elems])}>"

    def remove_const_ref(self) -> CType:
        return TupleCType([e.remove_const_ref() for e in self.elems])


@dataclass(frozen=True)
class MutRefCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f"{self.elem.cpp_type()} &"

    def remove_const_ref(self) -> CType:
        return self.elem.remove_const_ref()


# A NamedCType is short for Named C++ semantic type.  A NamedCType represents a C++ type, plus
# semantic information about what it represents.  For example, consider the
# argument "bool pin_memory"; its normal C++ type is "bool", but its C++
# semantic type also keeps track that this represents a "pin_memory"; you can't
# just use a random other boolean in a context where you need a "pin_memory"!
#


@dataclass(frozen=True)
class NamedCType:
    name: ArgName
    type: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return self.type.cpp_type(strip_ref=strip_ref)

    def remove_const_ref(self) -> NamedCType:
        return NamedCType(self.name, self.type.remove_const_ref())

    def with_name(self, name: str) -> NamedCType:
        return NamedCType(name, self.type)


# A binding represents any C++ binding site for a formal parameter.
# We don't distinguish between binding sites for different APIs;
# instead, all of the important distinctions are encoded in CType,
# which you can use to figure out if a given Binding is appropriate
# for use in another context.  (See torchgen.api.translate)


@dataclass(frozen=True)
class Binding:
    name: str
    nctype: NamedCType
    argument: Argument | TensorOptionsArguments | SelfArgument
    # TODO: maybe don't represent default here
    default: str | None = None

    def rename(self, name: str) -> Binding:
        return Binding(
            name=name,
            nctype=self.nctype,
            argument=self.argument,
            default=self.default,
        )

    @property
    def type(self) -> str:
        return self.nctype.cpp_type()

    def no_default(self) -> Binding:
        return Binding(
            name=self.name,
            nctype=self.nctype,
            default=None,
            argument=self.argument,
        )

    def decl(self, *, func_ptr_cast: bool = False) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"

        # casting only needs to know the type
        if func_ptr_cast:
            return f"{self.type}"
        else:
            return f"{self.type} {self.name}{mb_default}"

    def defn(self) -> str:
        return f"{self.type} {self.name}"

    def with_name(self, name: str) -> Binding:
        return Binding(
            name=name, nctype=self.nctype, argument=self.argument, default=self.default
        )


# An Expr is a C++ expression.  It has a C++ string representing its syntax,
# as well as a CType saying what it provides.


@dataclass(frozen=True)
class Expr:
    expr: str
    type: NamedCType
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.api.types.types_base import Binding, CType, Expr


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torchgen.model import (
        BackendIndex,
        FunctionSchema,
        NativeFunction,
        NativeFunctionsGroup,
        NativeFunctionsViewGroup,
    )


@dataclass(frozen=True)
class CppSignature:
    """
    A CppSignature represents a single overload in the C++ API.  For
    any given function schema, there may be multiple CppSignatures
    corresponding to it, based on how we desugar to C++.  See also
    CppSignatureGroup.
    """

    # The schema this signature is derived from
    func: FunctionSchema

    # Is this a C++ signature for a method, i.e. Tensor::my_op(...)?
    method: bool

    # Is this a faithful C++ signature (i.e. following the JIT schema) or a convenience API
    # (i.e. with a potential TensorOptions argument and out arguments in the front)
    faithful: bool

    # Is this a symint C++ signature.  For BC reasons, functions that take
    # SymInts still present as int64_t in C++, and the SymInt variant is
    # offered at a different overload name
    #
    # NB: If a function RETURNS a SymInt, this is ALWAYS false
    symint: bool

    # The set of C++ arguments which should not have defaults applied to them
    cpp_no_default_args: set[str]

    # Is this a fallback C++ binding?  Fallback bindings are enabled by
    # manual_cpp_binding: True and are alternate, non-public API that
    # lets manual C++ binding implementers access the binding that would
    # have been automatically generated
    fallback_binding: bool = False

    # Return the unpacked argument structure of this signature,
    # discarding information about which arguments are semantically
    # related to each other.
    def arguments(self) -> Sequence[Binding]:
        return cpp.arguments(
            self.func.arguments,
            faithful=self.faithful,
            symint=self.symint,
            method=self.method,
            cpp_no_default_args=self.cpp_no_default_args,
        )

    def name(self, *, suppress_symint_suffix: bool = False) -> str:
        n = cpp.name(
            self.func,
            faithful_name_for_out_overloads=self.faithful,
            symint_overload=False if suppress_symint_suffix else self.symint,
        )
        if self.fallback_binding:
            n = f"__dispatch_{n}"
        return n

    # Render the C++ declaration for this signature
    def decl(
        self,
        *,
        name: str | None = None,
        prefix: str = "",
        is_redispatching_fn: bool = False,
        suppress_symint_suffix: bool = False,
    ) -> str:
        returns_type = cpp.returns_type(
            self.func.returns, symint=self.symint
        ).cpp_type()
        cpp_args = [a.decl() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ["c10::DispatchKeySet dispatchKeySet"] + cpp_args
        cpp_args_str = ", ".join(cpp_args)
        if name is None:
            name = prefix + self.name(suppress_symint_suffix=suppress_symint_suffix)
        return f"{returns_type} {name}({cpp_args_str})"

    # Render the C++ definition for this signature, not including
    # the body (with curly braces)
    def defn(
        self,
        *,
        name: str | None = None,
        prefix: str = "",
        is_redispatching_fn: bool = False,
    ) -> str:
        returns_type = cpp.returns_type(
            self.func.returns, symint=self.symint
        ).cpp_type()
        cpp_args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ["c10::DispatchKeySet dispatchKeySet"] + cpp_args
        cpp_args_str = ", ".join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    def ptr_type(self) -> str:
        args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_types_str})"

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} ({args_types_str})"


# Represents group of all CppSignatures associated with a
# FunctionSchema.  Right now, that's the regular, user-visible
# signature, as well as a "faithful" signature which doesn't
# have grouping.
@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema
    signature: CppSignature
    faithful_signature: CppSignature | None
    symint_signature: CppSignature | None
    symint_faithful_signature: CppSignature | None

    def most_faithful_signature(self) -> CppSignature:
        if self.faithful_signature:
            return self.faithful_signature
        else:
            return self.signature

    def signatures(self, *, symint: bool = True) -> Iterator[CppSignature]:
        yield self.signature
        if self.faithful_signature:
            yield self.faithful_signature
        if symint:
            if self.symint_signature:
                yield self.symint_signature
            if self.symint_faithful_signature:
                yield self.symint_faithful_signature

    @staticmethod
    def from_native_function(
        f: NativeFunction, *, method: bool, fallback_binding: bool = False
    ) -> CppSignatureGroup:
        func = f.func

        def make_sig(*, faithful: bool, symint: bool) -> CppSignature:
            return CppSignature(
                func=func,
                faithful=faithful,
                symint=symint,
                method=method,
                fallback_binding=fallback_binding,
                cpp_no_default_args=f.cpp_no_default_args,
            )

        def make_sigs(*, symint: bool) -> tuple[CppSignature, CppSignature | None]:
            faithful_signature: CppSignature | None = None
            if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
                faithful_signature = make_sig(faithful=True, symint=symint)
            signature = make_sig(faithful=False, symint=symint)
            return signature, faithful_signature

        signature, faithful_signature = make_sigs(symint=False)
        symint_signature: CppSignature | None = None
        symint_faithful_signature: CppSignature | None = None
        if func.has_symint():
            symint_signature, symint_faithful_signature = make_sigs(symint=True)

        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
            symint_signature=symint_signature,
            symint_faithful_signature=symint_faithful_signature,
        )


@dataclass(frozen=True)
class DispatcherSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    # Allows you to prepend an arbitrary prefix to the signature name.
    # This is useful for parts of the codegen that generate wrappers around kernels,
    # and need to avoid naming collisions.
    prefix: str = ""

    symint: bool = True

    def arguments(self) -> list[Binding]:
        return dispatcher.arguments(self.func, symint=self.symint)

    def name(self) -> str:
        return self.prefix + dispatcher.name(self.func)

    def decl(self, name: str | None = None) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def defn(
        self, name: str | None = None, *, is_redispatching_fn: bool = False
    ) -> str:
        args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            args = ["c10::DispatchKeySet dispatchKeySet"] + args
        args_str = ", ".join(args)
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def exprs(self) -> list[Expr]:
        return [Expr(a.name, a.nctype) for a in self.arguments()]

    def returns_type(self) -> CType:
        return dispatcher.returns_type(self.func.returns, symint=self.symint)

    def ptr_type(self) -> str:
        dispatcher_args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{self.returns_type().cpp_type()} (*)({dispatcher_args_types_str})"

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        dispatcher_args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{self.returns_type().cpp_type()} ({dispatcher_args_types_str})"

    @staticmethod
    def from_schema(
        func: FunctionSchema, *, prefix: str = "", symint: bool = True
    ) -> DispatcherSignature:
        return DispatcherSignature(func, prefix, symint)


@dataclass(frozen=True)
class NativeSignature:
    # The schema this signature is derived from
    func: FunctionSchema

    symint: bool

    prefix: str = ""

    def name(self) -> str:
        return self.prefix + native.name(self.func)

    def decl(self, name: str | None = None) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    def defn(self, name: str | None = None) -> str:
        args_str = ", ".join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    def ptr_type(self) -> str:
        # don't include defaults in type signature!
        args_str = ", ".join(a.defn() for a in self.arguments())
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_str})"

    def arguments(self) -> list[Binding]:
        return native.arguments(self.func, symint=self.symint)

    def returns_type(self) -> CType:
        return native.returns_type(self.func.returns, symint=self.symint)

    def dispatcher_exprs(self) -> list[Expr]:
        return translate.translate(
            self.arguments(), dispatcher.arguments(self.func), method=False
        )


@dataclass(frozen=True)
class ViewInverseSignature:
    g: NativeFunctionsViewGroup

    def name(self) -> str:
        return functionalization.reverse_name(self.g.view, include_namespace=False)

    def decl(self) -> str:
        return_type = functionalization.returns_type(self.g.view.func)
        decls = [
            a.decl()
            for a in functionalization.op_arguments(self.g.view.func, is_reverse=True)
        ]
        return f"static {return_type.cpp_type()} {self.name()}({', '.join(decls)});"


@dataclass(frozen=True)
class StructuredImplSignature:
    g: NativeFunctionsGroup
    name: str

    def defn(self, name: str | None = None) -> str:
        args_str = ", ".join(a.defn() for a in self.arguments())
        return f"TORCH_IMPL_FUNC({self.name})({args_str})"

    def arguments(self) -> list[Binding]:
        return structured.impl_arguments(self.g)


# Helper functions


def kernel_signature(
    f: NativeFunction, backend_index: BackendIndex, *, prefix: str = ""
) -> NativeSignature | DispatcherSignature:
    # Note [External Backends Follow Dispatcher API]
    # Kernel signatures for in-tree backends follow the "native" API,
    # while kernels for out-of-tree backends follow the dispatcher API.
    # See the comments in `native.py` for details, but historically there have been
    # some small differences in schema convention between them and the Dispatcher API.
    # Any differences that require translating between the two will results in a runtime cost,
    # so we'd like to keep the differences as small as possible.
    # With external backends, we'd like to enforce that they write their kernels with schemas
    # that match the Dispatcher API directly, if they can.
    meta = backend_index.get_kernel(f)
    symint = meta is not None and meta.supports_symint()
    if symint:
        if not f.func.has_symint():
            raise AssertionError(
                f"attempted to define symint kernel for {backend_index.dispatch_key} "
                "without SymInt in schema"
            )
    if backend_index.external:
        return DispatcherSignature.from_schema(f.func, prefix=prefix, symint=symint)
    else:
        return NativeSignature(f.func, prefix=prefix, symint=symint)


# Functions only, no types
from torchgen.api import (
    cpp,
    dispatcher,
    functionalization,
    native,
    structured,
    translate,
)
<|endoftext|>#!/usr/bin/env python3

from __future__ import annotations

import os
from enum import Enum
from operator import itemgetter
from pathlib import Path
from typing import Any

import torch
from torch.jit.generate_bytecode import generate_upgraders_bytecode
from torchgen.code_template import CodeTemplate
from torchgen.operator_versions.gen_mobile_upgraders_constant import (
    MOBILE_UPGRADERS_HEADER_DESCRIPTION,
)


class ByteCode(Enum):
    instructions = 1
    constants = 2
    types = 3
    operators = 4
    register_size = 5


EXCLUDED_OP_SET = [
    "aten::full.names",
    "aten::full.out",
    "aten::full",
]

EXCLUE_UPGRADER_SET = ["full_0_4", "full_out_0_4"]

ONE_INSTRUCTION = CodeTemplate(
    """
    Instruction{OpCode::${operator_name}, ${X}, ${N}},"""
)

INSTRUCTION_LIST = CodeTemplate(
    """std::vector<Instruction>({
        ${instruction_list}
    }), // instructions list"""
)

ONE_CONSTANT = CodeTemplate(
    """
    c10::IValue(${constant}),"""
)

CONSTANT_LIST = CodeTemplate(
    """std::vector<c10::IValue>({
        ${constant_list}
    }), // constants list"""
)

CONSTANTS_LIST_EMPTY = """std::vector<c10::IValue>(), // constants list"""

ONE_TYPE = CodeTemplate("""c10::parseType("${type_str}"),""")

TYPE_LIST = CodeTemplate(
    """std::vector<c10::TypePtr>({
        ${type_list}
    }), // types list"""
)

TYPE_LIST_EMPTY = """std::vector<c10::TypePtr>(), // types list"""

ONE_OPERATOTR_STRING = CodeTemplate(
    """
    OperatorString({"${operator_name}", "${overload_name}", ${num_of_args}}),"""
)

OPERATOR_STRING_LIST = CodeTemplate(
    """
    std::vector<OperatorString>({
        ${operator_string_list}
    }), // operators list"""
)

ONE_UPGRADER_FUNCTION = CodeTemplate(
    """
    mobile::Function::registerFunc(
        "${upgrader_name}",
        ${instruction_list},
        ${constant_list},
        ${type_list},
        ${register_size}
    )"""
)

ONE_UPGRADER_SRC = CodeTemplate(
    """
    ByteCodeFunctionWithOperator({
        ${bytecode_function},
        ${operator_string_list}
    }),"""
)


ONE_UPGRADER_IN_VERSION_MAP = CodeTemplate(
    """Upgrader({${upgrader_min_version}, ${upgrader_max_version}, "${upgrader_name}", ${bytecode_func_index}})"""
)

ONE_OPERATOR_IN_VERSION_MAP = CodeTemplate(
    """
    {std::string("${operator_name}"),
        std::vector<Upgrader>({
            ${upgrader_list_in_version_map}
        })},"""
)


OPERATOR_VERSION_MAP = CodeTemplate(
    """
const std::unordered_map<std::string, std::vector<Upgrader>>
getOperatorVersionMapForMobile() {
  static std::unordered_map<std::string, std::vector<Upgrader>>
        operatorVersionMapForMobile({
            ${operator_list_in_version_map}
      });
  return operatorVersionMapForMobile;
}
"""
)


UPGRADER_CPP_SRC = CodeTemplate(
    MOBILE_UPGRADERS_HEADER_DESCRIPTION
    + """
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>

namespace torch {
namespace jit {

// clang-format off

// From operator_versions_map
${operator_version_map}

const std::vector<ByteCodeFunctionWithOperator>& getUpgraderBytecodeList() {
  auto generate_upgrader_bytecode_list = []() {
    std::vector<ByteCodeFunctionWithOperator> upgrader_function_list({
               ${upgrader_bytecode}
            });
    for (const auto& upgrader_function : upgrader_function_list) {
      for (const auto& op : upgrader_function.operators) {
        upgrader_function.function.append_operator(
            op.name,
            op.overload_name,
            op.num_specified_args);
      }
    }
    return upgrader_function_list;
  };
  static std::vector<ByteCodeFunctionWithOperator> upgraderBytecodeList =
      generate_upgrader_bytecode_list();
  return upgraderBytecodeList;
}

// clang-format on

} // namespace jit
} // namespace torch
"""
)

UPGRADER_MOBILE_FILE_NAME = "upgrader_mobile.cpp"

UPGRADER_ELEMENT = CodeTemplate(
    """\
Upgrader({${min_version}, ${max_version}, ${operator_name}, ${index}}),
"""
)

PER_OPERATOR_UPGRADER_LIST = CodeTemplate(
    """\
{
  std::string(${operator_name}),
  std::vector<Upgrader>({${upgrader_list}});
}
"""
)


def construct_instruction(instruction_list_from_yaml: list[Any]) -> str:
    instruction_list_part = [
        ONE_INSTRUCTION.substitute(
            operator_name=instruction[0],
            X=instruction[1],
            N=instruction[2],
        )
        for instruction in instruction_list_from_yaml
    ]
    return INSTRUCTION_LIST.substitute(
        instruction_list="".join(instruction_list_part).lstrip("\n")
    )


def construct_constants(constants_list_from_yaml: list[Any]) -> str:
    constants_list_part = []
    for constant_from_yaml in constants_list_from_yaml:
        convert_constant = None
        if isinstance(constant_from_yaml, str):
            # Add quotes if it's string
            convert_constant = f'"{constant_from_yaml}"'
        elif isinstance(constant_from_yaml, bool):
            convert_constant = "true" if constant_from_yaml else "false"
        elif constant_from_yaml is None:
            convert_constant = ""
        elif isinstance(constant_from_yaml, int):
            convert_constant = str(constant_from_yaml)
        else:
            raise ValueError(
                f"The type of {constant_from_yaml} is {type(constant_from_yaml)}. "
                "Please add change in construct_constants function in gen_mobile_upgraders.py."
            )
        constants_list_part.append(ONE_CONSTANT.substitute(constant=convert_constant))
    if len(constants_list_part) == 0:
        return CONSTANTS_LIST_EMPTY
    return CONSTANT_LIST.substitute(
        constant_list="".join(constants_list_part).lstrip("\n")
    )


def construct_operators(operator_list_from_yaml: list[Any]) -> str:
    operator_list_part = [
        ONE_OPERATOTR_STRING.substitute(
            operator_name=operator[0],
            overload_name=operator[1],
            num_of_args=operator[2],
        )
        for operator in operator_list_from_yaml
    ]
    return OPERATOR_STRING_LIST.substitute(
        operator_string_list="".join(operator_list_part).lstrip("\n")
    )


def construct_types(types_tr_list_from_yaml: list[Any]) -> str:
    types_tr_list_part = [
        ONE_TYPE.substitute(type_str=types_tr) for types_tr in types_tr_list_from_yaml
    ]
    if len(types_tr_list_part) == 0:
        return TYPE_LIST_EMPTY
    return TYPE_LIST.substitute(type_list="".join(types_tr_list_part).lstrip("\n"))


def construct_register_size(register_size_from_yaml: int) -> str:
    if not isinstance(register_size_from_yaml, int):
        raise ValueError(
            f"Input register size is {register_size_from_yaml} and"
            "it's type is {type(register_size_from_yaml)}. An int type is expected."
        )
    return str(register_size_from_yaml)


def construct_version_maps(
    upgrader_bytecode_function_to_index_map: dict[str, Any],
) -> str:
    version_map = torch._C._get_operator_version_map()
    sorted_version_map_ = sorted(version_map.items(), key=itemgetter(0))  # type: ignore[no-any-return]
    sorted_version_map = dict(sorted_version_map_)

    operator_list_in_version_map_part = []
    for op_name in sorted_version_map:
        upgraders_in_version_map_part = []
        # TODO: remove the skip after these two operators schemas are fixed
        if op_name in EXCLUDED_OP_SET:
            continue
        upgrader_ranges = torch._C._get_upgrader_ranges(op_name)
        upgrader_entries = sorted_version_map[op_name]
        if len(upgrader_ranges) != len(upgrader_entries):
            raise AssertionError(
                f"upgrader_ranges and upgrader_entries length mismatch for {op_name}: "
                f"{len(upgrader_ranges)} != {len(upgrader_entries)}"
            )
        for idx, upgrader_entry in enumerate(upgrader_entries):
            upgrader_name = upgrader_entry.upgrader_name
            bytecode_function_index = upgrader_bytecode_function_to_index_map[
                upgrader_name
            ]
            upgraders_in_version_map_part.append(
                ONE_UPGRADER_IN_VERSION_MAP.substitute(
                    upgrader_min_version=upgrader_ranges[idx].min_version,
                    upgrader_max_version=upgrader_ranges[idx].max_version,
                    upgrader_name=upgrader_name,
                    bytecode_func_index=bytecode_function_index,
                )
            )
        operator_list_in_version_map_part.append(
            ONE_OPERATOR_IN_VERSION_MAP.substitute(
                operator_name=op_name,
                upgrader_list_in_version_map="".join(upgraders_in_version_map_part),
            )
        )
    return OPERATOR_VERSION_MAP.substitute(
        operator_list_in_version_map="".join(operator_list_in_version_map_part).lstrip(
            "\n"
        )
    )


def get_upgrader_bytecode_function_to_index_map(
    upgrader_dict: list[dict[str, Any]],
) -> dict[str, Any]:
    upgrader_bytecode_function_to_index_map = {}
    index = 0
    for upgrader_bytecode in upgrader_dict:
        for upgrader_name in upgrader_bytecode:
            if upgrader_name in EXCLUE_UPGRADER_SET:
                continue
            upgrader_bytecode_function_to_index_map[upgrader_name] = index
            index += 1
    return upgrader_bytecode_function_to_index_map


def write_cpp(cpp_path: str, upgrader_dict: list[dict[str, Any]]) -> None:
    upgrader_bytecode_function_to_index_map = (
        get_upgrader_bytecode_function_to_index_map(upgrader_dict)
    )
    version_map_src = construct_version_maps(upgrader_bytecode_function_to_index_map)
    all_upgrader_src_string = []
    for upgrader_bytecode in upgrader_dict:
        for upgrader_name, bytecode in upgrader_bytecode.items():
            # TODO: remove the skip after these two operators schemas are fixed
            if upgrader_name in EXCLUE_UPGRADER_SET:
                continue
            instruction_list_str = ""
            constant_list_str = ""
            type_list_str = ""
            register_size_str = ""
            operator_list_str = ""
            for table_name, contents in bytecode.items():
                element = ByteCode[table_name]
                if element is ByteCode.instructions:
                    instruction_list_str = construct_instruction(contents)
                elif element is ByteCode.constants:
                    constant_list_str = construct_constants(contents)
                elif element is ByteCode.operators:
                    operator_list_str = construct_operators(contents)
                elif element is ByteCode.types:
                    type_list_str = construct_types(contents)
                elif element is ByteCode.register_size:
                    register_size_str = construct_register_size(contents)

            one_upgrader_function_string = ONE_UPGRADER_FUNCTION.substitute(
                upgrader_name=upgrader_name,
                instruction_list=instruction_list_str,
                constant_list=constant_list_str,
                type_list=type_list_str,
                register_size=register_size_str,
            )
            one_upgrader_src_string = ONE_UPGRADER_SRC.substitute(
                bytecode_function=one_upgrader_function_string.lstrip("\n"),
                operator_string_list=operator_list_str.lstrip("\n"),
            )
            all_upgrader_src_string.append(one_upgrader_src_string)

    upgrader_file_content = UPGRADER_CPP_SRC.substitute(
        operator_version_map=version_map_src,
        upgrader_bytecode="".join(all_upgrader_src_string).lstrip("\n"),
    )
    print("writing file to : ", cpp_path + "/" + UPGRADER_MOBILE_FILE_NAME)
    with open(os.path.join(cpp_path, UPGRADER_MOBILE_FILE_NAME), "wb") as out_file:
        out_file.write(upgrader_file_content.encode("utf-8"))


def sort_upgrader(upgrader_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_upgrader_list = sorted(
        upgrader_list, key=lambda one_upgrader: next(iter(one_upgrader))
    )
    return sorted_upgrader_list


def main() -> None:
    upgrader_list = generate_upgraders_bytecode()
    sorted_upgrader_list = sort_upgrader(upgrader_list)
    for up in sorted_upgrader_list:
        print("after sort upgrader : ", next(iter(up)))

    pytorch_dir = Path(__file__).resolve().parents[2]
    upgrader_path = pytorch_dir / "torch" / "csrc" / "jit" / "mobile"
    write_cpp(str(upgrader_path), sorted_upgrader_list)


if __name__ == "__main__":
    main()
<|endoftext|>"""
This module exports all latin and greek letters as Symbols, so you can
conveniently do

    >>> from sympy.abc import x, y

instead of the slightly more clunky-looking

    >>> from sympy import symbols
    >>> x, y = symbols('x y')

Caveats
=======

1. As of the time of writing this, the names ``O``, ``S``, ``I``, ``N``,
``E``, and ``Q`` are colliding with names defined in SymPy. If you import them
from both ``sympy.abc`` and ``sympy``, the second import will "win".
This is an issue only for * imports, which should only be used for short-lived
code such as interactive sessions and throwaway scripts that do not survive
until the next SymPy upgrade, where ``sympy`` may contain a different set of
names.

2. This module does not define symbol names on demand, i.e.
``from sympy.abc import foo`` will be reported as an error because
``sympy.abc`` does not contain the name ``foo``. To get a symbol named ``foo``,
you still need to use ``Symbol('foo')`` or ``symbols('foo')``.
You can freely mix usage of ``sympy.abc`` and ``Symbol``/``symbols``, though
sticking with one and only one way to get the symbols does tend to make the code
more readable.

The module also defines some special names to help detect which names clash
with the default SymPy namespace.

``_clash1`` defines all the single letter variables that clash with
SymPy objects; ``_clash2`` defines the multi-letter clashing symbols;
and ``_clash`` is the union of both. These can be passed for ``locals``
during sympification if one desires Symbols rather than the non-Symbol
objects for those names.

Examples
========

>>> from sympy import S
>>> from sympy.abc import _clash1, _clash2, _clash
>>> S("Q & C", locals=_clash1)
C & Q
>>> S('pi(x)', locals=_clash2)
pi(x)
>>> S('pi(C, Q)', locals=_clash)
pi(C, Q)

"""
from __future__ import annotations
from typing import Any

import string

from .core import Symbol, symbols
from .core.alphabets import greeks
from sympy.parsing.sympy_parser import null

##### Symbol definitions #####

# Implementation note: The easiest way to avoid typos in the symbols()
# parameter is to copy it from the left-hand side of the assignment.

a, b, c, d, e, f, g, h, i, j = symbols('a, b, c, d, e, f, g, h, i, j')
k, l, m, n, o, p, q, r, s, t = symbols('k, l, m, n, o, p, q, r, s, t')
u, v, w, x, y, z = symbols('u, v, w, x, y, z')

A, B, C, D, E, F, G, H, I, J = symbols('A, B, C, D, E, F, G, H, I, J')
K, L, M, N, O, P, Q, R, S, T = symbols('K, L, M, N, O, P, Q, R, S, T')
U, V, W, X, Y, Z = symbols('U, V, W, X, Y, Z')

alpha, beta, gamma, delta = symbols('alpha, beta, gamma, delta')
epsilon, zeta, eta, theta = symbols('epsilon, zeta, eta, theta')
iota, kappa, lamda, mu = symbols('iota, kappa, lamda, mu')
nu, xi, omicron, pi = symbols('nu, xi, omicron, pi')
rho, sigma, tau, upsilon = symbols('rho, sigma, tau, upsilon')
phi, chi, psi, omega = symbols('phi, chi, psi, omega')


##### Clashing-symbols diagnostics #####

# We want to know which names in SymPy collide with those in here.
# This is mostly for diagnosing SymPy's namespace during SymPy development.

_latin = list(string.ascii_letters)
# QOSINE should not be imported as they clash; gamma, pi and zeta clash, too
_greek = list(greeks) # make a copy, so we can mutate it
# Note: We import lamda since lambda is a reserved keyword in Python
_greek.remove("lambda")
_greek.append("lamda")

ns: dict[str, Any] = {}
exec('from sympy import *', ns)
_clash1: dict[str, Any] = {}
_clash2: dict[str, Any] = {}
while ns:
    _k, _ = ns.popitem()
    if _k in _greek:
        _clash2[_k] = null
        _greek.remove(_k)
    elif _k in _latin:
        _clash1[_k] = null
        _latin.remove(_k)
_clash = {}
_clash.update(_clash1)
_clash.update(_clash2)

del _latin, _greek, Symbol, _k, null
<|endoftext|>"""
SymPy is a Python library for symbolic mathematics. It aims to become a
full-featured computer algebra system (CAS) while keeping the code as simple
as possible in order to be comprehensible and easily extensible.  SymPy is
written entirely in Python. It depends on mpmath, and other external libraries
may be optionally for things like plotting support.

See the webpage for more information and documentation:

    https://sympy.org

"""


# Keep this in sync with setup.py/pyproject.toml
import sys
if sys.version_info < (3, 9):
    raise ImportError("Python version 3.9 or above is required for SymPy.")
del sys


try:
    import mpmath
except ImportError:
    raise ImportError("SymPy now depends on mpmath as an external library. "
    "See https://docs.sympy.org/latest/install.html#mpmath for more information.")

del mpmath

from sympy.release import __version__
from sympy.core.cache import lazy_function

if 'dev' in __version__:
    def enable_warnings():
        import warnings
        warnings.filterwarnings('default',   '.*',   DeprecationWarning, module='sympy.*')
        del warnings
    enable_warnings()
    del enable_warnings


def __sympy_debug():
    # helper function so we don't import os globally
    import os
    debug_str = os.getenv('SYMPY_DEBUG', 'False')
    if debug_str in ('True', 'False'):
        return eval(debug_str)
    else:
        raise RuntimeError("unrecognized value for SYMPY_DEBUG: %s" %
                           debug_str)
# Fails py2 test if using type hinting
SYMPY_DEBUG = __sympy_debug()  # type: bool


from .core import (sympify, SympifyError, cacheit, Basic, Atom,
        preorder_traversal, S, Expr, AtomicExpr, UnevaluatedExpr, Symbol,
        Wild, Dummy, symbols, var, Number, Float, Rational, Integer,
        NumberSymbol, RealNumber, igcd, ilcm, seterr, E, I, nan, oo, pi, zoo,
        AlgebraicNumber, comp, mod_inverse, Pow, integer_nthroot, integer_log,
        trailing, Mul, prod, Add, Mod, Rel, Eq, Ne, Lt, Le, Gt, Ge, Equality,
        GreaterThan, LessThan, Unequality, StrictGreaterThan, StrictLessThan,
        vectorize, Lambda, WildFunction, Derivative, diff, FunctionClass,
        Function, Subs, expand, PoleError, count_ops, expand_mul, expand_log,
        expand_func, expand_trig, expand_complex, expand_multinomial, nfloat,
        expand_power_base, expand_power_exp, arity, PrecisionExhausted, N,
        evalf, Tuple, Dict, gcd_terms, factor_terms, factor_nc, evaluate,
        Catalan, EulerGamma, GoldenRatio, TribonacciConstant, bottom_up, use,
        postorder_traversal, default_sort_key, ordered, num_digits)

from .logic import (to_cnf, to_dnf, to_nnf, And, Or, Not, Xor, Nand, Nor,
        Implies, Equivalent, ITE, POSform, SOPform, simplify_logic, bool_map,
        true, false, satisfiable)

from .assumptions import (AppliedPredicate, Predicate, AssumptionsContext,
        assuming, Q, ask, register_handler, remove_handler, refine)

from .polys import (Poly, PurePoly, poly_from_expr, parallel_poly_from_expr,
        degree, total_degree, degree_list, LC, LM, LT, pdiv, prem, pquo,
        pexquo, div, rem, quo, exquo, half_gcdex, gcdex, invert,
        subresultants, resultant, discriminant, cofactors, gcd_list, gcd,
        lcm_list, lcm, terms_gcd, trunc, monic, content, primitive, compose,
        decompose, sturm, gff_list, gff, sqf_norm, sqf_part, sqf_list, sqf,
        factor_list, factor, intervals, refine_root, count_roots, all_roots,
        real_roots, nroots, ground_roots, nth_power_roots_poly, cancel,
        reduced, groebner, is_zero_dimensional, GroebnerBasis, poly,
        symmetrize, horner, interpolate, rational_interpolate, viete, together,
        BasePolynomialError, ExactQuotientFailed, PolynomialDivisionFailed,
        OperationNotSupported, HeuristicGCDFailed, HomomorphismFailed,
        IsomorphismFailed, ExtraneousFactors, EvaluationFailed,
        RefinementFailed, CoercionFailed, NotInvertible, NotReversible,
        NotAlgebraic, DomainError, PolynomialError, UnificationFailed,
        GeneratorsError, GeneratorsNeeded, ComputationFailed,
        UnivariatePolynomialError, MultivariatePolynomialError,
        PolificationFailed, OptionError, FlagError, minpoly,
        minimal_polynomial, primitive_element, field_isomorphism,
        to_number_field, isolate, round_two, prime_decomp, prime_valuation,
        galois_group, itermonomials, Monomial, lex, grlex,
        grevlex, ilex, igrlex, igrevlex, CRootOf, rootof, RootOf,
        ComplexRootOf, RootSum, roots, Domain, FiniteField, IntegerRing,
        RationalField, RealField, ComplexField, PythonFiniteField,
        GMPYFiniteField, PythonIntegerRing, GMPYIntegerRing, PythonRational,
        GMPYRationalField, AlgebraicField, PolynomialRing, FractionField,
        ExpressionDomain, FF_python, FF_gmpy, ZZ_python, ZZ_gmpy, QQ_python,
        QQ_gmpy, GF, FF, ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX, EXRAW,
        construct_domain, swinnerton_dyer_poly, cyclotomic_poly,
        symmetric_poly, random_poly, interpolating_poly, jacobi_poly,
        chebyshevt_poly, chebyshevu_poly, hermite_poly, hermite_prob_poly,
        legendre_poly, laguerre_poly, apart, apart_list, assemble_partfrac_list,
        Options, ring, xring, vring, sring, field, xfield, vfield, sfield)

from .series import (Order, O, limit, Limit, gruntz, series, approximants,
        residue, EmptySequence, SeqPer, SeqFormula, sequence, SeqAdd, SeqMul,
        fourier_series, fps, difference_delta, limit_seq)

from .functions import (factorial, factorial2, rf, ff, binomial,
        RisingFactorial, FallingFactorial, subfactorial, carmichael,
        fibonacci, lucas, motzkin, tribonacci, harmonic, bernoulli, bell, euler,
        catalan, genocchi, andre, partition, divisor_sigma, legendre_symbol,
        jacobi_symbol, kronecker_symbol, mobius, primenu, primeomega,
        totient, reduced_totient, primepi, sqrt, root, Min, Max, Id,
        real_root, Rem, cbrt, re, im, sign, Abs, conjugate, arg, polar_lift,
        periodic_argument, unbranched_argument, principal_branch, transpose,
        adjoint, polarify, unpolarify, sin, cos, tan, sec, csc, cot, sinc,
        asin, acos, atan, asec, acsc, acot, atan2, exp_polar, exp, ln, log,
        LambertW, sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh,
        acoth, asech, acsch, floor, ceiling, frac, Piecewise, piecewise_fold,
        piecewise_exclusive, erf, erfc, erfi, erf2, erfinv, erfcinv, erf2inv,
        Ei, expint, E1, li, Li, Si, Ci, Shi, Chi, fresnels, fresnelc, gamma,
        lowergamma, uppergamma, polygamma, loggamma, digamma, trigamma,
        multigamma, dirichlet_eta, zeta, lerchphi, polylog, stieltjes, Eijk,
        LeviCivita, KroneckerDelta, SingularityFunction, DiracDelta, Heaviside,
        bspline_basis, bspline_basis_set, interpolating_spline, besselj,
        bessely, besseli, besselk, hankel1, hankel2, jn, yn, jn_zeros, hn1,
        hn2, airyai, airybi, airyaiprime, airybiprime, marcumq, hyper,
        meijerg, appellf1, legendre, assoc_legendre, hermite, hermite_prob,
        chebyshevt, chebyshevu, chebyshevu_root, chebyshevt_root, laguerre,
        assoc_laguerre, gegenbauer, jacobi, jacobi_normalized, Ynm, Ynm_c,
        Znm, elliptic_k, elliptic_f, elliptic_e, elliptic_pi, beta, mathieus,
        mathieuc, mathieusprime, mathieucprime, riemann_xi, betainc, betainc_regularized)

from .ntheory import (nextprime, prevprime, prime, primerange,
        randprime, Sieve, sieve, primorial, cycle_length, composite,
        compositepi, isprime, divisors, proper_divisors, factorint,
        multiplicity, perfect_power, factor_cache, pollard_pm1, pollard_rho, primefactors,
        divisor_count, proper_divisor_count,
        factorrat,
        mersenne_prime_exponent, is_perfect, is_mersenne_prime, is_abundant,
        is_deficient, is_amicable, is_carmichael, abundance, npartitions, is_primitive_root,
        is_quad_residue, n_order, sqrt_mod,
        quadratic_residues, primitive_root, nthroot_mod, is_nthpow_residue,
        sqrt_mod_iter, discrete_log, quadratic_congruence,
        binomial_coefficients, binomial_coefficients_list,
        multinomial_coefficients, continued_fraction_periodic,
        continued_fraction_iterator, continued_fraction_reduce,
        continued_fraction_convergents, continued_fraction, egyptian_fraction)

from .concrete import product, Product, summation, Sum

from .discrete import (fft, ifft, ntt, intt, fwht, ifwht, mobius_transform,
        inverse_mobius_transform, convolution, covering_product,
        intersecting_product)

from .simplify import (simplify, hypersimp, hypersimilar, logcombine,
        separatevars, posify, besselsimp, kroneckersimp, signsimp,
        nsimplify, FU, fu, sqrtdenest, cse, epath, EPath, hyperexpand,
        collect, rcollect, radsimp, collect_const, fraction, numer, denom,
        trigsimp, exptrigsimp, powsimp, powdenest, combsimp, gammasimp,
        ratsimp, ratsimpmodprime)

from .sets import (Set, Interval, Union, EmptySet, FiniteSet, ProductSet,
        Intersection, DisjointUnion, imageset, Complement, SymmetricDifference, ImageSet,
        Range, ComplexRegion, Complexes, Reals, Contains, ConditionSet, Ordinal,
        OmegaPower, ord0, PowerSet, Naturals, Naturals0, UniversalSet,
        Integers, Rationals)

from .solvers import (solve, solve_linear_system, solve_linear_system_LU,
        solve_undetermined_coeffs, nsolve, solve_linear, checksol, det_quick,
        inv_quick, check_assumptions, failing_assumptions, diophantine,
        rsolve, rsolve_poly, rsolve_ratio, rsolve_hyper, checkodesol,
        classify_ode, dsolve, homogeneous_order, solve_poly_system, factor_system,
        solve_triangulated, pde_separate, pde_separate_add, pde_separate_mul,
        pdsolve, classify_pde, checkpdesol, ode_order, reduce_inequalities,
        reduce_abs_inequality, reduce_abs_inequalities, solve_poly_inequality,
        solve_rational_inequalities, solve_univariate_inequality, decompogen,
        solveset, linsolve, linear_eq_to_matrix, nonlinsolve, substitution)

from .matrices import (ShapeError, NonSquareMatrixError, GramSchmidt,
        casoratian, diag, eye, hessian, jordan_cell, list2numpy, matrix2numpy,
        matrix_multiply_elementwise, ones, randMatrix, rot_axis1, rot_axis2,
        rot_axis3, symarray, wronskian, zeros, MutableDenseMatrix,
        DeferredVector, MatrixBase, Matrix, MutableMatrix,
        MutableSparseMatrix, banded, ImmutableDenseMatrix,
        ImmutableSparseMatrix, ImmutableMatrix, SparseMatrix, MatrixSlice,
        BlockDiagMatrix, BlockMatrix, FunctionMatrix, Identity, Inverse,
        MatAdd, MatMul, MatPow, MatrixExpr, MatrixSymbol, Trace, Transpose,
        ZeroMatrix, OneMatrix, blockcut, block_collapse, matrix_symbols,
        Adjoint, hadamard_product, HadamardProduct, HadamardPower,
        Determinant, det, diagonalize_vector, DiagMatrix, DiagonalMatrix,
        DiagonalOf, trace, DotProduct, kronecker_product, KroneckerProduct,
        PermutationMatrix, MatrixPermute, Permanent, per, rot_ccw_axis1,
        rot_ccw_axis2, rot_ccw_axis3, rot_givens)

from .geometry import (Point, Point2D, Point3D, Line, Ray, Segment, Line2D,
        Segment2D, Ray2D, Line3D, Segment3D, Ray3D, Plane, Ellipse, Circle,
        Polygon, RegularPolygon, Triangle, rad, deg, are_similar, centroid,
        convex_hull, idiff, intersection, closest_points, farthest_points,
        GeometryError, Curve, Parabola)

from .utilities import (flatten, group, take, subsets, variations,
        numbered_symbols, cartes, capture, dict_merge, prefixes, postfixes,
        sift, topological_sort, unflatten, has_dups, has_variety, reshape,
        rotations, filldedent, lambdify,
        threaded, xthreaded, public, memoize_property, timed)

from .integrals import (integrate, Integral, line_integrate, mellin_transform,
        inverse_mellin_transform, MellinTransform, InverseMellinTransform,
        laplace_transform, laplace_correspondence, laplace_initial_conds,
        inverse_laplace_transform, LaplaceTransform,
        InverseLaplaceTransform, fourier_transform, inverse_fourier_transform,
        FourierTransform, InverseFourierTransform, sine_transform,
        inverse_sine_transform, SineTransform, InverseSineTransform,
        cosine_transform, inverse_cosine_transform, CosineTransform,
        InverseCosineTransform, hankel_transform, inverse_hankel_transform,
        HankelTransform, InverseHankelTransform, singularityintegrate)

from .tensor import (IndexedBase, Idx, Indexed, get_contraction_structure,
        get_indices, shape, MutableDenseNDimArray, ImmutableDenseNDimArray,
        MutableSparseNDimArray, ImmutableSparseNDimArray, NDimArray,
        tensorproduct, tensorcontraction, tensordiagonal, derive_by_array,
        permutedims, Array, DenseNDimArray, SparseNDimArray)

from .parsing import parse_expr

from .calculus import (euler_equations, singularities, is_increasing,
        is_strictly_increasing, is_decreasing, is_strictly_decreasing,
        is_monotonic, finite_diff_weights, apply_finite_diff,
        differentiate_finite, periodicity, not_empty_in, AccumBounds,
        is_convex, stationary_points, minimum, maximum)

from .algebras import Quaternion

from .printing import (pager_print, pretty, pretty_print, pprint,
        pprint_use_unicode, pprint_try_use_unicode, latex, print_latex,
        multiline_latex, mathml, print_mathml, python, print_python, pycode,
        ccode, print_ccode, smtlib_code, glsl_code, print_glsl, cxxcode, fcode,
        print_fcode, rcode, print_rcode, jscode, print_jscode, julia_code,
        mathematica_code, octave_code, rust_code, print_gtk, preview, srepr,
        print_tree, StrPrinter, sstr, sstrrepr, TableForm, dotprint,
        maple_code, print_maple_code)

test = lazy_function('sympy.testing.runtests_pytest', 'test')
doctest = lazy_function('sympy.testing.runtests', 'doctest')

# This module causes conflicts with other modules:
# from .stats import *
# Adds about .04-.05 seconds of import time
# from combinatorics import *
# This module is slow to import:
#from physics import units
from .plotting import plot, textplot, plot_backends, plot_implicit, plot_parametric
from .interactive import init_session, init_printing, interactive_traversal

evalf._create_evalf_table()

__all__ = [
    '__version__',

    # sympy.core
    'sympify', 'SympifyError', 'cacheit', 'Basic', 'Atom',
    'preorder_traversal', 'S', 'Expr', 'AtomicExpr', 'UnevaluatedExpr',
    'Symbol', 'Wild', 'Dummy', 'symbols', 'var', 'Number', 'Float',
    'Rational', 'Integer', 'NumberSymbol', 'RealNumber', 'igcd', 'ilcm',
    'seterr', 'E', 'I', 'nan', 'oo', 'pi', 'zoo', 'AlgebraicNumber', 'comp',
    'mod_inverse', 'Pow', 'integer_nthroot', 'integer_log', 'trailing', 'Mul', 'prod',
    'Add', 'Mod', 'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality',
    'GreaterThan', 'LessThan', 'Unequality', 'StrictGreaterThan',
    'StrictLessThan', 'vectorize', 'Lambda', 'WildFunction', 'Derivative',
    'diff', 'FunctionClass', 'Function', 'Subs', 'expand', 'PoleError',
    'count_ops', 'expand_mul', 'expand_log', 'expand_func', 'expand_trig',
    'expand_complex', 'expand_multinomial', 'nfloat', 'expand_power_base',
    'expand_power_exp', 'arity', 'PrecisionExhausted', 'N', 'evalf', 'Tuple',
    'Dict', 'gcd_terms', 'factor_terms', 'factor_nc', 'evaluate', 'Catalan',
    'EulerGamma', 'GoldenRatio', 'TribonacciConstant', 'bottom_up', 'use',
    'postorder_traversal', 'default_sort_key', 'ordered', 'num_digits',

    # sympy.logic
    'to_cnf', 'to_dnf', 'to_nnf', 'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
    'Implies', 'Equivalent', 'ITE', 'POSform', 'SOPform', 'simplify_logic',
    'bool_map', 'true', 'false', 'satisfiable',

    # sympy.assumptions
    'AppliedPredicate', 'Predicate', 'AssumptionsContext', 'assuming', 'Q',
    'ask', 'register_handler', 'remove_handler', 'refine',

    # sympy.polys
    'Poly', 'PurePoly', 'poly_from_expr', 'parallel_poly_from_expr', 'degree',
    'total_degree', 'degree_list', 'LC', 'LM', 'LT', 'pdiv', 'prem', 'pquo',
    'pexquo', 'div', 'rem', 'quo', 'exquo', 'half_gcdex', 'gcdex', 'invert',
    'subresultants', 'resultant', 'discriminant', 'cofactors', 'gcd_list',
    'gcd', 'lcm_list', 'lcm', 'terms_gcd', 'trunc', 'monic', 'content',
    'primitive', 'compose', 'decompose', 'sturm', 'gff_list', 'gff',
    'sqf_norm', 'sqf_part', 'sqf_list', 'sqf', 'factor_list', 'factor',
    'intervals', 'refine_root', 'count_roots', 'all_roots', 'real_roots',
    'nroots', 'ground_roots', 'nth_power_roots_poly', 'cancel', 'reduced',
    'groebner', 'is_zero_dimensional', 'GroebnerBasis', 'poly', 'symmetrize',
    'horner', 'interpolate', 'rational_interpolate', 'viete', 'together',
    'BasePolynomialError', 'ExactQuotientFailed', 'PolynomialDivisionFailed',
    'OperationNotSupported', 'HeuristicGCDFailed', 'HomomorphismFailed',
    'IsomorphismFailed', 'ExtraneousFactors', 'EvaluationFailed',
    'RefinementFailed', 'CoercionFailed', 'NotInvertible', 'NotReversible',
    'NotAlgebraic', 'DomainError', 'PolynomialError', 'UnificationFailed',
    'GeneratorsError', 'GeneratorsNeeded', 'ComputationFailed',
    'UnivariatePolynomialError', 'MultivariatePolynomialError',
    'PolificationFailed', 'OptionError', 'FlagError', 'minpoly',
    'minimal_polynomial', 'primitive_element', 'field_isomorphism',
    'to_number_field', 'isolate', 'round_two', 'prime_decomp',
    'prime_valuation', 'galois_group', 'itermonomials', 'Monomial', 'lex', 'grlex',
    'grevlex', 'ilex', 'igrlex', 'igrevlex', 'CRootOf', 'rootof', 'RootOf',
    'ComplexRootOf', 'RootSum', 'roots', 'Domain', 'FiniteField',
    'IntegerRing', 'RationalField', 'RealField', 'ComplexField',
    'PythonFiniteField', 'GMPYFiniteField', 'PythonIntegerRing',
    'GMPYIntegerRing', 'PythonRational', 'GMPYRationalField',
    'AlgebraicField', 'PolynomialRing', 'FractionField', 'ExpressionDomain',
    'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy', 'QQ_python', 'QQ_gmpy',
    'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR', 'CC', 'EX', 'EXRAW',
    'construct_domain', 'swinnerton_dyer_poly', 'cyclotomic_poly',
    'symmetric_poly', 'random_poly', 'interpolating_poly', 'jacobi_poly',
    'chebyshevt_poly', 'chebyshevu_poly', 'hermite_poly', 'hermite_prob_poly',
    'legendre_poly', 'laguerre_poly', 'apart', 'apart_list', 'assemble_partfrac_list',
    'Options', 'ring', 'xring', 'vring', 'sring', 'field', 'xfield', 'vfield',
    'sfield',

    # sympy.series
    'Order', 'O', 'limit', 'Limit', 'gruntz', 'series', 'approximants',
    'residue', 'EmptySequence', 'SeqPer', 'SeqFormula', 'sequence', 'SeqAdd',
    'SeqMul', 'fourier_series', 'fps', 'difference_delta', 'limit_seq',

    # sympy.functions
    'factorial', 'factorial2', 'rf', 'ff', 'binomial', 'RisingFactorial',
    'FallingFactorial', 'subfactorial', 'carmichael', 'fibonacci', 'lucas',
    'motzkin', 'tribonacci', 'harmonic', 'bernoulli', 'bell', 'euler', 'catalan',
    'genocchi', 'andre', 'partition',  'divisor_sigma', 'legendre_symbol', 'jacobi_symbol',
    'kronecker_symbol', 'mobius', 'primenu', 'primeomega', 'totient', 'primepi',
    'reduced_totient', 'sqrt', 'root', 'Min', 'Max', 'Id', 'real_root',
    'Rem', 'cbrt', 're', 'im', 'sign', 'Abs', 'conjugate', 'arg', 'polar_lift',
    'periodic_argument', 'unbranched_argument', 'principal_branch',
    'transpose', 'adjoint', 'polarify', 'unpolarify', 'sin', 'cos', 'tan',
    'sec', 'csc', 'cot', 'sinc', 'asin', 'acos', 'atan', 'asec', 'acsc',
    'acot', 'atan2', 'exp_polar', 'exp', 'ln', 'log', 'LambertW', 'sinh',
    'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh',
    'acoth', 'asech', 'acsch', 'floor', 'ceiling', 'frac', 'Piecewise',
    'piecewise_fold', 'piecewise_exclusive', 'erf', 'erfc', 'erfi', 'erf2',
    'erfinv', 'erfcinv', 'erf2inv', 'Ei', 'expint', 'E1', 'li', 'Li', 'Si',
    'Ci', 'Shi', 'Chi', 'fresnels', 'fresnelc', 'gamma', 'lowergamma',
    'uppergamma', 'polygamma', 'loggamma', 'digamma', 'trigamma', 'multigamma',
    'dirichlet_eta', 'zeta', 'lerchphi', 'polylog', 'stieltjes', 'Eijk', 'LeviCivita',
    'KroneckerDelta', 'SingularityFunction', 'DiracDelta', 'Heaviside',
    'bspline_basis', 'bspline_basis_set', 'interpolating_spline', 'besselj',
    'bessely', 'besseli', 'besselk', 'hankel1', 'hankel2', 'jn', 'yn',
    'jn_zeros', 'hn1', 'hn2', 'airyai', 'airybi', 'airyaiprime',
    'airybiprime', 'marcumq', 'hyper', 'meijerg', 'appellf1', 'legendre',
    'assoc_legendre', 'hermite', 'hermite_prob', 'chebyshevt', 'chebyshevu',
    'chebyshevu_root', 'chebyshevt_root', 'laguerre', 'assoc_laguerre',
    'gegenbauer', 'jacobi', 'jacobi_normalized', 'Ynm', 'Ynm_c', 'Znm',
    'elliptic_k', 'elliptic_f', 'elliptic_e', 'elliptic_pi', 'beta',
    'mathieus', 'mathieuc', 'mathieusprime', 'mathieucprime', 'riemann_xi','betainc',
    'betainc_regularized',

    # sympy.ntheory
    'nextprime', 'prevprime', 'prime', 'primerange', 'randprime',
    'Sieve', 'sieve', 'primorial', 'cycle_length', 'composite', 'compositepi',
    'isprime', 'divisors', 'proper_divisors', 'factorint', 'multiplicity',
    'perfect_power', 'pollard_pm1', 'factor_cache', 'pollard_rho', 'primefactors',
    'divisor_count', 'proper_divisor_count',
    'factorrat',
    'mersenne_prime_exponent', 'is_perfect', 'is_mersenne_prime',
    'is_abundant', 'is_deficient', 'is_amicable', 'is_carmichael', 'abundance',
    'npartitions',
    'is_primitive_root', 'is_quad_residue',
    'n_order', 'sqrt_mod', 'quadratic_residues',
    'primitive_root', 'nthroot_mod', 'is_nthpow_residue', 'sqrt_mod_iter',
    'discrete_log', 'quadratic_congruence', 'binomial_coefficients',
    'binomial_coefficients_list', 'multinomial_coefficients',
    'continued_fraction_periodic', 'continued_fraction_iterator',
    'continued_fraction_reduce', 'continued_fraction_convergents',
    'continued_fraction', 'egyptian_fraction',

    # sympy.concrete
    'product', 'Product', 'summation', 'Sum',

    # sympy.discrete
    'fft', 'ifft', 'ntt', 'intt', 'fwht', 'ifwht', 'mobius_transform',
    'inverse_mobius_transform', 'convolution', 'covering_product',
    'intersecting_product',

    # sympy.simplify
    'simplify', 'hypersimp', 'hypersimilar', 'logcombine', 'separatevars',
    'posify', 'besselsimp', 'kroneckersimp', 'signsimp',
    'nsimplify', 'FU', 'fu', 'sqrtdenest', 'cse', 'epath', 'EPath',
    'hyperexpand', 'collect', 'rcollect', 'radsimp', 'collect_const',
    'fraction', 'numer', 'denom', 'trigsimp', 'exptrigsimp', 'powsimp',
    'powdenest', 'combsimp', 'gammasimp', 'ratsimp', 'ratsimpmodprime',

    # sympy.sets
    'Set', 'Interval', 'Union', 'EmptySet', 'FiniteSet', 'ProductSet',
    'Intersection', 'imageset', 'DisjointUnion', 'Complement', 'SymmetricDifference',
    'ImageSet', 'Range', 'ComplexRegion', 'Reals', 'Contains', 'ConditionSet',
    'Ordinal', 'OmegaPower', 'ord0', 'PowerSet', 'Naturals',
    'Naturals0', 'UniversalSet', 'Integers', 'Rationals', 'Complexes',

    # sympy.solvers
    'solve', 'solve_linear_system', 'solve_linear_system_LU',
    'solve_undetermined_coeffs', 'nsolve', 'solve_linear', 'checksol',
    'det_quick', 'inv_quick', 'check_assumptions', 'failing_assumptions',
    'diophantine', 'rsolve', 'rsolve_poly', 'rsolve_ratio', 'rsolve_hyper',
    'checkodesol', 'classify_ode', 'dsolve', 'homogeneous_order',
    'solve_poly_system', 'factor_system', 'solve_triangulated', 'pde_separate',
    'pde_separate_add', 'pde_separate_mul', 'pdsolve', 'classify_pde',
    'checkpdesol', 'ode_order', 'reduce_inequalities',
    'reduce_abs_inequality', 'reduce_abs_inequalities',
    'solve_poly_inequality', 'solve_rational_inequalities',
    'solve_univariate_inequality', 'decompogen', 'solveset', 'linsolve',
    'linear_eq_to_matrix', 'nonlinsolve', 'substitution',

    # sympy.matrices
    'ShapeError', 'NonSquareMatrixError', 'GramSchmidt', 'casoratian', 'diag',
    'eye', 'hessian', 'jordan_cell', 'list2numpy', 'matrix2numpy',
    'matrix_multiply_elementwise', 'ones', 'randMatrix', 'rot_axis1',
    'rot_axis2', 'rot_axis3', 'symarray', 'wronskian', 'zeros',
    'MutableDenseMatrix', 'DeferredVector', 'MatrixBase', 'Matrix',
    'MutableMatrix', 'MutableSparseMatrix', 'banded', 'ImmutableDenseMatrix',
    'ImmutableSparseMatrix', 'ImmutableMatrix', 'SparseMatrix', 'MatrixSlice',
    'BlockDiagMatrix', 'BlockMatrix', 'FunctionMatrix', 'Identity', 'Inverse',
    'MatAdd', 'MatMul', 'MatPow', 'MatrixExpr', 'MatrixSymbol', 'Trace',
    'Transpose', 'ZeroMatrix', 'OneMatrix', 'blockcut', 'block_collapse',
    'matrix_symbols', 'Adjoint', 'hadamard_product', 'HadamardProduct',
    'HadamardPower', 'Determinant', 'det', 'diagonalize_vector', 'DiagMatrix',
    'DiagonalMatrix', 'DiagonalOf', 'trace', 'DotProduct',
    'kronecker_product', 'KroneckerProduct', 'PermutationMatrix',
    'MatrixPermute', 'Permanent', 'per', 'rot_ccw_axis1', 'rot_ccw_axis2',
    'rot_ccw_axis3', 'rot_givens',

    # sympy.geometry
    'Point', 'Point2D', 'Point3D', 'Line', 'Ray', 'Segment', 'Line2D',
    'Segment2D', 'Ray2D', 'Line3D', 'Segment3D', 'Ray3D', 'Plane', 'Ellipse',
    'Circle', 'Polygon', 'RegularPolygon', 'Triangle', 'rad', 'deg',
    'are_similar', 'centroid', 'convex_hull', 'idiff', 'intersection',
    'closest_points', 'farthest_points', 'GeometryError', 'Curve', 'Parabola',

    # sympy.utilities
    'flatten', 'group', 'take', 'subsets', 'variations', 'numbered_symbols',
    'cartes', 'capture', 'dict_merge', 'prefixes', 'postfixes', 'sift',
    'topological_sort', 'unflatten', 'has_dups', 'has_variety', 'reshape',
    'rotations', 'filldedent', 'lambdify', 'threaded', 'xthreaded',
    'public', 'memoize_property', 'timed',

    # sympy.integrals
    'integrate', 'Integral', 'line_integrate', 'mellin_transform',
    'inverse_mellin_transform', 'MellinTransform', 'InverseMellinTransform',
    'laplace_transform', 'inverse_laplace_transform', 'LaplaceTransform',
    'laplace_correspondence', 'laplace_initial_conds',
    'InverseLaplaceTransform', 'fourier_transform',
    'inverse_fourier_transform', 'FourierTransform',
    'InverseFourierTransform', 'sine_transform', 'inverse_sine_transform',
    'SineTransform', 'InverseSineTransform', 'cosine_transform',
    'inverse_cosine_transform', 'CosineTransform', 'InverseCosineTransform',
    'hankel_transform', 'inverse_hankel_transform', 'HankelTransform',
    'InverseHankelTransform', 'singularityintegrate',

    # sympy.tensor
    'IndexedBase', 'Idx', 'Indexed', 'get_contraction_structure',
    'get_indices', 'shape', 'MutableDenseNDimArray', 'ImmutableDenseNDimArray',
    'MutableSparseNDimArray', 'ImmutableSparseNDimArray', 'NDimArray',
    'tensorproduct', 'tensorcontraction', 'tensordiagonal', 'derive_by_array',
    'permutedims', 'Array', 'DenseNDimArray', 'SparseNDimArray',

    # sympy.parsing
    'parse_expr',

    # sympy.calculus
    'euler_equations', 'singularities', 'is_increasing',
    'is_strictly_increasing', 'is_decreasing', 'is_strictly_decreasing',
    'is_monotonic', 'finite_diff_weights', 'apply_finite_diff',
    'differentiate_finite', 'periodicity', 'not_empty_in',
    'AccumBounds', 'is_convex', 'stationary_points', 'minimum', 'maximum',

    # sympy.algebras
    'Quaternion',

    # sympy.printing
    'pager_print', 'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode', 'latex', 'print_latex', 'multiline_latex',
    'mathml', 'print_mathml', 'python', 'print_python', 'pycode', 'ccode',
    'print_ccode', 'smtlib_code', 'glsl_code', 'print_glsl', 'cxxcode', 'fcode',
    'print_fcode', 'rcode', 'print_rcode', 'jscode', 'print_jscode',
    'julia_code', 'mathematica_code', 'octave_code', 'rust_code', 'print_gtk',
    'preview', 'srepr', 'print_tree', 'StrPrinter', 'sstr', 'sstrrepr',
    'TableForm', 'dotprint', 'maple_code', 'print_maple_code',

    # sympy.plotting
    'plot', 'textplot', 'plot_backends', 'plot_implicit', 'plot_parametric',

    # sympy.interactive
    'init_session', 'init_printing', 'interactive_traversal',

    # sympy.testing
    'test', 'doctest',
]


#===========================================================================#
#                                                                           #
# XXX: The names below were importable before SymPy 1.6 using               #
#                                                                           #
#          from sympy import *                                              #
#                                                                           #
# This happened implicitly because there was no __all__ defined in this     #
# __init__.py file. Not every package is imported. The list matches what    #
# would have been imported before. It is possible that these packages will  #
# not be imported by a star-import from sympy in future.                    #
#                                                                           #
#===========================================================================#


__all__.extend((
    'algebras',
    'assumptions',
    'calculus',
    'concrete',
    'discrete',
    'external',
    'functions',
    'geometry',
    'interactive',
    'multipledispatch',
    'ntheory',
    'parsing',
    'plotting',
    'polys',
    'printing',
    'release',
    'strategies',
    'tensor',
    'utilities',
))
<|endoftext|>"""
This module implements a method to find
Euler-Lagrange Equations for given Lagrangian.
"""
from itertools import combinations_with_replacement
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities.iterables import iterable


def euler_equations(L, funcs=(), vars=()):
    r"""
    Find the Euler-Lagrange equations [1]_ for a given Lagrangian.

    Parameters
    ==========

    L : Expr
        The Lagrangian that should be a function of the functions listed
        in the second argument and their derivatives.

        For example, in the case of two functions $f(x,y)$, $g(x,y)$ and
        two independent variables $x$, $y$ the Lagrangian has the form:

            .. math:: L\left(f(x,y),g(x,y),\frac{\partial f(x,y)}{\partial x},
                      \frac{\partial f(x,y)}{\partial y},
                      \frac{\partial g(x,y)}{\partial x},
                      \frac{\partial g(x,y)}{\partial y},x,y\right)

        In many cases it is not necessary to provide anything, except the
        Lagrangian, it will be auto-detected (and an error raised if this
        cannot be done).

    funcs : Function or an iterable of Functions
        The functions that the Lagrangian depends on. The Euler equations
        are differential equations for each of these functions.

    vars : Symbol or an iterable of Symbols
        The Symbols that are the independent variables of the functions.

    Returns
    =======

    eqns : list of Eq
        The list of differential equations, one for each function.

    Examples
    ========

    >>> from sympy import euler_equations, Symbol, Function
    >>> x = Function('x')
    >>> t = Symbol('t')
    >>> L = (x(t).diff(t))**2/2 - x(t)**2/2
    >>> euler_equations(L, x(t), t)
    [Eq(-x(t) - Derivative(x(t), (t, 2)), 0)]
    >>> u = Function('u')
    >>> x = Symbol('x')
    >>> L = (u(t, x).diff(t))**2/2 - (u(t, x).diff(x))**2/2
    >>> euler_equations(L, u(t, x), [t, x])
    [Eq(-Derivative(u(t, x), (t, 2)) + Derivative(u(t, x), (x, 2)), 0)]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation

    """

    funcs = tuple(funcs) if iterable(funcs) else (funcs,)

    if not funcs:
        funcs = tuple(L.atoms(Function))
    else:
        for f in funcs:
            if not isinstance(f, Function):
                raise TypeError('Function expected, got: %s' % f)

    vars = tuple(vars) if iterable(vars) else (vars,)

    if not vars:
        vars = funcs[0].args
    else:
        vars = tuple(sympify(var) for var in vars)

    if not all(isinstance(v, Symbol) for v in vars):
        raise TypeError('Variables are not symbols, got %s' % vars)

    for f in funcs:
        if not vars == f.args:
            raise ValueError("Variables %s do not match args: %s" % (vars, f))

    order = max([len(d.variables) for d in L.atoms(Derivative)
                        if d.expr in funcs] + [0])

    eqns = []
    for f in funcs:
        eq = diff(L, f)
        for i in range(1, order + 1):
            for p in combinations_with_replacement(vars, i):
                eq = eq + S.NegativeOne**i*diff(L, diff(f, *p), *p)
        new_eq = Eq(eq, 0)
        if isinstance(new_eq, Eq):
            eqns.append(new_eq)

    return eqns
<|endoftext|>from .accumulationbounds import AccumBounds, AccumulationBounds # noqa: F401
from .singularities import singularities
from sympy.core import Pow, S
from sympy.core.function import diff, expand_mul, Function
from sympy.core.kind import NumberKind
from sympy.core.mod import Mod
from sympy.core.numbers import equal_valued
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import frac
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
    TrigonometricFunction, sin, cos, tan, cot, csc, sec,
    asin, acos, acot, atan, asec, acsc)
from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, coth,
    sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.polys.polytools import degree, lcm_list
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
                             Complement)
from sympy.sets.fancysets import ImageSet
from sympy.sets.conditionset import ConditionSet
from sympy.utilities import filldedent
from sympy.utilities.iterables import iterable
from sympy.matrices.dense import hessian


def continuous_domain(f, symbol, domain):
    """
    Returns the domain on which the function expression f is continuous.

    This function is limited by the ability to determine the various
    singularities and discontinuities of the given function.
    The result is either given as a union of intervals or constructed using
    other set operations.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the intervals are to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the continuity of the symbol has to be checked.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, tan, log, pi, sqrt
    >>> from sympy.calculus.util import continuous_domain
    >>> x = Symbol('x')
    >>> continuous_domain(1/x, x, S.Reals)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    >>> continuous_domain(tan(x), x, Interval(0, pi))
    Union(Interval.Ropen(0, pi/2), Interval.Lopen(pi/2, pi))
    >>> continuous_domain(sqrt(x - 2), x, Interval(-5, 5))
    Interval(2, 5)
    >>> continuous_domain(log(2*x - 1), x, S.Reals)
    Interval.open(1/2, oo)

    Returns
    =======

    :py:class:`~.Interval`
        Union of all intervals where the function is continuous.

    Raises
    ======

    NotImplementedError
        If the method to determine continuity of such a function
        has not yet been developed.

    """
    from sympy.solvers.inequalities import solve_univariate_inequality

    if not domain.is_subset(S.Reals):
        raise NotImplementedError(filldedent('''
            Domain must be a subset of S.Reals.
            '''))
    implemented = [Pow, exp, log, Abs, frac,
                   sin, cos, tan, cot, sec, csc,
                   asin, acos, atan, acot, asec, acsc,
                   sinh, cosh, tanh, coth, sech, csch,
                   asinh, acosh, atanh, acoth, asech, acsch]
    used = [fct.func for fct in f.atoms(Function) if fct.has(symbol)]
    if any(func not in implemented for func in used):
        raise NotImplementedError(filldedent('''
            Unable to determine the domain of the given function.
            '''))

    x = Symbol('x')
    constraints = {
        log: (x > 0,),
        asin: (x >= -1, x <= 1),
        acos: (x >= -1, x <= 1),
        acosh: (x >= 1,),
        atanh: (x > -1, x < 1),
        asech: (x > 0, x <= 1)
    }
    constraints_union = {
        asec: (x <= -1, x >= 1),
        acsc: (x <= -1, x >= 1),
        acoth: (x < -1, x > 1)
    }

    cont_domain = domain
    for atom in f.atoms(Pow):
        den = atom.exp.as_numer_denom()[1]
        if atom.exp.is_rational and den.is_odd:
            pass    # 0**negative handled by singularities()
        else:
            constraint = solve_univariate_inequality(atom.base >= 0,
                                                        symbol).as_set()
            cont_domain = Intersection(constraint, cont_domain)

    for atom in f.atoms(Function):
        if atom.func in constraints:
            for c in constraints[atom.func]:
                constraint_relational = c.subs(x, atom.args[0])
                constraint_set = solve_univariate_inequality(
                    constraint_relational, symbol).as_set()
                cont_domain = Intersection(constraint_set, cont_domain)
        elif atom.func in constraints_union:
            constraint_set = S.EmptySet
            for c in constraints_union[atom.func]:
                constraint_relational = c.subs(x, atom.args[0])
                constraint_set += solve_univariate_inequality(
                    constraint_relational, symbol).as_set()
            cont_domain = Intersection(constraint_set, cont_domain)
        # XXX: the discontinuities below could be factored out in
        # a new "discontinuities()".
        elif atom.func == acot:
            from sympy.solvers.solveset import solveset_real
            # Sympy's acot() has a step discontinuity at 0. Since it's
            # neither an essential singularity nor a pole, singularities()
            # will not report it. But it's still relevant for determining
            # the continuity of the function f.
            cont_domain -= solveset_real(atom.args[0], symbol)
            # Note that the above may introduce spurious discontinuities, e.g.
            # for abs(acot(x)) at 0.
        elif atom.func == frac:
            from sympy.solvers.solveset import solveset_real
            r = function_range(atom.args[0], symbol, domain)
            r = Intersection(r, S.Integers)
            if r.is_finite_set:
                discont = S.EmptySet
                for n in r:
                    discont += solveset_real(atom.args[0]-n, symbol)
            else:
                discont = ConditionSet(
                    symbol, S.Integers.contains(atom.args[0]), cont_domain)
            cont_domain -= discont

    return cont_domain - singularities(f, symbol, domain)


def function_range(f, symbol, domain):
    """
    Finds the range of a function in a given domain.
    This method is limited by the ability to determine the singularities and
    determine limits.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the range of function is to be determined.
    domain : :py:class:`~.Interval`
        The domain under which the range of the function has to be found.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, exp, log, pi, sqrt, sin, tan
    >>> from sympy.calculus.util import function_range
    >>> x = Symbol('x')
    >>> function_range(sin(x), x, Interval(0, 2*pi))
    Interval(-1, 1)
    >>> function_range(tan(x), x, Interval(-pi/2, pi/2))
    Interval(-oo, oo)
    >>> function_range(1/x, x, S.Reals)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    >>> function_range(exp(x), x, S.Reals)
    Interval.open(0, oo)
    >>> function_range(log(x), x, S.Reals)
    Interval(-oo, oo)
    >>> function_range(sqrt(x), x, Interval(-5, 9))
    Interval(0, 3)

    Returns
    =======

    :py:class:`~.Interval`
        Union of all ranges for all intervals under domain where function is
        continuous.

    Raises
    ======

    NotImplementedError
        If any of the intervals, in the given domain, for which function
        is continuous are not finite or real,
        OR if the critical points of the function on the domain cannot be found.
    """

    if domain is S.EmptySet:
        return S.EmptySet

    period = periodicity(f, symbol)
    if period == S.Zero:
        # the expression is constant wrt symbol
        return FiniteSet(f.expand())

    from sympy.series.limits import limit
    from sympy.solvers.solveset import solveset

    if period is not None:
        if isinstance(domain, Interval):
            if (domain.inf - domain.sup).is_infinite:
                domain = Interval(0, period)
        elif isinstance(domain, Union):
            for sub_dom in domain.args:
                if isinstance(sub_dom, Interval) and \
                ((sub_dom.inf - sub_dom.sup).is_infinite):
                    domain = Interval(0, period)

    intervals = continuous_domain(f, symbol, domain)
    range_int = S.EmptySet
    if isinstance(intervals,(Interval, FiniteSet)):
        interval_iter = (intervals,)
    elif isinstance(intervals, Union):
        interval_iter = intervals.args
    else:
        raise NotImplementedError("Unable to find range for the given domain.")

    for interval in interval_iter:
        if isinstance(interval, FiniteSet):
            for singleton in interval:
                if singleton in domain:
                    range_int += FiniteSet(f.subs(symbol, singleton))
        elif isinstance(interval, Interval):
            vals = S.EmptySet
            critical_values = S.EmptySet
            bounds = ((interval.left_open, interval.inf, '+'),
                   (interval.right_open, interval.sup, '-'))

            for is_open, limit_point, direction in bounds:
                if is_open:
                    critical_values += FiniteSet(limit(f, symbol, limit_point, direction))
                    vals += critical_values
                else:
                    vals += FiniteSet(f.subs(symbol, limit_point))

            critical_points = solveset(f.diff(symbol), symbol, interval)

            if not iterable(critical_points):
                raise NotImplementedError(
                        'Unable to find critical points for {}'.format(f))
            if isinstance(critical_points, ImageSet):
                raise NotImplementedError(
                        'Infinite number of critical points for {}'.format(f))

            for critical_point in critical_points:
                vals += FiniteSet(f.subs(symbol, critical_point))

            left_open, right_open = False, False

            if critical_values is not S.EmptySet:
                if critical_values.inf == vals.inf:
                    left_open = True

                if critical_values.sup == vals.sup:
                    right_open = True

            range_int += Interval(vals.inf, vals.sup, left_open, right_open)
        else:
            raise NotImplementedError("Unable to find range for the given domain.")

    return range_int


def not_empty_in(finset_intersection, *syms):
    """
    Finds the domain of the functions in ``finset_intersection`` in which the
    ``finite_set`` is not-empty.

    Parameters
    ==========

    finset_intersection : Intersection of FiniteSet
        The unevaluated intersection of FiniteSet containing
        real-valued functions with Union of Sets
    syms : Tuple of symbols
        Symbol for which domain is to be found

    Raises
    ======

    NotImplementedError
        The algorithms to find the non-emptiness of the given FiniteSet are
        not yet implemented.
    ValueError
        The input is not valid.
    RuntimeError
        It is a bug, please report it to the github issue tracker
        (https://github.com/sympy/sympy/issues).

    Examples
    ========

    >>> from sympy import FiniteSet, Interval, not_empty_in, oo
    >>> from sympy.abc import x
    >>> not_empty_in(FiniteSet(x/2).intersect(Interval(0, 1)), x)
    Interval(0, 2)
    >>> not_empty_in(FiniteSet(x, x**2).intersect(Interval(1, 2)), x)
    Union(Interval(1, 2), Interval(-sqrt(2), -1))
    >>> not_empty_in(FiniteSet(x**2/(x + 2)).intersect(Interval(1, oo)), x)
    Union(Interval.Lopen(-2, -1), Interval(2, oo))
    """

    # TODO: handle piecewise defined functions
    # TODO: handle transcendental functions
    # TODO: handle multivariate functions
    if len(syms) == 0:
        raise ValueError("One or more symbols must be given in syms.")

    if finset_intersection is S.EmptySet:
        return S.EmptySet

    if isinstance(finset_intersection, Union):
        elm_in_sets = finset_intersection.args[0]
        return Union(not_empty_in(finset_intersection.args[1], *syms),
                     elm_in_sets)

    if isinstance(finset_intersection, FiniteSet):
        finite_set = finset_intersection
        _sets = S.Reals
    else:
        finite_set = finset_intersection.args[1]
        _sets = finset_intersection.args[0]

    if not isinstance(finite_set, FiniteSet):
        raise ValueError('A FiniteSet must be given, not %s: %s' %
                         (type(finite_set), finite_set))

    if len(syms) == 1:
        symb = syms[0]
    else:
        raise NotImplementedError('more than one variables %s not handled' %
                                  (syms,))

    def elm_domain(expr, intrvl):
        """ Finds the domain of an expression in any given interval """
        from sympy.solvers.solveset import solveset

        _start = intrvl.start
        _end = intrvl.end
        _singularities = solveset(expr.as_numer_denom()[1], symb,
                                  domain=S.Reals)

        if intrvl.right_open:
            if _end is S.Infinity:
                _domain1 = S.Reals
            else:
                _domain1 = solveset(expr < _end, symb, domain=S.Reals)
        else:
            _domain1 = solveset(expr <= _end, symb, domain=S.Reals)

        if intrvl.left_open:
            if _start is S.NegativeInfinity:
                _domain2 = S.Reals
            else:
                _domain2 = solveset(expr > _start, symb, domain=S.Reals)
        else:
            _domain2 = solveset(expr >= _start, symb, domain=S.Reals)

        # domain in the interval
        expr_with_sing = Intersection(_domain1, _domain2)
        expr_domain = Complement(expr_with_sing, _singularities)
        return expr_domain

    if isinstance(_sets, Interval):
        return Union(*[elm_domain(element, _sets) for element in finite_set])

    if isinstance(_sets, Union):
        _domain = S.EmptySet
        for intrvl in _sets.args:
            _domain_element = Union(*[elm_domain(element, intrvl)
                                      for element in finite_set])
            _domain = Union(_domain, _domain_element)
        return _domain


def periodicity(f, symbol, check=False):
    """
    Tests the given function for periodicity in the given symbol.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the period is to be determined.
    check : bool, optional
        The flag to verify whether the value being returned is a period or not.

    Returns
    =======

    period
        The period of the function is returned.
        ``None`` is returned when the function is aperiodic or has a complex period.
        The value of $0$ is returned as the period of a constant function.

    Raises
    ======

    NotImplementedError
        The value of the period computed cannot be verified.


    Notes
    =====

    Currently, we do not support functions with a complex period.
    The period of functions having complex periodic values such
    as ``exp``, ``sinh`` is evaluated to ``None``.

    The value returned might not be the "fundamental" period of the given
    function i.e. it may not be the smallest periodic value of the function.

    The verification of the period through the ``check`` flag is not reliable
    due to internal simplification of the given expression. Hence, it is set
    to ``False`` by default.

    Examples
    ========
    >>> from sympy import periodicity, Symbol, sin, cos, tan, exp
    >>> x = Symbol('x')
    >>> f = sin(x) + sin(2*x) + sin(3*x)
    >>> periodicity(f, x)
    2*pi
    >>> periodicity(sin(x)*cos(x), x)
    pi
    >>> periodicity(exp(tan(2*x) - 1), x)
    pi/2
    >>> periodicity(sin(4*x)**cos(2*x), x)
    pi
    >>> periodicity(exp(x), x)
    """
    if symbol.kind is not NumberKind:
        raise NotImplementedError("Cannot use symbol of kind %s" % symbol.kind)
    temp = Dummy('x', real=True)
    f = f.subs(symbol, temp)
    symbol = temp

    def _check(orig_f, period):
        '''Return the checked period or raise an error.'''
        new_f = orig_f.subs(symbol, symbol + period)
        if new_f.equals(orig_f):
            return period
        else:
            raise NotImplementedError(filldedent('''
                The period of the given function cannot be verified.
                When `%s` was replaced with `%s + %s` in `%s`, the result
                was `%s` which was not recognized as being the same as
                the original function.
                So either the period was wrong or the two forms were
                not recognized as being equal.
                Set check=False to obtain the value.''' %
                (symbol, symbol, period, orig_f, new_f)))

    orig_f = f
    period = None

    if isinstance(f, Relational):
        f = f.lhs - f.rhs

    f = f.simplify()

    if symbol not in f.free_symbols:
        return S.Zero

    if isinstance(f, TrigonometricFunction):
        try:
            period = f.period(symbol)
        except NotImplementedError:
            pass

    if isinstance(f, Abs):
        arg = f.args[0]
        if isinstance(arg, (sec, csc, cos)):
            # all but tan and cot might have a
            # a period that is half as large
            # so recast as sin
            arg = sin(arg.args[0])
        period = periodicity(arg, symbol)
        if period is not None and isinstance(arg, sin):
            # the argument of Abs was a trigonometric other than
            # cot or tan; test to see if the half-period
            # is valid. Abs(arg) has behaviour equivalent to
            # orig_f, so use that for test:
            orig_f = Abs(arg)
            try:
                return _check(orig_f, period/2)
            except NotImplementedError as err:
                if check:
                    raise NotImplementedError(err)
            # else let new orig_f and period be
            # checked below

    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        f = Pow(S.Exp1, expand_mul(f.exp))
        if im(f) != 0:
            period_real = periodicity(re(f), symbol)
            period_imag = periodicity(im(f), symbol)
            if period_real is not None and period_imag is not None:
                period = lcim([period_real, period_imag])

    if f.is_Pow and f.base != S.Exp1:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        if base_has_sym and not expo_has_sym:
            period = periodicity(base, symbol)

        elif expo_has_sym and not base_has_sym:
            period = periodicity(expo, symbol)

        else:
            period = _periodicity(f.args, symbol)

    elif f.is_Mul:
        coeff, g = f.as_independent(symbol, as_Add=False)
        if isinstance(g, TrigonometricFunction) or not equal_valued(coeff, 1):
            period = periodicity(g, symbol)
        else:
            period = _periodicity(g.args, symbol)

    elif f.is_Add:
        k, g = f.as_independent(symbol)
        if k is not S.Zero:
            return periodicity(g, symbol)

        period = _periodicity(g.args, symbol)

    elif isinstance(f, Mod):
        a, n = f.args

        if a == symbol:
            period = n
        elif isinstance(a, TrigonometricFunction):
            period = periodicity(a, symbol)
        #check if 'f' is linear in 'symbol'
        elif (a.is_polynomial(symbol) and degree(a, symbol) == 1 and
            symbol not in n.free_symbols):
                period = Abs(n / a.diff(symbol))

    elif isinstance(f, Piecewise):
        pass  # not handling Piecewise yet as the return type is not favorable

    elif period is None:
        from sympy.solvers.decompogen import compogen, decompogen
        g_s = decompogen(f, symbol)
        num_of_gs = len(g_s)
        if num_of_gs > 1:
            for index, g in enumerate(reversed(g_s)):
                start_index = num_of_gs - 1 - index
                g = compogen(g_s[start_index:], symbol)
                if g not in (orig_f, f): # Fix for issue 12620
                    period = periodicity(g, symbol)
                    if period is not None:
                        break

    if period is not None:
        if check:
            return _check(orig_f, period)
        return period

    return None


def _periodicity(args, symbol):
    """
    Helper for `periodicity` to find the period of a list of simpler
    functions.
    It uses the `lcim` method to find the least common period of
    all the functions.

    Parameters
    ==========

    args : Tuple of :py:class:`~.Symbol`
        All the symbols present in a function.

    symbol : :py:class:`~.Symbol`
        The symbol over which the function is to be evaluated.

    Returns
    =======

    period
        The least common period of the function for all the symbols
        of the function.
        ``None`` if for at least one of the symbols the function is aperiodic.

    """
    periods = []
    for f in args:
        period = periodicity(f, symbol)
        if period is None:
            return None

        if period is not S.Zero:
            periods.append(period)

    if len(periods) > 1:
        return lcim(periods)

    if periods:
        return periods[0]


def lcim(numbers):
    """Returns the least common integral multiple of a list of numbers.

    The numbers can be rational or irrational or a mixture of both.
    `None` is returned for incommensurable numbers.

    Parameters
    ==========

    numbers : list
        Numbers (rational and/or irrational) for which lcim is to be found.

    Returns
    =======

    number
        lcim if it exists, otherwise ``None`` for incommensurable numbers.

    Examples
    ========

    >>> from sympy.calculus.util import lcim
    >>> from sympy import S, pi
    >>> lcim([S(1)/2, S(3)/4, S(5)/6])
    15/2
    >>> lcim([2*pi, 3*pi, pi, pi/2])
    6*pi
    >>> lcim([S(1), 2*pi])
    """
    result = None
    if all(num.is_irrational for num in numbers):
        factorized_nums = [num.factor() for num in numbers]
        factors_num = [num.as_coeff_Mul() for num in factorized_nums]
        term = factors_num[0][1]
        if all(factor == term for coeff, factor in factors_num):
            common_term = term
            coeffs = [coeff for coeff, factor in factors_num]
            result = lcm_list(coeffs) * common_term

    elif all(num.is_rational for num in numbers):
        result = lcm_list(numbers)

    else:
        pass

    return result

def is_convex(f, *syms, domain=S.Reals):
    r"""Determines the  convexity of the function passed in the argument.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    syms : Tuple of :py:class:`~.Symbol`
        The variables with respect to which the convexity is to be determined.
    domain : :py:class:`~.Interval`, optional
        The domain over which the convexity of the function has to be checked.
        If unspecified, S.Reals will be the default domain.

    Returns
    =======

    bool
        The method returns ``True`` if the function is convex otherwise it
        returns ``False``.

    Raises
    ======

    NotImplementedError
        The check for the convexity of multivariate functions is not implemented yet.

    Notes
    =====

    To determine concavity of a function pass `-f` as the concerned function.
    To determine logarithmic convexity of a function pass `\log(f)` as
    concerned function.
    To determine logarithmic concavity of a function pass `-\log(f)` as
    concerned function.

    Currently, convexity check of multivariate functions is not handled.

    Examples
    ========

    >>> from sympy import is_convex, symbols, exp, oo, Interval
    >>> x = symbols('x')
    >>> is_convex(exp(x), x)
    True
    >>> is_convex(x**3, x, domain = Interval(-1, oo))
    False
    >>> is_convex(1/x**2, x, domain=Interval.open(0, oo))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convex_function
    .. [2] http://www.ifp.illinois.edu/~angelia/L3_convfunc.pdf
    .. [3] https://en.wikipedia.org/wiki/Logarithmically_convex_function
    .. [4] https://en.wikipedia.org/wiki/Logarithmically_concave_function
    .. [5] https://en.wikipedia.org/wiki/Concave_function

    """
    if len(syms) > 1 :
        return hessian(f, syms).is_positive_semidefinite
    from sympy.solvers.inequalities import solve_univariate_inequality
    f = _sympify(f)
    var = syms[0]
    if any(s in domain for s in singularities(f, var)):
        return False
    condition = f.diff(var, 2) < 0
    if solve_univariate_inequality(condition, var, False, domain):
        return False
    return True


def stationary_points(f, symbol, domain=S.Reals):
    """
    Returns the stationary points of a function (where derivative of the
    function is 0) in the given domain.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the stationary points are to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the stationary points have to be checked.
        If unspecified, ``S.Reals`` will be the default domain.

    Returns
    =======

    Set
        A set of stationary points for the function. If there are no
        stationary point, an :py:class:`~.EmptySet` is returned.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, sin, pi, pprint, stationary_points
    >>> x = Symbol('x')

    >>> stationary_points(1/x, x, S.Reals)
    EmptySet

    >>> pprint(stationary_points(sin(x), x), use_unicode=False)
              pi                              3*pi
    {2*n*pi + -- | n in Integers} U {2*n*pi + ---- | n in Integers}
              2                                2

    >>> stationary_points(sin(x),x, Interval(0, 4*pi))
    {pi/2, 3*pi/2, 5*pi/2, 7*pi/2}

    """
    from sympy.solvers.solveset import solveset

    if domain is S.EmptySet:
        return S.EmptySet

    domain = continuous_domain(f, symbol, domain)
    set = solveset(diff(f, symbol), symbol, domain)

    return set


def maximum(f, symbol, domain=S.Reals):
    """
    Returns the maximum value of a function in the given domain.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for maximum value needs to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the maximum have to be checked.
        If unspecified, then the global maximum is returned.

    Returns
    =======

    number
        Maximum value of the function in given domain.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, sin, cos, pi, maximum
    >>> x = Symbol('x')

    >>> f = -x**2 + 2*x + 5
    >>> maximum(f, x, S.Reals)
    6

    >>> maximum(sin(x), x, Interval(-pi, pi/4))
    sqrt(2)/2

    >>> maximum(sin(x)*cos(x), x)
    1/2

    """
    if isinstance(symbol, Symbol):
        if domain is S.EmptySet:
            raise ValueError("Maximum value not defined for empty domain.")

        return function_range(f, symbol, domain).sup
    else:
        raise ValueError("%s is not a valid symbol." % symbol)


def minimum(f, symbol, domain=S.Reals):
    """
    Returns the minimum value of a function in the given domain.

    Parameters
