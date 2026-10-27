class SparseTensorCreationSkipVariable(SkipFunctionVariable):
    """
    Skip variable for sparse tensor factory functions with clear messaging regarding lack of support.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        reason = "sparse tensor creation is not supported in torch.compile"
        super().__init__(value, reason=reason, **kwargs)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .. import graph_break_hints

        fn_name = getattr(self.value, "__name__", str(self.value))
        unimplemented(
            gb_type="Sparse tensor creation not supported",
            context=f"function: {fn_name}",
            explanation=(
                f"torch.compile does not support sparse tensor creation functions like {fn_name}. "
                "Sparse tensors require specialized handling that is not yet implemented in the compiler."
            ),
            hints=[*graph_break_hints.SPARSE_TENSOR],
        )


def emit_noargs_leaf_function_to_graph(
    tx: "InstructionTranslatorBase",
    real_impl: Callable[[], None],
    name: str,
) -> None:
    """Emit an invoke_leaf_function node for a side-effectful function with no
    tensor inputs or outputs.

    The function is captured as a closure inside _LeafCallable objects and
    registered as a static attribute on the graph module.  Because
    invoke_leaf_function is registered as EffectType.ORDERED, effect tokens
    prevent DCE and maintain execution ordering relative to other ops.

    Use this when Dynamo needs to preserve a pure-side-effect call (like
    setting global runtime state) in the compiled graph so that it replays
    at the correct position at runtime.
    """
    import torch.utils._pytree as pytree
    from torch._higher_order_ops.invoke_leaf_function import (
        _LeafCallable,
        invoke_leaf_function,
        make_leaf_function_wrappers,
    )

    def fake_impl():
        return None

    captured_out_spec: list[pytree.TreeSpec | None] = [None]
    wrapped_real, wrapped_fake = make_leaf_function_wrappers(
        real_impl, fake_impl, captured_out_spec
    )

    real_callable = _LeafCallable(wrapped_real)
    fake_callable = _LeafCallable(wrapped_fake)
    input_spec = pytree.tree_flatten(((), {}))[1]

    def make_proxy(attr_name: str, val: Any) -> Any:
        proxy = tx.output.register_static_attr_and_return_proxy(attr_name, val)
        proxy.node.type = type(val)
        return proxy

    invoke_args = (
        make_proxy(f"{name}_real_fn", real_callable),
        make_proxy(f"{name}_fake_fn", fake_callable),
        make_proxy(f"{name}_input_spec", input_spec),
        "",  # mutated_flat_indices
    )
    tx.output.create_proxy("call_function", invoke_leaf_function, invoke_args, {})


class TritonSetAllocatorVariable(VariableTracker):
    """Trace triton.set_allocator as an invoke_leaf_function node in the
    graph so that it executes at the right point at runtime, ordered by
    effect tokens."""

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        return type(self.value)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if len(args) != 1:
            raise AssertionError(f"expected exactly 1 arg, got {len(args)}")
        if kwargs:
            raise AssertionError("unexpected kwargs")
        alloc_fn = args[0].as_python_constant()

        # Emit an invoke_leaf_function node so it runs at runtime.
        set_allocator = self.value

        def real_impl():
            set_allocator(alloc_fn)
            return None

        emit_noargs_leaf_function_to_graph(tx, real_impl, "set_alloc")

        return ConstantVariable.create(None)


# ---------------------------------------------------------------------------
# CPython descriptor VTs
#
# Each class mirrors a CPython descriptor type (PyWrapperDescr_Type,
# PyMethodDescr_Type, etc.) and implements tp_descr_get_impl to model
# the descriptor binding step faithfully.
# ---------------------------------------------------------------------------


def _check_descriptor_obj_type(
    tx: "InstructionTranslatorBase",
    descriptor: types.MethodDescriptorType
    | types.WrapperDescriptorType
    | types.MemberDescriptorType
    | types.GetSetDescriptorType,
    obj: "VariableTracker",
) -> None:
    """Check that obj's type is compatible with descriptor.__objclass__.

    Mirrors CPython's descr_check which raises TypeError when a C descriptor
    is bound to an object whose type is not a subtype of the descriptor's
    __objclass__.

    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L79-L96
    """
    if obj is None:
        return
    try:
        obj_type = obj.python_type()
    except NotImplementedError:
        return
    if not issubclass(obj_type, descriptor.__objclass__):
        raise_type_error(
            tx,
            f"descriptor '{descriptor.__name__}' for "
            f"'{descriptor.__objclass__.__name__}' objects "
            f"doesn't apply to a '{obj_type.__name__}' object",
        )


# descr_members: __objclass__ and __name__ are PyMemberDef on all descriptor
# types. https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L641-L645
class WrapperDescriptorVariable(VariableTracker):
    """Unbound C slot wrapper (wrapper_descriptor on a type).

    CPython types define behavior through C-level slots on PyTypeObject
    (tp_richcompare, sq_length, nb_add, etc.).  When these slots are
    accessed from Python (e.g. list.__add__), CPython exposes them as
    wrapper_descriptor objects (PyWrapperDescr_Type).  A wrapper_descriptor
    is an unbound descriptor living on the type -- it is not tied to any
    instance.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L867

    When a wrapper_descriptor is accessed on an instance (e.g. [1,2].__add__),
    its tp_descr_get slot (wrapperdescr_get) is invoked, which calls
    PyWrapper_New to produce a bound method-wrapper (_PyMethodWrapper_Type).
    The tp_descr_get_impl method on this class mirrors that binding step.
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.WrapperDescriptorType,
        owner: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.owner = owner

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"WrapperDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.WrapperDescriptorType

    def as_python_constant(self) -> types.WrapperDescriptorType:
        return self.descriptor

    def get_real_python_backed_value(self) -> types.WrapperDescriptorType:
        return self.descriptor

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Unbound call: list.__add__([1,2], [3,4]) -- first arg is self.
        # Mirrors wrapperdescr_call which invokes the C slot directly.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L535
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        obj, *rest = args
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # Dispatch through the owner (UDCV for the defining class) rather
        # than obj.call_method, which would do MRO resolution from type(obj)
        # and find Python overrides on subclasses. Routing through the class
        # mirrors CPython's wrapperdescr_call which invokes the C slot directly.
        return self.owner.call_method(
            tx, self.descriptor.__name__, [obj, *rest], kwargs
        )

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> "MethodWrapperVariable":
        # Mirrors wrapperdescr_get which calls PyWrapper_New to produce
        # a bound method-wrapper.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L203-L213
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1489-L1505
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        return MethodWrapperVariable(self.descriptor, obj, source=self.source)


class MethodWrapperVariable(VariableTracker):
    """Bound method-wrapper (wrapper_descriptor bound to an instance).

    Produced by WrapperDescriptorVariable.tp_descr_get_impl, mirroring
    PyWrapper_New which stores a reference to the descriptor and the instance.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1450
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.WrapperDescriptorType,
        obj: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.obj = obj

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return (
            f"MethodWrapperVariable({cls_name}.{self.descriptor.__name__}, {self.obj})"
        )

    def python_type(self) -> type:
        return types.MethodWrapperType

    # A method-wrapper's own dunders come from the descriptor / bound obj and do
    # not depend on obj's contents. Resolving them via these tables (consulted
    # before the generic getset-descriptor path) avoids forcing self.obj to a
    # python constant, which would break e.g. a list holding non-constant items.
    tp_getset = {
        "__name__": GetSet(
            lambda s, tx: ConstantVariable.create(s.descriptor.__name__)
        ),
        "__qualname__": GetSet(
            lambda s, tx: ConstantVariable.create(s.descriptor.__qualname__)
        ),
    }
    tp_members = {
        "__self__": Member(lambda s, tx: s.obj),
    }

    def get_real_python_backed_value(self) -> types.MethodWrapperType:
        return self.as_python_constant()

    def is_python_constant(self) -> bool:
        return self.obj.is_python_constant()

    def as_python_constant(self) -> types.MethodWrapperType:
        obj_value = self.obj.as_python_constant()
        try:
            return self.descriptor.__get__(obj_value, type(obj_value))
        except TypeError:
            raise AsPythonConstantNotImplementedError(self) from None

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Only materialize the wrapper (which forces self.obj's python constant)
        # for the tensor-getter special case; the generic path below dispatches
        # via call_method and must not require a constant self.obj (e.g. a set /
        # frozenset whose as_python_constant() re-hashes its keys).
        if args and isinstance(args[0], variables.TensorVariable):
            try:
                method_wrapper = self.as_python_constant()
            except NotImplementedError:
                method_wrapper = None
            if method_wrapper is not None and is_tensor_base_attr_getter(
                method_wrapper
            ):
                if not (len(args) == 1 and len(kwargs) == 0):
                    raise_type_error(
                        tx, "tensor attribute getter takes exactly one argument"
                    )
                # Avoid the generic descriptor path's implicit owner lookup, which
                # would read __class__ on tensor subclasses during __torch_function__.
                descriptor = cast(Any, method_wrapper.__self__)
                return args[0].tp_getattro_impl(tx, descriptor.__name__)

        return self.obj.call_method(tx, self.descriptor.__name__, list(args), kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.descriptor.__name__))

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        try:
            # CPython wrapper_hash:
            # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1347
            return hash(self.as_python_constant()), False
        except NotImplementedError:
            return super().hash_impl(tx)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)


class MethodDescriptorVariable(VariableTracker):
    """Unbound C method descriptor (method_descriptor on a type).

    CPython types expose their PyMethodDef-based C methods as
    method_descriptor objects (PyMethodDescr_Type) in the type's tp_dict.
    For example, list.append and dict.get are method_descriptors.  Like
    wrapper_descriptors, these are unbound descriptors living on the type.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L716

    When a method_descriptor is accessed on an instance (e.g. [].append),
    its tp_descr_get slot (method_get) is invoked, which calls
    PyCFunction_NewEx to produce a bound builtin_function_or_method
    (PyCFunction_Type).  The tp_descr_get_impl method mirrors that step.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MethodDescriptorType,
        owner: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.owner = owner

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"MethodDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.MethodDescriptorType

    def as_python_constant(self) -> types.MethodDescriptorType:
        return self.descriptor

    def get_real_python_backed_value(self) -> types.MethodDescriptorType:
        return self.descriptor

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Mirrors methoddescr_call which invokes the C method directly.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L427
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        obj, *rest = args
        name = self.descriptor.__name__
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # Dispatch through the owner (UDCV for the defining class) rather
        # than obj.call_method, which would do MRO resolution from type(obj)
        # and find Python overrides on subclasses.
        return self.owner.call_method(tx, name, [obj, *rest], kwargs)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> "BoundBuiltinMethodVariable":
        # Mirrors method_get which calls PyCFunction_NewEx to produce a
        # bound builtin_function_or_method.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L40
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        return BoundBuiltinMethodVariable(self.descriptor, obj, source=self.source)


class BoundBuiltinMethodVariable(VariableTracker):
    """Bound builtin_function_or_method (PyCFunction_Type).

    Produced by MethodDescriptorVariable.tp_descr_get_impl (binding a
    method_descriptor to an instance, e.g. [].append) or created by the
    builder for bound C methods (e.g. frozenset().__contains__,
    tuple.__new__).  The backing descriptor can be a MethodDescriptorType
    or a BuiltinFunctionType (for methods stored directly in type dicts).
    https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L331
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MethodDescriptorType
        | types.BuiltinFunctionType
        | types.ClassMethodDescriptorType,
        obj: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.obj = obj

    def __repr__(self) -> str:
        cls_name = getattr(
            getattr(self.descriptor, "__objclass__", None), "__name__", "?"
        )
        return f"BoundBuiltinMethodVariable({cls_name}.{self.descriptor.__name__}, {self.obj})"

    def python_type(self) -> type:
        return types.BuiltinMethodType

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # meth_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/methodobject.c#L319
        try:
            return hash(self.as_python_constant()), False
        except AsPythonConstantNotImplementedError:
            return id(self), True

    def tp_richcompare_impl(self, tx, other, op):
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def as_python_constant(self) -> Any:
        obj = self.obj.as_python_constant()
        if isinstance(self.descriptor, types.ClassMethodDescriptorType):
            return self.descriptor.__get__(None, obj)
        if hasattr(self.descriptor, "__get__"):
            return self.descriptor.__get__(obj)  # type: ignore[union-attr]
        return getattr(obj, self.descriptor.__name__)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.obj.call_method(tx, self.descriptor.__name__, list(args), kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.descriptor.__name__))


class ClassMethodDescriptorVariable(VariableTracker):
    """C-level classmethod descriptor (classmethod_descriptor on a type).

    CPython exposes C classmethods defined via PyMethodDef with METH_CLASS
    as classmethod_descriptor objects (PyClassMethodDescr_Type).  For
    example, dict.fromkeys is a classmethod_descriptor.  Like
    method_descriptor, these live on the type and are unbound.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L756

    classmethod_get binds the C method to the class (ignoring obj) via
    PyCMethod_New, producing a bound builtin_function_or_method.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.ClassMethodDescriptorType,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"ClassMethodDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.ClassMethodDescriptorType

    def as_python_constant(self) -> types.ClassMethodDescriptorType:
        return self.descriptor

    def get_real_python_backed_value(self) -> types.ClassMethodDescriptorType:
        return self.descriptor

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> BoundBuiltinMethodVariable:
        # classmethod_get binds the C method to the class (ignoring obj),
        # producing a builtin_function_or_method via PyCMethod_New.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
        return BoundBuiltinMethodVariable(self.descriptor, owner, source=self.source)


class StaticMethodVariable(VariableTracker):
    """staticmethod descriptor wrapping a callable.

    CPython's staticmethod (PyStaticMethod_Type) is a non-data descriptor
    whose tp_descr_get (sm_descr_get) simply returns the wrapped callable,
    ignoring both obj and type.
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1520
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1418-L1428
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: staticmethod,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        func_name = getattr(self.descriptor.__func__, "__name__", "?")
        return f"StaticMethodVariable({func_name})"

    def python_type(self) -> type:
        return staticmethod

    def as_python_constant(self) -> staticmethod:  # type: ignore[type-arg]
        return self.descriptor

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # sm_descr_get returns sm->sm_callable unconditionally.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1418-L1428
        func_source = AttrSource(self.source, "__func__") if self.source else None
        return VariableTracker.build(tx, self.descriptor.__func__, func_source)


class ClassMethodVariable(VariableTracker):
    """classmethod descriptor wrapping a callable.

    CPython's classmethod (PyClassMethod_Type) is a non-data descriptor
    whose tp_descr_get (cm_descr_get) creates a bound method of the
    wrapped callable bound to the class (via PyMethod_New).
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1314
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1215-L1227
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: classmethod,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        func_name = getattr(self.descriptor.__func__, "__name__", "?")
        return f"ClassMethodVariable({func_name})"

    def python_type(self) -> type:
        return classmethod

    def as_python_constant(self) -> classmethod:  # type: ignore[type-arg]
        return self.descriptor

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # cm_descr_get binds the wrapped function to the class.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1215-L1227
        func_source = AttrSource(self.source, "__func__") if self.source else None
        bound_source = (
            AttrSource(owner.source, self.descriptor.__func__.__name__)
            if owner.source
            else None
        )
        return UserMethodVariable(
            self.descriptor.__func__,
            owner,
            source_fn=func_source,
            source=bound_source,
        )


class MemberDescriptorVariable(VariableTracker):
    """C struct field descriptor (member_descriptor on a type).

    CPython exposes C struct fields defined via PyMemberDef as
    member_descriptor objects (PyMemberDescr_Type).  These are data
    descriptors used by __slots__ and C extension types to provide
    direct access to struct members.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L793

    member_get reads the field via PyMember_GetOne.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L162-L180
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MemberDescriptorType,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"MemberDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.MemberDescriptorType

    def as_python_constant(self) -> types.MemberDescriptorType:
        return self.descriptor

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors member_get which calls PyMember_GetOne to read the
        # C struct field.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L162-L180
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        from .object_protocol import _UnhandledDescriptorError

        attr_name = self.descriptor.__name__
        obj_value = obj.get_real_python_backed_value()
        if obj_value is NO_SUCH_SUBOBJ:
            raise _UnhandledDescriptorError(
                f"Cannot resolve member_descriptor '{attr_name}' "
                f"on {type(obj).__name__}"
            )
        try:
            resolved = self.descriptor.__get__(obj_value, type(obj_value))
        except (AttributeError, TypeError):
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(obj_value).__name__}' object has no attribute '{attr_name}'"
                ],
            )
        result_source = obj.source and AttrSource(obj.source, attr_name)
        return VariableTracker.build(tx, resolved, result_source)

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # Mirrors member_set (PyMember_SetOne): store into the C struct field.
        # STORE_ATTR itself applies the descriptor, so replay via store_attr on
        # the target (mirrors the __slots__ path in UserDefinedObjectVariable).
        # value is None for __delete__.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L180-L196
        stored = variables.DeletedVariable() if value is None else value
        tx.output.side_effects.store_attr(obj, self.descriptor.__name__, stored)
        return variables.ConstantVariable.create(None)


class GetSetDescriptorVariable(VariableTracker):
    """C getter/setter descriptor (getset_descriptor on a type).

    CPython exposes C getter/setter pairs defined via PyGetSetDef as
    getset_descriptor objects (PyGetSetDescr_Type).  These are data
    descriptors used for computed attributes backed by C functions
    (e.g. object.__class__, type.__dict__).
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L830

    getset_get calls the C getter function.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, descriptor: types.GetSetDescriptorType, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"GetSetDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def get_real_python_backed_value(self) -> types.GetSetDescriptorType:
        return self.descriptor

    def _getset_descriptor_get(
        self, tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        if self.source is None:
            return None
        source = AttrSource(self.source, "__get__")
        return VariableTracker.build(tx, self.descriptor.__get__, source)

    tp_getset = {
        "__get__": GetSet(_getset_descriptor_get, None),
    }

    tp_members = {
        "__objclass__": Member(getset_build(lambda s: s.descriptor.__objclass__)),
        "__name__": Member(getset_build(lambda s: s.descriptor.__name__)),
    }

    def is_python_constant(self) -> bool:
        return True

    def as_python_constant(self) -> types.GetSetDescriptorType:
        return self.descriptor

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors getset_get which calls the C getter function.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
        attr_name = self.descriptor.__name__
        # Try to eagerly call the C getter when we can obtain the
        # concrete Python object (UDOV.value, or as_python_constant
        # for classes/constants). Fall back to tp_getattro_impl for
        # proxy-based VTs like TensorVariable.
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        obj_value = obj.get_real_python_backed_value()
        if obj_value is NO_SUCH_SUBOBJ:
            from .object_protocol import _UnhandledDescriptorError

            raise _UnhandledDescriptorError(
                f"Cannot resolve getset_descriptor '{attr_name}' "
                f"on {type(obj).__name__}"
            )
        try:
            resolved = self.descriptor.__get__(obj_value, type(obj_value))
        except (AttributeError, TypeError):
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(obj_value).__name__}' object has no attribute '{attr_name}'"
                ],
            )
        result_source = obj.source and AttrSource(obj.source, attr_name)
        if (
            obj.source
            and self.descriptor.__objclass__ is type
            and attr_name in ("__annotations__", "__dict__", "__mro__")
        ):
            # Direct descriptor calls still resolve the standard type slot even
            # when a metaclass shadows the same attribute. Only attach the
            # normal attribute source when runtime attribute lookup agrees.
            static_desc = inspect.getattr_static(type(obj_value), attr_name, None)
            if static_desc is not self.descriptor:
                result_source = None
            elif attr_name == "__mro__":
                result_source = TypeMROSource(obj.source)
        return VariableTracker.build(tx, resolved, result_source)


class PropertyVariable(VariableTracker):
    """Python property descriptor.

    The property type is a data descriptor with tp_descr_get =
    property_descr_get which calls fget(obj) to compute the value.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L2073
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: property,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        fget_name = getattr(self.descriptor.fget, "__name__", "?")
        return f"PropertyVariable({fget_name})"

    def python_type(self) -> type:
        return property

    def as_python_constant(self) -> property:
        return self.descriptor

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors property_descr_get: if obj is NULL or None, return self.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
        if obj is None:
            return self
        fget_source = AttrSource(self.source, "fget") if self.source else None
        fget_vt = VariableTracker.build(
            tx, self.descriptor.fget, source=fget_source, realize=True
        )
        return fget_vt.call_function(tx, [obj], {})


class TupleGetterVariable(VariableTracker):
    """_tuplegetter descriptor used by namedtuple for field access.

    _tuplegetter is a C data descriptor that stores an index and returns
    self[index] on instance access. When accessed on the class (obj=None),
    it returns the descriptor itself.
    https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2735
    https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2636-L2663
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: "_collections._tuplegetter",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        _, (idx, doc) = self.descriptor.__reduce__()
        return f"TupleGetterVariable(index={idx}, doc={doc!r})"

    def python_type(self) -> type:
        return _collections._tuplegetter

    def as_python_constant(self) -> "_collections._tuplegetter":
        return self.descriptor

    # _tuplegetter exposes __doc__ as a T_OBJECT member.
    # https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L2717-L2721
    tp_members = {"__doc__": Member(getset_build(lambda s: s.descriptor.__doc__))}

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2636-L2663
        if obj is None:
            return self
        _, (idx, _) = self.descriptor.__reduce__()
        return obj.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(idx)], {}
        )

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # _tuplegetter fields are read-only; tuplegetter_descr_set always
        # raises AttributeError for both set and delete.
        # https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2665-L2673
        msg = "can't delete attribute" if value is None else "can't set attribute"
        raise_observed_exception(AttributeError, tx, args=[msg])
<|endoftext|>"""
Core variable tracking functionality for Dynamo. This module defines the fundamental
classes and systems used to track and manage variables during Dynamo's operation.

The module provides:
1. VariableTracker - The base class for tracking variables during compilation
2. MutationType system - Classes for tracking and managing mutations to variables
3. Source type management - Utilities for tracking variable origins and scope
4. Variable state management - Tools for managing variable state and transformations

These components form the foundation of Dynamo's variable handling system,
enabling accurate tracking and transformation of Python code into optimized
computations.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import inspect
import logging
import textwrap
from collections.abc import Callable, ItemsView, KeysView, ValuesView
from contextvars import ContextVar
from enum import Enum, IntFlag
from typing import Any, NamedTuple, NoReturn, TYPE_CHECKING, TypeAlias

from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PyNumberSlots,
    PySequenceSlots,
    PyTypeSlots,
)

from .. import graph_break_hints, variables
from ..current_scope_id import current_scope_id
from ..exc import (
    ObservedAttributeError,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
    Unsupported,
)
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, Source
from ..utils import format_source_range, istype


_RICHCOMPARE_OPS = frozenset(
    {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"}
)


if TYPE_CHECKING:
    from torch._guards import Guard
    from torch.fx.proxy import Node

    from ..codegen import PyCodegen
    from ..side_effects import SideEffects
    from ..symbolic_convert import InstructionTranslatorBase
    from .constant import ConstantVariable
    from .functions import UserFunctionVariable


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SourceLocation:
    """Source position of the bytecode instruction that generated a VariableTracker."""

    filename: str
    lineno: int
    end_lineno: int | None = None
    col_offset: int | None = None
    end_col_offset: int | None = None

    def format(self) -> str:
        result = f'  File "{self.filename}", line {self.lineno}\n'
        source = format_source_range(
            self.filename,
            self.lineno,
            self.end_lineno,
            self.col_offset,
            self.end_col_offset,
        )
        if source:
            result += textwrap.indent(source.rstrip("\n"), "    ") + "\n"
        return result


# Tracks active method calls on VariableTracker instances to detect self-referential
# calls (e.g., as_python_constant on a list that contains itself). Maps
# (id(instance), id(original_method)) tuples to track which calls are in progress.
# We use id(original_method) rather than the method name string so that super()
# delegation within a class hierarchy (e.g. CustomClassObjectVariable.as_python_constant
# calling UserDefinedObjectVariable.as_python_constant) is not a false positive.
_vt_active_calls: ContextVar[set[tuple[int, int]] | None] = ContextVar(
    "_vt_active_calls", default=None
)


class SourceType(Enum):
    """
    This Enum divides VariableTracker into 2 cases, depending on the variable
    it represents:
    - already existed that Dynamo began tracking while introspection (Existing)
    - is a new variable that is created during Dynamo introspection (New)

    In general, we have these invariants:
    1. for `VariableTracker` associated with `Existing`, its `source` field must not be None.
    2. for `VariableTracker` associated with `New`, most of the time its
       `source` field is None, except for cases like side effect codegen for
       `AttributeMutationNew`, during which we generate a
       `LocalSource('tmp...')` for such variable, to facilitate codegen.
    """

    Existing = 0
    New = 1


class MutationType:
    """
    Base class for Variable.mutation_type. It encodes information about
    1. The type of mutation Dynamo allows on the variable.
    2. Whether the value represented by this variable already existed before
    Dynamo tracing.
    """

    def __init__(self, typ: SourceType) -> None:
        # In HigherOrderOperator tracing, we need to distinguish
        # between MutationTypes inside the HigherOrderOperator and
        # ones outside it. For example, it is not safe to mutate
        # `a` in the following example because it was constructed
        # in a different scope.
        #
        # def f(x):
        #     a = 1
        #     def g(x):
        #         nonlocal a
        #         a = 2
        #         return x
        #     return wrap(g, x) + a
        #
        # We use self.scope to distinguish this.
        # scope == 0: The object was an existing variable
        # scope == 1: The object was created while Dynamo
        #             was introspecting a function
        #             (and no HigherOrderOps were involved)
        # scope >= 2: The object was created through
        #             Dynamo introspection of a HigherOrderOp.
        #             The exact number corresponds to the level
        #             of nested HigherOrderOps.
        if typ is SourceType.Existing:
            self.scope = 0
        elif typ is SourceType.New:
            self.scope = current_scope_id()
        else:
            unimplemented(
                gb_type="Unsupported SourceType",
                context=f"MutationType.__init__ {self} {typ}",
                explanation=f"Dynamo does not support the type `{typ}`",
                hints=[
                    "This branch is not supposed to be reachable.",
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )


class AttrMutationKind(Enum):
    # Replay through STORE_ATTR/DELETE_ATTR, preserving descriptor-aware
    # setattr/delattr semantics. Used for slots, getset descriptors, and other
    # ordinary attribute writes that should not bypass descriptors.
    GENERIC_SETATTR = 0
    # Replay a direct object.__dict__ mutation while bypassing descriptors.
    INSTANCE_DICT = 1


class ValueMutationNew(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value itself (rather than its attributes).
    2. The value is created by the bytecode Dynamo is tracing through.

    For instance, Dynamo could model a newly created list with this marker,
    indicating that while we need to model mutations to this list, we don't have
    to emit bytecode for these mutations if the list doesn't escape into the
    Python world.
    """

    def __init__(self) -> None:
        super().__init__(SourceType.New)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


class ValueMutationExisting(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value itself (rather than its attributes).
    2. The value exists before Dynamo tracing started.

    For instance, Dynamo could model a pre-existing list with this marker,
    indicating that if we encounter mutations to this list, we need to buffer
    and re-apply those mutations after the graph runs, since the list might be
    used afterwards in Python.
    """

    # A flag to indicate whether mutation happened on the associated
    # `VariableTracker`. This enables SideEffects to accurately and quickly
    # filter out which pre-existing values it needs to generate mutation for.
    is_modified: bool

    def __init__(self, is_modified: bool = False) -> None:
        super().__init__(SourceType.Existing)
        self.is_modified = is_modified


class AttributeMutation(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates that Dynamo
    allows mutation on the value's attributes.
    """


class AttributeMutationExisting(AttributeMutation):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value's attributes.
    2. The value exists before Dynamo tracing started.

    For instance, Dynamo could model a pre-existing object with this marker,
    indicating that if we encounter mutations to this object, we need to buffer
    then re-apply those mutations after the graph runs, since the object might
    be used afterwards in Python.
    """

    def __init__(self) -> None:
        super().__init__(SourceType.Existing)


class AttributeMutationNew(AttributeMutation):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value's attributes.
    2. The value is created by the bytecode Dynamo is tracing through.

    For instance, Dynamo could model a newly created object with this marker,
    indicating that while we need to model mutations to this object, we don't
    have to emit bytecode for these mutations if the object doesn't escape into
    the Python world.
    """

    def __init__(self, cls_source: Source | None = None) -> None:
        super().__init__(SourceType.New)
        self.cls_source = cls_source


def _is_top_level_scope(scope_id: int) -> bool:
    return scope_id == 1


def is_side_effect_safe(m: MutationType) -> bool:
    scope_id = current_scope_id()

    # In the top-level scope (if no HigherOrderOperators are involved),
    # we are allowed to modify variables created in this scope as well
    # as existing variables.
    if _is_top_level_scope(scope_id):
        return True
    # Otherwise, only allow local mutation of variables created in the current scope
    return m.scope == scope_id


def maybe_get_python_type(obj: VariableTracker) -> type:
    try:
        return obj.python_type()
    except NotImplementedError:
        unimplemented(
            gb_type="Unsupported python_type() call",
            context=f"{obj} does not implement python_type()",
            explanation="This VariableTracker does not implement python_type(), "
            "which is required for object protocol operations.",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )


class NO_SUCH_SUBOBJ:
    """Sentinel indicating no concrete Python object is available."""


class MethodFlags(IntFlag):
    """Calling convention for a `Method` handler, mirroring CPython's
    PyMethodDef ml_flags. `call_method` enforces arity from these before
    invoking the handler, centralizing the raise_args_mismatch boilerplate."""

    VARARGS = 1  # any positional args; kwargs rejected unless KEYWORDS is set
    KEYWORDS = 2  # kwargs allowed (combine with VARARGS)
    NOARGS = 4  # exactly zero args and zero kwargs
    O = 8  # exactly one positional arg, no kwargs


def _check_method_arity(
    vt: VariableTracker,
    tx: InstructionTranslatorBase,
    name: str,
    flags: MethodFlags,
    args: Any,
    kwargs: Any,
) -> None:
    # Centralized arity check for a tp_methods handler, driven by MethodFlags,
    # raising the same TypeErrors CPython raises for builtin methods. Shared by
    # Method and callers that run the handler directly (e.g. tensor.py).
    n = len(args)
    qualname = f"{vt.python_type_name()}.{name}"
    if kwargs and not (flags & MethodFlags.KEYWORDS):
        raise_type_error(tx, f"{qualname}() takes no keyword arguments")
    if flags & (MethodFlags.NOARGS | MethodFlags.O):
        if flags & MethodFlags.NOARGS and args:
            raise_type_error(tx, f"{qualname}() takes no arguments ({n} given)")
        if flags & MethodFlags.O and n != 1:
            raise_type_error(tx, f"{qualname}() takes exactly one argument ({n} given)")


# CPython PyMethodDef.ml_flags bits (Include/methodobject.h); MethodFlags above
# deliberately mirrors the low four so the mapping is near-identity.
_METH_VARARGS, _METH_KEYWORDS, _METH_NOARGS, _METH_O, _METH_FASTCALL = 1, 2, 4, 8, 0x80


@functools.cache
def _flags_from_ml_flags(python_type: type, name: str) -> MethodFlags:
    """Arity convention for `python_type.name`, read from its CPython
    PyMethodDef.ml_flags. Falls back to VARARGS|KEYWORDS when python_type has no
    such attribute or it carries no ml_flags (slot wrappers, pure-Python
    functions)."""
    import torch

    default = MethodFlags.VARARGS | MethodFlags.KEYWORDS
    fn = getattr(python_type, name, None)
    ml = None if fn is None else torch._C._dynamo.eval_frame.get_method_ml_flags(fn)
    if ml is None:
        return default
    flags = MethodFlags(0)
    if ml & _METH_NOARGS:
        flags |= MethodFlags.NOARGS
    if ml & _METH_O:
        flags |= MethodFlags.O
    if ml & (_METH_VARARGS | _METH_FASTCALL):
        flags |= MethodFlags.VARARGS
    if ml & _METH_KEYWORDS:
        flags |= MethodFlags.KEYWORDS
    return flags or default


def _derive_method_flags(vt: VariableTracker, name: str) -> MethodFlags:
    """Resolve the arity flags for method `name` from the VT's arity-reference
    type (see VariableTracker.method_flags_type)."""
    return _flags_from_ml_flags(vt.method_flags_type(), name)


@dataclasses.dataclass(slots=True)
class Method:
    """Declarative entry in a VariableTracker's `tp_methods` table, analogous
    to CPython's PyMethodDef.

    Handlers have signature `(self, tx, args, kwargs)` and return the result
    VariableTracker, or None to decline the call (the equivalent of the old
    `super().call_method` fall-through) so `call_method` continues to the
    object-protocol dispatch below. The method name is the tp_methods key this
    entry is stored under; its arity convention is derived
    on demand from that method's ml_flags (see _derive_method_flags)."""

    handler: Callable[..., VariableTracker | None]

    def __call__(
        self,
        vt: VariableTracker,
        tx: InstructionTranslatorBase,
        name: str,
        args: Any,
        kwargs: Any,
    ) -> VariableTracker | None:
        flags = _derive_method_flags(vt, name)
        _check_method_arity(vt, tx, name, flags, args, kwargs)
        return self.handler(vt, tx, args, kwargs)


@dataclasses.dataclass(slots=True)
class GetSet:
    """`tp_getset` entry, analogous to CPython's PyGetSetDef. `getter`
    `(self, tx) -> VT | None` (None declines); `setter`
    `(self, tx, value) -> VT | None`, None for read-only."""

    getter: Callable[..., VariableTracker | None]
    setter: Callable[..., VariableTracker | None] | None = None


@dataclasses.dataclass(slots=True)
class Member:
    """`tp_members` entry, analogous to CPython's PyMemberDef. Same shape as
    GetSet; a distinct type so members and getsets never share a class."""

    getter: Callable[..., VariableTracker | None]
    setter: Callable[..., VariableTracker | None] | None = None


def getset_read(
    accessor: Callable[[Any], VariableTracker],
) -> Callable[..., VariableTracker]:
    """Getter for a GetSet/Member whose value is an already-built VT."""
    return lambda self, tx: accessor(self)


def getset_build(
    accessor: Callable[[Any], Any],
) -> Callable[..., VariableTracker]:
    """Getter that builds a VT from the raw value returned by `accessor`."""
    return lambda self, tx: VariableTracker.build(tx, accessor(self))


def unsupported_attr(name: str) -> Callable[..., VariableTracker | None]:
    def graph_break(
        vt: VariableTracker, tx: InstructionTranslatorBase
    ) -> VariableTracker | None:
        unimplemented(
            gb_type="Unsupported attribute",
            context=f"attr_unsupported {vt} {name}",
            explanation=f"{type(vt).__name__} does not support attribute '{name}'",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    return graph_break


# This helps users of `as_python_constant` to catch unimplemented error with
# more information; it inherits `NotImplementedError` for backward
# compatibility reasons.
class AsPythonConstantNotImplementedError(NotImplementedError):
    vt: VariableTracker

    def __init__(self, vt: VariableTracker, msg: str | None = None) -> None:
        msg = f"{vt} is not a constant" if msg is None else msg
        super().__init__(msg)
        self.vt = vt


# ---------------------------------------------------------------------------
# SLOTDEFS: Dynamo's analog of CPython's static slotdefs[] table
# (Objects/typeobject.c).  One global table, shared by every type, mapping a
# dunder name to the VariableTracker method that implements it (``impl``, the
# slotdefs "function" column) and a ``wrapper`` that adapts a generic call into
# the right impl invocation (the slotdefs "wrapper" column: arity + operand
# order).  CPython also stores a slot offset per entry; Dynamo does not need it
# -- dispatch only cares which impl to call, and MRO lookup already tells us
# whether a name resolves.  Per-VT variation is which impl method is overridden;
# the table itself never varies.  Lives here alongside the ``*_impl`` methods it
# names so mapping and implementations stay together; consumers (call_method
# here, getattro in object_protocol) both read it without an import cycle.
# ---------------------------------------------------------------------------


def _wrap_unaryfunc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 0:
        raise_type_error(tx, f"expected 0 arguments, got {len(args)}")
    return func(self, tx)


def _wrap_binaryfunc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    other = args[0]
    return func(self, tx, other)


def _wrap_binaryfunc_r(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # Reflected binary op (``__radd__``): same impl, operands from the right.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    other = args[0]
    return func(self, tx, other, reverse=True)


def _wrap_ternaryfunc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # __pow__ is binary or ternary: ``x.__pow__(y)`` or ``x.__pow__(y, z)`` (mod).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) not in (1, 2):
        raise_type_error(tx, f"expected 1 or 2 arguments, got {len(args)}")
    z = args[1] if len(args) == 2 else None
    return func(self, tx, args[0], z)


def _wrap_ternaryfunc_r(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # Reflected __rpow__: operands from the right.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) not in (1, 2):
        raise_type_error(tx, f"expected 1 or 2 arguments, got {len(args)}")
    z = args[1] if len(args) == 2 else None
    return func(self, tx, args[0], z, reverse=True)


def _wrap_sq_setitem(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    from .object_protocol import getindex

    if len(args) != 2:
        raise_type_error(tx, f"expected 2 arguments, got {len(args)}")
    [arg, value] = args
    i = getindex(tx, self, arg)
    return func(self, tx, i, value)


def _wrap_sq_item(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    from .object_protocol import getindex

    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")

    [arg] = args
    i = getindex(tx, self, arg)
    return func(self, tx, i)


def _wrap_indexargfunc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    from .object_protocol import pynumber_as_ssize_t

    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")

    [o] = args
    i = pynumber_as_ssize_t(tx, o, err=OverflowError)
    return func(self, tx, i)


def _wrap_objobjargproc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 2:
        raise_type_error(tx, f"expected 2 arguments, got {len(args)}")
    [key, value] = args
    return func(self, tx, key, value)


def _wrap_delitem(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    [key] = args
    return func(self, tx, key, None)


def _wrap_sq_delitem(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    from .object_protocol import getindex

    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    [arg] = args
    i = getindex(tx, self, arg)
    return func(self, tx, i, None)


def _wrap_objobjproc(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # sq_contains via __contains__: one arg, impl returns a bool.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    [other] = args
    return func(self, tx, other)


def _make_richcmp(
    op: str,
) -> Callable[
    [
        VariableTracker,
        InstructionTranslatorBase,
        Callable[..., VariableTracker],
        list[VariableTracker],
        dict[str, VariableTracker],
    ],
    VariableTracker,
]:
    # tp_richcompare via __lt__/.../__ge__: one impl, op baked per dunder.
    def wrapper(
        self: VariableTracker,
        tx: InstructionTranslatorBase,
        func: Callable[..., VariableTracker],
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if kwargs:
            raise_type_error(tx, "this method takes no keyword arguments")
        if len(args) != 1:
            raise_type_error(tx, f"expected 1 argument, got {len(args)}")
        return func(self, tx, args[0], op)

    return wrapper


richcmp_lt = _make_richcmp("__lt__")
richcmp_le = _make_richcmp("__le__")
richcmp_eq = _make_richcmp("__eq__")
richcmp_ne = _make_richcmp("__ne__")
richcmp_gt = _make_richcmp("__gt__")
richcmp_ge = _make_richcmp("__ge__")


def wrap_call(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # tp_call via __call__: variadic (kwargs handled by the call_method branch).
    return func(self, tx, args, kwargs)


def _wrap_init(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # tp_init via __init__: variadic, forwards kwargs (e.g. dict(a=1),
    # defaultdict(list, d, c=3)).
    return func(self, tx, args, kwargs)


def _wrap_setattr(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # tp_setattro via __setattr__: (name, value).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 2:
        raise_type_error(tx, f"expected 2 arguments, got {len(args)}")
    return func(self, tx, args[0], args[1])


def _wrap_delattr(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # tp_setattro via __delattr__: (name,).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    return func(self, tx, args[0])


def _wrap_getattro(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # tp_getattro via __getattribute__: impl takes the attribute name as a str.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    return func(self, tx, args[0].as_python_constant())


def _wrap_descr_get(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # tp_descr_get via __get__(obj, owner=None): owner defaults to type(obj).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) not in (1, 2):
        raise_type_error(tx, f"expected 1 or 2 arguments, got {len(args)}")
    obj = args[0]
    owner = args[1] if len(args) > 1 else obj.tp_getattro_impl(tx, "__class__")
    return func(self, tx, obj, owner)


def _wrap_descr_set(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # tp_descr_set via __set__(obj, value): 2 args.
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 2:
        raise_type_error(tx, f"expected 2 arguments, got {len(args)}")
    return func(self, tx, args[0], args[1])


def _wrap_descr_delete(
    self: VariableTracker,
    tx: InstructionTranslatorBase,
    func: Callable[..., VariableTracker],
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # tp_descr_set via __delete__(obj): value=None signals delete (CPython
    # passes value==NULL to the shared tp_descr_set slot).
    if kwargs:
        raise_type_error(tx, "this method takes no keyword arguments")
    if len(args) != 1:
        raise_type_error(tx, f"expected 1 argument, got {len(args)}")
    return func(self, tx, args[0], None)


class SlotGroup(Enum):
    """A CPython slot group.  The value is the group's index into
    get_type_slots()'s 4-tuple."""

    SEQUENCE = 0
    MAPPING = 1
    NUMBER = 2
    TYPE = 3


@dataclasses.dataclass(frozen=True)
class Slot:
    """A CPython type slot: the name of the VariableTracker method implementing
    the slot function.
    """

    impl: str

    def __call__(
        self,
        vt: VariableTracker,
        tx: InstructionTranslatorBase,
        *args: Any,
        **kwargs: Any,
    ) -> VariableTracker:
        return getattr(type(vt), self.impl)(vt, tx, *args, **kwargs)


class PyNumberMethods(NamedTuple):
    nb_bool: Slot | None
    nb_index: Slot | None
    nb_int: Slot | None
    nb_float: Slot | None
    nb_negative: Slot | None
    nb_positive: Slot | None
    nb_absolute: Slot | None
    nb_invert: Slot | None
    nb_add: Slot | None
    nb_inplace_add: Slot | None
    nb_subtract: Slot | None
    nb_inplace_subtract: Slot | None
    nb_multiply: Slot | None
    nb_inplace_multiply: Slot | None
    nb_remainder: Slot | None
    nb_inplace_remainder: Slot | None
    nb_power: Slot | None
    nb_inplace_power: Slot | None
    nb_lshift: Slot | None
    nb_inplace_lshift: Slot | None
    nb_rshift: Slot | None
    nb_inplace_rshift: Slot | None
    nb_and: Slot | None
    nb_inplace_and: Slot | None
    nb_xor: Slot | None
    nb_inplace_xor: Slot | None
    nb_or: Slot | None
    nb_inplace_or: Slot | None
    nb_floor_divide: Slot | None
    nb_inplace_floor_divide: Slot | None
    nb_true_divide: Slot | None
    nb_inplace_true_divide: Slot | None
    nb_divmod: Slot | None
    nb_matrix_multiply: Slot | None
    nb_inplace_matrix_multiply: Slot | None


class PySequenceMethods(NamedTuple):
    sq_length: Slot | None
    sq_concat: Slot | None
    sq_repeat: Slot | None
    sq_item: Slot | None
    sq_ass_item: Slot | None
    sq_contains: Slot | None
    sq_inplace_concat: Slot | None
    sq_inplace_repeat: Slot | None


class PyMappingMethods(NamedTuple):
    mp_length: Slot | None
    mp_subscript: Slot | None
    mp_ass_subscript: Slot | None


class PyTypeObject(NamedTuple):
    tp_repr: Slot | None
    tp_hash: Slot | None
    tp_call: Slot | None
    tp_str: Slot | None
    tp_getattro: Slot | None
    tp_setattro: Slot | None
    tp_richcompare: Slot | None
    tp_iter: Slot | None
    tp_iternext: Slot | None
    tp_descr_get: Slot | None
    tp_descr_set: Slot | None
    tp_init: Slot | None
    tp_as_number: PyNumberMethods
    tp_as_sequence: PySequenceMethods
    tp_as_mapping: PyMappingMethods


# The tp_ slot fields of PyTypeObject (everything but the sub-struct pointers).
_TYPE_FIELDS = tuple(f for f in PyTypeObject._fields if not f.startswith("tp_as_"))

# Per group: C slot bit -> struct field name, keyed via the enum's UPPERCASE name.
_BIT_FIELD: dict[SlotGroup, dict[int, str]] = {
    SlotGroup.NUMBER: {
        getattr(PyNumberSlots, f.upper()): f for f in PyNumberMethods._fields
    },
    SlotGroup.SEQUENCE: {
        getattr(PySequenceSlots, f.upper()): f for f in PySequenceMethods._fields
    },
    SlotGroup.MAPPING: {
        getattr(PyMappingSlots, f.upper()): f for f in PyMappingMethods._fields
    },
    SlotGroup.TYPE: {getattr(PyTypeSlots, f.upper()): f for f in _TYPE_FIELDS},
}


# The actual slot function (a VariableTracker method) each struct field dispatches to
_SLOT_FN: dict[SlotGroup, dict[str, str]] = {
    SlotGroup.NUMBER: {f: f"{f}_impl" for f in PyNumberMethods._fields},
    SlotGroup.SEQUENCE: {f: f"{f}_impl" for f in PySequenceMethods._fields},
    SlotGroup.MAPPING: {f: f"{f}_impl" for f in PyMappingMethods._fields},
    SlotGroup.TYPE: {
        **{f: f"{f}_impl" for f in _TYPE_FIELDS},
        "tp_call": "call_function",
        "tp_hash": "hash_impl",
    },
}

# _SLOT_FN must name a slot fn for exactly each struct field.
for _group, _fns in _SLOT_FN.items():
    if set(_fns) != set(_BIT_FIELD[_group].values()):
        raise AssertionError(f"{_group.name}: _SLOT_FN fields != struct fields")


def _fill(group: SlotGroup, masks: tuple[int, int, int, int]) -> dict[str, Any]:
    # {field -> Slot(slot fn) if the type fills the slot, else None}
    mask = masks[group.value]
    fn_map = _SLOT_FN[group]
    return {
        field: Slot(fn_map[field]) if has_slot(mask, bit) else None
        for bit, field in _BIT_FIELD[group].items()
    }


@functools.cache
def _tp_type(obj_type: type) -> PyTypeObject:
    # obj_type's whole slot table (PyTypeObject-like).  Independent of _tp_dict's
    # cross-group collapse, so a type with both sq_length and mp_length shows up in
    # both the sequence and mapping views (as CPython's separate sub-structs do).
    masks = get_type_slots(obj_type)
    return PyTypeObject(
        tp_as_number=PyNumberMethods(**_fill(SlotGroup.NUMBER, masks)),
        tp_as_sequence=PySequenceMethods(**_fill(SlotGroup.SEQUENCE, masks)),
        tp_as_mapping=PyMappingMethods(**_fill(SlotGroup.MAPPING, masks)),
        **_fill(SlotGroup.TYPE, masks),
    )


WrapperType: TypeAlias = Callable[
    [
        "VariableTracker",
        "InstructionTranslatorBase",
        Callable[..., "VariableTracker"],
        list["VariableTracker"],
        dict[str, "VariableTracker"],
    ],
    "VariableTracker",
]


@dataclasses.dataclass(frozen=True)
class SlotDef:
    """One slotdefs entry: dunder name -> (impl method, call wrapper).

    ``impl`` is the VariableTracker method implementing the behavior; ``wrapper``
    adapts a generic ``(self, tx, args)`` call to it (arity + operand order).  A
    reflected op reuses the forward impl with ``_wrap_binaryfunc_r``; an in-place
    op is a plain binary call into its own ``nb_inplace_*`` impl -- so no
    reverse/inplace flags are needed.

    ``group`` is the CPython slot group (``SlotGroup``) the entry lives in and
    ``slot`` is the bit within it; together they decide whether a real type
    implements the slot.
    """

    name: str
    impl: str
    group: SlotGroup
    slot: int
    wrapper: WrapperType

    def __call__(
        self,
        vt: VariableTracker,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """Call the impl via the wrapper, passing self, tx, and args."""
        func = getattr(type(vt), self.impl)
        return self.wrapper(vt, tx, func, args, kwargs)


def SQSLOT(name: str, impl: str, slot: int, wrapper: WrapperType) -> SlotDef:
    return SlotDef(name, impl, SlotGroup.SEQUENCE, slot, wrapper)


def MPSLOT(name: str, impl: str, slot: int, wrapper: WrapperType) -> SlotDef:
    return SlotDef(name, impl, SlotGroup.MAPPING, slot, wrapper)


def NBSLOT(name: str, impl: str, slot: int, wrapper: WrapperType) -> SlotDef:
    return SlotDef(name, impl, SlotGroup.NUMBER, slot, wrapper)


def TPSLOT(name: str, impl: str, slot: int, wrapper: WrapperType) -> SlotDef:
    return SlotDef(name, impl, SlotGroup.TYPE, slot, wrapper)


# The declarative slot table (mirrors CPython's static slotdefs[] array).  Each
# entry: dunder name, impl method, ``group`` + ``slot`` (which slot sub-struct
# and bit -- reflected ops share the forward slot's group/slot), and wrapper
# (call shape).
_SLOTDEFS: list[SlotDef] = [
    # SlotDef("__getattribute__", ),
    # SlotDef("__getattr__", ),
    # SlotDef("__setattr__", ),
    # SlotDef("__delattr__", ),
    TPSLOT("__repr__", "tp_repr_impl", PyTypeSlots.TP_REPR, _wrap_unaryfunc),
    # TPSLOT("__hash__", "tp_hash_impl", PyTypeSlots.TP_HASH, _wrap_unaryfunc),
    TPSLOT("__call__", "call_function", PyTypeSlots.TP_CALL, wrap_call),
    TPSLOT("__str__", "tp_str_impl", PyTypeSlots.TP_STR, _wrap_unaryfunc),
    TPSLOT(
        "__getattribute__",
        "tp_getattro_impl",
        PyTypeSlots.TP_GETATTRO,
        _wrap_getattro,
    ),
    TPSLOT(
        "__getattr__",
        "tp_getattro_impl",
        PyTypeSlots.TP_GETATTRO,
        _wrap_getattro,
    ),
    # This needs one to model tp_setattro first + PyObject_GenericSetAttr / PyObject_GenericDelAttr
    # TPSLOT(
    #     "__setattr__",
    #     "setattro_impl",
    #     PyTypeSlots.TP_SETATTRO,
    #     _wrap_setattr,
    # ),
    # TPSLOT(
    #     "__delattr__",
    #     "delattro_impl",
    #     PyTypeSlots.TP_SETATTRO,
    #     _wrap_delattr,
    # ),
    TPSLOT(
        "__lt__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_lt,
    ),
    TPSLOT(
        "__le__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_le,
    ),
    TPSLOT(
        "__eq__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_eq,
    ),
    TPSLOT(
        "__ne__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_ne,
    ),
    TPSLOT(
        "__gt__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_gt,
    ),
    TPSLOT(
        "__ge__",
        "tp_richcompare_impl",
        PyTypeSlots.TP_RICHCOMPARE,
        richcmp_ge,
    ),
    TPSLOT("__iter__", "tp_iter_impl", PyTypeSlots.TP_ITER, _wrap_unaryfunc),
    TPSLOT(
        "__next__",
        "tp_iternext_impl",
        PyTypeSlots.TP_ITERNEXT,
        _wrap_unaryfunc,
    ),
    TPSLOT(
        "__get__",
        "tp_descr_get_impl",
        PyTypeSlots.TP_DESCR_GET,
        _wrap_descr_get,
    ),
    TPSLOT(
        "__set__",
        "tp_descr_set_impl",
        PyTypeSlots.TP_DESCR_SET,
        _wrap_descr_set,
    ),
    TPSLOT(
        "__delete__",
        "tp_descr_set_impl",
        PyTypeSlots.TP_DESCR_SET,
        _wrap_descr_delete,
    ),
    TPSLOT("__init__", "tp_init_impl", PyTypeSlots.TP_INIT, _wrap_init),
    # SlotDef("__new__", ...), # missing
    # SlotDef("__del__", ...), # missing
    # SlotDef("__buffer__", ...), # missing
    # SlotDef("__release_buffer__", ...), # missing
    # SlotDef("__await__", ...), # missing
    # SlotDef("__aiter__", ...), # missing
    # SlotDef("__anext__", ...), # missing
    NBSLOT(
        "__add__",
        "nb_add_impl",
        PyNumberSlots.NB_ADD,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__radd__",
        "nb_add_impl",
        PyNumberSlots.NB_ADD,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__sub__",
        "nb_subtract_impl",
        PyNumberSlots.NB_SUBTRACT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rsub__",
        "nb_subtract_impl",
        PyNumberSlots.NB_SUBTRACT,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__mul__",
        "nb_multiply_impl",
        PyNumberSlots.NB_MULTIPLY,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rmul__",
        "nb_multiply_impl",
        PyNumberSlots.NB_MULTIPLY,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__mod__",
        "nb_remainder_impl",
        PyNumberSlots.NB_REMAINDER,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rmod__",
        "nb_remainder_impl",
        PyNumberSlots.NB_REMAINDER,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__divmod__",
        "nb_divmod_impl",
        PyNumberSlots.NB_DIVMOD,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rdivmod__",
        "nb_divmod_impl",
        PyNumberSlots.NB_DIVMOD,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__pow__",
        "nb_power_impl",
        PyNumberSlots.NB_POWER,
        _wrap_ternaryfunc,
    ),
    NBSLOT(
        "__rpow__",
        "nb_power_impl",
        PyNumberSlots.NB_POWER,
        _wrap_ternaryfunc_r,
    ),
    NBSLOT(
        "__neg__",
        "nb_negative_impl",
        PyNumberSlots.NB_NEGATIVE,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__pos__",
        "nb_positive_impl",
        PyNumberSlots.NB_POSITIVE,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__abs__",
        "nb_absolute_impl",
        PyNumberSlots.NB_ABSOLUTE,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__bool__",
        "nb_bool_impl",
        PyNumberSlots.NB_BOOL,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__invert__",
        "nb_invert_impl",
        PyNumberSlots.NB_INVERT,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__lshift__",
        "nb_lshift_impl",
        PyNumberSlots.NB_LSHIFT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rlshift__",
        "nb_lshift_impl",
        PyNumberSlots.NB_LSHIFT,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__rshift__",
        "nb_rshift_impl",
        PyNumberSlots.NB_RSHIFT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rrshift__",
        "nb_rshift_impl",
        PyNumberSlots.NB_RSHIFT,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__and__",
        "nb_and_impl",
        PyNumberSlots.NB_AND,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rand__",
        "nb_and_impl",
        PyNumberSlots.NB_AND,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__xor__",
        "nb_xor_impl",
        PyNumberSlots.NB_XOR,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rxor__",
        "nb_xor_impl",
        PyNumberSlots.NB_XOR,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__or__",
        "nb_or_impl",
        PyNumberSlots.NB_OR,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ror__",
        "nb_or_impl",
        PyNumberSlots.NB_OR,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__int__",
        "nb_int_impl",
        PyNumberSlots.NB_INT,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__float__",
        "nb_float_impl",
        PyNumberSlots.NB_FLOAT,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__iadd__",
        "nb_inplace_add_impl",
        PyNumberSlots.NB_INPLACE_ADD,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__isub__",
        "nb_inplace_subtract_impl",
        PyNumberSlots.NB_INPLACE_SUBTRACT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__imul__",
        "nb_inplace_multiply_impl",
        PyNumberSlots.NB_INPLACE_MULTIPLY,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__imod__",
        "nb_inplace_remainder_impl",
        PyNumberSlots.NB_INPLACE_REMAINDER,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ipow__",
        "nb_inplace_power_impl",
        PyNumberSlots.NB_INPLACE_POWER,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ilshift__",
        "nb_inplace_lshift_impl",
        PyNumberSlots.NB_INPLACE_LSHIFT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__irshift__",
        "nb_inplace_rshift_impl",
        PyNumberSlots.NB_INPLACE_RSHIFT,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__iand__",
        "nb_inplace_and_impl",
        PyNumberSlots.NB_INPLACE_AND,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ixor__",
        "nb_inplace_xor_impl",
        PyNumberSlots.NB_INPLACE_XOR,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__ior__",
        "nb_inplace_or_impl",
        PyNumberSlots.NB_INPLACE_OR,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__floordiv__",
        "nb_floor_divide_impl",
        PyNumberSlots.NB_FLOOR_DIVIDE,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rfloordiv__",
        "nb_floor_divide_impl",
        PyNumberSlots.NB_FLOOR_DIVIDE,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__truediv__",
        "nb_true_divide_impl",
        PyNumberSlots.NB_TRUE_DIVIDE,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rtruediv__",
        "nb_true_divide_impl",
        PyNumberSlots.NB_TRUE_DIVIDE,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__ifloordiv__",
        "nb_inplace_floor_divide_impl",
        PyNumberSlots.NB_INPLACE_FLOOR_DIVIDE,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__itruediv__",
        "nb_inplace_true_divide_impl",
        PyNumberSlots.NB_INPLACE_TRUE_DIVIDE,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__index__",
        "nb_index_impl",
        PyNumberSlots.NB_INDEX,
        _wrap_unaryfunc,
    ),
    NBSLOT(
        "__matmul__",
        "nb_matrix_multiply_impl",
        PyNumberSlots.NB_MATRIX_MULTIPLY,
        _wrap_binaryfunc,
    ),
    NBSLOT(
        "__rmatmul__",
        "nb_matrix_multiply_impl",
        PyNumberSlots.NB_MATRIX_MULTIPLY,
        _wrap_binaryfunc_r,
    ),
    NBSLOT(
        "__imatmul__",
        "nb_inplace_matrix_multiply_impl",
        PyNumberSlots.NB_INPLACE_MATRIX_MULTIPLY,
        _wrap_binaryfunc,
    ),
    # Mapping
    MPSLOT(
        "__len__",
        "mp_length_impl",
        PyMappingSlots.MP_LENGTH,
        _wrap_unaryfunc,
    ),
    MPSLOT(
        "__getitem__",
        "mp_subscript_impl",
        PyMappingSlots.MP_SUBSCRIPT,
        _wrap_binaryfunc,
    ),
    MPSLOT(
        "__setitem__",
        "mp_ass_subscript_impl",
        PyMappingSlots.MP_ASS_SUBSCRIPT,
        _wrap_objobjargproc,
    ),
    MPSLOT(
        "__delitem__",
        "mp_ass_subscript_impl",
        PyMappingSlots.MP_ASS_SUBSCRIPT,
        _wrap_delitem,
    ),
    # Sequence
    SQSLOT(
        "__len__",
        "sq_length_impl",
        PySequenceSlots.SQ_LENGTH,
        _wrap_unaryfunc,
    ),
    SQSLOT(
        "__add__",
        "sq_concat_impl",
        PySequenceSlots.SQ_CONCAT,
        _wrap_binaryfunc,
    ),
    SQSLOT(
        "__mul__",
        "sq_repeat_impl",
        PySequenceSlots.SQ_REPEAT,
        _wrap_indexargfunc,
    ),
    SQSLOT(
        "__rmul__",
        "sq_repeat_impl",
        PySequenceSlots.SQ_REPEAT,
        _wrap_indexargfunc,
    ),
    SQSLOT(
        "__getitem__",
        "sq_item_impl",
        PySequenceSlots.SQ_ITEM,
        _wrap_sq_item,
    ),
    SQSLOT(
        "__setitem__",
        "sq_ass_item_impl",
        PySequenceSlots.SQ_ASS_ITEM,
        _wrap_sq_setitem,
    ),
    SQSLOT(
        "__delitem__",
        "sq_ass_item_impl",
        PySequenceSlots.SQ_ASS_ITEM,
        _wrap_sq_delitem,
    ),
    SQSLOT(
        "__contains__",
        "sq_contains_impl",
        PySequenceSlots.SQ_CONTAINS,
        _wrap_objobjproc,
    ),
    SQSLOT(
        "__iadd__",
        "sq_inplace_concat_impl",
        PySequenceSlots.SQ_INPLACE_CONCAT,
        _wrap_binaryfunc,
    ),
    SQSLOT(
        "__imul__",
        "sq_inplace_repeat_impl",
        PySequenceSlots.SQ_INPLACE_REPEAT,
        _wrap_indexargfunc,
    ),
]


@functools.cache
def _compute_slotdefs(obj_type: type) -> dict[str, SlotDef]:
    masks = get_type_slots(obj_type)
    d: dict[str, SlotDef] = {}
    for sd in _SLOTDEFS:
        if sd.name in d:
            continue
        if has_slot(masks[sd.group.value], sd.slot):
            d[sd.name] = sd
    return d


class VariableTrackerMeta(type):
    all_subclasses: list[type] = []

    def __new__(
        mcs: type, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> type:
        # Determine which metaclass to use based on the class attributes
        # Classes with _no_implicit_realize = True should NOT implicitly realize
        # (they need standard isinstance behavior to avoid infinite recursion)
        # Check if any base class has _no_implicit_realize set, or if it's in attrs
        no_implicit_realize = attrs.get("_no_implicit_realize", False) or any(
            getattr(base, "_no_implicit_realize", False) for base in bases
        )
        if no_implicit_realize or name == "VariableTracker":
            # Use base VariableTrackerMeta (no custom __instancecheck__)
            return super().__new__(VariableTrackerMeta, name, bases, attrs)
        else:
            # Use ImplicitRealizingVariableTrackerMeta for all other subclasses
            return super().__new__(
                ImplicitRealizingVariableTrackerMeta, name, bases, attrs
            )

    def __init__(
        cls: type, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> None:
        super().__init__(name, bases, attrs)  # type: ignore[misc]
        VariableTrackerMeta.all_subclasses.append(cls)


class ImplicitRealizingVariableTrackerMeta(VariableTrackerMeta):
    def __instancecheck__(self, instance: object) -> bool:
        """Make isinstance work with LazyVariableTracker"""
        if instancecheck(LazyVariableTracker, instance):
            return instance.lazy_isinstance(self)  # pyrefly: ignore[missing-attribute]
        return instancecheck(self, instance)


class VariableTracker(metaclass=VariableTrackerMeta):
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.

    Prefer the factory function VariableTracker.build() over VariableTracker.__init__().
    """

    # The CPython type(s) this VT is designed to represent.
    # Single type or tuple of types. None means no static CPython type mapping
    # (e.g., dynamic types like UserDefinedObjectVariable, or Dynamo-internal VTs).
    _cpython_type: type | tuple[type, ...] | None = None

    # Declarative named-method table (analogous to CPython's tp_methods): maps a
    # method name to a Method describing its handler and calling convention.
    # `call_method` consults this before the object-protocol (tp_slot) dispatch.
    # Lookup walks the class MRO (see `_lookup_tp_table`), so a subclass table
    # adds to (rather than replaces) the entries it inherits, with the
    # most-derived class winning on name collisions.
    tp_methods: dict[str, Method] = {}
    # Declarative attribute tables, split to match CPython: tp_getset holds the
    # PyGetSetDef attributes, tp_members the PyMemberDef ones.
    tp_getset: dict[str, GetSet] = {}
    tp_members: dict[str, Member] = {}

    def _lookup_tp_table(self, name: str, *table_attrs: str) -> Any:
        """Resolve *name* in the given declarative tables across the class MRO.

        Mirrors CPython method/getset/member inheritance: each class contributes
        its own table entries into its own tp_dict, and lookup walks the MRO with
        the most-derived class winning. A subclass table therefore *adds* to
        (rather than replaces) the entries it inherits. Within a class the tables
        are consulted in the order given.
        """
        for klass in type(self).__mro__:
            d = klass.__dict__
            for attr in table_attrs:
                table = d.get(attr)
                if table is not None and name in table:
                    return table[name]
        return None

    def lookup_tp_getset_member(self, name: str) -> GetSet | Member | None:
        return self._lookup_tp_table(name, "tp_getset", "tp_members")

    def lookup_tp_method(self, name: str) -> Method | None:
        return self._lookup_tp_table(name, "tp_methods")

    def method_flags_type(self) -> type:
        """Type whose CPython ml_flags define this VT's tp_methods arities
        (see _derive_method_flags). Defaults to python_type(); a VT whose
        python_type() is a pure-Python stand-in without C ml_flags (e.g.
        OrderedSet) overrides this to the builtin whose method arities it
        mirrors, so MethodFlags still enforces arity."""
        return maybe_get_python_type(self)

    # fields to leave unmodified in apply()
    _nonvar_fields = {
        "value",
        "guards",
        "source",
        "source_location",
        "mutation_type",
        "parents_tracker",
        "user_code_variable_name",
        "dict_vt",
    }

    # Lazily-created view of the instance __dict__, backed by the side effects table
    dict_vt: variables.DunderDictVariable | None = None

    def clone(self, **kwargs: Any) -> VariableTracker:
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    @classmethod
    def visit(
        cls,
        fn: Callable[[VariableTracker], None],
        value: Any,
        cache: dict[int, Any] | None = None,
        side_effects: SideEffects | None = None,
    ) -> None:
        """
        Walk value and call fn on all the VariableTracker instances.

        When side_effects is provided, also walks attributes stored in
        store_attr_mutations (e.g. dataclass fields set during tracing
        that aren't in the VT's __dict__).

        Implemented with an explicit worklist rather than recursion so that
        deeply chained structures (e.g. the ~N-node buffer built by
        itertools.tee over a long iterable) do not overflow the Python stack.
        Children are pushed in reverse so they are popped in their original
        order, matching the pre-order DFS the previous recursive version
        produced (callers rely on the visitation order, e.g. for graph output
        ordering).
        """
        if cache is None:
            cache = {}

        worklist = [value]
        while worklist:
            cur = worklist.pop()
            idx = id(cur)
            if idx in cache:
                continue
            # save `cur` to keep it alive and ensure id() isn't reused
            cache[idx] = cur

            children: list[Any]
            if isinstance(cur, VariableTracker):
                cur = cur.unwrap()
                fn(cur)
                cur = cur.unwrap()  # calling fn() might have realized it
                nonvars = cur._nonvar_fields
                children = [
                    subvalue
                    for key, subvalue in cur.__dict__.items()
                    if key not in nonvars
                ]
                if (
                    side_effects is not None
                    and cur in side_effects.store_attr_mutations
                ):
                    children.extend(side_effects.store_attr_mutations[cur].values())
            elif istype(cur, (list, tuple)):
                children = list(cur)
            elif istype(cur, (dict, collections.OrderedDict)):
                children = list(cur.values())
            else:
                continue
            worklist.extend(reversed(children))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def debug_repr(self) -> str:
        # Intended to be overridden to provide more info
        try:
            return repr(self.as_python_constant())
        except NotImplementedError:
            return repr(self)

    def python_type(self) -> type:
        """
        Abstract method to be implemented by subclasses of VariableTracker.

        This method should return the type represented by the instance of the subclass.
        The purpose is to provide a standardized way to retrieve the Python type information
        of the variable being tracked.

        Returns:
            type: The Python type (such as int, str, list, etc.) of the variable tracked by
                the subclass. If the type cannot be determined or is not relevant,
                leaving it undefined or invoking super() is always sound.

        Note:
            This is an abstract method and may be overridden in subclasses.

        Example:
            class SetVariable(VariableTracker):
                def python_type(self):
                    return set

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        try:
            return type(self.as_python_constant())
        except NotImplementedError:
            raise NotImplementedError(f"{self} has no type") from None

    def python_type_name(self) -> str:
        """
        Return the type name for the Python type this VariableTracker represents.

        Mirrors CPython's tp_name slot (PyTypeObject.tp_name). In Python 3.10+,
        type.__name__ matches CPython's tp_name exactly (e.g., "list", "NoneType").

        Note: There are no external callers outside torch._dynamo that rely on this.
        Internal uses should prefer this over hardcoded type name strings.
        """
        try:
            return self.python_type().__name__
        except NotImplementedError:
            return "<unknown type>"

    def python_qualified_name(self) -> str:
        """
        Equivalent to _PyType_GetFullyQualifiedName

        See https://github.com/python/cpython/blob/v3.15.0b4/Objects/typeobject.c#L1658
        """
        try:
            type_ = self.python_type()
        except NotImplementedError:
            return "<unknown type>"
        # Direct attribute access is safe here because type objects use the getset protocol, which will only return str
        # (and not execute user code)
        mod = type_.__module__
        qn = type_.__qualname__
        if mod not in ("__main__", "builtins"):
            return f"{mod}.{qn}"
        else:
            return qn

    def as_python_constant(self) -> Any:
        """For constants"""
        raise AsPythonConstantNotImplementedError(self)

    def guard_as_python_constant(self) -> Any:
        """Similar to as_python_constant(), but add ID_MATCH guards to try to force things to become constants"""
        try:
            return self.as_python_constant()
        except NotImplementedError:
            unimplemented(
                gb_type="Not a Python constant",
                context=f"guard_as_python_constant {self}",
                explanation=f"Failed to convert {self} into a Python constant.",
                hints=[],
            )

    def is_python_constant(self) -> bool:
        try:
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

    def nb_bool_impl(self, tx: InstructionTranslatorBase) -> VariableTracker | None:
        # Mirrors CPython's tp_as_number->nb_bool slot.
        # https://github.com/python/cpython/blob/c09ccd9c429/Objects/object.c#L2135-L2158
        #
        # Returns None when the type has no nb_bool, causing generic_is_true to
        # fall through to length check, then truthy default.
        return None

    def is_hashable(self) -> bool:
        """Whether the underlying Python object is hashable.

        Defaults to True — in CPython, all objects are hashable unless the type
        explicitly sets __hash__ = None (PyObject_HashNotImplemented).
        https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L7288

        Subclasses override to return False for unhashable types (list, dict,
        set, etc.).  This is a fast, side-effect-free check used by
        HashableTracker and __contains__ before attempting to hash.
        """
        return True

    def hash_impl(self, tx: InstructionTranslatorBase) -> tuple[int, bool]:
        """Default tp_hash: object.__hash__ = PyObject_GenericHash (identity hash).

        PyObject_GenericHash: https://github.com/python/cpython/blob/e76aa128fe/Python/pyhash.c#L143-L147
        Assigned on object: https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L7288

        Returns (hash_value, is_fake).  is_fake=True means the hash depends on
        a sourceless object's identity and must not escape into output bytecode.

        VT subclasses override this for types with their own C tp_hash.
        UserDefinedObjectVariable.hash_impl handles the MRO walk for custom
        Python __hash__, builtin-inherited C hash, and unknown C tp_hash.
        """
        # object.__hash__ = Py_HashPointer(obj) — identity hash.
        if self.source:
            real_val = tx.output.resolve_source_value(self.source)
            if type(real_val).__hash__ is not object.__hash__:
                unimplemented(
                    gb_type="Missing hash_impl override",
                    context=f"hash_impl {self}",
                    explanation=f"{type(self).__name__} wraps "
                    f"{type(real_val).__name__} which has a non-default "
                    f"tp_hash. Add a hash_impl override to "
                    f"{type(self).__name__}.",
                    hints=[*graph_break_hints.DYNAMO_BUG],
                )
            # object.__hash__ = id(obj), so the hash is identity-based.
            # Return is_fake=True so the hash is only usable internally
            # (dict keys, set membership) and doesn't escape into bytecode
            # — avoids installing an ID_MATCH guard just for hashing.
            # TODO when LazyConstants are fully landed, we can use them here instead.
            return hash(real_val), True
        # Sourceless: no real object to hash — fake id.
        return id(self), True

    def tp_richcompare_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """Per-VT tp_richcompare slot. Subclasses must override.

        Analogous to CPython's tp_richcompare function pointer on PyTypeObject.
        Returns ConstantVariable(NotImplemented) when the type does not handle
        the comparison (signaling do_richcompare to try the other operand).

        Called from two paths:
        - call_method("__eq__") calls tp_richcompare_impl directly (like CPython's
          a.__eq__(b) calling tp_richcompare without do_richcompare).
        - generic_richcompare calls tp_richcompare_impl as part of the 4-step
          do_richcompare algorithm (subclass priority, forward, reflected,
          fallback).
        """
        unimplemented(
            gb_type="Missing tp_richcompare_impl override",
            context=f"tp_richcompare_impl {self} {op}",
            explanation=f"{type(self).__name__} does not implement "
            f"tp_richcompare_impl. Add a tp_richcompare_impl override to "
            f"{type(self).__name__}.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    def is_constant_match(self, *values: Any) -> bool:
        """
        Check if this variable is a python constant matching one of the given values.

        Examples:
            var.is_constant_match(None)  # True if var is constant None
            var.is_constant_match(True, False)  # True if var is constant True or False
            var.is_constant_match(NotImplemented)  # True if var is constant NotImplemented
        """
        return False

    def is_constant_none(self) -> bool:
        """Check if this variable is a constant None value."""
        return False

    def make_guard(self, fn: Callable[..., Any]) -> Guard:
        if self.source:
            return self.source.make_guard(fn)
        raise NotImplementedError

    def const_getattr(self, tx: InstructionTranslatorBase, name: str) -> Any:
        """getattr(self, name) returning a python constant"""
        raise NotImplementedError

    def is_symnode_like(self) -> bool:
        """Return True for values that can participate in SymNode operations"""
        return False

    def is_tensor(self) -> bool:
        """Return True for TensorVariable instances"""
        return False

    def get_handler_type_for_dispatch(self) -> type[VariableTracker]:
        """Return the VariableTracker type to use for builtin handler dispatch.

        This is used by BuiltinVariable to look up the appropriate handler for
        a given set of arguments. Most VariableTrackers just return their own
        type, but LazyConstantVariable returns ConstantVariable since it can
        be treated as a constant for most builtin operations.

        Subclasses that override this should install appropriate guards to
        ensure the dispatched handler remains valid.
        """
        return type(self)

    def get_value_for_setattr(self) -> object | None:
        """Return the wrapped Python object for generic STORE_ATTR mutation,
        or None to decline.  Only override for VTs with __dict__ and
        standard __setattr__."""
        return None

    def lookup_instance_dict(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker | None:
        """Look up *name* in the instance __dict__ (tp_dictoffset equivalent).

        Returns a VT if the attribute exists in the instance dict, None if
        the attribute is absent or the VT has no instance dict at all.
        UDOV overrides to check self.value.__dict__ + side effects.
        """
        return None

    def call_getattr_fallback(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker | None:
        """Call __getattr__ fallback (step 6 of GenericGetAttr).

        Returns a VT if the type defines __getattr__ and it succeeds,
        None otherwise.  The base returns None (most types have no
        __getattr__).  UDOV overrides to walk the MRO for __getattr__.
        """
        return None

    def tp_getattro_impl(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker:
        """Default attribute access via object_generic_getattr.

        Tries the MRO-based descriptor protocol first (PyObject_GenericGetAttr),
        falls back to const_getattr for VTs without a known python_type or
        when the descriptor protocol hits an unhandled type.
        """
        # tp_getset/tp_members are data descriptors: resolve ahead of the
        # object-protocol walk. A getter returning None declines.
        getset = self.lookup_tp_getset_member(name)
        if getset is not None:
            result = getset.getter(self, tx)
            if result is not None:
                return result

        # object.__class__: one shared getset on `object` rather than per-VT.
        if name == "__class__":
            return VariableTracker.build(tx, self.python_type())

        try:
            py_type = self.python_type()
        except NotImplementedError:
            py_type = None

        if py_type is not None:
            from .object_protocol import (
                _UnhandledDescriptorError,
                object_generic_getattr,
            )

            try:
                return object_generic_getattr(tx, self, name)
            except _UnhandledDescriptorError:
                pass

        value = self.const_getattr(tx, name)
        if not variables.ConstantVariable.is_literal(value):
            raise NotImplementedError
        source = self.source and AttrSource(self.source, name)
        if source and not self.is_python_constant():
            # Skip guards on const getattr objects like __code__.co_argcount
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
        return variables.ConstantVariable.create(value, source=source)

    def get_dict_vt(
        self, tx: InstructionTranslatorBase
    ) -> variables.DunderDictVariable:
        # Callers gate this on the object actually having an instance __dict__
        # (e.g. the per-VT `__dict__` attribute branches). If a VT without a
        # real instance dict reaches here, DunderDictVariable's example-value
        # lookup graph-breaks rather than fabricating a bogus dict.
        if self.dict_vt is None:
            self.dict_vt = variables.DunderDictVariable.create(tx, self)
        return self.dict_vt

    def is_proxy(self) -> bool:
        try:
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    def as_proxy(self) -> Any:
        raise NotImplementedError(str(self))

    def maybe_fx_node(self) -> Node | None:
        try:
            proxy = self.as_proxy()
            import torch.fx

            if isinstance(proxy, torch.fx.Proxy):
                return proxy.node
            return None
        except NotImplementedError:
            return None

    def _contains_self_reference(self) -> bool:
        """Check if this variable references itself (directly or indirectly)."""
        found_self = False

        def check(vt: VariableTracker) -> None:
            nonlocal found_self
            if vt is self:
                found_self = True

        # unwrap first iteration - otherwise we can't detect if we revisit self
        for key, subvalue in self.__dict__.items():
            if key not in self._nonvar_fields:
                VariableTracker.visit(check, subvalue)

        return found_self

    def reconstruct(self, codegen: PyCodegen) -> None:
        raise NotImplementedError

    def reconstruct_pycode(self, codegen: PyCodegen) -> str:
        raise NotImplementedError(f"reconstruct_pycode not implemented for {self}")

    def unpack_var_sequence(
        self, tx: InstructionTranslatorBase
    ) -> list[VariableTracker]:
        raise NotImplementedError

    def _hasattr_check_side_effects(
        self, tx: InstructionTranslatorBase, name: str
    ) -> ConstantVariable | None:
        """If *name* has a pending mutation, return the hasattr result; else None."""
        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            value = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
            return variables.ConstantVariable.create(
                not isinstance(value, variables.DeletedVariable)
            )
        return None

    def call_obj_hasattr(
        self, tx: InstructionTranslatorBase, name: str
    ) -> ConstantVariable:
        """Dynamo's hasattr(): try tp_getattro_impl, catch AttributeError.

        Mirrors CPython's PyObject_HasAttr (via PyObject_GetOptionalAttr):
        call tp_getattro, suppress AttributeError via PyErr_Clear, return
        True/False.
        https://github.com/python/cpython/blob/848cb25624ab44c9fef2966c777419376b65af1b/Objects/object.c#L1346
        """
        result = self._hasattr_check_side_effects(tx, name)
        if result is not None:
            return result

        try:
            self.tp_getattro_impl(tx, name)
            return variables.ConstantVariable.create(True)
        except ObservedAttributeError:
            tx.exn_vt_stack.clear_current_exception()
            return variables.ConstantVariable.create(False)
        except (NotImplementedError, Unsupported):
            pass

        unimplemented(
            gb_type="Unsupported hasattr call",
            context=f"call_obj_hasattr {self} {name}",
            explanation=f"Dynamo does not know how to trace the function `{self.debug_repr()}`",
            hints=[
                f"Avoid calling `hasattr({self.__class__.__name__}, {name})` in your code.",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def tp_iter_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """
        Implements PyObject_GetIter semantics (tp_iter slot).
        Subclasses override this to support iteration.
        """
        # PyObject_GetIter: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L2847-L2870
        # TODO: raise TypeError instead - Make sure the jit tests for ScriptDict/ScriptList works with
        # this change
        unimplemented(
            gb_type="missing tp_iter",
            context=f"tp_iter_impl not implemented for {self.python_type_name()}",
            explanation=f"Dynamo does not know how to iterate over `{self.debug_repr()}`.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def tp_init_impl(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """tp_init slot (__init__). VTs override to initialize instances."""
        unimplemented(
            gb_type="missing tp_init",
            context=f"tp_init_impl not implemented for {self.python_type_name()}",
            explanation=f"Dynamo does not know how to trace __init__ on `{self.debug_repr()}`.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    def call_function(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Reaching this base fallback means no subclass handled the call. If the
        # object's type has no tp_call slot, it is genuinely not callable, so
        # mirror CPython's PyObject_Call TypeError. Otherwise the type is
        # callable in principle but unsupported by Dynamo -> graph break.
        from .object_protocol import pycallable_check

        try:
            obj_type = self.python_type()
        except NotImplementedError:
            obj_type = None
        if obj_type is not None and not pycallable_check(obj_type):
            raise_type_error(tx, f"'{self.python_type_name()}' object is not callable")
        unimplemented(
            gb_type="Unsupported function call",
            context=f"call_function {self} {args} {kwargs}",
            explanation=f"Dynamo does not know how to trace the function `{self.debug_repr()}`",
            hints=[
                f"Avoid calling `{self.debug_repr()}` in your code.",
                "Please report an issue to PyTorch.",
            ],
        )

    def sq_length_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """Called when sq_length is not implemented."""
        raise_observed_exception(
            TypeError,
            tx,
            args=[f"object of type '{self.python_type_name()}' has no len()"],
        )

    def mp_length_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """Called when mp_length is not implemented."""
        raise_observed_exception(
            TypeError,
            tx,
            args=[f"object of type '{self.python_type_name()}' has no len()"],
        )

    def mp_subscript_impl(
        self,
        tx: InstructionTranslatorBase,
        key: VariableTracker,
    ) -> VariableTracker:
        # PyObject_GetItem: https://github.com/python/cpython/blob/62a6e898e01/Objects/abstract.c#L155-L206
        # generic_getitem handles dispatch and raises TypeError for non-subscriptable
        # objects.  This base fallback fires for types with mp_subscript at the
        # C level but no Dynamo override yet.
        unimplemented(
            gb_type="missing_mp_subscript",
            context=f"mp_subscript_impl not defined for {type(self).__name__}",
            explanation=f"Dynamo does not yet support subscripting '{self.python_type_name()}'.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def sq_item_impl(
        self,
        tx: InstructionTranslatorBase,
        key: VariableTracker,
    ) -> VariableTracker:
        # PyObject_GetItem Branch 2: tp_as_sequence->sq_item
        # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L168-L181
        # Key has already been converted to int via nb_index_impl by generic_getitem.
        unimplemented(
            gb_type="unsupported __getitem__ (sq_item)",
            context=f"sq_item_impl {self} {key}",
            explanation=f"Dynamo does not know how to handle sq_item on {self}",
            hints=[],
        )

    def sq_contains_impl(
        self, tx: InstructionTranslatorBase, item: VariableTracker
    ) -> VariableTracker:
        """Called when sq_contains is not implemented."""
        unimplemented(
            gb_type="missing sq_contains",
            context=f"sq_contains_impl not implemented for {self.python_type_name()}",
            explanation=f"Dynamo does not know how to check if `{item.debug_repr()}` is in `{self.debug_repr()}`.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def mp_ass_subscript_impl(
        self,
        tx: InstructionTranslatorBase,
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        unimplemented(
            gb_type="missing_mp_ass_subscript",
            context=f"mp_ass_subscript_impl not defined for {self.python_type_name()}",
            explanation=f"Dynamo does not yet support item assignment on '{self.python_type_name()}'.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def sq_ass_item_impl(
        self,
        tx: InstructionTranslatorBase,
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        unimplemented(
            gb_type="unsupported __setitem__ (sq_ass_item)",
            context=f"sq_ass_item_impl {self} {key} {value}",
            explanation=f"Dynamo does not know how to handle sq_ass_item on {self}",
            hints=[],
        )

    def sq_concat_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """Sequence concatenation via + operator (sq_concat slot)."""
        unimplemented(
            gb_type="missing sq_concat",
            context=f"sq_concat not defined for {type(self).__name__}",
            explanation=f"Dynamo does not yet support concatenating '{self.python_type_name()}' and '{other.python_type_name()}'",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def sq_inplace_concat_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """In-place sequence concatenation via += operator (sq_inplace_concat slot)."""
        unimplemented(
            gb_type="missing sq_inplace_concat",
            context=f"sq_inplace_concat not defined for {type(self).__name__}",
            explanation=f"Dynamo does not yet support inplace concat of '{self.python_type_name()}' and '{other.python_type_name()}'",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def call_method(
        self,
        tx: InstructionTranslatorBase,
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Declarative named-method dispatch (tp_methods). A handler returning
        # None declines the call and falls through to the object-protocol
        # (tp_slot) dispatch below, mirroring the old super().call_method path.
        method = self.lookup_tp_method(name)
        if method is not None:
            result = method(self, tx, name, args, kwargs)
            if result is not None:
                return result

        slotdef = self._slotdefs.get(name)
        if slotdef is not None:
            return slotdef(self, tx, args, kwargs)

        if name == "__hash__" and not args and not kwargs:
            from .object_protocol import generic_hash

            return generic_hash(tx, self)

        if name == "__subclasscheck__" and len(args) == 1 and not kwargs:
            if (self_py := self.as_python_constant()) and (
                derived_py := args[0].as_python_constant()
            ):
                if (
                    inspect.getattr_static(self_py, "__subclasscheck__", None)
                    is type.__subclasscheck__
                ):
                    return variables.ConstantVariable.create(
                        issubclass(derived_py, self_py)
                    )
        # __reduce_ex__ is a C builtin (object.__reduce_ex__) that Dynamo
        # cannot trace into.  Constant-fold it for VTs backed by a real
        # Python object so that copy.deepcopy can trace through.
        if (
            name == "__reduce_ex__"
            and len(args) == 1
            and not kwargs
            and self.is_python_constant()
        ):
            protocol = args[0].as_python_constant()
            return VariableTracker.build(
                tx, self.as_python_constant().__reduce_ex__(protocol)
            )

        hints = [
            f"Avoid calling `{self.python_type_name()}.{name}` in your code.",
            "Please report an issue to PyTorch.",
        ]
        # additional hint for method calls on improperly constructed iterators
        if isinstance(self, variables.UserDefinedObjectVariable) and name in (
            "__iter__",
            "__next__",
        ):
            if isinstance(self.value, (KeysView, ItemsView, ValuesView)):
                hints.append(
                    "Consider moving the creation of dict view object (e.g. `dict.keys()`, `dict.items()`,) "
                    "to the compiled region, instead of passing it as an input to the compiled region."
                )
            hints.append(
                "Dynamo does not fully support tracing builtin iterators (e.g. `map`, `zip`, `enumerate`) "
                "passed in from uncompiled to compiled regions (e.g. `torch.compile(fn)(enumerate(...))`). "
                "This can happen unintentionally if a previous graph break happens with a builtin iterator "
                "in the local scope."
            )
            hints.append(
                "List/dict comprehensions in Python <= 3.11 result in implicit function calls, which Dynamo "
                "cannot trace as a top level frame. Possible workarounds are (1) use a loop instead of a comprehension, "
                "(2) fix any graph breaks in the function above the comprehension, (3) wrap the comprehension in a "
                "function, or (4) use Python 3.12+."
            )
        unimplemented(
            gb_type="Unsupported method call",
            context=f"call_method {self} {name} {args} {kwargs}",
            explanation=f"Dynamo does not know how to trace method `{name}` of class `{self.python_type_name()}`",
            hints=hints,
        )

    def call_tree_map(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """Performance optimization to implement optree.tree_map faster than tracing it"""
        is_leaf_var = tree_map_kwargs.get("is_leaf")
        if is_leaf_var is not None and not is_leaf_var.is_constant_none():
            pred_result = is_leaf_var.call_function(tx, [self], {})
            try:
                leaf_decision = pred_result.as_python_constant()
            except NotImplementedError:
                return self._tree_map_fallback(
                    tx,
                    tree_map_fn,
                    map_fn,
                    rest,
                    tree_map_kwargs,
                )
            if leaf_decision:
                return map_fn.call_function(tx, [self, *rest], {})

        return self.call_tree_map_branch(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
        )

    def call_tree_map_branch(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """Emulate optree.tree_map without is_leaf/none_is_leaf checks (handled above)"""
        return self._tree_map_fallback(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
        )

    def _tree_map_fallback(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        tree_map_fn_copy = tree_map_fn.clone()
        tree_map_fn_copy._maybe_call_tree_map_fastpath = lambda *args, **kwargs: None  # type: ignore[missing-attribute]
        log.debug(
            "tree_map fastpath fallback triggered for %s (rest=%s, kwargs=%s)",
            self,
            rest,
            tree_map_kwargs,
        )
        return tree_map_fn_copy.call_function(
            tx,
            [map_fn, self, *rest],
            tree_map_kwargs,
        )

    def call_tree_map_with_path(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        """Performance optimization to implement tree_map_with_path faster than tracing it"""
        is_leaf_var = tree_map_kwargs.get("is_leaf")
        if is_leaf_var is not None and not is_leaf_var.is_constant_none():
            pred_result = is_leaf_var.call_function(tx, [self], {})
            try:
                leaf_decision = pred_result.as_python_constant()
            except NotImplementedError:
                return self._tree_map_with_path_fallback(
                    tx,
                    tree_map_fn,
                    map_fn,
                    rest,
                    tree_map_kwargs,
                    keypath,
                )
            if leaf_decision:
                keypath_var = variables.TupleVariable(
                    [VariableTracker.build(tx, k) for k in keypath]
                )
                return map_fn.call_function(tx, [keypath_var, self, *rest], {})

        return self.call_tree_map_with_path_branch(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
            keypath,
        )

    def call_tree_map_with_path_branch(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        """Handle tree_map_with_path for leaf nodes (default behavior)"""
        keypath_var = variables.TupleVariable(
            [VariableTracker.build(tx, k) for k in keypath]
        )
        return map_fn.call_function(tx, [keypath_var, self, *rest], {})

    def _tree_map_with_path_fallback(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        tree_map_fn_copy = tree_map_fn.clone()
        tree_map_fn_copy._maybe_call_tree_map_fastpath = lambda *args, **kwargs: None  # type: ignore[missing-attribute]
        log.debug(
            "tree_map_with_path fastpath fallback triggered for %s (rest=%s, kwargs=%s, keypath=%s)",
            self,
            rest,
            tree_map_kwargs,
            keypath,
        )
        return tree_map_fn_copy.call_function(
            tx,
            [map_fn, self, *rest],
            tree_map_kwargs,
        )

    def set_name_hint(self, name: str) -> None:
        pass

    def set_source_location(self, source_location: SourceLocation) -> None:
        self.source_location = source_location

    def realize(self) -> VariableTracker:
        """Used by LazyVariableTracker to build the real VariableTracker"""
        return self

    def unwrap(self) -> VariableTracker:
        """Used by LazyVariableTracker to return the real VariableTracker if it already exists"""
        return self

    def is_realized(self) -> bool:
        """Used by LazyVariableTracker to indicate an unrealized node"""
        return True

    def tp_iternext_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """
        Implements tp_iternext slot semantics.
        Subclasses override this to support next(). Reaching this base is a
        bug — it means tp_iternext is missing for that VariableTracker subclass.
        """
        unimplemented(
            gb_type="Missing tp_iternext",
            context=f"next({self.python_type_name()})",
            explanation=(
                f"Dynamo does not support next() on {self.python_type_name()}."
                " Add tp_iternext_impl to this VariableTracker subclass."
            ),
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def next_variable(self, tx: InstructionTranslatorBase) -> VariableTracker:
        return self.tp_iternext_impl(tx)

    def is_strict_mode(self, tx: InstructionTranslatorBase) -> bool:
        return bool(tx.strict_checks_fn and tx.strict_checks_fn(self))

    def is_mutable(self) -> bool:
        """Whether Dynamo allows mutation on this variable."""
        return not self.is_immutable()

    def is_immutable(self) -> bool:
        """Whether Dynamo bans mutation on this variable."""
        return self.mutation_type is None

    @staticmethod
    def build(
        tx: InstructionTranslatorBase,
        value: Any,
        source: Source | None = None,
        realize: bool = False,
    ) -> Any:
        """Create a new VariableTracker from a value and optional Source"""
        if source is None:
            return builder.SourcelessBuilder.create(tx, value)
        elif realize:
            return builder.VariableBuilder(tx, source)(value)
        elif type(value) in variables.LazyConstantVariable.supported_types:
            # Use LazyConstantVariable for primitives to enable deferred
            # guard installation - constants that are just passed through
            # won't cause recompilation when their values change.
            return variables.LazyConstantVariable.create(value, source)
        else:
            return variables.LazyVariableTracker.create(value, source, tx=tx)

    def get_id(self, tx: InstructionTranslatorBase) -> int | None:
        """Return id() of the underlying Python object, or None if unavailable.

        The base implementation uses source resolution for sourceful VTs.
        Subclasses override for special cases (e.g. NNModuleVariable uses
        get_submodule, ConstantVariable handles singletons).

        Returns None for sourceless VTs — callers use FakeIdVariable in
        that case (see generic_id in object_protocol.py).
        """
        if self.source:
            return id(tx.output.resolve_source_value(self.source))
        return None

    def get_id_guard_type(self) -> Callable[..., Any] | None:
        if self.source:
            return GuardBuilder.ID_MATCH
        return None

    def get_real_python_backed_value(self) -> object:
        """Return the Python object this VT wraps, for `is` comparison.

        Returns NO_SUCH_SUBOBJ if no concrete Python object is available.
        """

        try:
            return self.as_python_constant()
        except NotImplementedError:
            return NO_SUCH_SUBOBJ

    def nb_index_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Mirrors CPython's PyNumber_Index / nb_index slot.

        https://github.com/python/cpython/blob/c09ccd9c429/Objects/abstract.c#L1411-L1450

        The base implementation raises TypeError, matching CPython's behavior
        when tp_as_number->nb_index is NULL (_PyIndex_Check fails).
        """
        raise_observed_exception(
            TypeError,
            tx,
            args=[
                f"'{self.python_type_name()}' object cannot be interpreted as an integer"
            ],
        )

    def tp_repr_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Mirrors CPython's tp_repr slot.

        https://github.com/python/cpython/blob/v3.13.3/Objects/object.c#L745-L778

        Called when type_implements_tp_repr returns True for this type.
        Subclasses override to provide the actual repr implementation.
        """
        unimplemented(
            gb_type="tp_repr_impl not implemented",
            context=f"{type(self).__name__} has tp_repr slot but no tp_repr_impl override",
            explanation=f"The type {self.python_type_name()} has a tp_repr C slot but "
            "the corresponding VariableTracker doesn't implement tp_repr_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def tp_str_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Dynamo hook for VariableTrackers with dedicated str behavior.
        Subclasses override this for the per-type str result; generic_str()
        handles dispatch and repr fallback.
        """
        unimplemented(
            gb_type="tp_str_impl not implemented",
            context=f"{type(self).__name__} has no tp_str_impl override for {self.python_type_name()}",
            explanation=f"Dynamo does not implement __str__ for {self.python_type_name()} "
            f"in {type(self).__name__}.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_int_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Mirrors CPython's tp_as_number->nb_int slot.

        Called when type_implements_nb_int returns True for this type.
        Subclasses override to provide the actual conversion.
        """
        unimplemented(
            gb_type="nb_int_impl not implemented",
            context=f"{type(self).__name__} has nb_int slot but no nb_int_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_int C slot but "
            "the corresponding VariableTracker doesn't implement nb_int_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_float_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Mirrors CPython's tp_as_number->nb_float slot.

        Called when type_implements_nb_float returns True for this type.
        Subclasses override to provide the actual conversion.
        """
        unimplemented(
            gb_type="nb_float_impl not implemented",
            context=f"{type(self).__name__} has nb_float slot but no nb_float_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_float C slot but "
            "the corresponding VariableTracker doesn't implement nb_float_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def _nb_slot_not_implemented(
        self,
        tp_slot_name: str,
        other: VariableTracker,
        reverse: bool = False,
    ):
        unimplemented(
            gb_type="tp_as_number slot not implemented",
            context=f"{tp_slot_name}({self.python_type_name()}, {other.python_type_name()}, reverse={reverse})",
            explanation=f"The type {self.python_type_name()} does not implement the {tp_slot_name}",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_lshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_lshift slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rlshift__`` instead of ``__lshift__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_lshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_lshift slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_rshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_rshift slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rrshift__`` instead of ``__rshift__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_rshift_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_rshift slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_and_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_and slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rand__`` instead of ``__and__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_and_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_and slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_xor_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_xor slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rxor__`` instead of ``__xor__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_xor_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_xor slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_floor_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_floor_divide slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rfloordiv__`` instead of ``__floordiv__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_floor_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_floor_divide slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_true_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_true_divide slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rtruediv__`` instead of ``__truediv__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_true_divide_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_true_divide slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_remainder_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_remainder slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rmod__`` instead of ``__mod__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_remainder_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_remainder slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_divmod_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_divmod slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rdivmod__`` instead of ``__divmod__``). divmod has no
        in-place form.
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_power_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        z: VariableTracker | None,
        reverse: bool = False,
    ) -> VariableTracker:
        return variables.ConstantVariable(NotImplemented)

    def nb_power_z_impl(
        self,
        tx: InstructionTranslatorBase,
        v: VariableTracker,
        w: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_power slot when self is the modulus (z) operand.

        ``v`` is the base, ``w`` is the exponent, ``self`` is the modulus.
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_power_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        z: VariableTracker | None,
    ) -> VariableTracker:
        return variables.ConstantVariable(NotImplemented)

    def nb_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_multiply slot. Default: graph-breaks.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rmul__`` instead of ``__mul__``).  The base default
        graph-breaks so unmigrated VTs surface loudly; ``binary_op1`` only
        invokes this on types whose CPython type advertises the slot.
        """
        return self._nb_slot_not_implemented("nb_multiply_impl", other, reverse=reverse)

    def nb_inplace_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_multiply slot. Default: graph-breaks."""
        return self._nb_slot_not_implemented("nb_inplace_multiply_impl", other)

    def nb_matrix_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_matrix_multiply slot. Default: graph-breaks.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rmatmul__`` instead of ``__matmul__``).
        """
        return self._nb_slot_not_implemented(
            "nb_matrix_multiply_impl", other, reverse=reverse
        )

    def nb_inplace_matrix_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_matrix_multiply slot. Default: graph-breaks."""
        return self._nb_slot_not_implemented("nb_inplace_matrix_multiply_impl", other)

    def sq_repeat_impl(
        self,
        tx: InstructionTranslatorBase,
        count: VariableTracker,
    ) -> VariableTracker:
        """tp_as_sequence->sq_repeat slot.

        Implements sequence repetition (e.g. ``[1, 2] * 3``).  ``count`` is
        an int VariableTracker already produced by ``nb_index_impl``.
        """
        unimplemented(
            gb_type="sq_repeat_impl not implemented",
            context=f"{type(self).__name__} has sq_repeat slot but no sq_repeat_impl override",
            explanation=f"The type {self.python_type_name()} has an sq_repeat C slot but "
            "the corresponding VariableTracker doesn't implement sq_repeat_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def sq_inplace_repeat_impl(
        self,
        tx: InstructionTranslatorBase,
        count: VariableTracker,
    ) -> VariableTracker:
        """tp_as_sequence->sq_inplace_repeat slot."""
        unimplemented(
            gb_type="sq_inplace_repeat_impl not implemented",
            context=f"{type(self).__name__} has sq_inplace_repeat slot but no sq_inplace_repeat_impl override",
            explanation=f"The type {self.python_type_name()} has an sq_inplace_repeat C slot but "
            "the corresponding VariableTracker doesn't implement sq_inplace_repeat_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_or_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_or slot. Default: returns NotImplemented.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__ror__`` instead of ``__or__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_or_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_or slot. Default: returns NotImplemented."""
        return variables.ConstantVariable(NotImplemented)

    def nb_negative_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Mirrors CPython's tp_as_number->nb_negative slot.

        Called when type_implements_nb_negative returns True for this type.
        Subclasses override to provide the actual negation.
        """
        unimplemented(
            gb_type="nb_negative_impl not implemented",
            context=f"{type(self).__name__} has nb_negative slot but no nb_negative_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_negative C slot but "
            "the corresponding VariableTracker doesn't implement nb_negative_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_positive_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Mirrors CPython's tp_as_number->nb_positive slot.

        Called when type_implements_nb_positive returns True for this type.
        Subclasses override to provide the actual positive.
        """
        unimplemented(
            gb_type="nb_positive_impl not implemented",
            context=f"{type(self).__name__} has nb_positive slot but no nb_positive_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_positive C slot but "
            "the corresponding VariableTracker doesn't implement nb_positive_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_add_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_add slot. Default: graph break.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__radd__`` instead of ``__add__``).
        """
        # We need to return NotImplemented here because of sq_concat
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_add_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_add slot. Default: graph break."""
        # We need to return NotImplemented here because of sq_inplace_concat
        return variables.ConstantVariable(NotImplemented)

    def nb_subtract_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        """tp_as_number->nb_subtract slot. Default: graph break.

        ``reverse=True`` means self is the right-hand operand (CPython would
        look up ``__rsub__`` instead of ``__sub__``).
        """
        return variables.ConstantVariable(NotImplemented)

    def nb_inplace_subtract_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
    ) -> VariableTracker:
        """tp_as_number->nb_inplace_subtract slot. Default: graph break."""
        return variables.ConstantVariable(NotImplemented)

    def nb_absolute_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Mirrors CPython's tp_as_number->nb_absolute slot.

        Called when type_implements_nb_absolute returns True for this type.
        Subclasses override to provide the actual absolute value.
        """
        unimplemented(
            gb_type="nb_absolute_impl not implemented",
            context=f"{type(self).__name__} has nb_absolute slot but no nb_absolute_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_absolute C slot but "
            "the corresponding VariableTracker doesn't implement nb_absolute_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def nb_invert_impl(
        self,
        tx: InstructionTranslatorBase,
    ) -> VariableTracker:
        """Mirrors CPython's tp_as_number->nb_invert slot.

        Called when type_implements_nb_invert returns True for this type.
        Subclasses override to provide the actual bitwise inversion.
        """
        unimplemented(
            gb_type="nb_invert_impl not implemented",
            context=f"{type(self).__name__} has nb_invert slot but no nb_invert_impl override",
            explanation=f"The type {self.python_type_name()} has an nb_invert C slot but "
            "the corresponding VariableTracker doesn't implement nb_invert_impl.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def __init__(
        self,
        *,
        source: Source | None = None,
        mutation_type: MutationType | None = None,
        source_location: SourceLocation | None = None,
        dict_vt: variables.DunderDictVariable | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.source_location = source_location
        self.mutation_type = mutation_type
        # Carried so clone() round-trips the cached __dict__ view.
        self.dict_vt = dict_vt

        # NOTE sometimes mutation_type is set afterwards for implementation
        # convenience, we don't validate those cases at the moment.
        if mutation_type is not None:
            if isinstance(mutation_type, (ValueMutationNew, AttributeMutationNew)):
                # If this fails, it's either
                # 1. one mistakenly passed in a source
                # 2. `mutation_type` is incorrect
                if source is not None:
                    raise AssertionError(
                        "source must be None for ValueMutationNew/AttributeMutationNew"
                    )
            else:
                if not isinstance(
                    mutation_type, (ValueMutationExisting, AttributeMutationExisting)
                ):
                    raise AssertionError(
                        f"Expected ValueMutationExisting or AttributeMutationExisting, "
                        f"got {type(mutation_type)}"
                    )
                # If this fails, it's either
                # 1. one forgot to pass in a source
                # 2. `mutation_type` is incorrect
                if source is None:
                    raise AssertionError(
                        "source must not be None for ValueMutationExisting/AttributeMutationExisting"
                    )

    @property
    def tp_type(self) -> PyTypeObject:
        return _tp_type(maybe_get_python_type(self))

    @property
    def tp_as_number(self) -> PyNumberMethods:
        return _tp_type(maybe_get_python_type(self)).tp_as_number

    @property
    def tp_as_sequence(self) -> PySequenceMethods:
        return _tp_type(maybe_get_python_type(self)).tp_as_sequence

    @property
    def tp_as_mapping(self) -> PyMappingMethods:
        return _tp_type(maybe_get_python_type(self)).tp_as_mapping

    @property
    def tp_iter(self) -> Slot | None:
        return _tp_type(maybe_get_python_type(self)).tp_iter

    @property
    def tp_iternext(self) -> Slot | None:
        return _tp_type(maybe_get_python_type(self)).tp_iternext

    @property
    def tp_str(self) -> Slot | None:
        return _tp_type(maybe_get_python_type(self)).tp_str

    @property
    def tp_repr(self) -> Slot | None:
        return _tp_type(maybe_get_python_type(self)).tp_repr

    @property
    def _slotdefs(self):
        return _compute_slotdefs(maybe_get_python_type(self))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Wraps all subclasses' `as_python_constant` and `reconstruct` so that it cannot be
        called twice in the same call chain - i.e. self-referential objects.
        For `as_python_constant` - self-referential objects are NOT treated as constants.
        For `reconstruct` - we will graph break. The graph break can be avoided if the VT subclass
        can generate and cache itself before recursively `reconstruct`ing - see ListVariable for an example.
        """
        super().__init_subclass__(**kwargs)

        def as_python_constant_failure(self) -> NoReturn:
            raise AsPythonConstantNotImplementedError(
                self, msg=f"{self} is self-referential"
            )

        VariableTracker._add_call_once_guard(
            cls, "as_python_constant", as_python_constant_failure
        )

        def reconstruct_failure(self) -> NoReturn:
            unimplemented(
                gb_type="Reconstruction failure (self-referential)",
                context=str(self),
                explanation=f"Dynamo tried to reconstruct sourceless variable {self}, but it is self-referential. "
                "Dynamo must manually implement reconstruction rules for self-referentiable sourceless variables.",
                hints=[
                    "If Dynamo is attempting to trace a return statement and your code is attempting to return a variable "
                    "that Dynamo cannot reconstruct, then remove it from the return statement.",
                    "Remove the self-reference in the variable. A self-referring list, for example, is `l = []; l.append(l)`.",
                    *graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK,
                    "Report an issue to PyTorch if you need self-referential reconstrtuction support.",
                ],
            )

        VariableTracker._add_call_once_guard(cls, "reconstruct", reconstruct_failure)

    @staticmethod
    def _add_call_once_guard(
        cls: type[VariableTracker],
        method: str,
        callback: Callable[[VariableTracker], Any],
    ) -> None:
        original_method = getattr(cls, method)

        if original_method is getattr(VariableTracker, method) or hasattr(
            original_method, "_call_once_guarded"
        ):
            return

        @functools.wraps(original_method)
        def guarded_method(self, *args: Any, **kwargs: Any) -> VariableTracker:
            active = _vt_active_calls.get()
            if active is None:
                active = set()
                _vt_active_calls.set(active)

            key = (id(self), id(original_method))
            if key in active:
                callback(self)

            active.add(key)
            try:
                return original_method(self, *args, **kwargs)
            finally:
                active.discard(key)

        guarded_method._call_once_guarded = True  # pyrefly: ignore[missing-attribute]
        setattr(cls, method, guarded_method)


def typestr(*objs: object) -> str:
    if len(objs) == 1:
        (obj,) = objs
        if isinstance(obj, VariableTracker):
            return str(obj)
        else:
            return type(obj).__name__
    else:
        return " ".join(map(typestr, objs))


instancecheck = type.__instancecheck__


_CPYTHON_BASE_URL = "https://github.com/python/cpython/blob/v3.13.0/"


def print_cpython_to_vt_mapping() -> None:
    """Print the mapping from CPython types to Dynamo VariableTracker classes.

    Reads _cpython_type from each VT subclass (own attribute only, not inherited).
    """
    # Ensure all VT modules are imported so all_subclasses is complete
    from . import (  # noqa: F401
        builtin,
        constant,
        ctx_manager,
        dicts,
        distributed,
        functions,
        higher_order_ops,
        iter,
        lazy,
        lists,
        misc,
        nn_module,
        streams,
        tensor,
        torch,
        torch_function,
        user_defined,
    )

    mapping: dict[type, list[type]] = {}
    for vt_cls in VariableTrackerMeta.all_subclasses:
        cpython_type = vt_cls.__dict__.get("_cpython_type")
        if cpython_type is None:
            continue
        types = cpython_type if isinstance(cpython_type, tuple) else (cpython_type,)
        for t in types:
            mapping.setdefault(t, []).append(vt_cls)

    for py_type, vt_classes in sorted(mapping.items(), key=lambda x: x[0].__qualname__):
        for vt_cls in vt_classes:
            print(
                f"{py_type.__module__}.{py_type.__qualname__:30s} -> {vt_cls.__name__}"
            )


from . import builder
from .lazy import LazyVariableTracker
<|endoftext|>"""
This module provides iterator-related variable tracking functionality for Dynamo.
It implements variable classes for handling Python iterators and itertools functions
during symbolic execution and tracing.

The module includes:
- Base iterator variable classes for tracking iterator state
- Implementations of built-in iterators (zip, map, filter)
- Support for itertools functions (product, groupby, count, etc.)
- Mutation tracking and reconstruction capabilities for iterator operations

These classes integrate with Dynamo's variable tracking system to enable proper
handling of iterator operations during code transformation and optimization.
"""

import inspect
import itertools
import operator
import sys
from typing import Any, TYPE_CHECKING

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import (
    create_call_function,
    create_call_function_ex,
    create_instruction,
)
from ..exc import (
    handle_observed_exception,
    ObservedUserStopIteration,
    raise_observed_exception,
    raise_type_error,
    raise_value_error,
    unimplemented,
)
from ..utils import raise_args_mismatch, tracked_repr, unpack_iterable
from .base import GetSet, Method, ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .hashable import HashableTracker
from .object_protocol import generic_getiter, pyiter_next


# chain.from_iterable is a method descriptor that creates a new object on each
# attribute access (a is b → False). Capture once at import time for stable
# identity comparisons in ItertoolsVariable.call_function.
_CHAIN_FROM_ITERABLE = itertools.chain.from_iterable

# repeat(object, times=-1); itertools.repeat has no introspectable signature, so
# declare one to bind args/kwargs. `times` defaults to None to mark the unbounded
# form (CPython's cnt == -1).
_REPEAT_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter("object", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            "times", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        ),
    ]
)


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


MAX_ITERATOR_LIMIT = 100 * 1024  # 100k


def is_iterator_exhausted(
    tx: "InstructionTranslatorBase", iterator: VariableTracker
) -> bool:
    try:
        pyiter_next(tx, iterator)
        return False
    except ObservedUserStopIteration:
        handle_observed_exception(tx)
        return True


class ItertoolsVariable(VariableTracker):
    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def __repr__(self) -> str:
        return f"ItertoolsVariable({self.value})"

    def as_python_constant(self) -> Any:
        return self.value

    def get_real_python_backed_value(self) -> Any:
        return self.value

    def _get_from_iterable(
        self, tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        # Only itertools.chain has from_iterable; declining (None) falls
        # through to the generic protocol for other itertools callables.
        if self.value is itertools.chain:
            return ItertoolsVariable(_CHAIN_FROM_ITERABLE)
        return None

    tp_getset = {
        "from_iterable": GetSet(_get_from_iterable),
    }

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list["VariableTracker"],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # See also: module `torch._dynamo.polyfills.itertools`

        if self.value is itertools.chain and not kwargs:
            # Wrap args in a ListIteratorVariable so sub-iterables are pulled lazily.
            # generic_getiter on each sub-iterable is deferred to tp_iternext_impl,
            # matching CPython's behavior (iter() on each arg is lazy, not at construction).
            source = variables.ListIteratorVariable(
                list(args), mutation_type=ValueMutationNew()
            )
            return ChainVariable(source, mutation_type=ValueMutationNew())
        elif self.value is _CHAIN_FROM_ITERABLE and not kwargs and len(args) == 1:
            # Convert outer iterable to iterator; each sub-iterable converted lazily.
            source = generic_getiter(tx, args[0])
            return ChainVariable(source, mutation_type=ValueMutationNew())
        elif self.value is itertools.zip_longest:
            fillvalue_vt = kwargs.pop("fillvalue", ConstantVariable.create(None))
            if kwargs:
                unimplemented(
                    gb_type="Unsupported kwargs for itertools.zip_longest",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'fillvalue', but got "
                    f"{','.join(set(kwargs.keys()) - {'fillvalue'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )
            iterables = [generic_getiter(tx, arg) for arg in args]
            return ZipLongestVariable(
                iterables,
                fillvalue=fillvalue_vt,
                mutation_type=ValueMutationNew(),
            )
        elif self.value is itertools.product:
            if any(kw != "repeat" for kw in kwargs):
                unimplemented(
                    gb_type="Unsupported kwargs for itertools.product",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'repeat', but got "
                    f"{','.join(set(kwargs.keys()) - {'repeat'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            if "repeat" in kwargs:
                r = kwargs["repeat"].as_python_constant()
            else:
                r = 1
            seqs = [unpack_iterable(tx, arg) for arg in args]
            items = [
                variables.TupleVariable(list(item))
                for item in itertools.product(*seqs, repeat=r)
            ]
            return variables.ListIteratorVariable(
                items,  # type: ignore[arg-type]
                mutation_type=ValueMutationNew(),
            )
        elif self.value is itertools.groupby:
            if any(kw != "key" for kw in kwargs):
                unimplemented(
                    gb_type="Unsupported kwargs for itertools.groupby",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'key', but got "
                    f"{','.join(set(kwargs.keys()) - {'key'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            def retrieve_const_key(key: VariableTracker) -> Any:
                from ..utils import specialize_symnode

                # Unwrap LazyVariableTracker to get the underlying variable
                key = specialize_symnode(key)
                if isinstance(key, variables.SymNodeVariable):
                    return key.evaluate_expr()
                elif key.is_python_constant():
                    return key.as_python_constant()
                else:
                    unimplemented(
                        gb_type="Unsupported key type for itertools.groupby",
                        context=f"call_function {self} {args} {kwargs}",
                        explanation="Dynamo does not know how to trace "
                        f"itertools.groupby with key type: {str(type(key))}. "
                        "We only support grouping keys that are constants (int, float, str, etc.)",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )

            if len(args) != 1:
                unimplemented(
                    gb_type="Unsupported arguments for itertools.groupby",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation="Dynamo does not know how to trace "
                    f"itertools.groupby with args: {args} and kwargs: {kwargs}. "
                    "itertools.groupby expects an iterable to group and an "
                    "optional key function to determine groupings.",
                    hints=[
                        "Make sure the arguments to itertools.groupby are correct.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            seq = unpack_iterable(tx, args[0])

            if "key" in kwargs:

                def keyfunc(x: VariableTracker) -> Any:
                    return retrieve_const_key(
                        kwargs.get("key").call_function(tx, [x], {})  # type: ignore[union-attr]
                    )

            else:

                def keyfunc(x: VariableTracker) -> Any:
                    return retrieve_const_key(x)

            result = []
            try:
                for k, v in itertools.groupby(seq, key=keyfunc):
                    result.append(
                        variables.TupleVariable(
                            [
                                (
                                    variables.ConstantVariable.create(k)
                                    if variables.ConstantVariable.is_literal(k)
                                    else k
                                ),
                                variables.ListIteratorVariable(
                                    list(v), mutation_type=ValueMutationNew()
                                ),
                            ],
                            mutation_type=ValueMutationNew(),
                        )
                    )
            except Exception as e:
                unimplemented(
                    gb_type="Unexpected failure during itertools.groupby() iteration",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation="Unexpected failure in invoking function during groupby",
                    hints=[*graph_break_hints.SUPPORTABLE],
                    from_exc=e,
                )
            return variables.ListIteratorVariable(
                result,  # type: ignore[arg-type]
                mutation_type=ValueMutationNew(),
            )
        elif self.value is itertools.repeat:
            # ref: repeat_new in itertoolsmodule.c. A user-supplied negative
            # `times` is clamped to 0; an absent `times` means unbounded.
            try:
                bound = _REPEAT_SIGNATURE.bind(*args, **kwargs)
            except TypeError:
                raise_args_mismatch(tx, "repeat")
            item = bound.arguments["object"]
            times = bound.arguments.get("times")
            if times is None:
                return RepeatIteratorVariable(item, mutation_type=ValueMutationNew())
            if not times.is_python_constant():
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.repeat),
                    [item, times],
                    {},
                )
            times_val = times.as_python_constant()
            if not isinstance(times_val, int):
                raise_type_error(
                    tx,
                    f"'{times.python_type_name()}' object cannot be interpreted "
                    "as an integer",
                )
            return RepeatIteratorVariable(
                item, times=max(times_val, 0), mutation_type=ValueMutationNew()
            )
        elif self.value is itertools.count:
            # count(start=0, step=1): let Python's own argument binding validate
            # the call. Anything it rejects (extra args, duplicate/unknown
            # kwargs) falls through to a graph break so eager raises the
            # CPython TypeError.
            def count_sig(start: Any = 0, step: Any = 1) -> tuple[Any, Any]:
                return start, step

            try:
                item, step = count_sig(*args, **kwargs)
            except TypeError:
                return super().call_function(tx, args, kwargs)
            return variables.CountIteratorVariable(
                item, step, mutation_type=ValueMutationNew()
            )
        else:
            return super().call_function(tx, args, kwargs)


class IteratorVariable(VariableTracker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        unimplemented(
            gb_type="Unimplemented next() call",
            context=f"next({self})",
            explanation="This abstract method must be implemented",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "ConstantVariable":
        if name == "__iter__" or name == "__next__":
            return variables.ConstantVariable.create(True)
        return super().call_obj_hasattr(tx, name)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        """Iterators are their own iterator."""
        return self


class ChainVariable(IteratorVariable):
    """
    Represents itertools.chain(*iterables) — yields one item per tp_iternext_impl call.

    Uses a source_iterator that yields raw sub-iterables (not yet converted to
    iterators). Conversion happens lazily in tp_iternext_impl so that non-iterable
    args raise TypeError at iteration time, matching CPython's behavior.
    """

    # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L225-L310
    _cpython_type = itertools.chain

    def __init__(
        self,
        source_iterator: "VariableTracker",
        current: "VariableTracker | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # source_iterator yields raw sub-iterables; converted lazily per tp_iternext_impl
        self.source_iterator = source_iterator
        # current sub-iterator (None = not started or just exhausted a sub-iterable)
        self.current = current

    def python_type(self) -> type:
        return itertools.chain

    def _get_from_iterable(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        return ItertoolsVariable(_CHAIN_FROM_ITERABLE)

    tp_getset = {
        "from_iterable": GetSet(_get_from_iterable),
    }

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        if not self.is_mutable():
            raise AssertionError("ChainVariable must be mutable for next()")
        while True:
            if self.current is None:
                # Pull next sub-iterable from source (source is always an iterator)
                try:
                    next_raw = pyiter_next(tx, self.source_iterator)
                except ObservedUserStopIteration:
                    handle_observed_exception(tx)
                    raise_observed_exception(StopIteration, tx)
                # Convert sub-iterable to iterator lazily — may raise TypeError
                it = generic_getiter(tx, next_raw)
                tx.output.side_effects.mutation(self)
                self.current = it
            try:
                return pyiter_next(tx, self.current)
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                tx.output.side_effects.mutation(self)
                self.current = None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Reconstruct as itertools.chain(current, itertools.chain.from_iterable(source))
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("chain"),
                ]
            )
        )
        if self.current is not None:
            codegen(self.current)
        # chain.from_iterable(source_iterator) for remaining sub-iterables
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("chain"),
                    codegen.create_load_attr("from_iterable"),
                ]
            )
        )
        codegen(self.source_iterator)
        codegen.extend_output(create_call_function(1, False))
        n_args = (1 if self.current is not None else 0) + 1
        codegen.extend_output(create_call_function(n_args, False))


class RepeatIteratorVariable(IteratorVariable):
    # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c
    # repeat carries a `cnt` count: -1 means unbounded; CPython's repeat_new
    # clamps a user-supplied negative `times` to 0. We model that with `times`
    # (None == unbounded) and a `remaining` counter consumed by next().
    _nonvar_fields = {
        "times",
        "remaining",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        item: VariableTracker,
        times: int | None = None,
        remaining: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.item = item
        self.times = times
        self.remaining = times if remaining is None else remaining

    def python_type(self) -> type:
        return itertools.repeat

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c repeat_next
        if self.remaining is None:
            return self.item
        if not self.is_mutable():
            raise AssertionError("bounded RepeatIteratorVariable must be mutable")
        if self.remaining == 0:
            raise_observed_exception(StopIteration, tx)
        tx.output.side_effects.mutation(self)
        self.remaining -= 1
        return self.item

    def repeat_length_hint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # ref: repeat_len in itertoolsmodule.c (exposed as __length_hint__);
        # raises TypeError for the unbounded form ("len() of unsized object").
        # Not a C-level slot, so it lives in tp_methods rather than call_method.
        if self.times is None:
            raise_type_error(tx, "len() of unsized object")
        return ConstantVariable.create(self.remaining)

    tp_methods = {
        "__length_hint__": Method(repeat_length_hint),
    }

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        item_repr = tracked_repr(tx, self.item)
        if self.times is None:
            return ConstantVariable.create(f"repeat({item_repr})")
        return ConstantVariable.create(f"repeat({item_repr}, {self.remaining})")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("repeat"),
                ]
            )
        )
        codegen(self.item)
        if self.times is None:
            codegen.extend_output(create_call_function(1, False))
        else:
            codegen(ConstantVariable.create(self.remaining))
            codegen.extend_output(create_call_function(2, False))


class CountIteratorVariable(IteratorVariable):
    # advance_count tracks how many next() calls were made during tracing,
    # used by side_effects.py to replay them on the real iterator post-execution.
    _nonvar_fields = {
        "advance_count",
        *IteratorVariable._nonvar_fields,
    }

    def python_type(self) -> type:
        return itertools.count

    def __init__(
        self,
        item: int | VariableTracker = 0,
        step: int | VariableTracker = 1,
        advance_count: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(item, VariableTracker):
            item = ConstantVariable.create(item)
        if not isinstance(step, VariableTracker):
            step = ConstantVariable.create(step)
        self.item = item
        self.step = step
        self.advance_count = advance_count

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L4189-L4216
        if not self.is_mutable():
            raise AssertionError("CountIteratorVariable must be mutable for next()")
        old_item = self.item
        tx.output.side_effects.mutation(self)
        self.item = self.item.call_method(tx, "__add__", [self.step], {})
        self.advance_count += 1
        return old_item

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L4218-L4243
        if not (self.item.is_python_constant() and self.step.is_python_constant()):
            return super().tp_repr_impl(tx)
        cnt = self.item.as_python_constant()
        step = self.step.as_python_constant()
        # Suppress step in the repr when it is an integer equal to 1.
        if isinstance(step, int) and step == 1:
            result = f"count({cnt!r})"
        else:
            result = f"count({cnt!r}, {step!r})"
        return ConstantVariable.create(result)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("count"),
                ]
            )
        )
        codegen(self.item)
        codegen(self.step)
        codegen.extend_output(create_call_function(2, False))


class ZipVariable(IteratorVariable):
    """
    Represents zip(*iterables)
    """

    # PyZip_Type: https://github.com/python/cpython/blob/v3.13.0/Python/bltinmodule.c#L3011
    _cpython_type = zip

    _nonvar_fields = {
        "strict",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        iterable: VariableTracker,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(iterable, variables.TupleVariable):
            raise AssertionError(f"Expected a tuple of iterables, got {type(iterable)}")
        # can be list[Variable] or VariableTracker (with next_variable implemented)
        self.iterable = iterable
        self.strict = strict

    def python_type(self) -> type[zip]:  # type: ignore[type-arg]
        return zip

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L2906-L2994
        if not self.is_mutable():
            raise AssertionError("ZipVariable must be mutable for next()")
        tuplesize = len(self.iterable.items)

        if tuplesize == 0:
            raise_observed_exception(StopIteration, tx)

        items = []
        for i in range(tuplesize):
            it = self.iterable.items[i]
            try:
                items.append(pyiter_next(tx, it))
            except ObservedUserStopIteration:
                if not self.strict:
                    raise

                if i > 0:
                    raise_value_error(
                        tx, f"zip() argument {i} shorter than previous arguments"
                    )

                # In strict mode, if any iterable is exhausted, all must be exhausted
                for j in range(i + 1, tuplesize):
                    it_j = self.iterable.items[j]
                    if is_iterator_exhausted(tx, it_j):
                        continue
                    break
                else:
                    # all iterables exhausted, raise StopIteration
                    raise

                handle_observed_exception(tx)  # StopIteration
                raise_value_error(
                    tx, f"zip() argument {i} is longer than previous arguments"
                )

        return variables.TupleVariable(items)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "zip"), call_function_ex=True
        )
        codegen(self.iterable)
        codegen.extend_output(
            [
                codegen.create_load_const("strict"),
                codegen.create_load_const(self.strict),
                create_instruction("BUILD_MAP", arg=1),
                *create_call_function_ex(True, False),
            ]
        )


class ZipLongestVariable(IteratorVariable):
    """
    Represents itertools.zip_longest(*iterables, fillvalue=None)
    """

    # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L2822-L2887
    _cpython_type = itertools.zip_longest

    _nonvar_fields = {
        "exhausted",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        iterables: "list[VariableTracker]",
        fillvalue: "VariableTracker",
        exhausted: "list[bool] | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.iterables = iterables
        self.fillvalue = fillvalue
        self.exhausted = (
            exhausted if exhausted is not None else [False] * len(iterables)
        )

    def python_type(self) -> type:
        return itertools.zip_longest

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L2737-L2808
        if not self.is_mutable():
            raise AssertionError("ZipLongestVariable must be mutable for next()")
        if all(self.exhausted):
            raise_observed_exception(StopIteration, tx)
        values = []
        # CPython: when the last active iterator exhausts, return without yielding.
        any_active = False
        for i, it in enumerate(self.iterables):
            if self.exhausted[i]:
                values.append(self.fillvalue)
            else:
                try:
                    values.append(pyiter_next(tx, it))
                    any_active = True
                except ObservedUserStopIteration:
                    handle_observed_exception(tx)
                    tx.output.side_effects.mutation(self)
                    self.exhausted[i] = True
                    values.append(self.fillvalue)
        if not any_active:
            raise_observed_exception(StopIteration, tx)
        return variables.TupleVariable(values)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("zip_longest"),
                ]
            ),
            call_function_ex=True,
        )
        for i, it in enumerate(self.iterables):
            if not self.exhausted[i]:
                codegen(it)
            else:
                codegen.add_push_null(
                    lambda: codegen.append_output(
                        codegen.create_load_python_module(iter)  # type: ignore[arg-type]
                    )
                )
                codegen.extend_output(
                    [
                        create_instruction("BUILD_TUPLE", arg=0),
                        *create_call_function(1, False),
                    ]
                )
        codegen.extend_output(
            [create_instruction("BUILD_TUPLE", arg=len(self.iterables))]
        )
        codegen.extend_output([codegen.create_load_const("fillvalue")])
        codegen(self.fillvalue)
        codegen.extend_output(
            [
                create_instruction("BUILD_MAP", arg=1),
                *create_call_function_ex(True, False),
            ]
        )


class MapVariable(IteratorVariable):
    """
    Represents map(fn, *iterables)
    """

    # PyMap_Type: https://github.com/python/cpython/blob/v3.13.0/Python/bltinmodule.c#L1484
    _cpython_type = map

    _nonvar_fields = {
        "strict",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        fn: VariableTracker,
        iterables: VariableTracker,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.fn = fn
        if not isinstance(iterables, variables.TupleVariable):
            raise AssertionError(
                f"Expected a tuple of iterables, got {type(iterables)}"
            )
        self.iterable = iterables
        self.strict = strict

    def python_type(self) -> type:
        return map

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L1409-L1450
        if not self.is_mutable():
            raise AssertionError("MapVariable must be mutable for next()")

        tuplesize = len(self.iterable.items)

        items = []
        for i in range(tuplesize):
            it = self.iterable.items[i]
            try:
                items.append(pyiter_next(tx, it))
            except ObservedUserStopIteration:
                if not self.strict:
                    raise

                handle_observed_exception(tx)  # StopIteration

                if i:
                    raise_value_error(
                        tx,
                        f"map() argument {i + 1} shorter than argument {i}",
                    )

                # In strict mode, if any iterable is exhausted, all must be exhausted.
                for j in range(1, tuplesize):
                    it_j = self.iterable.items[j]
                    if not is_iterator_exhausted(tx, it_j):
                        raise_value_error(
                            tx, f"map() argument {j + 1} is longer than argument {j}"
                        )

                raise_observed_exception(StopIteration, tx)

        # type: ignore[attr-defined]
        return self.fn.call_function(tx, items, {})

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "map"), call_function_ex=True
        )
        codegen(variables.TupleVariable([self.fn, *self.iterable.items]))
        if self.strict:
            if sys.version_info < (3, 14):
                raise AssertionError(
                    "Unexpected bug: map(strict=True) requires Python 3.14+"
                )
            codegen.extend_output(
                [
                    codegen.create_load_const("strict"),
                    codegen.create_load_const(self.strict),
                    create_instruction("BUILD_MAP", arg=1),
                    *create_call_function_ex(True, False),
                ]
            )
        else:
            codegen.extend_output(create_call_function_ex(False, False))


class FilterVariable(IteratorVariable):
    """
    Represents filter(fn, iterable)
    """

    # PyFilter_Type: https://github.com/python/cpython/blob/v3.13.0/Python/bltinmodule.c#L630
    _cpython_type = filter

    def __init__(
        self,
        fn: VariableTracker,
        iterable: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.fn = fn
        self.iterable = iterable

    def python_type(self) -> type:
        return filter

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L573-L606
        # A do-while loop to find elements that make fn return true
        while True:
            item = pyiter_next(tx, self.iterable)
            if self.fn.is_constant_none():
                res = item
            else:
                res = self.fn.call_function(tx, [item], {})
            pred_res = variables.UserFunctionVariable(
                polyfills.predicate  # type: ignore[arg-type]
            ).call_function(tx, [res], {})
            if pred_res.as_python_constant():
                return item

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "filter"))
        codegen(self.fn)
        codegen(self.iterable)
        codegen.extend_output(create_call_function(2, False))


class DictViewIterator(IteratorVariable):
    """Base class for dict view iterators (keys, values, or items)."""

    _nonvar_fields = {
        "view_type",
        *IteratorVariable._nonvar_fields,
    }

    view_type: str = "keys"

    def __init__(
        self,
        items: dict[HashableTracker, VariableTracker],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if self.view_type == "keys":
            self._iter = iter(items)
        elif self.view_type == "values":
            self._iter = iter(items.values())  # type: ignore[bad-assignment]
        else:
            if self.view_type != "items":
                raise AssertionError(
                    f"Expected view_type 'items', got {self.view_type!r}"
                )
            self._iter = iter(items.items())  # type: ignore[bad-assignment]

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # dictiter_iternextitem: https://github.com/python/cpython/blob/v3.13.3/Objects/dictobject.c#L5538-L5578
        # dictiter_iternextkey: https://github.com/python/cpython/blob/v3.13.3/Objects/dictobject.c#L5125-L5144
        # dictiter_iternextvalue: https://github.com/python/cpython/blob/v3.13.3/Objects/dictobject.c#L5248-L5267
        try:
            item = next(self._iter)

            if self.view_type == "keys":
                return item.vt
            elif self.view_type == "values":
                return item  # type: ignore[bad-return]
            else:  # items
                k, v = item  # type: ignore[not-iterable]
                return VariableTracker.build(tx, (k.vt, v))
        except (RuntimeError, StopIteration) as e:
            raise_observed_exception(
                type(e),
                tx,
                args=[VariableTracker.build(tx, a) for a in e.args],
            )

    def dict_view_iter_length_hint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # dictiter_len/setiter_len: __length_hint__ returns the number of
        # not-yet-consumed elements. self._iter is a live Python iterator over
        # the captured items, so its own length hint already reflects any
        # next() calls made during tracing.
        # Not a C-level slot, so it lives in tp_methods rather than call_method.
        return ConstantVariable.create(operator.length_hint(self._iter))

    tp_methods = {
        "__length_hint__": Method(dict_view_iter_length_hint),
    }

    def python_type(self) -> type:
        if self.view_type == "keys":
            return type(iter({}))
        elif self.view_type == "values":
            return type(iter({}.values()))
        else:  # items
            return type(iter({}.items()))


class DictIterator(DictViewIterator):
    _cpython_type = type(iter({}))
    view_type = "keys"


class DictKeysIterator(DictViewIterator):
    _cpython_type = type(iter({}.keys()))
    view_type = "keys"


class DictValuesIterator(DictViewIterator):
    _cpython_type = type(iter({}.values()))
    view_type = "values"


class DictItemsIterator(DictViewIterator):
    _cpython_type = type(iter({}.items()))
    view_type = "items"


class SetIterator(DictViewIterator):
    _cpython_type = type(iter(set()))
    view_type = "keys"
<|endoftext|>"""
This file contains a collection of context manager classes used by Dynamo for tracking
and managing various PyTorch runtime states during graph compilation. These context
managers handle different aspects of PyTorch's execution environment, including:

- Autograd states (grad mode, inference mode)
- CUDA streams and events
- Profiling contexts
- Deterministic algorithms
- Forward/backward AD modes
- SDPA (Scaled Dot Product Attention) kernels
- FSDP (Fully Sharded Data Parallel) states
- AMP (Automatic Mixed Precision) autocast states

The context managers ensure proper state transitions during graph compilation by
tracking enter/exit points and managing cleanup operations. They help maintain
consistency between eager execution and compiled graph behavior by capturing and
restoring state changes.
"""

import contextlib
import inspect
import logging
import types
import warnings
from collections.abc import Callable, Sequence, Sized
from contextlib import AbstractContextManager, ExitStack
from typing import Any, TYPE_CHECKING

import torch._C
from torch._dynamo import config
from torch._guards import Guard
from torch._logging import warning_once

from .. import graph_break_hints, variables
from ..bytecode_transformation import create_call_function
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalStateSource
from ..utils import _get_error_on_graph_break, _set_error_on_graph_break
from .base import VariableTracker
from .functions import (
    NestedUserFunctionVariable,
    SkipFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
    WrappedNestedUserFunctionVariable,
    WrappedSkipFunctionVariable,
    WrappedUserFunctionVariable,
    WrappedUserMethodVariable,
)
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

log = logging.getLogger(__name__)


class ContextWrappingVariable(VariableTracker):
    _nonvar_fields = {
        "cm_obj",
        "target_values",
        "initial_values",
        "state",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self, target_values: Any, initial_values: Any | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.target_values = target_values
        self.initial_values = initial_values
        # target_values must be None or a Sequence for reconstruct / _call_func
        # etc. to work properly.
        if not (target_values is None or isinstance(target_values, Sequence)):
            raise TypeError(
                "ContextWrappingVariable.target_values must be None or a "
                f"Sequence, got {type(target_values).__name__}"
            )

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if hasattr(self, "_call_func"):
            self._call_func(tx, self.target_values)
        self.set_cleanup_hook(tx)
        return variables.ConstantVariable.create(None)

    def set_cleanup_hook(
        self, tx: "InstructionTranslatorBase", fn: Callable[..., Any] | None = None
    ) -> None:
        if fn is None:

            def fn() -> None:
                if hasattr(self, "_call_func"):
                    self._call_func(tx, self.initial_values)

        self.cleanup_fn: Callable[..., Any] | None = fn
        tx.output.add_cleanup_hook(self.cleanup)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        return variables.ConstantVariable.create(None)

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        codegen(
            AttrSource(codegen.tx.import_source(self.module_name()), self.fn_name())
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: self.reconstruct_type(codegen))
        target_values = self.target_values
        if not target_values:
            target_values = ()
        codegen.extend_output([codegen.create_load_const(val) for val in target_values])
        codegen.extend_output(create_call_function(len(target_values), False))

    def module_name(self) -> str:
        raise NotImplementedError("module_name called on base")

    def fn_name(self) -> str:
        raise NotImplementedError("fn_name called on base")

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if len(args) != 1:
            raise AssertionError(f"expected exactly 1 arg, got {len(args)}")
        if not isinstance(
            args[0],
            (
                NestedUserFunctionVariable,
                SkipFunctionVariable,
                UserMethodVariable,
                UserFunctionVariable,
            ),
        ):
            raise AssertionError(f"expected a function variable, got {type(args[0])}")

        if isinstance(args[0], NestedUserFunctionVariable):
            return WrappedNestedUserFunctionVariable(args[0], self)
        elif isinstance(args[0], SkipFunctionVariable):
            return WrappedSkipFunctionVariable(args[0], self)
        elif isinstance(args[0], UserMethodVariable):
            return WrappedUserMethodVariable(args[0], self)
        elif isinstance(args[0], UserFunctionVariable):
            return WrappedUserFunctionVariable(args[0], self)
        else:
            raise AssertionError("Unexpected arg type")

    def supports_graph_breaks(self) -> bool:
        return True

    def exit_on_graph_break(self) -> bool:
        return True

    def cleanup(self) -> None:
        if self.cleanup_fn is not None:
            self.cleanup_fn()
            self.cleanup_fn = None

    def cleanup_assert(self) -> None:
        if not self.cleanup_fn:
            raise AssertionError("multiple exits?")
        self.cleanup()


class GenericContextWrappingVariable(UserDefinedObjectVariable):
    # Some methods in ContextWrappingVariable assumes the arguments are
    # python constants. Which might not always be the case here.
    def __init__(self, cm_obj: AbstractContextManager[Any], **kwargs: Any) -> None:
        if cm_obj is None:
            raise AssertionError("cm_obj must not be None")
        super().__init__(
            value=cm_obj,
            value_type=cm_obj.__class__,
            **kwargs,
        )
        self.cm_obj = cm_obj

    def module_name(self) -> str:
        return self.cm_obj.__module__

    def fn_name(self) -> str:
        return type(self.cm_obj).__name__

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        source = None if self.source is None else AttrSource(self.source, "__enter__")
        return variables.UserMethodVariable(
            self.cm_obj.__enter__.__func__,  # type: ignore[attr-defined]
            self,
            source=source,
        ).call_function(tx, [], {})

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        source = None if self.source is None else AttrSource(self.source, "__exit__")
        x = variables.UserMethodVariable(
            self.cm_obj.__exit__.__func__,  # type: ignore[attr-defined]
            self,
            source=source,
        ).call_function(tx, list(args), {})
        tx.active_generic_context_managers.pop()
        return x

    def supports_graph_breaks(self) -> bool:
        return False

    def exit_on_graph_break(self) -> bool:
        return True


class RecordFunctionVariable(GenericContextWrappingVariable):
    def __init__(self, cm_obj: torch.profiler.record_function, **kwargs: Any) -> None:
        super().__init__(cm_obj, **kwargs)
        args = [variables.ConstantVariable.create(cm_obj.name)]
        record_kwargs = {}
        if cm_obj.args:
            record_kwargs["args"] = variables.ConstantVariable.create(cm_obj.args)
        self.record_fn = variables.torch.ProfilerRecordFunctionContextVariable.create(
            func=type(self.cm_obj), record_args=args, record_kwargs=record_kwargs
        )

    def enter(self, tx):
        tx.active_generic_context_managers.append(self)
        return self.record_fn.enter(tx)

    def exit(self, tx, *args):
        tx.active_generic_context_managers.pop()
        return self.record_fn.exit(tx, *args)


class RepararametrizeModuleContextVariable(GenericContextWrappingVariable):
    def __init__(self, ctx_manager_vt: ContextWrappingVariable, mod: Any) -> None:
        self.cm_vt = ctx_manager_vt
        self.mod = mod
        # We don't call super().__init__() because we're delegating most methods to cm_vt

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # Custom enter implementation with side effects. Additionally, do not
        # worry about side effects for the hop, as they cancel each other out.
