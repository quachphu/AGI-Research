        # https://github.com/python/cpython/blob/c09ccd9c429/Objects/longobject.c#L5200
        # https://github.com/python/cpython/blob/c09ccd9c429/Objects/floatobject.c#L853
        if isinstance(self.sym_num, torch.SymBool):
            return self
        if not isinstance(self.sym_num, (torch.SymInt, torch.SymFloat)):
            raise AssertionError(
                f"Expected SymInt or SymFloat, got {type(self.sym_num)}"
            )
        return SymNodeVariable.create(tx, self.as_proxy() != 0)

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """SymNode tp_richcompare: symbolic numeric comparison."""
        if not isinstance(other, (SymNodeVariable, ConstantVariable, TensorVariable)):
            if op in ("__eq__", "__ne__"):
                op_fn = cmp_name_to_op_mapping[op]
                return VariableTracker.build(tx, op_fn(object(), None))
            return ConstantVariable.create(NotImplemented)
        op_fn = cmp_name_to_op_mapping[op]
        proxy = tx.output.create_proxy(
            "call_function", op_fn, (self.as_proxy(), other.as_proxy()), {}
        )
        return SymNodeVariable.create(tx, proxy, sym_num=None)

    def as_tensor(self, tx: "InstructionTranslatorBase", dtype: Any) -> TensorVariable:
        if self._tensor_var is None:
            self._tensor_var = VariableTracker.build(
                tx, torch.scalar_tensor
            ).call_function(tx, [self], {"dtype": VariableTracker.build(tx, dtype)})
        return self._tensor_var

    def evaluate_expr(
        self, output_graph: Optional["OutputGraph"] = None
    ) -> bool | int | float:
        try:
            return guard_scalar(self.sym_num)
        except GuardOnDataDependentSymNode as e:
            if torch.fx.experimental._config.no_data_dependent_graph_break:
                raise

            raise UserError(  # noqa: B904
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using torch._check*(). {str(e)}",
                case_name="constrain_as_size_example",
            )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                name,
                *proxy_args_kwargs([self, *args], kwargs),
            ),
        )

    def nb_index_impl(
        self,
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        # SymInt / SymBool define __index__ as `self.node.int_()`, which
        # specializes the symbolic value to a concrete int (with guard).
        # SymFloat has no __index__.
        # ref: torch/__init__.py SymInt.__index__ / SymBool.__index__

        pytype = self.python_type()
        if pytype in (int, bool):
            return variables.ConstantVariable.create(self.evaluate_expr())
        return super().nb_index_impl(tx)

    def nb_int_impl(
        self,
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        # SymInt.__int__: https://github.com/pytorch/pytorch/blob/ee336ca5440939b8ad65e916d47421f849e56178/torch/__init__.py#L462
        # SymFloat.__int__: https://github.com/pytorch/pytorch/blob/ee336ca5440939b8ad65e916d47421f849e56178/torch/__init__.py#L682
        # SymBool.__int__: https://github.com/pytorch/pytorch/blob/ee336ca5440939b8ad65e916d47421f849e56178/torch/__init__.py#L784
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                sym_int,
                (self.as_proxy(),),
                {},
            ),
        )

    def method___int__(
        self, tx: "InstructionTranslatorBase", *args: Any, **kwargs: Any
    ) -> VariableTracker:
        return self.nb_int_impl(tx)

    def nb_add_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1139 (PyNumber_Add)
        if not _is_sym_arith_operand(other):
            return VariableTracker.build(tx, NotImplemented)
        args = [other, self] if reverse else [self, other]
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.add, *proxy_args_kwargs(args, {})
            ),
            sym_num=None,
        )

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if not other.is_symnode_like():
            return VariableTracker.build(tx, NotImplemented)
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.or_, *proxy_args_kwargs([self, other], {})
            ),
            sym_num=None,
        )

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if not other.is_symnode_like():
            return VariableTracker.build(tx, NotImplemented)
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.and_, *proxy_args_kwargs([self, other], {})
            ),
            sym_num=None,
        )

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if not other.is_symnode_like():
            return VariableTracker.build(tx, NotImplemented)
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.xor, *proxy_args_kwargs([self, other], {})
            ),
            sym_num=None,
        )

    def nb_lshift_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if not other.is_symnode_like():
            return VariableTracker.build(tx, NotImplemented)
        args = [other, self] if reverse else [self, other]
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.lshift, *proxy_args_kwargs(args, {})
            ),
            sym_num=None,
        )

    def nb_rshift_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if not other.is_symnode_like():
            return VariableTracker.build(tx, NotImplemented)
        args = [other, self] if reverse else [self, other]
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.rshift, *proxy_args_kwargs(args, {})
            ),
            sym_num=None,
        )

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1135 (PyNumber_Subtract)
        if not _is_sym_arith_operand(other):
            return VariableTracker.build(tx, NotImplemented)
        args = [other, self] if reverse else [self, other]
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.sub, *proxy_args_kwargs(args, {})
            ),
            sym_num=None,
        )

    def nb_multiply_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # sequence * sym_node must go through sq_repeat and not nb_multiply
        if not _is_sym_arith_operand(other):
            return VariableTracker.build(tx, NotImplemented)
        lhs, rhs = (other, self) if reverse else (self, other)
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.mul, *proxy_args_kwargs([lhs, rhs], {})
            ),
            sym_num=None,
        )

    def nb_floor_divide_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if not _is_sym_arith_operand(other):
            return VariableTracker.build(tx, NotImplemented)
        lhs, rhs = (other, self) if reverse else (self, other)
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.floordiv, *proxy_args_kwargs([lhs, rhs], {})
            ),
            sym_num=None,
        )

    def nb_true_divide_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if not _is_sym_arith_operand(other):
            return VariableTracker.build(tx, NotImplemented)
        lhs, rhs = (other, self) if reverse else (self, other)
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.truediv, *proxy_args_kwargs([lhs, rhs], {})
            ),
            sym_num=None,
        )

    def nb_remainder_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if not _is_sym_arith_operand(other):
            return VariableTracker.build(tx, NotImplemented)
        lhs, rhs = (other, self) if reverse else (self, other)
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.mod, *proxy_args_kwargs([lhs, rhs], {})
            ),
            sym_num=None,
        )

    def nb_float_impl(
        self,
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        # SymFloat.__float__: https://github.com/pytorch/pytorch/blob/ee336ca5440939b8ad65e916d47421f849e56178/torch/__init__.py#L679
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                sym_float,
                (self.as_proxy(),),
                {},
            ),
        )

    def method___float__(
        self, tx: "InstructionTranslatorBase", *args: Any, **kwargs: Any
    ) -> VariableTracker:
        return self.nb_float_impl(tx)

    def nb_negative_impl(
        self,
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        return SymNodeVariable.create(
            tx,
            operator.neg(self.as_proxy()),
            sym_num=None,
        )

    def method___neg__(
        self, tx: "InstructionTranslatorBase", *args: Any, **kwargs: Any
    ) -> VariableTracker:
        return self.nb_negative_impl(tx)

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        return hash(self.evaluate_expr()), False

    def nb_positive_impl(
        self,
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        return SymNodeVariable.create(
            tx,
            operator.pos(self.as_proxy()),
            sym_num=None,
        )

    def method___pos__(
        self, tx: "InstructionTranslatorBase", *args: Any, **kwargs: Any
    ) -> VariableTracker:
        return self.nb_positive_impl(tx)

    def nb_absolute_impl(
        self,
        tx: "InstructionTranslatorBase",
    ) -> VariableTracker:
        return SymNodeVariable.create(
            tx,
            operator.abs(self.as_proxy()),
            sym_num=None,
        )

    def nb_power_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        z: VariableTracker | None,
        reverse: bool = False,
    ) -> VariableTracker:
        if z is not None:
            return VariableTracker.build(tx, NotImplemented)
        if not _is_sym_arith_operand(other):
            return VariableTracker.build(tx, NotImplemented)
        lhs, rhs = (other, self) if reverse else (self, other)
        return SymNodeVariable.create(
            tx,
            tx.output.create_proxy(
                "call_function", operator.pow, *proxy_args_kwargs([lhs, rhs], {})
            ),
            sym_num=None,
        )

    def method___abs__(
        self, tx: "InstructionTranslatorBase", *args: Any, **kwargs: Any
    ) -> VariableTracker:
        return self.nb_absolute_impl(tx)

    def is_python_hashable(self) -> bool:
        return True

    def get_python_hash(self) -> int:
        # Essentially convert the SymNode to a constant variable whenever its
        # searched for a dict key.
        return hash(self.evaluate_expr())


class NumpyNdarrayVariable(TensorVariable):
    """
    Represents a np.ndarray, but backed by torch Tensor via torch._numpy.ndarray.
    Use this for Tensor.numpy() call.
    """

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", proxy: torch.fx.Proxy, **options: Any
    ) -> "NumpyNdarrayVariable":
        from .builder import wrap_fx_proxy_cls

        return wrap_fx_proxy_cls(
            target_cls=NumpyNdarrayVariable,
            tx=tx,
            proxy=proxy,
            **options,
        )

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        from ..exc import raise_type_error

        raise_type_error(tx, "unhashable type: 'numpy.ndarray'")

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """ndarray tp_richcompare: element-wise comparison via numpy_operator_wrapper."""
        from ..utils import numpy_operator_wrapper

        if not isinstance(other, (SymNodeVariable, ConstantVariable, TensorVariable)):
            return ConstantVariable.create(NotImplemented)
        op_fn = cmp_name_to_op_mapping[op]
        proxy = tx.output.create_proxy(
            "call_function",
            numpy_operator_wrapper(op_fn),
            (self.as_proxy(), other.as_proxy()),
            {},
        )
        return NumpyNdarrayVariable.create(tx, proxy)

    # NB: ndim/itemsize are ALWAYS specialized constants (numpy exposes them via
    # PyGetSetDef on ndarray), unlike size/shape which may carry free symbols.
    def _get_ndim(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        example_value = self.as_proxy().node.meta["example_value"]
        return VariableTracker.build(tx, tnp.ndarray(example_value).ndim)

    def _get_itemsize(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        example_value = self.as_proxy().node.meta["example_value"]
        return VariableTracker.build(tx, tnp.ndarray(example_value).itemsize)

    def _get_numpy_attr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        from ..utils import numpy_attr_wrapper

        proxy = tx.output.create_proxy(
            "call_function", numpy_attr_wrapper, (self.as_proxy(), name), {}
        )
        return NumpyNdarrayVariable.create(tx, proxy)

    tp_getset = {
        "ndim": GetSet(_get_ndim, None),
        "itemsize": GetSet(_get_itemsize, None),
        "T": GetSet(lambda s, tx: s._get_numpy_attr(tx, "T")),
        "real": GetSet(lambda s, tx: s._get_numpy_attr(tx, "real")),
        "imag": GetSet(lambda s, tx: s._get_numpy_attr(tx, "imag")),
        "flat": GetSet(lambda s, tx: s._get_numpy_attr(tx, "flat")),
    }

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # NB: This INTENTIONALLY does not call super(), because there is
        # no intrinsic reason ndarray properties are related to Tensor
        # properties.  The inheritance here is for implementation sharing.
        # tp_getset (ndim/itemsize/T/real/imag/flat) is resolved by
        # generic_getattr before this method is reached, so it is not
        # consulted here.
        from ..utils import numpy_attr_wrapper
        from .builder import wrap_fx_proxy

        example_value = self.as_proxy().node.meta["example_value"]
        example_ndarray = tnp.ndarray(example_value)

        def insert_into_graph() -> VariableTracker:
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_function", numpy_attr_wrapper, (self.as_proxy(), name), {}
                ),
            )

        # These are awkward to implement.  The standard playbook for torch._numpy
        # interop is to trace a call into the torch._numpy wrapper which works for
        # Tensor operations.  However, we don't want to do this for calls
        # that don't return Tensors, because in those cases we may not want
        # to trace the attribute access into the graph at all (it is sort
        # of harmless to do so, because AOTAutograd will eliminate them,
        # but it's best not to trace them in to begin with.)  But in any
        # case, tracing these into the graph is like trying to fit a square
        # peg into a round hole; best not to do it.  So instead we
        # painstakingly implement these by hand
        #
        # NB: only ALWAYS specialized attributes can go here; notably,
        # size/shape not allowed!
        if name in ("shape", "stride"):
            if not has_free_symbols(r := getattr(example_ndarray, name)):
                return VariableTracker.build(tx, tuple(int(r) for r in r))
            return insert_into_graph()
        elif name == "size":
            if not has_free_symbols(r := example_ndarray.size):
                return VariableTracker.build(tx, int(r))
            return insert_into_graph()
        elif name in ["base", "flags", "dtype"]:
            unimplemented(
                gb_type="Unsupported ndarray attribute access",
                context=f"tp_getattro_impl {self} {name}",
                explanation=f"Dynamo currently does not support tracing `ndarray.{name}`.",
                hints=[],
            )
        elif name == "__version__":
            unimplemented(
                gb_type="Unsupported ndarray.__version__ access",
                context=f"tp_getattro_impl {self} {name}",
                explanation=f"Dynamo currently does not support tracing `ndarray.{name}`.",
                hints=[],
            )
        raise NotImplementedError

    @staticmethod
    def patch_args(
        name: str, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> tuple[list[VariableTracker], dict[str, VariableTracker]]:
        if name == "clip":
            kwargs_rename = {"a_min": "min", "a_max": "max"}
            kwargs = {kwargs_rename.get(k, k): v for k, v in kwargs.items()}
        return args, kwargs

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..exc import unimplemented
        from ..utils import numpy_method_wrapper

        args, kwargs = self.patch_args(name, args, kwargs)

        if name == "astype":
            from .builtin import BuiltinVariable

            dtype_arg = None
            if "dtype" in kwargs:
                dtype_arg = kwargs["dtype"]
            elif len(args) > 0:
                dtype_arg = args[0]
            is_object_str = dtype_arg is not None and dtype_arg.is_constant_match("O")
            is_object_type = (
                isinstance(dtype_arg, BuiltinVariable) and dtype_arg.fn is object
            )
            if is_object_str or is_object_type:
                unimplemented(
                    gb_type="ndarray.astype(object)",
                    context=f"call_method {self} {name} {args} {kwargs}",
                    explanation=(
                        "`ndarray.astype('O')` or `ndarray.astype(object)` is not supported "
                        "by torch.compile, as there is no equivalent to object type in torch.Tensor. "
                        "This will be executed eagerly."
                    ),
                    hints=[*graph_break_hints.FUNDAMENTAL],
                )
        if name in ["__len__", "size", "tolist", "__iter__"]:
            # delegate back to TensorVariable
            return super().call_method(tx, name, args, kwargs)
        if name in ("tostring", "tobytes", "__delattr__"):
            unimplemented(
                gb_type="Unsupported ndarray method call",
                context=f"call_method {self} {name} {args} {kwargs}",
                explanation=f"`ndarray.{name}()` is not modelled in `torch._numpy`.",
                hints=[],
            )
        proxy = tx.output.create_proxy(
            "call_function",
            numpy_method_wrapper(name),
            *proxy_args_kwargs([self] + list(args), kwargs),
        )
        return NumpyNdarrayVariable.create(tx, proxy)

    def python_type(self) -> type:
        if np is not None:
            return np.ndarray
        else:
            return NoneType

    def mp_ass_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        if value is None:
            raise_value_error(tx, "cannot delete array elements")
        return self.call_method(tx, "__setitem__", [key, value], {})


class UnspecializedPythonVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized python float/int.
    """

    _nonvar_fields = {
        "raw_value",
        "need_unwrap",
        *TensorVariable._nonvar_fields,
    }

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        *,
        raw_value: float | int | None = None,
        need_unwrap: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(proxy, **kwargs)
        self.raw_value = raw_value
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(
        cls,
        tensor_variable: TensorVariable,
        raw_value: float | int | None,
        need_unwrap: bool = True,
    ) -> "UnspecializedPythonVariable":
        # Convert a `TensorVariable` instance into an `UnspecializedPythonVariable` instance.
        return UnspecializedPythonVariable(
            **dict(tensor_variable.__dict__),
            raw_value=raw_value,
            need_unwrap=need_unwrap,
        )


class FakeItemVariable(TensorVariable):
    """An unspecialized python variable which prevents access to the underlying raw value.
    This is needed if item is called on a FakeTensor."""

    _nonvar_fields = {
        "need_unwrap",
        *TensorVariable._nonvar_fields,
    }

    def __init__(self, proxy: torch.fx.Proxy, **kwargs: Any) -> None:
        need_unwrap = kwargs.pop("need_unwrap", False)
        super().__init__(proxy, **kwargs)
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(
        cls, tensor_variable: TensorVariable
    ) -> "FakeItemVariable":
        return FakeItemVariable(**dict(tensor_variable.__dict__))


class TensorSubclassVariable(UserDefinedClassVariable):
    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Handle `Subclass(existing_tensor, ...)` calls.
        from .torch_function import TensorWithTFOverrideVariable

        new_func = self.value.__new__
        if new_func is torch.Tensor.__new__:
            var = None
            if len(args) == 1 and args[0].is_tensor() and len(kwargs) == 0:
                data = args[0]
                # Simulate `torch.Tensor.__new__` as shallow-copying the input
                # tensor data with a new type. TODO polyfill?
                var = TensorWithTFOverrideVariable.from_tensor_var(
                    tx, data, self.value, self.source
                )
            else:
                unimplemented(
                    gb_type="Calling subclass default constructor with more than tensor argument",
                    context=f"{self.value}(args={args}, kwargs={kwargs})",
                    explanation="Currently not supported",
                    hints=[
                        "Avoid this constructor call or move it outside "
                        "`torch.compile` region",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
        else:
            # Let Dynamo trace through custom `__new__`
            var = VariableTracker.build(tx, new_func).call_function(
                tx, [self] + list(args), kwargs
            )
        if var is None:
            raise AssertionError("__new__ must return a non-None variable")
        # Let Dynamo trace through custom `__init__`
        init_func = self.value.__init__
        # TODO builder should be able to handle `torch.Tensor.__init__`,
        # which is `object.__init__`, so that we can remove this check.
        if init_func is not torch.Tensor.__init__:
            VariableTracker.build(tx, init_func).call_function(tx, [var], kwargs)

        # See NOTE [Side effect tracking for newly constructed tensor]
        tx.output.side_effects._track_obj(
            object(), var, mutation_type_cls=AttributeMutationNew
        )
        return var

    def as_python_constant(self) -> type:
        return self.value


class UntypedStorageVariable(VariableTracker):
    _nonvar_fields = {
        "example_value",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        from_tensor: TensorVariable,
        example_value: torch.UntypedStorage,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.from_tensor = from_tensor
        # Example_value will always have device="meta"
        self.example_value = example_value

    def python_type(self) -> type:
        return torch.UntypedStorage

    def method_size(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        result = self.example_value.size()
        if not has_free_symbols(result):
            # avoid creating a node in the graph
            return VariableTracker.build(tx, int(result))
        from ..external_utils import untyped_storage_size
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function",
                untyped_storage_size,
                (self.from_tensor.as_proxy(),),
                {},
            ),
        )

    def method_resize_(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        tx.output.create_proxy(
            "call_function",
            torch.ops.inductor.resize_storage_bytes_,
            (self.from_tensor.as_proxy(), args[0].as_proxy()),
            {},
        )
        return self

    tp_methods = {
        "size": Method(method_size),
        "resize_": Method(method_resize_),
    }

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.from_tensor)
        codegen.load_method("untyped_storage")
        codegen.call_method(0)


class DataPtrVariable(VariableTracker):
    _DATA_PTR_PRESERVING_TARGETS = {
        ("call_method", "detach"),
        ("call_function", torch.ops.aten.detach.default),
    }

    def __init__(
        self,
        from_tensor: TensorVariable,
        method_name: str = "data_ptr",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.from_tensor = from_tensor
        self.method_name = method_name
        self.tensor_version = from_tensor._get_fake_version()

    def python_type(self) -> type:
        return int

    @classmethod
    def _strip_data_ptr_preserving_aliases(cls, node: torch.fx.Node) -> torch.fx.Node:
        while (
            isinstance(node, torch.fx.Node)
            and (node.op, node.target) in cls._DATA_PTR_PRESERVING_TARGETS
            and len(node.args) == 1
            and isinstance(node.args[0], torch.fx.Node)
        ):
            node = node.args[0]
        return node

    def _is_same_data_ptr(self, other: "VariableTracker") -> bool:
        if not isinstance(other, DataPtrVariable):
            return False

        if self.tensor_version is None or self.tensor_version != other.tensor_version:
            return False

        self_root = self._strip_data_ptr_preserving_aliases(
            self.from_tensor.as_proxy().node
        )
        other_root = self._strip_data_ptr_preserving_aliases(
            other.from_tensor.as_proxy().node
        )
        return self_root is other_root

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """DataPtr tp_richcompare: identity-based eq/ne."""
        if op not in ("__eq__", "__ne__"):
            return ConstantVariable.create(NotImplemented)
        same_data_ptr = self._is_same_data_ptr(other)
        if same_data_ptr:
            return ConstantVariable.create(op == "__eq__")
        unimplemented(
            gb_type="Data pointer comparison",
            context=f"tp_richcompare_impl {self} {op} {other}",
            explanation="Dynamo can only trace data pointer comparisons "
            "when it can prove both operands have the same data pointer.",
            hints=[],
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.from_tensor)
        codegen.load_method(self.method_name)
        codegen.call_method(0)
<|endoftext|>"""
This module implements variable tracking for PyTorch nn.Module instances during Dynamo tracing.

It provides specialized handling for different types of nn.Module instances through several key classes:

- NNModuleVariable: Handles instance-specific module tracing, specializing on module id() and placing
  parameters directly on the torch.fx.GraphModule. This creates one graph per module instance.

- UnspecializedNNModuleVariable: Provides class-level module tracing, treating nn.Modules like other
  user-defined objects and passing parameters as inputs to the FX graph. This creates one graph per
  module class.

- UnspecializedBuiltinNNModuleVariable: Specifically handles built-in PyTorch modules (e.g. nn.Linear)
  with appropriate optimizations.

- FSDPManagedNNModuleVariable: Special handling for FSDP-wrapped modules with modified guarding behavior
  and parameter handling.

The module integrates with Dynamo's broader tracing functionality to handle module method calls,
parameter access, hooks, and other nn.Module behaviors while maintaining proper scoping and guarding
of module state.
"""

import collections
import functools
import inspect
import itertools
import re
import types
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from typing import Any, TYPE_CHECKING

import torch.nn
from torch._guards import Source

from .. import graph_break_hints, trace_rules, variables
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
    UnspecializeRestartAnalysis,
    Unsupported,
)
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..mutation_guard import GenerationTracker
from ..source import (
    AttrSource,
    ConstDictKeySource,
    DictGetItemSource,
    FSDPNNModuleSource,
    GetItemSource,
    NNModuleSource,
    UnspecializedNNModuleSource,
)
from ..utils import (
    enumerate_items_with_dict_position,
    get_custom_getattr,
    get_fake_value,
    is_lazy_module,
    is_namedtuple,
    is_safe_constant,
    istensor,
    istype,
    nnmodule_has_hooks,
    object_has_getattribute,
    proxy_args_kwargs,
    raise_args_mismatch,
    set_example_value,
    unpatched_nn_module_call,
    unpatched_nn_module_call_impl,
)
from .base import Method, typestr, ValueMutationNew, VariableTracker
from .functions import invoke_and_store_as_constant
from .lazy import LazyVariableTracker
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from .constant import ConstantVariable


def initialize_lazy_module(
    tx: "InstructionTranslatorBase",
    mod: torch.nn.Module,
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> None:
    """
    Fairly coupled helper used by NNModuleVariable and UnspecializedNNModuleVariable.

    Used to cause lazy module to be initialized (and delete its init hook) before tracing. Especially
    useful now that 'allowed' modules graph-break on hooks, calling this first ensures there is no hook
    by the time we trace __call__ and thus no graph-break for lazy allowed modules.
    """
    if inspect.getattr_static(mod, "_initialize_hook", None) is not None:

        def convert_to_fake(x: Any) -> Any:
            if is_namedtuple(x):
                return type(x)(*(convert_to_fake(elem) for elem in x))
            elif isinstance(x, dict):
                return {k: convert_to_fake(v) for k, v in x.items()}  # type: ignore[misc]
            elif isinstance(x, (list, tuple, set)):
                return type(x)(convert_to_fake(elem) for elem in x)
            elif isinstance(x, torch.fx.Proxy):
                fake = get_fake_value(x.node, tx)
                if isinstance(fake, torch.Tensor) and any(
                    isinstance(s, torch.SymInt) for s in fake.shape
                ):
                    # _infer_parameters runs real ops on the module, so
                    # symbolic shapes must be concretized to their hints.
                    shape = [
                        s.node.hint if isinstance(s, torch.SymInt) else s
                        for s in fake.shape
                    ]
                    if not all(isinstance(s, int) for s in shape):
                        raise AssertionError(
                            f"expected all shape entries to be int, got {shape}"
                        )
                    return torch.empty(  # pyrefly: ignore[no-matching-overload]
                        shape, dtype=fake.dtype, device=fake.device
                    )
                return fake
            else:
                return x

        proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)
        fake_args = [convert_to_fake(arg) for arg in proxy_args]
        fake_kwargs = {k: convert_to_fake(v) for k, v in proxy_kwargs.items()}
        try:
            mod._infer_parameters(mod, fake_args, fake_kwargs)  # type: ignore[operator]
        except AttributeError:
            # Re-raise with the original error message from the AttributeError
            raise_observed_exception(
                AttributeError,
                tx,
                args=["AttributeError during lazy module initialization"],
            )


@contextmanager
def record_nn_module_stack(
    module_key: str,
    source: Source,
    tx: "InstructionTranslatorBase",
    mod: torch.nn.Module,
) -> Any:
    fully_qualified_name = source.name
    # Remove redundant namings
    fully_qualified_name = re.sub(
        r"\._(?:modules|parameters|buffers)\[(['\"])([^'\"\]]+)\1\]",
        r".\2",
        fully_qualified_name,
    )
    num_calls = tx.num_calls.get(fully_qualified_name, 0)
    module_key = f"{module_key}@{num_calls}" if num_calls > 0 else module_key
    try:
        tx.nn_module_stack[module_key] = (fully_qualified_name, mod.__class__)
        tx.num_calls[fully_qualified_name] = num_calls + 1
        yield
    finally:
        del tx.nn_module_stack[module_key]


def guard_to_detect_forward_monkeypatching(
    source: Source | None, mod: torch.nn.Module
) -> None:
    # Users sometimes patch the forward method of a nn module instance to
    # perform optimizations like quantization. Though this is not a good
    # software practice, but python allows this and Dynamo needs to detect
    # this patching.
    #
    # One way to do this is to add an ID_MATCH guard on every function
    # getting inlined (https://github.com/pytorch/pytorch/pull/124975). But
    # this increased guard overhead by around 20%.
    #
    # To keep the guard overhead down, we just guard on the `forward` being
    # not present in the mod __dict__. The common case of patching forward
    # method adds `forward` in the instance __dict__, whereas the unpatched
    # `forward` sits in the type(mod).__dict__
    if source:
        if "forward" in mod.__dict__ and callable(mod.__dict__["forward"]):
            # Monkeypatched forward method, guard on call-relevant structure.
            fwd = mod.__dict__["forward"]
            forward_source = AttrSource(source, "forward")
            if type(fwd) is types.MethodType:
                forward_source = AttrSource(forward_source, "__func__")
                install_guard(forward_source.make_guard(GuardBuilder.CLOSURE_MATCH))
            elif isinstance(fwd, functools.partial):
                guard_to_detect_forward_partial_monkeypatching(
                    source, mod, forward_source, fwd
                )
            else:
                install_guard(forward_source.make_guard(GuardBuilder.CLOSURE_MATCH))
        else:
            # Common case - check that the forward key is absent in mod __dict__
            install_guard(
                source.make_guard(
                    functools.partial(
                        GuardBuilder.NOT_PRESENT_IN_GENERIC_DICT, attr="forward"
                    )
                )
            )


def guard_to_detect_forward_partial_monkeypatching(
    module_source: Source,
    mod: torch.nn.Module,
    partial_source: Source,
    partial_obj: functools.partial[Any],
) -> None:
    install_guard(partial_source.make_guard(GuardBuilder.TYPE_MATCH))

    func_source = AttrSource(partial_source, "func")
    if isinstance(partial_obj.func, functools.partial):
        guard_to_detect_forward_partial_monkeypatching(
            module_source, mod, func_source, partial_obj.func
        )
    else:
        install_guard(func_source.make_guard(GuardBuilder.CLOSURE_MATCH))

    args_source = AttrSource(partial_source, "args")
    install_guard(args_source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
    for i, arg in enumerate(partial_obj.args):
        guard_to_detect_forward_partial_value(
            module_source, mod, GetItemSource(args_source, i), arg
        )

    keywords_source = AttrSource(partial_source, "keywords")
    if partial_obj.keywords is None:
        install_guard(keywords_source.make_guard(GuardBuilder.NONE_MATCH))
        return

    install_guard(keywords_source.make_guard(GuardBuilder.DICT_KEYS_MATCH))
    for key, value in partial_obj.keywords.items():
        guard_to_detect_forward_partial_value(
            module_source, mod, DictGetItemSource(keywords_source, key), value
        )


def guard_to_detect_forward_partial_value(
    module_source: Source,
    mod: torch.nn.Module,
    value_source: Source,
    value: Any,
) -> None:
    if value is mod:
        dupe_guard = make_dupe_guard(value_source, module_source)
        install_guard(value_source.make_guard(dupe_guard or GuardBuilder.ID_MATCH))
    elif isinstance(value, functools.partial):
        guard_to_detect_forward_partial_monkeypatching(
            module_source, mod, value_source, value
        )
    elif type(value) is types.FunctionType:
        install_guard(value_source.make_guard(GuardBuilder.CLOSURE_MATCH))
    elif is_safe_constant(value):
        install_guard(value_source.make_guard(GuardBuilder.CONSTANT_MATCH))
    else:
        install_guard(value_source.make_guard(GuardBuilder.ID_MATCH))


class NNModuleVariable(VariableTracker):
    _nonvar_fields = {
        "module_type",
        "module_key",
        "value",
        "nn_module_stack_source",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self, module_type: type, module_key: str, value: torch.nn.Module, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.module_type = module_type
        self.module_key = module_key
        self.value = value
        # pyrefly: ignore[bad-override]
        # NOTE: Don't remove this; better than adding suppressions
        # everywhere else with asserts
        self.source: Source = self.source
        self.nn_module_stack_source = self.source

    def get_nn_module_stack_source(self) -> Source:
        res = self.nn_module_stack_source or self.source
        if not res:
            raise AssertionError("nn_module_stack_source must not be None")
        return res

    def set_nn_module_stack_source(self, source: Source) -> None:
        self.nn_module_stack_source = source

    def python_type(self) -> type:
        return self.module_type

    def get_id(self, tx: "InstructionTranslatorBase") -> int | None:
        return id(tx.output.get_submodule(self.module_key))

    def get_real_python_backed_value(self) -> object:
        return self.value

    def nb_bool_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        """nb_bool for nn.Module.

        nn.Module itself has no __bool__ or __len__, so bare modules are always
        truthy.  Subclasses like ModuleList/ModuleDict define __len__, so
        bool(module) calls PyObject_IsTrue which falls through nb_bool (NULL)
        to sq_length/mp_length.  We evaluate on the real module to capture this.
        """
        from .constant import ConstantVariable

        mod = tx.output.get_submodule(self.module_key)
        return ConstantVariable.create(bool(mod))

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        mod = tx.output.get_submodule(self.module_key)
        return VariableTracker.build(tx, repr(mod))

    def _wrap_submodule(
        self,
        tx: "InstructionTranslatorBase",
        source: Source,
        submod: torch.nn.Module,
        *key_extra: Any,
        **options: Any,
    ) -> None:
        return

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        # implement list/iter/tuple/etc calls
        base = tx.output.get_submodule(self.module_key)
        result: list[VariableTracker] = []
        if isinstance(base, torch.nn.ModuleDict):
            for name, submod in base.items():
                name_var = VariableTracker.build(tx, name)
                tx.output.register_attr_or_module(
                    submod,
                    self.module_key,
                    name,
                    source=NNModuleSource(GetItemSource(self.source, name)),  # type: ignore[arg-type]
                )
                result.append(name_var)
            return result

        if not isinstance(
            base, (torch.nn.ModuleList, torch.nn.ParameterList, torch.nn.Sequential)
        ):
            raise AssertionError(
                f"Expected ModuleList, ParameterList, or Sequential, got {typestr(base)}"
            )
        for idx, submod in enumerate(base):
            result.append(
                tx.output.register_attr_or_module(
                    submod,
                    self.module_key,
                    idx,
                    source=NNModuleSource(GetItemSource(self.source, idx)),
                )
            )
        return result

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "ConstantVariable":
        mod = tx.output.get_submodule(self.module_key)
        result = hasattr(mod, name)
        install_guard(
            self.source.make_guard(functools.partial(GuardBuilder.HASATTR, attr=name))
        )
        return VariableTracker.build(tx, result)

    def is_training(self, tx: "InstructionTranslatorBase") -> bool:
        mod = tx.output.get_submodule(self.module_key)
        return getattr(mod, "training", False)

    def convert_to_unspecialized(self, tx: "InstructionTranslatorBase") -> None:
        """Restart analysis treating this module as an UnspecializedNNModuleVariable"""
        mod = tx.output.get_submodule(self.module_key)
        GenerationTracker.tag(mod)

        # Mark the class dynamic unless its module initialization
        if tx.f_code.co_name != "__init__":
            GenerationTracker.mark_class_dynamic(type(mod))
        raise UnspecializeRestartAnalysis

    def _custom_getattr_fallback(
        self,
        base: torch.nn.Module,
        tx: "InstructionTranslatorBase",
        name: str,
        obj_source: Source,
    ) -> VariableTracker | None:
        """Check for a __getattr__ and handle it specially if it is implemented"""
        if object_has_getattribute(base):
            getattribute_fn = inspect.getattr_static(type(base), "__getattribute__")
            new_source = (
                AttrSource(obj_source, "__getattribute__") if obj_source else None
            )
            try:
                return variables.UserMethodVariable(
                    getattribute_fn,
                    self,
                    source=new_source,
                ).call_function(tx, [variables.ConstantVariable.create(name)], {})
            except ObservedAttributeError:
                handle_observed_exception(tx)
            except Unsupported:
                unimplemented(
                    gb_type="Custom __getattribute__ in nn.Module attribute access",
                    context=f"tp_getattro_impl {self} {name}",
                    explanation="Dynamo could not trace through the custom "
                    "`__getattribute__` method on this `nn.Module`.",
                    hints=[
                        "Simplify your `__getattribute__` implementation, "
                        "or replace it with a targeted `@property`.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

        getattr_fn = get_custom_getattr(base, ignore_nn_module_getattr=True)
        if getattr_fn is None:
            return None

        if not isinstance(getattr_fn, types.FunctionType):
            unimplemented(
                gb_type="torch.nn.Module with a non-function custom __getattr__",
                context=f"tp_getattro_impl {self} {name}",
                explanation=(
                    "Dynamo detected a nn.Module object with a custom "
                    "`__getattr__` method, but this method is not a standard "
                    "Python function (e.g., it might be implemented in C/C++). "
                    "Dynamo cannot currently trace into such non-standard "
                    "`__getattr__` methods."
                ),
                hints=[
                    "Avoid using objects with non-standard __getattr__ methods "
                    "within the compiled region. If possible, implement "
                    "__getattr__ as a standard Python function.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        options = {"source": AttrSource(obj_source, "__getattr__")}

        return variables.UserMethodVariable(getattr_fn, self, **options).call_function(
            tx, [VariableTracker.build(tx, name)], {}
        )

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        source = self.source and AttrSource(self.source, name)

        base = tx.output.get_submodule(self.module_key)
        # NB: We look up attributes in __dict__ directly, bypassing any custom
        # __getattribute__. Custom __getattribute__ is only traced through as a
        # fallback (via _custom_getattr_fallback) for attributes not found here.
        base_dict = object.__getattribute__(base, "__dict__")
        object_member = True
        all_class_attribute_names = set()
        for x in inspect.getmro(base.__class__):
            all_class_attribute_names.update(x.__dict__.keys())

        if not self.source:
            unimplemented(
                gb_type="getattr with no source",
                context=f"tp_getattro_impl {self} {name}",
                explanation="Dynamo does not know how to access an attribute "
                "on an `nn.Module` instance that lacks a source. This is "
                "usually an internal error in Dynamo.",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )

        if name == "__dict__":
            return self.get_dict_vt(tx)

        subobj = None
        if name in base_dict:
            subobj = base_dict[name]
        elif (
            "_modules" in base_dict
            and name in base_dict["_modules"]
            and name not in all_class_attribute_names
        ):
            subobj = base_dict["_modules"][name]
        elif "_parameters" in base_dict and name in base_dict["_parameters"]:
            subobj = base_dict["_parameters"][name]
        elif "_buffers" in base_dict and name in base_dict["_buffers"]:
            subobj = base_dict["_buffers"][name]
        else:
            try:
                subobj = inspect.getattr_static(base, name)
                object_member = False
            except AttributeError:
                # see if we can fallback to __getattr__, which is not checked by getattr_static
                result = self._custom_getattr_fallback(
                    base=base, tx=tx, name=name, obj_source=self.source
                )
                if result is not None:
                    return result
                # if we can't find a __getattr__, we can't parse this, raise attribute error
                raise_observed_exception(
                    AttributeError,
                    tx,
                    args=[f"'{type(base).__name__}' object has no attribute '{name}'"],
                )

        if name == "forward":
            guard_to_detect_forward_monkeypatching(self.source, base)

        if name == "__class__" and not object_member:
            return VariableTracker.build(tx, base.__class__, source=source)

        if object_member:
            out = VariableTracker.build(tx, subobj, NNModuleSource(source))  # type: ignore[arg-type]

            if isinstance(out, (NNModuleVariable, UnspecializedNNModuleVariable)):
                # nn_module_stack source is BC surface area. Ensure that
                # mod._modules["linear"] is reflected as mod.linear for
                # nn_module_stack.
                out.set_nn_module_stack_source(
                    AttrSource(self.get_nn_module_stack_source(), name)
                )
            return out

        else:
            if istype(subobj, property):
                if self.source:
                    # Read the class attribute to reach the property
                    source = AttrSource(AttrSource(self.source, "__class__"), name)
                    # Get the getter function
                    source = AttrSource(source, "fget")
                return variables.UserFunctionVariable(
                    subobj.fget,  # pyrefly: ignore[bad-argument-type]
                    source=source,
                ).call_function(tx, [(self)], {})
            elif istype(subobj, classmethod):
                return variables.UserMethodVariable(
                    subobj.__func__,
                    variables.UserDefinedObjectVariable(type(base)),
                    source=source,
                )
            elif istype(subobj, staticmethod):
                return variables.UserFunctionVariable(
                    subobj.__get__(base),
                    source=source,
                )
            elif istype(subobj, types.FunctionType):
                return variables.UserMethodVariable(subobj, self, source=source)
            elif is_safe_constant(subobj) or istensor(subobj):
                # Support possibly common cases of class members
                return VariableTracker.build(tx, subobj, NNModuleSource(source))  # type: ignore[arg-type]
            else:
                unimplemented(
                    gb_type="Unsupported nn.Module attribute type",
                    context=f"nn.Module subclass: {typestr(base)}, name: {name}, attribute type: {typestr(subobj)}",
                    explanation=f"Dynamo does not support tracing nn.Module attributes of type `{typestr(subobj)}`",
                    hints=[
                        f"Refactor your code so that `{name}` (type `{typestr(subobj)}`) is not an attribute of `{typestr(base)}`",
                        "Currently supported attribute types are methods, classmethods, staticmethods, "
                        "properties, constants, and tensors.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

        return super().tp_getattro_impl(tx, name)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        mod = tx.output.get_submodule(self.module_key)

        with record_nn_module_stack(
            self.module_key, self.get_nn_module_stack_source(), tx, mod
        ):
            is_lazy = is_lazy_module(mod)
            if (
                isinstance(mod, torch.nn.Sequential)
                and mod.__class__.forward is torch.nn.Sequential.forward
            ):
                if nnmodule_has_hooks(mod):
                    # We do not want to unroll sequential if it has hooks, since evaporating it
                    # will cause hooks to not fire!
                    # This terminates and restart the tracing process
                    self.convert_to_unspecialized(tx)

                # Unroll sequential
                if is_lazy:
                    raise AssertionError(
                        "Expected lazy sequential isn't a valid combination?"
                    )
                if kwargs:
                    raise_args_mismatch(
                        tx,
                        "torch.nn.Module.Sequential",
                        "0 kwargs",
                        f"{len(kwargs)} kwargs",
                    )
                (arg,) = args
                # TODO: Use named_children when it supports remove_duplicate=False.
                for child_name, submod in mod._modules.items():
                    tx.call_function(
                        tx.output.register_attr_or_module(
                            submod,
                            self.module_key,
                            child_name,
                            source=NNModuleSource(AttrSource(self.source, child_name)),  # type: ignore[arg-type]
                        ),
                        [arg],
                        {},
                    )
                    arg = tx.pop()
                return arg

            if is_lazy:
                # The module type will change after it is called
                if mod.cls_to_become is not None:
                    self.module_type = mod.cls_to_become  # type: ignore[assignment]

                # The pre-hook runs to initialize the module shapes, then deletes itself.  After this,
                # the module is more or less not lazy and can be treated as a normal module regardless of
                # is_allowed or other variations.
                initialize_lazy_module(tx, mod, args, kwargs)

            # If we are tracing the higher order op, we want Dynamo to step
            # inside the module call so that Dynamo can see the underlying
            # parameters and buffers and raise them as inputs to the graph.
            #
            # NB: torch.nn.utils.parametrize changes the class type of a
            # parametrized module such that its __module__ points to
            # "torch.nn.utils.parametrize".
            if (
                tx.output.is_root_tracer()
                and mod.__module__.startswith(("torch.nn.", "torch.ao."))
                and mod.__module__ != "torch.nn.utils.parametrize"
            ):
                if nnmodule_has_hooks(
                    mod, check_forward_hooks=True, check_backward_hooks=True
                ):
                    # End of fn, this bubbles up and restarts tracing.
                    self.convert_to_unspecialized(tx)

                from .builder import wrap_fx_proxy

                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_module",
                        self.module_key,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )
            else:
                if isinstance(mod, torch.fx.GraphModule):
                    # TODO: do we want to support __call__ for GM's?
                    # If so at least some changes are needed, we don't allow inlining
                    # the call_wrapped currently, and maybe other issues too
                    fn = mod.forward
                    fn_source = AttrSource(self.source, "forward")
                else:
                    fn = mod._call_impl
                    fn_source = AttrSource(self.source, "_call_impl")
                if istype(fn, types.MethodType):
                    fn = fn.__func__
                    fn_source = AttrSource(fn_source, "__func__")
                    args = [self] + list(args)
                else:
                    if not istype(fn, types.FunctionType):
                        raise AssertionError(f"Expected FunctionType, got {type(fn)}")
                return tx.inline_user_function_return(
                    variables.UserFunctionVariable(fn, source=fn_source),
                    args,
                    kwargs,
                    allow_nested_graph_breaks=True,
                )

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: "VariableTracker",
    ) -> "VariableTracker":
        # nn.Module containers (ModuleList/Dict/Sequential/ParameterDict/ParameterList)
        # These are Python-level __getitem__, not CPython C slots.
        from .lists import SliceVariable
        from .tensor import SymNodeVariable

        module = tx.output.get_submodule(self.module_key)

        builtin_supported = (
            torch.nn.ModuleDict.__getitem__,
            torch.nn.ModuleList.__getitem__,
            torch.nn.ParameterDict.__getitem__,
            torch.nn.ParameterList.__getitem__,
            torch.nn.Sequential.__getitem__,
        )
        # pyrefly: ignore[missing-attribute]
        if type(module).__getitem__ not in builtin_supported:
            if not (
                key.is_python_constant()
                and isinstance(key.as_python_constant(), (str, int))
            ):
                unimplemented(
                    gb_type="Invalid or non-const argument in nn.Module __getitem__",
                    context=f"mp_subscript_impl: {self} {key}",
                    explanation="Dynamo does not support calling "
                    f"method `__getitem__` of ``nn.Module`` {module} with a non-constant or non-(str, int) key.",
                    hints=["Use constant arguments of type str or int for __getitem__"],
                )
            fn = module.__getitem__.__func__  # pyrefly: ignore[missing-attribute]

            if not isinstance(fn, types.FunctionType):
                raise AssertionError(f"Expected FunctionType, got {type(fn)}")

            src = AttrSource(AttrSource(self.source, "__getitem__"), "__func__")  # type: ignore[arg-type]
            return tx.inline_user_function_return(
                variables.UserFunctionVariable(fn, source=src),
                [self, key],
                {},
                allow_nested_graph_breaks=True,
            )

        if isinstance(key, SliceVariable):
            # TODO(anijain2305,export-team) - Remove this if condition when inlining of inbuilt nn modules is
            # enabled for export.
            if tx.output.export:
                result = []
                keys = list(range(len(module)))[key.as_python_constant()]  # type: ignore[arg-type]
                for idx, submod in enumerate(module[key.as_python_constant()]):  # type: ignore[arg-type]
                    k = keys[idx]
                    src = NNModuleSource(GetItemSource(self.source, k))
                    result.append(
                        tx.output.register_attr_or_module(
                            submod,
                            k,
                            source=src,
                        )
                    )

                new_module = module[key.as_python_constant()]  # type: ignore[index]
                new_module_variable = tx.output.register_attr_or_module(
                    new_module,
                    f"{self}.__getitem__(slice)",
                    source=NNModuleSource(
                        GetItemSource(self.source, key.as_python_constant())
                    ),
                )
                return new_module_variable
            else:
                # slice on nn module results in a creation of new module instance, so we need to make it sourceless.
                # Convert to unspecialized so that UnspecializedNNModule variable can take care of it.
                self.convert_to_unspecialized(tx)

        key_value = 0
        if isinstance(key, SymNodeVariable):
            key_value = key.evaluate_expr(tx.output)
        elif key.is_python_constant():
            key_value = key.as_python_constant()
        else:
            unimplemented(
                gb_type="Unsupported key type for nn.Module.__getitem__",
                context=f"mp_subscript_impl: {self} {key}",
                explanation="Dynamo does not support getitem on "
                "`nn.Module` with non-constant key.",
                hints=[],
            )

        submod = module[key_value]  # type: ignore[index]
        return tx.output.register_attr_or_module(
            submod,
            self.module_key,
            key_value,
            source=NNModuleSource(GetItemSource(self.source, key_value)),
        )

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from . import ListIteratorVariable

        return ListIteratorVariable(
            self.unpack_var_sequence(tx), mutation_type=ValueMutationNew()
        )

    @staticmethod
    def _gen_source(source: Source, name: str) -> Source:
        name_split = name.split(".")
        if name_split[0] == "":
            return source
        while len(name_split) > 0:
            x = name_split.pop(0)
            source = AttrSource(source, x)
        return source

    def _generic_call_method_helper(
        self, tx: "InstructionTranslatorBase", name: str, args: Any, kwargs: Any
    ) -> VariableTracker:
        # Put a `call_method` node in the FX graph, with the nn.Module as the
        # first arg.
        module = tx.output.get_submodule(self.module_key)
        mod_proxy = tx.output.create_proxy("get_attr", self.module_key, (), {})
        set_example_value(mod_proxy.node, module)
        proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_method", name, args=(mod_proxy, *proxy_args), kwargs=proxy_kwargs
            ),
        )

    def _assert_all_args_kwargs_const(
        self, tx: "InstructionTranslatorBase", name: str, args: Any, kwargs: Any
    ) -> None:
        if not all(
            x.is_python_constant() for x in itertools.chain(args, kwargs.values())
        ):
            module = tx.output.get_submodule(self.module_key)
            unimplemented(
                gb_type="non-const argument in nn.Module method",
                context=f"call_method: {self} {name} {args} {kwargs}",
                explanation="Dynamo does not support calling "
                f"method `{name}` of ``nn.Module`` {module} with non-constant arguments.",
                hints=[],
            )

    def _get_kwargs(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: Any,
        kwargs: Any,
        *names: str,
    ) -> dict[str, Any]:
        self._assert_all_args_kwargs_const(tx, name, args, kwargs)
        module = tx.output.get_submodule(self.module_key)
        fn = getattr(module, name)
        bound_args = inspect.signature(fn).bind(
            *([x.as_python_constant() for x in args]),
            **{k: v.as_python_constant() for k, v in kwargs.items()},
        )
        bound_args.apply_defaults()
        bound_args = bound_args.arguments
        return {k: bound_args[k] for k in names}

    def _wrap_values(
        self, tx: "InstructionTranslatorBase", items: Iterable[tuple[Any, Any]]
    ) -> "variables.ListIteratorVariable":
        result: list[VariableTracker] = []
        for name, submod in items:
            result.append(
                tx.output.register_attr_or_module(
                    submod,
                    self.module_key,
                    name,
                    source=NNModuleSource(self._gen_source(self.source, name)),
                )
            )
        return variables.ListIteratorVariable(result, mutation_type=ValueMutationNew())

    def _named_embed(
        self, tx: "InstructionTranslatorBase", name: str, obj: Any
    ) -> "variables.TupleVariable":
        return variables.TupleVariable(
            [
                VariableTracker.build(tx, name),
                tx.output.register_attr_or_module(
                    obj,
                    self.module_key,
                    name,
                    source=NNModuleSource(self._gen_source(self.source, name)),
                ),
            ]
        )

    def _call_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Example: `self.layer.__call__(x)`. Dynamo inlines `__call__`,
        # including hooks.
        return self.call_function(tx, args, kwargs)

    def forward(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Example: `self.layer.forward(x)`. Dynamo puts a `call_method` node in
        # the FX graph and does not trigger hooks.
        module = tx.output.get_submodule(self.module_key)
        with record_nn_module_stack(
            self.module_key, self.get_nn_module_stack_source(), tx, module
        ):
            return self._generic_call_method_helper(tx, "forward", args, kwargs)

    def _get_item_by_idx(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        if not args[1].is_python_constant():
            raise_type_error(
                tx,
                f"``nn.Module`` {module}'s call method _get_item_by_idx requires a constant index argument",
            )
        if not isinstance(args[0], variables.TupleVariable):
            raise_type_error(
                tx,
                f"``nn.Module`` {module}'s call method _get_item_by_idx requires a tuple as first argument",
            )
        mod_var = args[0].items[args[1].as_python_constant()]
        if isinstance(mod_var, UnspecializedNNModuleVariable):
            return mod_var
        key = mod_var.module_key
        submod = tx.output.get_submodule(key)
        return tx.output.register_attr_or_module(
            submod, key, key, source=NNModuleSource(GetItemSource(self.source, key))
        )

    def named_children(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        tx.output.guard_on_key_order.add(AttrSource(self.source, "_modules"))
        if args or kwargs:
            raise_args_mismatch(
                tx,
                "named_children",
                "0 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        result: list[VariableTracker] = []
        for n, submod in module.named_children():
            result.append(self._named_embed(tx, n, submod))
        return variables.ListIteratorVariable(result, mutation_type=ValueMutationNew())

    def named_parameters(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        tx.output.guard_on_key_order.add(AttrSource(self.source, "_parameters"))
        kw = self._get_kwargs(tx, "named_parameters", args, kwargs, "prefix", "recurse")
        result: list[VariableTracker] = []
        for n, param in module.named_parameters(**kw):
            result.append(self._named_embed(tx, n, param))
        return variables.ListIteratorVariable(result, mutation_type=ValueMutationNew())

    def named_buffers(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        tx.output.guard_on_key_order.add(AttrSource(self.source, "_buffers"))
        kw = self._get_kwargs(
            tx, "named_buffers", args, kwargs, "prefix", "recurse", "remove_duplicate"
        )
        result: list[VariableTracker] = []
        for n, buffer in module.named_buffers(**kw):
            result.append(self._named_embed(tx, n, buffer))
        return variables.ListIteratorVariable(result, mutation_type=ValueMutationNew())

    def named_modules(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        tx.output.guard_on_key_order.add(AttrSource(self.source, "_modules"))
        kw = self._get_kwargs(
            tx, "named_modules", args, kwargs, "memo", "prefix", "remove_duplicate"
        )
        result: list[VariableTracker] = []
        for n, submod in module.named_modules(**kw):
            result.append(self._named_embed(tx, n, submod))
        return variables.ListIteratorVariable(result, mutation_type=ValueMutationNew())

    def children(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        tx.output.guard_on_key_order.add(AttrSource(self.source, "_modules"))
        if args or kwargs:
            raise_args_mismatch(
                tx,
                "children",
                "0 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        return self._wrap_values(tx, module.named_children())

    def modules(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        tx.output.guard_on_key_order.add(AttrSource(self.source, "_modules"))
        return self._wrap_values(tx, module.named_modules())

    def parameters(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        tx.output.guard_on_key_order.add(AttrSource(self.source, "_parameters"))
        kw = self._get_kwargs(tx, "parameters", args, kwargs, "recurse")
        return self._wrap_values(tx, module.named_parameters(**kw))

    def buffers(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        tx.output.guard_on_key_order.add(AttrSource(self.source, "_buffers"))
        kw = self._get_kwargs(tx, "buffers", args, kwargs, "recurse")
        return self._wrap_values(tx, module.named_buffers(**kw))

    def keys(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        if args or kwargs:
            raise_args_mismatch(
                tx,
                "keys",
                "0 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        result: list[VariableTracker] = []
        # pyrefly: ignore[not-iterable]
        for tmp in module:
            result.append(VariableTracker.build(tx, tmp))
        return variables.ListIteratorVariable(result, mutation_type=ValueMutationNew())

    def values(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        if args or kwargs:
            raise_args_mismatch(
                tx,
                "values",
                "0 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        return self._wrap_values(tx, module.items())  # type: ignore[operator]

    def items(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        module = tx.output.get_submodule(self.module_key)
        if args or kwargs:
            raise_args_mismatch(
                tx,
                "items",
                "0 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        result: list[VariableTracker] = []
        for n, submod in module.items():  # type: ignore[operator]
            result.append(self._named_embed(tx, n, submod))
        return variables.ListIteratorVariable(result, mutation_type=ValueMutationNew())

    tp_methods = {
        "_call_impl": Method(_call_impl),
        "_wrapped_call_impl": Method(_call_impl),
        "forward": Method(forward),
        "_get_item_by_idx": Method(_get_item_by_idx),
        "named_children": Method(named_children),
        "named_parameters": Method(named_parameters),
        "named_buffers": Method(named_buffers),
        "named_modules": Method(named_modules),
        "children": Method(children),
        "modules": Method(modules),
        "parameters": Method(parameters),
        "buffers": Method(buffers),
        "keys": Method(keys),
        "values": Method(values),
        "items": Method(items),
    }

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
        constant: bool = False,
    ) -> VariableTracker:
        from .constant import ConstantVariable

        module = tx.output.get_submodule(self.module_key)

        if name == "_check_input_dim" and trace_rules.is_torch_inline_allowed(
            inspect.getfile(module.__class__._check_input_dim)  # type: ignore[union-attr]
        ):
            return ConstantVariable.create(True)

        if constant:
            fn = getattr(module, name)
            const_name = f"{module.__class__.__name__}_{name}_result"
            return invoke_and_store_as_constant(tx, fn, const_name, args, kwargs)

        if name == "__iter__":
            return self.tp_iter_impl(tx)
        elif (
            name == "__contains__"
            and isinstance(module, (torch.nn.ModuleDict, torch.nn.ParameterDict))
            and args
            and args[0].is_python_constant()
        ):
            return VariableTracker.build(
                tx, args[0].as_python_constant() in module._modules
            )
        elif (
            name == "_get_abs_string_index"
            or (
                isinstance(module, torch.nn.modules.conv._ConvNd)
                and name == "_conv_forward"
            )
            or (
                isinstance(module, torch.nn.modules.conv._ConvTransposeNd)
                and name == "_output_padding"
            )
        ):
            # Inline the function
            fn = getattr(module, name).__func__
            fn_source = AttrSource(AttrSource(self.source, name), "__func__")  # type: ignore[arg-type]
            return tx.inline_user_function_return(
                variables.UserFunctionVariable(fn, source=fn_source),
                [self] + list(args),
                kwargs,
            )
        # A loose heuristic, but seems to be generally good before we drop into
        # the manual handling of inputs. Skip tp_methods names so they reach the
        # declarative dispatch in the base call_method.
        elif (
            name not in self.tp_methods
            and name in module.__class__.__dict__
            and callable(module.__class__.__dict__[name])
            and all(x.is_tensor() for x in itertools.chain(args, kwargs.values()))
        ):
            return self._generic_call_method_helper(tx, name, args, kwargs)
        else:
            return super().call_method(tx, name, list(args), kwargs)

    def sq_length_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        """Sequence length for container modules (e.g., nn.Sequential)."""
        module = tx.output.get_submodule(self.module_key)
        return VariableTracker.build(tx, len(module))  # type: ignore[arg-type]


class UnspecializedNNModuleVariable(UserDefinedObjectVariable):
    _nonvar_fields = {
        "value_type",
        "is_state_mutated",
        "nn_module_stack_source",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    """
    The above class will specialize on the id() of a module and place
    parameters on the torch.fx.GraphModule.  Giving one graph per
    module instance.  This version treats nn.Modules() like other user
    defined objects and will pass parameters into the FX graph as inputs.
    Giving one graph per module class.
    """

    def __init__(self, value: torch.nn.Module, **kwargs: Any) -> None:
        if "value_type" in kwargs:
            lazy_value_to_become = getattr(kwargs["value_type"], "cls_to_become", None)
            if type(value) is lazy_value_to_become:
                # We may have cloned a variabletracker for a LazyModule earlier (e.g. tracking side-effects)
                # and then later we called and mutated the LazyModule into a MaterializedModule.
                # We do not do the mutation upon first seeing a LazyModule since we preserve eager semantics to only
                # mutate upon first call, but this requires we update multiple copies of the VariableTracker post-mutation.
                kwargs["value_type"] = type(value)

        super().__init__(value=value, **kwargs)
        self.is_state_mutated = False
        # nn_module_stack_source is used to ensure BC for nn_module_stack.
        # Downstream users prefer mod.linear instead of mod._modules['linear']
        # as the module stack. When Dynamo inlines the __getattr__ method, we
        # cannot use self.source for nn_module_stack because it will be similar
        # to mod._modules['linear']. In these cases, we set the
        # nn_module_stack_source appropriately to resemble mod.linear.
        self.nn_module_stack_source = self.source

    def _wrap_source(self, attr_source: Source) -> Source:
        # the vt is already wrapped with UnspecializedNNModuleSource
        return attr_source

    def get_nn_module_stack_source(self) -> Source:
        res = self.nn_module_stack_source or self.source
        if not res:
            raise AssertionError("nn_module_stack_source must not be None")
        return res

    def set_nn_module_stack_source(self, source: Source) -> None:
        self.nn_module_stack_source = source

    @staticmethod
    @functools.cache
    def _nn_module_method_ids() -> set[int]:
        # Allow __setattr__ to fall through to base class handler
        supported = {
            torch.nn.Module.__setattr__,
            torch.nn.Module.__init__,
            torch.nn.Module.__delattr__,
        }
        return {
            id(x.__code__)
            for x in torch.nn.Module.__dict__.values()
            if hasattr(x, "__code__") and x not in supported
        }

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        try:
            fn = inspect.getattr_static(self.value_type, "__iter__")
        except AttributeError as e:
            raise NotImplementedError from e

        if fn in (
            torch.nn.ModuleList.__iter__,
            torch.nn.ParameterList.__iter__,
            torch.nn.Sequential.__iter__,
        ):
            # The program can mutate the nn module object but the saved `value`
            # will not reflect the mutations. So, trace through the `__iter__`
            # function to reflect any tracked mutations.
            return tx.inline_user_function_return(
                VariableTracker.build(tx, fn),
                [
                    self,
                ],
                {},
            ).unpack_var_sequence(tx)

        return super().unpack_var_sequence(tx)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        mod = self.value
        # see comment on lazy module handling in NNModuleVariable.call_function for context
        if is_lazy_module(mod):  # type: ignore[arg-type]
            if mod.cls_to_become is not None:  # type: ignore[attr-defined]
                self.value_type = mod.cls_to_become  # type: ignore[attr-defined,assignment]
            initialize_lazy_module(tx, mod, args, kwargs)  # type: ignore[arg-type]

        if not isinstance(mod, torch.fx.GraphModule):
            name = "__call__"
            fn = getattr(self.value_type, name)
        else:
            name = "_call_impl"
            fn = getattr(self.value_type, name)

        # Check if we can short circuit nn.Module._call_impl to the forward
        # method.  NB - This is done to reduce the compile time of Dynamo.
        if (
            istype(mod.__call__, types.MethodType)  # type: ignore[operator]
            and istype(mod._call_impl, types.MethodType)  # type: ignore[attr-defined]
            and mod.__call__.__func__ is unpatched_nn_module_call  # type: ignore[operator]
            and mod._call_impl.__func__ is unpatched_nn_module_call_impl  # type: ignore[attr-defined]
            # Consult pending STORE_ATTR side effects too. During tracing the
            # patched forward may not be visible in mod.__dict__ yet.
            and not self.has_key_in_generic_dict(tx, "forward")
        ):
            forward_method = inspect.getattr_static(mod, "forward")
            if isinstance(forward_method, types.FunctionType):
                globals_vt = tx.nn_modules_globals_vt

                def _hooks_dict_len(obj: VariableTracker, attr: str) -> int:
                    vt = obj.tp_getattro_impl(tx, attr)
                    vt = vt.realize() if hasattr(vt, "realize") else vt
                    return vt.len()  # type: ignore[union-attr]

                has_hooks = any(
                    _hooks_dict_len(self, attr)
                    for attr in (
                        "_backward_hooks",
                        "_backward_pre_hooks",
                        "_forward_hooks",
                        "_forward_hooks_with_kwargs",
                        "_forward_pre_hooks",
                        "_forward_pre_hooks_with_kwargs",
                    )
                ) or any(
                    _hooks_dict_len(globals_vt, attr)
                    for attr in (
                        "_global_backward_pre_hooks",
                        "_global_backward_hooks",
                        "_global_forward_hooks",
                        "_global_forward_pre_hooks",
                    )
                )

                if not has_hooks:
                    name = "forward"
                    fn = self.value_type.forward  # type: ignore[attr-defined]

        if self.source:
            source = self.get_source_by_walking_mro(tx, name)
        else:
            source = None

        guard_to_detect_forward_monkeypatching(self.source, mod)  # type: ignore[arg-type]

        ctx = (
            record_nn_module_stack(
                str(id(mod)),
                self.get_nn_module_stack_source(),
                tx,
                mod,  # type: ignore[arg-type]
            )
            if self.source
            else nullcontext()
        )
        with ctx:
            if not isinstance(fn, (types.FunctionType, torch.jit.ScriptFunction)):
                fn_vt = VariableTracker.build(tx, fn, source=source, realize=True)
                return fn_vt.call_function(tx, [self] + list(args), kwargs)
            else:
                # Ideally we would have just used VariableTracker.build(tx, fn,
                # source=source) but that introduces guard on the
                # `forward.__code__` object. Given that we already guard on the
                # forward not present in generic dict, we don't need this guard.
                return variables.UserFunctionVariable(fn, source=source).call_function(
                    tx, [self] + list(args), kwargs
                )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in ["_call_impl", "_wrapped_call_impl"]:
            fn = getattr(self.value_type, name)
            if self.source:
                source = self.get_source_by_walking_mro(tx, name)
            else:
                source = None

            fn_vt = VariableTracker.build(tx, fn, source=source, realize=True)
            return fn_vt.call_function(tx, [self] + list(args), kwargs)

        if not self.has_key_in_generic_dict(tx, name):
            try:
                method = inspect.getattr_static(type(self.value), name)
            except AttributeError:
                method = None

            if isinstance(method, staticmethod):
                source = AttrSource(
                    self.get_source_by_walking_mro(tx, name), "__func__"
                )
                fn_vt = VariableTracker.build(
                    tx, method.__func__, source=source, realize=True
                )
                return fn_vt.call_function(tx, args, kwargs)

            if (
                hasattr(method, "__code__")
                and id(method.__code__) in self._nn_module_method_ids()
            ):
                unimplemented(
                    gb_type="UnspecializedNNModuleVariable missing method",
                    context=f"call_method: {self} {name} {args} {kwargs}",
                    explanation=f"Dynamo does not support tracing method {name} of nn.Module {self.value}",
                    hints=[
                        "Dynamo does not really define unspecialized nn.Module very well.",
                        *graph_break_hints.DIFFICULT,
                    ],
                )

            # "_parameters" in self.value.__dict__ checks that module is initialized
            if name == "__setattr__" and "_parameters" in self.value.__dict__:
                # Record if mutations happens on parameters/buffers/modules. The
                # mutations on these are not tracked by base class
                # UserDefinedObject vt. This will be used later to graph break
                # on seeing a parameters() and family calls.
                # TODO(anijain2305) - This might not be needed if we let Dynamo
                # inline both getattr and setattr. In that case, it should see
                # the lowest level dicts - _parameters and family and
                # automatically track mutations on those. Investigate if that
                # can be done.
                attr_name = args[0].as_python_constant()
                value = args[1]

                # This is reverse engineered by looking at nn module __setattr__
                # logic.
                if (
                    value.is_tensor() and value.python_type() is torch.nn.Parameter
                ) or attr_name in self.value.__dict__["_parameters"]:
                    # Handle parameters
                    self.is_state_mutated = True
                elif attr_name in self.value.__dict__["_buffers"]:
                    # Handle buffers
                    self.is_state_mutated = True
                elif (
                    isinstance(
                        value,
                        (
                            variables.NNModuleVariable,
                            variables.UnspecializedNNModuleVariable,
                        ),
                    )
                    or attr_name in self.value.__dict__["_modules"]
                ):
                    # Handle submodules
                    self.is_state_mutated = True

            if (
                method is torch.nn.Module.__setattr__
                and isinstance(args[1], variables.DeletedVariable)
            ) or method is torch.nn.Module.__delattr__:
                # Trace through __delattr__ to track mutations on the module
                # members like `_modules``.
                fn_vt = VariableTracker.build(tx, torch.nn.Module.__delattr__)
                return fn_vt.call_function(tx, [self, args[0]], kwargs)

        return super().call_method(tx, name, list(args), kwargs)

    def getattr_helper(
        self, tx: "InstructionTranslatorBase", field: str, name_vt: VariableTracker
    ) -> VariableTracker | None:
        dict_vt = self.tp_getattro_impl(tx, field)
        if isinstance(dict_vt, variables.UserDefinedDictVariable):
            dict_vt = dict_vt._base_vt
        if isinstance(dict_vt, variables.ConstDictVariable):
            return dict_vt.maybe_getitem_const(name_vt)
        return None

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if (
            tx.output.side_effects.is_attribute_mutation(self)
            and name
            in (
                "named_parameters",
                "parameters",
                "named_buffers",
                "buffers",
                "named_modules",
                "modules",
            )
            and self.is_state_mutated
            and tx.output.side_effects.has_pending_mutation(self)
        ):
            unimplemented(
                gb_type="getattr() on nn.Module with pending mutation",
                context=f"getattr({self}, {name})",
                explanation="Intentionally graph breaking on getattr() on a nn.Module "
                "with a pending mutation",
                hints=[],
            )

        # Allow skipping of empty hook dict guards on inbuilt nn modules
        if name in (
            "_backward_hooks",
            "_backward_pre_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_forward_pre_hooks",
        ):
            # For empty hooks, make an EMPTY_NN_MODULE_HOOKS_DICT. This allows us to control the installation of empty
            # hooks guard via skip_nnmodule_hook_guards
            if not tx.output.side_effects.has_pending_mutation_of_attr(self, name):
                hooks_dict = getattr(self.value, name)
                if isinstance(hooks_dict, dict) and len(hooks_dict) == 0:
                    hooks_source = (
                        AttrSource(self.source, name) if self.source else None
                    )
                    if hooks_source:
                        install_guard(
                            hooks_source.make_guard(
                                GuardBuilder.EMPTY_NN_MODULE_HOOKS_DICT
                            )
                        )
                    hooks_vt_cls = (
                        variables.OrderedDictVariable
                        if isinstance(hooks_dict, collections.OrderedDict)
                        else variables.ConstDictVariable
                    )
                    return hooks_vt_cls({}, source=hooks_source)

        # For non-empty hook dicts, one way is to just fallback to VariableTracker.build() and create a ConstDictVariable.
        # However, ConstDictVariable guards on keys. This can cause recompiles when the same hook is installed for
        # different nn module instances, because the key keeps changing (look more into RemovableHandle to understand why
        # key changes - also related https://github.com/pytorch/pytorch/issues/125836). Here, we carefully craft a
        # NNModuleHooksDictVariable (a subclass of ConstDictVariable) to avoid any guard on the keys.
        if (
            self.source
            and name
            in (
                "_forward_pre_hooks_with_kwargs",
                "_forward_pre_hooks",
                "_forward_hooks_with_kwargs",
                "_forward_hooks",
            )
            and not tx.output.side_effects.has_pending_mutation_of_attr(self, name)
        ):
            hooks_dict = getattr(self.value, name)
            hooks_dict_source = AttrSource(self.source, name)
            install_guard(hooks_dict_source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            tx.output.guard_on_key_order.add(hooks_dict_source)

            def build_key_value(
                i: int, k: Any, v: Any
            ) -> tuple[VariableTracker, VariableTracker]:
                # Make key sourceless to avoid any guard on it
                key = VariableTracker.build(tx, k)

                # Instead of using dict[key] to access the value, use a dict[dict.keys()[index]] to access the
                # value. This removes the reliance on the actual key value.
                source_key = ConstDictKeySource(hooks_dict_source, i)
                source_value = DictGetItemSource(hooks_dict_source, source_key)
                value = LazyVariableTracker.create(v, source_value, tx=tx)
                return key, value

            result = dict(
                build_key_value(i, k, v)
                for i, k, v in enumerate_items_with_dict_position(hooks_dict)
            )

            return variables.NNModuleHooksDictVariable(result, source=hooks_dict_source)
        return super().tp_getattro_impl(tx, name)

    def manually_trace_nn_module_getattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        """
        Dynamo tracing of nn.Module __getattr__ can be expensive if the model
        has deep submodule hierarchy. Since the __getattr__ is stable, we can
        directly look into the underlying datastructures. This saves a lot of
        compilation time.
        """
        from .builder import SourcelessBuilder

        name_vt = SourcelessBuilder.create(tx, name)
        out = self.getattr_helper(tx, "_parameters", name_vt)
        if out is None:
            out = self.getattr_helper(tx, "_modules", name_vt)
        if out is None:
            out = self.getattr_helper(tx, "_buffers", name_vt)
        if out is None:
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(self.value).__name__}' object has no attribute '{name}'"
                ],
            )
        if out is None:
            raise AssertionError(
                f"manually_trace_nn_module_getattr failed to find attribute '{name}'"
            )
        return out


class UnspecializedBuiltinNNModuleVariable(UnspecializedNNModuleVariable):
    """
    Differentiates between builtin nn modules (e.g. torch.nn.Linear) and user defined nn modules.
    """

    def _wrap_source(self, attr_source: Source) -> Source:
        # vt is already wrapped with the UnspecializedBuiltinNNModuleSource
        return attr_source


class FSDPManagedNNModuleVariable(UnspecializedNNModuleVariable):
    """
    Tracing behavior: trace into submodules and treat them as Unspecialized, do not
    register parameters to the top-level, treat them as function inputs.

    Guards behavior: if 'skip_fsdp_guards', many guards that would be installed
    by a vanilla UnspecializedNNModuleVariable are simply dropped, on the basis
    that a user wrapping their model in FSDP(model) is already opting into a
    requirement to not modify internal model state, which would already break FSDP without
    compilation.
    """

    def __init__(self, value: torch.nn.Module, **kwargs: Any) -> None:
        source = kwargs.get("source")
        if source is None:
            raise AssertionError(
                "FSDPManagedNNModule depends on having an accurate source to control guarding."
            )

        super().__init__(value=value, **kwargs)
        self.source = source

    def _wrap_source(self, attr_source: Any) -> Any:
        if not isinstance(
            attr_source, (FSDPNNModuleSource, UnspecializedNNModuleSource)
        ):
            if torch._dynamo.config.skip_fsdp_guards:
                return FSDPNNModuleSource(attr_source)
            else:
                return UnspecializedNNModuleSource(attr_source)
        return attr_source
<|endoftext|>"""
Dictionary-related variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for different types of dictionary-like objects:
- Regular Python dictionaries (dict)
- Ordered dictionaries (collections.OrderedDict)
- Default dictionaries (collections.defaultdict)
- Dictionary views (keys and values)

These classes are responsible for tracking dictionary operations during graph compilation,
maintaining proper guards for dictionary mutations and key existence checks. They handle
dictionary creation, modification, key/value access, and view operations while ensuring
correct behavior in the compiled code through appropriate guard installation.

The implementation uses a special HashableTracker wrapper to handle
dictionary keys while preserving proper aliasing semantics. Set-related classes live
in sets.py.
"""

import collections
import functools
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any, cast, TYPE_CHECKING, Union

from torch.utils._pytree import MappingKey

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import (
    create_call_function,
    create_call_method,
    create_dup_top,
    create_instruction,
)
from ..exc import raise_observed_exception, raise_type_error, unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    DictGetItemSource,
    is_constant_source,
    is_from_local_source,
)
from ..utils import (
    _item_debug_repr,
    cmp_name_to_op_mapping,
    dict_items,
    dict_keys,
    dict_values,
    istype,
    raise_args_mismatch,
    tracked_repr,
    unpack_iterable,
)
from .base import (
    AttributeMutationExisting,
    AttributeMutationNew,
    AttrMutationKind,
    Method,
    NO_SUCH_SUBOBJ,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable
from .hashable import HashableTracker, is_hashable, raise_unhashable
from .object_protocol import (
    _is_method_type,
    generic_getitem,
    generic_richcompare_bool,
    mro_lookup,
)


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from .functions import UserFunctionVariable


# [Adding a new supported class within the keys of ConstDictVariable]
# - Implement hash_impl(self, tx) on the VariableTracker subclass (or rely on the
#   base class default which uses get_real_python_backed_value())
# - Implement tp_richcompare_impl() for key equality


def pydict_check(obj: VariableTracker) -> bool:
    # This is a simplified version of the CPython's PyDict_Check function:
    return issubclass(obj.python_type(), dict)


def _is_set_or_dictview(obj: VariableTracker) -> bool:
    """PyAnySet_Check(other) || PyDictViewSet_Check(other)"""
    try:
        t = obj.python_type()
    except NotImplementedError:
        return False
    return issubclass(t, (set, frozenset, dict_keys, dict_items))


class ConstDictVariable(VariableTracker):
    # PyDict_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4825
    _cpython_type = dict

    CONTAINS_GUARD = GuardBuilder.DICT_CONTAINS
    NOT_CONTAINS_GUARD = GuardBuilder.DICT_NOT_CONTAINS

    _nonvar_fields = {
        "_checking_python_constant",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        items: dict[VariableTracker, VariableTracker],
        **kwargs: Any,
    ) -> None:
        # .clone() pass these arguments in kwargs but they're recreated a few
        # lines below
        if "original_items" in kwargs:
            kwargs.pop("original_items")
        if "should_reconstruct_all" in kwargs:
            kwargs.pop("should_reconstruct_all")
        if "_checking_python_constant" in kwargs:
            kwargs.pop("_checking_python_constant")

        super().__init__(**kwargs)

        Hashable = HashableTracker

        # Keys will just be HashableTrackers when cloning, in any other case they'll be VariableTrackers
        if not all(
            isinstance(x, (VariableTracker, Hashable))
            and isinstance(v, VariableTracker)
            for x, v in items.items()
        ):
            raise AssertionError(
                "All keys must be VariableTracker or HashableTracker "
                "and all values must be VariableTracker"
            )

        def make_hashable(
            key: Union[VariableTracker, "HashableTracker"],
        ) -> "HashableTracker":
            return key if isinstance(key, Hashable) else Hashable(key)

        items_cls = cast(type, self._cpython_type)
        self.items = items_cls({make_hashable(x): v for x, v in items.items()})
        # need to reconstruct everything if the dictionary is an intermediate value
        # or if a pop/delitem was executed
        self.should_reconstruct_all = (
            not is_from_local_source(self.source) if self.source else True
        )
        self.original_items = items.copy()
        # Re-entrancy guard for is_python_constant against self-referential
        # dicts (d[k] = d, directly or via an OrderedDict/defaultdict wrapper
        # whose _base_vt is this instance). Both forms re-enter this same
        # instance's is_python_constant, so a per-instance flag suffices.
        self._checking_python_constant = False

    def as_proxy(self) -> dict[Any, Any]:
        return {k.vt.as_proxy(): v.as_proxy() for k, v in self.items.items()}

    def debug_repr(self) -> str:
        items: list[str] = []
        for k, v in self.items.items():
            key_str = _item_debug_repr(k.vt)
            val_str = _item_debug_repr(v)
            items.append(f"{key_str}: {val_str}")
        return "{" + ", ".join(items) + "}"

    def is_python_constant(self) -> bool:
        # Avoid the base implementation, which probes as_python_constant() and
        # thus rebuilds a real dict, re-hashing the keys (wrong for keys with a
        # side-effecting __hash__).  Check element constness directly instead.
        # Re-entrancy guard: a self-referential dict (d[k] = d, directly or via
        # an OrderedDict/defaultdict wrapper delegating to this _base_vt) would
        # otherwise recurse forever. A cyclic dict is not a flat python
        # constant; return False so reconstruct handles the cycle.
        if self._checking_python_constant:
            return False
        self._checking_python_constant = True
        try:
            return all(
                k.vt.is_python_constant() and v.is_python_constant()
                for k, v in self.items.items()
            )
        finally:
            self._checking_python_constant = False

    def as_python_constant(self) -> dict[Any, Any]:
        return {
            k.vt.as_python_constant(): v.as_python_constant()
            for k, v in self.items.items()
        }

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        items = [
            f"{tracked_repr(tx, key.vt)}: {tracked_repr(tx, value)}"
            for key, value in self.items.items()
        ]
        return VariableTracker.build(tx, "{" + ", ".join(items) + "}")

    def keys_as_python_constant(self) -> dict[Any, VariableTracker]:
        self.install_dict_keys_match_guard()
        return {k.vt.as_python_constant(): v for k, v in self.items.items()}

    def python_type(self) -> type:
        return cast(type, self._cpython_type)

    def __contains__(self, vt: VariableTracker) -> bool:
        if not isinstance(vt, VariableTracker):
            raise AssertionError(f"Expected VariableTracker, got {type(vt)}")
        # Use is_hashable as a side-effect-free pre-check.  We can't catch
        # ObservedTypeError from HashableTracker because it modifies
        # tx.exn_vt_stack as a side effect.
        if not is_hashable(vt):
            return False
        key = HashableTracker(vt)
        return key in self.items and not isinstance(
            self.items[key], variables.DeletedVariable
        )

    def call_tree_map_branch(
        self,
        tx: "InstructionTranslatorBase",
        tree_map_fn: "UserFunctionVariable",
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        other_dicts: list[ConstDictVariable] = []
        for candidate in rest:
            candidate = candidate.realize()
            if not isinstance(candidate, ConstDictVariable) or len(
                candidate.items
            ) != len(self.items):
                return self._tree_map_fallback(
                    tx, tree_map_fn, map_fn, rest, tree_map_kwargs
                )
            other_dicts.append(candidate)

        new_items_hashed = type(self.items)()
        for key_tracker, value in self.items.items():
            sibling_leaves: list[VariableTracker] = []
            for candidate in other_dicts:
                try:
                    sibling_leaves.append(candidate.items[key_tracker])
                except KeyError:
                    return self._tree_map_fallback(
                        tx, tree_map_fn, map_fn, rest, tree_map_kwargs
                    )
            new_items_hashed[key_tracker] = value.call_tree_map(
                tx,
                tree_map_fn,
                map_fn,
                sibling_leaves,
                tree_map_kwargs,
            )

        updated_original_items = {
            key_tracker.vt: new_items_hashed[key_tracker]
            for key_tracker in new_items_hashed
        }

        return self.clone(
            items=new_items_hashed,
            original_items=updated_original_items,
            should_reconstruct_all=True,
            source=None,
            mutation_type=ValueMutationNew(),
        )

    def call_tree_map_with_path_branch(
        self,
        tx: "InstructionTranslatorBase",
        tree_map_fn: "UserFunctionVariable",
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        other_dicts: list[ConstDictVariable] = []
        for candidate in rest:
            candidate = candidate.realize()
            if not isinstance(candidate, ConstDictVariable) or len(
                candidate.items
            ) != len(self.items):
                return self._tree_map_with_path_fallback(
                    tx, tree_map_fn, map_fn, rest, tree_map_kwargs, keypath
                )
            other_dicts.append(candidate)

        new_items_hashed = type(self.items)()
        for key_tracker, value in self.items.items():
            sibling_leaves: list[VariableTracker] = []
            for candidate in other_dicts:
                try:
                    sibling_leaves.append(candidate.items[key_tracker])
                except KeyError:
                    return self._tree_map_with_path_fallback(
                        tx, tree_map_fn, map_fn, rest, tree_map_kwargs, keypath
                    )
            key_const = key_tracker.vt.as_python_constant()
            child_keypath = keypath + (MappingKey(key_const),)
            new_items_hashed[key_tracker] = value.call_tree_map_with_path(
                tx,
                tree_map_fn,
                map_fn,
                sibling_leaves,
                tree_map_kwargs,
                child_keypath,
            )

        updated_original_items = {
            key_tracker.vt: new_items_hashed[key_tracker]
            for key_tracker in new_items_hashed
        }

        return self.clone(
            items=new_items_hashed,
            original_items=updated_original_items,
            should_reconstruct_all=True,
            source=None,
            mutation_type=ValueMutationNew(),
        )

    def len(self) -> int:
        return sum(
            not isinstance(x, variables.DeletedVariable) for x in self.items.values()
        )

    def has_new_items(self) -> bool:
        return self.should_reconstruct_all or any(
            self.is_new_item(self.original_items.get(key.vt), value)
            for key, value in self.items.items()
        )

    def is_new_item(
        self, value: VariableTracker | None, other: VariableTracker
    ) -> bool:
        # compare the id of the realized values if both values are not lazy VTs
        if value and value.is_realized() and other.is_realized():
            return id(value.realize()) != id(other.realize())
        return id(value) != id(other)

    def reconstruct_kvs_into_new_dict(self, codegen: "PyCodegen") -> None:
        # Build a dictionary that contains the keys and values.
        num_args = 0
        for key, value in self.items.items():
            # We can safely call realize() here as it won't introduce any new guards
            item = self.original_items.get(key.vt)
            if self.is_new_item(item, value) or self.should_reconstruct_all:
                codegen(key.vt)
                codegen(value)
                num_args += 1
        codegen.append_output(create_instruction("BUILD_MAP", arg=num_args))

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self._contains_self_reference():
            codegen.extend_output(
                [
                    create_instruction("BUILD_MAP", arg=0),
                    create_dup_top(),
                ]
            )
            codegen.add_cache(self)
            self.reconstruct_kvs_into_new_dict(codegen)
            codegen.append_output(create_instruction("DICT_UPDATE", arg=1))
        else:
            # Non-self-referential: use simple codegen
            self.reconstruct_kvs_into_new_dict(codegen)

    def reconstruct_pycode(self, codegen) -> str:
        if self._contains_self_reference():
            raise NotImplementedError
        return f"{{{', '.join(f'{k.vt.reconstruct_pycode(codegen)}: {v.reconstruct_pycode(codegen)}' for k, v in self.items.items())}}}"

    def getitem_const_raise_exception_if_absent(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        key = HashableTracker(arg)
        if key not in self.items:
            raise_observed_exception(KeyError, tx, args=[arg])
        return self.items[key]

    def getitem_const(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        key = HashableTracker(arg)
        if key not in self.items:
            msg = f"Dictionary key {arg.value} not found during tracing"  # type: ignore[attr-defined]
            unimplemented(
                gb_type="key not found in dict",
                context=f"Key {arg.value}",  # type: ignore[attr-defined]
                explanation=msg,
                hints=[
                    "Check if the key exists in the dictionary before accessing it.",
                    *graph_break_hints.USER_ERROR,
                ],
            )
        return self.items[key]

    def maybe_getitem_const(self, arg: VariableTracker) -> VariableTracker | None:
        key = HashableTracker(arg)
        if key not in self.items:
            return None
        return self.items[key]

    def realize_key_vt(self, arg: VariableTracker) -> None:
        # Realize the LazyVT on a particular index
        if arg not in self:
            raise AssertionError(f"Key {arg} not found in dict")
        key = HashableTracker(arg)
        index = tuple(self.items.keys()).index(key)
        original_key_vt = tuple(self.original_items.keys())[index]
        if isinstance(original_key_vt, variables.LazyVariableTracker):
            original_key_vt.realize()

    def install_dict_keys_match_guard(self) -> None:
        if self.source:
            install_guard(self.make_guard(GuardBuilder.DICT_KEYS_MATCH))

    def install_dict_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        # Key guarding - These are the cases to consider
        # 1) The dict has been mutated. In this case, we would have already
        # inserted a DICT_KEYS_MATCH guard, so we can skip.
        #
        # 2) args[0].source is None. This happens for const keys. Here, we
        # have to insert the DICT_CONTAINS guard.
        #
        # 3) args[0].source is not None. This can happen for non-const VTs.
        #   3a) contains=True. In this case, we can access the lazyVT from
        #   original_items and selectively realize it.
        #   3b) contains=False. There is no easy way to selectively apply this
        #   DICT_NOT_CONTAINS guard because our guard are represented via trees.
        #   Be conservative and add DICT_KEYS_MATCH guard.

        if not self.source:
            return

        if tx.output.side_effects.is_modified(self):
            return

        contains = args[0] in self
        if args[0].source is None and args[0].is_python_constant():
            guard_fn = (
                type(self).CONTAINS_GUARD if contains else type(self).NOT_CONTAINS_GUARD
            )
            install_guard(
                self.make_guard(
                    functools.partial(
                        guard_fn,
                        key=args[0].as_python_constant(),
                    )
                )
            )
        elif args[0].source:
            if contains:
                self.realize_key_vt(args[0])
            else:
                self.install_dict_keys_match_guard()

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # dict_subscript: https://github.com/python/cpython/blob/62a6e898e01/Objects/dictobject.c#L3673-L3706
        # Unhashable key check happens inside HashableTracker (hash_impl → TypeError).
        return self.getitem_const_raise_exception_if_absent(tx, key)

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4657-L4668
        if not is_hashable(item):
            raise_type_error(tx, f"unhashable type: '{item.python_type_name()}'")
        self.install_dict_contains_guard(tx, [item])
        contains = item in self
        return VariableTracker.build(tx, contains)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import DictIterator

        if self.source and not is_constant_source(self.source):
            install_guard(self.make_guard(GuardBuilder.DICT_KEYS_MATCH))
            tx.output.guard_on_key_order.add(self.source)
        return DictIterator(self.items.keys())

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from . import DictBuiltinVariable

        temp_dict_vt = DictBuiltinVariable.call_custom_dict(tx, dict, *args, **kwargs)
        tx.output.side_effects.mutation(self)
        self.items.update(temp_dict_vt.items)  # type: ignore[attr-defined]
        return ConstantVariable.create(None)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # NB - Both key and value are LazyVariableTrackers in the beginning. So,
        # we have to insert guards when a dict method is accessed. For this to
        # be simple, we are conservative and overguard. We skip guard only for
        # get/__getitem__ because the key guard will be inserted by the
        # corresponding value VT. For __contains__, we add a DICT_CONTAINS
        # guard. But for all the other methods, we insert the DICT_KEYS_MATCH
        # guard to be conservative.
        Hashable = HashableTracker

        if name == "items":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            return DictItemsVariable(self)
        elif name == "keys":
            if len(args):
                raise_args_mismatch(tx, name, "0 args", f"{len(args)} args")
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            return DictKeysVariable(self)
        elif name == "__reversed__":
            # dict.__reversed__: reverse insertion-order key iterator.
            # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c (dict___reversed___impl)
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            reversed_keys = [k.vt for k in reversed(list(self.items.keys()))]
            return variables.ListIteratorVariable(
                reversed_keys, mutation_type=ValueMutationNew()
            )
        elif name == "values":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            if args or kwargs:
                raise_observed_exception(TypeError, tx)
            return DictValuesVariable(self)
        elif name == "copy":
            self.install_dict_keys_match_guard()
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return self.clone(
                items=self.items.copy(), mutation_type=ValueMutationNew(), source=None
            )
        elif name == "get":
            if len(args) not in (1, 2):
                raise_args_mismatch(tx, name, "1 or 2 args", f"{len(args)} args")

            if args[0] not in self:
                self.install_dict_contains_guard(tx, args)
                if len(args) == 1:
                    # if default is not given, return None
                    return ConstantVariable.create(None)
                return args[1]
            # Key guarding - Nothing to do.
            return self.getitem_const(tx, args[0])
        elif name == "pop" and self.is_mutable():
            if len(args) not in (1, 2):
                raise_args_mismatch(tx, name, "1 or 2 args", f"{len(args)} args")

            if args[0] not in self:
                # missing item, return the default value. Install no DICT_CONTAINS guard.
                self.install_dict_contains_guard(tx, args)
                if len(args) == 1:
                    # if default is not given, raise KeyError
                    raise_observed_exception(KeyError, tx)
                return args[1]

            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            return self.items.pop(Hashable(args[0]))
        elif name == "popitem" and self.is_mutable():
            # dict.popitem() takes no args. OrderedDict.popitem(last=) is
            # handled by OrderedDictVariable.call_method.
            if len(args):
                raise_args_mismatch(tx, name)

            if not self.items:
                raise_observed_exception(
                    KeyError,
                    tx,
                    args=[
                        "popitem(): dictionary is empty",
                    ],
                )

            k, v = self.items.popitem()
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)

            return variables.TupleVariable([k.vt, v])
        elif name == "clear":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return ConstantVariable.create(None)
        elif name == "update" and self.is_mutable():
            # Mirrors CPython PyDict_Merge: if arg has keys(), iterate keys and
            # __getitem__; else treat arg as iterable of (key, value) pairs.
            # kwargs are always merged on top.
            # ref: https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Objects/dictobject.c#L3571
            self.install_dict_keys_match_guard()
            if len(args) > 1:
                raise_args_mismatch(tx, name, "at most 1 args", f"{len(args)} args")
            if args or kwargs:
                tx.output.side_effects.mutation(self)
            if args:
                other = args[0]
                if isinstance(other, ConstDictVariable):
                    # NB - Guard on all the keys of the other dict to ensure
                    # correctness.
                    other.install_dict_keys_match_guard()
                    self.items.update(other.items)
                elif (
                    isinstance(
                        other,
                        (variables.UserDefinedObjectVariable, MappingProxyVariable),
                    )
                    and other.call_obj_hasattr(tx, "keys").as_python_constant()
                ):
                    keys = other.call_method(tx, "keys", [], {})
                    for key in unpack_iterable(tx, keys):
                        self.items[Hashable(key)] = generic_getitem(tx, other, key)
                else:
                    for idx, item in enumerate(unpack_iterable(tx, other)):
                        pair = unpack_iterable(tx, item)
                        if len(pair) != 2:
                            raise_observed_exception(
                                ValueError,
                                tx,
                                args=[
                                    "dictionary update sequence element "
                                    f"#{idx} has length {len(pair)}; 2 is required"
                                ],
                            )
                        self.items[Hashable(pair[0])] = pair[1]
            for k, v in kwargs.items():
                self.items[Hashable(VariableTracker.build(tx, k))] = v
            return ConstantVariable.create(None)
        elif name == "setdefault" and self.is_mutable():
            if len(args) not in (1, 2):
                raise_args_mismatch(
                    tx,
                    name,
                    "1 or 2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            self.install_dict_keys_match_guard()
            if kwargs or len(args) > 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "at most 2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            value = self.maybe_getitem_const(args[0])
            if value is not None:
                return value
            else:
                if len(args) == 1:
                    x = ConstantVariable.create(None)
                else:
                    x = args[1]
                tx.output.side_effects.mutation(self)
                self.items[Hashable(args[0])] = x
                return x
        else:
            return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        self.install_dict_keys_match_guard()
        return [x.vt for x in self.items]

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/dictobject.c#L4643-L4658
        self_, other_ = (other, self) if reverse else (self, other)
        if pydict_check(self_) and pydict_check(other_):
            new = self_.call_method(tx, "copy", [], {})
            new.call_method(tx, "update", [other_], {})
            return new
        return ConstantVariable.create(NotImplemented)

    def nb_inplace_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/dictobject.c#L4660-L4667
        self.call_method(tx, "update", [other], {})
        return self

    def mp_ass_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c (dict_ass_sub)
        if not self.is_mutable():
            return super().mp_ass_subscript_impl(tx, key, value)

        # value=None signals delete (CPython NULL sentinel).
        if not is_hashable(key):
            raise_unhashable(key, tx)
        self.install_dict_keys_match_guard()
        hkey = HashableTracker(key)
        tx.output.side_effects.mutation(self)
        if value is None:
            if hkey not in self.items:
                raise_observed_exception(KeyError, tx, args=[key.as_python_constant()])
            self.should_reconstruct_all = True
            del self.items[hkey]
        else:
            self.items[hkey] = value
        return ConstantVariable.create(None)

    def mp_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        """Mapping length for dict objects."""
        self.install_dict_keys_match_guard()
        return VariableTracker.build(tx, len(self.items))

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        # dict not allow setting arbitrary attributes.  OrderedDict and
        # defaultdict allow arbitrary setattr, but not deletion of default attrs
        cls = self.python_type()
        if any(
            cls is t for t in (dict, collections.OrderedDict, collections.defaultdict)
        ):
            if hasattr(cls, name):
                return ConstantVariable.create(True)
            if cls is dict:
                return ConstantVariable.create(False)

        msg = f"hasattr on {cls} is not supported"
        unimplemented(
            gb_type="unsupported hasattr operation",
            context=f"Class {cls}",
            explanation=msg,
            hints=[
                "Consider using a regular dictionary instead",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def clone(self, **kwargs: Any) -> VariableTracker:
        self.install_dict_keys_match_guard()
        return super().clone(**kwargs)

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        from ..exc import raise_type_error

        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # dict_richcompare: https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L4198
        # Only supports eq/ne; returns NotImplemented for ordering.
        from .builder import SourcelessBuilder
        from .user_defined import UserDefinedDictVariable

        if op not in ("__eq__", "__ne__"):
            return ConstantVariable.create(NotImplemented)
        if not isinstance(other, ConstDictVariable):
            # Unwrap UserDefinedDictVariable to its base ConstDictVariable.
            # This is correct because CPython's dict_equal operates on the
            # internal C struct directly (ma_used, dk_entries, _Py_dict_lookup)
            # -- it never calls __getitem__ or __len__ on dict subclasses.
            # https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L4125-L4185
            if isinstance(other, UserDefinedDictVariable):
                if other._base_vt is None:
                    raise AssertionError("expected _base_vt to be set")
                other = other._base_vt
            else:
                return ConstantVariable.create(NotImplemented)
        eq_result = SourcelessBuilder.create(tx, polyfills.dict___eq__).call_function(
            tx, [self, other], {}
        )
        if op == "__ne__":
            return VariableTracker.build(tx, not eq_result.as_python_constant())
        return eq_result

    def tp_getattro_impl(self, tx: "InstructionTranslatorBase", name: str):
        if name == "__class__":
            return VariableTracker.build(tx, self.python_type())
        # DictGuardManager does not support getattr_manager for plain dicts,
        # so AttrSource chains through a dict source break guard creation.
        # Return CallMethodVariable directly for methods, bypassing the
        # MRO walk in object_generic_getattr that would create that chain.
        type_attr = mro_lookup(self.python_type(), name)
        if type_attr is not NO_SUCH_SUBOBJ and _is_method_type(type_attr):
            return variables.CallMethodVariable(self, name)
        return super().tp_getattro_impl(tx, name)


class OrderedDictVariable(ConstDictVariable):
    _cpython_type = collections.OrderedDict

    # OrderedDict-exclusive C methods, declared like CPython's odict tp_methods.
    # move_to_end is odict-only; popitem honors last= (vs dict's LIFO popitem).
    # OrderedDict exposes no tp_getset / tp_members.
    # https://github.com/python/cpython/blob/v3.13.0/Objects/odictobject.c

    def move_to_end(
        self: "OrderedDictVariable",
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.install_dict_keys_match_guard()
        tx.output.side_effects.mutation(self)
        if args[0] not in self:
            raise_observed_exception(KeyError, tx)

        last = True
        if len(args) == 2 and args[0].is_python_constant():
            last = args[1].as_python_constant()
        if "last" in kwargs and kwargs["last"].is_python_constant():
            last = kwargs["last"].as_python_constant()

        self.items.move_to_end(HashableTracker(args[0]), last=last)
        return ConstantVariable.create(None)

    def popitem(
        self: "OrderedDictVariable",
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not self.items:
            raise_observed_exception(
                KeyError, tx, args=["popitem(): dictionary is empty"]
            )

        last = True
        if len(args) == 1 and args[0].is_python_constant():
            last = args[0].as_python_constant()
        if "last" in kwargs and kwargs["last"].is_python_constant():
            last = kwargs["last"].as_python_constant()

        k, v = self.items.popitem(last=last)
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        return variables.TupleVariable([k.vt, v])

    tp_methods = {
        "move_to_end": Method(move_to_end),
        "popitem": Method(popitem),
    }

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Consult tp_methods before ConstDictVariable's own dispatch, whose
        # bare-dict popitem() branch would otherwise shadow the OrderedDict
        # popitem(last=) handler.
        method = self.lookup_tp_method(name)
        if method is not None:
            result = method(self, tx, name, args, kwargs)
            if result is not None:
                return result
        return super().call_method(tx, name, args, kwargs)

    def as_python_constant(self) -> "collections.OrderedDict[Any, Any]":
        return collections.OrderedDict(super().as_python_constant())

    @staticmethod
    def _ordered_dict_repr(items: list[tuple[str, str]]) -> str:
        if not items:
            return "OrderedDict()"
        if sys.version_info >= (3, 12):
            return (
                "OrderedDict({"
                + ", ".join(f"{key}: {value}" for key, value in items)
                + "})"
            )
        return (
            "OrderedDict(["
            + ", ".join(f"({key}, {value})" for key, value in items)
            + "])"
        )

    def debug_repr(self) -> str:
        items = [(k.vt.debug_repr(), v.debug_repr()) for k, v in self.items.items()]
        return self._ordered_dict_repr(items)

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # Python < 3.12 uses the historical list-of-pairs form, while 3.12+
        # uses dict-style formatting.
        items = [
            (tracked_repr(tx, key.vt), tracked_repr(tx, value))
            for key, value in self.items.items()
        ]
        return VariableTracker.build(tx, self._ordered_dict_repr(items))

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # OrderedDict.__or__/__ror__ preserve the OrderedDict type, unlike
        # dict.__or__ which returns a plain dict for a subclass operand.
        # ref: https://github.com/python/cpython/blob/3.13/Lib/collections/__init__.py#L327-L339
        if not issubclass(other.python_type(), dict):
            return ConstantVariable.create(NotImplemented)
        if reverse:
            new = VariableTracker.build(tx, self.python_type()).call_function(
                tx, [other], {}
            )
            new.call_method(tx, "update", [self], {})
        else:
            new = self.call_method(tx, "copy", [], {})
            new.call_method(tx, "update", [other], {})
        return new

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # emit `OrderedDict(constructed_dict)`
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(collections),
                    codegen.create_load_attr("OrderedDict"),
                ]
            )
        )
        if self._contains_self_reference():
            codegen.extend_output(
                [
                    *create_call_function(0, False),
                    create_dup_top(),
                ]
            )
            codegen.add_cache(self)

            codegen.append_output(create_dup_top())
            codegen.load_method("update")
            self.reconstruct_kvs_into_new_dict(codegen)
            codegen.extend_output(
                [
                    *create_call_method(1),
                    create_instruction("POP_TOP"),
                ]
            )
        else:
            self.reconstruct_kvs_into_new_dict(codegen)
            codegen.extend_output(create_call_function(1, False))

    def reconstruct_pycode(self, codegen) -> str:
        raise NotImplementedError


class MappingProxyVariable(VariableTracker):
    # PyDictProxy_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/descrobject.c#L1995
    _cpython_type = types.MappingProxyType

    # proxies to the original dict_vt
    def __init__(self, dv_dict: ConstDictVariable, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(dv_dict, ConstDictVariable):
            raise AssertionError(f"Expected ConstDictVariable, got {type(dv_dict)}")
        self.dv_dict = dv_dict

    def python_type(self) -> type:
        return types.MappingProxyType

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # mappingproxy.__hash__ delegates to the underlying dict, which
        # raises TypeError. Mirror that behavior.
        return self.dv_dict.hash_impl(tx)

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        return self.dv_dict.unpack_var_sequence(tx)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # load types.MappingProxyType
        if self.source:
            msg = (
                f"Preexisting MappingProxyVariable (source: {self.source}) cannot be reconstructed "
                "because the connection to the original dict will be lost."
            )
            unimplemented(
                gb_type="mapping proxy cannot be reconstructed",
                context=f"Source: {self.source}",
                explanation=msg,
                hints=[
                    "Use a mapping proxy constructed in the same `torch.compile` region.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(types),
                    codegen.create_load_attr("MappingProxyType"),
                ]
            )
        )
        codegen(self.dv_dict)
        codegen.extend_output(create_call_function(1, False))

    def _check_mutation_guard(self, tx: "InstructionTranslatorBase") -> None:
        if self.source and tx.output.side_effects.has_existing_dict_mutation():
            msg = (
                "A dict has been modified while we have an existing mappingproxy object. "
                "A mapping proxy object, as the name suggest, proxies a mapping "
                "object (usually a dict). If the original dict object mutates, it "
                "is reflected in the proxy object as well. For an existing proxy "
                "object, we do not know the original dict it points to. Therefore, "
                "for correctness we graph break when there is dict mutation and we "
                "are trying to access a proxy object."
            )

            unimplemented(
                gb_type="mapping proxy affected by dictionary mutation",
                context=f"Source: {self.source}, Dict mutation detected",
                explanation=msg,
                hints=[
                    "Avoid modifying dictionaries that might be referenced by mapping proxy objects",
                    "Or avoid using the mapping proxy objects after modifying its underlying dictionary",
                ],
            )

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # mappingproxy_getitem: https://github.com/python/cpython/blob/62a6e898e01/Objects/descrobject.c#L1052-L1056
        # TODO(follow-up): add tests for invalid key type, missing key
        self._check_mutation_guard(tx)
        return self.dv_dict.mp_subscript_impl(tx, key)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self._check_mutation_guard(tx)
        return self.dv_dict.call_method(tx, name, args, kwargs)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return self.dv_dict.tp_iter_impl(tx)

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # https://github.com/python/cpython/blob/60403a5409ff/Objects/descrobject.c#L1087-L1095
        return self.dv_dict.sq_contains_impl(tx, item)

    def mp_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return self.dv_dict.mp_length_impl(tx)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # mappingproxy_richcompare delegates to the underlying mapping:
        # https://github.com/python/cpython/blob/60403a5409ff/Objects/descrobject.c#L1231-L1236
        from .object_protocol import generic_richcompare

        return generic_richcompare(tx, self.dv_dict, other, op)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.python_type() is types.MappingProxyType:
            return VariableTracker.build(tx, name in types.MappingProxyType.__dict__)
        return super().call_obj_hasattr(tx, name)


class NNModuleHooksDictVariable(OrderedDictVariable):
    # Special class to avoid adding any guards on the nn module hook ids.
    # nn.Module hook dicts are OrderedDict, so back storage/reconstruct with it.
    def install_dict_keys_match_guard(self) -> None:
        pass

    def install_dict_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        pass


class DictViewVariable(VariableTracker):
    """
    Models _PyDictViewObject

    This is an "abstract" class. Subclasses will override kv and the items method
    """

    kv: str | None = None

    def __init__(self, dv_dict: ConstDictVariable, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.kv not in ("keys", "values", "items"):
            raise AssertionError(
                f"Expected kv to be 'keys', 'values', or 'items', got {self.kv!r}"
            )
        if not isinstance(dv_dict, ConstDictVariable):
            raise AssertionError(f"Expected ConstDictVariable, got {type(dv_dict)}")
        self.dv_dict = dv_dict

    @property
    def view_items(self) -> Any:
        if self.kv is None:
            raise AssertionError("kv must not be None")
        return getattr(self.dv_dict.items, self.kv)()

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        from ..exc import raise_type_error

        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        # Returns an iterable of the unpacked items
        # Implement in the subclasses
        raise NotImplementedError

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        return self.view_items_vt

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.kv is None:
            raise AssertionError("kv must not be None for reconstruct")
        codegen(self.dv_dict)
        codegen.load_method(self.kv)
        codegen.call_method(0)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.kv is None:
            raise AssertionError("kv must not be None for call_obj_hasattr")
        if name in self.python_type().__dict__:
            return ConstantVariable.create(True)
        return ConstantVariable.create(False)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # dictview_mapping getset returns a read-only mappingproxy of the
        # underlying dict for dict_keys/values/items.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L5032-L5040
        if name == "mapping":
            return MappingProxyVariable(self.dv_dict)
        return super().tp_getattro_impl(tx, name)

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if self.kv == "keys":
            items = ", ".join(tracked_repr(tx, key.vt) for key in self.view_items)
        elif self.kv == "values":
            items = ", ".join(tracked_repr(tx, value) for value in self.view_items)
        else:  # items
            items = ", ".join(
                f"({tracked_repr(tx, key.vt)}, {tracked_repr(tx, value)})"
                for key, value in self.view_items
            )
        return VariableTracker.build(tx, f"{self.python_type_name()}([{items}])")

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__reversed__":
            # dict_keys/values/items __reversed__: reverse insertion order.
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            if self.dv_dict.source and not is_constant_source(self.dv_dict.source):
                tx.output.guard_on_key_order.add(self.dv_dict.source)
            return variables.ListIteratorVariable(
                list(reversed(self.view_items_vt)),
                mutation_type=ValueMutationNew(),
            )
        return super().call_method(tx, name, args, kwargs)

    def sq_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        """Sequence length for dict view objects."""
        return VariableTracker.build(tx, len(self.view_items))

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6142-L6155 (dictviews_or)
        s = VariableTracker.build(tx, set).call_function(tx, [self], {})
        s.call_method(tx, "update", [other], {})
        return s

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6036 (dictviews_sub)
        self_, other_ = (other, self) if reverse else (self, other)
        s = VariableTracker.build(tx, set).call_function(tx, [self_], {})
        s.call_method(tx, "difference_update", [other_], {})
        return s

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/dictobject.c#L6105-L6188 (_PyDictView_Intersect)
        s = VariableTracker.build(tx, set).call_function(tx, [self], {})
        s.call_method(tx, "intersection_update", [other], {})
        return s

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/dictobject.c#L6305-L6325 (dictviews_xor)
        s = VariableTracker.build(tx, set).call_function(tx, [self], {})
        s.call_method(tx, "symmetric_difference_update", [other], {})
        return s


class DictKeysVariable(DictViewVariable):
    # PyDictKeys_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6365
    _cpython_type = dict_keys

    kv = "keys"

    @property
    def set_items(self) -> set[VariableTracker]:
        return set(self.view_items)

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        # Returns an iterable of the unpacked items
        return [x.vt for x in self.view_items]

    def python_type(self) -> type:
        return dict_keys

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import DictKeysIterator

        if self.dv_dict.source and not is_constant_source(self.dv_dict.source):
            tx.output.guard_on_key_order.add(self.dv_dict.source)
        return DictKeysIterator(self.dv_dict.items)

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_keys([])"
        else:
            items: list[str] = []
            for k in self.view_items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "dict_keys([" + ", ".join(items) + "])"

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L5998-L6005
        return self.dv_dict.sq_contains_impl(tx, item)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # dictview_richcompare: accepts set/frozenset and dict_keys/dict_items.
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L5952-L6010
        # Uses a polyfill with len() and ``in`` so that Dynamo traces through
        # sq_length/sq_contains, matching CPython's use of PyObject_Size and
        # PySequence_Contains.
        from .builder import SourcelessBuilder

        if not _is_set_or_dictview(other):
            return ConstantVariable.create(NotImplemented)
        return SourcelessBuilder.create(
            tx, polyfills.dictview_richcompare
        ).call_function(
            tx,
            [
                SourcelessBuilder.create(tx, cmp_name_to_op_mapping[op]),
                self,
                other,
            ],
            {},
        )


class DictValuesVariable(DictViewVariable):
    # PyDictValues_Type: https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L6615
    _cpython_type = dict_values

    # dict_values is hashable (identity hash), unlike dict_keys/dict_items.
    #
    # CPython only inherits tp_hash from the base when BOTH tp_richcompare
    # AND tp_hash are NULL on the type
    # (https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L7665-L7679).
    # dict_keys/dict_items set tp_richcompare (for set-like comparisons),
    # so their tp_hash is NOT inherited — it stays NULL, and
    # type_ready_set_hash sets it to PyObject_HashNotImplemented (unhashable)
    # (https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L8066-L8085).
    # dict_values does NOT set tp_richcompare
    # (https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L6630),
    # so it inherits object's tp_hash (PyObject_GenericHash) — hashable.
    #
    # Override DictViewVariable.hash_impl to restore the base identity hash.
    kv = "values"

    # dict.values() do not implement tp_as_number
    nb_or_impl = None  # type: ignore[bad-override]
    nb_inplace_or_impl = None  # type: ignore[bad-override]
    nb_subtract_impl = None  # type: ignore[bad-override]
    nb_inplace_subtract_impl = None  # type: ignore[bad-override]
    nb_and_impl = None  # type: ignore[bad-override]
    nb_inplace_and_impl = None  # type: ignore[bad-override]
    nb_xor_impl = None  # type: ignore[bad-override]
    nb_inplace_xor_impl = None  # type: ignore[bad-override]

    def is_hashable(self) -> bool:
        return True

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        return super(DictViewVariable, self).hash_impl(tx)

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        return list(self.view_items)

    def python_type(self) -> type:
        return dict_values

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import DictValuesIterator

        if self.dv_dict.source and not is_constant_source(self.dv_dict.source):
            tx.output.guard_on_key_order.add(self.dv_dict.source)
        return DictValuesIterator(self.dv_dict.items)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # dict_values has no tp_richcompare (inherits object's).
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_values([])"
        else:
            items: list[str] = []
            for v in self.view_items:
                val_str = _item_debug_repr(v)
                items.append(val_str)
            return "dict_values([" + ", ".join(items) + "])"


class DictItemsVariable(DictViewVariable):
    # PyDictItems_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6477
    _cpython_type = dict_items

    kv = "items"

    @property
    def set_items(self) -> set["HashableTracker"]:
        return {
            HashableTracker(variables.TupleVariable([k.vt, v]))
            for k, v in self.view_items
        }

    @property
    def view_items_vt(self) -> list[VariableTracker]:
        # Returns an iterable of the unpacked items
        return [variables.TupleVariable([k.vt, v]) for k, v in self.view_items]

    def python_type(self) -> type:
        return dict_items

    def debug_repr(self) -> str:
        if not self.view_items:
            return "dict_items([])"
        else:
            items: list[str] = []
            for k, v in self.view_items:
                key_str = _item_debug_repr(k.vt)
                val_str = _item_debug_repr(v)
                items.append(f"({key_str}, {val_str})")
            return "dict_items([" + ", ".join(items) + "])"

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L6433-L6451
        from ..utils import iter_contains

        if not is_hashable(item):
            raise_type_error(tx, f"unhashable type: '{item.python_type_name()}'")

        # Fast path: if item is a known (key, value) pair, use O(1) dict lookup.
        if isinstance(item, variables.TupleVariable) and len(item.items) == 2:
            key, val = item.items
            key_ht = HashableTracker(key)
            if key_ht not in self.dv_dict.items:
                return VariableTracker.build(tx, False)
            stored = self.dv_dict.items[key_ht]
            # dictitems_contains: PyObject_RichCompareBool(found, value, Py_EQ)
            # on the stored value, so a value whose __eq__ raises propagates.
            return generic_richcompare_bool(tx, stored, val, "__eq__")

        return iter_contains(self.view_items_vt, item, tx)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # dictview_richcompare: accepts set/frozenset and dict_keys/dict_items.
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L5952-L6010
        # Uses a polyfill with len() and ``in`` so that Dynamo traces through
        # sq_length/sq_contains, matching CPython's use of PyObject_Size and
        # PySequence_Contains.
        from .builder import SourcelessBuilder

        if not _is_set_or_dictview(other):
            return ConstantVariable.create(NotImplemented)
        return SourcelessBuilder.create(
            tx, polyfills.dictview_richcompare
        ).call_function(
            tx,
            [
                SourcelessBuilder.create(tx, cmp_name_to_op_mapping[op]),
                self,
                other,
            ],
            {},
        )

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import DictItemsIterator

        if self.dv_dict.source and not is_constant_source(self.dv_dict.source):
            tx.output.guard_on_key_order.add(self.dv_dict.source)
        return DictItemsIterator(self.dv_dict.items)


kV = HashableTracker | str


class SideEffectsProxyDict(collections.abc.MutableMapping[kV, VariableTracker]):
    """
    A proxy dict that allows us to track mutations to the dict using side
    effects table as storage.
    """

    @staticmethod
    def get_example_value_dict(vt: VariableTracker) -> dict[str, object]:
        if istype(vt, variables.NestedUserFunctionVariable):
            # NestedUserFunctionVariable is created with MAKE_FUNCTION and its
            # __dict__ starts empty. Any mutation will actually be recorded in
            # the side effects table.
            return {}
        elif isinstance(vt, variables.LocalGeneratorFunctionVariable):
            return SideEffectsProxyDict.get_example_value_dict(vt.vt)
        elif istype(vt, variables.ExceptionVariable):
            # Synthetic exceptions created during tracing have no backing
            # Python object; their __dict__ starts empty and any custom
            # attribute is recorded in the side effects table.
            return {}
        else:
            value = vt.get_real_python_backed_value()
            if value is not NO_SUCH_SUBOBJ:
                if isinstance(vt, variables.UserDefinedObjectVariable):
                    return vt._getattr_static("__dict__")  # type: ignore[bad-return]
                else:
                    return object.__getattribute__(value, "__dict__")
            else:
                unimplemented(
                    gb_type="unsupported variable type for __dict__ access",
                    context=f"VariableTracker type: {type(vt)}",
                    explanation=f"Dynamo does not know how to get __dict__ from {type(vt)}",
                    hints=[
                        *graph_break_hints.DYNAMO_BUG,
                    ],
                )

    @staticmethod
    def get_value___dict__(
        tx: "InstructionTranslatorBase", vt: VariableTracker
    ) -> dict[str, VariableTracker]:
        # GENERIC_SETATTR on "__dict__" means the whole instance dict was
        # replaced through the __dict__ getset descriptor (obj.__dict__ = ...).
        # Per-key obj.__dict__[name] mutations are tracked as INSTANCE_DICT.
        if isinstance(
            vt, variables.UserDefinedObjectVariable
        ) and tx.output.side_effects.has_pending_mutation_of_attr(
            vt, "__dict__", AttrMutationKind.GENERIC_SETATTR
        ):
            dict_vt = tx.output.side_effects.load_attr(vt, "__dict__")
            if isinstance(dict_vt, ConstDictVariable):
                result = {}
                for key, value in dict_vt.items.items():
                    try:
                        key_value = key.vt.as_python_constant()
                    except NotImplementedError:
                        unimplemented(
                            gb_type="non-constant key in object __dict__",
                            context=f"key={key.vt}",
                            explanation="Dynamo expects object __dict__ replacement keys to be constants.",
                            hints=[*graph_break_hints.SUPPORTABLE],
                        )
                    if not isinstance(key_value, str):
                        unimplemented(
                            gb_type="non-string key in object __dict__",
                            context=f"key={key_value!r}",
                            explanation="Dynamo expects object __dict__ keys to be strings.",
                            hints=[*graph_break_hints.USER_ERROR],
                        )
                    result[key_value] = value
                return result
            # Other replacement forms are not materialized here yet. Leave
            # them on the normal example-value path until Dynamo models their
            # contents explicitly.

        example_value_dict = SideEffectsProxyDict.get_example_value_dict(vt)

        return {
            key: VariableTracker.build(
                tx,
                value,
                source=vt.source
                and DictGetItemSource(AttrSource(vt.source, "__dict__"), key),
            )
            for key, value in example_value_dict.items()
        }

    def __init__(self, item: VariableTracker, tx: "InstructionTranslatorBase") -> None:
        self.item = item
        self.side_effects = tx.output.side_effects
        self.item_dict = self.get_value___dict__(tx, item)

    def _maybe_unwrap_key(self, key: kV) -> str:
        Hasher = HashableTracker
        return key.vt.as_python_constant() if istype(key, Hasher) else key

    def side_effects_table(self) -> dict[str, VariableTracker]:
        return {
            name: value
            for name, value in self.side_effects.store_attr_mutations.get(
                self.item, {}
            ).items()
            if self.side_effects.has_pending_mutation_of_attr(
                self.item, name, AttrMutationKind.INSTANCE_DICT
            )
        }

    def __getitem__(self, key: kV) -> VariableTracker:
        name = self._maybe_unwrap_key(key)
        if self.side_effects.has_pending_mutation_of_attr(
            self.item, name, AttrMutationKind.INSTANCE_DICT
        ):
            return self.side_effects.load_attr(self.item, name, deleted_ok=True)
        return self.item_dict[name]

    def __setitem__(self, key: kV, value: VariableTracker) -> None:
        # Find a way to not hash the key using HashableTracker. CPython's
        # instance __dict__ accepts arbitrary hashable keys when set via the
        # mapping API (only attribute access via setattr requires str), and
        # instance-dict mutations replay with object_setattr_ignore_descriptor
        # (a plain __dict__ store), so a non-str constant key is fine.
        name = self._maybe_unwrap_key(key)
        self.side_effects.store_instance_dict_attr(self.item, name, value)

    def __delitem__(self, key: kV) -> None:
        name = self._maybe_unwrap_key(key)
        self.side_effects.store_instance_dict_attr(
            self.item, name, variables.DeletedVariable()
        )

    def __contains__(self, key: kV) -> bool:  # type: ignore[bad-override]
        name = self._maybe_unwrap_key(key)
        if self.side_effects.has_pending_mutation_of_attr(
            self.item, name, AttrMutationKind.INSTANCE_DICT
        ):
            value = self.side_effects.load_attr(self.item, name, deleted_ok=True)
            return not isinstance(value, variables.DeletedVariable)
        return name in self.item_dict

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[HashableTracker]:
        Hasher = HashableTracker
        d = self.side_effects_table()
        for k, v in d.items():
            if isinstance(v, variables.DeletedVariable):
                continue
            yield Hasher(ConstantVariable.create(k))

        for k, v in self.item_dict.items():
            if k not in d:
                yield Hasher(ConstantVariable.create(k))


class DunderDictVariable(ConstDictVariable):
    """represents object.__dict__"""

    @classmethod
    def create(
        cls,
        tx: "InstructionTranslatorBase",
        vt: VariableTracker,
    ) -> "DunderDictVariable":
        mutation = AttributeMutationExisting() if vt.source else AttributeMutationNew()
        source = vt.source and AttrSource(vt.source, "__dict__")

        return cls(
            vt,
            tx=tx,
            mutation_type=mutation,
            source=source,
        )

    def __init__(
        self,
        vt: VariableTracker,
        tx: "InstructionTranslatorBase",
        **kwargs: Any,
    ) -> None:
        super().__init__({}, **kwargs)
        self.items = SideEffectsProxyDict(vt, tx)

    def setitem(self, name: str, value: VariableTracker) -> None:
        self.items[name] = value

    def delitem(self, name: str) -> None:
        if not self.contains(name):
            raise KeyError(name)
        del self.items[name]

    def getitem(self, name: str) -> VariableTracker:
        return self.items[name]

    def contains(self, name: str) -> bool:
        return name in self.items

    def getitem_or_default(
        self,
        name: str,
        default: Callable[[], VariableTracker],
    ) -> VariableTracker:
        if self.contains(name):
            return self.getitem(name)
        else:
            value = default()
            self.items[name] = value
            return value

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "copy":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return ConstDictVariable(
                dict(self.items), mutation_type=ValueMutationNew(), source=None
            )
        return super().call_method(tx, name, args, kwargs)

    # Mutations to __dict__ are tracked through side effects (SideEffectsProxyDict),
    # so we don't need to install guards. Guard installation is overridden to no-op.
    def install_dict_keys_match_guard(self) -> None:
        pass

    def install_dict_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        pass
<|endoftext|>"""
Variable tracking implementations for list-like data structures in Dynamo.

This module provides specialized variable tracking for various collection types:
- Lists and list subclasses (including torch.nn.ModuleList, ParameterList)
- Tuples and named tuples
- Ranges and slices
- Collections.deque
- torch.Size with special proxy handling

The implementations support both mutable and immutable collections, iteration,
and common sequence operations. Each collection type has a dedicated Variable
class that handles its unique behaviors while integrating with Dynamo's
variable tracking system.
"""

import collections
import operator
import sys
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.fx
from torch.utils._pytree import SequenceKey

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import (
    create_build_tuple,
    create_call_function,
    create_call_method,
    create_dup_top,
    create_instruction,
)
from ..exc import (
    ObservedException,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
    Unsupported,
)
from ..source import AttrSource
from ..utils import (
    check_positional,
    cmp_name_to_op_mapping,
    get_fake_value,
    guard_if_dyn,
    iter_contains,
    odict_values,
    raise_args_mismatch,
    range_iterator,
    set_example_value,
    tracked_repr,
    unpack_and_apply_fn,
    unpack_iterable,
)
from .base import (
    AsPythonConstantNotImplementedError,
    GetSet,
    Method,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable
from .functions import UserFunctionVariable
from .iter import IteratorVariable
from .object_protocol import (
    generic_richcompare_bool,
    maybe_get_python_type,
    pyindex_check,
    pylong_as_ssize_t,
    pynumber_as_ssize_t,
    pynumber_index,
    type_implements_nb_index,
    vt_is_iterable,
)


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


def pytuple_checkexact(obj: VariableTracker) -> bool:
    return obj.python_type() is tuple


def pytuple_check(obj: VariableTracker) -> bool:
    return issubclass(obj.python_type(), tuple)


def pylist_check(obj: VariableTracker) -> bool:
    return issubclass(obj.python_type(), list)


def pylist_checkexact(obj: VariableTracker) -> bool:
    return obj.python_type() is list


def pyslice_check(obj: VariableTracker) -> bool:
    return issubclass(obj.python_type(), slice)


def _cpython_has_simple_slice_bug() -> bool:
    # CPython gh-120384 fixed an array-out-of-bounds crash by moving the
    # PySequence_Fast check ahead of the step==1 branch in
    # list_ass_subscript, which as a side effect unified the non-iterable
    # error message for simple and extended slices. Before this landed,
    # simple slices (step is None or 1) raised "can only assign an
    # iterable" instead of "must assign iterable to extended slice".
    # Verified empirically against real CPython builds: the fix shipped in
    # 3.10.20, 3.11.15, and 3.12.5 (3.12.0 is the first 3.12 release, so
    # 3.12.0-3.12.4 predate it); 3.13+ never had the bug.
    v = sys.version_info
    if v[:2] == (3, 10):
        return v < (3, 10, 20)
    if v[:2] == (3, 11):
        return v < (3, 11, 15)
    if v[:2] == (3, 12):
        return v < (3, 12, 5)
    return False


class BaseListVariable(VariableTracker):
    @staticmethod
    def cls_for_instance(obj: Any) -> type["BaseListVariable"]:
        return BaseListVariable.cls_for(type(obj))

    @staticmethod
    def cls_for(obj: Any) -> type:
        return {
            iter: ListIteratorVariable,
            list: ListVariable,
            slice: SliceVariable,
            torch.Size: SizeVariable,
            tuple: TupleVariable,
            odict_values: ListVariable,
            torch.nn.ParameterList: ListVariable,
            torch.nn.ModuleList: ListVariable,
            collections.deque: DequeVariable,
        }[obj]

    def __init__(
        self,
        items: list[VariableTracker],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(items, list):
            raise AssertionError(f"items must be a list, got {type(items).__name__}")
        if not all(isinstance(x, VariableTracker) for x in items):
            raise AssertionError("all items must be VariableTracker instances")
        self.items: list[VariableTracker] = items

    def _as_proxy(self) -> list[Any]:
        return [x.as_proxy() for x in self.items]

    def modified(
        self, items: list[VariableTracker], **kwargs: Any
    ) -> "BaseListVariable":
        return type(self)(items, **kwargs)

    @property
    def value(self) -> Any:
        return self.as_python_constant()

    def debug_repr_helper(self, prefix: str, suffix: str) -> str:
        return prefix + ", ".join(i.debug_repr() for i in self.items) + suffix

    def as_python_constant(self) -> Any:
        return self.python_type()([x.as_python_constant() for x in self.items])

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        return VariableTracker.build(tx, self.debug_repr())

    def as_proxy(self) -> Any:
        if self.python_type() is SizeVariable:
            raise AssertionError(
                "SizeVariable should not use BaseListVariable.as_proxy"
            )
        return self.python_type()(self._as_proxy())

    def getitem_const(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        if pyindex_check(maybe_get_python_type(arg)):
            index = pynumber_as_ssize_t(tx, arg).as_python_constant()
            try:
                return self.items[index]
            except IndexError:
                raise_observed_exception(
                    IndexError, tx, args=["list index out of range"]
                )
        elif pyslice_check(arg):
            if not isinstance(arg, SliceVariable):
                raise AssertionError("Expected arg to be a SliceVariable")
            index = arg.as_index_slice(tx)
            if index.step == 0:
                raise_observed_exception(
                    ValueError, tx, args=["slice step cannot be zero"]
                )
            # Set source to None because slicing a list gives a new local
            return self.clone(
                items=self.items[index],
                source=None,
                mutation_type=ValueMutationNew() if self.mutation_type else None,
            )
        else:
            raise_type_error(
                tx,
                f"{self.python_type_name()} indices must be integers or slices, "
                f"not {arg.python_type_name()}",
            )

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        return list(self.items)

    def sq_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        """Sequence length for lists, tuples, and range objects."""
        return VariableTracker.build(tx, len(self.items))

    mp_length_impl = sq_length_impl

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/listobject.c#L635-L652

        # TODO(dynamo-team): Replace iter_contains by a proper impl. once we
        # implement PyObject_RichCompare
        return iter_contains(self.items, item, tx)

    def call_tree_map_branch(
        self,
        tx: "InstructionTranslatorBase",
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not isinstance(self, (ListVariable, TupleVariable)):
            return self._tree_map_fallback(
                tx, tree_map_fn, map_fn, rest, tree_map_kwargs
            )

        other_lists: list[BaseListVariable] = []
        for candidate in rest:
            if (
                not isinstance(candidate, BaseListVariable)
                or len(candidate.items) != len(self.items)
                or self.python_type() != candidate.python_type()
            ):
                return self._tree_map_fallback(
                    tx, tree_map_fn, map_fn, rest, tree_map_kwargs
                )
            other_lists.append(candidate)

        new_items: list[VariableTracker] = []
        for idx, item in enumerate(self.items):
            sibling_leaves = [candidate.items[idx] for candidate in other_lists]
            new_items.append(
                item.call_tree_map(
                    tx,
                    tree_map_fn,
                    map_fn,
                    sibling_leaves,
                    tree_map_kwargs,
                )
            )

        return self.clone(
            items=new_items,
            source=None,
            mutation_type=ValueMutationNew(),
        )

    def call_tree_map_with_path_branch(
        self,
        tx: "InstructionTranslatorBase",
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
        keypath: tuple[Any, ...],
    ) -> VariableTracker:
        if not isinstance(self, (ListVariable, TupleVariable)):
            return self._tree_map_with_path_fallback(
                tx, tree_map_fn, map_fn, rest, tree_map_kwargs, keypath
            )

        other_lists: list[BaseListVariable] = []
        for candidate in rest:
            if (
                not isinstance(candidate, BaseListVariable)
                or len(candidate.items) != len(self.items)
                or self.python_type() != candidate.python_type()
            ):
                return self._tree_map_with_path_fallback(
                    tx, tree_map_fn, map_fn, rest, tree_map_kwargs, keypath
                )
            other_lists.append(candidate)

        new_items: list[VariableTracker] = []
        for idx, item in enumerate(self.items):
            sibling_leaves = [candidate.items[idx] for candidate in other_lists]
            child_keypath = keypath + (SequenceKey(idx),)
            new_items.append(
                item.call_tree_map_with_path(
                    tx,
                    tree_map_fn,
                    map_fn,
                    sibling_leaves,
                    tree_map_kwargs,
                    child_keypath,
                )
            )

        return self.clone(
            items=new_items,
            source=None,
            mutation_type=ValueMutationNew(),
        )

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # list_subscript: https://github.com/python/cpython/blob/62a6e898e01/Objects/listobject.c#L3689-L3710
        return self.getitem_const(tx, key)

    def sq_item_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # list_item: https://github.com/python/cpython/blob/62a6e898e01/Objects/listobject.c#L335-L351
        # tuple_item: https://github.com/python/cpython/blob/62a6e898e01/Objects/tupleobject.c#L421-L430
        # CPython's sq_item takes Py_ssize_t (already int from generic_getitem's
        # nb_index_impl).  Unlike mp_subscript, sq_item never handles slices.
        index = key.as_python_constant()
        try:
            return self.items[index]
        except IndexError:
            raise_observed_exception(
                IndexError,
                tx,
                args=[f"{self.python_type_name()} index out of range"],
            )

    def sq_repeat_impl(
        self,
        tx: "InstructionTranslatorBase",
        count: VariableTracker,
    ) -> VariableTracker:
        n = count.as_python_constant()
        if pytuple_checkexact(self) and (n == 1 or not self.items):
            return self
        try:
            new_items = self.items * n
        except (MemoryError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        # self.items is a list, so we can't go through VariableTracker.build (which would wrap in a list variable)
        kwargs: dict[str, Any] = {}
        if issubclass(self.python_type(), list):
            kwargs["mutation_type"] = ValueMutationNew()
        return type(self)(new_items, **kwargs)

    def _seq_richcompare(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
        accepted_type: type,
    ) -> VariableTracker:
        """Shared list_richcompare / tuplerichcompare core.

        https://github.com/python/cpython/blob/e76aa128fe/Objects/listobject.c#L3352
        https://github.com/python/cpython/blob/e76aa128fe/Objects/tupleobject.c#L821
        CPython operates on the internal C array directly, so we compare
        VT items without going through a polyfill.
        """
        from .object_protocol import generic_richcompare
        from .tensor import SymNodeVariable

        try:
            other_type = other.python_type()
        except NotImplementedError:
            return ConstantVariable.create(NotImplemented)
        if not issubclass(other_type, accepted_type):
            return ConstantVariable.create(NotImplemented)

        left = self.items
        # CPython uses ob_item (the C array) directly, bypassing __iter__.
        # For user-defined subclasses (UserDefinedListVariable etc.),
        # the items live on the _base_vt.
        other_vt = getattr(other, "_base_vt", None) or other
        right = other_vt.items  # pyrefly: ignore[missing-attribute]

        cmp_op = cmp_name_to_op_mapping[op]

        if cmp_op is operator.eq and len(left) != len(right):
            return ConstantVariable.create(False)
        if cmp_op is operator.ne and len(left) != len(right):
            return ConstantVariable.create(True)

        sym_eq_acc = None
        for a, b in zip(left, right):
            # CPython uses PyObject_RichCompareBool per element, which has
            # an identity shortcut before the full do_richcompare dispatch.
            eq_result = generic_richcompare_bool(tx, a, b, "__eq__")
            if eq_result.is_python_constant():
                if not eq_result.as_python_constant():
                    if cmp_op in (operator.eq, operator.ne):
                        return ConstantVariable.create(cmp_op is operator.ne)
                    return generic_richcompare(tx, a, b, op)
            elif eq_result.is_symnode_like():
                if cmp_op not in (operator.eq, operator.ne):
                    # Ordering requires branching on element equality, which
                    # can't be expressed as a single symbolic expression.
                    # Specialize: install a guard on the equality result and
                    # proceed with the concrete value.
                    eq_bool = guard_if_dyn(eq_result)
                    if not eq_bool:
                        return generic_richcompare(tx, a, b, op)
                    continue
                if sym_eq_acc is None:
                    sym_eq_acc = eq_result
                else:
                    proxy = tx.output.create_proxy(
                        "call_function",
                        operator.and_,
                        (sym_eq_acc.as_proxy(), eq_result.as_proxy()),
                        {},
                    )
                    sym_eq_acc = SymNodeVariable.create(tx, proxy, sym_num=None)
            else:
                unimplemented(
                    gb_type="list_richcompare_nonconst",
                    context=f"element comparison produced {type(eq_result).__name__}",
                    explanation="Cannot determine list/tuple comparison at compile time when element comparison produces a non-constant result.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )

        if sym_eq_acc is not None:
            if cmp_op is operator.ne:
                proxy = tx.output.create_proxy(
                    "call_function",
                    operator.not_,
                    (sym_eq_acc.as_proxy(),),
                    {},
                )
                return SymNodeVariable.create(tx, proxy, sym_num=None)
            return sym_eq_acc

        return ConstantVariable.create(cmp_op(len(left), len(right)))

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "index":
            if not len(args):
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            try:
                # Speedup trace times for constant data structures
                items = [item.as_python_constant() for item in self.items]
                const_args = [arg.as_python_constant() for arg in args]
                const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
                try:
                    return VariableTracker.build(
                        tx, items.index(*const_args, **const_kwargs)
                    )
                except ValueError:
                    raise_observed_exception(
                        ValueError,
                        tx,
                        args=["tuple.index()"],
                    )
            except AsPythonConstantNotImplementedError:
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.index),
                    [self] + list(args),
                    kwargs,
                )
        elif name == "count":
            if len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return VariableTracker.build(tx, operator.countOf).call_function(
                tx,
                [self, args[0]],
                kwargs,
            )
        elif name == "__reversed__":
            # list/tuple/namedtuple __reversed__: reverse iterator over items.
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return ListIteratorVariable(
                list(reversed(self.items)),
                mutation_type=ValueMutationNew(),
            )
        return super().call_method(tx, name, args, kwargs)


class RangeVariable(BaseListVariable):
    # PyRange_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/rangeobject.c#L767
    _cpython_type = range

    def __init__(self, items: list[VariableTracker], **kwargs: Any) -> None:
        items_to_map = items
        start = variables.ConstantVariable.create(0)
        stop = None
        step = variables.ConstantVariable.create(1)

        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError

        def maybe_as_int(x: VariableTracker) -> VariableTracker:
            return (
                ConstantVariable.create(int(x.as_python_constant()))
                if x.is_python_constant()
                else x
            )

        # cast each argument to an integer
        start = maybe_as_int(start)
        step = maybe_as_int(step)
        stop = maybe_as_int(stop)

        if stop is None:
            raise AssertionError("stop must not be None after parsing range arguments")
        super().__init__([start, stop, step], **kwargs)

    def debug_repr(self) -> str:
        repr = f"range({self.start()}, {self.stop()}"
        if self.step() != 1:
            repr += f", {self.step()}"
        repr += ")"
        return repr

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: range_repr in https://github.com/python/cpython/blob/6280bb547840b609feedb78887c6491af75548e8/Objects/rangeobject.c#L673-L691
        return VariableTracker.build(tx, self.debug_repr())

    def python_type(self) -> type:
        return range

    # range math needs concrete bounds; with assume_static_by_default=False a
    # range object's start/stop/step are wrapped as symbolic ints, so specialize
    # them (installing a guard) rather than assuming a plain constant.
    def start(self) -> int:
        return guard_if_dyn(self.items[0])

    def stop(self) -> int:
        return guard_if_dyn(self.items[1])

    def step(self) -> int:
        return guard_if_dyn(self.items[2])

    def range_length(self) -> int:
        lo = self.start()
        hi = self.stop()
        step = self.step()

        if step == 0:
            raise AssertionError("step must not be zero")
        if step > 0 and lo < hi:
            return 1 + (hi - 1 - lo) // step
        elif step < 0 and lo > hi:
            return 1 + (lo - 1 - hi) // (0 - step)
        else:
            return 0

    def _get_slice_indices(self, length: int, slice: slice) -> list[int]:
        step_is_negative = 0

        if slice.step is None:
            step = 1
            step_is_negative = False
        else:
            step = slice.step
            step_is_negative = slice.step < 0

        # Find lower and upper bounds for start and stop.
        if step_is_negative:
            lower = -1
            upper = length + lower
        else:
            lower = 0
            upper = length

        # Compute start
        if slice.start is None:
            start = upper if step_is_negative else lower
        else:
            start = slice.start

        if start < 0:
            start += length
            if start < lower:
                start = lower
        else:
            if start > upper:
                start = upper

        # Compute stop.
        if slice.stop is None:
            stop = lower if step_is_negative else upper

        else:
            stop = slice.stop

            if stop < 0:
                stop += length
                if stop < lower:
                    stop = lower
            else:
                if stop > upper:
                    stop = upper

        return [start, stop, step]

    def apply_index(
        self, tx: "InstructionTranslatorBase", index: int
    ) -> VariableTracker:
        length = self.range_length()
        if index < 0:
            index = length + index

        if index < 0 or index >= length:
            raise_observed_exception(
                IndexError,
                tx,
                args=["range object index out of range"],
            )

        return VariableTracker.build(tx, self.start() + (index * self.step()))

    def apply_slice(self, slice: slice) -> "RangeVariable":
        (slice_start, slice_stop, slice_step) = self._get_slice_indices(
            self.range_length(), slice
        )

        def compute_item(index: int) -> int:
            return self.start() + (index * self.step())

        sub_step = self.step() * slice_step
        sub_start = compute_item(slice_start)
        sub_stop = compute_item(slice_stop)

        result = RangeVariable(
            [
                variables.ConstantVariable.create(x)
                for x in [sub_start, sub_stop, sub_step]
            ],
            mutation_type=ValueMutationNew() if self.mutation_type else None,
        )
        return result

    def as_python_constant(self) -> range:
        return range(*[x.as_python_constant() for x in self.items])

    def getitem_const(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # range_subscript: https://github.com/python/cpython/blob/62a6e898e01/Objects/rangeobject.c#L729-L748
        arg_type = maybe_get_python_type(arg)
        if pyindex_check(arg_type):
            i = pynumber_index(tx, arg)
            return self.apply_index(tx, i.as_python_constant())
        elif pyslice_check(arg):
            if not isinstance(arg, SliceVariable):
                raise AssertionError("Expected arg to be a SliceVariable")
            return self.apply_slice(arg.as_index_slice(tx))
        raise_type_error(
            tx,
            f"range indices must be integers or slices, not {arg.python_type_name()}",
        )

    def as_proxy(self) -> range:
        return self.python_type()(*self._as_proxy())

    def unpack_var_sequence(
        self, tx: Optional["InstructionTranslatorBase"] = None
    ) -> list[VariableTracker]:
        rng = range(self.start(), self.stop(), self.step())
        return [variables.ConstantVariable.create(x) for x in rng]

    def sq_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        """Sequence length for range objects."""
        length = self.range_length()
        if length > sys.maxsize:
            raise_observed_exception(OverflowError, tx)
        return VariableTracker.build(tx, length)

    def mp_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return self.sq_length_impl(tx)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if "range" in codegen.tx.f_globals:
            raise AssertionError("'range' must not be shadowed in f_globals")
        codegen.add_push_null(
            lambda: codegen.append_output(codegen.create_load_python_module(range))  # type: ignore[arg-type]
        )
        codegen.foreach(self.items)
        codegen.extend_output(create_call_function(3, False))

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.python_type() is range:
            return VariableTracker.build(tx, name in range.__dict__)
        return super().call_obj_hasattr(tx, name)

    def range_equals(self, other: "RangeVariable") -> bool:
        # ref: https://github.com/python/cpython/blob/62a6e898e017c9878490544f6a227b8a187a949c/Objects/rangeobject.c#L514-L553  (range_equals)
        r0, r1 = self, other
        if r0 is r1:
            return True
        if r0.range_length() != r1.range_length():
            return False
        if r0.range_length() == 0:
            return True
        if r0.start() != r1.start():
            return False
        if r0.range_length() == 1:
            return True
        return r0.step() == r1.step()

    def range_count(self, x: VariableTracker) -> int:
        # Based on CPython
        # https://github.com/guilhermeleobas/cpython/blob/baefaa6cba1d69efd2f930cdc56bca682c54b139/Objects/rangeobject.c#L442-L486
        x = x.as_python_constant()
        if type(x) not in (bool, int, float):
            return 0

        start, stop, step = self.start(), self.stop(), self.step()

        if step == 0:
            return 0

        in_range = (start <= x < stop) if step > 0 else (stop < x <= start)

        if in_range:
            re = ((x - start) % step) == 0
            return int(re)
        return 0

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # range_subscript: https://github.com/python/cpython/blob/62a6e898e01/Objects/rangeobject.c#L729-L748
        return self.getitem_const(tx, key)

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # range_contains: https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Objects/rangeobject.c#L482-L490
        if maybe_get_python_type(item) in (int, bool):
            return VariableTracker.build(tx, bool(self.range_count(item)))
        return iter_contains(self.unpack_var_sequence(tx), item, tx)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/rangeobject.c#L896-L927
        if not all(var.is_python_constant() for var in self.items):
            # Can't represent a `range_iterator` without well defined bounds
            return variables.misc.DelayGraphBreakVariable(
                msg="Cannot create range_iterator: bounds (start, stop, step) must be fully defined as concrete constants.",
            )
        return RangeIteratorVariable(
            self.start(), self.stop(), self.step(), self.range_length()
        )

    def sq_item_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # range_item: https://github.com/python/cpython/blob/62a6e898e01/Objects/rangeobject.c#L405-L416
        # CPython's sq_item takes Py_ssize_t (already int from generic_getitem's
        # nb_index_impl).  Unlike mp_subscript (range_subscript), no slices.
        index = key.as_python_constant()
        return self.apply_index(tx, index)

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """range_richcompare: eq/ne via range_equals, NotImplemented for ordering.

        https://github.com/python/cpython/blob/e76aa128fe/Objects/rangeobject.c#L472
        """
        from .builder import SourcelessBuilder

        if op not in ("__eq__", "__ne__"):
            return ConstantVariable.create(NotImplemented)
        try:
            if other.python_type() is not range:
                return ConstantVariable.create(NotImplemented)
        except NotImplementedError:
            return ConstantVariable.create(NotImplemented)

        if isinstance(other, RangeVariable):
            cmp = self.range_equals(other)
        else:
            cmp = False

        if op == "__eq__":
            return SourcelessBuilder.create(tx, cmp)
        else:
            return SourcelessBuilder.create(tx, not cmp)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import SourcelessBuilder

        if name == "count":
            return SourcelessBuilder.create(tx, self.range_count(*args))
        elif name == "__reversed__":
            # range.__reversed__: range_iterator with reversed bounds.
            # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/rangeobject.c (range_reverse)
            if args or kwargs:
