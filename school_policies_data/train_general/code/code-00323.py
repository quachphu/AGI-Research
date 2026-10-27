        with torch._dynamo.variables.higher_order_ops.dynamo_allow_side_effects_in_hop(
            tx
        ):
            self.old_parameters_var = self.mod.tp_getattro_impl(
                tx, "_parameters"
            ).realize()
            self.old_buffer_var = self.mod.tp_getattro_impl(tx, "_buffers").realize()
            tx.output.side_effects.ignore_mutations_on(self.old_parameters_var)
            tx.output.side_effects.ignore_mutations_on(self.old_buffer_var)
            return self.cm_vt.enter(tx)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        # Custom enter implementation with side effects. Additionally, do not
        # worry about side effects for the hop, as they cancel each other out.
        with torch._dynamo.variables.higher_order_ops.dynamo_allow_side_effects_in_hop(
            tx
        ):
            x = self.cm_vt.exit(tx, *args)
            tx.output.side_effects.stop_ignoring_mutations_on(self.old_buffer_var)
            tx.output.side_effects.stop_ignoring_mutations_on(self.old_parameters_var)
            return x

    # Forward all other method calls to self.cm_vt
    def __getattr__(self, name: str) -> Any:
        # This will be called for any attribute not explicitly defined in this class
        return getattr(self.cm_vt, name)


class GradInplaceRequiresGradCtxManagerVariable(ContextWrappingVariable):
    """represents torch grad requires grad"""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", target_values: Any, **kwargs: Any
    ) -> "GradInplaceRequiresGradCtxManagerVariable":
        return GradInplaceRequiresGradCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        [enabled] = self.target_values
        self.prev_state = torch._C._functorch.get_inplace_requires_grad_allowed()
        torch._C._functorch.set_inplace_requires_grad_allowed(enabled)
        self.set_cleanup_hook(
            tx,
            lambda: torch._C._functorch.set_inplace_requires_grad_allowed(
                self.prev_state
            ),
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch.set_inplace_requires_grad_allowed,
            (enabled,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._functorch.set_inplace_requires_grad_allowed,
            (self.prev_state,),
            {},
        )
        return variables.ConstantVariable.create(None)


class TemporarilyPopInterpreterStackCtxManagerVariable(ContextWrappingVariable):
    """represents torch._functorch.pyfunction.temporarily_pop_interpreter_stack()"""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", target_values: Any, **kwargs: Any
    ) -> "TemporarilyPopInterpreterStackCtxManagerVariable":
        return TemporarilyPopInterpreterStackCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        self.saved = torch._C._functorch.pop_dynamic_layer_stack()
        self.set_cleanup_hook(
            tx,
            lambda: torch._C._functorch.push_dynamic_layer_stack(self.saved),
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch.pop_dynamic_layer_stack,
            (),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._functorch.push_dynamic_layer_stack,
            (self.proxy,),
            {},
        )
        return variables.ConstantVariable.create(None)


class JvpIncrementNestingCtxManagerVariable(ContextWrappingVariable):
    """represents torch.func.jvp increment/decrement nesting"""

    # A guard is needed as the grad level is baked into the torch FX graph
    # This is fine if jvp is only called from within the function
    # being compiled. But the FX graph may be invalid in the case of a jvp
    # call from eager that calls the compiled function, as the jvp levels
    # may be different.
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", **kwargs: Any
    ) -> "JvpIncrementNestingCtxManagerVariable":
        var = JvpIncrementNestingCtxManagerVariable(
            target_values=None,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        install_guard(self._guards_singleton)
        jvp_level = torch._functorch.eager_transforms.enter_jvp_nesting()
        self.set_cleanup_hook(
            tx, lambda: torch._functorch.eager_transforms.exit_jvp_nesting()
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._functorch.predispatch._jvp_increment_nesting,
            (),
            {},
        )
        return variables.VariableTracker.build(tx, jvp_level)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function", torch._functorch.predispatch._jvp_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class SetFwdGradEnabledContextManager(ContextWrappingVariable):
    """represents torch.autograd.forward_ad._set_fwd_grad_enabled() to enable/disable fwd grad"""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", target_values: Any, **kwargs: Any
    ) -> "SetFwdGradEnabledContextManager":
        return SetFwdGradEnabledContextManager(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        [mode] = self.target_values
        self.prev_state = torch._C._is_fwd_grad_enabled()
        torch._C._set_fwd_grad_enabled(mode)
        self.set_cleanup_hook(
            tx,
            lambda: torch._C._set_fwd_grad_enabled(self.prev_state),
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._set_fwd_grad_enabled,
            (mode,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._C._set_fwd_grad_enabled,
            (self.prev_state,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def python_type(self) -> type:
        return torch.autograd.forward_ad._set_fwd_grad_enabled


class DualLevelContextManager(ContextWrappingVariable):
    """Represents torch.autograd.forward_ad.dual_level ctx manager"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.DUAL_LEVEL)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", **kwargs: Any
    ) -> "DualLevelContextManager":
        return DualLevelContextManager(
            target_values=None,
            initial_values=None,
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        install_guard(self._guards_singleton)
        self.new_level = torch.autograd.forward_ad.enter_dual_level()
        self.set_cleanup_hook(
            tx, lambda: torch.autograd.forward_ad.exit_dual_level(level=self.new_level)
        )
        self.proxy = tx.output.create_node(
            "call_function",
            torch._functorch.predispatch._enter_dual_level,
            (),
            {},
        )
        return variables.VariableTracker.build(tx, self.new_level)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._functorch.predispatch._exit_dual_level,
            (),
            {"level": self.new_level},
        )
        return variables.ConstantVariable.create(None)

    def python_type(self) -> type:
        return torch.autograd.forward_ad.dual_level


class GradIncrementNestingCtxManagerVariable(ContextWrappingVariable):
    """represents torch.func.grad increment/decrement nesting"""

    # A guard is needed as the grad level is baked into the torch FX graph
    # This is fine if grad is only called from within the function
    # being compiled. But the FX graph may be invalid in the case of a grad
    # call from eager that calls the compiled function, as the grad levels
    # may be different.
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", **kwargs: Any
    ) -> "GradIncrementNestingCtxManagerVariable":
        var = GradIncrementNestingCtxManagerVariable(
            target_values=None,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        install_guard(self._guards_singleton)
        grad_level = torch._C._functorch._grad_increment_nesting()
        self.set_cleanup_hook(tx, lambda: torch._C._functorch._grad_decrement_nesting())
        self.proxy = tx.output.create_node(
            "call_function",
            torch._C._functorch._grad_increment_nesting,
            (),
            {},
        )
        return variables.VariableTracker.build(tx, grad_level)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function", torch._C._functorch._grad_decrement_nesting, (), {}
        )
        return variables.ConstantVariable.create(None)


class CatchWarningsCtxManagerVariable(ContextWrappingVariable):
    """Delay a call to warnings.catch_warnings"""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", catch_warnings_args: dict[str, VariableTracker]
    ) -> "CatchWarningsCtxManagerVariable":
        return CatchWarningsCtxManagerVariable(
            catch_warnings_args=catch_warnings_args,
            target_values=None,
            initial_values=None,
        )

    def __init__(
        self,
        catch_warnings_args: dict[str, VariableTracker],
        target_values: Any | None = None,
        initial_values: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(catch_warnings_args, dict):
            raise AssertionError(
                f"catch_warnings_args must be a dict, got {catch_warnings_args}"
            )
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.catch_warnings_args = catch_warnings_args

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        kwargs = {
            k: v.as_python_constant() for k, v in self.catch_warnings_args.items()
        }
        ctx_val = warnings.catch_warnings(**kwargs)
        self.set_cleanup_hook(tx, lambda: ctx_val.__exit__(None, None, None))
        return variables.VariableTracker.build(tx, ctx_val.__enter__())

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("warnings", "catch_warnings")
        )
        codegen.foreach(self.catch_warnings_args.values())
        keys = tuple(self.catch_warnings_args.keys())
        codegen.extend_output(codegen.create_call_function_kw(len(keys), keys, False))

    def python_type(self) -> type:
        return warnings.catch_warnings


class VmapIncrementNestingCtxManagerVariable(ContextWrappingVariable):
    """represents torch VMap increment/decrement nesting"""

    # A guard is needed as the vmap level is baked into the torch FX graph
    # generated. This is fine if vmap is only called from within the function
    # being compiled. But the FX graph may be invalid in the case of a vmap
    # call from eager that calls the compiled function, as the vmap levels
    # may be different.
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FUNCTORCH_STACK_MATCH)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        target_values: list[VariableTracker],
        **kwargs: Any,
    ) -> "VmapIncrementNestingCtxManagerVariable":
        var = VmapIncrementNestingCtxManagerVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )
        return var

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        install_guard(self._guards_singleton)
        batch_size, randomness = self.target_values
        if isinstance(batch_size, variables.SymNodeVariable):
            batch_size_value = batch_size.sym_num
        else:
            batch_size_value = batch_size.as_python_constant()
        randomness = randomness.as_python_constant()
        vmap_level = torch._C._functorch._vmap_increment_nesting(
            batch_size_value, randomness
        )
        self.set_cleanup_hook(tx, lambda: torch._C._functorch._vmap_decrement_nesting())
        self.proxy = tx.output.create_proxy(
            "call_function",
            torch._functorch.predispatch._vmap_increment_nesting,
            (batch_size.as_proxy(), randomness),
            {},
        )
        return variables.VariableTracker.build(tx, vmap_level)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup()
        tx.output.create_node(
            "call_function",
            torch._functorch.predispatch._vmap_decrement_nesting,
            (),
            {},
        )
        return variables.ConstantVariable.create(None)


class GradModeVariable(ContextWrappingVariable):
    """represents torch.{no_grad,enable_grad,set_grad_mode}()"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.GRAD_MODE)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        target_value: Any,
        initialized: bool = False,
        **kwargs: Any,
    ) -> "GradModeVariable":
        var = GradModeVariable(
            target_values=[target_value],
            initial_values=[torch.is_grad_enabled()],
            **kwargs,
        )
        if initialized:
            var._call_func(tx, var.target_values)
        return var

    def __init__(
        self,
        target_values: Any,
        initial_values: Sequence[bool] | None = None,
        initialized: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self._call_func(tx, self.initial_values)
        return variables.ConstantVariable.create(None)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self._call_func(tx, self.initial_values)  # undo eager initialization
        return super().call_function(tx, args, kwargs)

    def _call_func(self, tx: "InstructionTranslatorBase", values: Any) -> None:
        if len(values) != 1:
            raise AssertionError(f"expected 1 value, got {len(values)}")
        value = values[0]
        # Coalesce grad mode mutations
        if torch.is_grad_enabled() != value:
            tx.output.create_node(
                "call_function", torch._C._set_grad_enabled, (value,), {}
            )
            torch._C._set_grad_enabled(value)

    def module_name(self) -> str:
        return "torch"

    def fn_name(self) -> str:
        return "set_grad_enabled"

    def python_type(self) -> type:
        return torch.set_grad_enabled


class InferenceModeVariable(ContextWrappingVariable):
    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", target_value: Any, **kwargs: Any
    ) -> "InferenceModeVariable":
        var = InferenceModeVariable(
            [target_value], initial_values=torch.is_inference_mode_enabled(), **kwargs
        )
        return var

    def __init__(
        self,
        target_values: Any,
        initial_values: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if initial_values is None:
            # This must be called here since function defaults are evaluated at import time
            initial_values = torch.is_inference_mode_enabled()
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        tx.output.create_node(
            "call_function",
            torch.autograd.grad_mode._exit_inference_mode,
            (self.proxy,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        disabled_inference_mode_forcibly = False
        if (
            torch._dynamo.config.fake_tensor_disable_inference_mode
            and self.target_values[0]
        ):
            # Do not set the inference mode because we keep it off during
            # compilation. Set the grad_enabled to False to reflect the relevant
            # part of inference_mode to torch.compile.
            disabled_inference_mode_forcibly = True
            prior = torch.is_grad_enabled()
            torch._C._set_grad_enabled(False)
        else:
            ctx = torch.autograd.grad_mode._enter_inference_mode(*self.target_values)

        def cleanup_hook() -> None:
            if disabled_inference_mode_forcibly:
                torch._C._set_grad_enabled(prior)
            else:
                torch.autograd.grad_mode._exit_inference_mode(ctx)

        self.set_cleanup_hook(tx, cleanup_hook)
        self.proxy = tx.output.create_node(
            "call_function",
            torch.autograd.grad_mode._enter_inference_mode,
            (*self.target_values,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch"

    def fn_name(self) -> str:
        return "inference_mode"

    def python_type(self) -> type:
        return torch.inference_mode


class GenericDeviceVariable(ContextWrappingVariable):
    """Abstract base for device context managers that swap the active device index."""

    _exchange_fn: Any
    _maybe_exchange_fn: Any
    _get_device_index_fn: Any

    @classmethod
    def create(
        cls, tx: "InstructionTranslatorBase", device: Any, **kwargs: Any
    ) -> "GenericDeviceVariable":
        return cls(
            target_values=[cls._get_device_index_fn(device, optional=True)],
            initial_values=None,
            **kwargs,
        )

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        tx.output.create_node(
            "call_function",
            self._maybe_exchange_fn,
            (self.proxy,),
            {},
        )
        return variables.ConstantVariable.create(False)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        prev_idx = self._exchange_fn(*self.target_values)
        self.set_cleanup_hook(tx, lambda: self._maybe_exchange_fn(prev_idx))
        self.proxy = tx.output.create_node(
            "call_function",
            self._exchange_fn,
            (*self.target_values,),
            {},
        )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        raise NotImplementedError

    def fn_name(self) -> str:
        return "device"

    def python_type(self) -> type:
        raise NotImplementedError


class CUDADeviceVariable(GenericDeviceVariable):
    """represents torch.cuda.device"""

    _exchange_fn = staticmethod(torch.cuda._exchange_device)
    _maybe_exchange_fn = staticmethod(torch.cuda._maybe_exchange_device)
    _get_device_index_fn = staticmethod(torch.cuda._get_device_index)

    def module_name(self) -> str:
        return "torch.cuda"

    def python_type(self) -> type:
        return torch.cuda.device


class XPUDeviceVariable(GenericDeviceVariable):
    """represents torch.xpu.device"""

    _exchange_fn = staticmethod(torch.xpu._exchange_device)
    _maybe_exchange_fn = staticmethod(torch.xpu._maybe_exchange_device)
    _get_device_index_fn = staticmethod(torch.xpu._get_device_index)

    def module_name(self) -> str:
        return "torch.xpu"

    def python_type(self) -> type:
        return torch.xpu.device


class AcceleratorDeviceIndexVariable(GenericDeviceVariable):
    """represents torch.accelerator.device_index"""

    _exchange_fn = staticmethod(torch._C._accelerator_exchangeDevice)
    _maybe_exchange_fn = staticmethod(torch._C._accelerator_maybeExchangeDevice)
    _get_device_index_fn = staticmethod(lambda device, **kwargs: device)

    def module_name(self) -> str:
        return "torch.accelerator"

    def fn_name(self) -> str:
        return "device_index"

    def python_type(self) -> type:
        return torch.accelerator.device_index


_device_context_manager_map: dict[type, type[GenericDeviceVariable]] = {
    torch.cuda.device: CUDADeviceVariable,
    torch.xpu.device: XPUDeviceVariable,
    torch.accelerator.device_index: AcceleratorDeviceIndexVariable,
}


def register_device_context_manager(
    device_context: type, variable_cls: type[GenericDeviceVariable]
) -> None:
    """Register a Dynamo variable class for a device context manager type.

    Raises ``ValueError`` if ``device_context`` is already registered.
    """
    if device_context in _device_context_manager_map:
        raise ValueError(
            f"Device context {device_context} already has a registered context manager"
        )
    _device_context_manager_map[device_context] = variable_cls


def get_device_context_manager(
    device_context: type,
) -> type[GenericDeviceVariable] | None:
    """Return the Dynamo variable class for ``device_context``, or ``None`` if not registered."""
    return _device_context_manager_map.get(device_context)


class TorchFunctionDisableVariable(ContextWrappingVariable):
    """represents whether torch function overrides are enabled or not"""

    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.TORCH_FUNCTION_STATE)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", **kwargs: Any
    ) -> "TorchFunctionDisableVariable":
        var = TorchFunctionDisableVariable(
            tx,
            target_values=[],
            initial_values=[],
            **kwargs,
        )
        return var

    def __init__(
        self,
        tx: "InstructionTranslatorBase",
        target_values: Sized,
        initial_values: Sized | None = None,
        only_subclass: bool = True,
        **kwargs: Any,
    ) -> None:
        if len(target_values) != 0:
            raise AssertionError(f"expected 0 target_values, got {len(target_values)}")
        if initial_values is None:
            raise AssertionError("initial_values must not be None")
        if len(initial_values) != 0:
            raise AssertionError(
                f"expected 0 initial_values, got {len(initial_values)}"
            )
        self.only_subclass = only_subclass
        self.initial_torch_function_subclass_enabled = (
            tx.symbolic_torch_function_state.torch_function_subclass_enabled
        )
        self.initial_torch_function_mode_enabled = (
            tx.symbolic_torch_function_state.torch_function_mode_enabled
        )

        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        install_guard(self._guards_singleton)

    def set_cleanup_hook(
        self,
        tx: "InstructionTranslatorBase",
        fn: Callable[..., Any] | None = None,
    ) -> None:
        if fn is None:

            def fn() -> None:
                tx.symbolic_torch_function_state.torch_function_subclass_enabled = (
                    self.initial_torch_function_subclass_enabled
                )
                if not self.only_subclass:
                    tx.symbolic_torch_function_state.torch_function_mode_enabled = (
                        self.initial_torch_function_mode_enabled
                    )

        self.cleanup_fn = fn
        tx.output.add_cleanup_hook(self.cleanup)

    def _call_func(self, tx: "InstructionTranslatorBase", values: Sized) -> None:
        if len(values) != 0:
            raise AssertionError(f"expected 0 values, got {len(values)}")
        tx.symbolic_torch_function_state.torch_function_subclass_enabled = False
        if not self.only_subclass:
            tx.symbolic_torch_function_state.torch_function_mode_enabled = False

    def module_name(self) -> str:
        return "torch._C"

    def fn_name(self) -> str:
        if self.only_subclass:
            return "DisableTorchFunctionSubclass"
        return "DisableTorchFunction"

    def python_type(self) -> type:
        if self.only_subclass:
            return torch._C.DisableTorchFunctionSubclass  # pyrefly: ignore[bad-return]
        return torch._C.DisableTorchFunction  # pyrefly: ignore[bad-return]


class DisabledSavedTensorsHooksVariable(ContextWrappingVariable):
    """represents torch.autograd.graph.disable_saved_tensors_hook."""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", target_value: str | None, **kwargs: Any
    ) -> "DisabledSavedTensorsHooksVariable":
        var = DisabledSavedTensorsHooksVariable(
            target_values=[target_value],
            initial_values=[
                torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
            ],
            **kwargs,
        )
        var._call_func(tx, [target_value])
        var.set_cleanup_hook(tx)
        return var

    def __init__(
        self,
        target_values: Sequence[str | None],
        initial_values: Sequence[str | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def _call_func(
        self, tx: "InstructionTranslatorBase", values: Sequence[str | None]
    ) -> None:
        if len(values) != 1:
            raise AssertionError(f"expected 1 value, got {len(values)}")
        value = values[0]
        if value is not None:
            # Disable `saved_tensors_hooks` with message (`value`)
            # OR
            # we are exiting this context and restoring the previous message.
            tx.output.create_node(
                "call_function",
                torch._C._autograd._saved_tensors_hooks_disable,
                (value,),
                {},
            )
            torch._C._autograd._saved_tensors_hooks_disable(value)
        else:
            # We are exiting this context and if prev_message was None, we re-enable `saved_tensors_hooks`.
            tx.output.create_node(
                "call_function", torch._C._autograd._saved_tensors_hooks_enable, (), {}
            )
            torch._C._autograd._saved_tensors_hooks_enable()

    def module_name(self) -> str:
        return "torch.autograd.graph"

    def fn_name(self) -> str:
        return "disable_saved_tensors_hooks"

    def python_type(self) -> type:
        return contextlib._GeneratorContextManager


class AutocastModeVariable(ContextWrappingVariable):
    @staticmethod
    def create(
        func: torch.amp.autocast_mode.autocast,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> "AutocastModeVariable":
        if func not in [
            torch.amp.autocast_mode.autocast,
            torch.cuda.amp.autocast,
            torch.cpu.amp.autocast,
        ]:
            raise AssertionError(f"unexpected autocast function: {func}")
        # device_type : str,
        # dtype : Optional[_dtype] = None,
        # enabled : bool = True,
        # cache_enabled : Optional[bool] = None):cache_enabled
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()
        target_values = []
        kwargs.clear()

        for key in ["device_type", "dtype", "enabled", "cache_enabled"]:
            if key == "device_type" and func in [
                torch.cuda.amp.autocast,
                torch.cpu.amp.autocast,
            ]:
                # pyrefly: ignore [unnecessary-comparison]
                arg = "cuda" if func is torch.cuda.amp.autocast else "cpu"
            else:
                arg = bound_args.arguments[key]
            if isinstance(arg, VariableTracker):
                target_values.append(arg.as_python_constant())
            else:
                target_values.append(arg)

        var = AutocastModeVariable(target_values, initial_values=None, **kwargs)
        return var

    def __init__(
        self,
        target_values: Sequence[Any],
        initial_values: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        tx.output.create_node(
            "call_function", torch.amp._exit_autocast, (self.proxy,), {}
        )
        return variables.ConstantVariable.create(None)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        ctx = torch.amp._enter_autocast(*self.target_values)
        self.set_cleanup_hook(tx, lambda: torch.amp._exit_autocast(ctx))
        self.proxy = tx.output.create_node(
            "call_function", torch.amp._enter_autocast, (*self.target_values,), {}
        )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.amp.autocast_mode"

    def fn_name(self) -> str:
        return "autocast"

    def python_type(self) -> type:
        return torch.amp.autocast_mode.autocast


class NullContextVariable(ContextWrappingVariable):
    """
    This class represents Python contextlib.nullcontext.
    """

    def __init__(
        self, enter_result: VariableTracker | None = None, **kwargs: Any
    ) -> None:
        self.enter_result = enter_result
        super().__init__(target_values=(), **kwargs)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if self.enter_result is None:
            return variables.ConstantVariable.create(None)
        return self.enter_result

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "contextlib"

    def fn_name(self) -> str:
        return "nullcontext"

    def python_type(self) -> type:
        return contextlib.nullcontext


class ProfilerContextVariable(ContextWrappingVariable):
    """
    This class represents a set of torch profiler context objects, where Dynamo
    ignores all the side-effects in the __init__, __enter__ and __exit__ methods
    by treating the object mostly as a `contextlib.nullcontext`, except for edge
    cases like the `__enter__` method which returns the object itself rather
    than `None`, per implementation of the torch objects.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(target_values=None, **kwargs)

    def python_type(self) -> type:
        return torch.profiler.profile

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return self

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "contextlib"

    def fn_name(self) -> str:
        return "nullcontext"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        unimplemented(
            gb_type="torch.profiler object escaped from compiled region",
            context=str(self),
            explanation="Dynamo doesn't support compiling a region that returns a torch.profiler context manager.",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class ProfilerRecordFunctionContextVariable(ContextWrappingVariable):
    """
    This class represents torch profiler context objects.

    For record_function: emits torch.ops.profiler._record_function_enter_new
    to the graph on enter, and torch.ops.profiler._record_function_exit on exit.
    But if emit_profiler_ops=False, behaves like nullcontext.

    For profile: behaves like nullcontext, ignoring all side-effects.
    """

    _nonvar_fields = {
        "emit_profiler_ops",
        *ContextWrappingVariable._nonvar_fields,
    }

    @staticmethod
    def create(
        func: Any,
        record_args: list[VariableTracker],
        record_kwargs: "dict[str, VariableTracker]",
        **kwargs: Any,
    ) -> "ProfilerRecordFunctionContextVariable":
        target_values = None
        if config.capture_profiler_record_function:
            # Extract name and args for record_function
            # record_function(name: str, args: Optional[str] = None)
            name = (
                record_args[0].as_python_constant()
                if record_args
                else record_kwargs.get(
                    "name", variables.ConstantVariable.create("unknown")
                ).as_python_constant()
            )
            record_args_const = None
            if len(record_args) > 1:
                record_args_const = record_args[1].as_python_constant()
            elif "args" in record_kwargs:
                record_args_const = record_kwargs["args"].as_python_constant()
            target_values = [name, record_args_const]
        else:
            warning_once(log, "Profiler record function %s will be ignored", func)
        return ProfilerRecordFunctionContextVariable(
            target_values=target_values,
            initial_values=None,
            **kwargs,
        )

    def __init__(
        self,
        target_values: Any = None,
        initial_values: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )

    def python_type(self) -> type:
        return torch.autograd.profiler.record_function

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if config.capture_profiler_record_function:
            name, args = self.target_values
            # Create the profiler entry node in the graph
            self.proxy = tx.output.create_node(
                "call_function",
                torch.ops.profiler._record_function_enter_new,
                (name, args),
                {},
            )
        return self

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        if config.capture_profiler_record_function:
            # Create the profiler exit node in the graph
            tx.output.create_node(
                "call_function",
                torch.ops.profiler._record_function_exit._RecordFunction,
                (self.proxy,),
                {},
            )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return (
            "torch.autograd.profiler"
            if config.capture_profiler_record_function
            else "contextlib"
        )

    def fn_name(self) -> str:
        return (
            "record_function"
            if config.capture_profiler_record_function
            else "nullcontext"
        )

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        # This will be called if we try to reconstruct the record_function type
        # when there's a graph break. The _set_error_on_graph_break(True) in enter()
        # will cause the graph break to raise an error before we get here.
        if config.capture_profiler_record_function:
            unimplemented(
                gb_type="record_function escaped from compiled region",
                context=str(self),
                explanation="Dynamo doesn't support graph break inside record_function region.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        else:
            # If capture is disabled, allow reconstruction (behaves like nullcontext)
            super().reconstruct_type(codegen)


class PreserveVersionContextVariable(ContextWrappingVariable):
    """
    Wraps torch.autograd._unsafe_preserve_version_counter
    """

    @staticmethod
    def _create_lambda_from_tensors(
        tx: "InstructionTranslatorBase",
        tensors: VariableTracker,
    ) -> "PreserveVersionContextVariable":
        if tensors.is_tensor():
            versions = variables.TupleVariable(
                [x.tp_getattro_impl(tx, "_version") for x in [tensors]]
            )
            tensors_tuple = variables.TupleVariable([tensors])
        else:
            if not isinstance(tensors, variables.TupleVariable):
                raise AssertionError(
                    f"tensors must be a TupleVariable, got {type(tensors)}"
                )
            versions = variables.TupleVariable(
                [x.tp_getattro_impl(tx, "_version") for x in tensors.items]
            )
            tensors_tuple = tensors
        return PreserveVersionContextVariable(tensors_tuple, versions)

    @staticmethod
    def constructor(tx: "InstructionTranslatorBase") -> VariableTracker:
        return variables.LambdaVariable(
            lambda tensors: PreserveVersionContextVariable._create_lambda_from_tensors(
                tx, tensors
            )
        )

    def __init__(
        self,
        tensors: VariableTracker,
        prev_versions: VariableTracker,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("target_values", None)
        super().__init__(**kwargs)
        self.tensors = tensors
        self.prev_versions = prev_versions
        # The context manager accepts Union[Tensor, Tuple[Tensor]]
        if self.tensors.is_tensor():
            self.tensors = variables.TupleVariable([self.tensors])
        if self.prev_versions.is_symnode_like():
            self.prev_versions = variables.TupleVariable([self.prev_versions])

    def python_type(self) -> type:
        return torch.autograd.grad_mode._unsafe_preserve_version_counter

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        from ..tensor_version_op import _unsafe_set_version_counter

        return variables.TorchInGraphFunctionVariable(
            _unsafe_set_version_counter
        ).call_function(tx, [self.tensors, self.prev_versions], {})

    def reconstruct(self, codegen: "PyCodegen") -> None:
        unimplemented(
            gb_type="torch.autograd._unsafe_preserve_version_counter escaped from compiled region",
            context=str(self),
            explanation=(
                "Dynamo doesn't support compiling a region that returns "
                "a torch.autograd._unsafe_preserve_version_counter context manager."
            ),
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class FSDPParamGroupUseTrainingStateVariable(ContextWrappingVariable):
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.FSDP_TRAINING_STATE)  # type: ignore[arg-type]

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        param_group_var: Any,
        target_value: Any,
        **kwargs: Any,
    ) -> "FSDPParamGroupUseTrainingStateVariable":
        var = FSDPParamGroupUseTrainingStateVariable(
            param_group_var=param_group_var,
            target_values=[target_value],
            initial_values=[param_group_var.value._training_state],
            **kwargs,
        )
        return var

    def __init__(
        self,
        param_group_var: Any,
        target_values: Sequence[Any],
        initial_values: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.param_group_var = param_group_var
        install_guard(self._guards_singleton)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        self._call_func(tx, self.target_values)
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self._call_func(tx, self.initial_values)  # type: ignore[arg-type]
        return variables.ConstantVariable.create(None)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # undo eager initialization
        self._call_func(tx, self.initial_values)  # type: ignore[arg-type]
        return super().call_function(tx, args, kwargs)

    def _call_func(
        self, tx: "InstructionTranslatorBase", values: Sequence[Any]
    ) -> None:
        if len(values) != 1:
            raise AssertionError(f"expected 1 value, got {len(values)}")
        value = values[0]
        if self.param_group_var.value._training_state != value:
            self.param_group_var.call_method(
                tx,
                "__setattr__",
                (
                    variables.VariableTracker.build(tx, "_training_state"),
                    variables.VariableTracker.build(tx, value),
                ),
                {},
            )
            self.param_group_var.value._training_state = value

    def module_name(self) -> str:
        return "torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup"

    def fn_name(self) -> str:
        return "use_training_state"


class SDPAKernelVariable(ContextWrappingVariable):
    """represents torch.nn.attention.sdpa_kernel"""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        backends: Any,
        set_priority: bool = False,
        **kwargs: Any,
    ) -> "SDPAKernelVariable":
        if isinstance(backends, torch.nn.attention.SDPBackend):
            backends = [backends]
        var = SDPAKernelVariable(
            target_values=backends,
            initial_values=None,
            set_priority=set_priority,
            **kwargs,
        )
        return var

    def __init__(
        self,
        target_values: list[torch.nn.attention.SDPBackend],
        initial_values: Any = None,
        set_priority: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.set_priority = set_priority

    @staticmethod
    def _backends_to_nodes(
        tx: "InstructionTranslatorBase",
        backends: list[Any],
    ) -> list[Any]:
        # convert to/from string in order to bake the backend into FX graph
        nodes = [
            tx.output.create_node(
                "call_function",
                torch.nn.attention._backend_from_string,
                (backend.name,),
                {},
            )
            for backend in backends
        ]
        return nodes

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        self.prev_backends = torch.nn.attention._cur_sdpa_kernel_backends(
            with_priority=self.set_priority
        )
        self.set_cleanup_hook(
            tx,
            lambda: torch.nn.attention._sdpa_kernel(
                self.prev_backends, set_priority=self.set_priority
            ),
        )
        torch.nn.attention._sdpa_kernel(
            self.target_values, set_priority=self.set_priority
        )
        arg = self._backends_to_nodes(tx, self.target_values)
        tx.output.create_node(
            "call_function",
            torch.nn.attention._sdpa_kernel,
            (arg, bool(self.set_priority)),
            {},
        )
        return variables.ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        self.cleanup_assert()
        arg = self._backends_to_nodes(tx, self.prev_backends)
        tx.output.create_node(
            "call_function",
            torch.nn.attention._sdpa_kernel,
            (arg, bool(self.set_priority)),
            {},
        )
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.nn.attention"

    # use a private version of sdpa_kernel that accepts variadic arguments
    # since dynamo reconstructs the contents of target_values one-by-one
    def fn_name(self) -> str:
        return "_sdpa_kernel_variadic"

    def python_type(self) -> type:
        return contextlib._GeneratorContextManager


class FxTracebackAnnotateVariable(ContextWrappingVariable):
    """
    fx.traceback.annotate is a context manager that allows users to annotate the
    fx graph nodes with custom metadata. In the context of Dynamo, we don't have
    to trace the body of the context manager. Instead we want to directly run
    the body of the context manager, so the Dynamo created Fx graphs have the
    right custom metadata. This variable tracker just runs __enter__ and
    __exit__ method (instead of tracing).
    """

    _nonvar_fields = {
        "annotation",
        *ContextWrappingVariable._nonvar_fields,
    }

    def __init__(
        self, annotation: dict[str, Any], initial_values: Any = None, **kwargs: Any
    ) -> None:
        self.annotation = annotation
        super().__init__(target_values=(), initial_values=initial_values, **kwargs)

    def enter(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        # Run the annotation ctx manager in eager. Also ensure that
        # preserve_node_meta context manager is setup. This is important to pass
        # on the metadata to the create_proxy nodes.
        stack = ExitStack()
        stack.enter_context(torch.fx.traceback.annotate(self.annotation))
        stack.enter_context(torch.fx.traceback.preserve_node_meta())
        self.set_cleanup_hook(tx, lambda: stack.close())
        return variables.ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.fx.traceback"

    def fn_name(self) -> str:
        return "annotate"

    def python_type(self) -> type:
        return contextlib._GeneratorContextManager

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        unimplemented(
            gb_type="torch.fx.traceback.annotate escaped from compiled region",
            context=str(self),
            explanation="Dynamo doesn't support graph break on torch.fx.traceback.annotate.",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )


class DynamoConfigPatchVariable(ContextWrappingVariable):
    """represents torch._dynamo.patch_dynamo_config"""

    # NOTE: no need to guard on dynamo config because dynamo config should not affect soundness
    # (though it may affect tracing behavior)
    def __init__(self, target_values: dict[str, Any], **kwargs: Any) -> None:
        target_values_tuple = tuple(target_values.items())
        super().__init__(
            target_values=(target_values_tuple,), initial_values=None, **kwargs
        )
        # pyrefly: ignore [implicit-any]
        initial_values_dict = {}
        for key, _ in target_values_tuple:
            initial_values_dict[key] = torch._dynamo.config.__getattr__(key)  # type: ignore[attr-defined]
        self.initial_values = (tuple(initial_values_dict.items()),)

    def _call_func(self, tx: "InstructionTranslatorBase", values: Any) -> None:
        if len(values) != 1:
            raise AssertionError(f"expected 1 value, got {len(values)}")
        value = values[0]
        # manually patch dynamo config
        for key, val in value:
            torch._dynamo.config.__setattr__(key, val)  # type: ignore[attr-defined]
        # No need to keep track of global side effects because
        # dynamo will properly restore this context manager for
        # unsupported instructions and continuation functions.
        # Dynamo config also should not affect the semantics of the compiled graph.

    def module_name(self) -> str:
        return "torch._dynamo"

    def fn_name(self) -> str:
        return "patch_dynamo_config"

    def python_type(self) -> type:
        from torch._dynamo.decorators import DynamoConfigPatchProxy

        return DynamoConfigPatchProxy


class ErrorOnGraphBreakVariable(ContextWrappingVariable):
    """represents torch._dynamo.error_on_graph_break"""

    def __init__(self, error_on_graph_break: bool, **kwargs: Any) -> None:
        super().__init__(
            target_values=(error_on_graph_break,),
            initial_values=(_get_error_on_graph_break(),),
            **kwargs,
        )

    def _call_func(
        self, tx: "InstructionTranslatorBase", values: Sequence[bool]
    ) -> None:
        if len(values) != 1:
            raise AssertionError(f"expected 1 value, got {len(values)}")
        _set_error_on_graph_break(values[0])

    def module_name(self) -> str:
        return "torch._dynamo"

    def fn_name(self) -> str:
        return "error_on_graph_break"

    def python_type(self) -> type:
        from torch._dynamo.decorators import ErrorOnGraphBreakDecoratorContextManager

        return ErrorOnGraphBreakDecoratorContextManager


class CudagraphOverrideVariable(ContextWrappingVariable):
    """represents torch._dynamo.override_cudagraphs"""

    def __init__(self, fwd: bool | None, bwd: bool | None, **kwargs: Any) -> None:
        super().__init__(
            target_values=(fwd, bwd),
            initial_values=None,  # Captured in enter()
            **kwargs,
        )

    def enter(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # Capture current annotation before overwriting
        prev = tx.output.cudagraph_annotation
        if prev is not None:
            self.initial_values = (prev.fwd, prev.bwd)
        else:
            self.initial_values = (None,)
        return super().enter(tx)

    def _call_func(
        self, tx: "InstructionTranslatorBase", values: tuple[Any, ...]
    ) -> None:
        from torch._inductor import _CudagraphAnnotation

        if len(values) == 1 and values[0] is None:
            # Restoring to no annotation
            tx.output.cudagraph_annotation = None
            tx.output.tracing_context.cudagraph_annotation = None
        else:
            if len(values) != 2:
                raise AssertionError(f"expected 2 values, got {len(values)}")
            fwd, bwd = values
            annotation = _CudagraphAnnotation(fwd=fwd, bwd=bwd)
            tx.output.cudagraph_annotation = annotation
            tx.output.tracing_context.cudagraph_annotation = annotation

    def module_name(self) -> str:
        return "torch._dynamo"

    def fn_name(self) -> str:
        return "override_cudagraphs"

    def python_type(self) -> type:
        from torch._dynamo.decorators import CudagraphOverrideContextManager

        return CudagraphOverrideContextManager

    def exit_on_graph_break(self) -> bool:
        # Annotation persists until graph is compiled; each resume function
        # will reconstruct the context manager and call enter() again
        return False

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> "VariableTracker":
        # Override to NOT call cleanup here. The cleanup will happen in
        # call_cleanup_hooks() during compile_subgraph, which is AFTER
        # the annotation is copied to gm.meta.
        return variables.ConstantVariable.create(None)


class WithEnterFunctionVariable(VariableTracker):
    def __init__(
        self,
        ctx: ContextWrappingVariable | GenericContextWrappingVariable,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ctx = ctx

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def python_type(self) -> type:
        return types.MethodType

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if args:
            raise AssertionError(
                "WithEnterFunctionVariable.call_function expects no args"
            )
        if kwargs:
            raise AssertionError(
                "WithEnterFunctionVariable.call_function expects no kwargs"
            )
        # NOTE: we assume that the instruction immediately after the current CALL instruction
        # is the first instruction of the block.

        return tx.enter_ctx(self.ctx, tx.current_instruction)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        try:
            type_str = f"{self.ctx.module_name()}.{self.ctx.fn_name()}"
        except NotImplementedError:
            type_str = str(type(self.ctx))
        unimplemented(
            gb_type="Attempted to reconstruct context manager's __enter__ method",
            context=str(self.ctx),
            explanation=f"Attempted to reconstruct context manager {type_str} while tracing `with ...:`",
            hints=[
                "It is likely there is a graph break while tracing `with ctx:` "
                "but outside the actual `ctx.__enter__()` method. "
                "`torch.compile` does not expect this to happen.",
                *graph_break_hints.DIFFICULT,
                *graph_break_hints.DYNAMO_BUG,
            ],
        )


class WithExitFunctionVariable(VariableTracker):
    _nonvar_fields = {
        "target",
        *VariableTracker._nonvar_fields,
    }

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def __init__(
        self,
        ctx: ContextWrappingVariable | GenericContextWrappingVariable,
        target: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(
            ctx, (ContextWrappingVariable, GenericContextWrappingVariable)
        ):
            raise AssertionError(
                f"ctx must be a ContextWrappingVariable or GenericContextWrappingVariable, got {type(ctx)}"
            )
        self.ctx = ctx
        self.target = target

    def python_type(self) -> type:
        return types.MethodType

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if kwargs:
            raise AssertionError(
                "WithExitFunctionVariable.call_function expects no kwargs"
            )
        return self.ctx.exit(tx, *args)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Note here we reconstruct the context manager rather than the
        # exit function.  The handler generated by BlockStackEntry
        # will re-enter the context in the resume function.
        self.ctx.reconstruct_type(codegen)  # type: ignore[union-attr]
<|endoftext|>"""
Utilities for debugging and reproducing issues in Ahead of Time with Inductor (AOTI) compilation.

This file provides tools and utilities for:
- Generating minimal reproducible test cases (minification)
- Handling exported programs and graph modules
- Creating debug repros for AOTI compilation issues
- Supporting both accuracy testing and error reproduction
- Managing configuration and environment for repro cases

The main components include:
- Minification tools to reduce test cases while preserving errors
- Repro generation utilities for exported programs
- Error handling specific to AOTI compilation
- Command-line interface for running and managing repros
"""

import argparse
import functools
import io
import logging
import os
import re
import shutil
import sys
import textwrap
from collections.abc import Sequence
from importlib import import_module
from typing import Any, IO

import torch
from torch._dynamo.debug_utils import (
    _cuda_system_info_comment,
    BuckTargetWriter,
    extra_imports,
    generate_config_string,
    generate_env_vars_string,
    helper_for_dump_minify,
    InputReader,
    minifier_dir,
    NNModuleToString,
    NopInputReader,
)
from torch.export import ExportedProgram
from torch.hub import tqdm


log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()


class AOTIMinifierError(Exception):
    def __init__(self, original_exception: str | Exception) -> None:
        additional_message = "This error is caused by a bug in the AOTI minifier, please report a bug to PyTorch"
        full_message = f"{additional_message}: {str(original_exception)}"
        super().__init__(full_message)
        self.original_exception = original_exception


def dump_to_minify(
    exported_program: ExportedProgram,
    compiler_name: str,
    command: str = "minify",
    options: dict[str, Any] | None = None,
) -> None:
    """
    If command is "minify":
        Dump exported_program to `debug_dir/minifier/minifier_launcher.py`, with minify command.
    If command is "run":
        Dump exported_program to `cwd/repro.py`, with run command.
    """
    if command not in ["minify", "run"]:
        raise AssertionError(f"command must be 'minify' or 'run', got {command!r}")

    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

    if command == "minify":
        out = io.StringIO()
        save_graph_repro_ep(
            out,
            compiler_name,
            exported_program=exported_program,
            save_dir=subdir,
            command="minify",
            config_patches=options,
        )
        return helper_for_dump_minify(out.getvalue())
    else:
        curdir = os.getcwd()
        file_name = os.path.join(curdir, "repro.py")
        try:
            with open(file_name, "w") as fd:
                save_graph_repro_ep(
                    fd,
                    compiler_name,
                    exported_program=exported_program,
                    config_patches=options,
                    save_dir=subdir,
                    command="run",
                    module_in_comment=True,
                )
            log.warning("Writing repro file to %s", file_name)
            if use_buck:
                BuckTargetWriter(file_name).write()
        except OSError:
            log.warning("No write permissions for %s", file_name)


def get_module_string(gm: torch.fx.GraphModule) -> str:
    def _convert_to_comment(s_: str) -> str:
        s = s_.split("\n")
        if len(s) == 1:
            return "# " + s_
        first = s.pop(0)
        for i in range(len(s)):
            line = s[i]
            if line.strip() != "":
                s[i] = "# " + line
            else:
                s[i] = ""
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    module_string = NNModuleToString.convert(gm, allow_unsafe_repr=True)
    return _convert_to_comment(module_string)


def save_graph_repro_ep(
    fd: IO[Any],
    compiler_name: str,
    *,
    exported_program: ExportedProgram | None = None,
    gm: torch.nn.Module | None = None,
    args: tuple[Any] | None = None,
    config_patches: dict[str, str] | None = None,
    stable_output: bool = False,
    save_dir: str | None = None,
    command: str = "run",
    accuracy: str | bool | None = None,
    check_str: str | None = None,
    module_in_comment: bool = False,
    strict: bool = False,
) -> None:
    # Save graph for reproducing the error.
    # Either exported_program or gm will be saved, depending on which one is defined.
    # Only one of exported_program and gm should be defined.

    if exported_program is None and gm is None:
        raise AOTIMinifierError("One of exported_program and gm must be defined")
    if exported_program is not None and gm is not None:
        raise AOTIMinifierError("Only one of exported_program and gm can be defined")
    if gm is not None and args is None:
        raise AOTIMinifierError("If gm is defined, args should also be defined")

    if exported_program is None:
        if gm is None:
            raise AssertionError("gm must not be None when exported_program is None")
        if args is None:
            raise AssertionError("args must not be None when exported_program is None")
        exported_program = torch.export.export(gm, args, strict=strict)
    elif gm is None:
        gm = exported_program.module(check_guards=False)

    # save a graph preview using gm
    module_string = get_module_string(gm)  # type: ignore[arg-type]
    fd.write(module_string)

    # save a graph repro using exported_program
    fd.write(
        generate_compiler_repro_exported_program(
            exported_program,
            options=config_patches,
            stable_output=stable_output,
            save_dir=save_dir,
        )
    )
    if accuracy is None:
        accuracy = "_accuracy" in compiler_name
    fd.write("if __name__ == '__main__':\n")
    fd.write("    from torch._dynamo.repro.aoti import run_repro\n")
    fd.write(
        f"    with torch.no_grad():\n"
        f"        run_repro(exported_program, config_patches=config_patches, accuracy={accuracy!r}, command={command!r}, "
        f"save_dir={save_dir!r}, check_str={check_str!r})\n"
    )


def dump_compiler_graph_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    config_patches: dict[str, str] | None = None,
    accuracy: str | bool | None = None,
    strict: bool = False,
) -> None:
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )
    with open(file_name, "w") as fd:
        save_graph_repro_ep(
            fd,
            compiler_name,
            gm=gm,
            args=tuple(args),
            config_patches=config_patches,
            save_dir=subdir,
            accuracy=accuracy,
            module_in_comment=True,
            strict=strict,
        )
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, "repro.py")
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning("Copying repro file for convenience to %s", repro_path)
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning("No write permissions for %s", repro_path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP REPROS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_compiler_repro_exported_program(
    exported_program: ExportedProgram,
    *,
    options: dict[str, str] | None = None,
    stable_output: bool = False,
    save_dir: str | None = None,
) -> str:
    model_str = textwrap.dedent(
        f"""
{generate_env_vars_string(stable_output=stable_output)}
import torch
import torch._inductor.inductor_prims

{generate_config_string(stable_output=stable_output)}

isolate_fails_code_str = None

{extra_imports}

        """
    )
    if not stable_output:
        model_str += f"# torch version: {torch.version.__version__}\n"
        if hasattr(torch.version, "cuda"):
            model_str += f"# torch cuda version: {torch.version.cuda}\n"
        if hasattr(torch.version, "git_version"):
            model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
        model_str += _cuda_system_info_comment()
    if save_dir:
        ep_path = os.path.join(save_dir, "exported_program.pt2")
    else:
        ep_path = "exported_program.pt2"
    torch.export.save(exported_program, ep_path)

    model_str += f"exported_program = torch.export.load('{ep_path}')\n"
    model_str += "# print(exported_program.graph)\n"
    model_str += f"config_patches={options}\n"
    return model_str


def repro_load_args(load_args: Any, save_dir: str | None) -> tuple[Any]:
    if not hasattr(load_args, "_version"):
        log.warning(
            "load_args does not have a _version attribute, please file a bug to PyTorch "
            "and describe how you generate this repro script"
        )
    else:
        if load_args._version > 0:
            log.warning(
                "load_args is version %s, but this version of PyTorch only supports "
                "version 0.  We will try to run it anyway but there may be an incompatibility; "
                "if so, try upgrading your version of PyTorch.",
                load_args._version,
            )

    nop_reader = NopInputReader()
    load_args(nop_reader)

    with tqdm(desc="Loading inputs", total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args

    return tuple(args)


def repro_common(
    options: Any, exported_program: ExportedProgram
) -> tuple[torch.fx.GraphModule, Any, Any]:
    # pyrefly: ignore [bad-assignment]
    torch._inductor.config.generate_intermediate_hooks = True
    mod = exported_program.module(check_guards=False)
    args, kwargs = exported_program.example_inputs
    return mod, args, kwargs  # type: ignore[return-value]


def repro_get_args(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: dict[str, Any] | None,
) -> tuple[torch.fx.GraphModule, Any, Any]:
    mod, args, kwargs = repro_common(options, exported_program)
    return mod, args, kwargs


def repro_run(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: dict[str, Any] | None,
) -> None:
    from torch._inductor import _aoti_compile_and_package_inner

    gm, args, kwargs = repro_common(options, exported_program)

    from torch.cuda import synchronize

    _aoti_compile_and_package_inner(
        gm,
        args,
        kwargs,
        load_and_run=True,
        check_accuracy=options.accuracy,
        inductor_configs=config_patches,
    )

    need_sync = False

    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            need_sync = True
            break

    if need_sync:
        synchronize()  # ensure segfaults are surfaced


def export_for_aoti_minifier(
    gm: torch.nn.Module,
    tuple_inputs: tuple[Any],
    strict: bool = False,
    skip_export_error: bool = True,
) -> torch.nn.Module | None:
    # Some graphs cannot be used for AOTI/export (illegal graphs), these should be
    # considered as graphs that don't fail in the minifier, so the minifier keeps searching.
    # In these cases, we return None. Otherwise, we return the exported graph module.
    # This won't affect the minifier result because the minifier is only responsible for catching
    # errors in AOTI, not export.
    #
    # Please add to this list of illegal graphs if you change the implementation here.
    # - graph output is not allowed by export
    #
    # If skip_export_error=True, then the errors in export will not be raised, and the minifier
    # will keep exploring and ignore this graph.
    from torch._dynamo.exc import UserError, UserErrorType

    try:
        ep = torch.export.export(gm, tuple_inputs, strict=strict)
        gm = ep.module(check_guards=False)
        return gm
    except Exception as e:
        if skip_export_error:
            return None
        if isinstance(e, UserError) and e.error_type == UserErrorType.INVALID_OUTPUT:
            # graph output is not allowed by export when strict=True
            return None
        if isinstance(e, RuntimeError):
            # graph output is not allowed by export when strict=False
            pattern = r"Found .* in output, which is not a known type\."
            if re.search(pattern, str(e)) is not None:
                return None
        raise AOTIMinifierError(e) from e
    # we should never reach here
    # pyrefly: ignore [unreachable]
    return None


def repro_minify(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: dict[str, Any] | None,
) -> None:
    from functorch.compile import minifier
    from torch._inductor import _aoti_compile_and_package_inner
    from torch._inductor.compile_fx import _aoti_flatten_inputs

    mod, args, kwargs = repro_common(options, exported_program)

    # update serialized_in_spec and serialized_out_spec
    flat_example_inputs, inductor_configs = _aoti_flatten_inputs(
        mod, args, kwargs, options=config_patches
    )
    compiler_name = "aot_inductor"
    if options.minifier_export_mode not in ["dynamo", "python"]:
        raise AssertionError(
            f"minifier_export_mode must be 'dynamo' or 'python', got {options.minifier_export_mode!r}"
        )
    strict = options.minifier_export_mode == "dynamo"
    skip_export_error = options.skip_export_error

    from torch.cuda import synchronize

    need_sync = False

    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            need_sync = True
            break

    def module_fails(
        gm: torch.fx.GraphModule,
        flat_example_inputs: list[Any],
        check_str: str | None = None,
    ) -> bool:
        # Need to export first so the in_spec and out_spec are populated
        tuple_inputs = tuple(flat_example_inputs)
        # pyrefly: ignore [bad-assignment]
        gm = export_for_aoti_minifier(
            gm, tuple_inputs, strict=strict, skip_export_error=skip_export_error
        )

        # Some graphs cannot be used for AOTI/export (illegal graphs), these should be
        # considered as graphs that don't fail in the minifier, so the minifier keeps searching.
        if gm is None:
            return False

        if not isinstance(gm, torch.fx.GraphModule):
            raise AssertionError(f"Expected torch.fx.GraphModule, got {type(gm)}")

        try:
            _aoti_compile_and_package_inner(
                gm,
                tuple_inputs,
                load_and_run=True,
                check_accuracy=options.accuracy,
                inductor_configs=inductor_configs,
            )
            if need_sync:
                synchronize()  # ensure segfaults are surfaced
            return False
        except Exception as e:
            if check_str is not None and check_str not in repr(e):
                return False
            return True

    minifier(
        mod,
        flat_example_inputs,
        module_fails=functools.partial(module_fails, check_str=options.check_str),
        dump_state=functools.partial(
            dump_compiler_graph_state,
            compiler_name=compiler_name,
            config_patches=config_patches,
            accuracy=options.accuracy,
            strict=strict,
        ),
        save_dir=options.save_dir,
        offload_to_disk=options.offload_to_disk,
        skip_offload=options.skip_saving_eager_intermediates,
        skip_sanity=options.skip_sanity,
        max_granularity=options.max_granularity,
    )


def run_repro(
    exported_program: ExportedProgram,
    *,
    config_patches: dict[str, str] | None = None,
    command: str = "run",
    accuracy: bool | str = "",
    save_dir: str | None = None,
    tracing_mode: str | None = None,
    check_str: str | None = None,
    minifier_export_mode: str = "python",
    skip_export_error: bool = True,
    **more_kwargs: Any,
) -> Any:
    for k in more_kwargs:
        log.warning(
            "Unrecognized kwarg %s; perhaps this repro was made on a newer version of PyTorch",
            k,
        )

    if accuracy is True:
        accuracy = "accuracy"
    elif accuracy is False:
        accuracy = ""

    parser = argparse.ArgumentParser(
        description=f"""\
An AOTI repro script, typically triggering a bug in PyTorch AOTInductor.
When run with no arguments, this script defaults to running '{command}'.
Extra flags may be available; to find out more, try '{command} --help'.
There are also alternate subcommands available, see below.

default settings on this script:
  {accuracy=}
  {tracing_mode=}
  {save_dir=}
  {check_str=}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def common_flags(parser: argparse.ArgumentParser) -> None:
        accuracy_group = parser.add_mutually_exclusive_group()
        accuracy_group.add_argument(
            "--no-accuracy",
            dest="accuracy",
            action="store_const",
            const="",
            default=accuracy,
            help="do not test accuracy, just run the module and see if it errors",
        )
        accuracy_group.add_argument(
            "--accuracy",
            action="store_const",
            const="accuracy",
            default=accuracy,
            help="""\
test if the RMSE between the compiled module and the fp64 reference is greater
than eager and the fp64 reference. This is usually more reliable than the
standard allclose test, as we expect numeric differences from compiling, often
improving accuracy over eager.  RMSE test allows for compiled module to
diverge greatly from eager, as long as this divergence moves it closer to the
'true' mathematical value of the network.  Caveats: (1) double precision can
still suffer from rounding error, so it is not a perfect reference (see for
example 'Herbie: Automatically Improving Floating Point Accuracy') for
approaches that detect the necessary working precision and compute it in
arbitrary precision floating point; unfortunately, this is not practical for
tensor computation; (2) if there are not enough samples in the output being
compared, we may get unlucky and have an unlucky greater RMSE than eager; this
could be overcome by applying a more rigorous statistical test at some
p-value, which we leave for future work.
""",
        )
        accuracy_group.add_argument(
            "--strict-accuracy",
            dest="accuracy",
            action="store_const",
            const="strict_accuracy",
            default=accuracy,
            help="""\
by default, when doing accuracy minification we will reject reductions which
change the divergence from a floating point divergence to a integral/boolean
divergence.  This is because some operations like ReLU involve temporarily
sharp boundaries that smooth out again afterwards; without requiring
divergence on floating point, the minifier will often fixate on divergent
boolean tensor even though this is not the true source of the divergence.
However, rejecting these reductions makes it more difficult for the minifier
to make process.  Using this option will let the minifier progress for ALL
divergences--you just might not end up with a useful repro in the end.""",
        )

        parser.add_argument(
            "--save-dir",
            type=str,
            default=save_dir,
            metavar="DIR",
            help="directory where saved inputs live",
        )
        parser.add_argument(
            "--no-save-dir",
            dest="save_dir",
            action="store_const",
            const=None,
            help="don't use any directory for saved inputs",
        )

    subparsers = parser.add_subparsers(
        dest="command", metavar="{run,minify}", required=True
    )

    parser_run = subparsers.add_parser(
        "run",
        help="just run the repro",
    )
    common_flags(parser_run)

    parser_minify = subparsers.add_parser(
        "minify", help="run the minifier on the repro"
    )
    common_flags(parser_minify)
    parser_get_args = subparsers.add_parser("get_args", help="get the args")
    common_flags(parser_get_args)
    parser_minify.add_argument(
        "--skip-saving-eager-intermediates",
        action="store_true",
        help="skip saving eager intermediates on --minify",
    )
    parser_minify.add_argument(
        "--offload-to-disk",
        action="store_true",
        help="during minification, offload delta debugging intermediates to disk.  Use if you're OOMing",
    )
    parser_minify.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip sanity check at beginning of minification on original graph",
    )
    parser_minify.add_argument(
        "--max-granularity",
        type=int,
        default=None,
        help="start at this granularity and work down; must be power of 2",
    )
    parser_minify.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )
    parser_minify.add_argument(
        "--minifier-export-mode",
        type=str,
        default=minifier_export_mode,
        help=(
            "The export mode used in minifier, either dynamo or python."
            "`dynamo` corresponds to strict=True, and `python` corresponds to strict=False."
        ),
    )
    parser_minify.add_argument(
        "--skip-export-error",
        type=bool,
        default=skip_export_error,
        help="Skip intermediate graphs that cannot be exported.",
    )

    # Run the repro in the context of minification, inverting exit code meaning
    parser_minifier_query = subparsers.add_parser(
        "minifier-query",
    )
    common_flags(parser_minifier_query)
    parser_minifier_query.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )

    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]

    options = parser.parse_args(args)
    COMMAND_FNS = {
        "minify": repro_minify,
        "run": repro_run,
        "get_args": repro_get_args,
    }
    return COMMAND_FNS[options.command](
        options, exported_program, config_patches=config_patches
    )
<|endoftext|>"""
Utilities for reproducing and debugging issues in Dynamo after graph capture.

This file provides tools and infrastructure for debugging problems that occur
after Dynamo has captured the graph but before/during backend compilation.
Key components include:

- Minification tools to reduce large graphs to minimal failing examples
- Accuracy testing to validate compiled graph outputs match eager mode
- Repro generation to create standalone reproduction scripts
- Debug backends for capturing and analyzing failures
- Utilities for saving/loading graph states and inputs

The tools here focus specifically on the post-graph-capture stage, making them
useful for debugging backend compilation issues, AOTAutograd problems, and
accuracy discrepancies between compiled and eager execution.
"""

import argparse
import copy
import functools
import logging
import os
import shutil
import sys
import textwrap
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any, cast

import torch
import torch.fx as fx
from torch._dynamo.debug_utils import (
    AccuracyError,
    backend_accuracy_fails,
    BUCK_CMD_PREFIX,
    BuckTargetWriter,
    extra_imports,
    generate_config_string,
    generate_env_vars_string,
    helper_for_dump_minify,
    InputReader,
    InputWriter,
    minifier_dir,
    NNModuleToString,
    NopInputReader,
    run_fwd_maybe_bwd,
    same_two_models,
)
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from torch.hub import tqdm

from .. import config
from ..backends.registry import CompilerFn, lookup_backend, register_debug_backend
from ..debug_utils import clone_inputs_retaining_gradness
from ..types import CompilerConfigProvider
from . import ReproOptions


log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _accuracy_fails(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    compiler_fn: Callable[[torch.fx.GraphModule, list[Any]], torch.fx.GraphModule],
) -> bool:
    return backend_accuracy_fails(
        gm,
        example_inputs,
        compiler_fn,
        only_fwd=config.repro_forward_only,
        ignore_non_fp=config.repro_ignore_non_fp,
    )


class WrapBackendDebug:
    def __init__(
        self, unconfigured_compiler_fn: CompilerFn, compiler_name: str | None
    ) -> None:
        functools.wraps(unconfigured_compiler_fn)(self)
        self._torchdynamo_orig_backend = unconfigured_compiler_fn
        self._compiler_name = compiler_name
        if hasattr(unconfigured_compiler_fn, "__name__"):
            self.__name__ = unconfigured_compiler_fn.__name__
        if hasattr(unconfigured_compiler_fn, "compiler_name"):
            self.__name__ = unconfigured_compiler_fn.compiler_name  # type: ignore[attr-defined]
        if hasattr(unconfigured_compiler_fn, "get_compiler_config"):
            self.get_compiler_config = cast(
                CompilerConfigProvider, unconfigured_compiler_fn
            ).get_compiler_config

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[Any], **kwargs: Any
    ) -> torch.fx.GraphModule:
        compiler_fn = functools.partial(self._torchdynamo_orig_backend, **kwargs)
        if config.repro_after not in ("dynamo", "aot", None):
            raise AssertionError(
                f"repro_after must be 'dynamo', 'aot', or None, got {config.repro_after!r}"
            )

        if config.repro_after == "dynamo":

            def add_paths(exc: Exception) -> None:
                exc.minifier_path = os.path.join(minifier_dir(), "minifier_launcher.py")  # type: ignore[attr-defined]
                if use_buck:
                    exc.buck_command = " ".join(  # type: ignore[attr-defined]
                        BUCK_CMD_PREFIX
                        + [BuckTargetWriter(exc.minifier_path).cmd_line_path]  # type: ignore[attr-defined]
                    )

            if config.repro_level == 3:
                dump_to_minify_after_dynamo(gm, example_inputs, self._compiler_name)

            # Check for either accuracy (level 4) or other type of failures.
            if config.repro_level == 4:
                # Check Accuracy
                compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                if _accuracy_fails(gm, example_inputs, compiler_fn):  # type: ignore[arg-type]
                    log.warning(
                        "Accuracy failed for the TorchDynamo produced graph. Creating script to minify the error."
                    )
                    dump_to_minify_after_dynamo(
                        fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                        example_inputs,
                        self._compiler_name,
                    )
                    exc = AccuracyError("Bad accuracy detected.")
                    add_paths(exc)
                    raise exc
            else:
                try:
                    compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                    run_fwd_maybe_bwd(compiled_gm, example_inputs)  # type: ignore[arg-type]
                except Exception as exc:
                    log.warning(
                        "Compiled Fx GraphModule failed. Creating script to minify the error."
                    )
                    if config.repro_level == 1:
                        dump_state_fn = functools.partial(
                            dump_backend_state, compiler_name=self._compiler_name
                        )
                        dump_state_fn(
                            fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs
                        )
                    elif config.repro_level == 2:
                        dump_to_minify_after_dynamo(
                            fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                            example_inputs,
                            self._compiler_name,
                        )
                    add_paths(exc)
                    raise
        else:
            compiled_gm = compiler_fn(gm, example_inputs)

        return compiled_gm  # type: ignore[return-value]


def wrap_backend_debug(
    unconfigured_compiler_fn: CompilerFn, compiler_name: str | None
) -> WrapBackendDebug:
    """
    A minifier decorator that wraps the TorchDynamo produced Fx graph modules.
    As opposed to wrap_compiler_debug, this wrapper intercepts at the
    TorchDynamo produced Fx Graph Module. This makes it backend-agnostic to some
    level, e.g., it is useful for minifying issues related to Aot Autograd
    tracing.  If an error is found, we minify and save the minified repro in
    repro.tar.gz.
    """
    return WrapBackendDebug(unconfigured_compiler_fn, compiler_name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO DUMPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_dynamo_fx_repro_string(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str | None,
    check_accuracy: bool = False,
    *,
    stable_output: bool = False,
    save_dir: str | None = None,
    command: str = "run",
) -> str:
    """
    Generate a repro string for backend-agnostic minified version.
    """

    model_str = NNModuleToString.convert(gm)

    # TODO: Figure out why torch.compile'd hash isn't work on this codepath
    writer = InputWriter(save_dir, stable_hash=True)
    for placeholder, arg in zip(fx_placeholder_targets(gm), args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            # TODO: improve these names with FQN
            writer.tensor(placeholder, arg)
        else:
            raise TypeError(f"arg is neither SymInt/int nor torch.Tensor, {arg}")
    load_args = "\n".join(writer.lines())

    return textwrap.dedent(
        f"""
{generate_env_vars_string(stable_output=stable_output)}
import math
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

{generate_config_string(stable_output=stable_output)}

{extra_imports}

{model_str}
mod = Repro()

{load_args}

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy={check_accuracy!r}, command={command!r},
        save_dir={save_dir!r}, autocast={torch.is_autocast_enabled()!r}, backend={compiler_name!r})
"""
    )


def dump_backend_repro_as_file(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str | None,
    check_accuracy: bool = False,
) -> None:
    """
    Saves the repro to a repro.py file
    """
    curdir = os.getcwd()
    subdir = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"minified_{len(gm.graph.nodes)}_nodes.py")
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )

    with open(file_name, "w") as fd:
        fd.write(
            generate_dynamo_fx_repro_string(
                gm, args, compiler_name, check_accuracy, save_dir=subdir
            )
        )
    latest_repro = os.path.join(curdir, "repro.py")
    log.warning("Copying %s to %s for convenience", file_name, latest_repro)

    if use_buck:
        BuckTargetWriter(latest_repro).write()

    shutil.copyfile(file_name, latest_repro)


def dump_backend_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str | None,
    check_accuracy: bool = False,
) -> None:
    """
    Dumps the dynamo graph to repro the issue.
    1) It tries to convert Fx GraphModule to a string. If we can, it writes to a
    repro.py file.
    2) If we can't convert Fx GraphModule to a string, we use to_folder to save
    the module and save a tar file.
    """
    if not NNModuleToString.can_convert_to_string(gm):
        raise AssertionError("GraphModule cannot be converted to string")
    return dump_backend_repro_as_file(gm, args, compiler_name, check_accuracy)
    # return dump_backend_repro_as_tarfile(gm, args, compiler_name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER DUMPER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def dump_to_minify_after_dynamo(
    gm: torch.fx.GraphModule, args: Sequence[Any], compiler_name: str | None
) -> None:
    # TODO: factor this out
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    helper_for_dump_minify(
        generate_dynamo_fx_repro_string(
            gm,
            args,
            compiler_name,
            check_accuracy=config.repro_level == 4,
            save_dir=subdir,
            command="minify",
        )
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER BACKENDS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@register_debug_backend  # type: ignore[arg-type]
def dynamo_minifier_backend(
    gm: fx.GraphModule, example_inputs: Sequence[Any], compiler_name: str | None
) -> fx.GraphModule:
    from functorch.compile import minifier

    compiler_fn = lookup_backend(compiler_name)  # type: ignore[arg-type]

    # TODO: It's inconsistent to pass SymInt inputs but REAL tensors.
    # We should pass ints and look at the GraphModule placeholders
    # to resolve them to SymInt (if necessary)
    example_inputs = [
        i.node.hint if isinstance(i, torch.SymInt) else i for i in example_inputs
    ]

    try:
        compiled_gm = compiler_fn(gm, example_inputs)
        run_fwd_maybe_bwd(compiled_gm, example_inputs)  # type: ignore[arg-type]
        raise ValueError("No issue was detected")
    except Exception as exc:
        orig_failure = str(exc)
        log.warning(
            "Compiled Fx GraphModule failed. Creating script to minify the error."
        )
        dump_state_fn = functools.partial(
            dump_backend_state, compiler_name=compiler_name
        )
        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)
        fails_fn = functools.partial(
            backend_fails,
            compiler_fn=compiler_fn,
            orig_failure=orig_failure,
        )
        minifier(
            gm,
            example_inputs,
            module_fails=fails_fn,
            dump_state=dump_state_fn,
        )
    return gm


@register_debug_backend  # type: ignore[arg-type]
def dynamo_accuracy_minifier_backend(
    gm: fx.GraphModule, example_inputs: Sequence[Any], compiler_name: str | None
) -> fx.GraphModule:
    from functorch.compile import minifier

    compiler_fn = lookup_backend(compiler_name)  # type: ignore[arg-type]

    # Set the eval mode to remove randomness.
    gm.eval()

    # Check Accuracy
    if _accuracy_fails(gm, example_inputs, compiler_fn):  # type: ignore[arg-type]
        log.warning("Accuracy failed for the TorchDynamo produced graph")
        dump_state_fn = functools.partial(
            dump_backend_state, compiler_name=compiler_name, check_accuracy=True
        )
        fails_fn = functools.partial(
            _accuracy_fails,
            compiler_fn=compiler_fn,  # type: ignore[arg-type]
        )
        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)
        minifier(
            gm,
            example_inputs,
            module_fails=fails_fn,
            dump_state=dump_state_fn,
        )
    else:
        log.error("Input graph does not fail accuracy testing")
    return gm


def backend_fails(
    gm: fx.GraphModule,
    example_inputs: Sequence[Any],
    compiler_fn: CompilerFn,
    orig_failure: Sequence[Any],
) -> bool:
    """
    Minifier uses this function to identify if the minified graph module fails
    with the same error.

    One caveat is that minifier can potentially go into a wrong direction when
    the resulting graph module fails for a different reason. To avoid this, we
    save the string for the original exception and check similarity between new
    and old exception. They can be somewhat different in some cases, when the
    exception string depends on the failing node information. So, we have a
    loose similarity metric to guide the minifier path.
    """
    from difflib import SequenceMatcher

    try:
        # Run the original gm to check eager validity
        run_fwd_maybe_bwd(gm, clone_inputs_retaining_gradness(example_inputs))
        compiled_gm = compiler_fn(gm, example_inputs)  # type: ignore[arg-type]
        run_fwd_maybe_bwd(compiled_gm, clone_inputs_retaining_gradness(example_inputs))  # type: ignore[arg-type]
    except Exception as e:
        new_failure = str(e)
        if SequenceMatcher(None, orig_failure, new_failure).ratio() > 0.5:
            return True
    return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def run_load_args(
    options: ReproOptions, mod: torch.nn.Module, load_args: Any
) -> list[Any]:
    if not hasattr(load_args, "_version"):
        log.warning(
            "load_args does not have a _version attribute, please file a bug to PyTorch "
            "and describe how you generate this repro script"
        )
    else:
        if load_args._version > 0:
            log.warning(
                "load_args is version %s, but this version of PyTorch only supports "
                "version 0.  We will try to run it anyway but there may be an incompatibility; "
                "if so, try upgrading your version of PyTorch.",
                load_args._version,
            )

    nop_reader = NopInputReader()
    load_args(nop_reader)

    with tqdm(desc="Loading inputs", total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args

    return args


def repro_minify(options: ReproOptions, mod: torch.nn.Module, load_args: Any) -> None:
    # Setup debug minifier compiler
    if not options.accuracy:
        compiler_fn = lookup_backend("dynamo_minifier_backend")
    else:
        compiler_fn = lookup_backend("dynamo_accuracy_minifier_backend")

    if options.backend is None:
        raise RuntimeError(
            "Compiler name is None - this likely means that a custom compiler "
            "was called by torchdynamo. Please remove this error, import your "
            "custom compiler function, and replace the backend=None "
            "line in run_repro to backend=<my_imported_custom_function>"
        )

    dynamo_minifier_backend = functools.partial(
        compiler_fn,
        compiler_name=options.backend,  # type: ignore[call-arg]
    )
    opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

    args = run_load_args(options, mod, load_args)
    device_type = next(
        (
            a.device.type
            for a in args
            if isinstance(a, torch.Tensor) and a.device.type != "cpu"
        ),
        "cpu",
    )
    with torch.amp.autocast(device_type, enabled=options.autocast):
        opt_mod(*args)


def repro_run(options: ReproOptions, mod: torch.nn.Module, load_args: Any) -> None:
    opt_mod = torch._dynamo.optimize(options.backend)(mod)

    if options.accuracy != "":
        mod.eval()
        opt_mod.eval()  # type: ignore[union-attr]

        args = run_load_args(options, mod, load_args)
        device_type = next(
            (
                a.device.type
                for a in args
                if isinstance(a, torch.Tensor) and a.device.type != "cpu"
            ),
            "cpu",
        )
        with torch.amp.autocast(device_type, enabled=options.autocast):
            # TODO: disable clone
            if not same_two_models(mod, mod, args):  # type: ignore[arg-type]
                raise AssertionError("Eager itself failed")
            if not same_two_models(
                mod,  # type: ignore[arg-type]
                opt_mod,  # type: ignore[arg-type]
                args,
                only_fwd=config.repro_forward_only,
                ignore_non_fp=config.repro_ignore_non_fp,
            ):
                raise AccuracyError("Dynamo failed")
    else:
        args = run_load_args(options, mod, load_args)
        device_type = next(
            (
                a.device.type
                for a in args
                if isinstance(a, torch.Tensor) and a.device.type != "cpu"
            ),
            "cpu",
        )
        with torch.amp.autocast(device_type, enabled=options.autocast):
            run_fwd_maybe_bwd(mod, args, only_fwd=options.only_fwd, disable_clone=True)  # type: ignore[arg-type]
            del args

            args = run_load_args(options, mod, load_args)
            run_fwd_maybe_bwd(
                opt_mod,  # type: ignore[arg-type]
                args,
                only_fwd=options.only_fwd,
                disable_clone=True,  # type: ignore[arg-type]
            )


def run_repro(
    mod: torch.nn.Module,
    load_args: Any,
    *,
    command: str = "run",
    accuracy: bool | str = "",
    save_dir: str | None = None,
    autocast: bool = False,
    backend: str = "inductor",
    **kwargs: Any,
) -> None:
    for k in kwargs:
        log.warning(
            "Unrecognized kwarg %s; perhaps this repro was made on a newer version of PyTorch",
            k,
        )

    if accuracy is True:
        accuracy = "accuracy"
    elif accuracy is False:
        accuracy = ""

    parser = argparse.ArgumentParser(
        description=f"""\
An after_dynamo repro script, typically triggering a bug in Dynamo or
AOTAutograd.  When run with no arguments, this script defaults to running
'{command}'.  Extra flags may be available; to find out more, try '{command}
--help'.  There are also alternate subcommands available, see below.

default settings on this script:
  {accuracy=}
  {save_dir=}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def common_flags(parser: argparse.ArgumentParser) -> None:
        accuracy_group = parser.add_mutually_exclusive_group()
        accuracy_group.add_argument(
            "--no-accuracy",
            dest="accuracy",
            action="store_const",
            const="",
            default=accuracy,
            help="do not test accuracy, just run the module and see if it errors",
        )
        accuracy_group.add_argument(
            "--accuracy",
            action="store_const",
            const="accuracy",
            default=accuracy,
            help="test accuracy",
        )
        parser.add_argument(
            "--save-dir",
            type=str,
            default=save_dir,
            metavar="DIR",
            help="directory where saved inputs live",
        )
        parser.add_argument(
            "--no-save-dir",
            dest="save_dir",
            action="store_const",
            const=None,
            help="don't use any directory for saved inputs",
        )
        parser.add_argument(
            "--no-isolate",
            dest="isolate",
            action="store_false",
            default=False,
            help="no isolate (doesn't do anything for after_dynamo)",
        )
        parser.add_argument(
            "--autocast",
            default=autocast,
            action="store_true",
            help="use torch.amp.autocast",
        )
        parser.add_argument(
            "--no-autocast",
            dest="autocast",
            action="store_false",
            help="don't use torch.amp.autocast",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default=backend,
            metavar="BACKEND",
            help="torch.compile backend to use",
        )

    subparsers = parser.add_subparsers(
        dest="command", metavar="{run,minify}", required=True
    )

    parser_run = subparsers.add_parser(
        "run",
        help="just run the repro",
    )
    common_flags(parser_run)
    parser_run.add_argument(
        "--only-fwd",
        action="store_true",
        help="don't run backwards compilation for testing",
    )

    parser_minify = subparsers.add_parser(
        "minify", help="run the minifier on the repro"
    )
    common_flags(parser_minify)

    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]

    options = cast(ReproOptions, parser.parse_args(args))
    COMMAND_FNS = {
        "minify": repro_minify,
        "run": repro_run,
    }
    COMMAND_FNS[options.command](options, mod, load_args)
<|endoftext|>from __future__ import annotations

from typing import Protocol


class ReproOptions(Protocol):
    """
    Read-only view of the argparse Namespace threaded through the repro_*
    entrypoints in after_aot.py and after_dynamo.py.

    argparse populates the Namespace dynamically, so a Protocol (rather than a
    concrete class) is the pragmatic fit: it documents the attributes those
    functions actually read without asserting how the Namespace is built. Not
    every attribute is populated for every subcommand -- e.g. after_dynamo
    never sets tracing_mode -- but each repro_* function only reads the subset
    that its own subparser defines.
    """

    # Shared across the after_aot and after_dynamo subparsers.
    command: str
    accuracy: str
    save_dir: str | None

    # after_aot-only attributes.
    tracing_mode: str | None
    check_str: str | None
    isolate: bool
    offload_to_disk: bool
    skip_saving_eager_intermediates: bool
    skip_sanity: bool
    max_granularity: int | None
    stable_hash: bool
    skip_saving_inductor_intermediates: bool
    skip_saving_float64_intermediates: bool
    skip_check_deterministic: bool

    # after_dynamo-only attributes.
    backend: str | None
    autocast: bool
    only_fwd: bool
<|endoftext|>"""
Utilities for reproducing and debugging issues in PyTorch's Dynamo AOT compilation.

This module provides tools and infrastructure for:
1. Generating minimal reproducible test cases ("repros") from failing compilations
2. Analyzing accuracy issues between eager and compiled execution
3. Minifying large models/inputs to isolate problematic patterns
4. Debugging compiler errors and accuracy divergences

The main components include:
- Repro generation: Creates standalone Python files that reproduce compiler issues
- Minification: Reduces large graphs to minimal failing examples
- Accuracy analysis: Compares compiled vs eager execution, with fp64 reference
- Debug tools: Dumps graph state, tracks intermediates, analyzes divergences

This is primarily used by PyTorch developers and researchers to debug issues in
the Dynamo AOT compilation pipeline, particularly for the Inductor backend.
"""

from __future__ import annotations

import argparse
import ast
import copy
import functools
import io
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import types
import typing
import uuid
from importlib import import_module
from tempfile import TemporaryFile
from typing import Any, IO, TYPE_CHECKING, TypedDict
from typing_extensions import Unpack

import sympy


try:
    import triton.language as tl
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.jit import JITFunction

    TritonConstexpr = tl.constexpr
except ImportError:

    class Autotuner:  # type: ignore[no-redef]
        pass

    class JITFunction:  # type: ignore[no-redef]
        pass

    class Heuristics:  # type: ignore[no-redef]
        pass

    class TritonConstexpr:  # type: ignore[no-redef]
        pass


import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo.debug_utils import (
    _cuda_system_info_comment,
    AccuracyError,
    backend_accuracy_fails,
    BuckTargetWriter,
    cast_to_fp64,
    extra_deps,
    extra_imports,
    generate_config_string,
    generate_env_vars_string,
    helper_for_dump_minify,
    InputReader,
    InputWriter,
    MAX_CONSTANT_NUMEL_INLINE,
    minifier_dir,
    NNModuleToString,
    NopInputReader,
    same_two_models,
)
from torch._dynamo.utils import clone_inputs, counters, same
from torch._environment import is_fbcode
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch._inductor.cpp_builder import normalize_path_separator
from torch._library.fake_class_registry import FakeScriptObject
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    fx_placeholder_targets,
    has_free_symbols,
)
from torch.hub import tqdm

from .. import config


def _find_repeat_interleave_constraints(
    gm: torch.fx.GraphModule,
) -> list[tuple[str, str]]:
    """
    Find repeat_interleave operations with output_size constraints.

    Returns list of (repeats_placeholder_name, output_size_placeholder_name) pairs.
    These represent constraints where sum(repeats) must equal output_size.
    """
    constraints = []
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or "repeat_interleave" not in str(node.target)
            or not node.args
        ):
            continue

        output_size_node = node.kwargs.get("output_size")
        repeats_node = node.args[0]

        # Both must be FX nodes (not constants) and direct placeholders
        if (
            isinstance(repeats_node, torch.fx.Node)
            and isinstance(output_size_node, torch.fx.Node)
            and repeats_node.op == "placeholder"
            and output_size_node.op == "placeholder"
        ):
            constraints.append((str(repeats_node.target), str(output_size_node.target)))

    return constraints


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch._inductor.compile_fx import _CompileFxCallable, _CompileFxKwargs
    from torch._inductor.output_code import OutputCode
    from torch._inductor.utils import InputType

    from . import ReproOptions


log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")


class GroupInfo(TypedDict):
    size: int
    rank: int


def _extract_distributed_info(
    gm: torch.fx.GraphModule,
) -> dict[str, GroupInfo]:
    """
    Extract process group information from distributed ops in the graph.

    Returns a dict mapping group names to dicts with 'size' and 'rank' keys.
    Example: {'tp': {'size': 4, 'rank': 0}, 'dp': {'size': 2, 'rank': 0}}
    """
    from torch.distributed import GroupName
    from torch.fx.operator_schemas import normalize_function

    group_info: dict[str, GroupInfo] = {}

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if not isinstance(node.target, OpOverload):
            continue
        if node.target.namespace not in {"_c10d_functional", "c10d_functional"}:
            continue

        opt_args_kwargs = normalize_function(
            node.target,
            args=node.args,
            kwargs=node.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        if opt_args_kwargs is None:
            continue
        _, kwargs = opt_args_kwargs

        group_name_ = kwargs.get("group_name")
        if not isinstance(group_name_, str):
            continue
        group_name = typing.cast(GroupName, group_name_)

        if group_name in group_info:
            continue

        from torch.distributed.distributed_c10d import (
            _get_group_size_by_name,
            _resolve_process_group,
        )

        group_size = _get_group_size_by_name(group_name)
        pg = _resolve_process_group(group_name)
        rank = pg.rank()
        group_info[group_name] = {"size": group_size, "rank": rank}

    return group_info


def setup_fake_process_groups(
    group_info: dict[str, GroupInfo],
) -> None:
    """
    Set up fake process groups for repro execution.

    Args:
        group_info: dict mapping group_name -> {'size': group_size, 'rank': rank}
    """
    import torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore

    if not group_info:
        return

    world_size = max(info["size"] for info in group_info.values())

    global_rank = 0
    for info in group_info.values():
        if info["size"] == world_size:
            global_rank = info["rank"]
            break

    store = FakeStore()
    dist.init_process_group(
        backend="fake",
        rank=global_rank,
        world_size=world_size,
        store=store,
    )

    default_pg = dist.distributed_c10d._get_default_group()
    torch._C._distributed_c10d._unregister_all_process_groups()

    for group_name, info in group_info.items():
        group_size = info["size"]
        if group_size == world_size:
            # pyrefly: ignore[bad-argument-type]
            torch._C._distributed_c10d._register_process_group(group_name, default_pg)
        else:
            ranks = list(range(group_size))
            new_pg = dist.new_group(ranks)
            # pyrefly: ignore[bad-argument-type]
            torch._C._distributed_c10d._register_process_group(group_name, new_pg)


def generate_standalone_repro(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    *,
    save_path: str | None = None,
) -> str:
    """
    Generate a self-contained repro script from an FX graph.
    """
    buf = io.StringIO()
    save_graph_repro(buf, gm, args, "inductor", save_dir=None)
    repro = buf.getvalue()

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(repro)
        log.info("Saved standalone repro to %s", save_path)

    return repro


use_buck = is_fbcode()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def wrap_compiler_debug(
    unconfigured_compiler_fn: _CompileFxCallable,
    compiler_name: str,
) -> _CompileFxCallable:
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler_fn - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstraction, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[InputType],
        compile_region_name: str | None = None,
        **kwargs: Unpack[_CompileFxKwargs],
    ) -> OutputCode:
        from torch._subclasses import FakeTensorMode

        compiler_fn = functools.partial(
            unconfigured_compiler_fn,
            compile_region_name=compile_region_name,
            **kwargs,
        )

        from torch._functorch.aot_autograd import get_aot_graph_name

        graph_name = get_aot_graph_name()

        # TODO: why do we need to deepcopy the original graph?
        orig_graph = copy.deepcopy(gm.graph)
        if config.repro_after not in ("dynamo", "aot", None):
            raise AssertionError(
                f"repro_after must be 'dynamo', 'aot', or None, got {config.repro_after!r}"
            )

        try:
            # Call the compiler_fn - which is either aot_autograd or inductor
            # with fake inputs
            inner_compiled_fn = compiler_fn(gm, example_inputs)
        except Exception:
            # TODO: Failures here are troublesome because no real inputs,
            # need a different serialization strategy
            if config.repro_after == "aot":
                if config.repro_level == 1:
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                elif config.repro_level == 2:
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                log.error("CompilerError")
            raise

        # We may run regular PyTorch compute that may trigger Dynamo, do NOT
        # recursively attempt to accuracy minify in that case!
        def deferred_for_real_inputs(
            real_inputs: Sequence[InputType], **_kwargs: object
        ) -> Any:
            # This is a bit obscure: if we recursively try to accuracy minify
            # the SAME function, this would trigger.  But most of the time
            # we should never hit this branch
            if _kwargs:
                raise AssertionError(f"Unexpected kwargs: {_kwargs}")
            if config.repro_after != "aot":
                if isinstance(inner_compiled_fn, str):
                    raise AssertionError("inner_compiled_fn should not be a string")
                return inner_compiled_fn(real_inputs)
            with config.patch(repro_after=None):
                return inner_debug_fn(real_inputs)

        def inner_debug_fn(real_inputs: Sequence[InputType]) -> Any:
            """
            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,
            example_inputs can be fake tensors. We can call compiler_fn (which is
            inductor or nvfuser) with fake tensors but the actually compiled_fn
            should be called with real tensors. Therefore, the actual invocation
            is deferred.
            """
            # Copy the tensor attrs like shape, stride etc by converting to Fake Tensor
            # because inductor clears the tensor list in its codegen. And example_inputs
            # are available only for the first invocation.
            fake_mode = FakeTensorMode()
            copy_tensor_attrs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in real_inputs
            ]
            if config.repro_level == 3:
                # Always dump the original module in case we have segfaults
                dump_to_minify(
                    fx.GraphModule(gm, orig_graph), real_inputs, compiler_name
                )

            if config.repro_level == 4:
                if compiler_name != "inductor":
                    raise NotImplementedError(
                        "Accuracy minification is supported for inductor only"
                    )
                failed = not same_two_models(
                    gm,
                    inner_compiled_fn,  # type: ignore[arg-type]
                    real_inputs,
                    only_fwd=True,
                    ignore_non_fp=config.repro_ignore_non_fp,
                )

                if failed:
                    log.warning(
                        "Accuracy failed for the AOT Autograd graph %s", graph_name
                    )
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        real_inputs,
                        f"{compiler_name}_accuracy",
                    )
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        real_inputs,
                        f"{compiler_name}_accuracy",
                    )
                    raise AccuracyError("Bad accuracy detected")
                else:
                    # Call the compiled function with real inputs
                    return inner_compiled_fn(real_inputs)  # type: ignore[operator]
            else:
                try:
                    # Call the compiled function with real inputs
                    out = inner_compiled_fn(real_inputs)  # type: ignore[operator]
                    # sync cuda kernels to ensure IMA detection
                    if (
                        any(
                            isinstance(arg, torch.Tensor) and arg.device.type != "cpu"
                            for arg in example_inputs
                        )
                        and torch.accelerator.is_available()
                    ):
                        torch.accelerator.synchronize()
                    return out
                except Exception:
                    if config.repro_level == 1:
                        dump_compiler_graph_state(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    elif config.repro_level == 2:
                        dump_to_minify(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    raise

        if config.repro_after == "aot":
            compiled_fn = deferred_for_real_inputs
            compiled_fn._boxed_call = True  # type: ignore[attr-defined]
            return compiled_fn  # type: ignore[return-value]
        else:
            return inner_compiled_fn

    return debug_wrapper


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP REPROS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def maybe_fbcode_instructions() -> str:
    if is_fbcode():
        extra_deps_formatted = "\n".join([f'        "{dep}",' for dep in extra_deps])
        if len(extra_deps_formatted) > 0:
            extra_deps_formatted = "\n" + extra_deps_formatted
        return f"""\
\"\"\"
To run this script in fbcode:
- Create a directory (//scripts/{{your_unixname}}/repro)
- Put this file in scripts/{{your_unixname}}/repro/fx_graph_runnable.py
- Add a TARGETS file that looks like the following
- `buck2 run //scripts/{{your_unixname}}/repro:repro`

NOTE: you may need additional deps to actually be able to run the script.
```
# Contents of TARGETS file
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

python_binary(
    name = "repro",
    main_src = "fx_graph_runnable.py",
    deps = [
        "//caffe2:torch",{extra_deps_formatted}
    ],
)
```
\"\"\"
"""
    else:
        return ""


def generate_custom_triton_kernel(kernel: Any) -> str:
    res = ""
    if isinstance(kernel, Autotuner):
        # pyrefly: ignore [missing-attribute]
        if isinstance(kernel.fn, Heuristics):
            res += "ERROR: Repro will not work as intended, "
            res += "triton.runtime.autotuner.Heuristics is not currently supported\n"
            return res

        config_strs = []
        # pyrefly: ignore [missing-attribute]
        for kernel_config in kernel.configs:
            config_strs.append(f"""triton.Config(
                    {str(kernel_config.kwargs)},
                    num_warps={kernel_config.num_warps},
                    num_stages={kernel_config.num_stages},
                )""")

        config_str = ",".join(config_strs)
        res += textwrap.dedent(f"""
        @triton.autotune(
            configs=[
                {config_str}
            ],
            key=[]
        )
        """).strip()

    # pyrefly: ignore [missing-attribute]
    src_code = kernel.src if isinstance(kernel, JITFunction) else kernel.fn.src
    res += "\n@triton.jit\n"
    res += src_code
    res += "\n"

    return res


def generate_compiler_repro_string(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    *,
    stable_output: bool = False,
    save_dir: str | None = None,
    stable_hash: bool = False,
    has_distributed_ops: bool = False,
) -> str:
    if save_dir is not None:
        save_dir = normalize_path_separator(save_dir)
    # Add distributed imports if needed
    distributed_imports = ""
    if has_distributed_ops:
        distributed_imports = textwrap.dedent(
            """
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
        """
        ).strip()

    triton_imports = ""

    if len(kernel_side_table.id_to_kernel) > 0:
        triton_imports = textwrap.dedent(
            """
import triton
import triton.language as tl
        """
        ).strip()

    model_str = textwrap.dedent(
        f"""
{generate_env_vars_string(stable_output=stable_output)}
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
import math
from math import inf
import torch._inductor.inductor_prims
{distributed_imports}
{triton_imports}

{generate_config_string(stable_output=stable_output)}

isolate_fails_code_str = None

{extra_imports}

{maybe_fbcode_instructions()}
     """
    )
    model_str += textwrap.dedent(
        """
if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
"""
    )
    if not stable_output:
        model_str += f"# torch version: {torch.version.__version__}\n"
        if hasattr(torch.version, "cuda"):
            model_str += f"# torch cuda version: {torch.version.cuda}\n"
        if hasattr(torch.version, "git_version"):
            model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
        model_str += _cuda_system_info_comment()

    kernel_side_table_prefix = (
        "torch._higher_order_ops.triton_kernel_wrap.kernel_side_table"
    )

    def get_fn_name(kernel: Any) -> str:
        fn: Any = kernel if isinstance(kernel, JITFunction) else kernel.fn
        return fn.__name__.split(".")[-1]

    def get_triton_import_line(name: str, val: Any) -> str | None:
        # User-defined Triton kernels are serialized from their source, not from
        # their original Python module.  If the source references a global
        # imported from Triton, such as `from triton.language.extra import
        # libdevice`, the standalone repro must recreate that import.
        if name in ("triton", "tl"):
            return None

        if isinstance(val, types.ModuleType):
            module_name = val.__name__
            if module_name == "triton" or module_name.startswith("triton."):
                return f"import {module_name} as {name}"
            return None

        module_name = getattr(val, "__module__", None)
        object_name = getattr(val, "__name__", None)
        if (
            isinstance(module_name, str)
            and (module_name == "triton" or module_name.startswith("triton."))
            and isinstance(object_name, str)
        ):
            if name == object_name:
                return f"from {module_name} import {object_name}"
            return f"from {module_name} import {object_name} as {name}"

        return None

    def write_kernel_dependencies(
        kernel: Any,
        written_constexpr_vars: set[str],
        written_nested_kernels: set[str],
        written_triton_imports: set[str],
    ) -> str:
        """Write out global triton imports, tl.constexpr vars, and nested kernels."""
        result = ""
        jit_fn = kernel if isinstance(kernel, JITFunction) else kernel.fn
        if not getattr(jit_fn, "fn", None) or not getattr(jit_fn, "src", None):
            return result

        fn_globals = getattr(jit_fn.fn, "__globals__", {})
        src = jit_fn.src
        full_src = src if src.strip().startswith("def ") else "def " + src

        referenced_names: set[str] = set()
        called_names: set[str] = set()
        for node in ast.walk(ast.parse(full_src)):
            if isinstance(node, ast.Name):
                referenced_names.add(node.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_names.add(node.func.id)

        # Only recreate globals that are safe and useful for a standalone
        # Triton repro: imports anchored in the Triton package and simple
        # constants.  Other user globals remain unsupported here.
        for name in sorted(referenced_names):
            if name in written_constexpr_vars:
                continue
            val = fn_globals.get(name)

            import_line = get_triton_import_line(name, val)
            if import_line is not None:
                if import_line not in written_triton_imports:
                    result += import_line + "\n"
                    written_triton_imports.add(import_line)
                continue

            if isinstance(val, TritonConstexpr) and getattr(val, "value", None):
                result += f"{name} = tl.constexpr({val.value})\n"
            elif isinstance(val, (int, float, str, bool)):
                result += f"{name} = {val!r}\n"
            else:
                continue
            written_constexpr_vars.add(name)

        # Write out nested kernel dependencies
        for name in called_names:
            val = fn_globals.get(name)
            if not isinstance(val, JITFunction) or val is jit_fn:
                continue
            nested_fn_name = get_fn_name(val)
            if nested_fn_name in written_nested_kernels:
                continue
            # Mark as written before recursing to prevent cycles
            written_nested_kernels.add(nested_fn_name)
            result += write_kernel_dependencies(
                val,
                written_constexpr_vars,
                written_nested_kernels,
                written_triton_imports,
            )
            result += generate_custom_triton_kernel(val)

        return result

    written_nested_kernels: set[str] = set()
    written_constexpr_vars: set[str] = set()
    written_triton_imports: set[str] = {
        "import triton",
        "import triton.language as tl",
    }

    model_str += f"{kernel_side_table_prefix}.reset_table()\n"

    for idx in kernel_side_table.id_to_kernel:
        kernel = kernel_side_table.get_kernel(idx)

        try:
            model_str += write_kernel_dependencies(
                kernel,
                written_constexpr_vars,
                written_nested_kernels,
                written_triton_imports,
            )
            fn_name = get_fn_name(kernel)

            unique_name = f"{fn_name}_{idx}"

            kernel_code = generate_custom_triton_kernel(kernel)
            kernel_code = kernel_code.replace(
                f"def {fn_name}(", f"def {unique_name}(", 1
            )
            model_str += kernel_code
            model_str += f"{kernel_side_table_prefix}.add_kernel({unique_name})\n"
        except AttributeError as e:
            model_str += "ERROR: Repro will not work as intended, "
            model_str += f"User defined triton kernel exception: {e}\n"

    if len(kernel_side_table.constant_args) > 0:
        model_str += f"{kernel_side_table_prefix}.constant_args={kernel_side_table.constant_args}\n"

    # This string is also emitted as a best-effort trace artifact during
    # normal compilation, so unsupported modules must not mask compile errors.
    model_str += NNModuleToString.convert(gm, allow_unsafe_repr=True)

    writer = InputWriter(save_dir, stable_hash=stable_hash)
    # pyrefly: ignore [implicit-any]
    used_syms = {}

    # Extract from graph placeholders and their corresponding arguments
    placeholder_targets = fx_placeholder_targets(gm)
    for placeholder, arg in zip(placeholder_targets, args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            # TODO: improve these names with FQN
            writer.tensor(placeholder, arg)
        elif arg is None:
            writer.const(placeholder)
        elif isinstance(arg, FakeScriptObject):
            writer.opaque(placeholder, arg.script_class_name)
        elif isinstance(arg, torch._C.Generator):
            writer.generator(placeholder, arg)
        else:
            writer.unsupported(placeholder, arg)

        # Extract symbolic variables from the same arguments

        if (
            isinstance(arg, torch.SymInt)
            # By checking sympy.Symbol, we are excluding any symbolic expressions.
            # TODO: we may need to solve expressions to extract symbol definitions.
            and isinstance(arg.node.expr, sympy.Symbol)
            and arg.node.hint is not None
        ):
            used_syms[str(arg.node)] = arg.node.hint
        elif isinstance(arg, torch.Tensor):
            # Extract symbolic variables from tensor shapes and strides
            for dim in arg.shape:
                if (
                    isinstance(dim, torch.SymInt)
                    and isinstance(dim.node.expr, sympy.Symbol)
                    and dim.node.hint is not None
                ):
                    used_syms[str(dim.node)] = dim.node.hint
            for stride in arg.stride():
                if (
                    isinstance(stride, torch.SymInt)
                    and isinstance(stride.node.expr, sympy.Symbol)
                    and stride.node.hint is not None
                ):
                    used_syms[str(stride.node)] = stride.node.hint
            # Extract symbols from storage nbytes (can be a symbolic expression)
            storage = arg.untyped_storage()
            nbytes = storage.nbytes()
            if isinstance(nbytes, torch.SymInt):
                expr = nbytes.node.expr
                shape_env = nbytes.node.shape_env
                for sym in expr.free_symbols:
                    sym_name = str(sym)
                    if sym_name not in used_syms and shape_env is not None:
                        hint = shape_env.backed_var_to_val.get(sym)
                        if hint is not None:
                            used_syms[sym_name] = int(hint)
    # Add symbolic variable definitions to the top of the generated code
    if used_syms:
        hint_lines = "\n".join(
            f"{name} = {hint}" for name, hint in sorted(used_syms.items())
        )
        model_str = f"{hint_lines}\n\n{model_str}"

    # Add fixup code for repeat_interleave constraints
    # When inputs are regenerated randomly, sum(repeats) != output_size
    # This fixup adjusts the repeats tensor to satisfy the constraint
    constraints = _find_repeat_interleave_constraints(gm)
    if constraints:
        placeholder_to_idx = {name: idx for idx, name in enumerate(placeholder_targets)}
        for repeats_name, output_size_name in constraints:
            repeats_idx = placeholder_to_idx.get(repeats_name)
            output_size_idx = placeholder_to_idx.get(output_size_name)
            if repeats_idx is not None and output_size_idx is not None:
                # Guard with hasattr since NopInputReader doesn't have args
                writer._lines.append(
                    "# Fixup: ensure sum(repeats) == output_size for repeat_interleave"
                )
                writer._lines.append("if hasattr(reader, 'args'):")
                writer._lines.append(f"    _repeats = reader.args[{repeats_idx}]")
                writer._lines.append(
                    f"    _output_size = reader.args[{output_size_idx}]"
                )
                writer._lines.append(
                    "    if isinstance(_repeats, torch.Tensor) and _repeats.dtype == torch.int64:"
                )
                writer._lines.append("        _n = _repeats.numel()")
                writer._lines.append("        _repeats.fill_(_output_size // _n)")
                writer._lines.append("        _repeats[:_output_size % _n] += 1")

    load_args_lines = writer.lines()
    load_args_code = "\n".join(load_args_lines)
    model_str += load_args_code + "\n"

    model_str += "mod = Repro()\n"
    return model_str


def save_graph_repro(
    fd: IO[Any],
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    stable_output: bool = False,
    save_dir: str | None = None,
    command: str = "run",
    accuracy: str | bool | None = None,
    tracing_mode: str | None = None,
    check_str: str | None = None,
    stable_hash: bool = False,
) -> None:
    if any(
        isinstance(arg, torch.fx.experimental._backward_state.BackwardState)
        for arg in args
    ):
        fd.write(
            "Repro is not generated due to existence of BackwardState in graph input"
        )
        return

    if save_dir is not None:
        save_dir = normalize_path_separator(save_dir)

    # Extract distributed info from the graph
    distributed_info = _extract_distributed_info(gm)
    has_distributed_ops = len(distributed_info) > 0

    fd.write(
        generate_compiler_repro_string(
            gm,
            args,
            stable_output=stable_output,
            save_dir=save_dir,
            stable_hash=stable_hash,
            has_distributed_ops=has_distributed_ops,
        )
    )
    if accuracy is None:
        accuracy = "_accuracy" in compiler_name
    if tracing_mode is None:
        tracing_mode = "real"
        if any(
            has_free_symbols(a) for a in args if not isinstance(a, FakeScriptObject)
        ):
            tracing_mode = "symbolic"
    fd.write("if __name__ == '__main__':\n")
    fd.write("    from torch._dynamo.repro.after_aot import run_repro\n")

    # Add distributed initialization before run_repro if needed
    if has_distributed_ops:
        fd.write(
            "    from torch._dynamo.repro.after_aot import setup_fake_process_groups\n"
        )
        fd.write(f"    setup_fake_process_groups({distributed_info!r})\n")

    fd.write(
        f"    with torch.no_grad():\n"
        f"        run_repro(mod, load_args, accuracy={accuracy!r}, command={command!r}, "
        f"save_dir={save_dir!r}, tracing_mode={tracing_mode!r}, check_str={check_str!r})\n"
        f"        # To run it separately, do \n"
        f"        # mod, args = run_repro(mod, load_args, accuracy={accuracy!r}, command='get_args', "
        f"save_dir={save_dir!r}, tracing_mode={tracing_mode!r}, check_str={check_str!r})\n"
        f"        # mod(*args)"
    )

    # Add distributed cleanup after run_repro
    if has_distributed_ops:
        fd.write("\n    dist.destroy_process_group()\n")


def dump_compiler_graph_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    accuracy: str | bool | None = None,
) -> None:
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )
    with open(file_name, "w") as fd:
        save_graph_repro(
            fd, gm, args, compiler_name, save_dir=subdir, accuracy=accuracy
        )
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, "repro.py")
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning("Copying repro file for convenience to %s", repro_path)
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning("No write permissions for %s", repro_path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP MINIFIER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def dump_to_minify(
    gm: torch.fx.GraphModule, args: Sequence[Any], compiler_name: str
) -> None:
    out = io.StringIO()
    # TODO: factor this out
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    save_graph_repro(out, gm, args, compiler_name, save_dir=subdir, command="minify")
    return helper_for_dump_minify(out.getvalue())


def isolate_fails(
    fx_g: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    env: dict[str, Any] | None = None,
    save_dir: str | None = None,
    accuracy: bool | str | None = None,
    tracing_mode: str | None = None,
    check_str: str | None = None,
) -> bool:
    if env is None:
        env = {}
    subdir = os.path.join(os.getcwd(), "isolate")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{str(uuid.uuid4())[:5]}.py")
    with open(file_name, "w") as fd:
        save_graph_repro(
            fd,
            fx_g,
            args,
            compiler_name,
            save_dir=save_dir,
            command="minifier-query",
            accuracy=accuracy,
            tracing_mode=tracing_mode,
            check_str=check_str,
        )
    # with open(file_name, "r") as fd:
    #     print(fd.read())
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    if use_buck:
        cmd = BuckTargetWriter(file_name).write(print_msg=False)
    else:
        cmd = [sys.executable, file_name]
    with (
        TemporaryFile() as stdout,
        TemporaryFile() as stderr,
        subprocess.Popen(
            cmd,
            cwd=subdir,
            stdout=stdout,
            stderr=stderr,
            env=new_env,
        ) as p,
    ):
        p.wait()

        stdout.seek(0)
        stderr.seek(0)
        print(
            textwrap.indent(stdout.read().decode("utf-8"), prefix=">>  "),
            file=sys.stdout,
        )
        print(
            textwrap.indent(stderr.read().decode("utf-8"), prefix=">>  "),
            file=sys.stderr,
        )
        # print(f"Isolated test failed - {file_name}")
        return p.returncode != 0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER TOOLS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def inductor_fails(
    fx_g: torch.fx.GraphModule, args: Sequence[Any], check_str: str | None = None
) -> bool:
    has_gpu = any(
        isinstance(arg, torch.Tensor) and arg.device.type != "cpu" for arg in args
    )

    def sync() -> None:
        if has_gpu and torch.accelerator.is_available():
            # Ensures that segfaults are surfaced
            torch.accelerator.synchronize()

    from torch._inductor.compile_fx import compile_fx_inner

    try:
        result = fx_g(*args)
        if not isinstance(result, (tuple, list)):
            raise AssertionError(
                f"Expected result to be a tuple or list, got {type(result)}"
            )
        if any(isinstance(x, (tuple, list)) for x in result):
            raise AssertionError("Result should not contain nested tuples or lists")
    except Exception:
        return False

    sync()

    try:
        compile_args = _get_compile_args(fx_g, args)
        compile_mod = compile_fx_inner(fx_g, compile_args)
        if isinstance(compile_mod, str):
            raise AssertionError("compile_fx_inner should not return a string")
        compile_mod(args)
        sync()
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def inductor_accuracy_fails(
    fx_g: torch.fx.GraphModule,
    args: Sequence[Any],
    check_str: str | None = None,
    *,
    require_fp64: bool = False,
    ignore_non_fp: bool = False,
) -> bool:
    from torch._inductor.compile_fx import compile_fx_inner

    def _compile_with_symbolic_args(
        gm: torch.fx.GraphModule, inputs: list[Any]
    ) -> torch.fx.GraphModule:
        return compile_fx_inner(gm, _get_compile_args(gm, inputs))  # type: ignore[return-value]

    return backend_aot_accuracy_fails(
        fx_g,
        args,  # type: ignore[arg-type]
        _compile_with_symbolic_args,  # type: ignore[arg-type]
        require_fp64=require_fp64,
        ignore_non_fp=ignore_non_fp,
    )


backend_aot_accuracy_fails = functools.partial(backend_accuracy_fails, only_fwd=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _build_symbolic_wrapper(
    mod: nn.Module,
    args: list[Any],
    symint_exprs: dict[int, str],
) -> tuple[nn.Module, list[Any]] | None:
    """Build a wrapper module that preserves symbolic relationships.

    Returns (wrapper_module, wrapper_args) where wrapper_args has carrier
    tensors in place of free-symbol symints, and the wrapper's forward()
    extracts SymInts from carrier.size(0), computes derived expressions,
    and delegates to the inner module.

    Returns None if the expressions don't contain free/derived structure
    (nothing to do).
    """
    import re

    free_sym_re = re.compile(r"^s\d+$")
    free_positions: dict[int, str] = {}
    derived_positions: dict[int, str] = {}

    for idx, expr_str in symint_exprs.items():
        if free_sym_re.fullmatch(expr_str):
            free_positions[idx] = expr_str
        else:
            derived_positions[idx] = expr_str

    if not free_positions:
        return None

    # Order: carrier tensors first, then all non-symint args in original order
    free_order = sorted(free_positions.keys())
    free_sym_names = [free_positions[i] for i in free_order]

    # Precompile derived expressions
    derived_compiled = {
        idx: compile(expr_str, f"<derived arg {idx}>", "eval")
        for idx, expr_str in derived_positions.items()
    }

    n_args = len(args)

    class _SymIntWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *flat_args: Any) -> Any:
            carriers = flat_args[: len(free_order)]
            other = flat_args[len(free_order) :]

            syms: dict[str, Any] = {}
            for i, sym_name in enumerate(free_sym_names):
                syms[sym_name] = carriers[i].size(0)

            for idx, code in sorted(derived_compiled.items()):
                derived_positions[idx]  # keep reference
                syms[f"_derived_{idx}"] = eval(code, {"__builtins__": {}}, syms)

            other_iter = iter(other)
            rebuilt = []
            for i in range(n_args):
                if i in free_positions:
                    rebuilt.append(syms[free_positions[i]])
                elif i in derived_positions:
                    rebuilt.append(syms[f"_derived_{i}"])
                else:
                    rebuilt.append(next(other_iter))

            return self.inner(*rebuilt)

    # Build wrapper args: carriers + non-symint args
    wrapper_args: list[Any] = []
    for i in free_order:
        hint = int(args[i]) if not isinstance(args[i], int) else args[i]
        device = "cpu"
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_cuda:
                device = a.device
                break
        wrapper_args.append(torch.empty(hint, dtype=torch.int8, device=device))

    for i, arg in enumerate(args):
        if i not in free_positions and i not in derived_positions:
            wrapper_args.append(arg)

    return _SymIntWrapper(mod), wrapper_args


def repro_common(
    options: ReproOptions, mod: nn.Module, load_args: Any
) -> tuple[torch.fx.GraphModule, list[Any]]:
    # Invariant for graphs we generate with the repro script
    if any(mod.named_parameters()):
        raise AssertionError("Repro graph should not have any parameters")
    for n, b in mod.named_buffers():
        if b.numel() > MAX_CONSTANT_NUMEL_INLINE:
            log.warning(
                "Constant %s was not serialized, generated random data instead. "
                "If you think this is affecting you, please comment on "
                "https://github.com/pytorch/pytorch/issues/100468",
                n,
            )

    if not hasattr(load_args, "_version"):
        log.warning(
            "load_args does not have a _version attribute, please file a bug to PyTorch "
            "and describe how you generate this repro script"
        )
    else:
        if load_args._version > 0:
            log.warning(
                "load_args is version %s, but this version of PyTorch only supports "
                "version 0.  We will try to run it anyway but there may be an incompatibility; "
                "if so, try upgrading your version of PyTorch.",
                load_args._version,
            )

    nop_reader = NopInputReader()
    load_args(nop_reader)

    with tqdm(desc="Loading inputs", total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args

    # Turn mod into a GraphModule the slow way
    # TODO: speed this up
    # NOTE: symint_exprs (from reader.symint(val, expr=...)) are preserved
    # in the repro script for documentation. A future improvement could use
    # _build_symbolic_wrapper to reconstruct algebraic relationships between
    # free and derived symints, but the arg reordering is fragile across
    # different graph structures so we skip it for now.
    mod = make_fx(mod, tracing_mode=options.tracing_mode)(*args)  # type: ignore[arg-type]

    # pyrefly: ignore [bad-assignment]
    torch._inductor.config.generate_intermediate_hooks = True

    return mod, args


def _get_compile_args(mod: torch.fx.GraphModule, args: Sequence[Any]) -> Sequence[Any]:
    """Extract FakeTensor/SymInt args from the traced graph for compilation.

    When repro_common traces with tracing_mode='symbolic', the resulting
    GraphModule's placeholder nodes carry FakeTensor/SymInt metadata.
    compile_fx_inner needs these (not the concrete args) so that Inductor
    generates proper symbolic-size bindings in the output code.

    For tracing_mode='real', concrete args are fine — we must NOT extract
    FakeTensor metadata because different nodes may have FakeTensors from
    different FakeTensorModes, causing a FakeTensorMode mismatch assertion
    in Inductor.  We detect symbolic tracing by checking for SymInt values,
    which only exist when tracing_mode='symbolic'.
    """
    placeholders = [n for n in mod.graph.nodes if n.op == "placeholder"]
    if not placeholders:
        return args
    # Only extract metadata if the graph was traced with symbolic mode.
    # SymInt values in placeholder metadata are the reliable indicator —
    # FakeTensors appear in both real and symbolic modes, but only symbolic
    # tracing creates SymInts for integer inputs.
    has_symint = any(isinstance(n.meta.get("val"), torch.SymInt) for n in placeholders)
    if not has_symint:
        return args
    return [n.meta.get("val", a) for n, a in zip(placeholders, args)]


ACCURACY_FAILS: dict[str, Callable[[torch.fx.GraphModule, Any], bool]] = {
    "": inductor_fails,
    # This might look inverted but it's not.  strict_accuracy means "we will
    # minify any time we see anything that diverges", whereas accuracy is more
    # conservative, and will only minify if there is a meaningful fp64
    # divergence
    "accuracy": functools.partial(
        inductor_accuracy_fails, require_fp64=True, ignore_non_fp=True
    ),
    "strict_accuracy": inductor_accuracy_fails,
}


def repro_minifier_query(options: ReproOptions, mod: nn.Module, load_args: Any) -> None:
    mod, args = repro_common(options, mod, load_args)
    fail_fn = functools.partial(
        ACCURACY_FAILS[options.accuracy],
        check_str=options.check_str,  # type: ignore[call-arg]
    )
    if fail_fn(mod, args):
        sys.exit(1)
    else:
        sys.exit(0)


def repro_minify(options: ReproOptions, mod: nn.Module, load_args: Any) -> None:
    from functorch.compile import minifier

    mod, args = repro_common(options, mod, load_args)
    compiler_name = "inductor_accuracy" if options.accuracy != "" else "inductor"

    favored_device = 1 if torch.cuda.device_count() >= 2 else 0
    env_variables = {"CUDA_VISIBLE_DEVICES": str(favored_device)}

    module_fails: Any
    if options.isolate:
        module_fails = functools.partial(
            isolate_fails,
            env=env_variables,
            compiler_name=compiler_name,
            save_dir=options.save_dir,
            accuracy=options.accuracy,
            tracing_mode=options.tracing_mode,
        )
    else:
        module_fails = ACCURACY_FAILS[options.accuracy]

    with config.patch(repro_after=None):
        minifier(
            mod,
            args,
            module_fails=functools.partial(module_fails, check_str=options.check_str),
            dump_state=functools.partial(
                dump_compiler_graph_state, compiler_name=compiler_name
            ),
            save_dir=options.save_dir,
            offload_to_disk=options.offload_to_disk,
            skip_offload=options.skip_saving_eager_intermediates,
            skip_sanity=options.skip_sanity,
            max_granularity=options.max_granularity,
        )


def repro_analyze(options: ReproOptions, mod: nn.Module, load_args: Any) -> None:
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.hooks import intermediate_hook

    mod, args = repro_common(options, mod, load_args)

    # TODO: The logic for cloning inputs/models here is intentionally
    # modeled off of run_fwd_maybe_bwd, but arguably it is better not to
    # clone inputs (as you are doubling your effective GPU memory usage).
    # It is certainly faster though!  It probably makes sense to let the
    # user specify the offload strategy.

    compile_mod = copy.deepcopy(mod)
    compile_args = _get_compile_args(compile_mod, args)
    with tqdm(desc="Compiling"):
        compiled = compile_fx_inner(compile_mod, compile_args)
    total = counters["inductor"]["intermediate_hooks"]

    known_names = set()

    def save_hook(name: str, val: Any) -> None:
        known_names.add(name)
        if not options.skip_saving_inductor_intermediates:
            writer.write_tensor(os.path.join("inductor", name), val)
        pbar.update(1)  # type: ignore[has-type]

    # analyze stores and re-reads intermediates on disk, so a save_dir is
    # required; --no-save-dir (save_dir=None) is invalid for this subcommand.
    save_dir = options.save_dir
    if save_dir is None:
        raise RuntimeError("analyze requires a save_dir; do not pass --no-save-dir")
    writer = torch.utils._content_store.ContentStoreWriter(
        save_dir, stable_hash=options.stable_hash
    )
    reader = torch.utils._content_store.ContentStoreReader(save_dir)

    new_args = clone_inputs(args)
    with (
        intermediate_hook(save_hook),
        tqdm(desc="Saving inductor intermediates", total=total) as pbar,
    ):
        if isinstance(compiled, str):
            raise AssertionError("compile_fx_inner should not return a string")
        compiled(new_args)  # type: ignore[arg-type]
        if new_args:
            raise AssertionError("new_args should be empty after compiled() call")

    def compare_tuples(tuple1: tuple[Any], tuple2: tuple[Any]) -> str | None:
        diff_indices = [i for i in range(len(tuple1)) if tuple1[i] != tuple2[i]]
        diff_values = [(tuple1[i], tuple2[i]) for i in diff_indices]

        if not diff_values:
            return None
        else:
            return " and ".join(f"{a} != {b}" for a, b in diff_values)

    def check_hook(name: str, val: Any) -> None:
        meta = writer.compute_tensor_metadata(val)
        meta2 = reader.read_tensor_metadata(os.path.join("inductor", name))
        reason = compare_tuples(meta, meta2)
        if reason is not None:
            pbar.write(f"NONDETERMINISTIC INDUCTOR at {name} ({reason})")
        pbar.update(1)

    if not options.skip_check_deterministic:
        new_args = clone_inputs(args)
        with (
            intermediate_hook(check_hook),
            tqdm(desc="Checking inductor determinism", total=total) as pbar,
        ):
            compiled(new_args)  # type: ignore[arg-type]
            if new_args:
                raise AssertionError("new_args should be empty after compiled() call")

    class WriterInterp(fx.Interpreter):
        def __init__(self, mod: torch.nn.Module, subdir: str) -> None:
            super().__init__(mod)
            self.subdir = subdir

        def run_node(self, n: torch.fx.Node) -> Any:
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                pbar.update(1)
                writer.write_tensor(os.path.join(self.subdir, name), r)
            return r

    # NB: the module cast doesn't actually do anything, since there are no
    # parameters/buffers on the module
    if not options.skip_saving_float64_intermediates:
        new_mod, new_args = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))  # type: ignore[arg-type]
        with tqdm(desc="Saving float64 intermediates", total=total) as pbar:
            WriterInterp(new_mod, "float64").boxed_run(new_args)
        if new_args:
            raise AssertionError("new_args should be empty after boxed_run() call")

    class ExactReaderInterp(fx.Interpreter):
        def run_node(self, n: torch.fx.Node) -> Any:
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                meta = writer.compute_tensor_metadata(r)
                meta2 = reader.read_tensor_metadata(os.path.join("float64", name))
                reason = compare_tuples(meta, meta2)
                if reason is not None:
                    pbar.write(f"NONDETERMINISTIC FLOAT64 at {name} ({reason})")
                pbar.update(1)
            return r

    # TODO: check eager determinism

    if not options.skip_check_deterministic:
        new_mod, new_args = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))  # type: ignore[arg-type]
        with tqdm(desc="Checking float64 determinism", total=total) as pbar:
            ExactReaderInterp(new_mod).boxed_run(new_args)
            if new_args:
                raise AssertionError("new_args should be empty after boxed_run() call")

    # Now that we've saved everything, interp through the eager graph
    # and do comparisons
    class ReaderInterp(fx.Interpreter):
        def run_node(self, n: torch.fx.Node) -> Any:
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                inductor = reader.read_tensor(os.path.join("inductor", name))
                float64 = reader.read_tensor(os.path.join("float64", name))
                logged = False

                def log_error(msg: str, *args: Any) -> None:
                    nonlocal logged
                    logged = True
                    pbar.write(f"DIVERGED at {name}: {msg % args}")

                if not same(
                    r,
                    inductor,
                    float64,
                    tol=torch._dynamo.config.repro_tolerance,
                    equal_nan=True,
                    log_error=log_error,
                ):
                    if not logged:
                        raise AssertionError("Divergence detected but not logged")
                pbar.update(1)
            return r

    with tqdm(desc="Checking divergence", total=total) as pbar:
        ReaderInterp(mod).boxed_run(args)
    if args:
        raise AssertionError("args should be empty after boxed_run() call")


def repro_get_args(
    options: ReproOptions, mod: nn.Module, load_args: Any
) -> tuple[torch.fx.GraphModule, list[Any]]:
    mod, args = repro_common(options, mod, load_args)
    return mod, args


def repro_run(options: ReproOptions, mod: nn.Module, load_args: Any) -> None:
    from torch._inductor.compile_fx import compile_fx_inner

    mod, args = repro_common(options, mod, load_args)

    compile_mod = copy.deepcopy(mod)
    compile_args = _get_compile_args(compile_mod, args)
    compiled = compile_fx_inner(compile_mod, compile_args)
    if isinstance(compiled, str):
        raise AssertionError("compile_fx_inner should not return a string")

    if options.accuracy != "":
        # We don't really respect --accuracy vs --strict-accuracy here, it
        # seems counterintuitive
        if not same_two_models(
            mod,
            compiled,  # type: ignore[arg-type]
            args,
            only_fwd=True,
            ignore_non_fp=config.repro_ignore_non_fp,
        ):
            raise AccuracyError("Bad accuracy detected")
    else:
        compiled(list(args))
        if (
            any(
                isinstance(arg, torch.Tensor) and arg.device.type != "cpu"
                for arg in args
            )
            and torch.accelerator.is_available()
        ):
            torch.accelerator.synchronize()  # ensure segfaults are surfaced


# TODO: lazily load the inputs or something, rather than cloning them
def run_repro(
    mod: nn.Module,
    load_args: Any,
    *,
    command: str = "run",
    accuracy: bool | str = "",
    save_dir: str | None = None,
    tracing_mode: str | None = None,
    patch_code: str | None = None,
    check_str: str | None = None,
    **kwargs: Any,
) -> Any:
    for k in kwargs:
        log.warning(
            "Unrecognized kwarg %s; perhaps this repro was made on a newer version of PyTorch",
            k,
        )

    if accuracy is True:
        accuracy = "accuracy"
    elif accuracy is False:
        accuracy = ""

    if patch_code is not None:
        log.warning(
            "patch_code no longer works on this version of PyTorch, silently ignoring"
        )

    parser = argparse.ArgumentParser(
        description=f"""\
An after_aot repro script, typically triggering a bug in PyTorch Inductor.
When run with no arguments, this script defaults to running '{command}'.
Extra flags may be available; to find out more, try '{command} --help'.
There are also alternate subcommands available, see below.

default settings on this script:
  {accuracy=}
  {tracing_mode=}
  {save_dir=}
  {check_str=}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def common_flags(parser: argparse.ArgumentParser) -> None:
        accuracy_group = parser.add_mutually_exclusive_group()
        accuracy_group.add_argument(
            "--no-accuracy",
            dest="accuracy",
            action="store_const",
            const="",
            default=accuracy,
            help="do not test accuracy, just run the module and see if it errors",
        )
        accuracy_group.add_argument(
            "--accuracy",
            action="store_const",
            const="accuracy",
            default=accuracy,
            help="""\
test if the RMSE between the compiled module and the fp64 reference is greater
than eager and the fp64 reference. This is usually more reliable than the
standard allclose test, as we expect numeric differences from compiling, often
improving accuracy over eager.  RMSE test allows for compiled module to
diverge greatly from eager, as long as this divergence moves it closer to the
'true' mathematical value of the network.  Caveats: (1) double precision can
still suffer from rounding error, so it is not a perfect reference (see for
example 'Herbie: Automatically Improving Floating Point Accuracy') for
approaches that detect the necessary working precision and compute it in
arbitrary precision floating point; unfortunately, this is not practical for
tensor computation; (2) if there are not enough samples in the output being
compared, we may get unlucky and have an unlucky greater RMSE than eager; this
could be overcome by applying a more rigorous statistical test at some
p-value, which we leave for future work.
""",
        )
        accuracy_group.add_argument(
            "--strict-accuracy",
            dest="accuracy",
            action="store_const",
            const="strict_accuracy",
            default=accuracy,
            help="""\
by default, when doing accuracy minification we will reject reductions which
change the divergence from a floating point divergence to an integral/boolean
divergence.  This is because some operations like ReLU involve temporarily
sharp boundaries that smooth out again afterwards; without requiring
divergence on floating point, the minifier will often fixate on divergent
boolean tensor even though this is not the true source of the divergence.
However, rejecting these reductions makes it more difficult for the minifier
to make progress.  Using this option will let the minifier progress for ALL
divergences--you just might not end up with a useful repro in the end.""",
        )

        parser.add_argument(
            "--save-dir",
            type=str,
            default=save_dir,
            metavar="DIR",
            help="directory where saved inputs live",
        )
        parser.add_argument(
            "--no-save-dir",
            dest="save_dir",
            action="store_const",
            const=None,
            help="don't use any directory for saved inputs",
        )
        parser.add_argument(
            "--tracing-mode",
            type=str,
            metavar="{real,fake,symbolic}",
            default=tracing_mode,
            help="how to trace the repro module into a GraphModule with metadata",
        )

    subparsers = parser.add_subparsers(
        dest="command", metavar="{run,minify,analyze}", required=True
    )

    parser_run = subparsers.add_parser(
        "run",
        help="just run the repro",
    )
    common_flags(parser_run)

    parser_minify = subparsers.add_parser(
        "minify", help="run the minifier on the repro"
    )
    common_flags(parser_minify)
    parser_get_args = subparsers.add_parser("get_args", help="get the args")
    common_flags(parser_get_args)
    parser_minify_isolate = parser_minify.add_mutually_exclusive_group()
    parser_minify_isolate.add_argument(
        "--isolate",
        action="store_true",
        default=True,
        help="run in separate processes to avoid interference (default)",
    )
    parser_minify_isolate.add_argument(
        "--no-isolate",
        dest="isolate",
        action="store_false",
        help="speed up by running all compilation in same process",
    )
    parser_minify.add_argument(
        "--skip-saving-eager-intermediates",
        action="store_true",
        help="skip saving eager intermediates on --minify",
    )
    # TODO: make this an option for --analyze too
    parser_minify.add_argument(
        "--offload-to-disk",
        action="store_true",
        help="during minification, offload delta debugging intermediates to disk.  Use if you're OOMing",
    )
    parser_minify.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip sanity check at beginning of minification on original graph",
    )
    parser_minify.add_argument(
        "--max-granularity",
        type=int,
        default=None,
        help="start at this granularity and work down; must be power of 2",
    )
    parser_minify.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )

    parser_analyze = subparsers.add_parser(
        "analyze", help="run the accuracy analyzer on the repro"
    )
    common_flags(parser_analyze)
    parser_analyze.add_argument(
        "--skip-saving-inductor-intermediates",
        action="store_true",
        help="skip saving inductor intermediates on --analyze",
    )
    parser_analyze.add_argument(
        "--skip-saving-float64-intermediates",
        action="store_true",
        help="skip saving float64 intermediates",
    )
    parser_analyze.add_argument(
        "--skip-check-deterministic",
        action="store_true",
        help="skip checking that the network is deterministic",
    )
    parser_analyze.add_argument(
        "--stable-hash",
        action="store_true",
        help="use SHA-1 checksum instead of fast (but possibly unsound) hash",
    )

    # Run the repro in the context of minification, inverting exit code meaning
    parser_minifier_query = subparsers.add_parser(
        "minifier-query",
    )
    common_flags(parser_minifier_query)
    parser_minifier_query.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )

    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]

    options = typing.cast("ReproOptions", parser.parse_args(args))
    COMMAND_FNS = {
        "minify": repro_minify,
        "analyze": repro_analyze,
        "minifier-query": repro_minifier_query,
        "run": repro_run,
        "get_args": repro_get_args,
    }
    return COMMAND_FNS[options.command](options, mod, load_args)
<|endoftext|>"""
Python polyfills for builtins
"""

from collections.abc import Iterable, MutableMapping
from typing import TypeVar

from ..decorators import substitute_in_graph


__all__ = []


T = TypeVar("T")


try:
    import _collections  # type: ignore[import-not-found]

    @substitute_in_graph(_collections._count_elements)
    def _count_elements(
        mapping: MutableMapping[T, int],
        iterable: Iterable[T],
    ) -> None:
        "Tally elements from the iterable."
        mapping_get = mapping.get
        for elem in iterable:
            mapping[elem] = mapping_get(elem, 0) + 1

    __all__.append("_count_elements")

except ImportError:
    pass
<|endoftext|>"""
Python polyfills for copy
"""

from __future__ import annotations

from typing import TypeVar

from ..decorators import substitute_in_graph


__all__ = [
    "reduce_ex_user_defined_object",
]

T = TypeVar("T")


@substitute_in_graph(object.__reduce_ex__, skip_signature_check=True)  # type: ignore[arg-type]
def reduce_ex_user_defined_object(obj: T, protocol: int, /) -> tuple:  # type: ignore[type-arg]
    """Traceable polyfill for object.__reduce_ex__ (protocol >= 2).

    Mirrors CPython's reduce_newobj (Objects/typeobject.c): builds the __new__
    arguments from __getnewargs_ex__/__getnewargs__, selects
    copyreg.__newobj_ex__ vs __newobj__ based on whether kwargs are present, and
    computes the pickle state (__getstate__ if overridden, else __dict__ if the
    object has one, else None). copy._reconstruct rebuilds the object via
    cls.__new__(cls, *args) and applies the state.

    This must not assume the object has a __dict__: tuple/slots objects such as
    namedtuples have __slots__ = () and no __dict__, and reduce to
    (copyreg.__newobj__, (cls, *values), None, None, None).
    """
    import copyreg

    cls = type(obj)

    args: tuple  # type: ignore[type-arg]
    kwargs: dict  # type: ignore[type-arg]
    if hasattr(cls, "__getnewargs_ex__"):
        args, kwargs = obj.__getnewargs_ex__()  # type: ignore[attr-defined]
    elif hasattr(cls, "__getnewargs__"):
        args = obj.__getnewargs__()  # type: ignore[attr-defined]
        kwargs = {}
    else:
        args = ()
        kwargs = {}

    if kwargs:
        func = copyreg.__newobj_ex__  # pyrefly: ignore[missing-attribute]
        newargs = (cls, args, kwargs)
    else:
        func = copyreg.__newobj__  # pyrefly: ignore[missing-attribute]
        newargs = (cls, *args)

    default_getstate = getattr(object, "__getstate__", None)
    if getattr(cls, "__getstate__", None) is not default_getstate:
        state = obj.__getstate__()
    else:
        try:
            state = obj.__dict__
        except AttributeError:
            state = None

    return (func, newargs, state, None, None)
<|endoftext|>"""
Python polyfills for sys
"""

from __future__ import annotations

import sys

from ..decorators import substitute_in_graph


__all__ = [
    "intern",
    "getrecursionlimit",
]


# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(sys.intern, can_constant_fold_through=True)
def intern(string: str, /) -> str:
    return string


@substitute_in_graph(sys.getrecursionlimit, can_constant_fold_through=True)
def getrecursionlimit() -> int:
    return sys.getrecursionlimit()


if sys.version_info >= (3, 11):

    @substitute_in_graph(sys.get_int_max_str_digits, can_constant_fold_through=True)
    def get_int_max_str_digits() -> int:
        return sys.get_int_max_str_digits()

    @substitute_in_graph(sys.set_int_max_str_digits, can_constant_fold_through=True)
    def set_int_max_str_digits(maxdigits: int) -> None:
        sys.set_int_max_str_digits(maxdigits)

    __all__ += ["get_int_max_str_digits", "set_int_max_str_digits"]
<|endoftext|>"""
Python polyfills for builtins
"""

from __future__ import annotations

import builtins
import functools
import operator
import sys
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "all",
    "any",
    "cast",
    "enumerate",
    "sum",
]


_T = TypeVar("_T")


@substitute_in_graph(builtins.all, can_constant_fold_through=True)
def all(iterable: Iterable[object], /) -> bool:
    for elem in iterable:
        if not elem:
            return False
    return True


@substitute_in_graph(builtins.any, can_constant_fold_through=True)
def any(iterable: Iterable[object], /) -> bool:
    for elem in iterable:
        if elem:
            return True
    return False


@substitute_in_graph(builtins.enumerate, is_embedded_type=True)  # type: ignore[arg-type]
def enumerate(iterable: Iterable[_T], start: int = 0) -> Iterable[tuple[int, _T]]:
    if not isinstance(start, int):
        raise TypeError(
            f"{type(start).__name__!r} object cannot be interpreted as an integer"
        )

    for x in iterable:
        yield start, x
        start += 1


@substitute_in_graph(builtins.sum, can_constant_fold_through=True)  # type: ignore[arg-type]
def sum(iterable: Iterable[_T], /, start: _T = 0) -> _T:  # type: ignore[assignment]
    return functools.reduce(operator.add, iterable, start)


# TODO(guilhermeleobas): Implement this iterator as a VariableTracker to see if
# it is faster than tracing through it
class _CallableIterator:
    def __init__(self, fn, sentinel):  # type: ignore[no-untyped-def]
        self.fn = fn
        self.sentinel = sentinel
        self.exhausted = False

    def __iter__(self):  # type: ignore[no-untyped-def]
        return self

    def __next__(self):  # type: ignore[no-untyped-def]
        if self.exhausted:
            raise StopIteration

        # The iterator created in this case will call object with no arguments
        # for each call to its __next__() method;
        r = self.fn()

        # CPython calliter_iternext (Objects/iterobject.c) re-reads it_callable /
        # it_sentinel after the call returns. If a reentrant next() exhausted
        # this iterator during fn() (gh-101892), it_sentinel is now NULL, so the
        # sentinel comparison is skipped and StopIteration is raised regardless
        # of the returned value.
        if self.exhausted:
            raise StopIteration

        # If the value returned is equal to sentinel, StopIteration will be raised
        if r == self.sentinel:
            self.exhausted = True
            raise StopIteration

        # otherwise the value will be returned.
        return r


class _SequenceIterator:
    def __init__(self, iterable) -> None:
        self.iterable = iterable
        self.index = 0
        self.exhausted = False

    def __iter__(self) -> _SequenceIterator:
        return self

    def __next__(self) -> object:
        if self.exhausted:
            raise StopIteration

        # CPython iter_iternext (Objects/iterobject.c): the index counter is a
        # Py_ssize_t, so once it reaches PY_SSIZE_T_MAX (sys.maxsize) it cannot
        # advance and every subsequent next() raises OverflowError.
        if self.index == sys.maxsize:
            raise OverflowError("iter index too large")

        try:
            result = self.iterable.__getitem__(self.index)
            self.index += 1
            return result
        except (IndexError, StopIteration):
            self.exhausted = True
            raise StopIteration from None

    def __setstate__(self, state) -> None:
        # CPython iter_setstate (Objects/iterobject.c): a negative index is
        # clamped to 0, and the state is only applied while the iterator is
        # live (it_seq != NULL), i.e. not yet exhausted.
        if not self.exhausted:
            self.index = max(state, 0)


def sequence_iterator(iterable) -> Iterable[object]:
    if hasattr(iterable, "__getitem__"):
        return _SequenceIterator(iterable)
    raise TypeError(f"'{type(iterable)}' object is not iterable")


class _ReversedSequenceIterator:
    # Mirrors CPython's reversed object (Objects/enumobject.c). Walks the
    # sequence backwards via __getitem__, starting from len(seq) - 1.
    def __init__(self, seq) -> None:
        self.seq = seq
        self.index = len(seq) - 1

    def __iter__(self) -> _ReversedSequenceIterator:
        return self

    def __next__(self) -> object:
        if self.index < 0:
            raise StopIteration
        try:
            result = self.seq[self.index]
        except (IndexError, StopIteration):
            self.index = -1
            raise StopIteration from None
        self.index -= 1
        return result


def reversed_sequence_iterator(seq) -> Iterable[object]:
    return _ReversedSequenceIterator(seq)


def callable_iterator(fn, sentinel, /):
    # If the second argument, sentinel, is given, then object must be a
    # callable object.
    if not isinstance(fn, Callable):  # type: ignore[arg-type]
        raise TypeError("iter(v, w): v must be a callable")

    return _CallableIterator(fn, sentinel)


@substitute_in_graph(typing.cast, can_constant_fold_through=True)
def cast(typ: type, val: _T) -> _T:  # type: ignore[type-var]
    return val
<|endoftext|>"""
Polyfills for torch._C._nn functions.
"""

from typing import Any

import torch
from torch.overrides import _is_torch_function_mode_enabled, _pop_mode_temporarily

from ..decorators import substitute_in_graph


@substitute_in_graph(torch._C._nn._parse_to, skip_signature_check=True)
def _parse_to_polyfill(
    *args: Any, **kwargs: Any
) -> tuple[torch.device, torch.dtype, bool, torch.memory_format]:
    """
    Polyfill for torch._C._nn._parse_to that parses arguments to nn.Module.to().

    Signature mirrors torch._C._nn._parse_to which accepts:
    - to(device) - device as string or torch.device
    - to(dtype) - dtype as torch.dtype
    - to(tensor) - extracts device and dtype from tensor
    - to(device=..., dtype=..., non_blocking=..., memory_format=...)

    Returns:
        tuple: (device, dtype, non_blocking, memory_format)
    """
    # Check for __torch_function__ mode and dispatch using handle_torch_function pattern
    if _is_torch_function_mode_enabled():
        with _pop_mode_temporarily() as mode:
            result = mode.__torch_function__(
                torch._C._nn._parse_to, tuple(), args, kwargs or {}
            )
        if result is not NotImplemented:
            return result

    # Default implementation
    device = None
    dtype = None
    non_blocking = False
    memory_format = None

    # Handle positional arguments
    if len(args) == 1:
        arg = args[0]
        # Check if it's a tensor
        if isinstance(arg, torch.Tensor):
            device = arg.device
            dtype = arg.dtype
        # Check if it's a dtype
        elif isinstance(arg, torch.dtype):
            dtype = arg
        # Check if it's a device (string or torch.device)
        elif isinstance(arg, (str, torch.device)):
            device = torch.device(arg) if isinstance(arg, str) else arg
        else:
            raise TypeError(
                f"to() received an invalid combination of arguments. Got: {type(arg)}"
            )
    elif len(args) > 1:
        raise TypeError(
            f"to() received too many positional arguments. Got {len(args)}, expected at most 1"
        )

    # Handle keyword arguments
    if "device" in kwargs:
        device_arg = kwargs["device"]
        if device_arg is not None:
            device = (
                torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            )

    if "dtype" in kwargs:
        dtype = kwargs["dtype"]

    if "non_blocking" in kwargs:
        non_blocking = kwargs["non_blocking"]

    if "memory_format" in kwargs:
        memory_format = kwargs["memory_format"]

    # pyrefly: ignore[bad-return]
    return (device, dtype, non_blocking, memory_format)


@substitute_in_graph(torch.__future__.get_swap_module_params_on_conversion)
def get_swap_module_params_on_conversion_polyfill() -> bool:
    """
    Polyfill for torch.__future__.get_swap_module_params_on_conversion.

    Returns the actual value from the underlying global variable.
    """
    # Access the module's global variable directly to avoid recursion
    import torch.__future__ as torch_future

    return torch_future._swap_module_params_on_conversion


@substitute_in_graph(torch._has_compatible_shallow_copy_type)
def _has_compatible_shallow_copy_type_polyfill(
    input: torch.Tensor, from_: torch.Tensor
) -> bool:
    """
    Polyfill for torch._has_compatible_shallow_copy_type.

    Checks if two tensors have compatible types for shallow copying.
    The C++ implementation checks if input's TensorImpl has compatible shallow copy type
    with from_'s key_set. We approximate this by checking if both tensors are the same type.
    """
    # Check if both tensors are the same type (handles both regular tensors and subclasses)
    # This is more permissive than checking exact torch.Tensor type equality
    # but properly handles subclasses by allowing same-type shallow copies
    return type(input) is type(from_)


__all__ = [
    "_parse_to_polyfill",
    "get_swap_module_params_on_conversion_polyfill",
    "_has_compatible_shallow_copy_type_polyfill",
]
<|endoftext|># Used to load and initialize polyfill handlers when importing torch._dynamo
# Please add a new import when adding a new polyfill module.

import importlib
from typing import TYPE_CHECKING

import torch.utils._pytree as python_pytree

from .. import polyfills, trace_rules


if TYPE_CHECKING:
    from types import ModuleType


# See also the TYPE_CHECKING block in torch/_dynamo/polyfills/__init__.py
POLYFILLED_MODULE_NAMES: tuple[str, ...] = (
    "_collections",
    "builtins",
    "copy",
    "functools",
    "io",
    "itertools",
    "operator",
    "os",
    "struct",
    "sys",
    "fx",
    "tensor",
    "torch_c_nn",
    "traceback",
)
if python_pytree._cxx_pytree_dynamo_traceable:
    POLYFILLED_MODULE_NAMES += ("pytree",)

POLYFILLED_MODULES: tuple["ModuleType", ...] = tuple(
    importlib.import_module(f".{submodule}", package=polyfills.__name__)
    for submodule in POLYFILLED_MODULE_NAMES
)


# Unregister the builtin functions from _builtin_function_ids to let them to be
# dispatched with the appropriate VariableTracker type. Otherwise, they will be
# dispatched with BuiltinVariable if present in _builtin_function_ids.
for polyfill_module in POLYFILLED_MODULES:
    for polyfill_name in polyfill_module.__all__:
        polyfill_handler = getattr(polyfill_module, polyfill_name)
        original_fn = polyfill_handler.__torch_dynamo_original__
        trace_rules._builtin_function_ids.remove(id(original_fn))
<|endoftext|>"""
Python polyfills for traceback
"""

import traceback
from traceback import StackSummary
from types import TracebackType

from ..decorators import substitute_in_graph


__all__ = ["extract_tb", "clear_frames"]


@substitute_in_graph(traceback.extract_tb, can_constant_fold_through=True)
def extract_tb(tb: TracebackType | None, limit: int | None = None) -> StackSummary:
    if tb is None:
        return traceback.StackSummary.from_list([])
    # pyrefly: ignore [implicit-any]
    frame_summary = []
    while tb is not None:
        if limit:
            if len(frame_summary) < limit:
                frame_summary.append(
                    # pyrefly: ignore[missing-attribute]
                    tb.frame_summary
                )
            else:
                break
        else:
            frame_summary.append(tb.frame_summary)  # pyrefly: ignore[missing-attribute]
        tb = tb.tb_next
    return traceback.StackSummary.from_list(frame_summary)


@substitute_in_graph(traceback.clear_frames, can_constant_fold_through=True)
def clear_frames(tb: TracebackType | None) -> None:
    # no-op
    return None
<|endoftext|>"""
Python polyfills for common builtins.
"""

# NOTE: 1. Please do not import any submodule in the directory here to avoid circular imports.
#       2. While adding a new polyfill module, also add it to POLYFILLED_MODULE_NAMES in loader.py.
#          Add it in the TYPE_CHECKING block below as well.

import types
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from itertools import repeat as _repeat
from operator import eq, ge, gt, le, lt, ne
from typing import Any, TYPE_CHECKING, TypeGuard, TypeVar
from typing_extensions import TypeIs

import torch


T = TypeVar("T")
U = TypeVar("U")
C = TypeVar("C")


if TYPE_CHECKING:
    from ..utils import dict_keys

    # Load by torch._dynamo.polyfills.loader
    # See also the POLYFILLED_MODULE_NAMES in torch/_dynamo/polyfills/loader.py
    # Put the submodules here to avoid circular imports
    from . import (
        _collections as _collections,
        builtins as builtins,
        functools as functools,
        io as io,
        itertools as itertools,
        operator as operator,
        os as os,
        pytree as pytree,
        struct as struct,
        sys as sys,
        torch_c_nn as torch_c_nn,
        traceback as traceback,
    )

from torch.overrides import BaseTorchFunctionMode


# These classes handle support for TorchFunctionModes across
# graph breaks
# Today the TorchFunctionMode enter (for the classes we support)
# simply pushes the mode onto the stack. Since after this occurs
# the stack is mutated, and we replay these mutations, we don't need
# any cleanup logic to be run once the graph break occurs, we simply replay
# these mutations to ensure at the graph break the torch function mode stack is correct
#  and reconstruct the torch function mode stack normally
# when we compile the resume function on the other side of the break.
# However, to ensure we exit properly
# in the resume function, we need to re-enter the contexts as we do other contexts.
# These contexts do nothing on enter, but provide the correct exit logic to ensure
# the stack state is correct.
class NoEnterTorchFunctionMode(BaseTorchFunctionMode):
    def __enter__(self) -> None:
        pass


# Used by WrappedUserFunctionVariable and similar to inline decorated function
# calls with bytecode backing. Without this, the context enter/exit happens in
# Python-level VT code, so a nested graph break inside `fn` would skip applying
# the context in the compiled fn/resume. By inlining through this polyfill, the
# `with` statement has real bytecode that the resume function can continue from.
def _fn_with_ctx(ctx: Any, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    with ctx:
        return fn(*args, **kwargs)


def index(
    iterator: Iterator[T], item: T, start: int = 0, end: int | None = None
) -> int:
    from itertools import islice

    for i, elem in islice(enumerate(iterator), start, end):
        if elem is item or elem == item:
            return i
    # This will not run in dynamo
    raise ValueError(f"{item} is not in {type(iterator)}")


def repeat(item: T, count: int) -> Iterator[T]:
    for _ in range(count):
        yield item


def radians(x: float) -> float:
    import math

    return math.pi / 180.0 * x


def impl_IS_MAPPING(a: object) -> TypeIs[Mapping[Any, Any]]:
    return isinstance(a, Mapping)


def impl_MATCH_SEQUENCE(a: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(a, Sequence) and not isinstance(a, (str, bytes, bytearray))


def _match_class_attr(obj: object, name: str, seen: set[str]) -> object:
    if name in seen:
        raise TypeError(f"{type(obj)} got multiple sub-patterns for attribute {name}")

    attr = getattr(obj, name)
    seen.add(name)
    return attr


def impl_MATCH_CLASS(
    subject: object, cls: type, nargs: int, kwargs: tuple[str, ...]
) -> tuple[object, ...] | None:
    if not isinstance(cls, type):
        raise TypeError("called match pattern must be a class")

    if not isinstance(subject, cls):
        return None

    typ = type(subject)
    match_self = False
    match_args = ()

    attrs = []
    seen = set()

    if nargs:
        if hasattr(typ, "__match_args__"):
            match_args = typ.__match_args__

            if not isinstance(match_args, tuple):
                raise TypeError(
                    f"{typ}.__match_args__ must be a tuple, (got {type(match_args)})"
                )

            for name in match_args[:nargs]:
                if not isinstance(name, str):
                    raise TypeError(
                        f"__match_args__ elements must be strings (got {type(name)})"
                    )
                attrs.append(_match_class_attr(subject, name, seen))
        else:
            # We should somehow check if the type has TPFLAGS_MATCH_SELF set
            # match_self is only true if TPFLAGS_MATCH_SELF is set, but there is
            # no way to check for it directly in Python. So we assume it is set
            # if there are no __match_args__
            match_self = True
            attrs.append(subject)

        allowed = 1 if match_self else len(match_args)
        if allowed < nargs:
            raise TypeError(
                f"accepts {allowed} positional sub-patterns ({nargs} given)"
            )

    for name in kwargs:
        attrs.append(_match_class_attr(subject, name, seen))

    return tuple(attrs)


def impl_MATCH_KEYS(obj: Mapping[T, U], keys: tuple[T, ...]) -> tuple[U, ...] | None:
    if not isinstance(obj, Mapping):
        raise AssertionError(f"Expected a Mapping, got {type(obj)}")
    if all(key in obj for key in keys):
        return tuple(obj[key] for key in keys)
    else:
        return None


def impl_CONTAINS_OP_fallback(a: T, b: Iterable[T]) -> bool:
    # performs fallback "a in b"
    # CPython: PySequence_Contains → _PySequence_IterSearch → PyObject_GetIter
    # PyObject_GetIter itself falls back to PySequence_GetItem when tp_iter is NULL.
    if hasattr(b, "__iter__"):
        for x in b:
            if x == a:
                return True
        return False
    if hasattr(b, "__getitem__"):
        i = 0
        while True:
            try:
                if b.__getitem__(i) == a:
                    return True
                i += 1
            except IndexError:
                return False
    raise TypeError(f"argument of type {type(b)} is not iterable")


def accumulate_grad(
    x: torch.Tensor,
    variable_grad: torch.Tensor | None,
    new_grad: torch.Tensor | None,
) -> torch.Tensor | None:
    # polyfills according to the Gradient Layout Contract
    if new_grad is None:
        return variable_grad
    new_grad_strided = torch.empty_like(x)
    new_grad_strided.copy_(new_grad)
    if variable_grad is None:
        return new_grad_strided
    elif variable_grad.is_sparse and not new_grad_strided.is_sparse:
        return new_grad_strided + variable_grad
    elif torch.is_grad_enabled():
        return variable_grad + new_grad_strided
    else:
        variable_grad.add_(new_grad_strided)
        return variable_grad


def accumulate_grad_no_alias(
    x: torch.Tensor,
    variable_grad: torch.Tensor | None,
    new_grad: torch.Tensor | None,
) -> torch.Tensor | None:
    # Mirrors inductor::accumulate_grad_: same gradient layout logic as
    # accumulate_grad, but the returned grad must not alias any input.
    if new_grad is None:
        if variable_grad is None:
            return None
        return variable_grad.clone()
    new_grad_strided = torch.empty_like(x)
    new_grad_strided.copy_(new_grad)
    if variable_grad is None:
        return new_grad_strided
    elif variable_grad.is_sparse and not new_grad_strided.is_sparse:
        return new_grad_strided + variable_grad
    elif torch.is_grad_enabled():
        return variable_grad + new_grad_strided
    else:
        result = variable_grad.clone()
        result.add_(new_grad_strided)
        return result


# This mirrors
# https://github.com/python/cpython/blob/a1c52d1265c65bcf0d9edf87e143843ad54f9b8f/Objects/listobject.c#L3352-L3413
def list_cmp(
    op: Callable[[Any, Any], bool], left: Sequence[T], right: Sequence[T]
) -> bool:
    """emulate `(1,2,3) > (1,2)` etc"""

    # Optimization: For equality, short-circuit if lengths differ
    # This avoids iterating through elements and triggering guards on SymInts
    left_len = len(left)
    right_len = len(right)

    if op is eq and left_len != right_len:
        return False
    if op is ne and left_len != right_len:
        return True

    # Apply `op` to the first pair that differ
    for a, b in zip(left, right):
        if a != b:
            return op(a, b)

    # No more pairs to compare, so compare sizes.
    return op(left_len, right_len)


def dict___eq__(d: dict[T, U], other: dict[T, U]) -> bool:
    if (len(d) != len(other)) or (d.keys() != other.keys()):
        return False

    if all(isinstance(a, OrderedDict) for a in (d, other)):
        return list(d.items()) == list(other.items())

    # CPython's dict_equal uses PyObject_RichCompareBool for value
    # comparison, which has an identity shortcut (if v is w, eq is True).
    # This matters for NaN: {k: nan} == {k: nan} is True when same nan.
    for k, v in d.items():
        ov = other[k]
        if v is not ov and v != ov:
            return False

    return True


def dictview_richcompare(
    op: Callable[[Any, Any], bool], self: Iterable[T], other: Iterable[T]
) -> bool:
    """Mirrors dictview_richcompare for dict_keys/dict_items vs set/frozenset.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L5952-L6010
    Uses len() and ``in`` so that Dynamo traces through sq_length/sq_contains.
    """
    len_self = len(self)  # type: ignore[arg-type]
    len_other = len(other)  # type: ignore[arg-type]

    if op is eq:
        if len_self != len_other:
            return False
        return all(item in other for item in self)
    if op is ne:
        if len_self != len_other:
            return True
        return not all(item in other for item in self)
    if op is lt:
        if len_self >= len_other:
            return False
        return all(item in other for item in self)
    if op is le:
        if len_self > len_other:
            return False
        return all(item in other for item in self)
    if op is gt:
        if len_self <= len_other:
            return False
        return all(item in self for item in other)
    # ge
    if len_self < len_other:
        return False
    return all(item in self for item in other)


def assert_dict_equal(
    self_: Any, d1: dict[T, U], d2: dict[T, U], msg: str | None = None
) -> None:
    self_.assertTrue(d1 == d2, msg)


def assert_multi_line_equal(
    self_: Any, first: T, second: T, msg: str | None = None
) -> None:
    return self_.assertTrue(first == second, msg)


# The original impl. uses difflib
def assert_sequence_equal(
    self_: Any,
    seq1: Sequence[T],
    seq2: Sequence[T],
    msg: str | None = None,
    seq_type: type[Any] | None = None,
) -> None:
    return self_.assertTrue(seq1 == seq2, msg)


def getattr_and_trace(*args: Any, **kwargs: Any) -> Any:
    wrapper_obj = args[0]
    attr_name = args[1]
    fn = getattr(wrapper_obj, attr_name)
    return fn(*args[2:], **kwargs)


def getattr_and_trace_no_nested_graph_breaks(*args: Any, **kwargs: Any) -> Any:
    with torch._dynamo.disable_nested_graph_breaks():
        return getattr_and_trace(*args, **kwargs)


def mapping_get(obj: Mapping[T, U], key: T, value: U | None = None, /) -> U | None:
    try:
        return obj.__getitem__(key)
    except KeyError:
        return value


def instantiate_user_defined_class_object(
    cls: type[T], /, *args: Any, **kwargs: Any
) -> T:
    obj = cls.__new__(cls, *args, **kwargs)

    # Only call __init__ if the object's type is a subclass of cls.
    # CPython uses PyType_IsSubtype(Py_TYPE(obj), type) at the C level, which does NOT
    # go through metaclass __instancecheck__. Using isinstance() here would be wrong
    # for classes with custom __instancecheck__ (e.g. torch.ByteStorage).
    # Reference: https://github.com/python/cpython/blob/3.12/Objects/typeobject.c#L1670-L1673
    if issubclass(type(obj), cls):
        obj.__init__(*args, **kwargs)
    return obj


def mutable_mapping_update(
    self,
    data: Mapping[T, U] | Iterable[tuple[T, U]] = (),
    /,
    **kwargs: Any,
) -> None:
    if isinstance(data, Mapping):
        # Merge standard mapping with PyMapping_Items
        for key, value in data.items():
            self[key] = value
    # FIXME: Enabling the `elif`-branch below needs too many `VariableClass.call_obj_hasattr` changes.
    #   >>> class Foo:
    #   ...     def __init__(self):
    #   ...         self.keys = lambda: ['a', 'b', 'c']  # not required to be a method
    #   ...
    #   ...     def __getitem__(self, key):
    #   ...         return 0
    #   ...
    #   >>> dict(Foo())
    #   {'a': 0, 'b': 0, 'c': 0}
    #
    # > This is a rare case, so we comment it out for now.
    #
    # elif hasattr(data, "keys"):
    #     # Merge mapping-like object with PyMapping_Keys + PyObject_GetItem
    #     for key in data.keys():
    #         self[key] = data[key]
    else:
        if not isinstance(data, Iterable):
            raise TypeError(f"{type(data).__name__!r} object is not iterable")
        # Likely a sequence of pairs
        for key, value in data:
            self[key] = value

    if kwargs:
        for key, value in kwargs.items():
            self[key] = value


# Used with something like dict(obj)
def construct_dict(
    cls: type[T],
    data: Mapping[object, object] | Iterable[tuple[object, object]] = (),
    /,
    **kwargs: Any,
) -> T:
    self = cls.__new__(cls)
    mutable_mapping_update(self, data, **kwargs)
    return self


def foreach_map_fn(*args: Any) -> Any:
    op = args[0]
    new_args: list[Any] = []
    at_least_one_list = False
    for arg in args[1:]:
        if not isinstance(arg, (list, tuple)):
            new_args.append(_repeat(arg))
        else:
            at_least_one_list = True
            new_args.append(arg)

    # Just apply op once to args if there are no lists
    if not at_least_one_list:
        return op(*args[1:])

    out = []
    for unpacked in zip(*new_args):
        out.append(op(*unpacked))

    return out


def foreach_lerp_inplace(
    self,
    end: list[torch.Tensor] | tuple[torch.Tensor, ...],
    weight: float | int | torch.Tensor,
) -> None:
    # Decompose lerp via addcmul_ for FMA.  Uses the same dual-formula
    # approach as CUDA's native lerp to get bitwise identical results:
    #   |w| <  0.5  (low):  fma(w, diff, start)
    #   |w| >= 0.5  (high): fma(-(1-w), diff, end)
    # For tensor weights (e.g. 0-dim tensor from tensor betas in Adam) the
    # low formula is always used because the native lerp_scalar lowering
    # would crash on float(weight) for symbolic expressions.
    diff = torch._foreach_sub(end, self)
    if isinstance(weight, torch.Tensor):
        # Select base and weight for the dual formula before a single addcmul:
        #   low  (|w| <  0.5): fma(w,      diff, self)
        #   high (|w| >= 0.5): fma(-(1-w), diff, end)
        mask = weight.abs() >= 0.5
        neg_omw = -(1.0 - weight)
        w = torch.where(mask, neg_omw, weight)
        bases = [torch.where(mask, e, s) for s, e in zip(self, end)]
        w_list = [w] * len(diff)
        torch._foreach_addcmul_(bases, w_list, diff)
        for s, b in zip(self, bases):
            s.copy_(b)
    else:
        abs_weight = weight if weight >= 0 else -weight
        if abs_weight >= 0.5:
            # High formula: end + (-(1-w)) * diff  →  fma(-(1-w), diff, end)
