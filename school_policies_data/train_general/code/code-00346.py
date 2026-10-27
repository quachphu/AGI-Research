    shifted_chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_w_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_w",
    ),
    hermite_polynomial_h=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_h_forward({x}, {y})",
        name="special_hermite_polynomial_h",
    ),
    hermite_polynomial_he=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_he_forward({x}, {y})",
        name="special_hermite_polynomial_he",
    ),
    laguerre_polynomial_l=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"laguerre_polynomial_l_forward({x}, {y})",
        name="special_laguerre_polynomial_l",
    ),
)


def is_buffer_removed(name: str) -> bool:
    return any(
        name in x
        for x in (
            V.graph.removed_buffers,
            V.kernel.removed_buffers,
            V.graph.inplaced_to_remove,
            V.kernel.inplaced_to_remove,
        )
    )


class DeferredLine(DeferredLineBase):
    """A line that can be 'unwritten' by adding name to V.graph.removed_buffers"""

    def __init__(self, name: str, line: str):
        super().__init__(line)
        self.name = name
        if isinstance(line, DeferredLineBase):
            raise AssertionError("line must not be a DeferredLineBase")

    def __call__(self) -> str | None:
        if not is_buffer_removed(self.name):
            return self.line
        return None

    def _new_line(self, line: str) -> DeferredLine:
        return DeferredLine(self.name, line)


class BracesBuffer(IndentedBuffer):
    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")

        return ctx()


class InplacedBuffer(NamedTuple):
    inner_name: str
    other_names: list[str]


@dataclasses.dataclass
class ArgName:
    name: str
    # is_constexpr=True is used to attach a " : tl.constexpr" into the argument list
    is_constexpr: bool = False

    def full_name(self) -> str:
        return f"{self.name}{' : tl.constexpr' if self.is_constexpr else ''}"


class RemovedArg:
    def __str__(self) -> str:
        return "REMOVED"


REMOVED = RemovedArg()


class KernelArgs:
    @staticmethod
    def _lookup(
        prefix: str,
        odict: dict[_T, str | RemovedArg] | dict[_T, str],
        name: _T,
    ) -> str:
        result: str | RemovedArg = odict.get(name, REMOVED)
        if isinstance(result, RemovedArg):
            odict[name] = new_result = f"{prefix}{len(odict)}"
            return new_result
        return result

    def __init__(self) -> None:
        self.input_buffers: dict[str, str] = {}
        self.output_buffers: dict[str, str | RemovedArg] = {}
        self.inplace_buffers: dict[str, InplacedBuffer | RemovedArg] = {}
        self.sizevars: dict[sympy.Expr, str] = {}
        self.workspace_args: list[WorkspaceArg] = []

    def __repr__(self) -> str:
        return "KernelArgs({})".format(
            ", ".join(
                map(
                    repr,
                    [
                        self.input_buffers,
                        self.output_buffers,
                        self.inplace_buffers,
                        self.sizevars,
                    ],
                )
            )
        )

    @staticmethod
    def _buffer_is_marked_removed(name: Any) -> bool:
        # this function is needed by MTIA
        return isinstance(name, RemovedArg)

    def input(self, name: str) -> str:
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        if name in V.graph.removed_buffers:
            raise AssertionError(name)
        if name in self.output_buffers:
            return cast(str, self.output_buffers[name])
        if name in self.inplace_buffers:
            return cast(InplacedBuffer, self.inplace_buffers[name]).inner_name
        if name.startswith("seed"):
            return self._lookup("seed", self.input_buffers, name)
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name: str) -> str:
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        if name in V.graph.removed_buffers:
            raise AssertionError(name)
        if name in self.inplace_buffers:
            return cast(InplacedBuffer, self.inplace_buffers[name]).inner_name
        return self._lookup("out_ptr", self.output_buffers, name)

    def make_inplace(self, input_name: str, output_name: str) -> None:
        if input_name in V.graph.unaligned_buffers:
            V.graph.unaligned_buffers.add(output_name)
        if output_name in self.inplace_buffers:
            raise AssertionError(output_name)
        if input_name in self.inplace_buffers:
            buf = self.inplace_buffers[input_name]
            if isinstance(buf, RemovedArg):
                raise AssertionError("buf must not be a RemovedArg")
            buf.other_names.append(output_name)
            self.inplace_buffers[output_name] = buf
        else:
            alive_buffers = [
                val
                for val in self.inplace_buffers.values()
                if not isinstance(val, RemovedArg)
            ]
            removed_buffers = [
                val
                for val in self.inplace_buffers.values()
                if isinstance(val, RemovedArg)
            ]
            inplace_buffer_idx = len(unique(alive_buffers)) + len(removed_buffers)
            buf = InplacedBuffer(
                f"in_out_ptr{inplace_buffer_idx}",
                [input_name, output_name],
            )
            self.inplace_buffers[input_name] = buf
            self.inplace_buffers[output_name] = buf

    def workspace(
        self, nelem: sympy.Expr, zero_fill: bool, dtype: torch.dtype = torch.uint8
    ) -> tuple[str, str, int]:
        """
        Allocate or extend a workspace buffer of nelem elements.

        This function manages the allocation of a workspace buffer. It either creates
        a new WorkspaceArg or extends an existing one.

        Note:
        - Calling this function will in-place mutate the args by adding or updating
        a WorkspaceArg.
        - The codegen for generating the Python argdefs and call_defs will check
        this field and allocate the buffer accordingly.
        - A new argument "ws_ptr" will be present in the generated code.

        Args:
            nelem (sympy.Expr): The number of elements to allocate.
            zero_fill (bool): Whether to initialize the buffer to zero.
            dtype (torch.dtype): the dtype of the workspace tensor

        Returns:
            Tuple[str, str, int]: A tuple containing:
                - "ws_ptr": A string identifier for the workspace pointer.
                - "workspace_{i}": a graph level unique identifier for
                    the workspace tensor.
                - offset: An integer representing the item offset in the workspace.
        """
        arg = WorkspaceArg(
            count=nelem,
            zero_mode=WorkspaceZeroMode.from_bool(zero_fill),
            device=V.graph.get_current_device_or_throw(),
            outer_name=WorkspaceArg.unique_name(),
            dtype=dtype,
        )
        for i, existing_arg in enumerate(self.workspace_args):
            if WorkspaceArg.can_join(existing_arg, arg):
                offset = existing_arg.count
                self.workspace_args[i] = WorkspaceArg.join(existing_arg, arg)
                return existing_arg.inner_name, existing_arg.outer_name, offset
            if not (
                existing_arg.inner_name != arg.inner_name
                and existing_arg.outer_name != arg.outer_name
            ):
                raise AssertionError(existing_arg)
        self.workspace_args.append(arg)
        return arg.inner_name, arg.outer_name, 0

    def semaphores(self, min_size: sympy.Expr) -> str:
        """
        Lazily allocate a graph-wide semaphores buffer with at least min_size.  This is a single buffer shared by
        all kernels and zero initialized once at graph start.  Each kernel must leave the buffer zeroed on exit.

        Warning: multiple calls to this function will return the same buffer.

        Args:
            min_size: the number of int32 semaphores required

        Returns:
            name of the semaphores buffer
        """
        from torch.fx.experimental.proxy_tensor import _coor_enabled

        current_device = V.graph.get_current_device_or_throw()
        # This name is emitted into the wrapper, so under compile-on-one-rank it cannot
        # carry the rank's device index (the graph is single-device there, so the type
        # alone still distinguishes buffers).
        suffix = "" if _coor_enabled() else f"_{current_device.index}"
        arg = WorkspaceArg(
            count=min_size,
            zero_mode=WorkspaceZeroMode.ZERO_PER_GRAPH,
            dtype=torch.uint32,
            inner_name="sem_ptr",
            outer_name=f"semaphores_{current_device.type}{suffix}",
            device=current_device,
        )
        for existing_arg in self.workspace_args:
            if existing_arg.inner_name == arg.inner_name:
                if arg != existing_arg:
                    raise AssertionError((arg, existing_arg))
        self.workspace_args.append(arg)
        return arg.inner_name

    def seed_offset(self, name: str, value: int) -> str:
        if not isinstance(value, int):
            raise AssertionError((type(value), value))
        # here we are lifting a constant integer into an arg to the kernel to try to get additional cache hits
        value = sympy.Integer(value)
        if value in self.sizevars:
            return self.sizevars[value]
        if name in self.sizevars.values():
            name = (
                f"{name}{sum(1 for v in self.sizevars.values() if v.startswith(name))}"
            )
        self.sizevars[value] = name
        return name

    def size(self, name: sympy.Symbol) -> str:
        if not isinstance(name, sympy.Symbol):
            raise AssertionError((type(name), name))
        if name.name == "seed":
            self.sizevars[name] = "seed"  # don't manage the name of seeds
            return "seed"
        return self._lookup("ks", self.sizevars, name)

    def call_names(self) -> Iterator[str]:
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def arg_name(self, name: str) -> str | None:
        """
        Returns inner name of a given outer name.
        """
        inplaced = self.inplace_buffers.get(name, None)
        if inplaced is not None and not isinstance(inplaced, RemovedArg):
            return inplaced.inner_name
        output_name = self.output_buffers.get(name, None)
        if output_name is not None and not isinstance(output_name, RemovedArg):
            return output_name
        return self.input_buffers.get(name, None)

    def wrap_ptr_arg(self, buf: str, dtype: torch.dtype) -> str:
        return buf

    def wrap_size_arg(self, size: SymbolLike) -> str:
        return str(size)

    def cpp_argdefs(
        self, dtype_to_cpp_type: dict[torch.dtype, str] | None = None
    ) -> tuple[list[str], list[str], list[str]]:
        from .cpp_utils import INDEX_TYPE

        if dtype_to_cpp_type is None:
            from .cpp_utils import DTYPE_TO_CPP

            dtype_to_cpp_type = DTYPE_TO_CPP

        call_args = []
        arg_defs = []
        arg_types = []
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"{cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"const {cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"const {cpp_dtype}*")
        for outer, maybe_inner in self.output_buffers.items():
            if outer in self.inplace_buffers or isinstance(maybe_inner, RemovedArg):
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"{cpp_dtype}* {maybe_inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.sizevars.items():
            if isinstance(outer, sympy.Symbol) and symbol_is_type(
                outer, (SymT.UNBACKED_FLOAT)
            ):
                arg_defs.append(f"const float {inner}")
                arg_types.append("const float")
            else:
                arg_defs.append(f"const {INDEX_TYPE} {inner}")
                arg_types.append(f"const {INDEX_TYPE}")
            call_args.append(self.wrap_size_arg(outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        if self.workspace_args:
            raise AssertionError("Workspace not supported on CPU ")
        return arg_defs, call_args, arg_types

    def python_argdefs(
        self,
    ) -> tuple[list[ArgName], list[str], list[KernelArgType], list[Any]]:
        arg_defs: list[ArgName] = []
        call_args: list[str] = []
        arg_types: list[Any] = []
        precompile_args: list[KernelArgType] = []
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            arg_defs.append(ArgName(inplaced.inner_name))
            call_args.append(inplaced.other_names[-1])
            arg_types.append(V.graph.get_dtype(inplaced.other_names[-1]))
            precompile_args.append(
                TensorArg(
                    name=inplaced.inner_name,
                    buffer=inplaced.other_names[-1],
                    dtype=V.graph.get_dtype(inplaced.other_names[-1]),
                )
            )
        for outer, inner in chain(
            self.input_buffers.items(),
            # pyrefly: ignore [bad-argument-type]
            self.output_buffers.items(),
        ):
            if outer in self.inplace_buffers or isinstance(inner, RemovedArg):
                continue
            arg_defs.append(ArgName(inner))
            call_args.append(outer)
            arg_types.append(V.graph.get_dtype(outer))
            precompile_args.append(
                TensorArg(
                    name=inner,
                    buffer=outer,
                    dtype=V.graph.get_dtype(outer),
                )
            )
        for outer, inner in self.sizevars.items():
            arg_defs.append(ArgName(inner))
            call_args.append(outer)
            arg_types.append(type(outer))
            precompile_args.append(SizeArg(inner, outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        for arg in self.workspace_args:
            arg_defs.append(ArgName(arg.inner_name))
            call_args.append(arg.outer_name)
            precompile_args.append(arg)
            arg_types.append(arg.dtype)
        return arg_defs, call_args, precompile_args, arg_types

    def aliases(self) -> Iterator[tuple[str, str]]:
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            for other in inplaced.other_names:
                if (
                    other in V.graph.inplaced_to_remove
                    or other in V.kernel.inplaced_to_remove
                ):
                    continue
                if other in self.input_buffers:
                    yield self.input_buffers[other], inplaced.inner_name
                if other in self.output_buffers:
                    yield cast(str, self.output_buffers[other]), inplaced.inner_name

    def is_removed(self, name: str) -> bool:
        return isinstance(
            self.output_buffers.get(name, REMOVED), RemovedArg
        ) and isinstance(self.inplace_buffers.get(name, REMOVED), RemovedArg)

    # Includes inplace buffers, excludes removed buffers.  Essentially,
    # after you do a call into this kernel, which buffers actually contain
    # updated data?  Modeled off of python_argdefs.
    def live_output_buffers(self) -> OrderedSet[str]:
        live_outs: OrderedSet[str] = OrderedSet()
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            live_outs.add(inplaced.other_names[-1])
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or isinstance(inner, RemovedArg):
                continue
            live_outs.add(outer)
        return live_outs


class CSEVariable:
    """A CSEVariable is just a name for an expression but it is useful to be able to annotate them on a backend dependent basis.
    To do so, the backends can simply overload `Kernel.create_cse_var`
    The "CSEVariable.update_on_args" method gives you a hook for annotations
    See example of TritonCSEVariable in triton.py
    """

    def __init__(
        self,
        name: str,
        bounds: ValueRanges[Any],
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ):
        super().__init__()
        if not isinstance(bounds, ValueRanges):
            raise AssertionError(type(bounds))
        self.name = name
        self.bounds = bounds
        self.use_count = 1  # track how many times this expression is used
        self.dtype = dtype
        self.shape = shape

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CSEVariable) and other.name == self.name

    def update_on_args(self, name: str, args: Any, kwargs: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


AugmentedKeyT = TypeVar("AugmentedKeyT", default=str)
CSEVariableType = TypeVar("CSEVariableType", bound=CSEVariable, default=CSEVariable)

if TYPE_CHECKING:
    ReductionCacheKey = tuple[
        torch.dtype,
        ReductionType,
        CSEVariable | tuple[CSEVariable, ...],
    ]


class CSE(Generic[CSEVariableType, AugmentedKeyT]):
    """Common subexpression elimination"""

    def __init__(
        self,
        prefix: str = "",
        suffix: str = "",
        name_prefix: str = "tmp",
        iter_buffers: itertools.count[int] | None = None,
        store_cache: MutableMapping[str, CSEVariableType] | None = None,
        reduction_cache: MutableMapping[ReductionCacheKey, CSEVariableType]
        | None = None,
        varname_map: dict[str, CSEVariableType] | None = None,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self._cache: MutableMapping[AugmentedKeyT, CSEVariableType] = {}
        self.name_prefix = name_prefix
        self.store_cache: MutableMapping[str, CSEVariableType] = store_cache or {}
        self.reduction_cache: MutableMapping[ReductionCacheKey, CSEVariableType] = (
            reduction_cache or {}
        )
        self.iter_buffer_ids: itertools.count[int] = iter_buffers or itertools.count()
        self.invalidated_stores: OrderedSet[str] = OrderedSet()
        self.varname_map: dict[str, CSEVariableType] = varname_map or {}

    def invalidate(self, keep_vars: OrderedSet[CSEVariable]) -> None:
        for name, tmp in [*self.store_cache.items()]:
            if tmp not in keep_vars:
                del self.store_cache[name]
                self.invalidated_stores.add(name)
        if keep_vars:
            self._cache = {k: v for k, v in self._cache.items() if v in keep_vars}
        else:
            self._cache = {}

    def clone(self) -> Self:
        return type(self)(
            prefix=self.prefix,
            suffix=self.suffix,
            name_prefix=self.name_prefix,
            iter_buffers=self.iter_buffer_ids,
            store_cache=self.store_cache,
            varname_map=self.varname_map,
            reduction_cache=self.reduction_cache,
        )

    def scoped_copy(self) -> Self:
        """Return a copy of using ScopedDict so changes to *_cache aren't visible in self"""
        new_cse = self.clone()
        new_cse._cache = ScopedDict(self._cache)
        new_cse.reduction_cache = ScopedDict(self.reduction_cache)
        new_cse.store_cache = ScopedDict(self.store_cache)
        return new_cse

    def augment_key(self, cache_key: str) -> AugmentedKeyT:
        "Override this method to augment cache key with backend specifics"
        return cast(AugmentedKeyT, cache_key)

    def put(self, cache_key: str, val: CSEVariableType) -> None:
        self._cache[self.augment_key(cache_key)] = val

    def contains(self, cache_key: str) -> bool:
        return self.augment_key(cache_key) in self._cache

    def try_get(self, cache_key: str) -> CSEVariableType | None:
        return self._cache.get(self.augment_key(cache_key), None)

    def get(self, cache_key: str) -> CSEVariableType:
        return self._cache[self.augment_key(cache_key)]

    def generate(
        self,
        buffer: IndentedBuffer,
        expr: str | CSEVariable | OpsValue | IndentedBuffer | DeferredLineBase,
        *,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        write: bool = True,
        assignment: bool = True,
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        if isinstance(expr, OpsValue):
            expr = expr.value

        if not (write or assignment):
            raise AssertionError("expected write or assignment to be set")
        if isinstance(expr, CSEVariable):
            # If the expressions were always created with all the information, we could
            # assert expr.bounds == bounds, but sometimes the expression is created
            # with the loose ValueRanges.unknown(), so we need to tighten the bounds
            expr.bounds = expr.bounds.tighten(bounds)
            expr.use_count += 1
            return cast(CSEVariableType, expr)
        elif isinstance(expr, IndentedBuffer):
            cache_key = expr.getvalue()
        elif isinstance(expr, DeferredLineBase):
            cache_key = expr.line
        else:
            if not isinstance(expr, str):
                raise AssertionError(f"expected str, got {type(expr)}")
            cache_key = expr
        var = self.try_get(cache_key)
        if shape is None and not assignment:
            # since there's no assignment to a variable, use any shape here
            # other than None to avoid the unknown shape failures
            shape = ()
        if not var:
            var = self.newvar(bounds, dtype, shape)
            self.put(cache_key, var)
            if write:
                if V.kernel.current_node:
                    V.kernel.current_node.codegen_originating_info(
                        buffer, only_once=True
                    )
                if isinstance(expr, IndentedBuffer):
                    if assignment:
                        buffer.writeline(f"{self.prefix}{var} =")
                    buffer.splice(expr)
                    buffer.writeline(self.suffix)
                elif isinstance(expr, DeferredLineBase):
                    if not assignment:
                        raise AssertionError("expected assignment to be set")
                    buffer.writeline(
                        expr._new_line(f"{self.prefix}{var} = {expr.line}{self.suffix}")
                    )
                else:
                    if assignment:
                        line = f"{self.prefix}{var} = {expr}{self.suffix}"
                    else:
                        line = f"{expr}{self.suffix}"
                    buffer.writeline(line)

                    # cpp backend cannot determine is_vec at this point
                    if (
                        assignment
                        and (
                            config.test_configs.runtime_triton_dtype_assert
                            or config.test_configs.static_cpp_dtype_assert
                        )
                        and dtype is not None
                        and get_current_backend() != "cpp"
                    ):
                        check_dtype(buffer, var, dtype)

        else:
            var.bounds = var.bounds.tighten(bounds)
            var.use_count += 1

        return var

    def newvar(
        self,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        var_name = f"{self.name_prefix}{next(self.iter_buffer_ids)}"
        var = cast(
            "CSEVariableType", V.kernel.create_cse_var(var_name, bounds, dtype, shape)
        )
        self.varname_map[var_name] = var
        return var

    def namedvar(
        self,
        name: str,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        torch._check_value(
            name not in self.varname_map, lambda: f"duplicate name: {name}"
        )
        var = cast(
            "CSEVariableType", V.kernel.create_cse_var(name, bounds, dtype, shape)
        )
        self.varname_map[name] = var
        return var


class CodeGen:
    def __init__(self) -> None:
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self) -> Self:
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class Kernel(CodeGen, Generic[CSEVariableType]):
    """Base class for generated kernels and their code buffers."""

    newvar_prefix: str = ""
    suffix: str = ""
    overrides: Callable[[], OpsHandler[Any]] | None = None

    def __init__(
        self, args: KernelArgs | None = None, increase_kernel_count: bool = True
    ) -> None:
        super().__init__()
        if increase_kernel_count:
            # pyrefly: ignore [bad-assignment]
            metrics.generated_kernel_count += 1
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()

        self.atomic_add_found = False
        self.num_load = 0
        self.num_store = 0
        self.num_reduction = 0

        self.cse: CSE[CSEVariableType, Any] = CSE(self.newvar_prefix, self.suffix)
        self.must_keep_buffers: OrderedSet[str] = OrderedSet()
        self.store_buffer_names: OrderedSet[str] = OrderedSet()
        self._load_mask: str | None = None
        self._load_other: None | int | float = None
        # OrderedSet in set_current_node
        self.current_node: SchedulerNode | None = None
        self.node_to_bounds: dict[torch.fx.Node, ValueRanges[Any]] | None = None

        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()

        # key: the buffer to write
        # value: the buffer to read and whose memory can be reused for
        #   the buffer specified by key
        self.inplace_update_buffers: dict[str, str] = {}
        # Set minimum number of elements processed per thread.
        self.min_elem_per_thread = 1
        self.kernel_name: str | None = None

    @contextlib.contextmanager
    def set_current_node(self, node: SchedulerNode) -> Iterator[None]:
        prior = self.current_node
        self.current_node = node
        self.node_to_bounds = node._body.bounds().get_bounds()
        try:
            yield
        finally:
            self.current_node = prior

    @contextlib.contextmanager
    def swap_buffers(
        self,
        lb: IndentedBuffer,
        cb: IndentedBuffer | None = None,
        sb: IndentedBuffer | None = None,
    ) -> Iterator[None]:
        if cb is None:
            cb = lb
        if disallow_stores := sb is None:
            sb = IndentedBuffer()
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = cse.scoped_copy()
        try:
            yield
        finally:
            self.loads = loads
            self.compute = compute
            self.stores = stores
            self.cse = cse
            # pyrefly: ignore [unbound-name]
            if disallow_stores:
                if sb:
                    raise AssertionError("unexpected store inside swap_buffers")

    def emit_kernel_override(
        self,
        wrapper,
        src_code: str,
        kernel_name: str,
        node_schedule,
        kernel_path: str,
        get_kernel_metadata,
    ) -> bool:
        """Override kernel emission. Return True if overridden, False to use default.

        External template handlers (e.g. Helion) can override this method
        to implement custom kernel emission to the wrapper.
        """
        return False

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        raise NotImplementedError

    def indirect_load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """A load that depends on an index we have read"""
        prior = self.loads
        try:
            # put the load in the compute section as it might have deps
            self.loads = self.compute
            return self.load(name, index)
        finally:
            self.loads = prior

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        raise NotImplementedError

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        raise NotImplementedError

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        raise NotImplementedError

    def partial_accumulate(
        self,
        name: str,
        reduction_type: ReductionType,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]], tuple[CSEVariable, ...]
        ],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        raise NotImplementedError

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[CSEVariable, ...]:
        raise NotImplementedError

    def var_ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
        raise NotImplementedError

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: CSEVariable | None = None,
    ) -> CSEVariable:
        """
        See [Note: Inductor bucketize op]
        """
        raise NotImplementedError

    @property
    def assert_function(self) -> str:
        raise NotImplementedError

    def indirect_assert(
        self,
        var: CSEVariable | str,
        lower: str | None,
        upper: str | None,
        mask: CSEVariable | str | None = None,
    ) -> str:
        if isinstance(var, CSEVariable):
            var = str(var)
        if not isinstance(var, str):
            raise AssertionError(type(var))
        if not (lower is None or isinstance(lower, str)):
            raise AssertionError(f"expected lower to be None or str, got {type(lower)}")
        if not (upper is None or isinstance(upper, str)):
            raise AssertionError(f"expected upper to be None or str, got {type(upper)}")
        if lower and upper:
            # The conditions need to be in parens because of Python's operator precedence.
            # It'd be less error-prone to use and/or/not, which is supported by triton
            cond = f"({lower} <= {var}) & ({var} < {upper})"
            cond_print = f"{lower} <= {var} < {upper}"
        elif lower:
            cond = f"{lower} <= {var}"
            cond_print = cond
        else:
            if not upper:
                raise AssertionError("expected upper to be set")
            cond = f"{var} < {upper}"
            cond_print = cond

        if mask:
            cond = f"({cond}) | ~({mask})"

        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        raise NotImplementedError

    def index_to_str(self, index: sympy.Expr) -> str:
        raise NotImplementedError

    def __enter__(self) -> Self:
        super().__enter__()
        if not self.overrides:
            raise AssertionError("expected overrides to be set")
        self.exit_stack.enter_context(
            V.set_ops_handler(CSEProxy(self, self.overrides()))
        )
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_kernel_local_buffers()
        super().__exit__(exc_type, exc_val, exc_tb)

    def remove_kernel_local_buffers(self) -> None:
        """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.

        Note that V.graph.scheduler can be None when codegening triton template
        kernels.
        """
        scheduler = V.graph.scheduler
        if not scheduler:
            return
        fused_node_names = OrderedSet(
            scheduler.name_to_buf[buf].defining_op_name()
            for buf in self.store_buffer_names
            if buf in scheduler.name_to_buf
        )
        names_to_remove: OrderedSet[str] = OrderedSet()
        for name in self.store_buffer_names:
            if (
                name not in self.must_keep_buffers
                and name not in self.args.input_buffers
                and scheduler.can_buffer_be_removed_through_fusion(
                    name, fused_node_names
                )
            ):
                self.num_store -= 1
                names_to_remove.add(name)

        for name in names_to_remove:
            if name in self.args.inplace_buffers:
                buf = self.args.inplace_buffers[name]
                if isinstance(buf, RemovedArg):
                    continue
                remove = all(n in names_to_remove for n in buf.other_names)
                if remove:
                    self.remove_inplace_buffer(name)
                self.inplaced_to_remove.add(name)
            else:
                self.remove_buffer(name)

    def remove_buffer(self, name: str) -> None:
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        log.debug("remove_buffer(%r)", name)
        self.args.output_buffers[name] = REMOVED
        self.removed_buffers.add(name)

    def remove_inplace_buffer(self, name: str) -> None:
        log.debug("removing_inplace_buffer(%r)", name)
        self.args.inplace_buffers[name] = REMOVED
        self.removed_buffers.add(name)

    def rename_indexing(
        self, index: list[sympy.Expr] | tuple[sympy.Expr, ...] | sympy.Expr
    ) -> sympy.Expr:
        # adds the necessary kernel args for index expressions
        # and renames variables in index expressions to kernel arg names
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        index = V.graph.sizevars.simplify(index)
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        replacements = {
            x: self.args.size(x)
            for x in sorted_symbols
            if symbol_is_type(
                x,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                    SymT.UNBACKED_FLOAT,
                ),
            )
        }
        return sympy_subs(index, replacements)

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:
        return CSEVariable(*args, **kwargs)

    def arg_name(self, node: IRNode) -> str | None:
        """
        Returns arg name of a given input or output node.
        """
        if node is None:
            return None
        return self.args.arg_name(node.get_name())

    def record_op_trace(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
    ) -> None:
        pass


@dataclasses.dataclass
class OptimizationContext:
    key: ClassVar[str] = "opt_ctx"

    dtype: torch.dtype | None = None
    ops_name: str = ""


@functools.cache
def jinja2_env() -> Any:
    try:
        import jinja2

        return jinja2.Environment(
            undefined=jinja2.StrictUndefined,
        )
    except ImportError:
        return None


class KernelTemplate:
    """
    Base class for defining kernel templates.

    Children classes: TritonTemplate, CUTLASSTemplate
    """

    @staticmethod
    def indent_except_first(
        source: str, num_indents: int, indents_spacing: int = 4
    ) -> str:
        lines = source.splitlines(True)
        if len(lines) > 1:
            lines[1:] = [
                (" " * indents_spacing * num_indents) + line for line in lines[1:]
            ]
        return "".join(lines)

    @staticmethod
    def _template_from_string(source: str) -> Any:
        env = jinja2_env()
        if env is None:
            return None
        env.filters["indent_except_first"] = KernelTemplate.indent_except_first
        from jinja2 import TemplateSyntaxError

        try:
            return env.from_string(source)
        except TemplateSyntaxError as e:

            class DetailedTemplateSyntaxError(TemplateSyntaxError):
                def __init__(self, original_error: TemplateSyntaxError) -> None:
                    super().__init__(
                        original_error.message,
                        original_error.lineno,
                        original_error.name,
                        original_error.filename,
                    )
                    self.original_error = original_error

                def __str__(self) -> str:
                    error_info = f"Error in template at line {self.lineno}\n"
                    error_info += f"Error message: {self.message}\n"
                    if hasattr(self.original_error, "source"):
                        lines = self.original_error.source.split("\n")
                        error_info += "Context:\n"
                        start = max(0, self.lineno - 2)
                        end = min(len(lines), self.lineno + 2)
                        for i in range(start, end):
                            if i == self.lineno - 1:
                                error_info += f"{i + 1}: --> {lines[i]}\n"
                                if hasattr(self.original_error, "column"):
                                    error_info += (
                                        "     "
                                        + " " * (self.original_error.column - 1)
                                        + "^\n"
                                    )
                            else:
                                error_info += f"{i + 1}:     {lines[i]}\n"
                    return error_info

            raise DetailedTemplateSyntaxError(e) from e

    @staticmethod
    def _fake_get_dtype(
        fake_outs: list[Buffer] | Buffer,
    ) -> Callable[[str], torch.dtype]:
        _get_dtype_real = V.graph.get_dtype
        if isinstance(fake_outs, (list, tuple)):
            lookup = {buf.get_name(): buf.get_dtype() for buf in fake_outs}
        else:
            lookup = {fake_outs.get_name(): fake_outs.get_dtype()}

        def get_dtype(name: str) -> torch.dtype:
            result = lookup.get(name)
            if result is not None:
                return result
            return _get_dtype_real(name)

        return get_dtype

    def __init__(self, name: str, hash: str | None = None) -> None:
        self.name = name
        self._hash = hash

    @property
    def uid(self) -> str:
        """
        entry point to override for templates to ensure a uid e.g. through a prefix

        the purpose of this is that every KernelTemplate/ExternKernelChoice is unique
        in the system, but reproducible e.g. restarting pytorch should yield the same id
        """
        # TODO(coconutruben): add some central registration to assert on global uniqueness
        return self.name

    @property
    def src_hash(self) -> str | None:
        """
        source hash for a Template.

        Templates can optionally provide a src hash to make it easier to cache/validate that
        a template has not changed from one version to another. Override this if that detection
        is different for your specific Template
        """
        return self._hash

    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        """
        Maybe generates a new ChoiceCaller and returns it, or None if generation fails.

        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """
        temp_choices: list[Any] = []
        result = self.maybe_append_choice(temp_choices, **kwargs)
        if result is None and len(temp_choices) == 1:
            return temp_choices[0]
        return None

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> NotImplementedError | None:
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """

        try:
            choices.append(self.generate(**kwargs))
            return None
        except NotImplementedError as e:
            log.info(
                "Cannot Append Choice: %s. KernelTemplate type is %s",
                e,
                type(self),
                stack_info=log.getEffectiveLevel() < logging.INFO,
            )
            return e

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        """
        Generates a ChoiceCaller instance from the given arguments.
        """

        raise NotImplementedError


class CSEProxy(DefaultHandler):
    """A ops handler that proxies calls to `kernel` and its
    handler and returns `CSEVariable`s with correct shape and dtype.
    """

    name = "CSEProxy"

    def __init__(self, kernel: Kernel[Any], parent_handler: OpsHandler[Any]):
        super().__init__()
        from ..bounds import ValueRangeAnalysis

        self.vr_analysis = ValueRangeAnalysis()
        self.kernel = kernel
        self.parent_handler = parent_handler

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bounds = self._bound_variable(name, *args, **kwargs)

        value = getattr(self.parent_handler, name)(*args, **kwargs)
        dtype_handler = DtypePropagationOpsHandler()
        shape_handler = ShapePropagationOpsHandler()

        backend = get_current_backend()

        shape_op = getattr(shape_handler, name)
        output_dtype = None
        output_shape = None

        if name == "masked" and backend == "triton":
            output_dtype = value.dtype
            output_shape = value.shape
        elif name == "masked" and backend == "cpp":
            output_dtype = V.interpreter.current_node.meta.get(
                OptimizationContext.key, None
            ).dtype
            # TODO: fix me
            output_shape = None
        elif backend in ("triton", "cpp", "mps"):
            dtype_op = getattr(dtype_handler, name)
            output_dtype = dtype_op(*args, **kwargs)
            output_shape = shape_op(*args, **kwargs)

        if backend in ("triton", "cpp"):
            # maybe there are some exceptions on mps?
            if output_dtype is None:
                raise AssertionError("expected output_dtype to be not None")

        output_idx = 0

        def do_cse(v: str | CSEVariable) -> CSEVariable:
            # we tree_map over the output, so we need to fetch corresponding dtype
            nonlocal output_idx
            var_dtype: torch.dtype | None = (
                output_dtype[output_idx]
                if isinstance(output_dtype, (list, tuple))
                else output_dtype
            )
            var_shape: BlockShapeType = (
                output_shape[output_idx]  # type: ignore[assignment]
                if isinstance(output_shape, (list, tuple))
                and len(output_shape) > 0
                and isinstance(output_shape[0], (list, tuple))
                else output_shape
            )
            output_idx += 1

            # some cpp op implementations don't set the dtype
            if isinstance(v, CSEVariable):
                if backend == "cpp" and v.dtype is None:
                    v.dtype = var_dtype
                if v.shape is None:
                    v.shape = var_shape

            csevar = V.kernel.cse.generate(
                V.kernel.compute,
                v,
                bounds=bounds,
                # pyrefly: ignore[bad-argument-type]
                dtype=output_dtype,
                shape=output_shape,
            )

            csevar.update_on_args(name, args, kwargs)

            if (
                config.test_configs.runtime_triton_dtype_assert
                or config.test_configs.static_cpp_dtype_assert
            ) and var_dtype is not None:
                check_dtype(V.kernel.compute, csevar, var_dtype)

            if (
                config.test_configs.runtime_triton_shape_assert
                and _is_runtime_triton_assert_target()
            ):
                shape_to_check = csevar.shape if csevar.shape is not None else var_shape
                if shape_to_check is not None:
                    check_shape(V.kernel.compute, csevar, shape_to_check)

            if config.runtime_triton_nan_asserts:
                check_nan(V.kernel.compute, csevar)

            return csevar

        result = pytree.tree_map(do_cse, value)
        self.kernel.record_op_trace(name, args, kwargs, result)
        return result

    def _bound_variable(self, name: str, *args: Any, **kwargs: Any) -> ValueRanges[Any]:
        """
        If the variable comes from an FX node, we forward the bound we have already computed
        Else, if the variable when codegen'ing another op, we try to compute its bounds
        """
        from ..bounds import ValueRangeAnalysis
        from ..select_algorithm import TritonTemplateKernel
        from .cutlass.kernel import CUTLASSTemplateKernel

        if isinstance(V.kernel, TritonTemplateKernel):
            return ValueRanges.unknown()

        if isinstance(V.kernel, CUTLASSTemplateKernel):
            return ValueRanges.unknown()

        if isinstance(V.interpreter, NullHandler):
            return ValueRanges.unknown()

        fx_node = V.interpreter.current_node
        if fx_node.target == name and self.kernel.node_to_bounds is not None:
            if not isinstance(self.kernel.node_to_bounds, dict):
                raise AssertionError(type(self.kernel.node_to_bounds))
            return self.kernel.node_to_bounds.get(fx_node, ValueRanges.unknown())
        elif config.compute_all_bounds and hasattr(ValueRangeAnalysis, name):
            # These create lots of inner strings. We would need to compute the bounds at the ops
            # We will also likely not get much from computing VRs on these nodes
            if any(s in fx_node.target for s in ("set_indirect", "reduction", "scan")):
                return ValueRanges.unknown()

            # We assume that the inputs come from `ops.` and are not strings. If you want to generate
            # intermediary strings, wrap them in CSE variables with properly initialised bounds.

            # If there is no FX bound but we know how to compute one we do so
            if kwargs:
                raise AssertionError("expected no kwargs")

            def arg_to_bound(x: Any) -> Any:
                if isinstance(x, CSEVariable):
                    return x.bounds
                elif isinstance(x, sympy.Expr):
                    return bound_sympy(x)
                else:
                    return x

            arg_bounds = list(map(arg_to_bound, args))
            return getattr(self.vr_analysis, name)(*arg_bounds)
        return ValueRanges.unknown()

    def indirect_indexing(
        self,
        var: CSEVariable,
        size: sympy.Expr | int,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        if isinstance(size, int):
            size = sympy.Integer(size)
        if not isinstance(size, sympy.Expr):
            raise AssertionError((type(size), size))
        # Skip CSE since this doesn't return an expression

        if var.bounds.lower < 0:
            if wrap_neg:
                stm = ops.add(var, ops.index_expr(size, torch.long))
                # Mixed negative and non-negative
                if var.bounds.upper >= 0:
                    lt = ops.lt(var, 0)
                    stm = ops.where(lt, stm, var)
            else:
                stm = var

            # Propagate bounds as we know how to compute them properly.
            # When wrap_neg=False, negative indices stay negative and must keep
            # their original bounds so lower-bound checks are not elided.
            new_bounds = var.bounds if not wrap_neg else ValueRanges.unknown()
            if (
                wrap_neg
                and var.bounds != ValueRanges.unknown()
                and isinstance(size, sympy.Number)
            ):
                # Take the negative part of the bound and add size to it
                # Then take union of that and the positive part
                # This is a tighter bound than that of a generic ops.where, as we have info on the cond
                neg_bounds = var.bounds & ValueRanges(-int_oo, -1)
                new_bounds = ValueRanges(
                    neg_bounds.lower + size, neg_bounds.upper + size
                )
                # We don't have a good way of representing the empty range
                if var.bounds.upper >= 0:
                    pos = var.bounds & ValueRanges(0, int_oo)
                    new_bounds = new_bounds | pos

            var = self.kernel.cse.generate(
                self.kernel.compute,
                stm,
                bounds=new_bounds,
                dtype=var.dtype,
                shape=var.shape,
            )

        sympy_var = self.parent_handler.indirect_indexing(var, size, check)
        if generate_assert(check):
            assert_lower = not (var.bounds.lower >= 0)
            # value ranges cannot x < s when x and s are symbols
            assert_upper = not isinstance(size, sympy.Number) or not (
                var.bounds.upper < size
            )
            self.kernel.check_bounds(sympy_var, size, assert_lower, assert_upper)
        return sympy_var

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        return self.kernel.check_bounds(expr, size, lower, upper)

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        if name in self.kernel.cse.invalidated_stores:
            # A load from an invalidated store requires us to
            # keep the actual buffer around
            V.kernel.must_keep_buffers.add(name)
        if free_symbol_is_type(index, SymT.TMP):
            return self.kernel.indirect_load(name, index)
        store_cache = self.kernel.cse.store_cache
        if name in store_cache:
            return store_cache[name]
        out = self.kernel.load(name, index)
        # count load that is not in the store_cache, and also not in the
        # cse cache.
        if out.use_count == 1:
            self.kernel.num_load += 1
        self.kernel.record_op_trace("load", (name, index), {}, out)
        return out

    def _update_store_cache(self, name: str, value: CSEVariable) -> None:
        self.kernel.cse.store_cache[name] = value
        if self.kernel.current_node and name in V.graph.name_to_buffer:
            buf = self.kernel.current_node.get_output(name)
            for other_name in buf.get_mutations():
                self.kernel.cse.store_cache[other_name] = value

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        self.kernel.store_buffer_names.add(name)
        # Update store cache when mode is None or "tma"
        if mode != "atomic_add":
            self._update_store_cache(name, value)
        if name not in V.graph.removed_buffers:
            self.kernel.store(name, index, value, mode=mode)
            self.kernel.num_store += 1
        self.kernel.record_op_trace("store", (name, index, value, mode), {})

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        self.kernel.device_assert_async(cond, msg)
        self.kernel.record_op_trace("device_assert_async", (cond, msg), {})

    # pyrefly: ignore [bad-override]
    def partial_accumulate(self, *args: Any) -> None:
        self.kernel.partial_accumulate(*args)

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        self.kernel.store_buffer_names.add(name)
        self._update_store_cache(name, value)

        if name not in V.graph.removed_buffers:
            self.kernel.num_store += 1
            return self.kernel.store_reduction(name, index, value)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        self.kernel.num_reduction += 1
        return self.kernel.reduction(dtype, src_dtype, reduction_type, value)

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]],
            tuple[CSEVariable, ...],
        ],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        return self.kernel.scan(dtypes, combine_fn, values)

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[CSEVariable, ...]:
        return self.kernel.sort(dtypes, values, stable, descending)

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: CSEVariable | None = None,
    ) -> CSEVariable:
        """
        [Note: Inductor bucketize op]

        Inputs:
        -------
        values: the values to be bucketized.
        boundaries: a tuple containing
          (a) the name of the boundaries tensor (which must be sorted, unless
          the sorting tensor is present),
          (b) the length of the tensor in the last dimension (i.e. the length of
          one set of boundaries),
          (c) the number of elements in the underlying storage (i.e. the length
          of the flattened tensor, ignoring striding), and
          (d) the stride of the tensor in the last dimension.
        boundary_indices: indices into a flattened version of the boundaries
        tensor, of the same size and shape as "values".  Each index points to
        the first element in the set of boundaries to be used for the
        corresponding value.
        indexing_dtype: the dtype to use when indexing into the boundaries
        tensor.  This must be int64 or int32.  This additionally specifies the
        dtype of the return value.
        right: see "Details" below.
        sorter: an optional tuple containing
          (a) the name of an optional sorting tensor, used to access unsorted
          boundaries without reordering the boundaries tensor, and
          (b) the stride of the tensor in the last dimension.
        The values in the sorting tensor are used as indices into the *last*
        dimension of the boundaries tensor, with all other indices matching.
        The size of the sorting and boundaries tensors must be equivalent.
        sorter_indices: must be present if the sorting array is present; see
        "boundary_indices" for the equivalent definition for the boundaries
        tensor.

        Output:
        -------
        The buckets each value belongs in, within a given set of boundaries.  0
        indicates a position before the first boundary, and len(boundaries_set)
        represents a position after the last boundary.

        Details:
        --------
        Given a value and a set of boundaries, calculate the bucket that each
        value belongs to.  This works differently in 1-D and N-D cases.

        for values [[-1, 0, 1, 2], [3, 4, 5, 9]], boundaries [0, 4, 4, 8], right=True
        return =   [[ 0, 1, 1, 1], [1, 3, 3, 4]].

        for values [[-1, 0, 1, 2], [3, 4, 5, 9]], boundaries [[0, 4], [4, 8]], right=True
        return =   [[ 0, 1, 1, 1], [0, 1, 1, 2]]

        Note that in the N-D boundaries case, the shape of "values" and
        "boundaries" must match in every dimension _except_ the last.

        When right == False, bucket i refers to range (boundaries[i], boundaries[i+1]].
        When right == True,  bucket i refers to range [boundaries[i], boundaries[i+1]).

        Boundaries must be non-decreasing, or a sorter must be provided which
        would re-index offsets in a non-decreasing order (e.g. the second output
        of torch.sort(offsets)).  Otherwise, the result is undefined.
        """
        return self.kernel.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )
<|endoftext|>import io
import itertools
import logging
import textwrap
import tokenize
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, cast, TYPE_CHECKING

import sympy
from sympy import Integer, Symbol

import torch
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    import triton

from .. import config, metrics
from ..runtime.hints import DeviceProperties, TritonMeta
from ..runtime.runtime_utils import next_power_of_2
from ..runtime.triton_heuristics import (
    RoundRobinComboKernelGrid,
    SequentialComboKernelGrid,
    SequentialFlattenComboKernelGrid,
)
from ..scheduler import BaseSchedulerNode
from ..stream_utils import (
    coor_benchmark_device_idx,
    coor_device_str,
    get_raw_stream_name,
)
from ..utils import (
    clear_on_fresh_cache,
    DeferredLineBase,
    Placeholder,
    triton_version_uses_attrs_dict,
)
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    IndentedBuffer,
    InplacedBuffer,
    Kernel,
    PythonPrinter,
    RemovedArg,
    SizeArg,
    TensorArg,
    WorkspaceArg,
)
from .simd import NodeInfo, prefix_is_reduction, SIMDScheduling
from .simd_kernel_features import SIMDKernelFeatures
from .triton import TritonKernel
from .triton_utils import (
    config_of,
    equal_1_arg_indices,
    is_unaligned_buffer,
    signature_to_meta,
)


# Default block sizes used when combo kernel autotuning is disabled.
DEFAULT_COMBO_BLOCK_SIZE_1D = 1024
DEFAULT_COMBO_BLOCK_SIZE_2D = 32


log = logging.getLogger(__name__)
pexpr = PythonPrinter().doprint
LARGE_NUMELS = 51_200_000
BLOCK_UTILIZATION = 0.8


def _size_hint(expr: Any) -> int:
    return V.graph.sizevars.optimization_hint(expr, fallback=1)


def _node_partition_log_context(
    node: BaseSchedulerNode, node_info_map: dict[BaseSchedulerNode, NodeInfo]
) -> tuple[Any, ...]:
    node_info = node_info_map[node]
    tiling_hints = tuple(
        (str(dim), _size_hint(numel))
        for dim, numel in sorted(
            node_info.tiling.items(), key=lambda item: str(item[0])
        )
    )
    return (
        bool(node_info.features.is_reduction()),
        node_info.is_persistent_reduction,
        tiling_hints,
        _size_hint(node_info.numel),
        _size_hint(node_info.rnumel),
    )


def _partition_separation_log_context(
    separated_nodes: list[BaseSchedulerNode],
    companion_nodes: list[BaseSchedulerNode],
    node_info_map: dict[BaseSchedulerNode, NodeInfo],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (
        tuple(_node_partition_log_context(n, node_info_map) for n in separated_nodes),
        tuple(_node_partition_log_context(n, node_info_map) for n in companion_nodes),
    )


def _log_partition_separation(
    log_message: str,
    separated_nodes: list[BaseSchedulerNode],
    companion_nodes: list[BaseSchedulerNode],
    node_info_map: dict[BaseSchedulerNode, NodeInfo],
) -> None:
    if log.isEnabledFor(logging.DEBUG):
        _log_partition_separation_once(
            log_message,
            len(separated_nodes),
            _partition_separation_log_context(
                separated_nodes, companion_nodes, node_info_map
            ),
        )


# This diagnostic is otherwise repeated once per equivalent recompile.
@clear_on_fresh_cache
@cache
def _log_partition_separation_once(
    log_message: str,
    num_nodes: int,
    partition_context: tuple[tuple[Any, ...], tuple[Any, ...]],
) -> None:
    log.debug(
        log_message,
        num_nodes,
    )


def _default_custom_combo_kernel_horizontal_partition(
    nodes: list[BaseSchedulerNode],
    triton_scheduling: SIMDScheduling,
    node_info_map: dict[BaseSchedulerNode, NodeInfo],
) -> list[list[BaseSchedulerNode]]:
    """Horizontally partition the given list of nodes into a list of list of nodes where each sublist
    represents a partition. Nodes in different partitions are implemented in different combo kernels.
    Nodes in the same partition are likely to be implemented
    in the same combo kernel, but subject to subsequent restrictions like CUDA limits for number of args.

    Input arguments:
        nodes: a list of fused scheduler nodes to partition.
        triton_scheduling: TritonScheduling instance.
        node_info_map: a map from node to NodeInfo NamedTuple
    Output:
        a list of list of nodes with each sublist representing a partition.

    The default algorithm is to partition nodes based on the following rules:
        1) nodes with the same number of block dimensions are grouped together.
        2) large pointwise nodes (numels greater than LARGE_NUMELS) are separated from other nodes.
        3) large reduce nodes are separated from other nodes.
    """

    if len(nodes) < 1:
        raise AssertionError(f"expected at least 1 node, got {len(nodes)}")

    # first partition nodes based on number of block dimensions
    tilings = [node_info_map[n].tiling for n in nodes]

    max_dims = max(len(t) for t in tilings)
    nodes_per_ndim: list[list[BaseSchedulerNode]] = []
    for i in range(2, max_dims + 1):
        group_per_dim = [n for n, t in zip(nodes, tilings) if len(t) == i]
        reduction = [
            n for n in group_per_dim if node_info_map[n].features.is_reduction()
        ]
        not_reduction = [n for n in group_per_dim if n not in reduction]
        # rnumel > 2048 usually has long execution time
        # BaseSchedulerNode.group[-1][-1] is rnumel for reduction nodes
        # Scheduling heuristic: separate long reductions (rnumel > 2048).
        # Uses optimization_hint with fallback=1 so unbacked defaults to short reduction.
        long_reduction = [
            n
            for n in reduction
            if V.graph.sizevars.optimization_hint(n.group[-1][-1], fallback=1) > 2048  # type: ignore[arg-type]
        ]
        short_reduction = [n for n in reduction if n not in long_reduction]
        very_large_reduction = [
            n
            for n in long_reduction
            if (
                V.graph.sizevars.optimization_hint(node_info_map[n].numel, fallback=1)
                * V.graph.sizevars.optimization_hint(
                    node_info_map[n].rnumel, fallback=1
                )
            )
            > LARGE_NUMELS
        ]
        long_reduction = [n for n in long_reduction if n not in very_large_reduction]
        if long_reduction:
            _log_partition_separation(
                "ComboKernels: %d long reduction nodes are separated",
                long_reduction,
                not_reduction + short_reduction,
                node_info_map,
            )
        if very_large_reduction:
            log.debug(
                "ComboKernels: %d very large reduction nodes are separated",
                len(very_large_reduction),
            )
            nodes_per_ndim.extend([node] for node in very_large_reduction)
        large_pointwise = [
            n
            for n in not_reduction
            if not node_info_map[n].features.is_reduction()
            and len(node_info_map[n].tiling) == 2
            and V.graph.sizevars.optimization_hint(
                node_info_map[n].tiling["x"], fallback=1
            )
            > LARGE_NUMELS
        ]
        if large_pointwise:
            companion_nodes = [n for n in not_reduction if n not in large_pointwise]
            # TODO benchmark the performance when large pointwise nodes combining with others
            # Include the non-large pointwise companions because the diagnostic
            # describes a partition decision for the current candidate group.
            _log_partition_separation(
                "ComboKernels: %d large pointwise nodes are separated",
                large_pointwise,
                companion_nodes,
                node_info_map,
            )
            not_reduction = companion_nodes
            nodes_per_ndim.extend([node] for node in large_pointwise)

        nodes_per_ndim.extend(
            g for g in (not_reduction, short_reduction, long_reduction) if g
        )

    if sum(len(p) for p in nodes_per_ndim) != len(nodes):
        raise AssertionError("partitioned node count must equal input node count")
    return nodes_per_ndim


_custom_combo_kernel_horizontal_partition_algorithm: Callable[
    [
        list[BaseSchedulerNode],
        SIMDScheduling,
        dict[BaseSchedulerNode, NodeInfo],
    ],
    list[list[BaseSchedulerNode]],
] = _default_custom_combo_kernel_horizontal_partition


def set_custom_combo_kernel_horizontal_partition(
    algorithm: Callable[
        [
            list[BaseSchedulerNode],
            SIMDScheduling,
            dict[BaseSchedulerNode, NodeInfo],
        ],
        list[list[BaseSchedulerNode]],
    ],
) -> None:
    """Sets the algorithm used to partition nodes into horizontal partitions. Nodes in different partitions
    are implemented in different combo kernels. Nodes in the same partition are likely to be implemented
    in the same combo kernel, but subject to subsequent restrictions like CUDA limits for number of args.

    The algorithm should take a list of nodes and return a list of list of nodes.

    The default algorithm is to partition nodes based on number of block dimensions.
    """
    global _custom_combo_kernel_horizontal_partition_algorithm
    _custom_combo_kernel_horizontal_partition_algorithm = algorithm


@dataclass
class PartitionState:
    partitions: list[list[BaseSchedulerNode]]
    cur_partition: list[BaseSchedulerNode]
    cur_count: int

    def finalize(self) -> None:
        if self.cur_partition:
            self.partitions.append(self.cur_partition)


@dataclass
class SubKernelSetup:
    uniquify_block_sizes: list[str]
    lhs_names: list[str]


@dataclass
class SubKernelCode:
    setup: IndentedBuffer
    body: IndentedBuffer
    setup_lhs_names: list[str]


@dataclass
class NoinlineSubKernelCall:
    params: list[ArgName]
    call_args: list[str]
    signature_key: tuple[Any, ...]


@dataclass
class SharedBody:
    body: IndentedBuffer
    placeholder_names: list[str]
    args_by_subkernel: list[list[str]]
    setup_lhs_names: list[str]


@dataclass
class ComboLaunchConfig:
    kwargs: dict[str, int]
    num_warps: int
    num_stages: int


class ComboKernel(Kernel):
    """
    A kernel that combines multiple sub-kernels into a single fused kernel.
    """

    @staticmethod
    def _update_partition(
        partition_state: PartitionState,
        node_rw_count: int,
        node_info: BaseSchedulerNode,
    ) -> None:
        if partition_state.cur_count + node_rw_count > config.combo_kernel_max_num_args:
            partition_state.partitions.append(partition_state.cur_partition)
            partition_state.cur_partition = [node_info]
            partition_state.cur_count = node_rw_count
        else:
            partition_state.cur_count += node_rw_count
            partition_state.cur_partition.append(node_info)

    @staticmethod
    def _base_horizontal_partition(
        subkernel_nodes: list[BaseSchedulerNode],
        triton_scheduling: SIMDScheduling,
        node_info_map: dict[BaseSchedulerNode, NodeInfo],
        custom_algorithm: bool,
    ) -> list[list[BaseSchedulerNode]]:
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
        # TODO support combination of kernels with different block dimensions
        if len(subkernel_nodes) < 1:
            raise AssertionError(
                f"expected at least 1 subkernel node, got {len(subkernel_nodes)}"
            )
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
            config.combo_kernel_allow_mixed_sizes == 1 and custom_algorithm
        )

        ndim_to_partition_state: dict[int, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )
        yelem_to_partition_state: dict[int, PartitionState] = defaultdict(
            lambda: PartitionState([], [], 0)
        )
        all_partitions = []

        for node in subkernel_nodes:
            tiled_groups = node_info_map[node].tiling
            node_info = node

            read_writes = node.read_writes
            read_write_count = len(read_writes.reads) + len(read_writes.writes)

            ndim = len(tiled_groups)
            if ndim < 2:
                raise AssertionError(f"Combokernel not support tile {tiled_groups}")

            # Skip 2d reductions (r0_,r1_) and 3D pointwise (x,y,z) from combo
            keys = tiled_groups.keys()
            if ("r0_" in keys and "r1_" in keys) or "z" in keys:
                all_partitions.append([node_info])
                continue

            if not mixed_sizes and ndim == 3:
                y_elem = tiled_groups["y"]
                partition_state = yelem_to_partition_state[y_elem]
                ComboKernel._update_partition(
                    partition_state, read_write_count, node_info
                )
            else:
                if not (mixed_sizes or ndim <= 3):
                    raise AssertionError(f"No mixed sizes: tile {tiled_groups}")
                partition_state = ndim_to_partition_state[ndim]
                ComboKernel._update_partition(
                    partition_state, read_write_count, node_info
                )

        for partition_state in ndim_to_partition_state.values():
            partition_state.finalize()
            all_partitions.extend(partition_state.partitions)
        for partition_state in yelem_to_partition_state.values():
            partition_state.finalize()
            all_partitions.extend(partition_state.partitions)
        return all_partitions

    @staticmethod
    def horizontal_partition(
        nodes: list[BaseSchedulerNode],
        triton_scheduling: SIMDScheduling,
        node_info_map: dict[BaseSchedulerNode, NodeInfo],
        custom_algorithm: bool = False,
    ) -> list[list[BaseSchedulerNode]]:
        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnum)
        for each subkernel node where each sublist forms a ComboKernel. It horizontally partitions nodes into
        sublists in the following way:
            1) call _custom_combo_kernel_horizontal_partition_algorithm() if custom_algorithm is True
            2) then, call _base_horizontal_partition() to partition nodes into sublists, each sublist is
               guaranteed to not exceed CUDA limits for number of args (read/writes) and to have the same
               2D or 1D blocking strategy.
        """
        if custom_algorithm:
            raw_partitions = _custom_combo_kernel_horizontal_partition_algorithm(
                nodes, triton_scheduling, node_info_map
            )
        else:
            raw_partitions = [nodes]

        """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
        all_partitions = []
        for raw_partition in raw_partitions:
            all_partitions.extend(
                ComboKernel._base_horizontal_partition(
                    raw_partition, triton_scheduling, node_info_map, custom_algorithm
                )
            )
        return all_partitions

    class SequentialDispatch:
        """
        The dispatcher which dispatches the subkernels in a sequential manner:
        the blocks are first dispatched to the 1st subkernel (until it is filled),
        then to the 2nd subkernel, and so on.
        The class defines the methods specific to the dispatch algorithm.
        Methods:
            codegen_pid_range(...): codegen the pid range for each subkernel.
            grid(...): codegen the grid size for launching the combo kernel.
        """

        grid_expr = SequentialComboKernelGrid

        @classmethod
        def codegen_pid_range(
            cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
        ) -> None:
            if num == 0:
                cls._calculate_xblocks(kernel, code)
                code.splice(f"if pid < num_xblocks_{num}:")
                with code.indent():
                    code.splice("pid_offset = pid")
            else:
                code.splice(f"elif pid < num_xblocks_{num}:")
                with code.indent():
                    code.splice(f"pid_offset = pid - num_xblocks_{num - 1}")

        @classmethod
        def _calculate_xblocks(
            cls, kernel: "ComboKernel", code: IndentedBuffer
        ) -> None:
            x_numels_list = kernel.x_numels_list
            for i in range(len(x_numels_list)):
                xnumels, no_x_dim = (
                    (x_numels_list[i], False)
                    if isinstance(x_numels_list[i], str)
                    and cast(str, x_numels_list[i])[0] != "-"
                    or (
                        isinstance(x_numels_list[i], int)
                        and cast(int, x_numels_list[i]) > 0
                    )
                    else (kernel.min_x_blocks_list[i], True)
                )
                xblock_str = (
                    f"tl.cdiv({xnumels}, XBLOCK)" if not no_x_dim else f"{xnumels}"
                )
                if i == 0:
                    code.splice(f"num_xblocks_{i} = {xblock_str}")
                else:
                    code.splice(f"num_xblocks_{i} = num_xblocks_{i - 1} + {xblock_str}")

    class SequentialFlattenGridDispatch:
        """
        Flattened grid dispatch for per-subkernel blocks.
        Uses flattened grid (sum of x*y blocks, 1, 1) and computes
        x_pid_offset, y_pid_offset from the flattened pid.
        """

        grid_expr = SequentialFlattenComboKernelGrid

        @classmethod
        def codegen_pid_range(
            cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
        ) -> None:
            if num == 0:
                cls._calculate_total_blocks(kernel, code)
                code.splice(f"if pid < num_blocks_{num}:")
            else:
                code.splice(f"elif pid < num_blocks_{num}:")

            with code.indent():
                # Compute local pid within this subkernel's block range
                if num == 0:
                    code.splice("local_pid = pid")
                else:
                    code.splice(f"local_pid = pid - num_blocks_{num - 1}")

                # Compute x/y indices from flattened local_pid
                if kernel.y_tree_list[num]:
                    code.splice(f"x_pid_offset = local_pid % x_blocks_{num}")
                    code.splice(f"y_pid_offset = local_pid // x_blocks_{num}")
                else:
                    code.splice("x_pid_offset = local_pid")

        @classmethod
        def _calculate_total_blocks(
            cls, kernel: "ComboKernel", code: IndentedBuffer
        ) -> None:
            """
            Calculate total blocks for each subkernel (x_blocks * y_blocks)
            and cumulative block counts for dispatch boundaries.
            """
            for i, sub_kernel in enumerate(kernel.sub_kernels):
                no_x_dim = sub_kernel.no_x_dim
                xnumel = (
                    kernel.min_x_blocks_list[i] if no_x_dim else kernel.x_numels_list[i]
                )
                x_blocks_str = (
                    f"tl.cdiv({xnumel}, XBLOCK_{i})" if not no_x_dim else f"{xnumel}"
                )
                code.splice(f"x_blocks_{i} = {x_blocks_str}")

                if kernel.y_tree_list[i]:
                    numel = V.graph.sizevars.simplify(kernel.y_tree_list[i].numel)
                    ynumel = (
                        int(numel)
                        if isinstance(numel, (Integer, int))
                        else f"ynumel_{i}"
                    )
                    code.splice(f"y_blocks_{i} = tl.cdiv({ynumel}, YBLOCK_{i})")

                blocks_expr = (
                    f"x_blocks_{i} * y_blocks_{i}"
                    if kernel.y_tree_list[i]
                    else f"x_blocks_{i}"
                )
                code.splice(
                    f"num_blocks_{i} = {blocks_expr}"
                    if i == 0
                    else f"num_blocks_{i} = num_blocks_{i - 1} + {blocks_expr}"
                )

    class RoundRobinDispatch:
        """
        The dispatcher which dispatches the subkernels in a round robin manner:
        the blocks are interleavedly dispatched to each subkernel to execute them
        in parallel.
        The class defines the methods specific to the dispatch algorithm.
        Methods:
            codegen_pid_range(...): codegen the pid range for each subkernel.
            grid(...): codegen the grid size for launching the combo kernel.
        """

        grid_expr = RoundRobinComboKernelGrid

        @classmethod
        def codegen_pid_range(
            cls, kernel: "ComboKernel", num: int, code: IndentedBuffer
        ) -> None:
            num_kernels = len(kernel.sub_kernels)
            if num == 0:
                cond = "if"
            else:
                cond = "elif"
            code.splice(f"{cond} pid % {num_kernels} == {num}:")
            with code.indent():
                code.splice(f"pid_offset = pid // {num_kernels}")

    def __init__(
        self,
        triton_kernel_cls: type[TritonKernel],
        enable_autotune: bool = False,
        mixed_sizes: bool = False,
        per_subkernel_blocks: bool = False,
    ) -> None:
        super().__init__()
        self.triton_kernel_cls = triton_kernel_cls
        self.sub_kernels: list[TritonKernel] = []
        self.iter_vars_count = itertools.count()
        self.grids: list[list[int]] = []
        self.min_x_blocks_list: list[int | str] = []
        self.x_numels_list: list[int | str] = []
        self.y_tree_list: list = []
        self.enable_autotune = enable_autotune
        self.mixed_sizes = mixed_sizes
        self.per_subkernel_blocks = per_subkernel_blocks
        self.dispatch_class: (
            type[
                ComboKernel.SequentialDispatch
                | ComboKernel.SequentialFlattenGridDispatch
                | ComboKernel.RoundRobinDispatch
            ]
            | None
        ) = None
        self.block_args: list[str] = []
        # the following are used when autotuning is disabled
        self.block_size_1d = DEFAULT_COMBO_BLOCK_SIZE_1D
        self.block_size_2d = DEFAULT_COMBO_BLOCK_SIZE_2D
        self.num_warps = 8
        self.block_size_reduce = 256
        self.dynamic_shape_args: list[str] = []
        self.no_bench_stitched_config: triton.Config | None = None
        self.combo_compile_time_autotune = False
        # Compile-time autotune: per-subkernel winning block sizes (XBLOCK_0, ...), passed as args.
        self.stitched_block_config: dict[str, int] | None = None
        # Distinct winner launch configs across the subkernels; seeds the combo's kernel-level autotune.
        self.combo_launch_candidates: list[ComboLaunchConfig] = []
        self.noinline_sub_kernel_calls: list[NoinlineSubKernelCall] = []
        self.noinline_arg_name_maps: list[dict[str, str]] = []

    @property
    def bake_blocks(self) -> bool:
        """Whether autotuning is disabled and default block values are required.

        Legacy combo kernels bake these values into the kernel body. Per-subkernel
        block combos pass them as constexpr arguments through default_config.
        """
        return not self.enable_autotune

    def create_sub_kernel(self, triton_kernel: TritonKernel) -> TritonKernel:
        sub_kernel = triton_kernel
        # pyrefly: ignore [bad-assignment]
        metrics.generated_kernel_count -= 1
        if not self.per_subkernel_blocks:
            sub_kernel.args = self.args
            sub_kernel.iter_vars_count = self.iter_vars_count
            sub_kernel.cse.iter_buffer_ids = self.cse.iter_buffer_ids
        self.sub_kernels.append(sub_kernel)
        return sub_kernel

    @staticmethod
    def create_triton_kernel(
        tiling: dict[str, sympy.Expr],
        features: SIMDKernelFeatures,
        optimize_mask: bool,
        triton_kernel_cls: type[TritonKernel],
        tiling_scores: dict[str, sympy.Expr] | None = None,
        per_subkernel_blocks: bool = False,
    ) -> TritonKernel:
        """
        Only allow optimize_mask=True when 1) sequential dispatch is used,
        2) numels except x dimension are the same for each sub kernel.
        """
        # Flattened dispatch: all dimensions derived from single pid
        if per_subkernel_blocks:
            pid_cache = {
                "tl.program_id(0)": "x_pid_offset",
                "tl.program_id(1)": "y_pid_offset",
            }
        else:
            pid_cache = {"tl.program_id(0)": "pid_offset"}

        kwargs: dict[str, Any] = dict(
            pid_cache=pid_cache,
            optimize_mask=optimize_mask,
            is_combo_kernel=True,
            per_subkernel_blocks=per_subkernel_blocks,
            # foreach kernels don't work with cooperative reductions
            override_cooperative_reduction=False,
            tiling_scores=tiling_scores,
        )
        triton_kernel_cls.apply_feature_required_overrides(features, kwargs)

        return triton_kernel_cls(tiling, features=features, **kwargs)

    def codegen_static_numels_sub_kernel(
        self, code: IndentedBuffer, sub_kernel: TritonKernel, num: int
    ) -> SubKernelSetup:
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        rnumel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
        grid = []
        lhs_names: list[str] = []
        uniquify_block_sizes = []
        for tree in sub_kernel.range_trees:
            simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
            if isinstance(simplified_tree_numel, (Integer, int)):
                lhs_name = f"{tree.prefix}numel"
                code.writeline(f"{lhs_name} = {int(simplified_tree_numel)}")
                lhs_names.append(lhs_name)
            else:
                if f"{tree.prefix}numel_{num}" not in self.dynamic_shape_args:
                    raise AssertionError(
                        f"{tree.prefix}numel_{num} not in dynamic_shape_args"
                    )
                uniquify_block_sizes.append(f"{tree.prefix}numel")

            if not tree.is_reduction:
                if isinstance(simplified_tree_numel, (Integer, int)):
                    grid.append(int(simplified_tree_numel))
                else:
                    # pyrefly: ignore [bad-argument-type]
                    grid.append(f"{tree.prefix}numel_{num}")

            if tree.is_reduction and sub_kernel.persistent_reduction:
                val = TritonKernel._get_persistent_RBLOCK(tree.numel)
                suffix = "" if self.per_subkernel_blocks else f"_{num}"
                lhs_names.append(f"{tree.prefix.upper()}BLOCK{suffix}")
                code.writeline(
                    f"{tree.prefix.upper()}BLOCK{suffix}: tl.constexpr = {val}"
                )

            if tree.prefix == "x" and sub_kernel.no_x_dim:
                suffix = "" if self.per_subkernel_blocks else f"_{num}"
                lhs_names.append(f"XBLOCK{suffix}")
                code.writeline(f"XBLOCK{suffix}: tl.constexpr = 1")
                uniquify_block_sizes.append("XBLOCK")
            elif tree.prefix in ("x", "y") and self.per_subkernel_blocks:
                uniquify_block_sizes.append(f"{tree.prefix.upper()}BLOCK")
            elif tree.is_reduction:
                if self.per_subkernel_blocks or sub_kernel.persistent_reduction:
                    uniquify_block_sizes.append(f"{tree.prefix.upper()}BLOCK")
        self.grids.append(grid)
        return SubKernelSetup(
            uniquify_block_sizes=uniquify_block_sizes,
            lhs_names=lhs_names,
        )

    def min_x_blocks_sub_kernel(self, sub_kernel: TritonKernel, num: int) -> None:
        """
        Kernels with no_x_dim being true has no tunable XBLOCK. They have a fixed number of X blocks.
        Grid calculation needs to make sure that they are assigned with enough number of blocks.
        """
        min_x_blocks: int | str = 0
        x_numels: int | str = 0
        for tree in sub_kernel.range_trees:
            simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
            if tree.prefix == "x":
                if isinstance(simplified_tree_numel, (Integer, int)):
                    x_numels = int(simplified_tree_numel)
                else:
                    x_numels = f"{tree.prefix}numel_{num}"
                if sub_kernel.no_x_dim:
                    min_x_blocks = x_numels
                    x_numels = (
                        # pyrefly: ignore [unsupported-operation]
                        -min_x_blocks
                        if isinstance(x_numels, int)
                        # pyrefly: ignore [redundant-cast]
                        else "-" + cast(str, x_numels)
                    )
                else:
                    if isinstance(simplified_tree_numel, (Integer, int)):
                        x_numels = int(simplified_tree_numel)
                    else:
                        x_numels = f"{tree.prefix}numel_{num}"
        self.min_x_blocks_list.append(min_x_blocks)
        self.x_numels_list.append(x_numels)

    def select_heuristics(self, sub_kernel: TritonKernel) -> tuple[str, dict[str, int]]:
        size_hints = {
            prefix: next_power_of_2(V.graph.sizevars.optimization_hint(numel))
            for prefix, numel in sub_kernel.numels.items()
            if not prefix_is_reduction(prefix) or sub_kernel.inside_reduction
        }
        if sub_kernel.persistent_reduction:
            if not sub_kernel.inside_reduction:
                raise AssertionError(
                    "persistent_reduction sub_kernel must be inside_reduction"
                )
            heuristics = "persistent_reduction"
        elif sub_kernel.inside_reduction:
            heuristics = "reduction"
        else:
            heuristics = "pointwise"
        return heuristics, size_hints

    def select_combo_heuristics(
        self, heuristics_list: list[str], size_hints_list: list[dict[str, int]]
    ) -> tuple[str, dict[str, int], TritonKernel]:
        if not self.enable_autotune and self.no_bench_stitched_config is None:
            return "foreach", size_hints_list[0], self.sub_kernels[0]
        if "reduction" in heuristics_list:
            i, _ = max(
                enumerate(size_hints_list),
                key=lambda x: x[1]["x"] if heuristics_list[x[0]] == "reduction" else 0,
            )
            return heuristics_list[i], size_hints_list[i], self.sub_kernels[i]
        elif "pointwise" in heuristics_list:
            i, _ = max(
                enumerate(size_hints_list),
                key=lambda x: x[1]["x"] if heuristics_list[x[0]] == "pointwise" else 0,
            )
            # modify size_hint to avoid oom check fail (may be a false alarm)
            num_pointwise = len([e for e in heuristics_list if e == "pointwise"])
            num_reduction = len([e for e in heuristics_list if e == "reduction"])
            num_persistent_reduction = len(
                [e for e in heuristics_list if e == "persistent_reduction"]
            )
            if num_reduction != 0:
                raise AssertionError(
                    "combining pointwise and reduction are not supported yet."
                )
            heuristics = (
                "pointwise_with_reduction"
                if num_persistent_reduction > 0
                else "pointwise"
            )
            if len(heuristics_list) - num_pointwise >= 4:
                size_hints = size_hints_list[i]
                size_hints["x"] = min(128, size_hints["x"])
            return heuristics, size_hints_list[i], self.sub_kernels[i]
        else:
            # find persistent_reduction with maximum rnumel
            i, _ = max(
                enumerate(size_hints_list),
                key=lambda x: max(
                    (v for k, v in x[1].items() if prefix_is_reduction(k))
                ),
            )
            return heuristics_list[i], size_hints_list[i], self.sub_kernels[i]

    def get_mutated_args_sub_kernels(self) -> list[str]:
        mutated_args: OrderedSet[str] = OrderedSet()
        for num, sub_kernel in enumerate(self.sub_kernels):
            arg_name_map = (
                self.noinline_arg_name_maps[num] if self.per_subkernel_blocks else {}
            )
            for mutation in sub_kernel.mutations:
                if mutation in sub_kernel.args.input_buffers:
                    name = sub_kernel.args.input_buffers[mutation]
                    mutated_args.add(arg_name_map.get(name, name))
                if (
                    mutation in sub_kernel.args.inplace_buffers
                    and mutation not in V.graph.removed_buffers
                    and mutation not in sub_kernel.removed_buffers
                ):
                    name = cast(
                        InplacedBuffer, sub_kernel.args.inplace_buffers[mutation]
                    ).inner_name
                    mutated_args.add(arg_name_map.get(name, name))
                if mutation in sub_kernel.args.output_buffers:
                    arg = sub_kernel.args.output_buffers[mutation]
                    if isinstance(arg, RemovedArg):
                        raise AssertionError("mutated output buffer arg was removed")
                    mutated_args.add(arg_name_map.get(arg, arg))
        return sorted(mutated_args)

    def select_dispatch_strategy(self) -> None:
        if self.dispatch_class is not None:
            return
        if self.per_subkernel_blocks:
            self.dispatch_class = ComboKernel.SequentialFlattenGridDispatch
            return
        # mixed_sizes is used for optimize_mask, so it only allows sequential dispatch
        # Not mixed sizes on y dim technically is ok to use round robin as wells.
        if not self.mixed_sizes or any(isinstance(e, str) for e in self.x_numels_list):
            # str in x_numels_list means a dynamic shape
            self.dispatch_class = ComboKernel.SequentialDispatch
            return
        # A negative x_blocks_list element means the kernel is not tunable,
        # i.e., no_x_dim = True
        x_numels_list = [abs(cast(int, e)) for e in self.x_numels_list]
        total = max(x_numels_list) * len(x_numels_list)
        needed = sum(x_numels_list)
        if needed / total > BLOCK_UTILIZATION:
            # Introduced overhead (masked blocks) is less than 20%
            self.dispatch_class = ComboKernel.RoundRobinDispatch
        else:
            self.dispatch_class = ComboKernel.SequentialDispatch

    def jit_line(
        self,
        heuristics: str,
        size_hints: dict[str, int],
        selected_kernel: TritonKernel,
        signature: list[Any],
        argdefs: list[ArgName],
        size_hints_list: list[dict[str, int]],
        pointwise_with_reduce: bool = False,
    ) -> str:
        """Write the @triton_heuristics.<heuristics> decorator line for the combo kernel."""

        can_use_32bit = all(k.index_dtype == "tl.int32" for k in self.sub_kernels)
        size_dtype = "tl.int32" if can_use_32bit else "tl.int64"
        for i, sub in enumerate(self.sub_kernels):
            self.min_x_blocks_sub_kernel(sub, i)
        self.select_dispatch_strategy()
        triton_meta: TritonMeta = cast(
            TritonMeta,
            {
                "signature": signature_to_meta(
                    signature, size_dtype=size_dtype, argdefs=argdefs
                ),
                "device": DeviceProperties.create(
                    V.graph.get_current_device_or_throw()
                ),
                "constants": {},
                # Inherit enable_fp_fusion, launch_pdl, disable_ftz so combo kernels
                # compile with the same Triton options as standalone kernels.
                **TritonKernel.triton_meta_common(),
            },
        )

        for arg_num in equal_1_arg_indices(signature):
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]

        triton_meta["configs"] = [
            config_of(signature, skip_cpp_wrapper_input_tensor_alignment=True)
        ]

        mutated_args = self.get_mutated_args_sub_kernels()
        dispatch = self.dispatch_class
        if dispatch is None:
            raise AssertionError("dispatch_class must not be None")

        # Compute the max persistent R0_BLOCK across sub-kernels.
        # This is used by _reduction_configs() to avoid generating configs
        # where XBLOCK * max_persistent_rblock creates pathologically large
        # tiles that cause extreme ROCm compilation times.
        # The max_persistent_rblock mirrors how R0_BLOCK is computed in
        # codegen_static_numels_sub_kernel() for persistent reductions.
        max_persistent_rblock = 0
        if not self.per_subkernel_blocks:
            max_persistent_rblock = max(
                (
                    TritonKernel._get_persistent_RBLOCK(tree.numel)
                    for sub in self.sub_kernels
                    if sub.persistent_reduction
                    for tree in sub.range_trees
                    if tree.is_reduction
                ),
                default=0,
            )

        inductor_meta = {
            "grid_type": dispatch.grid_expr.__name__,
            "combo_grid_meta": self.combo_grid_meta(size_hints_list),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            # Matches triton.py:codegen_kernel(): inference/backward graphs skip
            # CPU-copy of mutated args during autotune retries; training-forward
            # graphs must keep it to preserve benchmark inputs across retries.
            "optimize_mem": V.graph.is_inference or V.graph.is_backward,
            **self.triton_kernel_cls.inductor_meta_common(),
        }
        if max_persistent_rblock > 0:
            inductor_meta["max_persistent_rblock"] = max_persistent_rblock

        # Sum per-sub-kernel bandwidth / FLOP estimates for the combo launch.
        sub_metas = [sub.inductor_meta_per_kernel() for sub in self.sub_kernels]
        self._kernel_num_gb = sum(m.get("kernel_num_gb") or 0 for m in sub_metas)
        if config.benchmark_kernel or config.profile_bandwidth:
            inductor_meta["kernel_num_gb"] = self._kernel_num_gb
        if config.benchmark_kernel:
            inductor_meta["kernel_flop"] = sum(
                m.get("kernel_flop") or 0 for m in sub_metas
            )

        sub_kernel = selected_kernel
        if heuristics == "foreach":
            heuristics_line = f"""
                @triton_heuristics.foreach(
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                )
                @triton.jit
            """
        elif sub_kernel.inside_reduction:
            reduction_hint = sub_kernel.features.get_reduction_hint(
                sub_kernel.tiling_scores
            )
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                tile_hint = "tile_hint=TileHint.SQUARE,"
            else:
                tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """

        self.triton_meta = triton_meta
        self.inductor_meta = inductor_meta

        return heuristics_line

    def codegen_blocks(self, code: IndentedBuffer) -> None:
        has_yblock = any(self.y_tree_list)
        stitched_kwargs = (
            self.no_bench_stitched_config.kwargs
            if self.no_bench_stitched_config is not None
            else None
        )

        for block in self.block_args:
            if stitched_kwargs is not None and block in stitched_kwargs:
                size = stitched_kwargs[block]
            elif "YBLOCK" in block:
                size = self.block_size_2d
            elif "XBLOCK" in block:
                size = self.block_size_2d if has_yblock else self.block_size_1d
            elif "R0_BLOCK" in block:
                size = self.block_size_reduce
            else:
                raise AssertionError(f"{block} is not supported without autotuning")
            code.splice(f"{block}: tl.constexpr = {size}")

    def get_block_args(self) -> list[ConstexprArg]:
        """
        Calculate blocks from sub_kernels and range_trees.
        Update self.block_args, self.y_tree_list
        Return the block args
        """
        block_names = {}
        for i, sub_kernel in enumerate(self.sub_kernels):
            y_tree = None
            for tree in sub_kernel.range_trees:
                if tree.is_reduction and (
                    not sub_kernel.inside_reduction or sub_kernel.persistent_reduction
                ):
                    continue
                if tree.prefix == "x" and sub_kernel.no_x_dim:
                    continue
                if tree.prefix == "y":
                    y_tree = tree
                if self.per_subkernel_blocks:
                    block_names[f"{tree.prefix.upper()}BLOCK_{i}"] = tree.prefix
                else:
                    block_names[f"{tree.prefix.upper()}BLOCK"] = tree.prefix
            self.y_tree_list.append(y_tree)
        self.block_args = list(block_names.keys())

        return [ConstexprArg(x) for x in block_names]

    def add_numel_to_args(
        self, argdefs: list[ArgName], signature: list[Any]
    ) -> list[ArgName]:
        for num, sub_kernel in enumerate(self.sub_kernels):
            for tree in sub_kernel.active_range_trees():
                if not isinstance(tree.numel, (Integer, int)):
                    # only if it is a dynamic shape
                    sizearg = SizeArg(f"{tree.prefix}numel_{num}", tree.numel)
                    signature.append(sizearg)
                    argdefs.append(ArgName(f"{tree.prefix}numel_{num}"))
                    self.dynamic_shape_args.append(f"{tree.prefix}numel_{num}")
        return argdefs

    def add_numel_to_call_args(
        self, name: str, call_args: list[Any], arg_types: list[Any]
    ) -> None:
        for num, sub_kernel in enumerate(self.sub_kernels):
            for tree in sub_kernel.range_trees:
                numel_name = f"{tree.prefix}numel_{num}"
                if numel_name not in self.dynamic_shape_args:
                    continue
                if isinstance(tree.numel, (Integer, Symbol)):
                    expr = tree.numel
                else:
                    expr = V.graph.wrapper_code.generate_numel_expr(
                        name, tree, suffix=str(num)
                    )

                if not tree.is_reduction or sub_kernel.inside_reduction:
                    call_args.append(expr)
                    arg_types.append(type(expr))

    def kernel_benchmark_extra_args(self) -> list[str]:
        extra_args = []
        for num, sub_kernel in enumerate(self.sub_kernels):
            for tree in sub_kernel.range_trees:
                numel_name = f"{tree.prefix}numel_{num}"
                if numel_name not in self.dynamic_shape_args:
                    continue

                if not tree.is_reduction or sub_kernel.inside_reduction:
                    extra_args.append(
                        str(V.graph.sizevars.optimization_hint(tree.numel))
                    )
        return extra_args

    def _merge_noinline_kernel_args(self) -> None:
        for sub_kernel in self.sub_kernels:
            self._merge_noinline_kernel_arg(sub_kernel)

        self.noinline_arg_name_maps = [
            self._resolve_noinline_kernel_arg_names(sub_kernel)
            for sub_kernel in self.sub_kernels
        ]

    def _merge_noinline_kernel_arg(self, sub_kernel: TritonKernel) -> None:
        # Each noinline body codegens in a local namespace, so different members
        # can all have local names like in_ptr0/out_ptr0. First replay every
        # local arg into the combo-main namespace; names are resolved only after
        # all members have registered their in-place relationships.
        local_args = sub_kernel.args
        local_argdefs, _, local_signature, _ = local_args.python_argdefs()

        for argdef, signature in zip(local_argdefs, local_signature, strict=True):
            if isinstance(signature, TensorArg):
                buffer = signature.buffer
                inplaced = local_args.inplace_buffers.get(buffer)
                if (
                    inplaced is not None
                    and not isinstance(inplaced, RemovedArg)
                    and inplaced.inner_name == argdef.name
                ):
                    input_name, *output_names = inplaced.other_names
                    for output_name in output_names:
                        if output_name not in self.args.inplace_buffers:
                            self.args.make_inplace(input_name, output_name)
                elif local_args.output_buffers.get(buffer) == argdef.name:
                    self.args.output(buffer)
                else:
                    self.args.input(buffer)
            elif isinstance(signature, SizeArg):
                expr = signature.expr
                if isinstance(expr, Symbol):
                    self.args.size(expr)
                else:
                    self.args.seed_offset(argdef.name, int(expr))

    def _resolve_noinline_kernel_arg_names(
        self, sub_kernel: TritonKernel
    ) -> dict[str, str]:
        local_args = sub_kernel.args
        local_to_main: dict[str, str] = {}
        local_argdefs, _, local_signature, _ = local_args.python_argdefs()

        for argdef, signature in zip(local_argdefs, local_signature, strict=True):
            if isinstance(signature, TensorArg):
                buffer = signature.buffer
                if (
                    buffer in local_args.inplace_buffers
                    or local_args.output_buffers.get(buffer) == argdef.name
                ):
                    main_name = self.args.output(buffer)
                else:
                    main_name = self.args.input(buffer)
            elif isinstance(signature, SizeArg):
                main_name = self.args.sizevars[signature.expr]
            else:
                raise AssertionError(
                    f"unsupported combo noinline argument "
                    f"{argdef.name}: {type(signature).__name__}"
                )

            local_to_main[argdef.name] = main_name

        return local_to_main

    def _noinline_block_args_for_sub_kernel(
        self, sub_kernel: TritonKernel, num: int
    ) -> list[tuple[ArgName, str]]:
        block_args: list[tuple[ArgName, str]] = []
        for tree in sub_kernel.range_trees:
            if tree.prefix == "x" and sub_kernel.no_x_dim:
                continue
            if tree.prefix in ("x", "y") or (
                tree.is_reduction
                and sub_kernel.inside_reduction
                and not sub_kernel.persistent_reduction
            ):
                name = f"{tree.prefix.upper()}BLOCK"
                block_args.append((ArgName(name, is_constexpr=True), f"{name}_{num}"))
        return block_args

    def _signature_key_part(self, arg: Any) -> tuple[Any, ...]:
        if isinstance(arg, TensorArg):
            return ("tensor", arg.dtype, is_unaligned_buffer(arg))
        if isinstance(arg, SizeArg):
            return ("size", type(arg.expr).__name__)
        if isinstance(arg, ConstexprArg):
            return ("constexpr", arg.name)
        return (type(arg).__name__, getattr(arg, "name", None))

    def _prepare_noinline_sub_kernel_calls(self) -> None:
        calls: list[NoinlineSubKernelCall] = []
        for num, sub_kernel in enumerate(self.sub_kernels):
            local_to_main = self.noinline_arg_name_maps[num]
            local_argdefs, _, local_signature, _ = sub_kernel.args.python_argdefs()
            params = list(local_argdefs)
            call_args = [local_to_main[arg.name] for arg in local_argdefs]
            signature_key: list[Any] = [
                self._signature_key_part(arg) for arg in local_signature
            ]

            for tree in sub_kernel.active_range_trees():
                if isinstance(tree.numel, (Integer, int)):
                    continue
                if tree.is_reduction and not sub_kernel.inside_reduction:
                    continue
                params.append(ArgName(f"{tree.prefix}numel"))
                call_args.append(f"{tree.prefix}numel_{num}")
                signature_key.append(("range_numel", tree.prefix))

            params.append(ArgName("x_pid_offset"))
            call_args.append("x_pid_offset")
            signature_key.append(("dispatch", "x_pid_offset"))
            if any(tree.prefix == "y" for tree in sub_kernel.range_trees):
                params.append(ArgName("y_pid_offset"))
                call_args.append("y_pid_offset")
                signature_key.append(("dispatch", "y_pid_offset"))

            for param, call_arg in self._noinline_block_args_for_sub_kernel(
                sub_kernel, num
            ):
                params.append(param)
                call_args.append(call_arg)
                signature_key.append(("block", param.full_name()))

            calls.append(
                NoinlineSubKernelCall(
                    params=params,
                    call_args=call_args,
                    signature_key=tuple(signature_key),
                )
            )

        self.noinline_sub_kernel_calls = calls

    def _can_share_body(
        self,
        heuristics_list: list[str],
    ) -> bool:
        if len(self.sub_kernels) < 2:
            return False
        if self.enable_autotune or self.per_subkernel_blocks:
            return False
        if torch.version.hip is not None:
            # The shared-body form joins live pointer placeholders after a
            # many-way dispatch branch. HIP/Triton currently has pathological
            # compile times for that IR shape on large foreach lists, so keep
            # ROCm on the existing per-branch body emission path.
            return False
        if self.dispatch_class not in (
            ComboKernel.SequentialDispatch,
            ComboKernel.RoundRobinDispatch,
        ):
            return False
        if self.dynamic_shape_args or any(self.y_tree_list):
            return False
        if any(
            sub_kernel.no_x_dim
            or sub_kernel.inside_reduction
            or sub_kernel.persistent_reduction
            for sub_kernel in self.sub_kernels
        ):
            return False
        return all(heuristic == "pointwise" for heuristic in heuristics_list)

    @staticmethod
    def _plain_lines(code: IndentedBuffer) -> list[str] | None:
        lines: list[str] = []
        for line in code._lines:
            if isinstance(line, str):
                lines.append(line)
            elif isinstance(line, DeferredLineBase):
                evaluated = line()
                if evaluated is not None:
                    lines.append(evaluated)
            else:
                return None
        return lines

    @staticmethod
    def _replace_names_in_line(line: str, replacements: dict[str, str]) -> str | None:
        if not replacements:
            return line
        try:
            pieces: list[str] = []
            cursor = 0
            for token in tokenize.generate_tokens(io.StringIO(line).readline):
                if token.type == tokenize.NAME and token.string in replacements:
                    start, end = token.start[1], token.end[1]
                    pieces.append(line[cursor:start])
                    pieces.append(replacements[token.string])
                    cursor = end
            pieces.append(line[cursor:])
            return "".join(pieces)
        except tokenize.TokenError:
            return None

    @classmethod
    def _replace_names(
        cls, lines: list[str], replacements: dict[str, str]
    ) -> list[str] | None:
        replaced: list[str] = []
        for line in lines:
            new_line = cls._replace_names_in_line(line, replacements)
            if new_line is None:
                return None
            replaced.append(new_line)
        return replaced

    @staticmethod
    def _names_in_lines(lines: list[str]) -> OrderedSet[str] | None:
        names: OrderedSet[str] = OrderedSet()
        try:
            for line in lines:
                for token in tokenize.generate_tokens(io.StringIO(line).readline):
                    if token.type == tokenize.NAME:
                        names.add(token.string)
        except tokenize.TokenError:
            return None
        return names

    @staticmethod
    def _range_tree_names(sub_kernel: TritonKernel) -> list[str]:
        names: list[str] = []
        for tree in sub_kernel.range_trees:
            names.append(tree.name)
            names.extend(entry.name for entry in tree.nodes.values())
        return names

    @classmethod
    def _range_tree_name_replacements(
        cls, first: TritonKernel, sub_kernel: TritonKernel
    ) -> dict[str, str] | None:
        if len(first.range_trees) != len(sub_kernel.range_trees):
            return None

        for first_tree, tree in zip(
            first.range_trees, sub_kernel.range_trees, strict=True
        ):
            if (
                first_tree.prefix != tree.prefix
                or first_tree.tensor_dim != tree.tensor_dim
                or first_tree.is_reduction != tree.is_reduction
            ):
                return None

        first_names = cls._range_tree_names(first)
        names = cls._range_tree_names(sub_kernel)
        if len(first_names) != len(names):
            return None
        return {
            name: first_name
            for first_name, name in zip(first_names, names, strict=True)
            if name != first_name
        }

    @classmethod
    def _body_name_replacements(
        cls,
        first: TritonKernel,
        sub_kernel: TritonKernel,
        args: list[str],
    ) -> dict[str, str] | None:
        first_cse_names = {
            canonical: name for name, canonical in first._op_trace_cse_names.items()
        }
        replacements: dict[str, str] = {}
        for name, canonical in sub_kernel._op_trace_cse_names.items():
            first_name = first_cse_names.get(canonical)
            if first_name is None:
                return None
            if name != first_name:
                replacements[name] = first_name

        range_replacements = cls._range_tree_name_replacements(first, sub_kernel)
        if range_replacements is None:
            return None
        replacements.update(range_replacements)

        for i, arg in enumerate(args):
            replacements[arg] = f"foreach_arg{i}"
        return replacements

    @staticmethod
    def _compatible_shared_arg_properties(
        args_by_subkernel: list[list[str]],
        tensor_args: dict[str, TensorArg],
    ) -> bool:
        if not args_by_subkernel:
            return False
        num_args = len(args_by_subkernel[0])
        if num_args == 0:
            return False
        if any(len(args) != num_args for args in args_by_subkernel):
            return False
        if any(arg not in tensor_args for args in args_by_subkernel for arg in args):
            return False

        for arg_index in range(num_args):
            properties = OrderedSet(
                [
                    (
                        tensor_args[args[arg_index]].dtype,
                        is_unaligned_buffer(tensor_args[args[arg_index]]),
                    )
                    for args in args_by_subkernel
                ]
            )
            if len(properties) != 1:
                return False

        return True

    @classmethod
    def _setup_lhs_names(cls, sub_kernel_codes: list[SubKernelCode]) -> list[str]:
        names: list[str] = []
        seen: OrderedSet[str] = OrderedSet()
        for sub_kernel_code in sub_kernel_codes:
            for name in sub_kernel_code.setup_lhs_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def _try_get_shared_body(
        self,
        sub_kernel_codes: list[SubKernelCode],
        signature: list[Any],
        heuristics_list: list[str],
    ) -> SharedBody | None:
        if not self._can_share_body(heuristics_list):
            return None

        tensor_args = {arg.name: arg for arg in signature if isinstance(arg, TensorArg)}
        if not tensor_args:
            return None
        first_sub_kernel = self.sub_kernels[0]
        first_trace = first_sub_kernel.op_trace
        if not first_trace:
            return None
        if any(
            sub_kernel.op_trace != first_trace for sub_kernel in self.sub_kernels[1:]
        ):
            return None

        transformed_bodies: list[list[str]] = []
        args_by_subkernel: list[list[str]] = []

        for sub_kernel, sub_kernel_code in zip(
            self.sub_kernels, sub_kernel_codes, strict=True
        ):
            lines = self._plain_lines(sub_kernel_code.body)
            if lines is None:
                return None
            body_names = self._names_in_lines(lines)
            if body_names is None:
                return None
            if any(
                arg in body_names and arg not in tensor_args
                for arg in sub_kernel.op_trace_buffer_arg_names
            ):
                return None
            args = [
                arg
                for arg in sub_kernel.op_trace_buffer_arg_names
                if arg in tensor_args and arg in body_names
            ]
            # Every live tensor pointer in the emitted body must come from the
            # structured trace so placeholder substitution cannot miss it.
            if any(arg in body_names and arg not in args for arg in tensor_args):
                return None
            args_by_subkernel.append(args)

            replacements = self._body_name_replacements(
                first_sub_kernel, sub_kernel, args
            )
            if replacements is None:
                return None
            transformed = self._replace_names(lines, replacements)
            if transformed is None:
                return None

            transformed_bodies.append(transformed)

        if any(body != transformed_bodies[0] for body in transformed_bodies[1:]):
            return None
        if not self._compatible_shared_arg_properties(args_by_subkernel, tensor_args):
            return None

        setup_lhs_names = self._setup_lhs_names(sub_kernel_codes)

        body = IndentedBuffer()
        body.writelines(transformed_bodies[0])
        return SharedBody(
            body=body,
            placeholder_names=[
                f"foreach_arg{i}" for i in range(len(args_by_subkernel[0]))
            ],
            args_by_subkernel=args_by_subkernel,
            setup_lhs_names=setup_lhs_names,
        )

    def _codegen_sub_kernel_bodies(
        self,
    ) -> list[SubKernelCode]:
        sub_kernel_codes: list[SubKernelCode] = []
        for num, sub_kernel in enumerate(self.sub_kernels):
            setup = IndentedBuffer()
            sub_kernel_setup = self.codegen_static_numels_sub_kernel(
                setup, sub_kernel, num
            )
            sub_kernel.codegen_prologue(sub_kernel.body)
            sub_kernel.codegen_body()
            sub_kernel._filter_pdl(sub_kernel.body)
            uniquify = (
                []
                if self.per_subkernel_blocks
                else sub_kernel_setup.uniquify_block_sizes
            )
            body = self.uniquify_block_sizes(sub_kernel.body, num, uniquify)
            sub_kernel_codes.append(
                SubKernelCode(
                    setup=setup,
                    body=body,
                    setup_lhs_names=sub_kernel_setup.lhs_names,
                )
            )
        return sub_kernel_codes

    def _codegen_branch(
        self,
        code: IndentedBuffer,
        num: int,
        sub_kernel_code: SubKernelCode,
    ) -> None:
        if self.dispatch_class is None:
            raise AssertionError("dispatch_class must not be None")
        self.dispatch_class.codegen_pid_range(self, num, code)
        with code.indent():
            code.splice(sub_kernel_code.setup)
            code.splice(sub_kernel_code.body)

    def _codegen_noinline_sub_kernels(
        self,
        code: IndentedBuffer,
        kernel_name: str,
        sub_kernel_codes: list[SubKernelCode],
    ) -> list[str]:
        """Emit each sub-kernel body as a @triton.jit(noinline=True) device
        function and return the per-branch call lines."""
        call_lines: list[str] = []
        defs = IndentedBuffer()
        emitted: dict[tuple[Any, ...], str] = {}
        for num, sub_kernel_code in enumerate(sub_kernel_codes):
            noinline_call = self.noinline_sub_kernel_calls[num]
            params = noinline_call.params
            call_args = noinline_call.call_args
            key = (
                sub_kernel_code.setup.getvalue(),
                sub_kernel_code.body.getvalue(),
                noinline_call.signature_key,
            )

            sub_name = emitted.get(key)
            if sub_name is None:
                sub_name = f"{kernel_name}_body_{len(emitted)}"
                emitted[key] = sub_name
                defs.writeline("")
                defs.writeline("@triton.jit(noinline=True)")
                defs.writeline(
                    f"def {sub_name}({', '.join(p.full_name() for p in params)}):"
                )
                with defs.indent():
                    defs.splice(sub_kernel_code.setup)
                    defs.splice(sub_kernel_code.body)
            call_lines.append(f"{sub_name}({', '.join(call_args)})")
        defs.writeline("")
        code.splice(defs)
        return call_lines

    def _codegen_shared_branches(
        self,
        code: IndentedBuffer,
        sub_kernel_codes: list[SubKernelCode],
        shared_body: SharedBody,
    ) -> None:
        if self.dispatch_class is None:
            raise AssertionError("dispatch_class must not be None")
        for num, sub_kernel_code in enumerate(sub_kernel_codes):
            self.dispatch_class.codegen_pid_range(self, num, code)
            with code.indent():
                code.splice(sub_kernel_code.setup)
                for placeholder, arg in zip(
                    shared_body.placeholder_names,
                    shared_body.args_by_subkernel[num],
                    strict=True,
                ):
                    code.writeline(f"{placeholder} = {arg}")

        code.splice("else:")
        with code.indent():
            code.splice("pid_offset = 0")
            for name in shared_body.setup_lhs_names:
                code.writeline(f"{name} = 0")
            for placeholder, arg in zip(
                shared_body.placeholder_names,
                shared_body.args_by_subkernel[0],
                strict=True,
            ):
                code.writeline(f"{placeholder} = {arg}")

        code.splice(shared_body.body)

    def codegen_kernel(self, name: str | None = None) -> str:
        """Generate the triton code for a combo kernel that fuses multiple sub-kernels."""
        # TODO: is it correct to use the first sub kernel's heuristics?
        heuristics_list, size_hints_list = [], []
        for subkernel in self.sub_kernels:
            h, s = self.select_heuristics(subkernel)
            heuristics_list.append(h)
            size_hints_list.append(s)
        heuristics, size_hints, selected_kernel = self.select_combo_heuristics(
            heuristics_list, size_hints_list
        )
        pointwise_with_reduction, heuristics = (
            (True, "pointwise")
            if heuristics == "pointwise_with_reduction"
            else (False, heuristics)
        )
        code = IndentedBuffer()

        code.splice(self.triton_kernel_cls.gen_common_triton_imports())
        if config.benchmark_combo_kernel:
            code.splice(self.imports_for_benchmark_kernel())

        seen_helpers: OrderedSet[str] = OrderedSet()
        for sub_kernel in self.sub_kernels:
            for helper in sub_kernel.helper_functions:
                if helper not in seen_helpers:
                    code.writeline("")
                    code.splice(helper)
                    seen_helpers.add(helper)

        if self.per_subkernel_blocks:
            self._merge_noinline_kernel_args()

        argdefs, _, signature, _ = self.args.python_argdefs()
        argdefs = self.add_numel_to_args(argdefs, signature)
        if self.per_subkernel_blocks:
            main_arg_names = OrderedSet(arg.name for arg in argdefs)
            mapped_arg_names = OrderedSet(
                name
                for name_map in self.noinline_arg_name_maps
                for name in name_map.values()
            )
            missing_arg_names = mapped_arg_names - main_arg_names
            if missing_arg_names:
                raise AssertionError(
                    "combo noinline call arguments missing from main signature: "
                    f"{sorted(missing_arg_names)}"
                )
        block_args = self.get_block_args()
        if not self.bake_blocks or self.per_subkernel_blocks:
            argdefs.extend([ArgName(x.name, is_constexpr=True) for x in block_args])
            if triton_version_uses_attrs_dict():
                signature.extend(block_args)
        if self.per_subkernel_blocks:
            self._prepare_noinline_sub_kernel_calls()

        kernel_name = name or str(Placeholder.KERNEL_NAME)

        sub_kernel_codes = self._codegen_sub_kernel_bodies()
        # Sub-functions must be emitted before the main kernel's heuristics
        # decorator line (triton reads the decorated function's source by
        # inspection). PDL intrinsics move with each body and proton scopes
        # wrap the main kernel around the calls, so neither needs the inline
        # form.
        noinline_calls: list[str] | None = None
        if self.per_subkernel_blocks:
            noinline_calls = self._codegen_noinline_sub_kernels(
                code, kernel_name, sub_kernel_codes
            )

        code.splice(
            self.jit_line(
                heuristics,
                size_hints,
                selected_kernel,
                pointwise_with_reduce=pointwise_with_reduction,
                signature=signature,
                argdefs=argdefs,
                size_hints_list=size_hints_list,
            )
        )
        code.writeline(
            f"def {kernel_name}({', '.join(x.full_name() for x in argdefs)}):"
        )

        with code.indent():
            if config.triton.proton_profiling:
                code.writeline(f'pl.enter_scope("{kernel_name}")')
            code.splice("pid = tl.program_id(0)")
            if self.bake_blocks and not self.per_subkernel_blocks:
                self.codegen_blocks(code)

            shared_body = self._try_get_shared_body(
                sub_kernel_codes, signature, heuristics_list
            )
            if shared_body is not None:
                self._codegen_shared_branches(code, sub_kernel_codes, shared_body)
            elif noinline_calls is not None:
                if self.dispatch_class is None:
                    raise AssertionError("dispatch_class must not be None")
                for num, call_line in enumerate(noinline_calls):
                    self.dispatch_class.codegen_pid_range(self, num, code)
                    with code.indent():
                        code.writeline(call_line)
                code.splice("else:")
                with code.indent():
                    code.splice("pass")
            else:
                for num, sub_kernel_code in enumerate(sub_kernel_codes):
                    self._codegen_branch(code, num, sub_kernel_code)

                code.splice("else:")
                with code.indent():
                    code.splice("pass")
            if config.triton.proton_profiling:
                code.writeline(f'pl.exit_scope("{kernel_name}")')

        if config.benchmark_combo_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb=self._kernel_num_gb))

        return code.getvalue()

    def codegen_kernel_benchmark(self, num_gb: float) -> IndentedBuffer:
        """
        Generates Python code for benchmarking this combo kernel.
        - Creates example inputs (random tensors, constants, sizes).
        - Runs the kernel on the current GPU/stream.
        - Prints runtime (ms) and throughput (GB/s) using `num_gb`.
        Args:
            num_gb (float): The number of gigabytes to use for throughput calculation.
        Returns:
            IndentedBuffer: A buffer containing the generated Python benchmark code.
        """
        result = IndentedBuffer()
        _argdefs, call_args, signature, _ = self.args.python_argdefs()
        result.writelines(["", "", "def get_args():"])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for arg_name, arg_sig in zip(call_args, signature):
                var_name = f"arg_{next(name_cnt)}"
                buf = V.graph.try_get_buffer(arg_name)
                if buf:
                    size = V.graph.sizevars.optimization_hints(buf.get_size())
                    stride = V.graph.sizevars.optimization_hints(buf.get_stride())
                    result.writeline(
                        f"{var_name} = rand_strided({size}, {stride}, device='{coor_device_str(buf.get_device())}', dtype={buf.get_dtype()})"
                    )
                elif arg_name in V.graph.constants:
                    # note that random seed is put in V.graph.constants
                    const_tensor = V.graph.constants[arg_name]
                    size = V.graph.sizevars.optimization_hints(const_tensor.size())
                    stride = V.graph.sizevars.optimization_hints(const_tensor.stride())
                    result.writeline(
                        f"{var_name} = rand_strided({size}, {stride}, device='{coor_device_str(const_tensor.device)}', dtype={const_tensor.dtype})"  # type: ignore[arg-type]
                    )
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.optimization_hint(arg_sig.expr)

                    # Force the seed_offset to be 0 so calls to the same kernel
                    # using different seed offset will have the same benchmark harness.
                    # We can dedup kernel definitions in this case.
                    if "seed_offset" in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f"{var_name} = {symval_hint}")
                elif isinstance(arg_sig, WorkspaceArg):
                    device = V.graph.get_current_device_or_throw()
                    count = V.graph.sizevars.optimization_hint(arg_sig.count)
                    # for benchmark harness, we ignore arg_sig.zero_mode and always zero it
                    result.writeline(
                        f"{var_name} = torch.zeros({count}, device='{coor_device_str(device)}', dtype={arg_sig.dtype})"
                    )
                else:
                    raise KeyError(
                        f"Don't find the buffer or const tensor for {arg_name}"
                    )
                var_names.append(var_name)
            if self.dynamic_shape_args:
                var_names.extend(self.kernel_benchmark_extra_args())
            result.writeline(f"return {', '.join(var_names)},")

        result.writelines(["\n", "\n", "def call(args):"])
        device = V.graph.get_current_device_or_throw()
        coor_preamble, index = coor_benchmark_device_idx(device.index)
        with result.indent():
            if coor_preamble:
                result.writeline(coor_preamble)
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                stream_name = get_raw_stream_name(device.index)
                result.writeline(f"{stream_name} = get_raw_stream({index})")
                result.writeline(
                    f"{str(Placeholder.KERNEL_NAME)}.run(*args, stream={stream_name})"
                )

        # benchmark all configs
        result.writelines(["\n", "\n", "def benchmark_all_configs(args):"])
        with result.indent():
            if coor_preamble:
                result.writeline(coor_preamble)
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                result.writeline(
                    f"return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args)"
                )

        result.writelines(["\n", "\n", "if __name__ == '__main__':"])
        with result.indent():
            result.writeline(
                "from torch._inductor.runtime.benchmarking import benchmarker"
            )
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                f"ms = benchmarker.benchmark(call, fn_args=(args,), device='{device.type}',rep=40)"
            )
            result.writeline(f"num_gb = {num_gb}")
            result.writeline("gb_per_s = num_gb / (ms / 1e3)")
            result.writeline(
                'print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")'
            )

        return result

    def imports_for_benchmark_kernel(self) -> str:
        # Dedent BEFORE substituting get_raw_stream: a multi-line override would
        # otherwise collapse dedent's common prefix and misindent the imports.
        return textwrap.dedent(
            """
            from torch._dynamo.testing import rand_strided
            {}
            import torch
            """
        ).format(V.graph.device_ops.import_get_raw_stream_as("get_raw_stream"))

    def uniquify_block_sizes(
        self, code: IndentedBuffer, num_kernel: int, uniquify: list[str]
    ) -> IndentedBuffer:
        if not uniquify:
            return code
        modified = IndentedBuffer(initial_indent=code._indent)
        for line in code._lines:
            if isinstance(line, str) and (blocks := [e for e in uniquify if e in line]):
                modified_line = line
                for block in blocks:
                    modified_line = modified_line.replace(
                        block, f"{block}_{num_kernel}"
                    )
                modified.writeline(modified_line)
            elif isinstance(line, DeferredLineBase) and (
                blocks := [e for e in uniquify if e in line.line]
            ):
                modified_line = line.line
                for block in blocks:
                    modified_line = modified_line.replace(
                        block, f"{block}_{num_kernel}"
                    )
                new_line = line._new_line(modified_line)
                modified.writeline(new_line)
            else:
                modified.writeline(line)
        return modified

    def call_kernel(self, name: str) -> None:
        _, call_args, _, arg_types = self.args.python_argdefs()

        wrapper = V.graph.wrapper_code
        if self.dispatch_class is None:
            raise AssertionError("dispatch_class must not be None")
        if self.dynamic_shape_args:
            self.add_numel_to_call_args(name, call_args, arg_types)

        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=True,
            arg_types=arg_types,
            triton_meta=self.triton_meta,
            inductor_meta=self.inductor_meta,
        )

    def combo_grid_meta(self, size_hints_list: list[dict[str, int]]) -> dict[str, Any]:
        """
        Build metadata used by combo-kernel grid/dispatch/autotune helpers.
        """
        dynamic_shape = bool(self.dynamic_shape_args)
        num_kernels = len(self.sub_kernels)
        min_blocks = (
            max(self.min_x_blocks_list) * num_kernels if not dynamic_shape else None
        )

        meta: dict[str, Any] = {
            "num_kernels": num_kernels,
            "min_blocks": min_blocks,
            # Captured at codegen time so runtime sees the same value the
            # source was generated with, regardless of later config changes.
            "autotune_grouping": config.combo_kernel_autotune_grouping,
            "block_arg_names": tuple(self.block_args),
        }

        if self.bake_blocks or self.combo_compile_time_autotune:
            default_config: dict[str, int] = {}
            if self.combo_compile_time_autotune:
                # Compile-time autotune: per-subkernel winning block sizes are passed as args;
                # num_warps / num_stages / backend kwargs are autotuned over the distinct winner
                # launch candidates (flattened to tuples so the meta stays repr-serializable).
                if not self.combo_launch_candidates:
                    raise AssertionError(
                        "compile-time autotune requires at least one launch candidate"
                    )
                if self.stitched_block_config is not None:
                    default_config = dict(self.stitched_block_config)
                meta["stitched_launch_candidates"] = [
                    (c.kwargs, c.num_warps, c.num_stages)
                    for c in self.combo_launch_candidates
                ]
            elif self.no_bench_stitched_config is not None:
                stitched = self.no_bench_stitched_config
                default_config = {
                    k: int(v) for k, v in stitched.kwargs.items() if "BLOCK" in k
                }
                meta["stitched_backend_kwargs"] = {
                    k: v for k, v in stitched.kwargs.items() if "BLOCK" not in k
                }
                meta["stitched_num_warps"] = stitched.num_warps
                meta["stitched_num_stages"] = stitched.num_stages
            elif self.per_subkernel_blocks:
                # Per-subkernel block sizes: XBLOCK_0, XBLOCK_1, etc.
                for num, sub_kernel in enumerate(self.sub_kernels):
                    if sub_kernel.no_x_dim:
                        default_config[f"XBLOCK_{num}"] = 1
                    else:
                        block_size = (
                            self.block_size_2d
                            if any(self.y_tree_list)
                            else self.block_size_1d
                        )
                        default_config[f"XBLOCK_{num}"] = block_size

                    if self.y_tree_list[num]:
                        default_config[f"YBLOCK_{num}"] = self.block_size_2d
            else:
                if "YBLOCK" in self.block_args:
                    default_config = {
                        "XBLOCK": self.block_size_2d,
                        "YBLOCK": self.block_size_2d,
                    }
                else:
                    default_config = {"XBLOCK": self.block_size_1d}
            meta["default_config"] = default_config
        else:
            meta["default_config"] = None

        for num, sub_kernel in enumerate(self.sub_kernels):
            meta[f"no_x_dim_{num}"] = sub_kernel.no_x_dim

            if self.per_subkernel_blocks:
                meta[f"heuristic_{num}"] = (
                    "persistent_reduction"
                    if sub_kernel.persistent_reduction
                    else "reduction"
                    if sub_kernel.inside_reduction
                    else "pointwise"
                )

                meta[f"size_hints_{num}"] = size_hints_list[num]
                meta[f"inductor_meta_{num}"] = sub_kernel.inductor_meta_per_kernel()
                if meta[f"heuristic_{num}"] == "pointwise":
                    if len(size_hints_list[num]) == 2:
                        meta[f"tile_hint_{num}"] = "TileHint.SQUARE"
                    else:
                        meta[f"tile_hint_{num}"] = "TileHint.DEFAULT"
                else:
                    meta[f"reduction_hint_{num}"] = (
                        sub_kernel.features.get_reduction_hint(
                            sub_kernel.tiling_scores
                        ).name
                    )

            for tree in sub_kernel.range_trees:
                if not tree.is_reduction:
                    numel_name = f"{tree.prefix}numel_{num}"
                    if numel_name in self.dynamic_shape_args:
                        meta[numel_name] = None
                    else:
                        meta[numel_name] = int(V.graph.sizevars.simplify(tree.numel))

        return meta
<|endoftext|>from __future__ import annotations

from .common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    # MPS is single-device, so a non-zero index is unexpected. device_idx may be a
    # runtime expr (str) under compile-on-one-rank, but MPS never receives that (it has
    # no current_device_idx_expr override), so guard only the int case.
    def device_guard(self, device_idx: DeviceIdx) -> str:
        if isinstance(device_idx, int) and device_idx != 0:
            raise AssertionError(f"expected device_idx == 0, got {device_idx}")
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: DeviceIdx) -> str:
        if isinstance(device_idx, int) and device_idx != 0:
            raise AssertionError(f"expected device_idx == 0, got {device_idx}")
        return "pass  # MPS set device"

    def kernel_driver(self) -> str:
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
        """

    def cpp_kernel_type(self) -> str:
        return "MTLFunction_t"


register_device_op_overrides("mps", MPSDeviceOpOverrides())
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import logging
import math
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast, TypeVar
from unittest.mock import patch

import torch
import torch.utils
from torch.utils._ordered_set import OrderedSet

from ..._dynamo.utils import counters
from .. import config, ir, lowering as L
from ..kernel.mm_common import mm_args
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import (
    has_free_symbols,
    is_same_mkldnn_tensor,
    is_same_tensor,
    parallel_num_threads,
)
from ..virtualized import ops, V
from .cpp import get_export_declaration
from .cpp_micro_gemm import (
    CppMicroBrgemm,
    CppMicroGemm,
    CppMicroGemmAMX,
    CppMicroGemmFP32Vec,
    create_micro_gemm,
    is_int8_woq_gemm_small_m_dim_corner_case,
    LayoutType,
)
from .cpp_template import CppTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import (
    create_epilogue_with_attr,
    DTYPE_TO_CPP,
    GemmBlocking,
    get_gemm_template_output_and_compute_dtype,
)


log = logging.getLogger(__name__)

GEMM_TEMPLATE_INIT_BLOCKING_BASIC_BLOCK = r"""
    constexpr int64_t num_threads = {{num_threads}};
    constexpr int64_t N = {{N}};
    constexpr int64_t K = {{K}};
    constexpr int64_t Mr = {{micro_gemm.register_blocking.block_m}};
    constexpr int64_t Nr = {{micro_gemm.register_blocking.block_n}};
    constexpr int64_t Kr = {{micro_gemm.register_blocking.block_k}};
    constexpr int64_t Nr_blocks = (N + Nr - 1) / Nr;
    constexpr int64_t Kr_blocks = (K + Kr - 1) / Kr;
{%- if is_dynamic_M %}
    const int64_t M = {{kernel.size(GemmOut, 0)}};
    const int64_t Mr_blocks = (M + Mr - 1) / Mr;
{%- else %}
    constexpr int64_t M = {{kernel.size(GemmOut, 0)}};
    constexpr int64_t Mr_blocks = (M + Mr - 1) / Mr;
{%- endif %}
"""

GEMM_TEMPLATE_INIT_BLOCKING_EXTENDED = r"""
{%- if is_dynamic_M %}
    {%- if num_threads > 1 %}
    int64_t Mt_blocks, Nt_blocks, Kt_blocks;
    mm_get_thread_blocking(num_threads, {{config.cpp.gemm_max_k_slices}}, M, N, K, Mr, Nr, Kr, Mt_blocks, Nt_blocks, Kt_blocks);
    {%- else %}
    const auto Mt_blocks = Mr_blocks;
    const auto Nt_blocks = Nr_blocks;
    const auto Kt_blocks = Kr_blocks;
    {%- endif %}
    int64_t Mc_blocks, Nc_blocks, Kc_blocks;
    uint32_t L1_cache_size = {{L1_cache_size}};
    uint32_t L2_cache_size = {{L2_cache_size}};
    mm_get_cache_blocking<{{kernel.dtype(X)}}, {{kernel.dtype(W)}}>(
        num_threads,
        M,
        N,
        K,
        Mr,
        Nr,
        Kr,
        Mt_blocks,
        Nt_blocks,
        Kt_blocks,
        Mc_blocks,
        Nc_blocks,
        Kc_blocks,
        L1_cache_size,
        L2_cache_size
    );
    const int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    const int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    const int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    const int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    const int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;
{%- else %}
    constexpr int64_t Mt_blocks = {{template.thread_blocking(num_threads).block_m}};
    constexpr int64_t Nt_blocks = {{template.thread_blocking(num_threads).block_n}};
    constexpr int64_t Kt_blocks = {{template.thread_blocking(num_threads).block_k}};
    constexpr int64_t Mc_blocks = {{template.cache_blocking(num_threads).block_m}};
    constexpr int64_t Nc_blocks = {{template.cache_blocking(num_threads).block_n}};
    constexpr int64_t Kc_blocks = {{template.cache_blocking(num_threads).block_k}};
    constexpr int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    constexpr int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    constexpr int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    constexpr int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;
{%- endif %}
{%- if is_woq_int4 %}
    int64_t group_size = *q_group_size;
{%- endif %}

    // make sure all partitions are assigned
    {{kernel.assert_function}}(
        Mt_blocks * Nt_blocks * Kt_blocks * {{num_threads}} >= Mr_blocks * Nr_blocks * Kr_blocks,
        "Not all partitions are assigned."
    );
"""

GEMM_TEMPLATE_MULTI_THREADS_PARAMS = r"""
const int64_t k_group_id = tid / num_Kt_blocks;
const int64_t k_slice_id = tid % num_Kt_blocks;
const int64_t n_group_id = k_group_id / num_Nt_blocks;
const int64_t n_slice_id = k_group_id % num_Nt_blocks;
const int64_t k_block_start = k_slice_id * Kt_blocks;
const int64_t k_block_end = std::min(k_block_start + Kt_blocks, Kr_blocks);
const int64_t n_block_start = n_slice_id * Nt_blocks;
const int64_t n_block_end = std::min(n_block_start + Nt_blocks, Nr_blocks);
const int64_t m_block_start = std::min(n_group_id * Mt_blocks, Mr_blocks);
const int64_t m_block_end = std::min(m_block_start + Mt_blocks, Mr_blocks);
const int64_t num_Mc_blocks_per_thread = (m_block_end - m_block_start + Mc_blocks - 1) / Mc_blocks;
"""

GEMM_TEMPLATE_SINGLE_THREAD_PARAMS = r"""
constexpr int tid = 0;
constexpr int64_t k_group_id = 0;
constexpr int64_t k_slice_id = 0;
constexpr int64_t n_group_id = 0;
constexpr int64_t n_slice_id = 0;
constexpr int64_t m_block_start = 0;
constexpr int64_t n_block_start = 0;
constexpr int64_t n_block_end = Nr_blocks;
constexpr int64_t k_block_start = 0;
constexpr int64_t k_block_end = Kr_blocks;
{%- if is_dynamic_M %}
const int64_t num_Mc_blocks_per_thread = num_Mc_blocks;
const int64_t m_block_end = Mr_blocks;
{%- else %}
constexpr int64_t num_Mc_blocks_per_thread = num_Mc_blocks;
constexpr int64_t m_block_end = Mr_blocks;
{%- endif %}
"""

GEMM_TEMPLATE_M_LOOP_PARAMS = r"""
const int64_t my_mc_block_id = (mc_block_id + n_slice_id) % num_Mc_blocks_per_thread;
const int64_t mc = m_block_start + my_mc_block_id * Mc_blocks;
const int64_t m_start = mc * Mr;
const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
const int64_t m_size = m_end - m_start;
"""

GEMM_TEMPLATE_N_LOOP_PARAMS = r"""
const int64_t n_start = nc * Nr;
const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
const int64_t n_size = n_end - n_start;
// NB: assume we pad N, nc_block_end won't exceed padded N here.
const int64_t nc_block_end = std::min(nc + Nc_blocks, n_block_end);
"""

GEMM_TEMPLATE_MICROKERNEL_DEF = r"""
{{template.header().getvalue()}}

{{micro_gemm.codegen_define(kernel)}}
"""

GEMM_TEMPLATE_STUB_DEF = r"""
{%- if x_scale is not none %}
    {%- set kernel_args = {"X": X, "W": W, "inp": inp, "x_scale": x_scale, "x_zp": x_zp, "w_scale": w_scale, "w_zp": w_zp,} %}
{%- elif is_woq_int4 %}
    {%- set kernel_args = {"X": X, "W": W, "q_group_size": q_group_size, "qscale_and_zeros": qscale_and_zeros} %}
{%- else %}
    {%- set kernel_args = {"X": X, "W": W, "inp": inp} %}
{%- endif %}

extern "C" {{export_declaration}}
{{kernel.def_kernel(inputs=kernel_args, outputs={"Y": Y}, aliases=aliases)}}
"""

GEMM_TEMPLATE = r"""
{{ template.codegen_gemm_stub_def() }}
{
    {{ kernel.maybe_codegen_profile(template.get_kernel_prefix_name()) }}
    {{ template.codegen_blocks(
        num_threads, N, K, micro_gemm, is_dynamic_M, kernel, GemmOut, config, L1_cache_size, L2_cache_size, X, W
    ) }}

{%- if maybe_k_slicing %}
    std::unique_ptr<std::unique_ptr<{{DTYPE_TO_CPP[acc_buf_dtype]}}[]>[]> local_buf_ptrs;
    const int64_t num_Mc_blocks_per_thread_block = (Mt_blocks + Mc_blocks - 1) / Mc_blocks;
    const int64_t num_Nc_blocks_per_thread_block = (Nt_blocks + Nc_blocks - 1) / Nc_blocks;
    if (num_Kt_blocks > 1) {
        local_buf_ptrs.reset(new std::unique_ptr<{{DTYPE_TO_CPP[acc_buf_dtype]}}[]>[
            num_Mt_blocks * num_Nt_blocks *
            num_Mc_blocks_per_thread_block * num_Nc_blocks_per_thread_block *
            num_Kt_blocks
        ]);
    }
{%- endif %}

{%- if num_threads > 1 %}
    {%- set use_dynamic_threads = ((config.cpp.threads < 1) and (num_threads == cpu_count)) or config.cpp.dynamic_threads %}
    {%- if use_dynamic_threads %}
    #pragma omp parallel
    {%- else %}
    #pragma omp parallel num_threads({{num_threads}})
    {%- endif %}
    {
        {{ micro_gemm.codegen_init(kernel) }}
        #pragma omp for schedule(static, 1)
        for (int64_t tid = 0; tid < {{num_threads}}; tid++) {
            {{ template.codegen_multi_threads_params()|indent(12, false) }}
{%- else %}
    {
        {{ template.codegen_single_thread_params(is_dynamic_M)|indent(8, false) }}
        {{ micro_gemm.codegen_init(kernel) }}
{%- endif %}
{%- if use_local_acc %}
    {%- set acc_buf_name = "local_acc_buf" %}
        {{ kernel.define_buffer(acc_buf_name, ["Mc_blocks*Mr", "Nc_blocks*Nr"], acc_buf_dtype) }}
{%- endif %}
        for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
            {{ template.codegen_m_loop_params()|indent(12, false) }}
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                {{ template.codegen_n_loop_params()|indent(16, false) }}
{%- if use_local_acc %}
    {%- set acc = kernel.local_buffers[acc_buf_name] %}
                {{ kernel.reinit_buffer_if_null(acc_buf_name) }}
{%- else %}
    {%- set acc = kernel.slice_nd(GemmOut, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- endif %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
{%- set tile_X = kernel.slice_nd(X, [("m_start", "m_end"), ("k_start", "k_end")]) %}
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
{%- set acc_slice = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")]) %}
{%- if template.should_block_weights and not is_woq_int4 %}
{%- set tile_W_3d = kernel.slice_nd(W, [("nci", "nci + 1"), ("k_start", "k_end"), ()]) %}
{%- set tile_W = kernel.view(tile_W_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
{%- else %}
    {%- if is_woq_int4 %}
        {%- set tile_W = kernel.slice_nd(W, [("nci * Nr", "(nci + 1) * Nr"), ("k_start * Nr / 2", "k_end * Nr / 2")]) %}
        {%- set tile_qparam = kernel.slice_nd(
            qscale_and_zeros, [("k_start // group_size", "k_end // group_size"), ("nci * Nr", "(nci + 1) * Nr"), ()]) %}
    {%- else %}
        {%- set tile_W = kernel.slice_nd(W, [("k_start", "k_end"), ("n_start", "n_start + n_size")]) %}
        {%- set tile_qparam = None %}
    {%- endif %}
{%- endif %}
                        if (kc == k_block_start) {
                            {{ micro_gemm.codegen_call(kernel,
                                                       tile_X,
                                                       tile_W,
                                                       acc_slice,
                                                       accum=False,
                                                       qscale_and_zeros=tile_qparam)|indent(28, false)
                            }}
                        } else {
                            {{ micro_gemm.codegen_call(kernel,
                                                       tile_X,
                                                       tile_W,
                                                       acc_slice,
                                                       accum=True,
                                                       qscale_and_zeros=tile_qparam)|indent(28, false)
                            }}
                        }
                    }
                }
{%- if maybe_k_slicing %}
                if (num_Kt_blocks > 1) {
                    const int64_t n_cache_block_id = (nc - n_block_start) / Nc_blocks;
                    const int64_t mxn_cache_block_id =
                        ((n_group_id * num_Nt_blocks + n_slice_id) *
                         num_Mc_blocks_per_thread_block + my_mc_block_id) *
                            num_Nc_blocks_per_thread_block +
                        n_cache_block_id;
                    local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks + k_slice_id].reset(
                        {{ kernel.release_buffer(acc_buf_name) }});
                } else
{%- endif %}
                {
{%- set tile_Y = kernel.slice_nd(Y_2d, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- set tile_acc = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("0", "n_end - n_start")]) %}
                    {{ kernel.store_output(
                        tile_Y, tile_acc, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                    )|indent(20, false)
                    }}
                }
            }
        }
{%- if num_threads > 1 %}
            }
{%- if maybe_k_slicing %}
        if (num_Kt_blocks > 1) {
            #pragma omp for schedule(static, 1)
            for (int64_t tid = 0; tid < {{num_threads}}; tid++) {
                {{ template.codegen_multi_threads_params()|indent(16, false) }}
                for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
                    // We slice M-dim and each thread in the k-slicing group works on a slice
                    const int64_t m_start_unsliced = mc * Mr;
                    const int64_t m_end_unsliced = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
                    const int64_t m_size_unsliced = m_end_unsliced - m_start_unsliced;
                    const int64_t m_slice_size = (m_size_unsliced + num_Kt_blocks - 1) / num_Kt_blocks;
                    const int64_t m_start = std::min(m_start_unsliced + m_slice_size * k_slice_id, m_end_unsliced);
                    const int64_t m_end = std::min(m_start_unsliced + m_slice_size * (k_slice_id + 1), m_end_unsliced);
                    const int64_t m_size = m_end - m_start;
                    const int64_t m_offset = m_start - m_start_unsliced;
                    const int64_t m_cache_block_id = (mc - m_block_start) / Mc_blocks;
                    for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                        const int64_t n_start = nc * Nr;
                        const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
                        const int64_t n_size = n_end - n_start;
                        const int64_t n_cache_block_id = (nc - n_block_start) / Nc_blocks;
                        const int64_t mxn_cache_block_id =
                            ((n_group_id * num_Nt_blocks + n_slice_id) *
                             num_Mc_blocks_per_thread_block + m_cache_block_id) *
                                num_Nc_blocks_per_thread_block +
                            n_cache_block_id;
                        auto {{acc_buf_name}} = local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks].get();
                        for (int64_t other_slice = 1; other_slice < num_Kt_blocks; other_slice++) {
                            auto other_acc = local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks + other_slice].get();
                            for (int64_t m = m_offset; m < m_offset + m_size; m++) {
                                #pragma omp simd
                                for (int64_t n = 0; n < n_size; n++) {
                                    {{acc_buf_name}}[m*Nc_blocks*Nr + n] += other_acc[m*Nc_blocks*Nr + n];
                                }
                            }
                        }
        {%- set tile_Y_m_slice = kernel.slice_nd(Y_2d, [("m_start", "m_start + m_size"), ("n_start", "n_start + n_size")]) %}
        {%- set tile_acc_full = kernel.slice_nd(acc, [("0", "m_size_unsliced"), ("0", "n_size")]) %}
        {%- set tile_acc_m_slice = kernel.slice_nd(tile_acc_full, [("m_offset", "m_offset + m_size"), ()]) %}
                        {{ kernel.store_output(
                            tile_Y_m_slice, tile_acc_m_slice, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                        )|indent(20, false)
                        }}
                    }
                }
            }
        }
{%- endif %}
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
{%- else %}
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
{%- endif %}
}
"""

SMALL_M_GEMM_TEMPLATE = r"""
{{ template.codegen_gemm_stub_def() }}
{
    {{ kernel.maybe_codegen_profile() }}
    {{ template.codegen_blocks(
        num_threads, N, K, micro_gemm, is_dynamic_M, kernel, GemmOut, config, L1_cache_size, L2_cache_size, X, W
    ) }}
    # pragma omp parallel
    {
        #pragma omp for nowait
        for (int64_t nr_block_id = 0; nr_block_id < Nr_blocks; nr_block_id++) {
            // Handle one output M * Nr block in each thread
            int64_t n_start = nr_block_id * Nr;
            int64_t n_end = (nr_block_id + 1) * Nr;
{%- if use_local_acc %}
    {%- set acc_buf_name = "local_acc_buf" %}
            {{ kernel.define_stack_allocated_buffer(acc_buf_name, ["M", "Nr"], acc_buf_dtype) }}
    {%- set acc = kernel.local_buffers[acc_buf_name] %}
{%- else %}
    {%- set acc = kernel.slice_nd(GemmOut, [(0, "M"), ("n_start", "n_end")]) %}
{%- endif %}
            for (int64_t kr_block_id = 0; kr_block_id < Kr_blocks; kr_block_id++) {
                // this loop is not parallelized
                int64_t k_start = kr_block_id * Kr;
                int64_t k_end = std::min((kr_block_id + 1) * Kr, K);
{%- set tile_X = kernel.slice_nd(X, [(0, "M"), ("k_start", "k_end")]) %}
{%- set tile_W_3d = kernel.slice_nd(W, [("nr_block_id", "nr_block_id + 1"), ("k_start", "k_end"), ()]) %}
{%- set tile_W = kernel.view(tile_W_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
                if C10_UNLIKELY(kr_block_id == 0) {
                    {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=False, prefetch=True)|indent(20, false) }}
                } else if C10_UNLIKELY(k_end == K) {
                    {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=True, prefetch=False)|indent(20, false) }}
                } else {
                    {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc, accum=True, prefetch=True)|indent(20, false) }}
                }
            }
{%- set tile_Y = kernel.slice_nd(Y_2d, [("0", "M"), ("n_start", "n_end")]) %}
{%- set tile_acc = kernel.slice_nd(acc, [("0", "M"), ("0", "n_end - n_start")]) %}
            {{ kernel.store_output(
                tile_Y, tile_acc, GemmOut, epilogue_nodes, offsets=("0", "n_start"), reindexers=reindexers
            )|indent(20, false) }}
        }
    }
}
"""


def _is_int8_gemm(inputs):
    return (
        isinstance(inputs[0], ir.IRNode)
        and inputs[0].get_dtype() in [torch.uint8, torch.int8]
    ) or (
        isinstance(inputs[0], torch.Tensor)
        and inputs[0].dtype in [torch.uint8, torch.int8]
    )


def get_padded_n(n, block_n):
    return (n + block_n - 1) // block_n * block_n


_T = TypeVar("_T", ir.IRNode, torch.Tensor)


def transpose_w(W: _T, trans_w: bool) -> _T:
    """
    Transpose W based on the trans_w flag.
    """
    if isinstance(W, ir.IRNode):
        if trans_w:
            if not isinstance(W, ir.TensorBox):
                # pyrefly: ignore [bad-assignment]
                W = ir.TensorBox(W)
            W = L.permute(W, [1, 0])
    else:
        if trans_w:
            if not isinstance(W, torch.Tensor):
                raise AssertionError(f"expected torch.Tensor, got {type(W)}")
            # pyrefly: ignore [bad-assignment]
            W = W.transpose(0, 1)
    # pyrefly: ignore [bad-return]
    return W


def expand_bias(B: _T | None, X: _T) -> _T | None:
    """
    Expand Bias to the same size of X.
    """
    if B is not None:
        if isinstance(B, ir.IRNode):
            if not isinstance(B, ir.TensorBox):
                # pyrefly: ignore [bad-assignment]
                B = ir.TensorBox(B)
            if not hasattr(X, "get_size"):
                raise AssertionError("expected X to have a get_size attribute")
            # pyrefly: ignore [missing-attribute]
            B = L.expand(B, (X.get_size()[0], B.get_size()[-1]))
        else:
            if not isinstance(B, torch.Tensor):
                raise AssertionError(f"expected B to be torch.Tensor, got {type(B)}")
            if not isinstance(X, torch.Tensor):
                raise AssertionError(f"expected X to be torch.Tensor, got {type(X)}")
            # pyrefly: ignore [bad-assignment]
            B = B.expand(X.shape[0], B.shape[-1])
    return B


def prune_tensors(input_nodes: list[ir.IRNode], new_input_nodes: list[ir.IRNode]):
    """
    Prune unused tensors from `V.graph` since the GEMM Template use new packed weight.
    """

    def share_storage(base_tensor: torch.Tensor, comp_tensor: torch.Tensor):
        return base_tensor.is_mkldnn == comp_tensor.is_mkldnn and (
            is_same_tensor(base_tensor, comp_tensor)
            or is_same_mkldnn_tensor(base_tensor, comp_tensor)
        )

    def get_candidates(input_nodes, new_input_nodes):
        # Only Constant Buffer like weight and bias might be changed in GEMM Template.
        # The Inductor IR Node may changed, but still share the storage. For example:
        # bias in bfloat16 case which only do the expand
        return [
            node
            for node in input_nodes
            if (
                node not in new_input_nodes
                and isinstance(node, (ir.TensorBox, ir.StorageBox))
                and node.get_name() in V.graph.constants
                and not any(
                    (
                        isinstance(new_node, (ir.TensorBox, ir.StorageBox))
                        and new_node.get_name() in V.graph.constants
                        and share_storage(
                            V.graph.constants[node.get_name()],
                            V.graph.constants[new_node.get_name()],
                        )
                    )
                    for new_node in new_input_nodes
                )
            )
        ]

    for candidate_node in get_candidates(input_nodes, new_input_nodes):
        # By using the new packed weight for the GEMM template, we can prune the
        # old weight if it has no other users. This saves memory but makes the FX graph
        # non-retraceable. To support retracing, we can add a repack node to the
        # FX graph. For example:
        # mkldnn._linear_pointwise <- repack_linear_wgt <- packed_wgt_for_template
        candidate_tensor_users = 0
        candidate_tensor = V.graph.constants[candidate_node.get_name()]
        for node in reversed(V.graph.graph.nodes):
            # Case may happen when the candidate tensor is used by more than 1 get_attr node
            # https://github.com/pytorch/pytorch/issues/134998
            if node.op == "get_attr" and hasattr(
                V.graph.module, node.target
            ):  # candidate tensor might already be deleted
                comp_tensor = getattr(V.graph.module, node.target)
                if isinstance(comp_tensor, torch.Tensor) and share_storage(
                    candidate_tensor, comp_tensor
                ):
                    candidate_tensor_users += 1

        for node in reversed(V.graph.graph.nodes):
            # The get_attr node has only 1 user fx node
            # The candidate tensor has been used by only 1 get_attr node
            if (
                node.op == "get_attr"
                and node.target == candidate_node.get_name()
                and len(node.users) == 1
                and candidate_tensor_users == 1
            ):
                del V.graph.constants[node.target]
                delattr(V.graph.module, node.target)
                delattr(V.graph.graph.owning_module, node.target)
                counters["inductor"]["select_algorithm_weight_prune"] += 1


def gen_2d_view_of_epilogue_buf(
    Y: ir.Buffer,
    template_buffer: ir.Buffer,
    epilogue_nodes: list[ir.IRNode],
    reindexers: list[Callable[[list[Any]], list[Any]] | None],
    default_reindexers: list[Callable[[list[Any]], list[Any]] | None],
) -> tuple[
    ir.Buffer | ir.ReinterpretView,
    list[Callable[[list[Any]], list[Any]] | None],
]:
    """
    The dimension and the indexing could be different between the GEMM output, i.e. `template_buffer`, which is
    2D with MxN) and the output from the template after epilogues, i.e. `Y`. In the GEMM template code,
    we are not aware of the dimension and the indexing of the epilogues and always work on 2D tiles according to
    the indexing of the GEMM output.
    In this function, we return a 2D buffer (`Y_2d`) according to GEMM output (reinterpreted from `Y` if needed) and
    build a reindexer that converts the indexing of `Y` into `Y_2d`.
    """
    Y_2d: ir.Buffer | ir.ReinterpretView = Y
    if (
        Y.get_size() == template_buffer.get_size()
        and Y.get_stride() == template_buffer.get_stride()
    ):
        reindexers.extend(default_reindexers)
        Y_2d = Y
    else:

        def get_reindexer(epilogue_node, default_reindexer=None):
            # From template_buffer to epilogue_node_ordered (ordered by stride decreasingly, in dense format), for example:
            #   template_buffer:
            #       size (324, 512), stride (512, 1)
            #   epilogue_node_ordered (ordered by stride decreasingly, in dense format):
            #       size (1, 18, 18, 512), stride (165888, 9216, 512, 1)
            stride_order = list(
                ir.get_stride_order(
                    V.graph.sizevars.guarding_hints_or_throw(epilogue_node.get_stride())
                )
            )
            fill_order = ir.stride_order2fill_order(stride_order)
            reversed_fill_order = list(reversed(fill_order))
            size_with_stride_ordered_decreasingly = [
                epilogue_node.get_size()[i] for i in reversed_fill_order
            ]
            reshape_reindex = ir.View.dynamic_reshape_indexer(
                size_with_stride_ordered_decreasingly,
                template_buffer.get_size(),
            )
            if default_reindexer:
                reshape_reindex = ir.fuse_reindexing(reshape_reindex, default_reindexer)

            # From epilogue_node_ordered (ordered by stride decreasingly, in dense format) to epilogue_node, for example:
            #   epilogue_node_ordered (ordered by stride decreasingly, in dense format):
            #       size (1, 18, 18, 512), stride (165888, 9216, 512, 1)
            #   epilogue_node:
            #       size (1, 18, 18, 512), stride (165888, 1, 9216, 512)
            from_stride_ordered_decreasingly_to_epilogue_node_order = [
                (len(stride_order) - 1) - stride_order[i]
                for i in range(len(stride_order))
            ]
            stride_reindex = ir.same_reorder(
                from_stride_ordered_decreasingly_to_epilogue_node_order
            )

            reindexer = ir.fuse_reindexing(stride_reindex, reshape_reindex)  # type: ignore[var-annotated]
            return reindexer

        if default_reindexers is None:
            default_reindexers = [None] * len(epilogue_nodes)
        new_reindexers = [
            get_reindexer(epilogue_node, default_reindexer)
            for epilogue_node, default_reindexer in zip(
                epilogue_nodes, default_reindexers
            )
        ]
        reindexers.extend(new_reindexers)
        if isinstance(Y, ir.BaseView):
            storage = ir.StorageBox(Y.unwrap_view())
        else:
            if not isinstance(Y, ir.Buffer):
                raise AssertionError(f"expected ir.Buffer, got {type(Y)}")
            storage = ir.StorageBox(Y)
        Y_2d = ir.ReinterpretView(data=storage, layout=template_buffer.get_layout())
    return Y_2d, reindexers


class CppGemmTemplate(CppTemplate):
    """
    GEMM Template for Inductor CPP Backend.
    """

    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
        should_block_weights: bool = True,
        name="packed_gemm",
    ) -> None:
        if layout.dtype not in [
            torch.float,
            torch.bfloat16,
            torch.half,
            torch.uint8,
            torch.int8,
        ]:
            raise AssertionError(f"Unsupported layout dtype: {layout.dtype}")
        super().__init__(
            name,
            input_nodes,
            layout,
            num_threads,
            epilogue_creator=epilogue_creator,
        )
        self.beta = beta
        self.alpha = alpha
        self.has_bias = has_bias
        self.register_blocking = register_blocking
        m, n = layout.size[-2:]
        k = input_nodes[0].get_size()[-1]
        self.m, self.n, self.k = m, n, k
        self.padded_n = get_padded_n(n, self.register_blocking.block_n)
        self.is_dynamic_M = has_free_symbols((m,))
        self.should_block_weights = should_block_weights
        self.thread_blocking = self.make_thread_blocking_cache()
        self.cache_blocking = self.make_cache_blocking_cache()

    def make_thread_blocking_cache(self):
        cache = lru_cache()(self._thread_blocking)

        def thread_blocking(num_threads: int) -> GemmBlocking:
            return cache(num_threads)

        return thread_blocking

    def _thread_blocking(self, num_threads: int) -> GemmBlocking:
        """
        NOTE [Thread blocking in Cpp GEMM]
        We use simple heuristics to decide the thread blocking:
        1. Make sure all threads are occupied as much as possible.
        2. For (m, n) blocks, favor more square-sized thread blocks for better data reuse.
        3. If (m, n) blocks cannot occupy all the threads, we consider k-slicing.
        TODO(jgong5): allow tuning various blocking options
        """

        def get_factors(number):
            factors = []
            for i in range(int(number**0.5), 0, -1):
                if number % i == 0:
                    factors.append(number // i)
                    factors.append(i)
            return factors

        def get_blocking(m_factor, n_factor, k_factor, m_blocks, n_blocks, k_blocks):
            thread_block_k = math.ceil(k_blocks / k_factor)
            thread_block_n = math.ceil(n_blocks / n_factor)
            thread_block_m = math.ceil(m_blocks / m_factor)
            return GemmBlocking(thread_block_m, thread_block_n, thread_block_k)

        if self.is_dynamic_M:
            raise AssertionError("Unable to determine thread blocking for dynamic M.")
        register_blocking = self.register_blocking
        m_blocks = math.ceil(self.m / register_blocking.block_m)
        n_blocks = math.ceil(self.n / register_blocking.block_n)
        k_blocks = math.ceil(self.k / register_blocking.block_k)
        factors = get_factors(num_threads)
        if len(factors) == 0:
            raise AssertionError(f"expected at least one factor, got {factors}")

        if config.cpp.gemm_thread_factors is not None:
            factors = [int(i) for i in config.cpp.gemm_thread_factors.split(",")]
            if len(factors) != 3:
                raise AssertionError(f"expected 3 thread factors, got {len(factors)}")
            if math.prod(factors) != self.num_threads:
                raise AssertionError(
                    f"expected factors product == {self.num_threads}, "
                    f"got {math.prod(factors)}"
                )
            return get_blocking(
                factors[0], factors[1], factors[2], m_blocks, n_blocks, k_blocks
            )

        # we favor square-sized thread blocks for good data reuse
        def get_better_blocking(blocking, best_blocking):
            if best_blocking is None:
                best_blocking = blocking
            else:
                block_m_size = blocking.block_m * register_blocking.block_m
                block_n_size = blocking.block_n * register_blocking.block_n
                best_block_m_size = best_blocking.block_m * register_blocking.block_m
                best_block_n_size = best_blocking.block_n * register_blocking.block_n
                if blocking.block_k > best_blocking.block_k:
                    best_blocking = blocking
                elif (
                    blocking.block_k == best_blocking.block_k
                    and block_m_size + block_n_size
                    < best_block_m_size + best_block_n_size
                ):
                    best_blocking = blocking
            return best_blocking

        best_blocking = None
        # check if we can have a thread-blocking to occupy all threads without k-slicing
        for n_factor in factors:
            m_factor = num_threads // n_factor
            if n_blocks >= n_factor and m_blocks >= m_factor:
                blocking = get_blocking(
                    m_factor, n_factor, 1, m_blocks, n_blocks, k_blocks
                )
                best_blocking = get_better_blocking(blocking, best_blocking)

        if best_blocking is None:
            for k_factor in factors:
                if k_blocks >= k_factor and (
                    config.cpp.gemm_max_k_slices == 0
                    or k_factor <= config.cpp.gemm_max_k_slices
                ):
                    n_factors = get_factors(num_threads // k_factor)
                    for n_factor in n_factors:
                        m_factor = (num_threads // k_factor) // n_factor
                        if n_blocks >= n_factor and m_blocks >= m_factor:
                            blocking = get_blocking(
                                m_factor,
                                n_factor,
                                k_factor,
                                m_blocks,
                                n_blocks,
                                k_blocks,
                            )
                            best_blocking = get_better_blocking(blocking, best_blocking)

        if best_blocking is None:
            for n_factor in factors:
                m_factor = num_threads // n_factor
                if n_blocks >= n_factor or m_blocks >= m_factor:
                    blocking = get_blocking(
                        m_factor, n_factor, 1, m_blocks, n_blocks, k_blocks
                    )
                    best_blocking = get_better_blocking(blocking, best_blocking)

        if best_blocking is None:
            raise AssertionError("expected a valid thread blocking, got None")
        return best_blocking

    def make_cache_blocking_cache(self):
        cache = lru_cache()(self._cache_blocking)

        def cache_blocking(num_threads: int) -> GemmBlocking:
            return cache(num_threads)

        return cache_blocking

    def _cache_blocking(self, num_threads: int) -> GemmBlocking:
        def get_cache_blocking(register_blocking, thread_blocking):
            Mr = register_blocking.block_m
            Nr = register_blocking.block_n
            Kr = register_blocking.block_k

            Mt_blocks = thread_blocking.block_m
            Nt_blocks = thread_blocking.block_n
            Kt_blocks = thread_blocking.block_k

            if config.cpp.gemm_cache_blocking is not None:
                blockings = [int(i) for i in config.cpp.gemm_cache_blocking.split(",")]
                if len(blockings) != 3:
                    raise AssertionError(
                        f"expected 3 cache blockings, got {len(blockings)}"
                    )
                Mc_blocks, Nc_blocks, Kc_blocks = blockings
                return (
                    min(Mc_blocks, Mt_blocks),
                    min(Nc_blocks, Nt_blocks),
                    min(Kc_blocks, Kt_blocks),
                )

            # The ratios below are empirically determined to decide
            # the effective sizes of L1 and L2.
            # TODO: tune the factor here
            L1_limit_factor = 0.8
            L2_limit_factor = 0.5

            L1_cache_size = torch.cpu.get_capabilities().get(
                "l1d_cache_size", 0
            )  # per core cache size in Bytes
            if L1_cache_size <= 0:
                raise AssertionError(
                    f"Expect L1_cache_size > 0 but got {L1_cache_size}"
                )
            L1 = L1_cache_size * L1_limit_factor

            L2_cache_size = torch.cpu.get_capabilities().get(
                "l2_cache_size", 0
            )  # per core cache size in Bytes
            if L2_cache_size <= 0:
                raise AssertionError(
                    f"Expect L2_cache_size > 0 but got {L2_cache_size}"
                )
            L2 = L2_cache_size * L2_limit_factor

            def get_num_byte(dtype):
                return torch.tensor([], dtype=dtype).element_size()

            dtype_A = self.input_nodes[0].get_dtype()
            dtype_B = self.input_nodes[1].get_dtype()
            num_byte_A = get_num_byte(dtype_A)
            num_byte_B = get_num_byte(dtype_B)
            if dtype_A is torch.bfloat16 and dtype_B is torch.int8 and Kr != 1:
                # We will cache dequantized weights (BF16) in L1D for AMX micro-kernel.
                # In this case, the choice of the micro-kernel being used can't be decoupled from
                # the cache blocking.
                # TODO: Decouple the choice of micro-kernel from cache blocking
                num_byte_B *= num_byte_A

            # NOTE [CPP GEMM Cache Blocking Algorithm]
            # Our overall strategy is to
            # 1) Make cache blocks of B L1-reside and reused by multiple rows of A, i.e. Mc.
            #    Here, B is Kc x Nr where Nr is a single register block. We use L1 size to
            #    decide Kc. We want to make Mc large enough to better reuse B.
            # 2) Make cache blocks of A L2-reside, which would limit Mc. We want to reuse A
            #    along N, where we have two sub-strategies (see notes below) to decide Mc and Nc.

            # Step 1: Decide Kc assuming B block is L1-reside.
            size_cache_B = Kr * Kt_blocks * Nr * num_byte_B

            Kc_blocks = Kt_blocks
            if size_cache_B > L1:
                Kc_blocks = math.floor(L1 / (Kr * Nr * num_byte_B))

            if (
                config.cpp.use_small_dequant_buffer
                and dtype_A is torch.bfloat16
                and Mt_blocks == 1
            ):
                if dtype_B is torch.uint8:
                    # A16W4
                    # Make a small dequant_B buffer for woq int4 [q_group_size, Nr]
                    # Since when Mt_blocks == 1, L1-reside B block can't be reused by A.
                    if Kc_blocks * Kr >= self.q_group_size():
                        Kc_blocks = self.q_group_size() // Kr

                elif dtype_B is torch.int8:
                    # A16W8
                    # Make A, B, C buffer in L1
                    A_buf_size_div_K = self.m * num_byte_A
                    B_buf_size_div_K = Nr * num_byte_B
                    # assume acc in float32/int32 and Mc_blocks = Nc_blocks = 1
                    C_buf_size = Mr * Nr * 4
                    K_block_size = (L1 - C_buf_size) // (
                        A_buf_size_div_K + B_buf_size_div_K
                    )
                    if Kc_blocks * Kr >= K_block_size:
                        Kc_blocks = (K_block_size + Kr - 1) // Kr

            # Step 2: Decide Mc assuming A block is L2-reside.
            min_Mc_ratio = 2  # TODO(jgong5): something to tune?
            min_Mc_blocks = math.ceil(min_Mc_ratio * Mr / Nr)
            if min_Mc_blocks < 1:
                raise AssertionError(
                    f"expected min_Mc_blocks >= 1, got {min_Mc_blocks}"
                )
            Kt_bytes = Kt_blocks * Kr * num_byte_A
            if min_Mc_blocks * Mr * Kt_bytes < L2:
                # Strategy 1: A (Mc x Kt) resides in L2 and reused by all Nt
                # when Nc_blocks is kept 1. Mc should be large enough (>= min_Mc_blocks)
                # to reuse B (Kc x Nr) in L1. This makes C (Mc x Nr) small enough to reside
                # in L1.
                Mc_blocks = min(Mt_blocks, math.floor(L2 / (Mr * Kt_bytes)))
                Nc_blocks = 1
            else:
                # Strategy 2: Kt is too large to hold A (Mc x Kt) in L2, we reuse
                # A (Mc x Kc) in L2 by B (Kc x Nc). C (Mc x Nc) resides in L2.
                Mc_blocks = Mt_blocks
                Nc_blocks = min(math.ceil(Mc_blocks * Mr / Nr), Nt_blocks)
                Nc_bytes = Nc_blocks * Nr * 4  # assume C or acc is float32/int32
                Kc_bytes = Kc_blocks * Kr * num_byte_A
                if Mc_blocks * Mr * (Kc_bytes + Nc_bytes) > L2:
                    # The following is the solution for 4*Mc*Nc + Mc*Kc_bytes = L2,
                    # assuming Mc == Nc for good data reuse.
                    M_max = (math.sqrt(Kc_bytes * Kc_bytes + 16 * L2) - Kc_bytes) / 8
                    if M_max < Mc_blocks * Mr:
                        Mc_blocks = math.floor(M_max / Mr)
                        Nc_blocks = min(math.ceil(Mc_blocks * Mr / Nr), Nt_blocks)

            return Mc_blocks, Nc_blocks, Kc_blocks

        if self.is_dynamic_M:
            raise AssertionError("Unable to determine cache blocking for dynamic M.")
        register_blocking = self.register_blocking
        thread_blocking = self.thread_blocking(num_threads)

        return GemmBlocking(*get_cache_blocking(register_blocking, thread_blocking))

    def log_blockings(self):
        log.debug(f"Register blocking: {self.register_blocking}")  # noqa: G004
        if self.is_dynamic_M:
            # thread and cache blockings are determined at runtime for dynamic shapes
            return
        log.debug(
            f"Cache blocking: {self.cache_blocking(self.num_threads)}"  # noqa: G004
        )
        thread_blocking = self.thread_blocking(self.num_threads)
        log.debug(f"Thread blocking: {thread_blocking}")  # noqa: G004

        def get_occupancy():
            m_blocks = math.ceil(self.m / self.register_blocking.block_m)
            n_blocks = math.ceil(self.n / self.register_blocking.block_n)
            k_blocks = math.ceil(self.k / self.register_blocking.block_k)
            m = math.ceil(m_blocks / thread_blocking.block_m)
            n = math.ceil(n_blocks / thread_blocking.block_n)
            k = math.ceil(k_blocks / thread_blocking.block_k)
            return (m, n, k)

        log.debug(
            f"Number of threads: {self.num_threads}, occupancy: {get_occupancy()}"  # noqa: G004
        )

    def maybe_k_slicing(self):
        if self.num_threads == 1:
            return False
        if self.is_dynamic_M:
            # TODO(jgong5): perhaps use size hint to decide?
            return True
        register_blocking = self.register_blocking
        k_blocks = math.ceil(self.k / register_blocking.block_k)
        thread_blocking = self.thread_blocking(self.num_threads)
        return k_blocks > thread_blocking.block_k

    @classmethod
    def add_choices(
        cls,
        choices,
        layout,
        input_nodes,
        beta=1,
        alpha=1,
        has_bias=False,
        trans_w=False,
        input_indices=None,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = None,
        act_mapping: dict[int, ir.IRNode] | None = None,
    ):
        """
        Add choices for the GEMM template.
        """
        # Fast path to save the epilogue calculation when x_scale/x_zp/w_scale are constant
        use_int8_fast_compensation_path = _is_int8_gemm(input_nodes) and all(
            (
                isinstance(input_nodes[idx], ir.TensorBox)
                and isinstance(input_nodes[idx].data.data, ir.ConstantBuffer)
            )
            for idx in [1, 2, 4]
        )

        if input_indices is None:
            input_indices = list(range(len(input_nodes)))

        deduped_input_indices = []
        deduped_name_to_idx: dict[str, int] = {}
        for input_node in input_nodes:
            input_name = input_node.get_name()
            if input_name not in deduped_name_to_idx:
                deduped_name_to_idx[input_name] = len(deduped_name_to_idx)
            deduped_input_indices.append(deduped_name_to_idx[input_name])

        def indices_for_inputs(inputs):
            if len(inputs) == len(input_nodes):
                return input_indices
            if len(inputs) != len(deduped_name_to_idx):
                raise AssertionError(
                    f"expected {len(deduped_name_to_idx)} inputs, got {len(inputs)}"
                )
            return [deduped_input_indices[idx] for idx in input_indices]

        def reorder_and_filter(inputs, layout_or_out):
            current_input_indices = indices_for_inputs(inputs)
            if has_bias:
                if len(current_input_indices) < 3:
                    raise AssertionError(
                        f"expected >= 3 input indices, got {len(current_input_indices)}"
                    )
                # Assume the input order is [inp, x, w] and we reorder it to [x, w, inp]
                inp_idx = current_input_indices[0]
                x_idx = current_input_indices[1]
                w_idx = current_input_indices[2]
                return [
                    inputs[x_idx],
                    inputs[w_idx],
                    inputs[inp_idx],
                    *[inputs[idx] for idx in current_input_indices[3:]],
                ], layout_or_out
            else:
                if len(current_input_indices) < 2:
                    raise AssertionError(
                        f"expected >= 2 input indices, got {len(current_input_indices)}"
                    )
                return [inputs[idx] for idx in current_input_indices], layout_or_out

        new_inputs, new_layout = reorder_and_filter(input_nodes, layout)
        is_mkldnn_wgt = (
            new_inputs[1].get_name() in V.graph.constants
            and V.graph.constants[new_inputs[1].get_name()].is_mkldnn
        )
        if is_mkldnn_wgt:
            # It shouldn't happen as viewing an mkldnn tensor, we can extend the
            # implementation if it does.
            if isinstance(new_inputs[1], ir.BaseView):
                raise AssertionError("expected mkldnn weight to not be a BaseView")
        # Note that the layout of MKLDNN Tensor is with the wrong stride
        view_size = new_inputs[1].layout.size
        view_stride = new_inputs[1].layout.stride
        view_offset = new_inputs[1].layout.offset

        def maybe_to_dense(inputs, layout_or_out):
            new_inputs = list(inputs)
            if isinstance(inputs[1], torch.Tensor):
                W = inputs[1]
                new_inputs[1] = W.to_dense() if W.is_mkldnn else W
            return new_inputs, layout_or_out

        def normalize_shapes(inputs, layout_or_out):
            new_inputs = list(inputs)
            if not is_mkldnn_wgt and isinstance(new_inputs[1], torch.Tensor):
                if has_free_symbols(view_size):
                    # If batch size B is dynamic, we need to set the batch size and possibly stride
                    if has_free_symbols(view_size[1:]):
                        raise AssertionError(
                            "expected no free symbols in view_size[1:]"
                        )
                    view_size[:] = V.graph.sizevars.guarding_hints_or_throw(view_size)
                    view_stride[:] = V.graph.sizevars.guarding_hints_or_throw(
                        view_stride
                    )
                # With the assumptation that W is the storage of unwrap view
                # thus view it back here
                new_inputs[1] = new_inputs[1].as_strided(
                    view_size, view_stride, view_offset
                )

            if not trans_w:
                return new_inputs, layout_or_out
            X = new_inputs[0]
            W = new_inputs[1]
            B = new_inputs[2] if has_bias else None
            W = transpose_w(W, trans_w)
            B = expand_bias(B, X)  # type:ignore[arg-type]
            new_inputs[1] = W
            if B is not None:
                new_inputs[2] = B
            return new_inputs, layout_or_out

        # TODO(jgong5): decide proper number of threads per problem size
        num_threads = parallel_num_threads()
        new_inputs, _ = normalize_shapes(*maybe_to_dense(new_inputs, new_layout))
        m, n, k, *_ = mm_args(
            new_inputs[0],
            new_inputs[1],
            mat2_transposed=cls.is_woq_int4(),
            use_4x2_dim=cls.is_woq_int4(),
        )
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            new_inputs[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            "micro_gemm",
            m,
            n,
            k,
            input_dtype=new_inputs[0].get_dtype(),
            input2_dtype=new_inputs[1].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=alpha,
            num_threads=num_threads,
            use_ref=not cls.is_woq_int4(),
            q_group_size=cls.q_group_size(),
        )
        if micro_gemm is None:
            raise AssertionError("expected micro_gemm to be created, got None")
        pre_block_weights = cls.check_if_block_weight(new_inputs[1], micro_gemm)
        micro_gemm.use_local_vnni_blocking(not pre_block_weights)
        cpp_arg_indices: list[int] | None = None

        def dedupe_runtime_inputs(inputs):
            nonlocal cpp_arg_indices
            if not isinstance(inputs[0], torch.Tensor):
                cpp_arg_indices = []
                seen_names: OrderedSet[str] = OrderedSet()
                for idx, input_node in enumerate(inputs):
                    input_name = input_node.get_name()
                    if input_name not in seen_names:
                        seen_names.add(input_name)
                        cpp_arg_indices.append(idx)
                return inputs
            if cpp_arg_indices is None:
                raise AssertionError("expected cpp_arg_indices to be initialized")
            return [inputs[idx] for idx in cpp_arg_indices]

        def preprocessor(inputs, layout):
            new_inputs, new_layout = normalize_shapes(
                *maybe_to_dense(*reorder_and_filter(inputs, layout))
            )
            new_inputs, new_layout = cls.prep_weight(
                new_inputs,
                new_layout,
                # pyrefly: ignore [bad-argument-type]
                micro_gemm,
                pre_block_weights,
                use_int8_fast_compensation_path,
            )
            return dedupe_runtime_inputs(new_inputs), new_layout

        def postprocessor(output):
            if isinstance(output, ir.TensorBox):
                # prepack the weight as input to the template buffer
                template_buffer = ir.InputsKernel.unwrap_storage_for_input(output)
                if not isinstance(template_buffer, ir.CppTemplateBuffer):
                    raise AssertionError(
                        f"expected CppTemplateBuffer, got {type(template_buffer)}"
                    )
                new_input_nodes, _ = reorder_and_filter(input_nodes, layout)

                W_node = new_input_nodes[1]
                if W_node.get_name() not in V.graph.constants:
                    return output
                W = V.graph.constants[W_node.get_name()]
                new_input_nodes[1] = W
                new_input_nodes, new_layout = normalize_shapes(
                    *maybe_to_dense(new_input_nodes, layout)
                )
                new_input_nodes, _ = cls.prep_weight(
                    new_input_nodes,
                    new_layout,
                    # pyrefly: ignore [bad-argument-type]
                    micro_gemm,
                    pre_block_weights,
                    use_int8_fast_compensation_path,
                    skip_int8_compensation=True,
                )
                W_packed = new_input_nodes[1]
                W_packed_constant = V.graph.add_tensor_constant(W_packed)
                new_input_nodes[1] = W_packed_constant

                # Prune unused tensors
                prune_tensors(input_nodes, new_input_nodes)

                template_buffer.inputs[1] = ir.InputsKernel.unwrap_storage_for_input(
                    W_packed_constant
                )
            return output

        template = DataProcessorTemplateWrapper(
            cls,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            num_threads=num_threads,
            register_blocking=micro_gemm.register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            should_block_weights=pre_block_weights,
            name=micro_gemm.__class__.__name__,
        )
        template.maybe_append_choice(choices)
        return template

    @staticmethod
    def get_padded_size(n, block_n, k, should_block_weight):
        padded_n = get_padded_n(n, block_n)
        # We assume that all GEMM weight tensors should be blocked and padded
        new_size = [padded_n // block_n, k, block_n]
        return new_size, padded_n

    @staticmethod
    def _maybe_remove_storage_offset(node: ir.IRNode):
        if node.get_layout().offset == 0:
            return node
        # node may be contiguous but still have a non-zero storage offset.
        # GEMM_TEMPLATE emits code like:
        #   W.data_ptr[node.offset + ...]
        # but runtime W.data_ptr (after normalize_shapes()) already includes this offset.
        # To avoid double-offsetting, we remove the offset in the node also in the generated code.
        #   W.data_ptr[...]
        return ir.ExternKernel.copy_input(node)

    @classmethod
    def prep_weight(
        cls,
        inputs,
        layout: ir.Layout,
        micro_gemm: CppMicroGemm,
        should_block_weight: bool,
        use_int8_fast_compensation_path: bool = False,
        skip_int8_compensation: bool = False,
    ):
        """
        NOTE Weight prep consists of 2 separate steps:
        1. Blocking the weight tensor into a 3D shape: [n//block_n, k, block_n]
           This is always done if the weight tensor is constant, i.e. for all GEMM and some BMM.
           For BMM, we also block non-contiguous weight tensors, since they would be reshaped anyway.
           This assumes that blocked, contiguous weights will be more efficient for the GEMM kernel,
           and is worth the overhead of reshape and blocking.

           This blocking includes additional padding, when n is not a multiple of block_n.
           This padding allows a more efficient microkernel implementation. For BMM, this is only done
