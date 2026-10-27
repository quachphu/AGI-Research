        self.fetch_attr = submodules.__getitem__  # type: ignore[method-assign]
        self.current_node = None

    def run_node(self, n: torch.fx.Node) -> Any:
        # pyrefly: ignore [bad-assignment]
        self.current_node = n
        return super().run_node(n)

    def run(self, *args, **kwargs):
        with V.set_interpreter_handler(self):
            return super().run(*args, **kwargs)


# We don't need the nn.Module and constant handling in Tracer
class LightTracer(TracerBase):
    def __init__(self):
        super().__init__()
        self.graph = torch.fx.Graph(tracer_cls=self.__class__)  # type: ignore[arg-type]
        self.scope = Scope("", None)
        self.module_stack = {}  # type: ignore[assignment]
        self.node_name_to_scope = {}


class MemoryEntry(NamedTuple):
    index_name: str  # LoopBody.indexing_exprs[index_name]
    buffer_name: str | None
    mode: str | None  # V.ops.store(..., mode=mode)


class MemoryUsageType(Enum):
    # These are 1:1 with the opcode generating the usage
    LOAD = auto()
    LOAD_SEED = auto()
    STORE = auto()
    STORE_REDUCTION = auto()
    INDEX_EXPR = auto()
    CHECK_BOUNDS = auto()
    BUCKETIZE = auto()


class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    indexing_exprs: dict[str, sympy.Expr]
    submodules: dict[str, Any]
    subblocks: dict[str, LoopBodyBlock]
    indirect_vars: list[sympy.Symbol]
    indirect_var_ranges: dict[sympy.Symbol, sympy.Expr]
    root_block: LoopBodyBlock
    memory_usage: dict[MemoryUsageType, list[MemoryEntry]]
    op_counts: collections.Counter[str]

    # defined only temporarily
    indexing_exprs_name: dict[sympy.Expr, str]

    @staticmethod
    def _wrap_int_to_sympy_integer(expr):
        # Static sizes can enter indexing expressions as Python ints.
        if type(expr) is int:
            return sympy.Integer(expr)
        return expr

    def __init__(
        self,
        fn,
        args,
        var_ranges,
        iter_vars,
        reduce_vars,
        allow_same_symbol_in_index=False,
    ):
        super().__init__()

        _flat_sizes = tuple(var_ranges.values())
        self.sizes = (
            _flat_sizes[: len(iter_vars)],
            _flat_sizes[len(iter_vars) :],
        )

        self.iter_vars = iter_vars
        self.reduce_vars = reduce_vars
        self.var_ranges = var_ranges

        if isinstance(fn, LoopBody):
            self._init_with_copy(fn, args, allow_same_symbol_in_index)
        else:
            self._init_with_tracing(fn, args)

        self.indexing = None

    def get_original_num_rdims(self) -> int:
        if not self.has_partial_accumulate:
            raise AssertionError("Expected has_partial_accumulate to be set")
        node = self.root_block.graph.find_nodes(
            op="call_method", target="partial_accumulate"
        )[0]
        meta = node.args[-1]
        return meta["num_reduction_dims"]

    def extract_pw_from_reduction(self):
        self.root_block = self.root_block.extract_pw_from_reduction()
        self.has_partial_accumulate = True
        self.iter_vars = self.iter_vars + self.reduce_vars
        self.reduce_vars = []
        self.sizes = (self.sizes[0] + self.sizes[1], tuple())
        return self

    def _init_with_tracing(self, fn, args):
        """Do an FX trace of an arbitrary callable to construct self"""
        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.submodules = {"get_index": self.get_index}
        self.subblocks = {}
        self.indirect_vars = []
        self.indirect_var_ranges: dict[sympy.Symbol, sympy.Expr] = {}
        self.memory_usage = {t: [] for t in MemoryUsageType}
        self.op_counts = collections.Counter()
        self.root_block = LoopBodyBlock(self, fn, args)  # traces
        self.has_partial_accumulate = bool(
            self.root_block.graph.find_nodes(
                op="call_method", target="partial_accumulate"
            )
        )
        del self.indexing_exprs_name  # not used after _init_with_tracing

    def _init_with_copy(self, other: LoopBody, args, allow_same_symbol_in_index):
        """
        _init_with_tracing() is slow, so this is a fast path in the case
        where we are just reordering/merging/splitting the args of an
        existing LoopBody.
        """
        indexing_exprs = other.indexing_from_args(args, allow_same_symbol_in_index)
        self.indexing_exprs = {
            name: self._wrap_int_to_sympy_integer(
                V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges)
            )
            for name, expr in indexing_exprs.items()
        }
        self.subblocks = {k: v.clone(self) for k, v in other.subblocks.items()}
        self.indirect_vars = other.indirect_vars
        self.indirect_var_ranges = other.indirect_var_ranges
        self.memory_usage = other.memory_usage
        self.op_counts = other.op_counts
        self.root_block = other.root_block.clone(self)
        self.has_partial_accumulate = other.has_partial_accumulate

        submodules = {**other.submodules}
        submodules.pop("get_index")
        self.submodules = {
            "get_index": self.get_index,
            **{k: v.clone(self) for k, v in submodules.items()},  # type: ignore[attr-defined]
        }

    def has_op(self, name: str):
        return self.op_counts.get(name, 0) > 0

    def merge_loops(self) -> LoopBody:
        """
        Merge both iteration and reduction loops and return a new LoopBody.
        """
        old_body = self
        old_sizes = self.sizes
        old_iter_vars, old_reduce_vars = old_body.vars
        old_iter_sizes, old_reduce_sizes = old_sizes

        index_exprs = [*old_body.indexing_exprs.values()]

        iter_sizes, iter_reindex, _ = V.graph.sizevars._simplify_loops(
            old_iter_vars,
            old_iter_sizes,
            index_prevent_reordering(index_exprs, old_iter_vars, old_iter_sizes),
        )

        reduce_sizes, reduce_reindex, _ = V.graph.sizevars._simplify_loops(
            old_reduce_vars,
            old_reduce_sizes,
            index_prevent_reordering(index_exprs, old_reduce_vars, old_reduce_sizes),
        )

        if iter_sizes == old_iter_sizes and reduce_sizes == old_reduce_sizes:
            return old_body

        (
            (
                iter_vars,
                reduce_vars,
            ),
            var_ranges,
        ) = dependencies.index_vars_no_squeeze(iter_sizes, reduce_sizes, prefix="p")
        new_body = LoopBody(
            old_body,
            [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
            var_ranges,
            iter_vars,
            reduce_vars,
            allow_same_symbol_in_index=True,
        )

        return new_body

    def with_indexing_exprs(self, replacements: dict[str, sympy.Expr]) -> LoopBody:
        iter_vars, reduce_vars = self.vars
        new_body = LoopBody(
            self,
            (iter_vars, reduce_vars),
            self.var_ranges,
            iter_vars,
            reduce_vars,
            allow_same_symbol_in_index=True,
        )
        new_body.indexing_exprs.update(replacements)
        return new_body

    def expand_dimension_for_pointwise_node(
        self, dimension: int, new_range: int
    ) -> LoopBody:
        """
        Expand node on `dimension` to `new_range` and rely on index modular to avoid
        out-of-boundary access.
        """

        old_body = self
        old_sizes = self.sizes

        iter_size, reduce_size = old_sizes
        original_range = iter_size[dimension]
        new_iter_size = list(iter_size)
        new_iter_size[dimension] = new_range
        new_sizes = (new_iter_size, reduce_size)

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",  # type: ignore[arg-type]
        )

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            index = [*itertools.chain.from_iterable(indices)]
            if not len(index) == len(iter_size) + len(reduce_size):
                raise AssertionError(
                    f"Expected index length {len(index)} to equal "
                    f"iter_size + reduce_size ({len(iter_size)} + {len(reduce_size)})"
                )
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]

            new_iter_idx = list(iter_idx)
            new_iter_idx[dimension] = Mod(iter_idx[dimension], original_range)

            return old_body(new_iter_idx, reduce_idx)

        loop_body = LoopBody(
            new_body, (iter_vars, reduce_vars), var_ranges, iter_vars, reduce_vars
        )

        # use the original symbol prefix so we can do multiple round of reordering
        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",  # type: ignore[arg-type]
        )
        new_body = LoopBody(
            loop_body, (iter_vars2, reduce_vars2), var_ranges2, iter_vars2, reduce_vars2
        )
        return new_body

    def reindex_iter_loops(self, new_iter_sizes: Sequence[sympy.Expr]) -> LoopBody:
        """
        Reindex iteration loops into a different factorization of the same
        total numel. For example, [1024, 8192] -> [65536, 128].

        The old iteration vars are expressed as functions of the new vars via
        FloorDiv and ModularIndexing on the flat index.
        """
        old_body = self
        old_iter_sizes = self.sizes[0]
        reduce_sizes = self.sizes[1]

        new_sizes = (list(new_iter_sizes), list(reduce_sizes))

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",  # type: ignore[arg-type]
        )

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            index = [*itertools.chain.from_iterable(indices)]
            new_iter_idx = index[: len(new_iter_sizes)]
            reduce_idx = index[len(new_iter_sizes) :]
            flat = flatten_index(new_iter_idx, new_iter_sizes)
            old_iter_idx = decompose_index(flat, old_iter_sizes)
            return old_body(old_iter_idx, list(reduce_idx))

        loop_body = LoopBody(
            new_body, (iter_vars, reduce_vars), var_ranges, iter_vars, reduce_vars
        )

        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",  # type: ignore[arg-type]
        )
        return LoopBody(
            loop_body,
            (iter_vars2, reduce_vars2),
            var_ranges2,
            iter_vars2,
            reduce_vars2,
        )

    def reorder_iter_loops(self, new_order) -> LoopBody:
        """
        Reorder iteration loops and return a new LoopBody.
        """
        from .ir import same_reorder

        old_body = self
        old_sizes = self.sizes
        if not len(old_sizes[0]) == len(new_order):
            raise AssertionError(
                f"Expected old_sizes[0] length {len(old_sizes[0])} to equal "
                f"new_order length {len(new_order)}"
            )
        reorder_fn = same_reorder(new_order)

        iter_size, reduce_size = old_sizes
        new_iter_size = reorder_fn(iter_size)

        new_sizes = (new_iter_size, reduce_size)

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",  # type: ignore[arg-type]
        )

        inverse_order = {b: a for a, b in enumerate(new_order)}
        inverse_order = [inverse_order[i] for i in range(len(new_order))]

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            index = [*itertools.chain.from_iterable(indices)]
            if not len(index) == len(iter_size) + len(reduce_size):
                raise AssertionError(
                    f"Expected index length {len(index)} to equal "
                    f"iter_size + reduce_size ({len(iter_size)} + {len(reduce_size)})"
                )
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]
            iter_idx = [iter_idx[i] for i in inverse_order]
            return old_body(iter_idx, reduce_idx, allow_same_symbol_in_index=True)

        return LoopBody(
            new_body,
            (iter_vars, reduce_vars),
            var_ranges,
            iter_vars,
            reduce_vars,
        )

    @property
    def vars(self):
        if self.iter_vars is None:
            raise AssertionError("iter_vars is None")
        if self.reduce_vars is None:
            raise AssertionError("reduce_vars is None")
        return self.iter_vars, self.reduce_vars

    @cache_on_self
    def get_nodes(self):
        all_graphs = itertools.chain(
            (self.root_block.graph,),
            (block.graph for block in self.subblocks.values()),
        )
        return [node for graph in all_graphs for node in graph.nodes]

    @cache_on_self
    def bounds(self):
        # Doing a local import to avoid dumping all the code here
        from .bounds import BoundVars

        return BoundVars(self)

    def get_read_expr(self, buffer_name):
        # reversed to match old behavior
        for entry in reversed(self.memory_usage[MemoryUsageType.LOAD]):
            if entry.buffer_name == buffer_name:
                return self.indexing_exprs[entry.index_name]
        raise KeyError(buffer_name)

    def get_write_expr(self, buffer_name):
        for entry in itertools.chain(
            self.memory_usage[MemoryUsageType.STORE],
            self.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name == buffer_name:
                return self.indexing_exprs[entry.index_name]
        raise KeyError(buffer_name)

    def get_read_exprs(self):
        return [
            self.indexing_exprs[entry.index_name]
            for entry in self.memory_usage[MemoryUsageType.LOAD]
        ]

    def get_all_read_expr(self, buffer_name):
        # reversed to match old behavior
        out = []
        for entry in reversed(self.memory_usage[MemoryUsageType.LOAD]):
            if entry.buffer_name == buffer_name:
                out.append(self.indexing_exprs[entry.index_name])
        return out

    def get_write_exprs(self):
        return [
            self.indexing_exprs[entry.index_name]
            for entry in itertools.chain(
                self.memory_usage[MemoryUsageType.STORE],
                self.memory_usage[MemoryUsageType.STORE_REDUCTION],
            )
        ]

    def get_all_write_expr(self, buffer_name):
        out = []
        for entry in itertools.chain(
            self.memory_usage[MemoryUsageType.STORE],
            self.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name == buffer_name:
                out.append(self.indexing_exprs[entry.index_name])
        return out

    def debug_str(self):
        lines = [f"var_ranges = {dict(self.var_ranges)}"]
        lines.extend([f"{name} = {val}" for name, val in self.indexing_exprs.items()])
        lines.extend(
            [
                block.debug_str(name)
                for name, block in itertools.chain(
                    [("body", self.root_block)], self.subblocks.items()
                )
            ]
        )
        return "\n".join(lines)

    def is_memory_copy(self) -> bool:
        """
        True if this contains only a single loads and store.
        Note, this could involve a layout change.
        """
        return (
            len(self.memory_usage[MemoryUsageType.LOAD]) == 1
            and len(self.memory_usage[MemoryUsageType.STORE]) == 1
            and len(self.submodules) == 1  # get_index
            and self.root_block.contains_only_ops(("load", "store"))
        )

    __repr__ = debug_str

    def add_index_expr(
        self,
        expr: sympy.Expr,
        mtype: MemoryUsageType,
        buffer_name: str | None = None,
        mode: str | None = None,
    ):
        expr = self._wrap_int_to_sympy_integer(expr)
        name = self.indexing_exprs_name.get(expr)
        if not name:
            name = f"index{len(self.indexing_exprs)}"
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        self.memory_usage[mtype].append(MemoryEntry(name, buffer_name, mode))
        return name

    def add_submodule(self, block, prefix):
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f"{prefix}{len(self.submodules)}"
        self.submodules[name] = block
        return name

    def add_indirect(self, size):
        var = sympy_index_symbol_with_prefix(SymT.INDIRECT, len(self.indirect_vars))
        if var in self.indirect_var_ranges:
            raise AssertionError(f"Indirect var {var} already in indirect_var_ranges")
        self.indirect_vars.append(var)
        self.indirect_var_ranges[var] = size
        return var

    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        if str(old) == str(new):
            return
        if self.indexing is None:
            raise AssertionError("indexing must be set before replace_indirect")
        # pyrefly: ignore [bad-assignment]
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    def get_index(self, name):
        if self.indexing is None:
            raise AssertionError("indexing must be set before get_index")
        return self.indexing[name]

    def indexing_from_args(self, indices, allow_same_symbol_in_index=False):
        index = [*itertools.chain.from_iterable(indices)]
        if not len(index) == len(self.var_ranges):
            raise AssertionError(f"Index length mismatch: {index} vs {self.var_ranges}")
        if not allow_same_symbol_in_index and not all(
            v not in self.var_ranges for v in index
        ):
            raise AssertionError(
                f"Same symbol found in index: {self.var_ranges=}, {indices=}"
            )

        replacements = dict(zip(self.var_ranges.keys(), index))
        return {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }

    def __call__(self, *indices, allow_same_symbol_in_index=False):
        self.indexing = self.indexing_from_args(indices, allow_same_symbol_in_index)
        result = self.root_block()
        self.indexing = None
        return result

    def bind_set_indirect_shim(self, var, size, check, wrap_neg):
        def set_indirect(new_var):
            self.replace_indirect(
                var, V.ops.indirect_indexing(new_var, size, check, wrap_neg)
            )

        set_indirect.clone = functools.partial(  # type: ignore[attr-defined]
            LoopBody.bind_set_indirect_shim,
            var=var,
            size=size,
            check=check,
            wrap_neg=wrap_neg,
        )
        return set_indirect

    def bind_scan_shim(self, combine_fn):
        def shim(dtypes, values):
            return V.ops.scan(dtypes, combine_fn, values)

        shim.clone = functools.partial(LoopBody.bind_scan_shim, combine_fn=combine_fn)  # type: ignore[attr-defined]
        return shim

    def bind_masked_shim(self, name):
        def shim(mask, other):
            return V.ops.masked(mask, self.subblocks[name], other)

        shim.clone = functools.partial(LoopBody.bind_masked_shim, name=name)  # type: ignore[attr-defined]
        return shim


class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, however in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable[..., Any], args: list[Any]):
        self.body = body

        tracer = LightTracer()
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})

        from .index_propagation import IndexPropagation

        handler: Any = CountOps(
            CaptureIndexing(
                # pyrefly: ignore[bad-argument-type]
                proxy_ops,
                body,
                tracer,
            ),
            body.op_counts,
        )
        if config.constant_and_index_propagation:
            handler = IndexPropagation(
                handler, self.body.var_ranges, self.body.indirect_var_ranges
            )

        with V.set_ops_handler(handler):
            # This indirection is just a cute way to get IndexPropagation to
            # unwrap the return value.
            ops.output(fn(*args))
        self.graph = tracer.graph

    def extract_pw_from_reduction(self):
        red = None
        store = None
        for node in self.graph.nodes:
            if node.target == "reduction":
                if red:
                    raise AssertionError("Found multiple reduction nodes")
                red = node
            if node.target == "store_reduction":
                if store:
                    raise AssertionError("Found multiple store_reduction nodes")
                store = node
        if not red:
            raise AssertionError("No reduction node found in graph")
        if not store:
            raise AssertionError("No store_reduction node found in graph")
        reduction_type = red.args[-2]
        red_arg = red.args[-1]
        buf = store.args[1]
        ops = store.args[0]

        extra_meta = {
            "num_reduction_dims": len(self.body.reduce_vars),
        }
        with self.graph.inserting_after(store):
            self.graph.call_method(
                "partial_accumulate", (ops, buf, reduction_type, red_arg, extra_meta)
            )
        self.graph.erase_node(store)
        self.graph.erase_node(red)
        return self

    def __call__(self):
        graph = self.graph
        submodules = self.body.submodules

        return InterpreterShim(graph, submodules).run(V.get_ops_handler())

    def debug_str(self, name="block"):
        code = torch.fx.GraphModule(self.body.submodules, self.graph).code
        return re.sub(
            # strip `; del var0` suffixes to make output prettier
            r";[^\n]*",
            "",
            code.strip().replace("def forward(", f"def {name}("),
        )

    def contains_only_ops(self, allowed_ops) -> bool:
        return all(
            node.target in allowed_ops
            for node in self.graph.find_nodes(op="call_method")
        )

    def clone(self, body: LoopBody):
        """Shallow copy with a new parent LoopBody"""
        copy = LoopBodyBlock.__new__(LoopBodyBlock)
        copy.__dict__.update({**self.__dict__, "body": body})
        return copy


class CountOps(DefaultHandler):
    def __init__(self, inner: OpsHandler[Any], counts: collections.Counter[str]):
        self._inner = inner
        self._counts = counts

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._counts[name] += 1
        return getattr(self._inner, name)(*args, **kwargs)


class CaptureIndexing(WrapperHandler):
    name = "CaptureIndexing"

    def __init__(
        self,
        inner: OpsHandler[Any],
        body: LoopBody,
        tracer: LightTracer,
    ):
        super().__init__(inner)
        self.body = body
        self.tracer = tracer

    def _add_index(self, expr: sympy.Expr, mtype: MemoryUsageType, **kwargs: Any):
        return self.tracer.create_proxy(
            "call_module",
            "get_index",
            (self.body.add_index_expr(expr, mtype, **kwargs),),
            {},
        )

    def _simplify(self, expr: sympy.Expr) -> sympy.Expr:
        return V.graph.sizevars.simplify_with_ranges(expr, self.body.var_ranges)

    def load(self, name: str, index: sympy.Expr):
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.LOAD, buffer_name=name)
        return self._inner.load(name, index)

    def load_seed(self, name: str, index: int):
        if not isinstance(index, int):
            raise AssertionError(f"Expected int, got {type(index)}")
        self.body.add_index_expr(
            sympy.Integer(index), MemoryUsageType.LOAD_SEED, buffer_name=name
        )
        return self._inner.load_seed(name, index)

    def store(self, name, index, value, mode=None):
        index = self._simplify(index)
        index = self._add_index(
            index, MemoryUsageType.STORE, buffer_name=name, mode=mode
        )
        return self._inner.store(name, index, value, mode)

    def store_reduction(self, name, index, value):
        index = self._simplify(index)
        index = self._add_index(
            index, MemoryUsageType.STORE_REDUCTION, buffer_name=name
        )
        return self._inner.store_reduction(name, index, value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        result = self._inner.reduction(dtype, src_dtype, reduction_type, value)
        num_outputs = reduction_num_outputs(reduction_type)
        if num_outputs > 1:
            return tuple(result[i] for i in range(num_outputs))
        return result

    def index_expr(self, index, dtype):
        index = self._simplify(index)
        if isinstance(index, (int, sympy.Integer)):
            return self._inner.constant(int(index), dtype)
        index = self._add_index(index, MemoryUsageType.INDEX_EXPR)
        return self._inner.index_expr(index, dtype)

    def value_expr(self, index, dtype):
        index = self._simplify(index)
        if isinstance(index, (int, sympy.Integer)):
            return self._inner.constant(int(index), dtype)
        index = self._add_index(index, MemoryUsageType.INDEX_EXPR)
        return self._inner.value_expr(index, dtype)

    def check_bounds(self, index, size, lower, upper):
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.CHECK_BOUNDS)
        size = self.body._wrap_int_to_sympy_integer(size)
        size = self._add_index(size, MemoryUsageType.CHECK_BOUNDS)
        return self._inner.check_bounds(index, size, lower, upper)

    def bucketize(
        self,
        values: T,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: T | None = None,
    ) -> T:
        """
        See [Note: Inductor bucketize op]
        """
        boundaries = (
            boundaries[0],
            self._add_index(
                boundaries[1],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
            self._add_index(
                boundaries[2],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
            self._add_index(
                boundaries[3],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
        )
        if sorter is not None:
            sorter = (
                sorter[0],
                self._add_index(
                    sorter[1], MemoryUsageType.BUCKETIZE, buffer_name=sorter[0]
                ),
            )

        return self._inner.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )

    def masked(self, mask_proxy, masked_body: Callable[..., Any], other_proxy):
        """
        Recursively capture the masked out body in another LoopBodyBlock
        """
        name = self.body.add_submodule(None, "masked_subblock")
        self.body.submodules[name] = self.body.bind_masked_shim(name)
        self.body.subblocks[name] = LoopBodyBlock(self.body, masked_body, [])
        return self.tracer.create_proxy(
            "call_module", name, (mask_proxy, other_proxy), {}
        )

    def scan(
        self,
        dtype_proxy,
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        value_proxy,
    ):
        shim = self.body.bind_scan_shim(combine_fn)
        name = self.body.add_submodule(shim, "scan")
        result = self.tracer.create_proxy(
            "call_module",
            name,
            (dtype_proxy, value_proxy),
            {},
        )
        # Proxies are iterable, but some methods expect tuples/lists
        return tuple(result[i] for i in range(len(value_proxy)))

    def sort(self, dtypes, values, stable, descending):
        result = self._inner.sort(dtypes, values, stable, descending)
        # Proxies are iterable, but some methods expect tuples/lists
        return tuple(result[i] for i in range(len(values)))

    def frexp(self, value_proxy):
        result = self._inner.frexp(value_proxy)
        # Proxies are iterable, but some methods expect tuples/lists
        return (result[0], result[1])

    def indirect_indexing(self, index_proxy, size, check=True, wrap_neg=True):
        """
        Flow data from tensors into indexing formulas.
        Introduce a call_module to update the indexing.
        """

        var = self.body.add_indirect(size)
        set_indirect = self.body.bind_set_indirect_shim(var, size, check, wrap_neg)
        self.tracer.create_proxy(
            "call_module",
            self.body.add_submodule(set_indirect, f"set_{var}"),
            (index_proxy,),
            {},
        )
        return var

    def output(self, *result):
        self.tracer.create_proxy("output", "output", result, {})
<|endoftext|>import copy
import dataclasses
import logging
import os
import shutil
import uuid
from pathlib import Path

from torch._dynamo.utils import counters, dynamo_timed, set_feature_use
from torch._utils_internal import justknobs_check
from torch.utils._filelock import FileLock
from torch.utils._ordered_set import OrderedSet

from .runtime.runtime_utils import triton_cache_dir
from .utils import _IS_WINDOWS, GPU_KERNEL_BIN_EXTS


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TritonBundleEntry:
    """
    When we have compiled a triton kernel, we take note of that kernel by
    its triton generated hash, its device, and where this kernel is located.
    This is the minimum information we can use to later retrieve this kernel
    from file system.
    """

    kernel_hash: str
    device: int
    directory: str


@dataclasses.dataclass(frozen=True)
class TritonKernelArtifact:
    """
    Artifact for an individual kernel converted to bytes.
    Bytes could be a cubin, json, ttir, or ttgir.
    """

    filename: str
    payload: bytes = dataclasses.field(repr=False)  # Do not display binary


@dataclasses.dataclass(frozen=True)
class StaticallyLaunchedAutotuner:
    """
    Represents a statically compiled CachingAutotuner object that we can
    save directly in the cache. A CachingAutotuner is made up of a list of
    StaticTritonCompileResults, each of which uses the cubin from a TritonKernelArtifact.

    Statically saved here have their cubin files saved by a corresponding TritonBundleEntry.
    """

    cache_key: str
    kernel_name: str
    kernel: "CachingAutotuner"  # type: ignore[name-defined] # noqa: F821


@dataclasses.dataclass(frozen=True)
class TritonKernelArtifacts:
    """
    Collection of artifacts for a particular kernel.
    """

    kernel_hash: str
    device: int
    artifacts: list[TritonKernelArtifact]


@dataclasses.dataclass(frozen=True)
class TritonBundlerMetadata:
    """
    Metadata used for instrumentation
    """

    cached_kernel_names: list[str]
    statically_launched_kernel_names: list[str]


@dataclasses.dataclass(frozen=True)
class TritonBundle:
    """
    Serializable bundle to save into FXGraphCache
    """

    kernel_artifacts: list[TritonKernelArtifacts]
    static_autotuners: list[StaticallyLaunchedAutotuner]


class TritonBundler:
    """
    Lightweight Triton Kernel bundler that notes each time we compile a triton
    kernel. When collect is called, converts all the previously noted kernels and
    their artifacts into a structured bytes blob, and later when write is called
    it writes this structured blob back to file system.

    Intended Life cycle:
    - TritonBundler.begin_compile is called when we start compiling in Inductor
    - TritonBundler.put is called each time a Triton Kernel is compiled
    - TritonBundler.collect is called when a cache entry is being generated
    - TritonBundler.end_compile is called to indicate bundling is completed,
      collect will execute this function as well.
    - TritonBundler.read_and_emit is called when a cache entry is read
    """

    _entries: list[TritonBundleEntry] | None = None
    _static_autotuners: list[StaticallyLaunchedAutotuner] | None = None
    _winners: OrderedSet[str] | None = None

    # __grp__kernel_name.json contains metadata with source code paths
    # we use this as sentinel value for search and replace
    _REPLACE_BYTES: bytes = b"[REPLACE]"

    @staticmethod
    def is_enabled() -> bool:
        from torch._inductor import config

        if config.force_disable_caches:
            return False

        if (b := config.bundle_triton_into_fx_graph_cache) is not None:
            return b

        if not config.is_fbcode():
            return False

        return justknobs_check(
            "pytorch/remote_cache:bundle_triton_into_fx_graph_cache_v2"
        )

    @classmethod
    def begin_compile(cls) -> None:
        """
        Initializes the TritonBundler.
        The current TritonBundler bundle is finalized by TritonBundler.collect.
        """
        if not TritonBundler.is_enabled():
            return
        log.debug("TritonBundler.begin_compile is called")
        if cls._entries is not None:
            raise AssertionError(
                "TritonBundler.begin_compile called with active entries; "
                "expected cls._entries to be None"
            )
        cls._entries = []
        cls._static_autotuners = []
        cls._winners = OrderedSet()

    @classmethod
    def end_compile(cls) -> None:
        """
        Finalizes the TritonBundler. If collect is not yet called, it
        discards the current bundle.
        """
        log.debug("TritonBundler.end_compile is called")
        cls._entries = None
        cls._static_autotuners = None
        cls._winners = None

    @classmethod
    def put(cls, kernel_hash: str, device: int) -> None:
        """
        Lazily observes that we have seen a Triton kernel compilation. Remembers
        it for when collect is later called.
        """
        if (entries := cls._entries) is not None:
            entries.append(
                TritonBundleEntry(kernel_hash, device, triton_cache_dir(device))
            )

    @classmethod
    def put_winner(cls, kernel_hash: str) -> None:
        """
        Marks a kernel hash as a winning autotuning config. Only winning
        kernels are included in the bundle by collect(). If no winners are
        recorded (e.g. single-config kernels that skip autotuning), all
        entries are bundled.
        """
        if cls._winners is not None:
            cls._winners.add(kernel_hash)

    @classmethod
    def put_static_autotuner(cls, key: str, kernel: "CachingAutotuner") -> None:  # type: ignore[name-defined] # noqa: F821
        from torch._inductor import config

        if not config.use_static_triton_launcher:
            raise AssertionError("expected config.use_static_triton_launcher to be set")
        if (entries := cls._static_autotuners) is not None:
            # Clear a bunch of unpicklable values and make a copy to save
            # for FXGraphCache
            old_values = kernel.prepare_for_pickle()
            new_kernel = copy.deepcopy(kernel)
            new_kernel.prepare_for_caching()
            new_kernel._reload_kernel = None

            entries.append(
                StaticallyLaunchedAutotuner(
                    key,
                    new_kernel.inductor_meta.get("kernel_name", "unknown_kernel"),
                    new_kernel,
                )
            )

            # Put the values back since we need it to use now
            kernel.restore_after_unpickle(old_values)

    @classmethod
    def collect_static_autotuners(
        cls,
    ) -> tuple[list[StaticallyLaunchedAutotuner], list[str]]:
        if not cls._static_autotuners:
            return [], []
        else:
            log.info(
                "Saving %d statically launchable CachingAutotuners",
                len(cls._static_autotuners),
            )
            static_autotuner_names = [i.kernel_name for i in cls._static_autotuners]
            counters["inductor"]["triton_bundler_save_static_autotuner"] += 1
            return cls._static_autotuners, static_autotuner_names

    @classmethod
    def load_autotuners(
        cls, static_autotuners: list[StaticallyLaunchedAutotuner] | None
    ) -> list[str]:
        """
        Load statically launchable CachingAutotuners into async_compile.CompiledTritonKernels
        cache.
        """
        if not static_autotuners:
            return []

        from torch._inductor.async_compile import CompiledTritonKernels
        from torch._inductor.codecache import StaticAutotunerFuture

        log.info("Loading %d statically launchable autotuners", len(static_autotuners))
        kernel_names = []
        with dynamo_timed("TritonBundler.load_cached_static_autotuners"):
            for result in static_autotuners:
                try:
                    # Make sure the cubin path exists and is valid
                    for compile_result in result.kernel.compile_results:
                        compile_result.reload_cubin_path()
                except RuntimeError:
                    log.warning(
                        "Failed to reload cubin file statically launchable autotuner %s",
                        result.kernel_name,
                        exc_info=True,
                    )
                    continue
                # We make a future instead of returning the kernel here so that
                # kernels that are not statically launchable (i.e. cache miss)
                # can launch a worker without waiting on the blocking step of
                # StaticAutotunerFuture.result().
                CompiledTritonKernels._cache[result.cache_key] = StaticAutotunerFuture(
                    result.kernel
                )
                counters["inductor"]["triton_bundler_load_static_autotuner"] += 1
                kernel_names.append(result.kernel_name)
        return kernel_names

    @classmethod
    def collect(
        cls,
    ) -> tuple[TritonBundle, TritonBundlerMetadata | None]:
        """
        This is the main function called when a cache write happens. This function
        converts all the previously remembered kernels into bundled format so that
        it can be written into a cache entry.
        This function also finalizes the current bundle.
        """
        from torch._inductor import config

        if not TritonBundler.is_enabled():
            cls.end_compile()
            set_feature_use("triton_bundling", False)
            return TritonBundle([], []), None
        set_feature_use("triton_bundling", True)

        with dynamo_timed(key="TritonBundler.collect", log_pt2_compile_event=True):
            entries = cls._entries
            if entries is not None:
                # Only bundle winning autotuning configs. If _winners is
                # non-empty, skip entries whose kernel_hash is not a winner.
                # When _winners is empty (single-config kernels, or no
                # autotuning ran), bundle everything.
                winners = cls._winners
                result: list[TritonKernelArtifacts] = []
                kernel_names: list[str] = []
                for entry in entries:
                    if winners and entry.kernel_hash not in winners:
                        log.debug("Skipping non-winning kernel %s", entry.kernel_hash)
                        continue
                    artifacts: list[TritonKernelArtifact] = []
                    path = os.path.join(entry.directory, entry.kernel_hash)
                    if not os.path.exists(path):
                        continue
                    for filename in os.listdir(path):
                        filepath = os.path.join(path, filename)
                        try:
                            if not os.path.isfile(filepath):
                                raise AssertionError(
                                    f"expected a regular file, got {filepath}"
                                )
                            with open(filepath, "rb") as file:
                                payload = file.read()
                                if filepath.endswith(".json"):
                                    # Make sure there's no sentinel value
                                    if TritonBundler._REPLACE_BYTES in payload:
                                        log.warning(
                                            "Bundle contains illegal %s, payload: %s",
                                            TritonBundler._REPLACE_BYTES,
                                            payload,
                                        )
                                        raise AssertionError(
                                            "Bundle contains illegal bytes"
                                        )
                                    # Remove the path from payload
                                    payload = payload.replace(
                                        str.encode(path), TritonBundler._REPLACE_BYTES
                                    )
                                artifacts.append(
                                    TritonKernelArtifact(filename, payload)
                                )
                            counters["inductor"]["triton_bundler_save_kernel"] += 1
                        except Exception:
                            log.debug("failed to collect triton kernel", exc_info=True)
                        extension = os.path.splitext(filename)[1]
                        if extension in GPU_KERNEL_BIN_EXTS.values():
                            # Each kernel has bunch of files like .cubin(for cuda), .zebin(for xpu), .json, .ttir
                            # Just append one of them without the extension
                            kernel_names.append(Path(filename).stem)
                    if artifacts:
                        result.append(
                            TritonKernelArtifacts(
                                entry.kernel_hash,
                                entry.device,
                                artifacts,
                            )
                        )
                if config.use_static_triton_launcher:
                    static_autotuners, static_kernel_names = (
                        cls.collect_static_autotuners()
                    )
                else:
                    static_autotuners = []
                    static_kernel_names = []
                cls.end_compile()
                return TritonBundle(result, static_autotuners), TritonBundlerMetadata(
                    kernel_names, static_kernel_names
                )
            return TritonBundle([], []), None

    @staticmethod
    def read_and_emit(bundle: TritonBundle) -> TritonBundlerMetadata | None:
        """
        This is the main function called when a cache read happens. This function
        converts the bundled format back into individual files and writes them
        to the filesystem.

        NOTE: When we are writing to the filesystem, we assume exclusive access
        to the target directory.
        This means that if the target folder already exists and is non-empty,
        we bail out.
        Exclusive access means that no other process should be writing to
        or reading from the target directory.
        """
        from torch._inductor import config

        if not TritonBundler.is_enabled():
            return None

        with dynamo_timed(
            key="TritonBundler.read_and_emit", log_pt2_compile_event=True
        ):
            kernel_names: list[str] = []

            for artifacts in bundle.kernel_artifacts:
                basedir = triton_cache_dir(artifacts.device)
                directory = os.path.join(basedir, artifacts.kernel_hash)

                if os.path.exists(directory) and len(os.listdir(directory)) != 0:
                    # If directory already exists, we bail out and leave
                    # local disk to take care of caching
                    log.debug(
                        "Bailing out TritonBundler.read_and_emit, %s is non empty",
                        directory,
                    )
                    continue

                Path(basedir).mkdir(parents=True, exist_ok=True)

                # Random ID to avoid any collisions
                rnd_id = str(uuid.uuid4())
                tmp_dir = os.path.join(basedir, f"tmp.{rnd_id}")
                os.makedirs(tmp_dir)

                for artifact in artifacts.artifacts:
                    filepath = os.path.join(tmp_dir, artifact.filename)
                    with open(filepath, "wb") as file:
                        payload = artifact.payload
                        if artifact.filename.endswith(".json"):
                            payload = payload.replace(
                                TritonBundler._REPLACE_BYTES, str.encode(directory)
                            )
                        file.write(payload)
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"] += 1
                    extension = os.path.splitext(artifact.filename)[1]
                    if extension in GPU_KERNEL_BIN_EXTS.values():
                        # Each kernel has bunch of files like .cubin(for cuda), zebin(for xpu), .json, .ttir
                        # Just append one of them without the extension
                        kernel_names.append(Path(artifact.filename).stem)

                if _IS_WINDOWS:
                    with FileLock(directory + ".lock"):
                        if os.path.exists(directory):
                            shutil.rmtree(directory)
                        os.replace(tmp_dir, directory)
                else:
                    # Atomic on POSIX systems
                    try:
                        os.replace(tmp_dir, directory)
                    except OSError:
                        log.warning("Directory %s is not empty - skipping!", tmp_dir)

            if config.use_static_triton_launcher:
                static_kernel_names = TritonBundler.load_autotuners(
                    bundle.static_autotuners
                )
            else:
                static_kernel_names = []
            return TritonBundlerMetadata(kernel_names, static_kernel_names)
<|endoftext|>import contextlib
import threading
from collections.abc import Generator
from typing import Any

import torch


_TLS = threading.local()


def _freezing_active() -> bool:
    return getattr(_TLS, "freezing_active", False)


@contextlib.contextmanager
def enter_freezing() -> Generator[Any, None, None]:
    """
    Context manager to designate when freezing is active.
    """
    prev = _freezing_active()
    _TLS.freezing_active = True
    try:
        yield
    finally:
        _TLS.freezing_active = prev


def record_has_frozen_params(gm: torch.fx.GraphModule) -> None:
    """
    Mark the gm as having frozen params.
    """
    gm._has_frozen_params = True  # type: ignore[assignment]


def has_frozen_params(gm: torch.fx.GraphModule) -> bool:
    """
    Return True if the gm has frozen parameters.
    """
    return getattr(gm, "_has_frozen_params", False)


def maybe_set_is_frozen_param(t: torch.Tensor) -> None:
    """
    Mark the provided tensor as a frozen param if freezing is active.
    """
    if _freezing_active():
        t._is_frozen_param = True  # type: ignore[attr-defined]


def is_frozen_param(t: torch.Tensor) -> bool:
    """
    Return True if the tensor is a frozen param.
    """
    return getattr(t, "_is_frozen_param", False)
<|endoftext|>from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from typing_extensions import final, override

import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._inductor.output_code import (
    CompiledFxGraph,
    CompiledFxGraphConstants,
    OutputCode,
)

from .compile_fx import _CompileFxKwargs, _InProcessFxCompile, FxCompile


# When async compile works with cache, remove the disabling below
BUG_CACHES_DONT_WORK_WITH_ASYNC = True


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from concurrent.futures import Future

    from torch._inductor.utils import InputType
    from torch.fx import GraphModule

    from .compile_fx_ext import _OutOfProcessFxCompile, _WireProtocolPickledOutput


@dataclass
class _PostCompileData:
    example_inputs: Sequence[InputType]
    constants: CompiledFxGraphConstants
    graph_kwargs: _CompileFxKwargs


@dataclass
class ProgressiveCompilationState:
    progression_futures: deque[Future[_WireProtocolPickledOutput]]
    callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    post_compile_data: _PostCompileData | None

    def check_and_get_ready_stage(self) -> int:
        """Check if any progression stage is ready and return its index, or -1 if none are ready."""
        if not self.progression_futures:
            return -1

        stage_index = -1
        if self.post_compile_data:
            for i, future in enumerate(self.progression_futures):
                if future.done():
                    stage_index = i

        return stage_index

    def switch_to_progression_stage(self, stage_index: int) -> tuple[OutputCode, bool]:
        """
        Switch to the specified progression stage and return the optimized output code.
        Returns a tuple of (optimized_output_code, should_clear_compilation_state).
        """
        future = self.progression_futures[stage_index]
        if future is None:
            raise AssertionError("expected future to be not None")
        optimized_output_code = self.callback(future.result())

        if pcd := self.post_compile_data:
            optimized_output_code.post_compile(
                pcd.example_inputs, pcd.constants, pcd.graph_kwargs
            )

        # Clear earlier progression futures to free memory
        for _ in range(stage_index + 1):
            self.progression_futures.popleft()

        # Return whether all compilation state should be cleared
        should_clear_state = not self.progression_futures
        return optimized_output_code, should_clear_state


# _AsyncOutputCode handles the actual management of waiting for an
# out-of-process compile to finish and then switching over to it.
@final
class _AsyncOutputCode(OutputCode):
    _eager_fn: Callable[..., Any] | None
    _output_code: OutputCode | None
    _future: Future[_WireProtocolPickledOutput] | None
    _callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    _post_compile_data: _PostCompileData | None = None
    _boxed_call: bool  # Copied from the forward/output_code

    def __init__(
        self,
        # eager_fn is run until the future is finished.
        eager_fn: Callable[..., Any],
        # this responds with the result of the out-of-process compile when it's
        # ready.
        future: Future[_WireProtocolPickledOutput],
        # this callback gets called to turn the _WireProtocolPickledOutput into an OutputCode
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None:
        self._eager_fn = eager_fn
        self._boxed_call = getattr(eager_fn, "_boxed_call", False)
        self._output_code = None

        self._future = future
        self._callback = callback

    @override
    def __call__(self, *args: Any) -> Any:
        if self._future is not None and self._future.done():
            args = self._switch_to_compiled_fn(args)

        if eager_fn := self._eager_fn:
            _AsyncFxCompile._stat_eager_runs += 1
            return eager_fn(*args)

        else:
            _AsyncFxCompile._stat_compiled_runs += 1
            if self._output_code is None:
                raise AssertionError("expected self._output_code to be not None")
            return self._output_code.__call__(*args)

    # Takes and returns the args (converted to the "right" boxed mode)
    def _switch_to_compiled_fn(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if self._future is None:
            raise AssertionError("expected self._future to be not None")

        # TODO: If the future ended in an exception do we want to continue
        # running eager or hit the exception now?
        f, self._future = self._future, None
        output_code = self._callback(f.result())

        if pcd := self._post_compile_data:
            self._post_compile_data = None

            output_code.post_compile(
                pcd.example_inputs, pcd.constants, pcd.graph_kwargs
            )

        self._output_code = output_code
        self._eager_fn = None
        boxed_call = getattr(output_code, "_boxed_call", False)

        if self._boxed_call != boxed_call:
            if self._boxed_call:
                # Was boxed, now unboxed
                args = args[0] if len(args) > 0 else ()
            else:
                # Was unboxed, now boxed
                args = (args,)

        self._boxed_call = boxed_call
        return args

    @override
    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        if self._eager_fn is not None:
            self._post_compile_data = _PostCompileData(
                example_inputs, constants, graph_kwargs
            )
        else:
            if self._output_code is None:
                raise AssertionError("expected self._output_code to be not None")
            self._output_code.post_compile(example_inputs, constants, graph_kwargs)


# Given an FxCompile for an out-of-process compile _AsyncFxCompile will run
# eager until the compiled artifact is ready then it will automatically switch
# over to using the compiled version.
@final
class _AsyncFxCompile(FxCompile):
    _compile: _OutOfProcessFxCompile

    # Some debugging stats:
    # Number of times we started a background compile.
    _stat_bg_started: int = 0
    # Number of times we finished a background compile.
    _stat_bg_finished: int = 0
    # Number of times we ran "eager"
    _stat_eager_runs: int = 0
    # Number of times we ran our compiled (out-of-process) artifact
    _stat_compiled_runs: int = 0

    def __init__(self, compile: _OutOfProcessFxCompile) -> None:
        self._compile = compile

    @classmethod
    def _reset_stats(cls) -> None:
        cls._stat_bg_started = 0
        cls._stat_bg_finished = 0
        cls._stat_eager_runs = 0
        cls._stat_compiled_runs = 0

    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        eager_compile = _InProcessFxCompile()
        eager_compile.compile_region_name = self.compile_region_name
        eager_output_code = eager_compile.codegen_and_compile(
            gm, example_inputs, inputs_to_check, graph_kwargs
        )

        # This is similar to _SerializedFxCompile.codegen_and_compile() but
        # handles the async routing.

        serialized = self._compile.serialize_compile(
            gm, example_inputs, inputs_to_check, graph_kwargs
        )
        if not serialized:
            # We can't serialize - just return the eager OutputCode
            return eager_output_code

        inputs, constants = serialized

        _AsyncFxCompile._stat_bg_started += 1
        f = self._compile._send_to_child_async(inputs)

        # This is called by _switch_to_compiled_fn() when f has a result...
        def callback(pickled_output: _WireProtocolPickledOutput) -> OutputCode:
            _AsyncFxCompile._stat_bg_finished += 1
            output = pickled_output.deserialize(constants)
            if isinstance(output.graph, CompiledFxGraph):
                output.graph.compile_region_name = self.compile_region_name
            self._compile._postprocess(output)
            return output.graph

        return _AsyncOutputCode(eager_output_code, f, callback)


# _ProgressiveOutputCode handles running a fast compile first, then hot-swapping
# to a more optimized version when the expensive compile finishes.
@final
class _ProgressiveOutputCode(OutputCode):
    _fast_output_code: OutputCode | None
    _optimized_output_code: OutputCode | None
    _compilation_state: ProgressiveCompilationState | None
    # _boxed_call state is effectively cached (we sometimes wrap unboxed w/
    # lambdas to box them) so we can't change it mid-way. Since _boxed_call=True
    # is more common let's default to that and we'll convert if necessary.
    _boxed_call: bool = True

    def __init__(
        self,
        # Fast compile that runs faster than the progressive compiles
        fast_output_code: OutputCode,
        # Futures for the progressive optimized compiles
        progression_futures: Sequence[Future[_WireProtocolPickledOutput]],
        # Callback to convert the optimized result to OutputCode
        callback: Callable[[_WireProtocolPickledOutput], OutputCode],
    ) -> None:
        self._fast_output_code = fast_output_code
        self._optimized_output_code = None
        self._compilation_state = ProgressiveCompilationState(
            progression_futures=deque(progression_futures),
            callback=callback,
            post_compile_data=None,
        )

    @override
    def __call__(self, args: Sequence[Any]) -> Any:
        # Check if any newer progression stage is ready and switch to it
        self._check_and_switch_progression()

        if self._optimized_output_code is not None:
            _ProgressiveFxCompile._stat_optimized_runs += 1
            output_code = self._optimized_output_code
        else:
            _ProgressiveFxCompile._stat_fast_runs += 1
            if self._fast_output_code is None:
                raise AssertionError("expected self._fast_output_code to be not None")
            output_code = self._fast_output_code

        boxed_call = getattr(output_code, "_boxed_call", False)
        if boxed_call:
            res = output_code.__call__(args)
        else:
            res = output_code.__call__(*args)
        return res

    def _check_and_switch_progression(self) -> None:
        if not self._compilation_state:
            return

        stage_index = self._compilation_state.check_and_get_ready_stage()
        if stage_index == -1:
            # no futures are ready
            return

        self._switch_to_progression_stage(stage_index)

    def _switch_to_progression_stage(self, stage_index: int) -> None:
        if self._compilation_state is None:
            raise AssertionError("expected self._compilation_state to be not None")
        optimized_output_code, should_clear_state = (
            self._compilation_state.switch_to_progression_stage(stage_index)
        )

        self._optimized_output_code = optimized_output_code
        self._fast_output_code = None

        # Clear all compilation state if no more progression futures are left
        if should_clear_state:
            self._compilation_state = None

    @override
    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        if self._fast_output_code is None:
            raise AssertionError("expected self._fast_output_code to be not None")
        self._fast_output_code.post_compile(example_inputs, constants, graph_kwargs)

        if self._compilation_state is None:
            raise AssertionError("expected self._compilation_state to be not None")
        # Store for later when optimized version is ready
        self._compilation_state.post_compile_data = _PostCompileData(
            example_inputs, constants, graph_kwargs
        )


# _ProgressiveFxCompile runs a fast compile immediately, then kicks off
# progressive compiles in the background and hot-swaps when they're ready.
@final
class _ProgressiveFxCompile(FxCompile):
    _fast_compile: FxCompile
    _optimized_compile: _OutOfProcessFxCompile
    _progression_configs: list[dict[str, Any]]

    # Debugging stats
    _stat_bg_started: int = 0
    _stat_bg_finished: int = 0
    _stat_fast_runs: int = 0
    _stat_optimized_runs: int = 0

    def __init__(
        self,
        fast_compile: FxCompile,
        optimized_compile: _OutOfProcessFxCompile,
        progression_configs: list[dict[str, Any]],
    ) -> None:
        self._fast_compile = fast_compile
        self._optimized_compile = optimized_compile
        self._progression_configs = progression_configs

    @classmethod
    def _reset_stats(cls) -> None:
        cls._stat_bg_started = 0
        cls._stat_bg_finished = 0
        cls._stat_fast_runs = 0
        cls._stat_optimized_runs = 0

    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        import torch._inductor.config as inductor_config

        progression_futures: list[Future[_WireProtocolPickledOutput]] = []

        for config in self._progression_configs:
            with inductor_config.patch(config):
                _ProgressiveFxCompile._stat_bg_started += 1

                # Start the progressive compiles in the background
                serialized = self._optimized_compile.serialize_compile(
                    gm, example_inputs, inputs_to_check, graph_kwargs
                )

                if not serialized:
                    continue

                inputs, constants = serialized
                future = self._optimized_compile._send_to_child_async(inputs)
                progression_futures.append(future)

        fast_output_code = self._fast_compile.codegen_and_compile(
            gm, example_inputs, inputs_to_check, graph_kwargs
        )

        if not progression_futures:
            # All async compile attempts failed - just return the fast version
            return fast_output_code

        # Callback to handle the optimized result.
        # This callback may be called multiple times, once for each progressive level completed,
        # but may be skipped if a level either never completes or if a more optimal level
        # completes before a less optimal one is switched to.
        def callback(pickled_output: _WireProtocolPickledOutput) -> OutputCode:
            _ProgressiveFxCompile._stat_bg_finished += 1
            output = pickled_output.deserialize(constants)
            if isinstance(output.graph, CompiledFxGraph):
                output.graph.compile_region_name = self.compile_region_name
            self._optimized_compile._postprocess(output)
            return output.graph

        return _ProgressiveOutputCode(fast_output_code, progression_futures, callback)
<|endoftext|>from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from ast import literal_eval
from functools import cached_property
from hashlib import sha256
from os import getenv
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never, override, Self

from torch.utils._filelock import FileLock


if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor


# TypeVars can't be recursive, so generic types that fall within
# Key or Value can't be bound properly; for example, Key should
# only take tuples of other Key types: tuple[Key, ...]. this is
# a known shortcoming of torch's typing
Key = TypeVar("Key", str, int, tuple[Any, ...])
Value = TypeVar("Value", str, int, tuple[Any, ...], bytes, dict[Any, Any], list[Any])


class CacheError(ValueError):
    """
    Exception raised for errors encountered during cache operations.
    """


class Cache(ABC, Generic[Key, Value]):
    """
    Abstract base class for cache implementations.
    Provides the interface for cache operations.
    """

    @abstractmethod
    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache.
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present, else None.
        """

    @abstractmethod
    def insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        """


class InMemoryCache(Cache[Key, Value]):
    """
    In-memory cache implementation using a dictionary and thread lock.
    """

    def __init__(self: Self) -> None:
        """
        Initialize an empty in-memory cache.
        """
        self._cache: dict[Key, Value] = {}
        self._lock: Lock = Lock()

    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache.
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present, else None.
        """
        with self._lock:
            if (value := self._cache.get(key)) is not None:
                return value
            return None

    def insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        """
        with self._lock:
            if key in self._cache:
                # no overwrites for insert!
                return False
            self._cache[key] = value
            return True

    @classmethod
    def from_env_var(cls, env_var: str) -> Self:
        """
        Create an in-memory cache from an environment variable.
        Args:
            env_var (str): Name of the environment variable containing cache data.
        Returns:
            InMemoryCache: An instance populated from the environment variable.
        Raises:
            CacheError: If the environment variable is malformed or contains invalid data.
        """
        cache = cls()

        if (env_val := getenv(env_var)) is None:
            # env_var doesn't exist = empty cache
            return cache

        for kv_pair in env_val.split(";"):
            # ignore whitespace prefix/suffix
            kv_pair = kv_pair.strip()

            if not kv_pair:
                # kv_pair could be '' if env_val is '' or has ; suffix
                continue

            try:
                # keys and values should be comma separated
                key_bytes_repr, value_bytes_repr = kv_pair.split(",", 1)
            except ValueError as err:
                raise CacheError(
                    f"Malformed kv_pair {kv_pair!r} from env_var {env_var!r}, likely missing comma separator."
                ) from err

            # ignore whitespace prefix/suffix, again
            key_bytes_repr, value_bytes_repr = (
                key_bytes_repr.strip(),
                value_bytes_repr.strip(),
            )

            try:
                # check that key_bytes_str is an actual, legitimate encoding
                key_bytes = literal_eval(key_bytes_repr)
            except (ValueError, SyntaxError) as err:
                raise CacheError(
                    f"Malformed key_bytes_repr {key_bytes_repr!r} in kv_pair {kv_pair!r}, encoding is invalid."
                ) from err
            try:
                # check that value_bytes_str is an actual, legitimate encoding
                value_bytes = literal_eval(value_bytes_repr)
            except (ValueError, SyntaxError) as err:
                raise CacheError(
                    f"Malformed value_bytes_repr {value_bytes_repr!r} in kv_pair {kv_pair!r}, encoding is invalid."
                ) from err

            try:
                key = pickle.loads(key_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Malformed key_bytes_repr {key_bytes_repr!r} in kv_pair {kv_pair!r}, not un-pickle-able."
                ) from err
            try:
                value = pickle.loads(value_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Malformed value_bytes_repr {value_bytes_repr!r} in kv_pair {kv_pair!r}, not un-pickle-able."
                ) from err

            # true duplicates, i.e. multiple occurrences of the same key => value
            # mapping are ok and treated as a no-op; key duplicates with differing
            # values, i.e. key => value_1 and key => value_2 where value_1 != value_2,
            # are not okay since we don't allow overwriting cached values (it's bad regardless)
            if (not cache.insert(key, value)) and (cache.get(key) != value):
                raise CacheError(
                    f"Multiple values for key {key!r} found, got {cache.get(key)!r} and {value!r}."
                )

        return cache

    @classmethod
    def from_file_path(cls, fpath: Path) -> Self:
        """
        Create an in-memory cache from a file path.
        Args:
            fpath (Path): Path to the file containing pickled cache data.
        Returns:
            InMemoryCache: An instance populated from the file.
        Raises:
            CacheError: If the file is not a valid pickled dictionary.
        """
        cache = cls()

        if not fpath.is_file():
            # fpath doesn't exit = empty cache
            return cache

        try:
            with open(fpath, "rb") as fp:
                cache._cache = pickle.load(fp)
        except pickle.UnpicklingError as err:
            raise CacheError(
                f"Failed to create cache from file path {fpath}, file contents are un-pickle-able."
            ) from err

        if not isinstance(cache._cache, dict):
            raise CacheError(
                f"Failed to create cache from file path {fpath}, file contents not pickled dict[Key, Value]."
            )

        return cache


class AsyncCache(Cache[Key, Value]):
    """
    Asynchronous cache implementation using ThreadPoolExecutor.
    """

    def get_async(
        self: Self, key: Key, executor: ThreadPoolExecutor
    ) -> Future[Value | None]:
        """
        Retrieve a value from the cache asynchronously.
        Args:
            key (Key): The key to look up.
            executor (ThreadPoolExecutor): Executor for async execution.
        Returns:
            Future[Value | None]: Future for the cached value or None.
        """
        return executor.submit(self.get, key)

    def insert_async(
        self: Self, key: Key, value: Value, executor: ThreadPoolExecutor
    ) -> Future[bool]:
        """
        Insert a value into the cache asynchronously.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
            executor (ThreadPoolExecutor): Executor for async execution.
        Returns:
            Future[bool]: Future for the result of insertion.
        """
        return executor.submit(self.insert, key, value)


class OnDiskCache(AsyncCache[Key, Value]):
    """
    On-disk cache implementation using files and file locks.
    Stores cache data in files on disk, with atomic operations and versioning.
    Supports custom cache directory names.
    Attributes:
        version (int): The version used for cache versioning.
        name (str): The name of the cache directory.
    """

    version: int = 0

    def __init__(self: Self, name: str | None = None) -> None:
        """
        Initialize an on-disk cache instance.
        Args:
            name (str | None, optional): The name of the cache directory. If None,
                defaults to "on_disk_cache".
        """
        self.name = name or "on_disk_cache"

    @cached_property
    def base_dir(self: Self) -> Path:
        """
        Get the base directory for the cache.
        Returns:
            Path: The base directory path for storing cache files.
        """
        return Path(gettempdir()) / "cache" / self.name

    def _fpath_from_key(self: Self, key: Key) -> Path:
        """
        Get the file path for a given key.
        Args:
            key (Key): The key to convert to a file path.
        Returns:
            Path: The file path for the key.
        Raises:
            CacheError: If the key is not pickle-able.
        """
        try:
            return self.base_dir / sha256(pickle.dumps(key)).hexdigest()[:32]
        except (AttributeError, pickle.PicklingError) as err:
            raise CacheError(
                f"Failed to get fpath for key {key!r}, key is not pickle-able."
            ) from err
        # pyrefly: ignore [bad-argument-type]
        assert_never(key)

    def _flock_from_fpath(self: Self, fpath: Path) -> FileLock:
        """
        Get a file lock for a given file path.
        Args:
            fpath (Path): The file path.
        Returns:
            FileLock: The file lock for the path.
        """
        # fpath.name is a hex digest, meaning there are 16^4 potential values
        # for fpath.name[:4]; this is more than enough unique locks to not
        # cause additional overhead from shared locks and it also saves our
        # cache dir from becoming 50 percent locks
        # pyrefly: ignore [bad-return]
        return FileLock(str(fpath.parent / "locks" / fpath.name[:4]) + ".lock")

    @property
    def version_prefix(self: Self) -> bytes:
        """
        Get the version prefix for the cache.
        Returns:
            bytes: The version prefix as bytes, derived from the cache version string.
        """
        return sha256(str(OnDiskCache.version).encode()).digest()[:4]

    @override
    def get(self: Self, key: Key) -> Value | None:
        """
        Retrieve a value from the cache.
        Args:
            key (Key): The key to look up.
        Returns:
            Value | None: The cached value if present and version matches, else None.
        Raises:
            CacheError: If the value is corrupted or cannot be unpickled.
        Side Effects:
            Removes stale cache files if the version prefix does not match.
        """
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)

        with flock:
            if not fpath.is_file():
                return None

            value_bytes = None
            prefix_length = len(self.version_prefix)
            with open(fpath, "rb") as fp:
                if fp.read(prefix_length) == self.version_prefix:
                    value_bytes = fp.read()

            if value_bytes is None:
                # version_prefix did not match, so we can't read the stale
                # cached value; we should also remove the stale cached value,
                # so that key can be re-cached by the newer version
                fpath.unlink()
                return None

            try:
                value = pickle.loads(value_bytes)
            except pickle.UnpicklingError as err:
                raise CacheError(
                    f"Failed to get key {key!r}, value is potentially corrupted (value is not un-pickle-able)."
                ) from err

            return value

    @override
    def insert(self: Self, key: Key, value: Value) -> bool:
        """
        Insert a value into the cache.
        Args:
            key (Key): The key to insert.
            value (Value): The value to associate with the key.
        Returns:
            bool: True if the value was inserted, False if the key already exists.
        Raises:
            CacheError: If the value is not pickle-able.
        Side Effects:
            Creates the cache directory if it does not exist.
        """
        fpath = self._fpath_from_key(key)
        flock = self._flock_from_fpath(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" mode is exclusive creation, meaning the file will be created
            # iff the file does not already exist (atomic w/o overwrite); use
            # flock for added atomicity guarantee and to prevent partial writes
            with flock as _, open(fpath, "xb") as fp:
                fp.write(self.version_prefix)
                pickle.dump(value, fp)
        except pickle.PicklingError as err:
            raise CacheError(
                f"Failed to insert key {key!r} with value {value!r}, value is not pickle-able."
            ) from err
        except FileExistsError:
            return False
        return True


class InductorOnDiskCache(OnDiskCache[Key, Value]):
    """
    Inductor-specific on-disk cache implementation.
    Uses a custom base directory for Inductor cache files.
    """

    def __init__(self: Self) -> None:
        """
        Initialize an inductor on-disk cache instance.
        Sets the cache directory name to "inductor_on_disk_cache".
        """
        super().__init__("inductor_on_disk_cache")

    @cached_property
    def base_dir(self: Self) -> Path:
        """
        Get the base directory for the Inductor cache.
        Returns:
            Path: The base directory path for Inductor cache files.
        """
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "cache", self.name)
<|endoftext|>from __future__ import annotations

import csv
import dataclasses
import inspect
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from torch._inductor import config
from torch._inductor.utils import get_benchmark_name
from torch.utils._ordered_set import OrderedSet


# Prevent circular import
if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.triton_compat import Config
    from torch._inductor.scheduler import BaseSchedulerNode

# counter for tracking how many kernels have been generated
generated_kernel_count = 0
generated_cpp_vec_kernel_count = 0
num_bytes_accessed = 0
nodes_num_elem: list[
    tuple[
        BaseSchedulerNode,
        int,
    ]
] = []
node_runtimes: list[tuple[BaseSchedulerNode, float]] = []

# counters for tracking fusions
ir_nodes_pre_fusion = 0

# counters for tracking to_dtype inserted
cpp_to_dtype_count = 0


@dataclasses.dataclass
class CppOuterLoopFusedCount:
    inner_kernel_number: int
    local_buffer_number: int = 0


# The length counts the number of outer loop fusions.
cpp_outer_loop_fused_inner_counts: list[CppOuterLoopFusedCount] = []

num_comprehensive_padding = 0
num_matches_for_scatter_upon_const_tensor = 0

num_loop_reordering = 0
num_auto_chunking: int = 0

# counter for parallel reduction.
parallel_reduction_count = 0

codegen_mix_order_reduction: int = 0
rejected_mix_order_reduction_fusion: int = 0
codegen_nested_reduction: int = 0


# reset all counters
def reset() -> None:
    global generated_kernel_count
    global generated_cpp_vec_kernel_count
    global num_bytes_accessed, nodes_num_elem
    global ir_nodes_pre_fusion
    global cpp_to_dtype_count
    global cpp_outer_loop_fused_inner_counts
    global num_comprehensive_padding
    global num_matches_for_scatter_upon_const_tensor
    global num_loop_reordering
    global parallel_reduction_count
    global codegen_mix_order_reduction
    global rejected_mix_order_reduction_fusion
    global codegen_nested_reduction
    global num_auto_chunking

    generated_kernel_count = 0
    generated_cpp_vec_kernel_count = 0
    num_bytes_accessed = 0
    nodes_num_elem.clear()
    node_runtimes.clear()
    ir_nodes_pre_fusion = 0
    cpp_to_dtype_count = 0
    cpp_outer_loop_fused_inner_counts.clear()
    num_comprehensive_padding = 0
    num_matches_for_scatter_upon_const_tensor = 0
    num_loop_reordering = 0
    parallel_reduction_count = 0
    codegen_mix_order_reduction = 0
    rejected_mix_order_reduction_fusion = 0
    codegen_nested_reduction = 0
    num_auto_chunking = 0


@dataclass
class CachedMetricsDeltas:
    """
    The subset of metrics we want update across cache hits, e.g., the
    FxGraphCache.
    """

    generated_kernel_count: int
    generated_cpp_vec_kernel_count: int
    ir_nodes_pre_fusion: int
    cpp_to_dtype_count: int
    num_bytes_accessed: int
    num_matches_for_scatter_upon_const_tensor: int


def get_metric_fields() -> list[str]:
    return [field.name for field in dataclasses.fields(CachedMetricsDeltas)]


class CachedMetricsHelper:
    """
    A helper class to help calculate and apply counter deltas for those
    metrics we want to save with cache entries (e.g., FxGraphCache) and
    apply on a cache hit.
    """

    def __init__(self) -> None:
        self.cached_metrics = {}
        for metric in get_metric_fields():
            self.cached_metrics[metric] = globals()[metric]

    def get_deltas(self) -> CachedMetricsDeltas:
        delta_metrics = {}
        for metric in get_metric_fields():
            delta_metrics[metric] = globals()[metric] - self.cached_metrics[metric]

        return CachedMetricsDeltas(**delta_metrics)

    @staticmethod
    def apply_deltas(delta: CachedMetricsDeltas) -> None:
        for metric in get_metric_fields():
            globals()[metric] += getattr(delta, metric)


REGISTERED_METRIC_TABLES: dict[str, MetricTable] = {}


@dataclass
class MetricTable:
    table_name: str
    column_names: list[str]

    num_rows_added: int = 0

    def add_row(self, row_fn: Callable[[], dict[str, str | float | None]]) -> None:
        if self.table_name not in enabled_metric_tables():
            return

        row_dict = row_fn()
        if len(self.column_names) != len(row_dict):
            raise AssertionError(f"{len(self.column_names)} v.s. {len(row_dict)}")
        if OrderedSet(self.column_names) != OrderedSet(row_dict.keys()):
            raise AssertionError(
                f"{OrderedSet(self.column_names)} v.s. {OrderedSet(row_dict.keys())}"
            )

        bn = get_benchmark_name()
        # assert bn is not None
        row = [bn] + [row_dict[column_name] for column_name in self.column_names]
        if not all(isinstance(i, (str, float, type(None))) for i in row):
            raise AssertionError("expected all row values to be str, float, or None")
        self._write_row(row)

    def output_filename(self) -> str:
        return f"metric_table_{self.table_name}.csv"

    def write_header(self) -> None:
        filename = self.output_filename()
        with open(filename, "w") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            writer.writerow(["model_name"] + self.column_names)

    def _write_row(self, row: list[str | float | None]) -> None:
        filename = self.output_filename()
        if self.num_rows_added == 0 and not os.path.exists(filename):
            self.write_header()

        self.num_rows_added += 1

        for idx, orig_val in enumerate(row):
            if isinstance(orig_val, float):
                new_val = f"{orig_val:.6f}"
            elif orig_val is None:
                new_val = ""
            else:
                new_val = orig_val
            row[idx] = new_val

        with open(filename, "a") as fd:
            writer = csv.writer(fd, lineterminator="\n")
            writer.writerow(row)

    @staticmethod
    def register_table(name: str, column_names: list[str]) -> None:
        table = MetricTable(name, column_names)
        REGISTERED_METRIC_TABLES[name] = table


MetricTable.register_table(
    "slow_fusion",
    [
        "kernel1_path",
        "kernel1_latency",
        "kernel2_path",
        "kernel2_latency",
        "fused_kernel_path",
        "fused_kernel_latency",
        "slow_down_ratio",
    ],
)

# track the fusion statistics for each graph
MetricTable.register_table(
    "graph_stats",
    [
        "graph_id",
        "num_nodes_before_fusion",
        "num_nodes_after_fusion",
    ],
)

# track the perf difference between persistent reduction and non-persistent
# reductions
MetricTable.register_table(
    "persistent_red_perf",
    [
        "kernel0_path",
        "kernel1_path",
        "kernel2_path",
        "kernel3_path",
        "kernel0_latency",
        "kernel1_latency",
        "kernel2_latency",
        "kernel3_latency",
        "size_hints",
        "reduction_hint",
    ],
)

# Log the fusion failures due to indexing mismatch
MetricTable.register_table(
    "fusion_failure_due_to_indexing_mismatch",
    [
        "pre_grad_graph_id",
        "post_grad_graph_id",
        "node1_name",
        "node2_name",
        "node1_debug_str",
        "node2_debug_str",
        "common_buffer_names",
        "failure_reason",
    ],
)

# Log metadata for pointwise/reduction kernels. E.g., model name, kernel path, numel, rnumel, reduction hint
MetricTable.register_table(
    "kernel_metadata",
    [
        "kernel_name",
        "kernel_path",
        "kernel_category",  # pointwise/reduction/foreach etc.
        "size_hints",
        "reduction_hint",
        "line_of_code",
        "num_load",
        "num_store",
        "num_for_loop",
        "num_atomic_add",
        "num_args",
        # xyz numel can be different to size_hints since size_hints are rounded
        # up to the nearest power of 2.
        # Inductor kernel will burn in the xyz numel in kernel code for static
        # shape kernels.
        # Logging them will be helpful to find unaligned shape for reduction
        "xnumel",
        "ynumel",
        "rnumel",
        "kernel_args_num_gb",
    ],
)


def _parse_kernel_fn_code(kernel_module_code: str) -> str:
    """
    The kernel_module_code is the python module that contains kernel function code.
    kernel function is the proper triton kernel function annotated with
    @triton.jit
    """
    from .codecache import PyCodeCache
    from .wrapper_benchmark import get_triton_kernel

    mod = PyCodeCache.load(kernel_module_code)
    kernel = get_triton_kernel(mod)
    # kernel is a CachingAutotune; kernel.fn is the JITFunction;
    # kernel.fn.fn is the function being decorate by triton.jit
    return inspect.getsource(kernel.fn.fn)


def _parse_kernel_line_of_code(proper_kernel_fn_code: str) -> int:
    """
    Return the line of code for the kernel excluding the decorators.
    """
    return len(proper_kernel_fn_code.splitlines())


def _parse_size_hints(kernel_module_code: str, kernel_category: str) -> str | None:
    if kernel_category == "foreach":
        # foreach kernel does not have size_hints
        return None
    m = re.search(r"size_hints=(\[[0-9, ]*\]),", kernel_module_code)
    if not m:
        raise AssertionError("size_hints missing!")
    return m.group(1)


def _parse_reduction_hint(kernel_category: str, kernel_module_code: str) -> str | None:
    if kernel_category not in ("reduction", "persistent_reduction"):
        return None
    m = re.search(r"reduction_hint=ReductionHint\.(\w*),", kernel_module_code)
    if not m:
        raise AssertionError("reduction_hint not found in kernel source code!")
    return m.group(1)


def _count_pattern(proper_kernel_fn_code: str, pattern: str) -> int:
    return proper_kernel_fn_code.count(pattern)


def _count_args(proper_kernel_fn_code: str) -> int:
    def_line = proper_kernel_fn_code.splitlines()[0]
    if not def_line.startswith("def "):
        raise AssertionError(
            f"expected def line to start with 'def ', got {def_line!r}"
        )
    start_idx = def_line.index("(")
    end_idx = def_line.index("):")
    decl_csv = def_line[start_idx + 1 : end_idx]
    comps = decl_csv.split(",")
    return len(comps)


def _parse_proper_kernel_fn_code(kernel_fn_code: str) -> str:
    """
    Skip decorators.
    """
    start_pos = kernel_fn_code.index("def ")
    return kernel_fn_code[start_pos:]


def _parse_numel(proper_kernel_fn_code: str, numel_arg_name: str) -> int | None:
    m = re.search(f"{numel_arg_name} = ([\\d]+)", proper_kernel_fn_code)
    if m:
        return int(m.group(1))
    else:
        return None


def _parse_kernel_args_num_gb(
    kernel_fn_code: str, kernel_category: str
) -> float | None:
    """
    inductor meta looks like:
        inductor_meta={... 'mutated_arg_names': [], 'no_x_dim': False, 'kernel_num_gb': 2.0},
    """
    m = re.search(r".kernel_num_gb.:\s*([0-9.]+)", kernel_fn_code)
    if m:
        return float(m.group(1))
    else:
        """
        There are a few cases that kernel_num_gdb field can be missing:
        1. the field will be missing if config.benchmark_kernel and
           config.profile_bandwidth are false
        2. even if config.benchmark_kernel or config.profile_bandwidth is true.
           foreach kernel does not have kernel_num_gb field in the metadata
        """
        return None


def log_kernel_metadata(
    kernel_name: str, kernel_path: str, kernel_module_code: str
) -> None:
    """
    A utility to log kernel metadata. We may parse metadata from kernel source code here.

    It's fine to parse the generated kernel code here since the logging is
    disabled by default. It would hurt compilation time.
    """
    from .wrapper_benchmark import get_kernel_category_by_source_code

    kernel_category = get_kernel_category_by_source_code(kernel_module_code)
    reduction_hint = _parse_reduction_hint(kernel_category, kernel_module_code)
    size_hints = _parse_size_hints(kernel_module_code, kernel_category)
    kernel_fn_code = _parse_kernel_fn_code(kernel_module_code)

    proper_kernel_fn_code = _parse_proper_kernel_fn_code(kernel_fn_code)

    # the line of code excluding the decortors
    kernel_line_of_code = _parse_kernel_line_of_code(proper_kernel_fn_code)

    get_metric_table("kernel_metadata").add_row(
        lambda: {
            "kernel_name": kernel_name,
            "kernel_path": kernel_path,
            "kernel_category": kernel_category,
            "size_hints": size_hints,
            "reduction_hint": reduction_hint,
            "line_of_code": kernel_line_of_code,
            "num_load": _count_pattern(proper_kernel_fn_code, "tl.load"),
            "num_store": _count_pattern(proper_kernel_fn_code, "tl.store"),
            "num_for_loop": _count_pattern(proper_kernel_fn_code, "for "),
            "num_atomic_add": _count_pattern(proper_kernel_fn_code, "tl.atomic_add"),
            "num_args": _count_args(proper_kernel_fn_code),
            "xnumel": _parse_numel(proper_kernel_fn_code, "xnumel"),
            "ynumel": _parse_numel(proper_kernel_fn_code, "ynumel"),
            "rnumel": _parse_numel(proper_kernel_fn_code, "rnumel"),
            "kernel_args_num_gb": _parse_kernel_args_num_gb(
                kernel_fn_code, kernel_category
            ),
        }
    )


def purge_old_log_files() -> None:
    """
    Purge the old log file at the beginning when the benchmark script runs.
    Should do it in the parent process rather than the child processes running
    each individual model.
    """
    for name, table in REGISTERED_METRIC_TABLES.items():
        if name in enabled_metric_tables():
            filename = table.output_filename()
            if os.path.exists(filename):
                os.unlink(filename)

            table.write_header()


def enabled_metric_tables() -> OrderedSet[str]:
    return enabled_metric_tables_impl(config.enabled_metric_tables)


@lru_cache
def enabled_metric_tables_impl(config_str: str) -> OrderedSet[str]:
    enabled: OrderedSet[str] = OrderedSet()
    for name in config_str.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in REGISTERED_METRIC_TABLES:
            raise AssertionError(f"Metric table name {name} is not registered")
        enabled.add(name)
    return enabled


def is_metric_table_enabled(name: str) -> bool:
    return name in enabled_metric_tables()


def get_metric_table(name: str) -> MetricTable:
    if name not in REGISTERED_METRIC_TABLES:
        raise AssertionError(f"Metric table {name} is not defined")
    return REGISTERED_METRIC_TABLES[name]


MetricTable.register_table(
    "kernel_autotune",
    [
        "kernel_path",
        "kernel_name",
        "triton_config",
        "latency_ms",
    ],
)


def log_kernel_autotune_result(
    kernel_path: str, kernel_name: str, config: Config, latency: float
) -> None:
    get_metric_table("kernel_autotune").add_row(
        lambda: {
            "kernel_path": kernel_path,
            "kernel_name": kernel_name,
            "triton_config": str(config),
            "latency_ms": latency,
        }
    )
<|endoftext|>import asyncio
import sys
import weakref
from asyncio import AbstractEventLoop, Future
from collections.abc import Awaitable, Callable, Coroutine, Generator, Iterator
from contextlib import contextmanager, ExitStack
from contextvars import Context
from typing import Any, Protocol, TypeVar

from torch.utils._ordered_set import OrderedSet


T = TypeVar("T")
TCoro = Generator[Any, None, T]

if sys.version_info >= (3, 11):

    class TaskFactory(Protocol):
        def __call__(
            self,
            __loop: AbstractEventLoop,
            __factory: Coroutine[None, None, object] | Generator[None, None, object],
            __context: Context | None = None,
            /,
        ) -> asyncio.futures.Future[object]: ...

    TaskFactoryType = TaskFactory
else:
    TaskFactoryType = Callable[[AbstractEventLoop, Generator[TCoro, None, T]], Future]  # type: ignore[valid-type]


def await_sync(awaitable: Awaitable[T]) -> T:
    with get_loop() as loop:
        return loop.run_until_complete(awaitable)


@contextmanager
def get_loop(
    always_create_new_loop: bool = False,
) -> Iterator[AbstractEventLoop]:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as re:
        if "There is no current event loop in thread" in str(re):
            with _new_loop() as loop:
                yield loop
            return
        else:
            raise

    @contextmanager
    def _restore_loop(
        loop: asyncio.AbstractEventLoop,
    ) -> Iterator[None]:
        try:
            yield
        finally:
            asyncio.set_event_loop(loop)

    @contextmanager
    def _restore_running_loop() -> Iterator[None]:
        loop_from_events = asyncio.events._get_running_loop()
        asyncio.events._set_running_loop(None)
        try:
            yield
        finally:
            asyncio.events._set_running_loop(loop_from_events)

    with ExitStack() as stack:
        if loop.is_running():
            stack.enter_context(_restore_running_loop())
            stack.enter_context(_restore_loop(loop=loop))
            loop = stack.enter_context(_new_loop(loop.get_task_factory()))  # type: ignore[arg-type]
        elif loop.is_closed():
            loop = stack.enter_context(_new_loop())  # type: ignore[arg-type]
        elif always_create_new_loop:
            stack.enter_context(_restore_loop(loop=loop))
            loop = stack.enter_context(_new_loop())  # type: ignore[arg-type]
        yield loop


@contextmanager
def _new_loop(
    task_factory: TaskFactoryType | None = None,
) -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    tasks = _patch_loop(loop)

    if task_factory:
        # pyre-ignore[6]
        loop.set_task_factory(task_factory)  # type: ignore[arg-type]

    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        try:
            _cancel_all_tasks(loop, tasks)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def _cancel_all_tasks(
    loop: AbstractEventLoop,
    tasks: OrderedSet[Future],  # type: ignore[type-arg]
) -> None:
    to_cancel = [task for task in tasks if not task.done()]

    if not to_cancel:
        return

    # pyre-fixme[1001]: Awaitable assigned to `task` is never awaited.
    for task in to_cancel:
        task.cancel()

    # pyrefly: ignore [bad-argument-type]
    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def _patch_loop(loop: AbstractEventLoop) -> OrderedSet[Future]:  # type: ignore[type-arg]
    tasks: weakref.WeakSet[Future] = weakref.WeakSet()  # type: ignore[type-arg]

    task_factories: list[TaskFactoryType | None] = [None]

    def _set_task_factory(factory: TaskFactoryType | None) -> None:
        task_factories[0] = factory

    def _get_task_factory() -> TaskFactoryType | None:
        return task_factories[0]

    def _safe_task_factory(
        loop: AbstractEventLoop,
        coro: TCoro,  # type: ignore[type-arg]
        *,
        context: Context | None = None,
    ) -> asyncio.Future:  # type: ignore[valid-type, type-arg]
        task_factory = task_factories[0]
        if task_factory is None:
            if sys.version_info >= (3, 11):
                # pyrefly: ignore [bad-argument-type]
                task = asyncio.Task(coro, loop=loop, context=context)
            else:
                task = asyncio.Task(coro, loop=loop)
            # pyre-ignore[16]: `Task` has no attribute `_source_traceback`.
            if task._source_traceback:  # type: ignore[attr-defined]
                del task._source_traceback[  # type: ignore[attr-defined]
                    -1
                ]  # pragma: no cover  # type: ignore[attr-defined]
        else:
            if sys.version_info >= (3, 11):
                task = task_factory(loop, coro, context=context)  # type: ignore[arg-type, call-arg, assignment]
            else:
                task = task_factory(loop, coro)  # type: ignore[arg-type]
        #  `Union[Task[Any], Future[Any]]`.
        tasks.add(task)
        return task

    # pyre-ignore[6]
    loop.set_task_factory(_safe_task_factory)  # type: ignore[method-assign, arg-type]
    # pyre-ignore[8]
    loop.set_task_factory = _set_task_factory  # type: ignore[method-assign, assignment]
    # pyre-ignore[8]
    loop.get_task_factory = _get_task_factory  # type: ignore[method-assign, assignment]

    return tasks  # type: ignore[return-value]
<|endoftext|>import asyncio
from typing import TypeVar

import torch._inductor.config as config
from torch._inductor import ir


_T = TypeVar("_T")


def gen_best_config(mat1: ir.StorageBox, mat2: ir.StorageBox) -> asyncio.Task[_T]:
    """
    Generate the best GEMM autotune config for the given matrices.
    """
    if config.is_fbcode():
        from torch._inductor.fb.remote_gemm_autotune_cache import gen_best_config

        return gen_best_config(mat1, mat2)
    else:
        raise NotImplementedError("Function gen_best_config is not yet implemented")
<|endoftext|># mypy: allow-untyped-defs

import sympy

import torch

from .ir import Pointwise, TensorBox
from .virtualized import ops


# pyre-ignore[2,3]
def dense_idx_to_jagged_idx(batch_idx, seq_idx, offsets_loader, jagged_len):
    # jagged_len + 1 is used as the upper bound,
    # because the last sequence length may be zero
    begin_idx = ops.indirect_indexing(
        offsets_loader([batch_idx]),
        jagged_len + 1,
    )
    end_idx = offsets_loader([batch_idx + 1])
    jagged_idx = begin_idx + seq_idx
    return jagged_idx, end_idx


def get_inverse_offsets(
    offsets: TensorBox,
    jagged_len: int | sympy.Expr,
    realize: bool = True,
) -> TensorBox:
    """
    Returns "inverse_offsets" - the inverse of the offsets array.
    offsets maps batch index (dense) to jagged index (i.e. offset into jagged tensor).
    inverse_offsets maps jagged index to batch index.

    e.g. for offsets [0, 3, 4, 9, 10] this will return
    inverse_offsets = [0, 0, 0, 1, 2, 2, 2, 2, 2, 3]

    For the given offsets, the computed inverse_offsets are cached
    on the first call and reused in the further calls.
    """

    if hasattr(offsets, "inverse_offsets"):
        # inverse_offsets are already computed
        # for these offsets: can reuse
        return offsets.inverse_offsets

    # ops.bucketize takes offsets.get_name() which doesn't exist on Pointwise
    # kernels, i.e. we need to realize it before using. In other words, we need
    # offsets to be in global memory so that we can binary search over the
    # entire tensor
    offsets.realize()
    device: torch.device = offsets.get_device_or_error()
    dtype: torch.dtype = offsets.get_dtype()

    # pyre-ignore[2,3]
    def inner_fn(index):
        idx = index[0]
        bucket = ops.bucketize(
            values=ops.index_expr(idx, dtype),
            boundaries=(
                offsets.get_name(),
                offsets.get_size()[-1],
                offsets.get_size()[0] * offsets.get_stride()[0],
                offsets.get_stride()[-1],
            ),
            boundary_indices=0,
            indexing_dtype=dtype,
            right=True,
        )
        # ops.bucketize above returns 1-based bucket indices,
        # but we need 0-based, hence we subtract 1 from batch
        return bucket - 1

    inverse_offsets = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[jagged_len],
    )

    if realize:
        # "freeze" the node so that it doesn't get inlined downstream.
        inverse_offsets.realize()

    # cache inverse_offsets for further reuse
    offsets.inverse_offsets = inverse_offsets  # type: ignore[attr-defined]

    return inverse_offsets


def jagged_idx_to_dense_idx(
    jagged_idx,  # pyre-ignore[2]
    inverse_offsets_loader,  # pyre-ignore[2]
    offsets_loader,  # pyre-ignore[2]
    batch_size: int | sympy.Expr,
    max_seq_len: int | sympy.Expr,
    offsets_dtype: torch.dtype,
) -> tuple[sympy.Expr, sympy.Expr]:
    batch_idx = ops.indirect_indexing(
        inverse_offsets_loader([jagged_idx]),
        batch_size + 1,
    )
    batch_start = offsets_loader([batch_idx])
    seq = ops.index_expr(jagged_idx, offsets_dtype) - batch_start
    # check=False because there may be sequences longer than max_seq_len
    seq_idx = ops.indirect_indexing(seq, max_seq_len, check=False)
    return batch_idx, seq_idx


def register_jagged_ops():
    # Avoid circular import by importing here
    from .lowering import fallback_handler, is_integer_type, register_lowering

    # pyre-ignore[56]
    @register_lowering(torch.ops.aten._jagged_to_padded_dense_forward.default)
    def _jagged_to_padded_dense_forward(
        jagged_values: TensorBox,
        jagged_offsets: list[TensorBox],
        max_lengths: list[int],  # list of ints/SymInts
        padding_value: float = 0.0,
    ) -> TensorBox:
        device = jagged_values.get_device_or_error()
        dtype = jagged_values.get_dtype()

        jagged_values_size = jagged_values.get_size()

        # only handle the common case of a single jagged dimension
        if (
            len(jagged_offsets) != 1
            or device.type != "cuda"
            or device != jagged_offsets[0].get_device()
            or len(jagged_values_size) != 2
            or len(jagged_offsets[0].get_size()) != 1
            or len(max_lengths) != len(jagged_offsets)
            or not is_integer_type(jagged_offsets[0])
        ):
            return fallback_handler(
                torch.ops.aten._jagged_to_padded_dense_forward.default,
                add_to_fallback_set=False,
            )(
                jagged_values,
                jagged_offsets,
                max_lengths,
                padding_value,
            )

        offsets: TensorBox = jagged_offsets[0]  # type: ignore[assignment]
        offsets_len = offsets.get_size()[0]
        offsets_dtype = offsets.get_dtype()
        batch_size = offsets_len - 1
        max_seq_len = max_lengths[0]
        embedding_len = jagged_values_size[1]
        jagged_len = jagged_values_size[0]

        output_size = [batch_size, max_seq_len, embedding_len]

        values_loader = jagged_values.make_loader()
        offsets_loader = offsets.make_loader()

        # pyre-ignore[2,3,53]
        def inner_fn(index):
            # dense tensor size: [B, N, D]
            batch_idx, seq_idx, emb_idx = index
            jagged_idx, end_idx = dense_idx_to_jagged_idx(
                batch_idx=batch_idx,
                seq_idx=seq_idx,
                offsets_loader=offsets_loader,
                jagged_len=jagged_len,
            )
            return ops.masked(
                ops.lt(
                    ops.index_expr(jagged_idx, offsets_dtype),
                    end_idx,
                ),
                lambda: values_loader([jagged_idx, emb_idx]),
                padding_value,
            )

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=output_size,
        )

    def _dense_to_jagged_forward_impl(
        fallback_op,  # pyre-ignore[2]
        dense: TensorBox,
        jagged_offsets: list[TensorBox],
        jagged_len: int | None = None,
    ) -> TensorBox:
        device = dense.get_device_or_error()
        dtype = dense.get_dtype()

        dense_size = dense.get_size()

        # only handle the common case of a single jagged dimension
        if (
            len(jagged_offsets) != 1
            or device.type != "cuda"
            or device != jagged_offsets[0].get_device()
            or len(jagged_offsets[0].get_size()) != 1
            or len(dense_size) != 3
            or jagged_len is None
            or not is_integer_type(jagged_offsets[0])
        ):
            return fallback_handler(fallback_op, add_to_fallback_set=False)(
                dense,
                jagged_offsets,
                jagged_len,
            )

        offsets: TensorBox = jagged_offsets[0]  # type: ignore[assignment]
        offsets_dtype = offsets.get_dtype()
        batch_size = dense_size[0]
        max_seq_len = dense_size[1]
        embedding_len = dense_size[-1]

        output_size = [jagged_len, embedding_len]

        dense_loader = dense.make_loader()
        offsets_loader = offsets.make_loader()

        inverse_offsets = get_inverse_offsets(
            offsets=offsets,
            jagged_len=jagged_len,
        )
        inverse_offsets_loader = inverse_offsets.make_loader()

        # pyre-ignore[2,3,53]
        def inner_fn(index):
            # jagged tensor size: [sum_B(N_B), D]
            jagged_idx, emb_idx = index
            batch_idx, seq_idx = jagged_idx_to_dense_idx(
                jagged_idx=jagged_idx,
                offsets_loader=offsets_loader,
                inverse_offsets_loader=inverse_offsets_loader,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                offsets_dtype=offsets_dtype,
            )
            return ops.masked(
                ops.lt(
                    ops.index_expr(seq_idx, offsets_dtype),
                    ops.index_expr(max_seq_len, offsets_dtype),
                ),
                lambda: dense_loader([batch_idx, seq_idx, emb_idx]),
                0.0,  # jagged sequence longer than max_seq_len
            )

        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=output_size,
        )

    # pyre-ignore[56]
    @register_lowering(torch.ops.aten._padded_dense_to_jagged_forward)
    def _dense_to_jagged_forward(
        dense: TensorBox,
        jagged_offsets: list[TensorBox],
        jagged_len: int | None = None,
    ) -> TensorBox:
        return _dense_to_jagged_forward_impl(
            fallback_op=torch.ops.aten._padded_dense_to_jagged_forward.default,
            dense=dense,
            jagged_offsets=jagged_offsets,
            jagged_len=jagged_len,
        )
<|endoftext|>from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import torch
import torch._inductor.config
from torch._inductor import ir
from torch._inductor.virtualized import V

from .ir import FixedLayout, FlexibleLayout, Layout


if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy


class KernelInputs(ABC):
    """
    Class to store and provide access to input nodes for kernels.
    This class takes in a tuple of input nodes and provides methods to access
    information about these nodes, such as their device type and device.
    """

    def __init__(
        self,
        input_nodes: list[Any],
        scalars: dict[str, float | int] | None = None,
        out_dtype: torch.dtype | None = None,
    ):
        """
        Initialize with a tuple of input nodes.

        Args:
            input_nodes: A tuple of input nodes to store
            out_dtype: Optional output dtype to store
        """
        self._input_nodes = input_nodes
        self._device_name: str | None = None
        self._scalars = scalars if scalars is not None else {}
        self._out_dtype = out_dtype
        if len(input_nodes) <= 0:
            raise AssertionError("Expected at least one input node")

    def nodes(self, reorder: Sequence[int] | None = None) -> list[Any]:
        """
        Return the stored input nodes, optionally reordered.

        Args:
            reorder: Optional sequence of indices to reorder the nodes.
                    For example, (2, 0, 1) would return nodes in that order.

        Returns:
            The tuple of input nodes, optionally reordered
        """
        if reorder is None:
            return self._input_nodes
        if len(self._input_nodes) != len(reorder):
            raise AssertionError(
                f"Reorder length mismatch: {len(self._input_nodes)} vs {len(reorder)}"
            )
        return [self._input_nodes[i] for i in reorder]

    @property
    def count(self) -> int:
        """
        Get the number of input nodes.

        Returns:
            The number of input nodes
        """
        return len(self._input_nodes)

    @property
    def device_type(self) -> str | None:
        """
        Get the device type of the first node.

        Returns:
            The device type (e.g., 'cuda', 'cpu')
        """

        return ir.get_device_type(self._input_nodes[0])

    def device(self) -> torch.device:
        """
        Get the device of the first node.

        Returns:
            The device of the first node
        """
        return self._input_nodes[0].get_device()

    def device_name(self) -> str | None:
        """
        Get the device name information.

        Returns:
            A tuple of (gpu_name, vendor, model)
        """
        if self._device_name is None:
            device = self.device()
            if self.device_type == "cuda":
                device_properties = torch.cuda.get_device_properties(device)
                self._device_name = device_properties.gcnArchName
        return self._device_name

    def shapes_symbolic(self) -> tuple[tuple[Any, ...], ...]:
        """
        Get the symbolic shapes of all input nodes.

        Returns:
            A tuple of shape tuples for each input node
        """
        return tuple(node.get_size() for node in self._input_nodes)

    def shapes_hinted(self) -> tuple[tuple[int, ...], ...]:
        """
        Get the size hints for shapes of all input nodes.

        Returns:
            A tuple of shape tuples with integer hints for each input node
        """
        return tuple(
            V.graph.sizevars.optimization_hints(node.get_size())
            for node in self._input_nodes
        )

    def strides_symbolic(self) -> tuple[tuple[sympy.Integer, ...], ...]:
        """
        Get the symbolic strides of all input nodes.

        Returns:
            A tuple of stride tuples for each input node
        """
        return tuple(node.get_stride() for node in self._input_nodes)

    def strides_hinted(self) -> tuple[tuple[int, ...], ...]:
        """
        Get the size hints for strides of all input nodes.

        Returns:
            A tuple of stride tuples with integer hints for each input node
        """
        return tuple(
            V.graph.sizevars.optimization_hints(node.get_stride())
            for node in self._input_nodes
        )

    def dtypes(self) -> tuple[torch.dtype, ...]:
        """
        Get the dtypes of all input nodes.

        Returns:
            A tuple of dtypes for each input node
        """
        return tuple(node.get_dtype() for node in self._input_nodes)

    def dtype(self, idx: int = 0) -> torch.dtype:
        """
        Get the dtype of a specific input node.

        Args:
            idx: Index of the node to get the dtype from (default: 0)

        Returns:
            The dtype of the specified input node
        """
        return self._input_nodes[idx].get_dtype()

    @abstractmethod
    def out_dtype(self) -> torch.dtype:
        """
        Get the output dtype, whether passed in or inferred from the nodes

        Returns:
            The output dtype
        """

    def get_scalar(self, name: str) -> float | int:
        """
        Get the scalar value for a given name.

        Args:
            name: Name of the scalar to get

        Returns:
            The scalar value
        """
        if name not in self._scalars:
            raise AssertionError(f"Scalar {name} not found, but required")
        return self._scalars[name]

    @abstractmethod
    def output_layout(self, flexible: bool = True) -> Layout:
        """
        Abstract method to handle output layout generation.

        Args:
            out_dtype: Optional output dtype. If not provided, infer from inputs
            flexible: If True, return FlexibleLayout, otherwise FixedLayout
        """


class MMKernelInputs(KernelInputs):
    """
    Specialized KernelInputs for matrix multiplication operations.
    Provides additional methods to access M, N, K dimensions.
    """

    def __init__(
        self,
        input_nodes: list[Any],
        scalars: dict[str, float | int] | None = None,
        out_dtype: torch.dtype | None = None,
        mat1_idx: int = -2,
        mat2_idx: int = -1,
    ):
        """
        Initialize with a tuple of input nodes.

        By default, we assume the last 2 input nodes are mat1 and mat2, but
        the caller can adjust when necessary
        """
        super().__init__(input_nodes, scalars, out_dtype)
        # for mm, we need at least 2 nodes, and we need to know which nodes
        # are the main matrices e.g. addmm is (bias, mat1, mat2) whereas others
        # might be (mat1, mat2, scale), etc.
        if len(self._input_nodes) < 2:
            raise AssertionError("Expected at least 2 input nodes")

        # Adjust assertions to handle negative indices
        m1_idx, m2_idx = mat1_idx, mat2_idx
        if mat1_idx < 0:
            m1_idx += len(input_nodes)
        if mat2_idx < 0:
            m2_idx += len(input_nodes)

        if not (0 <= m1_idx < len(input_nodes)):
            raise AssertionError(f"Invalid mat1_idx: {mat1_idx}")
        if not (0 <= m2_idx < len(input_nodes)):
            raise AssertionError(f"Invalid mat2_idx: {mat2_idx}")

        self._mat1_idx = mat1_idx
        self._mat2_idx = mat2_idx

    def mnk_symbolic(
        self,
    ) -> tuple[sympy.Integer, sympy.Integer, sympy.Integer]:
        """
        Get the symbolic M, N, K dimensions for matrix multiplication.
        Handles both 2D (MM) and 3D (BMM) tensors.

        M is extracted from the second-to-last dimension of the first operand (mat1).
        N is extracted from the last dimension of the second operand (mat2).
        K is extracted from the last dimension of the first operand (mat1).

        Returns:
            A tuple of (M, N, K) dimensions
        """
        mat1 = self.nodes()[self._mat1_idx]
        mat2 = self.nodes()[self._mat2_idx]

        m = mat1.get_size()[-2]  # M from second-to-last dimension of mat1
        k = mat1.get_size()[-1]  # K from last dimension of mat1
        n = mat2.get_size()[-1]  # N from last dimension of mat2

        # Ensure K dimensions match between operands
        k0 = mat2.get_size()[-2]  # K from second-to-last dimension of mat2
        V.graph.sizevars.check_equals(k, k0)
        return (m, n, k)

    def out_dtype(self) -> torch.dtype:
        """
        Get the output dtype, whether passed in or inferred from the nodes

        Returns:
            The output dtype
        """
        if self._out_dtype is not None:
            return self._out_dtype
        return self.mat1mat2()[0].get_dtype()

    def output_layout(self, flexible: bool = True) -> Layout:
        """
        Handle output layout generation for matrix multiplication.

        Args:
            out_dtype: Optional output dtype. If not provided, infer from inputs
            flexible: If True, return FlexibleLayout, otherwise FixedLayout
        """
        mat1, mat2 = self.mat1mat2()
        out_dtype = self.out_dtype()
        # NOTE: taken from mm_common.mm_args
        *b1, m, k1 = mat1.get_size()
        *b2, k2, n = mat2.get_size()
        b = [V.graph.sizevars.check_equals_and_simplify(a, b) for a, b in zip(b1, b2)]
        size = [*b, m, n]
        if flexible:
            return FlexibleLayout(self.device(), out_dtype, size)
        else:
            return FixedLayout(self.device(), out_dtype, size)

    def mat1mat2(self) -> tuple[Any, Any]:
        """
        Get the mat1 and mat2 nodes.

        Returns:
            A tuple of (mat1, mat2) nodes
        """
        nodes = self.nodes()
        return nodes[self._mat1_idx], nodes[self._mat2_idx]

    def mnk_hinted(self) -> tuple[int, int, int]:
        """
        Get the hinted M, N, K dimensions for matrix multiplication.
        Handles both 2D (MM) and 3D (BMM) tensors.

        Uses shapes_hinted from the base class to get integer hints for dimensions.

        Returns:
            A tuple of (M, N, K) dimensions as integers
        """
        hinted_shapes = self.shapes_hinted()
        mat1_shape = hinted_shapes[self._mat1_idx]
        mat2_shape = hinted_shapes[self._mat2_idx]

        m = mat1_shape[-2]  # M from second-to-last dimension of mat1
        k = mat1_shape[-1]  # K from last dimension of mat1
        n = mat2_shape[-1]  # N from last dimension of mat2

        # Ensure K dimensions match between operands
        k_check = mat2_shape[-2]  # K from second-to-last dimension of mat2
        if k != k_check:
            raise AssertionError(f"K dimensions don't match: {k} vs {k_check}")

        return (m, n, k)

    def batch_hinted(self) -> int:
        """
        Get the hinted batch size for batched matrix multiplication.
        Returns 1 for non-batched (2D) operations.

        Returns:
            The batch size as an integer
        """
        hinted_shapes = self.shapes_hinted()
        mat1_shape = hinted_shapes[self._mat1_idx]

        if len(mat1_shape) >= 3:
            return mat1_shape[-3]  # Batch from third-to-last dimension
        return 1  # Non-batched operation
<|endoftext|>"""Rewrite raw ``c10d.{op}_`` inplace ops produced by ``make_fx`` into
``_c10d_functional.{op}`` + ``wait_tensor`` + ``aten.copy_`` form so
``standalone_compile`` (and Inductor's collective machinery underneath) can
recognize and lower them.

The structure separates concerns:
  * Arg resolution (PG, ReduceOp) lives in ``_resolve_*``.
  * The functional + wait + copy_ chain construction lives in
    ``_emit_collective_chain`` (used by every per-op rewrite).
  * Use redirection lives in ``_redirect_*_work_uses`` keyed by output schema.
  * Per-op rewrites (``_rewrite_<op>_``) glue these together.
  * The driver ``_functionalize_inplace_collectives`` walks the graph,
    dispatches by target, and runs final cleanup.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.graph_module import _assign_attr, _del_attr, _get_attr


def _resolve_reduce_op_str(
    gm: torch.fx.GraphModule, arg: torch.fx.node.Argument
) -> str:
    """Get the lower-case op string (``"sum"``/``"avg"``/...) for a c10d
    ReduceOp ``get_attr`` arg, converting from the torchbind ScriptObject
    form if needed.

    Only the ``get_attr`` (constant-baked) shape is supported today. Any
    other producer (placeholder, call_function, ...) would require
    ``_c10d_functional.*`` to accept ReduceOp directly (today it only
    takes the string form); deferred to a follow-up PR.
    """
    import torch.distributed as dist
    from torch.distributed._functional_collectives import REDUCE_OP_TO_STR

    if not isinstance(arg, torch.fx.Node) or arg.op != "get_attr":
        raise NotImplementedError(
            f"ReduceOp arg must be a constant ``get_attr``; got {arg!r}."
        )
    reduce_op = _get_attr(gm, arg.target)  # type: ignore[arg-type]
    if isinstance(reduce_op, torch.ScriptObject):
        reduce_op = dist.ReduceOp.RedOpType(reduce_op.op())  # type: ignore[attr-defined]
    return REDUCE_OP_TO_STR[reduce_op]


def _emit_collective_chain(
    gm: torch.fx.GraphModule,
    before: torch.fx.Node,
    input_t: torch.fx.Node,
    output_t: torch.fx.Node,
    functional_target: Any,
    extra_args: tuple[Any, ...],
    pg_arg: torch.fx.Node,
    custom: Any,
) -> torch.fx.Node:
    """Insert ``_c10d_functional.<op>(input, *extra_args, pg_arg)`` ->
    ``wait_tensor`` -> ``aten.copy_(output, wait)`` immediately before
    ``before`` and return the ``wait_tensor`` node.

    Propagates ``output_t.meta["val"]`` onto every new node — every functional
    collective we rewrite to produces a tensor with the same shape/dtype as
    ``output_t``, and ``aten.copy_`` returns ``output_t``. Also carries the
    rewritten node's ``custom`` annotation forward so region scooping continues
    to work.
    """
    val = output_t.meta.get("val")
    with gm.graph.inserting_before(before):
        ar = gm.graph.call_function(functional_target, (input_t, *extra_args, pg_arg))
        wait = gm.graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default, (ar,)
        )
        copy_ = gm.graph.call_function(torch.ops.aten.copy_.default, (output_t, wait))
        if val is not None:
            ar.meta["val"] = wait.meta["val"] = copy_.meta["val"] = val
        if custom is not None:
            ar.meta["custom"] = wait.meta["custom"] = copy_.meta["custom"] = custom
    return wait


def _redirect_inplace_collective_uses(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    wait_nodes: list[torch.fx.Node],
) -> None:
    """Re-route uses for an inplace c10d op with output schema
    ``(Tensor[], Work)`` — e.g. ``allreduce_``, ``broadcast_``,
    ``reduce_scatter_``, ``alltoall_``.

    ``make_fx`` lowers tensor consumers into ``getitem(node, 0)[i]``; replace
    each with ``wait_nodes[i]``. The work-handle ``getitem(node, 1)`` is erased
    if unused (live users raise — functional collectives have no Work object
    to forward to, so we can't preserve that synchronization).

    TODO: extend to ``(Tensor, Work)`` schemas (``_allgather_base_``,
    ``_reduce_scatter_base_``) when those rewrites are added. There the
    ``getitem(node, 0)`` is itself the tensor consumer (no inner
    ``getitem`` chain), so the inner loop below would need to fall back to
    ``use.replace_all_uses_with(wait_nodes[0])`` when ``use`` has no inner
    ``getitem`` users.
    """
    for use in list(node.users):
        if use.op != "call_function" or use.target is not operator.getitem:
            raise AssertionError(
                f"Cannot functionalize {node.target}: unexpected use {use} "
                f"(op={use.op!r}, target={use.target!r}). The c10d inplace "
                f"output is a tuple consumed via ``getitem`` in ``make_fx`` "
                f"graphs; non-getitem users are not supported by this pass."
            )
        if use.args[1] == 0:
            for sub_use in list(use.users):
                if (
                    sub_use.op != "call_function"
                    or sub_use.target is not operator.getitem
                ):
                    raise AssertionError(
                        f"Cannot functionalize {node.target}: tensor list "
                        f"consumed by non-getitem user {sub_use}. Extend "
                        f"_redirect_inplace_collective_uses to support this "
                        f"schema if needed."
                    )
                sub_use.replace_all_uses_with(wait_nodes[sub_use.args[1]])  # type: ignore[index]
                gm.graph.erase_node(sub_use)
            gm.graph.erase_node(use)
        elif use.args[1] == 1:
            if use.users:
                raise AssertionError(
                    f"Cannot functionalize {node.target}: Work handle has "
                    f"{len(use.users)} live use(s) ({list(use.users)}). "
                    f"Functional collectives express synchronization via "
                    f"``wait_tensor`` on the data tensor, so the Work object "
                    f"has no equivalent to forward to."
                )
            gm.graph.erase_node(use)


def _rewrite_allreduce_(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Reference rewrite — use this as a template for new collectives.

    Schema: ``c10d::allreduce_(Tensor[] tensors, ProcessGroup pg, ReduceOp op,
    Tensor? sparse_indices, bool async_op, int timeout=-1) -> (Tensor[], Work)``.

    Before::

        allreduce_ = c10d.allreduce_([t0, t1], pg, reduce_op, None, False)
        getitem    = allreduce_[0]
        getitem_1  = getitem[0]      # consumer of t0 post-allreduce
        getitem_2  = getitem[1]      # consumer of t1 post-allreduce
        ...        = allreduce_[1]   # work handle (unused)

    After (per input tensor; ``pg`` is the original ``get_attr`` Node — a
    later pass will bake it into a ``group_name`` string)::

        ar_i = _c10d_functional.all_reduce(t_i, op_str, pg)
        wait_i = _c10d_functional.wait_tensor(ar_i)
        _ = aten.copy_(t_i, wait_i)  # preserves inplace semantics
        # downstream uses of ``getitem_<i>`` are redirected to ``wait_i``
    """
    tensors = node.args[0]
    pg_arg = node.args[1]
    if not isinstance(tensors, (list, tuple)):
        raise AssertionError(f"expected ``Tensor[]`` arg list, got {type(tensors)}")
    if not isinstance(pg_arg, torch.fx.Node):
        raise AssertionError(f"expected ProcessGroup as fx.Node, got {type(pg_arg)}")
    op_str = _resolve_reduce_op_str(gm, node.args[2])
    custom = node.meta.get("custom")
    waits = []
    for t in tensors:
        if not isinstance(t, torch.fx.Node):
            raise AssertionError(f"expected tensor entry as fx.Node, got {type(t)}")
        waits.append(
            _emit_collective_chain(
                gm,
                node,
                t,
                t,
                torch.ops._c10d_functional.all_reduce.default,
                (op_str,),
                pg_arg,
                custom,
            )
        )
    _redirect_inplace_collective_uses(gm, node, waits)


_InplaceCollectiveRewrite = Callable[[torch.fx.GraphModule, torch.fx.Node], None]


def _inplace_c10d_rewrites() -> dict[Any, _InplaceCollectiveRewrite] | None:
    """Map inplace ``c10d.{op}_`` op targets to per-op rewrite functions.

    Returns ``None`` if ``torch.distributed`` is not available.

    To support a new collective: write ``_rewrite_<op>_`` (typically a few
    lines using ``_resolve_reduce_op_str`` / ``_emit_collective_chain`` /
    ``_redirect_inplace_collective_uses``), then register it here.
    """
    if not torch.distributed.is_available():
        return None
    return {
        torch.ops.c10d.allreduce_.default: _rewrite_allreduce_,
    }


def _functionalize_inplace_collectives(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Rewrite raw ``torch.ops.c10d.{op}_`` inplace calls in ``gm`` into
    ``_c10d_functional.{op}`` + ``wait_tensor`` + ``aten.copy_`` form so
    Inductor's collective machinery can recognize and lower them.
    """
    rewrites = _inplace_c10d_rewrites()
    if rewrites is None:
        return gm

    found = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        rewrite = rewrites.get(node.target, None)
        if rewrite is None:
            continue
        rewrite(gm, node)
        # Snapshot ``get_attr`` args before erasing the rewritten node.
        # Inplace c10d ops are impure; ``eliminate_dead_code`` keeps them
        # alive even with no users, so erase explicitly. Then drop any
        # ``get_attr`` arg that became orphan (e.g. the ReduceOp whose
        # value got baked into ``op_str``) along with its backing module
        # attribute. Live ones (e.g. the ProcessGroup, still referenced
        # by the new ``_c10d_functional`` call) are kept.
        get_attr_args = [
            a for a in node.args if isinstance(a, torch.fx.Node) and a.op == "get_attr"
        ]
        gm.graph.erase_node(node)
        for a in get_attr_args:
            if not a.users:
                _del_attr(gm, a.target)  # type: ignore[arg-type]
                gm.graph.erase_node(a)
        found = True

    if found:
        gm.recompile()
    return gm


def _unbox_process_group_torchbinds(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Replace ProcessGroup torchbind attrs with their unboxed Python
    ``dist.ProcessGroup``.

    ``make_fx`` traces ``dist.*`` collectives with a torchbind
    ``__torch__.torch.classes.c10d.ProcessGroup`` baked in as a graph attr.
    Inductor's collective ops (``_c10d_functional.{op}``) accept either a
    ``group_name`` string or a Python ``dist.ProcessGroup`` — but **not** a
    raw torchbind, which fails at runtime with::

        all_reduce_(): argument 'group_name' must be either a string ...
        but got __torch__.torch.classes.c10d.ProcessGroup

    Unboxing the attribute in-place (without changing the FX graph) makes
    the PG acceptable to the runtime collective op. The deepcopy
    share-by-reference hook in ``GraphModule.__deepcopy__`` is what lets
    ``standalone_compile``'s deepcopy of the gm survive the non-pickleable
    Python ``dist.ProcessGroup``.
    """
    import torch.distributed as dist

    if not torch.distributed.is_available():
        return gm

    for node in gm.graph.find_nodes(op="get_attr"):
        attr = _get_attr(gm, node.target)  # type: ignore[arg-type]
        if not isinstance(attr, torch.ScriptObject):
            continue
        if attr._type().qualified_name() != "__torch__.torch.classes.c10d.ProcessGroup":  # type: ignore[attr-defined]
            continue
        _assign_attr(dist.ProcessGroup.unbox(attr), gm, node.target)  # type: ignore[arg-type]
    return gm
<|endoftext|>import abc
import dataclasses
import itertools
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar
from typing_extensions import Self
from unittest.mock import patch

import sympy

import torch
from torch._inductor.utils import get_free_symbols
from torch.fx.experimental.symbolic_shapes import free_symbols, free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet

from ..utils._sympy.symbol import make_symbol, SymT
from .codegen.common import index_prevent_reordering
from .ops_handler import DefaultHandler
from .utils import (
    get_dtype_size,
    reduction_num_outputs,
    sympy_index_symbol,
    sympy_subs,
    VarRanges,
)
from .virtualized import ReductionType, V


T = TypeVar("T")

log = logging.getLogger(__name__)
is_indirect = re.compile(r"indirect|tmp").search


class Dep(abc.ABC):
    name: str
    index: sympy.Expr

    @abc.abstractmethod
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        pass

    @abc.abstractmethod
    def rename(self, renames: dict[str, str]) -> Self:
        pass

    @abc.abstractmethod
    def get_numel(self) -> sympy.Expr:
        pass

    @abc.abstractmethod
    def numbytes_hint(self) -> int:
        pass

    @abc.abstractmethod
    def numel_hint(self) -> int:
        pass

    @abc.abstractmethod
    def has_unbacked_symbols(self) -> bool:
        pass

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        pass

    def normalize_with_stride_order(self, prefix: str = "t") -> Self:
        return self


@dataclasses.dataclass(frozen=True)
class MemoryDep(Dep):
    # pyrefly: ignore [bad-override]
    name: str
    # pyrefly: ignore [bad-override]
    index: sympy.Expr
    var_names: tuple[sympy.Symbol, ...]
    size: tuple[sympy.Expr, ...]
    mode: str | None = None

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return (
            get_free_symbols(self.index, unbacked_only)
            | get_free_symbols(self.size, unbacked_only)
            | get_free_symbols(self.var_names, unbacked_only)
        )

    def __repr__(self) -> str:
        maybe_mode = ""
        if self.mode is not None:
            maybe_mode = f", {self.mode}"
        return f"MemoryDep({self.name!r}, {self.index}, {self.ranges}{maybe_mode})"

    @property
    def num_vars(self) -> int:
        return len(self.var_names)

    def decide_loop_order_to_match(self, other: "MemoryDep") -> list[int] | None:
        """
        Can return None if not able to decide loop orders.
        """
        if self.num_vars != other.num_vars:
            raise AssertionError(
                f"expected num_vars to match, got {self.num_vars} and {other.num_vars}"
            )

        # ignore broadcast for now since broadcast causes extra 0 strides
        # which makes it hard to decide the correct loop orders.
        if self.num_vars != len(self.index.free_symbols):
            return None
        if other.num_vars != len(other.index.free_symbols):
            return None

        # bail out if any size is 0 or 1
        # For size == 0, it's an empty tensor, any strides for that dimension
        # are equivalent. Skip for simplicity and it may not matter that much.
        #
        # For size == 1, it cause tie for strides of different dimensions.
        # Also when we first time create LoopBody in ComputedBuffer.simplify_and_reorder
        # we can dependencies.index_vars_squeeze which should already squeeze
        # the size == 1 dimensions.
        if any(s == 0 or s == 1 for s in itertools.chain(self.size, other.size)):
            return None

        # Extract strides for both expression
        self_strides = V.graph.sizevars.stride_hints(self.index, self.var_names)
        other_strides = V.graph.sizevars.stride_hints(other.index, other.var_names)

        # Even if the shape contains no 0/1, some complex index expression may
        # still have duplicate stride values. Here is an example:
        # https://gist.github.com/shunting314/511a7e1ec88aa2e1a8ec85d8445ab129
        # We don't reorder the loop for these cases for now, but in theory
        # we could improve the algorithm to detect the correct loop orders.
        if len(OrderedSet(self_strides)) != len(self_strides) or len(
            OrderedSet(other_strides)
        ) != len(other_strides):
            log.debug(
                "unable to decide loop order. self_dep=%s v.s. other_dep=%s, self_strides=%s v.s. other_strides=%s",
                self,
                other,
                self_strides,
                other_strides,
            )
            return None

        # May happen if self and other are as follows
        # MemoryDep('addmm_6', 393216*d0 + 768*d1 + d2, {d0: 16, d1: 512, d2: 768}, None)
        # MemoryDep('addmm_6', 98304*d0 + d1 + 768*d2, {d0: 64, d1: 768, d2: 128}, None)
        if OrderedSet(self_strides) != OrderedSet(other_strides):
            return None

        stride_to_index = {s: i for i, s in enumerate(self_strides)}
        order = [stride_to_index[s] for s in other_strides]

        if OrderedSet(order) != OrderedSet(range(self.num_vars)):
            raise AssertionError(
                f"expected order to be a permutation of range({self.num_vars}), got {order}"
            )
        return order

    def get_offset(self) -> sympy.Expr:
        """
        Return the offset by setting every variable to be 0.
        """
        return sympy_subs(self.index, dict.fromkeys(self.var_names, 0))

    def normalize(self) -> "MemoryDep":
        """
        Normalize by merging loops. The difference from normalize_with_stride_order is,
        this method does not reorder loops while normalize_with_stride_order reorders
        loops based on stride order.
        """
        return MemoryDep(
            self.name,
            *_RecordLoadStoreInner._normalize(self.index, self.ranges),  # type: ignore[arg-type]
            self.mode,
        )

    def normalize_with_stride_order(self, prefix: str = "t") -> "MemoryDep":
        r"""
        Used to decide if two MemoryDep does not equal due to different loop orders.
        More specifically, when dep1 and dep2 are not equal, we can normalize
        both and check if they are equal after that. If yes, then the mismatch is
        caused by different loop orders.
        """
        # import here to avoid circular import
        from torch._inductor import ir

        strides = V.graph.sizevars.stride_hints(self.index, self.var_names)

        # pick a loop order with stride ordered decreasingly
        order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        stride_reorder = ir.same_reorder(order)
        sizes = self.size
        var_names = self.var_names

        new_reordered_sizes = stride_reorder(sizes)
        new_reordered_var_names = stride_reorder(var_names)

        new_simplified_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            new_reordered_var_names,
            new_reordered_sizes,
            index_prevent_reordering(
                [self.index], new_reordered_var_names, new_reordered_sizes
            ),
        )

        # now let's create new symbols with the passed in prefix
        var_ranges, add_var = var_builder(prefix)
        replacement = dict(
            zip(
                new_reordered_var_names,
                reindex([add_var(x) for x in new_simplified_sizes]),
            )
        )
        new_index = sympy_subs(sympy.expand(self.index), replacement)  # type: ignore[arg-type] # next PR

        out = MemoryDep(
            self.name, new_index, tuple(var_ranges.keys()), tuple(var_ranges.values())
        )  # type: ignore[arg-type]
        return out

    @property
    def ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
        """{c0: 128, c1: 512, ...}"""
        return dict(zip(self.var_names, self.size))

    def simplify_with_ranges(self) -> "MemoryDep":
        return MemoryDep(
            name=self.name,
            index=V.graph.sizevars.simplify_with_ranges(self.index, self.ranges),
            var_names=self.var_names,
            size=self.size,
            mode=self.mode,
        )

    def get_numel(self) -> sympy.Expr:
        if self.is_indirect():
            numel = V.graph.get_numel(self.name)
        else:
            vars: OrderedSet[sympy.Basic] = OrderedSet(self.index.free_symbols)
            numel = sympy.S.One
            for var, size in zip(self.var_names, self.size):
                if var in vars:
                    numel = numel * size
        return numel  # type: ignore[return-value]

    def rename(self, renames: dict[str, str]) -> "MemoryDep":
        if self.name in renames:
            return MemoryDep(
                renames[self.name],
                self.index,
                var_names=self.var_names,
                size=self.size,
                mode=self.mode,
            )
        return self

    def numbytes_hint(self) -> int:
        try:
            return V.graph.sizevars.optimization_hint(
                self.get_numel(), fallback=0
            ) * get_dtype_size(V.graph.get_dtype(self.name))
        except NotImplementedError:  # NoneLayout
            return 0

    def numel_hint(self) -> int:
        try:
            return V.graph.sizevars.optimization_hint(self.get_numel(), fallback=0)
        except NotImplementedError:  # NoneLayout
            return 0

    def has_unbacked_symbols(self) -> bool:
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        if isinstance(self.index, sympy.Integer):
            return True
        return isinstance(self.index, sympy.Symbol) and self.index in self.var_names

    def stride1_for_last_dim(self, result_for_complex_expression: bool = True) -> bool:
        """
        Whether the stride for the last dimension is 1.
        """
        # python test/inductor/test_torchinductor_opinfo.py -k test_comprehensive_masked_scatter_cuda_float16
        # will exercise thru this corner case.
        if len(self.var_names) == 0:
            return True

        terms = self.index.args if isinstance(self.index, sympy.Add) else [self.index]

        last_sym = self.var_names[-1]
        for term in terms:
            if term == last_sym:
                return True

            # Having a >1 stride for the last dimension is bad for perf
            # return False.
            if (
                isinstance(term, sympy.Mul)
                and len(term.args) == 2
                and term.args[1] == last_sym
                and isinstance(term.args[0], (int, sympy.Integer))
                and term.args[0] > 1
            ):
                return False

        return result_for_complex_expression

    def is_scalar(self) -> bool:
        if isinstance(self.index, sympy.Symbol):
            return self.index not in self.var_names and not self.is_indirect()
        return isinstance(self.index, (int, sympy.Integer))

    def is_indirect(self) -> bool:
        return any(is_indirect(v.name) for v in self.index.free_symbols)  # type: ignore[attr-defined]


@dataclasses.dataclass(frozen=True)
class StarDep(Dep):
    # pyrefly: ignore [bad-override]
    name: str
    mode: str | None = None

    # depends on the entire buffer
    @property
    # pyrefly: ignore [bad-override]
    def index(self) -> sympy.Expr:
        raise NotImplementedError("StarDep does not have an index")

    def get_numel(self) -> sympy.Expr:
        return V.graph.get_numel(self.name)  # type: ignore[return-value]

    def rename(self, renames: dict[str, str]) -> "StarDep":
        if self.name in renames:
            return StarDep(renames[self.name], self.mode)
        return self

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def numbytes_hint(self) -> int:
        try:
            return V.graph.sizevars.optimization_hint(
                self.get_numel(), fallback=0
            ) * get_dtype_size(V.graph.get_dtype(self.name))
        except NotImplementedError:
            return 0  # NoneLayout, MultiOutputLayout, etc

    def numel_hint(self) -> int:
        try:
            return V.graph.sizevars.optimization_hint(self.get_numel(), fallback=0)
        except NotImplementedError:
            return 0  # NoneLayout, MultiOutputLayout, etc

    def has_unbacked_symbols(self) -> bool:
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        return False

    def is_scalar(self) -> bool:
        return False

    def is_indirect(self) -> bool:
        return False


# Used for tracking mutation ordering
# if A reads a buffer and B mutates it
# B must be ordered after A
#
# This is useful for a variety of reasons.
# For example, if A's read is never actually used, we can eliminate it.
# Another case is if A's buffer ends up being fused away, we never need to
# materialize that buffer
@dataclasses.dataclass(frozen=True)
class WeakDep(Dep):
    # Fake dependency on unused buffer
    # pyrefly: ignore [bad-override]
    name: str
    # Buffer that is doing the mutation
    mutating_buf: str
    # WeakDep's are also used to add dependencies to prevent some specific reordering,
    # E.g. collectives global ordering.
    # But if other pass guarantees proper ordering by its logic,
    # This additional "fake" deps will be holding optimizations.
    # This flag is used to identify those additional deps.
    is_fake: bool = False

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    @property
    # pyrefly: ignore [bad-override]
    def index(self) -> sympy.Expr:
        raise NotImplementedError("WeakDep does not have an index")

    def get_numel(self) -> sympy.Expr:
        return sympy.S.One

    def rename(self, renames: dict[str, str]) -> "WeakDep":
        if self.name in renames:
            return WeakDep(renames[self.name], self.mutating_buf, self.is_fake)
        return self

    def numbytes_hint(self) -> int:
        return 1  # Purely inserted for ordering, not an actual dep

    def numel_hint(self) -> int:
        return 1  # Purely inserted for ordering, not an actual dep

    def has_unbacked_symbols(self) -> bool:
        return False

    def is_contiguous(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class IndexExprDep:
    index: sympy.Expr  # type: ignore[assignment]
    var_names: tuple[sympy.Symbol, ...]
    size: tuple[sympy.Expr, ...]


@dataclasses.dataclass
class ReadWrites:
    reads: OrderedSet[Dep]
    writes: OrderedSet[Dep]
    index_exprs: OrderedSet[IndexExprDep]
    range_vars: list[sympy.Expr] | None = None
    var_ranges: VarRanges | None = None

    def rename(self, renames: dict[str, str]) -> "ReadWrites":
        return ReadWrites(
            OrderedSet(dep.rename(renames) for dep in self.reads),
            OrderedSet(dep.rename(renames) for dep in self.writes),
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def with_read(self, dep: Dep | OrderedSet[Dep]) -> "ReadWrites":
        if not isinstance(dep, (WeakDep, StarDep, OrderedSet)):
            raise AssertionError(
                f"expected WeakDep, StarDep, or OrderedSet, got {type(dep)}"
            )
        if not isinstance(dep, OrderedSet):
            dep = OrderedSet([dep])
        return ReadWrites(
            OrderedSet.union(self.reads, dep),
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def merge(self, other: "ReadWrites") -> "ReadWrites":
        reads = OrderedSet.union(self.reads, other.reads)
        writes = OrderedSet.union(self.writes, other.writes)
        index_exprs = OrderedSet.union(self.index_exprs, other.index_exprs)
        return ReadWrites(reads - writes, writes, index_exprs)

    @staticmethod
    def merge_list(read_writes: list["ReadWrites"]) -> "ReadWrites":
        all_writes = OrderedSet.union(*[rw.writes for rw in read_writes])
        all_reads = OrderedSet.union(*[rw.reads for rw in read_writes]) - all_writes
        all_index_exprs = OrderedSet.union(*[rw.index_exprs for rw in read_writes])
        return ReadWrites(all_reads, all_writes, all_index_exprs)

    def remove_reads(self, rem_reads: OrderedSet[Dep]) -> "ReadWrites":
        return ReadWrites(
            self.reads - rem_reads,
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def reads_and_writes(self) -> Iterable[Dep]:
        return itertools.chain(self.reads, self.writes)

    def buffer_names(self, ignore_integer_index: bool = True) -> OrderedSet[str]:
        """
        Integer index is used for load_seed.
        """
        names: OrderedSet[str] = OrderedSet()
        for dep in self.reads_and_writes():
            if not isinstance(dep, MemoryDep):
                continue
            if not ignore_integer_index or not isinstance(
                dep.index, (int, sympy.Integer)
            ):
                names.add(dep.name)
        return names

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        result: OrderedSet[sympy.Symbol] = OrderedSet()

        for dep in self.reads_and_writes():
            result |= dep.get_free_symbol_uses(unbacked_only)
        return result


class _RecordLoadStoreInner(V.MockHandler):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool) -> None:
        super().__init__()
        self._reads: OrderedSet[Dep] = OrderedSet()
        self._writes: OrderedSet[MemoryDep] = OrderedSet()
        self._index_exprs: OrderedSet[IndexExprDep] = OrderedSet()
        self._var_ranges: VarRanges = var_ranges
        self._should_normalize: bool = normalize

    @staticmethod
    def drop_unused_symbols(
        index: int | sympy.Expr,
        var_names: list[sympy.Expr],
        sizes: list[sympy.Expr],
    ) -> None:
        """
        Reduction has last (reduced) dim in its sizes, but
        downstream users won't.  Normalize this away.
        """
        if not isinstance(index, sympy.Expr):
            # index can be an int
            return
        free_symbols = index.free_symbols
        while var_names and var_names[-1] not in free_symbols:
            var_names.pop()
            sizes.pop()

    @classmethod
    def _normalize(
        cls, index: sympy.Expr, var_ranges: VarRanges
    ) -> tuple[sympy.Expr, tuple[sympy.Symbol, ...], tuple[sympy.Expr, ...]]:
        # Try to further simplify the indexes even if simplify_loops didn't
        # convert it to the simplest form because of the interference from
        # different indexing formulas.
        index_vars = [*var_ranges.keys()]
        sizes = tuple(var_ranges.values())  # type: ignore[assignment]
        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars,
            sizes,
            index_prevent_reordering([index], index_vars, sizes),
        )

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        new_vars, add_var = var_builder(canonicalization_prefix())
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))
        index = sympy_subs(sympy.expand(index), replacement)

        new_vars = [*new_vars.keys()]
        new_sizes = [*new_sizes]
        cls.drop_unused_symbols(index, new_vars, new_sizes)
        return index, tuple(new_vars), tuple(new_sizes)  # type: ignore[arg-type]

    def canonicalize(
        self, index: sympy.Expr
    ) -> tuple[sympy.Expr, tuple[sympy.Symbol, ...], tuple[sympy.Expr, ...]]:
        if not self._should_normalize:
            sizes = [V.graph.sizevars.simplify(x) for x in self._var_ranges.values()]
            var_names = [k for k, v in zip(self._var_ranges.keys(), sizes) if v != 1]
            sizes = [v for v in sizes if v != 1]

            self.drop_unused_symbols(index, var_names, sizes)

            return index, tuple(var_names), tuple(sizes)  # type: ignore[return-value, arg-type]
        var_ranges = {
            k: V.graph.sizevars.simplify(v)
            for k, v in self._var_ranges.items()
            # TODO(jansel): explore this further normalization
            # if k in free_symbols
        }
        return self._normalize(index, var_ranges)

    def load(self, name: str, index: sympy.Expr) -> None:
        self._reads.add(MemoryDep(name, *self.canonicalize(index)))

    def load_seed(self, name: str, index: int) -> None:
        if not isinstance(index, int):
            raise AssertionError(f"expected index to be int, got {type(index)}")
        self.load(name, sympy.Integer(index))

    def store(
        self, name: str, index: sympy.Expr, value: str, mode: str | None = None
    ) -> None:
        self._writes.add(MemoryDep(name, *self.canonicalize(index), mode=mode))

    def store_reduction(self, name: str, index: sympy.Expr, value: str) -> None:
        self.store(name, index, f"store_reduction({value})")

    def index_expr(self, index: sympy.Expr, dtype: torch.dtype | None) -> None:
        self._index_exprs.add(IndexExprDep(*self.canonicalize(index)))

    def value_expr(self, index: sympy.Expr, dtype: torch.dtype | None) -> None:
        self._index_exprs.add(IndexExprDep(*self.canonicalize(index)))

    def bucketize(
        self,
        values: T,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: T | None = None,
    ) -> None:
        """Records the names of the buffers that bucketize will read from."""
        self._reads.add(StarDep(boundaries[0]))
        if sorter is not None:
            self._reads.add(StarDep(sorter[0]))


class RecordLoadStore(V.KernelFormatterHandler):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool) -> None:
        parent_handler = _RecordLoadStoreInner(
            var_ranges=var_ranges, normalize=normalize
        )
        super().__init__(parent_handler=parent_handler)


# TODO: check call sites
def var_builder(prefix: str) -> tuple[VarRanges, Callable[[sympy.Expr], sympy.Symbol]]:
    cnt = itertools.count()
    var_ranges: VarRanges = {}

    def add_var(length: sympy.Expr) -> sympy.Symbol:
        v = sympy_index_symbol(f"{prefix}{next(cnt)}")
        var_ranges[v] = length
        return v

    return var_ranges, add_var


def index_vars_no_squeeze(
    *argsizes: Sequence[sympy.Expr], prefix: str
) -> tuple[list[list[sympy.Symbol]], VarRanges]:
    var_ranges, add_var = var_builder(prefix)
    args: list[list[sympy.Symbol]] = [list(map(add_var, size)) for size in argsizes]
    return args, var_ranges


def index_vars_squeeze(
    *argsizes: Sequence[sympy.Expr], prefix: str = "d"
) -> tuple[list[Sequence[sympy.Expr]], VarRanges]:
    from .ir import SqueezeView

    var_ranges, add_var = var_builder(prefix)
    args: list[Sequence[sympy.Expr]] = []
    for size in argsizes:
        new_size, reindex = SqueezeView.squeezer(size)
        args.append(reindex(list(map(add_var, new_size))))
    return args, var_ranges


def extract_read_writes(
    fn: Callable[..., Any],
    *argsizes: Sequence[sympy.Expr],
    normalize: bool = False,
    prefix: str = "d",
    hidden_args: Sequence[list[sympy.Expr]] = (),
) -> ReadWrites:
    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)

    from .loop_body import LoopBody

    if isinstance(fn, LoopBody):
        inner = extract_loop_body_with_args(
            fn,
            [*args, *hidden_args],  # type: ignore[list-item]
            var_ranges,
            normalize,
        )
    else:
        # Slow path tracing the function
        rw = RecordLoadStore(var_ranges, normalize=normalize)
        with V.set_ops_handler(rw):
            fn(*args, *hidden_args)
        inner = rw.parent_handler

    if normalize:
        range_vars = []  # Number of vars could differ due to normalization
    else:
        range_vars = [*itertools.chain.from_iterable(args)]

    return ReadWrites(
        # pyrefly: ignore [missing-attribute]
        OrderedSet(inner._reads),
        # pyrefly: ignore [missing-attribute]
        OrderedSet(inner._writes),
        # pyrefly: ignore [missing-attribute]
        inner._index_exprs,
        range_vars,
        var_ranges,
    )


def extract_loop_body_with_args(
    fn: Any,
    args: list[list[sympy.Expr]],
    var_ranges: VarRanges,
    normalize: bool = False,
) -> _RecordLoadStoreInner:
    from .loop_body import MemoryUsageType

    # Fast path to avoid tracing when we already have a LoopBody
    inner = _RecordLoadStoreInner(var_ranges=var_ranges, normalize=normalize)
    name_to_index = fn.indexing_from_args(args)
    if fn.indirect_vars:
        # mimic the `tmpX` naming tracing gives us
        repl = {v: make_symbol(SymT.TMP, i) for i, v in enumerate(fn.indirect_vars)}
        name_to_index = {k: sympy_subs(v, repl) for k, v in name_to_index.items()}  # type: ignore[arg-type]
    for entry in fn.memory_usage[MemoryUsageType.LOAD]:
        inner.load(entry.buffer_name, name_to_index[entry.index_name])  # type: ignore[arg-type]
    for entry in fn.memory_usage[MemoryUsageType.LOAD_SEED]:
        inner.load_seed(entry.buffer_name, int(name_to_index[entry.index_name]))  # type: ignore[arg-type]
    for entry in fn.memory_usage[MemoryUsageType.STORE]:
        inner.store(
            entry.buffer_name,
            name_to_index[entry.index_name],
            None,  # type: ignore[arg-type]
            entry.mode,
        )
    for entry in fn.memory_usage[MemoryUsageType.STORE_REDUCTION]:
        inner.store_reduction(
            entry.buffer_name,
            name_to_index[entry.index_name],
            None,  # type: ignore[arg-type]
        )
    for entry in fn.memory_usage[MemoryUsageType.INDEX_EXPR]:
        inner.index_expr(name_to_index[entry.index_name], None)
    for entry in fn.memory_usage[MemoryUsageType.BUCKETIZE]:
        # All that matters is that we record the buffer name, so place it in the
        # "boundaries" name position to ensure that it's recorded.
        inner.bucketize(
            None,
            (entry.buffer_name, None, None, None),
            None,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )
    # fn.memory_usage[MemoryUsageType.CHECK_BOUNDS] intentionally skipped
    return inner


def extract_input_node_reduction_ranges(
    input_node: "torch._inductor.ir.IRNode",
) -> tuple[list[sympy.Expr] | None, list[sympy.Expr] | None]:
    """
    Returns the size and reduction size of all inputs, if the sizes and reduction_sizes (if exist) are all the same.
    It's possible that a node has multiple inputs, some are Reduction nodes and others are Pointwise nodes.
    In this case, reduction_sizes of the Reduction nodes need to be the same.
    Otherwise returns (None, None).
    """

    from .ir import ComputedBuffer, ExternKernel, Loops

    size: list[sympy.Expr] | None
    reduction_size: list[sympy.Expr] | None

    if isinstance(input_node.get_defining_op(), ComputedBuffer):
        # Input node has already been realized. Return its size and reduction_size.
        size = [*input_node.get_size()]
        reduction_size = [*input_node.get_reduction_size()]
        if len(reduction_size) > 0:
            return (size, reduction_size)
        else:
            return (None, None)

    if not isinstance(input_node.data.data, Loops):  # type: ignore[attr-defined]
        # Other IRNodes do not have reduction_ranges.
        return (None, None)

    # There is one issue: what if there are views / permutations between the input node and its dependent realized nodes?
    # The current method still uses reduction ranges from the dependent realized node, which is not ideal.
    # Is there a way to check whether there are permutations in between?
    reads = input_node.get_reads()
    reduction_size: list[sympy.Expr] | None = None
    size: list[sympy.Expr] | None = None
    while reduction_size is None and len(reads) > 0:
        seen: OrderedSet[str] = OrderedSet()
        new_reads: list[Dep] = []
        for read in reads:
            if not isinstance(read, MemoryDep):
                continue
            if read.name in seen:
                continue
            seen.add(read.name)
            buffer = V.graph.try_get_buffer(read.name)
            if buffer is None:
                continue
            op = buffer.get_defining_op()
            if op is None or isinstance(op, ExternKernel):
                continue

            if isinstance(op, ComputedBuffer) and len(op.get_reduction_size()) > 0:
                if reduction_size is None:
                    reduction_size = [*op.get_reduction_size()]
                    size = [*op.get_size()]
                elif reduction_size != [*op.get_reduction_size()] or size != [
                    *op.get_size()
                ]:
                    return (None, None)
            else:
                new_reads.extend(op.get_reads())
        if reads == new_reads:
            return (size, reduction_size)
        else:
            reads = OrderedSet(new_reads)
    return (size, reduction_size)


def canonicalization_prefix() -> str:
    return "c"


# ops handler which computes all the free symbols for an IR
class FreeSymbolsOpsHandler(DefaultHandler):
    symbols: OrderedSet[sympy.Symbol]

    def __init__(self, unbacked_only: bool = True) -> None:
        self.symbols = OrderedSet()
        self.get_symbols = free_unbacked_symbols if unbacked_only else free_symbols

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        for a in itertools.chain(args, kwargs.values()):
            if isinstance(a, (sympy.Expr, sympy.logic.boolalg.Boolean)):
                self.symbols |= self.get_symbols(a)

    def indirect_indexing(
        self,
        index_var: Any,
        size: int | sympy.Expr,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        if isinstance(index_var, (sympy.Expr, sympy.logic.boolalg.Boolean)):
            raise AssertionError(
                f"index_var must not be a sympy Expr or Boolean, got {type(index_var)}"
            )
        self.symbols |= self.get_symbols(size)
        return sympy_index_symbol(f"({str(index_var)})")

    def frexp(self, x: Any) -> tuple[None, ...]:
        return (None,) * 2

    def scan(
        self, dtypes: Any, combine_fn: Any, values: Sequence[Any]
    ) -> tuple[None, ...]:
        return (None,) * len(values)

    def sort(
        self, dtypes: Any, values: Sequence[Any], stable: Any, descending: Any
    ) -> tuple[None, ...]:
        return (None,) * len(values)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: None | tuple[None, ...],
    ) -> None | tuple[None, ...]:
        num_values = reduction_num_outputs(reduction_type)
        return (None,) * num_values if num_values > 1 else None

    def masked(self, mask: Any, body: Callable[..., Any], other: Any) -> None:
        if not callable(body):
            raise AssertionError("masked body must always be callable.")
        # The body can make additional calls, for e.g. ops.indirect_indexing
        body()


def extract_free_symbols(
    fn: Callable[..., Any],
    index: Sequence[sympy.Expr],
    rindex: Sequence[sympy.Expr] | None = None,
    unbacked_only: bool = True,
) -> OrderedSet[sympy.Symbol]:
    from .ir import FlexibleLayout

    args = [index, rindex] if rindex is not None else [index]
    handler = FreeSymbolsOpsHandler(unbacked_only)
    # NB: I cargo culted the allow_indexing patch here, I don't understand why
    # people do this all over
    with (
        V.set_ops_handler(handler),
        patch.object(FlexibleLayout, "allow_indexing", True),
    ):
        fn(*args)
    return handler.symbols


class SymbolUsageCollectorOpsHandler(DefaultHandler):
    usages: OrderedSet[str]

    def __init__(self, symbol: sympy.Symbol) -> None:
        self.symbol = symbol
        self.usages = OrderedSet()

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        used_here = self.symbol in args or self.symbol in kwargs.values()
        if used_here:
            self.usages.add(name)
<|endoftext|>import math
from dataclasses import dataclass
from typing import Any

import sympy

import torch
from torch.fx.node import map_arg
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

from .loop_body import LoopBody
from .utils import dominated_nodes


def val_expressable_in_32_bits(val: Any) -> bool:
    if getattr(val, "is_Boolean", False):
        return True

    if isinstance(val, sympy.Expr):
        if not val.is_number:
            raise AssertionError(f"expected a number, got {val}")
        if val.is_Integer or val.is_Boolean:
            val = int(val)
        else:
            val = float(val)

    # bound within mantissa
    if isinstance(val, float):
        return val <= (2**24) and val >= -(2**24)

    if isinstance(val, int):
        iinfo = torch.iinfo(torch.int32)
        return val <= iinfo.max and val >= iinfo.min

    raise TypeError(f"Unexpected value {val}")


def range_expressable_in_32_bits(range: ValueRanges[sympy.Expr]) -> bool:
    return val_expressable_in_32_bits(range.lower) and val_expressable_in_32_bits(
        range.upper
    )


def try_to_reduce_precision(
    node: Any,
    bounds: dict[Any, Any],
    indirect_vars: list[Any],
    indices: dict[Any, sympy.Expr],
    replacement_vals: dict[Any, ValueRanges[sympy.Expr]],
) -> None:
    # if a downstream use of a node explicitly converts to int32, or float16/float32/float64,
    # then its precision is set for that chain of uses, and we don't need to consider those
    # dominated values
    def skip_filter(node: Any) -> bool:
        return node.target == "to_dtype" and node.args[2] in (
            torch.int32,
            torch.float32,
            torch.float64,
        )

    # TODO - there are dominated uses whose dtype does not depend on whether
    # we reduce the precision here, e.g. add(int64, int64) one of the args can be reduced to
    # int32 without changing the output precision of the node. this case hasn't shown up
    for dominated in dominated_nodes([node], skip_filter):
        if dominated.target in ["store", "output"]:
            continue

        if isinstance(dominated.target, str) and "set_indirect" in dominated.target:
            idx = int(dominated.target[len("set_indirect") :])
            indirect_var = indirect_vars[idx]

            # We check that we can compute all the indices it's involved in with int32
            for index, expr in indices.items():
                if indirect_var in expr.free_symbols:
                    index_val = replacement_vals[index]

                    if math.isinf(index_val.lower) or math.isinf(index_val.upper):
                        return

                    # all indices are integers, so make sure that we
                    # use the bounds of integers instead of floats.
                    # TODO - not sure if we should be doing int/float casts while tracing,
                    # might interfere with sympy.

                    index_val_int = ValueRanges[sympy.Expr](
                        int(index_val.lower), int(index_val.upper)
                    )
                    if not range_expressable_in_32_bits(index_val_int):
                        return

        if not range_expressable_in_32_bits(bounds[dominated]):
            return

    args = list(node.args)
    args[2] = torch.int32
    node.args = tuple(args)


def indexing_dtype_strength_reduction(loop_body: LoopBody) -> None:
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32
    """
    bv = loop_body.bounds()

    int64_dtype_nodes = [
        node
        for node in loop_body.get_nodes()
        if (
            node.target == "to_dtype"
            and node.args[2] == torch.int64
            and node not in bv.unbounded_vars
        )
    ]
    if not int64_dtype_nodes:
        return

    bounds = bv.get_bounds()

    # TODO - if dominated node of one to_dtype is not expressible in int32,
    # we should short circuit another to_dtype node if that node also dominates
    for node in int64_dtype_nodes:
        try_to_reduce_precision(
            node,
            bounds,
            loop_body.indirect_vars,
            loop_body.indexing_exprs,
            bv.replacement_vals,
        )


@dataclass(frozen=True)
class _ValueUseRule:
    # These fields are op arguments, so Any covers FX nodes, SymPy exprs,
    # scalars, and nested tuples/lists. value_sinks seed the backward walk
    # unconditionally. value_inputs propagate value demand only when the op's
    # result is already value-reachable. indexing_inputs block propagation
    # because they only affect addresses, bounds, masks, or other indexing-only
    # state.
    value_inputs: tuple[Any, ...] = ()
    value_sinks: tuple[Any, ...] = ()
    indexing_inputs: tuple[Any, ...] = ()


def _collect_fx_nodes(arg: Any) -> OrderedSet[torch.fx.Node]:
    nodes: OrderedSet[torch.fx.Node] = OrderedSet()

    def add_node(node: torch.fx.Node) -> torch.fx.Node:
        nodes.add(node)
        return node

    map_arg(arg, add_node)
    return nodes


def _collect_input_nodes(node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
    inputs = OrderedSet(node.all_input_nodes)
    if (
        node.op == "call_method"
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
    ):
        inputs.discard(node.args[0])
    return inputs


class _ValueUseRules:
    @staticmethod
    def default_rule(*args: Any, **kwargs: Any) -> _ValueUseRule | None:
        return None

    def load(self, name: str, index: sympy.Expr) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(index,))

    def load_seed(self, name: str, offset: int) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(offset,))

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: Any,
        mode: Any = None,
    ) -> _ValueUseRule:
        return _ValueUseRule(
            value_sinks=(value,),
            indexing_inputs=(index,),
        )

    def store_reduction(
        self, name: str, index: sympy.Expr, value: Any
    ) -> _ValueUseRule:
        return _ValueUseRule(
            value_sinks=(value,),
            indexing_inputs=(index,),
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: Any,
    ) -> _ValueUseRule:
        return _ValueUseRule(value_sinks=(value,))

    def partial_accumulate(
        self,
        name: str,
        reduction_type: str,
        value: Any,
        extra_meta: dict[str, Any],
    ) -> _ValueUseRule:
        return _ValueUseRule(value_sinks=(value,))

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[Any, ...],
        stable: bool,
        descending: bool,
    ) -> _ValueUseRule:
        return _ValueUseRule(value_sinks=(values,))

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn_or_values: Any,
        values: tuple[Any, ...] | None = None,
    ) -> _ValueUseRule:
        if values is None:
            values = combine_fn_or_values
        return _ValueUseRule(value_sinks=(values,))

    def bucketize(
        self,
        values: Any,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: Any,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: Any | None = None,
    ) -> _ValueUseRule:
        return _ValueUseRule(
            value_inputs=(values,),
            indexing_inputs=(boundaries, boundary_indices, sorter, sorter_indices),
        )

    def indirect_indexing(
        self, x: Any, size: sympy.Expr, check: bool = True, wrap_neg: bool = True
    ) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(x, size))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(expr, size))

    def masked(self, mask: Any, body: Any, other: Any) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(other,), indexing_inputs=(mask,))

    def masked_subblock(self, mask: Any, other: Any) -> _ValueUseRule:
        return _ValueUseRule(value_inputs=(other,), indexing_inputs=(mask,))

    def set_indirect(self, new_var: Any) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(new_var,))

    def device_assert_async(self, cond: Any, msg: str) -> _ValueUseRule:
        return _ValueUseRule(indexing_inputs=(cond,))


class _ValueUseAnalysis:
    def __init__(self, loop_body: LoopBody) -> None:
        self.loop_body = loop_body
        self.rules = _ValueUseRules()
        self.root_graph = loop_body.root_block.graph
        self.subblocks = getattr(loop_body, "subblocks", {})
        self.indirect_vars = getattr(loop_body, "indirect_vars", ())
        self.graphs = [
            self.root_graph,
            *(block.graph for block in self.subblocks.values()),
        ]
        self.install_call_module_rules()

        self.value_reachable: OrderedSet[torch.fx.Node] = OrderedSet()
        self.worklist: list[tuple[torch.fx.Graph, torch.fx.Node]] = []
        self.indirect_inputs: dict[sympy.Symbol, tuple[torch.fx.Graph, Any]] = {}

    def run(self) -> bool:
        if not self.has_index_expr():
            return False

        self.seed_value_reachable_nodes()
        return self.propagate_value_reachability()

    def has_index_expr(self) -> bool:
        return any(
            graph.find_nodes(op="call_method", target="index_expr", sort=False)
            for graph in self.graphs
        )

    def install_call_module_rules(self) -> None:
        for graph in self.graphs:
            for node in graph.find_nodes(op="call_module", sort=False):
                if not isinstance(node.target, str):
                    continue
                if node.target in self.subblocks:
                    setattr(self.rules, node.target, self.rules.masked_subblock)
                elif node.target.startswith("scan"):
                    setattr(self.rules, node.target, self.rules.scan)
                elif node.target.startswith("set_indirect"):
                    setattr(self.rules, node.target, self.rules.set_indirect)

    def seed_value_reachable_nodes(self) -> None:
        self._enqueue_graph_output(self.root_graph)
        for graph in self.graphs:
            for node in graph.nodes:
                if node.op == "output":
                    continue
                if (
                    node.op == "call_module"
                    and isinstance(node.target, str)
                    and node.target.startswith("set_indirect")
                ):
                    idx = int(node.target[len("set_indirect") :])
                    self.indirect_inputs[self.indirect_vars[idx]] = (graph, node.args)
                for sink_node in self.value_sink_nodes(node):
                    self.worklist.append((graph, sink_node))

    def propagate_value_reachability(self) -> bool:
        changed = False
        while self.worklist:
            graph, node = self.worklist.pop()
            if node in self.value_reachable:
                continue
            self.value_reachable.add(node)

            if node.target == "index_expr":
                node.target = "value_expr"
                changed = True

            if (
                node.op == "call_module"
                and isinstance(node.target, str)
                and node.target in self.subblocks
            ):
                self._enqueue_graph_output(self.subblocks[node.target].graph)

            if node.op == "call_module" and node.target == "get_index":
                index_name = node.args[0]
                if not isinstance(index_name, str):
                    raise AssertionError(f"expected str index name, got {index_name!r}")
                expr = self.loop_body.indexing_exprs[index_name]
                if isinstance(expr, sympy.Expr):
                    for symbol in expr.free_symbols:
                        indirect_input = self.indirect_inputs.get(symbol)
                        if indirect_input is not None:
                            ig, ia = indirect_input
                            map_arg(ia, lambda n: self.worklist.append((ig, n)))

            for input_node in self.value_input_nodes(node):
                self.worklist.append((graph, input_node))

        return changed

    def rule_for_node(self, node: torch.fx.Node) -> _ValueUseRule | None:
        if node.op == "call_method" and isinstance(node.target, str):
            rule_fn = getattr(self.rules, node.target, self.rules.default_rule)
            return rule_fn(*node.args[1:], **node.kwargs)
        if node.op == "call_module" and isinstance(node.target, str):
            rule_fn = getattr(self.rules, node.target, self.rules.default_rule)
            return rule_fn(*node.args, **node.kwargs)
        return None

    def value_input_nodes(self, node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
        rule = self.rule_for_node(node)
        if rule is None:
            return _collect_input_nodes(node)
        return _collect_fx_nodes(rule.value_inputs) | _collect_fx_nodes(
            rule.value_sinks
        )

    def value_sink_nodes(self, node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
        rule = self.rule_for_node(node)
        return _collect_fx_nodes(rule.value_sinks) if rule else OrderedSet()

    def _enqueue_graph_output(self, graph: torch.fx.Graph) -> None:
        output_nodes = graph.find_nodes(op="output", sort=False)
        if len(output_nodes) != 1:
            raise AssertionError(
                f"expected exactly 1 output node, got {len(output_nodes)}"
            )
        map_arg(output_nodes[0].args, lambda n: self.worklist.append((graph, n)))


def convert_index_expr_to_value_expr(loop_body: LoopBody) -> None:
    """
    Rewrite ``index_expr`` nodes that participate in value computation to
    ``value_expr``, so codegen can honor the requested dtype instead of
    narrowing to the kernel's indexing dtype.

    Classification: walk backward from value sinks (store value, reduction,
    output). Indexing inputs to ops such as ``load``, ``store``,
    ``check_bounds``, and ``set_indirect`` do not receive value demand. Any
    ``index_expr`` reachable from a value sink is converted in-place to
    ``value_expr``. Indirect indexing is handled by following value-reachable
    ``get_index`` expressions through any referenced ``indirect*`` symbols back
    to the corresponding ``set_indirect*`` input.

    Mixed-use policy: if the same ``index_expr`` feeds both an indexing use and
    a value use, rewrite it in-place to ``value_expr``. This is conservative and
    correct because the indexing path may compute at a wider dtype than needed,
    but the value path cannot lose precision. Cloning would preserve the
    narrower indexing path, but the mixed-use case is rare and the simpler
    in-place rewrite avoids extra CSE/register pressure.
    """
    if _ValueUseAnalysis(loop_body).run():
        LoopBody.get_nodes.clear_cache(loop_body)
        LoopBody.bounds.clear_cache(loop_body)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
import operator
from typing import TYPE_CHECKING

import torch
from torch import _prims, Tensor
from torch._utils import _get_device_index


if TYPE_CHECKING:
    from collections.abc import Sequence


log = logging.getLogger(__name__)


def make_prim(
    schema: str,
    impl_aten,
    return_type=_prims.RETURN_TYPE.NEW,
    doc: str = "",
    tags: Sequence[torch.Tag] | None = None,
):
    if isinstance(return_type, tuple):

        def meta(*args, **kwargs):
            return tuple(_prims.TensorMeta(o) for o in impl_aten(*args, **kwargs))

    else:

        def meta(*args, **kwargs):
            return _prims.TensorMeta(impl_aten(*args, **kwargs))

    return _prims._make_prim(
        schema=schema,
        return_type=return_type,
        meta=meta,
        impl_aten=impl_aten,
        doc=doc,
        tags=tags,
    )


def eager_force_stride(input_tensor: Tensor, stride) -> Tensor:
    if input_tensor.stride() == stride:
        return input_tensor
    new_tensor = input_tensor.clone().as_strided(
        input_tensor.shape,
        stride,
    )
    new_tensor.copy_(input_tensor)
    return new_tensor


def eager_prepare_softmax(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    amax = torch.amax(x, dim, keepdim=True)
    return amax, torch.sum(torch.exp(x - amax), dim, keepdim=True)


# Custom prims used for handling randomness
seed = make_prim(
    "inductor_seed(Device device) -> Tensor",
    lambda device: torch.randint(2**63 - 1, [], device=device),
    doc="create a fresh seed (one per call) for use with inductor_rand",
    tags=(torch.Tag.nondeterministic_seeded,),
)
seeds = make_prim(
    "inductor_seeds(int count, Device device) -> Tensor",
    lambda count, device: torch.randint(2**63 - 1, [count], device=device),
    doc="Horizontal fusion of many inductor_seed() calls",
    tags=(torch.Tag.nondeterministic_seeded,),
)
lookup_seed = make_prim(
    # if inductor_lookup_seed changes, update partitioners.py
    "inductor_lookup_seed(Tensor seeds, int index) -> Tensor",
    lambda seeds, index: seeds[index].clone(),
    doc="Extract a single seed from the result of inductor_seeds()",
)
# inductor_random() doesn't accept a dtype.
# instead, its lowering always burns in float32, and conversions to a different type
# are explicit in the graph. We therefore need this impl (used during tracing) to hardcode
# the dtype, so it always faithfully produces a float32 tensor during tracing,
# even if the default dtype is set to something else.
random = make_prim(
    "inductor_random(SymInt[] size, Tensor seed, str mode, *, ScalarType? align_dtype=None) -> Tensor",
    lambda size, seed, mode, *, align_dtype=None: getattr(torch, mode)(
        size, device=seed.device, dtype=torch.float32
    ),
    doc="torch.rand()/torch.randn() using backend-specific RNG that can be fused",
)
randint = make_prim(
    "inductor_randint(SymInt low, SymInt high, SymInt[] size, Tensor seed) -> Tensor",
    lambda low, high, size, seed: torch.randint(low, high, size, device=seed.device),
    doc="torch.randint() using backend-specific RNG that can be fused",
)


def _reserve_rng_state(device: torch.device, used_offset):
    """
    Reserve `used_offset` 32-bit Philox samples on the given CUDA device and
    return (seed, base), where base is in Philox-4x32 units.

    This mirrors how Inductor accounts for Philox consumption so compiled
    dropout kernels can reconstruct eager RNG state.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        # Only CUDA devices have Philox-based CUDAGenerator. For non-CUDA
        # devices this prim should be dead code and never actually run.
        return 0, 0

    dev_index = _get_device_index(dev, optional=True)
    if dev_index is None:
        dev_index = torch.cuda.current_device()

    gen = torch.cuda.default_generators[dev_index]
    seed_t, off_t, intra_t = torch.ops.inductor_prims.inductor_reserve_rng_state(
        gen, used_offset
    )

    # NOTE: for correctness in eager, intra_t should be 0.
    # Keep everything as tensor math to avoid host sync.
    if intra_t.device.type != off_t.device.type:
        intra = int(intra_t.item())
        base = torch.div(off_t + intra, 4, rounding_mode="floor")
    else:
        base = torch.div(off_t + intra_t, 4, rounding_mode="floor")
    return seed_t, base


def _rand_eager_offset_impl(offset, device: torch.device) -> Tensor:
    """
    Reserve `offset` 32-bit Philox samples and return a 1-element int64 tensor
    Place-holder: will be replaced by rand_eager_offsets
    In fx_passes/replace_random.py
        fuse_offset_creation_pass()
    """
    return torch.empty(2, dtype=torch.int64, device=device)


def _rand_eager_offsets_impl(offsets, device: torch.device) -> Tensor:
    """
    Batched version of _rand_eager_offset_impl. For each entry in `offsets`,
    reserve that many 32-bit Philox samples and return a 1D int64 tensor
    containing the packed (seed, base) values for each reservation.
    """
    states = [_reserve_rng_state(device, int(off)) for off in offsets]
    seeds = [s for s, _ in states]
    bases = [b for _, b in states]

    def _to_i64(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x, device=device, dtype=torch.int64)

    seeds_tensor = torch.stack([_to_i64(x) for x in seeds]).view(-1)
    bases_tensor = torch.stack([_to_i64(x) for x in bases]).view(-1)
    packed = torch.stack([seeds_tensor, bases_tensor], dim=1)
    return packed


def _rand_eager_offsets_meta(offsets, device: torch.device):
    return torch.empty((len(offsets), 2), dtype=torch.int64, device=device)


rand_eager_offset = make_prim(
    "inductor_rand_eager_offset(SymInt offset, Device device) -> Tensor",
    _rand_eager_offset_impl,
    doc=(
        "Reserve `offset` 32-bit Philox samples on `device` and return a "
        "1-element int64 tensor containing packed (seed, base)."
    ),
    tags=(torch.Tag.nondeterministic_seeded,),
)


rand_eager_offsets = _prims._make_prim(
    schema="inductor_rand_eager_offsets(SymInt[] offsets, Device device) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_rand_eager_offsets_meta,
    impl_aten=_rand_eager_offsets_impl,
    doc=(
        "Batched version of inductor_rand_eager_offset. For each entry in "
        "`offsets`, reserves that many 64-bit Philox samples and returns "
        "packed (seed, base) values."
    ),
    tags=(torch.Tag.nondeterministic_seeded,),
)


force_stride_order = make_prim(
    "inductor_force_stride_order(Tensor input, SymInt[] stride) -> Tensor",
    eager_force_stride,
    doc="Force the stride order for input tensor. No-op if the input tensor already has the stride. Do a copy otherwise",
)
_unsafe_index_put_ = make_prim(
    "_unsafe_index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
    lambda self, indices, values, accumulate=False: torch.ops.aten.index_put_(
        self, indices, values, accumulate
    ),
    doc="Unsafe index_put_ (doesn't issue device asserts)",
)
fma = make_prim(
    "fma(Tensor a, Tensor b, Tensor c) -> Tensor",
    lambda a, b, c: (a * b) + c,
    doc="Fused multiply add: fma(a, b, c) -> (a * b) + c without rounding after the multiplication",
    tags=(torch.Tag.pointwise,),
)


def _fma_setup_context(ctx, inputs, output):
    a, b, c = inputs
    ctx.a_shape = a.shape
    ctx.b_is_tensor = isinstance(b, Tensor)
    if ctx.b_is_tensor:
        ctx.b_shape = b.shape
        ctx.save_for_backward(a, b)
    else:
        ctx.b = b
        ctx.save_for_backward(a)
    ctx.c_shape = c.shape


def _fma_backward(ctx, grad):
    if ctx.b_is_tensor:
        a, b = ctx.saved_tensors
    else:
        (a,) = ctx.saved_tensors
        b = ctx.b

    grad_a = (grad * b).sum_to_size(ctx.a_shape) if ctx.needs_input_grad[0] else None
    grad_b = (
        (grad * a).sum_to_size(ctx.b_shape)
        if ctx.b_is_tensor and ctx.needs_input_grad[1]
        else None
    )
    grad_c = grad.sum_to_size(ctx.c_shape) if ctx.needs_input_grad[2] else None
    return grad_a, grad_b, grad_c


torch.library.register_autograd(
    fma,
    _fma_backward,
    setup_context=_fma_setup_context,
)


# Register DTensor sharding strategies for inductor prims ops.
# This must happen here (not in _pointwise_ops.py) because the ops
# don't exist until make_prim() is called above.
if torch.distributed.is_available():
    from torch.distributed.tensor._ops._pointwise_ops import register_inductor_prims

    register_inductor_prims()

prepare_softmax_online = make_prim(
    "prepare_softmax_online(Tensor a, int dim) -> (Tensor, Tensor)",
    eager_prepare_softmax,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Prepare the softmax by computing the max and sum.",
)


def _flattened_index_to_nd(indices, width):
    import sympy

    from torch.utils._sympy.functions import FloorDiv

    dim = len(width)

    if dim == 1:
        return [indices]
    elif dim >= 2:
        m = functools.reduce(operator.mul, width[1:])
        if isinstance(indices, sympy.Expr) or isinstance(m, sympy.Expr):
            ih = FloorDiv(indices, m)
        else:
            ih = indices // m
        indices_new = indices - (ih * m)
        return [ih, *_flattened_index_to_nd(indices_new, width[1:])]
    else:
        raise ValueError(f"Unknown dim: {dim}")


def _flatten_index(indices, width):
    result = indices[0]
    for d in range(1, len(indices)):
        result = width[d] * result + indices[d]
    return result


def _low_memory_max_pool_with_offsets_aten(
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    dim = len(kernel_size)
    if dim == 2:
        vals, indices = torch.ops.aten.max_pool2d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )
    else:
        vals, indices = torch.ops.aten.max_pool3d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode
        )

    idhw = _flattened_index_to_nd(indices, self.shape[-dim:])

    dhw_inc = []

    for d in range(dim):
        bh_shape = [1] * self.ndim
        bh_shape[-dim + d] = -1
        bh = torch.arange(
            indices.shape[-dim + d], dtype=torch.int64, device=self.device
        ).view(bh_shape)
        hbase = bh * stride[d] - padding[d]
        h_inc = (idhw[d] - hbase) // dilation[d]
        dhw_inc.append(h_inc)

    offsets = _flatten_index(dhw_inc, kernel_size)

    return vals, offsets.to(torch.int8)


def _low_memory_max_pool_offsets_to_indices_aten(
    offsets,
    kernel_size,
    input_size,
    stride,
    padding,
    dilation,
):
    dim = len(kernel_size)
    offsets = offsets.to(torch.int64)
    dhw_inc = _flattened_index_to_nd(offsets, kernel_size)

    idhw = []
    for d in range(dim):
        bh_shape = [1] * offsets.ndim
        bh_shape[-dim + d] = -1
        bh = torch.arange(
            offsets.shape[-dim + d], dtype=torch.int64, device=offsets.device
        ).view(bh_shape)
        hbase = bh * stride[d] - padding[d]
        idhw.append(hbase + dhw_inc[d] * dilation[d])

    return _flatten_index(idhw, input_size)


_low_memory_max_pool_with_offsets = make_prim(
    "_low_memory_max_pool_with_offsets(Tensor self, SymInt[] kernel_size, SymInt[] stride,  SymInt[] padding, SymInt[] dilation, bool ceil_mode) -> (Tensor, Tensor)",
    _low_memory_max_pool_with_offsets_aten,
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),
    doc="Instead of returning indices, returns indices offsets.",
)

_low_memory_max_pool_offsets_to_indices = make_prim(
    "_low_memory_max_pool_offsets_to_indices(Tensor self, SymInt[] kernel_size, SymInt[] input_size, SymInt[] stride, SymInt[] padding, SymInt[] dilation) -> Tensor",
    _low_memory_max_pool_offsets_to_indices_aten,
    doc="Convert small int offsets to regular indices.",
)


def _cvt_e8m0_rceil_aten(inp: Tensor) -> Tensor:
    """
    Convert float to e8m0 format with ceiling rounding and satfinite semantics.

    e8m0 format: 8-bit biased exponent (bias=127), no mantissa.
    For MX format scaling, this extracts the exponent with ceiling rounding.
    Uses satfinite semantics: inf is saturated to 254 (max finite e8m0).
    Accepts float32, float16, or bfloat16 (upcasted to float32 internally).
    """
    if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"cvt_e8m0_rceil requires float32, float16, or bfloat16 input, got {inp.dtype}"
        )
    if inp.dtype != torch.float32:
        inp = inp.to(torch.float32)
    inp_bits = inp.view(torch.int32)
    biased_exp = (inp_bits >> 23) & 0xFF
    mantissa = inp_bits & 0x7FFFFF
    needs_round_up = mantissa != 0
    e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
    # satfinite: clamp to max finite e8m0 value (254), not 255 (inf/nan)
    e8m0_biased = torch.clamp(e8m0_biased, 0, 254)
    return e8m0_biased.to(torch.uint8)


cvt_e8m0_rceil = make_prim(
    "inductor_cvt_e8m0_rceil(Tensor input) -> Tensor",
    _cvt_e8m0_rceil_aten,
    doc="Convert float to e8m0 with ceiling rounding. Uses PTX cvt.rp.satfinite.ue8m0x2.f32 on SM100+.",
)
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .heuristics.template.params import DictKernelTemplateParams


if TYPE_CHECKING:
    from collections.abc import Generator

    from .codegen.common import KernelTemplate
    from .heuristics.template.params import KernelTemplateParams
    from .ir import ChoiceCaller, Layout
    from .kernel_inputs import KernelInputs
    from .select_algorithm import ExternKernelChoice


class KernelTemplateChoice:
    """
    A class that encapsulates all the components needed to create a ChoiceCaller from a template.

    This class implements lazy evaluation for the choice property - the actual ChoiceCaller
    is only created when first accessed via the choice property.
    """

    def __init__(
        self,
        template: KernelTemplate | ExternKernelChoice,
        params: KernelTemplateParams,
        extra_kwargs: dict[str, Any],
        layout: Layout,
        inputs: KernelInputs,
    ):
        self.template = template
        self.params = params
        self.extra_kwargs = extra_kwargs
        self.layout = layout
        self.inputs = inputs
        self.annotations: dict[str, Any] = {"ktc": self}

    @property
    def choice(self) -> ChoiceCaller | None:
        """
        Lazily evaluate and return the ChoiceCaller for this template choice.

        On first access, calls template.choice_or_none() with the stored parameters.
        If successful, caches and returns the ChoiceCaller. If it fails, caches
        and returns None. Subsequent accesses return the cached value.

        Returns:
            ChoiceCaller if the template choice succeeds, None otherwise
        """
        if not hasattr(self, "_choice"):
            # First time accessing choice - try to generate it
            kwargs = self.params.to_kwargs()
            self._choice = self.template.choice_or_none(
                **kwargs,
                **self.extra_kwargs,
                layout=self.layout,
                input_nodes=self.inputs.nodes(),
            )
            if self._choice is not None:
                self._choice.annotations = self.annotations
        return self._choice


def make_ktc_generator(
    template: KernelTemplate | ExternKernelChoice,
    cs: Generator[KernelTemplateParams, None, None],
    extra_kwargs: dict[str, Any],
    overrides: dict[str, Any],
    layout: Layout,
    inputs: KernelInputs,
) -> Generator[KernelTemplateChoice, None, None]:
    """
    Create a generator of KernelTemplateChoice objects for a given template.

    Args:
        template: The template object (KernelTemplate or ExternKernelChoice)
        cs: Generator of KernelTemplateParams from template heuristic
        overrides: Override kwargs for the template
        layout: Layout value for the template
        inputs: KernelInputs for the op

    Yields:
        KernelTemplateChoice objects
    """
    for params in cs:
        # Apply overrides to params
        base_kwargs = params.to_kwargs()
        final_kwargs = {**base_kwargs, **overrides}
        final_params = DictKernelTemplateParams(final_kwargs)
        yield KernelTemplateChoice(
            template=template,
            params=final_params,
            extra_kwargs=extra_kwargs,
            layout=layout,
            inputs=inputs,
        )
<|endoftext|>from typing import Any

import torch.library
from torch import Tensor
from torch.autograd import Function


_test_lib_def = torch.library.Library("_inductor_test", "DEF")
_test_lib_def.define("realize(Tensor self) -> Tensor", tags=torch.Tag.pt2_compliant_tag)

_test_lib_impl = torch.library.Library("_inductor_test", "IMPL")
for dispatch_key in ("CPU", "CUDA", "MPS", "Meta"):
    _test_lib_impl.impl("realize", lambda x: x.clone(), dispatch_key)


class Realize(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx: object, x: Tensor) -> Tensor:
        return torch.ops._inductor_test.realize(x)

    @staticmethod
    # types need to stay consistent with _SingleLevelFunction
    def backward(ctx: Any, *grad_output: Any) -> Any:
        return grad_output[0]


def realize(x: Tensor) -> Tensor:
    return Realize.apply(x)
<|endoftext|>import importlib
import itertools
import logging
import pickle
import random
import signal
import string
import traceback
import types
from collections.abc import Callable, KeysView, Sequence
from enum import Enum
from functools import partial, wraps
from types import FrameType
from typing import Any, get_args, get_origin, Literal, TypeVar, Union

import torch
from functorch.compile import min_cut_rematerialization_partition
from torch._inductor.custom_graph_pass import CustomGraphPass, CustomPartitionerFn
from torch._inductor.scheduler import BaseSchedulerNode
from torch.utils._config_module import _ConfigEntry, ConfigModule
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def is_type(type_hint, comp_type) -> bool:  # type: ignore[no-untyped-def]
    """
    Determines if type_hint is comp_type. There are some type annotations that this doesn't work for.
    I think it's because some Type annotations are Type Objects and some are Special Forms, but not sure.
    There's definite room for improvement to make this more general for someone who deeply understands
    Python types.
    """
