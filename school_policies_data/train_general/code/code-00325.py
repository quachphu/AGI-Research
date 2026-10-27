def _arg_str(a: object) -> str:
    if isinstance(a, sympy.Expr):
        return sympy_str(a)
    return str(a)


# See OpDecompositions for superclass that desugars operations like reciprocal/square.
class OpsHandler(Generic[T]):
    """
    Protocol describing the set of valid operations on ``torch._inductor.virtualized.ops``,
    as well as the contract for op handlers.  The type T signifies the domain
    of the abstract analysis AKA what all the functions return / take as arguments
    anywhere compute occurs.

    While these operators are typically dtype polymorphic (e.g., you can use mul
    on both integers and floats), they do NOT do promotion and usually return the
    same dtype as the input.  You are expected to have handled type promotion
    during ATen decompositions.  Most operators correspond exactly to pointwise
    operations as defined by torch, so when in doubt about semantics, check the
    corresponding torch documentation.  These are all scalar operations (so they
    are defined to operate on a single element at a time.)

    For convenience, many operators take a src_dtype which indicates what the dtype
    of the input argument is.  Although in principle this can be derived by an
    analysis, providing this for ops where it is useful helps avoid having to repeatedly
    recompute dtype in code generation.

    Note that this often describes a class of static methods, for stateless
    ops handlers.

    Handlers are often defined using metaprogramming (e.g. _initialize_pointwise_overrides),
    which means you will not get type errors for those methods.  We have tests in
    test/inductor/test_op_completeness.py which check that all operators are implemented after
    all the metaprogramming has run.
    """

    def constant(self, value: bool | float | int, dtype: torch.dtype) -> T:
        """Produces a scalar constant of type dtype."""
        raise NotImplementedError

    def load_seed(self, name: str, offset: T) -> T:
        """Computes inductor_prims.lookup_seed."""
        raise NotImplementedError

    def rand(self, seed: T, offset: T) -> T:
        """Computes inductor_prims.random with mode="rand".  offset has dtype int32."""
        raise NotImplementedError

    def rand_eager(
        self, seed: T, base_offset: T, threads_per_round: T, tid: T, vec: T
    ) -> T:
        """Computes inductor_prims.random with mode="rand_eager".  offset has dtype int32."""
        raise NotImplementedError

    def randn(self, seed: T, offset: T) -> T:
        """Computes inductor_prims.random with mode="randn".  offset has dtype int32."""
        raise NotImplementedError

    def randint64(self, seed: T, offset: T, low: T, high: T) -> T:
        """Computes inductor_prims.randint.  offset has dtype int32."""
        raise NotImplementedError

    def masked(self, mask: T, body: Callable[[], T], other: T) -> T:
        """
        Computes body, but only perform loads/stores if the boolean mask
        evaluates to true.  For example, you would use this if you needed to
        perform an indirect load that may not be valid on some elements;
        without masking, invalid accesses can cause IMAs.  When mask is true,
        the result is the result of body; otherwise it is other. Here, `other`
        needs to be a constant.

        Contrast this with ops.where, which can multiplex between two values
        that have been unconditionally computed.
        """
        raise NotImplementedError

    def where(self, condition: T, input: T, other: T) -> T:
        """
        Computes torch.where: when condition is true, return input; otherwise return other.
        """
        raise NotImplementedError

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> T:
        """
        Converts a sympy expression into a scalar suitable for indexing memory.
        The kernel decides the actual computation dtype (typically int32) — the
        ``dtype`` argument is advisory and is not honored when it would conflict
        with the kernel's indexing dtype. For values that must respect the
        requested dtype, use ``value_expr`` instead.
        """
        raise NotImplementedError

    def value_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> T:
        """
        Converts a sympy expression into a scalar of type ``dtype`` that
        participates in tensor value computation. Unlike ``index_expr``, the
        result dtype is respected, so this is the right op when the user
        explicitly requested the dtype (e.g. ``arange(dtype=torch.int64)``
        whose result is added to a tensor rather than used as an index).
        """
        raise NotImplementedError

    def to_dtype(
        self,
        x: T,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> T:
        """
        Convert x to dtype.  src_dtype can be optionally set to specify what the original
        dtype of x was, which can improve code generation (used by torch to(dtype=dtype)).
        """
        raise NotImplementedError

    def trunc_to_int(self, x: T, dtype: torch.dtype) -> T:
        """
        Convert x to dtype with truncation semantics (similar to how the int
        constructor works in Python).  In Inductor codegen, this just decays
        to trunc and then to_dtype, but this composite operation helps
        roundtrips for Sympy evaluation.

        dtype is taken as an explicit parameter because the desired output
        dtype is typically the index dtype, which may vary between int32 and
        int64 depending on if we've shown that all the indexing operations can
        be done in int32.
        """
        raise NotImplementedError

    def ceil_to_int(self, x: T, dtype: torch.dtype) -> T:
        """
        Convert x to dtype with ceiling semantics.  See also trunc_to_int.
        """
        raise NotImplementedError

    def floor_to_int(self, x: T, dtype: torch.dtype) -> T:
        """
        Convert x to dtype with ceiling semantics.  See also trunc_to_int.
        """
        raise NotImplementedError

    def round_to_int(self, x: T, dtype: torch.dtype) -> T:
        """
        Convert x to dtype with round-to-even semantics.  See also trunc_to_int.
        """
        raise NotImplementedError

    def to_dtype_bitcast(self, x: T, dtype: torch.dtype, src_dtype: torch.dtype) -> T:
        """
        Reinterpret cast x to dtype (reinterpreting the bits in memory as another dtype.)
        src_dtype must be the original type of x.
        """
        raise NotImplementedError

    def identity(self, x: T) -> T:
        """
        Returns x as is.  This is used to trigger CSE.
        """
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These operations are only available in a "kernel" context.  Check
    # torch._inductor.codegen.common.CSEProxy for their typical implementation
    # in op handler (routing to their respective implementations in the kernel
    # handler)
    #
    # Importantly, inside a kernel, indexing and mask variables are available
    # in scope, which are typically used by sympy.Expr indexing.

    def indirect_indexing(
        self, x: T, size: sympy.Expr, check: bool = True, wrap_neg=True
    ) -> sympy.Expr:
        """
        Convert an integral x into a sympy.Expr that can be subsequently used in
        indexing computation.  'size' represents an upper bound on what valid
        indexes can be; when 'check' is True, we check that the x is in bounds.

        NB: This is typically mandatory to implement for any analysis, because you
        MUST return a valid sympy.Expr of some sort (even if it's a meaningless symbol).
        """
        raise NotImplementedError

    def load(self, name: str, index: sympy.Expr) -> T:
        """
        Load from the memory location 'name', offset by some indexing expression 'index'.
        """
        raise NotImplementedError

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: T,
        mode: StoreMode = None,
    ) -> None:
        """
        Store 'value' to the memory location 'name' offset by 'expr'.  If
        specified, 'mode' can require the store to be an atomic addition.
        """
        raise NotImplementedError

    # TODO: Better explain how the "collective" semantics of these ops;
    # remember that the input value is a scalar, you can't reduce on it in the
    # traditional sense!
    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: T,
    ) -> T | tuple[T, ...]:
        """
        Perform a 'reduction_type' reduction on 'value' of dtype 'src_dtype',
        using 'dtype' as the accumulation dtype for the reduction.  The result
        is an intermediate computation which should be stored to the final
        location using 'ops.store_reduction'.

        Valid reduction types are .  For Welford reduction types, this
        function returns multiple outputs; consult reduction_num_outputs to
        determine the amount in metaprogramming applications.
        """
        raise NotImplementedError

    # TODO: in practice, this seems to actually return None, but not returning
    # a T makes common __getattr__ idioms not type correctly.  Figure out if
    # this should be returning something.
    def store_reduction(self, name: str, index: sympy.Expr, value: T) -> None:
        """
        Store the fully accumulated result of 'reduction' to the memory
        location 'name' offset by 'expr'.
        """
        raise NotImplementedError

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[[tuple[T, ...], tuple[T, ...]], tuple[T, ...]],
        values: tuple[T, ...],
    ) -> tuple[T, ...]:
        """
        Perform an associative scan on 'value'.
        """
        # TODO: Improve the description with some pseudocode
        raise NotImplementedError

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[T, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[T, ...]:
        """
        Sort values along the reduction dimension.
        """
        raise NotImplementedError

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
        # See [Note: Inductor bucketize op]
        raise NotImplementedError

    def partial_accumulate(
        self,
        name: str,
        reduction_type: ReductionType,
        value: T,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The following ops have semantics that correspond exactly to the torch
    # operation with the same corresponding name.

    def abs(self, x0: T) -> T:
        raise NotImplementedError

    def exp(self, x0: T) -> T:
        raise NotImplementedError

    def exp2(self, x0: T) -> T:
        raise NotImplementedError

    def expm1(self, x0: T) -> T:
        raise NotImplementedError

    def sqrt(self, x0: T) -> T:
        raise NotImplementedError

    def relu(self, x0: T) -> T:
        raise NotImplementedError

    def minimum(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def maximum(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def fmaximum(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def cos(self, x0: T) -> T:
        raise NotImplementedError

    def sin(self, x0: T) -> T:
        raise NotImplementedError

    def lgamma(self, x0: T) -> T:
        raise NotImplementedError

    def erf(self, x0: T) -> T:
        raise NotImplementedError

    def cosh(self, x0: T) -> T:
        raise NotImplementedError

    def sinh(self, x0: T) -> T:
        raise NotImplementedError

    def acos(self, x0: T) -> T:
        raise NotImplementedError

    def acosh(self, x0: T) -> T:
        raise NotImplementedError

    def asin(self, x0: T) -> T:
        raise NotImplementedError

    def asinh(self, x0: T) -> T:
        raise NotImplementedError

    def atan2(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def atan(self, x0: T) -> T:
        raise NotImplementedError

    def atanh(self, x0: T) -> T:
        raise NotImplementedError

    def copysign(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def erfc(self, x0: T) -> T:
        raise NotImplementedError

    def erfinv(self, x0: T) -> T:
        raise NotImplementedError

    def frexp(self, x0: T):
        raise NotImplementedError

    def hypot(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def log10(self, x0: T) -> T:
        raise NotImplementedError

    def log2(self, x0: T) -> T:
        raise NotImplementedError

    def ldexp(self, x0: T, n: T) -> T:
        raise NotImplementedError

    def nextafter(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def logical_and(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def logical_not(self, x0: T) -> T:
        raise NotImplementedError

    def logical_or(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def logical_xor(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def bitwise_and(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def bitwise_not(self, x0: T) -> T:
        raise NotImplementedError

    def bitwise_or(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def bitwise_xor(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def bitwise_left_shift(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def bitwise_right_shift(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def rsqrt(self, x0: T) -> T:
        raise NotImplementedError

    def log1p(self, x0: T) -> T:
        raise NotImplementedError

    def tan(self, x0: T) -> T:
        raise NotImplementedError

    def tanh(self, x0: T) -> T:
        raise NotImplementedError

    def sigmoid(self, x0: T) -> T:
        raise NotImplementedError

    def signbit(self, x0: T) -> T:
        raise NotImplementedError

    def fmod(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def log(self, x0: T) -> T:
        raise NotImplementedError

    def isinf(self, x0: T) -> T:
        raise NotImplementedError

    def isnan(self, x0: T) -> T:
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    # This rounds half to even to break ties
    def round(self, x0: T) -> T:
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    def floor(self, x0: T) -> T:
        raise NotImplementedError

    def sign(self, x0: T) -> T:
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    def trunc(self, x0: T) -> T:
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    def ceil(self, x0: T) -> T:
        raise NotImplementedError

    def neg(self, x0: T) -> T:
        raise NotImplementedError

    def reciprocal(self, x0: T) -> T:
        raise NotImplementedError

    def eq(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def ne(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def lt(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def gt(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def le(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def ge(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def add(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def sub(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def mul(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    def pow(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def and_(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def or_(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def xor(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    # These are metaprogrammed by MockHandler._init_cls
    def lshift(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    def rshift(self, x0: T, x1: T) -> T:
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These are "special" operators.  These only exist if the target
    # language actually supports the operator.  Keep this in sync with
    # pointwise_overrides_data.

    def airy_ai(self, x: T) -> T:
        raise NotImplementedError

    def bessel_j0(self, x: T) -> T:
        raise NotImplementedError

    def bessel_j1(self, x: T) -> T:
        raise NotImplementedError

    def bessel_y0(self, x: T) -> T:
        raise NotImplementedError

    def bessel_y1(self, x: T) -> T:
        raise NotImplementedError

    def digamma(self, x: T) -> T:
        raise NotImplementedError

    def erfcx(self, x: T) -> T:
        raise NotImplementedError

    def fma(self, x: T, y: T, z: T) -> T:
        raise NotImplementedError

    def mul_rn(self, x: T, y: T) -> T:
        """Multiplication with round-to-nearest, preventing fusion with subsequent ops."""
        raise NotImplementedError

    def igamma(self, x: T, y: T) -> T:
        raise NotImplementedError

    def igammac(self, x: T, y: T) -> T:
        raise NotImplementedError

    def gammainc(self, x: T, y: T) -> T:
        raise NotImplementedError

    def gammaincc(self, x: T, y: T) -> T:
        raise NotImplementedError

    def i0(self, x: T) -> T:
        raise NotImplementedError

    def i0e(self, x: T) -> T:
        raise NotImplementedError

    def i1(self, x: T) -> T:
        raise NotImplementedError

    def i1e(self, x: T) -> T:
        raise NotImplementedError

    def log_ndtr(self, x: T) -> T:
        raise NotImplementedError

    def modified_bessel_i0(self, x: T) -> T:
        raise NotImplementedError

    def modified_bessel_i1(self, x: T) -> T:
        raise NotImplementedError

    def modified_bessel_k0(self, x: T) -> T:
        raise NotImplementedError

    def modified_bessel_k1(self, x: T) -> T:
        raise NotImplementedError

    def ndtr(self, x: T) -> T:
        raise NotImplementedError

    def ndtri(self, x: T) -> T:
        raise NotImplementedError

    def polygamma(self, x: T, y: T) -> T:
        raise NotImplementedError

    def scaled_modified_bessel_k0(self, x: T) -> T:
        raise NotImplementedError

    def scaled_modified_bessel_k1(self, x: T) -> T:
        raise NotImplementedError

    def spherical_bessel_j0(self, x: T) -> T:
        raise NotImplementedError

    def zeta(self, x: T, y: T) -> T:
        raise NotImplementedError

    def chebyshev_polynomial_t(self, x: T, y: T) -> T:
        raise NotImplementedError

    def chebyshev_polynomial_u(self, x: T, y: T) -> T:
        raise NotImplementedError

    def chebyshev_polynomial_v(self, x: T, y: T) -> T:
        raise NotImplementedError

    def chebyshev_polynomial_w(self, x: T, y: T) -> T:
        raise NotImplementedError

    def legendre_polynomial_p(self, x: T, y: T) -> T:
        raise NotImplementedError

    def shifted_chebyshev_polynomial_t(self, x: T, y: T) -> T:
        raise NotImplementedError

    def shifted_chebyshev_polynomial_u(self, x: T, y: T) -> T:
        raise NotImplementedError

    def shifted_chebyshev_polynomial_v(self, x: T, y: T) -> T:
        raise NotImplementedError

    def shifted_chebyshev_polynomial_w(self, x: T, y: T) -> T:
        raise NotImplementedError

    def hermite_polynomial_h(self, x: T, y: T) -> T:
        raise NotImplementedError

    def hermite_polynomial_he(self, x: T, y: T) -> T:
        raise NotImplementedError

    def laguerre_polynomial_l(self, x: T, y: T) -> T:
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These operators are a bit special, because they are conventionally
    # natively supported in both Python and C, but the semantics differ so
    # care must be taken

    def truncdiv(self, x0: T, x1: T) -> T:
        """C-style trunc division between integers only.  Computes the true
        division of two numbers and rounds the result to zero.
        """
        raise NotImplementedError

    def floordiv(self, x0: T, x1: T) -> T:
        """Python-style floor division between integers only.  Computes the
        true division of two numbers and floors the result.  If you want
        floor division for floats, do regular truediv and floor the result.
        """
        raise NotImplementedError

    def truediv(self, x0: T, x1: T) -> T:
        """True division between floats.  Integer inputs are NOT valid.  To
        do Python-style (int, int) -> float division, use int_truediv"""
        raise NotImplementedError

    def div_rn(self, x0: T, x1: T) -> T:
        """Division with round-to-nearest rounding mode.  Used for matching
        eager CUDA semantics where division uses IEEE round-to-nearest."""
        raise NotImplementedError

    def int_truediv(self, x0: T, x1: T) -> T:
        """True division between integers.  This is NOT the same as promoting
        to float and doing integer division, there is a bespoke algorithm for
        doing the division in higher precision than the above.
        """
        raise NotImplementedError

    def mod(self, x0: T, x1: T) -> T:
        """C-style modulus, take sign from LHS (x0)."""
        raise NotImplementedError

    def remainder(self, x0: T, x1: T) -> T:
        """Python-style modulus, take sign from RHS (x1)."""
        raise NotImplementedError

    def square(self, x0: T) -> T:
        raise NotImplementedError

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        raise NotImplementedError

    # halide-only
    def halide_clamp(self, value: T, size: sympy.Expr, check: bool) -> T:
        raise NotImplementedError

    # triton-only
    def dot(self, x: T, y: T) -> T:
        raise NotImplementedError

    # triton-only
    def inline_asm_elementwise(
        self,
        *inputs: T,
        asm: str,
        constraints: str | None = None,
        dtype: torch.dtype = torch.float32,
        is_pure: bool = True,
        pack: int = 1,
        input_dtypes: tuple[torch.dtype, ...] | None = None,
    ) -> T:
        raise NotImplementedError

    def output(self, *args: T) -> None:
        """This is a fake op used in analysis but not codegen"""
        raise NotImplementedError

    def placeholder(self, index: int) -> T:
        """This is a fake op used in analysis but not codegen"""
        raise NotImplementedError

    def device_assert_async(self, cond: T, msg: str) -> T:
        raise NotImplementedError


_ignore_op_re = re.compile(r"_.*|paren").fullmatch


def list_ops(cls: type[Any]):
    return OrderedSet([x for x in dir(cls) if not _ignore_op_re(x)])


OP_NAMES = list_ops(OpsHandler)


class DefaultHandler(OpsHandler[Any]):
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """
        Default implementation for all ops.  Override in a subclass to
        provide generic op behavior.

        Args:
            name: name of the op, see OpHandler.{name}
            args: positional args passed to the op
            kwargs: keyword args passed to the op

        Returns:
            return value of the op

        """
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        def fallback(*args: Any, **kwargs: Any) -> Any:
            return self._default(name, args, kwargs)

        # would like to remove this function entirely, but it's used in MTIA backend
        warnings.warn(f"undefined OpHandler.{name}, please add missing op schema")
        return fallback

    @staticmethod
    def _call_default(target: str):
        def call_default(self, *args, **kwargs):
            return self._default(target, args, kwargs)

        call_default.__name__ = target
        return call_default

    @classmethod
    def _init_cls(cls):
        """
        Here we codegen many functions of the form:

            def add(self, a, b):
                return self._default('add', (a, b), {})

        and install them in cls.  This is the same as _call_default above,
        but is about 1.2x faster since CPython varargs parsing is slow.
        """
        code = StringIO()
        for target in OP_NAMES:
            sig = inspect.signature(getattr(OpsHandler, target))
            if all(
                p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and p.default is inspect.Parameter.empty
                for p in sig.parameters.values()
            ):
                self_arg, *args = sig.parameters.keys()
                if self_arg != "self":
                    raise AssertionError(
                        f"expected first parameter 'self', got {self_arg!r}"
                    )
                code.write(
                    f"""
                    def {target}(self, {", ".join(args)}):
                        return self._default({target!r}, ({", ".join(args)}, ), {{}})
                    """.strip()
                )
                code.write("\n\n")
            else:
                # slower fallback for ops with default or variadic arguments
                setattr(cls, target, cls._call_default(target))

        ctx: dict[str, Any] = {}
        exec(code.getvalue(), ctx)
        for target, impl in ctx.items():
            if target in OP_NAMES:
                setattr(cls, target, impl)


DefaultHandler._init_cls()


class NoopHandler(DefaultHandler):
    name = "NoopHandler"

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return None

    @staticmethod
    def masked(mask, body, other) -> None:
        return None

    @staticmethod
    # pyrefly: ignore [bad-override]
    def frexp(x) -> tuple[None, None]:
        return (None, None)

    @staticmethod
    def scan(dtypes, combine_fn, values) -> tuple[None, ...]:
        return (None,) * len(values)

    @staticmethod
    def sort(dtypes, values, stable, descending) -> tuple[None, ...]:
        return (None,) * len(values)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def indirect_indexing(index_var, size, check=True, wrap_neg=True) -> sympy.Symbol:
        return sympy.S.Zero


class BasicMathOpsMixin:
    @staticmethod
    def add(a, b):
        return f"{a} + {b}"

    @staticmethod
    def sub(a, b):
        return f"{a} - {b}"

    @staticmethod
    def mul(a, b):
        return f"{a} * {b}"

    @staticmethod
    def floordiv(a, b):
        return f"{a} // {b}"

    @staticmethod
    def truediv(a, b):
        return f"{a} / {b}"

    @staticmethod
    def mod(a, b):
        # careful, depending on target semantics varies
        return f"{a} % {b}"

    @staticmethod
    def pow(a, b):
        return f"{a} ** {b}"

    @staticmethod
    def lshift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def rshift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def and_(a, b):
        return f"{a} & {b}"

    @staticmethod
    def or_(a, b):
        return f"{a} | {b}"

    @staticmethod
    def xor(a, b):
        return f"{a} ^ {b}"

    @staticmethod
    def eq(a, b):
        return f"{a} == {b}"

    @staticmethod
    def ne(a, b):
        return f"{a} != {b}"

    @staticmethod
    def lt(a, b):
        return f"{a} < {b}"

    @staticmethod
    def gt(a, b):
        return f"{a} > {b}"

    @staticmethod
    def le(a, b):
        return f"{a} <= {b}"

    @staticmethod
    def ge(a, b):
        return f"{a} >= {b}"

    @staticmethod
    def neg(a):
        return f"-{a}"


class MockHandler(BasicMathOpsMixin, DefaultHandler):
    name = "MockHandler"

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        fargs = [*map(_arg_str, args)]
        for k, v in kwargs.items():
            fargs.append(f"{k}={_arg_str(v)}")
        return f"ops.{name}({', '.join(fargs)})"

    @staticmethod
    def masked(mask, body, other) -> str:
        return f"ops.masked({mask}, {body()}, {other})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def frexp(x):
        return (f"ops.frexp({x})[0]", f"ops.frexp({x})[1]")

    @staticmethod
    def scan(dtypes, combine_fn, values):
        return tuple(
            f"ops.scan({dtypes}, {combine_fn}, {values})[{i}]"
            for i in range(len(values))
        )

    @staticmethod
    def sort(dtypes, values, stable, descending):
        return tuple(
            f"ops.sort({dtypes}, {values}, stable={stable}, descending={descending})[{i}]"
            for i in range(len(values))
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def indirect_indexing(index_var, size, check=True, wrap_neg=True) -> sympy.Symbol:
        return sympy_index_symbol(str(index_var))


class KernelFormatterHandler(DefaultHandler):
    def __init__(self, parent_handler: OpsHandler[Any]):
        self.parent_handler = parent_handler
        self._output = IndentedBuffer(1)
        self.var_counter = itertools.count()

    @staticmethod
    def ir_to_string(ir_fn, index, rindex=None) -> str:
        from .ir import FlexibleLayout
        from .virtualized import V

        args = [index, rindex] if rindex is not None else [index]
        names = ["index", "rindex"] if rindex is not None else ["index"]
        formatter = KernelFormatterHandler(MockHandler())

        with formatter._output.indent(-1):
            formatter._output.writeline(f"def inner_fn({', '.join(names)}):")
        for name, arg in zip(names, args):
            if arg:
                lhs = ", ".join(
                    [
                        str("_" if isinstance(v, (int, sympy.Integer)) else v)
                        for v in arg
                    ]
                )
                formatter._output.writeline(f"{lhs} = {name}")

        with (
            V.set_ops_handler(formatter),
            patch.object(FlexibleLayout, "allow_indexing", True),
        ):
            result = ir_fn(*args)
            return formatter.getvalue(result)

    def indirect_indexing(self, *args, **kwargs) -> sympy.Symbol:
        return self.parent_handler.indirect_indexing(*args, **kwargs)

    def _write(self, line):
        # replace line with a new variable name
        varname = f"tmp{next(self.var_counter)}"
        self._output.writeline(f"{varname} = {line}")
        return varname

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return pytree.tree_map(
            self._write, getattr(self.parent_handler, name)(*args, **kwargs)
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: str | tuple[str, ...],
    ) -> str | tuple[str, ...]:
        line = self.parent_handler.reduction(dtype, src_dtype, reduction_type, value)
        num_values = reduction_num_outputs(reduction_type)
        varnames = [f"tmp{next(self.var_counter)}" for _ in range(num_values)]
        self._output.writeline(f"{','.join(varnames)} = {line}")
        return tuple(varnames) if num_values > 1 else varnames[0]

    def getvalue(self, result):
        self._output.writeline(f"return {result}")
        return self._output.getvalue()


class WrapperHandler(DefaultHandler):
    def __init__(self, inner: OpsHandler[Any]):
        self._inner = inner

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return getattr(self._inner, name)(*args, **kwargs)


class AddParenHandler(WrapperHandler):
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        val = getattr(self._inner, name)(*args, **kwargs)
        if not val or isinstance(val, (sympy.Expr, tuple, list)):
            return val
        return f"({val})"


class OpCountResult(NamedTuple):
    num_ops: int
    used_ops: OrderedSet[str]
    read_buffers: list[str]
    nontrivial_read_count: int


class OpCounterCSE(DefaultHandler):
    """Shim to count how many ops are used"""

    def __init__(self, inner: OpsHandler[Any]):
        super().__init__()
        self.parent_handler = inner
        self.op_count = 0
        self.var_names: dict[str, str] = {}
        self._used_ops: OrderedSet[str] = OrderedSet()
        self._read_names: list[str] = []
        self._nontrivial_read_count = 0

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._used_ops.add(name)
        return pytree.tree_map(
            self._update_count, getattr(self.parent_handler, name)(*args, **kwargs)
        )

    def _update_count(self, val):
        varname = self.var_names.get(val)
        if not varname:
            varname = f"tmp{self.op_count}"
            self.op_count += 1
            self.var_names[val] = varname
        return varname

    def indirect_indexing(self, *args, **kwargs):
        self._used_ops.add("indirect_indexing")
        return self.parent_handler.indirect_indexing(*args, **kwargs)

    def load(self, name: str, index: sympy.Expr) -> str:
        val = self.parent_handler.load(name, index)
        if val not in self.var_names:
            self._used_ops.add("load")
            self._read_names.append(name)
            if not isinstance(index, (sympy.Integer, int)):
                self._nontrivial_read_count += 1
        return self._update_count(val)

    def load_seed(self, name: str, offset: T):
        val = self.parent_handler.load_seed(name, offset)
        if val not in self.var_names:
            self._used_ops.add("load_seed")
            self._read_names.append(name)
        return self._update_count(val)

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
        val = self.parent_handler.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )
        if val not in self.var_names:
            self._used_ops.add("bucketize")
            self._read_names.append(boundaries[0])
            if sorter is not None:
                self._read_names.append(sorter[0])
        return self._update_count(val)

    def getvalue(self):
        return OpCountResult(
            self.op_count, self._used_ops, self._read_names, self._nontrivial_read_count
        )


class ExtractConstantsHandler(NoopHandler):
    def __init__(self, device: torch.device | None):
        self.device = device

    def constant(self, value: Any, dtype: torch.dtype) -> torch._inductor.ir.Constant:
        from torch._inductor import ir

        return ir.Constant(
            value=value, dtype=dtype, device=self.device or torch.get_default_device()
        )


class SimpleCSEHandler(WrapperHandler):
    """Wraps the underlying handler with a CSE pass

    NOTE: Compared to codegen level CSE this is simplified as it
    doesn't support stores which require load cache invalidation.
    """

    def __init__(self, inner: Any):
        super().__init__(inner)
        self.cse_cache: dict[str, Any | tuple[Any, ...]] = {}
        self.mock = MockHandler()

    def indirect_indexing(self, *args, **kwargs) -> sympy.Expr:
        return super().indirect_indexing(*args, **kwargs)  # type: ignore[misc]

    def store(self, *args, **kwargs) -> None:
        raise NotImplementedError("store not implemented")

    def store_reduction(self, *args, **kwargs) -> None:
        raise NotImplementedError("store not implemented")

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        key = getattr(self.mock, name)(*args, **kwargs)
        val = self.cse_cache.get(key)
        if val is not None:
            return val

        val = getattr(self._inner, name)(*args, **kwargs)
        self.cse_cache[key] = val
        return val

    def device_assert_async(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )
<|endoftext|>"""
# Inductor Pattern Matcher

The pattern matcher enables search/replace within an FX graph.

The main entrypoint to the pattern matcher is register_replacement(). Given a
search function and a replacement function this will register a replacement with
a pass (such as torch._inductor.fx_passes.joint_graph.patterns).

Internally the pattern matcher represents patterns as a graph (a DAG). Creating
new patterns manually as a graph is cumbersome and error-prone so the standard
way to create patterns (using register_replacement()) is to provide a search
function and a replacement function which is traced and converted into a graph.

Because the search functions are built somewhat generic (they tend to ignore
tensor sizes, for example) register_replacement() allows you to specify an
`extra_check` function which performs additional checks to verify that the
matched pattern fully matches before returning it.

## Precompiled Patterns

New patterns are added using register_replacement(). Patterns added in this way
can have a compile-time overhead because they need to be traced before
use. Patterns can be precompiled and added using gen_register_replacement()
instead. To do this you call gen_register_replacement() instead of
register_replacement(). The arguments are the same except for an additional
unique name which is used as a lookup key.

## Internals

The match DAG is represented by a graph of `PatternExpr` nodes. Each PatternExpr
implements a `_match` method which returns either a `Match` object for a
successful match or a `FailedMatch` object for a failure to match.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
import inspect
import itertools
import logging
import math
import operator
import os
import re
import textwrap
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Collection, Generator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, Protocol, TypeVar
from typing_extensions import Self, TypeIs

import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch._subclasses.fake_tensor import (
    is_fake_tensor,
    maybe_get_fake_constant,
    unset_fake_temporarily,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true
from torch.fx.graph_module import _get_attr
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.traceback import preserve_node_meta
from torch.utils._ordered_set import OrderedSet

from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensor, FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

Constant = Any
NodeOrConstant = Constant | torch.fx.Node

backend = os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_BACKEND", "inductor")


_debug_nodes_cache: bool | OrderedSet[str] | None = None
_debug_nodes_env_value_cache: str | None = None


def _should_debug_node(node_name: str) -> bool:
    def _get_debug_nodes() -> bool | OrderedSet[str]:
        global _debug_nodes_cache, _debug_nodes_env_value_cache

        def parse_debug_env(env_value: str | None) -> bool | OrderedSet[str]:
            if not env_value:
                return False
            if env_value == "all":
                return True
            return OrderedSet(env_value.split(","))

        current_env = os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG")

        # Recompute only if env changed
        if current_env != _debug_nodes_env_value_cache or _debug_nodes_cache is None:
            _debug_nodes_cache = parse_debug_env(current_env)
            _debug_nodes_env_value_cache = current_env

        return _debug_nodes_cache

    debug_nodes = _get_debug_nodes()
    if isinstance(debug_nodes, bool):
        return debug_nodes
    return node_name in debug_nodes


class SearchFn(Protocol):
    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class ReplaceFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class TraceFn(Protocol):
    def __call__(
        self, fn: SearchFn | ReplaceFn, *args: Any, **kwargs: Any
    ) -> torch.fx.GraphModule: ...


T = TypeVar("T")

# What's a better name for this?
FnsType = torch.fx.node.Target | str


class Multiple:
    def __init__(self) -> None:
        # Ensure we're really a singleton.
        if "MULTIPLE" in globals() and self is not MULTIPLE:
            raise AssertionError("Multiple is not a singleton")


# Sentinel indicating multiple quantities can be matched
MULTIPLE = Multiple()


def _transfer_meta(
    new_meta: dict[str, Any], old_node: torch.fx.Node, pass_name: str = ""
) -> None:
    from torch.fx.traceback import NodeSource, NodeSourceAction

    # Transfer metadata after pattern matching occurs.
    # Copies _COPY_META_FIELDS, stack_trace, and (if missing) val/tensor_meta.
    if config.effective_provenance_tracking_level() == 1:
        new_from_node = new_meta.get("from_node", []).copy()
        new_from_node.append(NodeSource(old_node, pass_name, NodeSourceAction.REPLACE))
        new_meta.update(
            (k, v)
            for k, v in old_node.meta.items()
            if k in torch.fx.proxy._COPY_META_FIELDS
        )
        new_meta["from_node"] = new_from_node
    else:
        new_meta.update(
            (k, v)
            for k, v in old_node.meta.items()
            if k in torch.fx.proxy._COPY_META_FIELDS
        )
    if "stack_trace" in old_node.meta:
        new_meta["stack_trace"] = old_node.meta["stack_trace"]
    # Copy val/tensor_meta only when the new node doesn't already have them
    # (e.g. from tracing the replacement graph). Don't overwrite if present
    # since the replacement's own val is more accurate.
    if "val" not in new_meta and "val" in old_node.meta:
        new_meta["val"] = old_node.meta["val"]
    if "tensor_meta" not in new_meta and "tensor_meta" in old_node.meta:
        new_meta["tensor_meta"] = old_node.meta["tensor_meta"]


def _common_custom_context(nodes: Sequence[torch.fx.Node]) -> dict[str, Any]:
    if not nodes:
        return {}

    stream = nodes[0].meta.get("custom", {}).get("stream", 0)
    mempool = nodes[0].meta.get("custom", {}).get("mempool")
    mempool_device = nodes[0].meta.get("custom", {}).get("mempool_device")
    if any(
        (
            node.meta.get("custom", {}).get("stream", 0),
            node.meta.get("custom", {}).get("mempool"),
            node.meta.get("custom", {}).get("mempool_device"),
        )
        != (stream, mempool, mempool_device)
        for node in nodes[1:]
    ):
        return {}

    context: dict[str, Any] = {}
    if stream != 0 or any("stream" in node.meta.get("custom", {}) for node in nodes):
        context["stream"] = stream
    if mempool is not None:
        context["mempool"] = mempool
        context["mempool_device"] = mempool_device
    return context


def _merge_custom_context(
    new_meta: dict[str, Any], custom_context: dict[str, Any]
) -> None:
    # Replacement nodes inherit user stream/mempool context only at explicit
    # graph transform hooks. Other transforms must preserve meta["custom"] or
    # avoid moving context-tagged values across boundaries.
    if not custom_context:
        return
    custom = new_meta.setdefault("custom", {})
    for key, value in custom_context.items():
        custom.setdefault(key, value)


class Match:
    """
    Represents a successfully matched pattern.

    The `Match` object is returned to represent a successfully matched
    pattern. Included in the Match are the pattern that was matched, the graph
    nodes matched, and any args that were used during the matching.

    The args and kwargs are specific to the type of pattern that was matched and
    provide hints about what was matched.
    """

    pattern: PatternExpr
    args: list[Any]
    kwargs: dict[str, Any]
    nodes: list[torch.fx.Node]
    targets: dict[_TargetExpr, torch.fx.node.Target]
    ctx: MatchContext
    replacement_graph: torch.fx.GraphModule | None

    def __init__(
        self,
        ctx: MatchContext,
        pattern: PatternExpr,
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.pattern = pattern
        # The input nodes that must be passed in to the result
        self.args = list(args or [])
        self.kwargs = kwargs or {}
        # The nodes matched in this expression
        self.nodes = []
        # Mapping CallFunction to the node.target
        self.targets = {}
        self.ctx = ctx
        self.replacement_graph = None

    @property
    def graph(self) -> torch.fx.Graph:
        return self.ctx.graph

    def extend(self, other: Match) -> None:
        if self.kwargs:
            for key in OrderedSet(self.kwargs.keys()) & OrderedSet(other.kwargs.keys()):
                if self.kwargs[key] != other.kwargs[key]:
                    raise FailedMatch("kwarg mismatch: {}", key)
        self.args.extend(other.args)
        self.nodes.extend(other.nodes)
        self.kwargs.update(other.kwargs)
        self.targets.update(other.targets)

    def bundle(self) -> Match:
        # Wrap args in an extra list
        self.args = [tuple(self.args)] if self.args else []
        return self

    def __repr__(self) -> str:
        return f"Match(..., {self.args}, {self.kwargs})"

    def erase_nodes(self) -> None:
        graph = self.graph
        for n in reversed(self.nodes):
            if not n._erased and not n.users:
                graph.erase_node(n)

    def output_nodes(self) -> list[torch.fx.Node | None]:
        return [
            (self.ctx.pattern_to_node[p] if p is not None else None)
            for p in self.ctx.outputs
        ]

    def output_node(self) -> torch.fx.Node:
        return next(p for p in self.output_nodes() if p)

    def replace_with_graph(
        self, replacement_graph: torch.fx.Graph, args: Sequence[Any]
    ) -> list[torch.fx.Node]:
        return ReplacementPatternEntry.replace_with_graph(
            self, self.ctx.graph, replacement_graph, args
        )

    def replace_by_example(
        self,
        replacement_fn: ReplaceFn,
        args: Sequence[Any],
        trace_fn: TraceFn | None = None,
        run_functional_passes: bool = True,
    ) -> list[torch.fx.Node]:
        """Replace with a graph generated by tracing the replacement_fn.

        Args:
            run_functional_passes (bool). If we should run passes that
                assume functional IR (like DCE, remove_noop_ops), on the
                replacement graph.

        """
        from torch._inductor.virtualized import NullHandler, V

        context = (
            V.fake_mode
            if (not isinstance(V.fake_mode, NullHandler) or (V.fake_mode is None))
            else contextlib.nullcontext()
        )

        def should_propagate_eager_input_vals(nodes: list[torch.fx.Node]) -> bool:
            if len(nodes) != 1:
                return False
            node = nodes[0]
            if "eager_input_vals" not in node.meta:
                return False
            return node.target in OrderedSet(
                [
                    torch.ops.higher_order.auto_functionalized,
                    torch.ops.higher_order.auto_functionalized_v2,
                ]
            )

        # pyrefly: ignore [bad-context-manager]
        with context:
            if trace_fn is None:
                trace_fn = functools.partial(
                    fwd_only, run_functional_passes=run_functional_passes
                )

            if should_propagate_eager_input_vals(self.nodes):
                # Our strategy is:
                # 1) trace out the graph with eager_input_vals (which have accurate eager-mode metadata)
                # 2) trace out the graph with vals (which have the accurate Inductor metadata)
                # 3) Propagate the eager_input_vals from the first graph to the second.
                # 4) Use the second graph as the replacement graph.

                # Construct a map of node -> FakeTensor val in eager_input_vals
                node_to_val = {}

                fake_args, fake_kwargs = self.nodes[0].meta["eager_input_vals"]
                fake_kwargs = {**fake_kwargs}
                match_args, match_kwargs = tuple(self.args), self.kwargs

                def record(node: torch.fx.Node, val: Any) -> None:
                    if isinstance(node, torch.fx.Node):
                        node_to_val[node] = val

                torch.utils._pytree.tree_map(
                    record, (match_args, match_kwargs), (fake_args, fake_kwargs)
                )
                # map args to their FakeTensor val in eager_input_vals
                example_vals = torch.fx.map_arg(args, lambda arg: node_to_val[arg])

                # first graph
                graph_with_eager_vals = trace_fn(replacement_fn, example_vals)

                # second graph
                example_vals = torch.fx.map_arg(args, lambda arg: arg.meta["val"])
                # pyrefly: ignore [bad-argument-type]
                replacement = trace_fn(graph_with_eager_vals, example_vals)

                # propagate metadata from first graph to second
                # NB: This assertion might not be true in general, but it is true for
                # the auto_functionalized use cases we have
                if len(graph_with_eager_vals.graph.nodes) != len(
                    replacement.graph.nodes
                ):
                    raise AssertionError(
                        "graph node count mismatch between eager and replacement"
                    )
                for old_node, new_node in zip(
                    graph_with_eager_vals.graph.nodes, replacement.graph.nodes
                ):
                    if "eager_input_vals" in old_node.meta:
                        new_node.meta["eager_input_vals"] = old_node.meta[
                            "eager_input_vals"
                        ]

            else:
                example_vals = torch.fx.map_arg(
                    args,
                    lambda arg: arg.meta["val"]
                    if "val" in arg.meta
                    else arg.meta["example_value"],
                )
                fake_mode = torch._dynamo.utils.detect_fake_mode(example_vals)
                if fake_mode is not None:

                    def _convert_to_fake_mode(it):
                        if is_fake_tensor(it) and fake_mode is not None:
                            return fake_mode.from_tensor(it)
                        return it

                    example_vals = torch.fx.node.map_aggregate(
                        example_vals, _convert_to_fake_mode
                    )
                replacement = trace_fn(replacement_fn, example_vals)
            if len(self.nodes) == 1:
                for n in replacement.graph.nodes:
                    _transfer_meta(
                        new_meta=n.meta,
                        old_node=self.nodes[0],
                        pass_name="replace_by_example",
                    )

            return ReplacementPatternEntry.replace_with_graph(
                self,
                self.ctx.graph,
                replacement,
                args,
            )


class FailedMatch(RuntimeError):
    """
    Represents a unsuccessful match.

    The `FailedMatch` object is returned to represent a failure to match a
    pattern.
    """

    format_string: str

    def __init__(self, format_string: str, *args: Any, **kwargs: Any) -> None:
        self.format_string = format_string
        # We want to construct error messages lazily instead of eagerly, as
        # constructing them eagerly can significantly worsen compile times.
        if len(format_string) > 200:
            raise RuntimeError(
                f"Format string too long - use lazy construction of strings instead. Format string is\n {format_string}"
            )
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.format_string.format(*self.args, **self.kwargs)

    def __bool__(self) -> bool:
        return False


MatchResult = Match | FailedMatch


def is_match(m: MatchResult) -> TypeIs[Match]:
    """
    TypeIs cannot act on `self`. Thus this function exists to let mypy
    recognize FailedMatch.__bool__ as a TypeIs.
    """
    return bool(m)


class MatchContext:
    """
    Internal state needed while running PatternExpr._match().
    """

    outputs: list[PatternExpr | None]
    pattern_to_node: dict[PatternExpr, torch.fx.Node | None]
    graph: torch.fx.Graph
    exclusive_node_set: list[NodeOrConstant]

    def __init__(
        self,
        outputs: list[PatternExpr | None],
        pattern_to_node: dict[PatternExpr, torch.fx.Node] | None = None,
        *,
        graph: torch.fx.Graph,
    ) -> None:
        self.outputs = outputs
        self.pattern_to_node = {} if pattern_to_node is None else dict(pattern_to_node)
        self.graph = graph
        self.exclusive_node_set = []

    def match(self, pattern: PatternExpr, node: NodeOrConstant) -> MatchResult:
        """wrapper to check reused nodes in patterns"""
        if pattern in self.pattern_to_node:
            if self.pattern_to_node[pattern] == node:
                return Match(self, pattern)  # already checked this node
            else:
                return FailedMatch("repeated pattern differs")
        m = pattern._match(node, self)
        if pattern in self.pattern_to_node:
            raise AssertionError("pattern already present in pattern_to_node")
        self.pattern_to_node[pattern] = node if m else None
        return m

    def filter_multi_user_patterns(self) -> dict[PatternExpr, torch.fx.Node]:
        return {
            pattern: node
            for pattern, node in self.pattern_to_node.items()
            if pattern.has_multiple_users() and node is not None
        }


class PatternExpr(ABC):
    """
    Base class for types of patterns.
    """

    @abstractmethod
    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult: ...

    def match(self, node: torch.fx.Node) -> MatchResult:
        try:
            return MatchContext([self], graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

    def has_multiple_users(self) -> bool:
        return False

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None, None, None]:
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]

    def pattern_eq(self, other: Any) -> bool:
        """
        Compare two `PatternExpr`s and return true if they are the
        same. Note this is NOT matching a pattern - it is comparing the pattern
        structures (for debugging).
        """
        return isinstance(other, self.__class__)


class Arg(PatternExpr):
    """
    Capture an arg which will become an input to the handler.  Args are
    passed in depth first order.
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        return Match(ctx, self, args=[node])  # matches anything


class Ignored(PatternExpr):
    """
    Match an arg, but don't pass it to handler
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        return Match(ctx, self)  # matches anything

    def __repr__(self) -> str:
        return "*"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        return "Ignored()"


def _get_fake_tensor_constant(value: torch.Tensor) -> torch.Tensor | None:
    if is_fake_tensor(value):
        return maybe_get_fake_constant(value)
    return value


def _tensor_values_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if torch.equal(a, b):
        return True
    if a.dtype.is_complex:
        return _tensor_values_equal(a.real, b.real) and _tensor_values_equal(
            a.imag, b.imag
        )
    if a.dtype.is_floating_point:
        return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all().item())
    return False


def _constant_values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        if (
            a.dtype != b.dtype
            or a.shape != b.shape
            or a.device != b.device
            or a.layout != b.layout
            or a.requires_grad != b.requires_grad
        ):
            return False
        if a.layout == torch.strided and (
            a.stride() != b.stride() or a.storage_offset() != b.storage_offset()
        ):
            return False
        a_constant = _get_fake_tensor_constant(a)
        b_constant = _get_fake_tensor_constant(b)
        if a_constant is None or b_constant is None:
            return False
        try:
            with unset_fake_temporarily():
                return _tensor_values_equal(a_constant, b_constant)
        except (RuntimeError, TypeError, ValueError):
            return False

    try:
        result = a == b
    except (RuntimeError, TypeError, ValueError):
        return False
    return result if isinstance(result, bool) else False


def _python_constant_repr(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "-float('inf')"
    if isinstance(value, complex):
        return f"complex({_python_constant_repr(value.real)}, {_python_constant_repr(value.imag)})"
    if isinstance(value, list):
        return f"[{', '.join(map(_python_constant_repr, value))}]"
    if isinstance(value, tuple):
        trailing_comma = "," if len(value) == 1 else ""
        return f"({', '.join(map(_python_constant_repr, value))}{trailing_comma})"
    return repr(value)


def _tensor_constant_repr(value: torch.Tensor) -> str:
    if value.layout != torch.strided:
        raise NotImplementedError(
            f"NYI: serializing get_attr tensor with layout {value.layout}"
        )
    dtype = value.dtype
    device = value.device
    if is_fake_tensor(value):
        constant = maybe_get_fake_constant(value)
        if constant is None:
            raise NotImplementedError("NYI: serializing fake get_attr tensor")
        data_value = constant
    else:
        data_value = value
    if value.requires_grad:
        raise NotImplementedError("NYI: serializing tensor that requires grad")
    if not value.is_contiguous() or value.storage_offset() != 0:
        raise NotImplementedError("NYI: serializing non-contiguous get_attr tensor")
    with unset_fake_temporarily():
        if data_value.device.type != "cpu":
            cpu_value = data_value.detach().cpu()
        else:
            cpu_value = data_value.detach()
        value_list = cpu_value.tolist()

    args = [_python_constant_repr(value_list)]
    args.append(f"dtype={dtype}")
    if device.type != "cpu":
        args.append(f"device={str(device)!r}")
    return f"torch.tensor({', '.join(args)})"


class GetAttr(PatternExpr):
    """
    Match a get_attr node by comparing the attribute value instead of the target
    name, which is local to the traced GraphModule.
    """

    def __init__(self, value: Any, users: Multiple | int = 1) -> None:
        super().__init__()
        self.value = value
        self.users = users

    def __repr__(self) -> str:
        if self.users is MULTIPLE:
            comma_users = ", MULTIPLE"
        elif self.users != 1:
            comma_users = f", {self.users}"
        else:
            comma_users = ""
        return f"{self.__class__.__name__}({self.value!r}{comma_users})"

    def has_multiple_users(self) -> bool:
        return isinstance(self.users, Multiple) or self.users > 1

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        if not isinstance(node, torch.fx.Node) or node.op != "get_attr":
            return FailedMatch("get_attr_mismatch: node={}, pattern={}", node, self)
        if (
            self not in ctx.outputs
            and self.users is not MULTIPLE
            and len(node.users) != self.users
        ):
            return FailedMatch("multiple_users {}", self)
        if node.graph.owning_module is None:
            return FailedMatch("get_attr without owning module: {}", node)
        if not isinstance(node.target, str):
            return FailedMatch("get_attr target is not str: {}", node.target)
        value = _get_attr(node.graph.owning_module, node.target)
        if not _constant_values_equal(self.value, value):
            return FailedMatch("get_attr_value: {} != {}", value, self.value)
        m = Match(ctx, self)
        m.nodes.append(node)
        return m

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        args = [pp.pretty_print(self.value)]
        if self.users is MULTIPLE:
            args.append("MULTIPLE")
        elif self.users != 1:
            args.append(str(self.users))
        return f"{self.__class__.__name__}({', '.join(args)})"

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)
        return (
            super().pattern_eq(other)
            and _constant_values_equal(self.value, other.value)
            and self.users == other.users
        )


class KeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"KeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        return Match(ctx, self, kwargs={self.name: node})  # matches anything

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.name == other.name


class ExclusiveKeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"ExclusiveKeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        if node in ctx.exclusive_node_set:
            return FailedMatch("exclusive arg appears twice")

        ctx.exclusive_node_set.append(node)
        return Match(ctx, self, kwargs={self.name: node})  # matches anything

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.name == other.name


class _TargetExpr(PatternExpr):
    """
    Base class for filtering match by node.target
    """

    fns: list[FnsType]
    fns_set: OrderedSet[FnsType]

    def __init__(
        self, fns: FnsType | Sequence[FnsType], users: Multiple | int = 1
    ) -> None:
        super().__init__()
        fns = [fns] if callable(fns) or isinstance(fns, str) else list(fns)
        for fn in fns:
            if isinstance(fn, torch._ops.OpOverloadPacket):
                fns.extend(fn.op_overloads())  # noqa: B909

        self.fns = fns
        self.fns_set = OrderedSet(fns)
        self.users = users

    @property
    @abstractmethod
    def op(self) -> str: ...

    def fns_repr(self) -> str:
        first_repr = self.fns[0]
        if not isinstance(first_repr, str):
            first_repr = first_repr.__name__

        if len(self.fns) > 1:
            return f"[{first_repr}, ...]"
        elif self.fns[0] is getattr(torch, first_repr, None):
            return f"torch.{first_repr}"
        elif self.fns[0] is getattr(operator, first_repr, None):
            return f"operator.{first_repr}"
        elif isinstance(self.fns[0], torch._ops.OpOverload):
            return str(self.fns[0])
        else:
            return first_repr

    def __repr__(self) -> str:
        if self.users is MULTIPLE:
            comma_users = ", MULTIPLE"
        elif self.users != 1:
            comma_users = f", {self.users})"
        else:
            comma_users = ""
        return f"{self.__class__.__name__}({self.fns_repr()}{comma_users})"

    def has_multiple_users(self) -> bool:
        return isinstance(self.users, Multiple) or self.users > 1

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None, None, None]:
        raise NotImplementedError

    def _match_fns(self, node: torch.fx.Node) -> bool:
        return (
            isinstance(node, torch.fx.Node)
            and node.op == self.op
            and extract_target(node) in self.fns_set
        )

    def _match_users(self, node: torch.fx.Node, ctx: MatchContext) -> bool:
        return (
            self in ctx.outputs
            or self.users is MULTIPLE
            or len(node.users) == self.users
        )

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.op == other.op
            and self.fns == other.fns
            and self.users == other.users
        )


_SimpleSpec = tuple[Any, ...]


class _TargetArgsExpr(_TargetExpr):
    """
    Base class for filtering match by node.{target,args,kwargs}
    """

    def __init__(
        self,
        fns: torch.fx.node.Target | str | Sequence[Any],
        *args: Any,
        _users: int | Multiple = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(fns, _users)
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        if any(
            isinstance(x, (dict, list, tuple))
            for x in itertools.chain(args, kwargs.values())
        ):
            self.flatten = self.pytree_flatten
        else:
            self.flatten = self.simple_flatten
        self.flat_args_kwargs = self.flatten(self.args, self.kwargs)

    @staticmethod
    def simple_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> tuple[Sequence[Any], _SimpleSpec | pytree.TreeSpec]:
        values = (*args, *kwargs.values())
        spec = (len(args), *kwargs.keys())
        return values, spec

    @staticmethod
    def pytree_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> tuple[Sequence[Any], _SimpleSpec | pytree.TreeSpec]:
        type_mapping: dict[type, type] = {
            immutable_list: tuple,
            list: tuple,
            immutable_dict: dict,
        }

        def convert_type(x: Any) -> Any:
            cls = type(x)
            convert_fn = type_mapping.get(cls)
            if convert_fn is not None:
                return pytree.tree_map(
                    convert_type,
                    convert_fn(x),
                    is_leaf=lambda x: type(x) in type_mapping,
                )
            return x

        normalized_args_tree = pytree.tree_map(
            convert_type,
            (args, kwargs),
            is_leaf=lambda x: type(x) in type_mapping,
        )
        flat, spec = pytree.tree_flatten(normalized_args_tree)
        return flat, spec

    def __repr__(self) -> str:
        args = [
            self.fns_repr(),
            *map(repr, self.args),
            *[f"{k}={v}" for k, v in self.kwargs.items()],
        ]
        if self.users is MULTIPLE:
            args.append("_users=MULTIPLE")
        elif self.users != 1:
            args.append(f"_users={self.users}")
        return f"{self.__class__.__name__}({', '.join(args)})"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        args = [
            self.fns_repr(),
            *(pp.pretty_print(x) for x in self.args),
            *[f"{k}={pp.pretty_print(v)}" for k, v in self.kwargs.items()],
        ]
        if self.users is MULTIPLE:
            args.append("_users=MULTIPLE")
        elif self.users != 1:
            args.append(f"_users={self.users}")

        joiner_str = ", "
        return f"{self.__class__.__name__}({joiner_str.join(args)})"

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        if not self._match_fns(node) or len(node.args) != len(self.args):
            return FailedMatch("function_mismatch: node={}, pattern={}", node, self)

        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users {}", self)

        _args = node.args
        _kwargs = node.kwargs
        if len(_kwargs) < len(self.kwargs):
            from torch.fx.operator_schemas import normalize_function

            if not callable(node.target):
                raise AssertionError(
                    f"expected callable node.target, got {node.target}"
                )
            normalized_args_and_kwargs = normalize_function(
                node.target, node.args, node.kwargs
            )

            if normalized_args_and_kwargs is None:
                return FailedMatch("function_mismatch: node={}, pattern={}", node, self)
            else:
                _args, _kwargs = normalized_args_and_kwargs
                if len(_args) == len(self.args) and len(_kwargs) >= len(self.kwargs):
                    _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}
                else:
                    return FailedMatch(
                        "function_mismatch: node={}, pattern={}", node, self
                    )
        else:
            _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}

        node_items, node_spec = self.flatten(_args, _kwargs)
        self_items, self_spec = self.flat_args_kwargs
        if node_spec != self_spec:
            return FailedMatch("args_structure {} {}", node_spec, self_spec)
        if len(node_items) != len(self_items):
            raise AssertionError("node_items and self_items length mismatch")

        m = Match(ctx, self)
        for pattern, child_node in zip(self_items, node_items):
            if isinstance(pattern, PatternExpr):
                child_match = ctx.match(pattern, child_node)
                if not is_match(child_match):
                    return child_match
                m.extend(child_match)
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch(
                    "constant_args: {} {!r}!={pattern!r}",
                    node,
                    child_node,
                    pattern=pattern,
                )
        m.nodes.append(node)
        m.targets[self] = node.target
        return m

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None, None, None]:
        """
        This is used when we are matching a pattern with multiple outputs.
        There is a partial match (stored in ctx) and we want to walk
        this pattern to find a connection to an already-matched node.

        Yields candidate nodes that `self._match` might like.
        """
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]
            return

        for pattern in self.flat_args_kwargs[0]:
            if isinstance(pattern, PatternExpr):
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    for node in other_node.users:
                        if node not in searched:
                            if self._match_fns(node):
                                yield node
                                searched.add(node)

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.flat_args_kwargs[1] == other.flat_args_kwargs[1]
            and all(
                a.pattern_eq(b) if isinstance(a, PatternExpr) else a == b
                for a, b in zip(self.flat_args_kwargs[0], other.flat_args_kwargs[0])
            )
        )


class CallFunction(_TargetArgsExpr):
    """
    Matches a call_function node in the FX graphs: `fns[i](*args, **kwargs)`
    """

    op = "call_function"


class CallMethod(_TargetArgsExpr):
    """
    Matches a call_method node in the FX graphs: `fns[i].method(*args, **kwargs)`
    """

    op = "call_method"


class CallModule(_TargetArgsExpr):
    """
    Matches a call_module node in the FX graphs: `module(*args, **kwargs)`
    """

    op = "call_module"


class _TargetExprVarArgs(_TargetExpr):
    """
    Matches a call_function node with any arguments which are passed into the pattern
    """

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        if not self._match_fns(node):
            return FailedMatch("function_mismatch")

        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users")

        m = Match(ctx, self)
        m.nodes.append(node)
        m.targets[self] = node.target
        m.args.extend(node.args)
        m.kwargs.update(node.kwargs)
        return m


class CallFunctionVarArgs(_TargetExprVarArgs):
    op = "call_function"


class CallMethodVarArgs(_TargetExprVarArgs):
    op = "call_method"


class CallModuleVarArgs(_TargetExprVarArgs):
    op = "call_module"


class ListOf(PatternExpr):
    """
    Matches a repeated pattern
    """

    def __init__(self, pattern: PatternExpr, partial: bool = False) -> None:
        super().__init__()
        if not isinstance(pattern, PatternExpr):
            raise AssertionError(f"expected PatternExpr, got {type(pattern)}")
        self.pattern = pattern
        self.partial = partial

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern})"

    def _match(self, node: list[torch.fx.Node], ctx: MatchContext) -> MatchResult:  # type: ignore[override]
        if not isinstance(node, (list, tuple)) or len(node) == 0:
            return FailedMatch("non_list")
        m = Match(ctx, self)
        # Propagating patterns with multiple users will ensure we don't revisit
        # the same nodes
        pattern_to_node = ctx.filter_multi_user_patterns()
        matched = False
        for i, child_node in enumerate(node):
            child_ctx = MatchContext(
                ctx.outputs, pattern_to_node, graph=child_node.graph
            )
            child_match = child_ctx.match(self.pattern, child_node)
            pattern_to_node = child_ctx.filter_multi_user_patterns()
            if not is_match(child_match):
                if not self.partial:
                    return FailedMatch("list[{}]: {}", i, child_match)
                continue
            matched = True
            m.extend(child_match.bundle())
        if not matched:
            return FailedMatch("list: no_match")
        return m.bundle()

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.pattern.pattern_eq(other.pattern)
            and self.partial == other.partial
        )


class MultiOutputPattern(PatternExpr):
    outputs: list[PatternExpr | None]

    def __init__(self, outputs: Sequence[PatternExpr | None]) -> None:
        super().__init__()
        if not isinstance(outputs[0], _TargetExpr):
            raise AssertionError(f"expected _TargetExpr, got {type(outputs[0])}")
        if not all(x is None or isinstance(x, PatternExpr) for x in outputs):
            raise AssertionError(outputs)
        self.outputs = list(outputs)
        self.op = outputs[0].op

    @property
    def fns(self) -> Callable[..., Any] | str | Sequence[Any]:
        # This cast is checked above in __init__()
        output = typing.cast(_TargetExpr, self.outputs[0])
        return output.fns

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.outputs})"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        args = [pp.pretty_print(x) for x in self.outputs]
        joiner_str = f",\n{'  '}"
        str_out = f"{self.__class__.__name__}([{joiner_str.join(args)}"
        str_out = f"{str_out}\n])"
        return str_out

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        output = typing.cast(_TargetExpr, self.outputs[0])
        m = ctx.match(output, node)
        if not is_match(m):
            return m

        for pattern in self.outputs[1:]:
            if pattern is None:
                continue
            child_match = self._match_from_anchors(pattern, ctx)
            if not is_match(child_match):
                return child_match
            m.extend(child_match)

        return m

    def _match_from_anchors(
        self, pattern: PatternExpr, ctx: MatchContext
    ) -> MatchResult:
        prior = dict(ctx.pattern_to_node)
        m: MatchResult = FailedMatch("no anchor found")
        for node in pattern.find_anchor_nodes(ctx, OrderedSet()):
            m = ctx.match(pattern, node)
            if is_match(m):
                return m
            # revert any partial matches
            ctx.pattern_to_node = dict(prior)
        return m

    def match(self, node: torch.fx.Node) -> MatchResult:
        try:
            return MatchContext(self.outputs, graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and len(self.outputs) == len(other.outputs)
            and all(
                a.pattern_eq(b) if isinstance(a, PatternExpr) else a == b
                for a, b in zip(self.outputs, other.outputs)
            )
        )


class RepeatedExpr(PatternExpr):
    """
    Checks for a repeated pattern. Useful for repeated operations after a node such as `split` or `unbind`
    """

    def __init__(self, inner_pattern: _TargetExpr) -> None:
        super().__init__()
        self.inner_pattern = inner_pattern
        self.op = inner_pattern.op

    @property
    def fns(self) -> Sequence[FnsType]:
        return self.inner_pattern.fns

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        m = ctx.match(self.inner_pattern, node)
        if not is_match(m):
            return m
        ctx.pattern_to_node.pop(
            self.inner_pattern,
        )
        # Check all anchor nodes match the pattern
        for anchor_node in self.inner_pattern.find_anchor_nodes(
            ctx, OrderedSet([node])
        ):
            anchor_m = MatchContext([self], graph=node.graph).match(
                self.inner_pattern, anchor_node
            )
            if not is_match(anchor_m):
                return anchor_m
            m.extend(anchor_m)
        return m

    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.inner_pattern.pattern_eq(
            other.inner_pattern
        )


class PatternPrettyPrinter:
    """
    Serializes Patterns to executable python.
    XXX: currently only used and tested for fuse attention patterns. May not cover
    all patterns.
    """

    def __init__(self) -> None:
        self.namespace = torch.fx.graph._Namespace()
        self.memoized_objs_names: dict[PatternExpr, str] = {}
        self.memoized_objs_pp: dict[PatternExpr, str] = {}

    @staticmethod
    @functools.cache
    def run(obj: PatternExpr, output_name: str = "output") -> str:
        """
        Serializes obj to python code with obj written out to `output_name`
        """

        pp = PatternPrettyPrinter()
        if not hasattr(obj, "pretty_print"):
            raise AssertionError("obj has no pretty_print method")
        out_str = obj.pretty_print(pp=pp)

        output = [
            f"{pp.memoized_objs_names[key]} = {pp.memoized_objs_pp[key]}"
            for key in pp.memoized_objs_names
        ]

        output.append(f"{output_name} = {out_str}")

        return "\n".join(output)

    def pretty_print(self, obj: Any) -> str:
        if isinstance(obj, _TargetArgsExpr):
            if memoized_name := self.memoized_objs_names.get(obj):
                return memoized_name
            else:
                return self.memoize(obj)
        if isinstance(obj, torch.Tensor):
            return _tensor_constant_repr(obj)
        if hasattr(obj, "pretty_print"):
            return obj.pretty_print(self)

        return repr(obj)

    def memoize(self, obj: _TargetArgsExpr) -> str:
        obj_str = obj.pretty_print(self)
        obj_name = obj.fns_repr()
        for prefix in ("aten.", "torch.", "prims."):
            obj_name = obj_name.replace(prefix, "")

        tmp_name = self.namespace.create_name(obj_name, None)
        self.memoized_objs_names[obj] = tmp_name
        self.memoized_objs_pp[obj] = obj_str
        return tmp_name


class _PassDictsType(Protocol):
    def __getitem__(
        self, k: tuple[str, torch.fx.node.Target]
    ) -> list[PatternEntry]: ...


@dataclasses.dataclass
class PatternEntry:
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        raise NotImplementedError

    def register(
        self,
        pass_dicts: _PassDictsType | Sequence[_PassDictsType],
        target: torch.fx.node.Target | None = None,
        prepend: bool = False,
    ) -> None:
        if target is None:
            if not hasattr(self.pattern, "fns"):
                raise AssertionError("pattern has no fns attribute")
            for fn in self.pattern.fns:
                self.register(pass_dicts, fn, prepend=prepend)
        elif isinstance(pass_dicts, (dict, PatternMatcherPass)):
            if not hasattr(self.pattern, "op"):
                raise AssertionError("pattern has no op attribute")
            if prepend:
                pass_dicts[(self.pattern.op, target)].insert(0, self)
            else:
                pass_dicts[(self.pattern.op, target)].append(self)
        else:
            pass_dicts = typing.cast(Sequence[_PassDictsType], pass_dicts)
            for x in pass_dicts:
                self.register(x, target, prepend=prepend)


@dataclasses.dataclass
class LoweringPatternEntry(PatternEntry):
    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        handler = functools.wraps(self.handler)(functools.partial(self.handler, match))
        with graph.inserting_before(node):
            replacement = graph.call_function(handler, tuple(match.args), match.kwargs)
            _transfer_meta(replacement.meta, node)
            node.replace_all_uses_with(replacement)
        if match.nodes[-1] is not node:
            raise AssertionError("expected match.nodes[-1] to be node")
        match.erase_nodes()


@dataclasses.dataclass
class GraphPatternEntry(PatternEntry):
    """
    A pattern that runs a function on the FX graph
    """

    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        with graph.inserting_before(node):
            self.handler(match, *match.args, **match.kwargs)


@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    """
    The replacement pattern for the graph
    """

    normalize_args: Callable[..., list[Any]]
    pattern_name: str | None = None  # Unique identifier for per-pattern telemetry

    @staticmethod
    def replace_with_graph(
        match: Match,
        graph: torch.fx.Graph,
        replacement_graph: torch.fx.Graph | torch.fx.GraphModule,
        args: Sequence[torch.fx.Node],
        pass_name: str | None = None,
    ) -> list[torch.fx.Node]:
        """
        Inserts the replacement graph into the toplevel graph at the match
        """

        added_replacement_nodes: list[torch.fx.Node] = []
        custom_context = _common_custom_context(match.nodes)

        class Replacer(torch.fx.Interpreter):
            call_method = None  # type: ignore[assignment]
            call_module = None  # type: ignore[assignment]
            get_attr = None  # type: ignore[assignment]

            def run_node(self, node: torch.fx.Node) -> Any:
                if node.op in ("placeholder", "output"):
                    return super().run_node(node)
                target = node.target
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                if node.op == "call_function":
                    if not callable(target):
                        raise AssertionError(f"expected callable target, got {target}")
                    result = graph.call_function(target, args, kwargs)
                    added_replacement_nodes.append(result)
                    _transfer_meta(
                        new_meta=result.meta,
                        old_node=node,
                        pass_name=pass_name or "",
                    )
                    _merge_custom_context(result.meta, custom_context)
                    # This function copy-pastes the replacement graph into
                    # the graph. If the replacement graph had any eager_input_vals,
                    # we propagate those over (val/tensor_meta are handled by
                    # _transfer_meta above).
                    if "eager_input_vals" in node.meta:
                        result.meta["eager_input_vals"] = node.meta["eager_input_vals"]
                    return result
                if node.op == "get_attr":
                    # If the replacement graph contains a HOP, the subgraphs of the HOP are "get_attr" nodes.
                    # We need to fetch the subgraph of the HOP then register the subgraph to the replaced graph's root.
                    from torch._higher_order_ops.utils import (
                        unique_graph_name_with_root,
                    )

                    sub_gm = super().get_attr(target, args, kwargs)
                    if not isinstance(sub_gm, torch.fx.GraphModule):
                        raise NotImplementedError(
                            f"NYI: replacement_graph.{target} is not a graph module. Got {sub_gm}."
                        )
                    if graph.owning_module is None:
                        raise AssertionError("graph.owning_module is None")
                    graph_name = None
                    for n, mod in graph.owning_module.named_modules():
                        if sub_gm is mod:
                            graph_name = n
                            break
                    if graph_name is None:
                        if not isinstance(target, str):
                            raise AssertionError(
                                f"expected str target, got {type(target)}"
                            )
                        _, graph_name = unique_graph_name_with_root(
                            # pyrefly: ignore [unbound-name]
                            graph.owning_module,
                            target,
                        )
                        # pyrefly: ignore [unbound-name]
                        graph.owning_module.register_module(graph_name, sub_gm)
                    # pyrefly: ignore [unbound-name]
                    getattr_node = graph.get_attr(graph_name)
                    added_replacement_nodes.append(getattr_node)
                    return getattr_node

                raise NotImplementedError(f"unhandled {node}")

        output_nodes = match.output_nodes()

        if len(output_nodes) == 1:
            last_node = output_nodes[0]
        else:
            if not output_nodes[0]:
                raise AssertionError("output_nodes[0] is falsy")
            nodes = list(output_nodes[0].graph.nodes)
            indices = [
                (nodes.index(n), n)
                for n in output_nodes
                if isinstance(n, torch.fx.Node)
            ]
            last_node = min(indices, key=operator.itemgetter(0))[1]

        def percolate_tags(
            node: torch.fx.Node,
            tag_name: str,
            tag_value: str,
            input_stops: OrderedSet[torch.fx.Node],
        ) -> None:
            queue = [node]
            visited = OrderedSet[torch.fx.Node]()

            while queue:
                arg = queue.pop()
                if (
                    arg not in visited
                    and arg not in input_stops
                    and hasattr(arg, "meta")
                ):
                    visited.add(arg)
                    arg.meta[tag_name] = tag_value
                    queue.extend(arg.all_input_nodes)

        with graph.inserting_before(last_node):
            if not isinstance(replacement_graph, torch.fx.GraphModule):
                raise AssertionError(
                    f"expected GraphModule, got {type(replacement_graph)}"
                )
            replacement = Replacer(replacement_graph).run(*args)
            if isinstance(replacement, torch.fx.Node):
                replacement = [replacement]

            def maybe_getitem(node: torch.fx.Node) -> Any:
                if node.op != "call_function":
                    return None
                if node.target != operator.getitem:
                    return None
                if len(node.args) != 2:
                    raise AssertionError(f"expected 2 args, got {len(node.args)}")
                return node.args[1]

            def replace(
                old: torch.fx.Node | None,
                new: torch.fx.Node | Sequence[torch.fx.Node] | None,
            ) -> None:
                def filter_nodes_in_newly_added_nodes(node: torch.fx.Node) -> bool:
                    # Do not replace the use of a node if it is being used by
                    # nodes in the replaced graph
                    return node not in added_replacement_nodes

                if old is None:
                    if new is not None:
                        raise AssertionError("expected new to be None when old is None")
                    return
                if not isinstance(old, torch.fx.Node):
                    raise AssertionError(f"expected torch.fx.Node, got {type(old)}")
                if new is None:
                    old.replace_all_uses_with(
                        None,  # type: ignore[arg-type]
                        delete_user_cb=filter_nodes_in_newly_added_nodes,
                    )
                    if len(old.users) == 0:
                        graph.erase_node(old)
                    return
                if isinstance(new, torch.fx.Node):
                    _transfer_meta(new.meta, old, pass_name=pass_name or "")

                    # Preserve the recompute tags in the replacement graph. We
                    # look at the recompute tags of the original output node to
                    # propagate the tag from the output all the way to the input
                    args_set, _ = pytree.tree_flatten(args)
                    # args (named as args in the replace_with_graph).
                    # Note that this is best effort. Since patterns are from
                    # many to many, there is no easy way to correctly map the
                    # recomputable tags. It is possible in some scenarios that we
                    # incorrectly tag some nodes as recomputables.
                    for tag_name in ["recompute", "ac_graph_id", "custom"]:
                        if tag_name in old.meta:
                            percolate_tags(
                                new, tag_name, old.meta[tag_name], OrderedSet(args_set)
                            )

                    old.replace_all_uses_with(
                        new, delete_user_cb=filter_nodes_in_newly_added_nodes
                    )
                    if len(old.users) == 0:
                        graph.erase_node(old)
                    return

                # `new` is not a node: it's a list of nodes.
                #
                # This happens when we want to replace a node that has a single
                # packed return with multiple unpacked returns. We need to do
                # some graph surgery here.
                #
                # Example:
                #   def original_graph(x):
                #      a = op(x)
                #      b = a[0]
                #      c = a[1]
                #      ...
                #
                # Assume that we want to replace op(x) with the graph
                #   def new_op(x):
                #      w = x + 1
                #      z = x + 2
                #      return (w, z)
                #
                # We need to replace `op` with the contents of `new_op`,
                # and then rewrite a[0] to be w and a[1] to be z, as so:
                #   def new_graph(x):
                #     w = x + 1
                #     z = x + 2
                #     b = w
                #     c = z
                #     ...
                old_uses = list(old.users.keys())
                for user in old_uses:
                    idx = maybe_getitem(user)
                    if idx is None:
                        # Output is used directly
                        # pyrefly: ignore [bad-argument-type]
                        old.replace_all_uses_with(new)
                    else:
                        replace(user, new[idx])
                graph.erase_node(old)

            if len(output_nodes) == len(replacement):
                for old, new in zip(output_nodes, replacement):
                    replace(old, new)
            else:
                if len(output_nodes) != 1:
                    raise AssertionError(
                        f"expected 1 output node, got {len(output_nodes)}"
                    )
                replace(output_nodes[0], replacement)

        match.erase_nodes()

        # Remove dead replacement nodes so they don't inflate user counts
        # in later lowering heuristics (e.g. should_realize_on_reuse).
        for node in reversed(added_replacement_nodes):
            if (
                not node.users
                and not node.is_impure()
                and not isinstance(node.target, torch._ops.HigherOrderOperator)
            ):
                graph.erase_node(node)

        return [node for node in added_replacement_nodes if not node._erased]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        if match.replacement_graph is None:
            raise AssertionError("match.replacement_graph is None")
        self.replace_with_graph(
            match,
            graph,
            match.replacement_graph,
            self.normalize_args(*match.args, **match.kwargs),
            pass_name=self.pattern_name or "replace_with_graph",
        )


def _return_true(match: Match) -> bool:
    return True


def log_trace_failure(search_fn: Callable[..., Any], e: RuntimeError) -> None:
    log.info(
        "Replacement pattern %s failed to apply due to shape mismatch: %s",
        search_fn.__name__,
        e,
    )


@dataclasses.dataclass(frozen=True)
class _TraceArgInfo:
    argnames: list[str]
    flat_argnames: list[str]
    arg_specs: dict[str, Any]
    arg_flat_names: dict[str, list[str]]


def _unique_argname(name: str, used_names: OrderedSet[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name

    i = 0
    while f"{name}_{i}" in used_names:
        i += 1
    name = f"{name}_{i}"
    used_names.add(name)
    return name


def _trace_args_for_initial_trace(
    argnames: Sequence[str],
    example_inputs: Sequence[Any],
    scalar_workaround: dict[str, float | int],
) -> list[Any]:
    trace_args = []
    input_idx = 0
    for argname in argnames:
        if argname in scalar_workaround:
            trace_args.append(scalar_workaround[argname])
        else:
            trace_args.append(example_inputs[input_idx])
            input_idx += 1
    return trace_args


def _trace_arg_info(argnames: Sequence[str], args: Sequence[Any]) -> _TraceArgInfo:
    flat_argnames: list[str] = []
    arg_specs: dict[str, Any] = {}
    arg_flat_names: dict[str, list[str]] = {}
    used_names: OrderedSet[str] = OrderedSet()

    for argname, arg in zip(argnames, args):
        flat_args, spec = pytree.tree_flatten(arg)
        arg_specs[argname] = spec
        if len(flat_args) == 0:
            names: list[str] = []
        elif len(flat_args) == 1:
            names = [_unique_argname(argname, used_names)]
        else:
            names = [
                _unique_argname(f"{argname}_{i}", used_names)
                for i in range(len(flat_args))
            ]
        flat_argnames.extend(names)
        arg_flat_names[argname] = names

    return _TraceArgInfo(
        argnames=list(argnames),
        flat_argnames=flat_argnames,
        arg_specs=arg_specs,
        arg_flat_names=arg_flat_names,
    )


def _unflatten_match_kwargs(
    kwargs: Mapping[str, Any],
    arg_info: _TraceArgInfo,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = dict(kwargs)
    result: dict[str, Any] = {}

    for argname in arg_info.argnames:
        flat_names = arg_info.arg_flat_names[argname]
        found_names = [flat_name for flat_name in flat_names if flat_name in kwargs]
        if len(found_names) != len(flat_names):
            if defaults is not None and argname in defaults and not found_names:
                result[argname] = defaults[argname]
                continue
            missing_names = [
                flat_name for flat_name in flat_names if flat_name not in kwargs
            ]
            raise RuntimeError(
                f"Not all inputs to pattern found in match.kwargs. Perhaps one "
                f"of the inputs is unused? argname={argname}, "
                f"flat_argnames={flat_names}, missing_argnames={missing_names}, "
                f"match.kwargs={kwargs}"
            )

        flat_values = []
        for flat_name in flat_names:
            flat_values.append(kwargs.pop(flat_name))
        result[argname] = pytree.tree_unflatten(
            flat_values, arg_info.arg_specs[argname]
        )

    result.update(kwargs)
    return result


def _get_match_node_value(node: torch.fx.Node) -> Any:
    if "val" in node.meta:
        return node.meta["val"]
    return node.meta["example_value"]


def check_and_add_duplicate_pattern(
    pattern: PatternExpr,
    graph: torch.fx.Graph | None,
    seen_patterns: dict[str, list[str | None]],
    skip_duplicates: bool = False,
) -> bool:
    """
    Check if a pattern is a duplicate. Because we ignore certain types in searching, but not
    in matching, use the graph to distinguish equivalent search patterns.

    Returns True if a duplicate is found and `skip_duplicates=True` is passed in. Errors if
    `skip_duplicates` is False and a duplicate is found.
    """

    pattern_repr = PatternPrettyPrinter.run(pattern)
    equiv_pattern_reprs = seen_patterns.get(pattern_repr)
    if not equiv_pattern_reprs:
        seen_patterns[pattern_repr].append(str(graph) if graph else None)
        return False

    if graph is None:
        if skip_duplicates:
            return True
        torch._check(
            False,
            lambda: f"Duplicate pattern: {pattern_repr} with no graph",
        )

    new_graph_str = str(graph)
    for graph_str in equiv_pattern_reprs:
        if new_graph_str != graph_str:
            continue
        if skip_duplicates:
            return True
        torch._check(
            False,
            lambda: f"Duplicate pattern: {pattern_repr} with duplicated match graph {graph_str} ",
        )
    equiv_pattern_reprs.append(new_graph_str)
    return False


def register_replacement(
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: list[Any] | tuple[Any, ...],
    trace_fn: TraceFn,
    pass_dicts: _PassDictsType | Sequence[_PassDictsType],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
    search_fn_pattern: PatternExpr | None = None,
    skip_duplicates: bool = False,
    pattern_name: str | None = None,
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = select_decomp_table,
) -> bool:
    """
    Create a replacement rule based on example functions that get traced
    to create patterns.  This supports both training and inference when
    run on a joint forward+backward graph.

    Args:
        search_fn: traced to give original pattern
        replace_fn: traced to give replacement graph
        example_inputs: example inputs for initial trace
        trace_fn: fwd_only or joint_fwd_bwd
        pass_dict: dict of passes to register to
        extra_check: additional check to run on match(using real shapes)
    """
    argnames_static = [*inspect.signature(search_fn).parameters.keys()]

    if inspect.ismethod(search_fn):
        search_fn = _wrap_bound_method(search_fn, argnames_static)

    if inspect.ismethod(replace_fn):
        replace_argnames = [*inspect.signature(replace_fn).parameters.keys()]
        replace_fn = _wrap_bound_method(replace_fn, replace_argnames)

    if not isinstance(example_inputs, (list, tuple)):
        raise TypeError(
            f"example_inputs must be a list or tuple, got {type(example_inputs)}"
        )
    scalar_workaround = scalar_workaround or {}
    initial_trace_args = _trace_args_for_initial_trace(
        argnames_static, example_inputs, scalar_workaround
    )
    initial_arg_info = _trace_arg_info(argnames_static, initial_trace_args)
    requires_grad = pytree.tree_map(
        lambda x: isinstance(x, torch.Tensor) and x.requires_grad,
        initial_trace_args,
    )

    def check_fn(match: Match) -> bool:
        """
        Often shapes get burned into the pattern, so our initial match ran with
        `ignore_types=(int, ...)`.

        Recheck the match with the correct shapes.
        """
        structured_match_kwargs = _unflatten_match_kwargs(
            match.kwargs, initial_arg_info
        )

        args = list(
            torch.fx.map_arg(
                [structured_match_kwargs[name] for name in argnames_static],
                _get_match_node_value,
            )
        )

        sym_args: list[torch.SymInt] = []
        fake_mode = torch._dynamo.utils.detect_fake_mode(args)
        if fake_mode is None:
            raise AssertionError("fake_mode is None")
        with fake_mode:
            invalid_args = False
            requires_grad_values = iter(pytree.tree_leaves(requires_grad))

            def refresh_arg(arg: Any) -> Any:
                nonlocal invalid_args
                grad = next(requires_grad_values)
                if not isinstance(arg, torch.Tensor):
                    return arg

                if grad and is_integer_dtype(arg.dtype):
                    invalid_args = True
                    return arg

                refreshed_arg = torch.empty_strided(
                    arg.size(),
                    arg.stride(),
                    dtype=arg.dtype,
                    device=arg.device,
                    requires_grad=grad,
                )
                for v in itertools.chain(refreshed_arg.shape, refreshed_arg.stride()):
                    if isinstance(v, torch.SymInt) and all(
                        statically_known_true(v != a) for a in sym_args
                    ):
                        sym_args.append(v)
                return refreshed_arg

            args = pytree.tree_map(refresh_arg, args)
            if invalid_args:
                return False

            # If we were given a pre-traced pattern then use that instead of
            # retracing. Note that this means the pattern has to be independent
            # of its args.
            specific_pattern = search_fn_pattern
            specific_arg_info = _trace_arg_info(argnames_static, args)
            specific_argnames = specific_arg_info.flat_argnames

            if not specific_pattern:
                if sym_args:
                    # AOT Autograd and make fx will dedupe symbolic shape size
                    # accesses of sym ints that appear as inputs
                    # We don't want the sym_size uses to interfere with pattern matching
                    # so we provide them as inputs.
                    # Later, when we actually do the replacement, the symbolic shape
                    # sizes will get re-traced and added to the graph.

                    def search_fn_new(*args_new: Any, **_: Any) -> Any:
                        return search_fn(*args_new[len(args_new) - len(args) :])

                    try:
                        specific_graph = trace_fn(
                            search_fn_new,
                            sym_args + args,
                            get_decomp_fn=get_decomp_fn,
                        )
                    except RuntimeError as e:
                        log_trace_failure(search_fn, e)
                        return False

                    # correct argnames in the graph
                    sym_arg_names = []
                    placeholders = [
                        n for n in specific_graph.graph.nodes if n.op == "placeholder"
                    ]
                    for i, placeholder in enumerate(
                        placeholders[: len(sym_args) + len(specific_argnames)]
                    ):
                        if i < len(sym_args):
                            sym_arg_names.append(placeholder.target)
                            continue

                        with specific_graph.graph.inserting_after(placeholder):
                            new_node = specific_graph.graph.placeholder(
                                specific_argnames[i - len(sym_args)]
                            )
                            new_node.target = new_node.name
                            placeholder.replace_all_uses_with(new_node)
                            specific_graph.graph.erase_node(placeholder)

                    specific_argnames = sym_arg_names + specific_argnames
                else:
                    try:
                        specific_graph = trace_fn(
                            search_fn, args, get_decomp_fn=get_decomp_fn
                        )
                    except RuntimeError as e:
                        log_trace_failure(search_fn, e)
                        return False

                specific_pattern = fx_to_pattern(
                    specific_graph,
                    argnames=specific_argnames,
                    exclusive_arg_names=exclusive_arg_names,
                    scalar_workaround=scalar_workaround,
                )

            node = match.output_nodes()[0]
            if node is None:
                raise AssertionError("output node is None")
            specific_pattern_match = specific_pattern.match(node)

            if _should_debug_node(node.name):
                log.warning(
                    "Specific pattern match: %s%s %s %s",
                    node,
                    node.args,
                    specific_pattern_match,
                    specific_pattern,
                )

            if is_match(specific_pattern_match):
                specific_pattern_match.kwargs = _unflatten_match_kwargs(
                    specific_pattern_match.kwargs,
                    specific_arg_info,
                    defaults=structured_match_kwargs,
                )

            if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
                # trace the pattern using the shapes from the user program
                match.replacement_graph = trace_fn(
                    replace_fn, args, get_decomp_fn=get_decomp_fn
                )
                match.kwargs = structured_match_kwargs
                if len(match.nodes) == 1:
                    for n in match.replacement_graph.graph.nodes:
                        _transfer_meta(
                            new_meta=n.meta,
                            old_node=match.nodes[0],
                            pass_name="replacement",
                        )
                return True
            return False

    def normalize_args(**kwargs: Any) -> list[Any]:
        # Flat kwargs come from the initial match; structured kwargs are restored
        # after check_fn accepts the match.
        if not all(name in kwargs for name in argnames_static):
            kwargs = _unflatten_match_kwargs(kwargs, initial_arg_info)
        structured_args = [kwargs.pop(name) for name in argnames_static]
        args = []
        for arg in structured_args:
            args.extend(pytree.tree_leaves(arg))
        for i in range(1, len(kwargs) + 1):
            if f"tangents_{i}" not in kwargs:
                break
            args.append(kwargs.pop(f"tangents_{i}"))
        if kwargs:
            raise AssertionError(f"leftover kwargs: {kwargs!r}")
        return args

    if trace_fn is joint_fwd_bwd:
        # If inference mode is enabled during compilation, assume that we don't
        # want to match on any training graph patterns
        if torch.is_inference_mode_enabled():
            return False

    # TODO: Revisit the functionalize_rng_ops for lowmem dropout
    with functorch_config.patch(functionalize_rng_ops=False):
        if search_fn_pattern is None:
            pattern, gm = gen_pattern_and_search_gm(
                search_fn,
                example_inputs,
                trace_fn,
                scalar_workaround,
                exclusive_arg_names,
                get_decomp_fn=get_decomp_fn,
            )
        else:
            pattern = search_fn_pattern
            gm = None

        for pattern_matcher_pass in (
            pass_dicts if isinstance(pass_dicts, Sequence) else [pass_dicts]
        ):
            if isinstance(pattern_matcher_pass, PatternMatcherPass):
                if check_and_add_duplicate_pattern(
                    pattern,
                    gm.graph if gm else None,
                    pattern_matcher_pass.seen_patterns,
                    skip_duplicates=skip_duplicates,
                ):
                    return False

        pattern = ReplacementPatternEntry(
            pattern=pattern,
            extra_check=check_fn,
            normalize_args=normalize_args,
            pattern_name=pattern_name,
        )
        pattern.register(pass_dicts)
        return pattern.pattern  # type: ignore[return-value]


_serialized_patterns: OrderedSet[str] = OrderedSet()


def _serialize_pattern(
    unique_name: str,
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: dict[str, float | int] | None,
) -> PatternExpr:
    def get_file_template() -> str:
        auto_generated_msg = textwrap.dedent(
            """\
            # This is an auto-generated file. Please do not modify it by hand.
            # To re-generate, run:
            # cd ~/pytorch && python torchgen/fuse/gen_patterns.py
            """
        )

        file_template = textwrap.dedent(
            """\
            # noqa: F401, E501
            {msg}
            import torch
            import torch._inductor
            import operator

            aten = torch.ops.aten
            prims = torch.ops.prims

            """
        ).format(msg=auto_generated_msg)

        pattern_matcher_imports = []
        for name in dir(torch._inductor.pattern_matcher):
            attr = getattr(torch._inductor.pattern_matcher, name)
            try:
                if isinstance(attr, type) and issubclass(
                    attr, (PatternExpr, _TargetExpr)
                ):
                    # pyrefly: ignore [bad-argument-type]
                    pattern_matcher_imports.append(name)
            except TypeError:
                pass

        formatted_imports = ",\n   ".join(pattern_matcher_imports)
        formatted_imports = f"from torch._inductor.pattern_matcher import (\n   {formatted_imports},\n)\n"
        return f"{file_template}{formatted_imports}"

    if not SERIALIZED_PATTERN_PATH.is_dir():
        raise RuntimeError(
            f"Could not find serialized patterns directory at {SERIALIZED_PATTERN_PATH}"
        )

    pattern_name = search_fn.__name__

    from torch._functorch import config as functorch_config

    with functorch_config.patch(functionalize_rng_ops=False):
        pattern = gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround)

    serialized_pattern = PatternPrettyPrinter.run(pattern, output_name=unique_name)
    if pattern_name not in _serialized_patterns:
        write_mode = "w"
        _serialized_patterns.add(pattern_name)
    else:
        write_mode = "a"

    file_template = get_file_template()

    with open(SERIALIZED_PATTERN_PATH / f"{pattern_name}.py", write_mode) as f:
        if write_mode == "w":
            f.write(file_template)
        else:
            f.write("\n\n")
        f.write(serialized_pattern)
        f.write("\n")

    return pattern


SERIALIZED_PATTERN_PATH = Path(__file__).parent / "fx_passes" / "serialized_patterns"


# This is the set of serialized patterns that we've registered.  Used by
# test_serialized_patterns_up_to_date() to ensure the patterns are up
# to date.
@dataclasses.dataclass
class _PrecompiledPattern:
    search_fn: SearchFn
    example_inputs: Sequence[Any]
    trace_fn: TraceFn
    scalar_workaround: dict[str, float | int] | None
    search_fn_pattern: PatternExpr


_known_precompiled_patterns: list[_PrecompiledPattern] = []


def gen_register_replacement(
    unique_name: str,
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: _PassDictsType | Sequence[_PassDictsType],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
    skip_duplicates: bool = False,
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = select_decomp_table,
) -> None:
    # Make sure the example_inputs is materialized.
    example_inputs = tuple(example_inputs)

    if "PYTORCH_GEN_PATTERNS" in os.environ:
        pat = _serialize_pattern(
            unique_name, search_fn, example_inputs, trace_fn, scalar_workaround
        )
    else:
        pattern_name = search_fn.__name__
        m = importlib.import_module(
            f"torch._inductor.fx_passes.serialized_patterns.{pattern_name}"
        )
        if not m or not hasattr(m, unique_name):
            log.warning(
                "Precompiled pattern %r not found. Run torchgen/fuse/gen_patterns.py.",
                unique_name,
            )
        pat = getattr(m, unique_name)

    for arg in pytree.tree_iter(example_inputs):
        if isinstance(arg, FakeTensor) and maybe_get_fake_constant(arg) is not None:  # noqa: ISINSTANCE_FAKE_TENSOR
            # This can be a problem - small fake tensors (e.g. `tensor(2)`) will
            # hold onto their original constant value - and by stashing it here
            # will cause a memory leak if the constant value is on GPU.
            # Since this is just an optimization we can clear it out.
            # for c++ need to add a setter
            arg.constant = None

    _known_precompiled_patterns.append(
        _PrecompiledPattern(search_fn, example_inputs, trace_fn, scalar_workaround, pat)
    )
    register_replacement(
        search_fn,
        replace_fn,
        example_inputs,
        trace_fn,
        pass_dicts,
        extra_check,
        scalar_workaround,
        exclusive_arg_names,
        search_fn_pattern=pat,
        skip_duplicates=skip_duplicates,
        pattern_name=unique_name,
        get_decomp_fn=get_decomp_fn,
    )


@functorch_config.patch(functionalize_rng_ops=False)  # type: ignore[misc]
def gen_pattern_and_search_gm(
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = select_decomp_table,
) -> tuple[PatternExpr, torch.fx.GraphModule]:
    argnames = [*inspect.signature(search_fn).parameters.keys()]

    scalar_workaround = scalar_workaround or {}
    flat_inputs = _trace_args_for_initial_trace(
        argnames, example_inputs, scalar_workaround
    )
    arg_info = _trace_arg_info(argnames, flat_inputs)

    search_gm = trace_fn(search_fn, flat_inputs, get_decomp_fn=get_decomp_fn)
    return (
        fx_to_pattern(
            search_gm,
            ignore_types=(int, float, list, torch.device, torch.dtype),
            argnames=arg_info.flat_argnames,
            scalar_workaround=scalar_workaround,
            exclusive_arg_names=exclusive_arg_names,
        ),
        search_gm,
    )


def gen_pattern(
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
) -> PatternExpr:
    return gen_pattern_and_search_gm(
        search_fn, example_inputs, trace_fn, scalar_workaround, exclusive_arg_names
    )[0]


def register_lowering_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = _return_true,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = False,
    output_metadata_ignores_input_storage: bool = True,
    output_metadata_is_input: int | str | None = None,
    output_metadata_fn: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register an aten to inductor IR replacement pattern.  The decorated
    function is saved and then called at lowering time allowing direct
    pattern to inductor IR conversion.
    """

    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(handler):
            raise AssertionError(f"expected callable handler, got {handler}")
        LoweringPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        handler._inductor_lowering_function = True  # type: ignore[attr-defined]
        handler._inductor_lowering_output_metadata_ignores_input_storage = (  # type: ignore[attr-defined]
            output_metadata_ignores_input_storage
        )
        handler._inductor_lowering_output_metadata_is_input = (  # type: ignore[attr-defined]
            output_metadata_is_input
        )
        handler._inductor_lowering_output_metadata_fn = output_metadata_fn  # type: ignore[attr-defined]
        return handler

    return decorator


def register_graph_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = _return_true,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register a pattern that runs a function on the FX graph, allowing
    custom transformation code.
    """

    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(handler):
            raise AssertionError(f"expected callable handler, got {handler}")
        GraphPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        return handler

    return decorator


def is_start_of_fx_graph(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    # first node in the graph
    return node is next(iter(graph.nodes))


# match: copy_, relu_, _set_grad_enabled, manual_seed, _enter_autocast, etc
# doesn't match: __rshift__, etc
_mutation_op_re = re.compile(r"(?<!_)(_$|_[.]|(\b|_)(set|enter|exit|seed)(\b|_))(?!_)")


def fixme_incorrect_inductor_schema_op(op: torch._ops.OpOverload) -> bool:
    if op.namespace != "inductor":
        return False

    # TODO - fix schema
    # Don't add any more!
    return op in (torch.ops.inductor.resize_storage_bytes_.default,)


def is_mutation_op(node: torch.fx.Node) -> bool:
    if isinstance(
        node.target, torch._ops.OpOverload
    ) and not fixme_incorrect_inductor_schema_op(node.target):
        return node.target._schema.is_mutable
    elif isinstance(
        node.target, torch._higher_order_ops.auto_functionalize.AutoFunctionalized
    ):
        return False
    if node.op == "call_function":
        if not callable(node.target):
            raise AssertionError(f"expected callable node.target, got {node.target}")
        if _mutation_op_re.search(node.target.__name__):
            return True
    elif node.op == "call_method":
        if not isinstance(node.target, str):
            raise AssertionError(f"expected str node.target, got {type(node.target)}")
        if _mutation_op_re.search(node.target):
            return True
    return node.kwargs.get("out") is not None


def same_mutation_regions(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    if "mutation_region_id" not in a.meta:
        raise AssertionError("mutation_region_id missing from a.meta")
    if "mutation_region_id" not in b.meta:
        raise AssertionError("mutation_region_id missing from b.meta")
    return a.meta["mutation_region_id"] == b.meta["mutation_region_id"]


def get_mutation_region_id(graph: torch.fx.Graph, node: torch.fx.Node) -> int:
    n = node
    while "mutation_region_id" not in n.meta and not is_start_of_fx_graph(graph, n):
        n = n.prev
    mutation_region_id = n.meta.get("mutation_region_id", 0)
    while n is not node:
        n = n.next
        if is_mutation_op(n):
            mutation_region_id += 1
        n.meta["mutation_region_id"] = mutation_region_id
    return mutation_region_id


def should_compute_mutation_region_ids(graph: torch.fx.Graph) -> bool:
    return "mutation_region_id" not in next(iter(graph.nodes)).meta


def compute_mutation_region_ids(graph: torch.fx.Graph) -> None:
    mutation_region_id = 0
    for nd in graph.nodes:
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta["mutation_region_id"] = mutation_region_id


@dataclasses.dataclass(frozen=True)
class _MutationRegionRefresh:
    anchor: torch.fx.Node
    stop: torch.fx.Node
    mutation_region_id: int
    mutation_count: int


@dataclasses.dataclass
class _GraphMutationTracker:
    created_nodes: list[torch.fx.Node] = dataclasses.field(default_factory=list)
    erased_mutation_node: bool = False

    def changed_mutation_regions(self) -> bool:
        return self.erased_mutation_node or _contains_mutation_op(self.created_nodes)


@contextlib.contextmanager
def _track_graph_mutation_ops(
    graph: torch.fx.Graph,
    custom_context: dict[str, Any] | None = None,
) -> Generator[_GraphMutationTracker, None, None]:
    custom_context = custom_context or {}
    tracker = _GraphMutationTracker()
    create_node = graph.create_node
    erase_node = graph.erase_node
    sentinel = object()
    graph_dict = graph.__dict__
    create_node_attr = graph_dict.get("create_node", sentinel)
    erase_node_attr = graph_dict.get("erase_node", sentinel)

    def tracked_create_node(*args: Any, **kwargs: Any) -> torch.fx.Node:
        created_node = create_node(*args, **kwargs)
        _merge_custom_context(created_node.meta, custom_context)
        tracker.created_nodes.append(created_node)
        return created_node

    def tracked_erase_node(to_erase: torch.fx.Node) -> Any:
        if not to_erase._erased and is_mutation_op(to_erase):
            tracker.erased_mutation_node = True
        return erase_node(to_erase)

    graph_dict["create_node"] = tracked_create_node
    graph_dict["erase_node"] = tracked_erase_node
    try:
        yield tracker
    finally:
        if create_node_attr is sentinel:
            del graph_dict["create_node"]
        else:
            graph_dict["create_node"] = create_node_attr
        if erase_node_attr is sentinel:
            del graph_dict["erase_node"]
        else:
            graph_dict["erase_node"] = erase_node_attr


def _count_mutation_ops_until(start: torch.fx.Node, stop: torch.fx.Node) -> int:
    mutation_count = 0
    nd = start
    while nd is not stop:
        if nd.op == "root":
            raise AssertionError(f"unexpected root node reached: {nd}")
        if is_mutation_op(nd):
            mutation_count += 1
        nd = nd.next
    return mutation_count


def _contains_mutation_op(nodes: Iterable[torch.fx.Node]) -> bool:
    return any(is_mutation_op(nd) for nd in nodes)


def _replacement_changes_mutation_regions(
    entry: ReplacementPatternEntry, match: Match
) -> bool:
    if match.replacement_graph is None:
        raise AssertionError("match.replacement_graph must not be None")
    return _contains_mutation_op(match.nodes) or _contains_mutation_op(
        match.replacement_graph.graph.nodes
    )


def _mutation_region_refresh(
    graph: torch.fx.Graph,
    nodes: Sequence[torch.fx.Node],
) -> _MutationRegionRefresh:
    if nodes:
        first_matched_node = min(nodes)
        last_matched_node = max(nodes)
        anchor = first_matched_node.prev
        stop = last_matched_node.next
    else:
        first_node = next(iter(graph.nodes))
        anchor = first_node.prev
        stop = anchor

    mutation_region_id = (
        0 if anchor.op == "root" else get_mutation_region_id(graph, anchor)
    )
    return _MutationRegionRefresh(
        anchor,
        stop,
        mutation_region_id,
        _count_mutation_ops_until(anchor.next, stop),
    )


def _refresh_mutation_region_ids(
    graph: torch.fx.Graph,
    refresh: _MutationRegionRefresh,
) -> None:
    if refresh.anchor._erased or refresh.stop._erased:
        compute_mutation_region_ids(graph)
        return

    nd = refresh.anchor.next
    mutation_region_id = refresh.mutation_region_id
    mutation_count = 0
    while nd is not refresh.stop:
        if nd.op == "root":
            compute_mutation_region_ids(graph)
            return
        if is_mutation_op(nd):
            mutation_region_id += 1
            mutation_count += 1
        nd.meta["mutation_region_id"] = mutation_region_id
        nd = nd.next

    if mutation_count == refresh.mutation_count:
        return

    while nd.op != "root":
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta["mutation_region_id"] = mutation_region_id
        nd = nd.next


def _wrap_bound_method(fn: Any, argnames: list[str]) -> Any:
    """
    Wrap a bound method to remove 'self' from its signature for FX tracing.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in argnames
    ]
    wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    return wrapper


class PatternMatcherPass:
    """
    Registry of patterns to match and replace in FX graphs.
    """

    def __init__(
        self,
        pass_name: str | None = None,
        subsystem: str | None = None,
    ) -> None:
        super().__init__()
        self.patterns: defaultdict[
            tuple[str, torch.fx.node.Target], list[PatternEntry]
        ] = defaultdict(list)
        self.pass_name = pass_name
        self.subsystem = subsystem

        # For a particular generated pattern repr, store all of the str representations
        # of the graph used to generate them. Because we ignore certain patterns
        # in searching, but not in matching, use the graph to distinguish if two equivalent
        # searches are actually different.
        self.seen_patterns: dict[str, list[str | None]] = defaultdict(list)

    def __getitem__(self, item: tuple[str, torch.fx.node.Target]) -> list[PatternEntry]:
        return self.patterns[item]

    def apply(self, gm: torch.fx.GraphModule | torch.fx.Graph) -> int:
        """Apply all registered patterns to the graph, returning the number of matches."""
        if not self.patterns:
            return 0
        if isinstance(gm, torch.fx.GraphModule):
            graph = gm.graph
        elif isinstance(gm, torch.fx.Graph):
            graph = gm
            gm = graph.owning_module  # type: ignore[assignment]
        else:
            raise RuntimeError(
                f"The input to PatternMatcherPass must be a GraphModule or a Graph, but got {type(gm)}"
            )
        if should_compute_mutation_region_ids(graph):
            compute_mutation_region_ids(graph)
        get_mutation_region_id_partial = functools.partial(
            get_mutation_region_id, graph
        )
        count = 0
        nodes = []
        has_call_module = False
        for op, target in self.patterns:
            if op == "call_module":
                has_call_module = True
            else:
                nodes.append(graph.find_nodes(op=op, target=target, sort=False))
        if has_call_module:
            nodes.append(graph.find_nodes(op="call_module", sort=False))
        pass_name = self.pass_name if self.pass_name is not None else "pattern_matcher"
        if not isinstance(gm, torch.fx.GraphModule):
            raise AssertionError(f"expected GraphModule, got {type(gm)}")
        with GraphTransformObserver(gm, pass_name, self.subsystem):
            for node in sorted(itertools.chain.from_iterable(nodes), reverse=True):
                target = extract_target(node)
                if node.op == "call_module":
                    if (node.op, target) not in self.patterns:
                        continue

                # conservatively not applying pattern for cpu input,
                # since some of the patterns induce codegen and split nodes.
                # Note: we will only skip cpu compute if disable_cpp_codegen=True
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue

                for entry in self.patterns[(node.op, target)]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    # pattern match crosses mutation barrier - discard
                    if (
                        is_match(m)
                        and len(
                            OrderedSet(map(get_mutation_region_id_partial, m.nodes))
                        )
                        != 1
                    ):
                        continue
                    # pattern match crosses stream boundary - discard
                    if (
                        is_match(m)
                        and len(
                            OrderedSet(
                                (
                                    n.meta.get("custom", {}).get("stream", 0),
                                    n.meta.get("custom", {}).get("mempool"),
                                    n.meta.get("custom", {}).get("mempool_device"),
                                )
                                for n in m.nodes
                            )
                        )
                        != 1
                    ):
                        continue
                    if _should_debug_node(node.name):
                        log.warning("%s%s %s %s", node, node.args, m, entry.pattern)

                    if is_match(m) and guard_or_false(entry.extra_check(m)):
                        count += 1
                        if isinstance(entry, ReplacementPatternEntry):
                            mutation_region_refresh = (
                                _mutation_region_refresh(graph, m.nodes)
                                if _replacement_changes_mutation_regions(entry, m)
                                else None
                            )
                        elif isinstance(entry, GraphPatternEntry):
                            mutation_region_refresh = None
                        else:
                            mutation_region_refresh = _mutation_region_refresh(
                                graph, m.nodes
                            )
                        if isinstance(entry, GraphPatternEntry):
                            with _track_graph_mutation_ops(
                                graph, _common_custom_context(m.nodes)
                            ) as mutation_tracker:
                                entry.apply(m, graph, node)
                            if mutation_tracker.changed_mutation_regions():
                                compute_mutation_region_ids(graph)
                        elif mutation_region_refresh is not None:
                            entry.apply(m, graph, node)
                            _refresh_mutation_region_ids(graph, mutation_region_refresh)
                        else:
                            entry.apply(m, graph, node)
                        counters[backend]["pattern_matcher_count"] += 1
                        counters[backend]["pattern_matcher_nodes"] += len(m.nodes)

                        # Track per-pattern counts when debug mode is active
                        if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG"):
                            if getattr(entry, "pattern_name", None):
                                pattern_name = entry.pattern_name
                            else:
                                # Fallback: use pattern class name + operation target
                                pattern_class = entry.pattern.__class__.__name__
                                target = str(node.target)
                                pattern_name = f"{pattern_class}_{target}"

                            pattern_key = f"{backend}_pattern_matcher_per_pattern"
                            counters[pattern_key][pattern_name] += 1
        return count

    def clear(self) -> None:
        self.patterns.clear()


def _not_implemented(*args: Any, **kwargs: Any) -> NoReturn:
    raise NotImplementedError


def fx_to_pattern(
    gm: torch.fx.GraphModule | torch.fx.Graph,
    ignore_types: Sequence[type[Any]] = (),
    argnames: Sequence[str] = (),
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
) -> PatternExpr:
    """
    Convert an FX graph into a PatternExpr.  This is useful for simple
    patterns that can only match single functions and fixed-length lists.
    """
    # scalar_workaround is a hack to capture dropout_p
    # see https://github.com/pytorch/pytorch/issues/97894
    scalar_workaround = scalar_workaround or {}
    inv_scalar_workaround = {v: k for k, v in scalar_workaround.items()}
    if len(inv_scalar_workaround) != len(scalar_workaround):
        raise AssertionError("scalar_workaround has duplicate values")

    def process_arg(
        x: T, ignore_types_override: Sequence[type[Any]] | None = None
    ) -> T | KeywordArg | Ignored:
        current_ignore_types = (
            ignore_types_override if ignore_types_override is not None else ignore_types
        )
        if isinstance(x, (float, int)) and x in inv_scalar_workaround:
            return KeywordArg(inv_scalar_workaround[x])
        if type(x) in current_ignore_types:
            return Ignored()
        if isinstance(x, list) and all(isinstance(y, Ignored) for y in x) and x:
            return Ignored()
        return x

    argnum = itertools.count()

    class Converter(torch.fx.Interpreter):
        # pyrefly: ignore [bad-override]
        call_method = _not_implemented
        # pyrefly: ignore [bad-override]
        call_module = _not_implemented

        # pyrefly: ignore [bad-override]
        def placeholder(
            self,
            target: str,  # type: ignore[override]
            args: Sequence[Any],
            kwargs: Mapping[str, Any],
        ) -> ExclusiveKeywordArg | KeywordArg:
            n = next(argnum)
            if n < len(argnames):
                name = argnames[n]
            elif argnames:
                if not target.startswith("tangent"):
                    raise AssertionError(
                        f"expected target to start with 'tangent', got {target}"
                    )
                name = target
            else:
                target = re.sub(r"_\d+$", "", target)  # de-mangle arg name
                name = target
            if name in exclusive_arg_names:
                return ExclusiveKeywordArg(name)
            else:
                return KeywordArg(name)

        # pyrefly: ignore [bad-override]
        def get_attr(
            self,
            target: str,  # type: ignore[override]
            args: Sequence[Any],
            kwargs: Mapping[str, Any],
        ) -> GetAttr:
            if args or kwargs:
                raise AssertionError(f"unexpected get_attr args: {args}, {kwargs}")
            return GetAttr(_get_attr(self.module, target))

        # pyrefly: ignore [bad-override]
        def call_function(
            self,
            target: str,  # type: ignore[override]
            args: Sequence[Any],
            kwargs: Mapping[str, Any],
        ) -> PatternExpr:
            process_arg_fn = process_arg
            # Indexing is critical for matching getitem nodes, so we can't ignore int args here
            if target is operator.getitem:

                def process_arg_fn_impl(
                    x: T,
                    ignore_types_override: Sequence[type[Any]] | None = tuple(
                        t for t in ignore_types if t is not int
                    ),
                ) -> T | KeywordArg | Ignored:
                    return process_arg(x, ignore_types_override)

                process_arg_fn = process_arg_fn_impl

            args, kwargs = pytree.tree_map(process_arg_fn, (args, kwargs))
            if list in ignore_types:
                # Handle a burned in tensor size which are now [Ignored(), Ignored(), ...]
                args = [process_arg_fn(a) for a in args]
                kwargs = {k: process_arg_fn(a) for k, a in kwargs.items()}
            return CallFunction(target, *args, **kwargs)

        def run_node(self, n: torch.fx.Node) -> Any:
            rv = super().run_node(n)
            if n.op == "output" and isinstance(rv, tuple):
                args = n.args[0]
                if not isinstance(args, Collection):
                    raise AssertionError(f"expected Collection, got {type(args)}")
                if len(rv) != len(args):
                    raise AssertionError("rv and args length mismatch")
                for r, arg in zip(rv, args):
                    # pyrefly: ignore [missing-attribute]
                    r.users = len(arg.users)
            else:
                rv.users = len(n.users)
            return rv

    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(f"expected GraphModule, got {type(gm)}")
    pattern = Converter(gm).run()
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_leaves(pattern))
    return pattern


@torch.no_grad()
def fwd_only(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    run_functional_passes: bool = True,
    get_decomp_fn: Callable[..., Any] = select_decomp_table,
) -> torch.fx.GraphModule:
    """Build a normalized inference graph, for use with fx_to_pattern"""
    from torch.compiler import config as compiler_config

    # Patterns are device-agnostic templates traced with fixed example tensors; keep the
    # compile-on-one-rank device handling out of pattern tracing so make_fx's single-device
    # check only validates real user graphs, not these internal fixed-device templates.
    # TODO - look into using aot autograd, asserting no mutating ops here
    with (
        # pyrefly: ignore [missing-attribute]
        compiler_config.patch(compile_on_one_rank=False),
        enable_python_dispatcher(),
        preserve_node_meta(),
    ):
        gm = make_fx(fn, get_decomp_fn(), tracing_mode="real")(*args)

    from .fx_passes.post_grad import remove_noop_ops

    if run_functional_passes:
        remove_noop_ops(gm.graph)

        # NOTE: applying early_patterns to user patterns cause
        # duplicate patterns being found in vllm. Check
        # https://github.com/pytorch/pytorch/pull/170649#issuecomment-3693427775
        # for more details.
        # from .fx_passes.joint_graph import early_patterns
        # early_patterns.apply(gm.graph)

        gm.graph.eliminate_dead_code()

    gm.recompile()
    return gm


@torch.enable_grad()
def joint_fwd_bwd(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    get_decomp_fn: Callable[..., Any] = select_decomp_table,
) -> torch.fx.GraphModule:
    """Build a normalized training graph, for use with fx_to_pattern"""
    gm: torch.fx.GraphModule | None = None

    def record_joint_graph(
        joint_graph: torch.fx.GraphModule, inputs: Sequence[Any], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        nonlocal gm
        if gm:
            raise AssertionError("gm already set")
        gm = clone_graph(joint_graph)
        return default_partition(joint_graph, inputs, **kwargs)

    from torch.compiler import config as compiler_config

    # Keep compile-on-one-rank device handling out of pattern tracing (see fwd_only).
    # pyrefly: ignore [missing-attribute]
    with torch._guards.tracing(None), compiler_config.patch(compile_on_one_rank=False):
        aot_function(
            fn,
            # pyrefly: ignore[bad-argument-type]
            lambda gm, example_inputs: make_boxed_func(gm),
            partition_fn=record_joint_graph,
            decompositions=get_decomp_fn(),
            keep_inference_input_mutations=True,
            enable_log=False,
        )(*args)
    if not gm:
        raise AssertionError("gm was not set")

    from .fx_passes.post_grad import remove_noop_ops

    remove_noop_ops(gm.graph)

    from .fx_passes.joint_graph import early_patterns

    early_patterns.apply(gm.graph)

    # remove in/out specs
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _args(n: torch.fx.Node) -> list[torch.fx.node.Argument]:
    args: list[torch.fx.node.Argument] = []
    torch.fx.map_arg((n.args, n.kwargs), args.append)
    return args


def stable_topological_sort(graph: torch.fx.Graph) -> None:
    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = OrderedSet[torch.fx.Node]()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()
        waiting_for = [x for x in _args(node) if x not in ready]
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    if not (not waiting and len(ready) == len(graph.nodes)):
        raise AssertionError("topological sort did not consume all nodes")


def init_once_fakemode(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper around lazy init functions in fx_passes/"""

    _fn_params = inspect.signature(fn).parameters

    @functools.cache
    @functools.wraps(fn)
    def lazy_init(
        input_device: Any | None = None,
        get_decomp_fn: Callable[
            ..., dict[Any, Callable[..., Any]]
        ] = select_decomp_table,
    ) -> Any:
        counters_ref = counters[backend].copy()

        kwargs: dict[str, Any] = {}
        if "input_device" in _fn_params:
            kwargs["input_device"] = input_device
        if "get_decomp_fn" in _fn_params:
            kwargs["get_decomp_fn"] = get_decomp_fn

        with torch._guards.tracing(None), unset_fake_temporarily(), FakeTensorMode():
            result = fn(**kwargs)

        # clear view matches encountered during tracing
        counters[backend] = counters_ref

        return result

    return lazy_init


def config_flag(name: str) -> Callable[[Match], Any]:
    """Function for extra_check to put pass behind a flag"""

    def flag_check(match: Match) -> Any:
        return getattr(config, name)

    return flag_check


def clone_graph(input_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    class CopyGraph(Transformer):
        def run_node(self, old_node: torch.fx.Node) -> torch.fx.Node:
            new_node = super().run_node(old_node)
            if isinstance(new_node, torch.fx.Proxy):
                new_node.node.meta.update(old_node.meta)
                new_node.node.name = self.new_graph._graph_namespace.create_name(
                    old_node.name, None
                )

            return new_node

    return CopyGraph(input_graph).transform()


# TODO: remove in follow up diff, used internally
_seen_patterns: OrderedSet[str] = OrderedSet()


def get_arg_value(
    node: torch.fx.Node, arg_number: int, kwarg_name: str | None = None
) -> Any:
    if len(node.args) > arg_number:
        return node.args[arg_number]
    elif kwarg_name is None:
        return None
    else:
        return node.kwargs.get(kwarg_name)


def filter_nodes(nodes: Iterable[torch.fx.Node], fn: Any) -> list[torch.fx.Node]:
    fns = [fn]
    if isinstance(fn, torch._ops.OpOverloadPacket):
        fns.extend(fn.op_overloads())

    return [node for node in nodes if node.target in fns]


def extract_target(node: torch.fx.Node) -> torch.fx.node.Target:
    """For call_function and call_method, we directly use the target function;
    For call_module, the target is string, and we treat the module class
     as a function.
    """
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"expected str node.target, got {type(node.target)}")
        if node.graph.owning_module is None:
            raise AssertionError("node.graph.owning_module must not be None")
        return _get_attr(node.graph.owning_module, node.target).__class__
    return node.target
<|endoftext|>"""Stream utilities for Inductor codegen."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from torch._inductor.stream_constants import (
    DEFAULT_STREAM,
    DEFAULT_STREAM_IDX,
    STREAM_NAME_TEMPLATE,
)


if TYPE_CHECKING:
    import torch


__all__ = [
    "AOTI_SUPPORTED_STREAM_OP_NAMES",
    "AOTI_UNSUPPORTED_STREAM_OP_REASONS",
    "DEFAULT_STREAM",
    "DEFAULT_STREAM_IDX",
    "STREAM_NAME_TEMPLATE",
    "get_raw_stream_name",
    "get_stream_name",
]


AOTI_SUPPORTED_STREAM_OP_NAMES: dict[str, str] = {
    "torch.ops.streams.record_event.default": "record_event",
    "torch.ops.streams.wait_event.default": "wait_event",
    # A targeted single-event host wait (cudaEventSynchronize); required by e.g.
    # pinned non-blocking copies. Unlike synchronize_stream/device, it does not
    # block the whole stream/device, so it is safe to emit inside an AOTI Run().
    "torch.ops.streams.synchronize_event.default": "synchronize_event",
}


AOTI_UNSUPPORTED_STREAM_OP_REASONS: dict[str, str] = {
    "torch.ops.streams.synchronize_stream.default": (
        "Host-blocking sync ops are not supported inside an AOTI Run(). "
        "Use record_event + wait_event for device-side ordering instead."
    ),
    "torch.ops.streams.synchronize_device.default": (
        "Host-blocking device synchronization is not supported inside an AOTI Run()."
    ),
    "torch.ops.streams.wait_stream.default": (
        "wait_stream is not supported in AOTI cpp_wrapper. Use explicit "
        "record_event on the waited-on stream + wait_event on the waiting stream."
    ),
}


@functools.lru_cache
def get_stream_name(stream_idx: int) -> str:
    """Generate CUDA Stream name from stream index number.

    Args:
        stream_idx: Non-negative index number. 0 refers to the default stream, others refer to side
            streams.
    """
    if stream_idx == 0:
        return DEFAULT_STREAM
    else:
        return STREAM_NAME_TEMPLATE.format(stream_idx=stream_idx)


@functools.lru_cache
def _raw_stream_name_for_device(device_idx: int) -> str:
    return f"raw_stream{device_idx}"


def get_raw_stream_name(device_idx: int) -> str:
    """Generate variable name for a raw stream handle on the given device."""
    # Under compile-on-one-rank the wrapper must be byte-identical across ranks, so the
    # stream variable name cannot carry a rank-specific device index.
    from torch.fx.experimental.proxy_tensor import _coor_enabled

    if _coor_enabled():
        return "raw_stream"
    return _raw_stream_name_for_device(device_idx)


# [device-as-parameter] Name of the function-local variable holding the runtime current
# device index under compile-on-one-rank, so generated code is byte-identical across ranks.
COOR_DEVICE_IDX_VAR = "_coor_device_idx"


def coor_device_str(device: torch.device | None) -> str:
    """Device string to emit into generated code (benchmark and autotune harnesses).

    Under compile-on-one-rank drop the index (``cuda:0`` -> ``cuda``) so the emitted text
    is rank-identical and its tensors land on the running rank's current device.
    """
    from torch.fx.experimental.proxy_tensor import _coor_enabled

    if _coor_enabled() and device is not None:
        return device.type
    return str(device)


def coor_benchmark_device_idx(device_idx: int) -> tuple[str | None, int | str]:
    """(preamble line, index token) for a Triton kernel's benchmark harness.

    The harness is emitted into the kernel's own source, which is what the kernel hash
    covers, so under compile-on-one-rank it must resolve the device at run time rather
    than bake a rank-specific index -- otherwise kernels are not rank-identical whenever
    ``config.benchmark_kernel`` is on. The preamble line (when not None) must be emitted
    inside the harness function before the returned token is used.
    """
    from torch.fx.experimental.proxy_tensor import _coor_enabled

    if not _coor_enabled():
        return None, device_idx

    from torch._inductor.virtualized import V

    expr = V.graph.device_ops.current_device_idx_expr()
    return f"{COOR_DEVICE_IDX_VAR} = {expr}", COOR_DEVICE_IDX_VAR
<|endoftext|>import functools
from collections.abc import Callable, Sequence
from typing import Protocol

import sympy

import torch

from .virtualized import OpsValue, V


BlockShapeType = Sequence[int | str] | None


class ShapeVar(Protocol):
    @property
    def shape(self) -> BlockShapeType: ...


ShapeArg = ShapeVar | torch.types.Number | str | OpsValue | torch.dtype

# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


def get_broadcasted_shape(a: BlockShapeType, b: BlockShapeType) -> BlockShapeType:
    if not isinstance(a, Sequence):
        raise AssertionError(f"expected a to be a Sequence, got {type(a)}")
    if not isinstance(b, Sequence):
        raise AssertionError(f"expected b to be a Sequence, got {type(b)}")
    return _get_broadcasted_shape(tuple(a), tuple(b))


@functools.lru_cache(None)
def _get_broadcasted_shape(
    a: tuple[int | str, ...], b: tuple[int | str, ...]
) -> BlockShapeType:
    if len(a) > len(b):
        return _get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    elif len(a) < len(b):
        b, a = a, b
        return _get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    else:

        def _get_broadcasted_dim(d1: int | str, d2: int | str) -> int | str:
            if str(d1) == "1":
                return d2
            elif str(d2) == "1":
                return d1
            if str(d1) != str(d2):
                raise AssertionError(f"expected str(d1) == str(d2), got {d1} != {d2}")
            return d1

        return tuple(_get_broadcasted_dim(d1, d2) for d1, d2 in zip(a, b))


def broadcast_shapes_for_args(args: Sequence[ShapeArg]) -> BlockShapeType:
    result_shape: BlockShapeType = None

    for arg in args:
        if hasattr(arg, "shape"):
            shape = arg.shape
            if shape is None:
                return None
            elif result_shape is None:
                result_shape = tuple(shape)
            else:
                result_shape = get_broadcasted_shape(result_shape, tuple(shape))
        elif isinstance(arg, (int, float)):
            if result_shape is None:
                result_shape = ()
        elif isinstance(arg, torch.dtype):
            continue
        else:
            from torch._inductor.loop_body import LoopBody, LoopBodyBlock

            if isinstance(arg, (LoopBodyBlock, LoopBody, OpsValue)):
                # TODO: fix me
                return None
            raise TypeError(f"Unknown type: {type(arg)}")

    return result_shape


class ShapePropagationOpsHandler:
    """
    Propagate shape from args to output
    """

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> BlockShapeType:
        # TritonKernelOverrides.constant uses tl.full with shape=[1]*ndim for all types
        from torch._inductor.codegen.triton import TritonKernel

        if isinstance(V.kernel, TritonKernel):
            ndim = V.kernel.triton_tensor_ndim()
            return tuple([1] * ndim)
        else:
            return ()

    @staticmethod
    def store_reduction(name: str, index: int, value: ShapeArg) -> None:
        return None

    @staticmethod
    def reduction(
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: ShapeArg | tuple[ShapeArg, ...],
    ) -> BlockShapeType | tuple[BlockShapeType, ...]:
        raise NotImplementedError

    @staticmethod
    def store(name: str, index: int, value: ShapeArg, mode: str | None = None) -> None:
        return None

    @staticmethod
    def to_dtype(
        value: ShapeVar,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> BlockShapeType:
        return value.shape

    @staticmethod
    def dot(a: sympy.Expr, b: sympy.Expr) -> BlockShapeType:
        from torch._inductor.codegen.triton import TritonKernel

        if not isinstance(V.kernel, TritonKernel):
            raise AssertionError("dot supports Triton only")
        return ("YBLOCK", "XBLOCK")

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> BlockShapeType:
        # shape is implicitly embedded in expr.
        return None

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> BlockShapeType:
        # shape is implicitly embedded in expr.
        return None

    @staticmethod
    def load_seed(name: str, offset: int) -> BlockShapeType:
        return ()

    @staticmethod
    def indirect_indexing(
        var: ShapeArg,
        size: sympy.Expr | int,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> None:
        return None

    def __getattr__(self, name: str) -> Callable[..., BlockShapeType]:
        return lambda *args, **kwargs: broadcast_shapes_for_args(args)

    @staticmethod
    def device_assert_async(cond: ShapeArg, msg: str) -> None:
        return None
<|endoftext|>import collections
import contextlib
import copy
import dataclasses
import functools
import io
import itertools
import json
import logging
import os
import os.path
import pickle
import pstats
import shutil
import tempfile
import traceback
from collections.abc import Callable, Iterator, Sequence
from typing import Any, IO
from unittest.mock import patch

import torch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
from torch import fx
from torch._dynamo.repro.after_aot import save_graph_repro
from torch._dynamo.utils import get_debug_dir
from torch._functorch import config as functorch_config
from torch._inductor import utils
from torch._logging import getArtifactLogger
from torch._logging._internal import trace_structured
from torch._utils_internal import signpost_event
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.types import FileLike
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_leaves, tree_map

from . import config, ir
from .ir import ExternKernel
from .scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    OutputNode,
    SchedulerNode,
)
from .virtualized import V


log = logging.getLogger(__name__)

# Graph execution tracking for debugging
GRAPH_EXECUTION_ORDER: list[dict[str, object]] | None = None
RECORD_GRAPH_EXECUTION: bool = False
GRAPH_COMPILE_IDS: dict[int, str | None] | None = None

ir_pre_fusion_log = getArtifactLogger(__name__, "ir_pre_fusion")
ir_post_fusion_log = getArtifactLogger(__name__, "ir_post_fusion")
SchedulerNodeList = list[Any]
BufMeta = collections.namedtuple("BufMeta", ["name", "n_origin"])
GRAPHVIZ_COMMAND_SCALABLE = ["dot", "-Gnslimit=2", "-Gnslimit1=2", "-Gmaxiter=5000"]
_RAW_DOT_GRAPH_FORMATS = OrderedSet(["dot", "raw"])


@functools.cache
def has_dot() -> bool:
    return shutil.which("dot") is not None


def _get_graph_output_format(fname: str | None) -> str:
    if fname is not None:
        _, ext = os.path.splitext(fname)
        if ext:
            return ext.lstrip(".")
    return functorch_config.torch_compile_graph_format


def draw_buffers(
    nodes: list[BaseSchedulerNode],
    print_graph: bool = False,
    fname: str | None = None,
) -> None:
    """
    Draw a graph in fname.
    """
    graph_format = _get_graph_output_format(fname)
    if graph_format not in _RAW_DOT_GRAPH_FORMATS and not has_dot():
        log.warning("draw_buffers() requires `graphviz` package")
        return

    if fname is None:
        fname = get_graph_being_compiled()

    graph = create_fx_from_snodes(nodes)

    for node in graph.nodes:
        if "fusion_meta" not in node.meta:
            continue
        group = node.meta["fusion_meta"].group
        if isinstance(group, tuple):
            if isinstance(group[1], int):
                group = (group[1],)
            else:
                group = group[1]
        elif isinstance(group, str):
            # extern / template / nop nodes store a string like "extern"
            # as the group instead of a shape tuple.  Extract the real
            # output shape from the scheduler node's first output buffer.
            try:
                snode = node.meta["fusion_meta"].snode
                size = snode.get_outputs()[0].node.maybe_get_size()
                group = tuple(size) if size else ()
            except Exception:
                group = ()

        # gather meta data
        dtype = None
        if isinstance(node, ir.ComputedBuffer):
            dtype = node.data.dtype

        metadata = TensorMetadata(group, dtype, None, None, None, None, None)  # type: ignore[arg-type]

        node.meta["tensor_meta"] = metadata

    if print_graph:
        print(graph)

    gm = GraphModule({}, graph)
    legalize_graph(gm)
    gm.graph.lint()
    draw_graph(
        gm, fname, clear_meta=False, dot_graph_shape=config.trace.dot_graph_shape
    )


def create_fx_from_snodes(snodes: list[BaseSchedulerNode]) -> fx.Graph:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """

    def get_fake_func(name: str) -> Callable[..., int]:
        def func1(*args: Any) -> int:
            return 0

        func1.__name__ = name
        return func1

    FusionMeta = collections.namedtuple("FusionMeta", ["group", "snode", "type"])

    buf_to_fx_node = {}
    node_to_fx_node = {}
    graph = torch.fx.Graph()
    first_node = None

    outputs = []
    group: Any = None
    # create call_function node for each Buffer and Kernel
    # pyrefly: ignore [bad-assignment]
    for snode in snodes:
        if snode.is_extern():
            node_type = "extern"
            group = node_type
        elif snode.is_template():
            node_type = "template"
            group = node_type
        elif isinstance(snode, NopKernelSchedulerNode):
            node_type = "nop"
            group = node_type
        elif isinstance(snode, SchedulerNode):
            node_type = "compute"
            group = snode.group
        elif isinstance(snode, FusedSchedulerNode):
            node_type = "fused"
            group = snode.group
        else:
            raise RuntimeError("Unknown node type")

        fused_name = torch._inductor.utils.get_fused_kernel_name(
            snode.get_nodes(), "original_aten"
        )
        func_name = f"{node_type}: {fused_name}"
        node_func = get_fake_func(func_name)
        kwargs = {}
        if hasattr(snode, "get_device"):
            kwargs = {"device": snode.get_device()}
        fx_node = graph.call_function(node_func, args=(), kwargs=kwargs)  # type: ignore[arg-type]

        def in_output(snode: BaseSchedulerNode | FusedSchedulerNode) -> bool:
            if isinstance(snode, FusedSchedulerNode):
                return any(in_output(x) for x in snode.snodes)
            return any(
                isinstance(user.node, OutputNode)
                for buf in snode.get_outputs()
                for user in buf.users
            )

        if in_output(snode):
            outputs.append(fx_node)
        name = snode.get_name()
        fx_node.name = name

        fx_node.meta["fusion_meta"] = FusionMeta(group, snode, node_type)

        node_to_fx_node[name] = fx_node
        for buf in snode.get_outputs():
            buf_to_fx_node[buf.get_name()] = fx_node

        if first_node is None:
            first_node = fx_node

    # create edges between nodes
    for snode in snodes:
        name = snode.get_name()
        deps = snode.read_writes.reads

        fx_node = node_to_fx_node[name]
        new_args = []
        for dep in deps:
            if dep.name in buf_to_fx_node:
                dep_node = buf_to_fx_node[dep.name]
            else:
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)
                    buf_to_fx_node[dep.name] = dep_node
            if dep_node == fx_node:  # to avoid cycles
                continue
            new_args.append(dep_node)

        fx_node.args = tuple(new_args)

    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    return graph


def update_orig_fx_node_name_to_buf_name(
    nodes: SchedulerNodeList | None,
    node_name_to_buf_name: dict[str, str],
    parent_buf_name: str | None = None,
    n_origins: int = 0,
) -> None:
    if nodes is None:
        return
    for node in nodes:
        # for FusedSchedulerNode, traverse recursively into get_nodes()
        buf_name = node.get_name()
        children_nodes = node.get_nodes()
        if children_nodes is not None and len(children_nodes) > 1:
            update_orig_fx_node_name_to_buf_name(
                children_nodes,
                node_name_to_buf_name,
                buf_name if parent_buf_name is None else parent_buf_name,
            )
            continue
        else:
            # pyrefly: ignore [bad-argument-type, unsupported-operation]
            if not (len(children_nodes) == 1 and children_nodes[0] == node):
                raise AssertionError(
                    "expected a single child node equal to the current node"
                )

        ir_node = node.node
        if ir_node is None or ir_node.origins is None:
            continue
        for origin in ir_node.origins:
            node_name = origin.name
            # when buf1 and buf2 both have origin=node1
            # we draw node1 according to buf1
            if node_name not in node_name_to_buf_name:
                node_name_to_buf_name[node_name] = (
                    buf_name if parent_buf_name is None else parent_buf_name
                )


def get_node_name_to_buf_meta(
    node_name_to_buf_name: dict[str, str],
) -> dict[str, BufMeta]:
    buf_name_to_n_node = {}
    for node_name, buf_name in node_name_to_buf_name.items():
        if buf_name not in buf_name_to_n_node:
            buf_name_to_n_node[buf_name] = OrderedSet([node_name])
        else:
            # pyrefly: ignore [missing-attribute]
            buf_name_to_n_node[buf_name].add(node_name)

    node_name_to_buf_meta = {}
    for node_name, buf_name in node_name_to_buf_name.items():
        n_node = len(buf_name_to_n_node[buf_name])
        node_name_to_buf_meta[node_name] = BufMeta(buf_name, n_node)
    return node_name_to_buf_meta


def annotate_orig_fx_with_snodes(
    gm: torch.fx.GraphModule,
    snodes: SchedulerNodeList,
) -> None:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """
    node_name_to_buf_name: dict[str, str] = {}
    update_orig_fx_node_name_to_buf_name(snodes, node_name_to_buf_name)
    if node_name_to_buf_name is None:
        return
    node_name_to_buf_meta = get_node_name_to_buf_meta(node_name_to_buf_name)
    for node in gm.graph.nodes:
        if node.name in node_name_to_buf_meta:
            node.meta["buf_meta"] = node_name_to_buf_meta.get(node.name)


@contextlib.contextmanager
def enable_aot_logging() -> Iterator[None]:
    compile_debug = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    import torch._functorch.aot_autograd

    log = logging.getLogger(torch._functorch.aot_autograd.__name__)

    stack = contextlib.ExitStack()
    if not compile_debug:
        try:
            yield
        finally:
            stack.close()
        return

    # Enable all graphs to be logged to a file by setting the flags to True
    # and the log level of the file logger to DEBUG
    stack.enter_context(patch("functorch.compile.config.debug_partitioner", True))

    path = os.path.join(get_debug_dir(), "torchinductor")
    os.makedirs(path, exist_ok=True)

    fh = logging.FileHandler(
        os.path.join(
            path,
            f"aot_{get_aot_graph_name()}_debug.log",
        )
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
    )
    log.addHandler(fh)
    try:
        yield
    finally:
        log.removeHandler(fh)
        stack.close()


# Used for provenance tracking
# They are not stored in DebugContext because they are not set in
# _inductor_triton_kernel_to_post_grad_node_info's Debug Context
_inductor_post_to_pre_grad_nodes: dict[str, dict[str, list[str]]] = {}
_inductor_triton_kernel_to_post_grad_node_info: dict[str, list[str]] = {}
_pre_grad_graph_id: int | None = None
_inductor_pre_grad_node_stack_trace: dict[str, str] = {}
_inductor_kernel_stack_trace: dict[str, list[str]] = {}
_kernel_information_jsons: dict[str, dict[str, Any]] = {}
_inductor_kernel_provenance_debug_handle: int = 0


@dataclasses.dataclass
class _KernelExternInfo:
    is_extern: bool = False
    semantic_key: str | None = None
    shapes: dict[str, Any] | None = None


_inductor_kernel_extern_info: dict[str, _KernelExternInfo] = {}


def get_kernel_information_jsons() -> dict[str, dict[str, Any]]:
    return _kernel_information_jsons


def alias_kernel_provenance(original_kernel_name: str, alias_kernel_name: str) -> None:
    """Expose existing kernel provenance under an additional generated name."""
    if original_kernel_name in _inductor_kernel_stack_trace:
        _inductor_kernel_stack_trace[alias_kernel_name] = _inductor_kernel_stack_trace[
            original_kernel_name
        ]
    if original_kernel_name in _inductor_triton_kernel_to_post_grad_node_info:
        _inductor_triton_kernel_to_post_grad_node_info[alias_kernel_name] = (
            _inductor_triton_kernel_to_post_grad_node_info[original_kernel_name]
        )


def reset_inductor_kernel_provenance_debug_handle() -> None:
    global _inductor_kernel_provenance_debug_handle
    _inductor_kernel_provenance_debug_handle = 0


@contextlib.contextmanager
def reset_provenance_globals() -> Iterator[None]:
    """Context manager that resets provenance tracking globals upon entering
    and restores their original values when exiting."""
    global _pre_grad_graph_id
    global _inductor_post_to_pre_grad_nodes
    global _inductor_triton_kernel_to_post_grad_node_info
    global _inductor_pre_grad_node_stack_trace
    global _inductor_kernel_stack_trace
    global _inductor_kernel_provenance_debug_handle
    global _inductor_kernel_extern_info

    # Store original values
    original_pre_grad_graph_id = _pre_grad_graph_id
    original_post_to_pre_grad_nodes = _inductor_post_to_pre_grad_nodes.copy()
    original_triton_kernel_to_post_grad_node_info = (
        _inductor_triton_kernel_to_post_grad_node_info.copy()
    )
    original_inductor_pre_grad_node_stack_trace = (
        _inductor_pre_grad_node_stack_trace.copy()
    )
    original_inductor_kernel_stack_trace = _inductor_kernel_stack_trace.copy()
    original_inductor_kernel_provenance_debug_handle = (
        _inductor_kernel_provenance_debug_handle
    )
    original_inductor_kernel_extern_info = dict(_inductor_kernel_extern_info)

    # Reset to default values
    _pre_grad_graph_id = -1
    _inductor_post_to_pre_grad_nodes = {}
    _inductor_triton_kernel_to_post_grad_node_info = {}
    _inductor_pre_grad_node_stack_trace = {}
    _inductor_kernel_stack_trace = {}
    # Kernel timeline information is populated from compile_fx_inner while this
    # context is active.  It must outlive the reset scope so the profiler can
    # consume it during export_chrome_trace, which then clears it.
    _inductor_kernel_provenance_debug_handle = 0
    _inductor_kernel_extern_info = {}

    try:
        yield
    finally:
        # Restore original values
        _pre_grad_graph_id = original_pre_grad_graph_id
        _inductor_post_to_pre_grad_nodes = original_post_to_pre_grad_nodes
        _inductor_triton_kernel_to_post_grad_node_info = (
            original_triton_kernel_to_post_grad_node_info
        )
        _inductor_kernel_stack_trace = original_inductor_kernel_stack_trace
        _inductor_pre_grad_node_stack_trace = (
            original_inductor_pre_grad_node_stack_trace
        )
        _inductor_kernel_provenance_debug_handle = (
            original_inductor_kernel_provenance_debug_handle
        )
        _inductor_kernel_extern_info = original_inductor_kernel_extern_info


class DebugContext:
    _counter = itertools.count()

    @staticmethod
    def create_debug_dir(folder_name: str) -> str | None:
        debug_dir = config.trace.debug_dir or get_debug_dir()
        for n in DebugContext._counter:
            dirname = os.path.join(
                debug_dir,
                "torchinductor",
                f"{folder_name}.{n}",
            )
            try:
                os.makedirs(dirname, exist_ok=False)
                return dirname
            except FileExistsError:
                # Another process (e.g. a peer rank with the same debug_dir)
                # created this counter slot first; advance to the next n.
                continue
        return None

    def __init__(self) -> None:
        self._prof = None
        self._path = None
        self._stack = contextlib.ExitStack()

    def copy(self, new_path: str) -> None:
        if not self._path:
            return
        if not new_path.endswith(".debug"):
            raise AssertionError(new_path)
        from filelock import FileLock

        try:
            with FileLock(f"{new_path}.lock"):
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                shutil.copytree(self._path, new_path)
        except OSError:
            log.warning(
                "Failed to copy debug files from %s to %s", self._path, new_path
            )

    def fopen(
        self,
        filename: str,
        write_mode: str = "w",
        *args: Any,
        **kwargs: Any,
    ) -> IO[Any]:
        if not self._path:
            raise AssertionError("expected self._path to be set")
        return open(os.path.join(self._path, filename), write_mode, *args, **kwargs)

    @contextlib.contextmanager
    def fopen_context(
        self,
        filename: str,
        write_mode: str = "w",
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[IO[Any]]:
        if not self._path:
            raise AssertionError("expected self._path to be set")
        with open(os.path.join(self._path, filename), write_mode, *args, **kwargs) as f:
            yield f

    def filename(self, suffix: str) -> str:
        if not self._path:
            raise AssertionError("expected self._path to be set")
        return os.path.join(self._path, suffix)

    def upload_tar(self) -> None:
        if config.trace.upload_tar is not None:
            import tarfile

            if not self._path:
                raise AssertionError("expected self._path to be set")
            tar_file = os.path.join(
                self._path, f"{os.path.basename(self._path)}.tar.gz"
            )
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(self._path, arcname=os.path.basename(self._path))
            config.trace.upload_tar(tar_file)

    def __enter__(self) -> None:
        if config.debug:
            log = logging.getLogger("torch._dynamo")
            prev_level = log.level
            log.setLevel(logging.DEBUG)

            def reset_log_level(level: Any) -> None:
                log.setLevel(level)

            self._stack.callback(reset_log_level, prev_level)

        self._stack.enter_context(V.set_debug_handler(self))

        if not config.trace.enabled:
            return

        self._path = self.create_debug_dir(get_aot_graph_name())  # type: ignore[assignment]

        if config.trace.debug_log:
            self._setup_log_capture("debug.log", logging.DEBUG)
        if config.trace.info_log:
            self._setup_log_capture("info.log", logging.INFO)

    def _setup_log_capture(
        self,
        filename: str,
        level: int,
    ) -> None:
        log = logging.getLogger("torch._inductor")
        fd = self._stack.enter_context(self.fopen(filename))
        ch = logging.StreamHandler(fd)
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
        )
        log.addHandler(ch)
        log.setLevel(min(log.level, level))
        self._stack.callback(log.removeHandler, ch)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        if self._prof:
            self._prof.disable()
            self._save_profile_data()

        if self._path:
            self.upload_tar()
            log.warning("%s debug trace: %s", get_graph_being_compiled(), self._path)
        self._stack.close()

    def _save_profile_data(self) -> None:
        if not self._prof:
            raise AssertionError("expected self._prof to be set")
        self._prof.dump_stats(self.filename("compile.prof"))
        with self.fopen("compile.stats") as fd:
            stats = pstats.Stats(self._prof, stream=fd)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.print_stats(100)
            stats.sort_stats("tottime")
            stats.print_stats(100)

    def __getattr__(self, name: str) -> Callable[..., None] | None:
        if config.trace.enabled and getattr(config.trace, name):
            try:
                return getattr(DebugFormatter(self), name)
            except Exception:
                log.warning("Ignoring exception in debug code", exc_info=True)
                return None
        else:

            def ignored(*args: Any, **kwargs: Any) -> None:
                pass

            return ignored


class DebugFormatter:
    def __init__(self, handler: DebugContext) -> None:
        self.fopen = handler.fopen
        self.fopen_context = handler.fopen_context
        self.filename = handler.filename
        self.handler = handler

    def fx_graph(
        self,
        gm: torch.fx.GraphModule,
        inputs: list[torch.Tensor],
    ) -> None:
        with self.fopen("fx_graph_runnable.py") as fd:
            save_dir = None
            if torch._inductor.config.trace.save_real_tensors:
                inputs = torch._subclasses.fake_utils.try_convert_fake_to_real(inputs)
                save_dir = os.path.dirname(fd.name)

            # don't try to use stable hash torchinductor compilation if saving real tensors
            # and avoid recursively trying to save real tensors inside of the inductor compilation
            # regardless
            stable_hash = torch._inductor.config.trace.save_real_tensors
            with torch._inductor.config.patch(
                {"trace.enabled": False, "trace.save_real_tensors": False}
            ):
                save_graph_repro(
                    fd,
                    gm,
                    inputs,
                    "inductor",
                    save_dir=save_dir,
                    stable_hash=stable_hash,
                )

        with self.fopen("fx_graph_readable.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def fx_graph_transformed(
        self,
        gm: torch.fx.GraphModule,
        inputs: list[torch.Tensor],
    ) -> None:
        with self.fopen("fx_graph_transformed.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def ir_pre_fusion(self, nodes: SchedulerNodeList) -> None:
        with self.fopen("ir_pre_fusion.txt") as fd:
            fd.write(self._write_ir(nodes))

    def ir_post_fusion(self, nodes: SchedulerNodeList) -> None:
        with self.fopen("ir_post_fusion.txt") as fd:
            fd.write(self._write_ir(nodes))

    @staticmethod
    def _write_ir(nodes: SchedulerNodeList) -> str:
        buf = io.StringIO()
        for node in nodes:
            buf.write(node.debug_str())
            buf.write("\n\n\n")
        return buf.getvalue()

    def graph_diagram(self, nodes: SchedulerNodeList) -> None:
        draw_buffers(
            nodes,
            fname=self.filename(f"graph_diagram.{config.trace.graph_diagram_format}"),
        )

    def draw_orig_fx_graph(
        self,
        gm: torch.fx.GraphModule,
        nodes: SchedulerNodeList,
    ) -> None:
        annotate_orig_fx_with_snodes(gm, nodes)
        draw_graph(
            gm,
            fname=self.filename(
                f"orig_fx_graph_diagram.{config.trace.orig_fx_graph_diagram_format}"
            ),
            clear_meta=False,
            prog=GRAPHVIZ_COMMAND_SCALABLE,
            parse_stack_trace=True,
            dot_graph_shape=config.trace.dot_graph_shape,
        )

    def output_code(self, filename: str, extension: str = "py") -> None:
        shutil.copy(filename, self.filename(f"output_code.{extension}"))

    def log_autotuning_results(
        self,
        name: str,
        input_nodes: list[ir.IRNode],
        timings: dict["ChoiceCaller", float],  # type: ignore[name-defined] # noqa: F821
        elapse: float,
        precompile_elapse: float,
        prescreening_elapse: float | None,
    ) -> None:
        from .ir import FixedLayout

        def build_node_info(node: ir.IRNode) -> dict[str, str]:
            if hasattr(node, "name"):
                node_name = node.name
            else:
                node_name = ""
            node_info = {
                "name": node_name,
                "type": type(node).__name__,
            }
            try:
                layout = node.get_output_spec()
                if isinstance(layout, FixedLayout):
                    static_layout = FixedLayout(
                        layout.device,
                        dtype=layout.dtype,
                        size=V.graph.sizevars.optimization_hints(layout.size),
                        stride=V.graph.sizevars.optimization_hints(layout.stride),
                        offset=V.graph.sizevars.optimization_hint(
                            layout.offset, fallback=0
                        ),
                    )
                    node_info["layout"] = str(static_layout)
                else:
                    node_info["layout"] = str(layout)
            except Exception:
                pass
            try:
                node_info["dtype"] = str(node.get_dtype())
            except Exception:
                pass
            try:
                node_info["device"] = str(node.get_device())
            except Exception:
                pass
            try:
                node_info["stride"] = str(
                    V.graph.sizevars.optimization_hints(node.get_stride())
                )
            except Exception:
                pass
            try:
                node_info["size"] = str(
                    V.graph.sizevars.optimization_hints(node.get_size())
                )  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                node_info["numel"] = str(
                    V.graph.sizevars.optimization_hint(node.get_numel())
                )
            except Exception:
                pass
            if hasattr(node, "data") and isinstance(node.data, ir.IRNode):
                node_info["data"] = build_node_info(node.data)
            return node_info

        general_properties = {
            "op_name": name,
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_device_count": torch.cuda.device_count(),
            "input_nodes": [build_node_info(node) for node in input_nodes],
            "autotuning_time": elapse,
            "precompile_time": precompile_elapse,
            "prescreening_time": prescreening_elapse,
        }
        with self.fopen_context(
            "autotuning_result_json_list.txt", "at", encoding="utf-8"
        ) as fd:
            for caller, time in timings.items():
                info_dict = dict(caller.info_dict())
                info_dict.update(general_properties)
                info_dict["benchmark_result"] = time
                json.dump(info_dict, fd)
                fd.write("\n")


def log_ir_pre_fusion(nodes: SchedulerNodeList) -> None:
    if ir_pre_fusion_log.isEnabledFor(logging.INFO):
        ir_pre_fusion_log.info("BEFORE FUSION\n%s", DebugFormatter._write_ir(nodes))

    V.debug.ir_pre_fusion(nodes)


def log_ir_post_fusion(nodes: SchedulerNodeList) -> None:
    if ir_post_fusion_log.isEnabledFor(logging.INFO):
        ir_post_fusion_log.info("AFTER FUSION\n%s", DebugFormatter._write_ir(nodes))

    V.debug.ir_post_fusion(nodes)


def _dump_collective_schedule(schedule: list[str | None]) -> None:
    try:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_collective_schedule",
                "encoding": "json",
            },
            payload_fn=lambda: schedule,
        )
    except Exception:
        log.debug(
            "Failed to log inductor_collective_schedule via structured logging",
            exc_info=True,
        )


def log_collective_schedule(nodes: Sequence[BaseSchedulerNode]) -> None:
    schedule = [
        getattr(op, "python_kernel_name", None)
        for node in nodes
        if isinstance(op := getattr(node, "node", None), ir._CollectiveKernel)
    ]

    # Only log when there is at least one collective op
    if schedule:
        _dump_collective_schedule(schedule)


def log_runtime_and_tensor_meta(node_runtimes: Sequence[tuple[Any, float]]) -> None:
    """Log per-op runtime estimates and output tensor metadata for TLParse."""

    try:
        to_optimization_hints = V.graph.sizevars.optimization_hints

        def to_list(x: Sequence[Any] | None) -> list[Any]:
            return list(to_optimization_hints(x)) if x is not None else []

        def dtype_to_str(dtype: Any) -> str | None:
            if dtype is None:
                return None
            s = str(dtype)
            s = s.removeprefix("torch.")
            return s

        ops: list[dict[str, Any]] = []
        for s, runtime_ns in node_runtimes:
            name = getattr(s.node, "python_kernel_name", s.get_name())
            op_type = "collective" if utils.is_collective(s.node) else "compute"

            # Build outputs metadata if available
            outputs: list[dict[str, Any]] = []
            try:
                for buf in s.get_outputs():
                    irnode = buf.node
                    shape = irnode.maybe_get_size()
                    stride = (
                        irnode.get_stride()
                        if isinstance(irnode.layout, ir.Layout)
                        else None
                    )
                    dtype = irnode.maybe_get_dtype()
                    outputs.append(
                        {
                            "shape": to_list(shape),
                            "stride": to_list(stride),
                            "dtype": dtype_to_str(dtype),
                        }
                    )
            except Exception:
                pass

            ops.append(
                {
                    "name": name,
                    "type": op_type,
                    "estimated_runtime_ns": runtime_ns,
                    "outputs": outputs,
                }
            )

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "inductor_runtime_and_tensor_meta",
                "encoding": "json",
            },
            payload_fn=lambda: {"ops": ops},
        )
    except Exception:
        log.debug("Failed to log inductor_runtime_and_tensor_meta", exc_info=True)


def log_graph_execution() -> None:
    """Emit a structured artifact with the graph execution order."""
    if not GRAPH_EXECUTION_ORDER:
        return
    try:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "graph_execution",
                "encoding": "json",
            },
            payload_fn=lambda: {"graph_execution_order": GRAPH_EXECUTION_ORDER},
        )
    except Exception:
        log.debug("Failed to log graph_execution", exc_info=True)


@contextlib.contextmanager
def record_and_log_graph_execution_order() -> Iterator[None]:
    """Record graph execution order and log it once on exit."""
    global RECORD_GRAPH_EXECUTION, GRAPH_EXECUTION_ORDER, GRAPH_COMPILE_IDS
    GRAPH_EXECUTION_ORDER = []
    GRAPH_COMPILE_IDS = {}
    RECORD_GRAPH_EXECUTION = True
    try:
        yield
    finally:
        log_graph_execution()
        RECORD_GRAPH_EXECUTION = False
        GRAPH_EXECUTION_ORDER = None
        GRAPH_COMPILE_IDS = None


@dataclasses.dataclass
class TensorMetadataHolder:
    tensor_metadata: TensorMetadata
    device: torch.device


@dataclasses.dataclass
class SymExprMetadataHolder:
    pytype: str
    expr: Any
    hint: int | float | bool | None
    symbol_hints: dict[str, int | float | bool]


save_args_cnt = itertools.count()


def create_mapping_pre_post_grad_nodes(
    pre_grad_graph_id: int | None,
    post_to_pre_grad_nodes_json: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    """
    Create bidirectional mappings between pre_grad graph nodes
    and post_grad graph code nodes, and vice versa.
    """
    # return a dummy dict if there's any error
    empty_return: dict[str, dict[str, list[str]]] = {
        "preToPost": {},
        "postToPre": {},
    }

    if not isinstance(post_to_pre_grad_nodes_json, dict):
        log.error(
            "Provenance tracking error: post_to_pre_grad_nodes_json is not a dict"
        )
        return empty_return

    if not isinstance(pre_grad_graph_id, int):
        # pre_grad_graph_id may be empty if there's no pre_grad graph
        # and there's only a backward graph from backward pass engine
        return empty_return

    pre_to_post: dict[str, Any] = collections.defaultdict(OrderedSet)
    post_to_pre: dict[str, Any] = collections.defaultdict(OrderedSet)

    try:

        def check_format(node: dict[str, Any]) -> bool:
            if not isinstance(node, dict):
                log.error(
                    "Provenance tracking error: node provenance in post_to_pre_grad_nodes_json is not a dict"
                )
                return False
            if "graph_id" not in node or "name" not in node or "from_node" not in node:
                log.error(
                    "Provenance tracking error: node provenance in post_to_pre_grad_nodes_json has wrong format"
                )
                return False
            return True

        for outer_key, node_array in post_to_pre_grad_nodes_json.items():
            if not isinstance(node_array, list):
                log.error(
                    "Provenance tracking error: post_to_pre_grad_nodes_json value is not a list"
                )
                return empty_return
            for node in node_array:
                if not check_format(node):
                    return empty_return
                # Check the current node first
                if node.get("graph_id") == pre_grad_graph_id:
                    pre_to_post[node["name"]].add(outer_key)
                    post_to_pre[outer_key].add(node["name"])

                # Check nested from_node array recursively, add node with the right graph_id to the map
                stack = [(n, outer_key) for n in node.get("from_node", [])]
                while stack:
                    current_node, parent_key = stack.pop()
                    if not check_format(current_node):
                        return empty_return
                    if current_node.get("graph_id") == pre_grad_graph_id:
                        pre_to_post[current_node["name"]].add(parent_key)
                        post_to_pre[parent_key].add(current_node["name"])
                    stack.extend(
                        (n, parent_key) for n in current_node.get("from_node", [])
                    )

        def convert_sets_to_lists(d: dict[str, Any]) -> None:
            for key in d:
                d[key] = list(d[key])
            d = dict(d)

        # convert to list because set is not JSON serializable
        convert_sets_to_lists(pre_to_post)
        convert_sets_to_lists(post_to_pre)
        return {
            "preToPost": pre_to_post,
            "postToPre": post_to_pre,
        }
    except Exception as e:
        # Since this is just logging code, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "create_mapping_pre_post_grad_nodes",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        log.error("post_to_pre_grad_nodes_json:  %s", post_to_pre_grad_nodes_json)
        log.error("pre_grad_graph_id:  %s", pre_grad_graph_id)
        return empty_return


def create_node_mapping_kernel_to_post_grad(
    triton_kernel_to_post_grad_json: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Create bidirectional mappings between triton kernel name and post_grad
    graph code nodes, and vice versa.
    """

    # return a dummy dict if there's any error
    empty_return: dict[str, dict[str, Any]] = {
        "cppCodeToPost": {},
        "postToCppCode": {},
    }

    if not isinstance(triton_kernel_to_post_grad_json, dict):
        log.error(
            "Provenance tracking error: triton_kernel_to_post_grad_json is not a dict"
        )
        return empty_return

    post_to_cpp_code: dict[str, Any] = collections.defaultdict(OrderedSet)

    try:
        for outer_key, node_array in triton_kernel_to_post_grad_json.items():
            if not isinstance(node_array, list):
                log.error(
                    "Provenance tracking error: triton_kernel_to_post_grad_json value is not a list"
                )
                return empty_return
            for curr_node in node_array:
                post_to_cpp_code[curr_node].add(outer_key)

        def convert_sets_to_lists(d: dict[str, Any]) -> None:
            for key in d:
                d[key] = list(d[key])
            d = dict(d)

        # convert to list because set is not JSON serializable
        convert_sets_to_lists(post_to_cpp_code)
        return {
            "cppCodeToPost": triton_kernel_to_post_grad_json,
            "postToCppCode": post_to_cpp_code,
        }
    except Exception as e:
        # Since this is just logging code, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "create_mapping_kernel_to_post_grad",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        log.error(
            "triton_kernel_to_post_grad_json:  %s", triton_kernel_to_post_grad_json
        )
        return empty_return


def dump_inductor_provenance_info() -> dict[str, Any]:
    try:
        global _pre_grad_graph_id
        global _inductor_post_to_pre_grad_nodes
        global _inductor_triton_kernel_to_post_grad_node_info
        node_mapping: dict[str, Any] = {}
        if _pre_grad_graph_id:
            node_mapping_kernel = create_node_mapping_kernel_to_post_grad(
                _inductor_triton_kernel_to_post_grad_node_info
            )
            node_mapping = {
                **_inductor_post_to_pre_grad_nodes,
                **node_mapping_kernel,
            }
            if config.trace.enabled:
                with V.debug.fopen(
                    "inductor_provenance_tracking_node_mappings.json", "w"
                ) as fd:
                    json.dump(node_mapping, fd)
        # we need to update the node mapping version when node mapping format changes
        # so the tlparse tool knows which node mapping version it is looking at
        node_mapping["version"] = 2.0
        return node_mapping
    except Exception as e:
        # Since this is just debugging, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "dump_inductor_provenance_info",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        return {}


def _serialize_provenance_shape(size: Sequence[Any]) -> list[int | str]:
    shape: list[int | str] = []
    for dim in size:
        if ir._is_static(dim):
            shape.append(int(dim))
        else:
            shape.append(str(dim))
    return shape


def _get_tensor_metadata(node: ir.IRNode) -> tuple[list[int | str], str] | None:
    if not node.has_tensor_output():
        return None

    size = node.maybe_get_size()
    dtype = node.maybe_get_dtype()
    if size is None or dtype is None:
        return None

    return _serialize_provenance_shape(size), str(dtype)


def create_kernel_information_json() -> dict[str, dict[str, Any]]:
    """Create kernel information JSON with extended metadata for cross-hardware comparison."""
    try:
        global _inductor_post_to_pre_grad_nodes
        global _inductor_kernel_stack_trace
        global _inductor_triton_kernel_to_post_grad_node_info
        global _inductor_kernel_extern_info

        post_to_pre = _inductor_post_to_pre_grad_nodes.get("postToPre", {})
        all_kernels = OrderedSet(_inductor_kernel_stack_trace.keys()) | OrderedSet(
            _inductor_triton_kernel_to_post_grad_node_info.keys()
        )

        result = {}
        for kernel_key in all_kernels:
            post_grad_nodes = _inductor_triton_kernel_to_post_grad_node_info.get(
                kernel_key, []
            )

            pre_grad_nodes: OrderedSet[str] = OrderedSet()
            for post_node in post_grad_nodes:
                pre_grad_nodes.update(post_to_pre.get(post_node, []))

            stack_traces = _inductor_kernel_stack_trace.get(kernel_key, [])
            pre_grad_list = list(pre_grad_nodes)

