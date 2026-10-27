# Note: import order is significant here.

# submodules
from .runtime import (
    autotune,
    Config,
    heuristics,
    JITFunction,
    KernelInterface,
    reinterpret,
    TensorWrapper,
    OutOfResources,
    InterpreterError,
    MockTensor,
)
from .runtime.jit import constexpr_function, jit
from .runtime._async_compile import AsyncCompileMode, FutureKernel
from .compiler import compile, CompilationError
from .errors import TritonError
from .runtime._allocation import set_allocator

from . import language
from . import testing
from . import tools

must_use_result = language.core.must_use_result
aggregate = language.core._aggregate

__all__ = [
    "AsyncCompileMode",
    "aggregate",
    "autotune",
    "cdiv",
    "CompilationError",
    "compile",
    "Config",
    "constexpr_function",
    "FutureKernel",
    "heuristics",
    "InterpreterError",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "MockTensor",
    "must_use_result",
    "next_power_of_2",
    "OutOfResources",
    "reinterpret",
    "runtime",
    "set_allocator",
    "TensorWrapper",
    "TritonError",
    "testing",
    "tools",
]

# -------------------------------------
# misc. utilities that  don't fit well
# into any specific module
# -------------------------------------


@constexpr_function
def cdiv(x: int, y: int):
    return (x + y - 1) // y


@constexpr_function
def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n
<|endoftext|>from __future__ import annotations

from functools import reduce
from typing import Any, Callable, TYPE_CHECKING, Union, List, Dict

if TYPE_CHECKING:
    from .language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]

TRITON_MAX_TENSOR_NUMEL = 1048576


def get_iterable_path(iterable: IterableType, path: ObjPath) -> Any:
    return reduce(lambda a, idx: a[idx], path, iterable)  # type: ignore[index]


def set_iterable_path(iterable: IterableType, path: tuple[int, ...], val: Any):
    from .language import core
    assert len(path) != 0
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    assert isinstance(prev, core.tuple)
    prev._setitem(path[-1], val)


def is_iterable(x):
    from .language import core
    return isinstance(x, (list, tuple, core.tuple, core.tuple_type))


def apply_with_path(value: Any, fn: Callable[[ObjPath, Any], None], _path=None) -> None:
    if _path is None:
        _path = ()

    if is_iterable(value):
        for idx, item in enumerate(value):
            apply_with_path(item, fn, _path=(*_path, idx))
    else:
        fn(_path, value)


def find_paths_if(iterable: Union[IterableType, Any], pred: Callable[[ObjPath, Any], bool]) -> list[ObjPath]:
    # We need to use dict so that ordering is maintained, while set doesn't guarantee order
    ret: dict[ObjPath, None] = {}

    def _impl(path: tuple[int, ...], current: Any):
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl((*path, idx), item)
        elif pred(path, current):
            ret[path] = None

    _impl((), iterable)

    return list(ret.keys())


def is_power_of_two(x):
    # x & (x - 1) is also 0 for x == 0, but 0 is not a power of two, so guard it.
    return x > 0 and (x & (x - 1)) == 0


def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        if not is_power_of_two(d):
            raise ValueError(f"Shape element {i} must be a power of 2")
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel


type_canonicalisation_dict = {
    # we canonicalise all bools to be unsigned:
    "bool": "u1",
    "int1": "u1",
    "uint1": "u1",
    "i1": "u1",
    # floating-point dtypes:
    "float8e4nv": "fp8e4nv",
    "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    # signed integers:
    "int8": "i8",
    "int16": "i16",
    "int": "i32",
    "int32": "i32",
    "int64": "i64",
    # unsigned integers:
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "void": "void",
}

for v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[v] = v


def canonicalize_dtype(dtype):
    dtype_str = str(dtype).split(".")[-1]
    return type_canonicalisation_dict[dtype_str]


def canonicalize_ptr_dtype(dtype, is_const):
    return f"{'*k' if is_const else '*'}{canonicalize_dtype(dtype)}"


BITWIDTH_DICT: Dict[str, int] = {
    **{f"u{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"i{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"fp{n}": n
       for n in (16, 32, 64)},
    **{f"fp8{suffix}": 8
       for suffix in ("e4nv", "e4b15", "e4b8", "e5", "e5b16")},
    "bf16": 16,
    "void": 0,
}

for k, v in type_canonicalisation_dict.items():
    BITWIDTH_DICT[k] = BITWIDTH_DICT[v]


def get_primitive_bitwidth(dtype: str) -> int:
    return BITWIDTH_DICT[dtype]


def is_namedtuple(val):
    return isinstance(val, type) and issubclass(val, tuple) and hasattr(val, "_fields")


def _tuple_create(arg, contents):
    # NamedTuples and tuples have different construction semantics. NamedTuple
    # has a constructor that takes individual arguments, while tuple takes an
    # iterable. Both have type "tuple" making it difficult to distinguish
    # between them, but only NamedTuple has "_fields" and apparently this is how
    # everyone does the check.
    return type(arg)(*contents) if hasattr(arg, "_fields") else type(arg)(contents)
<|endoftext|>from ..runtime.jit import jit
from . import core as tl
from . import math

N_ROUNDS_DEFAULT = tl.constexpr(10)  # Default number of rounds for philox

# -------------------
# randint
# -------------------


@jit
def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Run `n_rounds` rounds of Philox for state (c0, c1, c2, c3) and key (k0, k1).
    """
    if c0.dtype == tl.uint32:
        PHILOX_KEY_A: tl.constexpr = 0x9E3779B9
        PHILOX_KEY_B: tl.constexpr = 0xBB67AE85
        PHILOX_ROUND_A: tl.constexpr = 0xD2511F53
        PHILOX_ROUND_B: tl.constexpr = 0xCD9E8D57
    else:
        tl.static_assert(c0.dtype == tl.uint64, "dtype not supported in philox_impl")
        PHILOX_KEY_A: tl.constexpr = 0x9E3779B97F4A7C15
        PHILOX_KEY_B: tl.constexpr = 0xBB67AE8584CAA73B
        PHILOX_ROUND_A: tl.constexpr = 0xD2E7470EE14C6C93
        PHILOX_ROUND_B: tl.constexpr = 0xCA5A826395121157

    for _ in tl.static_range(n_rounds):
        # for _ in range(n_rounds):
        # update random state
        A = PHILOX_ROUND_A
        B = PHILOX_ROUND_B
        _c0, _c2 = c0, c2
        c0 = math.umulhi(B, _c2) ^ c1 ^ k0
        c2 = math.umulhi(A, _c0) ^ c3 ^ k1
        c1 = tl.mul(B, _c2, sanitize_overflow=False)
        c3 = tl.mul(A, _c0, sanitize_overflow=False)
        # raise key
        k0 = tl.add(k0, PHILOX_KEY_A, sanitize_overflow=False)
        k1 = tl.add(k1, PHILOX_KEY_B, sanitize_overflow=False)
    return c0, c1, c2, c3


@jit
def philox(seed, c0, c1, c2, c3, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    seed = tl.to_tensor(seed)
    tl.static_assert(seed.dtype.is_int())
    seed = seed.to(tl.uint64)
    c0 = tl.to_tensor(c0)
    c1 = tl.to_tensor(c1)
    c2 = tl.to_tensor(c2)
    c3 = tl.to_tensor(c3)

    if tl.constexpr(c0.dtype.primitive_bitwidth) == 32:
        int_dtype = tl.uint32
        seed_hi = ((seed >> 32) & 0xffffffff).to(tl.uint32)
        seed_lo = (seed & 0xffffffff).to(tl.uint32)
    else:
        tl.static_assert(tl.constexpr(c0.dtype.primitive_bitwidth) == 64, "bitwidth not supported in philox")
        int_dtype = tl.uint64
        seed_hi = tl.full((1, ), 0, dtype=int_dtype)
        seed_lo = seed

    c0 = c0.to(int_dtype, bitcast=True)
    c1 = c1.to(int_dtype, bitcast=True)
    c2 = c2.to(int_dtype, bitcast=True)
    c3 = c3.to(int_dtype, bitcast=True)
    return philox_impl(c0, c1, c2, c3, seed_lo, seed_hi, n_rounds)


@jit
def randint(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, returns a single
    block of random :code:`int32`.

    If you need multiple streams of random numbers,
    using `randint4x` is likely to be faster than calling `randint` 4 times.

    :param seed: The seed for generating random numbers.
    :param offset: The offsets to generate random numbers for.
    """
    ret, _, _, _ = randint4x(seed, offset, n_rounds)
    return ret


@jit
def randint4x(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, returns four
    blocks of random :code:`int32`.

    This is the maximally efficient entry point
    to Triton's Philox pseudo-random number generator.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    # _0 = tl.zeros(offset.shape, offset.dtype)

    offset_lo = offset.to(tl.uint32)
    _0 = offset_lo * 0

    if tl.constexpr(offset.dtype.primitive_bitwidth) > 32:
        offset_hi = (offset >> 32).to(tl.uint32)
    else:
        offset_hi = _0

    return philox(seed, offset_lo, offset_hi, _0, _0, n_rounds)


# -------------------
# rand
# -------------------

# @jit
# def uint32_to_uniform_float(x):
#     """
#     Numerically stable function to convert a random uint32 into a random float uniformly sampled in [0, 1).
#     """
#     two_to_the_minus_32: tl.constexpr = 2.328306e-10
#     return x * two_to_the_minus_32


@jit
def uint_to_uniform_float(x):
    """
    Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
    """
    # TODO: fix frontend issues and cleanup
    # conditions can be simplified
    # scale is ((2**23 - 1) / 2**23) * 2**(N_BITS - 1)
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
        x = x.to(tl.int32, bitcast=True)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64))
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, ~x, x)
    return x * scale


@jit
def rand(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns a block of random :code:`float32` in :math:`U(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    source = randint(seed, offset, n_rounds)
    return uint_to_uniform_float(source)


@jit
def rand4x(seed, offsets, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offsets` block,
    returns 4 blocks of random :code:`float32` in :math:`U(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    i1, i2, i3, i4 = randint4x(seed, offsets, n_rounds)
    u1 = uint_to_uniform_float(i1)
    u2 = uint_to_uniform_float(i2)
    u3 = uint_to_uniform_float(i3)
    u4 = uint_to_uniform_float(i4)
    return u1, u2, u3, u4


# -------------------
# randn
# -------------------


@jit
def pair_uniform_to_normal(u1, u2):
    """Box-Muller transform"""
    u1 = tl.maximum(1.0e-7, u1)
    th = 6.283185307179586 * u2
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(th), r * math.sin(th)


@jit
def randn(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns a block of random :code:`float32` in :math:`\\mathcal{N}(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    i1, i2, _, _ = randint4x(seed, offset, n_rounds)
    u1 = uint_to_uniform_float(i1)
    u2 = uint_to_uniform_float(i2)
    n1, _ = pair_uniform_to_normal(u1, u2)
    return n1


@jit
def randn4x(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns 4 blocks of random :code:`float32` in :math:`\\mathcal{N}(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    u1, u2, u3, u4 = rand4x(seed, offset, n_rounds)
    n1, n2 = pair_uniform_to_normal(u1, u2)
    n3, n4 = pair_uniform_to_normal(u3, u4)
    return n1, n2, n3, n4
<|endoftext|>from __future__ import annotations  # remove after python 3.11
import builtins
import warnings

from typing import List, Optional, Sequence, Tuple, TypeVar, Generic, Type
import numbers

from triton.runtime import driver

from .._C.libtriton import ir
from . import core as tl

T = TypeVar('T')
TensorTy = TypeVar('TensorTy')


class IncompatibleTypeErrorImpl(Exception):

    def __init__(self, type_a, type_b):
        self.type_a = type_a
        self.type_b = type_b
        self.message = "invalid operands of type " + self.type_a.__repr__() + " and " + self.type_b.__repr__()
        super(IncompatibleTypeErrorImpl, self).__init__(self.message)


class TritonSemantic(Generic[TensorTy]):
    tensor: Type[TensorTy] = tl.tensor
    lang = tl

    builder: ir.builder

    def __init__(self, builder):
        self.builder = builder

# ===----------------------------------------------------------------------===##
# Programming Model
# ===----------------------------------------------------------------------===##

    def program_id(self, axis: int) -> TensorTy:
        if axis not in (0, 1, 2):
            raise ValueError(f"program_id axis must be 0, 1, or 2 but got {axis}")
        return self.tensor(self.builder.create_get_program_id(axis), tl.int32)

    def num_programs(self, axis: int) -> TensorTy:
        if axis not in (0, 1, 2):
            raise ValueError(f"num_programs axis must be 0, 1, or 2 but got {axis}")
        return self.tensor(self.builder.create_get_num_programs(axis), tl.int32)

# ===----------------------------------------------------------------------===//
#                               Implicit Casting Utilities
# ===----------------------------------------------------------------------===//

    def integer_promote_impl(self, a_ty: tl.dtype, b_ty: tl.dtype) -> tl.dtype:
        a_rank = a_ty.int_bitwidth
        b_rank = b_ty.int_bitwidth
        a_sn = a_ty.int_signedness
        b_sn = b_ty.int_signedness
        # Rules for signedness taken from "Usual arithmetic conversions" on
        # https://en.cppreference.com/w/c/language/conversion.
        if a_sn == b_sn:
            return a_ty if a_rank > b_rank else b_ty
        elif a_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
            return a_ty if a_rank >= b_rank else b_ty
        elif b_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
            return b_ty if b_rank >= a_rank else a_ty
        raise TypeError(f"unexpected signedness {a_sn} and {b_sn}")

    def computation_type_impl(self, a_ty: tl.dtype, a_is_scalar: bool, b_ty: tl.dtype, b_is_scalar: bool,
                              div_or_mod: bool) -> tl.dtype:
        # 0) For scalars we follow semantics similar to PyTorch, namely:
        # - If the scalar is of a lower or equal kind (bool < uint < int < fp),
        #   it doesn't participate in the promotion
        if a_is_scalar != b_is_scalar:
            scalar_ty, tensor_ty = (a_ty, b_ty) if a_is_scalar else (b_ty, a_ty)
            if scalar_ty.kind().value <= tensor_ty.kind().value:
                # Upcast because of 3) and 4) below!
                if div_or_mod and (tensor_ty in (tl.float16, tl.bfloat16)):
                    return tl.float32
                return tensor_ty

        # 1) if one operand is double, the other is implicitly
        #    converted to double
        if a_ty.is_fp64() or b_ty.is_fp64():
            return tl.float64
        # 2) if one operand is float, the other is implicitly
        #    converted to float
        if a_ty.is_fp32() or b_ty.is_fp32():
            return tl.float32
        # 3 ) if one operand is half, the other is implicitly converted to half
        #     unless we're doing / or %, which do not exist natively in PTX for fp16.
        #     Supported PTX op: add, sub, mul, fma, neg, abs, min, max, tanh, ex2, setp
        if a_ty.is_fp16() or b_ty.is_fp16():
            if div_or_mod:
                return tl.float32
            else:
                return tl.float16
        # 4) return bf16 only if both operands are of bf16
        if a_ty.is_bf16() and b_ty.is_bf16():
            if div_or_mod:
                return tl.float32
            else:
                return tl.bfloat16
        if a_ty.is_bf16() or b_ty.is_bf16():
            return tl.float32
        # 5) return fp16 if operands are different fp8
        if a_ty.is_fp8() and b_ty.is_fp8():
            return a_ty if a_ty == b_ty else tl.float16
        if not a_ty.is_int() or not b_ty.is_int():
            raise TypeError(f"unexpected type {a_ty} and {b_ty}")
        # 6 ) both operands are integer and undergo
        #    integer promotion
        if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
            raise TypeError("Cannot use /, #, or % with " + a_ty.__repr__() + " and " + b_ty.__repr__() +
                            " because they have different signedness;"
                            "this is unlikely to result in a useful answer. Cast them to the same signedness.")
        return self.integer_promote_impl(a_ty, b_ty)

    def to_tensor(self, x, check_type=True):
        if isinstance(x, self.tensor):
            return x
        x = x.value if isinstance(x, tl.constexpr) else x
        if isinstance(x, (int, float, bool)):
            dtype = self.to_tensor_type(x)
            return self.scalar_constant(x, dtype=dtype)
        elif check_type:
            raise TypeError(f"cannot convert {x} of type {type(x)} to tensor")
        return x

    def to_tensor_type(self, x):
        if isinstance(x, tl.dtype):
            return x
        elif isinstance(x, tl.constexpr_type):
            x = x.value

        if isinstance(x, bool):
            return tl.int1
        elif isinstance(x, int):
            if -2**31 <= x < 2**31:
                return tl.int32
            elif 2**31 <= x < 2**32:
                return tl.uint32
            elif -2**63 <= x < 2**63:
                return tl.int64
            elif 2**63 <= x < 2**64:
                return tl.uint64
            raise ValueError(f'Nonrepresentable integer {x}.')
        elif isinstance(x, float):
            min_float32 = 2**-126
            max_float32 = (2 - 2**-23) * 2**127
            abs_x = builtins.abs(x)
            if abs_x == float("inf") or\
               abs_x == 0.0 or \
               x != x or \
               min_float32 <= abs_x <= max_float32:
                return tl.float32
            else:
                return tl.float64
        raise TypeError(f"cannot convert {x} of type {type(x)} to tensor")

# ===----------------------------------------------------------------------===//
#                               Binary Operators
# ===----------------------------------------------------------------------===//

    def check_ptr_type_impl(self, type_a: tl.dtype, type_b: tl.dtype, allow_ptr_a: bool) -> None:
        if type_a.is_ptr():
            if not allow_ptr_a:
                raise IncompatibleTypeErrorImpl(type_a, type_b)
            # T* + U* with T != U
            if type_b.is_ptr() and (type_a != type_b):
                raise IncompatibleTypeErrorImpl(type_a, type_b)
            # T* + float
            if type_b.is_floating():
                raise IncompatibleTypeErrorImpl(type_a, type_b)

    def binary_op_type_checking_impl(self, lhs: TensorTy | numbers.Number, rhs: TensorTy | numbers.Number,
                                     allow_lhs_ptr=False, allow_rhs_ptr=False, arithmetic_check=True,
                                     div_or_mod=False) -> Tuple[TensorTy, TensorTy]:
        lhs_is_scalar = isinstance(lhs, numbers.Number)
        rhs_is_scalar = isinstance(rhs, numbers.Number)
        if lhs_is_scalar:
            lhs_scalar = lhs
            lhs = self.to_tensor(lhs)
        if rhs_is_scalar:
            rhs_scalar = rhs
            rhs = self.to_tensor(rhs)

        # implicit typecasting
        lhs_sca_ty = lhs.type.scalar
        rhs_sca_ty = rhs.type.scalar
        self.check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
        self.check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
        if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
            ret_sca_ty = self.computation_type_impl(lhs_sca_ty, lhs_is_scalar, rhs_sca_ty, rhs_is_scalar, div_or_mod)
            if (lhs_is_scalar and lhs_scalar < 0 and ret_sca_ty.is_int_unsigned()
                    or rhs_is_scalar and rhs_scalar < 0 and ret_sca_ty.is_int_unsigned()):
                raise ValueError("Cannot perform a binary operation between an unsigned tensor and a negative scalar. "
                                 "Perform a explicit cast on one of them.")
            if ret_sca_ty.is_int():
                if lhs_is_scalar and not (ret_sca_ty.get_int_min_value() <= lhs_scalar <=
                                          ret_sca_ty.get_int_max_value()):
                    raise ValueError(f"Scalar {lhs_scalar} is out of range for type {ret_sca_ty}")
                if rhs_is_scalar and not (ret_sca_ty.get_int_min_value() <= rhs_scalar <=
                                          ret_sca_ty.get_int_max_value()):
                    raise ValueError(f"Scalar {rhs_scalar} is out of range for type {ret_sca_ty}")
            lhs = self.scalar_constant(lhs_scalar, dtype=ret_sca_ty) if lhs_is_scalar else self.cast(lhs, ret_sca_ty)
            rhs = self.scalar_constant(rhs_scalar, dtype=ret_sca_ty) if rhs_is_scalar else self.cast(rhs, ret_sca_ty)

        # implicit broadcasting
        lhs, rhs = self.broadcast_impl_value(lhs, rhs)
        return lhs, rhs

    def binary_op_sanitize_overflow_impl(self, lhs: TensorTy, rhs: TensorTy, binary_op: callable):
        if lhs.type.scalar.int_bitwidth >= 64 or not self.builder.options.sanitize_overflow:
            return
        lhs_sca_ty = lhs.type.scalar
        rhs_sca_ty = rhs.type.scalar
        assert lhs_sca_ty == rhs_sca_ty, f"expected matching operand types, got {lhs_sca_ty} and {rhs_sca_ty}"
        assert lhs_sca_ty.is_int(), f"expected integer type, got {lhs_sca_ty}"
        lhs = self.cast(lhs, tl.int64)
        rhs = self.cast(rhs, tl.int64)
        ret = binary_op(lhs, rhs, False)
        max_value = lhs_sca_ty.get_int_max_value()
        max_value = self.scalar_constant(max_value, tl.int64)
        min_value = lhs_sca_ty.get_int_min_value()
        min_value = self.scalar_constant(min_value, tl.int64)
        cond = self.and_(self.less_equal(ret, max_value), self.greater_equal(ret, min_value))
        msg = f"int{lhs_sca_ty.int_bitwidth} overflow detected for operation {binary_op.__name__}"
        self.device_assert(cond, msg, None)

    def add(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number,
            sanitize_overflow: bool) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, True, True)
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        if input_scalar_ty.is_ptr() and other_scalar_ty.is_ptr():
            raise TypeError("cannot add pointers together")

        # offset + ptr
        # ptr + offset
        if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
            input, other = other, input
            input_scalar_ty = input.type.scalar
            other_scalar_ty = other.type.scalar
        if input_scalar_ty.is_ptr():
            other_handle = other.handle
            if other.dtype.is_int_unsigned() and other.dtype.int_bitwidth < 64:
                # addptr treats offset as signed. Zero-extend unsigned offsets to ensure they're positive
                i64_ty = other.type.with_element_ty(tl.int64).to_ir(self.builder)
                other_handle = self.builder.create_int_cast(other.handle, i64_ty, False)
            return self.tensor(self.builder.create_addptr(input.handle, other_handle), input.type)
        # float + float
        elif input_scalar_ty.is_floating():
            return self.tensor(self.builder.create_fadd(input.handle, other.handle), input.type)
        # int + int
        elif input_scalar_ty.is_int():
            if sanitize_overflow:
                self.binary_op_sanitize_overflow_impl(input, other, self.add)
            return self.tensor(self.builder.create_add(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {input_scalar_ty}")

    def sub(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number,
            sanitize_overflow: bool) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, True, False)
        scalar_ty = input.type.scalar
        # ptr - offset
        if scalar_ty.is_ptr():
            return self.add(input, self.minus(other), sanitize_overflow=False)
        # float - float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fsub(input.handle, other.handle), input.type)
        # int - int
        elif scalar_ty.is_int():
            if sanitize_overflow:
                self.binary_op_sanitize_overflow_impl(input, other, self.sub)
            return self.tensor(self.builder.create_sub(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {scalar_ty}")

    def mul(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number,
            sanitize_overflow: bool) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float * float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fmul(input.handle, other.handle), input.type)
        # int * int
        elif scalar_ty.is_int():
            if sanitize_overflow:
                self.binary_op_sanitize_overflow_impl(input, other, self.mul)
            return self.tensor(self.builder.create_mul(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {scalar_ty}")

    def truediv(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, False, False, True, True)
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        # float / int
        if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
            other = self.cast(other, input_scalar_ty)
        # int / float
        elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
            input = self.cast(input, other_scalar_ty)
        # int / int (cast to tl.float32)
        elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
            input = self.cast(input, tl.float32)
            other = self.cast(other, tl.float32)
        # float / float (cast to the highest exponent type)
        elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
            if input_scalar_ty.fp_mantissa_width > other_scalar_ty.fp_mantissa_width:
                other = self.cast(other, input_scalar_ty)
            else:
                input = self.cast(input, other_scalar_ty)
        # unreachable
        else:
            raise TypeError(f"unexpected type {input_scalar_ty}")
        return self.tensor(self.builder.create_fdiv(input.handle, other.handle), input.type)

    def floordiv(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, False, False, True, True)
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        if input_scalar_ty.is_int() and other_scalar_ty.is_int():
            ret_ty = self.integer_promote_impl(input_scalar_ty, other_scalar_ty)
            input = self.cast(input, ret_ty)
            other = self.cast(other, ret_ty)
            if ret_ty.is_int_signed():
                return self.tensor(self.builder.create_sdiv(input.handle, other.handle), input.type)
            else:
                return self.tensor(self.builder.create_udiv(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {input_scalar_ty}")

    def fdiv(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number, ieee_rounding: bool) -> TensorTy:
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
            raise TypeError("both operands of fdiv must have floating scalar type")
        input, other = self.binary_op_type_checking_impl(input, other, False, False, False, True)
        if ieee_rounding:
            ret = self.builder.create_precise_divf(input.handle, other.handle)
        else:
            ret = self.builder.create_fdiv(input.handle, other.handle)
        return self.tensor(ret, input.type)

    def mod(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, False, False, True, True)
        scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        # float % float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_frem(input.handle, other.handle), input.type)
        # % int
        elif scalar_ty.is_int():
            if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
                raise TypeError("Cannot mod " + scalar_ty.__repr__() + " by " + other_scalar_ty.__repr__() + " "
                                "because they have different signedness;"
                                "this is unlikely to result in a useful answer. Cast them to the same signedness.")
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_srem(input.handle, other.handle), input.type)
            else:
                return self.tensor(self.builder.create_urem(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {scalar_ty}")

##############
# other arithmetic ops
##############

    def minimum(self, x: TensorTy, y: TensorTy, propagate_nan: tl.PropagateNan):
        x, y = self.binary_op_type_checking_impl(x, y)
        dtype = x.dtype
        if dtype.is_floating():
            if propagate_nan == tl.PropagateNan.ALL:
                return self.tensor(self.builder.create_minimumf(x.handle, y.handle), x.type)
            elif propagate_nan == tl.PropagateNan.NONE:
                return self.tensor(self.builder.create_minnumf(x.handle, y.handle), x.type)
            else:
                raise ValueError(f"Unexpected propagate_nan {propagate_nan}")
        elif dtype.is_int_signed():
            return self.tensor(self.builder.create_minsi(x.handle, y.handle), x.type)
        elif dtype.is_int_unsigned():
            return self.tensor(self.builder.create_minui(x.handle, y.handle), x.type)
        else:
            raise TypeError(f"Unexpected dtype {dtype}")

    def maximum(self, x: TensorTy, y: TensorTy, propagate_nan: tl.PropagateNan):
        x, y = self.binary_op_type_checking_impl(x, y)
        dtype = x.dtype
        if dtype.is_floating():
            if propagate_nan == tl.PropagateNan.ALL:
                return self.tensor(self.builder.create_maximumf(x.handle, y.handle), x.type)
            elif propagate_nan == tl.PropagateNan.NONE:
                return self.tensor(self.builder.create_maxnumf(x.handle, y.handle), x.type)
            else:
                raise ValueError(f"Unexpected propagate_nan {propagate_nan}")
        elif dtype.is_int_signed():
            return self.tensor(self.builder.create_maxsi(x.handle, y.handle), x.type)
        elif dtype.is_int_unsigned():
            return self.tensor(self.builder.create_maxui(x.handle, y.handle), x.type)
        else:
            raise TypeError(f"Unexpected dtype {dtype}")

    def clamp(self, x: TensorTy, min: TensorTy, max: TensorTy, propagate_nan: tl.PropagateNan):
        min, max = self.binary_op_type_checking_impl(min, max)
        x, min = self.binary_op_type_checking_impl(x, min)
        x, max = self.binary_op_type_checking_impl(x, max)

        dtype = x.dtype
        if dtype.is_floating():
            return self.tensor(self.builder.create_clampf(x.handle, min.handle, max.handle, propagate_nan), x.type)
        else:
            raise TypeError(f"Unexpected dtype {dtype}. Only floating point clamp is supported")

##############
# bitwise ops
##############

    def bitwise_op_type_checking_impl(self, input: TensorTy, other: TensorTy) -> Tuple[TensorTy, TensorTy]:
        input, other = self.binary_op_type_checking_impl(input, other)
        input_sca_ty = input.type.scalar
        other_sca_ty = other.type.scalar
        if not input_sca_ty.is_int() or not other_sca_ty.is_int():
            raise IncompatibleTypeErrorImpl(input_sca_ty, other_sca_ty)
        ret_sca_ty = self.integer_promote_impl(input_sca_ty, other_sca_ty)
        if ret_sca_ty != input_sca_ty:
            input = self.cast(input, ret_sca_ty)
        if ret_sca_ty != other_sca_ty:
            other = self.cast(other, ret_sca_ty)
        return input, other

    def and_(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_and(input.handle, other.handle), input.type)

    def or_(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_or(input.handle, other.handle), input.type)

    def xor_(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_xor(input.handle, other.handle), input.type)

    def logical_and(self, input: TensorTy, other: TensorTy) -> TensorTy:
        if not input.type.is_int1():
            input = self.bitcast(input, tl.int1)
        if not other.type.is_int1():
            other = self.bitcast(other, tl.int1)
        return self.and_(input, other)

    def logical_or(self, input: TensorTy, other: TensorTy) -> TensorTy:
        if not input.type.is_int1():
            input = self.bitcast(input, tl.int1)
        if not other.type.is_int1():
            other = self.bitcast(other, tl.int1)
        return self.or_(input, other)

    def not_(self, input: TensorTy):
        if not input.type.is_int1():
            input = self.bitcast(input, tl.int1)
        return self.invert(input)

    def lshr(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_lshr(input.handle, other.handle), input.type)

    def ashr(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_ashr(input.handle, other.handle), input.type)

    def shl(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.bitwise_op_type_checking_impl(input, other)
        return self.tensor(self.builder.create_shl(input.handle, other.handle), input.type)

# ===----------------------------------------------------------------------===//
#                               Unary Operators
# ===----------------------------------------------------------------------===//

    def plus(self, input: TensorTy) -> TensorTy:
        return input

    def minus(self, input: TensorTy) -> TensorTy:
        input_sca_ty = input.type.scalar
        if input_sca_ty.is_ptr():
            raise ValueError("wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")")
        if input_sca_ty.is_floating():
            return self.tensor(self.builder.create_fneg(input.handle), input.type)
        _0 = self.tensor(self.builder.get_null_value(input_sca_ty.to_ir(self.builder)), input_sca_ty)
        return self.sub(_0, input, True)

    def invert(self, input: TensorTy) -> TensorTy:
        input_sca_ty = input.type.scalar
        if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
            raise ValueError("wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")")
        _1 = self.tensor(self.builder.get_all_ones_value(input_sca_ty.to_ir(self.builder)), input_sca_ty)
        return self.xor_(input, _1)

# ===----------------------------------------------------------------------===//
#                               Comparison Operators
# ===----------------------------------------------------------------------===//

    def _bool_like(self, v: TensorTy) -> tl.block_type:
        return v.type.with_element_ty(tl.int1)

    def greater_than(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float > float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOGT(input.handle, other.handle), self._bool_like(input))
        # > int
        elif scalar_ty.is_int():
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_icmpSGT(input.handle, other.handle), self._bool_like(input))
            else:
                return self.tensor(self.builder.create_icmpUGT(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def greater_equal(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float >= float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOGE(input.handle, other.handle), self._bool_like(input))
        # >= int
        elif scalar_ty.is_int():
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_icmpSGE(input.handle, other.handle), self._bool_like(input))
            else:
                return self.tensor(self.builder.create_icmpUGE(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def less_than(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float < float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOLT(input.handle, other.handle), self._bool_like(input))
        # < int
        elif scalar_ty.is_int():
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_icmpSLT(input.handle, other.handle), self._bool_like(input))
            else:
                return self.tensor(self.builder.create_icmpULT(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def less_equal(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float < float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOLE(input.handle, other.handle), self._bool_like(input))
        # < int
        elif scalar_ty.is_int():
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_icmpSLE(input.handle, other.handle), self._bool_like(input))
            else:
                return self.tensor(self.builder.create_icmpULE(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def equal(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float == float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpOEQ(input.handle, other.handle), self._bool_like(input))
        # == int
        elif scalar_ty.is_int():
            return self.tensor(self.builder.create_icmpEQ(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

    def not_equal(self, input: TensorTy, other: TensorTy) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other)
        scalar_ty = input.type.scalar
        # float == float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_fcmpUNE(input.handle, other.handle), self._bool_like(input))
        # == int
        elif scalar_ty.is_int():
            return self.tensor(self.builder.create_icmpNE(input.handle, other.handle), self._bool_like(input))
        raise TypeError(f"unexpected type {scalar_ty}")

# ===----------------------------------------------------------------------===//
#                               Block Creation
# ===----------------------------------------------------------------------===//

    def arange(self, start: int, end: int, *, ret_ty: tl.block_type = None) -> TensorTy:
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("arange's arguments must be of type tl.constexpr")
        is_start_int64 = bool(start >> 32)
        is_end_int64 = bool(end >> 32)
        if is_start_int64 or is_end_int64:
            raise ValueError("arange must fit in int32")
        if end <= start:
            raise ValueError("arange's end argument must be greater than the start argument")
        range = end - start
        if (range & (range - 1)) != 0:
            raise ValueError("arange's range must be a power of 2")
        shape = [range]
        if ret_ty is None:
            ret_ty = tl.block_type(tl.int32, shape)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        return self.tensor(self.builder.create_make_range(ret_ty_ir, start, end), ret_ty)

    def scalar_constant(self, value, dtype: tl.dtype) -> TensorTy:
        # scalar
        if dtype is None:
            raise ValueError("dtype must be specified when value is not a tensor")
        if value == 0:
            value = self.builder.get_null_value(dtype.to_ir(self.builder))
        elif dtype.is_fp8():
            value = self.builder.get_fp32(value)
            value = self.builder.create_fp_trunc(value, dtype.to_ir(self.builder))
        else:
            get_value_fn = getattr(self.builder, f"get_{dtype.name}")
            value = get_value_fn(value)
        return self.tensor(value, dtype)

    def make_scalar(self, value, dtype: tl.dtype) -> TensorTy:
        if isinstance(value, tl.tensor):
            assert value.numel.value == 1, "only accepts size-1 tensor"
            return self.cast(value, dtype)
        # scalar
        return self.scalar_constant(value, dtype)

    def full(self, shape: List[int], value, dtype: tl.dtype) -> TensorTy:
        return self.splat(self.make_scalar(value, dtype), shape)

# ===----------------------------------------------------------------------===//
#                               Shape Manipulation
# ===----------------------------------------------------------------------===//

    def splat(self, value: TensorTy, shape: List[int]) -> TensorTy:
        assert not value.type.is_block(), "Cannot splat a block tensor"
        if len(shape) == 0:
            return value
        ret_ty = tl.block_type(value.dtype, shape)
        return self.tensor(self.builder.create_splat(ret_ty.to_ir(self.builder), value.handle), ret_ty)

    def unsplat(self, value: TensorTy) -> TensorTy:
        return self.tensor(self.builder.create_unsplat(value.handle), value.dtype)

    def reshape(self, input: TensorTy, dst_shape: List[int], can_reorder: bool) -> TensorTy:
        numel = 1
        for s in dst_shape:
            numel *= s
        if input.type.numel != numel:
            raise ValueError("reshape() cannot change total number of elements in tensor")
        ret_ty = tl.block_type(input.type.scalar, dst_shape)
        return self.tensor(self.builder.create_reshape(input.handle, dst_shape, can_reorder), ret_ty)

    def expand_dims(self, input: TensorTy, axis: int) -> TensorTy:
        dst_shape = [tl._unwrap_if_constexpr(x) for x in input.shape]
        dst_shape.insert(axis, 1)

        if not input.type.is_block():
            return self.splat(input, shape=dst_shape)

        ret_ty = tl.block_type(input.type.scalar, dst_shape)
        return self.tensor(self.builder.create_expand_dims(input.handle, axis), ret_ty)

    def cat(self, lhs: TensorTy, rhs: TensorTy, can_reorder: bool) -> TensorTy:
        assert can_reorder, "current implementation of `cat` always may reorder elements"
        assert len(lhs.shape) == 1, f"expected 1D input for cat, got {len(lhs.shape)}D"
        ret_type = tl.block_type(lhs.type.scalar, [lhs.shape[0] + rhs.shape[0]])
        return self.tensor(self.builder.create_cat(lhs.handle, rhs.handle), ret_type)

    def join(self, a: TensorTy, b: TensorTy) -> TensorTy:
        a, b = self.broadcast_impl_value(a, b)

        # The IR can't handle joining two scalars, so upcast them to 1D tensors,
        # then downcast the result.
        was_rank_1 = a.shape == []
        if was_rank_1:
            a = self.expand_dims(a, 0)
            b = self.expand_dims(b, 0)

        if isinstance(a.shape[-1], tl.constexpr):
            two = tl.constexpr(2)
        else:
            two = 2
        new_shape = a.shape + [two]

        ret_type = tl.block_type(a.type.scalar, new_shape)
        ret = self.tensor(self.builder.create_join(a.handle, b.handle), ret_type)

        if was_rank_1:
            ret = self.reshape(ret, [2], can_reorder=False)

        return ret

    def split(self, a: TensorTy) -> Tuple[TensorTy, TensorTy]:
        assert (len(a.shape) > 0), "split requires a non-scalar tensor"
        assert (tl._unwrap_if_constexpr(a.shape[-1]) == 2), \
            f"expected last dimension to be 2 for split, got {tl._unwrap_if_constexpr(a.shape[-1])}"

        new_shape = a.shape[:-1]
        ret_type = tl.block_type(a.type.scalar, new_shape)
        outLHS, outRHS = self.builder.create_split(a.handle)
        return (
            self.tensor(outLHS, ret_type),
            self.tensor(outRHS, ret_type),
        )

    def permute(self, input: TensorTy, dims: Tuple[int]) -> TensorTy:
        if len(input.shape) != len(dims):
            raise ValueError(
                f"permute dims must have the same length as input shape, got {len(input.shape)} and {len(dims)}")
        if sorted(tl._unwrap_if_constexpr(d) for d in dims) != list(range(len(dims))):
            raise ValueError(f"permute dims must be a permutation of 0, 1, ..., n-1, but were {dims}")

        ret_type = tl.block_type(input.type.scalar, [input.shape[d] for d in dims])
        return self.tensor(self.builder.create_trans(input.handle, dims), ret_type)

    def broadcast_impl_shape(self, input: TensorTy, shape: Tuple[int]) -> TensorTy:
        if not input.type.is_block():
            return self.splat(input, shape)
        src_shape = input.type.get_block_shapes()
        if len(src_shape) != len(shape):
            raise ValueError(f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
        if shape == src_shape:
            return input
        for i, item in enumerate(src_shape):
            if shape[i] != item and item != 1:
                raise ValueError(f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                                 f" must match the existing size ({item}) at non-singleton dimension"
                                 f" {i}: {src_shape}, {shape}")
        ret_ty = tl.block_type(input.type.scalar, shape)
        return self.tensor(self.builder.create_broadcast(input.handle, shape), ret_ty)

    def broadcast_impl_value(self, lhs: TensorTy, rhs: TensorTy) -> TensorTy:
        lhs_ty = lhs.type
        rhs_ty = rhs.type

        # make_shape_compatible(block, scalar)
        if lhs_ty.is_block() and not rhs_ty.is_block():
            rhs_ty = lhs_ty.with_element_ty(rhs_ty.scalar)
            rhs = self.tensor(self.builder.create_splat(rhs_ty.to_ir(self.builder), rhs.handle), rhs_ty)
        # make_shape_compatible(scalar, block)
        elif not lhs_ty.is_block() and rhs_ty.is_block():
            lhs_ty = rhs_ty.with_element_ty(lhs_ty.scalar)
            lhs = self.tensor(self.builder.create_splat(lhs_ty.to_ir(self.builder), lhs.handle), lhs_ty)
        # make_shape_compatible(block, block)
        elif lhs_ty.is_block() and rhs_ty.is_block():
            lhs_shape = lhs_ty.get_block_shapes()
            rhs_shape = rhs_ty.get_block_shapes()

            if len(lhs_shape) < len(rhs_shape):
                # Add new axes to lhs
                for _ in range(len(lhs_shape), len(rhs_shape)):
                    lhs = self.tensor(self.builder.create_expand_dims(lhs.handle, 0),
                                      tl.block_type(lhs_ty.scalar, [1] + lhs_shape.values))
                    lhs_ty = lhs.type
                    lhs_shape = lhs_ty.get_block_shapes()
            elif len(rhs_shape) < len(lhs_shape):
                # Add new axes to rhs
                for _ in range(len(rhs_shape), len(lhs_shape)):
                    rhs = self.tensor(self.builder.create_expand_dims(rhs.handle, 0),
                                      tl.block_type(rhs_ty.scalar, [1] + rhs_shape.values))
                    rhs_ty = rhs.type
                    rhs_shape = rhs_ty.get_block_shapes()
            assert len(rhs_shape) == len(lhs_shape), \
                f"expected tensors of equal rank for broadcast, got {len(lhs_shape)} and {len(rhs_shape)}"

            ret_shape = []
            for i, left in enumerate(lhs_shape):
                right = rhs_shape[i]
                if left == 1:
                    ret_shape.append(right)
                elif (right == 1) or (right == left):
                    ret_shape.append(left)
                else:
                    raise ValueError("Cannot make_shape_compatible: incompatible dimensions "
                                     "at index " + str(i) + ": " + str(left) + " and " + str(right))
            if lhs_shape != ret_shape:
                ret_ty = tl.block_type(lhs_ty.scalar, ret_shape)
                lhs = self.tensor(self.builder.create_broadcast(lhs.handle, ret_shape), ret_ty)
            if rhs_shape != ret_shape:
                ret_ty = tl.block_type(rhs_ty.scalar, ret_shape)
                rhs = self.tensor(self.builder.create_broadcast(rhs.handle, ret_shape), ret_ty)
        # (scalar, scalar) => returns original blocks
        return lhs, rhs

#######
# cast
#######

    def _str_to_rounding_mode(self, rounding_mode: Optional[str]):
        if rounding_mode is None:
            return None
        if rounding_mode == 'rtne':
            return ir.ROUNDING_MODE.RTNE
        if rounding_mode == 'rtz':
            return ir.ROUNDING_MODE.RTZ
        raise ValueError(f"Invalid rounding mode: {rounding_mode}. Supported rounding modes are 'rtne' and 'rtz'.")

    def bitcast(self, input: TensorTy, dst_ty: tl.dtype) -> TensorTy:
        src_ty = input.type
        if src_ty.is_block():
            dst_ty = src_ty.with_element_ty(dst_ty.scalar)
        if src_ty == dst_ty:
            return input
        src_sca_ty = src_ty.scalar
        dst_sca_ty = dst_ty.scalar
        if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
            return self.cast(input, dst_ty)
        # Bitcast
        src_bits = src_sca_ty.primitive_bitwidth
        dst_bits = dst_sca_ty.primitive_bitwidth
        if src_bits != dst_bits:
            raise ValueError("Cannot bitcast data-type of size " + str(src_bits) + " to "
                             "data-type of size " + str(dst_bits))
        return self.tensor(self.builder.create_bitcast(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

    def cast(self, input: TensorTy, dst_ty: tl.dtype, fp_downcast_rounding: Optional[str] = None) -> TensorTy:
        src_ty = input.type
        src_sca_ty = src_ty.scalar
        dst_sca_ty = dst_ty.scalar
        if src_sca_ty == dst_sca_ty:
            return input
        if src_ty.is_block():
            dst_ty = src_ty.with_element_ty(dst_sca_ty)

        # For fp downcasting default rounding mode should be RTNE, for all other conversions it should
        # not be set
        fp_downcast_rounding = self._str_to_rounding_mode(fp_downcast_rounding)
        use_custom_rounding = False
        if dst_sca_ty.is_floating() and src_sca_ty.is_floating(
        ) and dst_sca_ty.primitive_bitwidth < src_sca_ty.primitive_bitwidth:
            if fp_downcast_rounding is None: fp_downcast_rounding = ir.ROUNDING_MODE.RTNE
            elif fp_downcast_rounding != ir.ROUNDING_MODE.RTNE: use_custom_rounding = True
        else:
            if fp_downcast_rounding is not None:
                raise ValueError("fp_downcast_rounding should be set only for truncating fp conversions. "
                                 "Source scalar type is " + str(src_sca_ty) + " and destination type is " +
                                 str(dst_sca_ty))

        if (src_sca_ty.is_fp8e4b15() or dst_sca_ty.is_fp8e4b15()):
            assert self.builder.codegen_fns.get(
                "convert_custom_types") is not None, "target doesn't provide conversion for this type."
            return self.builder.codegen_fns["convert_custom_types"](input, dst_ty, fp_downcast_rounding, _semantic=self)
        # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
        # and non-default rounding modes for downcasting
        if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
           (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()) or \
           use_custom_rounding:
            return self.tensor(
                self.builder.create_fp_to_fp(input.handle, dst_ty.to_ir(self.builder), fp_downcast_rounding), dst_ty)

        # bf16 <=> (not fp32)
        if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
           (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
            return self.cast(self.cast(input, tl.float32), dst_sca_ty)

        # Standard floating types' casting: truncation
        #   fp64 => fp32, fp16, bf16
        #   fp32 => fp16, bf16
        truncate_fp = src_sca_ty.is_floating() and \
            dst_sca_ty.is_floating() and \
            src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
        if truncate_fp:
            return self.tensor(self.builder.create_fp_trunc(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Standard floating types' casting: extension
        #   fp32 => fp64
        #   fp16 => fp32, fp64
        #   bf16 => fp32, fp64
        ext_fp = src_sca_ty.is_floating() and \
            dst_sca_ty.is_floating() and \
            src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
        if ext_fp:
            return self.tensor(self.builder.create_fp_ext(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting between integer types
        if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
           (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
            sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
            if dst_sca_ty.is_bool():
                ty = input.dtype.to_ir(self.builder)
                _0 = self.tensor(self.builder.get_null_value(ty), input.dtype)
                return self.not_equal(input, _0)
            else:
                return self.tensor(self.builder.create_int_cast(input.handle, dst_ty.to_ir(self.builder), sign_extend),
                                   dst_ty)

        # Casting standard floating types to integer types
        if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
            if dst_sca_ty.is_bool():
                ty = input.dtype.to_ir(self.builder)
                _0 = self.tensor(self.builder.get_null_value(ty), input.dtype)
                return self.not_equal(input, _0)
            elif dst_sca_ty.is_int_signed():
                return self.tensor(self.builder.create_fp_to_si(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            else:
                return self.tensor(self.builder.create_fp_to_ui(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting integer types to standard floating types
        if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
            if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
                return self.tensor(self.builder.create_ui_to_fp(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            else:
                return self.tensor(self.builder.create_si_to_fp(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting pointer types to integer types
        if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
            bitwidth = dst_sca_ty.int_bitwidth
            if bitwidth == 64:
                return self.tensor(self.builder.create_ptr_to_int(input.handle, dst_ty.to_ir(self.builder)), dst_ty)
            if bitwidth == 1:
                return self.not_equal(self.cast(input, tl.int64), self.tensor(self.builder.get_int64(0), tl.int64))

        # Casting integer types to pointer types
        if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
            return self.tensor(self.builder.create_int_to_ptr(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        # Casting pointer types to pointer types
        if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
            return self.tensor(self.builder.create_bitcast(input.handle, dst_ty.to_ir(self.builder)), dst_ty)

        assert False, f'cannot cast {input} to {dst_ty}'

# ===----------------------------------------------------------------------===//
#                               Memory Operators
# ===----------------------------------------------------------------------===//

    def _str_to_load_cache_modifier(self, cache_modifier):
        cache = ir.CACHE_MODIFIER.NONE  # default
        if cache_modifier:
            if cache_modifier == ".ca":
                cache = ir.CACHE_MODIFIER.CA
            elif cache_modifier == ".cg":
                cache = ir.CACHE_MODIFIER.CG
            elif cache_modifier == ".cs":
                cache = ir.CACHE_MODIFIER.CS
            elif cache_modifier == ".cv":
                cache = ir.CACHE_MODIFIER.CV
            else:
                raise ValueError(f"Cache modifier {cache_modifier} not supported")
        return cache

    def _str_to_store_cache_modifier(self, cache_modifier):
        cache = ir.CACHE_MODIFIER.NONE  # default
        if cache_modifier:
            if cache_modifier == ".wb":
                cache = ir.CACHE_MODIFIER.WB
            elif cache_modifier == ".cg":
                cache = ir.CACHE_MODIFIER.CG
            elif cache_modifier == ".cs":
                cache = ir.CACHE_MODIFIER.CS
            elif cache_modifier == ".wt":
                cache = ir.CACHE_MODIFIER.WT
            else:
                raise ValueError(f"Cache modifier {cache_modifier} not supported")
        return cache

    def _str_to_eviction_policy(self, eviction_policy):
        eviction = ir.EVICTION_POLICY.NORMAL  # default
        if eviction_policy:
            if eviction_policy == "evict_last":
                eviction = ir.EVICTION_POLICY.EVICT_LAST
            elif eviction_policy == "evict_first":
                eviction = ir.EVICTION_POLICY.EVICT_FIRST
            else:
                raise ValueError(f"Eviction policy {eviction_policy} not supported")
        return eviction

    def _str_to_padding_option(self, padding_option):
        padding = None  # default
        if padding_option:
            if padding_option == "zero":
                padding = ir.PADDING_OPTION.PAD_ZERO
            elif padding_option == "nan":
                padding = ir.PADDING_OPTION.PAD_NAN
            else:
                raise ValueError(f"Padding option {padding_option} not supported")
        return padding

    def _str_to_sem(self, sem_option):
        sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
        if sem_option:
            if sem_option == "acquire":
                sem = ir.MEM_SEMANTIC.ACQUIRE
            elif sem_option == "release":
                sem = ir.MEM_SEMANTIC.RELEASE
            elif sem_option == "acq_rel":
                sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
            elif sem_option == "relaxed":
                sem = ir.MEM_SEMANTIC.RELAXED
            else:
                raise ValueError(f"Memory semantic {sem_option} not supported")
        return sem

    def _str_to_scope(self, scope_option):
        scope = ir.MEM_SYNC_SCOPE.GPU
        if scope_option:
            if scope_option == "gpu":
                scope = ir.MEM_SYNC_SCOPE.GPU
            elif scope_option == "cta":
                scope = ir.MEM_SYNC_SCOPE.CTA
            elif scope_option == "sys":
                scope = ir.MEM_SYNC_SCOPE.SYSTEM
            else:
                raise ValueError(f"Memory semantic {scope_option} not supported")
        return scope

    def load(self, ptr: TensorTy, mask: Optional[TensorTy], other: Optional[TensorTy], boundary_check: Tuple,
             padding_option: str, cache_modifier: str, eviction_policy: str, is_volatile: bool) -> TensorTy:
        cache = self._str_to_load_cache_modifier(cache_modifier)
        eviction = self._str_to_eviction_policy(eviction_policy)
        padding = self._str_to_padding_option(padding_option)
        if not ptr.type.scalar.is_ptr():
            raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

        # Check `mask`, `other`, `boundary_check`, and `padding` arguments
        if mask is None and other is not None:
            raise ValueError("`other` cannot be provided without `mask`")
        if padding or boundary_check:
            raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                             "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                             "use block pointers (defined by `make_block_ptr`) instead")

        # For a pointer of scalar, check the type of `mask` and `other`
        if not ptr.type.is_block():
            if mask and mask.type.is_block():
                raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
            if other and other.type.is_block():
                raise ValueError("Other argument cannot be block type if pointer argument is not a block")

        # Make `mask` and `other` into the same shape as `ptr`
        if ptr.type.is_block():
            if mask is not None:
                ptr, mask = self.broadcast_impl_value(ptr, mask)
            if other is not None:
                ptr, other = self.broadcast_impl_value(ptr, other)

        # Get `pointer_type<elt_ty>` and `elt_ty`
        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty

        # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
        is_bool = elt_ty == tl.int1
        if is_bool:
            elt_ty = tl.int8
            ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
            ptr = self.cast(ptr, ptr_ty)

        # Cast `other` into `elt_ty` type
        if other is not None:
            other = self.cast(other, elt_ty)

        # Create loaded result type `dst_ty`
        if ptr.type.is_block():
            dst_ty = ptr.type.with_element_ty(elt_ty)
        else:
            # Load by de-referencing the pointer of scalar
            dst_ty = elt_ty

        # Build IR
        if mask is None:
            ret = self.tensor(self.builder.create_load(ptr.handle, cache, eviction, is_volatile), dst_ty)
        else:
            ret = self.tensor(
                self.builder.create_masked_load(ptr.handle, mask.handle, other.handle if other else None, cache,
                                                eviction, is_volatile), dst_ty)
        if is_bool:
            ret = self.cast(ret, tl.int1)
        return ret

    def descriptor_load(self, desc: tl.tensor_descriptor_base, offsets, cache_modifier: str,
                        eviction_policy: str) -> TensorTy:
        assert isinstance(desc, tl.tensor_descriptor_base), \
            f"expected a tensor descriptor, got {type(desc).__name__}"
        ndim = len(desc.block_shape)
        assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"

        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        x = self.builder.create_descriptor_load(desc.handle, offsets, self._str_to_load_cache_modifier(cache_modifier),
                                                self._str_to_eviction_policy(eviction_policy))
        return self.tensor(x, desc.block_type)

    def validate_store_like(self, desc: tl.tensor_descriptor_base, value: TensorTy, offsets) -> None:
        assert isinstance(desc, tl.tensor_descriptor_base), \
            f"expected a tensor descriptor, got {type(desc).__name__}"
        ndim = len(desc.block_shape)
        assert len(offsets) == ndim, f"expected {ndim} offsets, but got {len(offsets)}"
        assert value.shape == desc.block_shape, \
            f"expected value shape {desc.block_shape}, got {value.shape}"

    def descriptor_store(self, desc: tl.tensor_descriptor_base, value: TensorTy, offsets) -> TensorTy:
        self.validate_store_like(desc, value, offsets)
        # implicitly cast to the descriptor's type
        value = self.cast(value, desc.dtype)
        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        return self.tensor(self.builder.create_descriptor_store(desc.handle, value.handle, offsets), tl.void)

    def descriptor_atomic_add(self, desc: tl.tensor_descriptor_base, value: TensorTy, offsets) -> TensorTy:
        self.validate_store_like(desc, value, offsets)
        assert desc.dtype in {tl.uint32, tl.int32, tl.uint64, tl.float32, tl.float16, tl.bfloat16}, "Unsupported dtype"
        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        kind = ir.DESCRIPTOR_REDUCE_KIND.ADD
        return self.tensor(self.builder.create_descriptor_reduce(kind, desc.handle, value.handle, offsets), tl.void)

    def _has_native_tma(self, ):
        target = driver.active.get_current_target()
        return (target.backend == "cuda" and target.arch >= 90)

    def _descriptor_atomic_min_max_supported(self, dtype):
        assert dtype in {tl.uint32, tl.int32, tl.uint64, tl.int64, tl.float16, tl.bfloat16}, "Unsupported dtype"
        if dtype in {tl.float16, tl.bfloat16}:
            assert self._has_native_tma(), "16-bit float types require native tma support"

    def descriptor_atomic_min(self, desc: tl.tensor_descriptor_base, value: TensorTy, offsets) -> TensorTy:
        self.validate_store_like(desc, value, offsets)
        self._descriptor_atomic_min_max_supported(desc.dtype)
        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        kind = ir.DESCRIPTOR_REDUCE_KIND.MIN
        return self.tensor(self.builder.create_descriptor_reduce(kind, desc.handle, value.handle, offsets), tl.void)

    def descriptor_atomic_max(self, desc: tl.tensor_descriptor_base, value: TensorTy, offsets) -> TensorTy:
        self.validate_store_like(desc, value, offsets)
        self._descriptor_atomic_min_max_supported(desc.dtype)
        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        kind = ir.DESCRIPTOR_REDUCE_KIND.MAX
        return self.tensor(self.builder.create_descriptor_reduce(kind, desc.handle, value.handle, offsets), tl.void)

    def descriptor_atomic_and(self, desc: tl.tensor_descriptor_base, value: TensorTy, offsets) -> TensorTy:
        self.validate_store_like(desc, value, offsets)
        assert desc.dtype in {tl.uint32, tl.int32, tl.uint64, tl.int64}, "Unsupported dtype"
        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        kind = ir.DESCRIPTOR_REDUCE_KIND.AND
        return self.tensor(self.builder.create_descriptor_reduce(kind, desc.handle, value.handle, offsets), tl.void)

    def descriptor_atomic_or(self, desc: tl.tensor_descriptor_base, value: TensorTy, offsets) -> TensorTy:
        self.validate_store_like(desc, value, offsets)
        assert desc.dtype in {tl.uint32, tl.int32, tl.uint64, tl.int64}, "Unsupported dtype"
        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        kind = ir.DESCRIPTOR_REDUCE_KIND.OR
        return self.tensor(self.builder.create_descriptor_reduce(kind, desc.handle, value.handle, offsets), tl.void)

    def descriptor_atomic_xor(self, desc: tl.tensor_descriptor_base, value: TensorTy, offsets) -> TensorTy:
        self.validate_store_like(desc, value, offsets)
        assert desc.dtype in {tl.uint32, tl.int32, tl.uint64, tl.int64}, "Unsupported dtype"
        offsets = self._convert_to_ir_values(offsets, require_i64=False)
        kind = ir.DESCRIPTOR_REDUCE_KIND.XOR
        return self.tensor(self.builder.create_descriptor_reduce(kind, desc.handle, value.handle, offsets), tl.void)

    def descriptor_gather(self, desc, x_offsets, y_offset, cache_modifier: str, eviction_policy: str) -> TensorTy:
        assert isinstance(desc, tl.tensor_descriptor_base), \
            f"expected a tensor descriptor, got {desc.__class__.__name__}"
        assert cache_modifier == "", "cache modifier is not supported yet"
        assert eviction_policy == "", "eviction policy is not supported yet"

        # Validate descriptor.
        assert len(desc.block_shape) == 2, f"descriptor must be 2D, but got {desc.block_shape}"
        assert desc.block_shape[0] == 1, f"descriptor block must have 1 row, but got {desc.block_shape}"

        # Validate offsets.
        assert len(x_offsets.shape) == 1, f"x offsets must be 1D, but got {x_offsets.shape}"
        assert x_offsets.dtype in {tl.int16,
                                   tl.int32}, f"x offsets must have dtype int16 or int32, but got {x_offsets.dtype}"

        # Validate minimum block size.
        assert x_offsets.shape[0] >= 8, f"descriptor gather must have at least 8 rows, but got {x_offsets.shape}"
        dtype = desc.dtype
        min_cols = 32 // dtype.primitive_bitwidth * 8
        assert desc.block_shape[
            1] >= min_cols, f"descriptor gather of {dtype} must have at least {min_cols} columns, but got {desc.block_shape[1]}"

        type = tl.block_type(desc.dtype, [x_offsets.shape[0], desc.block_shape[1]])
        y_offset = self._convert_to_ir_values((y_offset, ), require_i64=False)[0]
        x = self.builder.create_descriptor_gather(desc.handle, x_offsets.handle, y_offset, type.to_ir(self.builder))
        return self.tensor(x, type)

    def descriptor_scatter(self, desc, value: TensorTy, x_offsets, y_offset) -> TensorTy:
        assert isinstance(desc, tl.tensor_descriptor_base), \
            f"expected a tensor descriptor, got {type(desc).__name__}"

        # Validate descriptor.
        assert len(desc.block_shape) == 2, f"descriptor must be 2D, but got {desc.block_shape}"
        assert desc.block_shape[0] == 1, f"descriptor block must have 1 row, but got {desc.block_shape}"

        # Validate offsets.
        assert len(x_offsets.shape) == 1, f"x offsets must be 1D, but got {x_offsets.shape}"
        assert x_offsets.dtype in {tl.int16,
                                   tl.int32}, f"x offsets must have dtype int16 or int32, but got {x_offsets.dtype}"

        # Validate minimum block size.
        assert x_offsets.shape[0] >= 8, f"descriptor scatter must have at least 8 rows, but got {x_offsets.shape}"
        dtype = desc.dtype
        min_cols = 32 // dtype.primitive_bitwidth * 8
        assert desc.block_shape[
            1] >= min_cols, f"descriptor scatter of {dtype} must have at least {min_cols} columns, but got {desc.block_shape[1]}"

        y_offset = self._convert_to_ir_values((y_offset, ), require_i64=False)[0]
        self.builder.create_descriptor_scatter(desc.handle, value.handle, x_offsets.handle, y_offset)
        return self.tensor(None, tl.void)

    def _broadcast_ptr_val_mask(self, ptr, val, mask):
        ptr_shape = ptr.shape
        if mask is None:
            ptr, val = self.broadcast_tensors(ptr, val)
        else:
            mask = self.to_tensor(mask)
            ptr, val, mask = self.broadcast_tensors(ptr, val, mask)
        if ptr_shape != ptr.shape:
            raise ValueError(f"Expected pointer argument to have shape {ptr.shape} but got {ptr_shape}")
        return ptr, val, mask

    def store(self, ptr: TensorTy, val: TensorTy, mask: Optional[TensorTy], boundary_check, cache_modifier: str,
              eviction_policy: str) -> TensorTy:
        cache = self._str_to_store_cache_modifier(cache_modifier)
        eviction = self._str_to_eviction_policy(eviction_policy)
        if ptr.type.is_const() or ptr.type.scalar.is_const():
            raise ValueError("Cannot store to a constant pointer")

        # Store by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        if not ptr.type.scalar.is_ptr():
            raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.store`")

        # Check `boundary_check` argument
        if boundary_check:
            raise ValueError("`boundary_check` argument is not supported for storing a tensor of pointers or storing a "
                             "scalar. Because the compiler does not know the boundary; please use block pointers "
                             "(defined by `make_block_ptr`) instead")

        # For a pointer of scalar, check the type of `val` and `mask`
        if not ptr.type.is_block():
            if val.type.is_block():
                raise ValueError("Value argument cannot be block type if pointer argument is not a block")
            if mask and mask.type.is_block():
                raise ValueError("Mask argument cannot be block type if pointer argument is not a block")

        # Make `mask` and `val` into the same shape as `ptr`
        if ptr.type.is_block():
            ptr, val, mask = self._broadcast_ptr_val_mask(ptr, val, mask)

        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty

        # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
        if elt_ty == tl.int1:
            elt_ty = tl.int8
            ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
            ptr = self.cast(ptr, ptr_ty)

        # Cast to target data type
        val = self.cast(val, elt_ty)

        # Build IR
        if mask is None:
            return self.tensor(self.builder.create_store(ptr.handle, val.handle, cache, eviction), tl.void)
        if not mask.type.scalar.is_bool():
            raise ValueError("Mask must have boolean scalar type")
        return self.tensor(self.builder.create_masked_store(ptr.handle, val.handle, mask.handle, cache, eviction),
                           tl.void)

#########
# atomic
#########

    def atomic_cas(self, ptr: TensorTy, cmp: TensorTy, val: TensorTy, sem: str, scope: str) -> TensorTy:
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        element_ty = ptr.type.scalar.element_ty
        if element_ty.primitive_bitwidth not in [16, 32, 64]:
            raise ValueError("atomic_cas only supports elements with width {16, 32, 64}")
        return self.tensor(self.builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle, sem, scope), val.type)

    def atom_red_typechecking_impl(self, ptr: TensorTy, val: TensorTy, mask: TensorTy,
                                   op: str) -> Tuple[TensorTy, TensorTy, TensorTy]:
        if not ptr.type.scalar.is_ptr():
            raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())
        if ptr.type.is_const() or ptr.type.element_ty.is_const():
            raise ValueError("Cannot store to a constant pointer")
        element_ty = ptr.type.scalar.element_ty
        if element_ty is tl.float16 and op != 'add':
            raise ValueError("atomic_" + op + " does not support fp16")
        if element_ty is tl.bfloat16 and op != 'add':
            raise ValueError("atomic_" + op + " does not support bf16")
        if element_ty in [tl.int16, tl.uint16] or element_ty.primitive_bitwidth < 16:
            raise ValueError("atomic_" + op + " does not support " + str(element_ty))
        if ptr.type.is_block():
            ptr, val, mask = self._broadcast_ptr_val_mask(ptr, val, mask)
        val = self.cast(val, ptr.type.scalar.element_ty)
        if mask is None:
            mask_ir = self.builder.get_int1(True)
            mask_ty = tl.int1
            if ptr.type.is_block():
                mask_ty = ptr.type.with_element_ty(tl.int1)
                mask_ir = self.builder.create_splat(mask_ty.to_ir(self.builder), mask_ir)
            mask = self.tensor(mask_ir, mask_ty)
        elif not mask.type.scalar.is_bool():
            raise ValueError("Mask must have boolean scalar type")
        return ptr, val, mask

    def _signbit(self, x: TensorTy) -> TensorTy:
        bitwidth = x.dtype.primitive_bitwidth
        idtype = tl.get_int_dtype(bitwidth=bitwidth, signed=False)
        ix = self.bitcast(x, idtype)
        signbit = self.lshr(ix, bitwidth - 1)
        return self.cast(signbit, tl.int1)

    def atomic_max(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'max')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        sca_ty = val.type.scalar
        # direct call to atomic_max for integers
        if sca_ty.is_int():
            if sca_ty.is_int_signed():
                return self.tensor(
                    self.builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope),
                    val.type)
            else:
                return self.tensor(
                    self.builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, ptr.handle, val.handle, mask.handle, sem, scope),
                    val.type)
        # for float
        # return atomic_smax(i_ptr, i_val) if val >= 0
        # return atomic_umin(i_ptr, i_val) if val < 0
        if sca_ty not in {tl.float32, tl.float64}:
            raise TypeError(f"atomic_max not supported for dtype {sca_ty}")

        i_type = tl.int32 if sca_ty == tl.float32 else tl.int64
        i_val = self.bitcast(val, i_type)
        i_ptr = self.bitcast(ptr, tl.pointer_type(i_type, 1))
        ui_type = tl.uint32 if sca_ty == tl.float32 else tl.uint64
        ui_val = self.bitcast(val, ui_type)
        ui_ptr = self.bitcast(ptr, tl.pointer_type(ui_type, 1))
        neg = self._signbit(val)
        pos = self.not_(neg)
        pos_ret = self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, i_ptr.handle, i_val.handle,
                                           self.and_(mask, pos).handle, sem, scope), i_val.type)
        neg_ret = self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, ui_ptr.handle, ui_val.handle,
                                           self.and_(mask, neg).handle, sem, scope), ui_val.type)
        ret = self.where(pos, pos_ret, neg_ret)
        return self.bitcast(ret, sca_ty)

    def atomic_min(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'min')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        sca_ty = val.type.scalar
        # direct call to atomic_min for integers
        if sca_ty.is_int():
            if sca_ty.is_int_signed():
                return self.tensor(
                    self.builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope),
                    val.type)
            else:
                return self.tensor(
                    self.builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, ptr.handle, val.handle, mask.handle, sem, scope),
                    val.type)
        # for float
        # return atomic_smin(i_ptr, i_val) if val >= 0
        # return atomic_umax(i_ptr, i_val) if val < 0
        if sca_ty not in {tl.float32, tl.float64}:
            raise TypeError(f"atomic_min not supported for dtype {sca_ty}")

        i_type = tl.int32 if sca_ty == tl.float32 else tl.int64
        i_val = self.bitcast(val, i_type)
        i_ptr = self.bitcast(ptr, tl.pointer_type(i_type, 1))
        ui_type = tl.uint32 if sca_ty == tl.float32 else tl.uint64
        ui_val = self.bitcast(val, ui_type)
        ui_ptr = self.bitcast(ptr, tl.pointer_type(ui_type, 1))
        neg = self._signbit(val)
        pos = self.not_(neg)
        pos_ret = self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, i_ptr.handle, i_val.handle,
                                           self.and_(mask, pos).handle, sem, scope), i_val.type)
        neg_ret = self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX, ui_ptr.handle, ui_val.handle,
                                           self.and_(mask, neg).handle, sem, scope), ui_val.type)
        ret = self.where(pos, pos_ret, neg_ret)
        return self.bitcast(ret, sca_ty)

    def atomic_add(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'add')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        sca_ty = val.type.scalar
        op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
        return self.tensor(self.builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle, sem, scope),
                           val.type)

    def atomic_and(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'and')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        return self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.AND, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    def atomic_or(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'or')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        return self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    def atomic_xor(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'xor')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        return self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.XOR, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

    def atomic_xchg(self, ptr: TensorTy, val: TensorTy, mask: TensorTy, sem: str, scope: str) -> TensorTy:
        ptr, val, mask = self.atom_red_typechecking_impl(ptr, val, mask, 'xchg')
        sem = self._str_to_sem(sem)
        scope = self._str_to_scope(scope)
        return self.tensor(
            self.builder.create_atomic_rmw(ir.ATOMIC_OP.XCHG, ptr.handle, val.handle, mask.handle, sem, scope),
            val.type)

# ===----------------------------------------------------------------------===//
#                               Linear Algebra
# ===----------------------------------------------------------------------===//

    def _str_to_dot_input_precision(self, input_precision):
        assert input_precision.lower() in self.builder.options.allowed_dot_input_precisions, \
            f"input_precision must be one of {self.builder.options.allowed_dot_input_precisions}. Got {input_precision}"
        input_precision = input_precision.upper()
        if input_precision == "TF32X3":
            input_precision = "TF32x3"
        if input_precision == "BF16X3":
            input_precision = "BF16x3"
        if input_precision == "BF16X6":
            input_precision = "BF16x6"
        return getattr(ir.INPUT_PRECISION, input_precision)

    def dot(self, lhs: TensorTy, rhs: TensorTy, acc: TensorTy, input_precision: Optional[str],
            max_num_imprecise_acc: int, out_dtype: tl.dtype | None) -> TensorTy:
        assert lhs.type.is_block() and rhs.type.is_block(), "dot operands must be block tensors (not scalars)"

        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
            # All combinations of supported fp8 x fp8 are permitted
            pass
        else:
            assert lhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                                 tl.float64), f"Unsupported lhs dtype {lhs.dtype}"
            assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
                                 tl.float64), f"Unsupported rhs dtype {rhs.dtype}"
            assert lhs.dtype == rhs.dtype, f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"

        if lhs.dtype.is_fp8e4b15() or rhs.dtype.is_fp8e4b15():
            if "fp8e4b15" in self.builder.options.deprecated_fp8_dot_operand_dtypes:
                warnings.warn(
                    "the use of fp8e4b15 is deprecated on Hopper and later architectures and can cause significant slow down. It will be removed in a future triton release"
                )
            # We upcast because there's no fp8e4b15 type in MLIR
            lhs = self.cast(lhs, tl.float16)
            rhs = self.cast(rhs, tl.float16)

        uses_fp8e4b8 = lhs.dtype.is_fp8e4b8() or rhs.dtype.is_fp8e4b8()
        uses_fp8e5b16 = lhs.dtype.is_fp8e5b16() or rhs.dtype.is_fp8e5b16()
        if uses_fp8e4b8 or uses_fp8e5b16:
            type_name = "fp8e4b8" if uses_fp8e4b8 else "fp8e5b16"
            if type_name in self.builder.options.deprecated_fp8_dot_operand_dtypes:
                arch = self.builder.options.arch
                warnings.warn(
                    f"{type_name} is AMD gfx942 specific and not supported on {arch} so it's upcasted to fp16 and can cause significant slow down. "
                    f"Please use OCP fp8 variants on {arch} for performance")
                lhs = self.cast(lhs, tl.float16)
                rhs = self.cast(rhs, tl.float16)

        if input_precision is None:
            input_precision = self.builder.options.default_dot_input_precision

        if out_dtype is None:
            out_dtype = tl.float32 if acc is None else acc.type.element_ty

        input_precision = self._str_to_dot_input_precision(input_precision)

        lhs_rank = len(lhs.shape)
        rhs_rank = len(rhs.shape)
        assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
        assert lhs.shape[-1].value == rhs.shape[
            -2].value, f"First input shape ({lhs.shape}) and second input shape {rhs.shape} are not compatible for matmul (second index of first shape ({lhs.shape[-1].value}) must be equal to first index of second shape ({rhs.shape[-2].value})"
        assert self.builder.codegen_fns.get(
            "min_dot_size") is not None, "target doesn't provide lower shape bounds for dot."
        min_dot_size = self.builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
        assert lhs.shape[-2].value >= min_dot_size[0] and lhs.shape[-1].value >= min_dot_size[2] \
            and rhs.shape[-1].value >= min_dot_size[1], \
                f"Input shapes should have M >= {min_dot_size[0]}, N >= {min_dot_size[1]} and K >= {min_dot_size[2]}"
        if lhs.type.scalar.is_int():
            assert lhs.type.scalar == tl.int8, "only int8 supported!"
            _0 = self.builder.get_int32(0)
            ret_scalar_ty = tl.int32
        elif out_dtype.is_bf16():
            raise ValueError(
                "out_dtype=bfloat16 is unsupported. Please use out_dtype=float32/float16 and cast with `.to(tl.bfloat16)`"
            )
        elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
            _0 = self.builder.get_fp32(0)
            ret_scalar_ty = tl.float32
        elif lhs.type.scalar.is_fp64():
            _0 = self.builder.get_fp64(0)
            ret_scalar_ty = tl.float64
        else:
            _0 = self.builder.get_fp16(0) if out_dtype.is_fp16() else self.builder.get_fp32(0)
            ret_scalar_ty = out_dtype

        M = lhs.type.shape[-2]
        N = rhs.type.shape[-1]
        K = lhs.type.shape[-1]
        B = lhs.type.shape[0] if lhs_rank == 3 else None
        ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
        if acc is None:
            acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
        else:
            acc_handle = acc.handle
            assert acc.type.shape == ret_ty.shape, \
                f"expected accumulator shape {ret_ty.shape}, got {acc.type.shape}"
            assert acc.type.element_ty == out_dtype, \
                f"expected accumulator dtype {out_dtype}, got {acc.type.element_ty}; " \
                f"pass out_dtype={acc.type.element_ty} to use this accumulator dtype"

        # max_num_imprecise_acc only applies to fp8 -> fp32 dot on sm_90
        if max_num_imprecise_acc is None:
            if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
                max_num_imprecise_acc = self.builder.options.max_num_imprecise_acc_default
            else:
                max_num_imprecise_acc = 0
        else:
            if lhs.dtype.is_fp8() and rhs.dtype.is_fp8() and max_num_imprecise_acc > K:
                raise ValueError(f"max_num_imprecise_acc ({max_num_imprecise_acc}) must be <= K ({K})")

        return self.tensor(
            self.builder.create_dot(lhs.handle, rhs.handle, acc_handle, input_precision, max_num_imprecise_acc), ret_ty)

    def _str_to_fp_type(self, float_format: str):
        ty_enum = getattr(ir.ScaleDotElemTypeTY, float_format.upper(), None)
        if ty_enum is None:
            raise ValueError(f"Invalid float format: {float_format}.")
        return ty_enum

    def _bitcast_to_fp_type(self, val: TensorTy, float_format: str):
        """
        If float_format is subbyte, make sure it's packed as uint8 and return it.
        Otherwise, return a tensor (perhaps bitcasting) of the specified float format.
        """
        triton_ty = {"e5m2": tl.float8e5, "e4m3": tl.float8e4nv, "bf16": tl.bfloat16, "fp16":
                     tl.float16}.get(float_format)
        if triton_ty is None:
            assert float_format == "e2m1", f"Internal Error: Unexpected float format: {float_format}"
            assert val.dtype == tl.uint8, f"e2m1 format must be packed as uint8. Got {val.dtype}"
            return val
        if val.dtype == triton_ty:
            return val
        else:
            unsigned_ty = {"e5m2": tl.uint8, "e4m3": tl.uint8, "bf16": tl.uint16, "fp16": tl.uint16}[float_format]
            assert val.dtype == unsigned_ty, f"Unexpected dtype for {float_format}. Got {val.dtype}"
            return self.bitcast(val, triton_ty)

    def deduce_scale_factor(self, lhs, lhs_scale, lhs_format, lhs_k_pack, rhs, rhs_scale, rhs_format, rhs_k_pack):

        def _to_scale_handle(scale):
            if isinstance(scale, tl.tensor) and scale.numel.value != 1:
                return scale.type.shape
            return None

        return ir.deduce_scale_factor(lhs.type.shape, _to_scale_handle(lhs_scale), self._str_to_fp_type(lhs_format),
                                      lhs_k_pack, rhs.type.shape, _to_scale_handle(rhs_scale),
                                      self._str_to_fp_type(rhs_format), rhs_k_pack)

    def verify_scaled_shape(self, M, N, K, lhs_scale, rhs_scale, scale_factor):
        if lhs_scale is not None:
            lhs_scale_shape = lhs_scale.type.shape
            assert lhs_scale_shape[-2:] == [
                M, K // scale_factor
            ], f"lhs_scale must be a tensor of shape [..., {M}, {K // scale_factor}]. Got {lhs_scale_shape}"
        if rhs_scale is not None:
            rhs_scale_shape = rhs_scale.type.shape
            assert rhs_scale_shape[-2:] == [
                N, K // scale_factor
            ], f"rhs_scale must be a tensor of shape [..., {N}, {K // scale_factor}]. Got {rhs_scale_shape}"

    def dot_scaled(self, lhs: TensorTy, lhs_scale: TensorTy, lhs_format: str, rhs: TensorTy,
                   rhs_scale: Optional[TensorTy], rhs_format: str, acc: TensorTy | None, fast_math: bool,
                   lhs_k_pack: bool, rhs_k_pack: bool, out_dtype: tl.dtype) -> TensorTy:
        assert lhs.type.is_block() and rhs.type.is_block(), "dot_scaled operands must be block tensors (not scalars)"
        #TODO: validate types.
        lhs_rank = len(lhs.shape)
        rhs_rank = len(rhs.shape)
        assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3, f"Both inputs must be either 2D or 3D; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
        lhs_format_enum = self._str_to_fp_type(lhs_format)
        rhs_format_enum = self._str_to_fp_type(rhs_format)
        allowed_formats = {"e2m1", "e4m3", "e5m2", "bf16", "fp16"}
        assert lhs_format in allowed_formats, f"NYI: lhs_format {lhs_format}"
        assert rhs_format in allowed_formats, f"NYI: rhs_format {rhs_format}"
        rhs_scale_is_none = rhs_scale is None or (isinstance(rhs_scale, tl.constexpr) and rhs_scale.value is None)
        lhs_scale_is_none = lhs_scale is None or (isinstance(lhs_scale, tl.constexpr) and lhs_scale.value is None)
        lhs = self._bitcast_to_fp_type(lhs, lhs_format)
        rhs = self._bitcast_to_fp_type(rhs, rhs_format)

        assert lhs_k_pack or lhs_format == "e2m1", "only mxfp4 inputs can be packed along a dimension different than K"
        assert rhs_k_pack or rhs_format == "e2m1", "only mxfp4 inputs can be packed along a dimension different than K"
        M, K_LHS = lhs.type.shape[-2:]
        K_RHS, N = rhs.type.shape[-2:]
        PACKED_A = 2 if lhs_format == "e2m1" else 1
        PACKED_B = 2 if rhs_format == "e2m1" else 1
        PACKED_A_DIM = PACKED_A * K_LHS if lhs_k_pack else K_LHS
        PACKED_B_DIM = PACKED_B * K_RHS if rhs_k_pack else K_RHS
        assert PACKED_B_DIM == PACKED_A_DIM, f"Reduction dimension should pack the same number of elements; (lhs: {lhs.shape} vs rhs: {rhs.shape})"
        #assert K * PACKED_B >= 64, f"scaled_dot NYI for K < 64. Got {K=}"
        B = lhs.type.shape[0] if lhs_rank == 3 else None
        K = K_LHS
        if not lhs_k_pack:
            M = M * PACKED_A
        else:
            K = K * PACKED_A
        if not rhs_k_pack:
            N = N * PACKED_B
        ret_ty = tl.block_type(out_dtype, [B, M, N] if B else [M, N])
        _0 = self.builder.get_fp32(0)
        if acc is None:
            acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
        else:
            acc_handle = acc.handle
            assert acc.type.shape == ret_ty.shape, \
                f"expected accumulator shape {ret_ty.shape}, got {acc.type.shape}"
            assert acc.type.element_ty == out_dtype, \
                f"expected accumulator dtype {out_dtype}, got {acc.type.element_ty}; " \
                f"pass out_dtype={acc.type.element_ty} to use this accumulator dtype"
        rhs_scale_handle = None if rhs_scale_is_none else rhs_scale.handle
        lhs_scale_handle = None if lhs_scale_is_none else lhs_scale.handle

        scale_factor = self.deduce_scale_factor(lhs, lhs_scale, lhs_format, lhs_k_pack, rhs, rhs_scale, rhs_format,
                                                rhs_k_pack)
        self.verify_scaled_shape(M, N, K, None if lhs_scale_is_none else lhs_scale,
                                 None if rhs_scale_is_none else rhs_scale, scale_factor)
        return self.tensor(
            self.builder.create_dot_scaled(lhs.handle, lhs_scale_handle, lhs_format_enum, rhs.handle, rhs_scale_handle,
                                           rhs_format_enum, fast_math, lhs_k_pack, rhs_k_pack, acc_handle), ret_ty)

# ===----------------------------------------------------------------------===//
#                               Indexing
# ===----------------------------------------------------------------------===//

    def where(self, condition: TensorTy, x: TensorTy, y: TensorTy) -> TensorTy:
        if condition.dtype != tl.int1:
            warnings.warn(
                f"tl.where with a non-boolean condition is deprecated and will error out in a future triton release. Got {condition.dtype}"
            )
        condition = self.cast(condition, tl.int1)
        x, y = self.binary_op_type_checking_impl(x, y, True, True)
        # x, y are broadcasted
        if condition.type.is_block():
            condition, x = self.broadcast_impl_value(condition, x)
            x, y = self.broadcast_impl_value(x, y)
        else:
            condition, _ = self.broadcast_impl_value(condition, x)
        ret_ty = x.type
        return self.tensor(self.builder.create_select(condition.handle, x.handle, y.handle), ret_ty)

# ===----------------------------------------------------------------------===//
#                               Reduction
# ===----------------------------------------------------------------------===

    def wrap_tensor(self, x, scalar_ty, ret_shape):
        if ret_shape:
            res_ty = tl.block_type(scalar_ty, ret_shape)
        else:
            # 0d-tensor -> scalar
            res_ty = scalar_ty
        return self.tensor(x, res_ty)

    def reduction(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn) -> Tuple[TensorTy, ...]:
        if axis is None:
            inputs = tuple(self.reshape(t, [t.numel.value], can_reorder=True) for t in inputs)
            axis = 0
        # get result shape
        shape = inputs[0].type.shape
        rank = len(shape)
        assert axis < rank, f"reduction axis must be < inputs rank ({rank})"
        ret_shape = [s for i, s in enumerate(shape) if i != axis]
        assert all(t.type.shape == shape for t in inputs), "all reduction inputs must have the same shape"

        reduce_op = self.builder.create_reduce([t.handle for t in inputs], axis)
        region_builder_fn(reduce_op)
        assert reduce_op.verify()

        return tuple(
            self.wrap_tensor(reduce_op.get_result(i), inputs[i].type.scalar, ret_shape) for i in range(len(inputs)))

# ===----------------------------------------------------------------------===
#                               Associative Scan
# ===----------------------------------------------------------------------===

    def associative_scan(self, inputs: Sequence[TensorTy], axis: int, region_builder_fn,
                         reverse: bool) -> Tuple[TensorTy, ...]:
        shape = inputs[0].type.shape
        rank = len(shape)

        assert -rank <= axis < rank, f"scan axis {axis} must be < inputs rank ({rank})"

        if axis < 0:
            axis += rank

        for t in inputs:
            assert t.type.shape == shape, "all scan inputs must have the same shape"

        scan_op = self.builder.create_scan([t.handle for t in inputs], axis, reverse)
        region_builder_fn(scan_op)
        assert scan_op.verify()

        return tuple(self.wrap_tensor(scan_op.get_result(i), inputs[i].type.scalar, shape) for i in range(len(inputs)))

# ===----------------------------------------------------------------------===
#                               Gather
# ===----------------------------------------------------------------------===

    def gather(self, src: TensorTy, index: TensorTy, axis: int) -> TensorTy:
        assert index.dtype.is_int(), "index must be an integer tensor"

        rank = len(src.type.shape)
        assert len(index.type.shape) == rank, "source and index tensors must have the same rank"

        assert -rank <= axis < rank, f"gather axis {axis} must be < source rank ({rank})"
        if axis < 0:
            axis += rank

        for d in range(rank):
            if d == axis:
                continue
            assert index.type.shape[d] == src.type.shape[d], f"index dim {axis} must match the corresponding source dim"

        gather = self.builder.create_gather(src.handle, index.handle, axis)
        return self.wrap_tensor(gather, src.type.scalar, index.type.shape)

# ===----------------------------------------------------------------------===
#                               Map Elementwise
# ===----------------------------------------------------------------------===

    def broadcast_tensors(self, *inputs):
        if not inputs:
            return ()
        head, *tail = inputs
        for i in range(len(tail)):
            head, tail[i] = self.broadcast_impl_value(head, tail[i])
        for i in range(len(tail) - 1):
            head, tail[i] = self.broadcast_impl_value(head, tail[i])
        return (head, *tail)

    def map_elementwise(self, inputs: Sequence[tl.tensor], result_types: Sequence[tl.dtype], pack: int,
                        region_builder_fn) -> Tuple[tl.tensor, ...]:
        inputs = self.broadcast_tensors(*inputs)

        assert len(inputs) > 0, "map_elementwise must have at least 1 input tensor"
        result_types = [inputs[0].type.with_element_ty(ty.scalar) for ty in result_types]
        elementwise_op = self.builder.create_map_elementwise(
            [t.handle for t in inputs],
            [ty.to_ir(self.builder) for ty in result_types],
            pack,
        )
        region_builder_fn(elementwise_op)
        assert elementwise_op.verify()

        return tuple(self.tensor(elementwise_op.get_result(i), ty) for i, ty in enumerate(result_types))


# ===----------------------------------------------------------------------===
#                               Histogram
# ===----------------------------------------------------------------------===

    def histogram(self, input: TensorTy, num_bins: int, mask: Optional[TensorTy]) -> TensorTy:
        assert len(input.shape) == 1, "histogram only supports 1D input"
        assert input.dtype.is_int(), "histogram only supports integer input"
        if mask is not None:
            mask = self.broadcast_impl_shape(mask, input.shape)
            if not mask.type.scalar.is_bool():
                raise ValueError("Mask must have boolean scalar type")
            mask = mask.handle
        return self.tensor(self.builder.create_histogram(input.handle, num_bins, mask),
                           tl.block_type(tl.int32, [num_bins]))

    def multiple_of(self, x: TensorTy, values: List[int]) -> TensorTy:
        if max(1, len(x.shape)) != len(values):
            raise ValueError("Shape of input to multiple_of does not match the length of values")
        x.handle.set_attr("tt.divisibility", ir.make_attr(values, x.handle.get_context()))
        return x

    def max_contiguous(self, x: TensorTy, values: List[int]) -> TensorTy:
        if len(x.shape) != len(values):
            raise ValueError("Shape of input to max_contiguous does not match the length of values")
        x.handle.set_attr("tt.contiguity", ir.make_attr(values, x.handle.get_context()))
        return x

    def max_constancy(self, x: TensorTy, values: List[int]) -> TensorTy:
        if len(x.shape) != len(values):
            raise ValueError("Shape of input to max_constancy does not match the length of values")
        x.handle.set_attr("tt.constancy", ir.make_attr(values, x.handle.get_context()))
        return x

    def debug_barrier(self) -> TensorTy:
        return self.tensor(self.builder.create_barrier(), tl.void)

    def device_print(self, prefix: str, args: List[TensorTy], hex: bool) -> TensorTy:
        # It makes sense visually for prefix to end in ": "; make it so.  Also,
        # non-empty prefixes should start with " ".
        if not prefix.endswith(" ") and args:
            prefix += " "
        if not prefix.endswith(": ") and args:
            prefix = prefix[:-1] + ": "
        if len(prefix) > 2 and not prefix.startswith(" "):
            prefix = " " + prefix

        new_args = [arg.handle for arg in args]
        is_signed = [arg.dtype.is_int_signed() for arg in args]
        return self.tensor(self.builder.create_print(prefix, hex, new_args, is_signed), tl.void)

    def device_assert(self, cond: TensorTy, msg: str, mask: Optional[TensorTy]) -> TensorTy:
        if not self.builder.options.debug:
            return
        if mask is not None:
            cond = self.or_(cond, self.not_(mask))
        return self.tensor(self.builder.create_assert(cond.handle, msg), tl.void)

    def assume(self, cond) -> TensorTy:
        return self.tensor(self.builder.create_assume(cond.handle), tl.void)

    def _convert_elem_to_ir_value(self, elem, require_i64):
        if isinstance(elem, int):
            elem = tl.constexpr(elem)
        if isinstance(elem, tl.constexpr):
            if isinstance(elem.value, bool):
                return self.builder.get_int1(elem.value)
            if require_i64:
                assert -2**63 <= elem.value < 2**63, f"Block pointers only support 64 bit `shape/strides`, " \
                    f"got a value {elem.value} which is out of the range"
                return self.builder.get_int64(elem.value)
            else:
                assert -2**31 <= elem.value < 2**31, f"Block pointers only support 32 bit `offsets/block_shape`, " \
                    f"got a value {elem.value} which is out of the range"
                return self.builder.get_int32(elem.value)
        elif isinstance(elem, tl.tensor):
            assert elem.numel.value == 1, "Expected a scalar in shape/strides/offsets"
            assert elem.dtype.is_int(), "Expected an integer scalar type in shape/strides/offsets"
            if elem.dtype != tl.int64 and require_i64:
                return self.builder.create_int_cast(elem.handle, self.builder.get_int64_ty(),
                                                    elem.dtype.is_int_signed())
            elif elem.dtype == tl.int64 and not require_i64:
                assert False, "Block pointers only support 32 bit `offsets/block_shape`, " \
                    "add a `.to(tl.int32)` or use regular indexing for 64 bit support"
            return elem.handle
        assert False, f"Unsupported element type in shape/strides/offsets: {type(elem)}"

    def _convert_to_ir_values(self, list_like, require_i64=True):
        if hasattr(list_like, "__iter__"):
            return [self._convert_elem_to_ir_value(elem, require_i64) for elem in list_like]
        return [self._convert_elem_to_ir_value(list_like, require_i64)]

    def make_tensor_descriptor(self, base: TensorTy, shape: List[TensorTy], strides: List[TensorTy],
                               block_shape: List[tl.constexpr], padding_option: str = "zero") -> tl.tensor_descriptor:
        ndim = len(shape)
        if not (1 <= ndim <= 5):
            raise ValueError(f"Expected 1 <= ndim <= 5 but got {ndim} dimensions")
        if len(strides) != ndim:
            raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
        if len(block_shape) != ndim:
            raise ValueError(f"Expected block_shape to have {ndim} dimensions but got {len(strides)}")
        assert isinstance(base.dtype, tl.pointer_type), f"base must be a pointer type, got {base.dtype}"
        elem_size = base.dtype.element_ty.primitive_bitwidth // 8
        contig_dim_size = tl._unwrap_if_constexpr(block_shape[-1])
        if contig_dim_size * elem_size < 16:
            raise ValueError(
                f"Descriptor block shape must have at least 16 bytes in the last dimension, but got {contig_dim_size} * {elem_size} = {contig_dim_size * elem_size} bytes"
            )

        last_stride = tl._unwrap_if_constexpr(strides[-1])
        if last_stride != 1:
            raise ValueError(f"Tensor descriptor last dim must be 1 but got {last_stride}")

        shape = [self.make_scalar(x, tl.int32) for x in shape]
        strides = [self.make_scalar(tl._unwrap_if_constexpr(x), tl.int64) for x in strides]

        # Check whether `block_shape` is static
        block_shape = tl._unwrap_shape(block_shape)

        assert isinstance(base.type, tl.pointer_type), f"base must be a pointer type, got {base.type}"
        type = tl.block_type(base.type.element_ty, block_shape)
        base_handle = base.handle
        is_signed_int = base.type.element_ty.is_int_signed()

        padding = self._str_to_padding_option(padding_option)

        if base.type.element_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
            raise ValueError("Padding option `nan` is not supported for integer blocks")

        handle = self.builder.create_make_tensor_descriptor(base_handle, [s.handle for s in shape],
                                                            [s.handle for s in strides], block_shape, is_signed_int,
                                                            padding)
        return tl.tensor_descriptor(handle, shape, strides, type)
<|endoftext|>from triton.runtime import driver
from triton.runtime.jit import constexpr_function

__all__ = ["current_target"]


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True


@constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


@constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.arch == "gfx942"


@constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.arch == "gfx950"


@constexpr_function
def is_hip_gfx1250():
    target = current_target()
    return target is not None and target.arch == "gfx1250"
<|endoftext|>"""isort:skip_file"""
# Import order is significant here.

from . import math
from . import extra
from .standard import (
    argmax,
    argmin,
    bitonic_merge,
    cdiv,
    cumprod,
    cumsum,
    flip,
    interleave,
    max,
    min,
    ravel,
    reduce_or,
    sigmoid,
    softmax,
    sort,
    squeeze,
    sum,
    swizzle2d,
    topk,
    unsqueeze,
    xor_sum,
    zeros,
    zeros_like,
)
from .core import (
    PropagateNan,
    TRITON_MAX_TENSOR_NUMEL,
    aggregate_replace,
    load_tensor_descriptor,
    store_tensor_descriptor,
    make_tensor_descriptor,
    tensor_descriptor,
    tensor_descriptor_type,
    add,
    advance,
    arange,
    associative_scan,
    assume,
    atomic_add,
    atomic_and,
    atomic_cas,
    atomic_max,
    atomic_min,
    atomic_or,
    atomic_xchg,
    atomic_xor,
    bfloat16,
    block_type,
    broadcast,
    broadcast_to,
    cat,
    cast,
    clamp,
    condition,
    const,
    constexpr,
    constexpr_type,
    debug_barrier,
    device_assert,
    device_print,
    dot,
    dot_scaled,
    dtype,
    expand_dims,
    float16,
    float32,
    float64,
    float8e4b15,
    float8e4nv,
    float8e4b8,
    float8e5,
    float8e5b16,
    full,
    gather,
    histogram,
    inline_asm_elementwise,
    int1,
    int16,
    int32,
    int64,
    int8,
    join,
    load,
    make_block_ptr,
    map_elementwise,
    max_constancy,
    max_contiguous,
    maximum,
    minimum,
    mul,
    multiple_of,
    num_programs,
    permute,
    pi32_t,
    pointer_type,
    program_id,
    range,
    reduce,
    reshape,
    slice,
    split,
    static_assert,
    static_print,
    static_range,
    store,
    sub,
    tensor,
    to_tensor,
    trans,
    tuple,
    tuple_type,
    uint16,
    uint32,
    uint64,
    uint8,
    expect_zero,
    view,
    void,
    where,
)
from .math import (umulhi, exp, exp2, fma, log, log2, cos, rsqrt, sin, sqrt, sqrt_rn, abs, fdiv, div_rn, erf, floor,
                   ceil)
from .random import (
    pair_uniform_to_normal,
    philox,
    philox_impl,
    rand,
    rand4x,
    randint,
    randint4x,
    randn,
    randn4x,
    uint_to_uniform_float,
)
from . import target_info

__all__ = [
    "PropagateNan",
    "TRITON_MAX_TENSOR_NUMEL",
    "load_tensor_descriptor",
    "store_tensor_descriptor",
    "make_tensor_descriptor",
    "tensor_descriptor",
    "abs",
    "add",
    "advance",
    "arange",
    "argmax",
    "argmin",
    "associative_scan",
    "assume",
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
    "bfloat16",
    "bitonic_merge",
    "block_type",
    "broadcast",
    "broadcast_to",
    "cat",
    "cast",
    "cdiv",
    "ceil",
    "clamp",
    "condition",
    "const",
    "constexpr",
    "constexpr_type",
    "cos",
    "cumprod",
    "cumsum",
    "debug_barrier",
    "device_assert",
    "device_print",
    "div_rn",
    "dot",
    "dot_scaled",
    "dtype",
    "erf",
    "exp",
    "exp2",
    "expand_dims",
    "extra",
    "fdiv",
    "flip",
    "float16",
    "float32",
    "float64",
    "float8e4b15",
    "float8e4nv",
    "float8e4b8",
    "float8e5",
    "float8e5b16",
    "floor",
    "fma",
    "full",
    "gather",
    "histogram",
    "inline_asm_elementwise",
    "interleave",
    "int1",
    "int16",
    "int32",
    "int64",
    "int8",
    "join",
    "load",
    "log",
    "log2",
    "make_block_ptr",
    "map_elementwise",
    "math",
    "max",
    "max_constancy",
    "max_contiguous",
    "maximum",
    "min",
    "minimum",
    "mul",
    "multiple_of",
    "num_programs",
    "pair_uniform_to_normal",
    "permute",
    "philox",
    "philox_impl",
    "pi32_t",
    "pointer_type",
    "program_id",
    "rand",
    "rand4x",
    "randint",
    "randint4x",
    "randn",
    "randn4x",
    "range",
    "ravel",
    "reduce",
    "reduce_or",
    "reshape",
    "rsqrt",
    "slice",
    "sigmoid",
    "sin",
    "softmax",
    "sort",
    "split",
    "sqrt",
    "sqrt_rn",
    "squeeze",
    "static_assert",
    "static_print",
    "static_range",
    "store",
    "sub",
    "sum",
    "swizzle2d",
    "target_info",
    "tensor",
    "topk",
    "to_tensor",
    "trans",
    "tuple",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "expect_zero",
    "uint_to_uniform_float",
    "umulhi",
    "unsqueeze",
    "view",
    "void",
    "where",
    "xor_sum",
    "zeros",
    "zeros_like",
]


def str_to_ty(name, c):
    from builtins import tuple

    if isinstance(name, tuple):
        fields = type(name).__dict__.get("_fields", None)
        return tuple_type([str_to_ty(x, c) for x in name], fields)

    if name[0] == "*":
        name = name[1:]
        const = False
        if name[0] == "k":
            name = name[1:]
            const = True
        ty = str_to_ty(name, c)
        return pointer_type(element_ty=ty, const=const)

    if name.startswith("tensordesc"):
        # Determine mode from type name: tensordesc_im2col vs tensordesc
        is_im2col = name.startswith("tensordesc_im2col")

        inner = name.split("<")[1].rstrip(">")
        dtype, rest = inner.split("[", maxsplit=1)
        block_shape, rest = rest.split("]", maxsplit=1)
        block_shape = [int(s.strip()) for s in block_shape.rstrip("]").split(",")]
        # For im2col, parse optional input_rank=N (e.g., ",input_rank=4,layout")
        tensor_rank = None
        import re as _re
        rank_match = _re.search(r",input_rank=(\d+)", rest)
        if rank_match:
            tensor_rank = int(rank_match.group(1))
            rest = rest[:rank_match.start()] + rest[rank_match.end():]
        layout_str = rest.lstrip(",")
        is_gluon = len(layout_str)
        dtype = str_to_ty(dtype, None)
        # For im2col with tensor_rank, use it for shape/stride types; otherwise use block_shape ndim
        ndim = tensor_rank if (is_im2col and tensor_rank is not None) else len(block_shape)
        shape_type = tuple_type([int32] * ndim)
        # FIXME: Last dim stride should be constexpr(1)
        stride_type = tuple_type(([int64] * ndim))
        block = block_type(dtype, block_shape)
        if is_gluon:
            from triton.experimental.gluon.language._layouts import NVMMASharedLayout, PaddedSharedLayout, SwizzledSharedLayout
            from triton.experimental.gluon.language.nvidia.hopper.tma import tensor_descriptor_type as nvidia_tensor_descriptor_type
            from triton.experimental.gluon.language.nvidia.hopper.tma import tensor_descriptor_im2col_type as nvidia_tensor_descriptor_im2col_type
            from triton.experimental.gluon.language.amd.gfx1250.tdm import tensor_descriptor_type as amd_tensor_descriptor_type
            layout = eval(
                layout_str,
                dict(NVMMASharedLayout=NVMMASharedLayout, PaddedSharedLayout=PaddedSharedLayout,
                     SwizzledSharedLayout=SwizzledSharedLayout))
            if isinstance(layout, NVMMASharedLayout):
                if is_im2col:
                    return nvidia_tensor_descriptor_im2col_type(block, shape_type, stride_type, layout)
                return nvidia_tensor_descriptor_type(block, shape_type, stride_type, layout)
            else:
                return amd_tensor_descriptor_type(block, shape_type, stride_type, layout)
        return tensor_descriptor_type(block, shape_type, stride_type)

    if name.startswith("constexpr"):
        return constexpr_type(c)

    tys = {
        "fp8e4nv": float8e4nv,
        "fp8e4b8": float8e4b8,
        "fp8e5": float8e5,
        "fp8e5b16": float8e5b16,
        "fp8e4b15": float8e4b15,
        "fp16": float16,
        "bf16": bfloat16,
        "fp32": float32,
        "fp64": float64,
        "i1": int1,
        "i8": int8,
        "i16": int16,
        "i32": int32,
        "i64": int64,
        "u1": int1,
        "u8": uint8,
        "u16": uint16,
        "u32": uint32,
        "u64": uint64,
        "B": int1,
    }
    return tys[name]
<|endoftext|>from . import core
from functools import wraps
from typing import List

T = core.TypeVar('T')


def _check_dtype(dtypes: List[str]) -> T:
    """
    We're following libdevice's convention to check accepted data types for math functions.
    It is not a good practice to support all data types as accelerators/GPUs don't support
    many float16 and bfloat16 math operations.
    We should let the users know that they are using and invoke explicit cast to convert
    the data type to the supported one.
    """

    def wrapper(fn):

        @wraps(fn)
        def check(*args, **kwargs):
            # concatenate args and kwargs
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, core.tensor)]:
                if arg.type.scalar.name not in dtypes:
                    raise ValueError(f"Expected dtype {dtypes} but got {arg.type.scalar.name}")
            return fn(*args, **kwargs)

        return check

    return wrapper


def _add_math_1arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`.

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _add_math_2arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x` and :code:`y`.

    :param x: the input values
    :type x: Block
    :param y: the input values
    :type y: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _add_math_3arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`, :code:`y`, and :code:`z`.

    :param x: the input values
    :type x: Block
    :param y: the input values
    :type y: Block
    :param z: the input values
    :type z: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@core.builtin
@_check_dtype(dtypes=["int32", "int64", "uint32", "uint64"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _semantic=None):
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x, y = core.binary_op_type_legalization(x, y, _semantic)
    return core.tensor(_semantic.builder.create_umulhi(x.handle, y.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential")
@core._tensor_member_fn
def exp(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_exp(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential (base 2)")
@core._tensor_member_fn
def exp2(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_exp2(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("natural logarithm")
@core._tensor_member_fn
def log(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_log(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("logarithm (base 2)")
@core._tensor_member_fn
def log2(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_log2(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("cosine")
@core._tensor_member_fn
def cos(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_cos(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("sine")
@core._tensor_member_fn
def sin(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_sin(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("fast square root")
@core._tensor_member_fn
def sqrt(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_sqrt(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_1arg_docstr("precise square root (rounding to nearest wrt the IEEE standard)")
@core._tensor_member_fn
def sqrt_rn(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_precise_sqrt(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("inverse square root")
@core._tensor_member_fn
def rsqrt(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_rsqrt(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_add_math_1arg_docstr("absolute value")
def abs(x, _semantic=None):
    x = _semantic.to_tensor(x)
    dtype = x.dtype
    if dtype.is_fp8e4b15():
        mask = core.full(x.shape, 0x7F, core.int8, _semantic=_semantic)
        return core.tensor(_semantic.builder.create_and(x.handle, mask.handle), x.type)
    elif dtype.is_floating():
        return core.tensor(_semantic.builder.create_fabs(x.handle), x.type)
    elif dtype.is_int_signed():
        return core.tensor(_semantic.builder.create_iabs(x.handle), x.type)
    elif dtype.is_int_unsigned():
        return x  # no-op
    else:
        assert False, f"Unexpected dtype {dtype}"


@core.builtin
@_add_math_2arg_docstr("fast division")
def fdiv(x, y, ieee_rounding=False, _semantic=None):
    ieee_rounding = core._unwrap_if_constexpr(ieee_rounding)
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    return _semantic.fdiv(x, y, ieee_rounding)


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_2arg_docstr("precise division (rounding to nearest wrt the IEEE standard)")
def div_rn(x, y, _semantic=None):
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x, y = core.binary_op_type_legalization(x, y, _semantic)
    return core.tensor(_semantic.builder.create_precise_divf(x.handle, y.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def erf(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_erf(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("floor")
@core._tensor_member_fn
def floor(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_floor(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("ceil")
@core._tensor_member_fn
def ceil(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_ceil(x.handle), x.type)


@core.builtin
@_add_math_3arg_docstr("fused multiply-add")
def fma(x, y, z, _semantic=None):
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    z = _semantic.to_tensor(z)
    x, y = core.binary_op_type_legalization(x, y, _semantic)
    z, x = core.binary_op_type_legalization(z, x, _semantic)
    z, y = core.binary_op_type_legalization(z, y, _semantic)
    return core.tensor(_semantic.builder.create_fma(x.handle, y.handle, z.handle), x.type)
<|endoftext|>from __future__ import annotations

import math
from warnings import warn
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps, cached_property
import typing
from typing import Union, Callable, List, Sequence, TypeVar, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import builtins
from .. import knobs
from ..runtime.jit import JITCallable
import inspect

from .._C.libtriton import ir
from .._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape, get_primitive_bitwidth, _tuple_create

T = TypeVar('T')

TRITON_BUILTIN = "__triton_builtin__"

PropagateNan = ir.PROPAGATE_NAN


def must_use_result(x, s=True):
    """If the result of this function is unused, throw an error."""
    if isinstance(x, str):
        return (lambda fn: must_use_result(fn, x))
    x._must_use_result = s
    return x


def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)
    wrapper.signature = inspect.signature(fn)

    return wrapper


def _tensor_member_fn(fn: T) -> T:
    """Decorator that adds this free function as a member fn on class tensor.

    When called as a member function on class tensor, the first argument to `fn`
    is `self`, i.e. the tensor object.

    If there are multiple decorators on a function, you probably want this one
    to be the highest one (i.e. furthest from the function's `def`), so it's
    applied last.

    Unfortunately you still need to add a type stub to the body of class tensor
    in order for pytype to know about it.
    """
    assert callable(fn)
    orig_sig = inspect.signature(fn)
    # Does fn take args other than _semantic, _generator, and the tensor itself?
    has_args = len(orig_sig.parameters.keys() - {"_semantic", "_generator"}) > 1

    if not fn.__doc__:
        fn.__doc__ = ""
    fn.__doc__ += f"""
    This function can also be called as a member function on :py:class:`tensor`,
    as :code:`x.{fn.__name__}({"..." if has_args else ""})` instead of
    :code:`{fn.__name__}(x{", ..." if has_args else ""})`.
    """

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    # Match the signature of `fn`, but change the first arg to `self` so the
    # docs are a little less weird.
    new_params = list(orig_sig.parameters.values())
    new_params[0] = new_params[0].replace(name='self')
    new_sig = orig_sig.replace(parameters=new_params)
    wrapper.__signature__ = new_sig
    wrapper.signature = new_sig
    wrapper.__doc__ = f"Forwards to :py:func:`{fn.__name__}` free function"
    # If fn is a builtin, mark the wrapper as a builtin too.
    if is_builtin(fn):
        setattr(wrapper, TRITON_BUILTIN, True)

    setattr(tensor, fn.__name__, fn if isinstance(fn, JITCallable) else wrapper)
    return fn


def _unwrap_iterable(x):
    """Returns x[0] if x has one element and x[0] is iterable."""
    x = _unwrap_if_constexpr(x)
    if len(x) == 1:
        # Determine whether x[0] is iterable.
        #
        # You might want to use collections.abc.Iterable instead of this
        # try/except block.  Unfortunately, this doesn't work with constexpr.
        #
        # The problem is that abc.Iterable checks for __iter__ on the *class*.
        # But we want constexpr to expose an __iter__ method if and only if the
        # wrapped *object* (i.e. self.value) is iterable.  Therefore there's no
        # right answer for whether the class constexpr defines __iter__, and
        # abc.Iterable doesn't work (at least not without some metaclass magic).
        try:
            iter(x[0])
            return x[0]
        except TypeError:
            pass

    return x


def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


@builtin
def to_tensor(x, _semantic=None):
    return _semantic.to_tensor(x)


# -----------------------
# constexpr
# -----------------------


class const:
    """
    This class is used as a type annotation to mark pointers to constant data.
    The `store` function cannot be called with a pointer to const. Constness
    is part of the pointer type and the usual Triton type consistency rules
    apply. For example you cannot have a function that returns constant pointer
    in one return statement and non-constant pointer in another.
    """
    pass


class base_value:
    """Base class of values that exist in the triton IR (i.e. not constexprs).
    """
    type: base_type

    def _set_name(self, builder: ir.builder, name: str) -> None:
        raise NotImplementedError

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        """Flatten frontend value into a sequence of mlir handles, which are appended
        to the output list
        """
        raise NotImplementedError


class base_type:

    def __eq__(self, other) -> bool:
        raise NotImplementedError("Types must implement __eq__")

    def __ne__(self, other) -> bool:
        return not (self == other)

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        """Build a frontend value with the current dtype, wrapping a list of existing handles.
        cursor is the index of the first handle relevant to this value, and the function
        should return the updated cursor position after any handles consumed by the created value.
        """
        raise NotImplementedError

    def mangle(self) -> str:
        raise NotImplementedError(f"NYI: Type mangling for type {self.__class__}")

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        raise NotImplementedError


class constexpr_type(base_type):

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, constexpr_type) and self.value == other.value

    def __repr__(self) -> str:
        return f"constexpr_type[{self.value}]"

    def __hash__(self):
        return hash(self.value)

    def mangle(self) -> str:
        if hasattr(self.value, "mangle"):
            val = self.value.mangle()
        else:
            val = repr(self.value)
        return f"c{val}"

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        return

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        return constexpr(self.value), cursor


class constexpr(base_value):
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        while isinstance(value, constexpr):
            value = value.value
        self.value = value
        self.type = constexpr_type(value)

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __hash__(self):
        return hash((self.value, self.type))

    def _set_name(self, builder: ir.builder, name: str) -> None:
        return

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        return

    def __index__(self):
        return self.value

    # In interpreter mode, constant values are not wrapped in constexpr,
    # and therefore do not have a .value attribute.
    # As a result, from here and below, we need to call the _unwrap_if_constexpr
    # function to obtain either constexpr.value or the value itself.
    def __add__(self, other):
        return constexpr(self.value + _unwrap_if_constexpr(other))

    def __radd__(self, other):
        return constexpr(_unwrap_if_constexpr(other) + self.value)

    def __sub__(self, other):
        return constexpr(self.value - _unwrap_if_constexpr(other))

    def __rsub__(self, other):
        return constexpr(_unwrap_if_constexpr(other) - self.value)

    def __mul__(self, other):
        return constexpr(self.value * _unwrap_if_constexpr(other))

    def __mod__(self, other):
        return constexpr(self.value % _unwrap_if_constexpr(other))

    def __rmul__(self, other):
        return constexpr(_unwrap_if_constexpr(other) * self.value)

    def __truediv__(self, other):
        return constexpr(self.value / _unwrap_if_constexpr(other))

    def __rtruediv__(self, other):
        return constexpr(_unwrap_if_constexpr(other) / self.value)

    def __floordiv__(self, other):
        return constexpr(self.value // _unwrap_if_constexpr(other))

    def __rfloordiv__(self, other):
        return constexpr(_unwrap_if_constexpr(other) // self.value)

    def __gt__(self, other):
        return constexpr(self.value > _unwrap_if_constexpr(other))

    def __rgt__(self, other):
        return constexpr(_unwrap_if_constexpr(other) > self.value)

    def __ge__(self, other):
        return constexpr(self.value >= _unwrap_if_constexpr(other))

    def __rge__(self, other):
        return constexpr(_unwrap_if_constexpr(other) >= self.value)

    def __lt__(self, other):
        return constexpr(self.value < _unwrap_if_constexpr(other))

    def __rlt__(self, other):
        return constexpr(_unwrap_if_constexpr(other) < self.value)

    def __le__(self, other):
        return constexpr(self.value <= _unwrap_if_constexpr(other))

    def __rle__(self, other):
        return constexpr(_unwrap_if_constexpr(other) <= self.value)

    def __eq__(self, other):
        return constexpr(self.value == _unwrap_if_constexpr(other))

    def __ne__(self, other):
        return constexpr(self.value != _unwrap_if_constexpr(other))

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return constexpr(-self.value)

    def __and__(self, other):
        return constexpr(self.value & _unwrap_if_constexpr(other))

    def logical_and(self, other):
        return constexpr(self.value and _unwrap_if_constexpr(other))

    def __or__(self, other):
        return constexpr(self.value | _unwrap_if_constexpr(other))

    def __xor__(self, other):
        return constexpr(self.value ^ _unwrap_if_constexpr(other))

    def logical_or(self, other):
        return constexpr(self.value or _unwrap_if_constexpr(other))

    def __pos__(self):
        return constexpr(+self.value)

    def __invert__(self):
        return constexpr(~self.value)

    def __pow__(self, other):
        return constexpr(self.value**_unwrap_if_constexpr(other))

    def __rpow__(self, other):
        return constexpr(_unwrap_if_constexpr(other)**self.value)

    def __rshift__(self, other):
        return constexpr(self.value >> _unwrap_if_constexpr(other))

    def __lshift__(self, other):
        return constexpr(self.value << _unwrap_if_constexpr(other))

    def __not__(self):
        return constexpr(not self.value)

    def __iter__(self):
        return iter(self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)

    def __getitem__(self, *args):
        args = (_unwrap_if_constexpr(x) for x in _normalize_tuple(args))
        return self.value.__getitem__(*args)


CONSTEXPR_0 = constexpr(0)


def _unwrap_if_constexpr(o):
    if isinstance(o, list):
        return [_unwrap_if_constexpr(x) for x in o]
    if isinstance(o, builtins.tuple):
        return _tuple_create(o, [_unwrap_if_constexpr(x) for x in o])
    if isinstance(o, tuple):
        return tuple([_unwrap_if_constexpr(x) for x in o], o.type)
    return o.value if isinstance(o, constexpr) else o


def _normalize_tuple(t):
    normalized_tuple = _unwrap_if_constexpr(t)
    if isinstance(normalized_tuple, (list, builtins.tuple)):
        normalized_tuple = tuple(normalized_tuple)
    return normalized_tuple


def check_bit_width(value, shift_value):
    if isinstance(value, tensor) and isinstance(shift_value, constexpr):
        bitwidth = value.type.scalar.primitive_bitwidth
        if shift_value.value >= bitwidth:
            warn(
                f"Value {shift_value.value} exceeds the maximum bitwidth ({bitwidth}) for type '{value.dtype}'. This may result in undefined behavior."
            )


# -----------------------
# dtype
# -----------------------


class dtype(base_type):
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    class KIND(Enum):
        BOOLEAN = 0
        INTEGRAL = 1
        FLOATING = 2

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        self.primitive_bitwidth = get_primitive_bitwidth(name)
        self.itemsize = self.primitive_bitwidth // 8
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = self.primitive_bitwidth
        elif name in dtype.FP_TYPES:
            if name == 'fp8e4b15':
                self.fp_mantissa_width = 3
                self.exponent_bias = 15
            elif name == 'fp8e4nv':
                self.fp_mantissa_width = 3
                self.exponent_bias = 7
            elif name == 'fp8e4b8':
                self.fp_mantissa_width = 3
                self.exponent_bias = 8
            elif name == 'fp8e5':
                self.fp_mantissa_width = 2
                self.exponent_bias = 15
            elif name == 'fp8e5b16':
                self.fp_mantissa_width = 2
                self.exponent_bias = 16
            elif name == 'fp16':
                self.fp_mantissa_width = 10
                self.exponent_bias = 15
            elif name == 'bf16':
                self.fp_mantissa_width = 7
                self.exponent_bias = 127
            elif name == 'fp32':
                self.fp_mantissa_width = 23
                self.exponent_bias = 127
            elif name == 'fp64':
                self.fp_mantissa_width = 52
                self.exponent_bias = 1023
            else:
                raise RuntimeError(f'Unsupported floating-point type {name}')

    def is_fp8(self):
        return 'fp8' in self.name

    def is_fp8e4nv(self):
        return self.name == 'fp8e4nv'

    def is_fp8e4b8(self):
        return self.name == 'fp8e4b8'

    def is_fp8e4b15(self):
        return self.name == 'fp8e4b15'

    def is_fp8e5(self):
        return self.name == 'fp8e5'

    def is_fp8e5b16(self):
        return self.name == 'fp8e5b16'

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def is_floating(self):
        return self.name in dtype.FP_TYPES

    def is_standard_floating(self):
        return self.name in dtype.STANDARD_FP_TYPES

    def is_int_signed(self):
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):
        return self.name in dtype.UINT_TYPES

    def is_int(self):
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        return self.is_int1()

    def kind(self):
        # Return int value following the type ordering bool < integer < fp
        if self.is_bool():
            return dtype.KIND.BOOLEAN
        elif self.is_int():
            return dtype.KIND.INTEGRAL
        else:
            assert self.is_floating()
            return dtype.KIND.FLOATING

    def get_int_max_value(self):
        if self.is_int_signed():
            return 2**(self.int_bitwidth - 1) - 1
        if self.is_int_unsigned():
            return 2**self.int_bitwidth - 1
        assert False

    def get_int_min_value(self):
        if self.is_int_signed():
            return -2**(self.int_bitwidth - 1)
        if self.is_int_unsigned():
            return 0
        assert False

    @staticmethod
    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES

    @staticmethod
    def is_void():
        raise RuntimeError("Not implemented")

    @staticmethod
    def is_block():
        return False

    @staticmethod
    def is_ptr():
        return False

    @staticmethod
    def is_const():
        return False

    def __eq__(self, other) -> bool:
        other = _unwrap_if_constexpr(other)
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash((self.name, ))

    @property
    def scalar(self):
        return self

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name.startswith("fp8"):
            if hasattr(builder, "options") and self.name not in builder.options.supported_fp8_dtypes:
                raise ValueError(f'type {self} not supported in this architecture. '
                                 f'The supported fp8 dtypes are {builder.options.supported_fp8_dtypes}')

        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name in ('int8', 'uint8'):
            return builder.get_int8_ty()
        elif self.name in ('int16', 'uint16'):
            return builder.get_int16_ty()
        elif self.name in ('int32', 'uint32'):
            return builder.get_int32_ty()
        elif self.name in ('int64', 'uint64'):
            return builder.get_int64_ty()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_ty()
        elif self.name == 'fp8e5b16':
            return builder.get_fp8e5b16_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b8':
            return builder.get_fp8e4b8_ty()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    def __str__(self):
        return self.name

    def codegen_name(self):
        if self.name.startswith("fp"):
            return "float" + self.name[2:]
        elif self.name.startswith("bf"):
            return "bfloat" + self.name[2:]
        else:
            return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        return tensor(handles[cursor], self), cursor + 1

    def mangle(self) -> str:
        if self.is_int():
            SIGNED = dtype.SIGNEDNESS.SIGNED
            prefix = 'i' if self.int_signedness == SIGNED else 'u'
            return prefix + str(self.int_bitwidth)
        if self.is_floating():
            return str(self)
        if self.is_void():
            return 'V'
        return super().mangle()

    def with_element_ty(self, element_ty: dtype):
        assert not self.is_block()
        return element_ty


# Some functions have a param named `dtype`, which shadows the `dtype` class.
# We can't change the param name because it is part of function's public API.
# Declare an alias so those functions can still reference the dtype class.
_DtypeClass = dtype


class pointer_type(dtype):

    def __init__(self, element_ty: dtype, address_space: int = 1, const: bool = False):
        element_ty = _unwrap_if_constexpr(element_ty)
        if not isinstance(element_ty, dtype):
            raise TypeError(f'element_ty has type `{type(element_ty).__name__}`; expected `dtype`.')
        self.element_ty = element_ty
        self.address_space = address_space
        self.const = const
        self.name = f'pointer<{element_ty}>' if not const else f'const_pointer<{element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return builder.get_ptr_ty(self.element_ty.to_ir(builder), self.address_space)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def is_const(self):
        return self.const

    def __eq__(self, other) -> bool:
        other = _unwrap_if_constexpr(other)
        if not isinstance(other, pointer_type):
            return False
        return self.element_ty == other.element_ty and self.address_space == other.address_space and self.const == other.const

    @property
    def scalar(self):
        return self

    def mangle(self) -> str:
        return f"P{self.element_ty.mangle()}"


class block_type(dtype):

    def __init__(self, element_ty: dtype, shape: List):
        self.element_ty = element_ty

        # Note that block_type's shape is a list of int
        # while tensor's shape is a list of constexpr.
        assert (isinstance(shape, (list, tuple)))

        # shape can be empty ([]) when an input is a 0D tensor.
        self.shape = tuple(_unwrap_shape(shape))
        if not self.shape:
            raise TypeError('0d block_type is forbidden')

        self.numel = validate_block_shape(self.shape)
        self.name = f'<{self.shape}, {self.element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shapes(self) -> Tuple[int]:
        return self.shape

    def with_element_ty(self, scalar_ty: dtype) -> block_type:
        return block_type(scalar_ty, self.shape)

    def __eq__(self, other) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    @property
    def scalar(self):
        return self.element_ty

    @property
    def nbytes(self):
        return self.numel * (self.element_ty.primitive_bitwidth // 8)

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        return f'{elt}S{shape}S'


class tuple_type(base_type):

    def __init__(self, types, fields=None):
        self.types = types
        self.fields = fields

    @cached_property
    def name(self):
        if self.fields is None:
            return '[' + ','.join(str(v) for v in self.types) + ']'
        return '[' + ','.join([f"{k}:{v}" for k, v in zip(self.fields, self.types)]) + ']'

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.types)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]):
        for ty in self.types:
            ty._flatten_ir_types(builder, out)

    def __getitem__(self, index: int) -> dtype:
        return self.types[index]

    def __eq__(self, other):
        return type(self) is type(other) and self.types == other.types and self.fields == other.fields

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tuple, int]:
        values = []
        for ty in self.types:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values.append(value)
        return tuple(values, self), cursor

    def mangle(self):
        return 'T' + '_'.join(ty.mangle() for ty in self.types) + 'T'


class slice_type(dtype):

    def __init__(self):
        self.name = 'slice_type'


# scalar types
void = dtype('void')
int1 = dtype('int1')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float8e5 = dtype('fp8e5')
float8e5b16 = dtype('fp8e5b16')
float8e4nv = dtype('fp8e4nv')
float8e4b8 = dtype('fp8e4b8')
float8e4b15 = dtype('fp8e4b15')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')
# pointer types
pi32_t = pointer_type(int32)


def get_int_dtype(bitwidth: int, signed: bool) -> dtype:
    if bitwidth == 1:
        return int1
    elif bitwidth == 8 and signed:
        return int8
    elif bitwidth == 8 and not signed:
        return uint8
    elif bitwidth == 16 and signed:
        return int16
    elif bitwidth == 16 and not signed:
        return uint16
    elif bitwidth == 32 and signed:
        return int32
    elif bitwidth == 32 and not signed:
        return uint32
    elif bitwidth == 64 and signed:
        return int64
    elif bitwidth == 64 and not signed:
        return uint64
    else:
        raise ValueError(f'Unsupported bitwidth {bitwidth} and signedness {signed}')


# -----------------------
# tensor
# -----------------------


class tensor(base_value):
    """Represents an N-dimensional array of values or pointers.

    :code:`tensor` is the fundamental data structure in Triton programs.  Most
    functions in :py:mod:`triton.language` operate on and return tensors.

    Most of the named member functions here are duplicates of the free functions
    in :code:`triton.language`.  For example, :code:`triton.language.sqrt(x)` is
    equivalent to :code:`x.sqrt()`.

    :code:`tensor` also defines most of the magic/dunder methods, so you can
    write :code:`x+y`, :code:`x << 2`, etc.

    .. rubric:: Constructors
    ..
       For some reason Sphinx includes __init__ before printing the full table
       of methods.  Not what I want, but I can't figure out how to fix it.  Give
       it its own section so it looks intentional. :)
    """

    def __init__(self, handle, type: dtype):
        """Not called by user code."""
        super().__init__()
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = type.shape if type.is_block() else ()
        self.numel = constexpr(math.prod(self.shape))
        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar
        self.shape = tuple([constexpr(s) for s in self.shape])

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    def __str__(self) -> str:
        # ex. "float32[16, 32]"
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

    @builtin
    def __add__(self, other, _semantic=None):
        return add(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __radd__(self, other, _semantic=None):
        return add(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __sub__(self, other, _semantic=None):
        return sub(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __rsub__(self, other, _semantic=None):
        return sub(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __mul__(self, other, _semantic=None):
        return mul(self, other, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __rmul__(self, other, _semantic=None):
        return mul(other, self, sanitize_overflow=True, _semantic=_semantic)

    @builtin
    def __truediv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.truediv(self, other)

    @builtin
    def __rtruediv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.truediv(other, self)

    @builtin
    def __floordiv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.floordiv(self, other)

    @builtin
    def __rfloordiv__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.floordiv(other, self)

    @builtin
    def __mod__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.mod(self, other)

    @builtin
    def __rmod__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.mod(other, self)

    # unary operators
    @builtin
    def __neg__(self, _semantic=None):
        return _semantic.minus(self)

    @builtin
    def __invert__(self, _semantic=None):
        return _semantic.invert(self)

    # bitwise operators

    @builtin
    def __and__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.and_(self, other)

    @builtin
    def __rand__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.and_(other, self)

    @builtin
    def __or__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.or_(self, other)

    @builtin
    def __ror__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.or_(other, self)

    @builtin
    def __xor__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.xor_(self, other)

    @builtin
    def __rxor__(self, other, _semantic=None):
        other = _unwrap_if_constexpr(other)
        return _semantic.xor_(other, self)

    @builtin
    def __lshift__(self, other, _semantic=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        return _semantic.shl(self, other)

    @builtin
    def __rlshift__(self, other, _semantic=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        return _semantic.shl(other, self)

    @builtin
    def __rshift__(self, other, _semantic=None):
        check_bit_width(self, other)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return _semantic.ashr(self, other)
        else:
            return _semantic.lshr(self, other)

    @builtin
    def __rrshift__(self, other, _semantic=None):
        check_bit_width(other, self)
        other = _unwrap_if_constexpr(other)
        if self.dtype.is_int_signed():
            return _semantic.ashr(other, self)
        else:
            return _semantic.lshr(other, self)

    # >
    @builtin
    def __gt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_than(self, other)

    @builtin
    def __rgt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_than(other, self)

    # >=
    @builtin
    def __ge__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_equal(self, other)

    @builtin
    def __rge__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.greater_equal(other, self)

    # <
    @builtin
    def __lt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_than(self, other)

    @builtin
    def __rlt__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_than(other, self)

    # <=
    @builtin
    def __le__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_equal(self, other)

    @builtin
    def __rle__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.less_equal(other, self)

    # ==
    @builtin
    def __eq__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.equal(self, other)

    @builtin
    def __req__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.equal(other, self)

    @builtin
    def __ne__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.not_equal(self, other)

    @builtin
    def __rne__(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.not_equal(other, self)

    @builtin
    def logical_and(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.logical_and(self, other)

    @builtin
    def logical_or(self, other, _semantic=None):
        other = _semantic.to_tensor(other)
        return _semantic.logical_or(self, other)

    # note: __not__ isn't actually a magic method in python
    # but it's ok because our ASTVisitor handles it
    @builtin
    def __not__(self, _semantic=None):
        return _semantic.not_(self)

    @builtin
    def __getitem__(self, slices, _semantic=None):
        if isinstance(slices, (builtins.slice, slice, constexpr)) or slices is None:
            slices = [slices]
        if isinstance(slices, tuple):
            slices = slices.values
        ret = self
        for dim, sl in enumerate(slices):
            if _unwrap_if_constexpr(sl) is None:
                ret = _semantic.expand_dims(ret, dim)
            elif isinstance(sl, (builtins.slice, slice)) and all(
                    _unwrap_if_constexpr(arg) is None for arg in (sl.start, sl.stop, sl.step)):
                pass  # an unsqueeze
            else:
                raise ValueError(f"unsupported tensor index: {sl}")
        return ret

    @property
    def T(self):
        """Transposes a 2D tensor."""
        assert False, "Transposition must be created by the AST Visitor"

    @builtin
    def to(self, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _semantic=None):
        """
        Alias for :py:func:`tensor.cast`.
        """
        return cast(self, dtype, fp_downcast_rounding, bitcast, _semantic=_semantic)

    # Type stubs for functions added by the _tensor_member_fn decorator.
    # (Unfortunately these can't be created automatically.)
    #
    # We couldn't write these definitions out even if we wanted to, because some
    # of these functions are defined in standard.py.
    def broadcast_to(self, *shape) -> tensor:
        ...

    def trans(self, *dims) -> tensor:
        ...

    def permute(self, *dims) -> tensor:
        ...

    def split(self) -> tuple[tensor, tensor]:
        ...

    def view(self, *shape) -> tensor:
        ...

    def reshape(self, *shape) -> tensor:
        ...

    def expand_dims(self, axis) -> tensor:
        ...

    def cast(self, dtype, fp_downcast_rounding=None, bitcast=False) -> tensor:
        ...

    def store(self, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="") -> tensor:
        ...

    def advance(self, offsets) -> tensor:
        ...

    def atomic_cas(self, cmp, val, sem=None, scope=None) -> tensor:
        ...

    def atomic_xchg(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_add(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_max(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_min(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_and(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_or(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_xor(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def exp(self) -> tensor:
        ...

    def log(self) -> tensor:
        ...

    def cos(self) -> tensor:
        ...

    def sin(self) -> tensor:
        ...

    def sqrt(self) -> tensor:
        ...

    def rsqrt(self) -> tensor:
        ...

    def abs(self) -> tensor:
        ...

    def reduce(self, axis, combine_fn, keep_dims=False) -> tensor:
        ...

    def associative_scan(self, axis, combine_fn, reverse=False) -> tensor:
        ...

    def gather(self, indices, axis) -> tensor:
        ...

    def histogram(self, num_bins) -> tensor:
        ...

    def cdiv(self, div) -> tensor:
        ...

    def sigmoid(self) -> tensor:
        ...

    def softmax(self, dim=None, keep_dims=False, ieee_rounding=False) -> tensor:
        ...

    def ravel(self) -> tensor:
        ...

    def max(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmax(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def min(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmin(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def sum(self, axis=None, keep_dims=False, dtype=None) -> tensor:
        ...

    def xor_sum(self, axis=None, keep_dims=False) -> tensor:
        ...

    def reduce_or(self, axis=None, keep_dims=False) -> tensor:
        ...

    def cumsum(self, axis=0, reverse=False) -> tensor:
        ...

    def cumprod(self, axis=0, reverse=False) -> tensor:
        ...

    def sort(self, dim: constexpr = None, descending: constexpr = CONSTEXPR_0) -> tensor:
        ...

    def flip(self, dim=None) -> tensor:
        ...


def _type_for_tuple_values(values, fields=None):
    return tuple_type([constexpr_type(x) if isinstance(x, (int, float, dtype)) else x.type for x in values], fields)


class tuple(base_value):

    def __init__(self, args: Sequence, type: Optional[tuple_type] = None):
        self.values = [i for i in args]
        if isinstance(type, tuple_type):
            self.type = type
        elif type is not None:  # make_template in ASTFunction.deserialize may pass us a list/tuple
            self.type = tuple_type(type)
        else:
            self.type = _type_for_tuple_values(self.values)

    def __getitem__(self, idx: constexpr):
        if isinstance(idx, int):
            idx = constexpr(idx)
        if isinstance(idx, constexpr):
            return self.values[idx]
        else:
            assert isinstance(idx, (slice, builtins.slice))
            return tuple(self.values[idx.start:idx.stop:idx.step])

    def __getattr__(self, name):
        fields = self.type.fields
        if fields is None or name not in fields:
            raise AttributeError(f"'tuple' object has no attribute {name}")
        return self.values[fields.index(name)]

    # TODO: remove
    def _setitem(self, idx, value):
        idx = _unwrap_if_constexpr(idx)
        assert isinstance(idx, int)
        self.values[idx] = value
        self.type = _type_for_tuple_values(self.values, self.type.fields)

    def __add__(self, other):
        other = _normalize_tuple(other)
        return tuple(self.values + other.values)
        # return tuple(a + b for a, b in zip(self.values, other.values))

    def __mul__(self, other):
        assert isinstance(other, constexpr)
        return tuple(self.values * other.value)

    def __eq__(self, other):
        other = _normalize_tuple(other)
        return constexpr(self.values == other.values)

    def __hash__(self):
        return hash(builtins.tuple(self.values))

    def __str__(self):
        return str([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        fields = self.type.fields
        if fields is not None:
            for field, v in zip(fields, self.values):
                v._set_name(builder, f"{name}.{field}")
        else:
            for i, v in enumerate(self.values):
                v._set_name(builder, f"{name}.{i}")

    def _flatten_ir(self, handles: List[ir.value]):
        for v in self.values:
            v._flatten_ir(handles)

    def __repr__(self):
        return f"({', '.join(repr(x) for x in self.values)})"


class slice:

    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
        self.type = slice_type()


class tensor_descriptor_base_type(base_type):

    def __init__(self, block_type: block_type):
        self.block_type = block_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor_base, int]:
        value = tensor_descriptor_base(handles[cursor], self.block_type)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        is_signed = self.block_type.element_ty.is_int_signed()
        out.append(builder.create_tensor_descriptor_type(self.block_type.to_ir(builder), is_signed))

    def __str__(self) -> str:
        # ex. "tensor_descriptor<float32[16, 32]>"
        return f"tensor_descriptor<{self.block_type}>"

    def __eq__(self, other) -> bool:
        if type(other) is not type(self):
            return False
        return self.block_type == other.block_type

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        return f"TD{self.block_type.mangle()}"


class tensor_descriptor_base(base_value):
    """"
    A tensor descriptor with unknown shape and strides
    """

    def __init__(self, handle, block_type: block_type):
        """Not called by user code."""
        super().__init__()

        self.handle = handle  # IR handle
        self.type = tensor_descriptor_base_type(block_type)  # Tensor type (block_type)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def block_type(self):
        return self.type.block_type

    @property
    def block_shape(self):
        return self.type.block_type.shape

    @property
    def dtype(self):
        return self.type.block_type.element_ty

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, offsets: Sequence[constexpr | tensor], _semantic=None) -> tensor:
        """Load a block from the descriptor starting at the given element offsets.

        Values outside of the tensor bounds will be filled with zeros.

        :note: Offset must be a multiple of 16-bytes
        """
        return _semantic.descriptor_load(self, offsets, "", "")

    @builtin
    def store(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        """Store a block from the descriptor starting at the given element offsets.

        Values outside of the tensor bounds will be ignored.

        :note: Offset must be a multiple of 16-bytes
        """
        return _semantic.descriptor_store(self, value, offsets)

    @builtin
    def atomic_add(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_add(self, value, offsets)

    @builtin
    def atomic_min(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_min(self, value, offsets)

    @builtin
    def atomic_max(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_max(self, value, offsets)

    @builtin
    def atomic_and(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_and(self, value, offsets)

    @builtin
    def atomic_or(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_or(self, value, offsets)

    @builtin
    def atomic_xor(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_atomic_xor(self, value, offsets)

    @builtin
    def gather(self, *args, _semantic=None) -> tensor:
        """Gather multiple descriptors worth of data"""
        assert len(args) == 2, f"descriptor gather only supports 2D indexing, but got {len(args)}"
        x_offsets = args[0]
        y_offset = args[1]
        return _semantic.descriptor_gather(self, x_offsets, y_offset, "", "")

    @builtin
    def scatter(self, value, *args, _semantic=None) -> tensor:
        """Scatter multiple descriptors worth of data"""
        assert len(args) == 2, f"descriptor scatter only supports 2D indexing, but got {len(args)}"
        x_offsets = args[0]
        y_offset = args[1]
        return _semantic.descriptor_scatter(self, value, x_offsets, y_offset)


class tensor_descriptor_type(tensor_descriptor_base_type):

    def __init__(self, block_type: block_type, shape_type: tuple_type, strides_type: tuple_type):
        self.block_type = block_type
        self.shape_type = shape_type
        self.strides_type = strides_type

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor_base, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        shape = shape.values
        strides = strides.values
        value = tensor_descriptor(handle, shape, strides, self.block_type)
        return value, cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        super()._flatten_ir_types(builder, out)
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def __eq__(self, other):
        return super().__eq__(other) and (self.shape_type == other.shape_type) and (self.strides_type
                                                                                    == other.strides_type)


class tensor_descriptor(tensor_descriptor_base):
    """A descriptor representing a tensor in global memory.
    """

    def __init__(self, handle, shape: List[tensor], strides: List[tensor], block_type: block_type):
        """Not called by user code."""
        # IR handle
        super().__init__(handle, block_type)
        # Global shape
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.type = tensor_descriptor_type(
            block_type,
            shape_type=self.shape.type,
            strides_type=self.strides.type,
        )

    def _set_name(self, builder: ir.builder, name: str) -> None:
        super()._set_name(builder, name)
        self.shape._set_name(builder, name + ".shape")
        self.strides._set_name(builder, name + ".stride")

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        super()._flatten_ir(handles)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)


# -----------------------
# aggregate
# -----------------------


@dataclass(frozen=True)
class _aggregate_type(base_type):
    """A generic base type for all Triton aggregate types.

    This class contains a reference to the original user-defined Python class
    and a list of class fields with their Triton types.
    """

    base_cls: type
    fields: List[Tuple[str, base_type]]

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[ir.value, int]:
        instance = self.base_cls._get_instance()
        for name, ty in self.fields:
            value, cursor = ty._unflatten_ir(handles, cursor)
            setattr(instance, name, value)
        return instance, cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        for name, ty in self.fields:
            ty._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        name = f"{self.base_cls.__module__}.{self.base_cls.__qualname__}"
        fields = [ty.mangle() for (name, ty) in self.fields]
        return f"{name}<{', '.join(fields)}>"


def _wrap_init_args(x):
    if isinstance(x, tuple):
        from triton.compiler.code_generator import _apply_to_tuple_values
        return _apply_to_tuple_values(x, _wrap_init_args)
    if isinstance(x, builtins.tuple):
        wrapped = builtins.tuple(_wrap_init_args(i) for i in x)
        fields = getattr(x, "_fields", None)
        ty = tuple_type([v.type for v in wrapped], fields)
        return tuple(wrapped, ty)
    if isinstance(x, base_value):
        return x
    return constexpr(x)


if TYPE_CHECKING:
    from typing_extensions import dataclass_transform
else:

    def dataclass_transform(**kwargs):
        return lambda obj: obj


_AGGREGATE_MISSING = object()


def _resolve_aggregate_fields(cls):
    all_annotations = {}
    all_defaults = {}
    # Inherit from oldest first, so child overrides parent
    for base in reversed(cls.__mro__[1:]):
        if base is base_value or base is object:
            continue
        if not getattr(base, "__triton_aggregate__", False):
            raise TypeError(f"Aggregates can only inherit from other aggregates, but got non-aggregate base: {base}")
        all_annotations.update(inspect.get_annotations(base))
        all_defaults.update(getattr(base, "__aggregate_defaults__", {}))

    # Add cls's own fields, resolving string annotations via typing.get_type_hints.
    own_names = inspect.get_annotations(cls)
    hints = typing.get_type_hints(cls)
    for name in own_names:
        all_annotations[name] = hints[name]
        val = cls.__dict__.get(name, _AGGREGATE_MISSING)
        if val is _AGGREGATE_MISSING:
            continue
        # Skip descriptors and methods - only plain values are defaults
        if not callable(val) or isinstance(val, base_value):
            all_defaults[name] = val
    return all_annotations, all_defaults


@dataclass_transform(eq_default=False)
def _aggregate(cls):
    all_annotations, all_defaults = _resolve_aggregate_fields(cls)

    init = cls.__dict__.get("__init__", None)

    if init is None:
        field_names = builtins.tuple(all_annotations.keys())

        def init(self, *args, **kwargs):
            if len(args) > len(field_names):
                raise TypeError(f"{cls.__name__}.__init__() takes {len(field_names) + 1} positional arguments "
                                f"but {len(args) + 1} were given")

            for index, name in enumerate(field_names):
                if index < len(args):
                    if name in kwargs:
                        raise TypeError(f"{cls.__name__}.__init__() got multiple values for argument '{name}'")
                    value = args[index]
                elif name in kwargs:
                    value = kwargs.pop(name)
                elif name in all_defaults:
                    value = all_defaults[name]
                else:
                    raise TypeError(f"{cls.__name__}.__init__() missing required argument: '{name}'")

                value = _wrap_init_args(value)
                setattr(self, name, value)

            if kwargs:
                unexpected = next(iter(kwargs))
                raise TypeError(f"{cls.__name__}.__init__() got an unexpected keyword argument '{unexpected}'")

        init.__triton_builtin__ = True

    # Define the wrapped Triton value type.
    class aggregate_value(base_value):
        __triton_builtin__ = True
        __triton_aggregate__ = True
        __annotations__ = all_annotations

        @classmethod
        def _get_instance(this_cls):
            return super().__new__(this_cls)

        def __new__(this_cls, *args, _semantic=None, _generator=None, **kwargs):
            # Call into the user-defined constructor.
            instance = this_cls._get_instance()
            # Track init phase so __setattr__ accepts writes during __init__
            # but rejects post-construction mutation.
            object.__setattr__(instance, "_aggregate_init_complete", False)
            extra_kwargs = {}
            if isinstance(init, JITCallable):
                # raise ValueError(f"{cls.__name__}.__init__ cannot be a @triton.jit function")
                pass
            else:
                if "_semantic" in inspect.signature(init).parameters:
                    extra_kwargs["_semantic"] = _semantic
                if "_generator" in inspect.signature(init).parameters:
                    extra_kwargs["_generator"] = _generator
            init(instance, *args, **extra_kwargs, **kwargs)

            # Require that the user-defined constructor initialized all fields.
            for name in all_annotations.keys():
                if not hasattr(instance, name):
                    raise AttributeError(f"constructor for {cls.__name__} did not initialize attribute '{name}'")

            # Lock further attribute assignment after __init__.
            object.__setattr__(instance, "_aggregate_init_complete", True)
            return instance

        # Only allow setting annotated attributes during __init__, and
        # only for attributes defined in the class annotations.
        def __setattr__(self, name, value):
            if name not in all_annotations:
                raise AttributeError(f"{cls.__name__} has no attribute '{name}'")
            if not isinstance(value, all_annotations[name]):
                raise TypeError(f"Expected {all_annotations[name]} for attribute '{name}', got {type(value)}")
            if getattr(self, "_aggregate_init_complete", False):
                raise AttributeError(f"cannot assign to field '{name}' on immutable aggregate {cls.__name__}; "
                                     f"use aggregate_replace() to construct a modified copy")
            super().__setattr__(name, value)

        def _set_name(self, builder: ir.builder, name: str) -> None:
            for key_name in all_annotations.keys():
                getattr(self, key_name)._set_name(builder, f"{name}.{key_name}")

        def _flatten_ir(self, handles: List[ir.value]) -> None:
            for name in all_annotations.keys():
                getattr(self, name)._flatten_ir(handles)

        @property
        def type(self):
            return _aggregate_type(aggregate_value,
                                   [(name, getattr(self, name).type) for name in all_annotations.keys()])

    hash_attrs = [init]

    for (name, member) in inspect.getmembers(cls):
        if inspect.isfunction(member) or inspect.ismethod(member) or isinstance(member, JITCallable):
            if name == "__init__":
                continue
            # __annotate__ is a Python 3.14+ internal; exclude from hash and
            # don't copy it onto the aggregate value type.
            if name == "__annotate__":
                continue
            # Don't override aggregate infrastructure methods inherited from
            # processed parent aggregates (e.g. __new__, __setattr__, _flatten_ir)
            if name in aggregate_value.__dict__:
                continue
            setattr(aggregate_value, name, member)

            # Exclude __annotate_func__ from hash — isn't user facing (Python 3.14+).
            if name != "__annotate_func__":
                hash_attrs.append(member)

    aggregate_value.hash_attrs = hash_attrs
    aggregate_value.__name__ = cls.__name__
    aggregate_value.__module__ = cls.__module__
    aggregate_value.__qualname__ = cls.__qualname__
    aggregate_value.__doc__ = cls.__doc__
    aggregate_value.__aggregate_fields__ = builtins.tuple(all_annotations.keys())
    aggregate_value.__aggregate_defaults__ = dict(all_defaults)

    return aggregate_value


def aggregate_replace(instance, **changes):
    """Create a copy of an aggregate instance with specified fields replaced.

    Similar to dataclasses.replace() — returns a new instance of the same
    aggregate type with the given fields updated and all other fields copied
    from the original instance.

    :param instance: The aggregate instance to copy
    :param changes: Keyword arguments for fields to replace
    :return: A new aggregate instance with the specified changes
    """
    if not getattr(type(instance), "__triton_aggregate__", False):
        raise TypeError(f"aggregate_replace() expects an aggregate instance, got {type(instance)}")

    field_names = type(instance).__aggregate_fields__
    for name in changes:
        if name not in field_names:
            raise TypeError(f"{type(instance).__name__} has no field '{name}'")

    kwargs = {name: getattr(instance, name) for name in field_names}
    kwargs.update(changes)

    return type(instance)(**kwargs)


def _is_block_ptr(value) -> bool:
    return isinstance(value, base_value) and getattr(value, "__triton_block_ptr__", False)


def _as_list_like(values):
    normalized = _normalize_tuple(values)
    if isinstance(normalized, (list, builtins.tuple, tuple)):
        return list(normalized)
    return [normalized]


def _canonicalize_block_ptr_static_tuple(values, name: str, *, positive: bool = False) -> tuple:
    converted = []
    for value in _as_list_like(values):
        value = _unwrap_if_constexpr(value)
        if not isinstance(value, int):
            raise ValueError(f"Expected `{name}` to contain only integers")
        if positive and value <= 0:
            raise ValueError(f"Expected `{name}` to contain only positive integers")
        converted.append(constexpr(value))
    return tuple(converted)


def _canonicalize_block_ptr_dynamic_tuple(values, name: str, _semantic) -> tuple:
    converted = []
    for value in _as_list_like(values):
        value = _semantic.to_tensor(value)
        if value.shape:
            raise ValueError(f"Expected `{name}` entries to be scalar tensors")
        if not value.dtype.is_int():
            raise ValueError(f"Expected `{name}` entries to be integers")
        if value.dtype != int64:
            value = _semantic.cast(value, int64)
        converted.append(value)
    return tuple(converted)


def _canonicalize_block_ptr_boundary_check(boundary_check, rank: int) -> builtins.tuple[int, ...]:
    checked = set()
    if boundary_check is None:
        return checked

    for dim in _as_list_like(boundary_check):
        dim = _unwrap_if_constexpr(dim)
        if not isinstance(dim, int) or not (0 <= dim < rank):
            raise ValueError(f"Expected `boundary_check` to contain dimensions in [0, {rank})")
        checked.add(dim)
    if len(checked) != len(boundary_check):
        raise ValueError("Duplicate dimension in `boundary_check`")
    return checked


@_aggregate
class _block_ptr:
    base: tensor
    shape: tuple
    strides: tuple
    offsets: tuple
    block_shape: tuple
    order: tuple

    __triton_block_ptr__ = True

    def __init__(self, base, shape, strides, offsets, block_shape, order, _semantic=None):
        if not base.type.is_ptr() or base.type.is_block():
            raise ValueError("Expected `base` to be a scalar pointer type")
        if isinstance(base.type.element_ty, block_type):
            raise ValueError("Expected `base` to point to a scalar element type")

        self.base = base
        self.shape = _canonicalize_block_ptr_dynamic_tuple(shape, "shape", _semantic)
        self.strides = _canonicalize_block_ptr_dynamic_tuple(strides, "strides", _semantic)
        self.offsets = _canonicalize_block_ptr_dynamic_tuple(offsets, "offsets", _semantic)
        self.block_shape = _canonicalize_block_ptr_static_tuple(block_shape, "block_shape", positive=True)
        self.order = _canonicalize_block_ptr_static_tuple(order, "order")

        rank = len(self.block_shape)
        if rank == 0:
            raise ValueError("Expected `make_block_ptr` to describe at least one dimension")
        for field_name, field_value in (("shape", self.shape), ("strides", self.strides), ("offsets", self.offsets),
                                        ("order", self.order)):
            if len(field_value) != rank:
                raise ValueError(
                    f"Expected `shape`, `strides`, `offsets`, `block_shape`, and `order` to have the same length; "
                    f"`{field_name}` has length {len(field_value)} but expected {rank}")
        order_values = [_unwrap_if_constexpr(value) for value in self.order]
        if sorted(order_values) != list(builtins.range(rank)):
            raise ValueError(f"Expected `order` to be a permutation of 0..{rank - 1}")

    def _tile_shape(self):
        return [_unwrap_if_constexpr(extent) for extent in self.block_shape]

    def _materialize(self, boundary_check=(), _semantic=None):
        tile_shape = self._tile_shape()
        checked_dims = _canonicalize_block_ptr_boundary_check(boundary_check, len(tile_shape))
        ptrs = self.base
        mask = None
        for dim, extent in enumerate(tile_shape):
            coord = add(self.offsets[dim], arange(0, extent, _semantic=_semantic), _semantic=_semantic)
            for _ in builtins.range(dim):
                coord = expand_dims(coord, 0, _semantic=_semantic)
            for _ in builtins.range(dim + 1, len(tile_shape)):
                coord = expand_dims(coord, -1, _semantic=_semantic)
            coord = broadcast_to(coord, tile_shape, _semantic=_semantic)
            ptrs = add(ptrs, mul(coord, self.strides[dim], _semantic=_semantic), _semantic=_semantic)
            if dim in checked_dims:
                valid = _semantic.and_(_semantic.greater_equal(coord, 0), _semantic.less_than(coord, self.shape[dim]))
                mask = valid if mask is None else _semantic.and_(mask, valid)
        return ptrs, mask

    def advance(self, offsets, _semantic=None):
        new_offsets = []
        offsets = _canonicalize_block_ptr_dynamic_tuple(offsets, "offsets", _semantic)
        if len(offsets) != len(self.offsets):
            raise ValueError(f"Expected `offsets` to have length {len(self.offsets)} but received {len(offsets)}")
        for old_offset, delta in zip(self.offsets, offsets):
            new_offsets.append(add(old_offset, delta, _semantic=_semantic))
        return _block_ptr(self.base, self.shape, self.strides, tuple(new_offsets), self.block_shape, self.order,
                          _semantic=_semantic)

    def load(self, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
             volatile=False, _semantic=None):
        if mask is not None or other is not None:
            raise ValueError("`mask` and `other` arguments cannot be specified for loading block pointers")

        padding_option = _unwrap_if_constexpr(padding_option)
        cache_modifier = _unwrap_if_constexpr(cache_modifier)
        eviction_policy = _unwrap_if_constexpr(eviction_policy)
        volatile = _unwrap_if_constexpr(volatile)
        if padding_option is None:
            padding_option = ""
        ptrs, mask = self._materialize(boundary_check, _semantic=_semantic)

        if padding_option == "":
            generated_other = None
        elif padding_option == "zero":
            # Use a float literal for float element types: `cast` cannot take an
            # integer to a non-standard float (e.g. fp8e4nv), but fp32 -> fp8 is fine.
            generated_other = 0.0 if self.base.dtype.element_ty.is_floating() else 0
        elif padding_option == "nan":
            if self.base.dtype.element_ty.is_int():
                raise ValueError("Padding option `nan` is not supported for integer block pointers")
            generated_other = float("nan")
        else:
            raise ValueError(f"Padding option {padding_option} not supported")

        return load(ptrs, mask=mask, other=generated_other, cache_modifier=cache_modifier,
                    eviction_policy=eviction_policy, volatile=volatile, _semantic=_semantic)

    def store(self, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _semantic=None):
        if mask is not None:
            raise ValueError("`mask` argument cannot be specified for storing block pointers")

        cache_modifier = _unwrap_if_constexpr(cache_modifier)
        eviction_policy = _unwrap_if_constexpr(eviction_policy)
        ptrs, mask = self._materialize(boundary_check, _semantic=_semantic)
        return store(ptrs, value, mask=mask, cache_modifier=cache_modifier, eviction_policy=eviction_policy,
                     _semantic=_semantic)


_block_ptr.__triton_block_ptr__ = True
_block_ptr.dtype = property(lambda self: self.base.dtype)

# -----------------------
# SPMD Programming Model
# -----------------------


@builtin
def program_id(axis, _semantic=None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = _semantic.program_id(0)
    #     pid1 = _semantic.program_id(1)
    #     pid2 = _semantic.program_id(2)
    #     npg0 = _semantic.num_programs(0)
    #     npg1 = _semantic.num_programs(1)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    axis = _unwrap_if_constexpr(axis)
    return _semantic.program_id(axis)


@builtin
def num_programs(axis, _semantic=None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = _unwrap_if_constexpr(axis)
    return _semantic.num_programs(axis)


# -----------------------
# Block Initialization
# -----------------------


@builtin
def arange(start, end, _semantic=None):
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    return _semantic.arange(start, end)


arange.__doc__ = f"""
    Returns contiguous values within the half-open interval :code:`[start,
    end)`.  :code:`end - start` must be less than or equal to
    :code:`TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}`

    :param start: Start of the interval. Must be a power of two.
    :type start: int32
    :param end: End of the interval. Must be a power of two greater than
        :code:`start`.
    :type end: int32
"""


def _unwrap_shape(shape):
    shape = _unwrap_if_constexpr(shape)
    return [_unwrap_if_constexpr(s) for s in shape]


def _shape_check_impl(shape):
    shape = _unwrap_shape(shape)
    validate_block_shape(shape)
    return shape


@builtin
def full(shape, value, dtype, _semantic=None):
    """
    Returns a tensor filled with the scalar value for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param value: A scalar value to fill the array with
    :type value: scalar
    :param dtype: Data type of the new array, e.g., :code:`tl.float16`
    :type dtype: tl.dtype
    """
    shape = _shape_check_impl(shape)
    value = _unwrap_if_constexpr(value)
    dtype = _unwrap_if_constexpr(dtype)
    return _semantic.full(shape, value, dtype)


# -----------------------
# Shape Manipulation
# -----------------------


@builtin
def broadcast(input, other, _semantic=None):
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return _semantic.broadcast_impl_value(input, other)


@_tensor_member_fn
@builtin
def broadcast_to(input, *shape, _semantic=None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape:

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        broadcast_to(x, (32, 32))
        broadcast_to(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return _semantic.broadcast_impl_shape(input, shape)


@_tensor_member_fn
@builtin
def trans(input: tensor, *dims, _semantic=None):
    """
    Permutes the dimensions of a tensor.

    If the parameter :code:`dims` is not specified, the function defaults to
    swapping the last two axes, thereby performing an (optionally batched)
    2D transpose.

    :param input: The input tensor.
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.

    :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        trans(x, (2, 1, 0))
        trans(x, 2, 1, 0)

    :py:func:`permute` is equivalent to this function, except it doesn't
    have the special case when no permutation is specified.
    """
    dims = _unwrap_iterable(dims)
    if not dims:
        n = len(input.shape)
        if n < 2:
            raise ValueError("tl.trans invoked with a 0- or 1-dimensional tensor")
        dims = list(builtins.range(n - 2)) + [n - 1, n - 2]
    return _semantic.permute(input, dims)


@_tensor_member_fn
@builtin
def permute(input, *dims, _semantic=None):
    """
    Permutes the dimensions of a tensor.

    :param input: The input tensor.
    :type input: Block
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a 3D tensor.

    :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        permute(x, (2, 1, 0))
        permute(x, 2, 1, 0)

    :py:func:`trans` is equivalent to this function, except when
    :code:`dims` is empty, it tries to swap the last two axes.
    """
    dims = _unwrap_iterable(dims)
    return _semantic.permute(input, dims)


@builtin
def cat(input, other, can_reorder=False, dim=0, _semantic=None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input: Tensor
    :param other: The second input tensor.
    :type other: Tensor
    :param can_reorder: Compiler hint. If true, the compiler is
        allowed to reorder elements while concatenating inputs.  Only use if the
        order does not matter (e.g., result is only used in reduction ops).
    :type can_reorder: bool
    :param dim: The dimension to concatenate along (used when can_reorder is False).
    :type dim: int
    """
    if can_reorder:
        return _semantic.cat(input, other, can_reorder)

    rank = len(input.shape)
    assert rank == len(other.shape), f"tensors must have the same rank, got {rank} and {len(other.shape)}"
