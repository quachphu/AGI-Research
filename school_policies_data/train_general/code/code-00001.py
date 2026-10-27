            # this includes types.Function, types.BuiltinFunctionType,
            # types.BuiltinMethodType, functools.partial, functools.singledispatch,
            # "class funclike" from Lib/test/test_inspect... on and on it goes.
            ann = getattr(obj, '__annotations__', None)
            obj_globals = getattr(obj, '__globals__', None)
            obj_locals = None
            unwrap = obj
        elif hasattr(obj, '__annotations__'):
            ann = obj.__annotations__
            obj_globals = obj_locals = unwrap = None
        else:
            raise TypeError(f"{obj!r} is not a module, class, or callable.")

        if ann is None:
            return {}

        if not isinstance(ann, dict):
            raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

        if not ann:
            return {}

        if not eval_str:
            if format is Format.STRING:
                return {
                    key: value if isinstance(value, str) else typing._type_repr(value)
                    for key, value in ann.items()
                }
            return dict(ann)

        if unwrap is not None:
            while True:
                if hasattr(unwrap, '__wrapped__'):
                    unwrap = unwrap.__wrapped__
                    continue
                if isinstance(unwrap, functools.partial):
                    unwrap = unwrap.func
                    continue
                break
            if hasattr(unwrap, "__globals__"):
                obj_globals = unwrap.__globals__

        if globals is None:
            globals = obj_globals
        if locals is None:
            locals = obj_locals or {}

        # "Inject" type parameters into the local namespace
        # (unless they are shadowed by assignments *in* the local namespace),
        # as a way of emulating annotation scopes when calling `eval()`
        if type_params := getattr(obj, "__type_params__", ()):
            locals = {param.__name__: param for param in type_params} | locals

        return_value = {key:
            value if not isinstance(value, str) else eval(value, globals, locals)
            for key, value in ann.items() }
        return return_value


if hasattr(typing, "evaluate_forward_ref"):
    evaluate_forward_ref = typing.evaluate_forward_ref
else:
    # Implements annotationlib.ForwardRef.evaluate
    def _eval_with_owner(
        forward_ref, *, owner=None, globals=None, locals=None, type_params=None
    ):
        if forward_ref.__forward_evaluated__:
            return forward_ref.__forward_value__
        if getattr(forward_ref, "__cell__", None) is not None:
            try:
                value = forward_ref.__cell__.cell_contents
            except ValueError:
                pass
            else:
                forward_ref.__forward_evaluated__ = True
                forward_ref.__forward_value__ = value
                return value
        if owner is None:
            owner = getattr(forward_ref, "__owner__", None)

        if (
            globals is None
            and getattr(forward_ref, "__forward_module__", None) is not None
        ):
            globals = getattr(
                sys.modules.get(forward_ref.__forward_module__, None), "__dict__", None
            )
        if globals is None:
            globals = getattr(forward_ref, "__globals__", None)
        if globals is None:
            if isinstance(owner, type):
                module_name = getattr(owner, "__module__", None)
                if module_name:
                    module = sys.modules.get(module_name, None)
                    if module:
                        globals = getattr(module, "__dict__", None)
            elif isinstance(owner, _types.ModuleType):
                globals = getattr(owner, "__dict__", None)
            elif callable(owner):
                globals = getattr(owner, "__globals__", None)

        # If we pass None to eval() below, the globals of this module are used.
        if globals is None:
            globals = {}

        if locals is None:
            locals = {}
            if isinstance(owner, type):
                locals.update(vars(owner))

        if type_params is None and owner is not None:
            # "Inject" type parameters into the local namespace
            # (unless they are shadowed by assignments *in* the local namespace),
            # as a way of emulating annotation scopes when calling `eval()`
            type_params = getattr(owner, "__type_params__", None)

        # Type parameters exist in their own scope, which is logically
        # between the locals and the globals. We simulate this by adding
        # them to the globals.
        if type_params is not None:
            globals = dict(globals)
            for param in type_params:
                globals[param.__name__] = param

        arg = forward_ref.__forward_arg__
        if arg.isidentifier() and not keyword.iskeyword(arg):
            if arg in locals:
                value = locals[arg]
            elif arg in globals:
                value = globals[arg]
            elif hasattr(builtins, arg):
                return getattr(builtins, arg)
            else:
                raise NameError(arg)
        else:
            code = forward_ref.__forward_code__
            value = eval(code, globals, locals)
        forward_ref.__forward_evaluated__ = True
        forward_ref.__forward_value__ = value
        return value

    def evaluate_forward_ref(
        forward_ref,
        *,
        owner=None,
        globals=None,
        locals=None,
        type_params=None,
        format=None,
        _recursive_guard=frozenset(),
    ):
        """Evaluate a forward reference as a type hint.

        This is similar to calling the ForwardRef.evaluate() method,
        but unlike that method, evaluate_forward_ref() also:

        * Recursively evaluates forward references nested within the type hint.
        * Rejects certain objects that are not valid type hints.
        * Replaces type hints that evaluate to None with types.NoneType.
        * Supports the *FORWARDREF* and *STRING* formats.

        *forward_ref* must be an instance of ForwardRef. *owner*, if given,
        should be the object that holds the annotations that the forward reference
        derived from, such as a module, class object, or function. It is used to
        infer the namespaces to use for looking up names. *globals* and *locals*
        can also be explicitly given to provide the global and local namespaces.
        *type_params* is a tuple of type parameters that are in scope when
        evaluating the forward reference. This parameter must be provided (though
        it may be an empty tuple) if *owner* is not given and the forward reference
        does not already have an owner set. *format* specifies the format of the
        annotation and is a member of the annotationlib.Format enum.

        """
        if format == Format.STRING:
            return forward_ref.__forward_arg__
        if forward_ref.__forward_arg__ in _recursive_guard:
            return forward_ref

        # Evaluate the forward reference
        try:
            value = _eval_with_owner(
                forward_ref,
                owner=owner,
                globals=globals,
                locals=locals,
                type_params=type_params,
            )
        except NameError:
            if format == Format.FORWARDREF:
                return forward_ref
            else:
                raise

        if isinstance(value, str):
            value = ForwardRef(value)

        # Recursively evaluate the type
        if isinstance(value, ForwardRef):
            if getattr(value, "__forward_module__", True) is not None:
                globals = None
            return evaluate_forward_ref(
                value,
                globals=globals,
                locals=locals,
                 type_params=type_params, owner=owner,
                _recursive_guard=_recursive_guard, format=format
            )
        if sys.version_info < (3, 12, 5) and type_params:
            # Make use of type_params
            locals = dict(locals) if locals else {}
            for tvar in type_params:
                if tvar.__name__ not in locals:  # lets not overwrite something present
                    locals[tvar.__name__] = tvar
        if sys.version_info < (3, 12, 5):
            return typing._eval_type(
                value,
                globals,
                locals,
                recursive_guard=_recursive_guard | {forward_ref.__forward_arg__},
            )
        else:
            return typing._eval_type(
                value,
                globals,
                locals,
                type_params,
                recursive_guard=_recursive_guard | {forward_ref.__forward_arg__},
            )


if sys.version_info >= (3, 14, 0, "beta"):
    type_repr = annotationlib.type_repr
else:
    def type_repr(value):
        """Convert a Python value to a format suitable for use with the STRING format.

        This is intended as a helper for tools that support the STRING format but do
        not have access to the code that originally produced the annotations. It uses
        repr() for most objects.

        """
        if isinstance(value, (type, _types.FunctionType, _types.BuiltinFunctionType)):
            if value.__module__ == "builtins":
                return value.__qualname__
            return f"{value.__module__}.{value.__qualname__}"
        if value is ...:
            return "..."
        return repr(value)


# Aliases for items that are in typing in all supported versions.
# We use hasattr() checks so this library will continue to import on
# future versions of Python that may remove these names.
_typing_names = [
    "AbstractSet",
    "AnyStr",
    "BinaryIO",
    "Callable",
    "Collection",
    "Container",
    "Dict",
    "FrozenSet",
    "Hashable",
    "IO",
    "ItemsView",
    "Iterable",
    "Iterator",
    "KeysView",
    "List",
    "Mapping",
    "MappingView",
    "Match",
    "MutableMapping",
    "MutableSequence",
    "MutableSet",
    "Optional",
    "Pattern",
    "Reversible",
    "Sequence",
    "Set",
    "Sized",
    "TextIO",
    "Tuple",
    "Union",
    "ValuesView",
    "cast",
    "no_type_check",
    # This is private, but it was defined by typing_extensions for a long time
    # and some users rely on it.
    "_AnnotatedAlias",
]

# Breakpoint: https://github.com/python/cpython/pull/133602
if sys.version_info < (3, 15, 0):
    _typing_names.append("no_type_check_decorator")
    __all__.append("no_type_check_decorator")

globals().update(
    {name: getattr(typing, name) for name in _typing_names if hasattr(typing, name)}
)
# These are defined unconditionally because they are used in
# typing-extensions itself.
Generic = typing.Generic
ForwardRef = typing.ForwardRef
Annotated = typing.Annotated
<|endoftext|>"""
certifi.py
~~~~~~~~~~

This module returns the installation location of cacert.pem or its contents.
"""
import sys
import atexit

def exit_cacert_ctx() -> None:
    _CACERT_CTX.__exit__(None, None, None)  # type: ignore[union-attr]


if sys.version_info >= (3, 11):

    from importlib.resources import as_file, files

    _CACERT_CTX = None
    _CACERT_PATH = None

    def where() -> str:
        # This is slightly terrible, but we want to delay extracting the file
        # in cases where we're inside of a zipimport situation until someone
        # actually calls where(), but we don't want to re-extract the file
        # on every call of where(), so we'll do it once then store it in a
        # global variable.
        global _CACERT_CTX
        global _CACERT_PATH
        if _CACERT_PATH is None:
            # This is slightly janky, the importlib.resources API wants you to
            # manage the cleanup of this file, so it doesn't actually return a
            # path, it returns a context manager that will give you the path
            # when you enter it and will do any cleanup when you leave it. In
            # the common case of not needing a temporary file, it will just
            # return the file system location and the __exit__() is a no-op.
            #
            # We also have to hold onto the actual context manager, because
            # it will do the cleanup whenever it gets garbage collected, so
            # we will also store that at the global level as well.
            _CACERT_CTX = as_file(files("certifi").joinpath("cacert.pem"))
            _CACERT_PATH = str(_CACERT_CTX.__enter__())
            atexit.register(exit_cacert_ctx)

        return _CACERT_PATH

    def contents() -> str:
        return files("certifi").joinpath("cacert.pem").read_text(encoding="ascii")

else:

    from importlib.resources import path as get_path, read_text

    _CACERT_CTX = None
    _CACERT_PATH = None

    def where() -> str:
        # This is slightly terrible, but we want to delay extracting the
        # file in cases where we're inside of a zipimport situation until
        # someone actually calls where(), but we don't want to re-extract
        # the file on every call of where(), so we'll do it once then store
        # it in a global variable.
        global _CACERT_CTX
        global _CACERT_PATH
        if _CACERT_PATH is None:
            # This is slightly janky, the importlib.resources API wants you
            # to manage the cleanup of this file, so it doesn't actually
            # return a path, it returns a context manager that will give
            # you the path when you enter it and will do any cleanup when
            # you leave it. In the common case of not needing a temporary
            # file, it will just return the file system location and the
            # __exit__() is a no-op.
            #
            # We also have to hold onto the actual context manager, because
            # it will do the cleanup whenever it gets garbage collected, so
            # we will also store that at the global level as well.
            _CACERT_CTX = get_path("certifi", "cacert.pem")
            _CACERT_PATH = str(_CACERT_CTX.__enter__())
            atexit.register(exit_cacert_ctx)

        return _CACERT_PATH

    def contents() -> str:
        return read_text("certifi", "cacert.pem", encoding="ascii")
<|endoftext|>import argparse

from certifi import contents, where

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--contents", action="store_true")
args = parser.parse_args()

if args.contents:
    print(contents())
else:
    print(where())
<|endoftext|>from .ctx_base import StandardBaseContext

import math
import cmath
from . import math2

from . import function_docs

from .libmp import mpf_bernoulli, to_float, int_types
from . import libmp

class FPContext(StandardBaseContext):
    """
    Context for fast low-precision arithmetic (53-bit precision, giving at most
    about 15-digit accuracy), using Python's builtin float and complex.
    """

    def __init__(ctx):
        StandardBaseContext.__init__(ctx)

        # Override SpecialFunctions implementation
        ctx.loggamma = math2.loggamma
        ctx._bernoulli_cache = {}
        ctx.pretty = False

        ctx._init_aliases()

    _mpq = lambda cls, x: float(x[0])/x[1]

    NoConvergence = libmp.NoConvergence

    def _get_prec(ctx): return 53
    def _set_prec(ctx, p): return
    def _get_dps(ctx): return 15
    def _set_dps(ctx, p): return

    _fixed_precision = True

    prec = property(_get_prec, _set_prec)
    dps = property(_get_dps, _set_dps)

    zero = 0.0
    one = 1.0
    eps = math2.EPS
    inf = math2.INF
    ninf = math2.NINF
    nan = math2.NAN
    j = 1j

    # Called by SpecialFunctions.__init__()
    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        if wrap:
            def f_wrapped(ctx, *args, **kwargs):
                convert = ctx.convert
                args = [convert(a) for a in args]
                return f(ctx, *args, **kwargs)
        else:
            f_wrapped = f
        f_wrapped.__doc__ = function_docs.__dict__.get(name, f.__doc__)
        setattr(cls, name, f_wrapped)

    def bernoulli(ctx, n):
        cache = ctx._bernoulli_cache
        if n in cache:
            return cache[n]
        cache[n] = to_float(mpf_bernoulli(n, 53, 'n'), strict=True)
        return cache[n]

    pi = math2.pi
    e = math2.e
    euler = math2.euler
    sqrt2 = 1.4142135623730950488
    sqrt5 = 2.2360679774997896964
    phi = 1.6180339887498948482
    ln2 = 0.69314718055994530942
    ln10 = 2.302585092994045684
    euler = 0.57721566490153286061
    catalan = 0.91596559417721901505
    khinchin = 2.6854520010653064453
    apery = 1.2020569031595942854
    glaisher = 1.2824271291006226369

    absmin = absmax = abs

    def is_special(ctx, x):
        return x - x != 0.0

    def isnan(ctx, x):
        return x != x

    def isinf(ctx, x):
        return abs(x) == math2.INF

    def isnormal(ctx, x):
        if x:
            return x - x == 0.0
        return False

    def isnpint(ctx, x):
        if type(x) is complex:
            if x.imag:
                return False
            x = x.real
        return x <= 0.0 and round(x) == x

    mpf = float
    mpc = complex

    def convert(ctx, x):
        try:
            return float(x)
        except:
            return complex(x)

    power = staticmethod(math2.pow)
    sqrt = staticmethod(math2.sqrt)
    exp = staticmethod(math2.exp)
    ln = log = staticmethod(math2.log)
    cos = staticmethod(math2.cos)
    sin = staticmethod(math2.sin)
    tan = staticmethod(math2.tan)
    cos_sin = staticmethod(math2.cos_sin)
    acos = staticmethod(math2.acos)
    asin = staticmethod(math2.asin)
    atan = staticmethod(math2.atan)
    cosh = staticmethod(math2.cosh)
    sinh = staticmethod(math2.sinh)
    tanh = staticmethod(math2.tanh)
    gamma = staticmethod(math2.gamma)
    rgamma = staticmethod(math2.rgamma)
    fac = factorial = staticmethod(math2.factorial)
    floor = staticmethod(math2.floor)
    ceil = staticmethod(math2.ceil)
    cospi = staticmethod(math2.cospi)
    sinpi = staticmethod(math2.sinpi)
    cbrt = staticmethod(math2.cbrt)
    _nthroot = staticmethod(math2.nthroot)
    _ei = staticmethod(math2.ei)
    _e1 = staticmethod(math2.e1)
    _zeta = _zeta_int = staticmethod(math2.zeta)

    # XXX: math2
    def arg(ctx, z):
        z = complex(z)
        return math.atan2(z.imag, z.real)

    def expj(ctx, x):
        return ctx.exp(ctx.j*x)

    def expjpi(ctx, x):
        return ctx.exp(ctx.j*ctx.pi*x)

    ldexp = math.ldexp
    frexp = math.frexp

    def mag(ctx, z):
        if z:
            return ctx.frexp(abs(z))[1]
        return ctx.ninf

    def isint(ctx, z):
        if hasattr(z, "imag"):   # float/int don't have .real/.imag in py2.5
            if z.imag:
                return False
            z = z.real
        try:
            return z == int(z)
        except:
            return False

    def nint_distance(ctx, z):
        if hasattr(z, "imag"):   # float/int don't have .real/.imag in py2.5
            n = round(z.real)
        else:
            n = round(z)
        if n == z:
            return n, ctx.ninf
        return n, ctx.mag(abs(z-n))

    def _convert_param(ctx, z):
        if type(z) is tuple:
            p, q = z
            return ctx.mpf(p) / q, 'R'
        if hasattr(z, "imag"):    # float/int don't have .real/.imag in py2.5
            intz = int(z.real)
        else:
            intz = int(z)
        if z == intz:
            return intz, 'Z'
        return z, 'R'

    def _is_real_type(ctx, z):
        return isinstance(z, float) or isinstance(z, int_types)

    def _is_complex_type(ctx, z):
        return isinstance(z, complex)

    def hypsum(ctx, p, q, types, coeffs, z, maxterms=6000, **kwargs):
        coeffs = list(coeffs)
        num = range(p)
        den = range(p,p+q)
        tol = ctx.eps
        s = t = 1.0
        k = 0
        while 1:
            for i in num: t *= (coeffs[i]+k)
            for i in den: t /= (coeffs[i]+k)
            k += 1; t /= k; t *= z; s += t
            if abs(t) < tol:
                return s
            if k > maxterms:
                raise ctx.NoConvergence

    def atan2(ctx, x, y):
        return math.atan2(x, y)

    def psi(ctx, m, z):
        m = int(m)
        if m == 0:
            return ctx.digamma(z)
        return (-1)**(m+1) * ctx.fac(m) * ctx.zeta(m+1, z)

    digamma = staticmethod(math2.digamma)

    def harmonic(ctx, x):
        x = ctx.convert(x)
        if x == 0 or x == 1:
            return x
        return ctx.digamma(x+1) + ctx.euler

    nstr = str

    def to_fixed(ctx, x, prec):
        return int(math.ldexp(x, prec))

    def rand(ctx):
        import random
        return random.random()

    _erf = staticmethod(math2.erf)
    _erfc = staticmethod(math2.erfc)

    def sum_accurately(ctx, terms, check_step=1):
        s = ctx.zero
        k = 0
        for term in terms():
            s += term
            if (not k % check_step) and term:
                if abs(term) <= 1e-18*abs(s):
                    break
            k += 1
        return s
<|endoftext|>"""
This module complements the math and cmath builtin modules by providing
fast machine precision versions of some additional functions (gamma, ...)
and wrapping math/cmath functions so that they can be called with either
real or complex arguments.
"""

import operator
import math
import cmath

# Irrational (?) constants
pi = 3.1415926535897932385
e = 2.7182818284590452354
sqrt2 = 1.4142135623730950488
sqrt5 = 2.2360679774997896964
phi = 1.6180339887498948482
ln2 = 0.69314718055994530942
ln10 = 2.302585092994045684
euler = 0.57721566490153286061
catalan = 0.91596559417721901505
khinchin = 2.6854520010653064453
apery = 1.2020569031595942854

logpi = 1.1447298858494001741

def _mathfun_real(f_real, f_complex):
    def f(x, **kwargs):
        if type(x) is float:
            return f_real(x)
        if type(x) is complex:
            return f_complex(x)
        try:
            x = float(x)
            return f_real(x)
        except (TypeError, ValueError):
            x = complex(x)
            return f_complex(x)
    f.__name__ = f_real.__name__
    return f

def _mathfun(f_real, f_complex):
    def f(x, **kwargs):
        if type(x) is complex:
            return f_complex(x)
        try:
            return f_real(float(x))
        except (TypeError, ValueError):
            return f_complex(complex(x))
    f.__name__ = f_real.__name__
    return f

def _mathfun_n(f_real, f_complex):
    def f(*args, **kwargs):
        try:
            return f_real(*(float(x) for x in args))
        except (TypeError, ValueError):
            return f_complex(*(complex(x) for x in args))
    f.__name__ = f_real.__name__
    return f

# Workaround for non-raising log and sqrt in Python 2.5 and 2.4
# on Unix system
try:
    math.log(-2.0)
    def math_log(x):
        if x <= 0.0:
            raise ValueError("math domain error")
        return math.log(x)
    def math_sqrt(x):
        if x < 0.0:
            raise ValueError("math domain error")
        return math.sqrt(x)
except (ValueError, TypeError):
    math_log = math.log
    math_sqrt = math.sqrt

pow = _mathfun_n(operator.pow, lambda x, y: complex(x)**y)
log = _mathfun_n(math_log, cmath.log)
sqrt = _mathfun(math_sqrt, cmath.sqrt)
exp = _mathfun_real(math.exp, cmath.exp)

cos = _mathfun_real(math.cos, cmath.cos)
sin = _mathfun_real(math.sin, cmath.sin)
tan = _mathfun_real(math.tan, cmath.tan)

acos = _mathfun(math.acos, cmath.acos)
asin = _mathfun(math.asin, cmath.asin)
atan = _mathfun_real(math.atan, cmath.atan)

cosh = _mathfun_real(math.cosh, cmath.cosh)
sinh = _mathfun_real(math.sinh, cmath.sinh)
tanh = _mathfun_real(math.tanh, cmath.tanh)

floor = _mathfun_real(math.floor,
    lambda z: complex(math.floor(z.real), math.floor(z.imag)))
ceil = _mathfun_real(math.ceil,
    lambda z: complex(math.ceil(z.real), math.ceil(z.imag)))


cos_sin = _mathfun_real(lambda x: (math.cos(x), math.sin(x)),
                        lambda z: (cmath.cos(z), cmath.sin(z)))

cbrt = _mathfun(lambda x: x**(1./3), lambda z: z**(1./3))

def nthroot(x, n):
    r = 1./n
    try:
        return float(x) ** r
    except (ValueError, TypeError):
        return complex(x) ** r

def _sinpi_real(x):
    if x < 0:
        return -_sinpi_real(-x)
    n, r = divmod(x, 0.5)
    r *= pi
    n %= 4
    if n == 0: return math.sin(r)
    if n == 1: return math.cos(r)
    if n == 2: return -math.sin(r)
    if n == 3: return -math.cos(r)

def _cospi_real(x):
    if x < 0:
        x = -x
    n, r = divmod(x, 0.5)
    r *= pi
    n %= 4
    if n == 0: return math.cos(r)
    if n == 1: return -math.sin(r)
    if n == 2: return -math.cos(r)
    if n == 3: return math.sin(r)

def _sinpi_complex(z):
    if z.real < 0:
        return -_sinpi_complex(-z)
    n, r = divmod(z.real, 0.5)
    z = pi*complex(r, z.imag)
    n %= 4
    if n == 0: return cmath.sin(z)
    if n == 1: return cmath.cos(z)
    if n == 2: return -cmath.sin(z)
    if n == 3: return -cmath.cos(z)

def _cospi_complex(z):
    if z.real < 0:
        z = -z
    n, r = divmod(z.real, 0.5)
    z = pi*complex(r, z.imag)
    n %= 4
    if n == 0: return cmath.cos(z)
    if n == 1: return -cmath.sin(z)
    if n == 2: return -cmath.cos(z)
    if n == 3: return cmath.sin(z)

cospi = _mathfun_real(_cospi_real, _cospi_complex)
sinpi = _mathfun_real(_sinpi_real, _sinpi_complex)

def tanpi(x):
    try:
        return sinpi(x) / cospi(x)
    except OverflowError:
        if complex(x).imag > 10:
            return 1j
        if complex(x).imag < 10:
            return -1j
        raise

def cotpi(x):
    try:
        return cospi(x) / sinpi(x)
    except OverflowError:
        if complex(x).imag > 10:
            return -1j
        if complex(x).imag < 10:
            return 1j
        raise

INF = 1e300*1e300
NINF = -INF
NAN = INF-INF
EPS = 2.2204460492503131e-16

_exact_gamma = (INF, 1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0,
  362880.0, 3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
  1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0,
  121645100408832000.0, 2432902008176640000.0)

_max_exact_gamma = len(_exact_gamma)-1

# Lanczos coefficients used by the GNU Scientific Library
_lanczos_g = 7
_lanczos_p = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
     771.32342877765313, -176.61502916214059, 12.507343278686905,
     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)

def _gamma_real(x):
    _intx = int(x)
    if _intx == x:
        if _intx <= 0:
            #return (-1)**_intx * INF
            raise ZeroDivisionError("gamma function pole")
        if _intx <= _max_exact_gamma:
            return _exact_gamma[_intx]
    if x < 0.5:
        # TODO: sinpi
        return pi / (_sinpi_real(x)*_gamma_real(1-x))
    else:
        x -= 1.0
        r = _lanczos_p[0]
        for i in range(1, _lanczos_g+2):
            r += _lanczos_p[i]/(x+i)
        t = x + _lanczos_g + 0.5
        return 2.506628274631000502417 * t**(x+0.5) * math.exp(-t) * r

def _gamma_complex(x):
    if not x.imag:
        return complex(_gamma_real(x.real))
    if x.real < 0.5:
        # TODO: sinpi
        return pi / (_sinpi_complex(x)*_gamma_complex(1-x))
    else:
        x -= 1.0
        r = _lanczos_p[0]
        for i in range(1, _lanczos_g+2):
            r += _lanczos_p[i]/(x+i)
        t = x + _lanczos_g + 0.5
        return 2.506628274631000502417 * t**(x+0.5) * cmath.exp(-t) * r

gamma = _mathfun_real(_gamma_real, _gamma_complex)

def rgamma(x):
    try:
        return 1./gamma(x)
    except ZeroDivisionError:
        return x*0.0

def factorial(x):
    return gamma(x+1.0)

def arg(x):
    if type(x) is float:
        return math.atan2(0.0,x)
    return math.atan2(x.imag,x.real)

# XXX: broken for negatives
def loggamma(x):
    if type(x) not in (float, complex):
        try:
            x = float(x)
        except (ValueError, TypeError):
            x = complex(x)
    try:
        xreal = x.real
        ximag = x.imag
    except AttributeError:   # py2.5
        xreal = x
        ximag = 0.0
    # Reflection formula
    # http://functions.wolfram.com/GammaBetaErf/LogGamma/16/01/01/0003/
    if xreal < 0.0:
        if abs(x) < 0.5:
            v = log(gamma(x))
            if ximag == 0:
                v = v.conjugate()
            return v
        z = 1-x
        try:
            re = z.real
            im = z.imag
        except AttributeError:   # py2.5
            re = z
            im = 0.0
        refloor = floor(re)
        if im == 0.0:
            imsign = 0
        elif im < 0.0:
            imsign = -1
        else:
            imsign = 1
        return (-pi*1j)*abs(refloor)*(1-abs(imsign)) + logpi - \
            log(sinpi(z-refloor)) - loggamma(z) + 1j*pi*refloor*imsign
    if x == 1.0 or x == 2.0:
        return x*0
    p = 0.
    while abs(x) < 11:
        p -= log(x)
        x += 1.0
    s = 0.918938533204672742 + (x-0.5)*log(x) - x
    r = 1./x
    r2 = r*r
    s += 0.083333333333333333333*r; r *= r2
    s += -0.0027777777777777777778*r; r *= r2
    s += 0.00079365079365079365079*r; r *= r2
    s += -0.0005952380952380952381*r; r *= r2
    s += 0.00084175084175084175084*r; r *= r2
    s += -0.0019175269175269175269*r; r *= r2
    s += 0.0064102564102564102564*r; r *= r2
    s += -0.02955065359477124183*r
    return s + p

_psi_coeff = [
0.083333333333333333333,
-0.0083333333333333333333,
0.003968253968253968254,
-0.0041666666666666666667,
0.0075757575757575757576,
-0.021092796092796092796,
0.083333333333333333333,
-0.44325980392156862745,
3.0539543302701197438,
-26.456212121212121212]

def _digamma_real(x):
    _intx = int(x)
    if _intx == x:
        if _intx <= 0:
            raise ZeroDivisionError("polygamma pole")
    if x < 0.5:
        x = 1.0-x
        s = pi*cotpi(x)
    else:
        s = 0.0
    while x < 10.0:
        s -= 1.0/x
        x += 1.0
    x2 = x**-2
    t = x2
    for c in _psi_coeff:
        s -= c*t
        if t < 1e-20:
            break
        t *= x2
    return s + math_log(x) - 0.5/x

def _digamma_complex(x):
    if not x.imag:
        return complex(_digamma_real(x.real))
    if x.real < 0.5:
        x = 1.0-x
        s = pi*cotpi(x)
    else:
        s = 0.0
    while abs(x) < 10.0:
        s -= 1.0/x
        x += 1.0
    x2 = x**-2
    t = x2
    for c in _psi_coeff:
        s -= c*t
        if abs(t) < 1e-20:
            break
        t *= x2
    return s + cmath.log(x) - 0.5/x

digamma = _mathfun_real(_digamma_real, _digamma_complex)

# TODO: could implement complex erf and erfc here. Need
# to find an accurate method (avoiding cancellation)
# for approx. 1 < abs(x) < 9.

_erfc_coeff_P = [
    1.0000000161203922312,
    2.1275306946297962644,
    2.2280433377390253297,
    1.4695509105618423961,
    0.66275911699770787537,
    0.20924776504163751585,
    0.045459713768411264339,
    0.0063065951710717791934,
    0.00044560259661560421715][::-1]

_erfc_coeff_Q = [
    1.0000000000000000000,
    3.2559100272784894318,
    4.9019435608903239131,
    4.4971472894498014205,
    2.7845640601891186528,
    1.2146026030046904138,
    0.37647108453729465912,
    0.080970149639040548613,
    0.011178148899483545902,
    0.00078981003831980423513][::-1]

def _polyval(coeffs, x):
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + x*p
    return p

def _erf_taylor(x):
    # Taylor series assuming 0 <= x <= 1
    x2 = x*x
    s = t = x
    n = 1
    while abs(t) > 1e-17:
        t *= x2/n
        s -= t/(n+n+1)
        n += 1
        t *= x2/n
        s += t/(n+n+1)
        n += 1
    return 1.1283791670955125739*s

def _erfc_mid(x):
    # Rational approximation assuming 0 <= x <= 9
    return exp(-x*x)*_polyval(_erfc_coeff_P,x)/_polyval(_erfc_coeff_Q,x)

def _erfc_asymp(x):
    # Asymptotic expansion assuming x >= 9
    x2 = x*x
    v = exp(-x2)/x*0.56418958354775628695
    r = t = 0.5 / x2
    s = 1.0
    for n in range(1,22,4):
        s -= t
        t *= r * (n+2)
        s += t
        t *= r * (n+4)
        if abs(t) < 1e-17:
            break
    return s * v

def erf(x):
    """
    erf of a real number.
    """
    x = float(x)
    if x != x:
        return x
    if x < 0.0:
        return -erf(-x)
    if x >= 1.0:
        if x >= 6.0:
            return 1.0
        return 1.0 - _erfc_mid(x)
    return _erf_taylor(x)

def erfc(x):
    """
    erfc of a real number.
    """
    x = float(x)
    if x != x:
        return x
    if x < 0.0:
        if x < -6.0:
            return 2.0
        return 2.0-erfc(-x)
    if x > 9.0:
        return _erfc_asymp(x)
    if x >= 1.0:
        return _erfc_mid(x)
    return 1.0 - _erf_taylor(x)

gauss42 = [\
(0.99839961899006235, 0.0041059986046490839),
(-0.99839961899006235, 0.0041059986046490839),
(0.9915772883408609, 0.009536220301748501),
(-0.9915772883408609,0.009536220301748501),
(0.97934250806374812, 0.014922443697357493),
(-0.97934250806374812, 0.014922443697357493),
(0.96175936533820439,0.020227869569052644),
(-0.96175936533820439, 0.020227869569052644),
(0.93892355735498811, 0.025422959526113047),
(-0.93892355735498811,0.025422959526113047),
(0.91095972490412735, 0.030479240699603467),
(-0.91095972490412735, 0.030479240699603467),
(0.87802056981217269,0.03536907109759211),
(-0.87802056981217269, 0.03536907109759211),
(0.8402859832618168, 0.040065735180692258),
(-0.8402859832618168,0.040065735180692258),
(0.7979620532554873, 0.044543577771965874),
(-0.7979620532554873, 0.044543577771965874),
(0.75127993568948048,0.048778140792803244),
(-0.75127993568948048, 0.048778140792803244),
(0.70049459055617114, 0.052746295699174064),
(-0.70049459055617114,0.052746295699174064),
(0.64588338886924779, 0.056426369358018376),
(-0.64588338886924779, 0.056426369358018376),
(0.58774459748510932, 0.059798262227586649),
(-0.58774459748510932, 0.059798262227586649),
(0.5263957499311922, 0.062843558045002565),
(-0.5263957499311922, 0.062843558045002565),
(0.46217191207042191, 0.065545624364908975),
(-0.46217191207042191, 0.065545624364908975),
(0.39542385204297503, 0.067889703376521934),
(-0.39542385204297503, 0.067889703376521934),
(0.32651612446541151, 0.069862992492594159),
(-0.32651612446541151, 0.069862992492594159),
(0.25582507934287907, 0.071454714265170971),
(-0.25582507934287907, 0.071454714265170971),
(0.18373680656485453, 0.072656175243804091),
(-0.18373680656485453, 0.072656175243804091),
(0.11064502720851986, 0.073460813453467527),
(-0.11064502720851986, 0.073460813453467527),
(0.036948943165351772, 0.073864234232172879),
(-0.036948943165351772, 0.073864234232172879)]

EI_ASYMP_CONVERGENCE_RADIUS = 40.0

def ei_asymp(z, _e1=False):
    r = 1./z
    s = t = 1.0
    k = 1
    while 1:
        t *= k*r
        s += t
        if abs(t) < 1e-16:
            break
        k += 1
    v = s*exp(z)/z
    if _e1:
        if type(z) is complex:
            zreal = z.real
            zimag = z.imag
        else:
            zreal = z
            zimag = 0.0
        if zimag == 0.0 and zreal > 0.0:
            v += pi*1j
    else:
        if type(z) is complex:
            if z.imag > 0:
                v += pi*1j
            if z.imag < 0:
                v -= pi*1j
    return v

def ei_taylor(z, _e1=False):
    s = t = z
    k = 2
    while 1:
        t = t*z/k
        term = t/k
        if abs(term) < 1e-17:
            break
        s += term
        k += 1
    s += euler
    if _e1:
        s += log(-z)
    else:
        if type(z) is float or z.imag == 0.0:
            s += math_log(abs(z))
        else:
            s += cmath.log(z)
    return s

def ei(z, _e1=False):
    typez = type(z)
    if typez not in (float, complex):
        try:
            z = float(z)
            typez = float
        except (TypeError, ValueError):
            z = complex(z)
            typez = complex
    if not z:
        return -INF
    absz = abs(z)
    if absz > EI_ASYMP_CONVERGENCE_RADIUS:
        return ei_asymp(z, _e1)
    elif absz <= 2.0 or (typez is float and z > 0.0):
        return ei_taylor(z, _e1)
    # Integrate, starting from whichever is smaller of a Taylor
    # series value or an asymptotic series value
    if typez is complex and z.real > 0.0:
        zref = z / absz
        ref = ei_taylor(zref, _e1)
    else:
        zref = EI_ASYMP_CONVERGENCE_RADIUS * z / absz
        ref = ei_asymp(zref, _e1)
    C = (zref-z)*0.5
    D = (zref+z)*0.5
    s = 0.0
    if type(z) is complex:
        _exp = cmath.exp
    else:
        _exp = math.exp
    for x,w in gauss42:
        t = C*x+D
        s += w*_exp(t)/t
    ref -= C*s
    return ref

def e1(z):
    # hack to get consistent signs if the imaginary part if 0
    # and signed
    typez = type(z)
    if type(z) not in (float, complex):
        try:
            z = float(z)
            typez = float
        except (TypeError, ValueError):
            z = complex(z)
            typez = complex
    if typez is complex and not z.imag:
        z = complex(z.real, 0.0)
    # end hack
    return -ei(-z, _e1=True)

_zeta_int = [\
-0.5,
0.0,
1.6449340668482264365,1.2020569031595942854,1.0823232337111381915,
1.0369277551433699263,1.0173430619844491397,1.0083492773819228268,
1.0040773561979443394,1.0020083928260822144,1.0009945751278180853,
1.0004941886041194646,1.0002460865533080483,1.0001227133475784891,
1.0000612481350587048,1.0000305882363070205,1.0000152822594086519,
1.0000076371976378998,1.0000038172932649998,1.0000019082127165539,
1.0000009539620338728,1.0000004769329867878,1.0000002384505027277,
1.0000001192199259653,1.0000000596081890513,1.0000000298035035147,
1.0000000149015548284]

_zeta_P = [-3.50000000087575873, -0.701274355654678147,
-0.0672313458590012612, -0.00398731457954257841,
-0.000160948723019303141, -4.67633010038383371e-6,
-1.02078104417700585e-7, -1.68030037095896287e-9,
-1.85231868742346722e-11][::-1]

_zeta_Q = [1.00000000000000000, -0.936552848762465319,
-0.0588835413263763741, -0.00441498861482948666,
-0.000143416758067432622, -5.10691659585090782e-6,
-9.58813053268913799e-8, -1.72963791443181972e-9,
-1.83527919681474132e-11][::-1]

_zeta_1 = [3.03768838606128127e-10, -1.21924525236601262e-8,
2.01201845887608893e-7, -1.53917240683468381e-6,
-5.09890411005967954e-7, 0.000122464707271619326,
-0.000905721539353130232, -0.00239315326074843037,
0.084239750013159168, 0.418938517907442414, 0.500000001921884009]

_zeta_0 = [-3.46092485016748794e-10, -6.42610089468292485e-9,
1.76409071536679773e-7, -1.47141263991560698e-6, -6.38880222546167613e-7,
0.000122641099800668209, -0.000905894913516772796, -0.00239303348507992713,
0.0842396947501199816, 0.418938533204660256, 0.500000000000000052]

def zeta(s):
    """
    Riemann zeta function, real argument
    """
    if not isinstance(s, (float, int)):
        try:
            s = float(s)
        except (ValueError, TypeError):
            try:
                s = complex(s)
                if not s.imag:
                    return complex(zeta(s.real))
            except (ValueError, TypeError):
                pass
            raise NotImplementedError
    if s == 1:
        raise ValueError("zeta(1) pole")
    if s >= 27:
        return 1.0 + 2.0**(-s) + 3.0**(-s)
    n = int(s)
    if n == s:
        if n >= 0:
            return _zeta_int[n]
        if not (n % 2):
            return 0.0
    if s <= 0.0:
        return 2.**s*pi**(s-1)*_sinpi_real(0.5*s)*_gamma_real(1-s)*zeta(1-s)
    if s <= 2.0:
        if s <= 1.0:
            return _polyval(_zeta_0,s)/(s-1)
        return _polyval(_zeta_1,s)/(s-1)
    z = _polyval(_zeta_P,s) / _polyval(_zeta_Q,s)
    return 1.0 + 2.0**(-s) + 3.0**(-s) + 4.0**(-s)*z
<|endoftext|>__version__ = '1.3.0'

from .usertools import monitor, timing

from .ctx_fp import FPContext
from .ctx_mp import MPContext
from .ctx_iv import MPIntervalContext

fp = FPContext()
mp = MPContext()
iv = MPIntervalContext()

fp._mp = mp
mp._mp = mp
iv._mp = mp
mp._fp = fp
fp._fp = fp
mp._iv = iv
fp._iv = iv
iv._iv = iv

# XXX: extremely bad pickle hack
from . import ctx_mp as _ctx_mp
_ctx_mp._mpf_module.mpf = mp.mpf
_ctx_mp._mpf_module.mpc = mp.mpc

make_mpf = mp.make_mpf
make_mpc = mp.make_mpc

extraprec = mp.extraprec
extradps = mp.extradps
workprec = mp.workprec
workdps = mp.workdps
autoprec = mp.autoprec
maxcalls = mp.maxcalls
memoize = mp.memoize

mag = mp.mag

bernfrac = mp.bernfrac

qfrom = mp.qfrom
mfrom = mp.mfrom
kfrom = mp.kfrom
taufrom = mp.taufrom
qbarfrom = mp.qbarfrom
ellipfun = mp.ellipfun
jtheta = mp.jtheta
kleinj = mp.kleinj
eta = mp.eta

qp = mp.qp
qhyper = mp.qhyper
qgamma = mp.qgamma
qfac = mp.qfac

nint_distance = mp.nint_distance

plot = mp.plot
cplot = mp.cplot
splot = mp.splot

odefun = mp.odefun

jacobian = mp.jacobian
findroot = mp.findroot
multiplicity = mp.multiplicity

isinf = mp.isinf
isnan = mp.isnan
isnormal = mp.isnormal
isint = mp.isint
isfinite = mp.isfinite
almosteq = mp.almosteq
nan = mp.nan
rand = mp.rand

absmin = mp.absmin
absmax = mp.absmax

fraction = mp.fraction

linspace = mp.linspace
arange = mp.arange

mpmathify = convert = mp.convert
mpc = mp.mpc

mpi = iv._mpi

nstr = mp.nstr
nprint = mp.nprint
chop = mp.chop

fneg = mp.fneg
fadd = mp.fadd
fsub = mp.fsub
fmul = mp.fmul
fdiv = mp.fdiv
fprod = mp.fprod

quad = mp.quad
quadgl = mp.quadgl
quadts = mp.quadts
quadosc = mp.quadosc
quadsubdiv = mp.quadsubdiv

invertlaplace = mp.invertlaplace
invlaptalbot = mp.invlaptalbot
invlapstehfest = mp.invlapstehfest
invlapdehoog = mp.invlapdehoog

pslq = mp.pslq
identify = mp.identify
findpoly = mp.findpoly

richardson = mp.richardson
shanks = mp.shanks
levin = mp.levin
cohen_alt = mp.cohen_alt
nsum = mp.nsum
nprod = mp.nprod
difference = mp.difference
diff = mp.diff
diffs = mp.diffs
diffs_prod = mp.diffs_prod
diffs_exp = mp.diffs_exp
diffun = mp.diffun
differint = mp.differint
taylor = mp.taylor
pade = mp.pade
polyval = mp.polyval
polyroots = mp.polyroots
fourier = mp.fourier
fourierval = mp.fourierval
sumem = mp.sumem
sumap = mp.sumap
chebyfit = mp.chebyfit
limit = mp.limit

matrix = mp.matrix
eye = mp.eye
diag = mp.diag
zeros = mp.zeros
ones = mp.ones
hilbert = mp.hilbert
randmatrix = mp.randmatrix
swap_row = mp.swap_row
extend = mp.extend
norm = mp.norm
mnorm = mp.mnorm

lu_solve = mp.lu_solve
lu = mp.lu
qr = mp.qr
unitvector = mp.unitvector
inverse = mp.inverse
residual = mp.residual
qr_solve = mp.qr_solve
cholesky = mp.cholesky
cholesky_solve = mp.cholesky_solve
det = mp.det
cond = mp.cond
hessenberg = mp.hessenberg
schur = mp.schur
eig = mp.eig
eig_sort = mp.eig_sort
eigsy = mp.eigsy
eighe = mp.eighe
eigh = mp.eigh
svd_r = mp.svd_r
svd_c = mp.svd_c
svd = mp.svd
gauss_quadrature = mp.gauss_quadrature

expm = mp.expm
sqrtm = mp.sqrtm
powm = mp.powm
logm = mp.logm
sinm = mp.sinm
cosm = mp.cosm

mpf = mp.mpf
j = mp.j
exp = mp.exp
expj = mp.expj
expjpi = mp.expjpi
ln = mp.ln
im = mp.im
re = mp.re
inf = mp.inf
ninf = mp.ninf
sign = mp.sign

eps = mp.eps
pi = mp.pi
ln2 = mp.ln2
ln10 = mp.ln10
phi = mp.phi
e = mp.e
euler = mp.euler
catalan = mp.catalan
khinchin = mp.khinchin
glaisher = mp.glaisher
apery = mp.apery
degree = mp.degree
twinprime = mp.twinprime
mertens = mp.mertens

ldexp = mp.ldexp
frexp = mp.frexp

fsum = mp.fsum
fdot = mp.fdot

sqrt = mp.sqrt
cbrt = mp.cbrt
exp = mp.exp
ln = mp.ln
log = mp.log
log10 = mp.log10
power = mp.power
cos = mp.cos
sin = mp.sin
tan = mp.tan
cosh = mp.cosh
sinh = mp.sinh
tanh = mp.tanh
acos = mp.acos
asin = mp.asin
atan = mp.atan
asinh = mp.asinh
acosh = mp.acosh
atanh = mp.atanh
sec = mp.sec
csc = mp.csc
cot = mp.cot
sech = mp.sech
csch = mp.csch
coth = mp.coth
asec = mp.asec
acsc = mp.acsc
acot = mp.acot
asech = mp.asech
acsch = mp.acsch
acoth = mp.acoth
cospi = mp.cospi
sinpi = mp.sinpi
sinc = mp.sinc
sincpi = mp.sincpi
cos_sin = mp.cos_sin
cospi_sinpi = mp.cospi_sinpi
fabs = mp.fabs
re = mp.re
im = mp.im
conj = mp.conj
floor = mp.floor
ceil = mp.ceil
nint = mp.nint
frac = mp.frac
root = mp.root
nthroot = mp.nthroot
hypot = mp.hypot
fmod = mp.fmod
ldexp = mp.ldexp
frexp = mp.frexp
sign = mp.sign
arg = mp.arg
phase = mp.phase
polar = mp.polar
rect = mp.rect
degrees = mp.degrees
radians = mp.radians
atan2 = mp.atan2
fib = mp.fib
fibonacci = mp.fibonacci
lambertw = mp.lambertw
zeta = mp.zeta
altzeta = mp.altzeta
gamma = mp.gamma
rgamma = mp.rgamma
factorial = mp.factorial
fac = mp.fac
fac2 = mp.fac2
beta = mp.beta
betainc = mp.betainc
psi = mp.psi
#psi0 = mp.psi0
#psi1 = mp.psi1
#psi2 = mp.psi2
#psi3 = mp.psi3
polygamma = mp.polygamma
digamma = mp.digamma
#trigamma = mp.trigamma
#tetragamma = mp.tetragamma
#pentagamma = mp.pentagamma
harmonic = mp.harmonic
bernoulli = mp.bernoulli
bernfrac = mp.bernfrac
stieltjes = mp.stieltjes
hurwitz = mp.hurwitz
dirichlet = mp.dirichlet
bernpoly = mp.bernpoly
eulerpoly = mp.eulerpoly
eulernum = mp.eulernum
polylog = mp.polylog
clsin = mp.clsin
clcos = mp.clcos
gammainc = mp.gammainc
gammaprod = mp.gammaprod
binomial = mp.binomial
rf = mp.rf
ff = mp.ff
hyper = mp.hyper
hyp0f1 = mp.hyp0f1
hyp1f1 = mp.hyp1f1
hyp1f2 = mp.hyp1f2
hyp2f1 = mp.hyp2f1
hyp2f2 = mp.hyp2f2
hyp2f0 = mp.hyp2f0
hyp2f3 = mp.hyp2f3
hyp3f2 = mp.hyp3f2
hyperu = mp.hyperu
hypercomb = mp.hypercomb
meijerg = mp.meijerg
appellf1 = mp.appellf1
appellf2 = mp.appellf2
appellf3 = mp.appellf3
appellf4 = mp.appellf4
hyper2d = mp.hyper2d
bihyper = mp.bihyper
erf = mp.erf
erfc = mp.erfc
erfi = mp.erfi
erfinv = mp.erfinv
npdf = mp.npdf
ncdf = mp.ncdf
expint = mp.expint
e1 = mp.e1
ei = mp.ei
li = mp.li
ci = mp.ci
si = mp.si
chi = mp.chi
shi = mp.shi
fresnels = mp.fresnels
fresnelc = mp.fresnelc
airyai = mp.airyai
airybi = mp.airybi
airyaizero = mp.airyaizero
airybizero = mp.airybizero
scorergi = mp.scorergi
scorerhi = mp.scorerhi
ellipk = mp.ellipk
ellipe = mp.ellipe
ellipf = mp.ellipf
ellippi = mp.ellippi
elliprc = mp.elliprc
elliprj = mp.elliprj
elliprf = mp.elliprf
elliprd = mp.elliprd
elliprg = mp.elliprg
agm = mp.agm
jacobi = mp.jacobi
chebyt = mp.chebyt
chebyu = mp.chebyu
legendre = mp.legendre
legenp = mp.legenp
legenq = mp.legenq
hermite = mp.hermite
pcfd = mp.pcfd
pcfu = mp.pcfu
pcfv = mp.pcfv
pcfw = mp.pcfw
gegenbauer = mp.gegenbauer
laguerre = mp.laguerre
spherharm = mp.spherharm
besselj = mp.besselj
j0 = mp.j0
j1 = mp.j1
besseli = mp.besseli
bessely = mp.bessely
besselk = mp.besselk
besseljzero = mp.besseljzero
besselyzero = mp.besselyzero
hankel1 = mp.hankel1
hankel2 = mp.hankel2
struveh = mp.struveh
struvel = mp.struvel
angerj = mp.angerj
webere = mp.webere
lommels1 = mp.lommels1
lommels2 = mp.lommels2
whitm = mp.whitm
whitw = mp.whitw
ber = mp.ber
bei = mp.bei
ker = mp.ker
kei = mp.kei
coulombc = mp.coulombc
coulombf = mp.coulombf
coulombg = mp.coulombg
barnesg = mp.barnesg
superfac = mp.superfac
hyperfac = mp.hyperfac
loggamma = mp.loggamma
siegeltheta = mp.siegeltheta
siegelz = mp.siegelz
grampoint = mp.grampoint
zetazero = mp.zetazero
riemannr = mp.riemannr
primepi = mp.primepi
primepi2 = mp.primepi2
primezeta = mp.primezeta
bell = mp.bell
polyexp = mp.polyexp
expm1 = mp.expm1
log1p = mp.log1p
powm1 = mp.powm1
unitroots = mp.unitroots
cyclotomic = mp.cyclotomic
mangoldt = mp.mangoldt
secondzeta = mp.secondzeta
nzeros = mp.nzeros
backlunds = mp.backlunds
lerchphi = mp.lerchphi
stirling1 = mp.stirling1
stirling2 = mp.stirling2
squarew = mp.squarew
trianglew = mp.trianglew
sawtoothw = mp.sawtoothw
unit_triangle = mp.unit_triangle
sigmoid = mp.sigmoid

# be careful when changing this name, don't use test*!
def runtests():
    """
    Run all mpmath tests and print output.
    """
    import os.path
    from inspect import getsourcefile
    from .tests import runtests as tests
    testdir = os.path.dirname(os.path.abspath(getsourcefile(tests)))
    importdir = os.path.abspath(testdir + '/../..')
    tests.testit(importdir, testdir)

def doctests(filter=[]):
    import sys
    from timeit import default_timer as clock
    for i, arg in enumerate(sys.argv):
        if '__init__.py' in arg:
            filter = [sn for sn in sys.argv[i+1:] if not sn.startswith("-")]
            break
    import doctest
    globs = globals().copy()
    for obj in globs: #sorted(globs.keys()):
        if filter:
            if not sum([pat in obj for pat in filter]):
                continue
        sys.stdout.write(str(obj) + " ")
        sys.stdout.flush()
        t1 = clock()
        doctest.run_docstring_examples(globs[obj], {}, verbose=("-v" in sys.argv))
        t2 = clock()
        print(round(t2-t1, 3))

if __name__ == '__main__':
    doctests()
<|endoftext|>"""
Implements the PSLQ algorithm for integer relation detection,
and derivative algorithms for constant recognition.
"""

from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed

# round to nearest integer (can be done more elegantly...)
def round_fixed(x, prec):
    return ((x + (1<<(prec-1))) >> prec) << prec

class IdentificationMethods(object):
    pass


def pslq(ctx, x, tol=None, maxcoeff=1000, maxsteps=100, verbose=False):
    r"""
    Given a vector of real numbers `x = [x_0, x_1, ..., x_n]`, ``pslq(x)``
    uses the PSLQ algorithm to find a list of integers
    `[c_0, c_1, ..., c_n]` such that

    .. math ::

        |c_1 x_1 + c_2 x_2 + ... + c_n x_n| < \mathrm{tol}

    and such that `\max |c_k| < \mathrm{maxcoeff}`. If no such vector
    exists, :func:`~mpmath.pslq` returns ``None``. The tolerance defaults to
    3/4 of the working precision.

    **Examples**

    Find rational approximations for `\pi`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> pslq([-1, pi], tol=0.01)
        [22, 7]
        >>> pslq([-1, pi], tol=0.001)
        [355, 113]
        >>> mpf(22)/7; mpf(355)/113; +pi
        3.14285714285714
        3.14159292035398
        3.14159265358979

    Pi is not a rational number with denominator less than 1000::

        >>> pslq([-1, pi])
        >>>

    To within the standard precision, it can however be approximated
    by at least one rational number with denominator less than `10^{12}`::

        >>> p, q = pslq([-1, pi], maxcoeff=10**12)
        >>> print(p); print(q)
        238410049439
        75888275702
        >>> mpf(p)/q
        3.14159265358979

    The PSLQ algorithm can be applied to long vectors. For example,
    we can investigate the rational (in)dependence of integer square
    roots::

        >>> mp.dps = 30
        >>> pslq([sqrt(n) for n in range(2, 5+1)])
        >>>
        >>> pslq([sqrt(n) for n in range(2, 6+1)])
        >>>
        >>> pslq([sqrt(n) for n in range(2, 8+1)])
        [2, 0, 0, 0, 0, 0, -1]

    **Machin formulas**

    A famous formula for `\pi` is Machin's,

    .. math ::

        \frac{\pi}{4} = 4 \operatorname{acot} 5 - \operatorname{acot} 239

    There are actually infinitely many formulas of this type. Two
    others are

    .. math ::

        \frac{\pi}{4} = \operatorname{acot} 1

        \frac{\pi}{4} = 12 \operatorname{acot} 49 + 32 \operatorname{acot} 57
            + 5 \operatorname{acot} 239 + 12 \operatorname{acot} 110443

    We can easily verify the formulas using the PSLQ algorithm::

        >>> mp.dps = 30
        >>> pslq([pi/4, acot(1)])
        [1, -1]
        >>> pslq([pi/4, acot(5), acot(239)])
        [1, -4, 1]
        >>> pslq([pi/4, acot(49), acot(57), acot(239), acot(110443)])
        [1, -12, -32, 5, -12]

    We could try to generate a custom Machin-like formula by running
    the PSLQ algorithm with a few inverse cotangent values, for example
    acot(2), acot(3) ... acot(10). Unfortunately, there is a linear
    dependence among these values, resulting in only that dependence
    being detected, with a zero coefficient for `\pi`::

        >>> pslq([pi] + [acot(n) for n in range(2,11)])
        [0, 1, -1, 0, 0, 0, -1, 0, 0, 0]

    We get better luck by removing linearly dependent terms::

        >>> pslq([pi] + [acot(n) for n in range(2,11) if n not in (3, 5)])
        [1, -8, 0, 0, 4, 0, 0, 0]

    In other words, we found the following formula::

        >>> 8*acot(2) - 4*acot(7)
        3.14159265358979323846264338328
        >>> +pi
        3.14159265358979323846264338328

    **Algorithm**

    This is a fairly direct translation to Python of the pseudocode given by
    David Bailey, "The PSLQ Integer Relation Algorithm":
    http://www.cecm.sfu.ca/organics/papers/bailey/paper/html/node3.html

    The present implementation uses fixed-point instead of floating-point
    arithmetic, since this is significantly (about 7x) faster.
    """

    n = len(x)
    if n < 2:
        raise ValueError("n cannot be less than 2")

    # At too low precision, the algorithm becomes meaningless
    prec = ctx.prec
    if prec < 53:
        raise ValueError("prec cannot be less than 53")

    if verbose and prec // max(2,n) < 5:
        print("Warning: precision for PSLQ may be too low")

    target = int(prec * 0.75)

    if tol is None:
        tol = ctx.mpf(2)**(-target)
    else:
        tol = ctx.convert(tol)

    extra = 60
    prec += extra

    if verbose:
        print("PSLQ using prec %i and tol %s" % (prec, ctx.nstr(tol)))

    tol = ctx.to_fixed(tol, prec)
    assert tol

    # Convert to fixed-point numbers. The dummy None is added so we can
    # use 1-based indexing. (This just allows us to be consistent with
    # Bailey's indexing. The algorithm is 100 lines long, so debugging
    # a single wrong index can be painful.)
    x = [None] + [ctx.to_fixed(ctx.mpf(xk), prec) for xk in x]

    # Sanity check on magnitudes
    minx = min(abs(xx) for xx in x[1:])
    if not minx:
        raise ValueError("PSLQ requires a vector of nonzero numbers")
    if minx < tol//100:
        if verbose:
            print("STOPPING: (one number is too small)")
        return None

    g = sqrt_fixed((4<<prec)//3, prec)
    A = {}
    B = {}
    H = {}
    # Initialization
    # step 1
    for i in xrange(1, n+1):
        for j in xrange(1, n+1):
            A[i,j] = B[i,j] = (i==j) << prec
            H[i,j] = 0
    # step 2
    s = [None] + [0] * n
    for k in xrange(1, n+1):
        t = 0
        for j in xrange(k, n+1):
            t += (x[j]**2 >> prec)
        s[k] = sqrt_fixed(t, prec)
    t = s[1]
    y = x[:]
    for k in xrange(1, n+1):
        y[k] = (x[k] << prec) // t
        s[k] = (s[k] << prec) // t
    # step 3
    for i in xrange(1, n+1):
        for j in xrange(i+1, n):
            H[i,j] = 0
        if i <= n-1:
            if s[i]:
                H[i,i] = (s[i+1] << prec) // s[i]
            else:
                H[i,i] = 0
        for j in range(1, i):
            sjj1 = s[j]*s[j+1]
            if sjj1:
                H[i,j] = ((-y[i]*y[j])<<prec)//sjj1
            else:
                H[i,j] = 0
    # step 4
    for i in xrange(2, n+1):
        for j in xrange(i-1, 0, -1):
            #t = floor(H[i,j]/H[j,j] + 0.5)
            if H[j,j]:
                t = round_fixed((H[i,j] << prec)//H[j,j], prec)
            else:
                #t = 0
                continue
            y[j] = y[j] + (t*y[i] >> prec)
            for k in xrange(1, j+1):
                H[i,k] = H[i,k] - (t*H[j,k] >> prec)
            for k in xrange(1, n+1):
                A[i,k] = A[i,k] - (t*A[j,k] >> prec)
                B[k,j] = B[k,j] + (t*B[k,i] >> prec)
    # Main algorithm
    for REP in range(maxsteps):
        # Step 1
        m = -1
        szmax = -1
        for i in range(1, n):
            h = H[i,i]
            sz = (g**i * abs(h)) >> (prec*(i-1))
            if sz > szmax:
                m = i
                szmax = sz
        # Step 2
        y[m], y[m+1] = y[m+1], y[m]
        for i in xrange(1,n+1): H[m,i], H[m+1,i] = H[m+1,i], H[m,i]
        for i in xrange(1,n+1): A[m,i], A[m+1,i] = A[m+1,i], A[m,i]
        for i in xrange(1,n+1): B[i,m], B[i,m+1] = B[i,m+1], B[i,m]
        # Step 3
        if m <= n - 2:
            t0 = sqrt_fixed((H[m,m]**2 + H[m,m+1]**2)>>prec, prec)
            # A zero element probably indicates that the precision has
            # been exhausted. XXX: this could be spurious, due to
            # using fixed-point arithmetic
            if not t0:
                break
            t1 = (H[m,m] << prec) // t0
            t2 = (H[m,m+1] << prec) // t0
            for i in xrange(m, n+1):
                t3 = H[i,m]
                t4 = H[i,m+1]
                H[i,m] = (t1*t3+t2*t4) >> prec
                H[i,m+1] = (-t2*t3+t1*t4) >> prec
        # Step 4
        for i in xrange(m+1, n+1):
            for j in xrange(min(i-1, m+1), 0, -1):
                try:
                    t = round_fixed((H[i,j] << prec)//H[j,j], prec)
                # Precision probably exhausted
                except ZeroDivisionError:
                    break
                y[j] = y[j] + ((t*y[i]) >> prec)
                for k in xrange(1, j+1):
                    H[i,k] = H[i,k] - (t*H[j,k] >> prec)
                for k in xrange(1, n+1):
                    A[i,k] = A[i,k] - (t*A[j,k] >> prec)
                    B[k,j] = B[k,j] + (t*B[k,i] >> prec)
        # Until a relation is found, the error typically decreases
        # slowly (e.g. a factor 1-10) with each step TODO: we could
        # compare err from two successive iterations. If there is a
        # large drop (several orders of magnitude), that indicates a
        # "high quality" relation was detected. Reporting this to
        # the user somehow might be useful.
        best_err = maxcoeff<<prec
        for i in xrange(1, n+1):
            err = abs(y[i])
            # Maybe we are done?
            if err < tol:
                # We are done if the coefficients are acceptable
                vec = [int(round_fixed(B[j,i], prec) >> prec) for j in \
                range(1,n+1)]
                if max(abs(v) for v in vec) < maxcoeff:
                    if verbose:
                        print("FOUND relation at iter %i/%i, error: %s" % \
                            (REP, maxsteps, ctx.nstr(err / ctx.mpf(2)**prec, 1)))
                    return vec
            best_err = min(err, best_err)
        # Calculate a lower bound for the norm. We could do this
        # more exactly (using the Euclidean norm) but there is probably
        # no practical benefit.
        recnorm = max(abs(h) for h in H.values())
        if recnorm:
            norm = ((1 << (2*prec)) // recnorm) >> prec
            norm //= 100
        else:
            norm = ctx.inf
        if verbose:
            print("%i/%i:  Error: %8s   Norm: %s" % \
                (REP, maxsteps, ctx.nstr(best_err / ctx.mpf(2)**prec, 1), norm))
        if norm >= maxcoeff:
            break
    if verbose:
        print("CANCELLING after step %i/%i." % (REP, maxsteps))
        print("Could not find an integer relation. Norm bound: %s" % norm)
    return None

def findpoly(ctx, x, n=1, **kwargs):
    r"""
    ``findpoly(x, n)`` returns the coefficients of an integer
    polynomial `P` of degree at most `n` such that `P(x) \approx 0`.
    If no polynomial having `x` as a root can be found,
    :func:`~mpmath.findpoly` returns ``None``.

    :func:`~mpmath.findpoly` works by successively calling :func:`~mpmath.pslq` with
    the vectors `[1, x]`, `[1, x, x^2]`, `[1, x, x^2, x^3]`, ...,
    `[1, x, x^2, .., x^n]` as input. Keyword arguments given to
    :func:`~mpmath.findpoly` are forwarded verbatim to :func:`~mpmath.pslq`. In
    particular, you can specify a tolerance for `P(x)` with ``tol``
    and a maximum permitted coefficient size with ``maxcoeff``.

    For large values of `n`, it is recommended to run :func:`~mpmath.findpoly`
    at high precision; preferably 50 digits or more.

    **Examples**

    By default (degree `n = 1`), :func:`~mpmath.findpoly` simply finds a linear
    polynomial with a rational root::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> findpoly(0.7)
        [-10, 7]

    The generated coefficient list is valid input to ``polyval`` and
    ``polyroots``::

        >>> nprint(polyval(findpoly(phi, 2), phi), 1)
        -2.0e-16
        >>> for r in polyroots(findpoly(phi, 2)):
        ...     print(r)
        ...
        -0.618033988749895
        1.61803398874989

    Numbers of the form `m + n \sqrt p` for integers `(m, n, p)` are
    solutions to quadratic equations. As we find here, `1+\sqrt 2`
    is a root of the polynomial `x^2 - 2x - 1`::

        >>> findpoly(1+sqrt(2), 2)
        [1, -2, -1]
        >>> findroot(lambda x: x**2 - 2*x - 1, 1)
        2.4142135623731

    Despite only containing square roots, the following number results
    in a polynomial of degree 4::

        >>> findpoly(sqrt(2)+sqrt(3), 4)
        [1, 0, -10, 0, 1]

    In fact, `x^4 - 10x^2 + 1` is the *minimal polynomial* of
    `r = \sqrt 2 + \sqrt 3`, meaning that a rational polynomial of
    lower degree having `r` as a root does not exist. Given sufficient
    precision, :func:`~mpmath.findpoly` will usually find the correct
    minimal polynomial of a given algebraic number.

    **Non-algebraic numbers**

    If :func:`~mpmath.findpoly` fails to find a polynomial with given
    coefficient size and tolerance constraints, that means no such
    polynomial exists.

    We can verify that `\pi` is not an algebraic number of degree 3 with
    coefficients less than 1000::

        >>> mp.dps = 15
        >>> findpoly(pi, 3)
        >>>

    It is always possible to find an algebraic approximation of a number
    using one (or several) of the following methods:

        1. Increasing the permitted degree
        2. Allowing larger coefficients
        3. Reducing the tolerance

    One example of each method is shown below::

        >>> mp.dps = 15
        >>> findpoly(pi, 4)
        [95, -545, 863, -183, -298]
        >>> findpoly(pi, 3, maxcoeff=10000)
        [836, -1734, -2658, -457]
        >>> findpoly(pi, 3, tol=1e-7)
        [-4, 22, -29, -2]

    It is unknown whether Euler's constant is transcendental (or even
    irrational). We can use :func:`~mpmath.findpoly` to check that if is
    an algebraic number, its minimal polynomial must have degree
    at least 7 and a coefficient of magnitude at least 1000000::

        >>> mp.dps = 200
        >>> findpoly(euler, 6, maxcoeff=10**6, tol=1e-100, maxsteps=1000)
        >>>

    Note that the high precision and strict tolerance is necessary
    for such high-degree runs, since otherwise unwanted low-accuracy
    approximations will be detected. It may also be necessary to set
    maxsteps high to prevent a premature exit (before the coefficient
    bound has been reached). Running with ``verbose=True`` to get an
    idea what is happening can be useful.
    """
    x = ctx.mpf(x)
    if n < 1:
        raise ValueError("n cannot be less than 1")
    if x == 0:
        return [1, 0]
    xs = [ctx.mpf(1)]
    for i in range(1,n+1):
        xs.append(x**i)
        a = ctx.pslq(xs, **kwargs)
        if a is not None:
            return a[::-1]

def fracgcd(p, q):
    x, y = p, q
    while y:
        x, y = y, x % y
    if x != 1:
        p //= x
        q //= x
    if q == 1:
        return p
    return p, q

def pslqstring(r, constants):
    q = r[0]
    r = r[1:]
    s = []
    for i in range(len(r)):
        p = r[i]
        if p:
            z = fracgcd(-p,q)
            cs = constants[i][1]
            if cs == '1':
                cs = ''
            else:
                cs = '*' + cs
            if isinstance(z, int_types):
                if z > 0: term = str(z) + cs
                else:     term = ("(%s)" % z) + cs
            else:
                term = ("(%s/%s)" % z) + cs
            s.append(term)
    s = ' + '.join(s)
    if '+' in s or '*' in s:
        s = '(' + s + ')'
    return s or '0'

def prodstring(r, constants):
    q = r[0]
    r = r[1:]
    num = []
    den = []
    for i in range(len(r)):
        p = r[i]
        if p:
            z = fracgcd(-p,q)
            cs = constants[i][1]
            if isinstance(z, int_types):
                if abs(z) == 1: t = cs
                else:           t = '%s**%s' % (cs, abs(z))
                ([num,den][z<0]).append(t)
            else:
                t = '%s**(%s/%s)' % (cs, abs(z[0]), z[1])
                ([num,den][z[0]<0]).append(t)
    num = '*'.join(num)
    den = '*'.join(den)
    if num and den: return "(%s)/(%s)" % (num, den)
    if num: return num
    if den: return "1/(%s)" % den

def quadraticstring(ctx,t,a,b,c):
    if c < 0:
        a,b,c = -a,-b,-c
    u1 = (-b+ctx.sqrt(b**2-4*a*c))/(2*c)
    u2 = (-b-ctx.sqrt(b**2-4*a*c))/(2*c)
    if abs(u1-t) < abs(u2-t):
        if b:  s = '((%s+sqrt(%s))/%s)' % (-b,b**2-4*a*c,2*c)
        else:  s = '(sqrt(%s)/%s)' % (-4*a*c,2*c)
    else:
        if b:  s = '((%s-sqrt(%s))/%s)' % (-b,b**2-4*a*c,2*c)
        else:  s = '(-sqrt(%s)/%s)' % (-4*a*c,2*c)
    return s

# Transformation y = f(x,c), with inverse function x = f(y,c)
# The third entry indicates whether the transformation is
# redundant when c = 1
transforms = [
  (lambda ctx,x,c: x*c, '$y/$c', 0),
  (lambda ctx,x,c: x/c, '$c*$y', 1),
  (lambda ctx,x,c: c/x, '$c/$y', 0),
  (lambda ctx,x,c: (x*c)**2, 'sqrt($y)/$c', 0),
  (lambda ctx,x,c: (x/c)**2, '$c*sqrt($y)', 1),
  (lambda ctx,x,c: (c/x)**2, '$c/sqrt($y)', 0),
  (lambda ctx,x,c: c*x**2, 'sqrt($y)/sqrt($c)', 1),
  (lambda ctx,x,c: x**2/c, 'sqrt($c)*sqrt($y)', 1),
  (lambda ctx,x,c: c/x**2, 'sqrt($c)/sqrt($y)', 1),
  (lambda ctx,x,c: ctx.sqrt(x*c), '$y**2/$c', 0),
  (lambda ctx,x,c: ctx.sqrt(x/c), '$c*$y**2', 1),
  (lambda ctx,x,c: ctx.sqrt(c/x), '$c/$y**2', 0),
  (lambda ctx,x,c: c*ctx.sqrt(x), '$y**2/$c**2', 1),
  (lambda ctx,x,c: ctx.sqrt(x)/c, '$c**2*$y**2', 1),
  (lambda ctx,x,c: c/ctx.sqrt(x), '$c**2/$y**2', 1),
  (lambda ctx,x,c: ctx.exp(x*c), 'log($y)/$c', 0),
  (lambda ctx,x,c: ctx.exp(x/c), '$c*log($y)', 1),
  (lambda ctx,x,c: ctx.exp(c/x), '$c/log($y)', 0),
  (lambda ctx,x,c: c*ctx.exp(x), 'log($y/$c)', 1),
  (lambda ctx,x,c: ctx.exp(x)/c, 'log($c*$y)', 1),
  (lambda ctx,x,c: c/ctx.exp(x), 'log($c/$y)', 0),
  (lambda ctx,x,c: ctx.ln(x*c), 'exp($y)/$c', 0),
  (lambda ctx,x,c: ctx.ln(x/c), '$c*exp($y)', 1),
  (lambda ctx,x,c: ctx.ln(c/x), '$c/exp($y)', 0),
  (lambda ctx,x,c: c*ctx.ln(x), 'exp($y/$c)', 1),
  (lambda ctx,x,c: ctx.ln(x)/c, 'exp($c*$y)', 1),
  (lambda ctx,x,c: c/ctx.ln(x), 'exp($c/$y)', 0),
]

def identify(ctx, x, constants=[], tol=None, maxcoeff=1000, full=False,
    verbose=False):
    r"""
    Given a real number `x`, ``identify(x)`` attempts to find an exact
    formula for `x`. This formula is returned as a string. If no match
    is found, ``None`` is returned. With ``full=True``, a list of
    matching formulas is returned.

    As a simple example, :func:`~mpmath.identify` will find an algebraic
    formula for the golden ratio::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> identify(phi)
        '((1+sqrt(5))/2)'

    :func:`~mpmath.identify` can identify simple algebraic numbers and simple
    combinations of given base constants, as well as certain basic
    transformations thereof. More specifically, :func:`~mpmath.identify`
    looks for the following:

        1. Fractions
        2. Quadratic algebraic numbers
        3. Rational linear combinations of the base constants
        4. Any of the above after first transforming `x` into `f(x)` where
           `f(x)` is `1/x`, `\sqrt x`, `x^2`, `\log x` or `\exp x`, either
           directly or with `x` or `f(x)` multiplied or divided by one of
           the base constants
        5. Products of fractional powers of the base constants and
           small integers

    Base constants can be given as a list of strings representing mpmath
    expressions (:func:`~mpmath.identify` will ``eval`` the strings to numerical
    values and use the original strings for the output), or as a dict of
    formula:value pairs.

    In order not to produce spurious results, :func:`~mpmath.identify` should
    be used with high precision; preferably 50 digits or more.

    **Examples**

    Simple identifications can be performed safely at standard
    precision. Here the default recognition of rational, algebraic,
    and exp/log of algebraic numbers is demonstrated::

        >>> mp.dps = 15
        >>> identify(0.22222222222222222)
        '(2/9)'
        >>> identify(1.9662210973805663)
        'sqrt(((24+sqrt(48))/8))'
        >>> identify(4.1132503787829275)
        'exp((sqrt(8)/2))'
        >>> identify(0.881373587019543)
        'log(((2+sqrt(8))/2))'

    By default, :func:`~mpmath.identify` does not recognize `\pi`. At standard
    precision it finds a not too useful approximation. At slightly
    increased precision, this approximation is no longer accurate
    enough and :func:`~mpmath.identify` more correctly returns ``None``::

        >>> identify(pi)
        '(2**(176/117)*3**(20/117)*5**(35/39))/(7**(92/117))'
        >>> mp.dps = 30
        >>> identify(pi)
        >>>

    Numbers such as `\pi`, and simple combinations of user-defined
    constants, can be identified if they are provided explicitly::

        >>> identify(3*pi-2*e, ['pi', 'e'])
        '(3*pi + (-2)*e)'

    Here is an example using a dict of constants. Note that the
    constants need not be "atomic"; :func:`~mpmath.identify` can just
    as well express the given number in terms of expressions
    given by formulas::

        >>> identify(pi+e, {'a':pi+2, 'b':2*e})
        '((-2) + 1*a + (1/2)*b)'

    Next, we attempt some identifications with a set of base constants.
    It is necessary to increase the precision a bit.

        >>> mp.dps = 50
        >>> base = ['sqrt(2)','pi','log(2)']
        >>> identify(0.25, base)
        '(1/4)'
        >>> identify(3*pi + 2*sqrt(2) + 5*log(2)/7, base)
        '(2*sqrt(2) + 3*pi + (5/7)*log(2))'
        >>> identify(exp(pi+2), base)
        'exp((2 + 1*pi))'
        >>> identify(1/(3+sqrt(2)), base)
        '((3/7) + (-1/7)*sqrt(2))'
        >>> identify(sqrt(2)/(3*pi+4), base)
        'sqrt(2)/(4 + 3*pi)'
        >>> identify(5**(mpf(1)/3)*pi*log(2)**2, base)
        '5**(1/3)*pi*log(2)**2'

    An example of an erroneous solution being found when too low
    precision is used::

        >>> mp.dps = 15
        >>> identify(1/(3*pi-4*e+sqrt(8)), ['pi', 'e', 'sqrt(2)'])
        '((11/25) + (-158/75)*pi + (76/75)*e + (44/15)*sqrt(2))'
        >>> mp.dps = 50
        >>> identify(1/(3*pi-4*e+sqrt(8)), ['pi', 'e', 'sqrt(2)'])
        '1/(3*pi + (-4)*e + 2*sqrt(2))'

    **Finding approximate solutions**

    The tolerance ``tol`` defaults to 3/4 of the working precision.
    Lowering the tolerance is useful for finding approximate matches.
    We can for example try to generate approximations for pi::

        >>> mp.dps = 15
        >>> identify(pi, tol=1e-2)
        '(22/7)'
        >>> identify(pi, tol=1e-3)
        '(355/113)'
        >>> identify(pi, tol=1e-10)
        '(5**(339/269))/(2**(64/269)*3**(13/269)*7**(92/269))'

    With ``full=True``, and by supplying a few base constants,
    ``identify`` can generate almost endless lists of approximations
    for any number (the output below has been truncated to show only
    the first few)::

        >>> for p in identify(pi, ['e', 'catalan'], tol=1e-5, full=True):
        ...     print(p)
        ...  # doctest: +ELLIPSIS
        e/log((6 + (-4/3)*e))
        (3**3*5*e*catalan**2)/(2*7**2)
        sqrt(((-13) + 1*e + 22*catalan))
        log(((-6) + 24*e + 4*catalan)/e)
        exp(catalan*((-1/5) + (8/15)*e))
        catalan*(6 + (-6)*e + 15*catalan)
        sqrt((5 + 26*e + (-3)*catalan))/e
        e*sqrt(((-27) + 2*e + 25*catalan))
        log(((-1) + (-11)*e + 59*catalan))
        ((3/20) + (21/20)*e + (3/20)*catalan)
        ...

    The numerical values are roughly as close to `\pi` as permitted by the
    specified tolerance:

        >>> e/log(6-4*e/3)
        3.14157719846001
        >>> 135*e*catalan**2/98
        3.14166950419369
        >>> sqrt(e-13+22*catalan)
        3.14158000062992
        >>> log(24*e-6+4*catalan)-1
        3.14158791577159

    **Symbolic processing**

    The output formula can be evaluated as a Python expression.
    Note however that if fractions (like '2/3') are present in
    the formula, Python's :func:`~mpmath.eval()` may erroneously perform
    integer division. Note also that the output is not necessarily
    in the algebraically simplest form::

        >>> identify(sqrt(2))
        '(sqrt(8)/2)'

    As a solution to both problems, consider using SymPy's
    :func:`~mpmath.sympify` to convert the formula into a symbolic expression.
    SymPy can be used to pretty-print or further simplify the formula
    symbolically::

        >>> from sympy import sympify # doctest: +SKIP
        >>> sympify(identify(sqrt(2))) # doctest: +SKIP
        2**(1/2)

    Sometimes :func:`~mpmath.identify` can simplify an expression further than
    a symbolic algorithm::

        >>> from sympy import simplify # doctest: +SKIP
        >>> x = sympify('-1/(-3/2+(1/2)*5**(1/2))*(3/2-1/2*5**(1/2))**(1/2)') # doctest: +SKIP
        >>> x # doctest: +SKIP
        (3/2 - 5**(1/2)/2)**(-1/2)
        >>> x = simplify(x) # doctest: +SKIP
        >>> x # doctest: +SKIP
        2/(6 - 2*5**(1/2))**(1/2)
        >>> mp.dps = 30 # doctest: +SKIP
        >>> x = sympify(identify(x.evalf(30))) # doctest: +SKIP
        >>> x # doctest: +SKIP
        1/2 + 5**(1/2)/2

    (In fact, this functionality is available directly in SymPy as the
    function :func:`~mpmath.nsimplify`, which is essentially a wrapper for
    :func:`~mpmath.identify`.)

    **Miscellaneous issues and limitations**

    The input `x` must be a real number. All base constants must be
    positive real numbers and must not be rationals or rational linear
    combinations of each other.

    The worst-case computation time grows quickly with the number of
    base constants. Already with 3 or 4 base constants,
    :func:`~mpmath.identify` may require several seconds to finish. To search
    for relations among a large number of constants, you should
    consider using :func:`~mpmath.pslq` directly.

    The extended transformations are applied to x, not the constants
    separately. As a result, ``identify`` will for example be able to
    recognize ``exp(2*pi+3)`` with ``pi`` given as a base constant, but
    not ``2*exp(pi)+3``. It will be able to recognize the latter if
    ``exp(pi)`` is given explicitly as a base constant.

    """

    solutions = []

    def addsolution(s):
        if verbose: print("Found: ", s)
        solutions.append(s)

    x = ctx.mpf(x)

    # Further along, x will be assumed positive
    if x == 0:
        if full: return ['0']
        else:    return '0'
    if x < 0:
        sol = ctx.identify(-x, constants, tol, maxcoeff, full, verbose)
        if sol is None:
            return sol
        if full:
            return ["-(%s)"%s for s in sol]
        else:
            return "-(%s)" % sol

    if tol:
        tol = ctx.mpf(tol)
    else:
        tol = ctx.eps**0.7
    M = maxcoeff

    if constants:
        if isinstance(constants, dict):
            constants = [(ctx.mpf(v), name) for (name, v) in sorted(constants.items())]
        else:
            namespace = dict((name, getattr(ctx,name)) for name in dir(ctx))
            constants = [(eval(p, namespace), p) for p in constants]
    else:
        constants = []

    # We always want to find at least rational terms
    if 1 not in [value for (name, value) in constants]:
        constants = [(ctx.mpf(1), '1')] + constants

    # PSLQ with simple algebraic and functional transformations
    for ft, ftn, red in transforms:
        for c, cn in constants:
            if red and cn == '1':
                continue
            t = ft(ctx,x,c)
            # Prevent exponential transforms from wreaking havoc
            if abs(t) > M**2 or abs(t) < tol:
                continue
            # Linear combination of base constants
            r = ctx.pslq([t] + [a[0] for a in constants], tol, M)
            s = None
            if r is not None and max(abs(uw) for uw in r) <= M and r[0]:
                s = pslqstring(r, constants)
            # Quadratic algebraic numbers
            else:
                q = ctx.pslq([ctx.one, t, t**2], tol, M)
                if q is not None and len(q) == 3 and q[2]:
                    aa, bb, cc = q
                    if max(abs(aa),abs(bb),abs(cc)) <= M:
                        s = quadraticstring(ctx,t,aa,bb,cc)
            if s:
                if cn == '1' and ('/$c' in ftn):
                    s = ftn.replace('$y', s).replace('/$c', '')
                else:
                    s = ftn.replace('$y', s).replace('$c', cn)
                addsolution(s)
                if not full: return solutions[0]

            if verbose:
                print(".")

    # Check for a direct multiplicative formula
    if x != 1:
        # Allow fractional powers of fractions
        ilogs = [2,3,5,7]
        # Watch out for existing fractional powers of fractions
        logs = []
        for a, s in constants:
            if not sum(bool(ctx.findpoly(ctx.ln(a)/ctx.ln(i),1)) for i in ilogs):
                logs.append((ctx.ln(a), s))
        logs = [(ctx.ln(i),str(i)) for i in ilogs] + logs
        r = ctx.pslq([ctx.ln(x)] + [a[0] for a in logs], tol, M)
        if r is not None and max(abs(uw) for uw in r) <= M and r[0]:
            addsolution(prodstring(r, logs))
            if not full: return solutions[0]

    if full:
        return sorted(solutions, key=len)
    else:
        return None

IdentificationMethods.pslq = pslq
IdentificationMethods.findpoly = findpoly
IdentificationMethods.identify = identify


if __name__ == '__main__':
    import doctest
    doctest.testmod()
<|endoftext|>"""
Plotting (requires matplotlib)
"""

from colorsys import hsv_to_rgb, hls_to_rgb
from .libmp import NoConvergence
from .libmp.backend import xrange

class VisualizationMethods(object):
    plot_ignore = (ValueError, ArithmeticError, ZeroDivisionError, NoConvergence)

def plot(ctx, f, xlim=[-5,5], ylim=None, points=200, file=None, dpi=None,
    singularities=[], axes=None):
    r"""
    Shows a simple 2D plot of a function `f(x)` or list of functions
    `[f_0(x), f_1(x), \ldots, f_n(x)]` over a given interval
    specified by *xlim*. Some examples::

        plot(lambda x: exp(x)*li(x), [1, 4])
        plot([cos, sin], [-4, 4])
        plot([fresnels, fresnelc], [-4, 4])
        plot([sqrt, cbrt], [-4, 4])
        plot(lambda t: zeta(0.5+t*j), [-20, 20])
        plot([floor, ceil, abs, sign], [-5, 5])

    Points where the function raises a numerical exception or
    returns an infinite value are removed from the graph.
    Singularities can also be excluded explicitly
    as follows (useful for removing erroneous vertical lines)::

        plot(cot, ylim=[-5, 5])   # bad
        plot(cot, ylim=[-5, 5], singularities=[-pi, 0, pi])  # good

    For parts where the function assumes complex values, the
    real part is plotted with dashes and the imaginary part
    is plotted with dots.

    .. note :: This function requires matplotlib (pylab).
    """
    if file:
        axes = None
    fig = None
    if not axes:
        import pylab
        fig = pylab.figure()
        axes = fig.add_subplot(111)
    if not isinstance(f, (tuple, list)):
        f = [f]
    a, b = xlim
    colors = ['b', 'r', 'g', 'm', 'k']
    for n, func in enumerate(f):
        x = ctx.arange(a, b, (b-a)/float(points))
        segments = []
        segment = []
        in_complex = False
        for i in xrange(len(x)):
            try:
                if i != 0:
                    for sing in singularities:
                        if x[i-1] <= sing and x[i] >= sing:
                            raise ValueError
                v = func(x[i])
                if ctx.isnan(v) or abs(v) > 1e300:
                    raise ValueError
                if hasattr(v, "imag") and v.imag:
                    re = float(v.real)
                    im = float(v.imag)
                    if not in_complex:
                        in_complex = True
                        segments.append(segment)
                        segment = []
                    segment.append((float(x[i]), re, im))
                else:
                    if in_complex:
                        in_complex = False
                        segments.append(segment)
                        segment = []
                    if hasattr(v, "real"):
                        v = v.real
                    segment.append((float(x[i]), v))
            except ctx.plot_ignore:
                if segment:
                    segments.append(segment)
                segment = []
        if segment:
            segments.append(segment)
        for segment in segments:
            x = [s[0] for s in segment]
            y = [s[1] for s in segment]
            if not x:
                continue
            c = colors[n % len(colors)]
            if len(segment[0]) == 3:
                z = [s[2] for s in segment]
                axes.plot(x, y, '--'+c, linewidth=3)
                axes.plot(x, z, ':'+c, linewidth=3)
            else:
                axes.plot(x, y, c, linewidth=3)
    axes.set_xlim([float(_) for _ in xlim])
    if ylim:
        axes.set_ylim([float(_) for _ in ylim])
    axes.set_xlabel('x')
    axes.set_ylabel('f(x)')
    axes.grid(True)
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()

def default_color_function(ctx, z):
    if ctx.isinf(z):
        return (1.0, 1.0, 1.0)
    if ctx.isnan(z):
        return (0.5, 0.5, 0.5)
    pi = 3.1415926535898
    a = (float(ctx.arg(z)) + ctx.pi) / (2*ctx.pi)
    a = (a + 0.5) % 1.0
    b = 1.0 - float(1/(1.0+abs(z)**0.3))
    return hls_to_rgb(a, b, 0.8)

blue_orange_colors = [
  (-1.0,  (0.0, 0.0, 0.0)),
  (-0.95, (0.1, 0.2, 0.5)),   # dark blue
  (-0.5,  (0.0, 0.5, 1.0)),   # blueish
  (-0.05, (0.4, 0.8, 0.8)),   # cyanish
  ( 0.0,  (1.0, 1.0, 1.0)),
  ( 0.05, (1.0, 0.9, 0.3)),   # yellowish
  ( 0.5,  (0.9, 0.5, 0.0)),   # orangeish
  ( 0.95, (0.7, 0.1, 0.0)),   # redish
  ( 1.0,  (0.0, 0.0, 0.0)),
  ( 2.0,  (0.0, 0.0, 0.0)),
]

def phase_color_function(ctx, z):
    if ctx.isinf(z):
        return (1.0, 1.0, 1.0)
    if ctx.isnan(z):
        return (0.5, 0.5, 0.5)
    pi = 3.1415926535898
    w = float(ctx.arg(z)) / pi
    w = max(min(w, 1.0), -1.0)
    for i in range(1,len(blue_orange_colors)):
        if blue_orange_colors[i][0] > w:
            a, (ra, ga, ba) = blue_orange_colors[i-1]
            b, (rb, gb, bb) = blue_orange_colors[i]
            s = (w-a) / (b-a)
            return ra+(rb-ra)*s, ga+(gb-ga)*s, ba+(bb-ba)*s

def cplot(ctx, f, re=[-5,5], im=[-5,5], points=2000, color=None,
    verbose=False, file=None, dpi=None, axes=None):
    """
    Plots the given complex-valued function *f* over a rectangular part
    of the complex plane specified by the pairs of intervals *re* and *im*.
    For example::

        cplot(lambda z: z, [-2, 2], [-10, 10])
        cplot(exp)
        cplot(zeta, [0, 1], [0, 50])

    By default, the complex argument (phase) is shown as color (hue) and
    the magnitude is show as brightness. You can also supply a
    custom color function (*color*). This function should take a
    complex number as input and return an RGB 3-tuple containing
    floats in the range 0.0-1.0.

    Alternatively, you can select a builtin color function by passing
    a string as *color*:

      * "default" - default color scheme
      * "phase" - a color scheme that only renders the phase of the function,
         with white for positive reals, black for negative reals, gold in the
         upper half plane, and blue in the lower half plane.

    To obtain a sharp image, the number of points may need to be
    increased to 100,000 or thereabout. Since evaluating the
    function that many times is likely to be slow, the 'verbose'
    option is useful to display progress.

    .. note :: This function requires matplotlib (pylab).
    """
    if color is None or color == "default":
        color = ctx.default_color_function
    if color == "phase":
        color = ctx.phase_color_function
    import pylab
    if file:
        axes = None
    fig = None
    if not axes:
        fig = pylab.figure()
        axes = fig.add_subplot(111)
    rea, reb = re
    ima, imb = im
    dre = reb - rea
    dim = imb - ima
    M = int(ctx.sqrt(points*dre/dim)+1)
    N = int(ctx.sqrt(points*dim/dre)+1)
    x = pylab.linspace(rea, reb, M)
    y = pylab.linspace(ima, imb, N)
    # Note: we have to be careful to get the right rotation.
    # Test with these plots:
    #   cplot(lambda z: z if z.real < 0 else 0)
    #   cplot(lambda z: z if z.imag < 0 else 0)
    w = pylab.zeros((N, M, 3))
    for n in xrange(N):
        for m in xrange(M):
            z = ctx.mpc(x[m], y[n])
            try:
                v = color(f(z))
            except ctx.plot_ignore:
                v = (0.5, 0.5, 0.5)
            w[n,m] = v
        if verbose:
            print(str(n) + ' of ' + str(N))
    rea, reb, ima, imb = [float(_) for _ in [rea, reb, ima, imb]]
    axes.imshow(w, extent=(rea, reb, ima, imb), origin='lower')
    axes.set_xlabel('Re(z)')
    axes.set_ylabel('Im(z)')
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()

def splot(ctx, f, u=[-5,5], v=[-5,5], points=100, keep_aspect=True, \
          wireframe=False, file=None, dpi=None, axes=None):
    """
    Plots the surface defined by `f`.

    If `f` returns a single component, then this plots the surface
    defined by `z = f(x,y)` over the rectangular domain with
    `x = u` and `y = v`.

    If `f` returns three components, then this plots the parametric
    surface `x, y, z = f(u,v)` over the pairs of intervals `u` and `v`.

    For example, to plot a simple function::

        >>> from mpmath import *
        >>> f = lambda x, y: sin(x+y)*cos(y)
        >>> splot(f, [-pi,pi], [-pi,pi])    # doctest: +SKIP

    Plotting a donut::

        >>> r, R = 1, 2.5
        >>> f = lambda u, v: [r*cos(u), (R+r*sin(u))*cos(v), (R+r*sin(u))*sin(v)]
        >>> splot(f, [0, 2*pi], [0, 2*pi])    # doctest: +SKIP

    .. note :: This function requires matplotlib (pylab) 0.98.5.3 or higher.
    """
    import pylab
    import mpl_toolkits.mplot3d as mplot3d
    if file:
        axes = None
    fig = None
    if not axes:
        fig = pylab.figure()
        axes = mplot3d.axes3d.Axes3D(fig)
    ua, ub = u
    va, vb = v
    du = ub - ua
    dv = vb - va
    if not isinstance(points, (list, tuple)):
        points = [points, points]
    M, N = points
    u = pylab.linspace(ua, ub, M)
    v = pylab.linspace(va, vb, N)
    x, y, z = [pylab.zeros((M, N)) for i in xrange(3)]
    xab, yab, zab = [[0, 0] for i in xrange(3)]
    for n in xrange(N):
        for m in xrange(M):
            fdata = f(ctx.convert(u[m]), ctx.convert(v[n]))
            try:
                x[m,n], y[m,n], z[m,n] = fdata
            except TypeError:
                x[m,n], y[m,n], z[m,n] = u[m], v[n], fdata
            for c, cab in [(x[m,n], xab), (y[m,n], yab), (z[m,n], zab)]:
                if c < cab[0]:
                    cab[0] = c
                if c > cab[1]:
                    cab[1] = c
    if wireframe:
        axes.plot_wireframe(x, y, z, rstride=4, cstride=4)
    else:
        axes.plot_surface(x, y, z, rstride=4, cstride=4)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    if keep_aspect:
        dx, dy, dz = [cab[1] - cab[0] for cab in [xab, yab, zab]]
        maxd = max(dx, dy, dz)
        if dx < maxd:
            delta = maxd - dx
            axes.set_xlim3d(xab[0] - delta / 2.0, xab[1] + delta / 2.0)
        if dy < maxd:
            delta = maxd - dy
            axes.set_ylim3d(yab[0] - delta / 2.0, yab[1] + delta / 2.0)
        if dz < maxd:
            delta = maxd - dz
            axes.set_zlim3d(zab[0] - delta / 2.0, zab[1] + delta / 2.0)
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()


VisualizationMethods.plot = plot
VisualizationMethods.default_color_function = default_color_function
VisualizationMethods.phase_color_function = phase_color_function
VisualizationMethods.cplot = cplot
VisualizationMethods.splot = splot
<|endoftext|>"""
This module defines the mpf, mpc classes, and standard functions for
operating with them.
"""
__docformat__ = 'plaintext'

import functools

import re

from .ctx_base import StandardBaseContext

from .libmp.backend import basestring, BACKEND

from . import libmp

from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
    round_floor, round_ceiling, dps_to_prec, round_nearest, prec_to_dps,
    ComplexResult, to_pickable, from_pickable, normalize,
    from_int, from_float, from_str, to_int, to_float, to_str,
    from_rational, from_man_exp,
    fone, fzero, finf, fninf, fnan,
    mpf_abs, mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul, mpf_mul_int,
    mpf_div, mpf_rdiv_int, mpf_pow_int, mpf_mod,
    mpf_eq, mpf_cmp, mpf_lt, mpf_gt, mpf_le, mpf_ge,
    mpf_hash, mpf_rand,
    mpf_sum,
    bitcount, to_fixed,
    mpc_to_str,
    mpc_to_complex, mpc_hash, mpc_pos, mpc_is_nonzero, mpc_neg, mpc_conjugate,
    mpc_abs, mpc_add, mpc_add_mpf, mpc_sub, mpc_sub_mpf, mpc_mul, mpc_mul_mpf,
    mpc_mul_int, mpc_div, mpc_div_mpf, mpc_pow, mpc_pow_mpf, mpc_pow_int,
    mpc_mpf_div,
    mpf_pow,
    mpf_pi, mpf_degree, mpf_e, mpf_phi, mpf_ln2, mpf_ln10,
    mpf_euler, mpf_catalan, mpf_apery, mpf_khinchin,
    mpf_glaisher, mpf_twinprime, mpf_mertens,
    int_types)

from . import function_docs
from . import rational

new = object.__new__

get_complex = re.compile(r'^\(?(?P<re>[\+\-]?\d*(\.\d*)?(e[\+\-]?\d+)?)??'
                         r'(?P<im>[\+\-]?\d*(\.\d*)?(e[\+\-]?\d+)?j)?\)?$')

if BACKEND == 'sage':
    from sage.libs.mpmath.ext_main import Context as BaseMPContext
    # pickle hack
    import sage.libs.mpmath.ext_main as _mpf_module
else:
    from .ctx_mp_python import PythonMPContext as BaseMPContext
    from . import ctx_mp_python as _mpf_module

from .ctx_mp_python import _mpf, _mpc, mpnumeric

class MPContext(BaseMPContext, StandardBaseContext):
    """
    Context for multiprecision arithmetic with a global precision.
    """

    def __init__(ctx):
        BaseMPContext.__init__(ctx)
        ctx.trap_complex = False
        ctx.pretty = False
        ctx.types = [ctx.mpf, ctx.mpc, ctx.constant]
        ctx._mpq = rational.mpq
        ctx.default()
        StandardBaseContext.__init__(ctx)

        ctx.mpq = rational.mpq
        ctx.init_builtins()

        ctx.hyp_summators = {}

        ctx._init_aliases()

        # XXX: automate
        try:
            ctx.bernoulli.im_func.func_doc = function_docs.bernoulli
            ctx.primepi.im_func.func_doc = function_docs.primepi
            ctx.psi.im_func.func_doc = function_docs.psi
            ctx.atan2.im_func.func_doc = function_docs.atan2
        except AttributeError:
            # python 3
            ctx.bernoulli.__func__.func_doc = function_docs.bernoulli
            ctx.primepi.__func__.func_doc = function_docs.primepi
            ctx.psi.__func__.func_doc = function_docs.psi
            ctx.atan2.__func__.func_doc = function_docs.atan2

        ctx.digamma.func_doc = function_docs.digamma
        ctx.cospi.func_doc = function_docs.cospi
        ctx.sinpi.func_doc = function_docs.sinpi

    def init_builtins(ctx):

        mpf = ctx.mpf
        mpc = ctx.mpc

        # Exact constants
        ctx.one = ctx.make_mpf(fone)
        ctx.zero = ctx.make_mpf(fzero)
        ctx.j = ctx.make_mpc((fzero,fone))
        ctx.inf = ctx.make_mpf(finf)
        ctx.ninf = ctx.make_mpf(fninf)
        ctx.nan = ctx.make_mpf(fnan)

        eps = ctx.constant(lambda prec, rnd: (0, MPZ_ONE, 1-prec, 1),
            "epsilon of working precision", "eps")
        ctx.eps = eps

        # Approximate constants
        ctx.pi = ctx.constant(mpf_pi, "pi", "pi")
        ctx.ln2 = ctx.constant(mpf_ln2, "ln(2)", "ln2")
        ctx.ln10 = ctx.constant(mpf_ln10, "ln(10)", "ln10")
        ctx.phi = ctx.constant(mpf_phi, "Golden ratio phi", "phi")
        ctx.e = ctx.constant(mpf_e, "e = exp(1)", "e")
        ctx.euler = ctx.constant(mpf_euler, "Euler's constant", "euler")
        ctx.catalan = ctx.constant(mpf_catalan, "Catalan's constant", "catalan")
        ctx.khinchin = ctx.constant(mpf_khinchin, "Khinchin's constant", "khinchin")
        ctx.glaisher = ctx.constant(mpf_glaisher, "Glaisher's constant", "glaisher")
        ctx.apery = ctx.constant(mpf_apery, "Apery's constant", "apery")
        ctx.degree = ctx.constant(mpf_degree, "1 deg = pi / 180", "degree")
        ctx.twinprime = ctx.constant(mpf_twinprime, "Twin prime constant", "twinprime")
        ctx.mertens = ctx.constant(mpf_mertens, "Mertens' constant", "mertens")

        # Standard functions
        ctx.sqrt = ctx._wrap_libmp_function(libmp.mpf_sqrt, libmp.mpc_sqrt)
        ctx.cbrt = ctx._wrap_libmp_function(libmp.mpf_cbrt, libmp.mpc_cbrt)
        ctx.ln = ctx._wrap_libmp_function(libmp.mpf_log, libmp.mpc_log)
        ctx.atan = ctx._wrap_libmp_function(libmp.mpf_atan, libmp.mpc_atan)
        ctx.exp = ctx._wrap_libmp_function(libmp.mpf_exp, libmp.mpc_exp)
        ctx.expj = ctx._wrap_libmp_function(libmp.mpf_expj, libmp.mpc_expj)
        ctx.expjpi = ctx._wrap_libmp_function(libmp.mpf_expjpi, libmp.mpc_expjpi)
        ctx.sin = ctx._wrap_libmp_function(libmp.mpf_sin, libmp.mpc_sin)
        ctx.cos = ctx._wrap_libmp_function(libmp.mpf_cos, libmp.mpc_cos)
        ctx.tan = ctx._wrap_libmp_function(libmp.mpf_tan, libmp.mpc_tan)
        ctx.sinh = ctx._wrap_libmp_function(libmp.mpf_sinh, libmp.mpc_sinh)
        ctx.cosh = ctx._wrap_libmp_function(libmp.mpf_cosh, libmp.mpc_cosh)
        ctx.tanh = ctx._wrap_libmp_function(libmp.mpf_tanh, libmp.mpc_tanh)
        ctx.asin = ctx._wrap_libmp_function(libmp.mpf_asin, libmp.mpc_asin)
        ctx.acos = ctx._wrap_libmp_function(libmp.mpf_acos, libmp.mpc_acos)
        ctx.atan = ctx._wrap_libmp_function(libmp.mpf_atan, libmp.mpc_atan)
        ctx.asinh = ctx._wrap_libmp_function(libmp.mpf_asinh, libmp.mpc_asinh)
        ctx.acosh = ctx._wrap_libmp_function(libmp.mpf_acosh, libmp.mpc_acosh)
        ctx.atanh = ctx._wrap_libmp_function(libmp.mpf_atanh, libmp.mpc_atanh)
        ctx.sinpi = ctx._wrap_libmp_function(libmp.mpf_sin_pi, libmp.mpc_sin_pi)
        ctx.cospi = ctx._wrap_libmp_function(libmp.mpf_cos_pi, libmp.mpc_cos_pi)
        ctx.floor = ctx._wrap_libmp_function(libmp.mpf_floor, libmp.mpc_floor)
        ctx.ceil = ctx._wrap_libmp_function(libmp.mpf_ceil, libmp.mpc_ceil)
        ctx.nint = ctx._wrap_libmp_function(libmp.mpf_nint, libmp.mpc_nint)
        ctx.frac = ctx._wrap_libmp_function(libmp.mpf_frac, libmp.mpc_frac)
        ctx.fib = ctx.fibonacci = ctx._wrap_libmp_function(libmp.mpf_fibonacci, libmp.mpc_fibonacci)

        ctx.gamma = ctx._wrap_libmp_function(libmp.mpf_gamma, libmp.mpc_gamma)
        ctx.rgamma = ctx._wrap_libmp_function(libmp.mpf_rgamma, libmp.mpc_rgamma)
        ctx.loggamma = ctx._wrap_libmp_function(libmp.mpf_loggamma, libmp.mpc_loggamma)
        ctx.fac = ctx.factorial = ctx._wrap_libmp_function(libmp.mpf_factorial, libmp.mpc_factorial)

        ctx.digamma = ctx._wrap_libmp_function(libmp.mpf_psi0, libmp.mpc_psi0)
        ctx.harmonic = ctx._wrap_libmp_function(libmp.mpf_harmonic, libmp.mpc_harmonic)
        ctx.ei = ctx._wrap_libmp_function(libmp.mpf_ei, libmp.mpc_ei)
        ctx.e1 = ctx._wrap_libmp_function(libmp.mpf_e1, libmp.mpc_e1)
        ctx._ci = ctx._wrap_libmp_function(libmp.mpf_ci, libmp.mpc_ci)
        ctx._si = ctx._wrap_libmp_function(libmp.mpf_si, libmp.mpc_si)
        ctx.ellipk = ctx._wrap_libmp_function(libmp.mpf_ellipk, libmp.mpc_ellipk)
        ctx._ellipe = ctx._wrap_libmp_function(libmp.mpf_ellipe, libmp.mpc_ellipe)
        ctx.agm1 = ctx._wrap_libmp_function(libmp.mpf_agm1, libmp.mpc_agm1)
        ctx._erf = ctx._wrap_libmp_function(libmp.mpf_erf, None)
        ctx._erfc = ctx._wrap_libmp_function(libmp.mpf_erfc, None)
        ctx._zeta = ctx._wrap_libmp_function(libmp.mpf_zeta, libmp.mpc_zeta)
        ctx._altzeta = ctx._wrap_libmp_function(libmp.mpf_altzeta, libmp.mpc_altzeta)

        # Faster versions
        ctx.sqrt = getattr(ctx, "_sage_sqrt", ctx.sqrt)
        ctx.exp = getattr(ctx, "_sage_exp", ctx.exp)
        ctx.ln = getattr(ctx, "_sage_ln", ctx.ln)
        ctx.cos = getattr(ctx, "_sage_cos", ctx.cos)
        ctx.sin = getattr(ctx, "_sage_sin", ctx.sin)

    def to_fixed(ctx, x, prec):
        return x.to_fixed(prec)

    def hypot(ctx, x, y):
        r"""
        Computes the Euclidean norm of the vector `(x, y)`, equal
        to `\sqrt{x^2 + y^2}`. Both `x` and `y` must be real."""
        x = ctx.convert(x)
        y = ctx.convert(y)
        return ctx.make_mpf(libmp.mpf_hypot(x._mpf_, y._mpf_, *ctx._prec_rounding))

    def _gamma_upper_int(ctx, n, z):
        n = int(ctx._re(n))
        if n == 0:
            return ctx.e1(z)
        if not hasattr(z, '_mpf_'):
            raise NotImplementedError
        prec, rounding = ctx._prec_rounding
        real, imag = libmp.mpf_expint(n, z._mpf_, prec, rounding, gamma=True)
        if imag is None:
            return ctx.make_mpf(real)
        else:
            return ctx.make_mpc((real, imag))

    def _expint_int(ctx, n, z):
        n = int(n)
        if n == 1:
            return ctx.e1(z)
        if not hasattr(z, '_mpf_'):
            raise NotImplementedError
        prec, rounding = ctx._prec_rounding
        real, imag = libmp.mpf_expint(n, z._mpf_, prec, rounding)
        if imag is None:
            return ctx.make_mpf(real)
        else:
            return ctx.make_mpc((real, imag))

    def _nthroot(ctx, x, n):
        if hasattr(x, '_mpf_'):
            try:
                return ctx.make_mpf(libmp.mpf_nthroot(x._mpf_, n, *ctx._prec_rounding))
            except ComplexResult:
                if ctx.trap_complex:
                    raise
                x = (x._mpf_, libmp.fzero)
        else:
            x = x._mpc_
        return ctx.make_mpc(libmp.mpc_nthroot(x, n, *ctx._prec_rounding))

    def _besselj(ctx, n, z):
        prec, rounding = ctx._prec_rounding
        if hasattr(z, '_mpf_'):
            return ctx.make_mpf(libmp.mpf_besseljn(n, z._mpf_, prec, rounding))
        elif hasattr(z, '_mpc_'):
            return ctx.make_mpc(libmp.mpc_besseljn(n, z._mpc_, prec, rounding))

    def _agm(ctx, a, b=1):
        prec, rounding = ctx._prec_rounding
        if hasattr(a, '_mpf_') and hasattr(b, '_mpf_'):
            try:
                v = libmp.mpf_agm(a._mpf_, b._mpf_, prec, rounding)
                return ctx.make_mpf(v)
            except ComplexResult:
                pass
        if hasattr(a, '_mpf_'): a = (a._mpf_, libmp.fzero)
        else: a = a._mpc_
        if hasattr(b, '_mpf_'): b = (b._mpf_, libmp.fzero)
        else: b = b._mpc_
        return ctx.make_mpc(libmp.mpc_agm(a, b, prec, rounding))

    def bernoulli(ctx, n):
        return ctx.make_mpf(libmp.mpf_bernoulli(int(n), *ctx._prec_rounding))

    def _zeta_int(ctx, n):
        return ctx.make_mpf(libmp.mpf_zeta_int(int(n), *ctx._prec_rounding))

    def atan2(ctx, y, x):
        x = ctx.convert(x)
        y = ctx.convert(y)
        return ctx.make_mpf(libmp.mpf_atan2(y._mpf_, x._mpf_, *ctx._prec_rounding))

    def psi(ctx, m, z):
        z = ctx.convert(z)
        m = int(m)
        if ctx._is_real_type(z):
            return ctx.make_mpf(libmp.mpf_psi(m, z._mpf_, *ctx._prec_rounding))
        else:
            return ctx.make_mpc(libmp.mpc_psi(m, z._mpc_, *ctx._prec_rounding))

    def cos_sin(ctx, x, **kwargs):
        if type(x) not in ctx.types:
            x = ctx.convert(x)
        prec, rounding = ctx._parse_prec(kwargs)
        if hasattr(x, '_mpf_'):
            c, s = libmp.mpf_cos_sin(x._mpf_, prec, rounding)
            return ctx.make_mpf(c), ctx.make_mpf(s)
        elif hasattr(x, '_mpc_'):
            c, s = libmp.mpc_cos_sin(x._mpc_, prec, rounding)
            return ctx.make_mpc(c), ctx.make_mpc(s)
        else:
            return ctx.cos(x, **kwargs), ctx.sin(x, **kwargs)

    def cospi_sinpi(ctx, x, **kwargs):
        if type(x) not in ctx.types:
            x = ctx.convert(x)
        prec, rounding = ctx._parse_prec(kwargs)
        if hasattr(x, '_mpf_'):
            c, s = libmp.mpf_cos_sin_pi(x._mpf_, prec, rounding)
            return ctx.make_mpf(c), ctx.make_mpf(s)
        elif hasattr(x, '_mpc_'):
            c, s = libmp.mpc_cos_sin_pi(x._mpc_, prec, rounding)
            return ctx.make_mpc(c), ctx.make_mpc(s)
        else:
            return ctx.cos(x, **kwargs), ctx.sin(x, **kwargs)

    def clone(ctx):
        """
        Create a copy of the context, with the same working precision.
        """
        a = ctx.__class__()
        a.prec = ctx.prec
        return a

    # Several helper methods
    # TODO: add more of these, make consistent, write docstrings, ...

    def _is_real_type(ctx, x):
        if hasattr(x, '_mpc_') or type(x) is complex:
            return False
        return True

    def _is_complex_type(ctx, x):
        if hasattr(x, '_mpc_') or type(x) is complex:
            return True
        return False

    def isnan(ctx, x):
        """
        Return *True* if *x* is a NaN (not-a-number), or for a complex
        number, whether either the real or complex part is NaN;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isnan(3.14)
            False
            >>> isnan(nan)
            True
            >>> isnan(mpc(3.14,2.72))
            False
            >>> isnan(mpc(3.14,nan))
            True

        """
        if hasattr(x, "_mpf_"):
            return x._mpf_ == fnan
        if hasattr(x, "_mpc_"):
            return fnan in x._mpc_
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnan(x)
        raise TypeError("isnan() needs a number as input")

    def isfinite(ctx, x):
        """
        Return *True* if *x* is a finite number, i.e. neither
        an infinity or a NaN.

            >>> from mpmath import *
            >>> isfinite(inf)
            False
            >>> isfinite(-inf)
            False
            >>> isfinite(3)
            True
            >>> isfinite(nan)
            False
            >>> isfinite(3+4j)
            True
            >>> isfinite(mpc(3,inf))
            False
            >>> isfinite(mpc(nan,3))
            False

        """
        if ctx.isinf(x) or ctx.isnan(x):
            return False
        return True

    def isnpint(ctx, x):
        """
        Determine if *x* is a nonpositive integer.
        """
        if not x:
            return True
        if hasattr(x, '_mpf_'):
            sign, man, exp, bc = x._mpf_
            return sign and exp >= 0
        if hasattr(x, '_mpc_'):
            return not x.imag and ctx.isnpint(x.real)
        if type(x) in int_types:
            return x <= 0
        if isinstance(x, ctx.mpq):
            p, q = x._mpq_
            if not p:
                return True
            return q == 1 and p <= 0
        return ctx.isnpint(ctx.convert(x))

    def __str__(ctx):
        lines = ["Mpmath settings:",
            ("  mp.prec = %s" % ctx.prec).ljust(30) + "[default: 53]",
            ("  mp.dps = %s" % ctx.dps).ljust(30) + "[default: 15]",
            ("  mp.trap_complex = %s" % ctx.trap_complex).ljust(30) + "[default: False]",
        ]
        return "\n".join(lines)

    @property
    def _repr_digits(ctx):
        return repr_dps(ctx._prec)

    @property
    def _str_digits(ctx):
        return ctx._dps

    def extraprec(ctx, n, normalize_output=False):
        """
        The block

            with extraprec(n):
                <code>

        increases the precision n bits, executes <code>, and then
        restores the precision.

        extraprec(n)(f) returns a decorated version of the function f
        that increases the working precision by n bits before execution,
        and restores the parent precision afterwards. With
        normalize_output=True, it rounds the return value to the parent
        precision.
        """
        return PrecisionManager(ctx, lambda p: p + n, None, normalize_output)

    def extradps(ctx, n, normalize_output=False):
        """
        This function is analogous to extraprec (see documentation)
        but changes the decimal precision instead of the number of bits.
        """
        return PrecisionManager(ctx, None, lambda d: d + n, normalize_output)

    def workprec(ctx, n, normalize_output=False):
        """
        The block

            with workprec(n):
                <code>

        sets the precision to n bits, executes <code>, and then restores
        the precision.

        workprec(n)(f) returns a decorated version of the function f
        that sets the precision to n bits before execution,
        and restores the precision afterwards. With normalize_output=True,
        it rounds the return value to the parent precision.
        """
        return PrecisionManager(ctx, lambda p: n, None, normalize_output)

    def workdps(ctx, n, normalize_output=False):
        """
        This function is analogous to workprec (see documentation)
        but changes the decimal precision instead of the number of bits.
        """
        return PrecisionManager(ctx, None, lambda d: n, normalize_output)

    def autoprec(ctx, f, maxprec=None, catch=(), verbose=False):
        r"""
        Return a wrapped copy of *f* that repeatedly evaluates *f*
        with increasing precision until the result converges to the
        full precision used at the point of the call.

        This heuristically protects against rounding errors, at the cost of
        roughly a 2x slowdown compared to manually setting the optimal
        precision. This method can, however, easily be fooled if the results
        from *f* depend "discontinuously" on the precision, for instance
        if catastrophic cancellation can occur. Therefore, :func:`~mpmath.autoprec`
        should be used judiciously.

        **Examples**

        Many functions are sensitive to perturbations of the input arguments.
        If the arguments are decimal numbers, they may have to be converted
        to binary at a much higher precision. If the amount of required
        extra precision is unknown, :func:`~mpmath.autoprec` is convenient::

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> mp.pretty = True
            >>> besselj(5, 125 * 10**28)    # Exact input
            -8.03284785591801e-17
            >>> besselj(5, '1.25e30')   # Bad
            7.12954868316652e-16
            >>> autoprec(besselj)(5, '1.25e30')   # Good
            -8.03284785591801e-17

        The following fails to converge because `\sin(\pi) = 0` whereas all
        finite-precision approximations of `\pi` give nonzero values::

            >>> autoprec(sin)(pi) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: autoprec: prec increased to 2910 without convergence

        As the following example shows, :func:`~mpmath.autoprec` can protect against
        cancellation, but is fooled by too severe cancellation::

            >>> x = 1e-10
            >>> exp(x)-1; expm1(x); autoprec(lambda t: exp(t)-1)(x)
            1.00000008274037e-10
            1.00000000005e-10
            1.00000000005e-10
            >>> x = 1e-50
            >>> exp(x)-1; expm1(x); autoprec(lambda t: exp(t)-1)(x)
            0.0
            1.0e-50
            0.0

        With *catch*, an exception or list of exceptions to intercept
        may be specified. The raised exception is interpreted
        as signaling insufficient precision. This permits, for example,
        evaluating a function where a too low precision results in a
        division by zero::

            >>> f = lambda x: 1/(exp(x)-1)
            >>> f(1e-30)
            Traceback (most recent call last):
              ...
            ZeroDivisionError
            >>> autoprec(f, catch=ZeroDivisionError)(1e-30)
            1.0e+30


        """
        def f_autoprec_wrapped(*args, **kwargs):
            prec = ctx.prec
            if maxprec is None:
                maxprec2 = ctx._default_hyper_maxprec(prec)
            else:
                maxprec2 = maxprec
            try:
                ctx.prec = prec + 10
                try:
                    v1 = f(*args, **kwargs)
                except catch:
                    v1 = ctx.nan
                prec2 = prec + 20
                while 1:
                    ctx.prec = prec2
                    try:
                        v2 = f(*args, **kwargs)
                    except catch:
                        v2 = ctx.nan
                    if v1 == v2:
                        break
                    err = ctx.mag(v2-v1) - ctx.mag(v2)
                    if err < (-prec):
                        break
                    if verbose:
                        print("autoprec: target=%s, prec=%s, accuracy=%s" \
                            % (prec, prec2, -err))
                    v1 = v2
                    if prec2 >= maxprec2:
                        raise ctx.NoConvergence(\
                        "autoprec: prec increased to %i without convergence"\
                        % prec2)
                    prec2 += int(prec2*2)
                    prec2 = min(prec2, maxprec2)
            finally:
                ctx.prec = prec
            return +v2
        return f_autoprec_wrapped

    def nstr(ctx, x, n=6, **kwargs):
        """
        Convert an ``mpf`` or ``mpc`` to a decimal string literal with *n*
        significant digits. The small default value for *n* is chosen to
        make this function useful for printing collections of numbers
        (lists, matrices, etc).

        If *x* is a list or tuple, :func:`~mpmath.nstr` is applied recursively
        to each element. For unrecognized classes, :func:`~mpmath.nstr`
        simply returns ``str(x)``.

        The companion function :func:`~mpmath.nprint` prints the result
        instead of returning it.

        The keyword arguments *strip_zeros*, *min_fixed*, *max_fixed*
        and *show_zero_exponent* are forwarded to :func:`~mpmath.libmp.to_str`.

        The number will be printed in fixed-point format if the position
        of the leading digit is strictly between min_fixed
        (default = min(-dps/3,-5)) and max_fixed (default = dps).

        To force fixed-point format always, set min_fixed = -inf,
        max_fixed = +inf. To force floating-point format, set
        min_fixed >= max_fixed.

            >>> from mpmath import *
            >>> nstr([+pi, ldexp(1,-500)])
            '[3.14159, 3.05494e-151]'
            >>> nprint([+pi, ldexp(1,-500)])
            [3.14159, 3.05494e-151]
            >>> nstr(mpf("5e-10"), 5)
            '5.0e-10'
            >>> nstr(mpf("5e-10"), 5, strip_zeros=False)
            '5.0000e-10'
            >>> nstr(mpf("5e-10"), 5, strip_zeros=False, min_fixed=-11)
            '0.00000000050000'
            >>> nstr(mpf(0), 5, show_zero_exponent=True)
            '0.0e+0'

        """
        if isinstance(x, list):
            return "[%s]" % (", ".join(ctx.nstr(c, n, **kwargs) for c in x))
        if isinstance(x, tuple):
            return "(%s)" % (", ".join(ctx.nstr(c, n, **kwargs) for c in x))
        if hasattr(x, '_mpf_'):
            return to_str(x._mpf_, n, **kwargs)
        if hasattr(x, '_mpc_'):
            return "(" + mpc_to_str(x._mpc_, n, **kwargs)  + ")"
        if isinstance(x, basestring):
            return repr(x)
        if isinstance(x, ctx.matrix):
            return x.__nstr__(n, **kwargs)
        return str(x)

    def _convert_fallback(ctx, x, strings):
        if strings and isinstance(x, basestring):
            if 'j' in x.lower():
                x = x.lower().replace(' ', '')
                match = get_complex.match(x)
                re = match.group('re')
                if not re:
                    re = 0
                im = match.group('im').rstrip('j')
                return ctx.mpc(ctx.convert(re), ctx.convert(im))
        if hasattr(x, "_mpi_"):
            a, b = x._mpi_
            if a == b:
                return ctx.make_mpf(a)
            else:
                raise ValueError("can only create mpf from zero-width interval")
        raise TypeError("cannot create mpf from " + repr(x))

    def mpmathify(ctx, *args, **kwargs):
        return ctx.convert(*args, **kwargs)

    def _parse_prec(ctx, kwargs):
        if kwargs:
            if kwargs.get('exact'):
                return 0, 'f'
            prec, rounding = ctx._prec_rounding
            if 'rounding' in kwargs:
                rounding = kwargs['rounding']
            if 'prec' in kwargs:
                prec = kwargs['prec']
                if prec == ctx.inf:
                    return 0, 'f'
                else:
                    prec = int(prec)
            elif 'dps' in kwargs:
                dps = kwargs['dps']
                if dps == ctx.inf:
                    return 0, 'f'
                prec = dps_to_prec(dps)
            return prec, rounding
        return ctx._prec_rounding

    _exact_overflow_msg = "the exact result does not fit in memory"

    _hypsum_msg = """hypsum() failed to converge to the requested %i bits of accuracy
using a working precision of %i bits. Try with a higher maxprec,
maxterms, or set zeroprec."""

    def hypsum(ctx, p, q, flags, coeffs, z, accurate_small=True, **kwargs):
        if hasattr(z, "_mpf_"):
            key = p, q, flags, 'R'
            v = z._mpf_
        elif hasattr(z, "_mpc_"):
            key = p, q, flags, 'C'
            v = z._mpc_
        if key not in ctx.hyp_summators:
            ctx.hyp_summators[key] = libmp.make_hyp_summator(key)[1]
        summator = ctx.hyp_summators[key]
        prec = ctx.prec
        maxprec = kwargs.get('maxprec', ctx._default_hyper_maxprec(prec))
        extraprec = 50
        epsshift = 25
        # Jumps in magnitude occur when parameters are close to negative
        # integers. We must ensure that these terms are included in
        # the sum and added accurately
        magnitude_check = {}
        max_total_jump = 0
        for i, c in enumerate(coeffs):
            if flags[i] == 'Z':
                if i >= p and c <= 0:
                    ok = False
                    for ii, cc in enumerate(coeffs[:p]):
                        # Note: c <= cc or c < cc, depending on convention
                        if flags[ii] == 'Z' and cc <= 0 and c <= cc:
                            ok = True
                    if not ok:
                        raise ZeroDivisionError("pole in hypergeometric series")
                continue
            n, d = ctx.nint_distance(c)
            n = -int(n)
            d = -d
            if i >= p and n >= 0 and d > 4:
                if n in magnitude_check:
                    magnitude_check[n] += d
                else:
                    magnitude_check[n] = d
                extraprec = max(extraprec, d - prec + 60)
            max_total_jump += abs(d)
        while 1:
            if extraprec > maxprec:
                raise ValueError(ctx._hypsum_msg % (prec, prec+extraprec))
            wp = prec + extraprec
            if magnitude_check:
                mag_dict = dict((n,None) for n in magnitude_check)
            else:
                mag_dict = {}
            zv, have_complex, magnitude = summator(coeffs, v, prec, wp, \
                epsshift, mag_dict, **kwargs)
            cancel = -magnitude
            jumps_resolved = True
            if extraprec < max_total_jump:
                for n in mag_dict.values():
                    if (n is None) or (n < prec):
                        jumps_resolved = False
                        break
            accurate = (cancel < extraprec-25-5 or not accurate_small)
            if jumps_resolved:
                if accurate:
                    break
                # zero?
                zeroprec = kwargs.get('zeroprec')
                if zeroprec is not None:
                    if cancel > zeroprec:
                        if have_complex:
                            return ctx.mpc(0)
                        else:
                            return ctx.zero

            # Some near-singularities were not included, so increase
            # precision and repeat until they are
            extraprec *= 2
            # Possible workaround for bad roundoff in fixed-point arithmetic
            epsshift += 5
            extraprec += 5

        if type(zv) is tuple:
            if have_complex:
                return ctx.make_mpc(zv)
            else:
                return ctx.make_mpf(zv)
        else:
            return zv

    def ldexp(ctx, x, n):
        r"""
        Computes `x 2^n` efficiently. No rounding is performed.
        The argument `x` must be a real floating-point number (or
        possible to convert into one) and `n` must be a Python ``int``.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> ldexp(1, 10)
            mpf('1024.0')
            >>> ldexp(1, -3)
            mpf('0.125')

        """
        x = ctx.convert(x)
        return ctx.make_mpf(libmp.mpf_shift(x._mpf_, n))

    def frexp(ctx, x):
        r"""
        Given a real number `x`, returns `(y, n)` with `y \in [0.5, 1)`,
        `n` a Python integer, and such that `x = y 2^n`. No rounding is
        performed.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> frexp(7.5)
            (mpf('0.9375'), 3)

        """
        x = ctx.convert(x)
        y, n = libmp.mpf_frexp(x._mpf_)
        return ctx.make_mpf(y), n

    def fneg(ctx, x, **kwargs):
        """
        Negates the number *x*, giving a floating-point result, optionally
        using a custom precision and rounding mode.

        See the documentation of :func:`~mpmath.fadd` for a detailed description
        of how to specify precision and rounding.

        **Examples**

        An mpmath number is returned::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fneg(2.5)
            mpf('-2.5')
            >>> fneg(-5+2j)
            mpc(real='5.0', imag='-2.0')

        Precise control over rounding is possible::

            >>> x = fadd(2, 1e-100, exact=True)
            >>> fneg(x)
            mpf('-2.0')
            >>> fneg(x, rounding='f')
            mpf('-2.0000000000000004')

        Negating with and without roundoff::

            >>> n = 200000000000000000000001
            >>> print(int(-mpf(n)))
            -200000000000000016777216
            >>> print(int(fneg(n)))
            -200000000000000016777216
            >>> print(int(fneg(n, prec=log(n,2)+1)))
            -200000000000000000000001
            >>> print(int(fneg(n, dps=log(n,10)+1)))
            -200000000000000000000001
            >>> print(int(fneg(n, prec=inf)))
            -200000000000000000000001
            >>> print(int(fneg(n, dps=inf)))
            -200000000000000000000001
            >>> print(int(fneg(n, exact=True)))
            -200000000000000000000001

        """
        prec, rounding = ctx._parse_prec(kwargs)
        x = ctx.convert(x)
        if hasattr(x, '_mpf_'):
            return ctx.make_mpf(mpf_neg(x._mpf_, prec, rounding))
        if hasattr(x, '_mpc_'):
            return ctx.make_mpc(mpc_neg(x._mpc_, prec, rounding))
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def fadd(ctx, x, y, **kwargs):
        """
        Adds the numbers *x* and *y*, giving a floating-point result,
        optionally using a custom precision and rounding mode.

        The default precision is the working precision of the context.
        You can specify a custom precision in bits by passing the *prec* keyword
        argument, or by providing an equivalent decimal precision with the *dps*
        keyword argument. If the precision is set to ``+inf``, or if the flag
        *exact=True* is passed, an exact addition with no rounding is performed.

        When the precision is finite, the optional *rounding* keyword argument
        specifies the direction of rounding. Valid options are ``'n'`` for
        nearest (default), ``'f'`` for floor, ``'c'`` for ceiling, ``'d'``
        for down, ``'u'`` for up.

        **Examples**

        Using :func:`~mpmath.fadd` with precision and rounding control::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fadd(2, 1e-20)
            mpf('2.0')
            >>> fadd(2, 1e-20, rounding='u')
            mpf('2.0000000000000004')
            >>> nprint(fadd(2, 1e-20, prec=100), 25)
            2.00000000000000000001
            >>> nprint(fadd(2, 1e-20, dps=15), 25)
            2.0
            >>> nprint(fadd(2, 1e-20, dps=25), 25)
            2.00000000000000000001
            >>> nprint(fadd(2, 1e-20, exact=True), 25)
            2.00000000000000000001

        Exact addition avoids cancellation errors, enforcing familiar laws
        of numbers such as `x+y-x = y`, which don't hold in floating-point
        arithmetic with finite precision::

            >>> x, y = mpf(2), mpf('1e-1000')
            >>> print(x + y - x)
            0.0
            >>> print(fadd(x, y, prec=inf) - x)
            1.0e-1000
            >>> print(fadd(x, y, exact=True) - x)
            1.0e-1000

        Exact addition can be inefficient and may be impossible to perform
        with large magnitude differences::

            >>> fadd(1, '1e-100000000000000000000', prec=inf)
            Traceback (most recent call last):
              ...
            OverflowError: the exact result does not fit in memory

        """
        prec, rounding = ctx._parse_prec(kwargs)
        x = ctx.convert(x)
        y = ctx.convert(y)
        try:
            if hasattr(x, '_mpf_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpf(mpf_add(x._mpf_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_add_mpf(y._mpc_, x._mpf_, prec, rounding))
            if hasattr(x, '_mpc_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpc(mpc_add_mpf(x._mpc_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_add(x._mpc_, y._mpc_, prec, rounding))
        except (ValueError, OverflowError):
            raise OverflowError(ctx._exact_overflow_msg)
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def fsub(ctx, x, y, **kwargs):
        """
        Subtracts the numbers *x* and *y*, giving a floating-point result,
        optionally using a custom precision and rounding mode.

        See the documentation of :func:`~mpmath.fadd` for a detailed description
        of how to specify precision and rounding.

        **Examples**

        Using :func:`~mpmath.fsub` with precision and rounding control::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fsub(2, 1e-20)
            mpf('2.0')
            >>> fsub(2, 1e-20, rounding='d')
            mpf('1.9999999999999998')
            >>> nprint(fsub(2, 1e-20, prec=100), 25)
            1.99999999999999999999
            >>> nprint(fsub(2, 1e-20, dps=15), 25)
            2.0
            >>> nprint(fsub(2, 1e-20, dps=25), 25)
            1.99999999999999999999
            >>> nprint(fsub(2, 1e-20, exact=True), 25)
            1.99999999999999999999

        Exact subtraction avoids cancellation errors, enforcing familiar laws
        of numbers such as `x-y+y = x`, which don't hold in floating-point
        arithmetic with finite precision::

            >>> x, y = mpf(2), mpf('1e1000')
            >>> print(x - y + y)
            0.0
            >>> print(fsub(x, y, prec=inf) + y)
            2.0
            >>> print(fsub(x, y, exact=True) + y)
            2.0

        Exact addition can be inefficient and may be impossible to perform
        with large magnitude differences::

            >>> fsub(1, '1e-100000000000000000000', prec=inf)
            Traceback (most recent call last):
              ...
            OverflowError: the exact result does not fit in memory

        """
        prec, rounding = ctx._parse_prec(kwargs)
        x = ctx.convert(x)
        y = ctx.convert(y)
        try:
            if hasattr(x, '_mpf_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpf(mpf_sub(x._mpf_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_sub((x._mpf_, fzero), y._mpc_, prec, rounding))
            if hasattr(x, '_mpc_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpc(mpc_sub_mpf(x._mpc_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_sub(x._mpc_, y._mpc_, prec, rounding))
        except (ValueError, OverflowError):
            raise OverflowError(ctx._exact_overflow_msg)
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def fmul(ctx, x, y, **kwargs):
        """
        Multiplies the numbers *x* and *y*, giving a floating-point result,
        optionally using a custom precision and rounding mode.

        See the documentation of :func:`~mpmath.fadd` for a detailed description
        of how to specify precision and rounding.

        **Examples**

        The result is an mpmath number::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fmul(2, 5.0)
            mpf('10.0')
            >>> fmul(0.5j, 0.5)
            mpc(real='0.0', imag='0.25')

        Avoiding roundoff::

            >>> x, y = 10**10+1, 10**15+1
            >>> print(x*y)
            10000000001000010000000001
            >>> print(mpf(x) * mpf(y))
            1.0000000001e+25
            >>> print(int(mpf(x) * mpf(y)))
            10000000001000011026399232
            >>> print(int(fmul(x, y)))
            10000000001000011026399232
            >>> print(int(fmul(x, y, dps=25)))
            10000000001000010000000001
            >>> print(int(fmul(x, y, exact=True)))
            10000000001000010000000001

        Exact multiplication with complex numbers can be inefficient and may
        be impossible to perform with large magnitude differences between
        real and imaginary parts::

            >>> x = 1+2j
            >>> y = mpc(2, '1e-100000000000000000000')
            >>> fmul(x, y)
            mpc(real='2.0', imag='4.0')
            >>> fmul(x, y, rounding='u')
            mpc(real='2.0', imag='4.0000000000000009')
            >>> fmul(x, y, exact=True)
            Traceback (most recent call last):
              ...
            OverflowError: the exact result does not fit in memory

        """
        prec, rounding = ctx._parse_prec(kwargs)
        x = ctx.convert(x)
        y = ctx.convert(y)
        try:
            if hasattr(x, '_mpf_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpf(mpf_mul(x._mpf_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_mul_mpf(y._mpc_, x._mpf_, prec, rounding))
            if hasattr(x, '_mpc_'):
                if hasattr(y, '_mpf_'):
                    return ctx.make_mpc(mpc_mul_mpf(x._mpc_, y._mpf_, prec, rounding))
                if hasattr(y, '_mpc_'):
                    return ctx.make_mpc(mpc_mul(x._mpc_, y._mpc_, prec, rounding))
        except (ValueError, OverflowError):
            raise OverflowError(ctx._exact_overflow_msg)
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def fdiv(ctx, x, y, **kwargs):
        """
        Divides the numbers *x* and *y*, giving a floating-point result,
        optionally using a custom precision and rounding mode.

        See the documentation of :func:`~mpmath.fadd` for a detailed description
        of how to specify precision and rounding.

        **Examples**

        The result is an mpmath number::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fdiv(3, 2)
            mpf('1.5')
            >>> fdiv(2, 3)
            mpf('0.66666666666666663')
            >>> fdiv(2+4j, 0.5)
            mpc(real='4.0', imag='8.0')

        The rounding direction and precision can be controlled::

            >>> fdiv(2, 3, dps=3)    # Should be accurate to at least 3 digits
            mpf('0.6666259765625')
            >>> fdiv(2, 3, rounding='d')
            mpf('0.66666666666666663')
            >>> fdiv(2, 3, prec=60)
            mpf('0.66666666666666667')
            >>> fdiv(2, 3, rounding='u')
            mpf('0.66666666666666674')

        Checking the error of a division by performing it at higher precision::

            >>> fdiv(2, 3) - fdiv(2, 3, prec=100)
            mpf('-3.7007434154172148e-17')

        Unlike :func:`~mpmath.fadd`, :func:`~mpmath.fmul`, etc., exact division is not
        allowed since the quotient of two floating-point numbers generally
        does not have an exact floating-point representation. (In the
        future this might be changed to allow the case where the division
        is actually exact.)

            >>> fdiv(2, 3, exact=True)
            Traceback (most recent call last):
              ...
            ValueError: division is not an exact operation

        """
        prec, rounding = ctx._parse_prec(kwargs)
        if not prec:
            raise ValueError("division is not an exact operation")
        x = ctx.convert(x)
        y = ctx.convert(y)
        if hasattr(x, '_mpf_'):
            if hasattr(y, '_mpf_'):
                return ctx.make_mpf(mpf_div(x._mpf_, y._mpf_, prec, rounding))
            if hasattr(y, '_mpc_'):
                return ctx.make_mpc(mpc_div((x._mpf_, fzero), y._mpc_, prec, rounding))
        if hasattr(x, '_mpc_'):
            if hasattr(y, '_mpf_'):
                return ctx.make_mpc(mpc_div_mpf(x._mpc_, y._mpf_, prec, rounding))
            if hasattr(y, '_mpc_'):
                return ctx.make_mpc(mpc_div(x._mpc_, y._mpc_, prec, rounding))
        raise ValueError("Arguments need to be mpf or mpc compatible numbers")

    def nint_distance(ctx, x):
        r"""
        Return `(n,d)` where `n` is the nearest integer to `x` and `d` is
        an estimate of `\log_2(|x-n|)`. If `d < 0`, `-d` gives the precision
        (measured in bits) lost to cancellation when computing `x-n`.

            >>> from mpmath import *
            >>> n, d = nint_distance(5)
            >>> print(n); print(d)
            5
            -inf
            >>> n, d = nint_distance(mpf(5))
            >>> print(n); print(d)
            5
            -inf
            >>> n, d = nint_distance(mpf(5.00000001))
            >>> print(n); print(d)
            5
            -26
            >>> n, d = nint_distance(mpf(4.99999999))
            >>> print(n); print(d)
            5
            -26
            >>> n, d = nint_distance(mpc(5,10))
            >>> print(n); print(d)
            5
            4
            >>> n, d = nint_distance(mpc(5,0.000001))
            >>> print(n); print(d)
            5
            -19

        """
        typx = type(x)
        if typx in int_types:
            return int(x), ctx.ninf
        elif typx is rational.mpq:
            p, q = x._mpq_
            n, r = divmod(p, q)
            if 2*r >= q:
                n += 1
            elif not r:
                return n, ctx.ninf
            # log(p/q-n) = log((p-nq)/q) = log(p-nq) - log(q)
            d = bitcount(abs(p-n*q)) - bitcount(q)
            return n, d
        if hasattr(x, "_mpf_"):
            re = x._mpf_
            im_dist = ctx.ninf
        elif hasattr(x, "_mpc_"):
            re, im = x._mpc_
            isign, iman, iexp, ibc = im
            if iman:
                im_dist = iexp + ibc
            elif im == fzero:
                im_dist = ctx.ninf
            else:
                raise ValueError("requires a finite number")
        else:
            x = ctx.convert(x)
            if hasattr(x, "_mpf_") or hasattr(x, "_mpc_"):
                return ctx.nint_distance(x)
            else:
                raise TypeError("requires an mpf/mpc")
        sign, man, exp, bc = re
        mag = exp+bc
        # |x| < 0.5
        if mag < 0:
            n = 0
            re_dist = mag
        elif man:
            # exact integer
            if exp >= 0:
                n = man << exp
                re_dist = ctx.ninf
            # exact half-integer
            elif exp == -1:
                n = (man>>1)+1
                re_dist = 0
            else:
                d = (-exp-1)
                t = man >> d
                if t & 1:
                    t += 1
                    man = (t<<d) - man
                else:
                    man -= (t<<d)
                n = t>>1   # int(t)>>1
                re_dist = exp+bitcount(man)
            if sign:
                n = -n
        elif re == fzero:
            re_dist = ctx.ninf
            n = 0
        else:
            raise ValueError("requires a finite number")
        return n, max(re_dist, im_dist)

    def fprod(ctx, factors):
        r"""
        Calculates a product containing a finite number of factors (for
        infinite products, see :func:`~mpmath.nprod`). The factors will be
        converted to mpmath numbers.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fprod([1, 2, 0.5, 7])
            mpf('7.0')

        """
        orig = ctx.prec
        try:
            v = ctx.one
            for p in factors:
                v *= p
        finally:
            ctx.prec = orig
        return +v

    def rand(ctx):
        """
        Returns an ``mpf`` with value chosen randomly from `[0, 1)`.
        The number of randomly generated bits in the mantissa is equal
        to the working precision.
        """
        return ctx.make_mpf(mpf_rand(ctx._prec))

    def fraction(ctx, p, q):
        """
        Given Python integers `(p, q)`, returns a lazy ``mpf`` representing
        the fraction `p/q`. The value is updated with the precision.

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> a = fraction(1,100)
            >>> b = mpf(1)/100
            >>> print(a); print(b)
            0.01
            0.01
            >>> mp.dps = 30
            >>> print(a); print(b)      # a will be accurate
            0.01
            0.0100000000000000002081668171172
            >>> mp.dps = 15
        """
        return ctx.constant(lambda prec, rnd: from_rational(p, q, prec, rnd),
            '%s/%s' % (p, q))

    def absmin(ctx, x):
        return abs(ctx.convert(x))

    def absmax(ctx, x):
        return abs(ctx.convert(x))

    def _as_points(ctx, x):
        # XXX: remove this?
        if hasattr(x, '_mpi_'):
            a, b = x._mpi_
            return [ctx.make_mpf(a), ctx.make_mpf(b)]
        return x

    '''
    def _zetasum(ctx, s, a, b):
        """
        Computes sum of k^(-s) for k = a, a+1, ..., b with a, b both small
        integers.
        """
        a = int(a)
        b = int(b)
        s = ctx.convert(s)
        prec, rounding = ctx._prec_rounding
        if hasattr(s, '_mpf_'):
            v = ctx.make_mpf(libmp.mpf_zetasum(s._mpf_, a, b, prec))
        elif hasattr(s, '_mpc_'):
            v = ctx.make_mpc(libmp.mpc_zetasum(s._mpc_, a, b, prec))
        return v
    '''

    def _zetasum_fast(ctx, s, a, n, derivatives=[0], reflect=False):
        if not (ctx.isint(a) and hasattr(s, "_mpc_")):
            raise NotImplementedError
        a = int(a)
        prec = ctx._prec
        xs, ys = libmp.mpc_zetasum(s._mpc_, a, n, derivatives, reflect, prec)
        xs = [ctx.make_mpc(x) for x in xs]
        ys = [ctx.make_mpc(y) for y in ys]
        return xs, ys

class PrecisionManager:
    def __init__(self, ctx, precfun, dpsfun, normalize_output=False):
        self.ctx = ctx
        self.precfun = precfun
        self.dpsfun = dpsfun
        self.normalize_output = normalize_output
    def __call__(self, f):
        @functools.wraps(f)
        def g(*args, **kwargs):
            orig = self.ctx.prec
            try:
                if self.precfun:
                    self.ctx.prec = self.precfun(self.ctx.prec)
                else:
                    self.ctx.dps = self.dpsfun(self.ctx.dps)
                if self.normalize_output:
                    v = f(*args, **kwargs)
                    if type(v) is tuple:
                        return tuple([+a for a in v])
                    return +v
                else:
                    return f(*args, **kwargs)
            finally:
                self.ctx.prec = orig
        return g
    def __enter__(self):
        self.origp = self.ctx.prec
        if self.precfun:
            self.ctx.prec = self.precfun(self.ctx.prec)
        else:
            self.ctx.dps = self.dpsfun(self.ctx.dps)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.prec = self.origp
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod()
<|endoftext|>import operator

from . import libmp

from .libmp.backend import basestring

from .libmp import (
    int_types, MPZ_ONE,
    prec_to_dps, dps_to_prec, repr_dps,
    round_floor, round_ceiling,
    fzero, finf, fninf, fnan,
    mpf_le, mpf_neg,
    from_int, from_float, from_str, from_rational,
    mpi_mid, mpi_delta, mpi_str,
    mpi_abs, mpi_pos, mpi_neg, mpi_add, mpi_sub,
    mpi_mul, mpi_div, mpi_pow_int, mpi_pow,
    mpi_from_str,
    mpci_pos, mpci_neg, mpci_add, mpci_sub, mpci_mul, mpci_div, mpci_pow,
    mpci_abs, mpci_pow, mpci_exp, mpci_log,
    ComplexResult,
    mpf_hash, mpc_hash)
from .matrices.matrices import _matrix

mpi_zero = (fzero, fzero)

from .ctx_base import StandardBaseContext

new = object.__new__

def convert_mpf_(x, prec, rounding):
    if hasattr(x, "_mpf_"): return x._mpf_
    if isinstance(x, int_types): return from_int(x, prec, rounding)
    if isinstance(x, float): return from_float(x, prec, rounding)
    if isinstance(x, basestring): return from_str(x, prec, rounding)
    raise NotImplementedError


class ivmpf(object):
    """
    Interval arithmetic class. Precision is controlled by iv.prec.
    """

    def __new__(cls, x=0):
        return cls.ctx.convert(x)

    def cast(self, cls, f_convert):
        a, b = self._mpi_
        if a == b:
            return cls(f_convert(a))
        raise ValueError

    def __int__(self):
        return self.cast(int, libmp.to_int)

    def __float__(self):
        return self.cast(float, libmp.to_float)

    def __complex__(self):
        return self.cast(complex, libmp.to_float)

    def __hash__(self):
        a, b = self._mpi_
        if a == b:
            return mpf_hash(a)
        else:
            return hash(self._mpi_)

    @property
    def real(self): return self

    @property
    def imag(self): return self.ctx.zero

    def conjugate(self): return self

    @property
    def a(self):
        a, b = self._mpi_
        return self.ctx.make_mpf((a, a))

    @property
    def b(self):
        a, b = self._mpi_
        return self.ctx.make_mpf((b, b))

    @property
    def mid(self):
        ctx = self.ctx
        v = mpi_mid(self._mpi_, ctx.prec)
        return ctx.make_mpf((v, v))

    @property
    def delta(self):
        ctx = self.ctx
        v = mpi_delta(self._mpi_, ctx.prec)
        return ctx.make_mpf((v,v))

    @property
    def _mpci_(self):
        return self._mpi_, mpi_zero

    def _compare(*args):
        raise TypeError("no ordering relation is defined for intervals")

    __gt__ = _compare
    __le__ = _compare
    __gt__ = _compare
    __ge__ = _compare

    def __contains__(self, t):
        t = self.ctx.mpf(t)
        return (self.a <= t.a) and (t.b <= self.b)

    def __str__(self):
        return mpi_str(self._mpi_, self.ctx.prec)

    def __repr__(self):
        if self.ctx.pretty:
            return str(self)
        a, b = self._mpi_
        n = repr_dps(self.ctx.prec)
        a = libmp.to_str(a, n)
        b = libmp.to_str(b, n)
        return "mpi(%r, %r)" % (a, b)

    def _compare(s, t, cmpfun):
        if not hasattr(t, "_mpi_"):
            try:
                t = s.ctx.convert(t)
            except:
                return NotImplemented
        return cmpfun(s._mpi_, t._mpi_)

    def __eq__(s, t): return s._compare(t, libmp.mpi_eq)
    def __ne__(s, t): return s._compare(t, libmp.mpi_ne)
    def __lt__(s, t): return s._compare(t, libmp.mpi_lt)
    def __le__(s, t): return s._compare(t, libmp.mpi_le)
    def __gt__(s, t): return s._compare(t, libmp.mpi_gt)
    def __ge__(s, t): return s._compare(t, libmp.mpi_ge)

    def __abs__(self):
        return self.ctx.make_mpf(mpi_abs(self._mpi_, self.ctx.prec))
    def __pos__(self):
        return self.ctx.make_mpf(mpi_pos(self._mpi_, self.ctx.prec))
    def __neg__(self):
        return self.ctx.make_mpf(mpi_neg(self._mpi_, self.ctx.prec))

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.ctx.almosteq(s, t, rel_eps, abs_eps)

class ivmpc(object):

    def __new__(cls, re=0, im=0):
        re = cls.ctx.convert(re)
        im = cls.ctx.convert(im)
        y = new(cls)
        y._mpci_ = re._mpi_, im._mpi_
        return y

    def __hash__(self):
        (a, b), (c,d) = self._mpci_
        if a == b and c == d:
            return mpc_hash((a, c))
        else:
            return hash(self._mpci_)

    def __repr__(s):
        if s.ctx.pretty:
            return str(s)
        return "iv.mpc(%s, %s)" % (repr(s.real), repr(s.imag))

    def __str__(s):
        return "(%s + %s*j)" % (str(s.real), str(s.imag))

    @property
    def a(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((a, a))

    @property
    def b(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((b, b))

    @property
    def c(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((c, c))

    @property
    def d(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((d, d))

    @property
    def real(s):
        return s.ctx.make_mpf(s._mpci_[0])

    @property
    def imag(s):
        return s.ctx.make_mpf(s._mpci_[1])

    def conjugate(s):
        a, b = s._mpci_
        return s.ctx.make_mpc((a, mpf_neg(b)))

    def overlap(s, t):
        t = s.ctx.convert(t)
        real_overlap = (s.a <= t.a <= s.b) or (s.a <= t.b <= s.b) or (t.a <= s.a <= t.b) or (t.a <= s.b <= t.b)
        imag_overlap = (s.c <= t.c <= s.d) or (s.c <= t.d <= s.d) or (t.c <= s.c <= t.d) or (t.c <= s.d <= t.d)
        return real_overlap and imag_overlap

    def __contains__(s, t):
        t = s.ctx.convert(t)
        return t.real in s.real and t.imag in s.imag

    def _compare(s, t, ne=False):
        if not isinstance(t, s.ctx._types):
            try:
                t = s.ctx.convert(t)
            except:
                return NotImplemented
        if hasattr(t, '_mpi_'):
            tval = t._mpi_, mpi_zero
        elif hasattr(t, '_mpci_'):
            tval = t._mpci_
        if ne:
            return s._mpci_ != tval
        return s._mpci_ == tval

    def __eq__(s, t): return s._compare(t)
    def __ne__(s, t): return s._compare(t, True)

    def __lt__(s, t): raise TypeError("complex intervals cannot be ordered")
    __le__ = __gt__ = __ge__ = __lt__

    def __neg__(s): return s.ctx.make_mpc(mpci_neg(s._mpci_, s.ctx.prec))
    def __pos__(s): return s.ctx.make_mpc(mpci_pos(s._mpci_, s.ctx.prec))
    def __abs__(s): return s.ctx.make_mpf(mpci_abs(s._mpci_, s.ctx.prec))

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.ctx.almosteq(s, t, rel_eps, abs_eps)

def _binary_op(f_real, f_complex):
    def g_complex(ctx, sval, tval):
        return ctx.make_mpc(f_complex(sval, tval, ctx.prec))
    def g_real(ctx, sval, tval):
        try:
            return ctx.make_mpf(f_real(sval, tval, ctx.prec))
        except ComplexResult:
            sval = (sval, mpi_zero)
            tval = (tval, mpi_zero)
            return g_complex(ctx, sval, tval)
    def lop_real(s, t):
        if isinstance(t, _matrix): return NotImplemented
        ctx = s.ctx
        if not isinstance(t, ctx._types): t = ctx.convert(t)
        if hasattr(t, "_mpi_"): return g_real(ctx, s._mpi_, t._mpi_)
        if hasattr(t, "_mpci_"): return g_complex(ctx, (s._mpi_, mpi_zero), t._mpci_)
        return NotImplemented
    def rop_real(s, t):
        ctx = s.ctx
        if not isinstance(t, ctx._types): t = ctx.convert(t)
        if hasattr(t, "_mpi_"): return g_real(ctx, t._mpi_, s._mpi_)
        if hasattr(t, "_mpci_"): return g_complex(ctx, t._mpci_, (s._mpi_, mpi_zero))
        return NotImplemented
    def lop_complex(s, t):
        if isinstance(t, _matrix): return NotImplemented
        ctx = s.ctx
        if not isinstance(t, s.ctx._types):
            try:
                t = s.ctx.convert(t)
            except (ValueError, TypeError):
                return NotImplemented
        return g_complex(ctx, s._mpci_, t._mpci_)
    def rop_complex(s, t):
        ctx = s.ctx
        if not isinstance(t, s.ctx._types):
            t = s.ctx.convert(t)
        return g_complex(ctx, t._mpci_, s._mpci_)
    return lop_real, rop_real, lop_complex, rop_complex

ivmpf.__add__, ivmpf.__radd__, ivmpc.__add__, ivmpc.__radd__ = _binary_op(mpi_add, mpci_add)
ivmpf.__sub__, ivmpf.__rsub__, ivmpc.__sub__, ivmpc.__rsub__ = _binary_op(mpi_sub, mpci_sub)
ivmpf.__mul__, ivmpf.__rmul__, ivmpc.__mul__, ivmpc.__rmul__ = _binary_op(mpi_mul, mpci_mul)
ivmpf.__div__, ivmpf.__rdiv__, ivmpc.__div__, ivmpc.__rdiv__ = _binary_op(mpi_div, mpci_div)
ivmpf.__pow__, ivmpf.__rpow__, ivmpc.__pow__, ivmpc.__rpow__ = _binary_op(mpi_pow, mpci_pow)

ivmpf.__truediv__ = ivmpf.__div__; ivmpf.__rtruediv__ = ivmpf.__rdiv__
ivmpc.__truediv__ = ivmpc.__div__; ivmpc.__rtruediv__ = ivmpc.__rdiv__

class ivmpf_constant(ivmpf):
    def __new__(cls, f):
        self = new(cls)
        self._f = f
        return self
    def _get_mpi_(self):
        prec = self.ctx._prec[0]
        a = self._f(prec, round_floor)
        b = self._f(prec, round_ceiling)
        return a, b
    _mpi_ = property(_get_mpi_)

class MPIntervalContext(StandardBaseContext):

    def __init__(ctx):
        ctx.mpf = type('ivmpf', (ivmpf,), {})
        ctx.mpc = type('ivmpc', (ivmpc,), {})
        ctx._types = (ctx.mpf, ctx.mpc)
        ctx._constant = type('ivmpf_constant', (ivmpf_constant,), {})
        ctx._prec = [53]
        ctx._set_prec(53)
        ctx._constant._ctxdata = ctx.mpf._ctxdata = ctx.mpc._ctxdata = [ctx.mpf, new, ctx._prec]
        ctx._constant.ctx = ctx.mpf.ctx = ctx.mpc.ctx = ctx
        ctx.pretty = False
        StandardBaseContext.__init__(ctx)
        ctx._init_builtins()

    def _mpi(ctx, a, b=None):
        if b is None:
            return ctx.mpf(a)
        return ctx.mpf((a,b))

    def _init_builtins(ctx):
        ctx.one = ctx.mpf(1)
        ctx.zero = ctx.mpf(0)
        ctx.inf = ctx.mpf('inf')
        ctx.ninf = -ctx.inf
        ctx.nan = ctx.mpf('nan')
        ctx.j = ctx.mpc(0,1)
        ctx.exp = ctx._wrap_mpi_function(libmp.mpi_exp, libmp.mpci_exp)
        ctx.sqrt = ctx._wrap_mpi_function(libmp.mpi_sqrt)
        ctx.ln = ctx._wrap_mpi_function(libmp.mpi_log, libmp.mpci_log)
        ctx.cos = ctx._wrap_mpi_function(libmp.mpi_cos, libmp.mpci_cos)
        ctx.sin = ctx._wrap_mpi_function(libmp.mpi_sin, libmp.mpci_sin)
        ctx.tan = ctx._wrap_mpi_function(libmp.mpi_tan)
        ctx.gamma = ctx._wrap_mpi_function(libmp.mpi_gamma, libmp.mpci_gamma)
        ctx.loggamma = ctx._wrap_mpi_function(libmp.mpi_loggamma, libmp.mpci_loggamma)
        ctx.rgamma = ctx._wrap_mpi_function(libmp.mpi_rgamma, libmp.mpci_rgamma)
        ctx.factorial = ctx._wrap_mpi_function(libmp.mpi_factorial, libmp.mpci_factorial)
        ctx.fac = ctx.factorial

        ctx.eps = ctx._constant(lambda prec, rnd: (0, MPZ_ONE, 1-prec, 1))
        ctx.pi = ctx._constant(libmp.mpf_pi)
        ctx.e = ctx._constant(libmp.mpf_e)
        ctx.ln2 = ctx._constant(libmp.mpf_ln2)
        ctx.ln10 = ctx._constant(libmp.mpf_ln10)
        ctx.phi = ctx._constant(libmp.mpf_phi)
        ctx.euler = ctx._constant(libmp.mpf_euler)
        ctx.catalan = ctx._constant(libmp.mpf_catalan)
        ctx.glaisher = ctx._constant(libmp.mpf_glaisher)
        ctx.khinchin = ctx._constant(libmp.mpf_khinchin)
        ctx.twinprime = ctx._constant(libmp.mpf_twinprime)

    def _wrap_mpi_function(ctx, f_real, f_complex=None):
        def g(x, **kwargs):
            if kwargs:
                prec = kwargs.get('prec', ctx._prec[0])
            else:
                prec = ctx._prec[0]
            x = ctx.convert(x)
            if hasattr(x, "_mpi_"):
                return ctx.make_mpf(f_real(x._mpi_, prec))
            if hasattr(x, "_mpci_"):
                return ctx.make_mpc(f_complex(x._mpci_, prec))
            raise ValueError
        return g

    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        if wrap:
            def f_wrapped(ctx, *args, **kwargs):
                convert = ctx.convert
                args = [convert(a) for a in args]
                prec = ctx.prec
                try:
                    ctx.prec += 10
                    retval = f(ctx, *args, **kwargs)
                finally:
                    ctx.prec = prec
                return +retval
        else:
            f_wrapped = f
        setattr(cls, name, f_wrapped)

    def _set_prec(ctx, n):
        ctx._prec[0] = max(1, int(n))
        ctx._dps = prec_to_dps(n)

    def _set_dps(ctx, n):
        ctx._prec[0] = dps_to_prec(n)
        ctx._dps = max(1, int(n))

    prec = property(lambda ctx: ctx._prec[0], _set_prec)
    dps = property(lambda ctx: ctx._dps, _set_dps)

    def make_mpf(ctx, v):
        a = new(ctx.mpf)
        a._mpi_ = v
        return a

    def make_mpc(ctx, v):
        a = new(ctx.mpc)
        a._mpci_ = v
        return a

    def _mpq(ctx, pq):
        p, q = pq
        a = libmp.from_rational(p, q, ctx.prec, round_floor)
        b = libmp.from_rational(p, q, ctx.prec, round_ceiling)
        return ctx.make_mpf((a, b))

    def convert(ctx, x):
        if isinstance(x, (ctx.mpf, ctx.mpc)):
            return x
        if isinstance(x, ctx._constant):
            return +x
        if isinstance(x, complex) or hasattr(x, "_mpc_"):
            re = ctx.convert(x.real)
            im = ctx.convert(x.imag)
            return ctx.mpc(re,im)
        if isinstance(x, basestring):
            v = mpi_from_str(x, ctx.prec)
            return ctx.make_mpf(v)
        if hasattr(x, "_mpi_"):
            a, b = x._mpi_
        else:
            try:
                a, b = x
            except (TypeError, ValueError):
                a = b = x
            if hasattr(a, "_mpi_"):
                a = a._mpi_[0]
            else:
                a = convert_mpf_(a, ctx.prec, round_floor)
            if hasattr(b, "_mpi_"):
                b = b._mpi_[1]
            else:
                b = convert_mpf_(b, ctx.prec, round_ceiling)
        if a == fnan or b == fnan:
            a = fninf
            b = finf
        assert mpf_le(a, b), "endpoints must be properly ordered"
        return ctx.make_mpf((a, b))

    def nstr(ctx, x, n=5, **kwargs):
        x = ctx.convert(x)
        if hasattr(x, "_mpi_"):
            return libmp.mpi_to_str(x._mpi_, n, **kwargs)
        if hasattr(x, "_mpci_"):
            re = libmp.mpi_to_str(x._mpci_[0], n, **kwargs)
            im = libmp.mpi_to_str(x._mpci_[1], n, **kwargs)
            return "(%s + %s*j)" % (re, im)

    def mag(ctx, x):
        x = ctx.convert(x)
        if isinstance(x, ctx.mpc):
            return max(ctx.mag(x.real), ctx.mag(x.imag)) + 1
        a, b = libmp.mpi_abs(x._mpi_)
        sign, man, exp, bc = b
        if man:
            return exp+bc
        if b == fzero:
            return ctx.ninf
        if b == fnan:
            return ctx.nan
        return ctx.inf

    def isnan(ctx, x):
        return False

    def isinf(ctx, x):
        return x == ctx.inf

    def isint(ctx, x):
        x = ctx.convert(x)
        a, b = x._mpi_
        if a == b:
            sign, man, exp, bc = a
            if man:
                return exp >= 0
            return a == fzero
        return None

    def ldexp(ctx, x, n):
        a, b = ctx.convert(x)._mpi_
        a = libmp.mpf_shift(a, n)
        b = libmp.mpf_shift(b, n)
        return ctx.make_mpf((a,b))

    def absmin(ctx, x):
        return abs(ctx.convert(x)).a

    def absmax(ctx, x):
        return abs(ctx.convert(x)).b

    def atan2(ctx, y, x):
        y = ctx.convert(y)._mpi_
        x = ctx.convert(x)._mpi_
        return ctx.make_mpf(libmp.mpi_atan2(y,x,ctx.prec))

    def _convert_param(ctx, x):
        if isinstance(x, libmp.int_types):
            return x, 'Z'
        if isinstance(x, tuple):
            p, q = x
            return (ctx.mpf(p) / ctx.mpf(q), 'R')
        x = ctx.convert(x)
        if isinstance(x, ctx.mpf):
            return x, 'R'
        if isinstance(x, ctx.mpc):
            return x, 'C'
        raise ValueError

    def _is_real_type(ctx, z):
        return isinstance(z, ctx.mpf) or isinstance(z, int_types)

    def _is_complex_type(ctx, z):
        return isinstance(z, ctx.mpc)

    def hypsum(ctx, p, q, types, coeffs, z, maxterms=6000, **kwargs):
        coeffs = list(coeffs)
        num = range(p)
        den = range(p,p+q)
        #tol = ctx.eps
        s = t = ctx.one
        k = 0
        while 1:
            for i in num: t *= (coeffs[i]+k)
            for i in den: t /= (coeffs[i]+k)
            k += 1; t /= k; t *= z; s += t
            if t == 0:
                return s
            #if abs(t) < tol:
            #    return s
            if k > maxterms:
                raise ctx.NoConvergence


# Register with "numbers" ABC
#     We do not subclass, hence we do not use the @abstractmethod checks. While
#     this is less invasive it may turn out that we do not actually support
#     parts of the expected interfaces.  See
#     http://docs.python.org/2/library/numbers.html for list of abstract
#     methods.
try:
    import numbers
    numbers.Complex.register(ivmpc)
    numbers.Real.register(ivmpf)
except ImportError:
    pass
<|endoftext|>from operator import gt, lt

from .libmp.backend import xrange

from .functions.functions import SpecialFunctions
from .functions.rszeta import RSCache
from .calculus.quadrature import QuadratureMethods
from .calculus.inverselaplace import LaplaceTransformInversionMethods
from .calculus.calculus import CalculusMethods
from .calculus.optimization import OptimizationMethods
from .calculus.odes import ODEMethods
from .matrices.matrices import MatrixMethods
from .matrices.calculus import MatrixCalculusMethods
from .matrices.linalg import LinearAlgebraMethods
from .matrices.eigen import Eigen
from .identification import IdentificationMethods
from .visualization import VisualizationMethods

from . import libmp

class Context(object):
    pass

class StandardBaseContext(Context,
    SpecialFunctions,
    RSCache,
    QuadratureMethods,
    LaplaceTransformInversionMethods,
    CalculusMethods,
    MatrixMethods,
    MatrixCalculusMethods,
    LinearAlgebraMethods,
    Eigen,
    IdentificationMethods,
    OptimizationMethods,
    ODEMethods,
    VisualizationMethods):

    NoConvergence = libmp.NoConvergence
    ComplexResult = libmp.ComplexResult

    def __init__(ctx):
        ctx._aliases = {}
        # Call those that need preinitialization (e.g. for wrappers)
        SpecialFunctions.__init__(ctx)
        RSCache.__init__(ctx)
        QuadratureMethods.__init__(ctx)
        LaplaceTransformInversionMethods.__init__(ctx)
        CalculusMethods.__init__(ctx)
        MatrixMethods.__init__(ctx)

    def _init_aliases(ctx):
        for alias, value in ctx._aliases.items():
            try:
                setattr(ctx, alias, getattr(ctx, value))
            except AttributeError:
                pass

    _fixed_precision = False

    # XXX
    verbose = False

    def warn(ctx, msg):
        print("Warning:", msg)

    def bad_domain(ctx, msg):
        raise ValueError(msg)

    def _re(ctx, x):
        if hasattr(x, "real"):
            return x.real
        return x

    def _im(ctx, x):
        if hasattr(x, "imag"):
            return x.imag
        return ctx.zero

    def _as_points(ctx, x):
        return x

    def fneg(ctx, x, **kwargs):
        return -ctx.convert(x)

    def fadd(ctx, x, y, **kwargs):
        return ctx.convert(x)+ctx.convert(y)

    def fsub(ctx, x, y, **kwargs):
        return ctx.convert(x)-ctx.convert(y)

    def fmul(ctx, x, y, **kwargs):
        return ctx.convert(x)*ctx.convert(y)

    def fdiv(ctx, x, y, **kwargs):
        return ctx.convert(x)/ctx.convert(y)

    def fsum(ctx, args, absolute=False, squared=False):
        if absolute:
            if squared:
                return sum((abs(x)**2 for x in args), ctx.zero)
            return sum((abs(x) for x in args), ctx.zero)
        if squared:
            return sum((x**2 for x in args), ctx.zero)
        return sum(args, ctx.zero)

    def fdot(ctx, xs, ys=None, conjugate=False):
        if ys is not None:
            xs = zip(xs, ys)
        if conjugate:
            cf = ctx.conj
            return sum((x*cf(y) for (x,y) in xs), ctx.zero)
        else:
            return sum((x*y for (x,y) in xs), ctx.zero)

    def fprod(ctx, args):
        prod = ctx.one
        for arg in args:
            prod *= arg
        return prod

    def nprint(ctx, x, n=6, **kwargs):
        """
        Equivalent to ``print(nstr(x, n))``.
        """
        print(ctx.nstr(x, n, **kwargs))

    def chop(ctx, x, tol=None):
        """
        Chops off small real or imaginary parts, or converts
        numbers close to zero to exact zeros. The input can be a
        single number or an iterable::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> chop(5+1e-10j, tol=1e-9)
            mpf('5.0')
            >>> nprint(chop([1.0, 1e-20, 3+1e-18j, -4, 2]))
            [1.0, 0.0, 3.0, -4.0, 2.0]

        The tolerance defaults to ``100*eps``.
        """
        if tol is None:
            tol = 100*ctx.eps
        try:
            x = ctx.convert(x)
            absx = abs(x)
            if abs(x) < tol:
                return ctx.zero
            if ctx._is_complex_type(x):
                #part_tol = min(tol, absx*tol)
                part_tol = max(tol, absx*tol)
                if abs(x.imag) < part_tol:
                    return x.real
                if abs(x.real) < part_tol:
                    return ctx.mpc(0, x.imag)
        except TypeError:
            if isinstance(x, ctx.matrix):
                return x.apply(lambda a: ctx.chop(a, tol))
            if hasattr(x, "__iter__"):
                return [ctx.chop(a, tol) for a in x]
        return x

    def almosteq(ctx, s, t, rel_eps=None, abs_eps=None):
        r"""
        Determine whether the difference between `s` and `t` is smaller
        than a given epsilon, either relatively or absolutely.

        Both a maximum relative difference and a maximum difference
        ('epsilons') may be specified. The absolute difference is
        defined as `|s-t|` and the relative difference is defined
        as `|s-t|/\max(|s|, |t|)`.

        If only one epsilon is given, both are set to the same value.
        If none is given, both epsilons are set to `2^{-p+m}` where
        `p` is the current working precision and `m` is a small
        integer. The default setting typically allows :func:`~mpmath.almosteq`
        to be used to check for mathematical equality
        in the presence of small rounding errors.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> almosteq(3.141592653589793, 3.141592653589790)
            True
            >>> almosteq(3.141592653589793, 3.141592653589700)
            False
            >>> almosteq(3.141592653589793, 3.141592653589700, 1e-10)
            True
            >>> almosteq(1e-20, 2e-20)
            True
            >>> almosteq(1e-20, 2e-20, rel_eps=0, abs_eps=0)
            False

        """
        t = ctx.convert(t)
        if abs_eps is None and rel_eps is None:
            rel_eps = abs_eps = ctx.ldexp(1, -ctx.prec+4)
        if abs_eps is None:
            abs_eps = rel_eps
        elif rel_eps is None:
            rel_eps = abs_eps
        diff = abs(s-t)
        if diff <= abs_eps:
            return True
        abss = abs(s)
        abst = abs(t)
        if abss < abst:
            err = diff/abst
        else:
            err = diff/abss
        return err <= rel_eps

    def arange(ctx, *args):
        r"""
        This is a generalized version of Python's :func:`~mpmath.range` function
        that accepts fractional endpoints and step sizes and
        returns a list of ``mpf`` instances. Like :func:`~mpmath.range`,
        :func:`~mpmath.arange` can be called with 1, 2 or 3 arguments:

        ``arange(b)``
            `[0, 1, 2, \ldots, x]`
        ``arange(a, b)``
            `[a, a+1, a+2, \ldots, x]`
        ``arange(a, b, h)``
            `[a, a+h, a+h, \ldots, x]`

        where `b-1 \le x < b` (in the third case, `b-h \le x < b`).

        Like Python's :func:`~mpmath.range`, the endpoint is not included. To
        produce ranges where the endpoint is included, :func:`~mpmath.linspace`
        is more convenient.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> arange(4)
            [mpf('0.0'), mpf('1.0'), mpf('2.0'), mpf('3.0')]
            >>> arange(1, 2, 0.25)
            [mpf('1.0'), mpf('1.25'), mpf('1.5'), mpf('1.75')]
            >>> arange(1, -1, -0.75)
            [mpf('1.0'), mpf('0.25'), mpf('-0.5')]

        """
        if not len(args) <= 3:
            raise TypeError('arange expected at most 3 arguments, got %i'
                            % len(args))
        if not len(args) >= 1:
            raise TypeError('arange expected at least 1 argument, got %i'
                            % len(args))
        # set default
        a = 0
        dt = 1
        # interpret arguments
        if len(args) == 1:
            b = args[0]
        elif len(args) >= 2:
            a = args[0]
            b = args[1]
        if len(args) == 3:
            dt = args[2]
        a, b, dt = ctx.mpf(a), ctx.mpf(b), ctx.mpf(dt)
        assert a + dt != a, 'dt is too small and would cause an infinite loop'
        # adapt code for sign of dt
        if a > b:
            if dt > 0:
                return []
            op = gt
        else:
            if dt < 0:
                return []
            op = lt
        # create list
        result = []
        i = 0
        t = a
        while 1:
            t = a + dt*i
            i += 1
            if op(t, b):
                result.append(t)
            else:
                break
        return result

    def linspace(ctx, *args, **kwargs):
        """
        ``linspace(a, b, n)`` returns a list of `n` evenly spaced
        samples from `a` to `b`. The syntax ``linspace(mpi(a,b), n)``
        is also valid.

        This function is often more convenient than :func:`~mpmath.arange`
        for partitioning an interval into subintervals, since
        the endpoint is included::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> linspace(1, 4, 4)
            [mpf('1.0'), mpf('2.0'), mpf('3.0'), mpf('4.0')]

        You may also provide the keyword argument ``endpoint=False``::

            >>> linspace(1, 4, 4, endpoint=False)
            [mpf('1.0'), mpf('1.75'), mpf('2.5'), mpf('3.25')]

        """
        if len(args) == 3:
            a = ctx.mpf(args[0])
            b = ctx.mpf(args[1])
            n = int(args[2])
        elif len(args) == 2:
            assert hasattr(args[0], '_mpi_')
            a = args[0].a
            b = args[0].b
            n = int(args[1])
        else:
            raise TypeError('linspace expected 2 or 3 arguments, got %i' \
                            % len(args))
        if n < 1:
            raise ValueError('n must be greater than 0')
        if not 'endpoint' in kwargs or kwargs['endpoint']:
            if n == 1:
                return [ctx.mpf(a)]
            step = (b - a) / ctx.mpf(n - 1)
            y = [i*step + a for i in xrange(n)]
            y[-1] = b
        else:
            step = (b - a) / ctx.mpf(n)
            y = [i*step + a for i in xrange(n)]
        return y

    def cos_sin(ctx, z, **kwargs):
        return ctx.cos(z, **kwargs), ctx.sin(z, **kwargs)

    def cospi_sinpi(ctx, z, **kwargs):
        return ctx.cospi(z, **kwargs), ctx.sinpi(z, **kwargs)

    def _default_hyper_maxprec(ctx, p):
        return int(1000 * p**0.25 + 4*p)

    _gcd = staticmethod(libmp.gcd)
    list_primes = staticmethod(libmp.list_primes)
    isprime = staticmethod(libmp.isprime)
    bernfrac = staticmethod(libmp.bernfrac)
    moebius = staticmethod(libmp.moebius)
    _ifac = staticmethod(libmp.ifac)
    _eulernum = staticmethod(libmp.eulernum)
    _stirling1 = staticmethod(libmp.stirling1)
    _stirling2 = staticmethod(libmp.stirling2)

    def sum_accurately(ctx, terms, check_step=1):
        prec = ctx.prec
        try:
            extraprec = 10
            while 1:
                ctx.prec = prec + extraprec + 5
                max_mag = ctx.ninf
                s = ctx.zero
                k = 0
                for term in terms():
                    s += term
                    if (not k % check_step) and term:
                        term_mag = ctx.mag(term)
                        max_mag = max(max_mag, term_mag)
                        sum_mag = ctx.mag(s)
                        if sum_mag - term_mag > ctx.prec:
                            break
                    k += 1
                cancellation = max_mag - sum_mag
                if cancellation != cancellation:
                    break
                if cancellation < extraprec or ctx._fixed_precision:
                    break
                extraprec += min(ctx.prec, cancellation)
            return s
        finally:
            ctx.prec = prec

    def mul_accurately(ctx, factors, check_step=1):
        prec = ctx.prec
        try:
            extraprec = 10
            while 1:
                ctx.prec = prec + extraprec + 5
                max_mag = ctx.ninf
                one = ctx.one
                s = one
                k = 0
                for factor in factors():
                    s *= factor
                    term = factor - one
                    if (not k % check_step):
                        term_mag = ctx.mag(term)
                        max_mag = max(max_mag, term_mag)
                        sum_mag = ctx.mag(s-one)
                        #if sum_mag - term_mag > ctx.prec:
                        #    break
                        if -term_mag > ctx.prec:
                            break
                    k += 1
                cancellation = max_mag - sum_mag
                if cancellation != cancellation:
                    break
                if cancellation < extraprec or ctx._fixed_precision:
                    break
                extraprec += min(ctx.prec, cancellation)
            return s
        finally:
            ctx.prec = prec

    def power(ctx, x, y):
        r"""Converts `x` and `y` to mpmath numbers and evaluates
        `x^y = \exp(y \log(x))`::

            >>> from mpmath import *
            >>> mp.dps = 30; mp.pretty = True
            >>> power(2, 0.5)
            1.41421356237309504880168872421

        This shows the leading few digits of a large Mersenne prime
        (performing the exact calculation ``2**43112609-1`` and
        displaying the result in Python would be very slow)::

            >>> power(2, 43112609)-1
            3.16470269330255923143453723949e+12978188
        """
        return ctx.convert(x) ** ctx.convert(y)

    def _zeta_int(ctx, n):
        return ctx.zeta(n)

    def maxcalls(ctx, f, N):
        """
        Return a wrapped copy of *f* that raises ``NoConvergence`` when *f*
        has been called more than *N* times::

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> f = maxcalls(sin, 10)
            >>> print(sum(f(n) for n in range(10)))
            1.95520948210738
            >>> f(10) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: maxcalls: function evaluated 10 times

        """
        counter = [0]
        def f_maxcalls_wrapped(*args, **kwargs):
            counter[0] += 1
            if counter[0] > N:
                raise ctx.NoConvergence("maxcalls: function evaluated %i times" % N)
            return f(*args, **kwargs)
        return f_maxcalls_wrapped

    def memoize(ctx, f):
        """
        Return a wrapped copy of *f* that caches computed values, i.e.
        a memoized copy of *f*. Values are only reused if the cached precision
        is equal to or higher than the working precision::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> f = memoize(maxcalls(sin, 1))
            >>> f(2)
            0.909297426825682
            >>> f(2)
            0.909297426825682
            >>> mp.dps = 25
            >>> f(2) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: maxcalls: function evaluated 1 times

        """
        f_cache = {}
        def f_cached(*args, **kwargs):
            if kwargs:
                key = args, tuple(kwargs.items())
            else:
                key = args
            prec = ctx.prec
            if key in f_cache:
                cprec, cvalue = f_cache[key]
                if cprec >= prec:
                    return +cvalue
            value = f(*args, **kwargs)
            f_cache[key] = (prec, value)
            return value
        f_cached.__name__ = f.__name__
        f_cached.__doc__ = f.__doc__
        return f_cached
<|endoftext|>#from ctx_base import StandardBaseContext

from .libmp.backend import basestring, exec_

from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
    round_floor, round_ceiling, dps_to_prec, round_nearest, prec_to_dps,
    ComplexResult, to_pickable, from_pickable, normalize,
    from_int, from_float, from_npfloat, from_Decimal, from_str, to_int, to_float, to_str,
    from_rational, from_man_exp,
    fone, fzero, finf, fninf, fnan,
    mpf_abs, mpf_pos, mpf_neg, mpf_add, mpf_sub, mpf_mul, mpf_mul_int,
    mpf_div, mpf_rdiv_int, mpf_pow_int, mpf_mod,
    mpf_eq, mpf_cmp, mpf_lt, mpf_gt, mpf_le, mpf_ge,
    mpf_hash, mpf_rand,
    mpf_sum,
    bitcount, to_fixed,
    mpc_to_str,
    mpc_to_complex, mpc_hash, mpc_pos, mpc_is_nonzero, mpc_neg, mpc_conjugate,
    mpc_abs, mpc_add, mpc_add_mpf, mpc_sub, mpc_sub_mpf, mpc_mul, mpc_mul_mpf,
    mpc_mul_int, mpc_div, mpc_div_mpf, mpc_pow, mpc_pow_mpf, mpc_pow_int,
    mpc_mpf_div,
    mpf_pow,
    mpf_pi, mpf_degree, mpf_e, mpf_phi, mpf_ln2, mpf_ln10,
    mpf_euler, mpf_catalan, mpf_apery, mpf_khinchin,
    mpf_glaisher, mpf_twinprime, mpf_mertens,
    int_types)

from . import rational
from . import function_docs

new = object.__new__

class mpnumeric(object):
    """Base class for mpf and mpc."""
    __slots__ = []
    def __new__(cls, val):
        raise NotImplementedError

class _mpf(mpnumeric):
    """
    An mpf instance holds a real-valued floating-point number. mpf:s
    work analogously to Python floats, but support arbitrary-precision
    arithmetic.
    """
    __slots__ = ['_mpf_']

    def __new__(cls, val=fzero, **kwargs):
        """A new mpf can be created from a Python float, an int, a
        or a decimal string representing a number in floating-point
        format."""
        prec, rounding = cls.context._prec_rounding
        if kwargs:
            prec = kwargs.get('prec', prec)
            if 'dps' in kwargs:
                prec = dps_to_prec(kwargs['dps'])
            rounding = kwargs.get('rounding', rounding)
        if type(val) is cls:
            sign, man, exp, bc = val._mpf_
            if (not man) and exp:
                return val
            v = new(cls)
            v._mpf_ = normalize(sign, man, exp, bc, prec, rounding)
            return v
        elif type(val) is tuple:
            if len(val) == 2:
                v = new(cls)
                v._mpf_ = from_man_exp(val[0], val[1], prec, rounding)
                return v
            if len(val) == 4:
                if val not in (finf, fninf, fnan):
                    sign, man, exp, bc = val
                    val = normalize(sign, MPZ(man), exp, bc, prec, rounding)
                v = new(cls)
                v._mpf_ = val
                return v
            raise ValueError
        else:
            v = new(cls)
            v._mpf_ = mpf_pos(cls.mpf_convert_arg(val, prec, rounding), prec, rounding)
            return v

    @classmethod
    def mpf_convert_arg(cls, x, prec, rounding):
        if isinstance(x, int_types): return from_int(x)
        if isinstance(x, float): return from_float(x)
        if isinstance(x, basestring): return from_str(x, prec, rounding)
        if isinstance(x, cls.context.constant): return x.func(prec, rounding)
        if hasattr(x, '_mpf_'): return x._mpf_
        if hasattr(x, '_mpmath_'):
            t = cls.context.convert(x._mpmath_(prec, rounding))
            if hasattr(t, '_mpf_'):
                return t._mpf_
        if hasattr(x, '_mpi_'):
            a, b = x._mpi_
            if a == b:
                return a
            raise ValueError("can only create mpf from zero-width interval")
        raise TypeError("cannot create mpf from " + repr(x))

    @classmethod
    def mpf_convert_rhs(cls, x):
        if isinstance(x, int_types): return from_int(x)
        if isinstance(x, float): return from_float(x)
        if isinstance(x, complex_types): return cls.context.mpc(x)
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return from_rational(p, q, cls.context.prec)
        if hasattr(x, '_mpf_'): return x._mpf_
        if hasattr(x, '_mpmath_'):
            t = cls.context.convert(x._mpmath_(*cls.context._prec_rounding))
            if hasattr(t, '_mpf_'):
                return t._mpf_
            return t
        return NotImplemented

    @classmethod
    def mpf_convert_lhs(cls, x):
        x = cls.mpf_convert_rhs(x)
        if type(x) is tuple:
            return cls.context.make_mpf(x)
        return x

    man_exp = property(lambda self: self._mpf_[1:3])
    man = property(lambda self: self._mpf_[1])
    exp = property(lambda self: self._mpf_[2])
    bc = property(lambda self: self._mpf_[3])

    real = property(lambda self: self)
    imag = property(lambda self: self.context.zero)

    conjugate = lambda self: self

    def __getstate__(self): return to_pickable(self._mpf_)
    def __setstate__(self, val): self._mpf_ = from_pickable(val)

    def __repr__(s):
        if s.context.pretty:
            return str(s)
        return "mpf('%s')" % to_str(s._mpf_, s.context._repr_digits)

    def __str__(s): return to_str(s._mpf_, s.context._str_digits)
    def __hash__(s): return mpf_hash(s._mpf_)
    def __int__(s): return int(to_int(s._mpf_))
    def __long__(s): return long(to_int(s._mpf_))
    def __float__(s): return to_float(s._mpf_, rnd=s.context._prec_rounding[1])
    def __complex__(s): return complex(float(s))
    def __nonzero__(s): return s._mpf_ != fzero

    __bool__ = __nonzero__

    def __abs__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_abs(s._mpf_, prec, rounding)
        return v

    def __pos__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_pos(s._mpf_, prec, rounding)
        return v

    def __neg__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpf_ = mpf_neg(s._mpf_, prec, rounding)
        return v

    def _cmp(s, t, func):
        if hasattr(t, '_mpf_'):
            t = t._mpf_
        else:
            t = s.mpf_convert_rhs(t)
            if t is NotImplemented:
                return t
        return func(s._mpf_, t)

    def __cmp__(s, t): return s._cmp(t, mpf_cmp)
    def __lt__(s, t): return s._cmp(t, mpf_lt)
    def __gt__(s, t): return s._cmp(t, mpf_gt)
    def __le__(s, t): return s._cmp(t, mpf_le)
    def __ge__(s, t): return s._cmp(t, mpf_ge)

    def __ne__(s, t):
        v = s.__eq__(t)
        if v is NotImplemented:
            return v
        return not v

    def __rsub__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if type(t) in int_types:
            v = new(cls)
            v._mpf_ = mpf_sub(from_int(t), s._mpf_, prec, rounding)
            return v
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t - s

    def __rdiv__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpf_ = mpf_rdiv_int(t, s._mpf_, prec, rounding)
            return v
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t / s

    def __rpow__(s, t):
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t ** s

    def __rmod__(s, t):
        t = s.mpf_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t % s

    def sqrt(s):
        return s.context.sqrt(s)

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.context.almosteq(s, t, rel_eps, abs_eps)

    def to_fixed(self, prec):
        return to_fixed(self._mpf_, prec)

    def __round__(self, *args):
        return round(float(self), *args)

mpf_binary_op = """
def %NAME%(self, other):
    mpf, new, (prec, rounding) = self._ctxdata
    sval = self._mpf_
    if hasattr(other, '_mpf_'):
        tval = other._mpf_
        %WITH_MPF%
    ttype = type(other)
    if ttype in int_types:
        %WITH_INT%
    elif ttype is float:
        tval = from_float(other)
        %WITH_MPF%
    elif hasattr(other, '_mpc_'):
        tval = other._mpc_
        mpc = type(other)
        %WITH_MPC%
    elif ttype is complex:
        tval = from_float(other.real), from_float(other.imag)
        mpc = self.context.mpc
        %WITH_MPC%
    if isinstance(other, mpnumeric):
        return NotImplemented
    try:
        other = mpf.context.convert(other, strings=False)
    except TypeError:
        return NotImplemented
    return self.%NAME%(other)
"""

return_mpf = "; obj = new(mpf); obj._mpf_ = val; return obj"
return_mpc = "; obj = new(mpc); obj._mpc_ = val; return obj"

mpf_pow_same = """
        try:
            val = mpf_pow(sval, tval, prec, rounding) %s
        except ComplexResult:
            if mpf.context.trap_complex:
                raise
            mpc = mpf.context.mpc
            val = mpc_pow((sval, fzero), (tval, fzero), prec, rounding) %s
""" % (return_mpf, return_mpc)

def binary_op(name, with_mpf='', with_int='', with_mpc=''):
    code = mpf_binary_op
    code = code.replace("%WITH_INT%", with_int)
    code = code.replace("%WITH_MPC%", with_mpc)
    code = code.replace("%WITH_MPF%", with_mpf)
    code = code.replace("%NAME%", name)
    np = {}
    exec_(code, globals(), np)
    return np[name]

_mpf.__eq__ = binary_op('__eq__',
    'return mpf_eq(sval, tval)',
    'return mpf_eq(sval, from_int(other))',
    'return (tval[1] == fzero) and mpf_eq(tval[0], sval)')

_mpf.__add__ = binary_op('__add__',
    'val = mpf_add(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_add(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_add_mpf(tval, sval, prec, rounding)' + return_mpc)

_mpf.__sub__ = binary_op('__sub__',
    'val = mpf_sub(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_sub(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_sub((sval, fzero), tval, prec, rounding)' + return_mpc)

_mpf.__mul__ = binary_op('__mul__',
    'val = mpf_mul(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_mul_int(sval, other, prec, rounding)' + return_mpf,
    'val = mpc_mul_mpf(tval, sval, prec, rounding)' + return_mpc)

_mpf.__div__ = binary_op('__div__',
    'val = mpf_div(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_div(sval, from_int(other), prec, rounding)' + return_mpf,
    'val = mpc_mpf_div(sval, tval, prec, rounding)' + return_mpc)

_mpf.__mod__ = binary_op('__mod__',
    'val = mpf_mod(sval, tval, prec, rounding)' + return_mpf,
    'val = mpf_mod(sval, from_int(other), prec, rounding)' + return_mpf,
    'raise NotImplementedError("complex modulo")')

_mpf.__pow__ = binary_op('__pow__',
    mpf_pow_same,
    'val = mpf_pow_int(sval, other, prec, rounding)' + return_mpf,
    'val = mpc_pow((sval, fzero), tval, prec, rounding)' + return_mpc)

_mpf.__radd__ = _mpf.__add__
_mpf.__rmul__ = _mpf.__mul__
_mpf.__truediv__ = _mpf.__div__
_mpf.__rtruediv__ = _mpf.__rdiv__


class _constant(_mpf):
    """Represents a mathematical constant with dynamic precision.
    When printed or used in an arithmetic operation, a constant
    is converted to a regular mpf at the working precision. A
    regular mpf can also be obtained using the operation +x."""

    def __new__(cls, func, name, docname=''):
        a = object.__new__(cls)
        a.name = name
        a.func = func
        a.__doc__ = getattr(function_docs, docname, '')
        return a

    def __call__(self, prec=None, dps=None, rounding=None):
        prec2, rounding2 = self.context._prec_rounding
        if not prec: prec = prec2
        if not rounding: rounding = rounding2
        if dps: prec = dps_to_prec(dps)
        return self.context.make_mpf(self.func(prec, rounding))

    @property
    def _mpf_(self):
        prec, rounding = self.context._prec_rounding
        return self.func(prec, rounding)

    def __repr__(self):
        return "<%s: %s~>" % (self.name, self.context.nstr(self(dps=15)))


class _mpc(mpnumeric):
    """
    An mpc represents a complex number using a pair of mpf:s (one
    for the real part and another for the imaginary part.) The mpc
    class behaves fairly similarly to Python's complex type.
    """

    __slots__ = ['_mpc_']

    def __new__(cls, real=0, imag=0):
        s = object.__new__(cls)
        if isinstance(real, complex_types):
            real, imag = real.real, real.imag
        elif hasattr(real, '_mpc_'):
            s._mpc_ = real._mpc_
            return s
        real = cls.context.mpf(real)
        imag = cls.context.mpf(imag)
        s._mpc_ = (real._mpf_, imag._mpf_)
        return s

    real = property(lambda self: self.context.make_mpf(self._mpc_[0]))
    imag = property(lambda self: self.context.make_mpf(self._mpc_[1]))

    def __getstate__(self):
        return to_pickable(self._mpc_[0]), to_pickable(self._mpc_[1])

    def __setstate__(self, val):
        self._mpc_ = from_pickable(val[0]), from_pickable(val[1])

    def __repr__(s):
        if s.context.pretty:
            return str(s)
        r = repr(s.real)[4:-1]
        i = repr(s.imag)[4:-1]
        return "%s(real=%s, imag=%s)" % (type(s).__name__, r, i)

    def __str__(s):
        return "(%s)" % mpc_to_str(s._mpc_, s.context._str_digits)

    def __complex__(s):
        return mpc_to_complex(s._mpc_, rnd=s.context._prec_rounding[1])

    def __pos__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_pos(s._mpc_, prec, rounding)
        return v

    def __abs__(s):
        prec, rounding = s.context._prec_rounding
        v = new(s.context.mpf)
        v._mpf_ = mpc_abs(s._mpc_, prec, rounding)
        return v

    def __neg__(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_neg(s._mpc_, prec, rounding)
        return v

    def conjugate(s):
        cls, new, (prec, rounding) = s._ctxdata
        v = new(cls)
        v._mpc_ = mpc_conjugate(s._mpc_, prec, rounding)
        return v

    def __nonzero__(s):
        return mpc_is_nonzero(s._mpc_)

    __bool__ = __nonzero__

    def __hash__(s):
        return mpc_hash(s._mpc_)

    @classmethod
    def mpc_convert_lhs(cls, x):
        try:
            y = cls.context.convert(x)
            return y
        except TypeError:
            return NotImplemented

    def __eq__(s, t):
        if not hasattr(t, '_mpc_'):
            if isinstance(t, str):
                return False
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
        return s.real == t.real and s.imag == t.imag

    def __ne__(s, t):
        b = s.__eq__(t)
        if b is NotImplemented:
            return b
        return not b

    def _compare(*args):
        raise TypeError("no ordering relation is defined for complex numbers")

    __gt__ = _compare
    __le__ = _compare
    __gt__ = _compare
    __ge__ = _compare

    def __add__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_add_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_add(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __sub__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_sub_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_sub(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __mul__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            if isinstance(t, int_types):
                v = new(cls)
                v._mpc_ = mpc_mul_int(s._mpc_, t, prec, rounding)
                return v
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_mul_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
            t = s.mpc_convert_lhs(t)
        v = new(cls)
        v._mpc_ = mpc_mul(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __div__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if not hasattr(t, '_mpc_'):
            t = s.mpc_convert_lhs(t)
            if t is NotImplemented:
                return t
            if hasattr(t, '_mpf_'):
                v = new(cls)
                v._mpc_ = mpc_div_mpf(s._mpc_, t._mpf_, prec, rounding)
                return v
        v = new(cls)
        v._mpc_ = mpc_div(s._mpc_, t._mpc_, prec, rounding)
        return v

    def __pow__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpc_ = mpc_pow_int(s._mpc_, t, prec, rounding)
            return v
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        v = new(cls)
        if hasattr(t, '_mpf_'):
            v._mpc_ = mpc_pow_mpf(s._mpc_, t._mpf_, prec, rounding)
        else:
            v._mpc_ = mpc_pow(s._mpc_, t._mpc_, prec, rounding)
        return v

    __radd__ = __add__

    def __rsub__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t - s

    def __rmul__(s, t):
        cls, new, (prec, rounding) = s._ctxdata
        if isinstance(t, int_types):
            v = new(cls)
            v._mpc_ = mpc_mul_int(s._mpc_, t, prec, rounding)
            return v
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t * s

    def __rdiv__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t / s

    def __rpow__(s, t):
        t = s.mpc_convert_lhs(t)
        if t is NotImplemented:
            return t
        return t ** s

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.context.almosteq(s, t, rel_eps, abs_eps)


complex_types = (complex, _mpc)


class PythonMPContext(object):

    def __init__(ctx):
        ctx._prec_rounding = [53, round_nearest]
        ctx.mpf = type('mpf', (_mpf,), {})
        ctx.mpc = type('mpc', (_mpc,), {})
        ctx.mpf._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.mpc._ctxdata = [ctx.mpc, new, ctx._prec_rounding]
        ctx.mpf.context = ctx
        ctx.mpc.context = ctx
        ctx.constant = type('constant', (_constant,), {})
        ctx.constant._ctxdata = [ctx.mpf, new, ctx._prec_rounding]
        ctx.constant.context = ctx

    def make_mpf(ctx, v):
        a = new(ctx.mpf)
        a._mpf_ = v
        return a

    def make_mpc(ctx, v):
        a = new(ctx.mpc)
        a._mpc_ = v
        return a

    def default(ctx):
        ctx._prec = ctx._prec_rounding[0] = 53
        ctx._dps = 15
        ctx.trap_complex = False

    def _set_prec(ctx, n):
        ctx._prec = ctx._prec_rounding[0] = max(1, int(n))
        ctx._dps = prec_to_dps(n)

    def _set_dps(ctx, n):
        ctx._prec = ctx._prec_rounding[0] = dps_to_prec(n)
        ctx._dps = max(1, int(n))

    prec = property(lambda ctx: ctx._prec, _set_prec)
    dps = property(lambda ctx: ctx._dps, _set_dps)

    def convert(ctx, x, strings=True):
        """
        Converts *x* to an ``mpf`` or ``mpc``. If *x* is of type ``mpf``,
        ``mpc``, ``int``, ``float``, ``complex``, the conversion
        will be performed losslessly.

        If *x* is a string, the result will be rounded to the present
        working precision. Strings representing fractions or complex
        numbers are permitted.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> mpmathify(3.5)
            mpf('3.5')
            >>> mpmathify('2.1')
            mpf('2.1000000000000001')
            >>> mpmathify('3/4')
            mpf('0.75')
            >>> mpmathify('2+3j')
            mpc(real='2.0', imag='3.0')

        """
        if type(x) in ctx.types: return x
        if isinstance(x, int_types): return ctx.make_mpf(from_int(x))
        if isinstance(x, float): return ctx.make_mpf(from_float(x))
        if isinstance(x, complex):
            return ctx.make_mpc((from_float(x.real), from_float(x.imag)))
        if type(x).__module__ == 'numpy': return ctx.npconvert(x)
        if isinstance(x, numbers.Rational): # e.g. Fraction
            try: x = rational.mpq(int(x.numerator), int(x.denominator))
            except: pass
        prec, rounding = ctx._prec_rounding
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return ctx.make_mpf(from_rational(p, q, prec))
        if strings and isinstance(x, basestring):
            try:
                _mpf_ = from_str(x, prec, rounding)
                return ctx.make_mpf(_mpf_)
            except ValueError:
                pass
        if hasattr(x, '_mpf_'): return ctx.make_mpf(x._mpf_)
        if hasattr(x, '_mpc_'): return ctx.make_mpc(x._mpc_)
        if hasattr(x, '_mpmath_'):
            return ctx.convert(x._mpmath_(prec, rounding))
        if type(x).__module__ == 'decimal':
            try: return ctx.make_mpf(from_Decimal(x, prec, rounding))
            except: pass
        return ctx._convert_fallback(x, strings)

    def npconvert(ctx, x):
        """
        Converts *x* to an ``mpf`` or ``mpc``. *x* should be a numpy
        scalar.
        """
        import numpy as np
        if isinstance(x, np.integer): return ctx.make_mpf(from_int(int(x)))
        if isinstance(x, np.floating): return ctx.make_mpf(from_npfloat(x))
        if isinstance(x, np.complexfloating):
            return ctx.make_mpc((from_npfloat(x.real), from_npfloat(x.imag)))
        raise TypeError("cannot create mpf from " + repr(x))

    def isnan(ctx, x):
        """
        Return *True* if *x* is a NaN (not-a-number), or for a complex
        number, whether either the real or complex part is NaN;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isnan(3.14)
            False
            >>> isnan(nan)
            True
            >>> isnan(mpc(3.14,2.72))
            False
            >>> isnan(mpc(3.14,nan))
            True

        """
        if hasattr(x, "_mpf_"):
            return x._mpf_ == fnan
        if hasattr(x, "_mpc_"):
            return fnan in x._mpc_
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnan(x)
        raise TypeError("isnan() needs a number as input")

    def isinf(ctx, x):
        """
        Return *True* if the absolute value of *x* is infinite;
        otherwise return *False*::

            >>> from mpmath import *
            >>> isinf(inf)
            True
            >>> isinf(-inf)
            True
            >>> isinf(3)
            False
            >>> isinf(3+4j)
            False
            >>> isinf(mpc(3,inf))
            True
            >>> isinf(mpc(inf,3))
            True

        """
        if hasattr(x, "_mpf_"):
            return x._mpf_ in (finf, fninf)
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            return re in (finf, fninf) or im in (finf, fninf)
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return False
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isinf(x)
        raise TypeError("isinf() needs a number as input")

    def isnormal(ctx, x):
        """
        Determine whether *x* is "normal" in the sense of floating-point
        representation; that is, return *False* if *x* is zero, an
        infinity or NaN; otherwise return *True*. By extension, a
        complex number *x* is considered "normal" if its magnitude is
        normal::

            >>> from mpmath import *
            >>> isnormal(3)
            True
            >>> isnormal(0)
            False
            >>> isnormal(inf); isnormal(-inf); isnormal(nan)
            False
            False
            False
            >>> isnormal(0+0j)
            False
            >>> isnormal(0+3j)
            True
            >>> isnormal(mpc(2,nan))
            False
        """
        if hasattr(x, "_mpf_"):
            return bool(x._mpf_[1])
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            re_normal = bool(re[1])
            im_normal = bool(im[1])
            if re == fzero: return im_normal
            if im == fzero: return re_normal
            return re_normal and im_normal
        if isinstance(x, int_types) or isinstance(x, rational.mpq):
            return bool(x)
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isnormal(x)
        raise TypeError("isnormal() needs a number as input")

    def isint(ctx, x, gaussian=False):
        """
        Return *True* if *x* is integer-valued; otherwise return
        *False*::

            >>> from mpmath import *
            >>> isint(3)
            True
            >>> isint(mpf(3))
            True
            >>> isint(3.2)
            False
            >>> isint(inf)
            False

        Optionally, Gaussian integers can be checked for::

            >>> isint(3+0j)
            True
            >>> isint(3+2j)
            False
            >>> isint(3+2j, gaussian=True)
            True

        """
        if isinstance(x, int_types):
            return True
        if hasattr(x, "_mpf_"):
            sign, man, exp, bc = xval = x._mpf_
            return bool((man and exp >= 0) or xval == fzero)
        if hasattr(x, "_mpc_"):
            re, im = x._mpc_
            rsign, rman, rexp, rbc = re
            isign, iman, iexp, ibc = im
            re_isint = (rman and rexp >= 0) or re == fzero
            if gaussian:
                im_isint = (iman and iexp >= 0) or im == fzero
                return re_isint and im_isint
            return re_isint and im == fzero
        if isinstance(x, rational.mpq):
            p, q = x._mpq_
            return p % q == 0
        x = ctx.convert(x)
        if hasattr(x, '_mpf_') or hasattr(x, '_mpc_'):
            return ctx.isint(x, gaussian)
        raise TypeError("isint() needs a number as input")

    def fsum(ctx, terms, absolute=False, squared=False):
        """
        Calculates a sum containing a finite number of terms (for infinite
        series, see :func:`~mpmath.nsum`). The terms will be converted to
        mpmath numbers. For len(terms) > 2, this function is generally
        faster and produces more accurate results than the builtin
        Python function :func:`sum`.

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> fsum([1, 2, 0.5, 7])
            mpf('10.5')

        With squared=True each term is squared, and with absolute=True
        the absolute value of each term is used.
        """
        prec, rnd = ctx._prec_rounding
        real = []
        imag = []
        for term in terms:
            reval = imval = 0
            if hasattr(term, "_mpf_"):
                reval = term._mpf_
            elif hasattr(term, "_mpc_"):
                reval, imval = term._mpc_
            else:
                term = ctx.convert(term)
                if hasattr(term, "_mpf_"):
                    reval = term._mpf_
                elif hasattr(term, "_mpc_"):
                    reval, imval = term._mpc_
                else:
                    raise NotImplementedError
            if imval:
                if squared:
                    if absolute:
                        real.append(mpf_mul(reval,reval))
                        real.append(mpf_mul(imval,imval))
                    else:
                        reval, imval = mpc_pow_int((reval,imval),2,prec+10)
                        real.append(reval)
                        imag.append(imval)
                elif absolute:
                    real.append(mpc_abs((reval,imval), prec))
                else:
                    real.append(reval)
                    imag.append(imval)
            else:
                if squared:
                    reval = mpf_mul(reval, reval)
                elif absolute:
